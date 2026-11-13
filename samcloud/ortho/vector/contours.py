"""Outdoor DTM raster accumulation, interpolation, smoothing, and 3D contour generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator
from shapely import make_valid
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union
from skimage.measure import find_contours
import yaml

from .raster_grid import RasterGrid

DEFAULT_GROUND_CODES: tuple[int, ...] = (
    2,    # Ground
    3,    # Low Vegetation
    9,    # Water
    11,   # Road Surface
    20,   # Ignored Ground
    66,   # Hardscape
    70,   # Natural terrain
    71,   # Sealed terrain
    73,   # Side walk
    74,   # Curbstone
    172,  # HCS asphalt road
    173,  # HCS asphalt hardscape
    174,  # HCS concrete road
    175,  # HCS concrete hardscape
    184,  # HCS curb
)


@dataclass(frozen=True, slots=True)
class ContourSettings:
    """Configuration parameters for DTM generation and contour extraction."""

    resolution: float = 0.20
    major_interval: float = 1.0
    minor_interval: float = 0.25
    max_gap: float = 3.0
    smoothing_sigma_m: float = 0.3
    min_length: float = 2.0
    simplify_tolerance: float = 0.03
    chaikin_iterations: int = 2

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.major_interval <= 0.0 or self.minor_interval <= 0.0:
            raise ValueError("contour intervals must be positive")
        if self.max_gap < 0.0:
            raise ValueError("max_gap cannot be negative")
        if self.smoothing_sigma_m < 0.0:
            raise ValueError("smoothing_sigma_m cannot be negative")
        if self.min_length < 0.0:
            raise ValueError("min_length cannot be negative")
        if self.simplify_tolerance < 0.0:
            raise ValueError("simplify_tolerance cannot be negative")
        if self.chaikin_iterations < 0:
            raise ValueError("chaikin_iterations cannot be negative")


class GroundDtmAccumulator:
    """Streaming accumulator computing per-cell mean ground elevation with O(cells) memory."""

    def __init__(self, bounds: tuple[float, float, float, float], resolution: float) -> None:
        min_x, min_y, max_x, max_y = (float(v) for v in bounds)
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("bounds max coordinates must exceed min coordinates")
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")

        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.resolution = float(resolution)

        self.width = max(1, math.ceil((self.max_x - self.min_x) / self.resolution))
        self.height = max(1, math.ceil((self.max_y - self.min_y) / self.resolution))
        self.max_y_grid = self.min_y + self.height * self.resolution

        self.grid = RasterGrid(
            min_x=self.min_x,
            max_y=self.max_y_grid,
            pixel_size=self.resolution,
            width=self.width,
            height=self.height,
        )
        total_cells = self.width * self.height
        self._sum_z = np.zeros(total_cells, dtype=np.float64)
        self._counts = np.zeros(total_cells, dtype=np.int64)

    def add(self, xyz_local: np.ndarray, classes: np.ndarray, ground_codes: Iterable[int]) -> None:
        """Accumulate ground-class points using vectorized operations."""
        xyz = np.asarray(xyz_local, dtype=np.float64)
        cls = np.asarray(classes, dtype=np.int64)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError("xyz_local must have shape (N, 3)")
        if cls.shape != (xyz.shape[0],):
            raise ValueError("classes shape must match xyz_local row count")

        code_array = np.fromiter(ground_codes, dtype=np.int64)
        if len(code_array) == 0:
            return

        ground_mask = np.isin(cls, code_array)
        if not ground_mask.any():
            return

        x = xyz[ground_mask, 0]
        y = xyz[ground_mask, 1]
        z = xyz[ground_mask, 2]

        cols = np.floor((x - self.min_x) / self.resolution).astype(np.int64)
        rows = np.floor((self.max_y_grid - y) / self.resolution).astype(np.int64)

        valid = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
        if not valid.any():
            return

        cols = cols[valid]
        rows = rows[valid]
        z = z[valid]

        flat_idx = rows * self.width + cols
        total_cells = self.width * self.height

        counts = np.bincount(flat_idx, minlength=total_cells)
        z_sums = np.bincount(flat_idx, weights=z, minlength=total_cells)

        self._counts += counts
        self._sum_z += z_sums

    def finish(self, settings: ContourSettings | None = None) -> tuple[np.ndarray, RasterGrid]:
        """Compute the final mean z DTM raster with NaN where cells are empty."""
        total_cells = self.width * self.height
        dtm_flat = np.full(total_cells, np.nan, dtype=np.float64)
        has_data = self._counts > 0
        dtm_flat[has_data] = self._sum_z[has_data] / self._counts[has_data]
        dtm = dtm_flat.reshape((self.height, self.width))
        return dtm, self.grid


def fill_dtm(
    dtm: np.ndarray,
    grid: RasterGrid,
    *,
    max_gap: float,
    smoothing_sigma_m: float,
) -> np.ndarray:
    """Linearly interpolate empty cells within max_gap, followed by normalized Gaussian smoothing."""
    array = np.asarray(dtm, dtype=np.float64)
    if array.shape != (grid.height, grid.width):
        raise ValueError("dtm shape must match RasterGrid height and width")

    valid_mask = ~np.isnan(array)
    if not valid_mask.any():
        return array.copy()

    filled = array.copy()

    # Interpolate empty cells within max_gap
    if not valid_mask.all() and max_gap > 0.0:
        distances_m = ndimage.distance_transform_edt(~valid_mask, sampling=grid.pixel_size)
        to_fill = (~valid_mask) & (distances_m <= max_gap)

        if to_fill.any():
            valid_rows, valid_cols = np.nonzero(valid_mask)
            valid_z = array[valid_rows, valid_cols]

            if len(valid_z) > 2_000_000:
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(valid_z), size=2_000_000, replace=False)
                valid_rows = valid_rows[sample_idx]
                valid_cols = valid_cols[sample_idx]
                valid_z = valid_z[sample_idx]

            if len(valid_z) >= 3:
                points = np.column_stack((valid_rows, valid_cols))
                interp = LinearNDInterpolator(points, valid_z)
                fill_rows, fill_cols = np.nonzero(to_fill)
                query_points = np.column_stack((fill_rows, fill_cols))
                interpolated = interp(query_points)

                valid_interp = ~np.isnan(interpolated)
                filled[fill_rows[valid_interp], fill_cols[valid_interp]] = interpolated[valid_interp]

    # Normalized convolution Gaussian smoothing
    if smoothing_sigma_m > 0.0:
        sigma_px = smoothing_sigma_m / grid.pixel_size
        valid_filled = (~np.isnan(filled)).astype(np.float64)
        values = np.nan_to_num(filled, nan=0.0)

        values_smooth = ndimage.gaussian_filter(values, sigma=sigma_px, mode="constant", cval=0.0)
        weights_smooth = ndimage.gaussian_filter(valid_filled, sigma=sigma_px, mode="constant", cval=0.0)

        smoothed = np.full_like(filled, np.nan)
        keep_mask = valid_filled > 0.0
        smoothed[keep_mask] = values_smooth[keep_mask] / np.maximum(weights_smooth[keep_mask], 1e-12)
        return smoothed

    return filled


def _chaikin_coords(
    coords: list[tuple[float, float]],
    iterations: int,
    *,
    closed: bool,
) -> list[tuple[float, float]]:
    """Apply Chaikin's corner-cutting algorithm to open or closed 2D coordinate lists."""
    if iterations <= 0 or len(coords) < 3:
        return coords

    current = [np.asarray(pt, dtype=np.float64) for pt in coords]

    for _ in range(iterations):
        if closed:
            pts = current[:-1] if np.allclose(current[0], current[-1]) else current
            n = len(pts)
            if n < 3:
                break
            refined: list[np.ndarray] = []
            for i in range(n):
                p0 = pts[i]
                p1 = pts[(i + 1) % n]
                refined.append(p0 * 0.75 + p1 * 0.25)
                refined.append(p0 * 0.25 + p1 * 0.75)
            refined.append(refined[0])
            current = refined
        else:
            n = len(current)
            if n < 2:
                break
            refined = [current[0]]
            for i in range(n - 1):
                p0 = current[i]
                p1 = current[i + 1]
                refined.append(p0 * 0.75 + p1 * 0.25)
                refined.append(p0 * 0.25 + p1 * 0.75)
            refined.append(current[-1])
            current = refined

    return [(float(pt[0]), float(pt[1])) for pt in current]


def contour_entities(
    dtm: np.ndarray,
    grid: RasterGrid,
    settings: ContourSettings,
    *,
    exclude: Iterable[Polygon] = (),
    z_offset: float = 0.0,
    z_scale: float = 1.0,
) -> list[dict]:
    """Extract 3D polyline contour entities from a DTM raster.

    ``z_offset`` / ``z_scale`` map local heights to absolute heights
    (``absolute = local / z_scale + z_offset``).  Contour levels are placed on absolute
    multiples of the intervals (so a 1 m contour is a whole absolute metre); the
    polyline vertices keep local z and every entity records its absolute ``elevation``.
    """
    array = np.asarray(dtm, dtype=np.float64) / float(z_scale) + float(z_offset)
    if array.shape != (grid.height, grid.width):
        raise ValueError("dtm shape must match RasterGrid height and width")
    if array.shape[0] < 2 or array.shape[1] < 2:
        return []

    valid_mask = ~np.isnan(array)
    if not valid_mask.any():
        return []

    min_val = float(np.nanmin(array))
    max_val = float(np.nanmax(array))
    if min_val >= max_val:
        return []

    k_min = math.ceil(min_val / settings.minor_interval - 1e-9)
    k_max = math.floor(max_val / settings.minor_interval + 1e-9)
    if k_min > k_max:
        return []

    exclude_polygons = [p for p in exclude if isinstance(p, Polygon) and not p.is_empty]
    exclude_geom = unary_union([make_valid(p) for p in exclude_polygons]) if exclude_polygons else None

    filled_image = np.nan_to_num(array, nan=min_val - 1.0)
    entities: list[dict] = []

    for k in range(k_min, k_max + 1):
        level = round(k * settings.minor_interval, 6)
        if level < min_val or level > max_val:
            continue

        quotient = level / settings.major_interval
        is_major = abs(quotient - round(quotient)) < 1e-6
        layer_role = "major" if is_major else "minor"

        raw_contours = find_contours(filled_image, level=level, mask=valid_mask)

        for contour in raw_contours:
            if len(contour) < 2:
                continue

            local_coords = [grid.pixel_to_local(float(r), float(c)) for r, c in contour]
            is_initially_closed = bool(np.allclose(contour[0], contour[-1]))

            smoothed = _chaikin_coords(
                local_coords,
                settings.chaikin_iterations,
                closed=is_initially_closed,
            )

            geom = LineString(smoothed)
            if settings.simplify_tolerance > 0.0:
                simplified = geom.simplify(settings.simplify_tolerance, preserve_topology=False)
            else:
                simplified = geom

            if simplified.is_empty:
                continue

            if exclude_geom is not None and not exclude_geom.is_empty:
                clipped = simplified.difference(exclude_geom)
            else:
                clipped = simplified

            if clipped.is_empty:
                continue

            parts: list[LineString] = []
            if isinstance(clipped, LineString):
                parts = [clipped]
            elif isinstance(clipped, MultiLineString):
                parts = list(clipped.geoms)
            elif hasattr(clipped, "geoms"):
                for g in clipped.geoms:
                    if isinstance(g, LineString) and not g.is_empty:
                        parts.append(g)

            for part in parts:
                if part.length < settings.min_length:
                    continue

                part_coords = list(part.coords)
                if len(part_coords) < 2:
                    continue

                is_closed = bool(
                    len(part_coords) >= 4
                    and math.dist(part_coords[0], part_coords[-1]) < 1e-6
                )

                local_z = (float(level) - float(z_offset)) * float(z_scale)
                points_3d = [[float(x), float(y), local_z] for x, y in part_coords]

                entities.append({
                    "type": "polyline3d",
                    "points": points_3d,
                    "closed": is_closed,
                    "layer_role": layer_role,
                    "elevation": float(level),
                })

    return entities


def _default_config_path() -> Path:
    direct = Path(__file__).resolve().parents[3] / "config" / "ortho_plan.yaml"
    if direct.exists():
        return direct
    cwd_candidate = Path("config/ortho_plan.yaml")
    return cwd_candidate if cwd_candidate.exists() else direct


def load_contour_settings(path: Path | str | None = None) -> ContourSettings:
    """Load and validate contour settings from ortho_plan.yaml or a custom path."""
    target_path = Path(path) if path is not None else _default_config_path()
    if not target_path.exists():
        return ContourSettings()

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "contours" not in data:
        return ContourSettings()

    section = data["contours"]
    if not isinstance(section, dict):
        return ContourSettings()

    return ContourSettings(
        resolution=float(section.get("resolution", 0.20)),
        major_interval=float(section.get("major_interval", 1.0)),
        minor_interval=float(section.get("minor_interval", 0.25)),
        max_gap=float(section.get("max_gap", 3.0)),
        smoothing_sigma_m=float(section.get("smoothing_sigma_m", 0.3)),
        min_length=float(section.get("min_length", 2.0)),
        simplify_tolerance=float(section.get("simplify_tolerance", 0.03)),
        chaikin_iterations=int(section.get("chaikin_iterations", 2)),
    )


def ground_codes_from_config(path: Path | str | None = None) -> tuple[int, ...]:
    """Read contours.ground_codes from ortho_plan.yaml, falling back to default Cyclone ground codes."""
    target_path = Path(path) if path is not None else _default_config_path()
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            contours_sec = data.get("contours", {})
            if isinstance(contours_sec, dict) and "ground_codes" in contours_sec:
                codes = contours_sec["ground_codes"]
                if isinstance(codes, Iterable):
                    return tuple(int(c) for c in codes)
    return DEFAULT_GROUND_CODES



@dataclass(frozen=True, slots=True)
class ContourLabelSettings:
    """Placement rules for major-contour elevation labels (metres)."""

    text_height_m: float = 1.0
    interval_m: float = 60.0
    min_length_m: float = 20.0
    gap_padding_m: float = 0.4
    char_width_factor: float = 0.62


def format_contour_level(elevation: float) -> str:
    """Whole metres without decimals, otherwise two decimals."""
    return str(int(round(elevation))) if abs(elevation - round(elevation)) < 1e-6 else f"{elevation:.2f}"


def label_major_contours(
    entities: list[dict],
    settings: ContourLabelSettings = ContourLabelSettings(),
    *,
    avoid: object | None = None,
) -> list[dict]:
    """Label major contours along their polyline and open a gap in the line behind each label.

    Each labelled 3D contour is cut into pieces around its labels; a TEXT entity is placed at
    the gap, rotated along the contour tangent (kept readable) and carrying the absolute
    elevation.  Labels never overlap each other or ``avoid`` (a shapely geometry); a label that
    would collide is skipped.  Contours shorter than ``min_length_m`` stay unlabelled.
    """
    from shapely import affinity
    from shapely.geometry import box

    output: list[dict] = []
    placed: list[Polygon] = []
    for entity in entities:
        if entity.get("type") != "polyline3d" or entity.get("layer_role") != "major" or "elevation" not in entity:
            output.append(entity)
            continue
        points = np.asarray(entity["points"], dtype=float)
        closed = bool(entity.get("closed"))
        if closed and np.allclose(points[0], points[-1]):
            points = points[:-1]
        coords = np.vstack((points, points[:1])) if closed else points
        steps = np.hypot(*(np.diff(coords[:, :2], axis=0)).T)
        length = float(steps.sum())
        text = format_contour_level(float(entity["elevation"]))
        height = settings.text_height_m
        half_gap = 0.5 * len(text) * height * settings.char_width_factor + settings.gap_padding_m
        if length < settings.min_length_m or length < 4.0 * half_gap:
            output.append(entity)
            continue
        cumulative = np.concatenate(([0.0], np.cumsum(steps)))
        count = max(1, int(round(length / settings.interval_m)))
        accepted: list[tuple[float, dict, Polygon]] = []
        for index in range(count):
            arc = (index + 0.5) * length / count
            before, after = _point_at(coords, cumulative, max(0.0, arc - half_gap)), _point_at(coords, cumulative, min(length, arc + half_gap))
            centre = _point_at(coords, cumulative, arc)
            tangent = after[:2] - before[:2]
            angle = math.degrees(math.atan2(tangent[1], tangent[0]))
            if angle > 90.0:
                angle -= 180.0
            elif angle <= -90.0:
                angle += 180.0
            shape = affinity.rotate(box(-half_gap + settings.gap_padding_m, -0.6 * height, half_gap - settings.gap_padding_m, 0.6 * height), angle, origin=(0.0, 0.0))
            shape = affinity.translate(shape, float(centre[0]), float(centre[1]))
            if any(shape.buffer(0.5 * height).intersects(other) for other in placed) or (avoid is not None and shape.intersects(avoid)):  # type: ignore[arg-type]
                continue
            placed.append(shape)
            accepted.append((arc, {"type": "text", "text": text, "point": [float(centre[0]), float(centre[1]), float(centre[2])], "height": height, "rotation": angle, "layer_role": "contour_label", "elevation": float(entity["elevation"])}, shape))
        if not accepted:
            output.append(entity)
            continue
        cuts = sorted((arc - half_gap, arc + half_gap) for arc, _text, _shape in accepted)
        base = {key: value for key, value in entity.items() if key not in {"points", "closed"}}
        start = 0.0
        pieces: list[tuple[float, float]] = []
        for low, high in cuts:
            if low - start > 1e-6:
                pieces.append((start, low))
            start = high
        if length - start > 1e-6:
            pieces.append((start, length))
        for low, high in pieces:
            inside = [coords[i] for i in range(len(coords)) if low < cumulative[i] < high]
            piece = np.vstack((_point_at(coords, cumulative, low), *inside, _point_at(coords, cumulative, high))) if inside else np.vstack((_point_at(coords, cumulative, low), _point_at(coords, cumulative, high)))
            if len(piece) >= 2:
                output.append({**base, "points": piece.tolist(), "closed": False})
        output.extend(text_entity for _arc, text_entity, _shape in accepted)
    return output


def _point_at(coords: np.ndarray, cumulative: np.ndarray, arc: float) -> np.ndarray:
    arc = float(min(max(arc, 0.0), cumulative[-1]))
    index = int(np.searchsorted(cumulative, arc, side="right")) - 1
    index = min(max(index, 0), len(coords) - 2)
    span = max(float(cumulative[index + 1] - cumulative[index]), 1e-12)
    fraction = (arc - float(cumulative[index])) / span
    return coords[index] * (1.0 - fraction) + coords[index + 1] * fraction
