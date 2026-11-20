"""Canopy-height watershed crown detection and neutral tree inserts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .raster_grid import RasterGrid


@dataclass(frozen=True, slots=True)
class Crown:
    """One measured tree crown in local DXF metres."""

    center: tuple[float, float]
    trunk: tuple[float, float]
    diameter: float
    height: float
    area: float


def detect_crowns(
    chm: np.ndarray,
    tree_mask: np.ndarray,
    grid: RasterGrid,
    *,
    min_height: float = 2.0,
    min_crown_diameter: float = 1.5,
    max_crown_diameter: float = 25.0,
    smoothing_sigma_m: float = 0.3,
    return_labels: bool = False,
) -> list[Crown] | tuple[list[Crown], np.ndarray]:
    """Detect crowns by peak-seeded watershed; oversized regions are re-seeded once.

    A region larger than the maximum diameter is split using half the original
    peak separation before its candidates are measured. This handles fused canopies
    without accepting a single implausibly large crown.

    With ``return_labels=True`` the result is ``(crowns, labels)``: ``labels`` is the
    int32 watershed raster aligned with the CHM grid (0 = background). Labels are not
    renumbered after oversized splits and may be non-contiguous; crowns rejected by
    the diameter limits keep their label in the raster. Find a crown's label by
    sampling ``labels`` at its ``trunk`` pixel (``grid.local_to_pixel``).
    """
    heights = np.asarray(chm, dtype=np.float32)
    trees = np.asarray(tree_mask, dtype=bool)
    if heights.shape != trees.shape or heights.shape != (grid.height, grid.width):
        raise ValueError("CHM and tree_mask must match RasterGrid")
    active = trees & np.isfinite(heights) & (heights >= min_height)
    if not active.any():
        return ([], np.zeros(heights.shape, dtype=np.int32)) if return_labels else []
    smoothed = gaussian_filter(np.where(active, heights, 0.0), sigma=max(0.0, smoothing_sigma_m / grid.pixel_size))
    distance = max(1, int(round(min_crown_diameter / (2.0 * grid.pixel_size))))
    labels = _watershed_labels(smoothed, active, distance, min_height)
    labels = _split_oversized(labels, smoothed, active, grid, max_crown_diameter, distance, min_height)
    crowns: list[Crown] = []
    areas = np.bincount(labels.ravel())
    for identifier, component_slice in enumerate(ndimage.find_objects(labels), start=1):
        if component_slice is None:
            continue
        area = float(areas[identifier]) * grid.pixel_size**2
        diameter = math.sqrt(4.0 * area / math.pi)
        if diameter < min_crown_diameter or diameter > max_crown_diameter:
            continue
        region = labels[component_slice] == identifier
        component_heights = heights[component_slice]
        local_row, local_col = np.unravel_index(np.argmax(np.where(region, component_heights, -np.inf)), region.shape)
        row = local_row + component_slice[0].start
        col = local_col + component_slice[1].start
        rows, cols = np.nonzero(region)
        rows = rows + component_slice[0].start
        cols = cols + component_slice[1].start
        center = grid.pixel_to_local(float(rows.mean()), float(cols.mean()))
        trunk = grid.pixel_to_local(float(row), float(col))
        crowns.append(Crown(center, trunk, diameter, float(heights[row, col]), area))
    if return_labels:
        return crowns, labels.astype(np.int32, copy=False)
    return crowns


def tree_entities(crowns: list[Crown]) -> list[dict[str, object]]:
    """Return one unit-diameter TREE insert per crown, scaled in metres."""
    return [
        {
            "type": "insert",
            "name": "TREE",
            "point": [float(crown.trunk[0]), float(crown.trunk[1])],
            "rotation": 0.0,
            "xscale": float(crown.diameter),
            "yscale": float(crown.diameter),
            "attributes": {"LABEL": f"Tree d={crown.diameter:.1f} h={crown.height:.1f}"},
        }
        for crown in crowns
    ]


def _watershed_labels(image: np.ndarray, mask: np.ndarray, distance: int, min_height: float) -> np.ndarray:
    peaks = peak_local_max(image, min_distance=distance, threshold_abs=min_height, labels=mask, exclude_border=False)
    markers = np.zeros(image.shape, dtype=np.int32)
    for index, (row, col) in enumerate(peaks, start=1):
        markers[row, col] = index
    if not len(peaks):
        row, col = np.unravel_index(np.argmax(np.where(mask, image, -np.inf)), image.shape)
        markers[row, col] = 1
    return watershed(-image, markers=markers, mask=mask)


def _split_oversized(
    labels: np.ndarray, image: np.ndarray, mask: np.ndarray, grid: RasterGrid, maximum: float, distance: int, min_height: float
) -> np.ndarray:
    result = labels.copy()
    next_identifier = int(result.max()) + 1
    areas = np.bincount(labels.ravel())
    for identifier, component_slice in enumerate(ndimage.find_objects(labels), start=1):
        if component_slice is None:
            continue
        diameter = math.sqrt(4.0 * areas[identifier] * grid.pixel_size**2 / math.pi)
        if diameter <= maximum:
            continue
        region = labels[component_slice] == identifier
        local = _watershed_labels(image[component_slice], region & mask[component_slice], max(1, distance // 2), min_height)
        local_ids = [value for value in np.unique(local) if value > 0]
        if len(local_ids) <= 1:
            continue
        result_component = result[component_slice]
        result_component[region] = 0
        for local_id in local_ids:
            result_component[local == local_id] = next_identifier
            next_identifier += 1
    return result


# ---------------------------------------------------------------------------
# Crown / trunk annotations (surveyor-style tree labels)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrunkSettings:
    """Trunk cylinder search and circle-fit thresholds (metres)."""

    search_radius_m: float = 1.0
    z_min_m: float = 0.4
    z_max_m: float = 1.8
    inlier_tolerance_m: float = 0.02
    min_points: int = 30
    min_inliers: int = 25
    min_inlier_fraction: float = 0.35
    min_coverage_deg: float = 120.0
    min_diameter_m: float = 0.05
    max_diameter_m: float = 1.5
    ransac_iterations: int = 400
    text_height_m: float = 0.5


def fit_trunk_diameter(
    xy: np.ndarray,
    approximate_centre: tuple[float, float],
    settings: TrunkSettings = TrunkSettings(),
) -> float | None:
    """Fit a trunk circle to horizontal-slice points; None unless a real cylinder is observed.

    RANSAC proposes circles from three points, the best consensus is refined by an
    algebraic (Kasa) least-squares fit on its inliers.  The result needs enough inliers,
    a large enough inlier fraction, an angular coverage of at least ``min_coverage_deg``
    around the fitted centre (partial arcs are rejected), a centre within the search
    radius of ``approximate_centre`` and a diameter between 0.05 and 1.5 m.
    """
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(points) < settings.min_points:
        return None
    rng = np.random.default_rng(7)
    best: tuple[int, np.ndarray] | None = None
    tolerance = settings.inlier_tolerance_m
    for _ in range(settings.ransac_iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        circle = _circle_from_three(sample)
        if circle is None:
            continue
        cx, cy, radius = circle
        if not settings.min_diameter_m / 2.0 <= radius <= settings.max_diameter_m / 2.0:
            continue
        inliers = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius) <= tolerance
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, inliers)
    if best is None:
        return None
    inliers = best[1]
    circle = None
    for _ in range(3):
        circle = _kasa_fit(points[inliers])
        if circle is None:
            return None
        cx, cy, radius = circle
        inliers = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius) <= tolerance
    assert circle is not None
    cx, cy, radius = circle
    count = int(inliers.sum())
    if count < settings.min_inliers or count < settings.min_inlier_fraction * len(points):
        return None
    if not settings.min_diameter_m / 2.0 <= radius <= settings.max_diameter_m / 2.0:
        return None
    if math.hypot(cx - approximate_centre[0], cy - approximate_centre[1]) > settings.search_radius_m:
        return None
    angles = np.sort(np.arctan2(points[inliers, 1] - cy, points[inliers, 0] - cx))
    gaps = np.diff(np.concatenate((angles, [angles[0] + 2.0 * math.pi])))
    coverage = 360.0 - math.degrees(float(gaps.max()))
    if coverage < settings.min_coverage_deg:
        return None
    return float(2.0 * radius)


def _circle_from_three(points: np.ndarray) -> tuple[float, float, float] | None:
    (ax, ay), (bx, by), (cx, cy) = points
    determinant = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(determinant) < 1e-9:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / determinant
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / determinant
    return float(ux), float(uy), float(math.hypot(ax - ux, ay - uy))


def _kasa_fit(points: np.ndarray) -> tuple[float, float, float] | None:
    if len(points) < 3:
        return None
    x, y = points[:, 0], points[:, 1]
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(points))))
    solution, *_ = np.linalg.lstsq(matrix, x * x + y * y, rcond=None)
    cx, cy, constant = solution
    squared = constant + cx * cx + cy * cy
    if squared <= 0.0:
        return None
    return float(cx), float(cy), float(math.sqrt(squared))


class TrunkCollector:
    """Collect tree-class points inside each trunk cylinder while a point cloud streams past.

    The cylinder is centred on the trunk (xy) with ``search_radius_m`` and spans
    ``z_min_m`` .. ``z_max_m`` above the local ground height of that tree.
    """

    def __init__(self, trunks_xy: np.ndarray, ground_z: np.ndarray, settings: TrunkSettings = TrunkSettings()) -> None:
        from scipy.spatial import cKDTree

        self._xy = np.asarray(trunks_xy, dtype=float).reshape(-1, 2)
        self._ground = np.asarray(ground_z, dtype=float).reshape(-1)
        self._settings = settings
        self._tree = cKDTree(self._xy) if len(self._xy) else None
        self._points: list[list[np.ndarray]] = [[] for _ in range(len(self._xy))]

    def add(self, xyz_local: np.ndarray, classes: np.ndarray, tree_codes: tuple[int, ...]) -> None:
        if self._tree is None:
            return
        xyz = np.asarray(xyz_local, dtype=float)
        keep = np.isin(np.asarray(classes), tree_codes)
        if not keep.any():
            return
        xyz = xyz[keep]
        distance, owner = self._tree.query(xyz[:, :2], distance_upper_bound=self._settings.search_radius_m)
        inside = np.isfinite(distance)
        if not inside.any():
            return
        xyz, owner = xyz[inside], owner[inside]
        height = xyz[:, 2] - self._ground[owner]
        band = np.isfinite(height) & (height >= self._settings.z_min_m) & (height <= self._settings.z_max_m)
        for index in np.unique(owner[band]):
            self._points[int(index)].append(xyz[band][owner[band] == index][:, :2])

    def diameters(self) -> list[float | None]:
        """Fitted trunk diameter per tree, None where no cylinder was observed."""
        output: list[float | None] = []
        for index, chunks in enumerate(self._points):
            if not chunks:
                output.append(None)
                continue
            output.append(fit_trunk_diameter(np.vstack(chunks), (float(self._xy[index, 0]), float(self._xy[index, 1])), self._settings))
        return output


def tree_annotation_entities(
    crowns: list[Crown],
    trunk_diameters: list[float | None] | None = None,
    *,
    text_height: float = 0.5,
) -> list[dict[str, object]]:
    """Crown diameter / top height text ("DK 4.8 m  H 6.5 m") and, only when measured, the trunk ("St. Ø 0.35 m")."""
    entities: list[dict[str, object]] = []
    for index, crown in enumerate(crowns):
        x = crown.trunk[0] + crown.diameter / 2.0 + 0.3
        y = crown.trunk[1]
        entities.append({"type": "text", "text": f"DK {crown.diameter:.1f} m  H {crown.height:.1f} m", "point": [float(x), float(y + 0.6 * text_height)], "height": text_height, "rotation": 0.0, "align": "MIDDLE_LEFT", "layer_role": "tree_label"})
        diameter = trunk_diameters[index] if trunk_diameters is not None and index < len(trunk_diameters) else None
        if diameter is not None:
            entities.append({"type": "text", "text": f"St. Ø {diameter:.2f} m", "point": [float(x), float(y - 0.6 * text_height)], "height": text_height, "rotation": 0.0, "align": "MIDDLE_LEFT", "layer_role": "trunk_label"})
    return entities
