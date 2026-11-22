"""Per-room floor / ceiling heights (flat or sloped) and per-wall heights from the classified points (metres)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import LineString, Polygon

from .common import CEILING, FLOOR, STAIRS_WALL, WALL, LevelPoints, StructureSettings, contains, histogram_mode, robust_sigma


@dataclass(slots=True)
class RoomLevel:
    """Measured floor and ceiling of one room (metres above the level floor; plane in metric plan coordinates)."""

    floor_m: float | None = None
    floor_source: str = "level"  # points | level
    ceiling_m: float | None = None  # ceiling height at the room's representative point
    ceiling_plane: tuple[float, float, float] | None = None  # z = a * x + b * y + c (sloped ceilings only)
    ceiling_source: str = "level"  # points | plane | planes | level
    slope_deg: float | None = None
    # Two-plane ceiling (gable / valley / slope + flat): both planes and how they combine (min = ridge, max = valley).
    ceiling_planes: list[tuple[float, float, float]] | None = None
    ceiling_envelope: str = "min"
    noise_m: float | None = None
    ceiling_points: int = 0
    coverage: float = 0.0
    notes: list[str] = field(default_factory=list)

    def ceiling_at(self, xy: np.ndarray, default: float) -> np.ndarray:
        """Ceiling height (metres) at plan positions; ``default`` when the room has no measured ceiling."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        if self.ceiling_planes:
            values = np.stack([a * xy[:, 0] + b * xy[:, 1] + c for a, b, c in self.ceiling_planes])
            return values.min(axis=0) if self.ceiling_envelope == "min" else values.max(axis=0)
        if self.ceiling_plane is not None:
            a, b, c = self.ceiling_plane
            return a * xy[:, 0] + b * xy[:, 1] + c
        value = self.ceiling_m if self.ceiling_m is not None else default
        return np.full(len(xy), float(value))


def _erode(polygon: Polygon, distance: float) -> Polygon:
    shrunk = polygon.buffer(-distance)
    if shrunk.is_empty or shrunk.area < 0.25 * polygon.area:
        return polygon
    if shrunk.geom_type != "Polygon":
        shrunk = max(shrunk.geoms, key=lambda item: item.area)
    return shrunk


def _cell_coverage(polygon: Polygon, xy: np.ndarray, cell: float) -> float:
    x0, y0, x1, y1 = polygon.bounds
    nx, ny = max(1, int(math.ceil((x1 - x0) / cell))), max(1, int(math.ceil((y1 - y0) / cell)))
    gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * cell, y0 + (np.arange(ny) + 0.5) * cell)
    centres = np.column_stack((gx.ravel(), gy.ravel()))
    inside = contains(polygon, centres)
    if not inside.any():
        return 0.0
    if len(xy) == 0:
        return 0.0
    cols = np.clip(((xy[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    rows = np.clip(((xy[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    hit = np.zeros((ny, nx), dtype=bool)
    hit[rows, cols] = True
    return float((hit.ravel() & inside).sum() / inside.sum())


def _ransac_plane(points: np.ndarray, threshold: float, iterations: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """Plane z = a x + b y + c with the most inliers (deterministic), refitted by least squares on its inliers."""
    if len(points) < 10:
        return None
    rng = np.random.default_rng(seed)
    sample = points if len(points) <= 20_000 else points[rng.choice(len(points), 20_000, replace=False)]
    best: tuple[int, np.ndarray] | None = None
    design = np.column_stack((sample[:, 0], sample[:, 1], np.ones(len(sample))))
    for _ in range(iterations):
        index = rng.choice(len(sample), 3, replace=False)
        trio = sample[index]
        matrix = np.column_stack((trio[:, 0], trio[:, 1], np.ones(3)))
        if abs(np.linalg.det(matrix)) < 1e-6:
            continue
        coefficients = np.linalg.solve(matrix, trio[:, 2])
        count = int((np.abs(design @ coefficients - sample[:, 2]) <= threshold).sum())
        if best is None or count > best[0]:
            best = (count, coefficients)
    if best is None:
        return None
    coefficients = best[1]
    full = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
    for _ in range(3):
        inliers = np.abs(full @ coefficients - points[:, 2]) <= threshold
        if inliers.sum() < 3:
            break
        coefficients, *_ = np.linalg.lstsq(full[inliers], points[inliers, 2], rcond=None)
    inliers = np.abs(full @ coefficients - points[:, 2]) <= threshold
    return coefficients, inliers


def _two_planes(
    points: np.ndarray, threshold: float, core: Polygon, first: tuple[np.ndarray, np.ndarray], settings: StructureSettings,
) -> tuple[list[np.ndarray], str, float] | None:
    """Best two-plane model (first RANSAC plane + RANSAC on its outliers), combined as lower (ridge) or upper (valley)
    envelope.  None unless both planes are clearly distinct, at least one is sloped, each owns a coherent part of the room
    and the ridge / valley line crosses the room.  Returns (planes, envelope, explained share)."""
    coefficients, inliers = first
    rest = points[~inliers]
    if len(rest) < max(20, settings.ceiling_two_plane_min_part * len(points)):
        return None
    second = _ransac_plane(rest, threshold, seed=1)
    if second is None:
        return None
    planes = [np.asarray(coefficients, dtype=float), np.asarray(second[0], dtype=float)]
    normals = [np.asarray((-a, -b, 1.0)) / math.sqrt(a * a + b * b + 1.0) for a, b, _ in planes]
    if math.degrees(math.acos(min(1.0, abs(float(normals[0] @ normals[1]))))) < settings.ceiling_two_plane_min_angle_deg:
        return None
    if max(math.degrees(math.atan(math.hypot(a, b))) for a, b, _ in planes) < settings.ceiling_slope_min_deg:
        return None
    design = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
    values = np.stack([design @ plane for plane in planes])
    best: tuple[float, str, np.ndarray] | None = None
    for envelope, surface in (("min", values.min(axis=0)), ("max", values.max(axis=0))):
        share = float((np.abs(surface - points[:, 2]) <= threshold).mean())
        if best is None or share > best[0]:
            best = (share, envelope, values.argmin(axis=0) if envelope == "min" else values.argmax(axis=0))
    assert best is not None
    share, envelope, owner = best
    explained = np.abs(np.choose(owner, values) - points[:, 2]) <= threshold
    parts = [float((explained & (owner == index)).mean()) for index in (0, 1)]
    if min(parts) < settings.ceiling_two_plane_min_part:
        return None
    # the ridge / valley line must split the room into two real parts
    (a0, b0, c0), (a1, b1, c1) = planes
    x0, y0, x1, y1 = core.bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, 25), np.linspace(y0, y1, 25))
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    grid = grid[contains(core, grid)]
    if len(grid) == 0:
        return None
    side = (a0 - a1) * grid[:, 0] + (b0 - b1) * grid[:, 1] + (c0 - c1) <= 0
    if min(side.mean(), 1.0 - side.mean()) < settings.ceiling_two_plane_min_part:
        return None
    # refit each plane on its own explained points
    for index in (0, 1):
        member = explained & (owner == index)
        if member.sum() >= 3:
            planes[index], *_ = np.linalg.lstsq(design[member], points[member, 2], rcond=None)
    order = sorted((0, 1), key=lambda index: -parts[index])
    return [planes[index] for index in order], envelope, share


def room_levels(points: LevelPoints, rooms: list[Polygon], settings: StructureSettings = StructureSettings()) -> list[RoomLevel]:
    """Robust floor (histogram mode of floor points) and ceiling (flat mode or RANSAC plane) of every metric room polygon."""
    storey = float(points.storey_height_m)
    ceiling_all = points.xyz[points.mask((CEILING,)) & (points.xyz[:, 2] >= settings.ceiling_min_clear_m) & (points.xyz[:, 2] <= 1.5 * storey)]
    floor_all = points.xyz[points.mask((FLOOR,)) & (np.abs(points.xyz[:, 2]) <= 0.25 * storey)]
    result: list[RoomLevel] = []
    for room in rooms:
        level = RoomLevel()
        core = _erode(room, settings.room_erode_m)
        floor = floor_all[contains(core, floor_all[:, :2])] if len(floor_all) else floor_all
        if len(floor) >= settings.room_min_points:
            level.floor_m = histogram_mode(floor[:, 2])
            level.floor_source = "points"
        ceiling = ceiling_all[contains(core, ceiling_all[:, :2])] if len(ceiling_all) else ceiling_all
        level.ceiling_points = int(len(ceiling))
        level.coverage = _cell_coverage(core, ceiling[:, :2], settings.room_cell_m) if len(ceiling) else 0.0
        if len(ceiling) < settings.room_min_points or level.coverage < settings.room_min_ceiling_coverage:
            level.notes.append(f"ceiling evidence too weak ({len(ceiling)} points, coverage {level.coverage:.2f}): level ceiling kept")
            result.append(level)
            continue
        mode = histogram_mode(ceiling[:, 2])
        near = ceiling[np.abs(ceiling[:, 2] - mode) <= 0.05, 2]
        sigma = max(robust_sigma(near), 0.003)
        threshold = max(0.02, 3.0 * sigma)
        flat_share = float((np.abs(ceiling[:, 2] - mode) <= threshold).mean())
        level.noise_m = sigma
        plane = _ransac_plane(ceiling, threshold)
        sloped = False
        if plane is not None:
            (a, b, c), inliers = plane
            slope = math.degrees(math.atan(math.hypot(a, b)))
            share = float(inliers.mean())
            x0, y0, x1, y1 = core.bounds
            corners = np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            rise = float(np.ptp(a * corners[:, 0] + b * corners[:, 1] + c))
            if slope >= settings.ceiling_slope_min_deg and share >= settings.ceiling_plane_min_inliers and share >= flat_share + 0.15 and rise >= settings.ceiling_plane_min_rise_m:
                sloped = True
                level.ceiling_plane = (float(a), float(b), float(c))
                level.slope_deg = float(slope)
                level.ceiling_source = "plane"
                point = room.representative_point()
                level.ceiling_m = float(a * point.x + b * point.y + c)
        single_share = max(flat_share, float(plane[1].mean()) if plane is not None and sloped else 0.0)
        double = _two_planes(ceiling, threshold, core, plane, settings) if plane is not None else None
        if double is not None and double[2] >= single_share + settings.ceiling_two_plane_gain and double[2] >= settings.ceiling_two_plane_min_share:
            planes, envelope, share = double
            sloped = True
            level.ceiling_planes = [tuple(float(v) for v in item) for item in planes]
            level.ceiling_envelope = envelope
            level.ceiling_plane = level.ceiling_planes[0]
            level.slope_deg = float(max(math.degrees(math.atan(math.hypot(a, b))) for a, b, _ in planes))
            level.ceiling_source = "planes"
            point = room.representative_point()
            level.ceiling_m = float(level.ceiling_at(np.asarray([[point.x, point.y]]), 0.0)[0])
            level.notes.append(f"two-plane ceiling ({envelope}): explains {share:.2f} of the points vs {single_share:.2f} for one surface")
        if not sloped:
            flat = ceiling[np.abs(ceiling[:, 2] - mode) <= threshold, 2]
            level.ceiling_m = float(np.median(flat)) if len(flat) else float(mode)
            level.ceiling_source = "points"
            level.slope_deg = 0.0
        floor_ref = level.floor_m if level.floor_m is not None else 0.0
        if level.ceiling_m is None or level.ceiling_m - floor_ref < settings.ceiling_min_clear_m:
            level.notes.append("measured ceiling too low above the floor: level ceiling kept")
            level.ceiling_m, level.ceiling_plane, level.ceiling_source, level.slope_deg = None, None, "level", None
            level.ceiling_planes = None
        result.append(level)
    return result


@dataclass(slots=True)
class WallHeight:
    height_m: float | None
    source: str  # room_ceiling | points | ceiling_occluded | adjacent_ceiling | storey
    top_m: float | None = None
    bottom_m: float | None = None
    stations: int = 0
    covered: int = 0
    reference_m: float | None = None


def _sorted_index(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(xyz[:, 0], kind="stable")
    return xyz[order], xyz[order, 0]


def _window(sorted_xyz: np.ndarray, sorted_x: np.ndarray, x0: float, x1: float) -> np.ndarray:
    lo, hi = np.searchsorted(sorted_x, x0, "left"), np.searchsorted(sorted_x, x1, "right")
    return sorted_xyz[lo:hi]


def wall_heights(
    points: LevelPoints, walls: list[tuple[np.ndarray, np.ndarray, float]], rooms: list[Polygon], levels: list[RoomLevel],
    level_ceiling_m: float, settings: StructureSettings = StructureSettings(),
) -> list[WallHeight]:
    """Height of every wall ``(p0, p1, thickness)`` (metres): measured top of its wall points, clipped to the adjacent ceilings."""
    spacing = points.spacing_m((WALL,))
    tolerance = max(0.04, 2.0 * spacing)
    wall_xyz, wall_x = _sorted_index(points.of((WALL, STAIRS_WALL)))
    all_xyz, all_x = _sorted_index(points.xyz[~points.mask((CEILING,))])
    result: list[WallHeight] = []
    for p0, p1, thickness in walls:
        p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
        length = float(np.linalg.norm(p1 - p0))
        if length < 1e-6:
            result.append(WallHeight(None, "storey"))
            continue
        direction = (p1 - p0) / length
        normal = np.asarray((-direction[1], direction[0]))
        half = max(float(thickness), 0.0) / 2.0 + tolerance
        x0, x1 = min(p0[0], p1[0]) - half, max(p0[0], p1[0]) + half
        candidates = _window(wall_xyz, wall_x, x0, x1)
        relative = candidates[:, :2] - p0
        along, offset = relative @ direction, relative @ normal
        member = candidates[(np.abs(offset) <= half) & (along >= 0.0) & (along <= length)]
        member_along = (member[:, :2] - p0) @ direction
        stations = max(1, int(math.ceil(length / settings.wall_bin_m)))
        bins = np.clip((member_along / settings.wall_bin_m).astype(int), 0, stations - 1)
        tops, bottoms = [], []
        for index in range(stations):
            z = member[bins == index, 2]
            if len(z) >= settings.wall_min_points_per_bin:
                tops.append(float(np.percentile(z, settings.wall_top_percentile)))
                bottoms.append(float(np.percentile(z, 100.0 - settings.wall_top_percentile)))
        # adjacent ceilings: rooms touching the wall band, sampled along the wall
        band = LineString([p0, p1]).buffer(float(thickness) / 2.0 + 0.20, cap_style=2)
        samples = np.asarray([p0 + direction * length * t for t in (0.05, 0.5, 0.95)])
        adjacent = [(room, level) for room, level in zip(rooms, levels, strict=True) if room.intersects(band)]
        references = [float(np.max(level.ceiling_at(samples, level_ceiling_m))) for _, level in adjacent]
        reference = max(references) if references else float(level_ceiling_m)
        measured_reference = any(level.ceiling_source != "level" for _, level in adjacent)
        covered = len(tops)
        if covered < max(2, int(math.ceil(settings.wall_min_bin_share * stations))):
            # Too few wall points of its own: the wall still ends at the measured ceiling of the rooms beside it.
            if measured_reference:
                result.append(WallHeight(reference, "adjacent_ceiling", stations=stations, covered=covered, reference_m=reference))
            else:
                result.append(WallHeight(None, "storey", stations=stations, covered=covered, reference_m=reference))
            continue
        top = float(np.percentile(tops, settings.wall_top_quantile))
        bottom = float(np.percentile(bottoms, 25.0))
        entry = WallHeight(reference, "room_ceiling", top_m=top, bottom_m=bottom, stations=stations, covered=covered, reference_m=reference)
        if top < reference - settings.wall_ceiling_tolerance_m:
            # A low top is a half-height wall only when the space above it is seen empty (no points of any class).
            window = _window(all_xyz, all_x, x0, x1)
            rel = window[:, :2] - p0
            inside = (np.abs(rel @ normal) <= half) & (rel @ direction >= 0.0) & (rel @ direction <= length)
            z = window[inside, 2]
            above_lo, above_hi = top + settings.wall_ceiling_tolerance_m, reference - settings.wall_ceiling_tolerance_m
            below_lo = max(0.0, top - 1.0)
            above = int(((z >= above_lo) & (z <= above_hi)).sum()) / max(above_hi - above_lo, 0.05)
            below = int(((z >= below_lo) & (z <= top)).sum()) / max(top - below_lo, 0.05)
            if below > 0 and above / below <= settings.wall_free_above_ratio and top >= settings.wall_min_height_m:
                entry.height_m, entry.source = top, "points"
            else:
                entry.source = "ceiling_occluded"
        result.append(entry)
    return result


def summarise_levels(levels: list[RoomLevel]) -> list[dict[str, Any]]:
    return [
        {
            "floor_m": None if level.floor_m is None else round(level.floor_m, 3), "floor_source": level.floor_source,
            "ceiling_m": None if level.ceiling_m is None else round(level.ceiling_m, 3), "ceiling_source": level.ceiling_source,
            "slope_deg": None if level.slope_deg is None else round(level.slope_deg, 2), "ceiling_points": level.ceiling_points,
            "planes": None if not level.ceiling_planes else [[round(v, 5) for v in plane] for plane in level.ceiling_planes],
            "envelope": level.ceiling_envelope if level.ceiling_planes else None,
            "coverage": round(level.coverage, 3), "notes": list(level.notes),
        }
        for level in levels
    ]
