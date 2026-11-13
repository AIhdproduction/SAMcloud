"""Built-in fronts: a measured wall line that is really the face of a tall built-in unit.

A floor-to-ceiling wardrobe, shelf or cabinet that stands against a wall hides
that wall.  The scanner sees (and the ceiling ends at) the unit's front, so the
wall evidence lands on the front instead of on the real wall behind it.  This
module recognises such fronts from general evidence only (no project
coordinates):

* a parallel wall line, ceiling edge or room-outline boundary 0.25-1.0 m behind
  the front, next to the unit;
* the strip between the two lines is filled with tall furniture (fine label
  wardrobe / shelf / bookcase / kitchen cabinet / refrigerator, class 117/120
  with a top height of at least 1.6 m, or wall-class pixels beside such a fine
  label) or is otherwise not open floor (the volume behind a tall front is never
  observed, so it is "filled" as long as no free floor was seen there);
* the strip spans at least 0.6 m along the wall.

The wall then moves to the back line, the strip becomes one A-EINB rectangle
and the furniture outlines inside it are absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .raster_grid import RasterGrid

TALL_FINE_LABELS = frozenset({"wardrobe", "shelf", "bookcase", "refrigerator", "fridge"})
TALL_IF_HIGH_FINE_LABELS = frozenset({"kitchen cabinet"})
TALL_CLASS_CODES = (117, 120)
BUILTIN_LABELS: dict[str, str] = {
    "wardrobe": "Schrank", "shelf": "Regal", "bookcase": "Bücherregal",
    "kitchen cabinet": "Küche", "refrigerator": "Küche", "fridge": "Küche",
}
ABSORBED_DISPLAY_NAMES = frozenset({"Schrank", "Regal", "Bücherregal", "Küche", "Kühlschrank", "Einbau"})
EINB_ID_BASE = 1_000_000


@dataclass(frozen=True, slots=True)
class BuiltinFrontSettings:
    min_depth_m: float = 0.25
    max_depth_m: float = 1.0
    min_length_m: float = 0.6
    fill_fraction: float = 0.5
    tall_height_m: float = 1.6
    angle_tolerance_deg: float = 4.0
    offset_tolerance_m: float = 0.10
    bridge_gap_m: float = 1.0
    adjacency_m: float = 0.3
    front_margin_m: float = 0.12
    front_reach_m: float = 0.40
    run_gap_m: float = 0.30
    sample_step_m: float = 0.05
    edge_min_length_m: float = 0.30
    back_support_m: float = 0.35
    overshoot_m: float = 0.30
    side_wall_reach_m: float = 0.20
    room_free_fraction: float = 0.5
    room_reach_m: float = 0.5
    run_end_fraction: float = 0.5


@dataclass(slots=True)
class _Segment:
    line: LineString
    source: str
    index: int | None
    angle: float
    offset: float = 0.0


@dataclass(slots=True)
class _Chain:
    angle: float
    unit: np.ndarray
    normal: np.ndarray
    offset: float
    start: float
    end: float
    members: list[_Segment]

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def measured_extent(self) -> tuple[float, float]:
        """Extent along the chain that measured wall lines cover (the whole chain when it has none).

        A collinear edge of the ceiling-derived room outline beyond a unit is not
        evidence that the front carries on as a wall; only measured walls are.
        """
        spans = [
            (min(float(self.unit @ np.asarray(point)) for point in (member.line.coords[0], member.line.coords[-1])),
             max(float(self.unit @ np.asarray(point)) for point in (member.line.coords[0], member.line.coords[-1])))
            for member in self.members if member.source == "wall"
        ]
        if not spans:
            return self.start, self.end
        return min(span[0] for span in spans), max(span[1] for span in spans)


def tall_furniture_mask(
    fine_classes: np.ndarray, fine_names: list[str], classes: np.ndarray, height_map: np.ndarray,
    structural_codes: Iterable[int], grid: RasterGrid, units_per_metre: float, settings: BuiltinFrontSettings = BuiltinFrontSettings(),
) -> np.ndarray:
    """Pixels that belong to tall furniture (see the module docstring)."""
    names = [str(name).strip().lower() for name in fine_names]
    tall_ids = [index + 1 for index, name in enumerate(names) if name in TALL_FINE_LABELS]
    high_ids = [index + 1 for index, name in enumerate(names) if name in TALL_IF_HIGH_FINE_LABELS]
    high = np.asarray(height_map) >= settings.tall_height_m * units_per_metre
    label_tall = np.isin(fine_classes, tall_ids) | (np.isin(fine_classes, high_ids) & high)
    class_tall = np.isin(classes, TALL_CLASS_CODES) & high
    radius = max(1, int(round(0.10 * units_per_metre / grid.pixel_size)))
    near = cv2.dilate(label_tall.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0
    wall_tall = np.isin(classes, tuple(structural_codes)) & near
    return label_tall | class_tall | wall_tall


def resolve_builtin_fronts(
    layers: dict[str, list[dict[str, Any]]],
    room_polygons: Iterable[Polygon],
    *,
    tall: np.ndarray,
    free_floor: np.ndarray,
    fine_classes: np.ndarray,
    fine_names: list[str],
    grid: RasterGrid,
    units_per_metre: float,
    settings: BuiltinFrontSettings = BuiltinFrontSettings(),
    exclude: Iterable[Polygon] = (),
) -> dict[str, Any]:
    """Move walls behind built-in fronts and turn each strip into one A-EINB rectangle.

    ``layers`` (A-WALL, A-EINB, A-FURN, A-FURN-DETL, A-ANNO) is modified in place.
    ``exclude`` lists strips that another resolver (recessed niches) already handled.
    """
    upm = max(float(units_per_metre), 1e-9)
    stats: dict[str, Any] = {"builtin_fronts": 0, "builtin_walls_added": 0, "builtin_walls_removed": 0, "builtin_furniture_absorbed": 0, "builtin_einb_absorbed": 0, "builtin_strips": []}
    walls = layers.setdefault("A-WALL", [])
    segments = _collect_segments(walls, room_polygons, upm, settings)
    chains = _build_chains(segments, upm, settings)
    accepted: list[Polygon] = list(exclude)
    free = np.asarray(free_floor, dtype=bool)
    tall_mask = np.asarray(tall, dtype=bool)
    expand = cv2.dilate(tall_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    for front in sorted(chains, key=lambda chain: -chain.length):
        if front.length < settings.min_length_m * upm or not any(member.source == "wall" or member.source == "room" for member in front.members):
            continue
        for side in (1.0, -1.0):
            candidates = _back_candidates(front, chains, side, upm, settings)
            if not candidates:
                continue
            runs = _tall_runs(front, side, expand, grid, upm, settings)
            if not runs:
                continue
            done = False
            for back, depth in candidates:
                for run in runs:
                    if _extent_gap(front, back, run) > settings.adjacency_m * upm:
                        continue
                    # A front that carries on beyond the unit is a real wall (the unit sits in a recess behind it).
                    measured_start, measured_end = front.measured_extent
                    if run[0] - measured_start > settings.overshoot_m * upm or measured_end - run[1] > settings.overshoot_m * upm:
                        continue
                    run = _extend_run_to_side_walls(layers.get("A-WALL", []), front, side, run, depth, upm, settings)
                    strip = _strip_polygon(front, side, run, depth)
                    if any(strip.intersection(other).area > 0.30 * strip.area for other in accepted):
                        continue
                    fill = _fill_fractions(strip, tall_mask, free, grid)
                    if fill["fill"] < settings.fill_fraction or fill["free"] > 1.0 - settings.fill_fraction:
                        continue
                    # The room (open floor) lies in front of the unit, never behind it.
                    room = _fill_fractions(_strip_polygon(front, -side, run, settings.room_reach_m * upm), tall_mask, free, grid)
                    if room["free"] < settings.room_free_fraction or room["free"] < fill["free"] + 0.30:
                        continue
                    fill["room_free"] = room["free"]
                    _apply_strip(layers, stats, front, back, side, run, depth, strip, fine_classes, fine_names, grid, upm, settings)
                    accepted.append(strip)
                    stats["builtin_fronts"] += 1
                    stats["builtin_strips"].append({"depth_m": depth / upm, "length_m": (run[1] - run[0]) / upm, **fill, "bounds": [float(value) for value in strip.bounds]})
                    done = True
                    break
                if done:
                    break
            if done:
                break
    return stats


# -- candidate lines ---------------------------------------------------------

def _angle_of(line: LineString) -> float:
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    return math.atan2(y1 - y0, x1 - x0) % math.pi


def _angle_gap(first: float, second: float) -> float:
    return abs((first - second + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _collect_segments(walls: list[dict[str, Any]], rooms: Iterable[Polygon], upm: float, settings: BuiltinFrontSettings) -> list[_Segment]:
    segments: list[_Segment] = []
    for index, entity in enumerate(walls):
        points = entity.get("points", [])
        if entity.get("type") != "polyline" or len(points) != 2 or entity.get("closed"):
            continue
        line = LineString(points)
        if line.length > 1e-9:
            segments.append(_Segment(line, "wall", index, _angle_of(line)))
    for room in rooms:
        coordinates = list(room.exterior.coords)
        for first, second in zip(coordinates, coordinates[1:], strict=False):
            line = LineString((first, second))
            if line.length >= settings.edge_min_length_m * upm:
                segments.append(_Segment(line, "room", None, _angle_of(line)))
    return segments


def _build_chains(segments: list[_Segment], upm: float, settings: BuiltinFrontSettings) -> list[_Chain]:
    count = len(segments)
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    angle_limit = math.radians(settings.angle_tolerance_deg)
    offset_limit = settings.offset_tolerance_m * upm
    for first in range(count):
        for second in range(first + 1, count):
            a, b = segments[first], segments[second]
            if _angle_gap(a.angle, b.angle) > angle_limit:
                continue
            if _line_distance(a.line, b.line.interpolate(0.5, normalized=True)) <= offset_limit and _line_distance(b.line, a.line.interpolate(0.5, normalized=True)) <= offset_limit:
                parent[find(second)] = find(first)
    groups: dict[int, list[_Segment]] = {}
    for index, segment in enumerate(segments):
        groups.setdefault(find(index), []).append(segment)
    chains: list[_Chain] = []
    for members in groups.values():
        weights = np.asarray([member.line.length for member in members])
        reference = members[int(np.argmax(weights))].angle
        vectors = np.asarray([[math.cos(2.0 * (member.angle - reference)), math.sin(2.0 * (member.angle - reference))] for member in members])
        angle = (reference + 0.5 * math.atan2(float((vectors[:, 1] * weights).sum()), float((vectors[:, 0] * weights).sum()))) % math.pi
        unit = np.asarray((math.cos(angle), math.sin(angle)))
        normal = np.asarray((-unit[1], unit[0]))
        offsets = np.asarray([float(normal @ np.asarray(member.line.interpolate(0.5, normalized=True).coords[0])) for member in members])
        offset = float((offsets * weights).sum() / weights.sum())
        spans = []
        for member in members:
            projections = [float(unit @ np.asarray(point)) for point in (member.line.coords[0], member.line.coords[-1])]
            spans.append((min(projections), max(projections), member))
        spans.sort(key=lambda item: (item[0], item[1]))
        current: list[Any] | None = None
        for start, end, member in spans:
            if current is not None and start - current[1] > settings.bridge_gap_m * upm:
                chains.append(_Chain(angle, unit, normal, offset, current[0], current[1], current[2]))
                current = None
            if current is None:
                current = [start, end, [member]]
            else:
                current[1] = max(current[1], end)
                current[2].append(member)
        if current is not None:
            chains.append(_Chain(angle, unit, normal, offset, current[0], current[1], current[2]))
    return chains


def _line_distance(line: LineString, point: Point) -> float:
    """Distance from ``point`` to the infinite line through ``line``."""
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 1e-12:
        return float("inf")
    return abs((x1 - x0) * (y0 - point.y) - (x0 - point.x) * (y1 - y0)) / length


def _back_candidates(front: _Chain, chains: list[_Chain], side: float, upm: float, settings: BuiltinFrontSettings) -> list[tuple[_Chain, float]]:
    """Parallel chains 0.25-1.0 m behind ``front`` on the given side, nearest first."""
    result: list[tuple[_Chain, float]] = []
    limit = math.radians(settings.angle_tolerance_deg)
    for other in chains:
        if other is front or _angle_gap(other.angle, front.angle) > limit:
            continue
        points = [np.asarray(point) for member in other.members for point in (member.line.coords[0], member.line.coords[-1])]
        depth = side * (float(np.mean([front.normal @ point for point in points])) - front.offset)
        support = sum(member.line.length for member in other.members)
        if settings.min_depth_m * upm <= depth <= settings.max_depth_m * upm and support >= settings.back_support_m * upm:
            result.append((other, depth))
    return sorted(result, key=lambda item: item[1])


def _extent_gap(front: _Chain, back: _Chain, run: tuple[float, float]) -> float:
    """Gap along the front's axis between the back chain and a strip run (0 when they overlap)."""
    projections = [float(front.unit @ np.asarray(point)) for member in back.members for point in (member.line.coords[0], member.line.coords[-1])]
    return max(0.0, min(projections) - run[1], run[0] - max(projections))


# -- strip evidence ----------------------------------------------------------

def _frame_point(chain: _Chain, side: float, along: float | np.ndarray, across: float | np.ndarray) -> np.ndarray:
    along = np.asarray(along, dtype=float)
    across = np.asarray(across, dtype=float)
    return chain.unit[None, :] * np.atleast_1d(along)[:, None] + chain.normal[None, :] * (chain.offset + side * np.atleast_1d(across))[:, None]


def _tall_runs(front: _Chain, side: float, tall: np.ndarray, grid: RasterGrid, upm: float, settings: BuiltinFrontSettings) -> list[tuple[float, float]]:
    """Stretches of the front with tall furniture right behind (or at) it."""
    step = settings.sample_step_m * upm
    samples = np.arange(front.start, front.end + 1e-9, step)
    if len(samples) < 2:
        return []
    depths = np.arange(-settings.front_margin_m * upm, settings.front_reach_m * upm + 1e-9, grid.pixel_size)
    along = np.repeat(samples, len(depths))
    across = np.tile(depths, len(samples))
    points = _frame_point(front, side, along, across)
    rows = np.rint((grid.max_y - points[:, 1]) / grid.pixel_size - 0.5).astype(int)
    cols = np.rint((points[:, 0] - grid.min_x) / grid.pixel_size - 0.5).astype(int)
    valid = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    hit = np.zeros(len(points), dtype=bool)
    hit[valid] = tall[rows[valid], cols[valid]]
    share = hit.reshape(len(samples), len(depths)).mean(axis=1)
    present = share > 0.0
    gap_samples = max(1, int(round(settings.run_gap_m * upm / step)))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last = -10**9
    for index, flag in enumerate(present):
        if flag:
            if start is None:
                start = index
            last = index
        elif start is not None and index - last > gap_samples:
            runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    trimmed: list[tuple[float, float]] = []
    for first, last_index in runs:
        # A shelf or cat tree standing beside the unit adds sparse tall columns at the run's ends: trim the ends
        # back to columns that are at least half as full as the run's typical column.
        limit = settings.run_end_fraction * float(np.median(share[first:last_index + 1][present[first:last_index + 1]]))
        while first < last_index and share[first] < limit:
            first += 1
        while last_index > first and share[last_index] < limit:
            last_index -= 1
        if (samples[last_index] - samples[first]) >= settings.min_length_m * upm:
            trimmed.append((float(samples[first]), float(samples[last_index])))
    return trimmed


def _extend_run_to_side_walls(walls: list[dict[str, Any]], front: _Chain, side: float, run: tuple[float, float], depth: float, upm: float, settings: BuiltinFrontSettings) -> tuple[float, float]:
    """Stretch the run to the perpendicular walls that close the unit at its ends."""
    reach = settings.side_wall_reach_m * upm
    start, stop = run
    best_start, best_stop = start, stop
    for entity in walls:
        points = entity.get("points", [])
        if entity.get("type") != "polyline" or len(points) != 2 or entity.get("closed"):
            continue
        if _angle_gap(_angle_of(LineString(points)), front.angle) < math.radians(90.0 - 8.0):
            continue
        for point in points:
            vector = np.asarray(point, dtype=float)
            along, across = float(front.unit @ vector), side * (float(front.normal @ vector) - front.offset)
            if not -0.15 * upm <= across <= depth + 0.15 * upm:
                continue
            if start - reach <= along < best_start:
                best_start = along
            if best_stop < along <= stop + reach:
                best_stop = along
    return best_start, best_stop


def _strip_polygon(front: _Chain, side: float, run: tuple[float, float], depth: float) -> Polygon:
    corners = _frame_point(front, side, [run[0], run[1], run[1], run[0]], [0.0, 0.0, depth, depth])
    return Polygon([tuple(point) for point in corners])


def _polygon_mask(polygon: Polygon, grid: RasterGrid) -> np.ndarray:
    output = np.zeros((grid.height, grid.width), dtype=np.uint8)
    points = np.asarray([((x - grid.min_x) / grid.pixel_size - 0.5, (grid.max_y - y) / grid.pixel_size - 0.5) for x, y in polygon.exterior.coords], dtype=np.float64)
    cv2.fillPoly(output, [np.rint(points).astype(np.int32)], 1)
    return output.astype(bool)


def _fill_fractions(strip: Polygon, tall: np.ndarray, free_floor: np.ndarray, grid: RasterGrid) -> dict[str, float]:
    mask = _polygon_mask(strip, grid)
    if not mask.any():
        return {"fill": 0.0, "tall": 0.0, "free": 1.0}
    return {
        "fill": float(np.mean(tall[mask] | ~free_floor[mask])),
        "tall": float(np.mean(tall[mask])),
        "free": float(np.mean(free_floor[mask] & ~tall[mask])),
    }


# -- applying a strip --------------------------------------------------------

def _apply_strip(
    layers: dict[str, list[dict[str, Any]]], stats: dict[str, Any], front: _Chain, back: _Chain | None, side: float, run: tuple[float, float],
    depth: float, strip: Polygon, fine_classes: np.ndarray, fine_names: list[str], grid: RasterGrid, upm: float, settings: BuiltinFrontSettings,
    *, kind: str = "builtin_front", label: str | None = None, identifier: int | None = None,
) -> None:
    """Rewrite the walls around ``strip``.  ``back`` is None for a niche whose back line is ``depth`` itself."""
    tolerance = 0.06 * upm
    connect = 0.15 * upm
    walls = layers["A-WALL"]
    unit, normal = front.unit, front.normal

    def frame(point: Iterable[float]) -> tuple[float, float]:
        vector = np.asarray(tuple(point), dtype=float)
        return float(unit @ vector), side * (float(normal @ vector) - front.offset)

    def world(along: float, across: float) -> list[float]:
        return [float(value) for value in _frame_point(front, side, along, across)[0]]

    kept: list[dict[str, Any]] = []
    extensions: list[tuple[dict[str, Any], int, list[float]]] = []
    back_intervals: list[tuple[float, float]] = []
    removed = 0
    for entity in walls:
        points = entity.get("points", [])
        if entity.get("type") != "polyline" or len(points) != 2 or entity.get("closed"):
            kept.append(entity)
            continue
        (s0, t0), (s1, t1) = frame(points[0]), frame(points[1])
        direction = LineString(points)
        angle_to_front = _angle_gap(_angle_of(direction), front.angle)
        parallel = angle_to_front <= math.radians(settings.angle_tolerance_deg)
        perpendicular = angle_to_front >= math.radians(90.0 - 8.0)
        if parallel and abs(t0) <= tolerance + 0.04 * upm and abs(t1) <= tolerance + 0.04 * upm:
            low, high = sorted((s0, s1))
            if not (high > run[0] and low < run[1]):
                kept.append(entity)
                continue
            removed += 1
            level = (t0 + t1) / 2.0
            for a, b in ((low, min(high, run[0] - tolerance)), (max(low, run[1] + tolerance), high)):
                if b - a > 0.05 * upm:
                    clone = dict(entity)
                    clone["points"] = [world(a, level), world(b, level)]
                    kept.append(clone)
            continue
        if parallel and abs(t0 - depth) <= tolerance + 0.04 * upm and abs(t1 - depth) <= tolerance + 0.04 * upm:
            back_intervals.append((min(s0, s1), max(s0, s1)))
            kept.append(entity)
            continue
        inside = all(run[0] - tolerance <= s <= run[1] + tolerance and -tolerance <= t <= depth + tolerance for s, t in ((s0, t0), (s1, t1)))
        if inside:
            removed += 1
            continue
        if perpendicular:
            # A side wall that stops at the front continues to the real wall.
            for end, (s, t) in enumerate(((s0, t0), (s1, t1))):
                other_t = t1 if end == 0 else t0
                near_end = min(abs(s - run[0]), abs(s - run[1])) <= connect
                if near_end and abs(t) <= connect and other_t < t - 0.05 * upm:
                    extensions.append((entity, end, world(s, depth)))
                    break
        kept.append(entity)
    for entity, end, target in extensions:
        entity["points"] = [list(map(float, point)) for point in entity["points"]]
        entity["points"][end] = target
    # Add the real wall along the back line, only where no wall exists yet.
    start, stop = run
    for a, b in sorted(back_intervals):
        if 0.0 <= start - b <= settings.adjacency_m * upm:
            start = min(start, b)
        if 0.0 <= a - stop <= settings.adjacency_m * upm:
            stop = max(stop, a)
    covered = sorted((max(a, start), min(b, stop)) for a, b in back_intervals if b > start and a < stop)
    cursor = start
    missing: list[tuple[float, float]] = []
    for a, b in covered:
        if a - cursor > 0.05 * upm:
            missing.append((cursor, a))
        cursor = max(cursor, b)
    if stop - cursor > 0.05 * upm:
        missing.append((cursor, stop))
    if back is None:
        back_frame_t = depth
    else:
        back_points = np.asarray([point for member in back.members for point in (member.line.coords[0], member.line.coords[-1])], dtype=float)
        back_frame_t = float(np.mean([side * (float(normal @ point) - front.offset) for point in back_points]))
    for a, b in missing:
        kept.append({"type": "polyline", "points": [world(a, back_frame_t), world(b, back_frame_t)], "closed": False, "_rms": 0.0, "_max_residual": 0.0, "_axis_snapped": True, "_source": "builtin_front"})
    layers["A-WALL"] = kept
    stats["builtin_walls_added"] += len(missing)
    stats["builtin_walls_removed"] += removed

    label = label if label is not None else _strip_label(strip, fine_classes, fine_names, grid)
    reach = strip.union(_strip_polygon(front, side, run, -settings.front_reach_m * upm * 0.75))
    reach_polygon = reach if isinstance(reach, Polygon) else strip.convex_hull
    layers["A-EINB"], absorbed_einb = _absorb_einb(layers.get("A-EINB", []), strip)
    stats["builtin_einb_absorbed"] += absorbed_einb
    identifier = identifier if identifier is not None else EINB_ID_BASE + int(stats["builtin_fronts"])
    layers["A-EINB"].append({"type": "polyline", "points": [[float(x), float(y)] for x, y in strip.exterior.coords[:-1]], "closed": True, "holes": [], "_einb_kind": kind, "_einb_id": identifier})
    layers["A-EINB"].append({"type": "text", "_einb_owner": identifier, "text": label, "point": list(strip.representative_point().coords[0]), "height": 0.10 * upm, "rotation": 0.0})
    absorbed = _absorb_furniture(layers, strip, reach_polygon)
    stats["builtin_furniture_absorbed"] += absorbed


def _strip_label(strip: Polygon, fine_classes: np.ndarray, fine_names: list[str], grid: RasterGrid) -> str:
    grown = strip.buffer(0.05)
    mask = _polygon_mask(grown, grid)
    counts: dict[str, int] = {}
    for index, name in enumerate(fine_names):
        key = str(name).strip().lower()
        if key in BUILTIN_LABELS:
            counts[key] = counts.get(key, 0) + int(np.count_nonzero(mask & (fine_classes == index + 1)))
    if not counts or max(counts.values()) == 0:
        return "Einbau"
    return BUILTIN_LABELS[max(counts, key=lambda key: counts[key])]


def _absorb_einb(entities: list[dict[str, Any]], strip: Polygon) -> tuple[list[dict[str, Any]], int]:
    drop_ids: set[Any] = set()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for entity in entities:
        if entity.get("type") != "polyline":
            kept.append(entity)
            continue
        points = entity.get("points", [])
        if entity.get("closed") and len(points) >= 3:
            polygon = Polygon(points).buffer(0)
            if polygon.area > 0.0 and polygon.intersection(strip).area >= 0.5 * polygon.area:
                dropped += 1
                if entity.get("_einb_id") is not None:
                    drop_ids.add(entity["_einb_id"])
                continue
        elif len(points) >= 2:
            line = LineString(points)
            if line.length > 0.0 and line.intersection(strip.buffer(0.03)).length >= 0.8 * line.length:
                dropped += 1
                continue
        kept.append(entity)
    kept = [entity for entity in kept if not (entity.get("type") == "text" and entity.get("_einb_owner") in drop_ids)]
    inside_texts = [entity for entity in kept if entity.get("type") == "text" and entity.get("_einb_owner") is None and entity.get("point") is not None and strip.contains(Point(entity["point"]))]
    return [entity for entity in kept if not any(entity is item for item in inside_texts)], dropped


def _absorb_furniture(layers: dict[str, list[dict[str, Any]]], strip: Polygon, reach: Polygon) -> int:
    """Remove furniture outlines (and their labels / detail lines) that are the built-in itself."""
    absorbed = 0
    names: list[str] = []
    kept: list[dict[str, Any]] = []
    for entity in layers.get("A-FURN", []):
        points = entity.get("points", [])
        if entity.get("type") == "polyline" and entity.get("closed") and len(points) >= 3:
            polygon = Polygon(points).buffer(0)
            if polygon.area > 0.0:
                in_strip = polygon.intersection(strip).area / polygon.area
                in_reach = polygon.intersection(reach).area / polygon.area
                label = str(entity.get("_label", ""))
                if in_strip >= 0.5 or (in_reach >= 0.5 and label in ABSORBED_DISPLAY_NAMES):
                    absorbed += 1
                    names.append(label)
                    continue
        kept.append(entity)
    layers["A-FURN"] = kept
    if absorbed:
        detail = layers.get("A-FURN-DETL", [])
        layers["A-FURN-DETL"] = [
            entity for entity in detail
            if not (len(entity.get("points", [])) >= 2 and LineString(entity["points"]).length > 0.0 and LineString(entity["points"]).intersection(reach.buffer(0.02)).length >= 0.8 * LineString(entity["points"]).length)
        ]
        layers["A-ANNO"] = [
            entity for entity in layers.get("A-ANNO", [])
            if not (
                entity.get("type") == "text" and entity.get("point") is not None and reach.buffer(0.05).contains(Point(entity["point"]))
                and any(str(entity.get("text", "")).startswith(name) for name in names if name)
            )
        ]
    return absorbed
