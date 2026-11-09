"""Wall centre-line, estimated wall-width and opening geometry.

Wall thickness is inferred from the perpendicular spread of cut-band points.
Only visible inner faces are commonly present in a scan, therefore the polygon
width is an estimate and centre-lines remain the preferred measured output.
"""

from __future__ import annotations

import math

import numpy as np

from scipy.spatial import cKDTree
from shapely.geometry import LineString


def wall_lines(
    points: np.ndarray,
    cell_size: float,
    *,
    direction_tolerance_degrees: float = 5.0,
    merge_gap: float = 0.15,
    corner_distance: float = 0.2,
    max_thickness: float = 0.5,
    min_length: float = 0.5,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Extract wall centre-lines from cut-band points.

    1. Thin the points to one per ``cell_size`` cell.
    2. Estimate the dominant wall direction from local PCA orientations
       (folded modulo 90 degrees) and use it plus its perpendicular.
    3. For each direction, repeatedly take the strongest offset band and split it
       into runs along the wall; gaps above ``merge_gap`` (or three times the
       point spacing) separate runs, so door and window gaps stay open.
    4. Remaining points are searched for walls in other directions (RANSAC);
       a residual line within the tolerance snaps to a dominant direction.
    5. Parallel faces closer than ``max_thickness`` become one centre-line.
    6. Segment ends near an intersection are extended or trimmed to it.
    """
    xy = _thin_xy(points, cell_size)
    if len(xy) < 8:
        return []
    spacing = _median_spacing(xy)
    gap = max(merge_gap, spacing * 3.0)
    band = max(cell_size * 2.0, 0.03)
    min_points = max(6, int(0.5 * min_length / max(spacing, cell_size)))
    base = _dominant_angle(xy, max(0.15, spacing * 4.0))
    used = np.zeros(len(xy), dtype=bool)
    lines: list[tuple[float, float, float, float]] = []
    for angle in (base, base + math.pi / 2.0):
        lines.extend(_direction_lines(xy, used, angle, band, gap, min_length, min_points))
    lines.extend(_residual_lines(xy, used, (base, base + math.pi / 2.0), math.radians(direction_tolerance_degrees), band, gap, min_length, min_points))
    lines = _pair_faces(lines, max_thickness)
    segments = [_line_to_segment(line) for line in lines]
    return _close_corners(segments, corner_distance)


def bridge_collinear(segments: list[tuple[tuple[float, float], tuple[float, float]]], gap: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Join collinear wall pieces across gaps up to ``gap`` (used to locate openings)."""
    return _merge_collinear(list(segments), gap)


def wall_polygons(lines: list[tuple[tuple[float, float], tuple[float, float]]], points: np.ndarray, *, minimum_thickness: float) -> list[dict[str, object]]:
    """Buffer centre-lines by their perpendicular point spread into wall polygons."""
    values = np.asarray(points, dtype=np.float64)
    result: list[dict[str, object]] = []
    for start, end in lines:
        vector = np.asarray(end) - np.asarray(start)
        length = float(np.linalg.norm(vector))
        if length == 0:
            continue
        normal = np.array((-vector[1], vector[0])) / length
        unit = vector / length
        relative = values[:, :2] - np.asarray(start)
        along = relative @ unit
        offsets = np.abs(relative @ normal)
        nearby = (along >= -minimum_thickness) & (along <= length + minimum_thickness) & (offsets <= max(0.5, minimum_thickness * 4.0))
        offsets = offsets[nearby]
        thickness = max(minimum_thickness, float(np.percentile(offsets, 90)) * 2.0) if len(offsets) else minimum_thickness
        polygon = LineString([start, end]).buffer(thickness / 2.0, cap_style=2, join_style=2)
        result.append({"type": "polyline", "points": [list(value) for value in polygon.exterior.coords], "holes": [], "closed": True})
    return result


def opening_specs(opening_points: np.ndarray, walls: list[tuple[tuple[float, float], tuple[float, float]]], *, kind: str, cluster_eps: float, max_distance: float) -> tuple[list[dict[str, object]], str | None]:
    """Project 1D DBSCAN opening clusters to walls and return neutral specs."""
    values = np.asarray(opening_points, dtype=np.float64)
    if values.size == 0 or not walls:
        return [], "no wall geometry is available"
    assigned: dict[int, list[tuple[float, np.ndarray]]] = {}
    for point in values:
        wall_index = min(range(len(walls)), key=lambda index: _distance_to_segment(point[:2], *walls[index]))
        start, end = (np.asarray(value, dtype=np.float64) for value in walls[wall_index])
        distance = _distance_to_segment(point[:2], start, end)
        if distance > max_distance:
            continue
        unit = (end - start) / max(float(np.linalg.norm(end - start)), 1e-12)
        assigned.setdefault(wall_index, []).append((float(np.dot(point[:2] - start, unit)), point))
    result: list[dict[str, object]] = []
    for wall_index, records in assigned.items():
        locations = np.asarray([item[0] for item in records]).reshape(-1, 1)
        labels = _cluster_1d(locations, cluster_eps)
        start, end = (np.asarray(value, dtype=np.float64) for value in walls[wall_index])
        unit = (end - start) / max(float(np.linalg.norm(end - start)), 1e-12)
        for label in sorted(set(int(value) for value in labels if value >= 0)):
            cluster = [records[index] for index, value in enumerate(labels) if value == label]
            positions = np.asarray([item[0] for item in cluster])
            centre = start + unit * float(positions.mean())
            width = max(0.3, float(positions.max() - positions.min()))
            result.append({"kind": kind, "wall_index": wall_index, "wall": [start.tolist(), end.tolist()], "center": centre.tolist(), "width": width, "start": (centre - unit * width / 2).tolist(), "end": (centre + unit * width / 2).tolist(), "direction": unit.tolist()})
    return result, None if result else "no opening is within the wall distance threshold"


def opening_entities(
    specs: list[dict[str, object]],
    *,
    swing_arc: bool = False,
    wall_thickness: float = 0.2,
) -> list[dict[str, object]]:
    """Return jambs and window glazing lines, plus optional door swing arcs."""
    if wall_thickness <= 0:
        raise ValueError("Wall thickness must be positive")
    entities: list[dict[str, object]] = []
    for spec in specs:
        direction = np.asarray(spec["direction"], dtype=np.float64)
        normal = np.array((-direction[1], direction[0]))
        start, end = np.asarray(spec["start"], dtype=np.float64), np.asarray(spec["end"], dtype=np.float64)
        half_thickness = wall_thickness / 2.0
        entities.extend(
            [
                {"type": "line", "start": (start - normal * half_thickness).tolist(), "end": (start + normal * half_thickness).tolist()},
                {"type": "line", "start": (end - normal * half_thickness).tolist(), "end": (end + normal * half_thickness).tolist()},
            ]
        )
        if spec["kind"] == "window":
            offset = wall_thickness / 4.0
            entities.extend(
                [
                    {"type": "line", "start": (start - normal * offset).tolist(), "end": (end - normal * offset).tolist()},
                    {"type": "line", "start": (start + normal * offset).tolist(), "end": (end + normal * offset).tolist()},
                ]
            )
        elif swing_arc:
            jamb = np.asarray(spec["start"])
            angle = math.degrees(math.atan2(direction[1], direction[0]))
            entities.append({"type": "arc", "center": jamb.tolist(), "radius": float(spec["width"]), "start_angle": angle, "end_angle": angle + 90.0})
    return entities


def _merge_collinear(segments: list[tuple[tuple[float, float], tuple[float, float]]], gap: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    result = list(segments)
    changed = True
    while changed:
        changed = False
        for first_index in range(len(result)):
            for second_index in range(first_index + 1, len(result)):
                if not _same_line(result[first_index], result[second_index], gap):
                    continue
                result[first_index:first_index + 1] = [_unite_collinear(result[first_index], result[second_index])]
                del result[second_index]
                changed = True
                break
            if changed:
                break
    return result


def _same_line(first: tuple[tuple[float, float], tuple[float, float]], second: tuple[tuple[float, float], tuple[float, float]], gap: float) -> bool:
    a, b = np.asarray(first[0]), np.asarray(first[1])
    c, d = np.asarray(second[0]), np.asarray(second[1])
    vector = b - a
    length = np.linalg.norm(vector)
    other_vector = d - c
    other_length = np.linalg.norm(other_vector)
    direction_cross = float(vector[0] * other_vector[1] - vector[1] * other_vector[0])
    cross = float(vector[0] * (c - a)[1] - vector[1] * (c - a)[0])
    if length == 0 or other_length == 0 or abs(direction_cross / (length * other_length)) > 0.05 or abs(cross / length) > 0.1:
        return False
    unit = vector / length
    first_min, first_max = sorted((0.0, length))
    second_values = [float(np.dot(value - a, unit)) for value in (c, d)]
    second_min, second_max = sorted(second_values)
    return max(first_min, second_min) <= min(first_max, second_max) + gap


def _distance_to_segment(point: np.ndarray, start: tuple[float, float] | np.ndarray, end: tuple[float, float] | np.ndarray) -> float:
    start_array, end_array = np.asarray(start), np.asarray(end)
    vector = end_array - start_array
    denominator = float(np.dot(vector, vector))
    if denominator == 0:
        return float(np.linalg.norm(point - start_array))
    factor = np.clip(np.dot(point - start_array, vector) / denominator, 0.0, 1.0)
    return float(np.linalg.norm(point - (start_array + factor * vector)))


def _angle_distance(first: float, second: float) -> float:
    return abs((first - second + math.pi / 2) % math.pi - math.pi / 2)


def _unite_collinear(first: tuple[tuple[float, float], tuple[float, float]], second: tuple[tuple[float, float], tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    start, end = (np.asarray(value, dtype=np.float64) for value in first)
    direction = (end - start) / max(float(np.linalg.norm(end - start)), 1e-12)
    candidates = [np.asarray(value, dtype=np.float64) for value in (*first, *second)]
    values = [float(np.dot(candidate - start, direction)) for candidate in candidates]
    return tuple((start + direction * value).tolist() for value in (min(values), max(values)))  # type: ignore[return-value]


def _close_corners(segments: list[tuple[tuple[float, float], tuple[float, float]]], distance: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    result = [[np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)] for start, end in segments]
    for first_index, first in enumerate(result):
        for second_index, second in enumerate(result[first_index + 1:], start=first_index + 1):
            intersection = _line_intersection(first[0], first[1], second[0], second[1])
            if intersection is None:
                continue
            for segment in (first, second):
                endpoint = min(range(2), key=lambda index: float(np.linalg.norm(segment[index] - intersection)))
                if np.linalg.norm(segment[endpoint] - intersection) <= distance:
                    segment[endpoint] = intersection
    return [(tuple(start), tuple(end)) for start, end in result]


def _line_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    matrix = np.column_stack((b - a, c - d))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-9:
        return None
    factors = np.linalg.solve(matrix, c - a)
    return a + (b - a) * factors[0]


def _cluster_1d(values: np.ndarray, eps: float) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=1).fit_predict(values)


def _thin_xy(points: np.ndarray, cell_size: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    xy = values[:, :2]
    _, index = np.unique(np.floor(xy / cell_size).astype(np.int64), axis=0, return_index=True)
    return xy[np.sort(index)]


def _median_spacing(xy: np.ndarray) -> float:
    sample = xy if len(xy) <= 20_000 else xy[np.linspace(0, len(xy) - 1, 20_000, dtype=np.int64)]
    distances, _ = cKDTree(sample).query(sample, k=2)
    return float(np.median(distances[:, 1]))


def _dominant_angle(xy: np.ndarray, radius: float) -> float:
    """Return the dominant wall direction in [0, pi/2) from local PCA orientations."""
    sample = xy if len(xy) <= 6_000 else xy[np.linspace(0, len(xy) - 1, 6_000, dtype=np.int64)]
    neighbourhoods = cKDTree(xy).query_ball_point(sample, r=radius)
    angles: list[float] = []
    weights: list[float] = []
    for members in neighbourhoods:
        if len(members) < 5:
            continue
        local = xy[members]
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov((local - local.mean(axis=0)).T))
        if eigenvalues[1] <= 1e-12:
            continue
        linearity = float((eigenvalues[1] - eigenvalues[0]) / eigenvalues[1])
        direction = eigenvectors[:, 1]
        angles.append(math.atan2(direction[1], direction[0]))
        weights.append(linearity**2)
    if not angles:
        return 0.0
    vector = np.sum(np.asarray(weights) * np.exp(1j * np.asarray(angles) * 4.0))
    return float(np.angle(vector) / 4.0) % (math.pi / 2.0)


def _direction_lines(xy: np.ndarray, used: np.ndarray, angle: float, band: float, gap: float, min_length: float, min_points: int) -> list[tuple[float, float, float, float]]:
    """Find wall runs parallel to ``angle``; returns (angle, offset, start, end) tuples."""
    unit = np.array((math.cos(angle), math.sin(angle)))
    normal = np.array((-unit[1], unit[0]))
    along = xy @ unit
    across = xy @ normal
    result: list[tuple[float, float, float, float]] = []
    blocked = np.zeros(len(xy), dtype=bool)
    while True:
        candidates = ~used & ~blocked
        if candidates.sum() < min_points:
            break
        values = across[candidates]
        bins = np.floor(values / band).astype(np.int64)
        labels, counts = np.unique(bins, return_counts=True)
        if counts.max() < min_points:
            break
        peak = labels[np.argmax(counts)]
        centre = float(np.median(values[(bins >= peak - 1) & (bins <= peak + 1)]))
        member = candidates & (np.abs(across - centre) <= band)
        offset = float(np.median(across[member]))
        member = candidates & (np.abs(across - offset) <= band)
        blocked |= member
        order = np.sort(along[member])
        steps = np.diff(order)
        moving = steps[steps > 1e-3]
        run_gap = max(gap, 3.0 * float(np.median(moving))) if len(moving) else gap
        breaks = np.flatnonzero(steps > run_gap)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(order) - 1]))
        found = False
        for first, last in zip(starts, ends, strict=True):
            if order[last] - order[first] >= min_length and last - first + 1 >= min_points:
                result.append((angle, offset, float(order[first]), float(order[last])))
                found = True
        if found:
            used |= member
    return result


def _residual_lines(xy: np.ndarray, used: np.ndarray, dominant: tuple[float, float], tolerance: float, band: float, gap: float, min_length: float, min_points: int) -> list[tuple[float, float, float, float]]:
    """Sequential RANSAC for walls outside the dominant directions (deterministic seed)."""
    rng = np.random.default_rng(0)
    result: list[tuple[float, float, float, float]] = []
    for _ in range(12):
        remaining = np.flatnonzero(~used)
        if len(remaining) < max(min_points, 2):
            break
        points = xy[remaining]
        best: np.ndarray | None = None
        best_angle = 0.0
        for _ in range(200):
            first, second = points[rng.choice(len(points), 2, replace=False)]
            direction = second - first
            length = float(np.linalg.norm(direction))
            if length < band:
                continue
            normal = np.array((-direction[1], direction[0])) / length
            inliers = np.abs((points - first) @ normal) <= band
            if best is None or inliers.sum() > best.sum():
                best, best_angle = inliers, math.atan2(direction[1], direction[0])
        if best is None or best.sum() < min_points:
            break
        for candidate in dominant:
            if _angle_distance(best_angle % math.pi, candidate % math.pi) <= tolerance:
                best_angle = candidate
        result.extend(_direction_lines(xy[remaining[best]], np.zeros(int(best.sum()), dtype=bool), best_angle, band, gap, min_length, min_points))
        used[remaining[best]] = True
    return result


def _pair_faces(lines: list[tuple[float, float, float, float]], max_thickness: float) -> list[tuple[float, float, float, float]]:
    """Merge two parallel, overlapping wall faces into one centre-line."""
    remaining = list(lines)
    result: list[tuple[float, float, float, float]] = []
    while remaining:
        angle, offset, start, end = remaining.pop(0)
        partner_index = None
        for index, (other_angle, other_offset, other_start, other_end) in enumerate(remaining):
            if _angle_distance(angle % math.pi, other_angle % math.pi) > 1e-6:
                continue
            overlap = min(end, other_end) - max(start, other_start)
            shorter = min(end - start, other_end - other_start)
            if 0 < abs(offset - other_offset) <= max_thickness and overlap >= 0.5 * shorter:
                partner_index = index
                break
        if partner_index is None:
            result.append((angle, offset, start, end))
            continue
        _, other_offset, other_start, other_end = remaining.pop(partner_index)
        result.append((angle, (offset + other_offset) / 2.0, min(start, other_start), max(end, other_end)))
    return result


def _line_to_segment(line: tuple[float, float, float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    angle, offset, start, end = line
    unit = np.array((math.cos(angle), math.sin(angle)))
    normal = np.array((-unit[1], unit[0]))
    first = unit * start + normal * offset
    second = unit * end + normal * offset
    return (float(first[0]), float(first[1])), (float(second[0]), float(second[1]))
