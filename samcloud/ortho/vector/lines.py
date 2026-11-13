"""Skeleton tracing for curbs, road edges, fences, and railings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from scipy.spatial import cKDTree
from shapely import line_merge
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union
from skimage.morphology import skeletonize

from .raster_grid import RasterGrid

Pixel = tuple[int, int]


def trace_lines(
    mask: np.ndarray,
    grid: RasterGrid,
    *,
    min_length: float,
    gap_max: float,
    angle_tol_deg: float,
    smooth_iterations: int,
    simplify_tolerance: float,
) -> list[LineString]:
    """Trace a skeleton graph, bridge compatible gaps, smooth, and simplify it."""
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != (grid.height, grid.width):
        raise ValueError("mask shape must match RasterGrid")
    skeleton = skeletonize(binary)
    graph = _pixel_graph(skeleton)
    branches = _prune_spurs(_branches(graph), graph, grid, min_length)
    local = [_to_line(branch, grid) for branch in branches]
    joined = _join_gaps(local, gap_max, math.radians(angle_tol_deg), grid.pixel_size)
    result: list[LineString] = []
    for line in joined:
        smoothed = _chaikin(line, smooth_iterations)
        simplified = smoothed.simplify(simplify_tolerance, preserve_topology=False) if simplify_tolerance > 0 else smoothed
        if isinstance(simplified, LineString) and simplified.length >= min_length:
            result.append(simplified)
    return result


def boundary_lines(
    partition: dict[str, list[Polygon]],
    layer_a: str,
    layer_b: str,
    *,
    min_length: float = 0.0,
    simplify_tolerance: float = 0.0,
) -> list[LineString]:
    """Return merged shared boundaries between two partition layers."""
    if not partition.get(layer_a) or not partition.get(layer_b):
        return []
    first = unary_union(partition[layer_a])
    second = unary_union(partition[layer_b])
    shared = first.boundary.intersection(second.boundary)
    merged = line_merge(shared)
    candidates = list(merged.geoms) if isinstance(merged, MultiLineString) else [merged]
    output: list[LineString] = []
    for line in candidates:
        if not isinstance(line, LineString):
            continue
        candidate = line.simplify(simplify_tolerance, preserve_topology=False) if simplify_tolerance > 0.0 else line
        if candidate.length >= min_length:
            output.append(candidate)
    return output


def _pixel_graph(skeleton: np.ndarray) -> dict[Pixel, set[Pixel]]:
    rows, cols = np.nonzero(skeleton)
    nodes = {(int(row), int(col)) for row, col in zip(rows.tolist(), cols.tolist(), strict=True)}
    graph: dict[Pixel, set[Pixel]] = {node: set() for node in nodes}
    for row, col in nodes:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbour = (row + dr, col + dc)
                if dr != 0 and dc != 0 and ((row + dr, col) in nodes or (row, col + dc) in nodes):
                    continue
                if neighbour in nodes:
                    graph[(row, col)].add(neighbour)
    return graph


def _branches(graph: dict[Pixel, set[Pixel]]) -> list[list[Pixel]]:
    if not graph:
        return []
    terminals = {node for node, neighbours in graph.items() if len(neighbours) != 2}
    visited: set[frozenset[Pixel]] = set()
    output: list[list[Pixel]] = []
    for start in terminals:
        for neighbour in graph[start]:
            edge = frozenset((start, neighbour))
            if edge in visited:
                continue
            path = [start, neighbour]
            visited.add(edge)
            previous, current = start, neighbour
            while current not in terminals:
                choices = [node for node in graph[current] if node != previous]
                if not choices:
                    break
                following = choices[0]
                visited.add(frozenset((current, following)))
                path.append(following)
                previous, current = current, following
            output.append(path)
    for start in graph:
        for neighbour in graph[start]:
            edge = frozenset((start, neighbour))
            if edge in visited:
                continue
            path = [start, neighbour]
            visited.add(edge)
            previous, current = start, neighbour
            while True:
                choices = [node for node in graph[current] if node != previous]
                if not choices:
                    break
                following = choices[0]
                if following == start:
                    path.append(following)
                    visited.add(frozenset((current, following)))
                    break
                if frozenset((current, following)) in visited:
                    break
                visited.add(frozenset((current, following)))
                path.append(following)
                previous, current = current, following
            output.append(path)
    return [path for path in output if len(path) >= 2]


def _to_line(path: list[Pixel], grid: RasterGrid) -> LineString:
    return LineString([grid.pixel_to_local(row, col) for row, col in path])


def _prune_spurs(
    branches: list[list[Pixel]], graph: dict[Pixel, set[Pixel]], grid: RasterGrid, min_length: float
) -> list[list[Pixel]]:
    """Drop short branches that terminate in a skeleton leaf before gap joining."""
    minimum = min_length / 2.0
    result: list[list[Pixel]] = []
    for branch in branches:
        has_leaf = len(graph[branch[0]]) == 1 or len(graph[branch[-1]]) == 1
        if has_leaf and _to_line(branch, grid).length < minimum:
            continue
        result.append(branch)
    return result


@dataclass(slots=True)
class _GapLine:
    """One active line and the source endpoints that still bound it."""

    coordinates: list[tuple[float, float]]
    start_source: int
    end_source: int
    order: int
    start_tangent: np.ndarray
    end_tangent: np.ndarray


def _join_gaps(lines: list[LineString], gap_max: float, angle_tolerance: float, pixel_size: float) -> list[LineString]:
    """Join globally nearest compatible endpoints without repeatedly scanning all lines."""
    if len(lines) < 2 or gap_max <= 0.0:
        return lines[:]

    endpoint_coordinates = np.asarray(
        [point for line in lines for point in (line.coords[0], line.coords[-1])], dtype=float
    )
    endpoint_tree = cKDTree(endpoint_coordinates)
    active: dict[int, _GapLine] = {}
    endpoint_owner: dict[int, tuple[int, bool]] = {}
    for index, line in enumerate(lines):
        coordinates = [(float(x), float(y)) for x, y in line.coords]
        start_source, end_source = 2 * index, 2 * index + 1
        active[index] = _gap_line(coordinates, start_source, end_source, index, pixel_size)
        endpoint_owner[start_source] = (index, True)
        endpoint_owner[end_source] = (index, False)

    candidates: list[tuple[float, int, int, int, int, bool, bool]] = []

    def add_candidate(first_source: int, second_source: int) -> None:
        first_owner = endpoint_owner.get(first_source)
        second_owner = endpoint_owner.get(second_source)
        if first_owner is None or second_owner is None or first_owner[0] == second_owner[0]:
            return
        first_id, first_at_start = first_owner
        second_id, second_at_start = second_owner
        first = active[first_id]
        second = active[second_id]
        if first.order > second.order:
            first_id, second_id = second_id, first_id
            first_at_start, second_at_start = second_at_start, first_at_start
            first, second = second, first
        candidate = _gap_candidate(first, first_at_start, second, second_at_start, gap_max, angle_tolerance)
        if candidate is not None:
            distance = candidate
            heappush(candidates, (distance, first.order, second.order, first_id, second_id, first_at_start, second_at_start))

    for first_source, second_source in endpoint_tree.query_pairs(gap_max):
        add_candidate(int(first_source), int(second_source))

    next_id = len(lines)
    while candidates:
        _distance, _first_order, _second_order, first_id, second_id, first_at_start, second_at_start = heappop(candidates)
        first = active.get(first_id)
        second = active.get(second_id)
        if first is None or second is None:
            continue
        candidate = _gap_candidate(first, first_at_start, second, second_at_start, gap_max, angle_tolerance)
        if candidate is None:
            continue

        joined_coordinates, start_source, end_source = _joined_coordinates(first, first_at_start, second, second_at_start)
        del active[first_id]
        del active[second_id]
        for source in (first.start_source, first.end_source, second.start_source, second.end_source):
            endpoint_owner.pop(source, None)

        joined = _gap_line(joined_coordinates, start_source, end_source, first.order, pixel_size)
        active[next_id] = joined
        endpoint_owner[start_source] = (next_id, True)
        endpoint_owner[end_source] = (next_id, False)
        for source in (start_source, end_source):
            for other_source in endpoint_tree.query_ball_point(endpoint_coordinates[source], gap_max):
                add_candidate(source, int(other_source))
        next_id += 1

    return [LineString(line.coordinates) for line in sorted(active.values(), key=lambda line: line.order)]


def _gap_line(
    coordinates: list[tuple[float, float]], start_source: int, end_source: int, order: int, pixel_size: float
) -> _GapLine:
    return _GapLine(
        coordinates=coordinates,
        start_source=start_source,
        end_source=end_source,
        order=order,
        start_tangent=_endpoint_tangent(coordinates, at_start=True, pixel_size=pixel_size),
        end_tangent=_endpoint_tangent(coordinates, at_start=False, pixel_size=pixel_size),
    )


def _gap_candidate(
    first: _GapLine,
    first_at_start: bool,
    second: _GapLine,
    second_at_start: bool,
    gap_max: float,
    angle_tolerance: float,
) -> float | None:
    first_point = np.asarray(first.coordinates[0 if first_at_start else -1], dtype=float)
    second_point = np.asarray(second.coordinates[0 if second_at_start else -1], dtype=float)
    gap = second_point - first_point
    distance = float(np.hypot(*gap))
    if distance > gap_max or distance <= 1e-9:
        return None
    direction = gap / distance
    first_tangent = -first.start_tangent if first_at_start else first.end_tangent
    second_tangent = second.start_tangent if second_at_start else -second.end_tangent
    if _vector_angle(first_tangent, direction) > angle_tolerance:
        return None
    if _vector_angle(second_tangent, direction) > angle_tolerance:
        return None
    return distance


def _joined_coordinates(
    first: _GapLine, first_at_start: bool, second: _GapLine, second_at_start: bool
) -> tuple[list[tuple[float, float]], int, int]:
    left = first.coordinates[::-1] if first_at_start else first.coordinates
    right = second.coordinates if second_at_start else second.coordinates[::-1]
    start_source = first.end_source if first_at_start else first.start_source
    end_source = second.end_source if second_at_start else second.start_source
    return left + right, start_source, end_source


def _try_join(
    first: LineString, second: LineString, gap_max: float, angle_tolerance: float, pixel_size: float
) -> tuple[float, LineString] | None:
    candidates = [
        (list(first.coords), list(second.coords)),
        (list(first.coords), list(second.coords)[::-1]),
        (list(first.coords)[::-1], list(second.coords)),
        (list(first.coords)[::-1], list(second.coords)[::-1]),
    ]
    compatible: list[tuple[float, LineString]] = []
    for left, right in candidates:
        gap = np.asarray(right[0], dtype=float) - np.asarray(left[-1], dtype=float)
        distance = float(np.hypot(*gap))
        if distance > gap_max or distance <= 1e-9:
            continue
        direction = gap / distance
        tangent_a = _endpoint_tangent(left, at_start=False, pixel_size=pixel_size)
        tangent_b = _endpoint_tangent(right, at_start=True, pixel_size=pixel_size)
        if _vector_angle(tangent_a, direction) <= angle_tolerance and _vector_angle(tangent_b, direction) <= angle_tolerance:
            compatible.append((distance, LineString(left + right)))
    return min(compatible, key=lambda candidate: candidate[0]) if compatible else None


def _endpoint_tangent(points: list[tuple[float, float]], *, at_start: bool, pixel_size: float) -> np.ndarray:
    """Fit an oriented least-squares tangent over at least 0.5 m of skeleton."""
    count = min(len(points), max(5, int(math.ceil(0.5 / pixel_size))))
    sample = np.asarray(points[:count] if at_start else points[-count:], dtype=float)
    centred = sample - sample.mean(axis=0)
    _values, vectors = np.linalg.eigh(centred.T @ centred)
    tangent = vectors[:, -1]
    progression = sample[1] - sample[0] if at_start else sample[-1] - sample[-2]
    if float(np.dot(tangent, progression)) < 0.0:
        tangent = -tangent
    return _unit(tangent)


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.hypot(*vector))
    return vector / length if length > 1e-12 else np.asarray((1.0, 0.0))


def _vector_angle(first: np.ndarray, second: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))


def _chaikin(line: LineString, iterations: int) -> LineString:
    coordinates = [np.asarray(point, dtype=float) for point in line.coords]
    for _ in range(max(0, iterations)):
        if len(coordinates) < 3:
            break
        refined = [coordinates[0]]
        for first, second in zip(coordinates, coordinates[1:], strict=False):
            refined.append(first * 0.75 + second * 0.25)
            refined.append(first * 0.25 + second * 0.75)
        refined.append(coordinates[-1])
        coordinates = refined
    return LineString(coordinates)
