"""Roof-plane labelling for footprint partition cells."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import networkx as nx
import numpy as np
from shapely import distance, points
from shapely.geometry import LineString

from samcloud.buildings.partition import Partition
from samcloud.drone.contracts import RoofPlane

logger = logging.getLogger(__name__)


@dataclass
class LabelResult:
    """Result of the pruned alpha-expansion labelling."""

    labels: np.ndarray
    energy: float
    cycles: int
    timed_out: bool
    candidates_per_cell: list[list[int]]
    seconds: float


def _height(plane: RoofPlane, xy: np.ndarray) -> np.ndarray:
    normal = np.asarray(plane.normal, dtype=float)
    if abs(normal[2]) < 1e-10:
        return np.full(len(xy), np.nan)
    return -(xy @ normal[:2] + float(plane.d)) / normal[2]


def _unique_adjacency(partition: Partition) -> list[tuple[int, int, float]]:
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int, float]] = []
    for first, second, length in partition.adjacency:
        key = (min(first, second), max(first, second))
        if first == second or key in seen or length <= 0:
            continue
        seen.add(key)
        result.append((key[0], key[1], float(length)))
    return result


def _pair_cost(first: int, second: int, edge: tuple[int, int, float], cells, planes, complexity: float, cache: dict | None = None) -> float:
    if first == second or first == 0 or second == 0:
        return 0.0
    key = (edge[0], edge[1], min(first, second), max(first, second))
    if cache is not None and key in cache:
        return cache[key]
    line = cells[edge[0]].boundary.intersection(cells[edge[1]].boundary)
    if isinstance(line, LineString):
        coordinates = np.asarray(line.coords, dtype=float)
    else:
        segments = [item for item in getattr(line, "geoms", []) if hasattr(item, "coords")]
        coordinates = max((np.asarray(item.coords, dtype=float) for item in segments), key=len, default=np.empty((0, 2)))
    if len(coordinates) < 2:
        midpoint = cells[edge[0]].boundary.intersection(cells[edge[1]].boundary).centroid
        xy = np.array([[midpoint.x, midpoint.y]])
    else:
        fractions = np.linspace(0.0, 1.0, 5)
        xy = coordinates[0] + fractions[:, None] * (coordinates[-1] - coordinates[0])
    delta = np.abs(_height(planes[first - 1], xy) - _height(planes[second - 1], xy))
    value = float(edge[2] * complexity * np.nanmean(np.minimum(delta, 5.0)))
    if cache is not None:
        cache[key] = value
    return value


def _energy(labels: np.ndarray, data: np.ndarray, edges, cells, planes, complexity: float, cache: dict | None = None) -> float:
    total = float(data[np.arange(len(labels)), labels].sum())
    for edge in edges:
        total += _pair_cost(int(labels[edge[0]]), int(labels[edge[1]]), edge, cells, planes, complexity, cache)
    return total


def _cut_move(labels, alpha, candidates, data, edges, cells, planes, complexity, cache) -> np.ndarray:
    """Return one alpha-expansion move using the KZ binary construction."""
    variable = [index for index, options in enumerate(candidates) if alpha in options and labels[index] != alpha]
    if not variable:
        return labels
    position = {cell: index for index, cell in enumerate(variable)}
    # [cost for keep, cost for switching], accumulated with pairwise fixed terms.
    unary = np.array([[data[cell, labels[cell]], data[cell, alpha]] for cell in variable], dtype=float)
    graph = nx.DiGraph()
    source, sink = "source", "sink"
    graph.add_nodes_from((source, sink))
    constant = 0.0
    for first, second, length in edges:
        first_variable, second_variable = first in position, second in position
        current_a, current_b = int(labels[first]), int(labels[second])
        if not first_variable and not second_variable:
            continue
        if first_variable and not second_variable:
            node = position[first]
            unary[node, 0] += _pair_cost(current_a, current_b, (first, second, length), cells, planes, complexity, cache)
            unary[node, 1] += _pair_cost(alpha, current_b, (first, second, length), cells, planes, complexity, cache)
            continue
        if second_variable and not first_variable:
            node = position[second]
            unary[node, 0] += _pair_cost(current_a, current_b, (first, second, length), cells, planes, complexity, cache)
            unary[node, 1] += _pair_cost(current_a, alpha, (first, second, length), cells, planes, complexity, cache)
            continue
        left, right = position[first], position[second]
        e00 = _pair_cost(current_a, current_b, (first, second, length), cells, planes, complexity, cache)
        e01 = _pair_cost(current_a, alpha, (first, second, length), cells, planes, complexity, cache)
        e10 = _pair_cost(alpha, current_b, (first, second, length), cells, planes, complexity, cache)
        e11 = 0.0
        cut = 0.5 * (e01 + e10 - e00 - e11)
        if cut < -1e-9:
            logger.debug("non-submodular alpha-expansion term clamped for cells %s/%s", first, second)
            cut = 0.0
        unary[left, 1] += e10 - e00 - cut
        unary[right, 1] += e01 - e00 - cut
        constant += e00
        if cut > 0:
            graph.add_edge(left, right, capacity=float(cut))
            graph.add_edge(right, left, capacity=float(cut))
    for node, (keep, switch) in enumerate(unary):
        offset = min(keep, switch)
        constant += offset
        graph.add_edge(source, node, capacity=float(max(0.0, switch - offset)))
        graph.add_edge(node, sink, capacity=float(max(0.0, keep - offset)))
    _value, partition = nx.minimum_cut(graph, source, sink, capacity="capacity")
    source_side, _sink_side = partition
    result = labels.copy()
    for cell, node in position.items():
        if node not in source_side:
            result[cell] = alpha
    return result


def label_cells(
    partition: Partition,
    roof_planes: list[RoofPlane],
    roof_points_xyz: np.ndarray,
    *,
    buffer: float,
    complexity_factor: float,
    time_limit: float,
) -> LabelResult:
    """Assign partition cells to pruned roof-plane labels with alpha expansion."""
    started = time.perf_counter()
    cells = partition.cells
    count, labels_count = len(cells), len(roof_planes)
    xyz = np.asarray(roof_points_xyz, dtype=float)
    candidates: list[list[int]] = []
    supports: list[np.ndarray] = []
    support_bounds: list[tuple[float, float, float, float] | None] = []
    for plane in roof_planes:
        indices = np.asarray(plane.inliers, dtype=int)
        indices = indices[(indices >= 0) & (indices < len(xyz))]
        support = xyz[indices, :2]
        supports.append(support)
        if len(support):
            support_bounds.append((float(support[:, 0].min()), float(support[:, 1].min()), float(support[:, 0].max()), float(support[:, 1].max())))
        elif plane.outline is not None and not plane.outline.is_empty:
            support_bounds.append(tuple(float(value) for value in plane.outline.bounds))
        else:
            support_bounds.append(None)
    for cell in cells:
        options = [0]
        cmin_x, cmin_y, cmax_x, cmax_y = cell.bounds
        for plane_index, (plane, support, bounds) in enumerate(zip(roof_planes, supports, support_bounds, strict=False), start=1):
            if bounds is None:
                continue
            pmin_x, pmin_y, pmax_x, pmax_y = bounds
            dx = max(cmin_x - pmax_x, pmin_x - cmax_x, 0.0)
            dy = max(cmin_y - pmax_y, pmin_y - cmax_y, 0.0)
            if dx * dx + dy * dy > buffer * buffer:
                continue
            near = bool(len(support) and np.any(distance(cell, points(support)) <= buffer))
            if not near and plane.outline is not None and not plane.outline.is_empty:
                near = plane.outline.distance(cell) <= buffer
            if near:
                options.append(plane_index)
        candidates.append(options)
    data = np.full((count, labels_count + 1), np.inf, dtype=float)
    for cell_index, cell in enumerate(cells):
        min_x, min_y, max_x, max_y = cell.bounds
        boxed = np.flatnonzero((xyz[:, 0] >= min_x) & (xyz[:, 0] <= max_x) & (xyz[:, 1] >= min_y) & (xyz[:, 1] <= max_y))
        if len(boxed):
            inside = np.asarray(cell.covers(points(xyz[boxed, :2])), dtype=bool)
            local = xyz[boxed[inside]]
        else:
            local = xyz[:0]
        area = max(float(cell.area), 1e-6)
        data[cell_index, 0] = 0.5 * area if len(local) else 0.02 * area
        for label in candidates[cell_index][1:]:
            if len(local):
                residual = np.minimum(np.abs(roof_planes[label - 1].distance(local)), 0.5)
                data[cell_index, label] = float(residual.mean() * area)
            else:
                data[cell_index, label] = 0.021 * area
    labels = np.array([options[int(np.argmin(data[index, options]))] for index, options in enumerate(candidates)], dtype=int)
    edges = _unique_adjacency(partition)
    pair_cache: dict = {}
    current_energy = _energy(labels, data, edges, cells, roof_planes, complexity_factor, pair_cache)
    all_labels = range(labels_count + 1)
    cycles, timed_out = 0, False
    while True:
        changed = False
        cycles += 1
        for alpha in all_labels:
            if time.perf_counter() - started >= time_limit:
                timed_out = True
                break
            proposal = _cut_move(labels, alpha, candidates, data, edges, cells, roof_planes, complexity_factor, pair_cache)
            proposal_energy = _energy(proposal, data, edges, cells, roof_planes, complexity_factor, pair_cache)
            if proposal_energy < current_energy - 1e-9:
                labels, current_energy, changed = proposal, proposal_energy, True
        if timed_out or not changed or cycles >= max(2, labels_count + 1):
            break
    return LabelResult(labels, float(current_energy), cycles, timed_out, candidates, time.perf_counter() - started)
