"""Feature geometry for furniture and outdoor symbols."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize
from shapely.geometry import LineString

from samcloud.dxf.geometry.raster import rasterize_points


def furniture_inserts(points: np.ndarray, codes: np.ndarray | int, *, eps: float = 0.25) -> list[dict[str, object]]:
    """Fit per-class furniture clusters as rotated rectangle block inserts."""
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return []
    class_values = np.full(len(values), int(codes), dtype=np.uint8) if np.isscalar(codes) else np.asarray(codes, dtype=np.uint8)
    if len(class_values) != len(values):
        raise ValueError("Furniture codes must have one value per point")
    result: list[dict[str, object]] = []
    for code in sorted(set(int(value) for value in class_values)):
        members = values[class_values == code]
        labels = DBSCAN(eps=eps, min_samples=min(5, len(members))).fit_predict(members[:, :2])
        for label in sorted(set(int(value) for value in labels if value >= 0)):
            cluster = members[labels == label]
            rectangle = cv2.minAreaRect(cluster[:, :2].astype(np.float32))
            (x, y), (width, height), angle = rectangle
            result.append({"type": "insert", "name": "FURNITURE", "point": [float(x), float(y)], "rotation": float(angle), "xscale": float(width), "yscale": float(height), "attributes": {"class": str(code), "height": f"{float(cluster[:, 2].max()):.3f}"}})
    return result


def circles_from_clusters(points: np.ndarray, *, eps: float = 0.5, minimum_radius: float = 0.05, radius_percentile: float = 100.0, maximum_radius: float | None = None) -> list[dict[str, object]]:
    """Return circles from DBSCAN clusters using the horizontal extent.

    With ``maximum_radius``, a cluster that is larger (e.g. a row of touching
    tree crowns) is re-clustered with a smaller ``eps`` until every circle fits
    or ``eps`` drops below 0.2 m.
    """
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return []
    labels = DBSCAN(eps=eps, min_samples=max(1, min(4, len(values)))).fit_predict(values[:, :2])
    result = []
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        cluster = values[labels == label]
        centre = cluster[:, :2].mean(axis=0)
        radius = max(minimum_radius, float(np.percentile(np.linalg.norm(cluster[:, :2] - centre, axis=1), radius_percentile)))
        if maximum_radius is not None and radius > maximum_radius and eps * 0.6 >= 0.2:
            result.extend(circles_from_clusters(cluster, eps=eps * 0.6, minimum_radius=minimum_radius, radius_percentile=radius_percentile, maximum_radius=maximum_radius))
            continue
        result.append({"type": "circle", "center": centre.tolist(), "radius": radius})
    return result


def points_from_cluster(points: np.ndarray, *, eps: float = 0.5) -> list[dict[str, object]]:
    """Represent every DBSCAN feature cluster as a centroid point."""
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return []
    labels = DBSCAN(eps=eps, min_samples=max(1, min(3, len(values)))).fit_predict(values[:, :2])
    return [{"type": "point", "point": values[labels == label, :2].mean(axis=0).tolist()} for label in sorted(set(int(value) for value in labels if value >= 0))]


def skeleton_lines(points: np.ndarray, cell_size: float, *, simplify_tolerance: float = 0.0, min_length: float = 0.0) -> list[dict[str, object]]:
    """Trace a skeleton graph into metre polylines.

    The mask is closed (3x3) so noisy one-cell-wide lines stay connected, then
    skeletonised. Paths run between endpoints and junctions on an 8-connected
    graph without redundant diagonal edges; closed loops are traced too.
    Short spurs are pruned and paths meeting end to end are joined before
    simplification.
    """
    raster = rasterize_points(points, cell_size, closing_kernel=3)
    if raster.mask.size == 0:
        return []
    skeleton = skeletonize(raster.mask > 0)
    nodes = {(int(row), int(column)) for row, column in np.argwhere(skeleton)}
    neighbours = {node: _graph_neighbours(node, nodes) for node in nodes}
    anchors = {node for node, linked in neighbours.items() if len(linked) != 2}
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[list[tuple[int, int]]] = []
    for anchor in sorted(anchors):
        for next_node in neighbours[anchor]:
            if _edge(anchor, next_node) not in seen:
                paths.append(_walk(anchor, next_node, neighbours, anchors, seen))
    for node in sorted(nodes):
        for next_node in neighbours[node]:
            if _edge(node, next_node) not in seen:
                paths.append(_walk(node, next_node, neighbours, {node}, seen))
    degree = {anchor: len(neighbours[anchor]) for anchor in anchors}
    spur = max(3.0 * cell_size, min_length)
    metre_paths = []
    for path in paths:
        coordinates = [(raster.origin[0] + column * cell_size, raster.origin[1] + row * cell_size) for row, column in path]
        length = LineString(coordinates).length if len(coordinates) > 1 else 0.0
        ends = (degree.get(path[0], 2), degree.get(path[-1], 2))
        if length < spur and 1 in ends and max(ends) >= 3:
            continue
        metre_paths.append(coordinates)
    lines: list[dict[str, object]] = []
    for coordinates in _join_paths(metre_paths, 1.5 * cell_size):
        if len(coordinates) < 2:
            continue
        line = LineString(coordinates).simplify(max(simplify_tolerance, cell_size), preserve_topology=False)
        if line.length >= min_length and len(line.coords) >= 2:
            lines.append({"type": "polyline", "points": [list(value) for value in line.coords], "closed": False})
    if lines:
        return lines
    return _sparse_line_fit(points, min_length)


def _graph_neighbours(node: tuple[int, int], nodes: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """8-neighbours, skipping a diagonal when an orthogonal pixel already links both."""
    row, column = node
    result = [candidate for candidate in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)) if candidate in nodes]
    for row_offset in (-1, 1):
        for column_offset in (-1, 1):
            diagonal = (row + row_offset, column + column_offset)
            if diagonal in nodes and (row + row_offset, column) not in nodes and (row, column + column_offset) not in nodes:
                result.append(diagonal)
    return result


def _walk(start: tuple[int, int], following: tuple[int, int], neighbours: dict[tuple[int, int], list[tuple[int, int]]], stops: set[tuple[int, int]], seen: set[tuple[tuple[int, int], tuple[int, int]]]) -> list[tuple[int, int]]:
    path = [start]
    previous, current = start, following
    seen.add(_edge(previous, current))
    while current not in stops:
        path.append(current)
        candidates = [item for item in neighbours[current] if item != previous and _edge(current, item) not in seen]
        if not candidates:
            return path
        previous, current = current, candidates[0]
        seen.add(_edge(previous, current))
    path.append(current)
    return path


def _join_paths(paths: list[list[tuple[float, float]]], tolerance: float) -> list[list[tuple[float, float]]]:
    """Join paths whose endpoints meet when exactly two path ends share that point.

    Endpoints are indexed on a ``tolerance`` grid, so joining is linear in the
    number of paths (large curb or marking layers produce thousands of paths).
    """
    def key(point: tuple[float, float]) -> tuple[int, int]:
        return (int(round(point[0] / tolerance)), int(round(point[1] / tolerance)))

    stored: dict[int, list[tuple[float, float]]] = {index: list(path) for index, path in enumerate(paths) if len(path) >= 2}
    index: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for identifier, path in stored.items():
        index.setdefault(key(path[0]), set()).add((identifier, 0))
        index.setdefault(key(path[-1]), set()).add((identifier, -1))
    queue = [cell for cell, ends in index.items() if len(ends) == 2]
    next_identifier = len(paths)
    while queue:
        cell = queue.pop()
        ends = index.get(cell, set())
        if len(ends) != 2:
            continue
        (first_id, first_end), (second_id, second_end) = sorted(ends)
        if first_id == second_id:
            continue
        first, second = stored.pop(first_id), stored.pop(second_id)
        for identifier, path in ((first_id, first), (second_id, second)):
            index[key(path[0])].discard((identifier, 0))
            index[key(path[-1])].discard((identifier, -1))
        head = first if first_end == -1 else first[::-1]
        tail = second if second_end == 0 else second[::-1]
        merged = head + tail[1:]
        stored[next_identifier] = merged
        for end, point in ((0, merged[0]), (-1, merged[-1])):
            ends_here = index.setdefault(key(point), set())
            ends_here.add((next_identifier, end))
            if len(ends_here) == 2:
                queue.append(key(point))
        next_identifier += 1
    return list(stored.values())

def _sparse_line_fit(points: np.ndarray, min_length: float) -> list[dict[str, object]]:
    """Return a fitted polyline when sparse sampling cannot form a pixel skeleton."""
    values = np.asarray(points, dtype=np.float64)
    xy = values[:, :2]
    if len(xy) < 2:
        return []
    centre = xy.mean(axis=0)
    _, singular, right = np.linalg.svd(xy - centre, full_matrices=False)
    if not len(singular) or singular[0] <= 1e-12:
        return []
    direction = right[0]
    projected = (xy - centre) @ direction
    start = centre + direction * float(projected.min())
    end = centre + direction * float(projected.max())
    if float(np.linalg.norm(end - start)) < min_length:
        return []
    return [{"type": "polyline", "points": [start.tolist(), end.tolist()], "closed": False}]


MAX_CROWN_RADIUS = 8.0
"""Larger crown clusters are split; single crowns above 8 m radius are rare."""


def tree_entities(points: np.ndarray, codes: np.ndarray) -> list[dict[str, object]]:
    """Return one crown circle and one trunk point for every tree cluster."""
    values, class_values = np.asarray(points, dtype=np.float64), np.asarray(codes, dtype=np.uint8)
    crowns = values[np.isin(class_values, [4, 5])]
    trunks = values[class_values == 95]
    circles = circles_from_clusters(crowns, eps=1.5, minimum_radius=0.2, radius_percentile=90.0, maximum_radius=MAX_CROWN_RADIUS)
    result: list[dict[str, object]] = []
    for circle in circles:
        centre = np.asarray(circle["center"], dtype=np.float64)
        nearby = trunks[np.linalg.norm(trunks[:, :2] - centre, axis=1) <= float(circle["radius"]) + 0.5] if len(trunks) else np.empty((0, 3))
        result.append(circle)
        result.append({"type": "point", "point": (nearby[:, :2].mean(axis=0) if len(nearby) else centre).tolist()})
    return result


def _neighbours(node: tuple[int, int]) -> list[tuple[int, int]]:
    row, column = node
    return [(row + row_offset, column + column_offset) for row_offset in (-1, 0, 1) for column_offset in (-1, 0, 1) if row_offset or column_offset]


def _edge(first: tuple[int, int], second: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (first, second) if first < second else (second, first)
