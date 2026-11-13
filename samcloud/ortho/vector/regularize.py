"""Direction-aware polygon regularisation for building and wall footprints."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

import numpy as np
from shapely import make_valid
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon


def dominant_direction(polygons: Iterable[Polygon] | Iterable[LineString]) -> float:
    """Return the length-weighted dominant edge direction modulo ninety degrees."""
    sine = 0.0
    cosine = 0.0
    for item in polygons:
        sequences: list[Sequence[tuple[float, float]]] = []
        if isinstance(item, Polygon):
            sequences.append(list(item.exterior.coords))
            sequences.extend(list(ring.coords) for ring in item.interiors)
        elif isinstance(item, LineString):
            sequences.append(list(item.coords))
        for sequence in sequences:
            for first, second in zip(sequence, sequence[1:], strict=False):
                dx, dy = second[0] - first[0], second[1] - first[1]
                length = math.hypot(dx, dy)
                if length <= 1e-9:
                    continue
                angle = math.atan2(dy, dx)
                cosine += length * math.cos(4.0 * angle)
                sine += length * math.sin(4.0 * angle)
    if abs(sine) + abs(cosine) <= 1e-12:
        return 0.0
    return (0.25 * math.atan2(sine, cosine)) % (math.pi / 2.0)


def regularize_polygon(
    poly: Polygon,
    *,
    angle_tolerance_deg: float = 15.0,
    min_edge_length: float = 0.1,
    direction: float | None = None,
) -> Polygon:
    """Simplify, snap near-orthogonal edges, intersect them, and return a valid polygon."""
    if poly.is_empty:
        return poly
    if min_edge_length <= 0.0:
        raise ValueError("min_edge_length must be positive")
    base_direction = _polygon_direction(poly) if direction is None else float(direction)
    exterior = _regularize_ring(list(poly.exterior.coords), base_direction, angle_tolerance_deg, min_edge_length)
    holes = [
        _regularize_ring(list(ring.coords), base_direction, angle_tolerance_deg, min_edge_length)
        for ring in poly.interiors
    ]
    candidate = Polygon(exterior, [ring for ring in holes if len(ring) >= 4])
    valid = make_valid(candidate)
    return _largest_polygon(valid, fallback=poly)


def regularize_polygons(
    polys: Iterable[Polygon],
    *,
    angle_tolerance_deg: float = 15.0,
    min_edge_length: float = 0.1,
) -> list[Polygon]:
    """Regularise connected polygon groups with one length-weighted direction each."""
    source = [polygon for polygon in polys if not polygon.is_empty]
    if not source:
        return []
    unseen = set(range(len(source)))
    output: list[Polygon] = []
    while unseen:
        seed = unseen.pop()
        group = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            touching = [index for index in unseen if source[current].buffer(1e-9).intersects(source[index].buffer(1e-9))]
            for index in touching:
                unseen.remove(index)
                group.add(index)
                queue.append(index)
        members = [source[index] for index in sorted(group)]
        shared = dominant_direction(members)
        output.extend(
            regularize_polygon(member, angle_tolerance_deg=angle_tolerance_deg, min_edge_length=min_edge_length, direction=shared)
            for member in members
        )
    return output


def _regularize_ring(
    coordinates: list[tuple[float, float]], direction: float, angle_tolerance_deg: float, min_edge_length: float
) -> list[tuple[float, float]]:
    if len(coordinates) < 4:
        return coordinates
    source = Polygon(coordinates).simplify(min_edge_length * 1.5, preserve_topology=True)
    points = list(source.exterior.coords)[:-1]
    points = _drop_short_vertices(points, min_edge_length)
    if len(points) < 3:
        return coordinates
    tolerance = math.radians(angle_tolerance_deg)
    lines: list[tuple[np.ndarray, np.ndarray, float]] = []
    for first, second in zip(points, points[1:] + points[:1], strict=False):
        start, end = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        vector = end - start
        angle = math.atan2(float(vector[1]), float(vector[0]))
        target, error = _nearest_orthogonal(angle, direction)
        used_angle = target if error <= tolerance else angle
        unit = np.asarray((math.cos(used_angle), math.sin(used_angle)), dtype=float)
        midpoint = (start + end) * 0.5
        lines.append((midpoint, unit, float(np.hypot(*vector))))
    lines = _merge_collinear_edges(lines, tolerance, min_edge_length)
    vertices: list[tuple[float, float]] = []
    for index, current in enumerate(lines):
        previous = lines[index - 1]
        intersection = _line_intersection(previous, current)
        if intersection is None or not np.isfinite(intersection).all():
            intersection = current[0]
        vertices.append((float(intersection[0]), float(intersection[1])))
    vertices = _drop_short_vertices(vertices, min_edge_length * 0.35)
    return vertices + [vertices[0]] if len(vertices) >= 3 else coordinates


def _drop_short_vertices(points: list[tuple[float, float]], threshold: float) -> list[tuple[float, float]]:
    changed = True
    result = points[:]
    while changed and len(result) > 3:
        changed = False
        for index in range(len(result)):
            previous, current, following = result[index - 1], result[index], result[(index + 1) % len(result)]
            if math.dist(previous, current) < threshold or math.dist(current, following) < threshold:
                result.pop(index)
                changed = True
                break
    return result


def _nearest_orthogonal(angle: float, direction: float) -> tuple[float, float]:
    candidates = [direction + step * math.pi / 2.0 for step in range(4)]
    best = min(candidates, key=lambda candidate: _angle_distance(angle, candidate))
    return best, _angle_distance(angle, best)


def _polygon_direction(poly: Polygon) -> float:
    """Use the oriented envelope for mostly rectangular raster footprints."""
    envelope = poly.minimum_rotated_rectangle
    rectangularity = poly.area / max(envelope.area, 1e-12)
    if rectangularity >= 0.8:
        coordinates = list(envelope.exterior.coords)[:-1]
        edges = [np.asarray(coordinates[(index + 1) % 4]) - np.asarray(point) for index, point in enumerate(coordinates)]
        edge = max(edges, key=lambda value: float(np.hypot(*value)))
        return math.atan2(float(edge[1]), float(edge[0])) % (math.pi / 2.0)
    return dominant_direction([poly])


def _angle_distance(first: float, second: float) -> float:
    return abs((first - second + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _line_intersection(
    first: tuple[np.ndarray, np.ndarray, float], second: tuple[np.ndarray, np.ndarray, float]
) -> np.ndarray | None:
    origin_a, vector_a, _length_a = first
    origin_b, vector_b, _length_b = second
    determinant = float(vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0])
    if abs(determinant) < 1e-8:
        return None
    delta = origin_b - origin_a
    factor = float((delta[0] * vector_b[1] - delta[1] * vector_b[0]) / determinant)
    return origin_a + factor * vector_a


def _merge_collinear_edges(
    lines: list[tuple[np.ndarray, np.ndarray, float]], tolerance: float, min_edge_length: float
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Merge consecutive parallel edges using a length-weighted normal offset."""
    if len(lines) < 2:
        return lines
    result = lines[:]
    changed = True
    parallel_tolerance = max(math.radians(1.0), tolerance * 0.25)
    while changed and len(result) > 3:
        changed = False
        for index in range(len(result)):
            first = result[index]
            second = result[(index + 1) % len(result)]
            if _angle_distance(math.atan2(first[1][1], first[1][0]), math.atan2(second[1][1], second[1][0])) > parallel_tolerance:
                continue
            unit = first[1]
            normal = np.asarray((-unit[1], unit[0]))
            offset_first = float(np.dot(first[0], normal))
            offset_second = float(np.dot(second[0], normal))
            if abs(offset_first - offset_second) > max(min_edge_length, 1e-6):
                continue
            length = first[2] + second[2]
            offset = (offset_first * first[2] + offset_second * second[2]) / length
            along = (float(np.dot(first[0], unit)) * first[2] + float(np.dot(second[0], unit)) * second[2]) / length
            merged = (unit * along + normal * offset, unit, length)
            result[index] = merged
            result.pop((index + 1) % len(result))
            changed = True
            break
    return result


def _largest_polygon(geometry: object, *, fallback: Polygon) -> Polygon:
    if isinstance(geometry, Polygon):
        return geometry
    candidates: list[Polygon] = []
    if isinstance(geometry, MultiPolygon):
        candidates = list(geometry.geoms)
    elif isinstance(geometry, GeometryCollection):
        candidates = [item for item in geometry.geoms if isinstance(item, Polygon)]
    return max(candidates, key=lambda item: item.area) if candidates else fallback
