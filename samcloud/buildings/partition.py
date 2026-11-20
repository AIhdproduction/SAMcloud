"""Split building footprints along roof-plane boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon
from shapely.ops import nearest_points, polygonize, unary_union

from samcloud.drone.contracts import RoofPlane
from samcloud.ortho.vector.regularize import dominant_direction


@dataclass
class Partition:
    """Footprint cells and directed shared-edge adjacency."""

    cells: list[Polygon]
    adjacency: list[tuple[int, int, float]]


def _height(plane: RoofPlane, x: float, y: float) -> float:
    if abs(float(plane.normal[2])) < 1e-10:
        return float("nan")
    return -float(plane.normal[0] * x + plane.normal[1] * y + plane.d) / float(plane.normal[2])


def _lines(geometry: object) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 1e-7 else []
    if isinstance(geometry, MultiLineString):
        return [line for line in geometry.geoms if line.length > 1e-7]
    if isinstance(geometry, GeometryCollection):
        answer: list[LineString] = []
        for item in geometry.geoms:
            answer.extend(_lines(item))
        return answer
    return []


def _extend(line: LineString, footprint: Polygon) -> LineString | None:
    coordinates = np.asarray(line.coords)
    if len(coordinates) < 2:
        return None
    vector = coordinates[-1] - coordinates[0]
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        return None
    vector /= length
    center = coordinates.mean(axis=0)
    extent = max(footprint.bounds[2] - footprint.bounds[0], footprint.bounds[3] - footprint.bounds[1]) * 3.0 + 1.0
    extended = LineString([tuple(center - vector * extent), tuple(center + vector * extent)])
    clipped = extended.intersection(footprint)
    candidates = _lines(clipped)
    return max(candidates, key=lambda item: item.length) if candidates else None


def _snap_direction(line: LineString, direction: float, tolerance: float = 10.0) -> LineString:
    points = np.asarray(line.coords)
    vector = points[-1] - points[0]
    angle = math.atan2(float(vector[1]), float(vector[0]))
    candidates = [direction + step * math.pi / 2.0 for step in range(4)]
    target = min(candidates, key=lambda item: abs((angle - item + math.pi / 2.0) % math.pi - math.pi / 2.0))
    error = abs((angle - target + math.pi / 2.0) % math.pi - math.pi / 2.0)
    if error > math.radians(tolerance):
        return line
    center = points.mean(axis=0)
    half = float(np.linalg.norm(vector)) * 0.5
    unit = np.array((math.cos(target), math.sin(target)))
    return LineString([tuple(center - unit * half), tuple(center + unit * half)])


def plane_lines(roof_planes: list[RoofPlane], footprint: Polygon) -> list[LineString]:
    """Return clipped ridge and height-step lines between adjacent roof planes."""
    if footprint.is_empty:
        return []
    result: list[LineString] = []
    direction = dominant_direction([footprint])
    for index, first in enumerate(roof_planes):
        if first.outline is None or first.outline.is_empty:
            continue
        for second in roof_planes[index + 1:]:
            if second.outline is None or second.outline.is_empty or first.outline.distance(second.outline) > 0.5:
                continue
            cross = np.cross(first.normal, second.normal)
            # Plane fits of two horizontal stepped roofs differ very slightly
            # because of measurement noise.  Treating them as intersecting
            # creates a line far outside the footprint and skips the outline
            # boundary that marks the actual height step.
            if np.linalg.norm(cross[:2]) > 1e-3:
                # One least-norm point on both 3-D planes, projected to XY.
                matrix = np.vstack((first.normal, second.normal))
                if np.linalg.matrix_rank(matrix) == 2:
                    point3d = np.linalg.lstsq(matrix, -np.array([first.d, second.d]), rcond=None)[0]
                    point = point3d[:2]
                    unit = cross[:2] / np.linalg.norm(cross[:2])
                    raw = LineString([tuple(point - unit), tuple(point + unit)])
                    clipped = _extend(_snap_direction(raw, direction), footprint)
                    if clipped is not None:
                        result.append(clipped)
                    continue
            shared = first.outline.boundary.intersection(second.outline.boundary)
            for boundary in _lines(shared):
                midpoint = boundary.interpolate(0.5, normalized=True)
                jump = abs(_height(first, midpoint.x, midpoint.y) - _height(second, midpoint.x, midpoint.y))
                if jump > 0.5:
                    clipped = _extend(_snap_direction(boundary, direction), footprint)
                    if clipped is not None:
                        result.append(clipped)
            if shared.is_empty:
                first_point, second_point = nearest_points(first.outline, second.outline)
                delta = np.array((second_point.x - first_point.x, second_point.y - first_point.y))
                length = float(np.linalg.norm(delta))
                if length > 1e-8:
                    midpoint = (np.array((first_point.x, first_point.y)) + np.array((second_point.x, second_point.y))) * 0.5
                    tangent = np.array((-delta[1], delta[0])) / length
                    raw = LineString([tuple(midpoint - tangent), tuple(midpoint + tangent)])
                    jump = abs(_height(first, midpoint[0], midpoint[1]) - _height(second, midpoint[0], midpoint[1]))
                    if jump > 0.5:
                        clipped = _extend(_snap_direction(raw, direction), footprint)
                        if clipped is not None:
                            result.append(clipped)
    unique: list[LineString] = []
    for line in result:
        if not any(line.hausdorff_distance(old) < 1e-5 for old in unique):
            unique.append(line)
    return unique


def _adjacency(cells: list[Polygon]) -> list[tuple[int, int, float]]:
    result: list[tuple[int, int, float]] = []
    for first, cell in enumerate(cells):
        for second in range(first + 1, len(cells)):
            shared = cell.boundary.intersection(cells[second].boundary).length
            if shared > 1e-8:
                result.append((first, second, float(shared)))
                result.append((second, first, float(shared)))
    return result


def partition_footprint(footprint: Polygon, lines: list[LineString], *, min_cell_area: float = 0.5) -> Partition:
    """Polygonize an extended roof-line arrangement and return cells with adjacency."""
    if footprint.is_empty:
        return Partition([], [])
    extended = [_extend(line, footprint) for line in lines]
    # Let cutters cross the footprint boundary very slightly.  Clipping a line
    # to a noisy facade-derived footprint can leave its endpoints a few microns
    # short of the boundary, which prevents polygonize from creating cells.
    noded: list[LineString] = []
    for line in extended:
        if line is None:
            continue
        coordinates = np.asarray(line.coords, dtype=float)
        vector = coordinates[-1] - coordinates[0]
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        unit = vector / length
        noded.append(LineString([tuple(coordinates[0] - unit * 1e-4), tuple(coordinates[-1] + unit * 1e-4)]))
    geometry = unary_union([footprint.boundary, *noded])
    cells = [poly.intersection(footprint) for poly in polygonize(geometry)]
    cells = [cell for cell in cells if isinstance(cell, Polygon) and cell.area > 1e-8]
    if not cells:
        cells = [footprint]
    changed = True
    while changed:
        changed = False
        for index, cell in enumerate(cells):
            if cell.area >= min_cell_area or len(cells) == 1:
                continue
            candidates = [(cell.boundary.intersection(other.boundary).length, other_index) for other_index, other in enumerate(cells) if other_index != index]
            length, neighbour = max(candidates, default=(0.0, -1))
            if length <= 1e-8:
                continue
            merged = cell.union(cells[neighbour])
            if not isinstance(merged, Polygon):
                continue
            cells[neighbour] = merged
            cells.pop(index)
            changed = True
            break
    cells.sort(key=lambda item: (item.centroid.x, item.centroid.y))
    return Partition(cells, _adjacency(cells))
