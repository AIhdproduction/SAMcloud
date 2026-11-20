"""Watertight LoD building mesh extrusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import triangulate, unary_union

from samcloud.buildings.partition import Partition
from samcloud.drone.contracts import RoofPlane


def _ground(ground_z: float | Callable, x: float, y: float) -> float:
    if not callable(ground_z):
        return float(ground_z)
    try:
        return float(ground_z(x, y))
    except TypeError:
        return float(ground_z((x, y)))


_XY_TOLERANCE = 1e-6
_WELD_TOLERANCE = 1e-3
_NODE_TOLERANCE = 0.02


def _xy_key(point: np.ndarray) -> tuple[int, int]:
    return tuple(np.rint(np.asarray(point, dtype=float) / _XY_TOLERANCE).astype(int))


def _vertex_key(vertex: np.ndarray, tolerance: float = _WELD_TOLERANCE) -> tuple[int, int, int]:
    return tuple(np.rint(np.asarray(vertex, dtype=float) / tolerance).astype(int))


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray, tolerance: float = _XY_TOLERANCE) -> tuple[bool, float]:
    vector = end - start
    squared_length = float(np.dot(vector, vector))
    if squared_length <= tolerance * tolerance:
        return (float(np.linalg.norm(point - start)) <= tolerance, 0.0)
    position = float(np.dot(point - start, vector) / squared_length)
    closest = start + min(1.0, max(0.0, position)) * vector
    return (float(np.linalg.norm(point - closest)) <= tolerance, position)


def _node_points(points: list[np.ndarray], tolerance: float = _NODE_TOLERANCE) -> list[np.ndarray]:
    """Coalesce small partitioning jitter before splitting the common graph."""
    result: list[np.ndarray] = []
    for point in points:
        point = np.asarray(point, dtype=float)
        if not any(np.linalg.norm(point - known) <= tolerance for known in result):
            result.append(point)
    return result


def _split_ring(ring: np.ndarray, all_points: list[np.ndarray], tolerance: float = _NODE_TOLERANCE) -> list[np.ndarray]:
    """Split every ring edge at all known partition and footprint vertices."""
    split: list[np.ndarray] = []
    for start, end in zip(ring, np.roll(ring, -1, axis=0), strict=False):
        candidates: list[tuple[float, np.ndarray]] = []
        for point in all_points:
            on_segment, position = _point_on_segment(point, start, end, tolerance)
            if on_segment:
                candidates.append((position, point))
        candidates.sort(key=lambda item: item[0])
        previous: tuple[int, int] | None = None
        for _position, point in candidates:
            key = _xy_key(point)
            if key != previous:
                split.append(np.asarray(point, dtype=float))
                previous = key
    if len(split) > 1 and _xy_key(split[0]) == _xy_key(split[-1]):
        split.pop()
    return split


def _signed_area(ring: list[np.ndarray]) -> float:
    coordinates = np.asarray(ring, dtype=float)
    return float(np.dot(coordinates[:, 0], np.roll(coordinates[:, 1], -1)) - np.dot(coordinates[:, 1], np.roll(coordinates[:, 0], -1))) / 2.0


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _inside_triangle(point: np.ndarray, first: np.ndarray, second: np.ndarray, third: np.ndarray) -> bool:
    def cross(left: np.ndarray, right: np.ndarray) -> float:
        return _cross2(left, right)

    values = (cross(second - first, point - first), cross(third - second, point - second), cross(first - third, point - third))
    return min(values) >= -1e-10 or max(values) <= 1e-10


def _triangulate_ring(ring: list[np.ndarray]) -> list[tuple[int, int, int]]:
    """Ear-clip a simple polygon while preserving every supplied boundary vertex."""
    if len(ring) < 3:
        return []
    indices = list(range(len(ring)))
    if _signed_area(ring) < 0:
        indices.reverse()
    triangles: list[tuple[int, int, int]] = []
    while len(indices) > 3:
        found_ear = False
        for position, middle in enumerate(indices):
            first, third = indices[position - 1], indices[(position + 1) % len(indices)]
            left, center, right = ring[first], ring[middle], ring[third]
            if _cross2(center - left, right - center) <= 1e-10:
                continue
            if any(_inside_triangle(ring[other], left, center, right) for other in indices if other not in (first, middle, third)):
                continue
            triangles.append((first, middle, third))
            indices.pop(position)
            found_ear = True
            break
        if not found_ear:
            for position, middle in enumerate(indices):
                first, third = indices[position - 1], indices[(position + 1) % len(indices)]
                if abs(_cross2(ring[middle] - ring[first], ring[third] - ring[middle])) <= 1e-10:
                    indices.pop(position)
                    break
            else:
                raise ValueError("could not triangulate partition cell")
    triangles.append(tuple(indices))
    return triangles


def _triangles(polygon: Polygon) -> list[np.ndarray]:
    """Triangulate a simple footprint polygon without optional earcut."""
    result: list[np.ndarray] = []
    for triangle in triangulate(polygon):
        if polygon.covers(triangle.representative_point()) and triangle.area > 1e-10:
            result.append(np.asarray(triangle.exterior.coords[:-1], dtype=float))
    return result


def _lines(geometry) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 1e-8 else []
    if isinstance(geometry, MultiLineString):
        return [line for line in geometry.geoms if line.length > 1e-8]
    return [part for part in getattr(geometry, "geoms", []) if isinstance(part, LineString) and part.length > 1e-8]


def _weld(vertices: list[np.ndarray], faces: list[tuple[int, int, int]], roles: list[int], tolerance: float = _WELD_TOLERANCE):
    lookup: dict[tuple[int, int, int], int] = {}
    welded: list[np.ndarray] = []
    inverse = np.empty(len(vertices), dtype=int)
    for index, vertex in enumerate(vertices):
        key = _vertex_key(vertex, tolerance)
        inverse[index] = lookup.setdefault(key, len(welded))
        if inverse[index] == len(welded):
            welded.append(np.asarray(vertex, dtype=float))
    compact_faces, compact_roles = [], []
    seen: set[tuple[int, int, int]] = set()
    for face, role in zip(faces, roles, strict=False):
        compact = tuple(int(inverse[value]) for value in face)
        if len(set(compact)) < 3:
            continue
        key = tuple(sorted(compact))
        if key in seen:
            continue
        seen.add(key)
        compact_faces.append(compact)
        compact_roles.append(role)
    return np.asarray(welded, dtype=float), np.asarray(compact_faces, dtype=int), np.asarray(compact_roles, dtype=np.uint8)


def _orient_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if not len(faces):
        return faces
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        trimesh.repair.fix_normals(mesh, multibody=True)
        return np.asarray(mesh.faces, dtype=int)
    except Exception:
        return faces


def _seal_open_edges(vertices: np.ndarray, faces: np.ndarray, roles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close residual boundary loops left by nearly coincident partition segments."""
    edge_faces: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for first, second, third in faces:
        edge_faces[tuple(sorted((int(first), int(second))))].append((int(first), int(second)))
        edge_faces[tuple(sorted((int(second), int(third))))].append((int(second), int(third)))
        edge_faces[tuple(sorted((int(third), int(first))))].append((int(third), int(first)))
    boundary = [entries[0] for entries in edge_faces.values() if len(entries) == 1]
    if not boundary:
        return vertices, faces, roles
    starts: dict[int, list[int]] = defaultdict(list)
    for index, (first, _second) in enumerate(boundary):
        starts[first].append(index)
    unused = set(range(len(boundary)))
    extra_vertices: list[np.ndarray] = []
    extra_faces: list[tuple[int, int, int]] = []
    extra_roles: list[int] = []
    while unused:
        first_index = next(iter(unused))
        first, second = boundary[first_index]
        loop = [(first, second)]
        unused.remove(first_index)
        current = second
        while current != first:
            candidates = [index for index in starts[current] if index in unused]
            if not candidates:
                loop = []
                break
            index = candidates[0]
            edge = boundary[index]
            loop.append(edge)
            unused.remove(index)
            current = edge[1]
        if len(loop) < 3:
            continue
        cycles = [loop]
        split = True
        while split:
            split = False
            for cycle_index, cycle in enumerate(cycles):
                nodes = [edge[0] for edge in cycle]
                repeated: tuple[int, int] | None = None
                for left, node in enumerate(nodes):
                    for right in range(left + 3, len(nodes)):
                        if nodes[right] == node:
                            repeated = (left, right)
                            break
                    if repeated is not None:
                        break
                if repeated is None:
                    continue
                left, right = repeated
                cycles[cycle_index : cycle_index + 1] = [cycle[left:right], cycle[right:] + cycle[:left]]
                split = True
                break
        for cycle in cycles:
            if len(cycle) < 3:
                continue
            node_indices = np.asarray([edge[0] for edge in cycle], dtype=int)
            center = len(vertices) + len(extra_vertices)
            extra_vertices.append(vertices[node_indices].mean(axis=0))
            for start, end in cycle:
                extra_faces.append((center, start, end))
                extra_roles.append(1)
    if not extra_faces:
        return vertices, faces, roles
    return (
        np.vstack((vertices, np.asarray(extra_vertices, dtype=float))),
        np.vstack((faces, np.asarray(extra_faces, dtype=int))),
        np.concatenate((roles, np.asarray(extra_roles, dtype=np.uint8))),
    )


def _extrude(partition: Partition, top, footprint: Polygon, ground_z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    roles: list[int] = []

    def add(vertex) -> int:
        vertices.append(np.asarray(vertex, dtype=float))
        return len(vertices) - 1

    def triangle(a, b, c, role: int) -> None:
        faces.append((add(a), add(b), add(c)))
        roles.append(role)

    # Roofs point up; the ground face is deliberately reversed to point down.
    for index, cell in enumerate(partition.cells):
        for coords in _triangles(cell):
            lifted = [np.array((x, y, top(index, x, y)), dtype=float) for x, y in coords]
            triangle(*lifted, 2)
    # Split the ground along the same cells as the roof.  This avoids T-junctions
    # where a partition edge reaches the exterior wall.
    for cell in partition.cells:
        for coords in _triangles(cell):
            bottom = [np.array((x, y, _ground(ground_z, x, y)), dtype=float) for x, y in coords]
            triangle(bottom[0], bottom[2], bottom[1], 0)

    # One façade polygon per original footprint edge.  Its top profile can contain
    # a vertical jump, which is essential where a stepped annex reaches the façade.
    ring = np.asarray(footprint.exterior.coords, dtype=float)
    for start, end in zip(ring[:-1], ring[1:], strict=False):
        edge = LineString([start, end])
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 1e-8:
            continue
        direction = vector / length
        pieces: list[tuple[float, float, int]] = []
        for index, cell in enumerate(partition.cells):
            shared = edge.intersection(cell.boundary)
            for line in _lines(shared):
                coordinates = np.asarray(line.coords, dtype=float)
                if len(coordinates) < 2:
                    continue
                first_t = float(np.dot(coordinates[0] - start, direction))
                second_t = float(np.dot(coordinates[-1] - start, direction))
                if abs(first_t - second_t) > 1e-8:
                    pieces.append((min(first_t, second_t), max(first_t, second_t), index))
        pieces.sort(key=lambda item: item[0])
        if not pieces:
            continue
        profile = [(0.0, _ground(ground_z, *start))]
        for _first_t, second_t, _index in pieces:
            xy = start + second_t * direction
            profile.append((second_t, _ground(ground_z, *xy)))
        for first_t, second_t, index in reversed(pieces):
            profile.append((second_t, top(index, *(start + second_t * direction))))
            profile.append((first_t, top(index, *(start + first_t * direction))))
        wall = Polygon(profile)
        for local in _triangles(wall):
            lifted = [np.array((*((start + direction * value[0]).tolist()), value[1])) for value in local]
            triangle(*lifted, 1)

    seen_pairs: set[tuple[int, int]] = set()
    for first, second, _length in partition.adjacency:
        key = (min(first, second), max(first, second))
        if first == second or key in seen_pairs:
            continue
        seen_pairs.add(key)
        shared = partition.cells[first].boundary.intersection(partition.cells[second].boundary)
        for line in _lines(shared):
            coordinates = np.asarray(line.coords, dtype=float)
            for start, end in zip(coordinates[:-1], coordinates[1:], strict=False):
                h_a0, h_a1 = top(first, *start), top(first, *end)
                h_b0, h_b1 = top(second, *start), top(second, *end)
                if max(abs(h_a0 - h_b0), abs(h_a1 - h_b1)) <= 1e-5:
                    continue
                low0, high0 = sorted((h_a0, h_b0))
                low1, high1 = sorted((h_a1, h_b1))
                a = np.array((start[0], start[1], low0))
                b = np.array((end[0], end[1], low1))
                ta = np.array((start[0], start[1], high0))
                tb = np.array((end[0], end[1], high1))
                triangle(a, b, tb, 1)
                triangle(a, tb, ta, 1)
    vertices_array, faces_array, roles_array = _weld(vertices, faces, roles)
    return vertices_array, _orient_outward(vertices_array, faces_array), roles_array


def _extrude_shared(partition: Partition, top, footprint: Polygon, ground_z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one closed shell from a shared, fully noded partition boundary graph."""
    original_rings = [np.asarray(cell.exterior.coords[:-1], dtype=float) for cell in partition.cells]
    all_points = list(np.asarray(footprint.exterior.coords[:-1], dtype=float))
    all_points.extend(point for ring in original_rings for point in ring)
    points = _node_points(all_points)
    rings = [_split_ring(ring, points) for ring in original_rings]
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    roles: list[int] = []
    vertex_lookup: dict[tuple[int, int, int], int] = {}

    def add(vertex: np.ndarray) -> int:
        key = _vertex_key(vertex)
        index = vertex_lookup.get(key)
        if index is None:
            index = len(vertices)
            vertices.append(np.asarray(vertex, dtype=float))
            vertex_lookup[key] = index
        return index

    def face(first: int, second: int, third: int, role: int) -> None:
        if len({first, second, third}) == 3:
            faces.append((first, second, third))
            roles.append(role)

    roof_vertices: list[list[int]] = []
    ground_vertices: list[list[int]] = []
    edge_owners: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int, int]]] = defaultdict(list)
    for cell_index, ring in enumerate(rings):
        roof = [add(np.array((point[0], point[1], top(cell_index, *point)), dtype=float)) for point in ring]
        ground = [add(np.array((point[0], point[1], _ground(ground_z, *point)), dtype=float)) for point in ring]
        roof_vertices.append(roof)
        ground_vertices.append(ground)
        for first, second, third in _triangulate_ring(ring):
            face(roof[first], roof[second], roof[third], 2)
            face(ground[first], ground[third], ground[second], 0)
        for position in range(len(ring)):
            next_position = (position + 1) % len(ring)
            edge_key = tuple(sorted((_xy_key(ring[position]), _xy_key(ring[next_position]))))
            edge_owners[edge_key].append((cell_index, position, next_position))

    for owners in edge_owners.values():
        if len(owners) == 1:
            continue
        for owner_index, (first_cell, first_start, first_end) in enumerate(owners):
            for second_cell, second_start, second_end in owners[owner_index + 1:]:
                a0, a1 = roof_vertices[first_cell][first_start], roof_vertices[first_cell][first_end]
                b0, b1 = roof_vertices[second_cell][second_start], roof_vertices[second_cell][second_end]
                if _xy_key(rings[first_cell][first_start]) == _xy_key(rings[second_cell][second_end]):
                    b0, b1 = b1, b0
                face(a0, a1, b1, 1)
                face(a0, b1, b0, 1)

    # A single façade surface per footprint edge shares every height transition
    # at cell junctions.  Independent cell quads would leave open vertical
    # T-junctions wherever a roof step meets the outside wall.
    footprint_ring = np.asarray(footprint.exterior.coords, dtype=float)
    for start, end in zip(footprint_ring[:-1], footprint_ring[1:], strict=False):
        edge = LineString([start, end])
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 1e-8:
            continue
        direction = vector / length
        pieces: list[tuple[float, float, int]] = []
        for owners in edge_owners.values():
            if len(owners) != 1:
                continue
            cell_index, first, second = owners[0]
            first_point, second_point = rings[cell_index][first], rings[cell_index][second]
            segment = LineString([first_point, second_point])
            if edge.distance(segment) > _NODE_TOLERANCE:
                continue
            first_t = float(np.dot(first_point - start, direction))
            second_t = float(np.dot(second_point - start, direction))
            if max(first_t, second_t) < -_NODE_TOLERANCE or min(first_t, second_t) > length + _NODE_TOLERANCE:
                continue
            first_t, second_t = max(0.0, min(length, first_t)), max(0.0, min(length, second_t))
            if abs(first_t - second_t) > 1e-8:
                pieces.append((min(first_t, second_t), max(first_t, second_t), cell_index))
        pieces.sort(key=lambda item: item[0])
        if not pieces:
            continue
        profile = [(0.0, _ground(ground_z, *start))]
        for _first_t, second_t, _cell_index in pieces:
            profile.append((second_t, _ground(ground_z, *(start + second_t * direction))))
        for first_t, second_t, cell_index in reversed(pieces):
            profile.append((second_t, top(cell_index, *(start + second_t * direction))))
            profile.append((first_t, top(cell_index, *(start + first_t * direction))))
        for triangle in _triangles(Polygon(profile)):
            indices = [add(np.array((*((start + direction * value[0]).tolist()), value[1]), dtype=float)) for value in triangle]
            face(*indices, 1)

    vertices_array, faces_array, roles_array = _weld(vertices, faces, roles)
    faces_array = _orient_outward(vertices_array, faces_array)
    vertices_array, faces_array, roles_array = _seal_open_edges(vertices_array, faces_array, roles_array)
    return vertices_array, _orient_outward(vertices_array, faces_array), roles_array


def extrude_lod22(
    partition: Partition, labels: np.ndarray, roof_planes: list[RoofPlane], footprint: Polygon, ground_z: float | Callable
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrude roof-labelled partition cells into an LoD2.2 triangular mesh."""
    labels = np.asarray(labels, dtype=int)

    def top(index: int, x: float, y: float) -> float:
        label = int(labels[index]) if index < len(labels) else 0
        if label <= 0 or label > len(roof_planes):
            return _ground(ground_z, x, y)
        plane = roof_planes[label - 1]
        return -float(plane.normal[0] * x + plane.normal[1] * y + plane.d) / float(plane.normal[2])

    return _extrude_shared(partition, top, footprint, ground_z)


def extrude_lod13(partition: Partition, cell_heights: np.ndarray, ground_z: float | Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrude partition cells as flat LoD1.3 roof slabs."""
    heights = np.asarray(cell_heights, dtype=float)
    footprint = partition.cells[0].union(partition.cells[0]) if len(partition.cells) == 1 else None
    if footprint is None:
        from shapely.ops import unary_union

        footprint = unary_union(partition.cells)
    if not isinstance(footprint, Polygon):
        footprint = max((item for item in footprint.geoms if isinstance(item, Polygon)), key=lambda item: item.area)
    return _extrude_shared(partition, lambda index, _x, _y: float(heights[index]), footprint, ground_z)


def extrude_lod1(footprint: Polygon, height: float, ground_z: float | Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrude a single flat footprint as an LoD1.2 mesh."""
    return _extrude_shared(Partition([footprint], []), lambda _index, _x, _y: float(height), footprint, ground_z)
