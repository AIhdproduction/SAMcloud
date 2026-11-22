"""Small geometry helpers of the room objects: components, planes, outlines, closed prism meshes, mirror images."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import shapely
from scipy import ndimage
from shapely import affinity
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import unary_union

Mesh = tuple[np.ndarray, np.ndarray]


def voxel_components(points: np.ndarray, cell: float) -> tuple[np.ndarray, int]:
    """Component id (0..count-1) per point: occupied voxels linked through their 26 neighbours."""
    if not len(points):
        return np.zeros(0, dtype=np.int64), 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    cells = np.floor((points - points.min(axis=0)) / cell).astype(np.int64) + 1
    dims = cells.max(axis=0) + 2
    codes, inverse = np.unique((cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2], return_inverse=True)
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                delta = (dx * dims[1] + dy) * dims[2] + dz
                if delta <= 0:
                    continue
                position = np.minimum(np.searchsorted(codes, codes + delta), len(codes) - 1)
                hit = codes[position] == codes + delta
                first.append(np.flatnonzero(hit))
                second.append(position[hit])
    rows, cols = np.concatenate(first), np.concatenate(second)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(codes), len(codes)))
    count, labels = connected_components(graph, directed=False)
    return labels[inverse.reshape(-1)].astype(np.int64), int(count)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(centre, unit normal, rms distance) of the least-squares plane."""
    centre = points.mean(axis=0)
    centred = points - centre
    values, vectors = np.linalg.eigh(centred.T @ centred / max(len(points), 1))
    return centre, vectors[:, 0], math.sqrt(max(float(values[0]), 0.0))


def upright_plane(points: np.ndarray, tolerance: float, max_tilt_deg: float, rounds: int = 300) -> np.ndarray | None:
    """Inlier mask of the best plane that is at most ``max_tilt_deg`` away from vertical (deterministic RANSAC)."""
    count = len(points)
    if count < 3:
        return None
    rng = np.random.default_rng(count)
    sample = points if count <= 4000 else points[np.sort(rng.choice(count, 4000, replace=False))]
    triples = rng.integers(0, len(sample), size=(rounds, 3))
    origin = sample[triples[:, 0]]
    normal = np.cross(sample[triples[:, 1]] - origin, sample[triples[:, 2]] - origin)
    length = np.linalg.norm(normal, axis=1)
    valid = length > 1e-6
    normal, origin = normal[valid] / length[valid, None], origin[valid]
    upright = np.abs(normal[:, 2]) <= math.sin(math.radians(max_tilt_deg))
    normal, origin = normal[upright], origin[upright]
    if not len(normal):
        return None
    best, best_score = 0, -1
    for index in range(len(normal)):
        score = int(np.count_nonzero(np.abs((sample - origin[index]) @ normal[index]) <= tolerance))
        if score > best_score:
            best, best_score = index, score
    inlier = np.abs((points - origin[best]) @ normal[best]) <= tolerance
    for _ in range(2):  # refine on the inliers
        if np.count_nonzero(inlier) < 3:
            return None
        centre, axis, _rms = fit_plane(points[inlier])
        if abs(float(axis[2])) > math.sin(math.radians(max_tilt_deg)):
            break
        inlier = np.abs((points - centre) @ axis) <= tolerance
    return inlier


def panel_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(horizontal axis u, upward axis v, unit normal) of an upright panel."""
    normal = normal / np.linalg.norm(normal)
    u = np.asarray((-normal[1], normal[0], 0.0))
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(normal, u)
    if v[2] < 0:
        v, u = -v, -u
    return u, v, normal


def ceiling_plane(points: np.ndarray, tolerance: float = 0.03, max_slope_deg: float = 55.0) -> np.ndarray | None:
    """(a, b, c) of ``z = a x + b y + c`` through the dominant upper surface; None without enough points."""
    if len(points) < 30:
        return None
    rng = np.random.default_rng(len(points))
    sample = points if len(points) <= 3000 else points[np.sort(rng.choice(len(points), 3000, replace=False))]
    xy_centre = sample[:, :2].mean(axis=0)
    design = np.column_stack((sample[:, 0] - xy_centre[0], sample[:, 1] - xy_centre[1], np.ones(len(sample))))
    limit = math.tan(math.radians(max_slope_deg))
    best = np.asarray((0.0, 0.0, float(np.median(sample[:, 2]))))
    edges = np.arange(sample[:, 2].min(), sample[:, 2].max() + 0.01, 0.01)
    if len(edges) > 2:
        counts, edges = np.histogram(sample[:, 2], bins=edges)
        peak = int(np.argmax(np.convolve(counts, np.ones(3), mode="same")))
        best[2] = 0.5 * float(edges[peak] + edges[peak + 1])
        best[2] = float(np.median(sample[np.abs(sample[:, 2] - best[2]) <= tolerance, 2]))
    level, level_score = best.copy(), int(np.count_nonzero(np.abs(sample[:, 2] - best[2]) <= tolerance))
    best_score = level_score
    for triple in rng.integers(0, len(sample), size=(150, 3)):
        matrix = design[triple]
        if abs(np.linalg.det(matrix)) < 1e-3:
            continue
        solution = np.linalg.solve(matrix, sample[triple, 2])
        if math.hypot(solution[0], solution[1]) > limit:
            continue
        score = int(np.count_nonzero(np.abs(sample[:, 2] - design @ solution) <= tolerance))
        if score > best_score:
            best, best_score = solution, score
    for _ in range(3):
        inlier = np.abs(sample[:, 2] - design @ best) <= tolerance
        if np.count_nonzero(inlier) < 15:
            break
        solution = np.linalg.lstsq(design[inlier], sample[inlier, 2], rcond=None)[0]
        if math.hypot(solution[0], solution[1]) > limit:
            break
        best = solution
    # a slope needs clearly more support than a level ceiling (a real one has many times more): glare under lamps tilts a free fit
    if int(np.count_nonzero(np.abs(sample[:, 2] - design @ best) <= tolerance)) < 1.15 * level_score:
        best = level
    return np.asarray((best[0], best[1], best[2] - best[0] * xy_centre[0] - best[1] * xy_centre[1]))


def rectangle_sides(polygon: Polygon) -> tuple[float, float, np.ndarray]:
    """(long side, short side, unit long axis) of the minimum rotated rectangle."""
    rectangle = polygon.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon):
        return 0.0, 0.0, np.asarray((1.0, 0.0))
    corners = np.asarray(rectangle.exterior.coords)[:3]
    first, second = corners[1] - corners[0], corners[2] - corners[1]
    a, b = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    if a >= b:
        return a, b, first / max(a, 1e-12)
    return b, a, second / max(b, 1e-12)


def hull_of(xy: np.ndarray) -> Polygon | None:
    if len(xy) < 3:
        return None
    hull = MultiPoint(np.asarray(xy, dtype=float)).convex_hull
    return hull if isinstance(hull, Polygon) and hull.area > 0 else None


def occupancy_outline(xy: np.ndarray, cell: float, density_share: float) -> Polygon | None:
    """Filled outline of where the points are dense; thin glare and stray points fall away.

    Cells below ``density_share`` of the typical density are dropped, small holes are closed, and of separate parts
    only those of at least a quarter of the largest one stay (arms of a lamp seen in pieces are bridged).
    """
    if len(xy) < 3:
        return None
    origin = xy.min(axis=0) - 5 * cell
    index = np.floor((xy - origin) / cell).astype(np.int64)
    counts = np.zeros(tuple(index.max(axis=0) + 6), dtype=np.float64)
    np.add.at(counts, (index[:, 0], index[:, 1]), 1.0)
    smooth = ndimage.uniform_filter(counts, 3)
    typical = float(np.percentile(smooth[counts > 0], 75))
    body = (counts > 0) & (smooth >= density_share * typical)
    disc = ndimage.generate_binary_structure(2, 1)
    body = ndimage.binary_closing(body, disc, iterations=2)
    body = ndimage.binary_fill_holes(body)
    labels, count = ndimage.label(body, np.ones((3, 3)))
    if not count:
        return None
    sizes = np.bincount(labels.ravel())[1:]
    squares = []
    for number in np.flatnonzero(sizes >= 0.25 * sizes.max()) + 1:
        cols, rows = np.nonzero(labels == number)
        x, y = origin[0] + cols * cell, origin[1] + rows * cell
        squares.extend(shapely.box(x, y, x + cell, y + cell))
    merged = unary_union(squares)
    for reach in (0.0, 0.03, 0.06, 0.12):
        closed = merged.buffer(reach, join_style=2).buffer(-reach, join_style=2) if reach else merged
        if isinstance(closed, Polygon):
            break
    else:
        closed = merged.convex_hull
    outline = Polygon(closed.exterior).simplify(0.75 * cell, preserve_topology=True)
    return outline if outline.is_valid and outline.area > 0 else None


def regular_outline(outline: Polygon) -> tuple[Polygon, str]:
    """A round or rectangular outline becomes a true circle or rectangle; anything else keeps its measured shape."""
    hull = outline.convex_hull
    long_side, short_side, _axis = rectangle_sides(outline)
    if long_side <= 0 or short_side <= 0:
        return outline, "outline"
    solidity = outline.area / hull.area
    rectangularity = outline.area / (long_side * short_side)
    if solidity >= 0.85 and long_side / short_side <= 1.35:
        radius = math.sqrt(hull.area / math.pi)
        circle = hull.centroid.buffer(radius, quad_segs=6)
        iou = circle.intersection(hull).area / circle.union(hull).area
        if iou >= 0.85 and (rectangularity < 0.90 or long_side <= 0.20):
            return circle, "circle"
    if rectangularity >= 0.80:
        rectangle = outline.minimum_rotated_rectangle
        scale = math.sqrt(outline.area / rectangle.area)
        shrunk = affinity.scale(rectangle, scale, scale, origin="centroid")
        return shrunk, "rectangle"
    return outline, "outline"


def prism_mesh(outline: Polygon, bottom: Callable[[np.ndarray], np.ndarray], top: Callable[[np.ndarray], np.ndarray]) -> Mesh | None:
    """Closed prism over a hole-free outline between two height functions (planes: ``f(xy) -> z``), normals outwards."""
    ring = np.asarray(shapely.geometry.polygon.orient(Polygon(outline.exterior), 1.0).exterior.coords[:-1], dtype=float)
    count = len(ring)
    if count < 3:
        return None
    position = {(float(x), float(y)): index for index, (x, y) in enumerate(ring)}
    caps: list[tuple[int, int, int]] = []
    for triangle in shapely.constrained_delaunay_triangles(Polygon(ring)).geoms:
        corners = [position.get((float(x), float(y))) for x, y in list(triangle.exterior.coords)[:3]]
        if None in corners or len(set(corners)) != 3:
            return None
        a, b, c = ring[corners]
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
            corners.reverse()
        caps.append(tuple(corners))
    low, high = np.column_stack((ring, bottom(ring))), np.column_stack((ring, top(ring)))
    if np.any(high[:, 2] - low[:, 2] < 1e-3):
        return None
    faces = [(a + count, b + count, c + count) for a, b, c in caps] + [(c, b, a) for a, b, c in caps]
    for index in range(count):
        following = (index + 1) % count
        faces += [(index, following, following + count), (index, following + count, index + count)]
    return np.vstack((low, high)), np.asarray(faces, dtype=np.int64)


def box_mesh(centre: np.ndarray, axes: Sequence[np.ndarray], half: Sequence[float]) -> Mesh:
    """Closed oriented box from its centre, three orthogonal unit axes and half sizes."""
    a, b, c = (np.asarray(axis, dtype=float) * size for axis, size in zip(axes, half, strict=True))
    if float(np.dot(np.cross(a, b), c)) < 0:
        a, b = b, a
    vertices = np.asarray([centre + sx * a + sy * b + sz * c for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
    faces = np.asarray([
        (0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
        (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5),
    ], dtype=np.int64)
    return vertices, faces


def join_meshes(meshes: Iterable[Mesh]) -> Mesh:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for points, triangles in meshes:
        vertices.append(points)
        faces.append(triangles + offset)
        offset += len(points)
    return np.vstack(vertices), np.vstack(faces)


def nearest_line(point_xy: np.ndarray, lines: list[LineString]) -> tuple[float, LineString | None]:
    if not lines:
        return math.inf, None
    point = shapely.Point(float(point_xy[0]), float(point_xy[1]))
    line = min(lines, key=lambda item: item.distance(point))
    return float(line.distance(point)), line


def line_direction(line: LineString) -> np.ndarray:
    start, end = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
    return (end - start) / max(float(np.linalg.norm(end - start)), 1e-12)


def reflection_of(candidates: list[dict[str, Any]], mirrors: list[Any], settings: Any) -> dict[int, int]:
    """{index of a reflection: index of the object it shows}.

    A reflection lies behind a mirror and its mirror image coincides with a real object of similar size that has at
    least as much evidence.  Candidates carry ``centre`` (3,), ``size`` (largest side) and ``points``.
    """
    found: dict[int, int] = {}
    for index, candidate in enumerate(candidates):
        centre = np.asarray(candidate["centre"], dtype=float)
        for mirror in mirrors:
            offset = centre - mirror.centre
            depth = float(offset @ mirror.normal)
            if depth >= -0.02:
                continue
            if float(np.linalg.norm((offset - depth * mirror.normal)[:2])) > settings.lateral_m:
                continue
            image = centre - 2.0 * depth * mirror.normal
            for other, real in enumerate(candidates):
                if other == index or other in found:
                    continue
                real_centre = np.asarray(real["centre"], dtype=float)
                if float((real_centre - mirror.centre) @ mirror.normal) <= 0.0:
                    continue
                if float(np.linalg.norm(real_centre - image)) > settings.match_m or real["points"] < candidate["points"]:
                    continue
                small, large = sorted((float(real["size"]), float(candidate["size"])))
                if small >= settings.size_ratio * large:
                    found[index] = other
                    break
            if index in found:
                break
    return found
