"""Compact closed meshes of irregular objects (WC, washbasin, bathtub, lamp, plant, cat tree) from stacked slice hulls.

The cleaned points are cut into horizontal slices; every slice gets the convex hull of its points, sampled as a
star-shaped ring with a fixed number of vertices (rays from the slice centre).  Consecutive rings are lofted into quads,
the lowest and highest ring are closed with fans.  The result is watertight, has consistent outward normals, no
degenerate faces and a bounded triangle count (2 * K * slices).  Nearly identical consecutive rings are merged.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from samcloud.bim.furniture_recon.geometry import Mesh


def _hull_ring(xy: np.ndarray, centre: np.ndarray, count: int) -> np.ndarray | None:
    """Radius of the convex hull of ``xy`` along ``count`` rays from ``centre`` (ring of 2D points)."""
    try:
        hull = ConvexHull(xy)
    except (QhullError, ValueError):
        return None
    vertices = xy[hull.vertices]
    rel = vertices - centre
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    ring = np.empty((count, 2))
    edges = np.column_stack((rel, np.roll(rel, -1, axis=0)))
    for index, d in enumerate(directions):
        best = None
        for x0, y0, x1, y1 in edges:
            # ray t*d intersects segment p0 + s*(p1-p0)
            e = np.asarray((x1 - x0, y1 - y0))
            denominator = d[0] * (-e[1]) - d[1] * (-e[0])
            if abs(denominator) < 1e-12:
                continue
            t = (x0 * (-e[1]) - y0 * (-e[0])) / denominator
            s = (d[0] * y0 - d[1] * x0) / denominator
            if t > 0 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t > best):
                best = t
        ring[index] = centre + d * (best if best is not None else 0.0)
    return ring


def slice_rings(points: np.ndarray, z0: float, z1: float, slice_m: float, count: int, *, min_points: int = 6, counts: list[int] | None = None) -> list[tuple[float, np.ndarray]]:
    """(z, ring) per slice; slices with too few points are bridged by their neighbours (``counts`` receives the point counts)."""
    height = z1 - z0
    slices = int(min(max(math.ceil(height / max(slice_m, 1e-3)), 3), 40))
    edges = np.linspace(z0, z1, slices + 1)
    rings: list[tuple[float, np.ndarray]] = []
    for index in range(slices):
        lo, hi = edges[index], edges[index + 1]
        member = points[(points[:, 2] >= lo - 1e-9) & (points[:, 2] <= hi + 1e-9)]
        if len(member) < min_points:
            continue
        xy = member[:, :2]
        # trim stray points of the slice (2D): farther than 1.5 x the p90 distance from the median
        centre = np.median(xy, axis=0)
        distance = np.hypot(*(xy - centre).T)
        limit = 1.5 * float(np.percentile(distance, 90)) + 1e-3
        xy = xy[distance <= limit]
        if len(xy) < 3:
            continue
        ring = _hull_ring(xy, xy.mean(axis=0), count)
        if ring is None:
            # degenerate (collinear) slice: a thin rectangle around it
            spread = np.ptp(xy, axis=0)
            if float(spread.max()) <= 0:
                continue
            ring = _hull_ring(np.vstack((xy, xy + 0.005)), xy.mean(axis=0) + 0.0025, count)
            if ring is None:
                continue
        rings.append(((lo + hi) / 2.0, ring))
        if counts is not None:
            counts.append(int(len(member)))
    return rings


def _ring_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def trim_terminal_rings(rings: list[tuple[float, np.ndarray]], counts: list[int], *, area_share: float = 0.10, count_share: float = 0.02) -> tuple[list[tuple[float, np.ndarray]], list[int]]:
    """Drop tiny slices at the top and the bottom (a tap above a bathtub rim, a stray point column under a chair): a
    terminal slice whose hull is below ``area_share`` of the median slice area and that holds few points."""
    if len(rings) <= 3:
        return rings, counts
    areas = [_ring_area(ring) for _, ring in rings]
    median = float(np.median(areas))
    total = max(sum(counts), 1)
    rings, counts, areas = list(rings), list(counts), list(areas)
    while len(rings) > 3 and areas[-1] < area_share * median and counts[-1] < count_share * total:
        rings.pop(); counts.pop(); areas.pop()
    while len(rings) > 3 and areas[0] < area_share * median and counts[0] < count_share * total:
        rings.pop(0); counts.pop(0); areas.pop(0)
    return rings, counts


def smooth_rings(rings: list[tuple[float, np.ndarray]]) -> list[tuple[float, np.ndarray]]:
    """Moving median of three consecutive rings (vertex-wise): removes single-slice spikes, keeps the end rings."""
    if len(rings) < 3:
        return rings
    stacked = np.stack([ring for _, ring in rings])
    smoothed = stacked.copy()
    smoothed[1:-1] = np.median(np.stack((stacked[:-2], stacked[1:-1], stacked[2:])), axis=0)
    return [(z, smoothed[index]) for index, (z, _) in enumerate(rings)]


def _merge_similar(rings: list[tuple[float, np.ndarray]], tolerance: float) -> list[tuple[float, np.ndarray]]:
    if len(rings) <= 2:
        return rings
    kept = [rings[0]]
    for index in range(1, len(rings) - 1):
        z, ring = rings[index]
        previous = kept[-1][1]
        following = rings[index + 1][1]
        # drop a ring that is (nearly) the linear interpolation of its neighbours
        t = (z - kept[-1][0]) / max(rings[index + 1][0] - kept[-1][0], 1e-9)
        expected = previous + t * (following - previous)
        if float(np.abs(ring - expected).max()) > tolerance:
            kept.append(rings[index])
    kept.append(rings[-1])
    return kept


def loft_mesh(rings: list[tuple[float, np.ndarray]], z_bottom: float, z_top: float) -> Mesh | None:
    """Closed mesh through the rings; the first ring is moved to ``z_bottom``, the last to ``z_top``."""
    if len(rings) < 1:
        return None
    if len(rings) == 1:
        rings = [(z_bottom, rings[0][1]), (z_top, rings[0][1])]
    count = len(rings[0][1])
    levels = [z for z, _ in rings]
    levels[0], levels[-1] = z_bottom, z_top
    vertices = [np.column_stack((ring, np.full(count, z))) for (_, ring), z in zip(rings, levels)]
    stack = np.vstack(vertices)
    bottom_centre = np.append(rings[0][1].mean(axis=0), z_bottom)
    top_centre = np.append(rings[-1][1].mean(axis=0), z_top)
    stack = np.vstack((stack, bottom_centre, top_centre))
    b_index, t_index = len(stack) - 2, len(stack) - 1
    faces: list[tuple[int, int, int]] = []
    for level in range(len(rings) - 1):
        a0, b0 = level * count, (level + 1) * count
        for i in range(count):
            j = (i + 1) % count
            faces.append((a0 + i, a0 + j, b0 + j))
            faces.append((a0 + i, b0 + j, b0 + i))
    top0 = (len(rings) - 1) * count
    for i in range(count):
        j = (i + 1) % count
        faces.append((b_index, j, i))
        faces.append((t_index, top0 + i, top0 + j))
    mesh = Mesh(stack, np.asarray(faces, dtype=np.int64))
    return clean_mesh(mesh)


def clean_mesh(mesh: Mesh) -> Mesh | None:
    """Drop degenerate triangles, merge duplicate vertices, make the normals point outwards."""
    vertices = np.round(mesh.vertices, 6)
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[mesh.faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[keep]
    tri = unique[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    faces = faces[area > 1e-10]
    if len(faces) < 4:
        return None
    used = np.unique(faces)
    remap = -np.ones(len(unique), dtype=np.int64)
    remap[used] = np.arange(len(used))
    result = Mesh(unique[used], remap[faces])
    if result.volume() < 0:
        result = Mesh(result.vertices, result.faces[:, ::-1].copy())
    return result


def is_watertight(mesh: Mesh) -> bool:
    """Every edge is shared by exactly two faces with opposite orientation."""
    edges = np.vstack((mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]))
    forward = {tuple(edge) for edge in edges.tolist()}
    if len(forward) != len(edges):
        return False
    return all((b, a) in forward for a, b in forward)


def fit_mesh(points: np.ndarray, *, z_bottom: float, z_top: float, ring_vertices: int = 24, slice_m: float = 0.05, max_triangles: int = 3000) -> Mesh | None:
    """Stacked slice-hull mesh of ``points`` between ``z_bottom`` and ``z_top`` (metres, item frame)."""
    if len(points) < 20 or z_top - z_bottom < 0.02:
        return None
    count = int(max(8, ring_vertices))
    point_counts: list[int] = []
    rings = slice_rings(points, z_bottom, z_top, slice_m, count, counts=point_counts)
    if not rings:
        return None
    slice_height = (z_top - z_bottom) / int(min(max(math.ceil((z_top - z_bottom) / max(slice_m, 1e-3)), 3), 40))
    kept_top, kept_bottom = rings[-1][0], rings[0][0]
    rings, point_counts = trim_terminal_rings(rings, point_counts)
    if rings[-1][0] < kept_top:
        z_top = rings[-1][0] + slice_height / 2.0
    if rings[0][0] > kept_bottom and z_bottom > 0.0:
        # a grounded object (z_bottom 0) keeps standing on the floor; a hanging one starts at its lowest kept slice
        z_bottom = rings[0][0] - slice_height / 2.0
    rings = smooth_rings(rings)
    rings = _merge_similar(rings, tolerance=0.01)
    while 2 * count * (len(rings) + 1) > max_triangles and len(rings) > 3:
        rings = rings[::2] if len(rings) % 2 else rings[:-1:2] + [rings[-1]]
    return loft_mesh(rings, z_bottom, z_top)
