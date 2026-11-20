"""3D curb axes from CURB points and a straight/arc splitter for the IFC writer."""
from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from samcloud.drone import contracts as c

LOGGER = logging.getLogger(__name__)


def douglas_peucker_3d(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker thinning of an (M,3) polyline using 3D point-to-segment distances."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= 2:
        return pts.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        seg_len2 = float(seg @ seg)
        rel = pts[a + 1 : b] - pts[a]
        if seg_len2 < 1e-18:
            dist = np.linalg.norm(rel, axis=1)
        else:
            t = np.clip((rel @ seg) / seg_len2, 0.0, 1.0)
            dist = np.linalg.norm(rel - t[:, None] * seg, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            idx = a + 1 + k
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return pts[keep]


def _components(points: np.ndarray, eps: float) -> list[np.ndarray]:
    """Index arrays of the connected components of the eps-neighbour graph."""
    n = len(points)
    if n == 0:
        return []
    tree = cKDTree(points)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    count, labels = csgraph.connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(count + 1))
    return [order[bounds[i] : bounds[i + 1]] for i in range(count)]


def _trace_cluster(
    pts: np.ndarray, eps: float, step: float, branch_offset: float
) -> tuple[np.ndarray | None, np.ndarray]:
    """Main axis (sampled every ``step``) of one cluster and the indices left over (branches)."""
    # Thin the cloud for the graph search; assignment of all points happens afterwards.
    voxel = eps / 4.0
    keys = np.floor(pts / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    thin = pts[np.sort(first)]
    tree = cKDTree(thin)
    graph = tree.sparse_distance_matrix(tree, eps * 1.5, output_type="coo_matrix").tocsr()

    d0 = csgraph.dijkstra(graph, directed=False, indices=0)
    finite = np.isfinite(d0)
    a = int(np.argmax(np.where(finite, d0, -1.0)))
    dist_a, pred = csgraph.dijkstra(graph, directed=False, indices=a, return_predecessors=True)
    b = int(np.argmax(np.where(np.isfinite(dist_a), dist_a, -1.0)))
    path = [b]
    while path[-1] != a:
        nxt = int(pred[path[-1]])
        if nxt < 0:
            break
        path.append(nxt)
    path_idx = np.asarray(path[::-1])
    if len(path_idx) < 2:
        return None, np.arange(len(pts))

    # Densify the path (nodes can be eps apart) so the branch test measures the real offset.
    nodes, node_s = thin[path_idx], dist_a[path_idx]
    seg_len = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    reps = np.maximum(np.ceil(seg_len / (voxel / 2.0)).astype(int), 1)
    path_pts = [nodes[:1]]
    path_s = [node_s[:1]]
    for k, m in enumerate(reps):
        u = (np.arange(1, m + 1) / m)[:, None]
        path_pts.append(nodes[k] * (1 - u) + nodes[k + 1] * u)
        path_s.append((node_s[k] * (1 - u) + node_s[k + 1] * u)[:, 0])
    path_pts = np.vstack(path_pts)
    path_s = np.concatenate(path_s)
    ptree = cKDTree(path_pts)
    off, nearest = ptree.query(pts)
    main = off <= branch_offset
    s = path_s[nearest[main]]
    main_pts = pts[main]
    bins = np.floor((s - s.min()) / step).astype(np.int64)
    axis = []
    for bi in np.unique(bins):
        sel = bins == bi
        if sel.sum() < 3:
            continue
        cell = main_pts[sel]
        axis.append([cell[:, 0].mean(), cell[:, 1].mean(), np.percentile(cell[:, 2], 90)])
    leftover = np.flatnonzero(~main)
    if len(axis) < 2:
        return None, leftover
    return np.asarray(axis), leftover


def curb_axes(
    ps: c.PointSet,
    *,
    cluster_eps: float = 0.25,
    min_points: int = 30,
    dp_tolerance: float = 0.015,
    step: float = 0.25,
    branch_offset: float = 0.2,
) -> list[c.CurbAxis]:
    """Ordered, Douglas-Peucker thinned 3D axes of the curb (class CURB) points.

    Points are clustered (eps-neighbour components). Each cluster's longest geodesic path is the
    main axis: points are binned by arc length every ``step`` m (XY = mean, Z = 90th percentile
    = top edge) and thinned in 3D with ``dp_tolerance``. Points farther than ``branch_offset``
    from the main path (T-junction branches) are re-clustered and become separate axes.
    """
    sel = ps.classes == c.CURB
    cloud = ps.xyz[sel]
    if len(cloud) < min_points:
        return []
    queue = [cloud[idx] for idx in _components(cloud, cluster_eps) if len(idx) >= min_points]
    axes: list[c.CurbAxis] = []
    while queue:
        pts = queue.pop(0)
        axis, leftover = _trace_cluster(pts, cluster_eps, step, branch_offset)
        if axis is not None:
            thin = douglas_peucker_3d(axis, dp_tolerance)
            axes.append(c.CurbAxis(
                curb_id=len(axes), points=thin,
                evidence=c.Evidence(points=int(len(pts) - len(leftover))), status="ok",
            ))
        if len(leftover) >= min_points:
            rest = pts[leftover]
            for idx in _components(rest, cluster_eps):
                if len(idx) >= min_points:
                    queue.append(rest[idx])
    return axes


def _line_deviation(p: np.ndarray, i: int, j: int) -> float:
    """Max distance of p[i..j] from the chord p[i]-p[j] (XY)."""
    chord = p[j] - p[i]
    length = float(np.hypot(*chord))
    if length < 1e-12:
        return float("inf")
    rel = p[i : j + 1] - p[i]
    return float(np.max(np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0])) / length)


def _circle_fit(p: np.ndarray) -> tuple[float, float]:
    """Algebraic (Kasa) circle fit: (radius, max radial residual)."""
    x, y = p[:, 0], p[:, 1]
    mx, my = x.mean(), y.mean()
    xc, yc = x - mx, y - my
    A = np.column_stack([2 * xc, 2 * yc, np.ones(len(p))])
    rhs = xc**2 + yc**2
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, cc = sol
    r2 = cc + a * a + b * b
    if r2 <= 0:
        return float("inf"), float("inf")
    r = float(np.sqrt(r2))
    resid = float(np.max(np.abs(np.hypot(xc - a, yc - b) - r)))
    return r, resid


def split_straight_and_arcs(
    points: np.ndarray,
    *,
    arc_max_residual: float = 0.01,
    arc_max_radius: float = 50.0,
    arc_min_points: int = 4,
) -> list[tuple[str, slice]]:
    """Split an ordered axis into ``("straight"|"arc", slice)`` parts (XY geometry).

    A run is straight while all points stay within ``arc_max_residual`` of its chord; it is an arc
    when >= ``arc_min_points`` consecutive points fit one circle within ``arc_max_residual`` and
    radius <= ``arc_max_radius``. Whichever covers more points is taken first; neighbouring
    parts share their boundary vertex (slice stop - 1 == next slice start); consecutive straight
    parts are merged.
    """
    p = np.asarray(points, dtype=np.float64)[:, :2]
    n = len(p)
    if n < 2:
        return []
    parts: list[tuple[str, int, int]] = []
    i = 0
    while i < n - 1:
        js = i + 1
        while js + 1 < n and _line_deviation(p, i, js + 1) <= arc_max_residual:
            js += 1
        ja = -1
        j0 = i + arc_min_points - 1
        if j0 < n:
            r, res = _circle_fit(p[i : j0 + 1])
            if res <= arc_max_residual and r <= arc_max_radius:
                ja = j0
                while ja + 1 < n:
                    r, res = _circle_fit(p[i : ja + 2])
                    if res <= arc_max_residual and r <= arc_max_radius:
                        ja += 1
                    else:
                        break
        if ja >= 0 and ja >= js:
            parts.append(("arc", i, ja))
            i = ja
        else:
            parts.append(("straight", i, js))
            i = js
    merged: list[tuple[str, int, int]] = []
    for kind, a, b in parts:
        if merged and kind == "straight" and merged[-1][0] == "straight":
            merged[-1] = ("straight", merged[-1][1], b)
        else:
            merged.append((kind, a, b))
    return [(kind, slice(a, b + 1)) for kind, a, b in merged]
