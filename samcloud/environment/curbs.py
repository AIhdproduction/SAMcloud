"""3D curb axes from CURB points and a straight/arc splitter for the IFC writer."""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import shapely
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from samcloud.drone import contracts as c

LOGGER = logging.getLogger(__name__)

_TOP_BAND_M = 0.03  # points this close to the p90 height form the curb top edge


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
        top_z = np.percentile(cell[:, 2], 90)
        # XY from the top edge only: lower points (curb face, gutter band) must not pull the axis sideways
        upper = cell[cell[:, 2] >= top_z - _TOP_BAND_M]
        ref = upper if len(upper) >= 3 else cell
        axis.append([ref[:, 0].mean(), ref[:, 1].mean(), top_z])
    leftover = np.flatnonzero(~main)
    if len(axis) < 2:
        return None, leftover
    return np.asarray(axis), leftover


def _trace_axes(
    ps: c.PointSet, *, cluster_eps: float, min_points: int, dp_tolerance: float, step: float, branch_offset: float,
) -> list[c.CurbAxis]:
    """Raw curb axes (clusters, longest path, branches) without dedupe or cross-section analysis."""
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


def curb_axes(
    ps: c.PointSet,
    *,
    cluster_eps: float = 0.25,
    min_points: int = 30,
    dp_tolerance: float = 0.015,
    step: float = 0.25,
    branch_offset: float = 0.2,
    config: dict | None = None,
    report: dict | None = None,
    sections: bool = True,
) -> list[c.CurbAxis]:
    """Ordered, Douglas-Peucker thinned 3D axes of the curb (class CURB) points.

    Points are clustered (eps-neighbour components). Each cluster's longest geodesic path is the
    main axis: points are binned by arc length every ``step`` m (XY = mean of the top-edge points,
    Z = 90th percentile = top edge) and thinned in 3D with ``dp_tolerance``. Points farther than
    ``branch_offset`` from the main path (T-junction branches) are re-clustered and become separate
    axes. Parallel duplicates (``curb.dedupe_*``) are removed (:func:`dedupe_curb_axes`); with
    ``sections`` every axis gets its road-side foot height, gutter width and flush state
    (:func:`analyse_curb_sections`). ``report`` receives ``duplicates_removed`` / ``duplicate_length_m``
    / ``removed_axes`` and the section statistics.
    """
    cfg = (config if config is not None else c.load_drone_config()).get("curb", {})
    axes = _trace_axes(ps, cluster_eps=cluster_eps, min_points=min_points, dp_tolerance=dp_tolerance,
                       step=step, branch_offset=branch_offset)
    kept, removed = dedupe_curb_axes(
        axes, offset_m=float(cfg.get("dedupe_offset_m", 0.5)), angle_deg=float(cfg.get("dedupe_angle_deg", 10.0)),
        overlap=float(cfg.get("dedupe_overlap", 0.5)))
    kept = [replace(axis, curb_id=i) for i, axis in enumerate(kept)]
    if report is not None:
        report["duplicates_removed"] = len(removed)
        report["duplicate_length_m"] = round(float(sum(_xy_length(a.points) for a in removed)), 3)
        report["removed_axes"] = [a.points for a in removed]
    if sections and kept:
        kept = analyse_curb_sections(ps, kept, cfg, dp_tolerance=dp_tolerance, report=report)
    return kept


def _xy_length(points: np.ndarray) -> float:
    return float(np.hypot(*np.diff(np.asarray(points)[:, :2], axis=0).T).sum())


def dedupe_curb_axes(
    axes: list[c.CurbAxis], *, offset_m: float = 0.5, angle_deg: float = 10.0, overlap: float = 0.5,
    sample_m: float = 0.25,
) -> tuple[list[c.CurbAxis], list[c.CurbAxis]]:
    """Remove curb axes that run parallel behind another one; returns ``(kept, removed)``.

    The shorter axis A of a pair is a duplicate of B when at least ``overlap`` of A's samples lie within
    ``offset_m`` of B with a direction difference below ``angle_deg``. Of two axes of similar length (within
    a factor 2) the line at the height jump (the higher median axis Z, more than 3 cm apart) stays, otherwise
    the one with more evidence points; next to a much longer axis the short piece always goes.
    """
    n = len(axes)
    lines = [LineString(np.asarray(a.points)[:, :2]) for a in axes]
    gone: set[int] = set()
    cos_limit = float(np.cos(np.radians(angle_deg)))
    order = sorted(range(n), key=lambda i: lines[i].length)  # shortest first
    for ia in order:
        if ia in gone or lines[ia].length <= 1e-9:
            continue
        la = lines[ia]
        count = max(int(la.length / sample_m), 1) + 1
        dist = np.linspace(0.0, la.length, count)
        pa = shapely.line_interpolate_point(la, dist)
        da = _directions(la, dist)
        for ib in range(n):
            if ib == ia or ib in gone or lines[ib].length < la.length - 1e-9:
                continue
            lb = lines[ib]
            near = shapely.distance(pa, lb) <= offset_m
            if not near.any():
                continue
            loc = shapely.line_locate_point(lb, pa)
            db = _directions(lb, loc)
            parallel = np.abs(np.sum(da * db, axis=1)) >= cos_limit
            if float(np.mean(near & parallel)) < overlap:
                continue
            za, zb = float(np.median(axes[ia].points[:, 2])), float(np.median(axes[ib].points[:, 2]))
            if lb.length > 2.0 * la.length:
                loser = ia  # a short piece alongside a much longer axis is a fragment of the same curb
            elif abs(za - zb) > 0.03:
                loser = ia if za < zb else ib
            elif axes[ia].evidence.points != axes[ib].evidence.points:
                loser = ia if axes[ia].evidence.points < axes[ib].evidence.points else ib
            else:
                loser = ia
            gone.add(loser)
            if loser == ia:
                break
    kept = [a for i, a in enumerate(axes) if i not in gone]
    return kept, [axes[i] for i in sorted(gone)]


def _directions(line: LineString, dist: np.ndarray, delta: float = 0.1) -> np.ndarray:
    """Unit tangent of ``line`` at the arc lengths ``dist``."""
    total = line.length
    lo = np.clip(np.asarray(dist, dtype=float) - delta, 0.0, total)
    hi = np.clip(np.asarray(dist, dtype=float) + delta, 0.0, total)
    a = shapely.get_coordinates(shapely.line_interpolate_point(line, lo))
    b = shapely.get_coordinates(shapely.line_interpolate_point(line, hi))
    v = b - a
    norm = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-12)
    return v / norm[:, None]


_SURFACE_CLASSES = (c.GROUND, c.ROAD, c.SEALED_TERRAIN, c.SIDEWALK, c.PATH)


def _station_arclengths(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Arc lengths of the original vertices united with a regular ``step`` grid, and the vertex arc lengths."""
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points[:, :2], axis=0).T))]
    total = float(cum[-1])
    regular = np.linspace(0.0, total, max(int(np.ceil(total / step)), 1) + 1)
    merged = np.unique(np.round(np.r_[cum, regular], 6))
    keep = np.r_[True, np.diff(merged) > 1e-4]
    return merged[keep], cum


def _runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Boolean mask of the runs of ``mask`` that are at least ``min_run`` long (short arrays: whole run)."""
    out = np.zeros(len(mask), dtype=bool)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        if j - i + 1 >= min(min_run, len(mask)):
            out[i : j + 1] = True
        i = j + 1
    return out


def analyse_curb_sections(
    ps: c.PointSet, axes: list[c.CurbAxis], cfg: dict, *, dp_tolerance: float = 0.015, report: dict | None = None,
) -> list[c.CurbAxis]:
    """Cross-section analysis of every curb axis: road-side foot height, gutter width, flush stretches.

    Per station (``curb.section_step_m``) the ground/pavement/curb points within ``section_halfwidth_m``
    on both sides are taken; the lower side is the road side, its median height the foot level (reveal =
    top - foot). Curb-class points lying flat at foot level on the road side form the gutter band
    (measured when at least ``gutter_min_band_m`` wide, else ``gutter_width_m`` is assumed). Station runs
    with a reveal below ``flush_max_m`` are flush: the top is lowered to the surface so nothing sticks out.
    """
    step = float(cfg.get("section_step_m", 0.25))
    half = float(cfg.get("section_halfwidth_m", 0.6))
    hw = float(cfg.get("section_width_m", 0.12)) / 2.0
    excl = hw + 0.02
    flush_max = float(cfg.get("flush_max_m", 0.03))
    z_tol = float(cfg.get("gutter_z_tol_m", 0.03))
    default_reveal = float(cfg.get("default_reveal_m", 0.12))
    max_reveal = float(cfg.get("max_reveal_m", 0.35))
    xyz = np.asarray(ps.xyz, dtype=np.float64)
    cls = np.asarray(ps.classes)
    use = np.isin(cls, (*_SURFACE_CLASSES, c.CURB))
    if not use.any():
        return axes
    pts, pcl = xyz[use], cls[use]
    tree = cKDTree(pts[:, :2])
    win = step / 2.0 + 0.05
    radius = float(np.hypot(half, win))
    out: list[c.CurbAxis] = []
    stats = {"stations": 0, "flush_stations": 0, "gutters_measured": 0, "gutters_assumed": 0, "foot_assumed": 0}
    for axis in axes:
        ap = np.asarray(axis.points, dtype=np.float64)
        if len(ap) < 2 or _xy_length(ap) < 1e-6:
            out.append(axis)
            continue
        s_st, cum = _station_arclengths(ap, step)
        line = LineString(ap[:, :2])
        sp = shapely.get_coordinates(shapely.line_interpolate_point(line, s_st))
        tang = _directions(line, s_st)
        nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
        top = np.interp(s_st, cum, ap[:, 2])
        k = len(s_st)
        zl = np.full(k, np.nan)
        zr = np.full(k, np.nan)
        local: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i, nb in enumerate(tree.query_ball_point(sp, radius)):
            if not nb:
                local.append((np.empty(0), np.empty(0), np.empty(0, dtype=int)))
                continue
            nb = np.asarray(nb)
            rel = pts[nb, :2] - sp[i]
            along, off = rel @ tang[i], rel @ nrm[i]
            sel = (np.abs(along) <= win) & (np.abs(off) <= half)
            off, z, cl = off[sel], pts[nb[sel], 2], pcl[nb[sel]]
            local.append((off, z, cl))
            surf = cl != c.CURB
            for sign, target in ((1.0, zl), (-1.0, zr)):
                side_mask = sign * off > excl
                good = side_mask & surf
                if good.sum() >= 3:
                    target[i] = float(np.median(z[good]))
                else:
                    fall = side_mask & (z <= top[i] - 0.02)
                    if fall.sum() >= 3:
                        target[i] = float(np.median(z[fall]))
        both = np.isfinite(zl) & np.isfinite(zr)
        votes = np.sign(zr[both] - zl[both]) * (np.abs(zr[both] - zl[both]) > 0.02)
        side = int(np.sign(votes.sum())) if both.any() else 0
        if not both.any():
            one_l, one_r = np.isfinite(zl), np.isfinite(zr)
            if one_l.any() != one_r.any():
                z_side = float(np.nanmedian(zl if one_l.any() else zr))
                if z_side < float(np.median(top)) - 0.03:
                    side = 1 if one_l.any() else -1
        foot = np.fmin(zl, zr)  # lower side = road side (fmin ignores a missing side)
        foot[(top - foot) > max_reveal] = np.nan  # a step that high is no curb reveal (stairs, retaining wall): not trusted
        valid = np.isfinite(foot)
        assumed = not valid.any()
        if assumed:
            foot = top - default_reveal
        else:
            foot = np.interp(s_st, s_st[valid], foot[valid])
            foot = ndimage.median_filter(foot, size=5, mode="nearest") if k >= 5 else foot
        foot = np.minimum(foot, top)
        reveal = top - foot
        flush_mask = _runs(reveal < flush_max, int(cfg.get("flush_min_run", 3))) if not assumed else np.zeros(k, bool)
        new_top = np.where(flush_mask, np.minimum(top, foot), top)
        n_flush = int(flush_mask.sum())
        flush = n_flush >= 0.5 * k
        width, measured = 0.0, False
        if side != 0 and not flush:
            band, covered = [], 0
            for i, (off, z, cl) in enumerate(local):
                if len(off) == 0:
                    continue
                m = (cl == c.CURB) & (side * off > excl) & (np.abs(z - foot[i]) <= z_tol)
                if m.sum() >= 2:
                    band.append(side * off[m])
                    covered += 1
            if band and covered >= 0.3 * k and sum(len(b) for b in band) >= 10:
                measured_w = float(np.percentile(np.concatenate(band), 98)) - hw
                if measured_w >= float(cfg.get("gutter_min_band_m", 0.15)):
                    width, measured = min(measured_w, half - hw), True
            if not measured:
                width = float(cfg.get("gutter_width_m", 0.30))
        if n_flush:
            new_axis = douglas_peucker_3d(np.column_stack([sp, new_top]), dp_tolerance)
            new_cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(new_axis[:, :2], axis=0).T))]
            foot_v = np.interp(new_cum, s_st, foot)
        else:
            new_axis = ap
            foot_v = np.interp(cum, s_st, foot)
        notes = dict(axis.evidence.notes)
        notes.update(reveal_m=round(float(np.median(reveal)), 3), flush_stations=n_flush, stations=int(k),
                     gutter="measured" if measured else ("assumed" if width > 0 else "none"),
                     foot="assumed" if assumed else "measured")
        out.append(replace(
            axis, points=new_axis, foot_z=foot_v, road_side=side, gutter_width=float(width),
            gutter_measured=measured, flush=bool(flush), flush_stations=n_flush,
            evidence=c.Evidence(views=axis.evidence.views, points=axis.evidence.points, rmse=axis.evidence.rmse, notes=notes)))
        stats["stations"] += int(k)
        stats["flush_stations"] += n_flush
        stats["gutters_measured"] += int(measured)
        stats["gutters_assumed"] += int(width > 0 and not measured)
        stats["foot_assumed"] += int(assumed)
    if report is not None:
        report.update(stats)
    return out


def transfer_curb_section(old: c.CurbAxis, new_points: np.ndarray) -> c.CurbAxis:
    """``old`` with the geometry replaced by ``new_points`` (e.g. the line/arc fit), section data carried over.

    The foot height is re-sampled by projecting the new vertices onto the old axis; a reversed direction
    flips ``road_side``.
    """
    new_points = np.asarray(new_points, dtype=np.float64)
    if old.foot_z is None or len(old.points) < 2 or len(new_points) < 2:
        return replace(old, points=new_points)
    old_xy = np.asarray(old.points)[:, :2]
    line = LineString(old_xy)
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(old_xy, axis=0).T))]
    along = shapely.line_locate_point(line, shapely.points(new_points[:, :2]))
    foot = np.interp(along, cum, np.asarray(old.foot_z, dtype=np.float64))
    side = old.road_side if along[-1] >= along[0] else -old.road_side
    return replace(old, points=new_points, foot_z=np.minimum(foot, new_points[:, 2]), road_side=side)


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
