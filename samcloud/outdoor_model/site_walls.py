"""Survey-derived site walls and fences.

* Fences are traced as polylines (longest path + branches per XY cluster, so neighbouring fences stay
  separate and corners are kept). Per station along the axis the BOTTOM edge is the terrain height and
  the TOP edge a robust upper percentile of the fence points; both are median-smoothed and thinned with
  Douglas-Peucker. The IFC body is an upright ribbon between the two edges, the DXF line the top edge.
* Site walls follow their traced axis: the footprint is the mitred buffer of the whole axis with the
  measured thickness, used identically for IFC and DXF.

Thresholds live in ``site.fence.*`` / ``site.wall.*`` / ``dxf.fence_bottom_line`` of the drone config.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient

from samcloud.bim.core.model import FenceElement, FenceParams, ParamOrigin, SiteWallElement, SiteWallParams, Status
from samcloud.drone import contracts as c

LOGGER = logging.getLogger(__name__)

LAYER_WALL = "S-WALL"
LAYER_FENCE = "S-FENC"
DtmSampler = Callable[[np.ndarray], np.ndarray]


@dataclass
class SiteWallFeature:
    wall_id: int
    axis: np.ndarray                      # (M,2) centre line
    thickness: float
    base_z: float
    top_z: float
    evidence: c.Evidence = field(default_factory=c.Evidence)
    status: c.Status = "ok"
    reason: str = ""
    top_z_axis: np.ndarray | None = None  # (M,) top height per axis vertex
    base_z_axis: np.ndarray | None = None  # (M,) base (terrain) height per axis vertex


@dataclass
class FenceFeature:
    fence_id: int
    axis: np.ndarray                      # (M,3) top edge polyline (z = top)
    height: float                         # median(top - bottom)
    evidence: c.Evidence = field(default_factory=c.Evidence)
    status: c.Status = "ok"
    reason: str = ""
    bottom_z: np.ndarray | None = None    # (M,) bottom edge (terrain) per vertex


@dataclass
class _Trace:
    xy: np.ndarray                        # (M,2) station centres in path order
    members: list[np.ndarray]             # point indices per station


def _components(xyz: np.ndarray, eps: float = 0.75) -> list[np.ndarray]:
    if len(xyz) == 0:
        return []
    pairs = cKDTree(xyz[:, :2]).query_pairs(eps, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz)))
    count, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == i) for i in range(count)]


def _dtm(sample: DtmSampler | None, xy: np.ndarray, fallback: float) -> float:
    if sample is None:
        return fallback
    values = np.asarray(sample(np.asarray(xy, float).reshape(-1, 2)), float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else fallback


def _trace_one(pts: np.ndarray, eps: float, step: float, branch_offset: float, min_station: int) -> tuple[_Trace | None, np.ndarray]:
    """Main axis stations of one XY cluster and the local indices left over (branches)."""
    voxel = eps / 4.0
    keys = np.floor(pts / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    thin = pts[np.sort(first)]
    if len(thin) < 2:
        return None, np.arange(len(pts))
    tree = cKDTree(thin)
    graph = tree.sparse_distance_matrix(tree, eps * 2.0, output_type="coo_matrix").tocsr()
    d0 = dijkstra(graph, directed=False, indices=0)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1.0)))
    dist_a, pred = dijkstra(graph, directed=False, indices=a, return_predecessors=True)
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
    nodes, node_s = thin[path_idx], dist_a[path_idx]
    seg_len = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    reps = np.maximum(np.ceil(seg_len / (voxel / 2.0)).astype(int), 1)
    path_pts, path_s = [nodes[:1]], [node_s[:1]]
    for k, m in enumerate(reps):
        u = (np.arange(1, m + 1) / m)[:, None]
        path_pts.append(nodes[k] * (1 - u) + nodes[k + 1] * u)
        path_s.append((node_s[k] * (1 - u) + node_s[k + 1] * u)[:, 0])
    path_pts, path_s = np.vstack(path_pts), np.concatenate(path_s)
    off, nearest = cKDTree(path_pts).query(pts)
    main = np.flatnonzero(off <= branch_offset)
    leftover = np.flatnonzero(off > branch_offset)
    if len(main) == 0:
        return None, np.arange(len(pts))
    s = path_s[nearest[main]]
    bins = np.floor((s - s.min()) / step).astype(np.int64)
    xy, members = [], []
    for bi in np.unique(bins):
        sel = main[bins == bi]
        if len(sel) < min_station:
            continue
        xy.append(pts[sel].mean(axis=0))
        members.append(sel)
    if len(xy) < 2:
        return None, leftover
    xy = np.asarray(xy)
    for end, inner in ((0, 1), (-1, -2)):  # the end stations reach the outermost points along the line
        d = xy[end] - xy[inner]
        norm = float(np.hypot(*d))
        if norm > 1e-9:
            d /= norm
            reach = float(((pts[members[end]] - xy[end]) @ d).max())
            xy[end] = xy[end] + d * max(reach, 0.0)
    return _Trace(xy, members), leftover


def _trace(xyz: np.ndarray, *, eps: float, step: float, branch_offset: float, min_points: int, min_station: int) -> list[_Trace]:
    """Polyline traces of all XY clusters of ``xyz``; branches (T-junctions) become their own traces."""
    traces: list[_Trace] = []
    queue = [ids for ids in _components(xyz, eps) if len(ids) >= min_points]
    while queue:
        ids = queue.pop(0)
        trace, leftover = _trace_one(xyz[ids, :2], eps, step, branch_offset, min_station)
        if trace is not None:
            traces.append(_Trace(trace.xy, [ids[m] for m in trace.members]))
        if len(leftover) >= min_points and len(leftover) < len(ids):
            rest = ids[leftover]
            for sub in _components(xyz[rest], eps):
                if len(sub) >= min_points:
                    queue.append(rest[sub])
    return traces


def _dp_indices(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices kept by a Douglas-Peucker thinning of an (M,D) polyline."""
    n = len(points)
    keep = np.zeros(n, dtype=bool)
    keep[[0, n - 1]] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = points[j] - points[i]
        seg2 = float(seg @ seg)
        rel = points[i + 1 : j] - points[i]
        if seg2 < 1e-18:
            dist = np.linalg.norm(rel, axis=1)
        else:
            t = np.clip((rel @ seg) / seg2, 0.0, 1.0)
            dist = np.linalg.norm(rel - t[:, None] * seg, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return np.flatnonzero(keep)


def _snap_corners(xy: np.ndarray, keep: np.ndarray, min_angle: float = 20.0, group_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Kept vertices ``(xy, station indices)`` with every corner restored as the intersection of its two straight runs.

    Station means cut a corner (up to half a station, often over two vertices). Neighbouring corner vertices
    closer than ``group_m`` form one corner; a line is fitted to the stations of each straight run (the stations
    next to a corner are left out) and neighbouring lines are intersected. The corner group is replaced by the
    intersection (station index of the group's middle vertex).
    """
    keep = np.asarray(keep)
    pts = xy[keep].copy()
    if len(keep) < 3:
        return pts, keep
    vec = np.diff(pts, axis=0)
    ang = np.degrees(np.abs(np.arctan2(vec[:-1, 0] * vec[1:, 1] - vec[:-1, 1] * vec[1:, 0], np.einsum("ij,ij->i", vec[:-1], vec[1:]))))
    corner_pos = [int(k) + 1 for k in np.flatnonzero(ang > min_angle)]
    if not corner_pos:
        return pts, keep
    groups: list[list[int]] = [[corner_pos[0]]]
    for k in corner_pos[1:]:
        if float(np.hypot(*(pts[k] - pts[groups[-1][-1]]))) < group_m:
            groups[-1].append(k)
        else:
            groups.append([k])
    last = len(xy) - 1
    bounds = [0]
    for g in groups:
        bounds += [int(keep[g[0]]), int(keep[g[-1]])]
    bounds.append(last)
    lines = []
    for i in range(len(groups) + 1):  # run i lies between group i-1 and group i
        a0, b0 = bounds[2 * i], bounds[2 * i + 1]
        sel = xy[a0 + (1 if i > 0 else 0) : b0 - (1 if i < len(groups) else 0) + 1]
        if len(sel) < 3:
            lines.append(None)
            continue
        centre = sel.mean(axis=0)
        lines.append((centre, np.linalg.svd(sel - centre)[2][0]))
    drop: set[int] = set()
    for i, g in enumerate(groups):
        first, second = lines[i], lines[i + 1]
        if first is None or second is None:
            continue
        (c1, d1), (c2, d2) = first, second
        det = -d1[0] * d2[1] + d1[1] * d2[0]
        if abs(det) < 0.2:
            continue
        rhs = c2 - c1
        t = (-rhs[0] * d2[1] + rhs[1] * d2[0]) / det
        point = c1 + d1 * t
        mid = g[len(g) // 2]
        if float(np.hypot(*(point - pts[mid]))) < 1.5:
            pts[mid] = point
            drop.update(k for k in g if k != mid)
    mask = np.array([k not in drop for k in range(len(keep))])
    return pts[mask], keep[mask]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    return ndimage.median_filter(values, size=window, mode="nearest") if window > 1 and len(values) >= window else values


def _cfg(config: dict | None, group: str) -> dict:
    cfg = config if config is not None else c.load_drone_config()
    return dict(cfg.get("site", {}).get(group, {}))


def _disc_median(sampler: DtmSampler, xy: np.ndarray, radius: float) -> np.ndarray:
    """Terrain height at ``xy`` as the median of samples in a disc (centre, half radius, full radius rings): smooth."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if radius <= 0.0:  # direct sampling (surface sampler of the final TIN)
        return np.asarray(sampler(xy), float).reshape(-1)
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles)])
    offsets = np.vstack([[0.0, 0.0], ring * radius * 0.5, ring * radius])
    pts = (xy[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    values = np.asarray(sampler(pts), float).reshape(len(xy), len(offsets))
    values = np.where(np.isfinite(values), values, np.nan)
    with np.errstate(all="ignore"):
        out = np.nanmedian(values, axis=1)
    return out


def site_wall_features(ps: c.PointSet, dtm_sampler: DtmSampler | None, *, config: dict | None = None) -> list[SiteWallFeature]:
    """Trace class-72 points into wall axes (polylines, corners kept) with measured thickness, base and top per vertex.

    The base follows the (smooth, disc-median) terrain per station, the top is an upper percentile of the station
    points; both are median-smoothed. ``base_z`` / ``top_z`` stay scalar summaries (median base, p95 top).
    """
    cfg = _cfg(config, "wall")
    xyz = np.asarray(ps.xyz, float)[np.asarray(ps.classes) == c.SITE_WALL]
    window = int(cfg.get("smooth_window", 5))
    output: list[SiteWallFeature] = []
    for trace in _trace(xyz, eps=float(cfg.get("eps_m", 0.75)), step=float(cfg.get("step_m", 0.4)),
                        branch_offset=float(cfg.get("branch_offset_m", 0.8)), min_points=int(cfg.get("min_points", 4)),
                        min_station=int(cfg.get("min_station_points", 3))):
        all_idx = np.concatenate(trace.members)
        pts = xyz[all_idx]
        top_st = _smooth(np.array([np.percentile(xyz[m, 2], float(cfg.get("top_percentile", 95.0))) for m in trace.members]), window)
        low_st = np.array([np.percentile(xyz[m, 2], 5.0) for m in trace.members])
        base_st = low_st
        if dtm_sampler is not None:
            terrain = _disc_median(dtm_sampler, trace.xy, float(cfg.get("terrain_radius_m", 0.5)))
            base_st = np.where(np.isfinite(terrain), terrain, low_st)
        base_st = np.minimum(_smooth(base_st, window), top_st - 0.01)
        tol = float(cfg.get("dp_m", 0.02))
        keep = np.union1d(_dp_indices(np.column_stack([trace.xy, np.zeros(len(trace.xy))]), tol),
                          np.union1d(_dp_indices(np.column_stack([trace.xy, top_st]), 0.05),
                                     _dp_indices(np.column_stack([trace.xy, base_st]), 0.05)))
        axis, keep = _snap_corners(trace.xy, keep)
        line = LineString(axis)
        loc = shapely.line_locate_point(line, shapely.points(pts[:, :2]))
        on = shapely.get_coordinates(shapely.line_interpolate_point(line, loc))
        dirs = _tangents(line, loc)
        cross = np.einsum("ij,ij->i", pts[:, :2] - on, np.column_stack([-dirs[:, 1], dirs[:, 0]]))
        thickness = max(float(cfg.get("min_thickness_m", 0.1)), float(np.percentile(cross, 99) - np.percentile(cross, 1)))
        base = float(np.median(base_st))
        top = float(np.percentile(pts[:, 2], float(cfg.get("top_percentile", 95.0))))
        cum_axis = np.r_[0.0, np.cumsum(np.hypot(*np.diff(axis, axis=0).T))]
        top_axis, base_axis = top_st[keep], base_st[keep]
        status, reason = ("ok", "") if top > base + 0.05 else ("rejected", "top is not above terrain")
        evidence = c.Evidence(points=len(pts), notes={"top_variation_m": round(float(np.ptp(top_st)), 3),
                                                      "base_variation_m": round(float(np.ptp(base_st)), 3),
                                                      "length_m": round(float(cum_axis[-1]), 3)})
        output.append(SiteWallFeature(len(output), axis, thickness, base, top, evidence, status, reason, top_axis, base_axis))
    return output


def _tangents(line: LineString, dist: np.ndarray, delta: float = 0.1) -> np.ndarray:
    total = line.length
    lo = np.clip(np.asarray(dist, float) - delta, 0.0, total)
    hi = np.clip(np.asarray(dist, float) + delta, 0.0, total)
    a = shapely.get_coordinates(shapely.line_interpolate_point(line, lo))
    b = shapely.get_coordinates(shapely.line_interpolate_point(line, hi))
    v = b - a
    return v / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-12)[:, None]


@dataclass
class _FenceTrace:
    axis: np.ndarray        # (M,3) top edge
    bottom: np.ndarray      # (M,)
    points: int
    stations: int


def _end_direction(axis: np.ndarray, at_end: bool, reach: float = 0.5) -> np.ndarray:
    """Outward unit direction of a polyline end (from the vertex about ``reach`` m back to the end)."""
    xy = axis[:, :2][::-1] if not at_end else axis[:, :2]
    end = xy[-1]
    back = xy[0]
    for p in xy[::-1]:
        if float(np.hypot(*(end - p))) >= reach:
            back = p
            break
    v = end - back
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def _merge_fences(parts: list[_FenceTrace], *, gap_m: float, angle_deg: float, height_diff_m: float) -> list[_FenceTrace]:
    """Join collinear fence fragments: end gap < ``gap_m``, directions within ``angle_deg``, similar height."""
    parts = list(parts)
    while True:
        best = None
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                hi = float(np.median(parts[i].axis[:, 2] - parts[i].bottom))
                hj = float(np.median(parts[j].axis[:, 2] - parts[j].bottom))
                if abs(hi - hj) >= height_diff_m:
                    continue
                for ei in (False, True):
                    for ej in (False, True):
                        pi = parts[i].axis[-1 if ei else 0, :2]
                        pj = parts[j].axis[-1 if ej else 0, :2]
                        gap = float(np.hypot(*(pi - pj)))
                        if gap >= gap_m:
                            continue
                        di, dj = _end_direction(parts[i].axis, ei), _end_direction(parts[j].axis, ej)
                        if _angle(di, -dj) >= angle_deg:
                            continue
                        if gap > 0.1 and _angle(di, (pj - pi) / gap) >= 2 * angle_deg:
                            continue
                        if best is None or gap < best[0]:
                            best = (gap, i, j, ei, ej)
        if best is None:
            return parts
        _gap, i, j, ei, ej = best
        a, b = parts[i], parts[j]
        a_axis, a_bot = (a.axis, a.bottom) if ei else (a.axis[::-1], a.bottom[::-1])
        b_axis, b_bot = (b.axis, b.bottom) if not ej else (b.axis[::-1], b.bottom[::-1])
        merged = _FenceTrace(np.vstack([a_axis, b_axis]), np.concatenate([a_bot, b_bot]), a.points + b.points, a.stations + b.stations)
        parts = [p for k, p in enumerate(parts) if k not in (i, j)] + [merged]


def fence_features(ps: c.PointSet, dtm_sampler: DtmSampler | None, *, config: dict | None = None,
                   report: dict | None = None) -> list[FenceFeature]:
    """Trace class-77 points into fences: top edge (upper percentile) and bottom edge per station.

    Bottom edge: the lower envelope of the station points (p3 of z, smoothed). Within ``ground_snap_m`` of
    the smooth terrain (disc median of the DTM) the bottom is that terrain, otherwise (fence on a wall or
    plinth, points not reaching the ground) the point envelope - the bottom never hangs far below the lowest
    point. Top and bottom are median-smoothed and thinned; bottom <= top - 0.2 m. Collinear fragments are
    merged (``merge_gap_m``, ``merge_angle_deg``, ``merge_height_diff_m``); ``report`` receives
    ``traced`` / ``merged`` counts.
    """
    cfg = _cfg(config, "fence")
    xyz = np.asarray(ps.xyz, float)[np.asarray(ps.classes) == c.FENCE]
    window = int(cfg.get("smooth_window", 5))
    bwindow = int(cfg.get("bottom_smooth_window", 9))
    snap = float(cfg.get("ground_snap_m", 0.15))
    min_h = float(cfg.get("min_height_m", 0.2))
    parts: list[_FenceTrace] = []
    for trace in _trace(xyz, eps=float(cfg.get("eps_m", 0.3)), step=float(cfg.get("step_m", 0.25)),
                        branch_offset=float(cfg.get("branch_offset_m", 0.3)), min_points=int(cfg.get("min_points", 4)),
                        min_station=int(cfg.get("min_station_points", 2))):
        top = _smooth(np.array([np.percentile(xyz[m, 2], float(cfg.get("top_percentile", 95.0))) for m in trace.members]), window)
        low = _smooth(np.array([np.percentile(xyz[m, 2], float(cfg.get("bottom_percentile", 3.0))) for m in trace.members]), bwindow)
        bottom = low.copy()
        if dtm_sampler is not None:
            terrain = _disc_median(dtm_sampler, trace.xy, float(cfg.get("terrain_radius_m", 0.5)))
            on_ground = np.isfinite(terrain) & (np.abs(terrain - low) <= snap)
            if len(on_ground) >= bwindow > 1:  # majority vote (mirrored ends): sparse stations follow their neighbours
                on_ground = ndimage.median_filter(on_ground.astype(int), size=bwindow, mode="mirror") > 0
            bottom = np.where(on_ground & np.isfinite(terrain), terrain, low)
        bottom = _smooth(bottom, bwindow)
        top = np.maximum(top, bottom)
        tol = float(cfg.get("dp_m", 0.02))
        keep = np.union1d(_dp_indices(np.column_stack([trace.xy, top]), tol),
                          _dp_indices(np.column_stack([trace.xy, bottom]), float(cfg.get("bottom_dp_m", 0.03))))
        axis_xy, keep = _snap_corners(trace.xy, keep)
        parts.append(_FenceTrace(np.column_stack([axis_xy, top[keep]]), bottom[keep],
                                 int(sum(len(m) for m in trace.members)), int(len(top))))
    traced = len(parts)
    parts = _merge_fences(parts, gap_m=float(cfg.get("merge_gap_m", 1.0)), angle_deg=float(cfg.get("merge_angle_deg", 15.0)),
                          height_diff_m=float(cfg.get("merge_height_diff_m", 0.3)))
    if report is not None:
        report.update(traced=traced, merged=len(parts))
    output: list[FenceFeature] = []
    for part in parts:
        axis, bottom = part.axis, part.bottom
        height = float(np.median(axis[:, 2] - bottom))
        bottom = np.minimum(bottom, axis[:, 2] - 0.2)  # a fence is at least 0.2 m tall
        length = float(np.hypot(*np.diff(axis[:, :2], axis=0).T).sum())
        status, reason = ("ok", "") if height > min_h else ("rejected", "height is not above terrain")
        if status == "ok" and length < float(cfg.get("min_length_m", 0.5)):
            status, reason = "rejected", f"fence length {length:.2f} m below {float(cfg.get('min_length_m', 0.5)):.2f} m"
        output.append(FenceFeature(len(output), axis, height, c.Evidence(points=part.points, notes={"stations": part.stations}), status, reason, bottom))
    return output


def _state(raw: str) -> Status:
    return {"ok": Status.OK, "warning": Status.WARNING}.get(raw, Status.FAIL)


def wall_polygon(axis_xy: np.ndarray, thickness: float) -> np.ndarray:
    """Footprint ring (counter-clockwise, no repeated end point) of the whole axis: mitre joins, flat caps."""
    poly = LineString(np.asarray(axis_xy, float)[:, :2]).buffer(float(thickness) / 2.0, cap_style=2, join_style=2)
    if poly.geom_type != "Polygon":
        poly = max((g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"), key=lambda g: g.area)
    return np.asarray(orient(Polygon(poly.exterior), 1.0).exterior.coords, float)[:-1]


def build_site_elements(
    walls: Sequence[SiteWallFeature], fences: Sequence[FenceFeature], *, config: dict | None = None,
) -> tuple[list[object], list[c.QAEntry]]:
    """Site wall and fence elements; rejected features become ``QAEntry``."""
    cfg = config if config is not None else c.load_drone_config()
    wall_cfg, fence_cfg = _cfg(cfg, "wall"), _cfg(cfg, "fence")
    bottom_line = bool(cfg.get("dxf", {}).get("fence_bottom_line", False))
    elements: list[object] = []
    qa: list[c.QAEntry] = []
    for wall in walls:
        ident = f"site-wall-{wall.wall_id}"
        if wall.status not in ("ok", "warning"):
            qa.append(c.QAEntry("site_wall", ident, wall.reason or "rejected"))
            continue
        axis = np.asarray(wall.axis, float)
        ring = wall_polygon(axis, wall.thickness)
        top_axis = base_axis = None
        limit = float(wall_cfg.get("top_variation_m", 0.10))
        if wall.top_z_axis is not None and float(np.ptp(wall.top_z_axis)) > limit:
            top_axis = [float(v) for v in wall.top_z_axis]
        if wall.base_z_axis is not None and float(np.ptp(wall.base_z_axis)) > limit:
            base_axis = [float(v) for v in wall.base_z_axis]
        if top_axis is not None or base_axis is not None:  # one varying edge: both per-vertex lists are written
            top_axis = [float(v) for v in (wall.top_z_axis if wall.top_z_axis is not None else np.full(len(axis), wall.top_z))]
            base_axis = [float(v) for v in (wall.base_z_axis if wall.base_z_axis is not None else np.full(len(axis), wall.base_z))]
        elements.append(SiteWallElement(
            id=ident, tag=str(wall.wall_id), name="Mauer",
            params=SiteWallParams(axis=[tuple(map(float, p[:2])) for p in axis], footprint=[tuple(map(float, p)) for p in ring],
                                  thickness=wall.thickness, base_z=wall.base_z, top_z=wall.top_z, top_z_axis=top_axis, base_z_axis=base_axis),
            param_origin={"axis": ParamOrigin.FITTED, "footprint": ParamOrigin.FITTED, "thickness": ParamOrigin.FITTED,
                          "base_z": ParamOrigin.MEASURED, "top_z": ParamOrigin.FITTED},
            evidence={"points": wall.evidence.points, "thickness_m": round(wall.thickness, 3), "vertices": int(len(axis)),
                      "top_varies": top_axis is not None}, status=_state(wall.status)))
    for fence in fences:
        ident = f"fence-{fence.fence_id}"
        if fence.status not in ("ok", "warning"):
            qa.append(c.QAEntry("fence", ident, fence.reason or "rejected"))
            continue
        bottom = None if fence.bottom_z is None else [float(v) for v in fence.bottom_z]
        elements.append(FenceElement(
            id=ident, tag=str(fence.fence_id), name="Zaun",
            params=FenceParams(axis=[tuple(map(float, p)) for p in fence.axis], height=fence.height, bottom_z=bottom,
                               thickness=float(fence_cfg.get("thickness_m", 0.04))),
            param_origin={"axis": ParamOrigin.FITTED, "height": ParamOrigin.FITTED, "bottom_z": ParamOrigin.MEASURED},
            evidence={"points": fence.evidence.points, "height_m": round(fence.height, 3), "vertices": int(len(fence.axis)),
                      "dxf_bottom_line": bottom_line}, status=_state(fence.status)))
    return elements, qa


def to_dxf_entities(elements: Sequence[object], frame: c.LocalFrame) -> dict[str, list[dict]]:
    """``S-WALL`` closed LWPOLYLINE (the IFC footprint) and ``S-FENC`` polyline3d of the fence top edge.

    The fence bottom edge is added only when the element carries ``dxf_bottom_line`` (``dxf.fence_bottom_line``).
    """
    output: dict[str, list[dict]] = {}
    for el in elements:
        if isinstance(el, SiteWallElement):
            ring = np.asarray(el.params.footprint if el.params.footprint else wall_polygon(np.asarray(el.params.axis, float), float(el.params.thickness)), float)
            z = np.full((len(ring), 1), float(el.params.base_z))
            output.setdefault(LAYER_WALL, []).append({
                "type": "lwpolyline", "points": frame.to_lv95(np.column_stack([ring, z])).tolist(), "closed": True,
                "elevation": float(frame.origin_lv95[2] + el.params.base_z)})
        elif isinstance(el, FenceElement):
            axis = np.asarray(el.params.axis, float)
            output.setdefault(LAYER_FENCE, []).append({"type": "polyline3d", "points": frame.to_lv95(axis).tolist(), "closed": False})
            if el.evidence.get("dxf_bottom_line") and el.params.bottom_z is not None:
                bottom = np.column_stack([axis[:, :2], np.asarray(el.params.bottom_z, float)])
                output[LAYER_FENCE].append({"type": "polyline3d", "points": frame.to_lv95(bottom).tolist(), "closed": False})
    return output


def category_counts(elements: Sequence[object]) -> dict[str, int]:
    return {"site_wall": sum(isinstance(x, SiteWallElement) for x in elements), "fence": sum(isinstance(x, FenceElement) for x in elements)}
