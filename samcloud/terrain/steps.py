"""Vertical steps in the 2.5D terrain TIN: curb faces, path edges, retaining walls.

A TIN over (x, y) cannot hold a height jump along a breakline.  A step is therefore modelled as two lines:

* the TOP edge exactly on the line at the height of the upper side, and
* the BOTTOM edge ``offset_m`` (1 cm) outside, towards the lower side, at the height of the lower side.

The 1 cm strip between them is the near-vertical face; the surface stays 2.5D and closed.

:func:`detect_steps` finds the step runs (along every region boundary and along steep linear DTM features),
:func:`apply_steps` splits an already built, constrained :class:`~samcloud.drone.contracts.LabeledTin` along them.
The split is a pure mesh operation (duplicate the vertices of the lower fan, add the strip triangles), so it does not
depend on a constrained Delaunay: it only needs the step line to exist as TIN edges, which the region boundaries and
the breaklines handed to ``build_labeled_tin`` guarantee.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import shapely
from scipy import ndimage
from samcloud.drone.contracts import TIN_LABELS
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import linemerge, unary_union

LOGGER = logging.getLogger(__name__)
STEP_LABEL = TIN_LABELS["step"]
_EPS = 1e-9
_ON_LINE_M = 1.2e-2         # TIN vertices within this distance of a step line belong to it
_MIN_AREA_M2 = 1e-8        # triangles below this XY area are degenerate
DEFAULTS: dict = {
    "enabled": True, "min_slope_deg": 60.0, "min_length_m": 1.0, "probe_m": 0.15, "min_jump_m": 0.04,
    "offset_m": 0.01, "min_run_m": 0.5, "sample_m": 0.25, "dtm_features": True, "wall_min_jump_m": 0.25,
    "curb_snap_m": 0.15, "curb_max_jump_m": 0.5, "pavement_jump_m": 0.08,
}


@dataclass
class StepRun:
    """One continuous stretch of a line with a height jump.

    ``xy`` is the ordered polyline on the line itself (the TOP edge), ``z_top`` / ``z_bottom`` the heights of the upper
    and the lower side per vertex, ``lower_side`` is ``+1`` when the lower side is left of the walking direction
    (``-1`` right). ``closed`` marks a ring that is a step all around.
    """

    xy: np.ndarray
    z_top: np.ndarray
    z_bottom: np.ndarray
    lower_side: int
    closed: bool = False
    source: str = "region"

    @property
    def length(self) -> float:
        """Length of the run in metres."""
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum()) if len(self.xy) > 1 else 0.0

    @property
    def mean_jump(self) -> float:
        """Mean height difference upper minus lower side."""
        return float(np.mean(self.z_top - self.z_bottom)) if len(self.xy) else 0.0

    def top_line(self) -> np.ndarray:
        """(N, 3) top edge, usable as TIN breakline (finite z)."""
        return np.column_stack((self.xy, self.z_top))


def step_config(config: Mapping | None) -> dict:
    """``terrain.step`` of ``config`` merged over the defaults."""
    node = ((config or {}).get("terrain", {}) or {}).get("step", {}) or {}
    return {**DEFAULTS, **{k: v for k, v in dict(node).items() if v is not None}}


# ------------------------------------------------------------------------------------------------ detection


def _polygon_parts(value) -> list[Polygon]:
    if hasattr(value, "polygon"):
        value = value.polygon
    if value is None or value.is_empty:
        return []
    if value.geom_type == "Polygon":
        return [value]
    if hasattr(value, "geoms"):
        return [part for geom in value.geoms for part in _polygon_parts(geom)]
    return []


def _region_lines(regions: Mapping[str, Iterable]) -> list[LineString]:
    """Noded, merged boundary linework of all regions (shared edges appear once)."""
    rings = []
    for name, polygons in regions.items():
        if name in ("lawn", "terrain"):
            continue
        for value in polygons or ():
            for poly in _polygon_parts(value):
                rings.extend([poly.exterior, *poly.interiors])
    if not rings:
        return []
    noded = unary_union(rings)
    merged = noded if noded.geom_type == "LineString" else linemerge(noded)
    geoms = [merged] if merged.geom_type == "LineString" else list(getattr(merged, "geoms", ()))
    return [g for g in geoms if g.geom_type == "LineString" and g.length > 1e-6]


def _sample_positions(line: LineString, spacing: float) -> np.ndarray:
    """Arc positions: every original vertex plus a regular grid of at most ``spacing``."""
    coords = np.asarray(line.coords)[:, :2]
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
    pieces: list[np.ndarray] = [cumulative[:1]]
    for a, b in zip(cumulative[:-1], cumulative[1:]):
        count = max(1, int(math.ceil((b - a) / spacing - 1e-9)))
        pieces.append(np.linspace(a, b, count + 1)[1:])
    return np.concatenate(pieces)


def _normals(points: np.ndarray, closed: bool) -> np.ndarray:
    """Unit left normals at the samples (mitred between the neighbouring segments)."""
    ext = np.vstack((points, points[:1])) if closed else points
    seg = np.diff(ext, axis=0)
    length = np.linalg.norm(seg, axis=1)
    length[length < _EPS] = 1.0
    unit = seg / length[:, None]
    left = np.column_stack((-unit[:, 1], unit[:, 0]))
    if closed:
        normals = left + np.roll(left, 1, axis=0)
    else:
        normals = np.empty((len(points), 2))
        normals[0], normals[-1] = left[0], left[-1]
        normals[1:-1] = left[:-1] + left[1:]
    size = np.linalg.norm(normals, axis=1)
    fallback = np.vstack((left, left[-1:]))[: len(points)] if not closed else left
    bad = size < 1e-6
    normals[bad] = fallback[bad]
    return normals / np.maximum(np.linalg.norm(normals, axis=1), _EPS)[:, None]


def _median3(values: np.ndarray, closed: bool) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    return ndimage.median_filter(values, size=3, mode="wrap" if closed else "nearest")


def _stretches(sign: np.ndarray, closed: bool) -> list[np.ndarray]:
    """Index arrays of maximal stretches with one non-zero sign (cyclic for closed lines; no stretch wraps the start)."""
    n = len(sign)
    start = 0
    if closed:
        change = np.flatnonzero(sign != np.roll(sign, 1))
        if len(change) == 0:
            return [np.arange(n)] if sign[0] != 0 else []
        start = int(change[0])
    order = (np.arange(n) + start) % n if closed else np.arange(n)
    out: list[np.ndarray] = []
    current: list[int] = []
    for position in order:
        s = int(sign[position])
        if s != 0 and current and s == int(sign[current[-1]]):
            current.append(int(position))
            continue
        if current:
            out.append(np.asarray(current))
            current = []
        if s != 0:
            current.append(int(position))
    if current:
        out.append(np.asarray(current))
    return out


def _runs_of_line(line: LineString, sampler: Callable, cfg: Mapping, probe: float, source: str,
                  min_jump: float | None = None, confirm_probe: float | None = None, accept: Callable | None = None) -> list[StepRun]:
    spacing = max(float(cfg["sample_m"]), 0.05)
    coords = np.asarray(line.coords)
    closed = len(coords) > 3 and bool(np.allclose(coords[0, :2], coords[-1, :2]))
    positions = _sample_positions(line, spacing)
    if closed:
        positions = positions[:-1]
    if len(positions) < 2:
        return []
    pts = np.asarray([line.interpolate(float(d)).coords[0][:2] for d in positions], dtype=float)
    normals = _normals(pts, closed)
    zl = np.asarray(sampler(pts + probe * normals), dtype=float).reshape(-1)
    zr = np.asarray(sampler(pts - probe * normals), dtype=float).reshape(-1)
    ok = np.isfinite(zl) & np.isfinite(zr)
    jump = np.where(ok, zl - zr, 0.0)
    if confirm_probe is not None:   # clear evidence: the jump must be there at both probe distances
        far = np.asarray(sampler(pts + confirm_probe * normals), dtype=float).reshape(-1) - np.asarray(sampler(pts - confirm_probe * normals), dtype=float).reshape(-1)
        far = np.where(np.isfinite(far), far, 0.0)
        jump = np.where(np.sign(far) == np.sign(jump), np.sign(jump) * np.minimum(np.abs(jump), np.abs(far)), 0.0)
    jump = _median3(jump, closed)
    sign = np.where(np.abs(jump) >= float(cfg["min_jump_m"] if min_jump is None else min_jump), np.sign(jump), 0).astype(int)
    if accept is not None:
        sign = np.where(accept(pts, normals), sign, 0)
    runs: list[StepRun] = []
    for idx in _stretches(sign, closed):
        whole = closed and len(idx) == len(pts)
        if len(idx) < 2 and not whole:
            continue
        xy = pts[idx]
        if not whole and float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) < float(cfg["min_run_m"]):
            continue
        runs.append(_make_run(xy, zl[idx], zr[idx], int(sign[idx[0]]), whole, source))
    return runs


def _make_run(xy: np.ndarray, zl: np.ndarray, zr: np.ndarray, sign: int, closed: bool, source: str) -> StepRun:
    """Run from samples: upper side = the higher probe, lower side on the other side; heights lightly smoothed."""
    left_higher = sign > 0
    top = _median3(np.asarray(zl if left_higher else zr, float), closed)
    bottom = _median3(np.asarray(zr if left_higher else zl, float), closed)
    xy = np.asarray(xy, dtype=float)
    if closed:
        xy = np.vstack((xy, xy[:1]))
        top, bottom = np.r_[top, top[:1]], np.r_[bottom, bottom[:1]]
    return StepRun(xy=xy, z_top=top, z_bottom=bottom, lower_side=-1 if left_higher else 1, closed=closed, source=source)


def detect_dtm_step_lines(
    dtm: np.ndarray, bounds: tuple[float, float, float, float], resolution: float, config: Mapping | None = None,
) -> list[LineString]:
    """Centre lines of steep linear DTM features (gradient > ``min_slope_deg`` over at least ``min_length_m``)."""
    from skimage.morphology import skeletonize

    cfg = step_config(config)
    z = np.asarray(dtm, dtype=float)
    valid = np.isfinite(z)
    if valid.sum() < 9:
        return []
    filled = np.where(valid, z, np.nanmean(z[valid]))
    gy, gx = np.gradient(filled, float(resolution))
    steep = (np.hypot(gx, gy) >= math.tan(math.radians(float(cfg["min_slope_deg"])))) & valid
    steep = ndimage.binary_closing(steep, iterations=1) & ndimage.binary_erosion(valid, iterations=3)  # not at the data edge
    if not steep.any():
        return []
    skeleton = skeletonize(steep)
    labels, count = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=int))
    xmin, _ymin, _xmax, ymax = (float(v) for v in bounds)
    lines: list[LineString] = []
    for index in range(1, count + 1):
        rows, cols = np.nonzero(labels == index)
        if len(rows) * float(resolution) < float(cfg["min_length_m"]):
            continue
        chain = _order_pixels(rows, cols)
        if len(chain) < 3:
            continue
        xy = np.column_stack((xmin + (chain[:, 1] + 0.5) * resolution, ymax - (chain[:, 0] + 0.5) * resolution))
        line = LineString(xy).simplify(0.5 * float(resolution))
        if line.length >= float(cfg["min_length_m"]):
            lines.append(line)
    return lines


def _order_pixels(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Longest path through a skeleton component (pixels, 8-neighbourhood), as (row, col) pairs."""
    import networkx as nx

    coords = {(int(r), int(c)) for r, c in zip(rows, cols)}
    graph = nx.Graph()
    for r, c in coords:
        graph.add_node((r, c))
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if (r + dr, c + dc) in coords:
                graph.add_edge((r, c), (r + dr, c + dc))
    start = next(iter(coords))
    far = max(nx.single_source_shortest_path_length(graph, start).items(), key=lambda kv: kv[1])[0]
    lengths = nx.single_source_shortest_path_length(graph, far)
    other = max(lengths.items(), key=lambda kv: kv[1])[0]
    return np.asarray(nx.shortest_path(graph, far, other), dtype=float)


def _resample_along(values, fractions: np.ndarray) -> np.ndarray | None:
    """Per-vertex ``values`` resampled at arc-length ``fractions`` (0..1) of the same polyline."""
    if values is None:
        return None
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) < 2:
        return None
    return np.interp(fractions, np.linspace(0.0, 1.0, len(values)), values)


def _curb_xyz(curb) -> np.ndarray:
    return np.asarray(getattr(curb, "points", curb), dtype=float)


def _curb_runs(curbs: Iterable, sampler: Callable, cfg: Mapping, boundary=None) -> list[StepRun]:
    """Steps along curb axes: top = curb top (axis z), bottom = road-side surface (``foot_z`` or a probe).

    With ``boundary`` (pavement outlines) only the parts of an axis within ``curb_snap_m`` of an outline count, and they are
    snapped onto it, so the step lies exactly on a TIN constraint; curbs that stand elsewhere get no step.
    """
    probe, spacing = float(cfg["probe_m"]), max(float(cfg["sample_m"]), 0.05)
    runs: list[StepRun] = []
    for curb in curbs:
        axis = _curb_xyz(curb)
        if axis.ndim != 2 or axis.shape[1] < 3 or len(axis) < 2:
            continue
        line = LineString(axis[:, :2])
        if line.length < float(cfg["min_run_m"]):
            continue
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(axis[:, :2], axis=0), axis=1))]
        arc = _sample_positions(line, spacing)
        pts = np.asarray([line.interpolate(float(d)).coords[0][:2] for d in arc], dtype=float)
        near = np.ones(len(pts), dtype=bool)
        if boundary is not None:
            dist = np.asarray(shapely.distance(shapely.points(pts[:, 0], pts[:, 1]), boundary), dtype=float)
            near = dist <= float(cfg["curb_snap_m"])
            if near.any():
                from shapely.ops import nearest_points

                pts[near] = np.asarray([nearest_points(boundary, shapely.Point(p))[0].coords[0][:2] for p in pts[near]], dtype=float)
        normals = _normals(pts, False)
        zl = np.asarray(sampler(pts + probe * normals), dtype=float).reshape(-1)
        zr = np.asarray(sampler(pts - probe * normals), dtype=float).reshape(-1)
        road_side = int(getattr(curb, "road_side", 0) or 0)
        lower = road_side if road_side else (1 if np.nanmean(zl) < np.nanmean(zr) else -1)
        probe_low = zl if lower > 0 else zr
        probe_up = zr if lower > 0 else zl
        top = np.interp(arc, cum, axis[:, 2])
        top = np.where(np.isfinite(top), top, probe_up)
        foot = _resample_along(getattr(curb, "foot_z", None), arc / max(line.length, _EPS))
        bottom = probe_low if foot is None else np.where(np.isfinite(foot), foot, probe_low)
        jump = _median3(top - bottom, False)
        sign = np.where((jump >= float(cfg["min_jump_m"])) & (jump <= float(cfg["curb_max_jump_m"])) & near, lower, 0).astype(int)
        for idx in _stretches(sign, False):
            if len(idx) < 2 or float(np.linalg.norm(np.diff(pts[idx], axis=0), axis=1).sum()) < float(cfg["min_run_m"]):
                continue
            runs.append(StepRun(xy=pts[idx], z_top=_median3(top[idx], False), z_bottom=_median3(bottom[idx], False),
                                lower_side=lower, closed=False, source="curb"))
    return runs


def _curb_runs_on_lines(lines: Iterable[LineString], curbs: Iterable, sampler: Callable, cfg: Mapping) -> list[StepRun]:
    """Curb steps on the pavement outlines themselves (so the step lies exactly on the TIN constraints).

    Every sample of an outline within ``curb_snap_m`` of a curb axis takes the curb data of the nearest axis position:
    top = curb top (axis z), bottom = ``foot_z`` (or the lower probe), lower side = ``road_side`` (turned into the walking
    direction of the outline), else the lower probe side.
    """
    probe, spacing, snap = float(cfg["probe_m"]), max(float(cfg["sample_m"]), 0.05), float(cfg["curb_snap_m"])
    axes = []
    for curb in curbs:
        xyz = _curb_xyz(curb)
        if xyz.ndim == 2 and xyz.shape[1] >= 3 and len(xyz) >= 2 and LineString(xyz[:, :2]).length >= float(cfg["min_run_m"]):
            axes.append((curb, xyz, LineString(xyz[:, :2])))
    if not axes:
        return []
    tree = shapely.STRtree([a[2] for a in axes])
    runs: list[StepRun] = []
    for line in lines:
        coords = np.asarray(line.coords)
        closed = len(coords) > 3 and bool(np.allclose(coords[0, :2], coords[-1, :2]))
        positions = _sample_positions(line, spacing)
        if closed:
            positions = positions[:-1]
        if len(positions) < 2:
            continue
        pts = np.asarray([line.interpolate(float(d)).coords[0][:2] for d in positions], dtype=float)
        points = shapely.points(pts[:, 0], pts[:, 1])
        nearest = np.asarray(tree.nearest(points), dtype=int)
        normals = _normals(pts, closed)
        top = np.full(len(pts), np.nan)
        foot = np.full(len(pts), np.nan)
        lower = np.zeros(len(pts), dtype=int)
        near = np.zeros(len(pts), dtype=bool)
        for k, (point, index) in enumerate(zip(points, nearest)):
            curb, xyz, axis = axes[index]
            if axis.distance(point) > snap:
                continue
            near[k] = True
            arc = axis.project(point)
            cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
            top[k] = np.interp(arc, cum, xyz[:, 2])
            foot_all = getattr(curb, "foot_z", None)
            if foot_all is not None and len(np.asarray(foot_all)) >= 2:
                foot[k] = np.interp(arc / max(axis.length, _EPS), np.linspace(0.0, 1.0, len(np.asarray(foot_all))), np.asarray(foot_all, dtype=float))
            side = int(getattr(curb, "road_side", 0) or 0)
            if side:
                ahead = np.asarray(axis.interpolate(min(arc + 0.1, axis.length)).coords[0][:2]) - np.asarray(axis.interpolate(max(arc - 0.1, 0.0)).coords[0][:2])
                mine = np.array([normals[k][1], -normals[k][0]])        # walking direction of the outline (normal turned back)
                lower[k] = side if float(np.dot(ahead, mine)) >= 0.0 else -side
        if not near.any():
            continue
        zl = np.asarray(sampler(pts + probe * normals), dtype=float).reshape(-1)
        zr = np.asarray(sampler(pts - probe * normals), dtype=float).reshape(-1)
        probe_side = np.where(zl < zr, 1, -1)
        lower = np.where(lower != 0, lower, probe_side)
        probe_low = np.where(lower > 0, zl, zr)
        probe_up = np.where(lower > 0, zr, zl)
        top = np.where(np.isfinite(top), top, probe_up)
        bottom = np.where(np.isfinite(foot), foot, probe_low)
        bottom = np.where(np.isfinite(bottom), bottom, probe_low)
        jump = _median3(top - bottom, closed)
        ok = near & (jump >= float(cfg["min_jump_m"])) & (jump <= float(cfg["curb_max_jump_m"]))
        sign = np.where(ok, lower, 0).astype(int)
        for idx in _stretches(sign, closed):
            whole = closed and len(idx) == len(pts)
            if len(idx) < 2 and not whole:
                continue
            if not whole and float(np.linalg.norm(np.diff(pts[idx], axis=0), axis=1).sum()) < float(cfg["min_run_m"]):
                continue
            xy, z_top, z_bottom = pts[idx], _median3(top[idx], False), _median3(bottom[idx], False)
            if whole:
                xy, z_top, z_bottom = np.vstack((xy, xy[:1])), np.r_[z_top, z_top[:1]], np.r_[z_bottom, z_bottom[:1]]
            runs.append(StepRun(xy=xy, z_top=z_top, z_bottom=z_bottom, lower_side=int(sign[idx[0]]), closed=whole, source="curb"))
    return runs


def detect_steps(
    regions: Mapping[str, Iterable] | None,
    dtm_sampler: Callable,
    *,
    curbs: Iterable = (),
    extra_lines: Iterable[LineString] = (),
    dtm: np.ndarray | None = None,
    dtm_bounds: tuple[float, float, float, float] | None = None,
    dtm_resolution: float | None = None,
    config: Mapping | None = None,
) -> list[StepRun]:
    """Step runs, only where a real vertical object is: curb axes and clearly evidenced walls.

    * ``curbs`` (``CurbAxis`` or (N,3) arrays, z = curb top): top edge = curb top, bottom edge = road-side surface
      (``foot_z`` / ``road_side`` when present, else the lower probe), kept where the jump is at least ``min_jump_m``.
    * Steep DTM lines (``terrain.step.dtm_features``, gradient over ``min_slope_deg`` and ``min_length_m`` long) and
      ``extra_lines`` count as retaining walls only when the jump is at least ``wall_min_jump_m`` (0.25 m) at both
      ``probe_m`` and twice that distance.
    Pavement outlines alone never create a step: the pavement edge is flush with the terrain
    (see :mod:`samcloud.terrain.flush`). ``regions`` is accepted for compatibility and ignored.
    """
    cfg = step_config(config)
    if not cfg["enabled"]:
        return []
    curbs = list(curbs)
    probe = float(cfg["probe_m"])
    outlines = _region_lines({k: v for k, v in regions.items()}) if regions else []
    runs: list[StepRun] = _curb_runs_on_lines(outlines, curbs, dtm_sampler, cfg) if outlines else _curb_runs(curbs, dtm_sampler, cfg)
    if regions:   # a clear height jump between two pavements (a curb the curb detection missed) is a step too
        pave = [part for name in ("road", "sidewalk", "path") for value in regions.get(name, []) or [] for part in _polygon_parts(value)]
        if pave:
            curb_geom = unary_union([LineString(r.xy) for r in runs]) if runs else None
            union = unary_union(pave)
            shapely.prepare(union)

            def both_pavement(pts: np.ndarray, normals: np.ndarray) -> np.ndarray:
                left, right = pts + probe * normals, pts - probe * normals
                both = np.asarray(shapely.contains_xy(union, left[:, 0], left[:, 1]) & shapely.contains_xy(union, right[:, 0], right[:, 1]), dtype=bool)
                if curb_geom is not None:
                    both &= ~np.asarray(shapely.dwithin(shapely.points(pts[:, 0], pts[:, 1]), curb_geom, 0.1), dtype=bool)
                return both

            for line in _region_lines({k: v for k, v in regions.items() if k in ("road", "sidewalk", "path")}):
                runs.extend(_runs_of_line(line, dtm_sampler, cfg, probe, "pavement", min_jump=float(cfg["pavement_jump_m"]),
                                          confirm_probe=2.0 * probe, accept=both_pavement))
    feature_lines = list(extra_lines)
    if cfg["dtm_features"] and dtm is not None and dtm_bounds is not None and dtm_resolution:
        feature_lines.extend(detect_dtm_step_lines(dtm, dtm_bounds, dtm_resolution, config))
    wide = max(probe, 1.5 * float(dtm_resolution or 0.0))
    curb_lines = [LineString(_curb_xyz(cb)[:, :2]) for cb in curbs if len(_curb_xyz(cb)) >= 2]
    curb_geom = unary_union(curb_lines).buffer(0.3) if curb_lines else None
    for line in feature_lines:
        if curb_geom is not None and line.intersection(curb_geom).length > 0.5 * line.length:
            continue  # the curb step already covers it
        runs.extend(_runs_of_line(line, dtm_sampler, cfg, wide, "wall", min_jump=float(cfg["wall_min_jump_m"]),
                                  confirm_probe=2.0 * wide))
    LOGGER.info("%d step run(s), %.1f m", len(runs), sum(r.length for r in runs))
    return runs


# ------------------------------------------------------------------------------------------------ TIN split


def _signed_area2(xy: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _chains(run: StepRun, vertices: np.ndarray, edges: set[tuple[int, int]]) -> list[list[int]]:
    """Ordered TIN vertex chains along the run line (broken where the TIN lacks the edge)."""
    line = LineString(run.xy)
    pts = shapely.points(vertices[:, 0], vertices[:, 1])
    near = np.flatnonzero(shapely.dwithin(pts, line, _ON_LINE_M))
    if len(near) < 2:
        return []
    arc = shapely.line_locate_point(line, pts[near])
    order = near[np.argsort(arc, kind="stable")]
    chains: list[list[int]] = []
    current: list[int] = [int(order[0])]
    for a, b in zip(order[:-1], order[1:]):
        if tuple(sorted((int(a), int(b)))) in edges:
            current.append(int(b))
        else:
            if len(current) >= 2:
                chains.append(current)
            current = [int(b)]
    if len(current) >= 2:
        chains.append(current)
    if run.closed and chains:
        first, last = chains[0], chains[-1]
        if first[0] != last[-1] and tuple(sorted((first[0], last[-1]))) in edges:
            if len(chains) == 1:
                chains[0] = first + [first[0]]
            else:
                chains[0] = last + first
                chains.pop()
    return chains


class _Vertices:
    """Growable vertex array."""

    def __init__(self, vertices: np.ndarray) -> None:
        self.n = len(vertices)
        self.buf = np.empty((self.n + 1024, 3))
        self.buf[: self.n] = vertices

    @property
    def view(self) -> np.ndarray:
        return self.buf[: self.n]

    def add(self, vertex: np.ndarray) -> int:
        if self.n == len(self.buf):
            self.buf = np.vstack((self.buf, np.empty((len(self.buf), 3))))
        self.buf[self.n] = vertex
        self.n += 1
        return self.n - 1


def _base_ids(array: np.ndarray, original: Mapping[int, int]) -> np.ndarray:
    """Vertex ids with duplicates mapped back to their original vertex."""
    if not original:
        return array
    out = array.copy()
    for new, old in original.items():
        out[array == new] = old
    return out


def apply_steps(tin, runs: Iterable[StepRun], config: Mapping | None = None):
    """Split ``tin`` along ``runs``: top edge at the upper height, bottom edge ``offset_m`` outside at the lower height.

    Operates in place and returns ``tin``. Interior chain vertices are duplicated: the original vertex keeps the
    triangles of the upper side (z = top), the copy (moved ``offset_m`` to the lower side, z = bottom) takes the lower
    fan. Two strip triangles per chain edge close the gap and take the label ``step``. Chain ends stay single
    vertices (z = top), so the surface remains closed. ``tin.step_report`` lists segments, length and mean jump.

    The strip triangles get their own label ``TIN_LABELS['step']``: they belong to no pavement body (a vertical face of a
    pavement outline must not enter its top surface) and are written with the terrain.
    """
    cfg = step_config(config)
    runs = [r for r in runs if len(r.xy) >= 2]
    report = {"runs": len(runs), "segments": 0, "length_m": 0.0, "mean_jump_m": 0.0, "duplicated_vertices": 0, "skipped_vertices": 0}
    tin.step_report = report
    if not runs or not len(tin.triangles):
        return tin
    offset = float(cfg["offset_m"])
    store = _Vertices(np.asarray(tin.vertices, dtype=float))
    triangles = np.asarray(tin.triangles, dtype=np.int64).copy()
    labels = np.asarray(tin.labels).copy()
    new_tris: list[tuple[int, int, int]] = []
    new_labels: list[int] = []
    duplicated: dict[int, int] = {}
    original: dict[int, int] = {}      # duplicate id -> original id
    total_length, segments = 0.0, 0
    base_count = len(tin.vertices)

    for run in runs:
        mapped = _base_ids(triangles, original)
        pairs_all = np.vstack((mapped[:, [0, 1]], mapped[:, [1, 2]], mapped[:, [2, 0]]))
        pairs_all.sort(axis=1)
        edges = {(int(a), int(b)) for a, b in pairs_all}
        run_line = LineString(run.xy)
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(run.xy, axis=0), axis=1))]
        for chain in _chains(run, store.view[:base_count], edges):
            closed = len(chain) > 2 and chain[0] == chain[-1]
            ring = chain[:-1] if closed else chain
            count = len(ring)
            xy_ring = store.view[ring][:, :2]
            arc = shapely.line_locate_point(run_line, shapely.points(xy_ring[:, 0], xy_ring[:, 1]))
            top, bottom = np.interp(arc, cum, run.z_top), np.interp(arc, cum, run.z_bottom)
            for v, zt in zip(ring, top):
                store.buf[v, 2] = zt
            incident = _incident(_base_ids(triangles, original), ring)
            chain_dup: dict[int, int] = {}
            for k, v in enumerate(ring):
                if not (closed or 0 < k < count - 1) or v in duplicated:
                    continue
                prev_v, next_v = ring[(k - 1) % count], ring[(k + 1) % count]
                lower = _lower_fan(store.view, triangles, incident[v], v, prev_v, next_v, run.lower_side, original)
                if lower is None:
                    report["skipped_vertices"] += 1
                    continue
                n1 = _lower_normal(store.buf[v, :2] - store.buf[prev_v, :2], run.lower_side)
                n2 = _lower_normal(store.buf[next_v, :2] - store.buf[v, :2], run.lower_side)
                miter = (n1 + n2) / max(1.0 + float(np.dot(n1, n2)), 0.5)
                for factor in (1.0, 0.5, 0.25):
                    candidate = np.array([store.buf[v, 0] + offset * factor * miter[0],
                                          store.buf[v, 1] + offset * factor * miter[1], float(bottom[k])])
                    if _fan_ok(store.view, triangles, lower, v, candidate):
                        new_id = store.add(candidate)
                        for t in lower:
                            triangles[t][triangles[t] == v] = new_id
                        duplicated[v], chain_dup[v], original[new_id] = new_id, new_id, v
                        break
                else:
                    report["skipped_vertices"] += 1
            for a, b in zip(ring, ring[1:] + ([ring[0]] if closed else [])):
                da, db = chain_dup.get(a), chain_dup.get(b)
                if da is None and db is None:
                    continue
                label = STEP_LABEL
                if da is not None and db is not None:
                    strip = [(a, b, db), (a, db, da)]
                else:
                    strip = [(a, b, da)] if da is not None else [(a, b, db)]
                new_tris.extend(strip)
                new_labels.extend([label] * len(strip))
                segments += 1
                total_length += float(np.linalg.norm(store.buf[a, :2] - store.buf[b, :2]))
    vertices = store.view.copy()
    if new_tris:
        add = np.asarray(new_tris, dtype=np.int64)
        flip = _signed_area2(vertices[:, :2], add) < 0
        add[flip] = add[flip][:, [0, 2, 1]]
        triangles = np.vstack((triangles, add))
        labels = np.concatenate((labels, np.asarray(new_labels, dtype=labels.dtype)))
    jumps = [float(vertices[v, 2] - vertices[d, 2]) for v, d in duplicated.items()]
    tin.vertices, tin.triangles, tin.labels = vertices, triangles, labels
    report.update(segments=int(segments), length_m=round(total_length, 3),
                  mean_jump_m=round(float(np.mean(jumps)), 4) if jumps else 0.0, duplicated_vertices=len(duplicated))
    return tin


def _incident(mapped: np.ndarray, vertex_ids: list[int]) -> dict[int, list[int]]:
    """Triangle indices per vertex of ``vertex_ids`` (``mapped``: triangles with duplicates mapped to originals)."""
    wanted = list(set(vertex_ids))
    out: dict[int, list[int]] = {v: [] for v in wanted}
    for column in range(3):
        for t in np.flatnonzero(np.isin(mapped[:, column], wanted)):
            out[int(mapped[t, column])].append(int(t))
    return out


def _lower_normal(direction: np.ndarray, lower_side: int) -> np.ndarray:
    length = float(np.hypot(direction[0], direction[1]))
    if length < _EPS:
        return np.zeros(2)
    return np.array([-direction[1], direction[0]]) / length * lower_side


def _lower_fan(
    vertices: np.ndarray, triangles: np.ndarray, incident: list[int], v: int, prev_v: int, next_v: int,
    lower_side: int, original: Mapping[int, int],
) -> list[int] | None:
    """Triangles around ``v`` on the lower side of the chain (None unless the chain splits the fan in exactly two)."""
    def base(w) -> int:
        return original.get(int(w), int(w))

    by_neighbour: dict[int, list[int]] = {}
    for t in incident:
        for w in triangles[t]:
            if base(w) != v:
                by_neighbour.setdefault(base(w), []).append(t)
    parent = {t: t for t in incident}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for w, tris in by_neighbour.items():
        if w in (prev_v, next_v):
            continue
        for other in tris[1:]:
            parent[find(other)] = find(tris[0])
    groups: dict[int, list[int]] = {}
    for t in incident:
        groups.setdefault(find(t), []).append(t)
    if len(groups) != 2:
        return None
    direction = vertices[next_v, :2] - vertices[v, :2]
    for t in by_neighbour.get(next_v, []):
        third = [int(w) for w in triangles[t] if base(w) not in (v, next_v)]
        if not third:
            continue
        rel = vertices[third[0], :2] - vertices[v, :2]
        if (direction[0] * rel[1] - direction[1] * rel[0]) * lower_side > 0:
            return sorted(groups[find(t)])
    return None


def _fan_ok(vertices: np.ndarray, triangles: np.ndarray, lower: list[int], v: int, candidate: np.ndarray) -> bool:
    """The lower fan keeps its orientation and a positive area when ``v`` moves to ``candidate``."""
    tri = triangles[lower]
    before = _signed_area2(vertices[:, :2], tri)
    moved = vertices.copy()
    moved[v] = candidate
    after = _signed_area2(moved[:, :2], tri)
    return bool(np.all(np.sign(before) == np.sign(after)) and np.all(np.abs(after) > 2 * _MIN_AREA_M2))


def summarize(runs: Iterable[StepRun]) -> dict:
    """Number, total length and mean jump of ``runs``."""
    runs = list(runs)
    length = sum(r.length for r in runs)
    mean = float(sum(r.mean_jump * r.length for r in runs) / length) if length > 0 else 0.0
    return {"runs": len(runs), "length_m": round(length, 3), "mean_jump_m": round(mean, 4)}


__all__ = ["StepRun", "apply_steps", "detect_dtm_step_lines", "detect_steps", "step_config", "summarize"]
