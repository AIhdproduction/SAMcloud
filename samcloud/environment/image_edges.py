"""Refine the boundaries between surfaces (road, sidewalk, path, terrain) with the top-down image.

The class raster gives boundaries that are good to a few decimetres.  For every boundary chain
between two different surfaces this module searches perpendicular to the chain for the strongest
colour (CIE Lab) and height step in the ortho raster, finds a coherent offset curve with dynamic
programming, fits lines and arcs to it and accepts it when enough of the chain is supported by the
image.  Shared edges are refined once (both neighbours get the identical line), curb/gutter chains
(master breaklines) are never moved, and nodes where chains meet are moved to the intersection of
the refined lines so T-junctions keep a single node.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge
from shapely.validation import make_valid

from samcloud.environment.ortho_rgb import OrthoRGB
from samcloud.terrain.edges import BulgeLine, _curb_xyz, _explode, _fit_circle, _resample, densify, fit_chain, fit_settings, join_lines

LOGGER = logging.getLogger(__name__)

_GRID = 1e-3
_TRIM = 2  # samples dropped at both ends of an open chain before fitting
_DEFAULTS: dict = {
    "enabled": True,
    "cell_m": 0.03,             # ortho cell size
    "z_percentile": 0.7,        # colour of the points above this height percentile of a cell
    "hole_fill_cells": 3,       # holes up to this many cells are filled
    "search_m": 0.30,           # perpendicular search distance
    "step_m": 0.02,             # offset step
    "window_m": 0.06,           # half width of the before/after windows of the step detector
    "along_m": 0.6,             # profiles are averaged over this length along the chain
    "sigma_m": 0.03,            # Gaussian smoothing of the Lab image
    "weight_color": 1.0,        # weight of the Lab step
    "weight_height": 1.0,       # weight of the height step (1 cm = this many Lab units)
    "min_contrast": 5.0,        # a sample counts as supported when its combined step is at least this
    "prior_per_m": 1.0,         # cost per metre of shift (only breaks ties near the class-raster edge)
    "smooth_cost": 1.5,         # cost per step of offset change between neighbouring samples
    "max_step": 3,              # largest offset change (in steps) between neighbouring samples
    "passes": 2,                # the search is repeated along the refined line (removes the along-chain blur)
    "pass2_search_m": 0.10,      # search distance of the repeat
    "straight": True,           # robust line segmentation of the refined samples
    "min_segment_m": 1.5,       # shortest straight segment
    "inlier_m": 0.04,           # RANSAC inlier tolerance
    "merge_gap_samples": 8,     # segments this close (samples) may be merged
    "merge_deg": 4.0,           # consecutive segments with a smaller direction change are merged
    "snap_deg": 6.0,            # segment directions snap to the site's dominant orientations within this
    "min_arc_radius_m": 2.0,    # arcs only from this radius and only when clearly better than lines
    "ransac_trials": 150,       # hypotheses per RANSAC search
    "hough": True,              # image line segments near the boundary are extra line hypotheses
    "hough_parallel_deg": 8.0,  # ... when parallel to the class boundary within this
    "hough_min_len_m": 1.0,
    "min_support": 0.60,        # share of supported samples that must agree with the fitted line
    "fit_tolerance_m": 0.03,    # line/arc fit tolerance for image-derived samples (tighter than edges.fit)
    "agree_m": 0.04,            # a sample agrees when it lies this close to the fitted line
    "max_area_change": 0.05,    # per class; above this the class-raster outlines are kept
    "min_edge_m": 0.5,          # short straight edges that are only zig-zags are removed
    "zigzag_dev_m": 0.05,       # ... when removing the vertex moves the line by less than this
    "node_shift_m": 0.45,       # nodes move at most this far
    "min_chain_m": 1.0,         # shorter chains stay as they are
    "side_offset_m": 0.15,      # distance of the probes that tell which surface lies on each side
}


_CLEAN_DEFAULTS: dict = {
    "enabled": True,
    "spike_m": 0.25,            # a spike is narrower than this at its base ...
    "spike_protrude_m": 0.10,   # ... and sticks out further than this
    "line_tol_m": 0.08,         # RANSAC inlier tolerance without image support
    "min_segment_m": 1.5,
    "merge_gap_samples": 40,
    "max_shift_m": 0.25,        # from the class-raster edge
    "max_area_change": 0.05,    # per polygon; cleaned chains that break it are reverted
}


def clean_settings(config: Mapping | None) -> dict:
    """``edges.clean`` of the drone config merged over the built-in defaults."""
    out = dict(_CLEAN_DEFAULTS)
    node = (config or {}).get("edges", {}) if isinstance(config, Mapping) else {}
    out.update(node.get("clean", {}) if isinstance(node, Mapping) else {})
    return out


def image_settings(config: Mapping | None) -> dict:
    """``edges.image`` of the drone config merged over the built-in defaults."""
    out = dict(_DEFAULTS)
    node = (config or {}).get("edges", {}) if isinstance(config, Mapping) else {}
    out.update(node.get("image", {}) if isinstance(node, Mapping) else {})
    return out


@dataclass
class ImageEdgeResult:
    """``regions[name][i]`` lists the polygons that replace input polygon ``i`` of class ``name``."""

    regions: dict[str, list[list[Polygon]]]
    report: dict = field(default_factory=dict)


@dataclass
class _Chain:
    xy: np.ndarray
    closed: bool
    start: tuple | None = None
    end: tuple | None = None
    pair: tuple[str, str] | None = None
    fixed: bool = False
    fit: BulgeLine | None = None
    support: float = 0.0
    shift: float = 0.0
    note: str = ""
    final: np.ndarray | None = None
    trimmed: bool = False
    p: np.ndarray | None = None
    nrm: np.ndarray | None = None
    hyps: list = field(default_factory=list)
    segs: list = field(default_factory=list)
    spikes: int = 0


def _key(point) -> tuple[int, int]:
    return (int(round(point[0] / _GRID)), int(round(point[1] / _GRID)))


def _tangents(p: np.ndarray) -> np.ndarray:
    t = np.gradient(p, axis=0)
    if len(p) > 5:
        t = ndimage.uniform_filter1d(t, 3, axis=0, mode="nearest")
    n = np.hypot(t[:, 0], t[:, 1])
    return t / np.maximum(n, 1e-12)[:, None]


def _window_means(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``k`` entries before and after every position along axis 1 (nan where the window is cut)."""
    csum = np.concatenate((np.zeros_like(values[:, :1]), np.cumsum(values, axis=1)), axis=1)
    nt = values.shape[1]
    j = np.arange(nt)
    before = np.full(values.shape, np.nan)
    after = np.full(values.shape, np.nan)
    ok_b = j >= k
    before[:, ok_b] = (csum[:, j[ok_b]] - csum[:, j[ok_b] - k]) / k
    ok_a = j + k < nt
    after[:, ok_a] = (csum[:, j[ok_a] + k + 1] - csum[:, j[ok_a] + 1]) / k
    return before, after


def _along_smooth(values: np.ndarray, covered: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Average over ``size`` samples along the chain (axis 0) using only covered samples."""
    w = covered.astype(float)
    if values.ndim == 3:
        w3 = w[..., None]
        num = ndimage.uniform_filter1d(np.where(covered[..., None], values, 0.0), size, axis=0, mode="nearest")
        den = ndimage.uniform_filter1d(w3, size, axis=0, mode="nearest")
    else:
        num = ndimage.uniform_filter1d(np.where(covered, values, 0.0), size, axis=0, mode="nearest")
        den = ndimage.uniform_filter1d(w, size, axis=0, mode="nearest")
    ok = den > 0.5
    return np.where(ok, num / np.maximum(den, 1e-9), 0.0), ok[..., 0] if values.ndim == 3 else ok


def _step_strength(p: np.ndarray, n: np.ndarray, ortho: OrthoRGB, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combined step strength for every sample and candidate offset.

    Returns ``(strength (m, nc), covered (m, nc), offsets (nc,))`` with candidates in ``[-search, search]``.
    """
    dt = float(cfg["step_m"])
    k = max(1, int(round(float(cfg["window_m"]) / dt)))
    s = int(round(float(cfg["search_m"]) / dt))
    offsets_all = np.arange(-(s + k), s + k + 1) * dt
    m, nt = len(p), len(offsets_all)
    pts = (p[:, None, :] + offsets_all[None, :, None] * n[:, None, :]).reshape(-1, 2)
    lab, cov = ortho.sample_lab(pts, float(cfg["sigma_m"]))
    lab, cov = lab.reshape(m, nt, 3), cov.reshape(m, nt)
    size = max(1, int(round(float(cfg["along_m"]) / max(float(np.median(np.hypot(*np.diff(p, axis=0).T))), 1e-3))))
    lab_s, cov_s = _along_smooth(lab, cov, size)
    nb = np.stack([_window_means(lab_s[..., ch], k) for ch in range(3)], axis=0)  # (3, 2, m, nt)
    diff = nb[:, 1] - nb[:, 0]
    color = np.sqrt(np.nansum(diff ** 2, axis=0))
    covered = cov_s
    # a step needs covered samples on both sides
    cov_b = _window_means(covered.astype(float), k)
    covered_all = (cov_b[0] > 0.99) & (cov_b[1] > 0.99) & covered
    strength = float(cfg["weight_color"]) * np.nan_to_num(color)
    if ortho.height is not None and float(cfg["weight_height"]) > 0:
        z, zcov = ortho.sample_height(pts)
        z, zcov = z.reshape(m, nt), zcov.reshape(m, nt)
        z_s, zok = _along_smooth(np.nan_to_num(z), zcov, size)
        zb, za = _window_means(z_s, k)
        zc_b = _window_means(zok.astype(float), k)
        zstep = np.abs(np.nan_to_num(za - zb)) * ((zc_b[0] > 0.99) & (zc_b[1] > 0.99) & zok)
        strength = strength + float(cfg["weight_height"]) * 100.0 * zstep
    sel = slice(k, nt - k)
    return strength[:, sel], covered_all[:, sel], offsets_all[sel]


def _best_path(strength: np.ndarray, covered: np.ndarray, offsets: np.ndarray, cfg: dict) -> np.ndarray:
    """Offset index per sample: maximise the step strength with a smoothness and a small distance prior."""
    m, nc = strength.shape
    cost = -np.where(covered, strength, 0.0) + float(cfg["prior_per_m"]) * np.abs(offsets)[None, :]
    ms = int(cfg["max_step"])
    smooth = float(cfg["smooth_cost"])
    acc = np.empty_like(cost)
    arg = np.zeros((m, nc), dtype=np.int16)
    acc[0] = cost[0]
    for i in range(1, m):
        best = np.full(nc, np.inf)
        choice = np.zeros(nc, dtype=np.int16)
        for d in range(-ms, ms + 1):
            lo, hi = max(0, d), min(nc, nc + d)  # target j in [lo, hi), source j - d
            cand = acc[i - 1, lo - d:hi - d] + smooth * abs(d)
            better = cand < best[lo:hi]
            best[lo:hi] = np.where(better, cand, best[lo:hi])
            choice[lo:hi] = np.where(better, np.arange(lo, hi) - d, choice[lo:hi])
        acc[i] = cost[i] + best
        arg[i] = choice
    path = np.empty(m, dtype=np.int64)
    path[-1] = int(np.argmin(acc[-1]))
    for i in range(m - 1, 0, -1):
        path[i - 1] = arg[i, path[i]]
    return path


def _dev(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Distance of ``b`` from the chord ``a -> c``."""
    chord = c - a
    length = float(np.hypot(*chord))
    if length < 1e-12:
        return float(np.hypot(*(b - a)))
    return abs(chord[0] * (b - a)[1] - chord[1] * (b - a)[0]) / length


def drop_short_edges(line: BulgeLine, min_edge: float, max_dev: float) -> BulgeLine:
    """Remove zig-zags: straight edges shorter than ``min_edge`` that deviate less than ``max_dev``.

    A short edge is removed together with its two vertices when both lie within ``max_dev`` of the
    chord of their neighbours, or one vertex goes when it is within ``max_dev`` of its neighbours'
    chord.  Arcs and real corners (larger deviation) stay.
    """
    pts, bulges = [np.asarray(q, float) for q in line.pts], [float(b) for b in line.bulges]
    changed = True
    while changed and len(pts) > 2:
        changed = False
        for i in range(1, len(pts) - 2):
            if bulges[i - 1] != 0.0 or bulges[i] != 0.0 or bulges[i + 1] != 0.0:
                continue
            a, b, c, d = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
            if float(np.hypot(*(c - b))) >= min_edge:
                continue
            if max(_dev(a, b, d), _dev(a, c, d)) <= max_dev:
                del pts[i:i + 2], bulges[i:i + 2]
                changed = True
                break
        if changed:
            continue
        for i in range(1, len(pts) - 1):
            if bulges[i - 1] != 0.0 or bulges[i] != 0.0:
                continue
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            if min(float(np.hypot(*(b - a))), float(np.hypot(*(c - b)))) >= min_edge:
                continue
            if _dev(a, b, c) <= max_dev:
                del pts[i], bulges[i]
                changed = True
                break
    return BulgeLine(np.asarray(pts), np.asarray(bulges), dict(line.report))


def _end_line(dense: np.ndarray, at_end: bool, reach: float = 0.8) -> tuple[np.ndarray, np.ndarray] | None:
    """Point and unit direction of the chain's end piece (``reach`` metres from the end)."""
    d = dense[::-1] if at_end else dense
    seg = np.hypot(*np.diff(d, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    part = d[cum <= max(reach, cum[1] if len(cum) > 1 else 0.0)]
    if len(part) < 2:
        return None
    centre = part.mean(axis=0)
    _u, _s, vt = np.linalg.svd(part - centre, full_matrices=False)
    return centre, vt[0]


def _node_position(node: np.ndarray, lines: list[tuple[np.ndarray, np.ndarray]], max_shift: float) -> np.ndarray:
    """Least-squares intersection of the refined end lines, or the projection onto the single line."""
    if not lines:
        return node
    if len(lines) == 1:
        c, u = lines[0]
        new = c + u * float((node - c) @ u)
    else:
        angles = [abs(float(a[1][0] * b[1][1] - a[1][1] * b[1][0])) for ia, a in enumerate(lines) for b in lines[ia + 1:]]
        if max(angles) < math.sin(math.radians(10.0)):
            c, u = lines[0]
            new = c + u * float((node - c) @ u)
        else:
            mat, rhs = np.zeros((2, 2)), np.zeros(2)
            for c, u in lines:
                proj = np.eye(2) - np.outer(u, u)
                mat += proj
                rhs += proj @ c
            new = np.linalg.solve(mat, rhs)
    return new if float(np.hypot(*(new - node))) <= max_shift else node


def _line_dev(q: np.ndarray, c: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.abs((q - c) @ np.array([-u[1], u[0]]))


def _longest_run(mask: np.ndarray, d: np.ndarray, gap: int = 2) -> tuple[int, int] | None:
    """Longest run (by arc length ``d``) of True samples, bridging gaps of up to ``gap`` samples."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    splits = np.flatnonzero(np.diff(idx) > gap + 1) + 1
    best, best_len = None, -1.0
    for part in np.split(idx, splits):
        length = float(d[part[-1]] - d[part[0]])
        if length > best_len:
            best, best_len = (int(part[0]), int(part[-1])), length
    return best


def _best_line(q: np.ndarray, d: np.ndarray, lo: int, hi: int, cfg: dict, rng, hyps) -> tuple | None:
    """Longest straight run of ``q[lo:hi+1]`` within ``inlier_m``: ``(i0, i1, centre, direction)``."""
    n = hi - lo + 1
    base = float(cfg["inlier_m"])
    tol = base * float(cfg.get("expand_factor", 1.0))   # a run may wobble up to this far if half of it is within ``inlier_m``
    if n < 6 or d[hi] - d[lo] < float(cfg["min_segment_m"]):
        return None
    sub, ds = q[lo:hi + 1], d[lo:hi + 1]
    a = rng.integers(0, n, int(cfg["ransac_trials"]))
    b = rng.integers(0, n, int(cfg["ransac_trials"]))
    ok = np.abs(ds[a] - ds[b]) >= 0.5
    cands = [(sub[i], (sub[j] - sub[i]) / max(float(np.hypot(*(sub[j] - sub[i]))), 1e-9)) for i, j in zip(a[ok], b[ok])]
    cands += [(c, u) for c, u in hyps]
    centre0 = sub.mean(axis=0)
    cands.append((centre0, np.linalg.svd(sub - centre0, full_matrices=False)[2][0]))   # one line through everything
    best, best_len = None, 0.0
    for c, u in cands:
        dev = _line_dev(sub, c, u)
        run = _longest_run(dev <= tol, ds)
        if run is not None and ds[run[1]] - ds[run[0]] > best_len:
            if tol > base and float((dev[run[0]:run[1] + 1] <= base).mean()) < 0.5:
                continue
            best, best_len = run, float(ds[run[1]] - ds[run[0]])
    if best is None or best_len < float(cfg["min_segment_m"]):
        return None
    for _ in range(2):  # refit on the run and re-collect the inliers
        pts = sub[best[0]:best[1] + 1]
        c = pts.mean(axis=0)
        u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
        run = _longest_run(_line_dev(sub, c, u) <= tol, ds)
        if run is None:
            return None
        best = run
    pts = sub[best[0]:best[1] + 1]
    length = float(ds[best[1]] - ds[best[0]])
    if length < float(cfg["min_segment_m"]):
        return None
    c = pts.mean(axis=0)
    u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    # the ends of a run that leaves into a curve stay within the tolerance for a while: trim them
    first, last = best
    for _ in range(3):
        dev = _line_dev(sub[first:last + 1], c, u)
        lo_i, hi_i = 0, len(dev) - 1
        while lo_i < hi_i and dev[lo_i] > 0.5 * tol:
            lo_i += 1
        while hi_i > lo_i and dev[hi_i] > 0.5 * tol:
            hi_i -= 1
        if lo_i == 0 and hi_i == len(dev) - 1:
            break
        first, last = first + lo_i, first + hi_i
        if last - first < 4:
            return None
        pts = sub[first:last + 1]
        c = pts.mean(axis=0)
        u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    best = (first, last)
    length = float(ds[best[1]] - ds[best[0]])
    if length < float(cfg["min_segment_m"]):
        return None
    if float((pts[-1] - pts[0]) @ u) < 0:
        u = -u
    # an arc that fits clearly better than the line is not a straight segment
    line_dev = float(_line_dev(pts, c, u).max())
    if line_dev > 0.012 and len(pts) >= 8:
        try:
            _cc, radius, circ_dev = _fit_circle(pts)
        except Exception:  # noqa: BLE001
            radius, circ_dev = 0.0, line_dev
        if float(cfg["min_arc_radius_m"]) <= radius <= 200.0 and circ_dev < 0.5 * line_dev:
            return None
    return best[0] + lo, best[1] + lo, c, u


def straight_segments(q: np.ndarray, cfg: dict, hyps=(), seed: int = 7) -> list[tuple]:
    """Straight segments ``(i0, i1, centre, direction)`` of the ordered samples ``q`` (RANSAC, recursive)."""
    q = np.asarray(q, float)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
    rng = np.random.default_rng(seed)
    out: list[tuple] = []
    stack = [(0, len(q) - 1)]
    while stack:
        lo, hi = stack.pop()
        found = _best_line(q, d, lo, hi, cfg, rng, hyps)
        if found is None:
            continue
        out.append(found)
        stack += [(lo, found[0] - 1), (found[1] + 1, hi)]
    out.sort(key=lambda t: t[0])
    # merge consecutive segments that are (nearly) collinear
    merged = True
    tol = float(cfg["inlier_m"])
    while merged and len(out) > 1:
        merged = False
        for k in range(len(out) - 1):
            a, b = out[k], out[k + 1]
            ang = math.degrees(math.acos(min(1.0, abs(float(a[3] @ b[3])))))
            if ang >= float(cfg["merge_deg"]) or b[0] - a[1] > int(cfg["merge_gap_samples"]):
                continue
            pts = q[a[0]:b[1] + 1]
            c = pts.mean(axis=0)
            u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
            if float(_line_dev(pts, c, u).max()) <= 1.5 * tol:
                if float((pts[-1] - pts[0]) @ u) < 0:
                    u = -u
                out[k:k + 2] = [(a[0], b[1], c, u)]
                merged = True
                break
    return out


def dominant_orientations(segments: list[tuple[float, float]], *, min_share: float = 0.08, min_length_m: float = 10.0) -> list[float]:
    """Peaks (radians, mod pi) of the length-weighted direction histogram of ``(angle, length)`` pairs."""
    if not segments:
        return []
    deg = np.array([math.degrees(a) % 180.0 for a, _l in segments])
    wgt = np.array([l for _a, l in segments], float)
    hist = np.bincount(np.minimum(deg.astype(int), 179), weights=wgt, minlength=180)
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 1.5) ** 2)
    smooth = np.convolve(np.r_[hist[-6:], hist, hist[:6]], kernel, mode="same")[6:-6]
    total = float(hist.sum())
    bins = np.arange(180)
    peaks: list[float] = []
    while True:
        i = int(np.argmax(smooth))
        if smooth[i] < max(min_share * total, min_length_m):
            break
        near = (bins - i + 90) % 180 - 90
        delta = (deg - (i + 0.5) + 90.0) % 180.0 - 90.0
        sel = np.abs(delta) <= 3.5
        centre = (i + 0.5) + (float((delta[sel] * wgt[sel]).sum() / wgt[sel].sum()) if sel.any() else 0.0)
        peaks.append(float(centre % 180.0))
        smooth[np.abs(near) <= 6] = 0.0
    return [math.radians(a) for a in peaks]


def _snap_direction(u: np.ndarray, dirs: Sequence[float], snap_deg: float) -> np.ndarray | None:
    ang = math.atan2(u[1], u[0]) % math.pi
    best, best_diff = None, math.radians(snap_deg)
    for d in dirs:
        diff = abs((ang - d + math.pi / 2) % math.pi - math.pi / 2)
        if diff <= best_diff:
            best, best_diff = d, diff
    if best is None:
        return None
    v = np.array([math.cos(best), math.sin(best)])
    return v if float(v @ u) > 0 else -v


def _fit_res(points: np.ndarray, tight: dict) -> BulgeLine:
    """Lines/arcs through ``points`` (first and last stay exactly)."""
    pts = np.asarray(points, float)
    pts = pts[np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-9]] if len(pts) > 1 else pts
    if len(pts) < 4:
        return BulgeLine(pts, np.zeros(max(len(pts) - 1, 0)))
    return fit_chain(pts, config=tight)


def _fillet(a, b, x: np.ndarray, gap: np.ndarray, tol: float, rmin: float):
    """Arc tangent to lines ``a`` and ``b`` (meeting at ``x``) that fits the ``gap`` samples: ``(T1, T2, bulge, radius)``."""
    u1, u2 = a[3], b[3]
    cross = float(u1[0] * u2[1] - u1[1] * u2[0])
    theta = math.acos(max(-1.0, min(1.0, float(u1 @ u2))))
    if theta < math.radians(3.0) or theta > math.radians(150.0) or len(gap) < 3:
        return None
    sign = 1.0 if cross > 0 else -1.0
    n1 = sign * np.array([-u1[1], u1[0]])
    best = None
    for r in np.geomspace(rmin, 200.0, 90):
        t = r * math.tan(theta / 2.0)
        t1, t2 = x - u1 * t, x + u2 * t
        if t > 60.0:
            continue
        centre = t1 + n1 * r
        dev = float(np.abs(np.hypot(*(gap - centre).T) - r).max())
        if best is None or dev < best[0]:
            best = (dev, t1, t2, r)
    if best is None or best[0] > tol:
        return None
    dev, t1, t2, r = best
    return t1, t2, sign * math.tan(theta / 4.0), r


def straighten_samples(q: np.ndarray, cfg: dict, tight: dict, *, dirs: Sequence[float] = (), hyps=()) -> tuple[BulgeLine, list]:
    """Fit lines (RANSAC, snapped to ``dirs``) joined by corners or tangent arcs to the ordered samples ``q``.

    Straight segments are fitted robustly; between two segments the samples become one tangent arc when it
    fits (so an arc is entered tangentially, without a short line piece), otherwise lines/arcs by ``fit_chain``.
    """
    segs = straight_segments(q, cfg, hyps)
    if not segs:
        return fit_chain(q, config=tight), []
    tol = float(cfg["inlier_m"])
    lines = []
    for i0, i1, c, u in segs:
        v = _snap_direction(u, dirs, float(cfg["snap_deg"])) if dirs else None
        if v is not None and float(_line_dev(q[i0:i1 + 1], c, v).max()) <= 1.5 * tol:
            u = v
        lines.append([i0, i1, c, u])
    n = len(lines)
    joins: list[tuple] = []
    for k in range(n - 1):
        a, b = lines[k], lines[k + 1]
        cross = float(a[3][0] * b[3][1] - a[3][1] * b[3][0])
        if abs(cross) < 1e-3:
            joins.append(("free",))
            continue
        rhs = b[2] - a[2]
        x = a[2] + ((rhs[0] * b[3][1] - rhs[1] * b[3][0]) / cross) * a[3]
        gap = b[0] - a[1]
        if gap <= 6:
            ok = float(np.hypot(*(x - q[a[1]]))) <= 0.5 and float(np.hypot(*(x - q[b[0]]))) <= 0.5
            joins.append(("corner", x) if ok else ("free",))
            continue
        fil = _fillet(a, b, x, q[a[1] + 1:b[0]], 0.75 * tol, float(cfg["min_arc_radius_m"]))
        if fil is not None:
            # the tangent points must stay on the segments' extent
            t1, t2 = fil[0], fil[1]
            if float((t1 - q[a[0]]) @ a[3]) > 0.2 and float((q[b[1]] - t2) @ b[3]) > 0.2:
                joins.append(("fillet", t1, t2, fil[2]))
                continue
        joins.append(("free",))

    def project(k: int, index: int) -> np.ndarray:
        c, u = lines[k][2], lines[k][3]
        return c + u * float((q[index] - c) @ u)

    starts, ends = [], []
    for k in range(n):
        st = project(k, lines[k][0])
        if k > 0 and joins[k - 1][0] == "corner":
            st = joins[k - 1][1]
        elif k > 0 and joins[k - 1][0] == "fillet":
            st = joins[k - 1][2]
        en = project(k, lines[k][1])
        if k < n - 1 and joins[k][0] == "corner":
            en = joins[k][1]
        elif k < n - 1 and joins[k][0] == "fillet":
            en = joins[k][1]
        starts.append(st)
        ends.append(en)
    parts: list[BulgeLine] = []
    first = q[0]
    if lines[0][0] >= 3:
        parts.append(_fit_res(np.vstack((first, q[1:lines[0][0]], starts[0])), tight))
    for k in range(n):
        if float(np.hypot(*(ends[k] - starts[k]))) > 1e-9:
            parts.append(BulgeLine(np.vstack((starts[k], ends[k])), np.zeros(1)))
        if k < n - 1:
            j = joins[k]
            if j[0] == "fillet":
                parts.append(BulgeLine(np.vstack((j[1], j[2])), np.array([j[3]])))
            elif j[0] == "free":
                mid = q[lines[k][1] + 1:lines[k + 1][0]]
                parts.append(_fit_res(np.vstack((ends[k], mid, starts[k + 1])), tight) if len(mid) >= 3
                             else BulgeLine(np.vstack((ends[k], starts[k + 1])), np.zeros(1)))
    last = lines[-1][1]
    if len(q) - 1 - last >= 3:
        parts.append(_fit_res(np.vstack((ends[-1], q[last + 1:-1], q[-1])), tight))
    line = join_lines(parts)
    info = [(math.atan2(u[1], u[0]), float(np.hypot(*(q[i1] - q[i0]))), i0, i1) for i0, i1, _c, u in lines]
    return line, info


def remove_spikes(p: np.ndarray, spike_m: float, protrude_m: float, window: int = 24) -> tuple[np.ndarray, int]:
    """Cut excursions that are narrower than ``spike_m`` at their base and stick out more than ``protrude_m``."""
    pts = np.asarray(p, float)
    removed = 0
    changed = True
    while changed:
        changed = False
        n = len(pts)
        for i in range(n - 2):
            for j in range(i + 2, min(n, i + window)):
                chord = pts[j] - pts[i]
                if float(np.hypot(*chord)) >= spike_m:
                    continue
                seg = LineString([pts[i], pts[j]]) if float(np.hypot(*chord)) > 1e-9 else None
                mid = pts[i + 1:j]
                dist = shapely.distance(shapely.points(mid), seg) if seg is not None else np.hypot(*(mid - pts[i]).T)
                if float(dist.max()) > protrude_m:
                    pts = np.vstack((pts[:i + 1], pts[j:]))
                    removed += 1
                    changed = True
                    break
            if changed:
                break
    return pts, removed


def _image_lines(p: np.ndarray, ortho: OrthoRGB, cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Strong straight image edges (Hough on the Lab gradient) within ``search_m`` of the chain and roughly parallel to it."""
    try:
        from skimage.feature import canny
        from skimage.transform import probabilistic_hough_line
        import cv2
    except Exception:  # noqa: BLE001 - optional
        return []
    cell = ortho.cell
    f = max(1, int(round(0.05 / cell)))
    margin = float(cfg["search_m"]) + 0.2
    x0, y0 = p.min(axis=0) - margin
    x1, y1 = p.max(axis=0) + margin
    r0, r1 = int((ortho.ymax - y1) / cell), int(np.ceil((ortho.ymax - y0) / cell))
    c0, c1 = int((x0 - ortho.xmin) / cell), int(np.ceil((x1 - ortho.xmin) / cell))
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, ortho.shape[0]), min(c1, ortho.shape[1])
    if r1 - r0 < 10 or c1 - c0 < 10:
        return []
    lum = ortho.lab(float(cfg["sigma_m"]))[r0:r1:f, c0:c1:f, 0]
    cs = cell * f
    mask = np.zeros(lum.shape, np.uint8)
    pix = np.column_stack(((p[:, 0] - (ortho.xmin + c0 * cell)) / cs, (ortho.ymax - r0 * cell - p[:, 1]) / cs)).astype(np.int32)
    cv2.polylines(mask, [pix.reshape(-1, 1, 2)], False, 1, 1)
    mask = cv2.dilate(mask, np.ones((2 * int(float(cfg["search_m"]) / cs) + 1,) * 2, np.uint8)) > 0
    edges = canny(lum, sigma=1.0, low_threshold=float(cfg["min_contrast"]) * 0.4, high_threshold=float(cfg["min_contrast"])) & mask
    if edges.sum() < 20:
        return []
    segments = probabilistic_hough_line(edges, threshold=10, line_length=max(5, int(float(cfg["hough_min_len_m"]) / cs)),
                                        line_gap=3, rng=0)
    tang = _tangents(p)
    out = []
    for (ca, ra), (cb, rb) in segments:
        a = np.array([ortho.xmin + c0 * cell + (ca + 0.5) * cs, ortho.ymax - r0 * cell - (ra + 0.5) * cs])
        b = np.array([ortho.xmin + c0 * cell + (cb + 0.5) * cs, ortho.ymax - r0 * cell - (rb + 0.5) * cs])
        u = (b - a) / max(float(np.hypot(*(b - a))), 1e-9)
        mid = (a + b) / 2
        k = int(np.argmin(np.hypot(*(p - mid).T)))
        if float(np.hypot(*(p[k] - mid))) > float(cfg["search_m"]):
            continue
        if math.degrees(math.acos(min(1.0, abs(float(u @ tang[k]))))) <= float(cfg["hough_parallel_deg"]):
            out.append((a, u))
    return out[:60]


def close_slivers(fitted: dict[str, list[list[Polygon]]], tolerance: float = 0.05) -> int:
    """Give slivers narrower than ``tolerance`` between faces to the neighbour with the longest shared edge (in place)."""
    from shapely.ops import unary_union

    from samcloud.environment.topology import _gap_pieces

    entries = [(name, i, k) for name, groups in fitted.items() for i, faces in enumerate(groups) for k in range(len(faces))]
    if len(entries) < 2:
        return 0
    union = unary_union([fitted[n][i][k] for n, i, k in entries])
    closed = 0
    for piece in _gap_pieces(union, tolerance):
        if piece.area < 2e-5:  # < 0.2 cm2: harmless, and leaving it keeps the shared edges bit-identical
            continue
        ring = piece.buffer(1e-6)
        best, best_len = None, 1e-4
        for n, i, k in entries:
            shared = ring.intersection(fitted[n][i][k].boundary).length
            if shared > best_len:
                best, best_len = (n, i, k), shared
        if best is None:
            continue
        n, i, k = best
        merged = unary_union([fitted[n][i][k], piece.buffer(2e-6)])
        parts = _explode(make_valid(merged))
        if parts:
            fitted[n][i][k] = max(parts, key=lambda g: g.area)
            closed += 1
    return closed


def _identity(regions: Mapping[str, Sequence[Polygon]], report: dict) -> ImageEdgeResult:
    return ImageEdgeResult({n: [[p] if p is not None and not p.is_empty else [] for p in polys] for n, polys in regions.items()}, report)


def refine_partition(
    regions: Mapping[str, Sequence[Polygon]],
    ortho: OrthoRGB | None,
    curbs: Sequence = (),
    *,
    config: Mapping | None = None,
) -> ImageEdgeResult:
    """Refine the boundaries between the classes of a planar partition with the ortho image.

    ``regions`` maps a class name (road, sidewalk, path, lawn) to its polygons; space outside every
    polygon is terrain.  The result has the structure of ``EdgeFit.regions`` (faces per input
    polygon) and can be passed on to ``fit_surfaces``.  Without ortho (or colours) or with
    ``edges.image.enabled: false`` only the geometric clean-up (``edges.clean``) acts; with that off as well
    the input is returned unchanged.
    """
    cfg = image_settings(config)
    ccfg = clean_settings(config)
    use_image = bool(cfg["enabled"]) and ortho is not None and bool(ortho.valid.any())
    if not use_image and not ccfg["enabled"]:
        return _identity(regions, {"fallback": "no ortho" if ortho is None else "disabled"})
    fit_cfg = fit_settings(config)
    tight = {"edges": {"fit": {**fit_cfg, "tolerance_m": min(float(cfg["fit_tolerance_m"]), float(fit_cfg["tolerance_m"]))}}}
    sample = float(fit_cfg["sample_m"])
    curb_match = float(fit_cfg["curb_match_m"])
    originals: list[tuple[str, int, Polygon]] = []
    for name, polys in regions.items():
        for i, poly in enumerate(polys):
            if poly is None or poly.is_empty:
                continue
            for part in _explode(make_valid(shapely.set_precision(poly, _GRID))):
                originals.append((name, i, part))
    if not originals:
        return _identity(regions, {"fallback": "no outlines"})
    boundary = shapely.set_precision(shapely.union_all([p.boundary for _n, _i, p in originals]), _GRID)
    merged = linemerge(boundary)
    lines = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    tree = shapely.STRtree([p for _n, _i, p in originals])
    names = np.asarray([n for n, _i, _p in originals] + ["terrain"], dtype=object)

    def classes_at(points: np.ndarray) -> np.ndarray:
        out = np.full(len(points), len(originals), dtype=int)
        hits = tree.query(shapely.points(points), predicate="within")
        out[hits[0]] = hits[1]
        return names[out]

    curb_lines = []
    for curb in curbs:
        xyz = _curb_xyz(curb)
        if xyz.ndim == 2 and len(xyz) >= 2:
            curb_lines.append(LineString(xyz[:, :2]))
    curb_geom = shapely.union_all(curb_lines) if curb_lines else None

    chains: list[_Chain] = []
    degree: Counter = Counter()
    for line in lines:
        xy = np.asarray(line.coords, dtype=float)
        closed = len(xy) > 3 and bool(np.allclose(xy[0], xy[-1]))
        chain = _Chain(xy, closed, _key(xy[0]), _key(xy[-1]))
        degree[chain.start] += 1
        degree[chain.end] += 1
        chains.append(chain)

    stats = Counter()
    pair_stats: dict[str, Counter] = {}
    shifts: list[float] = []
    clean_shifts: list[float] = []
    for chain in chains:
        length = float(LineString(chain.xy).length)
        if length < float(cfg["min_chain_m"]) or len(chain.xy) < 2:
            chain.note = "short"
            continue
        p = _resample(chain.xy, sample)
        if len(p) < 4:
            chain.note = "short"
            continue
        tang = _tangents(p)
        nrm = np.column_stack((-tang[:, 1], tang[:, 0]))
        off = float(cfg["side_offset_m"])
        left, right = classes_at(p + off * nrm), classes_at(p - off * nrm)
        lc, rc = Counter(left).most_common(1)[0][0], Counter(right).most_common(1)[0][0]
        if lc == rc:
            chain.note = "same surface"
            continue
        chain.pair = tuple(sorted((str(lc), str(rc))))
        if curb_geom is not None and float((shapely.distance(shapely.points(p), curb_geom) <= curb_match).mean()) >= 0.5:
            chain.fixed = True
            chain.note = "curb"
            continue
        if chain.closed and degree[chain.start] > 2:
            chain.note = "ring at junction"
            continue
        chain.p, chain.nrm, chain.note = p, nrm, "todo"

    tight_arc = {"edges": {"fit": {**tight["edges"]["fit"], "min_radius_m": max(float(fit_cfg["min_radius_m"]), float(cfg["min_arc_radius_m"]))}}}

    def refine_chain(chain: _Chain, dirs: Sequence[float]) -> None:
        p, nrm = chain.p, chain.nrm
        chain.fit, chain.segs = None, []
        supported = np.zeros(len(p), bool)
        q, fit, dense = p, None, None
        cur, cur_n = p, nrm
        if cfg["straight"] and cfg["hough"] and not chain.closed and not chain.hyps:
            chain.hyps = _image_lines(p, ortho, cfg)
        for number in range(max(1, int(cfg["passes"]))):
            sub = cfg if number == 0 else {**cfg, "search_m": float(cfg["pass2_search_m"])}
            strength, covered, offsets = _step_strength(cur, cur_n, ortho, sub)
            path = _best_path(strength, covered, offsets, sub)
            idx = np.arange(len(cur))
            o = offsets[path]
            supported = covered[idx, path] & (strength[idx, path] >= float(cfg["min_contrast"]))
            if number == 0 and supported.mean() < float(cfg["min_support"]):
                break
            q = cur + o[:, None] * cur_n
            # the first/last samples are biased by the image border at the node: fit without them
            chain.trimmed = (not chain.closed) and len(q) >= 20
            qf = q[_TRIM:-_TRIM] if chain.trimmed else q
            try:
                if cfg["straight"] and not chain.closed and len(qf) >= 8:
                    fit, chain.segs = straighten_samples(qf, cfg, tight_arc, dirs=dirs, hyps=chain.hyps)
                else:
                    fit = fit_chain(qf, closed=chain.closed, free_start=chain.closed, config=tight)
            except Exception as exc:  # noqa: BLE001 - keep the class-raster edge
                LOGGER.warning("image edge fit failed: %s", exc)
                fit = None
                break
            dense = densify(fit, float(fit_cfg["sagitta_m"]))[0]
            if number + 1 < int(cfg["passes"]):
                cur = _resample(dense, sample)
                if len(cur) < 4:
                    break
                tg = _tangents(cur)
                cur_n = np.column_stack((-tg[:, 1], tg[:, 0]))
        if fit is None:
            chain.note = "low contrast" if supported.mean() < float(cfg["min_support"]) else "fit failed"
            return
        keep = slice(_TRIM, -_TRIM) if chain.trimmed else slice(None)
        dist = shapely.distance(shapely.points(q[keep]), LineString(dense))
        support = float((supported[keep] & (dist <= float(cfg["agree_m"]))).mean())
        moved = shapely.distance(shapely.points(_resample(dense, sample)), LineString(chain.xy))
        shift = float(moved.mean())
        chain.support, chain.shift = support, shift
        if support < float(cfg["min_support"]) or shift > float(cfg["search_m"]) or float(moved.max()) > 1.2 * float(cfg["search_m"]):
            chain.note = "not coherent"
            return
        chain.fit, chain.note = fit, "refined"

    tight_clean = {"edges": {"fit": {**fit_cfg, "min_radius_m": max(float(fit_cfg["min_radius_m"]), float(cfg["min_arc_radius_m"]))}}}
    seg_cfg = {**cfg, "inlier_m": float(ccfg["line_tol_m"]), "min_segment_m": float(ccfg["min_segment_m"]),
               "merge_gap_samples": int(ccfg["merge_gap_samples"]), "hough": False, "expand_factor": 1.5}

    def clean_chain(chain: _Chain, dirs: Sequence[float]) -> None:
        """Spike removal, robust lines, snapping and tangent arcs without image support."""
        if chain.closed or chain.p is None or len(chain.p) < 8:
            return
        pts, spikes = remove_spikes(chain.p, float(ccfg["spike_m"]), float(ccfg["spike_protrude_m"]))
        try:
            fit, segs = straighten_samples(pts, seg_cfg, tight_clean, dirs=dirs)
        except Exception as exc:  # noqa: BLE001 - keep the class-raster edge
            LOGGER.warning("edge clean-up failed: %s", exc)
            return
        dense = densify(fit, float(fit_cfg["sagitta_m"]))[0]
        moved = shapely.distance(shapely.points(_resample(dense, sample)), LineString(chain.xy))
        if float(moved.max()) > float(ccfg["max_shift_m"]):
            chain.note = "clean too far"
            return
        chain.fit, chain.segs, chain.note, chain.trimmed = fit, segs, "cleaned", False
        chain.shift, chain.spikes = float(moved.mean()), spikes

    def work(chain: _Chain, dirs: Sequence[float]) -> None:
        if chain.note == "refined" or (use_image and chain.note == "todo"):
            refine_chain(chain, dirs)
        if chain.fit is None and ccfg["enabled"] and chain.note in ("todo", "low contrast", "not coherent", "fit failed", "clean too far"):
            clean_chain(chain, dirs)

    todo = [ch for ch in chains if ch.note == "todo"]
    for ch in todo:
        work(ch, ())
    dirs: list[float] = []
    if cfg["straight"] and float(cfg["snap_deg"]) > 0:
        pairs = [(a, l) for ch in todo if ch.fit is not None for a, l, _i0, _i1 in ch.segs]
        for curb in curb_lines:
            xy = np.asarray(curb.coords)
            seg = np.diff(xy, axis=0)
            ln = np.hypot(*seg.T)
            pairs += [(math.atan2(v[1], v[0]), float(w)) for v, w in zip(seg, ln) if w >= 0.5]
        dirs = dominant_orientations(pairs)
        if dirs:
            for ch in todo:
                if ch.fit is not None and ch.segs:
                    backup = (ch.fit, ch.segs, ch.note, ch.support, ch.shift, ch.trimmed)
                    before = ch.note
                    ch.fit = None
                    if before == "refined":
                        refine_chain(ch, dirs)
                    else:
                        clean_chain(ch, dirs)
                    if ch.fit is None:
                        ch.fit, ch.segs, ch.note, ch.support, ch.shift, ch.trimmed = backup
    report_dirs = [round(math.degrees(d), 1) for d in dirs]
    for chain in chains:
        stats[chain.note or "kept"] += 1
        if chain.pair is not None:
            row = pair_stats.setdefault("|".join(chain.pair), Counter())
            row[chain.note] += 1
        if chain.fit is not None:
            (shifts if chain.note == "refined" else clean_shifts).append(chain.shift)

    # nodes: move to the intersection of the refined end lines; nodes on curbs stay
    fixed_nodes = {k for ch in chains if ch.fixed and not ch.closed for k in (ch.start, ch.end)}
    node_xy: dict[tuple, np.ndarray] = {}
    node_lines: dict[tuple, list] = {}
    for ch in chains:
        if ch.closed:
            continue
        node_xy.setdefault(ch.start, ch.xy[0])
        node_xy.setdefault(ch.end, ch.xy[-1])
        if ch.fit is not None:
            dense = densify(ch.fit, float(fit_cfg["sagitta_m"]))[0]
            for at_end, k in ((False, ch.start), (True, ch.end)):
                piece = _end_line(dense, at_end)
                if piece is not None:
                    node_lines.setdefault(k, []).append(piece)
    image_nodes = {k for ch in chains if ch.note == "refined" and not ch.closed for k in (ch.start, ch.end)}
    new_node = {k: (xy if k in fixed_nodes else _node_position(
        xy, node_lines.get(k, []), float(cfg["node_shift_m"]) if k in image_nodes else min(float(cfg["node_shift_m"]), float(ccfg["max_shift_m"]))))
        for k, xy in node_xy.items()}

    def build_finals(skip: set[int]) -> list[np.ndarray]:
        finals: list[np.ndarray] = []
        for idx, ch in enumerate(chains):
            use = ch.fit is not None and idx not in skip
            if ch.closed:
                if use:
                    line = drop_short_edges(ch.fit, float(cfg["min_edge_m"]), float(cfg["zigzag_dev_m"]))
                    finals.append(densify(line, float(fit_cfg["sagitta_m"]))[0])
                else:
                    finals.append(ch.xy)
                continue
            a, b = new_node[ch.start], new_node[ch.end]
            if not use:
                xy = ch.xy.copy()
                xy[0], xy[-1] = a, b
                finals.append(xy)
                continue
            pts, bul = np.asarray(ch.fit.pts, float).copy(), np.asarray(ch.fit.bulges, float).copy()
            if ch.trimmed:
                pts, bul = np.vstack((a, pts, b)), np.r_[0.0, bul, 0.0]
            else:
                moved_a, moved_b = float(np.hypot(*(pts[0] - a))), float(np.hypot(*(pts[-1] - b)))
                if (bul[0] != 0.0 and moved_a > 0.02) or (bul[-1] != 0.0 and moved_b > 0.02):
                    pts, bul = np.vstack((a, pts, b)), np.r_[0.0, bul, 0.0]
                else:
                    pts[0], pts[-1] = a, b
            line = drop_short_edges(BulgeLine(pts, bul), float(cfg["min_edge_m"]), float(cfg["zigzag_dev_m"]))
            finals.append(densify(line, float(fit_cfg["sagitta_m"]))[0])
        return finals

    area_old = {n: 0.0 for n in regions}
    key_old: dict[tuple[str, int], float] = {}
    key_geom: dict[tuple[str, int], list[Polygon]] = {}
    for name, i, poly in originals:
        area_old[name] += poly.area
        key_old[(name, i)] = key_old.get((name, i), 0.0) + poly.area
        key_geom.setdefault((name, i), []).append(poly)

    def assemble(skip: set[int]):
        finals = build_finals(skip)
        noded = shapely.union_all([LineString(f) for f in finals if len(f) >= 2])
        faces = [f for f in shapely.polygonize([noded]).geoms if f.area > 1e-6]
        fitted: dict[str, list[list[Polygon]]] = {n: [[] for _ in polys] for n, polys in regions.items()}
        area_new = {n: 0.0 for n in regions}
        key_new: dict[tuple[str, int], float] = {}
        for face in faces:
            hits = tree.query(face.representative_point(), predicate="within")
            if len(hits) == 0:
                continue
            name, i, _p = originals[int(hits[0])]
            fitted[name][i].append(face)
            area_new[name] += face.area
            key_new[(name, i)] = key_new.get((name, i), 0.0) + face.area
        worst_class = max((abs(area_new[n] - area_old[n]) / area_old[n] for n in area_old if area_old[n] > 0), default=0.0)
        worst_key = {k: abs(key_new.get(k, 0.0) - v) / v for k, v in key_old.items() if v > 0}
        return fitted, worst_class, worst_key

    skip: set[int] = set()
    for _attempt in range(3):
        fitted, worst, worst_key = assemble(skip)
        bad = [k for k, v in worst_key.items() if v > float(ccfg["max_area_change"])]
        cleaned = [idx for idx, ch in enumerate(chains) if ch.note == "cleaned" and idx not in skip]
        if not bad or not cleaned:
            break
        borders = [shapely.union_all([g.boundary for g in key_geom[k]]) for k in bad]
        revert = {idx for idx in cleaned
                  if any(shapely.distance(b, shapely.Point(chains[idx].xy[len(chains[idx].xy) // 2])) < 2e-3 for b in borders)}
        if not revert:
            break
        skip |= revert
    cleaned_ok = [ch for idx, ch in enumerate(chains) if ch.note == "cleaned" and idx not in skip]
    report = {"chains": len(chains), "stats": dict(stats), "pairs": {k: dict(v) for k, v in pair_stats.items()},
              "refined": int(len(shifts)), "mean_shift_m": round(float(np.mean(shifts)), 4) if shifts else 0.0,
              "max_shift_m": round(float(np.max(shifts)), 4) if shifts else 0.0,
              "cleaned": len(cleaned_ok), "reverted_clean": len(skip), "spikes_removed": int(sum(ch.spikes for ch in cleaned_ok)),
              "clean_mean_shift_m": round(float(np.mean(clean_shifts)), 4) if clean_shifts else 0.0,
              "clean_max_shift_m": round(float(np.max(clean_shifts)), 4) if clean_shifts else 0.0,
              "area_change": round(float(worst), 4), "dominant_deg": report_dirs,
              "segments": int(sum(len(ch.segs) for ch in chains if ch.fit is not None))}
    if worst > float(cfg["max_area_change"]):
        LOGGER.warning("image edge refinement changed a class area by %.1f %%; keeping the class-raster outlines", worst * 100)
        report["fallback"] = f"area change {worst * 100:.1f} %"
        return _identity(regions, report)
    if not shifts and not cleaned_ok:
        report["fallback"] = "no chain refined"
        return _identity(regions, report)
    report["slivers_closed"] = close_slivers(fitted)
    return ImageEdgeResult(fitted, report)
