"""Class-specific cleaning of a classified drone point cloud.

Nothing is invented: points are either removed (outliers, reported per class and
reason) or moved a small, bounded distance onto a robust local plane (flagged in
``PointSet.adjusted``). Vegetation is never moved.

Implementation notes
--------------------
* All neighbourhood work uses ``scipy.spatial.cKDTree`` on 10 m XY tiles with a
  halo (default 1 m). Every per-point decision depends only on points inside the
  halo, so tiled and untiled runs give the same result. Statistical outlier
  statistics are gathered per point in the tiles and thresholded globally per
  class group; neighbour distances are capped at the halo, which keeps the
  result tile-independent and makes isolated points clearly stand out.
* Local planes are fitted robustly (PCA + three trimmed re-fits at 2.5 robust
  sigma), so a point on one side of a ridge, a curb step or another height
  discontinuity is never pulled towards the plane of the other side.
"""
from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator

import numpy as np
from scipy.spatial import cKDTree

from samcloud.drone.contracts import (
    BUILDING,
    CURB,
    GROUND,
    LOW_VEGETATION,
    MEDIUM_VEGETATION,
    PATH,
    PointSet,
    ROAD,
    ROAD_MARKING,
    ROOF_FLAT,
    ROOF_INCLINED,
    SEALED_TERRAIN,
    SHRUB,
    SIDEWALK,
    TREE,
    TREE_TRUNK,
    load_drone_config,
)

logger = logging.getLogger(__name__)

VEGETATION_CODES: tuple[int, ...] = (LOW_VEGETATION, MEDIUM_VEGETATION, TREE, SHRUB, TREE_TRUNK)
PLANAR_CODES: tuple[int, ...] = (BUILDING, ROOF_INCLINED, ROOF_FLAT)
GROUND_CODES: tuple[int, ...] = (GROUND, ROAD, SEALED_TERRAIN, SIDEWALK, PATH, ROAD_MARKING)

# Outlier statistics are computed per group so sparse classes are not wiped by dense ones.
_GROUPS: dict[int, str] = {
    **{c: "building" for c in PLANAR_CODES},
    **{c: "ground" for c in GROUND_CODES},
    **{c: "vegetation" for c in VEGETATION_CODES},
}

DEFAULT_CLOUD_FILTER: dict = {
    "tile_size_m": 10.0,
    "halo_m": 1.0,
    "outliers": {"nb_neighbors": 20, "std_ratio": 2.0, "radius": None, "min_points": None},
    "planar": {
        "class_codes": list(PLANAR_CODES), "sigma_factor": 1.0, "curvature_keep": 0.05,
        "radius": 0.3, "min_inlier_fraction": 0.95, "max_rms_m": 0.015,
    },
    "ground": {
        "class_codes": list(GROUND_CODES), "radius": 0.5, "normal_jump_deg": 12.0,
        "curb_clearance_m": 0.15, "curb_code": CURB, "sigma_factor": 1.0,
        "min_inlier_fraction": 0.95, "max_rms_m": 0.015,
    },
}

Report = dict[int, dict]


# --------------------------------------------------------------------------- tiling
def _iter_tiles(xy: np.ndarray, tile: float, halo: float) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (core, context) index arrays: core = points in a tile, context = core + halo points."""
    if tile <= 0:
        raise ValueError("tile_size must be positive")
    if len(xy) == 0:
        return
    origin = xy.min(axis=0)
    ij = np.floor((xy - origin) / tile).astype(np.int64)
    big = np.int64(1 << 31)
    keys = ij[:, 0] * big + ij[:, 1]
    order = np.argsort(keys, kind="stable")
    uniq, start = np.unique(keys[order], return_index=True)
    stop = np.append(start[1:], len(order))
    buckets = {int(k): order[a:b] for k, a, b in zip(uniq, start, stop)}
    ring = int(math.ceil(halo / tile))
    for k, core in buckets.items():
        ti, tj = divmod(k, int(big))
        cand = [buckets[(ti + di) * int(big) + (tj + dj)]
                for di in range(-ring, ring + 1) for dj in range(-ring, ring + 1)
                if (ti + di) * int(big) + (tj + dj) in buckets]
        cand = np.concatenate(cand)
        lo = origin + np.array([ti, tj]) * tile - halo
        hi = origin + np.array([ti + 1, tj + 1]) * tile + halo
        pts = xy[cand]
        sel = cand[np.all((pts >= lo) & (pts <= hi), axis=1)]
        yield np.sort(core), np.sort(sel)


def _run_tiles(xy: np.ndarray, tile: float, halo: float,
               fn: Callable[[np.ndarray, np.ndarray], None], workers: int | None) -> None:
    """Run ``fn(core, ctx)`` for each tile; ``fn`` writes to disjoint (core) output slices."""
    tasks = list(_iter_tiles(xy, tile, halo))
    n_workers = max(1, workers if workers else (os.cpu_count() or 1))
    if n_workers == 1 or len(tasks) <= 1:
        for core, ctx in tasks:
            fn(core, ctx)
        return
    with ThreadPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
        list(pool.map(lambda t: fn(*t), tasks))


# --------------------------------------------------------------------------- plane fit
def _fit_planes(q: np.ndarray, src: np.ndarray, tree: cKDTree, radius: float, k_max: int,
                min_neighbors: int, trim: float = 2.5, iters: int = 3, block: int = 20000) -> dict:
    """Robust local plane at each query point from the source points within ``radius``.

    Returns arrays: ``ok`` (enough neighbours), ``normal`` (n,3, nz >= 0), ``dist`` (signed
    distance of the query to its plane), ``rms`` (inlier RMS), ``curv`` (smallest eigenvalue
    over the sum), ``frac`` (inlier fraction), ``idx`` (n,K neighbour index, ``len(src)`` = none).
    """
    n = len(q)
    n_src = len(src)
    out = {
        "ok": np.zeros(n, dtype=bool), "normal": np.tile([0.0, 0.0, 1.0], (n, 1)),
        "dist": np.zeros(n), "rms": np.zeros(n), "curv": np.ones(n), "frac": np.zeros(n),
        "idx": np.full((n, k_max), n_src, dtype=np.int64),
    }
    if n == 0 or n_src == 0:
        return out
    k = min(k_max, n_src)
    for a in range(0, n, block):
        b = min(a + block, n)
        qb = q[a:b]
        _, idx = tree.query(qb, k=k, distance_upper_bound=radius, workers=1)
        idx = np.asarray(idx).reshape(b - a, k)
        valid = idx < n_src
        pts = src[np.where(valid, idx, 0)] - qb[:, None, :]
        w = valid.astype(np.float64)
        nvalid = w.sum(axis=1)
        for it in range(iters):
            cnt = np.maximum(w.sum(axis=1), 1.0)
            c = (w[..., None] * pts).sum(axis=1) / cnt[:, None]
            x = pts - c[:, None, :]
            cov = np.einsum("nk,nki,nkj->nij", w, x, x) / cnt[:, None, None]
            evals, evecs = np.linalg.eigh(cov)
            normal = evecs[:, :, 0]
            d = np.einsum("nki,ni->nk", x, normal)
            if it < iters - 1:
                ad = np.where(valid, np.abs(d), np.inf)
                ad.sort(axis=1)
                mid = np.clip((nvalid.astype(int) - 1) // 2, 0, k - 1)
                med = np.take_along_axis(ad, mid[:, None], axis=1)[:, 0]
                med = np.where(np.isfinite(med), med, 0.0)
                thr = trim * np.maximum(1.4826 * med, 1e-4)
                w = (valid & (np.abs(d) <= thr[:, None])).astype(np.float64)
        flip = normal[:, 2] < 0
        normal = np.where(flip[:, None], -normal, normal)
        cnt = np.maximum(w.sum(axis=1), 1.0)
        out["ok"][a:b] = (nvalid >= min_neighbors) & (w.sum(axis=1) >= min_neighbors)
        out["normal"][a:b] = normal
        out["dist"][a:b] = -np.einsum("ni,ni->n", c, normal)
        out["rms"][a:b] = np.sqrt((w * d * d).sum(axis=1) / cnt)
        out["curv"][a:b] = evals[:, 0] / np.maximum(evals.sum(axis=1), 1e-18)
        out["frac"][a:b] = w.sum(axis=1) / np.maximum(nvalid, 1.0)
        full = np.full((b - a, k_max), n_src, dtype=np.int64)
        full[:, :k] = np.where(valid, idx, n_src)
        out["idx"][a:b] = full
    return out


# --------------------------------------------------------------------------- reporting
def _build_report(classes: np.ndarray, removed: np.ndarray, reason: np.ndarray | None,
                  shift: np.ndarray, moved: np.ndarray) -> Report:
    """Per-class report in the index space of the *input* arrays."""
    report: Report = {}
    for code in np.unique(classes):
        m = classes == code
        mv = m & moved & ~removed
        shifts = shift[mv]
        by_reason = {"statistical": 0, "radius": 0}
        if reason is not None:
            by_reason["statistical"] = int((m & removed & (reason == 1)).sum())
            by_reason["radius"] = int((m & removed & (reason == 2)).sum())
        report[int(code)] = {
            "input": int(m.sum()),
            "removed": int((m & removed).sum()),
            "moved": int(mv.sum()),
            "max_shift_m": float(shifts.max()) if len(shifts) else 0.0,
            "median_shift_m": float(np.median(shifts)) if len(shifts) else 0.0,
            "removed_by_reason": by_reason,
        }
    return report


def _with_adjusted(ps: PointSet, xyz: np.ndarray, moved: np.ndarray) -> PointSet:
    adjusted = np.zeros(len(ps), dtype=bool) if ps.adjusted is None else ps.adjusted.copy()
    adjusted |= moved
    out = ps.subset(np.arange(len(ps)))
    out.xyz = xyz
    out.adjusted = adjusted
    return out


def _check_codes(class_codes: Iterable[int], forbidden: Iterable[int], what: str) -> np.ndarray:
    codes = [int(c) for c in class_codes]
    bad = sorted(set(codes) & set(int(f) for f in forbidden))
    if bad:
        logger.warning("%s: classes %s are never moved, ignored", what, bad)
    return np.array([c for c in codes if c not in bad], dtype=np.int64)


# --------------------------------------------------------------------------- outliers
def _outlier_masks(ps: PointSet, nb_neighbors: int, std_ratio: float, radius: float | None,
                   min_points: int | None, tile: float, halo: float,
                   workers: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (remove mask, reason array 0/1 statistical/2 radius)."""
    n = len(ps)
    remove = np.zeros(n, dtype=bool)
    reason = np.zeros(n, dtype=np.int8)
    use_radius = radius is not None and min_points is not None and min_points > 0
    cap = max(halo, float(radius) if use_radius else 0.0)
    keys = np.array([_GROUPS.get(int(c), f"class_{int(c)}") for c in np.unique(ps.classes)])
    group_of = {int(c): k for c, k in zip(np.unique(ps.classes), keys)}
    point_group = np.array([group_of[int(c)] for c in ps.classes]) if n else np.array([])
    for group in np.unique(point_group):
        rows = np.flatnonzero(point_group == group)
        xyz = ps.xyz[rows]
        mean_d = np.zeros(len(rows))
        counts = np.zeros(len(rows), dtype=np.int64)

        def work(core: np.ndarray, ctx: np.ndarray) -> None:
            tree = cKDTree(xyz[ctx])
            q = xyz[core]
            if nb_neighbors > 0:
                ks = list(range(2, min(nb_neighbors + 1, len(ctx)) + 1))
                if ks:
                    dist, _ = tree.query(q, k=ks, distance_upper_bound=cap, workers=1)
                    d = np.where(np.isfinite(dist), dist, cap).reshape(len(core), len(ks))
                else:
                    d = np.zeros((len(core), 0))
                if d.shape[1] < nb_neighbors:
                    d = np.hstack([d, np.full((len(core), nb_neighbors - d.shape[1]), cap)])
                mean_d[core] = d.mean(axis=1)
            if use_radius:
                counts[core] = tree.query_ball_point(q, float(radius), return_length=True, workers=1) - 1

        _run_tiles(xyz[:, :2], tile, cap, work, workers)
        if nb_neighbors > 0 and len(rows) > 1:
            thr = mean_d.mean() + std_ratio * mean_d.std()
            stat = mean_d > thr + 1e-9
            remove[rows[stat]] = True
            reason[rows[stat]] = 1
        if use_radius:
            rad = (counts < int(min_points)) & ~remove[rows]
            remove[rows[rad]] = True
            reason[rows[rad]] = 2
    return remove, reason


def remove_outliers(ps: PointSet, *, nb_neighbors: int = 20, std_ratio: float = 2.0,
                    radius: float | None = None, min_points: int | None = None,
                    tile_size: float = 10.0, halo: float = 1.0,
                    workers: int | None = None) -> tuple[PointSet, Report]:
    """Statistical (and optional radius) outlier removal per class group.

    A point is removed if its mean distance to ``nb_neighbors`` neighbours exceeds the group's
    mean + ``std_ratio`` * std (distances capped at the halo), or if ``radius``/``min_points``
    are given and it has fewer than ``min_points`` neighbours within ``radius``. Groups are
    building/roof, ground/pavement, vegetation and every other class on its own.
    """
    n = len(ps)
    if n == 0:
        return ps.subset(np.arange(0)), {}
    remove, reason = _outlier_masks(ps, nb_neighbors, std_ratio, radius, min_points,
                                    tile_size, halo, workers)
    report = _build_report(ps.classes, remove, reason, np.zeros(n), np.zeros(n, dtype=bool))
    return ps.subset(~remove), report


# --------------------------------------------------------------------------- planar
def _planar_move(xyz: np.ndarray, classes: np.ndarray, codes: np.ndarray, *, sigma_factor: float,
                 curvature_keep: float, radius: float, min_inlier_fraction: float, max_rms_m: float,
                 min_neighbors: int, k_max: int, tile: float, halo: float,
                 workers: int | None) -> tuple[np.ndarray, np.ndarray]:
    new_xyz = xyz.copy()
    moved = np.zeros(len(xyz), dtype=bool)
    rows = np.flatnonzero(np.isin(classes, codes))
    if len(rows) == 0:
        return new_xyz, moved
    pts = xyz[rows]
    halo = max(halo, radius)

    def work(core: np.ndarray, ctx: np.ndarray) -> None:
        src = pts[ctx]
        fit = _fit_planes(pts[core], src, cKDTree(src), radius, k_max, min_neighbors)
        move = (fit["ok"] & (fit["rms"] <= max_rms_m) & (fit["frac"] >= min_inlier_fraction) & (fit["curv"] < curvature_keep)
                & (np.abs(fit["dist"]) <= sigma_factor * fit["rms"]))
        target = pts[core] - fit["dist"][:, None] * fit["normal"]
        sel = rows[core[move]]
        new_xyz[sel] = target[move]
        moved[sel] = True

    _run_tiles(pts[:, :2], tile, halo, work, workers)
    return new_xyz, moved


def project_planar(ps: PointSet, class_codes: Iterable[int], *, sigma_factor: float = 1.0,
                   curvature_keep: float = 0.05, radius: float = 0.3,
                   min_inlier_fraction: float = 0.95, max_rms_m: float = 0.015, min_neighbors: int = 8, k_max: int = 64,
                   tile_size: float = 10.0, halo: float = 1.0,
                   workers: int | None = None) -> tuple[PointSet, Report]:
    """Project building/roof points onto their robust local plane.

    A point is moved only if |distance| <= ``sigma_factor`` * local plane RMS, the local
    curvature is below ``curvature_keep`` and at least ``min_inlier_fraction`` of the
    neighbourhood lies on the plane and the plane RMS is <= ``max_rms_m`` (edges, ridges and
    corners therefore stay untouched).
    Moved points get ``adjusted=True``. Vegetation codes are ignored.
    """
    codes = _check_codes(class_codes, VEGETATION_CODES, "project_planar")
    new_xyz, moved = _planar_move(
        ps.xyz, ps.classes, codes, sigma_factor=sigma_factor, curvature_keep=curvature_keep,
        radius=radius, min_inlier_fraction=min_inlier_fraction, max_rms_m=max_rms_m, min_neighbors=min_neighbors,
        k_max=k_max, tile=tile_size, halo=halo, workers=workers)
    shift = np.linalg.norm(new_xyz - ps.xyz, axis=1)
    report = _build_report(ps.classes, np.zeros(len(ps), dtype=bool), None, shift, moved)
    return _with_adjusted(ps, new_xyz, moved), report


# --------------------------------------------------------------------------- ground
def _ground_move(xyz: np.ndarray, classes: np.ndarray, codes: np.ndarray, *, radius: float,
                 normal_jump_deg: float, curb_clearance_m: float, curb_code: int,
                 sigma_factor: float, min_inlier_fraction: float, max_rms_m: float, min_neighbors: int,
                 k_max: int, tile: float, halo: float,
                 workers: int | None) -> tuple[np.ndarray, np.ndarray]:
    new_xyz = xyz.copy()
    moved = np.zeros(len(xyz), dtype=bool)
    codes = codes[codes != curb_code]
    rows = np.flatnonzero(np.isin(classes, np.append(codes, curb_code)))
    if len(rows) == 0:
        return new_xyz, moved
    pts = xyz[rows]
    is_curb = classes[rows] == curb_code
    is_ground = np.isin(classes[rows], codes)
    halo = max(halo, 2.0 * radius + 1e-6, curb_clearance_m)
    cos_jump = math.cos(math.radians(normal_jump_deg))

    def work(core: np.ndarray, ctx: np.ndarray) -> None:
        g_ctx = ctx[is_ground[ctx]]
        g_core = core[is_ground[core]]
        if len(g_core) == 0:
            return
        gpts = pts[g_ctx]
        fit = _fit_planes(gpts, gpts, cKDTree(gpts), radius, k_max, min_neighbors)
        pos = np.searchsorted(g_ctx, g_core)
        # breakline protection 1: normal jump towards any neighbour
        nb = fit["idx"][pos]
        nvalid = nb < len(g_ctx)
        n_nb = fit["normal"][np.where(nvalid, nb, 0)]
        cosang = np.abs(np.einsum("nki,ni->nk", n_nb, fit["normal"][pos]))
        informative = nvalid & fit["ok"][np.where(nvalid, nb, 0)]
        jump = np.any(informative & (cosang < cos_jump), axis=1) if nb.shape[1] else np.zeros(len(pos), bool)
        # breakline protection 2: curb points nearby
        c_ctx = ctx[is_curb[ctx]]
        near_curb = np.zeros(len(g_core), dtype=bool)
        if len(c_ctx):
            dist, _ = cKDTree(pts[c_ctx]).query(pts[g_core], k=1, distance_upper_bound=curb_clearance_m,
                                                workers=1)
            near_curb = np.isfinite(dist)
        f = {k: v[pos] for k, v in fit.items() if k != "idx"}
        move = (f["ok"] & (f["rms"] <= max_rms_m) & (f["frac"] >= min_inlier_fraction) & ~jump & ~near_curb
                & (np.abs(f["dist"]) <= sigma_factor * f["rms"]))
        target = pts[g_core] - f["dist"][:, None] * f["normal"]
        sel = rows[g_core[move]]
        new_xyz[sel] = target[move]
        moved[sel] = True

    _run_tiles(pts[:, :2], tile, halo, work, workers)
    return new_xyz, moved


def smooth_ground(ps: PointSet, class_codes: Iterable[int], *, radius: float = 0.5,
                  normal_jump_deg: float = 12.0, curb_clearance_m: float = 0.15,
                  curb_code: int = CURB, sigma_factor: float = 1.0,
                  min_inlier_fraction: float = 0.95, max_rms_m: float = 0.015, min_neighbors: int = 8, k_max: int = 64,
                  tile_size: float = 10.0, halo: float = 1.0,
                  workers: int | None = None) -> tuple[PointSet, Report]:
    """Smooth road/sidewalk/ground points onto a robust local plane without crossing breaklines.

    Points are left untouched if they are within ``curb_clearance_m`` of a ``curb_code`` point,
    if a neighbour's local normal deviates by more than ``normal_jump_deg`` (normal jump), or if
    the neighbourhood is not one coherent surface (inlier fraction < ``min_inlier_fraction``,
    i.e. a height step). Curb points themselves are never moved; vegetation codes are ignored.
    """
    codes = _check_codes(class_codes, VEGETATION_CODES, "smooth_ground")
    new_xyz, moved = _ground_move(
        ps.xyz, ps.classes, codes, radius=radius, normal_jump_deg=normal_jump_deg,
        curb_clearance_m=curb_clearance_m, curb_code=curb_code, sigma_factor=sigma_factor,
        min_inlier_fraction=min_inlier_fraction, max_rms_m=max_rms_m, min_neighbors=min_neighbors, k_max=k_max,
        tile=tile_size, halo=halo, workers=workers)
    shift = np.linalg.norm(new_xyz - ps.xyz, axis=1)
    report = _build_report(ps.classes, np.zeros(len(ps), dtype=bool), None, shift, moved)
    return _with_adjusted(ps, new_xyz, moved), report


# --------------------------------------------------------------------------- orchestrator
def _merge(base: dict, override: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def filter_drone_cloud(ps: PointSet, config: dict | None = None, *, mode: str = "full",
                       workers: int | None = None) -> tuple[PointSet, Report]:
    """Run outlier removal, planar projection and ground smoothing on a drone cloud.

    ``mode="full"`` runs all three stages; ``mode="outliers_only"`` only removes outliers and
    moves nothing. How to decide: use ``outliers_only`` whenever an upstream step has already
    projected planes onto the same points (to avoid projecting twice). The existing geometry
    refinement v2 (``samcloud/engine/geometry/refine.py``, ``refine_classification``) only
    re-labels points (class ids from rules and neighbour votes) and never changes coordinates,
    also for ``drone_gps``; with it alone, ``mode="full"`` is the right choice. Check again if
    refine.py ever starts writing to ``xyz``.

    ``config`` is the drone config (``load_drone_config()``); optional ``cloud_filter`` section
    overrides ``DEFAULT_CLOUD_FILTER``. The report maps class code to ``input``, ``removed``,
    ``moved``, ``max_shift_m``, ``median_shift_m`` (+ ``removed_by_reason``) relative to the input;
    ``input == points left in the output + removed`` for every class.
    """
    if mode not in ("full", "outliers_only"):
        raise ValueError(f"unknown mode {mode!r}")
    full_cfg = config if config is not None else load_drone_config()
    cfg = _merge(DEFAULT_CLOUD_FILTER, full_cfg.get("cloud_filter", {}) or {})
    tile, halo = float(cfg["tile_size_m"]), float(cfg["halo_m"])
    n = len(ps)
    if n == 0:
        return ps.subset(np.arange(0)), {}
    o = cfg["outliers"]
    remove, reason = _outlier_masks(ps, int(o["nb_neighbors"]), float(o["std_ratio"]), o["radius"],
                                    o["min_points"], tile, halo, workers)
    kept = np.flatnonzero(~remove)
    cur = ps.subset(kept)
    moved_kept = np.zeros(len(kept), dtype=bool)
    if mode == "full" and len(kept):
        p = cfg["planar"]
        codes = _check_codes(p["class_codes"], VEGETATION_CODES, "planar")
        xyz, mv = _planar_move(
            cur.xyz, cur.classes, codes, sigma_factor=float(p["sigma_factor"]),
            curvature_keep=float(p["curvature_keep"]), radius=float(p["radius"]),
            min_inlier_fraction=float(p["min_inlier_fraction"]), max_rms_m=float(p["max_rms_m"]), min_neighbors=8, k_max=64,
            tile=tile, halo=halo, workers=workers)
        cur.xyz, moved_kept = xyz, moved_kept | mv
        g = cfg["ground"]
        codes = _check_codes(g["class_codes"], VEGETATION_CODES, "ground")
        xyz, mv = _ground_move(
            cur.xyz, cur.classes, codes, radius=float(g["radius"]),
            normal_jump_deg=float(g["normal_jump_deg"]), curb_clearance_m=float(g["curb_clearance_m"]),
            curb_code=int(g["curb_code"]), sigma_factor=float(g["sigma_factor"]),
            min_inlier_fraction=float(g["min_inlier_fraction"]), max_rms_m=float(g["max_rms_m"]), min_neighbors=8, k_max=64,
            tile=tile, halo=halo, workers=workers)
        cur.xyz, moved_kept = xyz, moved_kept | mv
    moved = np.zeros(n, dtype=bool)
    moved[kept] = moved_kept
    shift = np.zeros(n)
    shift[kept] = np.linalg.norm(cur.xyz - ps.xyz[kept], axis=1)
    cur.adjusted = (np.zeros(len(kept), dtype=bool) if cur.adjusted is None else cur.adjusted) | moved_kept
    return cur, _build_report(ps.classes, remove, reason, shift, moved)
