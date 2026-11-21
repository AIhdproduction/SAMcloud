"""Clean terrain model: remove residual non-ground bumps, smooth pavements, report the deviation.

* :func:`clean_dtm` removes bumps (vegetation, objects) that survived the ground filter: cells that
  stand more than ``terrain.clean.max_bump_m`` above a robust local surface (morphological opening)
  over an area below ``terrain.clean.max_bump_area_m2`` are re-interpolated from the surrounding
  ground.  Long steep features (retaining walls) and cells next to curbs/pavements are never touched.
* :func:`smooth_pavement_surfaces` fits a robust local plane per pavement element.
* :func:`tin_deviation` measures the final TIN against the original ground points per label.

All rasters are north-up (row 0 = ``ymax``), bounds are ``(xmin, ymin, xmax, ymax)`` in metres.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import shapely
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator

from samcloud.drone.contracts import TIN_LABELS

LOGGER = logging.getLogger(__name__)


def _cfg(config: Mapping | None, *path: str) -> dict:
    node = config if config is not None else {}
    for key in path:
        node = node.get(key, {}) if isinstance(node, Mapping) else {}
    return dict(node) if isinstance(node, Mapping) else {}


def _odd(cells: float, minimum: int = 3) -> int:
    n = max(int(round(cells)), minimum)
    return n if n % 2 else n + 1


def _fill_nearest(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid cells by the nearest valid value (filter input only)."""
    if valid.all() or not valid.any():
        return np.where(valid, z, 0.0)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def protect_mask(xyz: np.ndarray, bounds, resolution: float, shape: tuple[int, int], dilate_m: float) -> np.ndarray:
    """Raster mask of cells that contain ``xyz`` points, dilated by ``dilate_m``."""
    mask = np.zeros(shape, dtype=bool)
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(pts):
        xmin, _ymin, _xmax, ymax = bounds
        col = np.floor((pts[:, 0] - xmin) / resolution).astype(np.int64)
        row = np.floor((ymax - pts[:, 1]) / resolution).astype(np.int64)
        ok = (col >= 0) & (col < shape[1]) & (row >= 0) & (row < shape[0])
        mask[row[ok], col[ok]] = True
    cells = int(round(dilate_m / resolution))
    return ndimage.binary_dilation(mask, iterations=cells) if cells > 0 and mask.any() else mask


def _interpolate_removed(z: np.ndarray, removed: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``removed`` cells from ``sources`` (nearest fallback)."""
    out = z.copy()
    ring = ndimage.binary_dilation(removed, iterations=10) & sources & ~removed
    if ring.sum() < 3:
        ring = sources
    rows, cols = np.nonzero(ring)
    tr, tc = np.nonzero(removed)
    if len(rows) < 3 or len(tr) == 0:
        return out
    pts = np.column_stack((cols, rows)).astype(float)
    target = np.column_stack((tc, tr)).astype(float)
    try:
        est = LinearNDInterpolator(pts, z[rows, cols])(target)
    except Exception:  # noqa: BLE001 - degenerate (collinear) source cells
        est = np.full(len(tr), np.nan)
    bad = ~np.isfinite(est)
    if bad.any():
        nearest = ndimage.distance_transform_edt(~ring, return_distances=False, return_indices=True)
        est[bad] = z[nearest[0][tr[bad], tc[bad]], nearest[1][tr[bad], tc[bad]]]
    out[tr, tc] = est
    return out


def clean_dtm(
    dtm: np.ndarray,
    validity: np.ndarray,
    resolution: float,
    *,
    protect: np.ndarray | None = None,
    config: Mapping | None = None,
) -> tuple[np.ndarray, dict]:
    """Remove residual non-ground bumps and apply a light noise filter.

    Returns ``(cleaned dtm, report)``.  ``protect`` marks cells that must never change (pavement
    and curb cells).  The report lists the removed bump count, cell count, area and maximum height.
    """
    cfg = _cfg(config, "terrain", "clean")
    res = float(resolution)
    z = np.asarray(dtm, dtype=np.float64)
    valid = np.isfinite(z)
    report: dict = {"enabled": bool(cfg.get("enabled", True)), "bumps_removed": 0, "cells_removed": 0,
                    "area_removed_m2": 0.0, "max_bump_removed_m": 0.0, "walls_protected": 0,
                    "median_cells_changed": 0}
    if not report["enabled"] or not valid.any():
        return np.asarray(dtm).copy(), report
    protect = np.zeros(z.shape, dtype=bool) if protect is None else np.asarray(protect, dtype=bool)
    max_bump = float(cfg.get("max_bump_m", 0.15))
    max_area = float(cfg.get("max_bump_area_m2", 25.0))
    filled = _fill_nearest(z, valid)

    # robust local surface: opening of a median-filtered DTM keeps planes, slopes and steps
    window = _odd(float(cfg.get("window_m", 5.0)) / res)
    ref = ndimage.grey_opening(ndimage.median_filter(filled, size=3), size=(window, window))
    bump = filled - ref

    # long steep linear features are retaining walls
    gy, gx = np.gradient(filled, res)
    steep = (np.hypot(gx, gy) > float(cfg.get("steep_slope", 1.0))) & valid
    labels, count = ndimage.label(steep, structure=np.ones((3, 3), dtype=int))
    wall = np.zeros(z.shape, dtype=bool)
    min_len = float(cfg.get("wall_min_length_m", 2.0))
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        extent = float(np.hypot((box[0].stop - box[0].start) * res, (box[1].stop - box[1].start) * res))
        if extent < min_len:
            continue
        rows, cols = np.nonzero(labels[box] == index)
        if len(rows) < 6:
            continue
        eig = np.linalg.eigvalsh(np.cov(np.vstack((rows, cols)).astype(float)))
        # a wall is a linear feature; the steep rim of a round mound is not
        if np.sqrt(max(eig[1], 1e-12) / max(eig[0], 1e-12)) >= float(cfg.get("wall_elongation", 2.5)):
            wall |= labels == index
    report["walls_protected"] = int(ndimage.label(wall, structure=np.ones((3, 3), dtype=int))[1]) if wall.any() else 0
    wall_zone = ndimage.binary_dilation(wall, iterations=2) if wall.any() else wall

    seeds = valid & (bump > max_bump)
    if seeds.any():
        margin = max(1, int(round(float(cfg.get("grow_margin_m", 1.0)) / res)))
        grown = valid & (bump > float(cfg.get("grow_fraction", 0.25)) * max_bump)
        grown &= ndimage.binary_dilation(seeds, iterations=margin)
        glabels, gcount = ndimage.label(grown | seeds, structure=np.ones((3, 3), dtype=int))
        sizes = np.bincount(glabels.ravel(), minlength=gcount + 1) * res * res
        seeded = np.zeros(gcount + 1, dtype=bool)
        seeded[np.unique(glabels[seeds])] = True
        seeded[0] = False
        touches_wall = np.zeros(gcount + 1, dtype=bool)
        if wall_zone.any():
            touches_wall[np.unique(glabels[wall_zone])] = True
        peak = np.asarray(ndimage.maximum(bump, glabels, index=np.arange(gcount + 1)), dtype=float)
        # tall objects (shrub groups, hedges, piles) may be larger than a low bump and still be clutter
        limit = max_area * np.where(peak >= float(cfg.get("tall_bump_m", 1.0)), float(cfg.get("tall_area_factor", 4.0)), 1.0)
        selected = seeded & (sizes < limit)
        for index in np.flatnonzero(selected & touches_wall):
            # next to a wall only a compact blob is a mound; a long strip is a terrace or the wall's own top
            rows, cols = np.nonzero(glabels == index)
            eig = np.linalg.eigvalsh(np.cov(np.vstack((rows, cols)).astype(float))) if len(rows) > 5 else np.array([1.0, 1.0])
            if np.sqrt(max(eig[1], 1e-12) / max(eig[0], 1e-12)) >= float(cfg.get("wall_elongation", 2.5)):
                selected[index] = False
        report["bumps_removed"] = int(selected.sum())
        removal = selected[glabels]
        removal = ndimage.binary_dilation(removal, structure=np.ones((3, 3), dtype=bool)) & valid & ~protect
        if removal.any():
            sources = valid & ~removal
            z = _interpolate_removed(np.where(valid, z, 0.0), removal, sources)
            z[~valid] = np.nan
            report["cells_removed"] = int(removal.sum())
            report["area_removed_m2"] = round(float(removal.sum()) * res * res, 3)
            report["max_bump_removed_m"] = round(float(bump[removal].max()), 3)
    median = int(cfg.get("median_cells", 3))
    if median >= 3:
        measured = (np.asarray(validity) == 1) & np.isfinite(z) & ~protect & ~wall_zone
        smooth = ndimage.median_filter(_fill_nearest(z, np.isfinite(z)), size=median)
        report["median_cells_changed"] = int((measured & (np.abs(smooth - z) > 1e-6)).sum())
        z = np.where(measured, smooth, z)
    return z.astype(np.asarray(dtm).dtype if np.issubdtype(np.asarray(dtm).dtype, np.floating) else np.float64), report


# ---------------------------------------------------------------------------------- pavements


def _robust_plane(z: np.ndarray, weight: np.ndarray, window: int, iterations: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell robust local plane value at the cell centre (and a validity mask)."""
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    zz = np.where(weight > 0, z, 0.0)
    w0 = (weight > 0).astype(np.float64)
    w = w0.copy()
    value = zz.copy()
    ok = np.zeros(z.shape, dtype=bool)

    def box(a):
        return ndimage.uniform_filter(a, size=window, mode="constant")

    for _ in range(max(1, iterations)):
        m1, mx, my = box(w), box(w * xx), box(w * yy)
        mz = box(w * zz)
        mxx, mxy, myy = box(w * xx * xx), box(w * xx * yy), box(w * yy * yy)
        mxz, myz = box(w * xx * zz), box(w * yy * zz)
        sx, sy = mx - xx * m1, my - yy * m1
        sxx = mxx - 2 * xx * mx + xx * xx * m1
        sxy = mxy - xx * my - yy * mx + xx * yy * m1
        syy = myy - 2 * yy * my + yy * yy * m1
        sxz, syz = mxz - xx * mz, myz - yy * mz
        a = np.zeros(z.shape + (3, 3))
        a[..., 0, 0], a[..., 0, 1], a[..., 0, 2] = m1, sx, sy
        a[..., 1, 0], a[..., 1, 1], a[..., 1, 2] = sx, sxx + 1e-3 * m1, sxy
        a[..., 2, 0], a[..., 2, 1], a[..., 2, 2] = sy, sxy, syy + 1e-3 * m1
        rhs = np.stack((mz, sxz, syz), axis=-1)
        enough = box(w0) * window * window >= 6.0
        a[~enough] = np.eye(3)
        sol = np.linalg.solve(a, rhs[..., None])[..., 0]
        value = sol[..., 0]
        ok = enough & (weight > 0)
        r = np.abs(zz - value) / max(scale, 1e-6)
        w = w0 * np.clip(1.0 - r * r, 0.0, 1.0) ** 2
        w = np.where(w0 > 0, np.maximum(w, 0.02), 0.0)
    return value, ok


def smooth_pavement_surfaces(
    dtm: np.ndarray,
    bounds,
    resolution: float,
    polygons: Sequence,
    *,
    config: Mapping | None = None,
) -> tuple[np.ndarray, dict]:
    """Robust local-plane smoothing of every pavement polygon (road/sidewalk/path elements).

    Each polygon is smoothed on its own cells only, so the curb step between neighbouring elements
    is kept.  Returns ``(dtm, report)``.
    """
    cfg = _cfg(config, "terrain", "smooth")
    z = np.asarray(dtm, dtype=np.float64).copy()
    report = {"enabled": bool(cfg.get("enabled", True)), "elements": 0, "cells": 0, "rms_change_m": 0.0}
    if not report["enabled"]:
        return z, report
    res = float(resolution)
    xmin, _ymin, _xmax, ymax = bounds
    window = _odd(float(cfg.get("window_m", 2.0)) / res)
    iterations = int(cfg.get("iterations", 3))
    scale = float(cfg.get("robust_scale_m", 0.08))
    max_change = float(cfg.get("max_change_m", 0.06))
    changes: list[np.ndarray] = []
    for item in polygons:
        poly = getattr(item, "polygon", item)
        if poly is None or poly.is_empty:
            continue
        x0, y0, x1, y1 = poly.bounds
        c0 = max(int(np.floor((x0 - xmin) / res)) - 1, 0)
        c1 = min(int(np.ceil((x1 - xmin) / res)) + 1, z.shape[1])
        r0 = max(int(np.floor((ymax - y1) / res)) - 1, 0)
        r1 = min(int(np.ceil((ymax - y0) / res)) + 1, z.shape[0])
        if c1 <= c0 or r1 <= r0:
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        inside = shapely.contains_xy(poly, xmin + (cc + 0.5) * res, ymax - (rr + 0.5) * res) & np.isfinite(z[r0:r1, c0:c1])
        if inside.sum() < 6:
            continue
        sub = z[r0:r1, c0:c1]
        value, ok = _robust_plane(sub, inside.astype(np.float64), window, iterations, scale)
        use = inside & ok & np.isfinite(value)
        # a plane never moves a cell further than the noise level: larger jumps are real relief
        value = sub + np.clip(value - sub, -max_change, max_change)
        changes.append((value - sub)[use])
        sub[use] = value[use]
        report["elements"] += 1
        report["cells"] += int(use.sum())
    if changes:
        report["rms_change_m"] = round(float(np.sqrt(np.mean(np.concatenate(changes) ** 2))), 4)
    return z, report


# ---------------------------------------------------------------------------------- former lawn areas


def _cell_flat(xyz: np.ndarray, bounds, resolution: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    xmin, _ymin, _xmax, ymax = bounds
    col = np.floor((xyz[:, 0] - xmin) / resolution).astype(np.int64)
    row = np.floor((ymax - xyz[:, 1]) / resolution).astype(np.int64)
    inside = (col >= 0) & (col < shape[1]) & (row >= 0) & (row < shape[0])
    return row * shape[1] + col, inside


def lawn_cell_mask(
    lawn_xyz: np.ndarray, bounds, resolution: float, shape: tuple[int, int], *,
    protect: np.ndarray | None = None, min_points: int = 3,
) -> np.ndarray:
    """Raster mask of the former lawn: cells with at least ``min_points`` low-vegetation points, closed by one cell."""
    mask = np.zeros(shape, dtype=bool)
    pts = np.asarray(lawn_xyz, dtype=float).reshape(-1, 3)
    if len(pts):
        flat, inside = _cell_flat(pts, bounds, float(resolution), shape)
        counts = np.bincount(flat[inside], minlength=shape[0] * shape[1]).reshape(shape)
        mask = counts >= int(min_points)
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    if protect is not None:
        mask &= ~np.asarray(protect, dtype=bool)
    return mask


def points_in_mask(xyz: np.ndarray, mask: np.ndarray, bounds, resolution: float) -> np.ndarray:
    """Boolean per point: lies in a ``True`` cell of the north-up raster ``mask``."""
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    flat, inside = _cell_flat(pts, bounds, float(resolution), mask.shape)
    out = np.zeros(len(pts), dtype=bool)
    out[inside] = np.asarray(mask, dtype=bool).ravel()[flat[inside]]
    return out


def lawn_ground_dtm(
    dtm: np.ndarray,
    bounds,
    resolution: float,
    lawn_xyz: np.ndarray,
    support_xyz: np.ndarray,
    *,
    protect: np.ndarray | None = None,
    config: Mapping | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ground height of former lawn areas: grass tops are not ground.

    Lawn points contain grass and shrub tops (several cm above the soil), so the per-cell median of the ground
    points over-estimates the terrain.  In the lawn cells (see :func:`lawn_cell_mask`) the DTM becomes the
    ``terrain.lawn_ground_percentile`` (default 20) of the ``support_xyz`` heights per cell (never higher than
    before), cells with fewer than ``terrain.lawn_ground_min_points`` points are interpolated (a cell never drops more than ``terrain.lawn_ground_max_lowering_m``, default 0.25 m)
     from the surrounding
    ground (every non-lawn DTM cell, pavement edges included), and a 3x3 median removes single-cell noise.
    Returns ``(dtm, lawn_mask, report)``.
    """
    from samcloud.terrain.rasters import _cell_percentile

    tcfg = _cfg(config, "terrain")
    percentile = float(tcfg.get("lawn_ground_percentile", 20.0))
    min_points = int(tcfg.get("lawn_ground_min_points", 4))
    max_lowering = float(tcfg.get("lawn_ground_max_lowering_m", 0.25))
    z = np.asarray(dtm, dtype=np.float64).copy()
    shape = z.shape
    mask = lawn_cell_mask(lawn_xyz, bounds, resolution, shape, protect=protect)
    report: dict = {"percentile": percentile, "cells": int(mask.sum()), "interpolated_cells": 0,
                    "mean_lowering_m": 0.0, "max_lowering_m": 0.0}
    if not mask.any():
        return z, mask, report
    before = z.copy()
    pts = np.asarray(support_xyz, dtype=float).reshape(-1, 3)
    flat, inside = _cell_flat(pts, bounds, float(resolution), shape)
    sel = inside & mask.ravel()[np.where(inside, flat, 0)]
    have = np.zeros(shape, dtype=bool)
    if sel.any():
        counts = np.bincount(flat[sel], minlength=shape[0] * shape[1])
        cells, values = _cell_percentile(flat[sel], pts[sel, 2], percentile)
        enough = counts[cells] >= min_points
        cells, values = cells[enough], values[enough]
        old = before.ravel()[cells]
        new = np.where(np.isfinite(old), np.clip(values, old - max_lowering, old), values)  # never higher, never absurdly lower
        z.ravel()[cells] = new
        have.ravel()[cells] = True
    holes = mask & ~have
    if holes.any():
        sources = np.isfinite(z) & ~holes
        if sources.sum() >= 3:
            z = _interpolate_removed(np.where(np.isfinite(z), z, 0.0), holes, sources)
        report["interpolated_cells"] = int(holes.sum())
    valid = np.isfinite(z)
    smooth = ndimage.median_filter(np.where(valid, z, np.nanmean(z[valid]) if valid.any() else 0.0), size=3)
    use = mask & valid
    z[use] = smooth[use]
    changed = mask & np.isfinite(before) & np.isfinite(z)
    if changed.any():
        delta = (before - z)[changed]
        report["mean_lowering_m"] = round(float(delta.mean()), 4)
        report["max_lowering_m"] = round(float(delta.max()), 4)
    z[~np.isfinite(np.asarray(dtm, dtype=np.float64))] = np.nan  # the measured/clipped extent of the DTM stays
    return z, mask, report


# ---------------------------------------------------------------------------------- deviation


def tin_deviation(tin, ground_xyz: np.ndarray, *, max_points: int = 400_000, trim_m: float = 0.3) -> dict:
    """RMS / P95 / max vertical deviation between ``tin`` and ground points, per TIN label.

    Points are sampled deterministically (every k-th) down to ``max_points``; each point is assigned to
    the label of the triangle that contains it.  Points outside the TIN are counted as ``outside``; ``rms_trimmed_m`` ignores errors above ``trim_m``.
    """
    pts = np.asarray(ground_xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) > max_points:
        pts = pts[:: int(np.ceil(len(pts) / max_points))]
    vertices = np.asarray(tin.vertices, dtype=np.float64)
    triangles = np.asarray(tin.triangles, dtype=np.int64)
    labels = np.asarray(tin.labels)
    names = {code: name for name, code in TIN_LABELS.items()}
    out: dict = {"points": int(len(pts)), "labels": {}}
    if len(pts) == 0 or len(triangles) == 0:
        return out
    tree = shapely.STRtree(shapely.polygons(vertices[triangles][:, :, :2]))
    pidx, tidx = tree.query(shapely.points(pts[:, :2]), predicate="intersects")
    _, first = np.unique(pidx, return_index=True)
    pidx, tidx = pidx[first], tidx[first]
    tri = vertices[triangles[tidx]]
    p = pts[pidx, :2]
    v0, v1, v2 = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
    den = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
    ok = np.abs(den) > 1e-18
    den = np.where(ok, den, 1.0)
    w0 = ((v1[:, 1] - v2[:, 1]) * (p[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (p[:, 1] - v2[:, 1])) / den
    w1 = ((v2[:, 1] - v0[:, 1]) * (p[:, 0] - v2[:, 0]) + (v0[:, 0] - v2[:, 0]) * (p[:, 1] - v2[:, 1])) / den
    zz = w0 * tri[:, 0, 2] + w1 * tri[:, 1, 2] + (1.0 - w0 - w1) * tri[:, 2, 2]
    err = (pts[pidx, 2] - zz)[ok]
    lab = labels[tidx][ok]
    out["outside"] = int(len(pts) - len(err))
    for code, name in names.items():
        sel = np.abs(err[lab == code])
        if len(sel):
            core = sel[sel <= trim_m]
            out["labels"][name] = {"points": int(len(sel)), "rms_m": round(float(np.sqrt(np.mean(sel ** 2))), 4),
                                   "p95_m": round(float(np.percentile(sel, 95)), 4), "max_m": round(float(sel.max()), 4),
                                   # without gross outliers (non-ground points that passed the ground filter)
                                   "rms_trimmed_m": round(float(np.sqrt(np.mean(core ** 2))), 4) if len(core) else 0.0,
                                   "outliers_over_trim": int(len(sel) - len(core))}
    return out


__all__ = ["clean_dtm", "lawn_cell_mask", "lawn_ground_dtm", "points_in_mask", "protect_mask", "smooth_pavement_surfaces", "tin_deviation"]
