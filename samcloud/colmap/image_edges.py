"""Image edges, sparse COLMAP anchors and local, edge-aware alignment of the Depth-Anything prior.

Used by ``depth_fill`` (indoor step 6). Everything here works on a small
"working" raster (longest side ``work_px``) of one view:

* ``load_sparse_anchors``: reliable sparse points per view (reprojection error
  and track length filtered; hooks, switches and handles mis-triangulate).
* ``detect_edges``: long LSD lines (+ Canny), strong ones preferably on SAM
  label boundaries; micro lines (jackets, wood grain) are ignored.
* ``fit_prior_regions``: the (already globally aligned) prior is re-fitted per
  region = (area bounded by strong lines) x (image tile) against COLMAP depth and
  sparse anchors. Regions never borrow scale across a strong edge; regions without
  anchors in reach or whose residual stays above the tolerance are distrusted.
* ``edge_consistency``: halo (prior jump without image edge) and wrong-prior
  (image edge with COLMAP depth jump where the prior stays smooth) masks.
* ``lines_to_3d``: image lines with >= 2 anchors along them become 3D segments.
* ``complete_planes``: RANSAC planes from anchors per textureless SAM region.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

EDGE_DEFAULTS: dict[str, Any] = {
    "work_px": 480,                 # longest side of the working raster
    "min_line_frac": 60.0 / 1440.0,  # minimum line length relative to the image size
    "strong_line_frac": 2.5,        # lines this many times the minimum are strong even without a SAM boundary
    "sam_boundary_share": 0.3,      # share of a line that lies on a label boundary to count as strong
    "line_px": 3,                   # drawn line thickness on the working raster
    "canny_low": 60, "canny_high": 140,
    "tiles": 8,
    "min_samples": 20.0,            # weighted anchor samples a region needs for a fit
    "min_count": 2,
    "sparse_weight": 10.0,
    "reach_frac": 0.25,             # fit reach (fraction of the working size) for regions without own anchors
    "anchor_radius_frac": 0.05,     # "anchor nearby" distance (fraction of the working size)
    "residual_tolerance": 0.03,
    "sparse_max_error_px": 1.5,
    "sparse_min_track": 3,
    "halo_px": 2,                   # working pixels around a prior jump rejected when no image edge explains it
    "halo_jump_rel": 0.06,
    "edge_match_px": 3,
    "wrong_jump_rel": 0.10,
    "wrong_band_px": 6,
}


def _cv2():
    import cv2

    return cv2


# ----------------------------------------------------------------------------- sparse anchors


def load_sparse_anchors(sparse_dir: Path, views, max_error_px: float = 1.5, min_track: int = 3,
                        reconstruction=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (uv Nx2 in view pixels, camera depth N) of reliable sparse points seen by the view."""
    if reconstruction is None:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    by_id: dict[int, list[np.ndarray]] = {}
    for point in reconstruction.points3D.values():
        if point.error > max_error_px or point.track.length() < min_track:
            continue
        xyz = np.asarray(point.xyz, dtype=np.float64)
        for element in point.track.elements:
            by_id.setdefault(int(element.image_id), []).append(xyz)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    by_name = {view.name: view for view in views}
    for image_id, points in by_id.items():
        image = reconstruction.images[image_id]
        view = by_name.get(image.name)
        if view is None:
            continue
        local = np.asarray(points) @ view.rotation.T + view.translation
        z = local[:, 2]
        ok = z > 1e-3
        u = view.fx * local[ok, 0] / z[ok] + view.cx
        v = view.fy * local[ok, 1] / z[ok] + view.cy
        inside = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)
        result[view.name] = (np.column_stack((u[inside], v[inside])).astype(np.float32), z[ok][inside].astype(np.float32))
    return result


# ----------------------------------------------------------------------------- lines


def working_shape(shape: tuple[int, int], settings: dict[str, Any]) -> tuple[int, int]:
    scale = min(1.0, float(settings["work_px"]) / max(shape))
    return max(8, int(round(shape[0] * scale))), max(8, int(round(shape[1] * scale)))


def detect_edges(image_rgb: np.ndarray | None, labels: np.ndarray | None, shape: tuple[int, int],
                 settings: dict[str, Any]) -> dict[str, Any]:
    """Strong lines (mask at working ``shape``), all long lines and a Canny edge mask.

    Lines shorter than ``min_line_frac`` of the image width are ignored. A line is
    strong when it lies on a SAM label boundary or is much longer than the minimum.
    """
    cv2 = _cv2()
    h, w = shape
    strong = np.zeros(shape, np.uint8)
    canny = np.zeros(shape, bool)
    lines: list[tuple[float, float, float, float, bool]] = []
    if image_rgb is None:
        return {"strong": strong > 0, "canny": canny, "lines": lines}
    gray = cv2.cvtColor(np.ascontiguousarray(image_rgb[:, :, :3]), cv2.COLOR_RGB2GRAY)
    detect_size = (min(960, gray.shape[1]), min(960, gray.shape[0]))
    small = cv2.resize(gray, detect_size, interpolation=cv2.INTER_AREA)
    boundary = None
    if labels is not None:
        lab = cv2.resize(np.asarray(labels).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        grad = (cv2.dilate(lab, np.ones((3, 3), np.uint8)) != cv2.erode(lab, np.ones((3, 3), np.uint8)))
        boundary = cv2.dilate(grad.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = detector.detect(small)[0]
    min_len = float(settings["min_line_frac"]) * detect_size[0]
    sx, sy = w / detect_size[0], h / detect_size[1]
    if found is not None:
        for x1, y1, x2, y2 in found[:, 0, :]:
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_len:
                continue
            p1, p2 = (float(x1) * sx, float(y1) * sy), (float(x2) * sx, float(y2) * sy)
            on_boundary = False
            if boundary is not None:
                count = max(2, int(length // 4))
                xs = np.clip(np.linspace(p1[0], p2[0], count).astype(int), 0, w - 1)
                ys = np.clip(np.linspace(p1[1], p2[1], count).astype(int), 0, h - 1)
                on_boundary = float(boundary[ys, xs].mean()) >= float(settings["sam_boundary_share"])
            is_strong = on_boundary or length >= float(settings["strong_line_frac"]) * min_len
            lines.append((p1[0], p1[1], p2[0], p2[1], is_strong))
            if is_strong:
                cv2.line(strong, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), 1,
                         int(settings["line_px"]))
    edges = cv2.Canny(cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA), int(settings["canny_low"]),
                      int(settings["canny_high"]))
    canny = edges > 0
    return {"strong": strong > 0, "canny": canny, "lines": lines}


def edge_components(strong: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected areas of the working raster that strong lines do not cross."""
    cv2 = _cv2()
    count, labels = cv2.connectedComponents((~strong).astype(np.uint8), connectivity=4)
    return labels.astype(np.int32), int(count)


# ----------------------------------------------------------------------------- alignment


def _weighted_median_by_group(keys: np.ndarray, values: np.ndarray, weights: np.ndarray, groups: int) -> np.ndarray:
    result = np.full(groups, np.nan)
    if not len(keys):
        return result
    order = np.lexsort((values, keys))
    k, v, w = keys[order], values[order], weights[order]
    boundaries = np.flatnonzero(np.r_[True, k[1:] != k[:-1], True])
    cumulative = np.cumsum(w)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        base = cumulative[start - 1] if start else 0.0
        half = base + 0.5 * (cumulative[stop - 1] - base)
        result[k[start]] = v[start + int(np.searchsorted(cumulative[start:stop], half))]
    return result


def _anchor_samples(geometric_w: np.ndarray, sparse_w: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                    settings: dict[str, Any]):
    """rows, cols, depth, weight of dense COLMAP pixels (weight 1) and sparse anchors (high weight)."""
    rows, cols = np.nonzero(geometric_w > 0)
    depth = geometric_w[rows, cols].astype(np.float64)
    weight = np.ones(len(rows))
    if sparse_w is not None and len(sparse_w[0]):
        sr, sc, sz = sparse_w
        rows = np.concatenate([rows, sr])
        cols = np.concatenate([cols, sc])
        depth = np.concatenate([depth, sz.astype(np.float64)])
        weight = np.concatenate([weight, np.full(len(sr), float(settings["sparse_weight"]))])
    return rows, cols, depth, weight


def fit_prior_regions(prior_w: np.ndarray, geometric_w: np.ndarray, sparse_w, components: np.ndarray | None,
                      settings: dict[str, Any]):
    """Local scale per region; returns (scale map at working res, 0 = distrusted), near-anchor mask, info.

    ``prior_w`` / ``geometric_w`` are at working resolution (geometric 0 where COLMAP has no valid
    depth). ``components`` is the strong-line component map (None = no edge awareness).
    """
    from scipy import ndimage

    h, w = prior_w.shape
    tiles = max(1, int(settings["tiles"]))
    th, tw = math.ceil(h / tiles), math.ceil(w / tiles)
    comp = components if components is not None else np.zeros((h, w), np.int32)
    n_comp = int(comp.max()) + 1
    tile_id = (np.arange(h)[:, None] // th) * tiles + (np.arange(w)[None, :] // tw)
    region = comp.astype(np.int64) * (tiles * tiles) + tile_id
    groups = n_comp * tiles * tiles
    rows, cols, depth, weight = _anchor_samples(geometric_w, sparse_w, settings)
    prior_at = prior_w[rows, cols].astype(np.float64)
    ok = prior_at > 0
    rows, cols, depth, weight, prior_at = rows[ok], cols[ok], depth[ok], weight[ok], prior_at[ok]
    keys = region[rows, cols]
    ratio = depth / prior_at
    scale_g = _weighted_median_by_group(keys, ratio, weight, groups)
    total_w = np.bincount(keys, weights=weight, minlength=groups)
    count = np.bincount(keys, minlength=groups)
    resid = np.abs(prior_at * scale_g[keys] - depth) / np.maximum(depth, 1e-6)
    median_resid = _weighted_median_by_group(keys, resid, weight, groups)
    good = (total_w >= float(settings["min_samples"])) & (count >= int(settings["min_count"])) & (
        median_resid <= float(settings["residual_tolerance"]))
    scale_g = np.where(good, scale_g, np.nan)
    scale_map = scale_g[region]  # nan where the region has no good fit
    anchor_mask = np.zeros((h, w), bool)
    anchor_mask[rows, cols] = True
    dist_anchor = ndimage.distance_transform_edt(~anchor_mask) if anchor_mask.any() else np.full((h, w), np.inf)
    reach = float(settings["reach_frac"]) * max(h, w)
    # whole edge-bounded area as a coarser region: few reliable anchors (corner points) can still align a textureless wall
    ckeys = comp[rows, cols].astype(np.int64)
    cscale = _weighted_median_by_group(ckeys, ratio, weight, n_comp)
    ctotal = np.bincount(ckeys, weights=weight, minlength=n_comp)
    ccount = np.bincount(ckeys, minlength=n_comp)
    cresid = _weighted_median_by_group(ckeys, np.abs(prior_at * cscale[ckeys] - depth) / np.maximum(depth, 1e-6), weight, n_comp)
    cgood = (ctotal >= float(settings["min_samples"])) & (ccount >= int(settings["min_count"])) & (
        cresid <= float(settings["residual_tolerance"]))
    coarse = np.where(cgood, cscale, np.nan)[comp]
    use_coarse = np.isnan(scale_map) & ~np.isnan(coarse) & (dist_anchor <= reach)
    scale_map = np.where(use_coarse, coarse, scale_map)
    fitted = ~np.isnan(scale_map)
    if fitted.any() and not fitted.all():
        dist, (ri, ci) = ndimage.distance_transform_edt(~fitted, return_indices=True)
        same_comp = comp[ri, ci] == comp
        borrow = ~fitted & same_comp & (dist <= reach)
        scale_map = np.where(borrow, scale_map[ri, ci], scale_map)
    # final check: every region's own samples must agree with the scale it ended up with
    sample_scale = scale_map[rows, cols]
    have = ~np.isnan(sample_scale)
    final_resid = np.abs(prior_at[have] * sample_scale[have] - depth[have]) / np.maximum(depth[have], 1e-6)
    region_resid = _weighted_median_by_group(keys[have], final_resid, weight[have], groups)
    bad_region = region_resid > float(settings["residual_tolerance"])
    trusted = ~np.isnan(scale_map) & ~bad_region[region]
    scale_out = np.where(trusted, scale_map, 0.0).astype(np.float32)
    radius = float(settings["anchor_radius_frac"]) * max(h, w)
    near = dist_anchor <= radius if anchor_mask.any() else np.zeros((h, w), bool)
    info = {"regions": int(np.count_nonzero(good)), "bad_regions": int(np.count_nonzero(bad_region)),
            "trusted_share": float(trusted.mean()), "anchors": int(len(rows))}
    return scale_out, near, info


def sparse_to_working(sparse, view, shape_w: tuple[int, int]):
    """Sparse anchors (view pixels) -> (rows, cols, depth) on the working raster."""
    if sparse is None or not len(sparse[0]):
        return None
    uv, z = sparse
    h, w = shape_w
    cols = np.clip((uv[:, 0] * w / view.width).astype(np.int64), 0, w - 1)
    rows = np.clip((uv[:, 1] * h / view.height).astype(np.int64), 0, h - 1)
    return rows, cols, z


# ----------------------------------------------------------------------------- edge consistency


def edge_consistency(prior_w: np.ndarray, geometric_w: np.ndarray, edges: dict[str, Any], settings: dict[str, Any]):
    """Returns (halo mask, wrong-prior mask) on the working raster.

    halo: a prior depth jump no image edge explains, widened by ``halo_px``.
    wrong: a strong image line with a COLMAP depth jump across it where the prior stays smooth.
    """
    cv2 = _cv2()
    h, w = prior_w.shape
    safe = np.maximum(prior_w, 1e-6)
    jump = np.zeros((h, w), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        right = np.abs(np.diff(prior_w, axis=1)) / safe[:, :-1]
        down = np.abs(np.diff(prior_w, axis=0)) / safe[:-1, :]
    limit = float(settings["halo_jump_rel"])
    jump[:, :-1] |= (right > limit) & (prior_w[:, :-1] > 0) & (prior_w[:, 1:] > 0)
    jump[:-1, :] |= (down > limit) & (prior_w[:-1, :] > 0) & (prior_w[1:, :] > 0)
    k = 2 * int(settings["edge_match_px"]) + 1
    explained = cv2.dilate((edges["canny"] | edges["strong"]).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    unexplained = jump & ~explained
    r = int(settings["halo_px"])
    halo = cv2.dilate(unexplained.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0 if r > 0 else unexplained
    wrong = np.zeros((h, w), bool)
    band = int(settings["wrong_band_px"])
    for x1, y1, x2, y2, is_strong in edges["lines"]:
        if not is_strong:
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 4:
            continue
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
        count = max(3, int(length // 4))
        ts = np.linspace(0.1, 0.9, count)
        px, py = x1 + ts * (x2 - x1), y1 + ts * (y2 - y1)
        sides = []
        for sign in (1, -1):
            sx = np.clip((px + sign * nx * band).round().astype(int), 0, w - 1)
            sy = np.clip((py + sign * ny * band).round().astype(int), 0, h - 1)
            sides.append((geometric_w[sy, sx], prior_w[sy, sx]))
        (ga, pa), (gb, pb) = sides
        ok = (ga > 0) & (gb > 0) & (pa > 0) & (pb > 0)
        if ok.sum() < 3:
            continue
        gj = np.median(np.abs(ga[ok] - gb[ok]) / np.minimum(ga[ok], gb[ok]))
        pj = np.median(np.abs(pa[ok] - pb[ok]) / np.minimum(pa[ok], pb[ok]))
        if gj > float(settings["wrong_jump_rel"]) and pj < 0.5 * gj:
            cv2.line(wrong_img := np.zeros((h, w), np.uint8), (int(round(x1)), int(round(y1))),
                     (int(round(x2)), int(round(y2))), 1, 2 * band + 1)
            wrong |= wrong_img > 0
    return halo, wrong


# ----------------------------------------------------------------------------- 3D lines


def lines_to_3d(edges: dict[str, Any], sparse_w, view, shape_w: tuple[int, int], min_anchors: int = 2,
                tol_px: float = 3.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Strong image lines with >= ``min_anchors`` sparse anchors along them -> world segments."""
    if sparse_w is None:
        return []
    rows, cols, z = sparse_w
    h, w = shape_w
    segments = []
    for x1, y1, x2, y2, is_strong in edges["lines"]:
        if not is_strong:
            continue
        d = np.array([x2 - x1, y2 - y1])
        length = float(np.hypot(*d))
        if length < 4:
            continue
        rel = np.column_stack((cols - x1, rows - y1))
        t = rel @ d / length**2
        dist = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / length
        on = (dist <= tol_px) & (t >= -0.05) & (t <= 1.05)
        if on.sum() < min_anchors:
            continue
        tt = t[on]
        zz = z[on]
        a, b = np.argmin(tt), np.argmax(tt)
        if tt[b] - tt[a] <= 0.0:
            continue
        ends = []
        for index in (a, b):
            u = (x1 + tt[index] * d[0]) * view.width / w
            v = (y1 + tt[index] * d[1]) * view.height / h
            cam = np.array([(u - view.cx) / view.fx * zz[index], (v - view.cy) / view.fy * zz[index], zz[index]])
            ends.append((cam - view.translation) @ view.rotation)
        segments.append((ends[0], ends[1]))
    return segments


# ----------------------------------------------------------------------------- plane completion


def ransac_plane(points: np.ndarray, tol: float, iterations: int = 200, rng=None):
    """Best plane (normal, offset) and inlier mask for ``points`` (n x 3) or None."""
    rng = rng or np.random.default_rng(0)
    n = len(points)
    if n < 4:
        return None
    best, best_count = None, 0
    for _ in range(iterations):
        a, b, c = points[rng.choice(n, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        offset = -float(normal @ a)
        inliers = np.abs(points @ normal + offset) <= tol
        if inliers.sum() > best_count:
            best, best_count = (normal, offset), int(inliers.sum())
    if best is None:
        return None
    inliers = np.abs(points @ best[0] + best[1]) <= tol
    if inliers.sum() >= 3:  # refine
        centre = points[inliers].mean(axis=0)
        normal = np.linalg.svd(points[inliers] - centre)[2][-1]
        best = (normal, -float(normal @ centre))
        inliers = np.abs(points @ best[0] + best[1]) <= tol
    return best[0], best[1], inliers


def complete_planes(prior_w: np.ndarray, geometric_w: np.ndarray, sparse_w, region_mask: np.ndarray,
                    components: np.ndarray | None, view, settings: dict[str, Any], tol: float):
    """Plane depth for textureless SAM regions from anchors; returns (depth map at working res, planes used).

    Anchors (COLMAP dense pixels + sparse) inside the region and its boundary ring are clustered by
    successive RANSAC planes (each needs ``plane_min_inliers`` and ``plane_min_inlier_share`` of its
    region's anchors left); anchors on different sides of a strong edge never share a plane (planes are
    fitted per edge component). Pixels take the plane of the nearest anchor of their own component
    within ``plane_reach_frac``; the prior only refines within ``tol``, otherwise the plane wins.
    """
    from scipy import ndimage
    import cv2

    h, w = prior_w.shape
    out = np.zeros((h, w), np.float32)
    rows, cols, depth, weight = _anchor_samples(geometric_w, sparse_w, settings)
    ring = cv2.dilate(region_mask.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    keep = ring[rows, cols]
    rows, cols, depth = rows[keep], cols[keep], depth[keep]
    if len(rows) < int(settings.get("plane_min_inliers", 12)):
        return out, 0
    comp = components if components is not None else np.zeros((h, w), np.int32)
    anchor_comp = comp[rows, cols]
    u = (cols + 0.5) * view.width / w
    v = (rows + 0.5) * view.height / h
    cam = np.column_stack(((u - view.cx) / view.fx * depth, (v - view.cy) / view.fy * depth, depth))
    world = (cam - view.translation) @ view.rotation
    plane_of = np.full(len(rows), -1)
    planes = []
    rng = np.random.default_rng(0)
    for cid in np.unique(anchor_comp):
        idx = np.flatnonzero(anchor_comp == cid)
        free = idx.copy()
        for _ in range(int(settings.get("plane_max_planes", 3))):
            if len(free) < int(settings.get("plane_min_inliers", 12)):
                break
            if len(free) > 4000:
                free = rng.choice(free, 4000, replace=False)
            fit = ransac_plane(world[free], tol, rng=rng)
            if fit is None:
                break
            normal, offset, inl = fit
            if inl.sum() < int(settings.get("plane_min_inliers", 12)):
                break
            full_inl = np.abs(world[idx] @ normal + offset) <= tol
            planes.append((normal, offset))
            plane_of[idx[full_inl & (plane_of[idx] < 0)]] = len(planes) - 1
            free = idx[plane_of[idx] < 0]
        # an edge component whose anchors do not mostly sit on its planes is not a plane
    ok = plane_of >= 0
    if not ok.any():
        return out, 0
    plane_mask = np.zeros((h, w), bool)
    plane_mask[rows[ok], cols[ok]] = True
    dist, (ri, ci) = ndimage.distance_transform_edt(~plane_mask, return_indices=True)
    index_map = np.full((h, w), -1, np.int32)
    index_map[rows[ok], cols[ok]] = plane_of[ok]
    nearest = index_map[ri, ci]
    reach = float(settings.get("plane_reach_frac", 0.15)) * max(h, w)
    use = region_mask & (dist <= reach) & (comp[ri, ci] == comp) & (nearest >= 0)
    yy, xx = np.nonzero(use)
    if not len(yy):
        return out, len(planes)
    pu = (xx + 0.5) * view.width / w
    pv = (yy + 0.5) * view.height / h
    rays = np.column_stack(((pu - view.cx) / view.fx, (pv - view.cy) / view.fy, np.ones(len(pu))))
    pl = np.asarray([planes[i] for i in nearest[yy, xx]], dtype=object)
    normals = np.stack([p[0] for p in pl])
    offsets = np.array([p[1] for p in pl])
    # plane in camera frame: n_w . (R^T (X - t)... ) -> convert normal/offset to camera coordinates
    n_cam = normals @ view.rotation.T
    o_cam = offsets - normals @ (view.rotation.T @ view.translation)
    denom = np.einsum("ij,ij->i", n_cam, rays)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -o_cam / denom
    valid = np.isfinite(z) & (z > 0.05) & (np.abs(denom) > 0.05)
    p = prior_w[yy, xx]
    z = np.where((p > 0) & (np.abs(p - z) <= tol * np.maximum(z, 1e-6)), p, z)
    out[yy[valid], xx[valid]] = z[valid].astype(np.float32)
    return out, len(planes)
