"""Merge all depth maps into one surface: one point layer, like a mesh (indoor, step 6).

Each camera contributes its own depth: COLMAP's geometric depth where it
measured something, and - only where it agrees with COLMAP - the Depth-Anything-V2
prior for the textureless gaps in between:

* Local fit: the prior is fitted to the camera's own COLMAP depth per image
  tile (``prior_tiles`` x ``prior_tiles``, robust median scale) and blended
  smoothly between tiles. A tile whose fitted prior still deviates from COLMAP
  by more than ``prior_agree_tolerance`` is not used, nor are tiles more than
  ``prior_reach_tiles`` away from any measured tile. Views without COLMAP depth
  contribute no prior. Depth-Anything never adds surfaces COLMAP does not back.
* Every depth map then goes into a truncated signed distance field (TSDF, as in
  KinectFusion / scanner software): each voxel holds the weighted average
  signed distance to the surface all cameras see (measured depth weighs
  ``geometric_weight``, prior ``prior_weight``). Space any camera looks through
  becomes "empty", so nothing can stay where a camera sees free space.

``mode: extract`` (default) builds the cloud from the field itself: one point
on every voxel edge where the field crosses zero between two observed voxels.
That is a single layer by construction - no copies of a wall behind each
other - spaced about one voxel apart. Colour is the weighted mean of the images,
the class a weighted majority vote of the SAM label images seen at the surface.

``mode: snap`` keeps the existing classified points and pulls them onto the zero
level (earlier behaviour).

Visibility (``visibility_check``): every point is projected into all COLMAP
depth maps. A point that at least ``visibility_min_through`` cameras look
straight through (they measure a surface clearly behind it) more often than
cameras confirm it cannot exist and is removed - whatever Depth-Anything says.
After plane snapping a second pass also uses the agreeing prior: points seen by
fewer than ``point_min_seen_views`` cameras but hidden behind a nearer surface
in ``point_hidden_views`` (behind walls, above the ceiling), and wall planes
whose points the cameras hardly see (``wall_min_seen_views``), are removed.
The same pass runs the ``hidden_filter``: a point no COLMAP depth map confirms
(measured depth equal to it or beyond it, at its pixel or in a small window around
it), that at most ``max_prior_views`` agreeing priors back, and that at least
``min_closed_views`` cameras see only through a clearly nearer COLMAP surface
(floor behind a wall, an unscanned room), is removed. Points without any such
evidence (textureless ceilings no camera measured) stay.

Afterwards ``plane_classes`` (ceiling, floor, walls) are fitted with planes
(horizontal for ceiling/floor, vertical for walls) and points within
``plane_snap_m`` are put exactly onto their plane, so a ceiling has one height
instead of several. Walls follow the building's two main directions
(``plane_manhattan``), only snap along their measured length, and parallel
copies facing the same way closer than ``plane_duplicate_m`` are merged (the
two faces of an interior wall face opposite ways and stay apart).

Class coherence: for the large classes, a point whose class hardly occurs in
its surroundings while another class dominates takes that class (no wall
points on the sofa); tiny isolated fragments are dropped. Points with fewer
than ``min_neighbors`` neighbours (haze) are removed last.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from samcloud.engine.ply_io import read_vertex_array
from samcloud.colmap.plane_fill import View, read_colmap_array

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "mode": "extract",
    "voxel_m": 0.015,
    "truncation_m": 0.12,
    "max_prior_depth": 4.0,
    "max_geometric_depth": 10.0,
    "geometric_weight": 10.0,
    "prior_weight": 1.0,
    "min_weight": 2.0,
    "prior_tiles": 8,
    "prior_min_samples": 150,
    "prior_agree_tolerance": 0.02,
    "prior_reach_tiles": 2,
    "edge_tolerance": 0.02,
    "max_incidence_deg": 70.0,
    "exclude_classes": ["window"],
    "depth_edge_filter": {"enabled": True, "max_incidence_deg": 80.0, "edge_px": 1,
                          "edge_jump_rel": 0.05, "edge_jump_m": 0.03, "drop_lonely": False},
    "mirror_filter": {"enabled": True, "ring_inner_px": 4, "ring_outer_px": 12, "min_depth_jump_m": 0.25,
                      "merge_angle_deg": 3, "merge_offset_m": 0.03, "min_views": 2, "min_baseline_m": 0.3,
                      "min_mask_share": 0.002, "mirror_dilate_px": 6, "through_share": 0.8, "fill_plane": False,
                      "prior_tolerance": 0.03, "prior_edge_tolerance": 0.02,
                      "free_space_views": 2, "visibility_ignore_masks": True,
                      "ring_label_ids": [2, 5, 17]},
    "behind_wall_filter": {"enabled": False, "keep_depth_m": 1.0, "cell_m": 0.05, "floor_band_m": 0.15,
                           "extent_margin_m": 0.25, "blocked_share": 0.9, "min_support_views": 2,
                           "mirror_image": {"enabled": False, "search_front_m": 1.2, "search_back_m": 0.2,
                                            "step_m": 0.01, "match_m": 0.04, "min_score": 0.35,
                                            "min_points": 2000}},
    # snap mode
    "iterations": 5,
    "max_residual_m": 0.02,
    "dedupe_m": 0.01,
    "drop_free_space": True,
    "drop_behind_surface": True,
    # visibility: COLMAP depth only, Depth-Anything has no say here
    "visibility_check": True,
    "visibility_margin_m": 0.03,
    "visibility_margin_rel": 0.01,
    "visibility_min_through": 2,
    "visibility_through_ratio": 1.0,
    "wall_min_seen_views": 3,
    "point_min_seen_views": 2,   # points seen by fewer views ...
    "point_hidden_views": 10,    # ... but hidden behind a nearer surface in this many are removed  # median number of views seeing a wall plane; fewer: a wall behind a wall
    "units_per_metre": 1.0,
    # points no COLMAP depth confirms, few priors back and many cameras only see through a nearer COLMAP surface
    "hidden_filter": {"enabled": False, "window_frac": 0.0556, "window_grid": 5, "min_window_valid": 5,
                      "open_share": 0.2, "closed_share": 0.8, "min_closed_views": 10, "max_prior_views": 2},
    # occupancy ray-march: every sampled supporting view of a point looks through a solid surface
    "raymarch_filter": {"enabled": True, "voxel_m": 0.04, "dilate_voxels": 2, "structure_classes": ["wall", "door", "ceiling", "floor"],
                        "occ_min_support": 30, "own_skip_voxels": 3, "run_samples": 3, "max_steps": 200, "min_false": 3,
                        "max_views": 32, "chunk": 20000, "max_voxels": 400_000_000,
                        # tall furniture (built-in wardrobes, shelf walls) occludes like a wall; low furniture stays permeable
                        "tall_furniture_enabled": True, "tall_furniture_classes": ["furniture"], "tall_furniture_min_height_m": 1.6,
                        "tall_furniture_cell_m": 0.08, "tall_furniture_min_fill": 0.5, "tall_furniture_min_extent_m": 0.5,
                        "tall_furniture_min_points": 2, "floor_cell_m": 1.0,
                        # holes in tall vertical bodies narrower than close_gap_m are closed (textureless wall, glossy panels)
                        "close_gaps_enabled": True, "body_fill": 0.35, "close_gap_m": 0.5, "fill_body_columns": False,
                        # a point whose line of sight runs longer than this inside solid voxels is buried in a body: that view counts as blocked
                        "bury_depth_m": 0.0},
    # planes
    "plane_classes": ["ceiling", "floor", "wall"],
    "plane_manhattan": True,
    "plane_manhattan_deg": 4.0,
    "plane_duplicate_m": 0.20,
    "plane_duplicate_deg": 10.0,
    "plane_extent_margin_m": 0.25,
    # class coherence
    "class_coherence": True,
    "class_coherence_classes": ["wall", "ceiling", "floor", "furniture", "door", "window", "unknown"],
    "class_cell_m": 0.05,
    "class_window_cells": 5,
    "class_min_share": 0.2,
    "class_majority_share": 0.5,
    "class_min_points": 20,
    "plane_snap_m": 0.04,
    "plane_inlier_m": 0.02,
    "plane_min_points": 5000,
    "plane_max_failures": 15,
    "plane_merge_m": {"ceiling": 0.25, "floor": 0.10},
    "clip_to_levels": True,
    "plane_level_min_share": 0.05,
    "plane_max_planes": 40,
    "plane_hypotheses": 256,
    "plane_angle_deg": 8.0,
    # haze
    "neighbor_cell_m": 0.03,
    "min_neighbors": 30,
    "neighbor_passes": 2,
    "max_voxels": 250_000_000,
    "bounds_percentile": 0.2,
}


# ----------------------------------------------------------------------------- per-view depth


def _edge_mask_t(depth, tolerance: float):
    import torch

    smooth = torch.ones_like(depth, dtype=torch.bool)
    safe = depth.clamp(min=1e-6)
    right = (depth[:, 1:] - depth[:, :-1]).abs() / safe[:, :-1]
    down = (depth[1:, :] - depth[:-1, :]).abs() / safe[:-1, :]
    smooth[:, :-1] &= right <= tolerance
    smooth[:, 1:] &= right <= tolerance
    smooth[:-1, :] &= down <= tolerance
    smooth[1:, :] &= down <= tolerance
    return smooth


def _incidence_mask_t(depth, view: View, max_incidence_deg: float):
    import torch

    h, w = depth.shape
    dev = depth.device
    cols = (torch.arange(w, device=dev, dtype=torch.float32) + 0.5) * view.width / w
    rows = (torch.arange(h, device=dev, dtype=torch.float32) + 0.5) * view.height / h
    x = ((cols[None, :] - view.cx) / view.fx) * depth
    y = ((rows[:, None] - view.cy) / view.fy) * depth
    points = torch.stack((x, y, depth), dim=-1)
    du = torch.zeros_like(points)
    dv = torch.zeros_like(points)
    du[:, :-1] = points[:, 1:] - points[:, :-1]
    dv[:-1, :] = points[1:, :] - points[:-1, :]
    normal = torch.cross(du, dv, dim=-1)
    cosine = (normal * points).sum(-1).abs() / (normal.norm(dim=-1) * points.norm(dim=-1)).clamp(min=1e-12)
    return cosine >= math.cos(math.radians(max_incidence_deg))


def fit_prior_locally(prior, geometric, settings: dict[str, Any]):
    """Fit the prior to COLMAP depth per tile; returns (corrected prior, usable mask) on the same device.

    Only tiles where the fitted prior agrees with COLMAP (median relative
    deviation <= ``prior_agree_tolerance``), plus up to ``prior_reach_tiles``
    tiles around them, may use the prior.
    """
    import torch
    import torch.nn.functional as F

    h, w = geometric.shape
    tiles = max(1, int(settings["prior_tiles"]))
    th, tw = math.ceil(h / tiles), math.ceil(w / tiles)
    # all tiles at once: (tiles, tiles, th * tw) blocks, invalid pixels as NaN
    pad = (0, tiles * tw - w, 0, tiles * th - h)
    g = F.pad(geometric[None, None], pad)[0, 0].reshape(tiles, th, tiles, tw).permute(0, 2, 1, 3).reshape(tiles, tiles, -1)
    p = F.pad(prior[None, None], pad)[0, 0].reshape(tiles, th, tiles, tw).permute(0, 2, 1, 3).reshape(tiles, tiles, -1)
    both = (g > 0) & (p > 0)
    nan = torch.tensor(float("nan"), device=geometric.device)
    ratio = torch.where(both, g / torch.where(both, p, torch.ones_like(p)), nan)
    s = torch.nanmedian(ratio, dim=-1).values
    deviation = torch.nanmedian(torch.where(both, (p * s[..., None] - g).abs() / torch.where(both, g, torch.ones_like(g)), nan),
                                dim=-1).values
    enough = both.sum(dim=-1) >= int(settings["prior_min_samples"])
    scale = torch.where(enough & (deviation <= float(settings["prior_agree_tolerance"])), s, nan)
    measured = ~torch.isnan(scale)
    if not bool(measured.any()):
        return prior, torch.zeros_like(prior, dtype=torch.bool)
    # spread fitted scales to neighbouring tiles (reach), averaging known neighbours
    usable_tiles = measured.clone()
    filled = torch.nan_to_num(scale, nan=0.0)
    count = measured.float()
    for _ in range(max(0, int(settings["prior_reach_tiles"]))):
        pad_v = F.pad(filled[None, None], (1, 1, 1, 1))
        pad_c = F.pad(count[None, None], (1, 1, 1, 1))
        kernel = torch.ones((1, 1, 3, 3), device=scale.device)
        sum_v = F.conv2d(pad_v, kernel)[0, 0]
        sum_c = F.conv2d(pad_c, kernel)[0, 0]
        grow = ~usable_tiles & (sum_c > 0)
        filled = torch.where(grow, sum_v / sum_c.clamp(min=1e-6), filled)
        count = torch.where(grow, torch.ones_like(count), count)
        usable_tiles |= grow
    # smooth per-pixel scale (bilinear between tile centres), unknown tiles take the global median
    global_scale = scale[measured].median()
    grid = torch.where(usable_tiles, filled, global_scale)
    per_pixel = F.interpolate(grid[None, None], size=(tiles * th, tiles * tw), mode="bilinear",
                              align_corners=False)[0, 0, :h, :w]
    usable = F.interpolate(usable_tiles[None, None].float(), size=(tiles * th, tiles * tw),
                           mode="nearest")[0, 0, :h, :w] > 0.5
    return prior * per_pixel, usable


def view_depth(
    view: View,
    geometric: np.ndarray | None,
    prior: np.ndarray | None,
    labels: np.ndarray | None,
    excluded: tuple[int, ...],
    settings: dict[str, Any],
    device: str = "cpu",
    mirror_mask: np.ndarray | None = None,
    mirror_plane: Any | None = None,
    edge_report: dict[str, int] | None = None,
):
    """Combined depth and weights (torch, at the geometric map size) or None.

    Geometric depth where COLMAP measured; elsewhere the locally fitted prior,
    but only where it agrees with COLMAP and passes the edge, grazing-angle
    and class (window) filters.
    """
    import torch
    import torch.nn.functional as F

    if geometric is None:
        return None  # without measurements a prior cannot be checked
    g = torch.as_tensor(np.asarray(geometric, dtype=np.float32), device=device)
    h, w = g.shape
    if mirror_mask is not None:
        mask = torch.as_tensor(np.asarray(mirror_mask, dtype=bool), device=device)
        if mask.shape != g.shape:
            mask = F.interpolate(mask[None, None].float(), size=(h, w), mode="nearest")[0, 0] > 0.5
        if bool(settings.get("mirror_filter", {}).get("fill_plane", False)) and mirror_plane is not None:
            from samcloud.colmap.mirror_filter import plane_depth_map

            plane_depth = torch.as_tensor(plane_depth_map(view, (h, w), mirror_plane), device=device)
            g = torch.where(mask & (plane_depth > 0), plane_depth, g)
        else:
            g = torch.where(mask, torch.zeros_like(g), g)
    from samcloud.colmap.depth_edges import geometric_valid_mask_torch

    raw_valid = (g > 0) & (g <= float(settings["max_geometric_depth"]))
    g_ok = geometric_valid_mask_torch(g, view, settings) & (g <= float(settings["max_geometric_depth"]))
    if edge_report is not None:
        edge_report["valid"] = edge_report.get("valid", 0) + int(raw_valid.sum())
        edge_report["dropped"] = edge_report.get("dropped", 0) + int((raw_valid & ~g_ok).sum())
    depth = torch.where(g_ok, g, torch.zeros_like(g))
    weight = torch.where(g_ok, torch.full_like(g, float(settings["geometric_weight"])), torch.zeros_like(g))
    if prior is not None:
        p = torch.as_tensor(np.asarray(prior, dtype=np.float32), device=device)
        if p.shape != g.shape:
            p = F.interpolate(p[None, None], size=(h, w), mode="nearest")[0, 0]
        p, usable = fit_prior_locally(p, torch.where(g_ok, g, torch.zeros_like(g)), settings)
        ok = usable & ~g_ok & ~mask if mirror_mask is not None else usable & ~g_ok
        ok &= (p > 0) & (p <= float(settings["max_prior_depth"]))
        ok &= _edge_mask_t(p, float(settings["edge_tolerance"]))
        ok &= _incidence_mask_t(p, view, float(settings["max_incidence_deg"]))
        if labels is not None and excluded:
            lab = torch.as_tensor(np.asarray(labels).astype(np.int64), device=device)
            if lab.shape != g.shape:
                lab = F.interpolate(lab[None, None].float(), size=(h, w), mode="nearest")[0, 0].long()
            ok &= ~torch.isin(lab, torch.as_tensor(excluded, device=device))
        depth = torch.where(ok, p, depth)
        weight = torch.where(ok, torch.full_like(p, float(settings["prior_weight"])), weight)
    return depth, weight


# ----------------------------------------------------------------------------- TSDF


class Tsdf:
    """Dense TSDF grid (values in units of the truncation, -1..1) with weights, colour and class votes."""

    def __init__(self, origin: np.ndarray, voxel: float, dims: tuple[int, int, int], truncation: float, device: str,
                 *, attributes: bool = False):
        import torch

        self.origin = np.asarray(origin, dtype=np.float64)
        self.voxel = float(voxel)
        self.dims = tuple(int(d) for d in dims)
        self.truncation = float(truncation)
        self.device = device
        self.tsdf = torch.ones(self.dims, dtype=torch.float32, device=device)
        self.weight = torch.zeros(self.dims, dtype=torch.float32, device=device)
        self.attributes = attributes
        if attributes:
            self.colour = torch.zeros((*self.dims, 3), dtype=torch.float16, device=device)
            self.colour_weight = torch.zeros(self.dims, dtype=torch.float16, device=device)
            # weighted Boyer-Moore majority vote per voxel
            self.label = torch.zeros(self.dims, dtype=torch.uint8, device=device)
            self.label_count = torch.zeros(self.dims, dtype=torch.float16, device=device)

    def centers(self, x0: int, x1: int):
        import torch

        nx, ny, nz = x1 - x0, self.dims[1], self.dims[2]
        ix = torch.arange(x0, x1, device=self.device, dtype=torch.float32)
        iy = torch.arange(ny, device=self.device, dtype=torch.float32)
        iz = torch.arange(nz, device=self.device, dtype=torch.float32)
        gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
        grid = torch.stack((gx, gy, gz), dim=-1).reshape(nx * ny * nz, 3)
        origin = torch.as_tensor(self.origin, dtype=torch.float32, device=self.device)
        return origin + (grid + 0.5) * self.voxel

    def box_centers(self, lo: tuple[int, int, int], hi: tuple[int, int, int]):
        """World centres and flat grid indices of the voxels in [lo, hi)."""
        import torch

        axes = [torch.arange(lo[k], hi[k], device=self.device) for k in range(3)]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        flat = ((gx * self.dims[1] + gy) * self.dims[2] + gz).reshape(-1)
        grid = torch.stack((gx, gy, gz), dim=-1).reshape(-1, 3).to(torch.float32)
        origin = torch.as_tensor(self.origin, dtype=torch.float32, device=self.device)
        return origin + (grid + 0.5) * self.voxel, flat

    def frustum_box(self, view: View, reach: float) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
        """Voxel index box around everything the view can see up to ``reach`` (camera + image corners)."""
        corners = [np.zeros(3)]
        for u in (0.0, float(view.width)):
            for v in (0.0, float(view.height)):
                corners.append(np.array([(u - view.cx) / view.fx, (v - view.cy) / view.fy, 1.0]) * reach)
        world = (np.asarray(corners) - np.asarray(view.translation)) @ np.asarray(view.rotation)
        lo = np.floor((world.min(axis=0) - self.origin) / self.voxel).astype(int) - 1
        hi = np.ceil((world.max(axis=0) - self.origin) / self.voxel).astype(int) + 2
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(self.dims))
        if np.any(hi <= lo):
            return None
        return tuple(int(x) for x in lo), tuple(int(x) for x in hi)

    def integrate(self, view: View, depth, weight, *, image=None, labels=None, slab: int = 48) -> None:
        """Fuse one view: depth/weight maps (torch or numpy, 0 = no depth), optional image (h,w,3) and label map."""
        import torch

        d = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        wt = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
        h, w = d.shape
        img = None if image is None else torch.as_tensor(image, device=self.device)
        lab = None if labels is None else torch.as_tensor(labels, device=self.device)
        rotation = torch.as_tensor(view.rotation, dtype=torch.float32, device=self.device)
        translation = torch.as_tensor(view.translation, dtype=torch.float32, device=self.device)
        reach = float(d.max()) + self.truncation if d.numel() else 0.0
        if reach <= self.truncation:
            return
        box = self.frustum_box(view, reach)  # only voxels the camera can see, not the whole grid
        if box is None:
            return
        (lo_x, lo_y, lo_z), (hi_x, hi_y, hi_z) = box
        near = 0.5 * self.truncation
        flat_tsdf = self.tsdf.view(-1)
        flat_weight = self.weight.view(-1)
        for x0 in range(lo_x, hi_x, slab):
            x1 = min(hi_x, x0 + slab)
            world, grid_index = self.box_centers((x0, lo_y, lo_z), (x1, hi_y, hi_z))
            local = world @ rotation.T + translation
            z = local[:, 2]
            front = (z > 0.05) & (z < reach)
            if not bool(front.any()):
                continue
            index = torch.nonzero(front, as_tuple=False).squeeze(1)
            zl = z[index]
            u = (view.fx * local[index, 0] / zl + view.cx) * (w / view.width)
            v = (view.fy * local[index, 1] / zl + view.cy) * (h / view.height)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            index, zl, u, v = index[inside], zl[inside], u[inside], v[inside]
            if not len(index):
                continue
            ui, vi = u.long(), v.long()
            dd = d[vi, ui]
            ww = wt[vi, ui]
            sdf = dd - zl
            valid = (dd > 0) & (ww > 0) & (sdf > -self.truncation)
            if not bool(valid.any()):
                continue
            index, sdf, ww, ui, vi, zl = index[valid], sdf[valid], ww[valid], ui[valid], vi[valid], zl[valid]
            value = torch.clamp(sdf / self.truncation, max=1.0)
            index = grid_index[index]
            old_w = flat_weight[index]
            new_w = old_w + ww
            flat_tsdf[index] = (flat_tsdf[index] * old_w + value * ww) / new_w
            flat_weight[index] = new_w
            if not self.attributes:
                continue
            at = sdf.abs() < near
            if not bool(at.any()):
                continue
            idx, aw = index[at], ww[at]
            if img is not None:
                iu = (ui[at].float() * img.shape[1] / w).long().clamp(0, img.shape[1] - 1)
                iv = (vi[at].float() * img.shape[0] / h).long().clamp(0, img.shape[0] - 1)
                rgb = img[iv, iu].float()
                flat_colour = self.colour.view(-1, 3)
                flat_cw = self.colour_weight.view(-1)
                cw_old = flat_cw[idx].float()
                cw_new = (cw_old + aw).clamp(max=60000)
                flat_colour[idx] = ((flat_colour[idx].float() * cw_old[:, None] + rgb * aw[:, None]) / cw_new[:, None]).half()
                flat_cw[idx] = cw_new.half()
            if lab is not None:
                lu = (ui[at].float() * lab.shape[1] / w).long().clamp(0, lab.shape[1] - 1)
                lv = (vi[at].float() * lab.shape[0] / h).long().clamp(0, lab.shape[0] - 1)
                cls = lab[lv, lu].to(torch.uint8)
                flat_label = self.label.view(-1)
                flat_count = self.label_count.view(-1)
                cand = flat_label[idx]
                cnt = flat_count[idx].float()
                same = cand == cls
                cnt_same = cnt + aw
                replace = ~same & (cnt < aw)
                cnt_other = torch.where(replace, aw - cnt, cnt - aw)
                flat_label[idx] = torch.where(same | ~replace, cand, cls)
                flat_count[idx] = torch.where(same, cnt_same, cnt_other).clamp(max=60000).half()

    def sample(self, points: np.ndarray):
        """Trilinear (tsdf, weight, gradient) at world points (numpy, float64)."""
        import torch
        import torch.nn.functional as F

        p = torch.as_tensor((points - self.origin) / self.voxel - 0.5, dtype=torch.float32, device=self.device)
        dims = torch.as_tensor(self.dims, dtype=torch.float32, device=self.device)

        def grid_values(volume, q):
            norm = 2.0 * q / (dims - 1).clamp(min=1) - 1.0
            g = norm[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3)
            return F.grid_sample(volume[None, None], g, mode="bilinear", padding_mode="border", align_corners=True).reshape(-1)

        tsdf = grid_values(self.tsdf, p)
        weight = grid_values(self.weight, p)
        grad = torch.zeros((len(p), 3), dtype=torch.float32, device=self.device)
        for axis in range(3):
            step = torch.zeros(3, device=self.device)
            step[axis] = 1.0
            grad[:, axis] = (grid_values(self.tsdf, p + step) - grid_values(self.tsdf, p - step)) / 2.0
        outside = ((p < 0) | (p > dims - 1)).any(dim=1)
        weight[outside] = 0.0
        return tsdf, weight, grad

    def extract(self, min_weight: float):
        """One point per voxel edge where the field crosses zero between two observed voxels.

        Returns numpy (xyz float64, rgb uint8 or None, label uint8 or None).
        """
        import torch

        xyz_blocks, rgb_blocks, lab_blocks = [], [], []
        origin = torch.as_tensor(self.origin, dtype=torch.float64, device=self.device)
        for axis in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(0, -1)
            b[axis] = slice(1, None)
            va, vb = self.tsdf[tuple(a)], self.tsdf[tuple(b)]
            wa, wb = self.weight[tuple(a)], self.weight[tuple(b)]
            crossing = (wa >= min_weight) & (wb >= min_weight) & (va.abs() < 0.999) & (vb.abs() < 0.999)
            crossing &= (va > 0) != (vb > 0)
            index = torch.nonzero(crossing, as_tuple=False)
            if not len(index):
                continue
            fa = va[crossing].double()
            fb = vb[crossing].double()
            t = (fa / (fa - fb)).clamp(0, 1)
            pos = index.double() + 0.5
            pos[:, axis] += t
            xyz_blocks.append((origin + pos * self.voxel).cpu().numpy())
            if self.attributes:
                ia = tuple(index.T)
                ib = list(index.T)
                ib[axis] = ib[axis] + 1
                ib = tuple(ib)
                ca, cb = self.colour[ia].float(), self.colour[ib].float()
                wca, wcb = self.colour_weight[ia].float(), self.colour_weight[ib].float()
                rgb = (ca * wca[:, None] + cb * wcb[:, None]) / (wca + wcb).clamp(min=1e-6)[:, None]
                rgb_blocks.append(rgb.clamp(0, 255).to(torch.uint8).cpu().numpy())
                closer_a = va[crossing].abs() <= vb[crossing].abs()
                lab_blocks.append(torch.where(closer_a, self.label[ia], self.label[ib]).cpu().numpy())
        if not xyz_blocks:
            return np.empty((0, 3)), None, None
        xyz = np.concatenate(xyz_blocks)
        rgb = np.concatenate(rgb_blocks) if rgb_blocks else None
        lab = np.concatenate(lab_blocks) if lab_blocks else None
        return xyz, rgb, lab


def grid_for_points(points: np.ndarray, settings: dict[str, Any]) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    q = float(settings["bounds_percentile"])
    lo = np.percentile(points, q, axis=0) - 3 * float(settings["truncation_m"])
    hi = np.percentile(points, 100 - q, axis=0) + 3 * float(settings["truncation_m"])
    voxel = float(settings["voxel_m"])
    while True:
        dims = tuple(int(math.ceil(s / voxel)) + 1 for s in (hi - lo))
        if int(np.prod(dims)) <= int(settings["max_voxels"]):
            return lo, voxel, dims
        voxel *= 1.25


def fuse_views(
    views: list[View],
    stereo_dir: Path,
    prior_reader: Callable[[str], np.ndarray | None] | None,
    tsdf: Tsdf,
    settings: dict[str, Any],
    *,
    label_reader: Callable[[str], np.ndarray | None] | None = None,
    image_reader: Callable[[str], np.ndarray | None] | None = None,
    excluded: tuple[int, ...] = (),
    mirror_masks: dict[str, np.ndarray] | None = None,
    mirror_planes: dict[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
    depth_filter_report: dict[str, int] | None = None,
) -> int:
    """Integrate every view; disk reads run one view ahead in a background thread."""
    from concurrent.futures import ThreadPoolExecutor

    def load(view: View):
        path = Path(stereo_dir) / "depth_maps" / f"{view.name}.geometric.bin"
        geometric = read_colmap_array(path) if path.is_file() else None
        if geometric is None:
            return None
        prior = prior_reader(view.name) if prior_reader is not None else None
        labels = label_reader(view.name) if label_reader is not None else None
        image = image_reader(view.name) if image_reader is not None and tsdf.attributes else None
        return geometric, prior, labels, image

    used = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(load, views[0]) if views else None
        for number, view in enumerate(views, start=1):
            if progress is not None:
                progress(number, len(views))
            loaded = pending.result() if pending is not None else None
            pending = pool.submit(load, views[number]) if number < len(views) else None
            if loaded is None:
                continue
            geometric, prior, labels, image = loaded
            combined = view_depth(view, geometric, prior, labels, excluded, settings, device=tsdf.device,
                                  mirror_mask=(mirror_masks or {}).get(view.name),
                                  mirror_plane=(mirror_planes or {}).get(view.name), edge_report=depth_filter_report)
            if combined is None:
                continue
            tsdf.integrate(view, *combined, image=image,
                           labels=None if labels is None or not tsdf.attributes else labels.astype(np.uint8))
            used += 1
    return used


# ----------------------------------------------------------------------------- snap mode


def snap_points(points: np.ndarray, tsdf: Tsdf, settings: dict[str, Any], chunk: int = 1_000_000
                ) -> tuple[np.ndarray, np.ndarray, dict[str, int], np.ndarray]:
    """Move points onto the fused surface; returns (new points, keep mask, stats, unit normals towards the cameras)."""
    import torch

    out = np.array(points, dtype=np.float64, copy=True)
    keep = np.ones(len(points), dtype=bool)
    normals = np.zeros((len(points), 3), dtype=np.float32)
    min_weight = float(settings["min_weight"])
    stats = {"snapped": 0, "unsupported": 0, "free_space": 0, "behind_surface": 0, "not_converged": 0,
             "residual_sum_cm": 0.0}
    for start in range(0, len(points), chunk):
        part = out[start:start + chunk]
        value, weight, _ = tsdf.sample(part)
        supported = (weight >= min_weight).cpu().numpy()
        value_np = value.cpu().numpy()
        free = supported & (value_np >= 0.999)
        behind = supported & (value_np <= -0.999)
        movable = supported & ~free & ~behind
        moved = part.copy()
        for _ in range(max(1, int(settings["iterations"]))):
            sel = np.flatnonzero(movable)
            if not len(sel):
                break
            v, _, g = tsdf.sample(moved[sel])
            # Newton step to the zero level: averaging views that disagree makes the
            # field flatter than 1/truncation, so step by value / |gradient|.
            slope = torch.linalg.norm(g, dim=1) / tsdf.voxel
            direction = (g / slope.clamp(min=1e-6)[:, None] / tsdf.voxel).cpu().numpy().astype(np.float64)
            distance = torch.clamp(v / slope.clamp(min=0.5 / tsdf.truncation), -tsdf.truncation, tsdf.truncation)
            moved[sel] -= direction * distance.cpu().numpy().astype(np.float64)[:, None]
        out[start:start + chunk] = np.where(movable[:, None], moved, part)
        if np.any(movable):
            residual, _, g = tsdf.sample(out[start:start + chunk][movable])
            # the field rises towards the cameras: its gradient is the side the surface faces
            normals[start + np.flatnonzero(movable)] = (g / torch.linalg.norm(g, dim=1, keepdim=True).clamp(min=1e-9)).cpu().numpy()
            slope = (torch.linalg.norm(g, dim=1) / tsdf.voxel).clamp(min=0.5 / tsdf.truncation)
            distance = (torch.abs(residual) / slope).cpu().numpy()
            stats["residual_sum_cm"] += float(distance.sum()) * 100
            stuck = np.zeros(len(part), dtype=bool)
            stuck[np.flatnonzero(movable)] = distance > float(settings["max_residual_m"])
            keep[start:start + chunk] &= ~stuck
            stats["not_converged"] += int(stuck.sum())
        if settings.get("drop_free_space", True):
            keep[start:start + chunk] &= ~free
        if settings.get("drop_behind_surface", True):
            keep[start:start + chunk] &= ~behind
        stats["snapped"] += int(movable.sum())
        stats["unsupported"] += int((~supported).sum())
        stats["free_space"] += int(free.sum())
        stats["behind_surface"] += int(behind.sum())
        del value, weight
    residual_sum = stats.pop("residual_sum_cm")
    stats["mean_residual_cm"] = round(residual_sum / max(stats["snapped"], 1), 3)
    return out, keep, stats, normals


def _window_evidence(depth, u, v, z, tolerance, config: dict[str, Any]):
    """Per point: (open, closed) from the COLMAP depth around its pixel (torch, ``depth`` 0 where invalid).

    A grid of ``window_grid`` x ``window_grid`` samples spans ``window_frac`` of the image either way.
    With at least ``min_window_valid`` measured samples the view is *closed* when ``closed_share`` of
    them lie clearly nearer than the point (a wall in front of it, seen over a whole neighbourhood, even
    where the wall itself is textureless at the point's own pixel) and *open* when ``open_share`` of them
    reach the point's depth or lie beyond it.
    """
    import torch

    h, w = depth.shape
    grid = max(2, int(config["window_grid"]))
    radius = float(config["window_frac"]) * min(h, w)
    offsets = torch.round(torch.linspace(-radius, radius, grid, device=depth.device)).long()
    dx, dy = torch.meshgrid(offsets, offsets, indexing="ij")
    dx, dy = dx.reshape(-1), dy.reshape(-1)
    uu = (u.long()[:, None] + dx[None, :]).clamp(0, w - 1)
    vv = (v.long()[:, None] + dy[None, :]).clamp(0, h - 1)
    sample = depth[vv, uu]
    valid = sample > 0
    measured = valid.sum(1)
    front = (valid & (sample < (z - tolerance)[:, None])).sum(1)
    enough = measured >= int(config["min_window_valid"])
    share = front.float() / measured.clamp(min=1).float()
    return enough & ((1.0 - share) >= float(config["open_share"])), enough & (share >= float(config["closed_share"]))


def unconfirmed_mask(evidence: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    """True for points no COLMAP depth confirms and many cameras only see through a nearer COLMAP surface."""
    confirmed = (evidence["support"] + evidence["through"] > 0) | (evidence["window_open"] > 0)
    return (~confirmed & (evidence["prior_open"] <= int(config["max_prior_views"]))
            & (evidence["window_closed"] >= int(config["min_closed_views"])))


def visibility_mask(points: np.ndarray, views: list[View], stereo_dir: Path, settings: dict[str, Any],
                    device: str = "cpu", progress: Callable[[int, int], None] | None = None,
                    prior_reader: Callable[[str], np.ndarray | None] | None = None, count_seen: bool = False,
                    evidence: dict[str, np.ndarray] | None = None,
                    mirror_mask_reader: Callable[[View, tuple[int, int]], np.ndarray | None] | None = None):
    """True for points no camera disproves: COLMAP depth maps only (see module docstring).

    ``mirror_mask_reader(view, shape)`` returns a boolean mask (or None): pixels inside it give no evidence,
    because a mirror pixel's depth is the virtual depth of the reflected scene.

    Returns (keep, stats); with ``count_seen`` (keep, stats, seen) where ``seen`` is
    the number of views per point that see it (COLMAP depth, plus the
    Depth-Anything prior where it agrees with COLMAP). With ``count_seen`` the
    ``hidden_filter`` also runs; ``evidence`` (a dict) receives the per-point counters it uses.
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor

    keep = np.ones(len(points), dtype=bool)
    if not len(points) or not views:
        empty = {"removed_visibility": 0}
        return (keep, empty, np.zeros(len(points), np.int32)) if count_seen else (keep, empty)
    p = torch.as_tensor(points, dtype=torch.float32, device=device)
    through = torch.zeros(len(p), dtype=torch.int32, device=device)
    support = torch.zeros(len(p), dtype=torch.int32, device=device)
    seen = torch.zeros(len(p), dtype=torch.int32, device=device)
    hidden = torch.zeros(len(p), dtype=torch.int32, device=device)
    prior_open = torch.zeros(len(p), dtype=torch.int32, device=device)
    window_open = torch.zeros(len(p), dtype=torch.int32, device=device)
    window_closed = torch.zeros(len(p), dtype=torch.int32, device=device)
    hidden_config = {**DEFAULTS["hidden_filter"], **(settings.get("hidden_filter") or {})}
    use_window = bool(count_seen and hidden_config.get("enabled", False))
    margin_abs = float(settings["visibility_margin_m"]) * float(settings.get("units_per_metre", 1.0))
    margin_rel = float(settings["visibility_margin_rel"])
    max_depth = float(settings["max_geometric_depth"])
    chunk = 262_144

    def load(view: View):
        path = Path(stereo_dir) / "depth_maps" / f"{view.name}.geometric.bin"
        if not path.is_file():
            return None
        prior = prior_reader(view.name) if count_seen and prior_reader is not None else None
        array = read_colmap_array(path)
        mask = mirror_mask_reader(view, array.shape[:2]) if mirror_mask_reader is not None else None
        return array, prior, mask

    def confirm(view: View, depth, weight, local) -> None:
        h, w = depth.shape
        z = local[:, 2]
        u = (view.fx * local[:, 0] / z + view.cx) * (w / view.width)
        v = (view.fy * local[:, 1] / z + view.cy) * (h / view.height)
        index = torch.nonzero((z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h), as_tuple=False).squeeze(1)
        if not len(index):
            return
        pixel_v, pixel_u = v[index].long(), u[index].long()
        dd = depth[pixel_v, pixel_u]
        zz = z[index]
        tolerance = margin_abs + margin_rel * zz
        seen[index[(dd > 0) & ((dd - zz).abs() <= tolerance)]] += 1
        hidden[index[(dd > 0) & (dd < zz - tolerance)]] += 1
        from_prior = weight[pixel_v, pixel_u] < float(settings["geometric_weight"])
        prior_open[index[(dd > 0) & from_prior & (dd >= zz - tolerance)]] += 1

    used = 0
    masked_views = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(load, views[0])
        for number, view in enumerate(views, start=1):
            if progress is not None:
                progress(number, len(views))
            loaded = pending.result()
            pending = pool.submit(load, views[number]) if number < len(views) else None
            if loaded is None:
                continue
            depth, prior, mask = loaded
            d = torch.as_tensor(np.asarray(depth, dtype=np.float32), device=device)
            from samcloud.colmap.depth_edges import geometric_valid_mask_torch

            d = torch.where(geometric_valid_mask_torch(d, view, settings) & (d <= max_depth), d, torch.zeros_like(d))
            if mask is not None:
                d = torch.where(torch.as_tensor(np.asarray(mask, dtype=bool), device=device), torch.zeros_like(d), d)
                masked_views += 1
            h, w = d.shape
            rotation = torch.as_tensor(view.rotation, dtype=torch.float32, device=device)
            translation = torch.as_tensor(view.translation, dtype=torch.float32, device=device)
            local = p @ rotation.T + translation
            if count_seen:
                combined = view_depth(view, depth, prior, None, (), settings, device=device, mirror_mask=mask)
                if combined is not None:
                    confirm(view, combined[0], combined[1], local)
            z = local[:, 2]
            u = (view.fx * local[:, 0] / z + view.cx) * (w / view.width)
            v = (view.fy * local[:, 1] / z + view.cy) * (h / view.height)
            index = torch.nonzero((z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h), as_tuple=False).squeeze(1)
            if not len(index):
                continue
            dd = d[v[index].long(), u[index].long()]
            zz = z[index]
            measured = dd > 0
            margin = margin_abs + margin_rel * zz
            through[index[measured & (dd > zz + margin)]] += 1
            support[index[measured & ((dd - zz).abs() <= margin)]] += 1
            if use_window:
                for start in range(0, len(index), chunk):
                    part = index[start:start + chunk]
                    is_open, is_closed = _window_evidence(d, u[part], v[part], z[part], margin_abs + margin_rel * z[part],
                                                          hidden_config)
                    window_open[part[is_open]] += 1
                    window_closed[part[is_closed]] += 1
            used += 1
    disproved = (through >= int(settings["visibility_min_through"])) & \
        (through.float() > float(settings["visibility_through_ratio"]) * support.float())
    keep = ~disproved.cpu().numpy()
    stats = {"removed_visibility": int((~keep).sum()), "visibility_views": used, "mirror_masked_views": masked_views}
    if count_seen:
        # never (or once) seen, but hidden behind a nearer surface in many views: behind a wall/ceiling
        unseen = (seen < int(settings["point_min_seen_views"])) & (hidden >= int(settings["point_hidden_views"]))
        stats["removed_hidden"] = int((unseen & ~disproved).sum())
        keep &= ~unseen.cpu().numpy()
        counters = {"support": support, "through": through, "prior_open": prior_open, "window_open": window_open,
                    "window_closed": window_closed, "seen": seen, "hidden": hidden}
        counters = {name: value.cpu().numpy() for name, value in counters.items()}
        if evidence is not None:
            evidence.update(counters)
        unconfirmed = unconfirmed_mask(counters, hidden_config) if use_window else np.zeros(len(keep), dtype=bool)
        stats["removed_unconfirmed"] = int((unconfirmed & keep).sum())
        keep &= ~unconfirmed
        return keep, stats, counters["seen"]
    return keep, stats


def _march_blocked(grid, origin, voxel: float, dims, points, camera, config: dict[str, Any]):
    """True where the segment point -> camera crosses a solid occupied run after leaving the point's own surface.

    ``grid`` is a dense boolean occupancy grid (torch). Marching starts at the point: occupied voxels that
    touch the first ``own_skip_voxels`` belong to the point's own surface and are skipped until the first empty
    sample; after that a run of ``run_samples`` occupied samples blocks the line of sight.
    """
    import torch

    steps = int(config["max_steps"])
    delta = camera - points
    length = delta.norm(dim=1).clamp(min=1e-3)
    direction = delta / length[:, None]
    step = torch.maximum(torch.full_like(length, 0.75 * voxel), length / steps)
    index = torch.arange(steps, device=points.device)[None, :]
    along = index * step[:, None]
    position = points[:, None, :] + direction[:, None, :] * along[..., None]
    cell = torch.floor((position - origin) / voxel).long()
    limit = torch.as_tensor(dims, device=points.device)
    inside = ((cell >= 0) & (cell < limit)).all(-1) & (along <= length[:, None])
    cell = torch.minimum(cell.clamp(min=0), limit - 1)
    occupied = grid[cell[..., 0], cell[..., 1], cell[..., 2]] & inside
    near = along <= float(config["own_skip_voxels"]) * voxel
    own_end = torch.where(occupied & near, index.expand_as(occupied), torch.full_like(index.expand_as(occupied), -1)).max(1).values
    free_after = (~occupied) & (index > own_end[:, None])
    first_free = torch.where(free_after.any(1), free_after.float().argmax(1), torch.full_like(own_end, steps))
    run = int(config["run_samples"])
    solid = occupied[:, : steps - run + 1].clone()
    for offset in range(1, run):
        solid &= occupied[:, offset: steps - run + 1 + offset]
    valid = (index[:, : steps - run + 1] >= first_free[:, None]) & \
        (along[:, : steps - run + 1] <= (length - float(config["own_skip_voxels"]) * voxel)[:, None])
    blocked = (solid & valid).any(1)
    bury = float(config.get("bury_depth_m", 0.0)) * float(config.get("units_per_metre", 1.0))
    if bury > 0:  # the point sits deep inside a solid mass: its own run of occupied samples is longer than a surface can be
        leading = occupied.long().cumprod(1).sum(1)
        blocked |= leading.float() * step > bury
    return blocked


def _local_floor_z(points: np.ndarray, floor_points: np.ndarray, query_xy: np.ndarray, xy0: np.ndarray, fcell: float) -> np.ndarray:
    """Local floor height at ``query_xy``: 10th-percentile z of the floor points per ``fcell`` cell, nearest known cell elsewhere."""
    from scipy import ndimage

    fshape = np.floor((points[:, :2].max(axis=0) - xy0) / fcell).astype(np.int64) + 1
    fidx = np.floor((floor_points[:, :2] - xy0) / fcell).astype(np.int64)
    flat = fidx[:, 0] * fshape[1] + fidx[:, 1]
    order = np.argsort(flat, kind="stable")
    flat_sorted, z_sorted = flat[order], floor_points[order, 2]
    cells, starts = np.unique(flat_sorted, return_index=True)
    ends = np.append(starts[1:], len(flat_sorted))
    floor_grid = np.full(tuple(int(v) for v in fshape), np.nan)
    known = np.zeros(tuple(int(v) for v in fshape), dtype=bool)
    for c, a, b in zip(cells, starts, ends):
        floor_grid[c // fshape[1], c % fshape[1]] = np.percentile(z_sorted[a:b], 10)
        known[c // fshape[1], c % fshape[1]] = True
    nearest = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    floor_grid = floor_grid[nearest[0], nearest[1]]
    q = np.floor((query_xy - xy0) / fcell).astype(np.int64)
    return floor_grid[q[:, 0].clip(0, fshape[0] - 1), q[:, 1].clip(0, fshape[1] - 1)]


def tall_furniture_members(points: np.ndarray, labels: np.ndarray, class_ids: dict[str, int], unit: float,
                           config: dict[str, Any]) -> np.ndarray:
    """Boolean mask of furniture points that belong to a tall, vertical body (built-in wardrobe, shelf wall, tall cabinet).

    Scale-free: every metric key is multiplied by ``unit`` (units per metre). The furniture points are binned into
    XY columns of ``tall_furniture_cell_m``; a column counts when its occupied height bins (>= ``tall_furniture_min_points``
    points each) fill at least ``tall_furniture_min_fill`` of the band from 0.15 m above the *local* floor up to
    ``tall_furniture_min_height_m`` and its top reaches that height. Requiring a filled band that starts near the floor
    keeps floor lamps, plants, coat stands (thin), cat trees (sparse) and wall-mounted upper cabinets (no body near the
    floor) out. The local floor is the 10th percentile z of the floor-class points per ``floor_cell_m`` cell, filled from the
    nearest cell that has floor points. A connected component (8-neighbourhood) of such columns must span at least
    ``tall_furniture_min_extent_m`` (its largest bounding-box side) to be an occluder.
    """
    from scipy import ndimage

    members = np.zeros(len(points), dtype=bool)
    ids = [class_ids[name] for name in config.get("tall_furniture_classes", []) if name in class_ids]
    floor_id = class_ids.get("floor")
    if not ids or floor_id is None or not config.get("tall_furniture_enabled", False):
        return members
    fur = np.flatnonzero(np.isin(labels, ids))
    floor_points = points[labels == floor_id]
    if len(fur) < 10 or len(floor_points) < 10:
        return members
    cell = float(config["tall_furniture_cell_m"]) * unit
    xy0 = points[:, :2].min(axis=0) - cell
    col = np.floor((points[fur, :2] - xy0) / cell).astype(np.int64)
    shape = col.max(axis=0) + 1
    floor_z = _local_floor_z(points, floor_points, points[fur, :2], xy0, float(config["floor_cell_m"]) * unit)
    height = points[fur, 2] - floor_z
    tall = float(config["tall_furniture_min_height_m"]) * unit
    band_lo = 0.15 * unit
    bin_size = 0.08 * unit
    zbin = np.floor((height - band_lo) / bin_size).astype(np.int64)
    n_bins = max(1, int(np.floor((tall - band_lo) / bin_size)))
    key = (col[:, 0] * shape[1] + col[:, 1]) * 4096 + np.clip(zbin, -1, 2000) + 1
    keys, counts = np.unique(key, return_counts=True)
    keys = keys[counts >= int(config["tall_furniture_min_points"])]
    zb, column_key = keys % 4096 - 1, keys // 4096
    n_cols = int(shape[0] * shape[1])
    in_band = (zb >= 0) & (zb < n_bins)
    fill = np.bincount(column_key[in_band], minlength=n_cols)[:n_cols] / n_bins
    top = np.full(n_cols, -np.inf)
    np.maximum.at(top, column_key, band_lo + (zb + 1) * bin_size)
    good = (fill >= float(config["tall_furniture_min_fill"])) & (top >= tall - bin_size)
    mask2d = good.reshape(tuple(int(v) for v in shape))
    comp, count = ndimage.label(mask2d, structure=np.ones((3, 3)))
    if not count:
        return members
    extent_min = float(config["tall_furniture_min_extent_m"]) * unit
    valid = np.zeros(count + 1, dtype=bool)
    for number, sl in enumerate(ndimage.find_objects(comp), start=1):
        valid[number] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * cell >= extent_min
    label_per_point = comp[col[:, 0], col[:, 1]]
    members[fur[valid[label_per_point]]] = True
    return members


def closed_body_columns(points: np.ndarray, labels: np.ndarray, class_ids: dict[str, int], solid: np.ndarray, unit: float,
                        config: dict[str, Any]):
    """Vertical bodies (walls, built-ins) with holes: columns that only a *gap* in the surface separates are solid too.

    White textureless wall or glossy panels leave holes in an otherwise tall, vertical sheet of points; rays slip through
    the holes to points on the far side. Columns of 0.08 m (``tall_furniture_cell_m``) whose solid points (any class but
    floor, ceiling and window) fill >= ``body_fill`` of the band from 0.15 m above the local floor to 1.6 m are body
    columns. A binary closing along x and along y with ``close_gap_m`` fills gaps narrower than that between body columns.
    Returns (xy cell centres, floor z) of the *added* columns only; empty arrays if nothing was added.
    """
    from scipy import ndimage

    empty = (np.zeros((0, 2)), np.zeros(0))
    floor_id = class_ids.get("floor")
    if not config.get("close_gaps_enabled", False) or floor_id is None:
        return empty
    skip = [class_ids[n] for n in ("floor", "ceiling", "window") if n in class_ids]
    use = np.flatnonzero(solid & ~np.isin(labels, skip))
    floor_points = points[labels == floor_id]
    if len(use) < 10 or len(floor_points) < 10:
        return empty
    cell = float(config["tall_furniture_cell_m"]) * unit
    xy0 = points[:, :2].min(axis=0) - cell
    col = np.floor((points[use, :2] - xy0) / cell).astype(np.int64)
    shape = np.floor((points[:, :2].max(axis=0) - xy0) / cell).astype(np.int64) + 2
    floor_z = _local_floor_z(points, floor_points, points[use, :2], xy0, float(config["floor_cell_m"]) * unit)
    band_lo, bin_size = 0.15 * unit, 0.08 * unit
    tall = float(config["tall_furniture_min_height_m"]) * unit
    n_bins = max(1, int(np.floor((tall - band_lo) / bin_size)))
    zbin = np.floor((points[use, 2] - floor_z - band_lo) / bin_size).astype(np.int64)
    inside = (zbin >= 0) & (zbin < n_bins)
    key = (col[inside, 0] * shape[1] + col[inside, 1]) * 4096 + zbin[inside]
    keys, counts = np.unique(key, return_counts=True)
    keys = keys[counts >= int(config["tall_furniture_min_points"])]
    n_cols = int(shape[0] * shape[1])
    fill = np.bincount(keys // 4096, minlength=n_cols)[:n_cols] / n_bins
    body = (fill >= float(config["body_fill"])).reshape(tuple(int(v) for v in shape))
    length = int(round(float(config["close_gap_m"]) * unit / cell)) + 1
    closed = ndimage.binary_closing(np.pad(body, length), structure=np.ones((1, length), bool))[length:-length, length:-length]
    closed |= ndimage.binary_closing(np.pad(body, length), structure=np.ones((length, 1), bool))[length:-length, length:-length]
    added = np.argwhere(closed if config.get("fill_body_columns", True) else closed & ~body)
    if not len(added):
        return empty
    centres = xy0 + (added + 0.5) * cell
    z = _local_floor_z(points, floor_points, centres, xy0, float(config["floor_cell_m"]) * unit)
    return centres, z


def raymarch_mask(points: np.ndarray, labels: np.ndarray, class_ids: dict[str, int], views: list[View], stereo_dir: Path,
                  settings: dict[str, Any], evidence: dict[str, np.ndarray], device: str = "cpu",
                  prior_reader: Callable[[str], np.ndarray | None] | None = None):
    """Remove points whose every sampled supporting view looks through a solid surface (``raymarch_filter``).

    Occupancy is built from the cloud itself: points of the ``structure_classes`` (walls, doors, ceiling, floor)
    plus points that at least ``occ_min_support`` COLMAP depth maps confirm, dilated by ``dilate_voxels`` so
    walls are watertight. A view *supports* a point when COLMAP (or the agreeing prior) sees it there; the
    segment view centre -> point is marched through the grid. A point goes when it has no clear supporting view
    and at least ``min_false`` blocked ones. Only points with fewer than ``occ_min_support`` COLMAP views
    are tested; at most ``max_views`` supporting views per point are sampled.
    Returns (remove, stats).
    """
    import torch

    config = {**DEFAULTS["raymarch_filter"], **(settings.get("raymarch_filter") or {})}
    config["units_per_metre"] = float(settings.get("units_per_metre", 1.0))
    stats = {"raymarch_candidates": 0, "raymarch_removed": 0}
    remove = np.zeros(len(points), dtype=bool)
    if not config.get("enabled", False) or not len(points) or not views:
        return remove, stats
    unit = float(settings.get("units_per_metre", 1.0))
    voxel = float(config["voxel_m"]) * unit
    structure = [class_ids[name] for name in config["structure_classes"] if name in class_ids]
    solid_points = (evidence["support"] >= int(config["occ_min_support"])) | np.isin(labels, structure)
    tall = tall_furniture_members(points, labels, class_ids, unit, config)
    stats["raymarch_tall_furniture_points"] = int(tall.sum())
    solid_points |= tall
    low = points.min(axis=0) - 0.5 * unit
    dims = np.ceil((points.max(axis=0) + 0.5 * unit - low) / voxel).astype(np.int64)
    while int(np.prod(dims)) > int(config["max_voxels"]):
        voxel *= 1.5
        dims = np.ceil((points.max(axis=0) + 0.5 * unit - low) / voxel).astype(np.int64)
    grid = torch.zeros(tuple(int(d) for d in dims), dtype=torch.bool, device=device)
    cells = torch.as_tensor(np.floor((points[solid_points] - low) / voxel).astype(np.int64), device=device)
    grid[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    gap_xy, gap_floor = closed_body_columns(points, labels, class_ids, solid_points, unit, config)
    stats["raymarch_closed_columns"] = int(len(gap_xy))
    if len(gap_xy):
        steps_z = np.arange(0.0, float(config["tall_furniture_min_height_m"]) * unit, 0.5 * voxel)
        zz = gap_floor[:, None] + 0.1 * unit + steps_z[None, :]
        fill = np.floor((np.dstack((np.repeat(gap_xy[:, None, 0], len(steps_z), 1), np.repeat(gap_xy[:, None, 1], len(steps_z), 1), zz))
                         .reshape(-1, 3) - low) / voxel).astype(np.int64)
        fill = fill[((fill >= 0) & (fill < np.asarray(dims))).all(1)]
        grid[torch.as_tensor(fill[:, 0], device=device), torch.as_tensor(fill[:, 1], device=device),
             torch.as_tensor(fill[:, 2], device=device)] = True
    radius = int(config["dilate_voxels"])
    if radius > 0:
        grid = torch.nn.functional.max_pool3d(grid[None, None].float(), 2 * radius + 1, 1, radius)[0, 0] > 0.5
    candidate = (evidence["support"] < int(config["occ_min_support"])) & ~tall  # an occluder never removes itself
    rows = np.flatnonzero(candidate)
    stats["raymarch_candidates"] = int(len(rows))
    if not len(rows):
        return remove, stats
    total = (evidence["seen"][rows] + evidence["prior_open"][rows]).astype(np.float64)
    quota = torch.as_tensor(np.minimum(1.0, int(config["max_views"]) / np.maximum(total, 1.0)), dtype=torch.float32, device=device)
    pts = torch.as_tensor(points[rows], dtype=torch.float32, device=device)
    row_id = torch.arange(len(rows), device=device, dtype=torch.int64)
    origin = torch.as_tensor(low, dtype=torch.float32, device=device)
    clear = torch.zeros(len(rows), dtype=torch.int32, device=device)
    blocked = torch.zeros(len(rows), dtype=torch.int32, device=device)
    margin_abs = float(settings["visibility_margin_m"]) * unit
    margin_rel = float(settings["visibility_margin_rel"])
    chunk = int(config["chunk"])
    for number, view in enumerate(views):
        path = Path(stereo_dir) / "depth_maps" / f"{view.name}.geometric.bin"
        if not path.is_file():
            continue
        depth_np = read_colmap_array(path)
        prior = prior_reader(view.name) if prior_reader is not None else None
        combined = view_depth(view, depth_np, prior, None, (), settings, device=device)
        if combined is None:
            continue
        depth, weight = combined
        h, w = depth.shape
        rotation = torch.as_tensor(view.rotation, dtype=torch.float32, device=device)
        translation = torch.as_tensor(view.translation, dtype=torch.float32, device=device)
        local = pts @ rotation.T + translation
        z = local[:, 2]
        u = (view.fx * local[:, 0] / z + view.cx) * (w / view.width)
        v = (view.fy * local[:, 1] / z + view.cy) * (h / view.height)
        index = torch.nonzero((z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h), as_tuple=False).squeeze(1)
        if not len(index):
            continue
        pixel_v, pixel_u = v[index].long(), u[index].long()
        measured = depth[pixel_v, pixel_u]
        zz = z[index]
        tolerance = margin_abs + margin_rel * zz
        from_prior = weight[pixel_v, pixel_u] < float(settings["geometric_weight"])
        supports = (measured > 0) & (((measured - zz).abs() <= tolerance) | (from_prior & (measured >= zz - tolerance)))
        hashed = ((row_id[index] * 2654435761 + (number + 1) * 40503) % 65536).float() / 65536.0
        index = index[supports & (hashed < quota[index])]
        camera = torch.as_tensor(view.center, dtype=torch.float32, device=device)
        for start in range(0, len(index), chunk):
            part = index[start:start + chunk]
            stopped = _march_blocked(grid, origin, voxel, dims, pts[part], camera[None, :].expand(len(part), 3), config)
            blocked[part[stopped]] += 1
            clear[part[~stopped]] += 1
    gone = ((clear == 0) & (blocked >= int(config["min_false"]))).cpu().numpy()
    remove[rows[gone]] = True
    stats["raymarch_removed"] = int(remove.sum())
    return remove, stats


def unseen_wall_mask(plane_index: np.ndarray, seen: np.ndarray, min_seen: float) -> tuple[np.ndarray, list[int]]:
    """False for points of wall planes that (median) fewer than ``min_seen`` views see: walls behind walls."""
    keep = np.ones(len(plane_index), dtype=bool)
    removed: list[int] = []
    for plane in np.unique(plane_index[plane_index >= 0]):
        rows = plane_index == plane
        if float(np.median(seen[rows])) < min_seen:
            keep[rows] = False
            removed.append(int(plane))
    return keep, removed


def class_coherence(points: np.ndarray, labels: np.ndarray, class_ids: dict[str, int], settings: dict[str, Any],
                    device: str = "cpu") -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Make the large classes spatially consistent; returns (labels, keep mask, stats)."""
    import torch
    import torch.nn.functional as F

    labels = np.asarray(labels).copy()
    never_classified = labels == 255
    labels[never_classified] = 0
    keep = np.ones(len(points), dtype=bool)
    stats = {"relabelled": 0, "removed_fragments": 0}
    group = []
    for name in settings.get("class_coherence_classes") or []:
        if name == "unknown":
            group.append(0)
        elif name in class_ids:
            group.append(class_ids[name])
    present = [c for c in group if np.any(labels == c)]
    if len(points) < 2 or len(present) < 2:
        labels[never_classified] = 255
        return labels, keep, stats
    cell = float(settings["class_cell_m"])
    lo = points.min(axis=0)
    while True:
        cells = np.floor((points - lo) / cell).astype(np.int64)
        dims = cells.max(axis=0) + 1
        if int(np.prod(dims)) <= 60_000_000:
            break
        cell *= 1.5
    flat = torch.as_tensor((cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2], device=device)
    lab = torch.as_tensor(labels.astype(np.int64), device=device)
    k = max(1, int(settings["class_window_cells"])) | 1
    n = len(points)
    total = torch.zeros(n, device=device)
    own = torch.zeros(n, device=device)
    best = torch.zeros(n, device=device)
    best_class = torch.full((n,), -1, dtype=torch.int64, device=device)
    for c in present:
        grid = torch.zeros(int(np.prod(dims)), device=device)
        grid.index_add_(0, flat[lab == c], torch.ones(int((lab == c).sum()), device=device))
        window = F.avg_pool3d(grid.view(1, 1, *[int(x) for x in dims]), k, stride=1, padding=k // 2,
                              count_include_pad=True).view(-1) * (k ** 3)
        count = window[flat]
        total += count
        mine = lab == c
        own = torch.where(mine, count, own)
        better = ~mine & (count > best) & (c != 0)  # nothing becomes "unknown"
        best = torch.where(better, count, best)
        best_class = torch.where(better, torch.full_like(best_class, c), best_class)
    in_group = torch.isin(lab, torch.as_tensor(present, device=device))
    share = own / total.clamp(min=1)
    relabel = in_group & (share < float(settings["class_min_share"])) & \
        (best / total.clamp(min=1) >= float(settings["class_majority_share"])) & (best_class >= 0)
    fragment = in_group & (own < float(settings["class_min_points"])) & (best == 0) & (lab != 0)
    lab = torch.where(relabel, best_class, lab)
    labels = lab.cpu().numpy().astype(labels.dtype)
    labels[never_classified] = 255
    keep = ~fragment.cpu().numpy()
    stats["relabelled"] = int(relabel.sum())
    stats["removed_fragments"] = int((~keep).sum())
    return labels, keep, stats


def dedupe_first(points: np.ndarray, keep: np.ndarray, spacing: float) -> np.ndarray:
    """Indices of kept points, one per ``spacing`` voxel (lowest index wins)."""
    index = np.flatnonzero(keep)
    if spacing <= 0 or not len(index):
        return index
    cells = np.floor((points[index] - points[index].min(axis=0)) / spacing).astype(np.int64)
    span = cells.max(axis=0) + 1
    key = (cells[:, 0] * span[1] + cells[:, 1]) * span[2] + cells[:, 2]
    _, first = np.unique(key, return_index=True)
    return index[np.sort(first)]


# ----------------------------------------------------------------------------- planes and haze


def _horizontal_axes(normal) -> list:
    """The building's two main wall directions given one wall normal."""
    import torch

    n = normal.clone()
    n[2] = 0
    n = n / n.norm().clamp(min=1e-12)
    perpendicular = torch.stack((-n[1], n[0], torch.zeros_like(n[0])))
    return [n, perpendicular]


def dominant_wall_axes(normals: np.ndarray) -> np.ndarray | None:
    """Unit normal of the building's main wall direction from point normals (angle modulo 90 deg), or None."""
    n = np.asarray(normals, dtype=np.float64)
    flat = np.hypot(n[:, 0], n[:, 1])
    n = n[(flat > 0.9) & (np.abs(n[:, 2]) < 0.3)]
    if len(n) < 100:
        return None
    angle = np.mod(np.degrees(np.arctan2(n[:, 1], n[:, 0])), 90.0)
    hist, edges = np.histogram(angle, bins=180, range=(0.0, 90.0))
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)  # 1.5 deg smoothing, circular
    peak = float(edges[int(np.argmax(hist))] + 0.25)
    offset = (angle - peak + 45.0) % 90.0 - 45.0
    near = np.abs(offset) <= 3.0
    if near.sum() < 50:
        return None
    theta = math.radians(peak + float(np.median(offset[near])))
    return np.array([math.cos(theta), math.sin(theta), 0.0])


def snap_to_planes(points: np.ndarray, labels: np.ndarray, class_ids: dict[str, int], settings: dict[str, Any],
                   device: str = "cpu", seed: int = 0, normals: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit horizontal planes to ceiling/floor and vertical planes to walls; put nearby points onto them.

    Greedy RANSAC per class: the plane with most inliers (``plane_inlier_m``)
    is accepted when it carries ``plane_min_points``; points of that class
    within ``plane_snap_m`` are projected onto it and removed from the search.
    Up to ``plane_max_planes`` planes per class.
    """
    import torch

    out = np.array(points, dtype=np.float64, copy=True)
    report: dict[str, Any] = {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=device)
    cos_limit = math.cos(math.radians(float(settings["plane_angle_deg"])))
    inlier = float(settings["plane_inlier_m"])
    snap = float(settings["plane_snap_m"])
    minimum = int(settings["plane_min_points"])
    plane_index = np.full(len(points), -1, dtype=np.int64)  # wall plane per point (-1: none)
    wall_ids: list[int] = []
    for name in settings.get("plane_classes") or []:
        cls = class_ids.get(name)
        if cls is None:
            continue
        member = np.flatnonzero(labels == cls)
        if len(member) < minimum:
            continue
        horizontal = name in ("ceiling", "floor")
        pts = torch.as_tensor(out[member], dtype=torch.float64, device=device)
        nrm = None if normals is None else torch.as_tensor(normals[member], dtype=torch.float64, device=device)
        free = torch.ones(len(member), dtype=torch.bool, device=device)
        planes, snapped = [], 0
        failures = 0
        axes = None  # main wall directions: from the point normals, else fixed by the first (largest) wall
        manhattan = (not horizontal) and bool(settings.get("plane_manhattan", False))
        if manhattan and nrm is not None:
            main_axis = dominant_wall_axes(normals[member])
            if main_axis is not None:
                axes = _horizontal_axes(torch.as_tensor(main_axis, dtype=torch.float64, device=device))
        axis_cos = math.cos(math.radians(float(settings.get("plane_manhattan_deg", 4.0))))

        def on_axis(normal_rows):
            if axes is None:
                return torch.ones(len(normal_rows), dtype=torch.bool, device=device)
            flat = normal_rows[:, :2] / normal_rows[:, :2].norm(dim=1, keepdim=True).clamp(min=1e-12)
            return ((flat @ axes[0][:2]).abs() >= axis_cos) | ((flat @ axes[1][:2]).abs() >= axis_cos)

        for _ in range(int(settings["plane_max_planes"]) * 3):
            if len(planes) >= int(settings["plane_max_planes"]) or failures >= int(settings.get("plane_max_failures", 15)):
                break
            candidates = torch.nonzero(free, as_tuple=False).squeeze(1)
            if len(candidates) < minimum:
                break
            # local hypotheses: two partners within 0.6 m of a random seed point, so the
            # three points usually lie on the same wall segment of a whole apartment
            hypotheses = int(settings["plane_hypotheses"])
            seeds = candidates[torch.randint(len(candidates), (hypotheses,), generator=generator).to(device)]
            sub = candidates[torch.randperm(len(candidates), generator=generator)[:20_000].to(device)]
            close = torch.cdist(pts[seeds], pts[sub]) <= 0.6
            weights = close.double() + 1e-9
            partners = sub[torch.multinomial(weights.cpu(), 2, replacement=True, generator=generator).to(device)]
            a, b, c = pts[seeds], pts[partners[:, 0]], pts[partners[:, 1]]
            normal = torch.cross(b - a, c - a, dim=1)
            normal = normal / normal.norm(dim=1, keepdim=True).clamp(min=1e-12)
            vertical_part = (normal @ up).abs()
            ok = vertical_part >= cos_limit if horizontal else vertical_part <= math.sin(math.radians(float(settings["plane_angle_deg"])))
            if manhattan:
                ok &= on_axis(normal)
            if not bool(ok.any()):
                failures += 1
                continue
            normal, a = normal[ok], a[ok]
            offset = -(normal * a).sum(dim=1)
            sample = candidates[torch.randperm(len(candidates), generator=generator)[:200_000].to(device)]
            dist = (pts[sample] @ normal.T + offset[None, :]).abs()
            counts = (dist <= inlier).sum(dim=0)
            best = int(torch.argmax(counts))
            n, d0 = normal[best], offset[best]
            # refine: a few rounds of (inliers -> constrained least squares -> median offset);
            # the first round uses the wide snap band so a slightly rotated hypothesis recovers
            good = True
            for round_index in range(4):
                band = snap if round_index == 0 else inlier
                near = free & ((pts @ n + d0).abs() <= band)
                if int(near.sum()) < minimum:
                    good = False
                    break
                q = pts[near]
                if horizontal:  # ceilings and floors are exactly level (model is gravity aligned)
                    n = up.clone()
                else:  # walls exactly vertical: direction from the horizontal spread only
                    centroid = q.mean(dim=0)
                    _, _, vh = torch.linalg.svd((q - centroid)[:, :2], full_matrices=False)
                    n = torch.zeros(3, dtype=torch.float64, device=device)
                    n[:2] = vh[-1]
                    n = n / n.norm()
                    if axes is not None:  # exactly along the building's main direction
                        if bool(on_axis(n[None, :])[0]):
                            dots = torch.stack([n @ axes[0], n @ axes[1]])
                            pick = int(torch.argmax(dots.abs()))
                            n = axes[pick] * torch.sign(dots[pick])
                d0 = -(q @ n).median()  # centre of the band, robust to a second nearby layer
            if good and manhattan and not bool(on_axis(n[None, :])[0]):
                good = False  # refinement drifted off the building's main directions
            if not good:
                failures += 1
                continue
            distance = pts @ n + d0
            take = free & (distance.abs() <= snap)
            extent = None
            if not horizontal:
                # a wall is a segment, not an infinite line: snap only along its measured length
                tangent = torch.stack((-n[1], n[0], torch.zeros_like(n[0])))
                along = pts @ tangent
                core = along[free & (distance.abs() <= inlier)]
                margin = float(settings.get("plane_extent_margin_m", 0.25))
                lo_s = torch.quantile(core[torch.randperm(len(core), generator=generator)[:100_000].to(device)], 0.01) - margin
                hi_s = torch.quantile(core[torch.randperm(len(core), generator=generator)[:100_000].to(device)], 0.99) + margin
                take &= (along >= lo_s) & (along <= hi_s)
                extent = (float(lo_s), float(hi_s))
            if int(take.sum()) < minimum:
                failures += 1
                continue
            pts[take] -= distance[take][:, None] * n[None, :]
            free &= ~take
            snapped += int(take.sum())
            facing = 0.0
            if nrm is not None:
                facing = float(torch.sign((nrm[take] @ n).median()))
            if manhattan and axes is None:
                axes = _horizontal_axes(n)
            planes.append({"normal": [round(float(x), 4) for x in n.cpu()], "offset": round(float(d0), 4),
                           "points": int(take.sum()), "facing": facing, "extent": extent,
                           "_n": n.clone(), "_d": d0.clone(),
                           "_take": torch.nonzero(take, as_tuple=False).squeeze(1)})
        duplicates = 0
        if not horizontal and len(planes) > 1 and float(settings.get("plane_duplicate_m", 0.0)) > 0:
            # parallel copies of one wall seen from the same side (thick depth bands) -> onto the bigger one;
            # the two faces of an interior wall face opposite ways and stay
            dup_m = float(settings["plane_duplicate_m"])
            dup_cos = math.cos(math.radians(float(settings.get("plane_duplicate_deg", 10.0))))
            order = sorted(range(len(planes)), key=lambda k: -planes[k]["points"])
            for rank, k in enumerate(order[1:], start=1):
                small = planes[k]
                rows = small["_take"]
                q = pts[rows]
                for big_index in order[:rank]:
                    big = planes[big_index]
                    if big.get("merged_into") is not None:
                        continue
                    if abs(float(small["_n"] @ big["_n"])) < dup_cos:
                        continue
                    if small["facing"] and big["facing"] and \
                            small["facing"] * big["facing"] * float(torch.sign(small["_n"] @ big["_n"])) < 0:
                        continue
                    offset = q @ big["_n"] + big["_d"]
                    if float(offset.median().abs()) > dup_m:
                        continue
                    if big["extent"] is not None:
                        tangent = torch.stack((-big["_n"][1], big["_n"][0], torch.zeros_like(big["_n"][0])))
                        along = q @ tangent
                        inside = (along >= big["extent"][0]) & (along <= big["extent"][1])
                        if float(inside.double().mean()) < 0.5:
                            continue
                    pts[rows] -= offset[:, None] * big["_n"][None, :]
                    small["merged_into"] = big_index
                    duplicates += small["points"]
                    break
        merged = 0
        merge = float((settings.get("plane_merge_m") or {}).get(name, 0.0)) if horizontal else 0.0
        if merge > 0 and len(planes) > 1:
            # one ceiling (floor) height: smaller levels within plane_merge_m of a bigger one are
            # put onto it - several heights of one textureless ceiling are depth errors
            order = sorted(range(len(planes)), key=lambda k: -planes[k]["points"])
            for rank, k in enumerate(order[1:], start=1):
                height = -planes[k]["offset"]
                for big in order[:rank]:
                    target = -planes[big]["offset"]
                    if planes[big].get("merged_into") is None and abs(height - target) <= merge:
                        pts[planes[k]["_take"], 2] = target
                        planes[k]["merged_into"] = big
                        merged += planes[k]["points"]
                        break
        out[member] = pts.cpu().numpy()
        if not horizontal:
            # which wall plane every point now lies on (copies count for the plane they were put onto)
            for k, plane in enumerate(planes):
                target = k
                while planes[target].get("merged_into") is not None:
                    target = planes[target]["merged_into"]
                plane_index[member[plane["_take"].cpu().numpy()]] = len(wall_ids) + target
            wall_ids.extend(range(len(planes)))
        report[name] = {"planes": len(planes), "snapped_points": snapped, "class_points": int(len(member)),
                        "merged_points": merged, "duplicate_points": duplicates,
                        "walls": [{k: v for k, v in pl.items() if not k.startswith("_")} for pl in planes]
                        if not horizontal else None,
                        "levels": sorted({round(-pl["offset"], 3) for pl in planes if pl.get("merged_into") is None})
                        if horizontal else None,
                        "level_points": _level_points(planes) if horizontal else None}
    report["_wall_index"] = plane_index
    return out, report


def _level_points(planes: list[dict[str, Any]]) -> dict[str, int]:
    """Points per remaining height (merged planes count for the level they were put onto)."""
    totals: dict[str, int] = {}
    for plane in planes:
        target = plane
        while target.get("merged_into") is not None:
            target = planes[target["merged_into"]]
        key = f"{-target['offset']:.3f}"
        totals[key] = totals.get(key, 0) + int(plane["points"])
    return totals


def levels_mask(points: np.ndarray, plane_report: dict[str, Any], settings: dict[str, Any]) -> np.ndarray:
    """With one floor and one ceiling height, nothing exists below the floor or above the ceiling.

    Such points (wall copies poking through the ceiling, layers behind it) are
    invisible from the inside and therefore removed. Skipped when a class has
    several levels (split-level rooms, more than one storey).
    """
    keep = np.ones(len(points), dtype=bool)
    if not settings.get("clip_to_levels", True):
        return keep
    share = float(settings.get("plane_level_min_share", 0.05))

    def real_levels(name: str) -> list[float]:
        entry = plane_report.get(name) or {}
        counts = entry.get("level_points") or {}
        levels = entry.get("levels") or []
        if not counts:
            return levels
        total = max(sum(counts.values()), 1)
        # a "level" carrying only a few percent of the class is stray points, not a split-level room
        return [lv for lv in levels if counts.get(f"{lv:.3f}", total) / total >= share]

    ceiling = real_levels("ceiling")
    floor = real_levels("floor")
    margin = float(settings["plane_snap_m"])
    if len(ceiling) == 1:
        keep &= points[:, 2] <= ceiling[0] + margin
    if len(floor) == 1:
        keep &= points[:, 2] >= floor[0] - margin
    return keep


def _voxel_neighbor_counts_torch(xyz: np.ndarray, cell: float, device: str) -> np.ndarray:
    """Same counts as ``sparse_filter._voxel_neighbor_counts``, computed with torch."""
    import torch

    p = torch.as_tensor(xyz, dtype=torch.float64, device=device)
    cells = torch.floor((p - p.min(dim=0).values) / cell).to(torch.int64)
    span = cells.max(dim=0).values + 3
    key = ((cells[:, 0] + 1) * span[1] + (cells[:, 1] + 1)) * span[2] + (cells[:, 2] + 1)
    unique, counts = torch.unique(key, return_counts=True)
    total = torch.zeros(len(p), dtype=torch.int64, device=device)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                probe = key + (dx * span[1] + dy) * span[2] + dz
                position = torch.searchsorted(unique, probe).clamp(max=len(unique) - 1)
                total += torch.where(unique[position] == probe, counts[position], torch.zeros_like(counts[position]))
    return (total - 1).cpu().numpy()


def sparse_point_mask(points: np.ndarray, settings: dict[str, Any], device: str | None = None) -> np.ndarray:
    """True for points to keep: at least ``min_neighbors`` others in the surrounding 3x3x3 cells."""
    from samcloud.viewer.sparse_filter import _voxel_neighbor_counts

    def counts_for(xyz: np.ndarray, cell: float) -> np.ndarray:
        if device is None or device == "numpy":
            return _voxel_neighbor_counts(xyz, cell)
        return _voxel_neighbor_counts_torch(xyz, cell, device)

    keep = np.ones(len(points), dtype=bool)
    minimum = int(settings.get("min_neighbors", 0))
    if minimum <= 0 or len(points) < 2:
        return keep
    for _ in range(max(1, int(settings.get("neighbor_passes", 1)))):
        index = np.flatnonzero(keep)
        if len(index) < 2:
            break
        counts = counts_for(points[index], float(settings["neighbor_cell_m"]))
        keep[index[counts < minimum]] = False
    return keep


# ----------------------------------------------------------------------------- entry point


def derive_units_per_metre(z: np.ndarray, labels: np.ndarray, class_ids: dict[str, int],
                           nominal_storey_height_m: float = 2.5) -> dict[str, Any]:
    """Local units per metre of an unscaled indoor project from the storey height (floor vs ceiling class).

    ``z`` are the point heights, ``labels`` their class ids. Median floor and ceiling heights give the
    storey height in local units; a standard storey (``nominal_storey_height_m``, default 2.5 m) turns it
    into a scale (the ortho plan uses the same approximation, ``ortho/plan.py _indoor_scale_and_bands``).
    Fallback 1.0 without both classes or when the result is implausible (outside 0.3..3).
    """
    fallback = {"units_per_metre": 1.0, "source": "fallback"}
    floor_id, ceiling_id = class_ids.get("floor"), class_ids.get("ceiling")
    if floor_id is None or ceiling_id is None or not len(z):
        return fallback
    floor = z[labels == floor_id]
    ceiling = z[labels == ceiling_id]
    if len(floor) < 1000 or len(ceiling) < 1000:
        return fallback
    floor_z = float(np.median(floor))
    above = ceiling[ceiling > floor_z]
    if len(above) < 1000:
        return fallback
    height = float(np.median(above)) - floor_z
    unit = height / float(nominal_storey_height_m)
    if not (0.3 <= unit <= 3.0):
        return fallback
    return {"units_per_metre": round(unit, 4), "source": "storey_height", "storey_height_local": round(height, 4),
            "nominal_storey_height_m": float(nominal_storey_height_m)}


def _class_ids(class_names: list[str] | None) -> dict[str, int]:
    short = [name.split(",", 1)[0].strip().lower() for name in (class_names or [])]
    return {name: index + 1 for index, name in enumerate(short)}


def _guarded_mirror_report(builder, mirrors, removed_by_mirror) -> dict[str, Any]:
    """Mirror numbers are report-only: a failure here must never discard the fused cloud."""
    try:
        return builder(mirrors, removed_by_mirror)
    except Exception as error:
        print(f"WARNING: Mirror report failed: {error}")
        return {"mirrors": [], "mirror_points_removed": int(np.asarray(removed_by_mirror).sum()),
                "mirror_report_error": str(error)}


def consolidate_classified_cloud(
    dense_dir: Path,
    classified_ply: Path,
    views: list[View],
    prior_reader: Callable[[str], np.ndarray | None] | None,
    settings: dict[str, Any] | None = None,
    *,
    scores_path: Path | None = None,
    labels_dir: Path | None = None,
    class_names: list[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    mirrors: list[Any] | None = None,
) -> dict[str, Any]:
    """Fuse all depth maps into a TSDF and rewrite the classified cloud as one surface layer (in place)."""
    started = time.perf_counter()
    import cv2
    import torch
    from plyfile import PlyData, PlyElement

    settings = {**DEFAULTS, **(settings or {})}
    dense_dir, classified_ply = Path(dense_dir), Path(classified_ply)
    if not bool(settings.get("enabled", True)):
        return {"enabled": False}
    stereo_dir = dense_dir / "stereo"
    if not (stereo_dir / "depth_maps").is_dir():
        return {"enabled": True, "reason": "depth maps missing"}
    vertex = read_vertex_array(classified_ply)
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    if not len(points):
        return {"enabled": True, "reason": "empty cloud"}
    input_point_count = len(points)
    source_indices = np.arange(len(points))
    mirror_settings = {
        **settings.get("mirror_filter", {}),
        "depth_edge_filter": settings.get("depth_edge_filter", {}),
        "units_per_metre": settings.get("units_per_metre", 1.0),
        "visibility_margin_m": settings.get("visibility_margin_m", 0.03),
        "visibility_margin_rel": settings.get("visibility_margin_rel", 0.01),
        "max_geometric_depth": settings.get("max_geometric_depth", np.inf),
    }
    supplied_mirrors = mirrors is not None
    mirrors = [] if mirrors is None else mirrors
    mirror_masks: dict[str, np.ndarray] = {}
    mirror_planes: dict[str, Any] = {}
    removed_by_mirror = np.zeros(len(points), dtype=bool)
    if bool(mirror_settings.get("enabled", True)):
        from samcloud.colmap.mirror_filter import detect_mirrors, mirror_report, phantom_mask

        if not supplied_mirrors:
            mirrors = detect_mirrors(views, stereo_dir, dense_dir / "sam_mirror", mirror_settings, labels_dir)
        for mirror in mirrors:
            for name, mask in mirror.masks.items():
                mirror_masks[name] = mask if name not in mirror_masks else (mirror_masks[name] | mask)
                mirror_planes.setdefault(name, mirror)
        if mirrors:
            removed_by_mirror = phantom_mask(points, mirrors, views, stereo_dir, mirror_settings,
                                             masks_dir=dense_dir / "sam_mirror")
            keep = ~removed_by_mirror
            vertex, points, source_indices = vertex[keep], points[keep], source_indices[keep]
    mirror_reader = None
    if mirrors and bool(mirror_settings.get("enabled", True)) and bool(mirror_settings.get("visibility_ignore_masks", True)):
        from samcloud.colmap.mirror_filter import mirror_mask_reader

        mirror_reader = mirror_mask_reader(dense_dir / "sam_mirror", int(mirror_settings.get("mirror_dilate_px", 6)))
    mode = str(settings.get("mode", "extract"))
    ids = _class_ids(class_names)
    excluded = tuple(ids[n.lower()] for n in settings.get("exclude_classes") or [] if n.lower() in ids)

    def read_labels(name: str) -> np.ndarray | None:
        if labels_dir is None:
            return None
        path = Path(labels_dir) / f"{name}.png"
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None

    def read_image(name: str) -> np.ndarray | None:
        image = cv2.imread(str(dense_dir / "images" / name), cv2.IMREAD_COLOR)
        return None if image is None else image[:, :, ::-1].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timings: dict[str, float] = {"read": round(time.perf_counter() - started, 2)}
    clock = [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter()
        timings[name] = round(timings.get(name, 0.0) + now - clock[0], 2)
        clock[0] = now

    origin, voxel, dims = grid_for_points(points, settings)
    tsdf = Tsdf(origin, voxel, dims, float(settings["truncation_m"]), device, attributes=(mode == "extract"))
    lap('setup')
    depth_filter_report: dict[str, int] = {}
    used = fuse_views(views, stereo_dir, prior_reader, tsdf, settings, label_reader=read_labels,
                      image_reader=read_image, excluded=excluded, mirror_masks=mirror_masks,
                      mirror_planes=mirror_planes, progress=progress, depth_filter_report=depth_filter_report)
    report: dict[str, Any] = {
        "enabled": True, "mode": mode,
        "settings": {key: settings[key] for key in DEFAULTS if key != "enabled"},
        "voxel_m": voxel, "grid": list(dims), "device": device, "views_fused": used,
        "excluded_label_ids": list(excluded), "original_points": int(input_point_count),
        "depth_edge_filter": {"dropped_pixels": depth_filter_report.get("dropped", 0),
                              "valid_pixels": depth_filter_report.get("valid", 0),
                              "dropped_share": depth_filter_report.get("dropped", 0) / max(depth_filter_report.get("valid", 0), 1)},
    }
    if mirrors:
        report.update(_guarded_mirror_report(mirror_report, mirrors, removed_by_mirror))
    else:
        report.update({"mirrors": [], "mirror_points_removed": 0})
    report["visibility_mirror_masks"] = mirror_reader is not None
    template = vertex[:0]
    if mode == "extract":
        xyz, rgb, lab = tsdf.extract(float(settings["min_weight"]))
        del tsdf
        if device == "cuda":
            torch.cuda.empty_cache()
        if rgb is None:
            rgb = np.full((len(xyz), 3), 200, np.uint8)
        if lab is None:
            lab = np.zeros(len(xyz), np.uint8)
        report["extracted_points"] = int(len(xyz))
        normals = None
        selected = np.arange(len(xyz))
    else:
        lap('tsdf_fusion')
        xyz, keep, stats, normals = snap_points(points, tsdf, settings)
        del tsdf
        if device == "cuda":
            torch.cuda.empty_cache()
        selected = source_indices[dedupe_first(xyz, keep, float(settings["dedupe_m"]))]
        report.update(stats)
        rgb = np.column_stack((vertex["red"], vertex["green"], vertex["blue"])) if "red" in vertex.dtype.names else None
        lab = np.asarray(vertex["scalar_class"]) if "scalar_class" in vertex.dtype.names else np.zeros(len(points), np.uint8)
        selected_local = np.searchsorted(source_indices, selected)
        xyz, lab, normals = xyz[selected_local], lab[selected_local], normals[selected_local]
        rgb = rgb[selected_local] if rgb is not None else np.full((len(selected), 3), 200, np.uint8)
    lap('snap')
    if settings.get("visibility_check", True):
        visible, vis_stats = visibility_mask(xyz, views, stereo_dir, settings, device=device,
                                             mirror_mask_reader=mirror_reader)
        report.update(vis_stats)
        xyz, rgb, lab, selected = xyz[visible], rgb[visible], lab[visible], selected[visible]
        normals = normals[visible] if normals is not None else None
    lap('visibility')
    wall_index = np.full(len(xyz), -1, dtype=np.int64)
    if settings.get("plane_classes"):
        xyz, report["planes"] = snap_to_planes(xyz, lab, ids, settings, device=device, normals=normals)
        wall_index = report["planes"].pop("_wall_index")
        inside = levels_mask(xyz, report["planes"], settings)
        report["removed_outside_levels"] = int((~inside).sum())
        xyz, rgb, lab, selected = xyz[inside], rgb[inside], lab[inside], selected[inside]
        wall_index = wall_index[inside]
    lap('planes')
    seen = None
    if settings.get("visibility_check", True) and settings.get("plane_classes"):
        # again at the final positions: a plane in front of the real wall is seen through as well,
        # and a plane behind it is seen by (almost) no camera at all
        evidence: dict[str, np.ndarray] = {}
        visible, vis_stats, seen = visibility_mask(xyz, views, stereo_dir, settings, device=device,
                                                   prior_reader=prior_reader, count_seen=True, evidence=evidence,
                                                   mirror_mask_reader=mirror_reader)
        report["removed_visibility_after_planes"] = vis_stats["removed_visibility"]
        report["removed_hidden"] = vis_stats["removed_hidden"]
        report["removed_unconfirmed"] = vis_stats["removed_unconfirmed"]
        seen_walls, unseen = unseen_wall_mask(wall_index, seen, float(settings["wall_min_seen_views"]))
        report["removed_unseen_walls"] = {"planes": unseen, "points": int((~seen_walls).sum())}
        visible &= seen_walls
        xyz, rgb, lab, selected = xyz[visible], rgb[visible], lab[visible], selected[visible]
        wall_index, seen = wall_index[visible], seen[visible]
        evidence = {name: values[visible] for name, values in evidence.items()}
        marched, march_stats = raymarch_mask(xyz, lab, ids, views, stereo_dir, settings, evidence, device=device,
                                             prior_reader=prior_reader)
        report.update(march_stats)
        if marched.any():
            keep_march = ~marched
            xyz, rgb, lab, selected = xyz[keep_march], rgb[keep_march], lab[keep_march], selected[keep_march]
            wall_index, seen = wall_index[keep_march], seen[keep_march]
    lap('visibility_after_planes')
    if settings.get("plane_classes"):
        from samcloud.colmap.behind_wall import behind_wall_mask

        fine_terms = None
        fine_labels = None
        if scores_path is not None:
            # Fine terms are snapshotted before step 6.  ``selected`` still maps
            # every surviving snap-mode point to that source cloud, so no nearest
            # neighbour transfer (and no semantic drift) is needed here.
            from samcloud.engine.fine_terms import SOURCE_NAME

            source_path = Path(scores_path).with_name(SOURCE_NAME)
            if mode != "extract" and source_path.is_file():
                with np.load(source_path) as archive:
                    source_terms = np.asarray(archive.get("fine_term", ()), dtype=np.uint8)
                    source_labels = archive.get("fine_labels")
                if len(source_terms) == input_point_count and source_labels is not None and len(selected):
                    fine_terms = source_terms[selected]
                    fine_labels = [str(value) for value in source_labels.tolist()]
        behind_keep, behind_stats = behind_wall_mask(xyz, lab, wall_index, report["planes"], views, stereo_dir, settings,
                                                      mirrors=mirrors, class_ids=ids, seen=seen, fine_terms=fine_terms,
                                                      fine_labels=fine_labels, device=device)
        report.update(behind_stats)
        xyz, rgb, lab, selected = xyz[behind_keep], rgb[behind_keep], lab[behind_keep], selected[behind_keep]
        wall_index = wall_index[behind_keep]
    else:
        report.update({"behind_wall_removed": 0, "behind_wall_planes": []})
    lap('behind_wall')
    if settings.get("class_coherence", True):
        lab, coherent, class_stats = class_coherence(xyz, lab, ids, settings, device=device)
        report.update(class_stats)
        xyz, rgb, lab, selected = xyz[coherent], rgb[coherent], lab[coherent], selected[coherent]
    lap('class_coherence')
    dense_enough = sparse_point_mask(xyz, settings, device=device)
    report["removed_sparse"] = int((~dense_enough).sum())
    xyz, rgb, lab, selected = xyz[dense_enough], rgb[dense_enough], lab[dense_enough], selected[dense_enough]
    lap('haze')
    result = np.zeros(len(xyz), dtype=template.dtype)
    for axis, name in enumerate(("x", "y", "z")):
        result[name] = xyz[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        if name in result.dtype.names:
            result[name] = rgb[:, axis]
    if "scalar_class" in result.dtype.names:
        result["scalar_class"] = lab
    del vertex
    PlyData([PlyElement.describe(result, "vertex")], text=False).write(str(classified_ply))
    report["final_points"] = int(len(result))
    report["per_class"] = {str(int(k)): int(c) for k, c in zip(*np.unique(lab, return_counts=True))}
    if scores_path is not None and Path(scores_path).is_file():
        if mode == "extract":
            Path(scores_path).unlink()  # per-point SAM scores do not exist for extracted points
            report["scores_removed"] = True
        else:
            with np.load(scores_path) as archive:
                arrays = {name: archive[name] for name in archive.files}
            # ``selected`` always indexes the original classified cloud. Mirror
            # removal happens before snapping, so checking the shortened cloud
            # here would silently leave every score array out of sync.
            if all(len(values) == input_point_count for values in arrays.values()):
                np.savez(scores_path, **{name: values[selected] for name, values in arrays.items()})
                report["scores_filtered"] = True
    lap('write')
    report['seconds'] = timings
    (dense_dir / "surface_fusion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
