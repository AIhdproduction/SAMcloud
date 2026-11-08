"""Fill measurement gaps with Depth-Anything-V2 depth confirmed by several views (indoor).

COLMAP leaves textureless surfaces (white walls, ceilings) empty. The aligned
Depth-Anything-V2 prior (``depth_prior``: fitted per view to COLMAP's own depth
map) predicts depth there. A predicted pixel only becomes a point when the
other cameras confirm it, much like COLMAP's own fusion:

1. Reference view: pixels where COLMAP measured nothing but the prior has a
   depth, away from depth edges (``edge_tolerance``; edges produce the "flying"
   points between foreground and background), within ``max_depth``.
2. The pixel is back-projected and checked in up to ``neighbors`` nearby views
   looking in a similar direction (``max_view_angle_deg``) that stand at least
   ``min_baseline_m`` away. Views from almost the same spot (consecutive video
   frames) repeat Depth-Anything's own errors, e.g. smoothed slopes across a
   doorway, and would "confirm" them; real parallax does not:
   * a COLMAP depth clearly *behind* the point (free space) rejects it,
   * a COLMAP depth clearly in front means the view does not see it (ignored),
   * otherwise the neighbour's prior must agree within ``tolerance`` (vote) or
     sees through it (vote against).
3. Accepted: >= ``min_agreeing_views`` agreeing neighbours, they are at least
   ``min_agree_ratio`` of all neighbours that could judge, and no free-space
   violation.
4. Points are thinned to one per ``spacing_m`` voxel and never placed in a
   voxel that already holds a measured point. Colour comes from the reference
   photo, the class from its SAM label raster.

Pixels whose surface the camera sees at a grazing angle (``max_incidence_deg``)
are skipped as well. Pixels SAM labels as one of ``exclude_classes`` (windows: glass and the view
through it give Depth-Anything wrong depths, seen as fanned-out rays) are
never used as reference pixels.

Nothing is laid on a regular grid: points sit where real pixels look.

The checks run on the GPU (torch) when one is available; maps are kept in a
small least-recently-used cache there (``gpu_cache_maps``), since the
neighbours of a view are mostly the views just before and after it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from samcloud.engine.ply_io import read_vertex_array
from samcloud.colmap.plane_fill import View, read_colmap_array

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "stride": 2,
    "neighbors": 8,
    "max_view_angle_deg": 60.0,
    "max_neighbor_distance_m": 3.0,
    "min_baseline_m": 0.3,
    "tolerance": 0.03,
    "free_space_margin": 0.05,
    "edge_tolerance": 0.02,
    "min_agreeing_views": 2,
    "min_agree_ratio": 0.6,
    "max_depth": 8.0,
    "spacing_m": 0.01,
    "max_points": 20_000_000,
    "exclude_classes": ["window"],
    "max_incidence_deg": 75.0,
    "device": "auto",          # auto: cuda when available, else cpu
    "gpu_cache_maps": 256,     # per-view maps kept on the device (LRU)
}


def _axis(view: View) -> np.ndarray:
    return view.rotation.T @ np.array([0.0, 0.0, 1.0])


def neighbor_views(views: list[View], settings: dict[str, Any]) -> list[list[int]]:
    """Nearest views looking in a similar direction, per view (indices into ``views``)."""
    centers = np.array([view.center for view in views])
    axes = np.array([_axis(view) for view in views])
    cos_limit = math.cos(math.radians(float(settings["max_view_angle_deg"])))
    result = []
    for index in range(len(views)):
        distance = np.linalg.norm(centers - centers[index], axis=1)
        similar = (axes @ axes[index] >= cos_limit) & (distance <= float(settings["max_neighbor_distance_m"]))
        similar &= distance >= float(settings.get("min_baseline_m", 0.0))
        similar[index] = False
        candidates = np.flatnonzero(similar)
        order = candidates[np.argsort(distance[candidates], kind="stable")]
        result.append(order[: int(settings["neighbors"])].tolist())
    return result


def _edge_mask(depth: np.ndarray, tolerance: float) -> np.ndarray:
    """True where the depth is smooth (no jump to the right/lower neighbour)."""
    smooth = np.ones(depth.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        right = np.abs(np.diff(depth, axis=1)) / np.maximum(depth[:, :-1], 1e-6)
        down = np.abs(np.diff(depth, axis=0)) / np.maximum(depth[:-1, :], 1e-6)
    smooth[:, :-1] &= right <= tolerance
    smooth[:, 1:] &= right <= tolerance
    smooth[:-1, :] &= down <= tolerance
    smooth[1:, :] &= down <= tolerance
    return smooth


def _incidence_mask(depth: np.ndarray, view: View, max_incidence_deg: float) -> np.ndarray:
    """True where the camera looks at the predicted surface steeper than ``max_incidence_deg``.

    Surfaces seen at grazing angles get depth errors along the viewing ray;
    neighbouring cameras at almost the same spot repeat them, which shows up
    as fans of rays. The normal comes from the prior's own surface.
    """
    height, width = depth.shape[:2]
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    x = (cols * view.width / width - view.cx) / view.fx * depth
    y = (rows * view.height / height - view.cy) / view.fy * depth
    points = np.stack((x, y, depth), axis=-1)
    du = np.zeros_like(points)
    dv = np.zeros_like(points)
    du[:, :-1] = points[:, 1:] - points[:, :-1]
    dv[:-1, :] = points[1:, :] - points[:-1, :]
    normal = np.cross(du, dv)
    normal_length = np.linalg.norm(normal, axis=-1)
    ray_length = np.linalg.norm(points, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.abs(np.sum(normal * points, axis=-1)) / (normal_length * ray_length)
    return np.nan_to_num(cosine) >= np.cos(np.radians(float(max_incidence_deg)))


def _voxel_keys(points: np.ndarray, origin: np.ndarray, spacing: float) -> np.ndarray:
    cells = np.floor((points - origin) / spacing).astype(np.int64)
    return (cells[:, 0] * 2_000_003 + cells[:, 1]) * 2_000_003 + cells[:, 2]


class _MapCache:
    """Per-view arrays kept in memory (depth maps are reused by many reference views)."""

    def __init__(self, loader: Callable[[str], np.ndarray | None]) -> None:
        self.loader = loader
        self.items: dict[str, np.ndarray | None] = {}

    def get(self, name: str) -> np.ndarray | None:
        if name not in self.items:
            value = self.loader(name)
            self.items[name] = None if value is None else np.asarray(value, dtype=np.float32)
        return self.items[name]


def _sample(array: np.ndarray, view: View, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample a per-view map given camera pixel coordinates (any map resolution)."""
    height, width = array.shape[:2]
    x = np.clip((u * width / view.width).astype(np.int64), 0, width - 1)
    y = np.clip((v * height / view.height).astype(np.int64), 0, height - 1)
    return array[y, x]


class _TensorCache:
    """Least-recently-used per-view maps as float32 tensors on the compute device."""

    def __init__(self, loader: Callable[[str], np.ndarray | None], device: str, capacity: int) -> None:
        from collections import OrderedDict

        self.loader, self.device, self.capacity = loader, device, max(4, int(capacity))
        self.items: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, name: str):
        import torch

        if name in self.items:
            self.items.move_to_end(name)
            return self.items[name]
        value = self.loader(name)
        tensor = None if value is None else torch.as_tensor(np.asarray(value, dtype=np.float32), device=self.device)
        self.items[name] = tensor
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return tensor


def _sample_t(array, view: View, u, v):
    height, width = array.shape[:2]
    x = (u * width / view.width).long().clamp(0, width - 1)
    y = (v * height / view.height).long().clamp(0, height - 1)
    return array[y, x]


def _edge_mask_torch(depth, tolerance: float):
    smooth = depth.new_ones(depth.shape, dtype=bool)
    safe = depth.clamp(min=1e-6)
    right = (depth[:, 1:] - depth[:, :-1]).abs() / safe[:, :-1]
    down = (depth[1:, :] - depth[:-1, :]).abs() / safe[:-1, :]
    smooth[:, :-1] &= right <= tolerance
    smooth[:, 1:] &= right <= tolerance
    smooth[:-1, :] &= down <= tolerance
    smooth[1:, :] &= down <= tolerance
    return smooth


def _incidence_mask_torch(depth, view: View, max_incidence_deg: float):
    import torch

    height, width = depth.shape[:2]
    dev = depth.device
    cols = (torch.arange(width, device=dev, dtype=torch.float32) + 0.5)[None, :].expand(height, width)
    rows = (torch.arange(height, device=dev, dtype=torch.float32) + 0.5)[:, None].expand(height, width)
    x = (cols * view.width / width - view.cx) / view.fx * depth
    y = (rows * view.height / height - view.cy) / view.fy * depth
    points = torch.stack((x, y, depth), dim=-1)
    du = torch.zeros_like(points)
    dv = torch.zeros_like(points)
    du[:, :-1] = points[:, 1:] - points[:, :-1]
    dv[:-1, :] = points[1:, :] - points[:-1, :]
    normal = torch.cross(du, dv, dim=-1)
    cosine = (normal * points).sum(-1).abs() / (normal.norm(dim=-1) * points.norm(dim=-1))
    return torch.nan_to_num(cosine) >= math.cos(math.radians(float(max_incidence_deg)))


def _fill_device(settings: dict[str, Any]) -> str:
    wanted = str(settings.get("device", "auto"))
    if wanted != "auto":
        return wanted
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "numpy"


def fill_from_depth_prior(
    views: list[View],
    stereo_dir: Path,
    prior_reader: Callable[[str], np.ndarray | None],
    existing_points: np.ndarray,
    settings: dict[str, Any] | None = None,
    *,
    image_reader: Callable[[str], np.ndarray | None] | None = None,
    label_reader: Callable[[str], np.ndarray | None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    exclude_label_ids: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (xyz, rgb, class ids, agreeing view count, report) of confirmed prior points."""
    settings = {**DEFAULTS, **(settings or {})}
    device = _fill_device(settings)
    if device != "numpy":
        candidates = _candidates_torch(views, stereo_dir, prior_reader, settings, device, image_reader=image_reader,
                                       label_reader=label_reader, progress=progress, exclude_label_ids=exclude_label_ids)
        report_stats, blocks = candidates
        return _thin(blocks, existing_points, settings, report_stats, device)
    priors = _MapCache(prior_reader)

    def load_geometric(name: str) -> np.ndarray | None:
        path = Path(stereo_dir) / "depth_maps" / f"{name}.geometric.bin"
        return read_colmap_array(path) if path.is_file() else None

    geometric = _MapCache(load_geometric)
    neighbors = neighbor_views(views, settings)
    stride = max(1, int(settings["stride"]))
    tolerance = float(settings["tolerance"])
    margin = float(settings["free_space_margin"])
    blocks_xyz, blocks_rgb, blocks_cls, blocks_votes = [], [], [], []
    stats = {"reference_views": 0, "candidates": 0, "rejected_free_space": 0, "rejected_votes": 0, "accepted": 0}
    for number, view in enumerate(views, start=1):
        if progress is not None:
            progress(number, len(views))
        prior = priors.get(view.name)
        if prior is None or not neighbors[number - 1]:
            continue
        stats["reference_views"] += 1
        ph, pw = prior.shape[:2]
        rows, cols = np.mgrid[0:ph:stride, 0:pw:stride]
        rows, cols = rows.ravel(), cols.ravel()
        depth = prior[rows, cols]
        keep = (depth > 0) & (depth <= float(settings["max_depth"]))
        keep &= _edge_mask(prior, float(settings["edge_tolerance"]))[rows, cols]
        keep &= _incidence_mask(prior, view, float(settings["max_incidence_deg"]))[rows, cols]
        u = (cols + 0.5) * view.width / pw
        v = (rows + 0.5) * view.height / ph
        measured = geometric.get(view.name)
        if measured is not None:
            keep &= _sample(measured, view, u, v) <= 0
        labels = label_reader(view.name) if label_reader is not None else None
        pixel_labels = _sample(labels, view, u, v).astype(np.uint8) if labels is not None else None
        if pixel_labels is not None and exclude_label_ids:
            keep &= ~np.isin(pixel_labels, np.asarray(exclude_label_ids, dtype=np.uint8))
        if not np.any(keep):
            continue
        u, v, depth = u[keep], v[keep], depth[keep]
        if pixel_labels is not None:
            pixel_labels = pixel_labels[keep]
        cam = np.column_stack(((u - view.cx) / view.fx * depth, (v - view.cy) / view.fy * depth, depth))
        world = (cam - view.translation) @ view.rotation
        stats["candidates"] += len(world)
        agree = np.zeros(len(world), dtype=np.int32)
        disagree = np.zeros(len(world), dtype=np.int32)
        violations = np.zeros(len(world), dtype=np.int32)
        for index in neighbors[number - 1]:
            other = views[index]
            other_prior = priors.get(other.name)
            other_geo = geometric.get(other.name)
            if other_prior is None and other_geo is None:
                continue
            local = world @ other.rotation.T + other.translation
            z = local[:, 2]
            front = z > 1e-3
            ou = np.where(front, other.fx * local[:, 0] / np.where(front, z, 1.0) + other.cx, -1.0)
            ov = np.where(front, other.fy * local[:, 1] / np.where(front, z, 1.0) + other.cy, -1.0)
            inside = front & (ou >= 0) & (ou < other.width) & (ov >= 0) & (ov < other.height)
            sel = np.flatnonzero(inside)
            if not len(sel):
                continue
            zs = z[sel]
            occluded = np.zeros(len(sel), dtype=bool)
            if other_geo is not None:
                g = _sample(other_geo, other, ou[sel], ov[sel])
                violations[sel[(g > 0) & (g > zs * (1.0 + margin))]] += 1
                occluded = (g > 0) & (g < zs * (1.0 - margin))
            if other_prior is not None:
                p = _sample(other_prior, other, ou[sel], ov[sel])
                judge = ~occluded & (p > 0)
                agree[sel[judge & (np.abs(p - zs) <= tolerance * zs)]] += 1
                disagree[sel[judge & (p > zs * (1.0 + tolerance))]] += 1
        free_ok = violations == 0
        votes_ok = (agree >= int(settings["min_agreeing_views"])) & (
            agree >= float(settings["min_agree_ratio"]) * np.maximum(agree + disagree, 1)
        )
        stats["rejected_free_space"] += int(np.count_nonzero(~free_ok))
        stats["rejected_votes"] += int(np.count_nonzero(free_ok & ~votes_ok))
        accepted = free_ok & votes_ok
        if not np.any(accepted):
            continue
        stats["accepted"] += int(np.count_nonzero(accepted))
        blocks_xyz.append(world[accepted])
        blocks_votes.append(agree[accepted])
        image = image_reader(view.name) if image_reader is not None else None
        if image is not None:
            blocks_rgb.append(_sample(image, view, u[accepted], v[accepted])[:, :3].astype(np.uint8))
        else:
            blocks_rgb.append(np.full((int(np.count_nonzero(accepted)), 3), 200, dtype=np.uint8))
        if pixel_labels is not None:
            blocks_cls.append(pixel_labels[accepted])
        else:
            blocks_cls.append(np.zeros(int(np.count_nonzero(accepted)), dtype=np.uint8))
    return _thin((blocks_xyz, blocks_rgb, blocks_cls, blocks_votes), existing_points, settings, stats, "numpy")


def _candidates_torch(views, stereo_dir, prior_reader, settings, device, *, image_reader, label_reader, progress,
                      exclude_label_ids):
    """GPU version of the per-view confirmation loop; returns (stats, numpy blocks)."""
    import torch

    def load_geometric(name: str) -> np.ndarray | None:
        path = Path(stereo_dir) / "depth_maps" / f"{name}.geometric.bin"
        return read_colmap_array(path) if path.is_file() else None

    capacity = int(settings.get("gpu_cache_maps", 256))
    priors = _TensorCache(prior_reader, device, capacity)
    geometric = _TensorCache(load_geometric, device, capacity)
    neighbors = neighbor_views(views, settings)
    stride = max(1, int(settings["stride"]))
    tolerance = float(settings["tolerance"])
    margin = float(settings["free_space_margin"])
    excluded = torch.as_tensor(list(exclude_label_ids), dtype=torch.long, device=device) if exclude_label_ids else None
    blocks_xyz, blocks_rgb, blocks_cls, blocks_votes = [], [], [], []
    stats = {"reference_views": 0, "candidates": 0, "rejected_free_space": 0, "rejected_votes": 0, "accepted": 0,
             "device": device}
    f32 = dict(dtype=torch.float32, device=device)
    for number, view in enumerate(views, start=1):
        if progress is not None:
            progress(number, len(views))
        prior = priors.get(view.name)
        if prior is None or not neighbors[number - 1]:
            continue
        stats["reference_views"] += 1
        ph, pw = prior.shape[:2]
        rows = torch.arange(0, ph, stride, device=device)[:, None].expand(-1, (pw + stride - 1) // stride).reshape(-1)
        cols = torch.arange(0, pw, stride, device=device)[None, :].expand((ph + stride - 1) // stride, -1).reshape(-1)
        depth = prior[rows, cols]
        keep = (depth > 0) & (depth <= float(settings["max_depth"]))
        keep &= _edge_mask_torch(prior, float(settings["edge_tolerance"]))[rows, cols]
        keep &= _incidence_mask_torch(prior, view, float(settings["max_incidence_deg"]))[rows, cols]
        u = (cols.to(torch.float32) + 0.5) * view.width / pw
        v = (rows.to(torch.float32) + 0.5) * view.height / ph
        measured = geometric.get(view.name)
        if measured is not None:
            keep &= _sample_t(measured, view, u, v) <= 0
        labels = label_reader(view.name) if label_reader is not None else None
        pixel_labels = None
        if labels is not None:
            lab = torch.as_tensor(np.asarray(labels).astype(np.int64), device=device)
            pixel_labels = _sample_t(lab, view, u, v)
            if excluded is not None:
                keep &= ~torch.isin(pixel_labels, excluded)
        if not bool(keep.any()):
            continue
        u, v, depth = u[keep], v[keep], depth[keep]
        if pixel_labels is not None:
            pixel_labels = pixel_labels[keep]
        rotation = torch.as_tensor(view.rotation, **f32)
        translation = torch.as_tensor(view.translation, **f32)
        cam = torch.stack(((u - view.cx) / view.fx * depth, (v - view.cy) / view.fy * depth, depth), dim=1)
        world = (cam - translation) @ rotation
        stats["candidates"] += int(len(world))
        agree = torch.zeros(len(world), dtype=torch.int32, device=device)
        disagree = torch.zeros_like(agree)
        violations = torch.zeros_like(agree)
        for index in neighbors[number - 1]:
            other = views[index]
            other_prior = priors.get(other.name)
            other_geo = geometric.get(other.name)
            if other_prior is None and other_geo is None:
                continue
            local = world @ torch.as_tensor(other.rotation, **f32).T + torch.as_tensor(other.translation, **f32)
            z = local[:, 2]
            front = z > 1e-3
            safe_z = torch.where(front, z, torch.ones_like(z))
            ou = torch.where(front, other.fx * local[:, 0] / safe_z + other.cx, torch.full_like(z, -1.0))
            ov = torch.where(front, other.fy * local[:, 1] / safe_z + other.cy, torch.full_like(z, -1.0))
            inside = front & (ou >= 0) & (ou < other.width) & (ov >= 0) & (ov < other.height)
            sel = torch.nonzero(inside, as_tuple=False).squeeze(1)
            if not len(sel):
                continue
            zs = z[sel]
            occluded = torch.zeros(len(sel), dtype=torch.bool, device=device)
            if other_geo is not None:
                g = _sample_t(other_geo, other, ou[sel], ov[sel])
                violations[sel[(g > 0) & (g > zs * (1.0 + margin))]] += 1
                occluded = (g > 0) & (g < zs * (1.0 - margin))
            if other_prior is not None:
                p = _sample_t(other_prior, other, ou[sel], ov[sel])
                judge = ~occluded & (p > 0)
                agree[sel[judge & ((p - zs).abs() <= tolerance * zs)]] += 1
                disagree[sel[judge & (p > zs * (1.0 + tolerance))]] += 1
        free_ok = violations == 0
        votes_ok = (agree >= int(settings["min_agreeing_views"])) & (
            agree.float() >= float(settings["min_agree_ratio"]) * (agree + disagree).clamp(min=1).float()
        )
        stats["rejected_free_space"] += int((~free_ok).sum())
        stats["rejected_votes"] += int((free_ok & ~votes_ok).sum())
        accepted = free_ok & votes_ok
        count = int(accepted.sum())
        if not count:
            continue
        stats["accepted"] += count
        blocks_xyz.append(world[accepted].double().cpu().numpy())
        blocks_votes.append(agree[accepted].cpu().numpy())
        ua, va = u[accepted].cpu().numpy(), v[accepted].cpu().numpy()
        image = image_reader(view.name) if image_reader is not None else None
        if image is not None:
            blocks_rgb.append(_sample(image, view, ua, va)[:, :3].astype(np.uint8))
        else:
            blocks_rgb.append(np.full((count, 3), 200, dtype=np.uint8))
        if pixel_labels is not None:
            blocks_cls.append(pixel_labels[accepted].cpu().numpy().astype(np.uint8))
        else:
            blocks_cls.append(np.zeros(count, dtype=np.uint8))
    return stats, (blocks_xyz, blocks_rgb, blocks_cls, blocks_votes)


def _thin(blocks, existing_points, settings, stats, device):
    """One point per ``spacing_m`` voxel (best confirmed), never in a voxel with a measured point."""
    blocks_xyz, blocks_rgb, blocks_cls, blocks_votes = blocks
    report: dict[str, Any] = {"settings": {key: settings[key] for key in DEFAULTS if key != "enabled"}, **stats}
    if not blocks_xyz:
        report["added_points"] = 0
        empty = np.empty((0, 3))
        return empty, np.empty((0, 3), np.uint8), np.empty(0, np.uint8), np.empty(0, np.int32), report
    xyz = np.concatenate(blocks_xyz)
    rgb = np.concatenate(blocks_rgb)
    cls = np.concatenate(blocks_cls)
    votes = np.concatenate(blocks_votes)
    spacing = float(settings["spacing_m"])
    existing = np.asarray(existing_points, dtype=np.float64)
    origin = np.minimum(xyz.min(axis=0), existing.min(axis=0) if len(existing) else xyz.min(axis=0)) - spacing
    keys = _voxel_keys(xyz, origin, spacing)
    existing_keys = np.unique(_voxel_keys(existing, origin, spacing)) if len(existing) else None
    # Prefer the best-confirmed candidate per voxel, skip voxels with measured points.
    if device in ("numpy", "cpu"):
        order = np.lexsort((-votes, keys))
        sorted_keys = keys[order]
        first = np.ones(len(keys), dtype=bool)
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        if existing_keys is not None:
            first &= ~np.isin(sorted_keys, existing_keys)
        chosen = order[first]
    else:  # same selection, sorted on the GPU (tens of millions of candidates)
        import torch

        k = torch.as_tensor(keys, device=device)
        by_votes = torch.sort(-torch.as_tensor(votes, device=device), stable=True).indices
        order = by_votes[torch.sort(k[by_votes], stable=True).indices]
        sorted_keys = k[order]
        first = torch.ones(len(order), dtype=torch.bool, device=device)
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        if existing_keys is not None:
            first &= ~torch.isin(sorted_keys, torch.as_tensor(existing_keys, device=device))
        chosen = order[first].cpu().numpy()
        del k, by_votes, order, sorted_keys, first
    xyz, rgb, cls, votes = xyz[chosen], rgb[chosen], cls[chosen], votes[chosen]
    limit = int(settings["max_points"])
    if len(xyz) > limit:
        pick = np.sort(np.random.default_rng(0).choice(len(xyz), limit, replace=False))
        xyz, rgb, cls, votes = xyz[pick], rgb[pick], cls[pick], votes[pick]
        report["limited_to_max_points"] = True
    report["added_points"] = int(len(xyz))
    return xyz, rgb, cls, votes, report


def fill_classified_from_prior(
    dense_dir: Path,
    classified_ply: Path,
    views: list[View],
    prior_reader: Callable[[str], np.ndarray | None],
    settings: dict[str, Any] | None = None,
    *,
    scores_path: Path | None = None,
    labels_dir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Append prior points confirmed by several views to a classified cloud in place."""
    import cv2
    from plyfile import PlyData, PlyElement

    dense_dir, classified_ply = Path(dense_dir), Path(classified_ply)
    settings = {**DEFAULTS, **(settings or {})}
    if not bool(settings.get("enabled", True)):
        return {"enabled": False, "added_points": 0}
    if not (dense_dir / "stereo" / "depth_maps").is_dir():
        return {"enabled": True, "added_points": 0, "reason": "depth maps missing"}
    vertex = read_vertex_array(classified_ply)
    existing = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)

    def read_image(name: str) -> np.ndarray | None:
        image = cv2.imread(str(dense_dir / "images" / name), cv2.IMREAD_COLOR)
        return None if image is None else image[:, :, ::-1]

    def read_labels(name: str) -> np.ndarray | None:
        if labels_dir is None:
            return None
        path = Path(labels_dir) / f"{name}.png"
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None

    short = [name.split(",", 1)[0].strip().lower() for name in (class_names or [])]
    excluded = tuple(short.index(name.lower()) + 1 for name in settings.get("exclude_classes") or [] if name.lower() in short)
    xyz, rgb, cls, votes, report = fill_from_depth_prior(
        views, dense_dir / "stereo", prior_reader, existing, settings,
        image_reader=read_image, label_reader=read_labels, progress=progress, exclude_label_ids=excluded,
    )
    report["excluded_label_ids"] = list(excluded)
    report = {"enabled": True, "rule": "depth_anything_multiview", "original_points": int(len(existing)), **report}
    if len(xyz):
        extra = np.zeros(len(xyz), dtype=vertex.dtype)
        for axis, name in enumerate(("x", "y", "z")):
            extra[name] = xyz[:, axis]
        for axis, name in enumerate(("red", "green", "blue")):
            extra[name] = rgb[:, axis]
        extra["scalar_class"] = cls
        combined = np.concatenate([vertex, extra])
        del vertex
        PlyData([PlyElement.describe(combined, "vertex")], text=False).write(str(classified_ply))
        report["added_per_class_id"] = {str(int(k)): int(c) for k, c in zip(*np.unique(cls, return_counts=True))}
        if scores_path is not None and Path(scores_path).is_file():
            with np.load(scores_path) as archive:
                arrays = {name: archive[name] for name in archive.files}
            if all(len(values) == len(existing) for values in arrays.values()):
                added = len(xyz)
                extra_scores = {
                    "top1_class": cls.astype(np.uint8),
                    "top1_conf": np.where(cls > 0, 0.5, 0.0).astype(np.float16),
                    "top2_class": np.zeros(added, dtype=np.uint8),
                    "top2_conf": np.zeros(added, dtype=np.float16),
                    "view_count": np.minimum(votes + 1, 65535).astype(np.uint16),
                }
                np.savez(scores_path, **{
                    name: np.concatenate([values, extra_scores.get(name, np.zeros(added, values.dtype)).astype(values.dtype)])
                    for name, values in arrays.items()
                })
                report["scores_extended"] = True
    (dense_dir / "depth_fill.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
