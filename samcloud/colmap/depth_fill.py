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
    "units_per_metre": 1.0,    # project scale: max_depth, min_baseline_m, spacing_m, max_neighbor_distance_m, plane tolerance
    # Round 5: local, edge-aware re-alignment of the prior against COLMAP depth and reliable sparse points.
    "local_alignment": False,  # (config/colmap.yml turns the round-5 rules on)   per region (strong image lines x tiles) instead of one global fit per view
    "use_edges": True,         # regions never cross strong image lines
    "use_sparse": True,        # reliable sparse points (error / track length filtered) as high-weight anchors
    "halo_filter": False,      # prior depth jump without image edge, image edge with COLMAP jump but smooth prior
    "anchor_rule": False,      # >= min_anchor_views COLMAP/sparse-confirmed views (breaks "Depth-Anything confirms itself")
    "min_anchor_views": 1,
    "plane_completion": False,  # RANSAC planes from anchors for textureless wall/ceiling SAM regions
    "plane_label_ids": [],      # SAM label ids of wall/ceiling (filled from class names)
    "plane_tolerance_m": 0.03,
    "plane_min_inliers": 12,
    "plane_max_planes": 3,
    "plane_reach_frac": 0.15,
    "image_edges": {},          # overrides of image_edges.EDGE_DEFAULTS
}

_SCALED_KEYS = ("max_depth", "min_baseline_m", "spacing_m", "max_neighbor_distance_m", "plane_tolerance_m")


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


def _mirror_mask(view: View, shape: tuple[int, int], mirrors) -> np.ndarray | None:
    """Union and dilate the confirmed mirror masks for one view."""
    if not mirrors:
        return None
    import cv2

    combined = np.zeros(shape, dtype=bool)
    radius = 0
    for mirror in mirrors:
        mask = mirror.masks.get(view.name)
        if mask is None:
            continue
        source = np.asarray(mask, dtype=np.uint8)
        if source.shape != shape:
            source = cv2.resize(source, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        combined |= source > 0
        radius = max(radius, int(getattr(mirror, "dilate_px", 0)))
    if not combined.any():
        return None
    return cv2.dilate(combined.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0 if radius else combined


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


class PriorRefiner:
    """Per-view prior after local, edge-aware alignment (round 5); ``depth`` and ``near`` maps share one computation.

    ``depth(name)``: float32 map at prior resolution, 0 where the prior is distrusted (no anchors in
    reach, residual above tolerance, halo, wrong against an image edge). ``near(name)``: 1 where a
    COLMAP pixel or reliable sparse anchor lies within ``anchor_radius_frac``. Without
    ``local_alignment`` the raw prior is returned and ``near`` is all ones.
    """

    def __init__(self, views, stereo_dir, prior_reader, settings, *, image_reader=None, label_reader=None, anchors=None):
        from collections import OrderedDict

        self.view_by_name = {view.name: view for view in views}
        self.stereo_dir, self.prior_reader, self.settings = Path(stereo_dir), prior_reader, settings
        self.image_reader, self.label_reader = image_reader, label_reader
        self.anchors = anchors or {}
        from samcloud.colmap.image_edges import EDGE_DEFAULTS

        self.edge_settings = {**EDGE_DEFAULTS, **(settings.get("image_edges") or {})}
        self._geo: "OrderedDict[str, Any]" = OrderedDict()
        self._done: "OrderedDict[str, Any]" = OrderedDict()
        self.info = {"views_refined": 0, "trusted_share_sum": 0.0, "halo_share_sum": 0.0, "wrong_share_sum": 0.0,
                     "plane_views": 0, "planes": 0, "plane_pixels": 0}

    def geometric(self, name: str) -> np.ndarray | None:
        if name in self._geo:
            self._geo.move_to_end(name)
            return self._geo[name]
        path = self.stereo_dir / "depth_maps" / f"{name}.geometric.bin"
        value = None
        if path.is_file():
            from samcloud.colmap.depth_edges import geometric_valid_mask

            depth = read_colmap_array(path)
            value = np.where(geometric_valid_mask(depth, self.view_by_name[name], self.settings), depth, 0.0).astype(np.float32)
        self._geo[name] = value
        if len(self._geo) > 16:
            self._geo.popitem(last=False)
        return value

    def _compute(self, name: str):
        if name in self._done:
            return self._done[name]
        prior = self.prior_reader(name)
        result = (None, None)
        if prior is not None:
            prior = np.asarray(prior, dtype=np.float32)
            align = bool(self.settings.get("local_alignment", True))
            if align or bool(self.settings.get("anchor_rule", False)):
                result = self._refine(name, prior, align)
            else:
                result = (prior, np.ones(prior.shape, np.float32))
        self._done[name] = result
        if len(self._done) > 4:
            self._done.popitem(last=False)
        return result

    def _refine(self, name: str, prior: np.ndarray, align: bool = True):
        import cv2
        from samcloud.colmap import image_edges as ie

        view = self.view_by_name[name]
        es = self.edge_settings
        shape = prior.shape[:2]
        wh, ww = ie.working_shape(shape, es)
        geo = self.geometric(name)
        if geo is None:
            geo_w = np.zeros((wh, ww), np.float32)
        else:
            geo_w = cv2.resize(geo, (ww, wh), interpolation=cv2.INTER_NEAREST)
        prior_w = cv2.resize(prior, (ww, wh), interpolation=cv2.INTER_NEAREST)
        labels = self.label_reader(name) if self.label_reader is not None else None
        edges, components = None, None
        if bool(self.settings.get("use_edges", True)) and self.image_reader is not None:
            edges = ie.detect_edges(self.image_reader(name), labels, (wh, ww), es)
            components, _ = ie.edge_components(edges["strong"])
        sparse_w = None
        if bool(self.settings.get("use_sparse", True)):
            sparse_w = ie.sparse_to_working(self.anchors.get(name), view, (wh, ww))
        scale_w, near_w, _info = ie.fit_prior_regions(prior_w, geo_w, sparse_w, components, es)
        distrust = np.zeros((wh, ww), bool)
        halo_share = wrong_share = 0.0
        if edges is not None and bool(self.settings.get("halo_filter", True)):
            halo, wrong = ie.edge_consistency(prior_w, geo_w, edges, es)
            distrust = halo | wrong
            halo_share, wrong_share = float(halo.mean()), float(wrong.mean())
        if align:
            refined_w = np.where((scale_w > 0) & ~distrust, prior_w * scale_w, 0.0).astype(np.float32)
        else:  # only the "anchor nearby" map is wanted: the prior stays as it is
            refined_w = prior_w.astype(np.float32)
        if bool(self.settings.get("plane_completion", False)) and labels is not None and \
                self.settings.get("plane_label_ids"):
            lab_w = cv2.resize(np.asarray(labels).astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST)
            region = np.isin(lab_w, np.asarray(self.settings["plane_label_ids"], dtype=np.uint8)) & (geo_w <= 0)
            plane_settings = {**es, "plane_min_inliers": self.settings["plane_min_inliers"],
                              "plane_max_planes": self.settings["plane_max_planes"],
                              "plane_reach_frac": self.settings["plane_reach_frac"]}
            plane_depth, used = ie.complete_planes(
                refined_w, geo_w, sparse_w, region, components, view, plane_settings,
                float(self.settings["plane_tolerance_m"]))
            if used:
                has = plane_depth > 0
                refined_w = np.where(has, plane_depth, refined_w)
                near_w = near_w | has
                self.info["plane_views"] += 1
                self.info["planes"] += used
                self.info["plane_pixels"] += int(has.sum())
        refined = cv2.resize(refined_w, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        near = cv2.resize(near_w.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        self.info["views_refined"] += 1
        self.info["trusted_share_sum"] += float((refined_w > 0).mean())
        self.info["halo_share_sum"] += halo_share
        self.info["wrong_share_sum"] += wrong_share
        return refined, near

    def depth(self, name: str):
        return self._compute(name)[0]

    def near(self, name: str):
        return self._compute(name)[1]

    def report(self) -> dict[str, Any]:
        n = max(1, self.info["views_refined"])
        return {"views_refined": self.info["views_refined"],
                "mean_trusted_share": round(self.info["trusted_share_sum"] / n, 4),
                "mean_halo_share": round(self.info["halo_share_sum"] / n, 4),
                "mean_wrong_share": round(self.info["wrong_share_sum"] / n, 4),
                "plane_views": self.info["plane_views"], "planes": self.info["planes"],
                "plane_pixels": self.info["plane_pixels"]}


def _scaled(settings: dict[str, Any]) -> dict[str, Any]:
    try:
        unit = float(settings.get("units_per_metre", 1.0))
    except (TypeError, ValueError):  # "auto" is resolved by the pipeline; direct callers get metric units
        unit = 1.0
    return {**settings, "units_per_metre": unit,
            **{key: float(settings[key]) * unit for key in _SCALED_KEYS if key in settings}}


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
    mirrors=None,
    anchors: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (xyz, rgb, class ids, agreeing view count, report) of confirmed prior points.

    ``anchors``: name -> (uv, depth) of reliable sparse points (``image_edges.load_sparse_anchors``).
    """
    settings = _scaled({**DEFAULTS, **(settings or {})})
    device = _fill_device(settings)
    refiner = PriorRefiner(views, stereo_dir, prior_reader, settings, image_reader=image_reader,
                           label_reader=label_reader, anchors=anchors)
    if device != "numpy":
        candidates = _candidates_torch(views, stereo_dir, refiner, settings, device, image_reader=image_reader,
                                       label_reader=label_reader, progress=progress, exclude_label_ids=exclude_label_ids,
                                       mirrors=mirrors)
        report_stats, blocks = candidates
        report_stats["alignment"] = refiner.report()
        return _thin(blocks, existing_points, settings, report_stats, device)
    priors = _MapCache(refiner.depth)
    nears = _MapCache(refiner.near)
    geometric = _MapCache(refiner.geometric)
    anchor_rule = bool(settings["anchor_rule"])
    min_anchor = int(settings["min_anchor_views"])
    only = set(settings["reference_names"]) if settings.get("reference_names") else None
    neighbors = neighbor_views(views, settings)
    stride = max(1, int(settings["stride"]))
    tolerance = float(settings["tolerance"])
    margin = float(settings["free_space_margin"])
    blocks_xyz, blocks_rgb, blocks_cls, blocks_votes = [], [], [], []
    stats = {"reference_views": 0, "candidates": 0, "rejected_free_space": 0, "rejected_votes": 0,
             "rejected_anchor": 0, "accepted": 0, "units_per_metre": float(settings["units_per_metre"])}
    for number, view in enumerate(views, start=1):
        if progress is not None:
            progress(number, len(views))
        if only is not None and view.name not in only:
            continue
        prior = priors.get(view.name)
        near_ref = nears.get(view.name)
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
        mirror = _mirror_mask(view, prior.shape, mirrors)
        if mirror is not None:
            keep &= ~mirror[rows, cols]
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
        if mirrors:
            from samcloud.colmap.mirror_filter import behind_mirror_cone

            keep_cone = ~behind_mirror_cone(world, view, mirrors, settings)
            u, v, depth, world = u[keep_cone], v[keep_cone], depth[keep_cone], world[keep_cone]
            if pixel_labels is not None:
                pixel_labels = pixel_labels[keep_cone]
            if not len(world):
                continue
        stats["candidates"] += len(world)
        agree = np.zeros(len(world), dtype=np.int32)
        disagree = np.zeros(len(world), dtype=np.int32)
        violations = np.zeros(len(world), dtype=np.int32)
        anchored = (_sample(near_ref, view, u, v) > 0).astype(np.int32) if near_ref is not None else np.zeros(len(world), np.int32)
        for index in neighbors[number - 1]:
            other = views[index]
            other_prior = priors.get(other.name)
            other_near = nears.get(other.name)
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
                other_mirror = _mirror_mask(other, other_geo.shape, mirrors)
                if other_mirror is not None:
                    blocked = _sample(other_mirror, other, ou[sel], ov[sel]).astype(bool)
                    g = np.where(blocked, 0.0, g)
                violations[sel[(g > 0) & (g > zs * (1.0 + margin))]] += 1
                occluded = (g > 0) & (g < zs * (1.0 - margin))
            if other_prior is not None:
                p = _sample(other_prior, other, ou[sel], ov[sel])
                judge = ~occluded & (p > 0)
                agreeing = judge & (np.abs(p - zs) <= tolerance * zs)
                agree[sel[agreeing]] += 1
                disagree[sel[judge & (p > zs * (1.0 + tolerance))]] += 1
                if other_near is not None:
                    anchored[sel[agreeing & (_sample(other_near, other, ou[sel], ov[sel]) > 0)]] += 1
        free_ok = violations == 0
        votes_ok = (agree >= int(settings["min_agreeing_views"])) & (
            agree >= float(settings["min_agree_ratio"]) * np.maximum(agree + disagree, 1)
        )
        anchor_ok = (anchored >= min_anchor) if anchor_rule else np.ones(len(world), dtype=bool)
        stats["rejected_free_space"] += int(np.count_nonzero(~free_ok))
        stats["rejected_votes"] += int(np.count_nonzero(free_ok & ~votes_ok))
        stats["rejected_anchor"] += int(np.count_nonzero(free_ok & votes_ok & ~anchor_ok))
        accepted = free_ok & votes_ok & anchor_ok
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
    stats["alignment"] = refiner.report()
    return _thin((blocks_xyz, blocks_rgb, blocks_cls, blocks_votes), existing_points, settings, stats, "numpy")


def _candidates_torch(views, stereo_dir, refiner, settings, device, *, image_reader, label_reader, progress,
                      exclude_label_ids, mirrors=None):
    """GPU version of the per-view confirmation loop; returns (stats, numpy blocks)."""
    import torch

    capacity = int(settings.get("gpu_cache_maps", 256))
    priors = _TensorCache(refiner.depth, device, capacity)
    nears = _TensorCache(refiner.near, device, capacity)
    geometric = _TensorCache(refiner.geometric, device, capacity)
    anchor_rule = bool(settings["anchor_rule"])
    min_anchor = int(settings["min_anchor_views"])
    only = set(settings["reference_names"]) if settings.get("reference_names") else None
    neighbors = neighbor_views(views, settings)
    stride = max(1, int(settings["stride"]))
    tolerance = float(settings["tolerance"])
    margin = float(settings["free_space_margin"])
    excluded = torch.as_tensor(list(exclude_label_ids), dtype=torch.long, device=device) if exclude_label_ids else None
    blocks_xyz, blocks_rgb, blocks_cls, blocks_votes = [], [], [], []
    stats = {"reference_views": 0, "candidates": 0, "rejected_free_space": 0, "rejected_votes": 0,
             "rejected_anchor": 0, "accepted": 0, "device": device, "units_per_metre": float(settings["units_per_metre"])}
    f32 = dict(dtype=torch.float32, device=device)
    for number, view in enumerate(views, start=1):
        if progress is not None:
            progress(number, len(views))
        if only is not None and view.name not in only:
            continue
        prior = priors.get(view.name)
        near_ref = nears.get(view.name)
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
        mirror = _mirror_mask(view, (ph, pw), mirrors)
        if mirror is not None:
            mirror_t = torch.as_tensor(mirror, dtype=torch.bool, device=device)
            keep &= ~mirror_t[rows, cols]
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
        if mirrors:
            from samcloud.colmap.mirror_filter import behind_mirror_cone

            cone = torch.as_tensor(behind_mirror_cone(world.detach().cpu().numpy(), view, mirrors, settings), device=device)
            keep_cone = ~cone
            u, v, depth, world = u[keep_cone], v[keep_cone], depth[keep_cone], world[keep_cone]
            if pixel_labels is not None:
                pixel_labels = pixel_labels[keep_cone]
            if not len(world):
                continue
        stats["candidates"] += int(len(world))
        agree = torch.zeros(len(world), dtype=torch.int32, device=device)
        disagree = torch.zeros_like(agree)
        violations = torch.zeros_like(agree)
        anchored = (_sample_t(near_ref, view, u, v) > 0).to(torch.int32) if near_ref is not None else torch.zeros_like(agree)
        for index in neighbors[number - 1]:
            other = views[index]
            other_prior = priors.get(other.name)
            other_near = nears.get(other.name)
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
                other_mirror = _mirror_mask(other, tuple(other_geo.shape), mirrors)
                if other_mirror is not None:
                    mirror_t = torch.as_tensor(other_mirror, dtype=torch.bool, device=device)
                    blocked = _sample_t(mirror_t, other, ou[sel], ov[sel]).bool()
                    g = torch.where(blocked, torch.zeros_like(g), g)
                violations[sel[(g > 0) & (g > zs * (1.0 + margin))]] += 1
                occluded = (g > 0) & (g < zs * (1.0 - margin))
            if other_prior is not None:
                p = _sample_t(other_prior, other, ou[sel], ov[sel])
                judge = ~occluded & (p > 0)
                agreeing = judge & ((p - zs).abs() <= tolerance * zs)
                agree[sel[agreeing]] += 1
                disagree[sel[judge & (p > zs * (1.0 + tolerance))]] += 1
                if other_near is not None:
                    anchored[sel[agreeing & (_sample_t(other_near, other, ou[sel], ov[sel]) > 0)]] += 1
        free_ok = violations == 0
        votes_ok = (agree >= int(settings["min_agreeing_views"])) & (
            agree.float() >= float(settings["min_agree_ratio"]) * (agree + disagree).clamp(min=1).float()
        )
        anchor_ok = (anchored >= min_anchor) if anchor_rule else torch.ones_like(free_ok)
        stats["rejected_free_space"] += int((~free_ok).sum())
        stats["rejected_votes"] += int((free_ok & ~votes_ok).sum())
        stats["rejected_anchor"] += int((free_ok & votes_ok & ~anchor_ok).sum())
        accepted = free_ok & votes_ok & anchor_ok
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
    mirrors=None,
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
    if not settings.get("plane_label_ids"):
        settings["plane_label_ids"] = [short.index(name) + 1 for name in ("wall", "ceiling") if name in short]
    anchors = None
    if bool(settings.get("local_alignment")) and bool(settings.get("use_sparse", True)) and (dense_dir / "sparse").is_dir():
        from samcloud.colmap.image_edges import EDGE_DEFAULTS, load_sparse_anchors

        edge = {**EDGE_DEFAULTS, **(settings.get("image_edges") or {})}
        try:
            anchors = load_sparse_anchors(dense_dir / "sparse", views, float(edge["sparse_max_error_px"]),
                                          int(edge["sparse_min_track"]))
        except Exception as error:  # noqa: BLE001 - the dense COLMAP pixels still anchor the alignment
            print(f"WARNING: Sparse anchors were skipped: {error}")
    xyz, rgb, cls, votes, report = fill_from_depth_prior(
        views, dense_dir / "stereo", prior_reader, existing, settings,
        image_reader=read_image, label_reader=read_labels, progress=progress, exclude_label_ids=excluded,
        mirrors=mirrors, anchors=anchors,
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
