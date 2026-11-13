"""Detect planar mirrors from SAM masks and remove reflected phantom geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from samcloud.colmap.plane_fill import View, read_colmap_array


DEFAULTS = {
    "enabled": True, "ring_inner_px": 4, "ring_outer_px": 12, "min_depth_jump_m": 0.25,
    "merge_angle_deg": 3.0, "merge_offset_m": 0.03, "min_views": 2, "through_share": 0.8,
    "min_baseline_m": 0.3, "min_mask_share": 0.002, "mirror_dilate_px": 6, "merge_gap_m": 0.10,
    "ring_label_ids": [2, 5, 17], "fill_plane": False,
    # Kept with the mirror settings because the filter runs after depth fill.
    # These are the depth_fill tolerance and edge_tolerance defaults.
    "prior_tolerance": 0.03, "prior_edge_tolerance": 0.02,
    # The Depth-Anything prior counts as evidence only where COLMAP validates it: in a window
    # (fraction of the shorter image side) around the pixel at least ``prior_trust_min_valid``
    # of the pixels have valid COLMAP depth and their median relative residual is within
    # ``prior_tolerance``.  Elsewhere the prior is "unknown", neither support nor contradiction.
    "prior_trust_window_frac": 0.04, "prior_trust_min_valid": 0.05,
    "depth_edge_filter": {"enabled": True, "max_incidence_deg": 80.0, "edge_px": 1,
                          "edge_jump_rel": 0.05, "edge_jump_m": 0.03, "drop_lonely": False},
}


@dataclass
class MirrorCandidate:
    view: View
    normal: np.ndarray
    offset: float
    center: np.ndarray
    mask: np.ndarray
    outline: np.ndarray
    ring_points: np.ndarray
    ring_depths: np.ndarray


@dataclass
class Mirror:
    normal: np.ndarray
    offset: float
    center: np.ndarray
    polygons: list[np.ndarray]
    views: list[View]
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    dilate_px: int = 6
    filter_stats: dict[str, int] = field(default_factory=dict)

    @property
    def area(self) -> float:
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union

            return float(unary_union([Polygon(points) for points in self.polygons]).area)
        except ImportError:
            return float(sum(abs(_polygon_area(points[:, :2])) for points in self.polygons))

    @property
    def baseline_m(self) -> float:
        if len(self.views) < 2:
            return 0.0
        centres = np.stack([view.center for view in self.views])
        return float(np.max(np.linalg.norm(centres[:, None] - centres[None, :], axis=2)))


def read_mirror_mask(masks_dir: Path | str | None, image_name: str) -> np.ndarray | None:
    if masks_dir is None:
        return None
    path = Path(masks_dir) / f"{image_name}.png"
    if not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if mask is None else mask > 0


def _resized(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool, copy=False)
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _world_points(view: View, pixels: np.ndarray, depths: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    height, width = depth_shape
    u = (pixels[:, 0] + 0.5) * view.width / width
    v = (pixels[:, 1] + 0.5) * view.height / height
    local = np.column_stack(((u - view.cx) * depths / view.fx, (v - view.cy) * depths / view.fy, depths))
    return (local - view.translation) @ view.rotation


def _fit_plane_ransac(points: np.ndarray, depths: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray] | None:
    if len(points) < 20:
        return None
    threshold = 0.02 + 0.005 * float(np.median(depths))
    rng = np.random.default_rng(seed)
    best = np.zeros(len(points), dtype=bool)
    trials = min(512, max(96, len(points) // 3))
    for _ in range(trials):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) <= threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 3:
        return None
    centered = points[best] - points[best].mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ points[best].mean(axis=0))
    best = np.abs(points @ normal + offset) <= threshold
    return normal, offset, best


def _mask_hull(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(np.vstack(contours)).reshape(-1, 2).astype(float)
    return hull if len(hull) >= 3 else None


def mirror_candidate(view: View, geometric: np.ndarray, mask: np.ndarray, settings: dict[str, Any],
                     labels: np.ndarray | None = None) -> MirrorCandidate | None:
    """Fit one mirror plane from a SAM mask and its surrounding geometric-depth ring."""
    from samcloud.colmap.surface_fusion import view_depth

    settings = {**DEFAULTS, **(settings or {})}
    if mask.mean() < float(settings["min_mask_share"]):
        return None
    mask = _resized(mask, geometric.shape)
    outer = cv2.dilate(mask.astype(np.uint8), np.ones((2 * int(settings["ring_outer_px"]) + 1,) * 2, np.uint8)) > 0
    inner = cv2.dilate(mask.astype(np.uint8), np.ones((2 * int(settings["ring_inner_px"]) + 1,) * 2, np.uint8)) > 0
    ring = outer & ~inner
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != geometric.shape:
            labels = cv2.resize(labels, (geometric.shape[1], geometric.shape[0]), interpolation=cv2.INTER_NEAREST)
        ring &= np.isin(labels, np.asarray(settings["ring_label_ids"], dtype=np.uint8))
    combined = view_depth(view, geometric, None, None, (), {"max_geometric_depth": np.inf,
                          "geometric_weight": 1.0, "prior_weight": 0.0,
                          "depth_edge_filter": settings["depth_edge_filter"]}, device="cpu")
    if combined is None:
        return None
    depth = combined[0].cpu().numpy()
    pixels_yx = np.argwhere(ring & (depth > 0))
    if len(pixels_yx) < 20:
        return None
    pixels = pixels_yx[:, ::-1]
    depths = depth[pixels_yx[:, 0], pixels_yx[:, 1]]
    points = _world_points(view, pixels, depths, depth.shape)
    fitted = _fit_plane_ransac(points, depths, view.index)
    if fitted is None:
        return None
    normal, offset, inliers = fitted
    if inliers.mean() < 0.60:
        return None
    if abs(float(normal[2])) > np.sin(np.deg2rad(10.0)):
        return None
    center = points[inliers].mean(axis=0)
    if normal @ (view.center - center) < 0:
        normal, offset = -normal, -offset
    inside = mask & (depth > 0)
    py, px = np.nonzero(inside)
    if not len(px):
        return None
    observed = depth[py, px]
    u = (px + 0.5) * view.width / depth.shape[1]
    v = (py + 0.5) * view.height / depth.shape[0]
    directions = np.column_stack(((u - view.cx) / view.fx, (v - view.cy) / view.fy, np.ones(len(u)))) @ view.rotation
    denominator = directions @ normal
    plane_depth = -(view.center @ normal + offset) / denominator
    valid = np.isfinite(plane_depth) & (plane_depth > 0)
    if not valid.any() or float(np.median(observed[valid] - plane_depth[valid])) < float(settings["min_depth_jump_m"]) * float(settings.get("units_per_metre", 1.0)):
        return None
    hull = _mask_hull(mask)
    if hull is None:
        return None
    directions = np.column_stack((((hull[:, 0] + 0.5) * view.width / depth.shape[1] - view.cx) / view.fx,
                                  ((hull[:, 1] + 0.5) * view.height / depth.shape[0] - view.cy) / view.fy,
                                  np.ones(len(hull)))) @ view.rotation
    distances = -(view.center @ normal + offset) / (directions @ normal)
    outline = view.center + directions * distances[:, None]
    return MirrorCandidate(view, normal, offset, center, mask, outline, points, depths)


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def _polygon_area(points: np.ndarray) -> float:
    return 0.5 * float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1)))


def _point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    crossing = False
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (first[1] > y) != (second[1] > y):
            crossing ^= x < (second[0] - first[0]) * (y - first[1]) / (second[1] - first[1] + 1e-12) + first[0]
    return crossing


def _segment_distance_2d(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    direction = second - first
    length_sq = float(direction @ direction)
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - first))
    fraction = np.clip(float((point - first) @ direction) / length_sq, 0.0, 1.0)
    return float(np.linalg.norm(point - (first + fraction * direction)))


def _polygon_distance_2d(first: np.ndarray, second: np.ndarray) -> float:
    if any(_point_in_polygon_2d(point, second) for point in first) or any(_point_in_polygon_2d(point, first) for point in second):
        return 0.0
    return min(_segment_distance_2d(point, edge_first, edge_second)
               for polygon, other in ((first, second), (second, first))
               for point in polygon
               for edge_first, edge_second in zip(other, np.roll(other, -1, axis=0)))


def _candidate_polygons(candidate: MirrorCandidate, normal: np.ndarray, center: np.ndarray) -> np.ndarray:
    first, second = _basis(normal)
    relative = candidate.outline - center
    return np.column_stack((relative @ first, relative @ second))


def _candidates_merge(first: MirrorCandidate, second: MirrorCandidate, settings: dict[str, Any]) -> bool:
    """Return whether two local mirror observations represent one physical mirror."""
    alignment = float(first.normal @ second.normal)
    if abs(alignment) < np.cos(np.deg2rad(float(settings["merge_angle_deg"]))):
        return False
    second_normal = second.normal if alignment >= 0 else -second.normal
    second_offset = second.offset if alignment >= 0 else -second.offset
    maximum_offset = float(settings["merge_offset_m"]) * float(settings.get("units_per_metre", 1.0))
    # Global plane offsets vary with the arbitrary reconstruction origin when normals differ.
    # Compare each fitted plane at the other observation's local polygon centre instead.
    if abs(float(first.normal @ second.center + first.offset)) > maximum_offset:
        return False
    if abs(float(second_normal @ first.center + second_offset)) > maximum_offset:
        return False
    first_polygon = _candidate_polygons(first, first.normal, first.center)
    second_polygon = _candidate_polygons(second, first.normal, first.center)
    maximum_gap = float(settings["merge_gap_m"]) * float(settings.get("units_per_metre", 1.0))
    return _polygon_distance_2d(first_polygon, second_polygon) <= maximum_gap


def detect_mirrors(views: list[View], stereo_dir: Path, masks_dir: Path | str | None, settings: dict[str, Any],
                   labels_dir: Path | str | None = None) -> list[Mirror]:
    """Return confirmed mirrors; an absent mask folder intentionally yields no mirrors."""
    settings = {**DEFAULTS, **(settings or {})}
    if not bool(settings["enabled"]) or masks_dir is None or not Path(masks_dir).is_dir():
        return []
    candidates: list[MirrorCandidate] = []
    for view in views:
        mask = read_mirror_mask(masks_dir, view.name)
        path = Path(stereo_dir) / "depth_maps" / f"{view.name}.geometric.bin"
        if mask is None or mask.mean() < float(settings["min_mask_share"]) or not path.is_file():
            continue
        labels_path = Path(labels_dir) / f"{view.name}.png" if labels_dir is not None else None
        labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED) if labels_path is not None and labels_path.is_file() else None
        candidate = mirror_candidate(view, read_colmap_array(path), mask, settings, labels)
        if candidate is not None:
            candidates.append(candidate)
    groups: list[list[MirrorCandidate]] = []
    for candidate in candidates:
        for group in groups:
            if any(_candidates_merge(candidate, other, settings) for other in group):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    mirrors: list[Mirror] = []
    for group in groups:
        independent: list[MirrorCandidate] = []
        for candidate in group:
            if not independent or all(np.linalg.norm(candidate.view.center - other.view.center) >= float(settings["min_baseline_m"]) * float(settings.get("units_per_metre", 1.0))
                                      for other in independent):
                independent.append(candidate)
        if len(independent) < int(settings["min_views"]):
            continue
        all_points = np.concatenate([candidate.ring_points for candidate in group])
        all_depths = np.concatenate([candidate.ring_depths for candidate in group])
        fitted = _fit_plane_ransac(all_points, all_depths, min(candidate.view.index for candidate in group))
        if fitted is None or fitted[2].mean() < 0.60:
            continue
        normal, offset, inliers = fitted
        if abs(float(normal[2])) > np.sin(np.deg2rad(10.0)):
            continue
        center = all_points[inliers].mean(axis=0)
        if normal @ (group[0].view.center - center) < 0:
            normal, offset = -normal, -offset
        first, second = _basis(normal)
        polygons = []
        for candidate in group:
            relative = candidate.outline - center
            polygons.append(np.column_stack((relative @ first, relative @ second)))
        mirrors.append(Mirror(normal, offset, center, polygons, [candidate.view for candidate in group],
                              {candidate.view.name: candidate.mask for candidate in group},
                              int(settings["mirror_dilate_px"])))
    return mirrors


def plane_depth_map(view: View, shape: tuple[int, int], mirror: Mirror) -> np.ndarray:
    """Depth of a confirmed mirror plane in one geometric-depth-map frame."""
    height, width = shape
    y, x = np.mgrid[:height, :width]
    u = (x + 0.5) * view.width / width
    v = (y + 0.5) * view.height / height
    directions = np.stack(((u - view.cx) / view.fx, (v - view.cy) / view.fy, np.ones_like(u)), axis=-1) @ view.rotation
    denominator = directions @ mirror.normal
    return (-(view.center @ mirror.normal + mirror.offset) / denominator).astype(np.float32)


def _inside_polygons(points: np.ndarray, mirror: Mirror) -> np.ndarray:
    first, second = _basis(mirror.normal)
    relative = points - mirror.center
    x, y = relative @ first, relative @ second
    inside = np.zeros(len(points), dtype=bool)
    for polygon in mirror.polygons:
        crossing = np.zeros(len(points), dtype=bool)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            crossing ^= ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
        inside |= crossing
    return inside


def _inside_polygons_torch(points, mirror: Mirror):
    import torch

    first, second = _basis(mirror.normal)
    first = torch.as_tensor(first, dtype=points.dtype, device=points.device)
    second = torch.as_tensor(second, dtype=points.dtype, device=points.device)
    centre = torch.as_tensor(mirror.center, dtype=points.dtype, device=points.device)
    relative = points - centre
    x, y = relative @ first, relative @ second
    inside = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for polygon in mirror.polygons:
        polygon = torch.as_tensor(polygon, dtype=points.dtype, device=points.device)
        crossing = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        for index in range(len(polygon)):
            a, b = polygon[index], polygon[(index + 1) % len(polygon)]
            crossing ^= ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
        inside |= crossing
    return inside


def behind_mirror_cone(points: np.ndarray, view: View, mirrors: list[Mirror], settings: dict[str, Any]) -> np.ndarray:
    """Return points beyond a mirror plane whose ray from ``view`` crosses its polygon."""
    points = np.asarray(points, dtype=np.float64)
    result = np.zeros(len(points), dtype=bool)
    margin = float(settings.get("visibility_margin_m", 0.03)) * float(settings.get("units_per_metre", 1.0))
    margin_rel = float(settings.get("visibility_margin_rel", 0.01))
    for mirror in mirrors:
        ray = points - view.center
        distance = np.linalg.norm(ray, axis=1)
        behind = points @ mirror.normal + mirror.offset < -(margin + margin_rel * distance)
        denominator = ray @ mirror.normal
        fraction = -(view.center @ mirror.normal + mirror.offset) / np.where(np.abs(denominator) > 1e-12, denominator, np.inf)
        crossing = behind & (fraction > 0) & (fraction < 1)
        if np.any(crossing):
            intersections = view.center + ray[crossing] * fraction[crossing, None]
            result[np.flatnonzero(crossing)[_inside_polygons(intersections, mirror)]] = True
    return result


def _dilated_mask(mask: np.ndarray | None, shape: tuple[int, int], radius: int) -> np.ndarray | None:
    if mask is None:
        return None
    resized = _resized(mask, shape)
    if radius <= 0:
        return resized
    return cv2.dilate(resized.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0


def _prior_valid_mask_torch(depth, tolerance: float):
    """Return valid, non-edge pixels for an aligned Depth-Anything prior."""
    import torch

    valid = depth > 0
    safe = depth.clamp(min=1e-6)
    right = (depth[:, 1:] - depth[:, :-1]).abs() / safe[:, :-1]
    down = (depth[1:, :] - depth[:-1, :]).abs() / safe[:-1, :]
    valid[:, :-1] &= right <= tolerance
    valid[:, 1:] &= right <= tolerance
    valid[:-1, :] &= down <= tolerance
    valid[1:, :] &= down <= tolerance
    return valid


def _prior_trust_torch(prior, prior_ok, depth, depth_ok, settings: dict[str, Any]):
    """Return (trusted, both, agree): pixels where COLMAP validates the prior locally."""
    import torch
    import torch.nn.functional as functional

    both = prior_ok & depth_ok
    agree = both & ((prior - depth).abs() <= float(settings["prior_tolerance"]) * depth)
    window = max(9, int(float(settings["prior_trust_window_frac"]) * min(prior.shape)) | 1)
    valid_share = functional.avg_pool2d(both.float()[None, None], window, 1, window // 2, count_include_pad=False)[0, 0]
    agree_share = functional.avg_pool2d(agree.float()[None, None], window, 1, window // 2, count_include_pad=False)[0, 0]
    # Median residual <= tolerance means at least half of the valid pixels agree.
    trusted = prior_ok & (valid_share >= float(settings["prior_trust_min_valid"])) & (agree_share >= 0.5 * valid_share)
    return trusted, both, agree


def phantom_mask(points: np.ndarray, mirrors: list[Mirror], views: list[View], stereo_dir: Path,
                 settings: dict[str, Any], chunk: int = 500_000, masks_dir: Path | str | None = None) -> np.ndarray:
    """Return reflected points that lack sufficient direct, unoccluded observations."""
    settings = {**DEFAULTS, **(settings or {})}
    import torch
    from samcloud.colmap.depth_edges import geometric_valid_mask_torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    removed = np.zeros(len(points), dtype=bool)
    margin_abs = float(settings.get("visibility_margin_m", 0.03)) * float(settings.get("units_per_metre", 1.0))
    margin_rel = float(settings.get("visibility_margin_rel", 0.01))
    max_depth = float(settings.get("max_geometric_depth", np.inf))
    masks_dir = Path(masks_dir) if masks_dir is not None else Path(stereo_dir).parent / "sam_mirror"
    candidate_masks: list[np.ndarray] = []
    for mirror in mirrors:
        centres = np.stack([view.center for view in mirror.views])
        camera_distance = np.median(np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2), axis=1)
        behind = points @ mirror.normal + mirror.offset < -(margin_abs + margin_rel * camera_distance)
        candidate_masks.append(behind)
    if not any(mask.any() for mask in candidate_masks):
        return removed
    candidate_matrix = np.column_stack(candidate_masks)
    candidate_rows = np.flatnonzero(candidate_matrix.any(axis=1))
    candidate_flags = torch.as_tensor(candidate_matrix[candidate_rows], dtype=torch.bool, device=device)
    candidate_points = torch.as_tensor(points[candidate_rows], dtype=torch.float32, device=device)
    through_counts = torch.zeros((len(candidate_rows), len(mirrors)), dtype=torch.int16, device=device)
    prior_through_counts = torch.zeros_like(through_counts)
    polygon_counts = torch.zeros_like(through_counts)
    direct_counts = torch.zeros_like(through_counts)
    blocked_counts = torch.zeros_like(through_counts)
    # Cull the point cloud by conservative spatial cells before projecting.  This keeps the
    # exact point test below, but avoids scanning every behind-plane point for every cubemap face.
    cell_size = max(0.10, 0.25 * float(settings.get("units_per_metre", 1.0)))
    cell_origin = points[candidate_rows].min(axis=0)
    cell_index = np.floor((points[candidate_rows] - cell_origin) / cell_size).astype(np.int32)
    cell_keys, inverse = np.unique(cell_index, axis=0, return_inverse=True)
    cell_order = np.argsort(inverse, kind="stable")
    cell_starts = np.r_[0, np.flatnonzero(np.diff(inverse[cell_order])) + 1, len(cell_order)]
    cell_centres = torch.as_tensor(cell_origin + (cell_keys.astype(np.float64) + 0.5) * cell_size,
                                   dtype=torch.float32, device=device)
    cell_radius = float(np.sqrt(3.0) * cell_size * 0.5)
    depth_maps_read = 0
    prior_maps_read = 0
    prior_pairs = {"in": [0, 0], "out": [0, 0]}
    prior_untrusted_views = 0
    for view in views:
        rotation = torch.as_tensor(view.rotation, dtype=torch.float32, device=device)
        translation = torch.as_tensor(view.translation, dtype=torch.float32, device=device)
        camera = torch.as_tensor(view.center, dtype=torch.float32, device=device)
        camera_front = torch.as_tensor([float(mirror.normal @ view.center + mirror.offset) > margin_abs for mirror in mirrors],
                                       dtype=torch.bool, device=device)
        cell_local = cell_centres @ rotation.T + translation
        cell_z = cell_local[:, 2]
        x_slope = max(view.cx, view.width - view.cx) / view.fx
        y_slope = max(view.cy, view.height - view.cy) / view.fy
        cell_visible = (cell_z + cell_radius > 0.1) & \
                       (cell_local[:, 0].abs() <= x_slope * (cell_z + cell_radius) + cell_radius) & \
                       (cell_local[:, 1].abs() <= y_slope * (cell_z + cell_radius) + cell_radius)
        selected_cells = torch.nonzero(cell_visible, as_tuple=False).squeeze(1).cpu().numpy()
        if not len(selected_cells):
            continue
        selected_indices = np.concatenate([cell_order[cell_starts[index]:cell_starts[index + 1]] for index in selected_cells])
        depth_loaded = False
        depth = None
        depth_ok = None
        prior = None
        prior_ok = None
        prior_trust = None
        shape: tuple[int, int] = (view.height, view.width)
        mask_t = None
        path = Path(stereo_dir) / "depth_maps" / f"{view.name}.geometric.bin"
        for start in range(0, len(selected_indices), chunk):
            local_indices = torch.as_tensor(selected_indices[start:start + chunk], dtype=torch.long, device=device)
            part = candidate_points[local_indices]
            local = part @ rotation.T + translation
            z = local[:, 2]
            u_frame = view.fx * local[:, 0] / z + view.cx
            v_frame = view.fy * local[:, 1] / z + view.cy
            in_frame = (z > 0.1) & (u_frame >= 0) & (u_frame < view.width) & (v_frame >= 0) & (v_frame < view.height)
            eligible = candidate_flags[local_indices]
            in_frame &= eligible.any(dim=1)
            if not bool(in_frame.any()):
                continue
            if not depth_loaded:
                if path.is_file():
                    depth = torch.as_tensor(read_colmap_array(path), dtype=torch.float32, device=device)
                    depth = torch.where(depth <= max_depth, depth, torch.zeros_like(depth))
                    depth_ok = geometric_valid_mask_torch(depth, view, settings) & (depth <= max_depth)
                    depth_maps_read += 1
                    shape = tuple(depth.shape)
                prior_path = Path(stereo_dir).parent / "depth_prior" / f"{view.name}.npy"
                if prior_path.is_file():
                    prior_array = np.load(prior_path).astype(np.float32, copy=False)
                    if prior_array.shape != shape:
                        prior_array = cv2.resize(prior_array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                    prior = torch.as_tensor(prior_array, dtype=torch.float32, device=device)
                    prior_ok = _prior_valid_mask_torch(prior, float(settings["prior_edge_tolerance"]))
                    prior_maps_read += 1
                mirror_mask = _dilated_mask(read_mirror_mask(masks_dir, view.name), shape, int(settings["mirror_dilate_px"]))
                mask_t = torch.as_tensor(mirror_mask, dtype=torch.bool, device=device) if mirror_mask is not None else None
                if prior is not None and depth is not None:
                    prior_trust, both, agree = _prior_trust_torch(prior, prior_ok, depth, depth_ok, settings)
                    regions = {"out": torch.ones_like(both) if mask_t is None else ~mask_t}
                    if mask_t is not None:
                        regions["in"] = mask_t
                    for key, region in regions.items():
                        prior_pairs[key][0] += int((both & region).sum().item())
                        prior_pairs[key][1] += int((agree & region).sum().item())
                    if int(both.sum().item()) and int(agree.sum().item()) < 0.5 * int(both.sum().item()):
                        prior_untrusted_views += 1
                depth_loaded = True
            positions = torch.nonzero(in_frame, as_tuple=False).squeeze(1)
            visible_local = local[positions]
            visible_flags = eligible[positions]
            height, width = shape
            u = ((view.fx * visible_local[:, 0] / visible_local[:, 2] + view.cx) * width / view.width).long().clamp(0, width - 1)
            v = ((view.fy * visible_local[:, 1] / visible_local[:, 2] + view.cy) * height / view.height).long().clamp(0, height - 1)
            ray = part[positions] - camera
            crossings = []
            plane_crossed = []
            for mirror_index, mirror in enumerate(mirrors):
                normal = torch.as_tensor(mirror.normal, dtype=torch.float32, device=device)
                fraction = -(camera @ normal + float(mirror.offset)) / (ray @ normal)
                crossed = (fraction > 0) & (fraction < 1)
                # The ray passes through this mirror's wall plane, so the view looks through that wall.
                plane_crossed.append(crossed & visible_flags[:, mirror_index])
                crossed &= _inside_polygons_torch(camera + ray * fraction[:, None], mirror)
                crossings.append(crossed)
            through_wall = torch.stack(plane_crossed).any(dim=0)
            inside_mask = mask_t[v, u] if mask_t is not None else torch.zeros(len(positions), dtype=torch.bool, device=device)
            target_rows = local_indices[positions]
            measured = depth[v, u] if depth is not None else torch.zeros(len(positions), dtype=torch.float32, device=device)
            valid_depth = depth_ok[v, u] if depth_ok is not None else torch.zeros(len(positions), dtype=torch.bool, device=device)
            tolerance = margin_abs + margin_rel * visible_local[:, 2]
            prior_match = torch.zeros(len(positions), dtype=torch.bool, device=device)
            prior_open = torch.zeros(len(positions), dtype=torch.bool, device=device)
            if prior is not None and prior_ok is not None:
                prior_depth = prior[v, u]
                prior_match = ~valid_depth & prior_ok[v, u] & ((prior_depth - visible_local[:, 2]).abs()
                                                               <= float(settings["prior_tolerance"]) * visible_local[:, 2])
                # Where COLMAP has no depth (textureless ceilings) the prior still shows whether the
                # point is occluded: it is open to this view when the prior is not closer than the point.
                # The prior is only believed where COLMAP validates it locally, or where the ray stays on
                # the point's side of every mirror wall (a wrong prior cannot hide a wall in between).
                open_prior = ~valid_depth & prior_ok[v, u] & (prior_depth >= visible_local[:, 2] * (1.0 - float(settings["prior_tolerance"])))
                trusted = prior_trust[v, u] if prior_trust is not None else torch.zeros_like(open_prior)
                prior_open = open_prior & (trusted | ~through_wall)
            for mirror_index, crossed in enumerate(crossings):
                through_colmap = valid_depth & ((measured - visible_local[:, 2]).abs() <= tolerance)
                through = visible_flags[:, mirror_index] & camera_front[mirror_index] & crossed & (through_colmap | prior_match)
                through_counts[target_rows[through], mirror_index] += 1
                polygon_through = visible_flags[:, mirror_index] & camera_front[mirror_index] & crossed
                polygon_counts[target_rows[polygon_through], mirror_index] += 1
                through_prior = visible_flags[:, mirror_index] & camera_front[mirror_index] & crossed & prior_match
                prior_through_counts[target_rows[through_prior], mirror_index] += 1
            open_view = (valid_depth & (measured >= visible_local[:, 2] - tolerance)) | prior_open
            # A view that does not look through a mirror polygon but measures a surface clearly in
            # front of the point (COLMAP, or the prior) proves the point hidden from that view.
            closed_view = (valid_depth & (measured < visible_local[:, 2] - tolerance))
            if prior is not None and prior_ok is not None:
                closed_view = closed_view | (~valid_depth & prior_ok[v, u]
                                             & (prior[v, u] < visible_local[:, 2] * (1.0 - float(settings["prior_tolerance"]))))
            plain_view = ~torch.stack(crossings).any(dim=0) & ~inside_mask
            direct = plain_view & open_view
            blocked = plain_view & closed_view
            for mirror_index in range(len(mirrors)):
                if bool(direct.any()):
                    kept = direct & visible_flags[:, mirror_index]
                    direct_counts[target_rows[kept], mirror_index] += 1
                if bool(blocked.any()):
                    hidden = blocked & visible_flags[:, mirror_index]
                    blocked_counts[target_rows[hidden], mirror_index] += 1
    for mirror_index, mirror in enumerate(mirrors):
        rows = candidate_masks[mirror_index][candidate_rows]
        through = through_counts[:, mirror_index][rows]
        prior_through = prior_through_counts[:, mirror_index][rows]
        direct = direct_counts[:, mirror_index][rows]
        polygon = polygon_counts[:, mirror_index][rows]
        blocked = blocked_counts[:, mirror_index][rows]
        enough_through = through >= 2
        depth_rule = enough_through & (through.float() >= float(settings["through_share"]) * (through + direct).float())
        # A reflection is seen through the mirror polygon but never directly: every other view that
        # could see the point is blocked by the wall in front of it (at least two such views).
        occluded_rule = (polygon >= 2) & (direct == 0) & (blocked >= 2)
        rejected = depth_rule | occluded_rule
        removed[candidate_rows[rows]] |= rejected.cpu().numpy()
        mirror.filter_stats = {
            "through_points": int((through > 0).sum().item()),
            "prior_through": int((prior_through > 0).sum().item()),
            "removed": int(rejected.sum().item()),
            "direct_kept": int((enough_through & ~rejected & (direct > 0)).sum().item()),
            "depth_maps_read": depth_maps_read,
            "prior_maps_read": prior_maps_read,
            "prior_disagree_share_out_mask": 1.0 - prior_pairs["out"][1] / max(1, prior_pairs["out"][0]),
            "prior_disagree_share_in_mask": 1.0 - prior_pairs["in"][1] / max(1, prior_pairs["in"][0]),
            "prior_untrusted_views": prior_untrusted_views,
        }
    return removed


def mirror_report(mirrors: list[Mirror], removed: np.ndarray) -> dict[str, Any]:
    return {
        "mirrors": [{"normal": [float(value) for value in mirror.normal], "centre": [float(value) for value in mirror.center],
                     "area": mirror.area, "baseline_m": mirror.baseline_m, "views": [view.name for view in mirror.views],
                     "through_points": int(mirror.filter_stats.get("through_points", 0)),
                     "prior_through": int(mirror.filter_stats.get("prior_through", 0)),
                     "removed": int(mirror.filter_stats.get("removed", 0)),
                     "direct_kept": int(mirror.filter_stats.get("direct_kept", 0))} for mirror in mirrors],
        "mirror_points_removed": int(removed.sum()),
        "mirror_depth_maps_read": max((int(mirror.filter_stats.get("depth_maps_read", 0)) for mirror in mirrors), default=0),
        "mirror_prior_maps_read": max((int(mirror.filter_stats.get("prior_maps_read", 0)) for mirror in mirrors), default=0),
    }
