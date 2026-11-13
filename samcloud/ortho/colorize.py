"""Tile-based visibility-tested colour sampling and feathered multi-view blending."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from samcloud.dxf.cancel import CancelToken, ProgressCallback, check_cancel, report_progress
from samcloud.ortho.cameras import OrthoCamera, project_points, scaled
from samcloud.ortho.config import OrthoConfig
from samcloud.ortho.dsm import DsmResult
from samcloud.ortho.grid import OrthoGrid
from samcloud.ortho.photos import PhotoLoader
from samcloud.ortho.zbuffer import build_zbuffer, visible_depth


@dataclass(slots=True)
class ColourResult:
    rgba: np.ndarray
    view_count: np.ndarray
    colour_source: np.ndarray
    missing_photos: list[str]


class _ZBufferCache:
    """Byte-bounded LRU cache that prevents large multi-camera projects exhausting GPU memory."""

    def __init__(self, budget_mb: int) -> None:
        self._budget = budget_mb * 1024 * 1024
        self._entries: OrderedDict[str, tuple[torch.Tensor, float]] = OrderedDict()
        self._bytes = 0

    def get(self, camera: OrthoCamera, points: torch.Tensor, config: OrthoConfig, indoor: bool) -> tuple[torch.Tensor, float]:
        cached = self._entries.pop(camera.name, None)
        if cached is not None:
            self._entries[camera.name] = cached
            return cached
        created = build_zbuffer(camera, points, config, indoor=indoor)
        size = created[0].numel() * created[0].element_size()
        while self._entries and self._bytes + size > self._budget:
            old, _factor = self._entries.popitem(last=False)[1]
            self._bytes -= old.numel() * old.element_size()
        self._entries[camera.name] = created
        self._bytes += size
        return created


def colorize(grid: OrthoGrid, dsm: DsmResult, cameras: list[OrthoCamera], config: OrthoConfig, *, indoor: bool, device: str, camera_local_centres: dict[str, np.ndarray] | None = None, progress: ProgressCallback | None = None, cancel: CancelToken | None = None) -> ColourResult:
    """Blend visible source-photo samples on independent north-up raster tiles."""
    target = torch.device(device)
    all_points = torch.as_tensor(dsm.full_colmap_xyz, dtype=torch.float32, device=target)
    cache = _ZBufferCache(config.zbuffer.cache_mb)
    usable: list[OrthoCamera] = []
    missing: list[str] = []
    for camera in cameras:
        check_cancel(cancel)
        if camera.image_path is None:
            missing.append(camera.name)
        else:
            usable.append(camera)
    rgba = np.zeros((grid.height, grid.width, 4), dtype=np.uint8)
    views = np.zeros((grid.height, grid.width), dtype=np.uint8)
    colour_source = np.zeros((grid.height, grid.width), dtype=np.uint8)
    dsm_heights = torch.as_tensor(np.asarray(dsm.dsm), dtype=torch.float32, device=target)
    loader = PhotoLoader(str(target))
    tiles_y = range(0, grid.height, config.tile_size)
    tiles_x = range(0, grid.width, config.tile_size)
    total_tiles = max(1, len(tiles_y) * len(tiles_x))
    tile_number = 0
    max_incidence = config.blend.max_incidence_deg_indoor if indoor else config.blend.max_incidence_deg
    min_cos = float(np.cos(np.deg2rad(max_incidence)))
    cand_count = min(max(config.blend.candidates, config.blend.top_k), len(usable))
    for y0 in tiles_y:
        for x0 in tiles_x:
            check_cancel(cancel)
            y1, x1 = min(grid.height, y0 + config.tile_size), min(grid.width, x0 + config.tile_size)
            xyz = torch.as_tensor(np.asarray(dsm.colmap_xyz[y0:y1, x0:x1]).reshape(-1, 3), dtype=torch.float32, device=target)
            normal = torch.as_tensor(dsm.normals[y0:y1, x0:x1].reshape(-1, 3), dtype=torch.float32, device=target)
            cloud_rgb = torch.as_tensor(dsm.top_rgb[y0:y1, x0:x1].reshape(-1, 3), dtype=torch.uint8, device=target)
            surface_fill = torch.as_tensor((dsm.validity[y0:y1, x0:x1].reshape(-1) == 3), dtype=torch.bool, device=target)
            rows, cols = np.indices((y1 - y0, x1 - x0))
            local_xy = grid.pixel_to_local(np.column_stack((cols.ravel() + x0, rows.ravel() + y0)))
            local_xy_t = torch.as_tensor(local_xy, dtype=torch.float32, device=target)
            local_z = torch.as_tensor(np.asarray(dsm.dsm[y0:y1, x0:x1]).reshape(-1), dtype=torch.float32, device=target)
            valid_cell = torch.isfinite(xyz).all(dim=1)
            cand_scores = torch.full((len(xyz), cand_count), float("-inf"), dtype=torch.float32, device=target)
            cand_rgb = torch.zeros((len(xyz), cand_count, 3), dtype=torch.float32, device=target)
            cand_feather = torch.zeros((len(xyz), cand_count), dtype=torch.float32, device=target)
            count = torch.zeros(len(xyz), dtype=torch.uint8, device=target)
            for camera_index, camera in enumerate(usable):
                check_cancel(cancel)
                report_progress(progress, 0.25 + 0.65 * (tile_number + (camera_index + 1) / max(1, len(usable))) / total_tiles, "Colouring orthophoto camera batches")
                if not _camera_sees_tile(camera, xyz, valid_cell):
                    continue
                u, v, depth, valid = project_points(camera, xyz)
                zbuf, factor = cache.get(camera, all_points, config, indoor)
                reduced = scaled(camera, factor)
                ru, rv = u * factor, v * factor
                col = torch.clamp(torch.floor(ru).to(torch.long), 0, reduced.width - 1)
                row = torch.clamp(torch.floor(rv).to(torch.long), 0, reduced.height - 1)
                sampled_z = zbuf[row, col]
                voxel = config.zbuffer.voxel_indoor if indoor else config.zbuffer.voxel_outdoor
                ray = torch.as_tensor(camera.centre, dtype=torch.float32, device=target) - xyz
                distance = torch.linalg.vector_norm(ray, dim=1).clamp_min(1e-6)
                cos_incidence = (normal * (ray / distance[:, None])).sum(dim=1)
                incidence_ok = cos_incidence >= min_cos
                visible = valid_cell & valid & incidence_ok & visible_depth(depth, sampled_z, voxel, config, cos_incidence, float(reduced.params[0]))
                if not indoor and config.blend.ray_check_enabled and camera_local_centres is not None:
                    visible &= ~dsm_ray_occluded(dsm_heights, grid, local_xy_t, local_z, torch.as_tensor(camera_local_centres[camera.name], dtype=torch.float32, device=target), config)
                if not torch.any(visible):
                    continue
                image = loader.get(camera.image_path).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
                gx = (u + 0.5) * 2.0 / camera.width - 1.0
                gy = (v + 0.5) * 2.0 / camera.height - 1.0
                sampled = functional.grid_sample(image, torch.stack((gx, gy), dim=1).reshape(1, -1, 1, 2), mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, :, 0].T
                angle = torch.clamp(cos_incidence, min=0.0)
                centre_distance = torch.sqrt(((u - camera.width / 2) / (camera.width / 2)) ** 2 + ((v - camera.height / 2) / (camera.height / 2)) ** 2)
                score = config.blend.weight_angle * angle + config.blend.weight_distance / distance + config.blend.weight_center * (1.0 - centre_distance.clamp(0, 1))
                if not indoor and config.blend.nadir_penalty > 0:
                    incidence_deg = torch.rad2deg(torch.acos(torch.clamp(angle, 0.0, 1.0)))
                    excess = ((incidence_deg - config.blend.nadir_soft_limit_deg) / max(1e-6, 90.0 - config.blend.nadir_soft_limit_deg)).clamp(0.0, 1.0)
                    score -= config.blend.nadir_penalty * excess * excess
                edge = torch.minimum(torch.minimum(u, camera.width - 1 - u), torch.minimum(v, camera.height - 1 - v)).clamp_min(0) / max(1, config.blend.feather_px)
                border_feather = _smoothstep(edge)
                edge_distance = cv2.distanceTransform(visible.reshape(y1 - y0, x1 - x0).detach().cpu().numpy().astype(np.uint8), cv2.DIST_L2, 3).reshape(-1)
                visibility_feather = torch.as_tensor(_smoothstep_np(edge_distance / max(1, config.blend.feather_px)), dtype=torch.float32, device=target)
                candidate_score = torch.where(visible, score, torch.full_like(score, float("-inf")))
                combined_scores = torch.cat((cand_scores, candidate_score[:, None]), dim=1)
                combined_rgb = torch.cat((cand_rgb, sampled[:, None, :]), dim=1)
                combined_feather = torch.cat((cand_feather, (border_feather * visibility_feather)[:, None]), dim=1)
                cand_scores, selected = torch.topk(combined_scores, k=cand_count, dim=1)
                cand_rgb = torch.gather(combined_rgb, 1, selected[:, :, None].expand(-1, -1, 3))
                cand_feather = torch.gather(combined_feather, 1, selected)
                count += visible.to(torch.uint8)
            if cand_count > 0:
                valid_cands = torch.isfinite(cand_scores)
                num_valid = valid_cands.sum(dim=1, keepdim=True)
                masked_rgb = cand_rgb.masked_fill(~valid_cands.unsqueeze(-1), float("nan"))
                med = torch.nanmedian(masked_rgb, dim=1, keepdim=True).values
                rgb_dist = torch.linalg.vector_norm(cand_rgb - med, dim=2)
                reject = (num_valid >= 3) & (rgb_dist > config.blend.consistency_threshold)
                surviving_scores = torch.where(reject, torch.tensor(float("-inf"), dtype=cand_scores.dtype, device=target), cand_scores)

                count_top = min(config.blend.top_k, cand_count)
                top_scores, selected = torch.topk(surviving_scores, k=count_top, dim=1)
                top_rgb = torch.gather(cand_rgb, 1, selected[:, :, None].expand(-1, -1, 3))
                top_feather = torch.gather(cand_feather, 1, selected)

                finite = torch.isfinite(top_scores)
                best = torch.where(finite[:, :1], top_scores[:, :1], torch.zeros_like(top_scores[:, :1]))
                gap = torch.exp(-(best - top_scores.clamp(max=best)) / config.blend.score_gap_sigma)
                weights = torch.where(finite, torch.clamp(top_scores, min=0) ** config.blend.score_power * top_feather * gap, torch.zeros_like(top_scores))
                denominator = weights.sum(dim=1)
                good = denominator > 0
                rgb = torch.where(good[:, None], (top_rgb * weights[:, :, None]).sum(dim=1) / denominator[:, None].clamp_min(1e-8), torch.zeros((len(xyz), 3), device=target)).clamp(0, 255).to(torch.uint8)
            else:
                good = torch.zeros(len(xyz), dtype=torch.bool, device=target)
                rgb = torch.zeros((len(xyz), 3), dtype=torch.uint8, device=target)
            fallback = valid_cell & ~good & ~surface_fill
            rgb = torch.where(fallback[:, None], cloud_rgb, rgb)
            block = torch.zeros((len(xyz), 4), dtype=torch.uint8, device=target)
            block[:, :3], block[:, 3] = rgb, (good | fallback).to(torch.uint8) * 255
            source = torch.where(good, torch.ones(len(xyz), dtype=torch.uint8, device=target), torch.where(fallback, torch.full((len(xyz),), 2, dtype=torch.uint8, device=target), torch.zeros(len(xyz), dtype=torch.uint8, device=target)))
            count = torch.where(fallback, torch.zeros_like(count), count)
            rgba[y0:y1, x0:x1] = block.reshape(y1 - y0, x1 - x0, 4).cpu().numpy()
            views[y0:y1, x0:x1] = count.reshape(y1 - y0, x1 - x0).cpu().numpy()
            colour_source[y0:y1, x0:x1] = source.reshape(y1 - y0, x1 - x0).cpu().numpy()
            tile_number += 1
            report_progress(progress, 0.25 + 0.75 * tile_number / total_tiles, "Colouring orthophoto tiles")
    return ColourResult(rgba, views, colour_source, missing)


def dsm_ray_occluded(dsm_heights: torch.Tensor, grid: OrthoGrid, surface_xy: torch.Tensor, surface_z: torch.Tensor, camera_local: torch.Tensor, config: OrthoConfig) -> torch.Tensor:
    """Reject outdoor candidates whose DSM rises above the ray from a surface cell to its camera."""
    samples = config.blend.ray_check_samples
    direction = camera_local[:2] - surface_xy
    horizontal = torch.linalg.vector_norm(direction, dim=1).clamp_min(1e-6)
    distance = torch.minimum(horizontal, torch.full_like(horizontal, config.blend.ray_check_distance))
    unit = direction / horizontal[:, None]
    fractions = torch.arange(1, samples + 1, dtype=torch.float32, device=surface_xy.device) / float(samples + 1)
    offsets = distance[:, None] * fractions[None, :]
    sample_xy = surface_xy[:, None, :] + unit[:, None, :] * offsets[:, :, None]
    sample_z = surface_z[:, None] + (camera_local[2] - surface_z)[:, None] * (offsets / horizontal[:, None])
    u = (sample_xy[..., 0] - grid.min_x) / grid.pixel_size - 0.5
    v = (grid.max_y - sample_xy[..., 1]) / grid.pixel_size - 0.5
    gx = (u + 0.5) * 2.0 / grid.width - 1.0
    gy = (v + 0.5) * 2.0 / grid.height - 1.0
    raster = torch.nan_to_num(dsm_heights, nan=-float("inf")).unsqueeze(0).unsqueeze(0)
    sampled = functional.grid_sample(raster, torch.stack((gx, gy), dim=-1).reshape(1, -1, 1, 2), mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0, :, 0].reshape(-1, samples)
    return torch.any(sampled > sample_z + config.blend.ray_check_clearance, dim=1)


def _camera_sees_tile(camera: OrthoCamera, xyz: torch.Tensor, valid: torch.Tensor) -> bool:
    """Conservatively reject cameras whose projected tile box is wholly outside their expanded frustum."""
    selected = xyz[valid]
    if len(selected) == 0:
        return False
    lower, upper = selected.min(dim=0).values, selected.max(dim=0).values
    corners = torch.cartesian_prod(*[torch.stack((lower[index], upper[index])) for index in range(3)])
    points = torch.cat((corners, selected[len(selected) // 2:len(selected) // 2 + 1]), dim=0)
    u, v, depth, _projected = project_points(camera, points)
    pad_x, pad_y = camera.width * 0.1, camera.height * 0.1
    return bool(torch.any((depth > 0) & (u >= -pad_x) & (u < camera.width + pad_x) & (v >= -pad_y) & (v < camera.height + pad_y)).item())


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    clipped = value.clamp(0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _smoothstep_np(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)
