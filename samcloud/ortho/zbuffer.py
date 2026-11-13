"""GPU-capable reduced camera depth buffers from the full classified cloud."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from samcloud.ortho.cameras import OrthoCamera, project_points, scaled
from samcloud.ortho.config import OrthoConfig


def voxel_downsample(points: torch.Tensor, voxel: float) -> torch.Tensor:
    """Keep one deterministic point per voxel without requiring optional point-cloud libraries."""
    keys = torch.floor(points / voxel).to(torch.int64)
    # torch.unique lacks return_index in the installed API; use a CPU key pass only for the compact index list.
    key_cpu = keys.detach().cpu().numpy()
    _, indices = np.unique(key_cpu, axis=0, return_index=True)
    return points[torch.as_tensor(indices, device=points.device, dtype=torch.long)]


def build_zbuffer(camera: OrthoCamera, full_xyz: torch.Tensor, config: OrthoConfig, *, indoor: bool) -> tuple[torch.Tensor, float]:
    """Splat full geometry and erode only at real reduced-image depth discontinuities."""
    factor = min(1.0, config.zbuffer.max_long_side / max(camera.width, camera.height))
    reduced = scaled(camera, factor)
    u, v, depth, valid = project_points(reduced, full_xyz)
    h, w = reduced.height, reduced.width
    buffer = torch.full((h * w,), float("inf"), dtype=full_xyz.dtype, device=full_xyz.device)
    if torch.any(valid):
        col = torch.floor(u[valid]).to(torch.long)
        row = torch.floor(v[valid]).to(torch.long)
        voxel = config.zbuffer.voxel_indoor if indoor else config.zbuffer.voxel_outdoor
        radius = torch.clamp(torch.round(voxel * reduced.params[0] / depth[valid]).to(torch.long), 2, 4)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                use = (radius >= max(abs(dx), abs(dy)))
                if not torch.any(use):
                    continue
                rr, cc = row[use] + dy, col[use] + dx
                inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                if torch.any(inside):
                    buffer.scatter_reduce_(0, rr[inside] * w + cc[inside], depth[valid][use][inside], reduce="amin", include_self=True)
    result = buffer.reshape(h, w)
    if config.zbuffer.min_filter_px > 0:
        padding = config.zbuffer.min_filter_px
        finite = torch.isfinite(result)
        nearest = -functional.max_pool2d(torch.where(finite, -result, torch.full_like(result, float("-inf"))).unsqueeze(0).unsqueeze(0), 2 * padding + 1, stride=1, padding=padding)[0, 0]
        farthest = functional.max_pool2d(torch.where(finite, result, torch.full_like(result, float("-inf"))).unsqueeze(0).unsqueeze(0), 2 * padding + 1, stride=1, padding=padding)[0, 0]
        voxel = config.zbuffer.voxel_indoor if indoor else config.zbuffer.voxel_outdoor
        focal = max(float(reduced.params[0]), 1.0)
        smooth_surface_range = (2 * padding + 1) * farthest / focal + config.zbuffer.discontinuity_voxels * voxel
        discontinuity = finite & torch.isfinite(nearest) & torch.isfinite(farthest) & ((farthest - nearest) > smooth_surface_range)
        result = torch.where(discontinuity, nearest, result)
    return result, factor


def visible_depth(depth: torch.Tensor, sampled_z: torch.Tensor, voxel: float, config: OrthoConfig, cos_incidence: torch.Tensor, focal_length: float) -> torch.Tensor:
    """Return visibility with a surface-footprint tolerance rather than a fixed depth percentage."""
    cosine = torch.clamp(cos_incidence.abs(), min=0.05, max=1.0)
    tangent = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0)) / cosine
    footprint = sampled_z / max(float(focal_length), 1.0)
    tolerance = config.zbuffer.abs_tol_voxels * voxel + config.zbuffer.footprint_tolerance_scale * footprint * tangent
    return torch.isfinite(sampled_z) & (depth <= sampled_z + tolerance)
