"""Shared in-memory quality filter for COLMAP geometric depth maps."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from samcloud.colmap.plane_fill import View


DEFAULTS = {
    "enabled": True,
    "max_incidence_deg": 80.0,
    "edge_px": 1,
    "edge_jump_rel": 0.05,
    "edge_jump_m": 0.03,
    "drop_lonely": False,
}


def _settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    values = settings.get("depth_edge_filter") if settings and "depth_edge_filter" in settings else {"enabled": False}
    return {**DEFAULTS, **(values or {})}


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            source_y = slice(max(0, -dy), mask.shape[0] - max(0, dy))
            source_x = slice(max(0, -dx), mask.shape[1] - max(0, dx))
            target_y = slice(max(0, dy), mask.shape[0] - max(0, -dy))
            target_x = slice(max(0, dx), mask.shape[1] - max(0, -dx))
            result[target_y, target_x] |= mask[source_y, source_x]
    return result


def geometric_valid_mask(depth: np.ndarray, view: View, settings: dict[str, Any] | None = None) -> np.ndarray:
    """Return usable geometric-depth pixels without modifying ``depth``."""
    active = _settings(settings)
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0)
    if not bool(active["enabled"]):
        return valid
    jump = np.zeros(values.shape, dtype=bool)
    absolute = float(active["edge_jump_m"]) * float((settings or {}).get("units_per_metre", 1.0))
    relative = float(active["edge_jump_rel"])
    for first, second, target_first, target_second in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        a, b = values[first], values[second]
        pair = valid[first] & valid[second]
        difference = np.abs(a - b)
        pair &= (difference > absolute) & (difference / np.maximum(np.minimum(a, b), 1e-6) > relative)
        jump[target_first] |= pair
        jump[target_second] |= pair
    edge = _dilate(jump, max(0, int(active["edge_px"]))) & valid
    isolated = np.zeros(values.shape, dtype=bool)
    neighbours = np.zeros(values.shape, dtype=np.int8)
    similar = np.zeros(values.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            source_y = slice(max(0, -dy), values.shape[0] - max(0, dy))
            source_x = slice(max(0, -dx), values.shape[1] - max(0, dx))
            target_y = slice(max(0, dy), values.shape[0] - max(0, -dy))
            target_x = slice(max(0, dx), values.shape[1] - max(0, -dx))
            pair = valid[target_y, target_x] & valid[source_y, source_x]
            neighbours[target_y, target_x] += pair
            difference = np.abs(values[target_y, target_x] - values[source_y, source_x])
            similar[target_y, target_x] |= pair & (difference / np.maximum(values[target_y, target_x], 1e-6) <= relative)
    isolated = valid & ~similar & ((neighbours > 0) | bool(active["drop_lonely"]))
    incidence = _incidence_mask(values, view, float(active["max_incidence_deg"]), absolute, relative)
    return valid & ~edge & ~isolated & incidence


def _incidence_mask(depth: np.ndarray, view: View, maximum: float,
                    absolute: float = 0.03, relative: float = 0.05) -> np.ndarray:
    """True where incidence is acceptable or no reliable normal exists (unknown != grazing).

    Only central differences (both neighbours valid, no depth jump) count as reliable; one-sided
    differences next to holes are too noisy at 1 px baseline and are treated as unknown.
    """
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    safe = np.where(valid, depth, 0.0).astype(np.float32)
    cols, rows = np.meshgrid(np.arange(width, dtype=np.float32) + 0.5, np.arange(height, dtype=np.float32) + 0.5)
    points = np.stack(((cols * view.width / width - view.cx) / view.fx * safe,
                       (rows * view.height / height - view.cy) / view.fy * safe, safe), axis=-1)

    def tangent(axis: int):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        a_d, b_d = safe[first], safe[second]
        difference = np.abs(a_d - b_d)
        link_jump = (difference > absolute) & (difference / np.maximum(np.minimum(a_d, b_d), 1e-6) > relative)
        pair = valid[first] & valid[second] & ~link_jump
        forward_ok = np.zeros(depth.shape, dtype=bool)
        backward_ok = np.zeros(depth.shape, dtype=bool)
        forward_ok[first] = pair
        backward_ok[second] = pair
        step = points[second] - points[first]
        forward = np.zeros_like(points)
        backward = np.zeros_like(points)
        forward[first] = step
        backward[second] = step
        both = forward_ok & backward_ok
        vector = np.where(both[..., None], 0.5 * (forward + backward),
                          np.where(forward_ok[..., None], forward, backward))
        return vector, both

    du, ok_u = tangent(1)
    dv, ok_v = tangent(0)
    normal = np.cross(du, dv)
    normal_len = np.linalg.norm(normal, axis=-1)
    ray_len = np.linalg.norm(points, axis=-1)
    reliable = ok_u & ok_v & (normal_len > 1e-12) & (ray_len > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.abs(np.sum(normal * points, axis=-1)) / (normal_len * ray_len)
    return ~reliable | (np.nan_to_num(cosine) >= math.cos(math.radians(maximum)))


def geometric_valid_mask_torch(depth, view: View, settings: dict[str, Any] | None = None):
    """Torch counterpart of :func:`geometric_valid_mask`, kept on the active device."""
    import torch
    import torch.nn.functional as F

    active = _settings(settings)
    valid = torch.isfinite(depth) & (depth > 0)
    if not bool(active["enabled"]):
        return valid
    absolute = float(active["edge_jump_m"]) * float((settings or {}).get("units_per_metre", 1.0))
    relative = float(active["edge_jump_rel"])
    jump = torch.zeros_like(valid)
    right_valid = valid[:, :-1] & valid[:, 1:]
    right_difference = (depth[:, :-1] - depth[:, 1:]).abs()
    right = right_valid & (right_difference > absolute) & (right_difference / torch.minimum(depth[:, :-1], depth[:, 1:]).clamp(min=1e-6) > relative)
    jump[:, :-1] |= right
    jump[:, 1:] |= right
    down_valid = valid[:-1, :] & valid[1:, :]
    down_difference = (depth[:-1, :] - depth[1:, :]).abs()
    down = down_valid & (down_difference > absolute) & (down_difference / torch.minimum(depth[:-1, :], depth[1:, :]).clamp(min=1e-6) > relative)
    jump[:-1, :] |= down
    jump[1:, :] |= down
    radius = max(0, int(active["edge_px"]))
    edge = (F.max_pool2d(jump.float()[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0] > 0 if radius else jump) & valid
    neighbours = torch.zeros_like(depth, dtype=torch.int8)
    similar = torch.zeros_like(valid)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            source_y = slice(max(0, -dy), depth.shape[0] - max(0, dy))
            source_x = slice(max(0, -dx), depth.shape[1] - max(0, dx))
            target_y = slice(max(0, dy), depth.shape[0] - max(0, -dy))
            target_x = slice(max(0, dx), depth.shape[1] - max(0, -dx))
            pair = valid[target_y, target_x] & valid[source_y, source_x]
            neighbours[target_y, target_x] += pair.to(torch.int8)
            difference = (depth[target_y, target_x] - depth[source_y, source_x]).abs()
            similar[target_y, target_x] |= pair & (difference / depth[target_y, target_x].clamp(min=1e-6) <= relative)
    isolated = valid & ~similar & ((neighbours > 0) | bool(active["drop_lonely"]))
    h, w = depth.shape
    safe = torch.where(valid, depth, torch.zeros_like(depth))
    cols = (torch.arange(w, device=depth.device, dtype=torch.float32) + 0.5) * view.width / w
    rows = (torch.arange(h, device=depth.device, dtype=torch.float32) + 0.5) * view.height / h
    points = torch.stack((((cols[None, :] - view.cx) / view.fx) * safe,
                          ((rows[:, None] - view.cy) / view.fy) * safe, safe), dim=-1)

    def tangent(axis: int):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        a_d, b_d = safe[first], safe[second]
        difference = (a_d - b_d).abs()
        link_jump = (difference > absolute) & (difference / torch.minimum(a_d, b_d).clamp(min=1e-6) > relative)
        pair = valid[first] & valid[second] & ~link_jump
        forward_ok = torch.zeros_like(valid)
        backward_ok = torch.zeros_like(valid)
        forward_ok[first] = pair
        backward_ok[second] = pair
        step = points[second] - points[first]
        forward = torch.zeros_like(points)
        backward = torch.zeros_like(points)
        forward[first] = step
        backward[second] = step
        both = forward_ok & backward_ok
        vector = torch.where(both[..., None], 0.5 * (forward + backward),
                             torch.where(forward_ok[..., None], forward, backward))
        return vector, both

    du, ok_u = tangent(1)
    dv, ok_v = tangent(0)
    normal = torch.cross(du, dv, dim=-1)
    normal_len = normal.norm(dim=-1)
    ray_len = points.norm(dim=-1)
    reliable = ok_u & ok_v & (normal_len > 1e-12) & (ray_len > 1e-12)
    cosine = (normal * points).sum(-1).abs() / (normal_len * ray_len).clamp(min=1e-12)
    incidence = ~reliable | (cosine >= math.cos(math.radians(float(active["max_incidence_deg"]))))
    return valid & ~edge & ~isolated & incidence
