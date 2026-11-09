"""Streaming voxel preview generation for classified LAS files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np

from samcloud.dxf.cancel import CancelToken, ProgressCallback, check_cancel, report_progress
from samcloud.dxf.config import DxfExportConfig, load_dxf_config
from samcloud.dxf.las_source import Frame, LasInfo, iter_las_chunks


@dataclass(frozen=True, slots=True)
class PreviewCloud:
    """Voxel-reduced local point cloud for interaction and explicit analysis actions."""

    xyz: np.ndarray
    cls: np.ndarray
    rgb: np.ndarray | None
    voxel_size: float


@dataclass(slots=True)
class _VoxelSamples:
    """Sorted voxel keys and their first source-point representatives."""

    keys: np.ndarray
    xyz: np.ndarray
    cls: np.ndarray
    rgb: np.ndarray | None


def create_preview(
    path: Path,
    frame: Frame,
    *,
    info: LasInfo | None = None,
    max_points: int | None = None,
    voxel_size: float | None = None,
    chunk_size: int | None = None,
    config: DxfExportConfig | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> PreviewCloud:
    """Create a bounded voxel preview, preserving one class and RGB sample per voxel."""
    settings = config or load_dxf_config()
    limit = settings.preview.max_points if max_points is None else max_points
    if limit < 1:
        raise ValueError("Maximum preview point count must be at least 1")
    source_path = Path(path)
    check_cancel(cancel)
    if info is None:
        with laspy.open(source_path) as reader:
            header = reader.header
            extent = np.asarray(header.maxs, dtype=np.float64) - np.asarray(header.mins, dtype=np.float64)
            point_count = int(header.point_count)
            has_rgb = all(name in header.point_format.dimension_names for name in ("red", "green", "blue"))
    else:
        extent = info.extent
        point_count = info.point_count
        has_rgb = info.has_rgb
    chosen_chunk_size = settings.loading.chunk_size if chunk_size is None else chunk_size
    initial_voxel = voxel_size if voxel_size is not None else settings.preview.voxel_size
    if initial_voxel is None:
        initial_voxel = _estimate_voxel_size(extent, point_count, limit, frame)
    if initial_voxel <= 0:
        raise ValueError("Preview voxel size must be positive")
    capacity_voxel = _capacity_voxel_size(extent, limit, frame)
    current_voxel = max(float(initial_voxel), capacity_voxel)
    samples = _voxel_samples(source_path, frame, current_voxel, chosen_chunk_size, has_rgb, progress, cancel)
    if len(samples.keys) > limit:
        samples = _limit_samples(samples, limit)
    return _as_preview(samples, has_rgb, current_voxel)


def _voxel_samples(
    path: Path,
    frame: Frame,
    voxel_size: float,
    chunk_size: int,
    has_rgb: bool,
    progress: ProgressCallback | None,
    cancel: CancelToken | None,
) -> _VoxelSamples:
    samples = _empty_samples(has_rgb)
    for chunk in iter_las_chunks(path, chunk_size, progress=progress, cancel=cancel, stage="Voxel-downsampling preview"):
        check_cancel(cancel)
        original = np.column_stack((chunk.x, chunk.y, chunk.z))
        local = frame.to_local(original)
        classes = np.asarray(chunk.classification, dtype=np.uint8)
        rgb = _chunk_rgb(chunk) if has_rgb else None
        keys = np.floor(local / voxel_size).astype(np.int64)
        unique_keys, indices = np.unique(keys, axis=0, return_index=True)
        check_cancel(cancel)
        samples = _merge_samples(samples, unique_keys, local[indices].astype(np.float32), classes[indices], None if rgb is None else rgb[indices])
        check_cancel(cancel)
    return samples


def _chunk_rgb(chunk: object) -> np.ndarray:
    return np.column_stack((np.asarray(chunk.red), np.asarray(chunk.green), np.asarray(chunk.blue))).astype(np.uint16) // 257


def _as_preview(
    samples: _VoxelSamples,
    has_rgb: bool,
    voxel_size: float,
) -> PreviewCloud:
    if len(samples.keys) == 0:
        return PreviewCloud(np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint8), None if not has_rgb else np.empty((0, 3), dtype=np.uint8), voxel_size)
    return PreviewCloud(samples.xyz, samples.cls, samples.rgb, voxel_size)


def _estimate_voxel_size(extent: np.ndarray, point_count: int, limit: int, frame: Frame) -> float:
    """Estimate a voxel edge from header bounds and point count before streaming."""
    local_extent = _local_extent_bound(extent, frame)
    target_count = max(1, min(point_count, limit))
    return float((np.prod(np.maximum(local_extent, 1e-6)) / target_count) ** (1.0 / 3.0))


def _capacity_voxel_size(extent: np.ndarray, limit: int, frame: Frame) -> float:
    """Return a conservative voxel size whose bounding grid fits the limit."""
    local_extent = _local_extent_bound(extent, frame)
    target = max(limit, 8)
    lower = 0.0
    upper = max(float(np.max(local_extent)), 1e-6)
    while _voxel_capacity(local_extent, upper) > target:
        upper *= 2.0
    for _ in range(48):
        middle = (lower + upper) / 2.0
        if _voxel_capacity(local_extent, middle) <= target:
            upper = middle
        else:
            lower = middle
    return upper


def _local_extent_bound(extent: np.ndarray, frame: Frame) -> np.ndarray:
    """Bound aligned local extents without reading the source points again."""
    source_extent = np.maximum(np.asarray(extent, dtype=np.float64), 0.0) * frame.scale_factor
    return np.maximum(source_extent @ np.abs(frame.rotation).T, 1e-9)


def _voxel_capacity(extent: np.ndarray, voxel_size: float) -> int:
    capacity = 1
    for axis_extent in extent:
        capacity *= int(np.ceil(axis_extent / voxel_size)) + 2
    return capacity


def _empty_samples(has_rgb: bool) -> _VoxelSamples:
    return _VoxelSamples(
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3), dtype=np.float32),
        np.empty(0, dtype=np.uint8),
        np.empty((0, 3), dtype=np.uint8) if has_rgb else None,
    )


def _merge_samples(
    existing: _VoxelSamples,
    keys: np.ndarray,
    xyz: np.ndarray,
    classes: np.ndarray,
    rgb: np.ndarray | None,
) -> _VoxelSamples:
    """Merge sorted chunk keys without a Python loop over individual points."""
    if len(existing.keys) == 0:
        return _VoxelSamples(keys, xyz, classes, rgb)
    existing_keys = _structured_keys(existing.keys)
    chunk_keys = _structured_keys(keys)
    positions = np.searchsorted(existing_keys, chunk_keys)
    in_bounds = positions < len(existing_keys)
    present = np.zeros(len(keys), dtype=bool)
    present[in_bounds] = existing_keys[positions[in_bounds]] == chunk_keys[in_bounds]
    if np.all(present):
        return existing
    new_keys, new_xyz, new_classes = keys[~present], xyz[~present], classes[~present]
    new_rgb = None if rgb is None else rgb[~present]
    insert_at = positions[~present]
    total = len(existing.keys) + len(new_keys)
    merged_keys = np.empty((total, 3), dtype=np.int64)
    merged_xyz = np.empty((total, 3), dtype=np.float32)
    merged_classes = np.empty(total, dtype=np.uint8)
    existing_positions = np.arange(len(existing.keys)) + np.searchsorted(insert_at, np.arange(len(existing.keys)), side="right")
    new_positions = insert_at + np.arange(len(new_keys))
    merged_keys[existing_positions], merged_keys[new_positions] = existing.keys, new_keys
    merged_xyz[existing_positions], merged_xyz[new_positions] = existing.xyz, new_xyz
    merged_classes[existing_positions], merged_classes[new_positions] = existing.cls, new_classes
    if existing.rgb is None:
        merged_rgb = None
    else:
        merged_rgb = np.empty((total, 3), dtype=np.uint8)
        merged_rgb[existing_positions], merged_rgb[new_positions] = existing.rgb, new_rgb
    return _VoxelSamples(merged_keys, merged_xyz, merged_classes, merged_rgb)


def _structured_keys(keys: np.ndarray) -> np.ndarray:
    dtype = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    return np.ascontiguousarray(keys).view(dtype).reshape(-1)


def _limit_samples(samples: _VoxelSamples, limit: int) -> _VoxelSamples:
    """Keep deterministic, spatially distributed sorted-key representatives."""
    indices = np.linspace(0, len(samples.keys) - 1, limit, dtype=np.intp)
    return _VoxelSamples(
        samples.keys[indices],
        samples.xyz[indices],
        samples.cls[indices],
        None if samples.rgb is None else samples.rgb[indices],
    )
