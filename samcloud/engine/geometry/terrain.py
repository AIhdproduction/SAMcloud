"""Robust raster terrain estimation and height-above-ground sampling."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _separable_filter(
    grid: np.ndarray,
    radius: int,
    operation,
    padding_value: float,
) -> np.ndarray:
    """Apply a square filter while ignoring positions outside the raster."""
    if radius <= 0:
        return grid.copy()
    size = radius * 2 + 1
    padded_x = np.pad(
        grid,
        ((0, 0), (radius, radius)),
        mode="constant",
        constant_values=padding_value,
    )
    first = operation(sliding_window_view(padded_x, size, axis=1), axis=-1)
    padded_y = np.pad(
        first,
        ((radius, radius), (0, 0)),
        mode="constant",
        constant_values=padding_value,
    )
    return operation(sliding_window_view(padded_y, size, axis=0), axis=-1)


def morphological_opening(grid: np.ndarray, window_cells: int) -> np.ndarray:
    """Remove plateaus narrower than ``window_cells`` while retaining broad terrain."""
    radius = max(0, int(np.ceil(window_cells / 2.0)))
    eroded = _separable_filter(grid, radius, np.min, np.inf)
    return _separable_filter(eroded, radius, np.max, -np.inf)


def fill_empty_cells(grid: np.ndarray, max_iterations: int | None = None) -> np.ndarray:
    """Fill missing raster cells from their valid eight-connected neighbours."""
    result = grid.astype(np.float64, copy=True)
    if not np.isnan(result).any():
        return result
    limit = max_iterations or max(result.shape) * 2
    for _ in range(limit):
        missing = ~np.isfinite(result)
        if not missing.any():
            break
        total = np.zeros_like(result)
        count = np.zeros_like(result, dtype=np.int16)
        valid = np.isfinite(result)
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            source_y = slice(max(0, -dy), result.shape[0] - max(0, dy))
            source_x = slice(max(0, -dx), result.shape[1] - max(0, dx))
            target_y = slice(max(0, dy), result.shape[0] - max(0, -dy))
            target_x = slice(max(0, dx), result.shape[1] - max(0, -dx))
            neighbour_valid = valid[source_y, source_x]
            total[target_y, target_x] += np.where(neighbour_valid, result[source_y, source_x], 0.0)
            count[target_y, target_x] += neighbour_valid
        fill = missing & (count > 0)
        if not fill.any():
            break
        result[fill] = total[fill] / count[fill]
    if not np.isfinite(result).any():
        raise ValueError("No valid terrain cells are available")
    # Any remaining islands are assigned the global robust terrain level.
    result[~np.isfinite(result)] = np.nanmedian(result)
    return result


def build_dtm(
    xyz: np.ndarray,
    ground_mask: np.ndarray,
    cell_size: float,
    percentile: float,
    min_points: int,
    opening_window_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a low-percentile, morphologically opened terrain raster.

    The returned origin is the lower x/y corner of the raster and sampling uses
    cell centres. A percentile rank deliberately skips isolated lowest outliers.
    """
    if cell_size <= 0:
        raise ValueError("terrain.cell_size must be greater than zero")
    if not np.any(ground_mask):
        raise ValueError("No points belong to configured terrain classes")
    origin = np.floor(np.min(xyz[:, :2], axis=0) / cell_size) * cell_size
    indices = np.floor((xyz[:, :2] - origin) / cell_size).astype(np.int64)
    shape = tuple(np.max(indices, axis=0) + 1)
    terrain_indices = indices[ground_mask]
    terrain_z = xyz[ground_mask, 2]
    flat = terrain_indices[:, 1] * shape[0] + terrain_indices[:, 0]
    order = np.argsort(flat, kind="stable")
    flat, terrain_z = flat[order], terrain_z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    ends = np.r_[starts[1:], len(flat)]
    raster_flat = np.full(shape[0] * shape[1], np.nan, dtype=np.float64)
    for start, end in zip(starts, ends):
        values = terrain_z[start:end]
        if len(values) < min_points:
            continue
        # ``ceil`` makes a single very low outlier in a small cell insufficient
        # to set the terrain level, unlike a raw minimum.
        rank = min(len(values) - 1, max(0, int(np.ceil(len(values) * percentile / 100.0))))
        raster_flat[flat[start]] = np.partition(values, rank)[rank]
    raster = raster_flat.reshape((shape[1], shape[0]))
    filled = fill_empty_cells(raster)
    opened = morphological_opening(filled, int(np.ceil(opening_window_m / cell_size)))
    return opened, origin.astype(np.float64), float(cell_size)


def populated_terrain_cells(xyz: np.ndarray, ground_mask: np.ndarray, cell_size: float, min_points: int) -> int:
    """Count raster cells that have enough selected terrain observations."""
    if not np.any(ground_mask):
        return 0
    origin = np.floor(np.min(xyz[:, :2], axis=0) / cell_size) * cell_size
    indices = np.floor((xyz[ground_mask, :2] - origin) / cell_size).astype(np.int64)
    _, counts = np.unique(indices, axis=0, return_counts=True)
    return int(np.count_nonzero(counts >= min_points))


def sample_dtm(dtm: np.ndarray, origin: np.ndarray, cell_size: float, xy: np.ndarray) -> np.ndarray:
    """Bilinearly sample a DTM at point positions, clamping at its boundary."""
    coordinates = (xy - origin) / cell_size - 0.5
    x = np.clip(coordinates[:, 0], 0, dtm.shape[1] - 1)
    y = np.clip(coordinates[:, 1], 0, dtm.shape[0] - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, dtm.shape[1] - 1)
    y1 = np.minimum(y0 + 1, dtm.shape[0] - 1)
    tx, ty = x - x0, y - y0
    return ((1 - tx) * (1 - ty) * dtm[y0, x0] + tx * (1 - ty) * dtm[y0, x1]
            + (1 - tx) * ty * dtm[y1, x0] + tx * ty * dtm[y1, x1])


def height_above_ground(xyz: np.ndarray, dtm: np.ndarray, origin: np.ndarray, cell_size: float) -> np.ndarray:
    """Return per-point height above the robust terrain surface."""
    return xyz[:, 2].astype(np.float64) - sample_dtm(dtm, origin, cell_size, xyz[:, :2])
