"""Local horizontal support estimation for vegetation and roof surfaces."""

from __future__ import annotations

import numpy as np

from .terrain import fill_empty_cells, morphological_opening, sample_dtm


def build_local_support(
    xyz: np.ndarray,
    support_mask: np.ndarray,
    cell_size: float,
    percentile: float,
    min_points: int,
    window_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rasterise low horizontal support surfaces and bridge small empty gaps."""
    if cell_size <= 0:
        raise ValueError("support.cell_size must be greater than zero")
    if not np.any(support_mask):
        raise ValueError("No horizontal support points are available")
    origin = np.floor(np.min(xyz[:, :2], axis=0) / cell_size) * cell_size
    indices = np.floor((xyz[:, :2] - origin) / cell_size).astype(np.int64)
    shape = tuple(np.max(indices, axis=0) + 1)
    selected = indices[support_mask]
    flat = selected[:, 1] * shape[0] + selected[:, 0]
    values = xyz[support_mask, 2]
    order = np.argsort(flat, kind="stable")
    flat, values = flat[order], values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    ends = np.r_[starts[1:], len(flat)]
    raster_flat = np.full(shape[0] * shape[1], np.nan, dtype=np.float64)
    for start, end in zip(starts, ends):
        cell_values = values[start:end]
        if len(cell_values) < min_points:
            continue
        rank = min(len(cell_values) - 1, max(0, int(np.ceil(len(cell_values) * percentile / 100.0))))
        raster_flat[flat[start]] = np.partition(cell_values, rank)[rank]
    raster = fill_empty_cells(raster_flat.reshape((shape[1], shape[0])))
    opened = morphological_opening(raster, int(np.ceil(window_m / cell_size)))
    return opened, origin.astype(np.float64), float(cell_size)


def height_above_support(
    xyz: np.ndarray, support: np.ndarray, origin: np.ndarray, cell_size: float
) -> np.ndarray:
    """Return each point's height above its locally horizontal support surface."""
    return xyz[:, 2].astype(np.float64) - sample_dtm(support, origin, cell_size, xyz[:, :2])
