"""One-pass voxel local-geometry features."""

from __future__ import annotations

import numpy as np


def voxel_coordinates(xyz: np.ndarray, voxel_size: float, origin: np.ndarray | None = None) -> np.ndarray:
    """Return integer voxel coordinates with an optional shared origin."""
    if voxel_size <= 0:
        raise ValueError("features.voxel_size must be greater than zero")
    base = np.min(xyz, axis=0) if origin is None else origin
    return np.floor((xyz - base) / voxel_size).astype(np.int64)


def _void_keys(coordinates: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(coordinates)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


def compute_voxel_features(xyz: np.ndarray, voxel_size: float, min_points: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compute normal orientation, planarity and roughness once per occupied voxel."""
    origin = np.min(xyz, axis=0).astype(np.float64)
    coordinates = voxel_coordinates(xyz, voxel_size, origin)
    _, inverse = np.unique(_void_keys(coordinates), return_inverse=True)
    voxel_count = int(inverse.max()) + 1 if len(inverse) else 0
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)
    sums = np.stack([np.bincount(inverse, weights=xyz[:, axis], minlength=voxel_count) for axis in range(3)], axis=1)
    means = sums / counts[:, None]
    moments = np.empty((voxel_count, 3, 3), dtype=np.float64)
    for row in range(3):
        for col in range(row, 3):
            values = np.bincount(inverse, weights=xyz[:, row] * xyz[:, col], minlength=voxel_count)
            moments[:, row, col] = moments[:, col, row] = values / counts - means[:, row] * means[:, col]
    valid = counts >= min_points
    normal_z_voxel = np.full(voxel_count, np.nan)
    planarity_voxel = np.full(voxel_count, np.nan)
    roughness_voxel = np.full(voxel_count, np.nan)
    linearity_voxel = np.full(voxel_count, np.nan)
    if valid.any():
        eigenvalues, eigenvectors = np.linalg.eigh(moments[valid])
        smallest = np.clip(eigenvalues[:, 0], 0.0, None)
        largest = eigenvalues[:, 2]
        normal_z_voxel[valid] = np.abs(eigenvectors[:, 2, 0])
        planarity_voxel[valid] = np.divide(
            eigenvalues[:, 1] - eigenvalues[:, 0], largest,
            out=np.zeros_like(largest), where=largest > 1e-12,
        )
        linearity_voxel[valid] = np.divide(
            largest - eigenvalues[:, 1], largest,
            out=np.zeros_like(largest), where=largest > 1e-12,
        )
        roughness_voxel[valid] = np.sqrt(smallest)
    return {
        "normal_z": normal_z_voxel[inverse],
        "planarity": planarity_voxel[inverse],
        "linearity": linearity_voxel[inverse],
        "roughness": roughness_voxel[inverse],
    }, coordinates, origin
