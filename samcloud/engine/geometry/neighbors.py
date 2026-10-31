"""Vectorized voxel-neighbourhood class histograms."""

from __future__ import annotations

from itertools import product

import numpy as np

from .features import _void_keys


def build_voxel_neighbour_index(
    voxel_coords: np.ndarray,
    class_ids: np.ndarray,
    passed_rules: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate valid point labels once for repeated voxel neighbourhood queries."""
    valid = passed_rules & (class_ids > 0) & (class_ids <= class_count)
    if not valid.any():
        return _void_keys(voxel_coords[:0]), np.zeros((0, class_count + 1), dtype=np.int32)
    unique_keys, inverse = np.unique(_void_keys(voxel_coords[valid]), return_inverse=True)
    labels = class_ids[valid].astype(np.int64)
    counts = np.bincount(
        inverse * (class_count + 1) + labels,
        minlength=len(unique_keys) * (class_count + 1),
    ).reshape(len(unique_keys), class_count + 1)
    return unique_keys, counts


def voxel_neighbour_histogram(
    voxel_coords: np.ndarray,
    class_ids: np.ndarray,
    passed_rules: np.ndarray,
    query_indices: np.ndarray,
    class_count: int,
    voxel_radius: int,
    voxel_index: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Count valid class labels in surrounding voxels for selected points.

    Work is vectorized over points; the only loop visits the fixed voxel offsets.
    """
    if not len(query_indices):
        return np.zeros((0, class_count + 1), dtype=np.int32)
    if voxel_index is None:
        voxel_index = build_voxel_neighbour_index(voxel_coords, class_ids, passed_rules, class_count)
    unique_keys, counts = voxel_index
    if not len(unique_keys):
        return np.zeros((len(query_indices), class_count + 1), dtype=np.int32)
    result = np.zeros((len(query_indices), class_count + 1), dtype=np.int32)
    queries = voxel_coords[query_indices]
    for offset in product(range(-voxel_radius, voxel_radius + 1), repeat=3):
        keys = _void_keys(queries + np.asarray(offset, dtype=np.int64))
        positions = np.searchsorted(unique_keys, keys)
        found = positions < len(unique_keys)
        found[found] &= unique_keys[positions[found]] == keys[found]
        if found.any():
            result[found] += counts[positions[found]]
    return result
