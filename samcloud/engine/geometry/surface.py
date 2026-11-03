"""Surface roughness/colour rules and neighbourhood smoothing for terrain-level classes."""

from __future__ import annotations

import numpy as np


def _cell_roughness(xyz: np.ndarray, mask: np.ndarray, cell: float, offset: float, min_points: int) -> np.ndarray:
    rough = np.full(len(xyz), np.nan)
    if not mask.any():
        return rough
    p = xyz[mask]
    origin = p[:, :2].min(0) - offset
    ij = np.floor((p[:, :2] - origin) / cell).astype(np.int64)
    key = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    centre = (ij + 0.5) * cell + origin
    x, y, z = p[:, 0] - centre[:, 0], p[:, 1] - centre[:, 1], p[:, 2]

    def total(values: np.ndarray) -> np.ndarray:
        return np.bincount(inv, values, minlength=len(uniq))

    n = counts.astype(float)
    matrix = np.stack([
        np.stack([total(x * x), total(x * y), total(x)], -1),
        np.stack([total(x * y), total(y * y), total(y)], -1),
        np.stack([total(x), total(y), n], -1),
    ], 1) + np.eye(3) * 1e-6
    rhs = np.stack([total(x * z), total(y * z), total(z)], -1)
    coef = np.linalg.solve(matrix, rhs[..., None])[..., 0]
    residual = z - (coef[inv, 0] * x + coef[inv, 1] * y + coef[inv, 2])
    std = np.sqrt(total(residual * residual) / n)
    std[counts < min_points] = np.nan
    rough[np.flatnonzero(mask)] = std[inv]
    return rough


def surface_roughness(xyz: np.ndarray, mask: np.ndarray, cell: float = 0.5, min_points: int = 10) -> np.ndarray:
    """Per-point std of residuals to a local plane, averaged over two half-cell shifted grids (less blocky)."""
    first = _cell_roughness(xyz, mask, cell, 0.0, min_points)
    second = _cell_roughness(xyz, mask, cell, cell / 2.0, min_points)
    return np.nanmean(np.stack([first, second]), axis=0)


def excess_green(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    return (2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]) / np.maximum(rgb.sum(1), 1.0)


def apply_surface_rule(
    class_ids: np.ndarray, xyz: np.ndarray, rgb: np.ndarray | None, hag: np.ndarray,
    name_to_id: dict[str, int], config: dict,
) -> tuple[np.ndarray, dict[str, int]]:
    """Lawn at terrain level -> ground, rough green -> low vegetation, rough soil -> ground."""
    out = class_ids.copy()
    stats = {"to_low_vegetation": 0, "to_ground": 0}
    if not config.get("enabled", True) or "ground" not in name_to_id or "low vegetation" not in name_to_id:
        return out, stats
    ground, vegetation = name_to_id["ground"], name_to_id["low vegetation"]
    classes = [name_to_id[name] for name in config.get("classes", ["ground", "low vegetation", "sidewalk"]) if name in name_to_id]
    low = np.isfinite(hag) & (hag < float(config.get("max_hag", 2.5)))
    rough = surface_roughness(xyz, low, float(config.get("cell_size", 0.5)), int(config.get("min_points", 10)))
    green = excess_green(rgb) > float(config.get("green_index_min", 0.05)) if rgb is not None else np.ones(len(out), bool)
    candidates = np.isin(class_ids, classes) & low & np.isfinite(rough)
    rough_limit = float(config.get("rough_vegetation_m", 0.025))
    to_vegetation = candidates & (rough > rough_limit) & green
    to_ground = candidates & (class_ids == vegetation) & (
        ((rough > rough_limit) & ~green)
        | ((rough <= float(config.get("smooth_ground_m", 0.025))) & (hag <= float(config.get("lawn_max_hag", 0.25))))
    )
    out[to_vegetation] = vegetation
    out[to_ground] = ground
    stats["to_low_vegetation"] = int(np.count_nonzero(to_vegetation & (class_ids != vegetation)))
    stats["to_ground"] = int(np.count_nonzero(to_ground))
    return out, stats


def neighbourhood_majority(
    xyz: np.ndarray, classes: np.ndarray, n_classes: int, *, voxel: float = 0.3, iterations: int = 2,
    own_max_share: float = 0.25, majority_min_share: float = 0.6, protected: tuple[int, ...] = (),
) -> np.ndarray:
    """Relabel isolated points to the clear majority of their 3x3x3 voxel neighbourhood."""
    if len(xyz) == 0:
        return classes.copy()
    keys3 = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    dims = keys3.max(0) + 3

    def key(values: np.ndarray) -> np.ndarray:
        return (values[:, 0] + 1) * dims[1] * dims[2] + (values[:, 1] + 1) * dims[2] + (values[:, 2] + 1)

    uniq, inv = np.unique(key(keys3), return_inverse=True)
    ukeys3 = np.column_stack(np.unravel_index(uniq, (int(dims[0]), int(dims[1]), int(dims[2])))) - 1
    offsets = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
    neighbours = []
    for offset in offsets:
        wanted = key(ukeys3 + offset)
        position = np.clip(np.searchsorted(uniq, wanted), 0, len(uniq) - 1)
        neighbours.append(np.where(uniq[position] == wanted, position, -1))
    neighbours = np.stack(neighbours, 1)
    out = classes.astype(np.int64).copy()
    protected_ids = np.asarray(protected, dtype=np.int64)
    locked = np.isin(out, protected_ids)
    for _ in range(max(1, iterations)):
        counts = np.zeros((len(uniq), n_classes), np.int32)
        np.add.at(counts, (inv, out), 1)
        totals = np.zeros_like(counts)
        for column in range(neighbours.shape[1]):
            valid = neighbours[:, column] >= 0
            totals[valid] += counts[neighbours[valid, column]]
        sums = totals.sum(1)
        major = totals.argmax(1)
        share = totals.max(1) / np.maximum(sums, 1)
        own_share = totals[inv, out] / np.maximum(sums[inv], 1)
        change = (~locked) & (own_share < own_max_share) & (share[inv] >= majority_min_share) & ~np.isin(major[inv], protected_ids)
        out[change] = major[inv][change]
    return out.astype(classes.dtype)
