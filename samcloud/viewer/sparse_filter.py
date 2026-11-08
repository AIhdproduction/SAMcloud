"""Remove stray points from the sparse (COLMAP) cloud before it is shown.

The sparse model itself stays untouched: dense reconstruction, georeferencing
and control points keep using every tie point. Only the viewer and the preview
show the cleaned version. A tie point is kept when

* it was seen in at least ``min_track_length`` images (two-view points are
  the main source of scattered noise),
* its reprojection error is at most ``max_reprojection_error`` pixels,
* its rays meet at ``min_triangulation_angle_deg`` or more (narrow angles give
  points smeared along the viewing ray),
* it is not much farther from the nearest camera than almost every other
  point (``max_camera_distance_factor`` x the 95th percentile), which drops
  the far streaks seen through windows or into the sky, and
* it has at least ``min_neighbors`` other points around it.

Clouds below ``min_points`` are shown unfiltered, and so is any cloud the
filter would cut below ``min_keep_ratio``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from samcloud.core.configuration import load_global_config

FILTER_VERSION = 1
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "min_track_length": 3,
    "max_reprojection_error": 1.5,
    "min_triangulation_angle_deg": 3.0,
    "max_camera_distance_factor": 2.0,
    "min_neighbors": 4,
    # Tiny models (tests, failed runs) are shown as they are; a filter that would
    # hide most of a cloud is treated as misconfigured and skipped.
    "min_points": 1000,
    "min_keep_ratio": 0.2,
}


def sparse_cleaning_settings() -> dict[str, Any]:
    """Defaults merged with ``viewer.sparse_cleaning`` from ``config/defaults.yml``."""
    settings = dict(DEFAULTS)
    try:
        configured = load_global_config("defaults.yml").get("viewer", {}).get("sparse_cleaning", {})
    except (OSError, ValueError, AttributeError):
        configured = {}
    if isinstance(configured, dict):
        settings.update({key: value for key, value in configured.items() if key in DEFAULTS})
    return settings


def _read_points_binary(points_path: Path) -> dict[str, np.ndarray]:
    """Parse ``points3D.bin`` without needing cameras/images (fallback path)."""
    xyz, rgb, error, track, owners, image_ids = [], [], [], [], [], []
    with Path(points_path).open("rb") as file:
        total = struct.unpack("<Q", file.read(8))[0]
        for index in range(total):
            _point_id, x, y, z, red, green, blue, err = struct.unpack("<QdddBBBd", file.read(43))
            length = struct.unpack("<Q", file.read(8))[0]
            elements = np.frombuffer(file.read(length * 8), dtype="<i4").reshape(-1, 2) if length else np.zeros((0, 2), "<i4")
            xyz.append((x, y, z))
            rgb.append((red, green, blue))
            error.append(err)
            track.append(length)
            owners.append(np.full(len(elements), index, np.int64))
            image_ids.append(elements[:, 0].astype(np.int64))
    return {
        "xyz": np.asarray(xyz, float).reshape(-1, 3),
        "rgb": np.asarray(rgb, np.uint8).reshape(-1, 3),
        "error": np.asarray(error, float),
        "track": np.asarray(track, np.int64),
        "owner": np.concatenate(owners) if owners else np.zeros(0, np.int64),
        "image_id": np.concatenate(image_ids) if image_ids else np.zeros(0, np.int64),
    }


def _camera_centers(model_directory: Path) -> dict[int, np.ndarray]:
    try:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(model_directory)
        return {int(image_id): np.asarray(image.projection_center(), float) for image_id, image in reconstruction.images.items()}
    except Exception:  # missing/partial model files: filter without camera geometry
        return {}


def triangulation_angles(xyz: np.ndarray, owner: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Approximate largest ray angle per point in degrees (2 x max angle to the mean ray)."""
    angles = np.zeros(len(xyz))
    if not len(owner):
        return angles
    rays = centers - xyz[owner]
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    mean = np.zeros_like(xyz)
    np.add.at(mean, owner, rays)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    spread = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", rays, mean[owner]), -1.0, 1.0)))
    np.maximum.at(angles, owner, spread)
    return np.minimum(2.0 * angles, 180.0)


def nearest_camera_distance(xyz: np.ndarray, centers: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    unique = np.unique(np.round(centers, 3), axis=0)
    best = np.full(len(xyz), np.inf)
    for start in range(0, len(xyz), chunk):
        part = xyz[start:start + chunk]
        for center in unique:
            np.minimum(best[start:start + chunk], np.linalg.norm(part - center, axis=1), out=best[start:start + chunk])
    return best


def _voxel_neighbor_counts(xyz: np.ndarray, cell: float) -> np.ndarray:
    cells = np.floor((xyz - xyz.min(axis=0)) / cell).astype(np.int64)
    span = cells.max(axis=0) + 3
    key = ((cells[:, 0] + 1) * span[1] + (cells[:, 1] + 1)) * span[2] + (cells[:, 2] + 1)
    unique, counts = np.unique(key, return_counts=True)
    total = np.zeros(len(xyz), np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                probe = key + (dx * span[1] + dy) * span[2] + dz
                position = np.clip(np.searchsorted(unique, probe), 0, len(unique) - 1)
                total += np.where(unique[position] == probe, counts[position], 0)
    return total - 1


def neighbor_counts(xyz: np.ndarray, typical: int = 8) -> np.ndarray:
    """Other points in the surrounding 3x3x3 voxels.

    The voxel starts at the mean 3D spacing and grows until a typical point
    has ``typical`` neighbours, so flat clouds (a wall, drone ground) are
    judged at their real surface spacing, not at a volume-based one.
    """
    if len(xyz) < 2:
        return np.zeros(len(xyz), np.int64)
    extent = np.maximum(np.percentile(xyz, 95, axis=0) - np.percentile(xyz, 5, axis=0), 1e-6)
    cell = max(float((np.prod(extent) / len(xyz)) ** (1.0 / 3.0)) * 1.5, 1e-6)
    counts = _voxel_neighbor_counts(xyz, cell)
    for _ in range(12):
        if np.median(counts) >= typical:
            break
        cell *= 1.5
        counts = _voxel_neighbor_counts(xyz, cell)
    return counts


def sparse_keep_mask(points: dict[str, np.ndarray], centers: dict[int, np.ndarray], settings: dict[str, Any]) -> np.ndarray:
    xyz = points["xyz"]
    keep = np.ones(len(xyz), bool)
    if not settings.get("enabled", True) or not len(xyz):
        return keep
    keep &= points["track"] >= int(settings["min_track_length"])
    keep &= points["error"] <= float(settings["max_reprojection_error"])
    if centers:
        known = np.isin(points["image_id"], np.fromiter(centers, np.int64))
        owner = points["owner"][known]
        ray_centers = np.asarray([centers[int(i)] for i in points["image_id"][known]], float).reshape(-1, 3)
        keep &= triangulation_angles(xyz, owner, ray_centers) >= float(settings["min_triangulation_angle_deg"])
        factor = float(settings["max_camera_distance_factor"])
        if factor > 0 and keep.any():
            distance = nearest_camera_distance(xyz, np.asarray(list(centers.values())))
            keep &= distance <= factor * np.percentile(distance[keep], 95)
    if int(settings["min_neighbors"]) > 0 and keep.sum() > 1:
        kept = np.flatnonzero(keep)
        keep[kept[neighbor_counts(xyz[kept]) < int(settings["min_neighbors"])]] = False
    return keep


def load_clean_sparse_points(points_path: Path, settings: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz, rgb)`` of the cleaned sparse cloud for display."""
    settings = sparse_cleaning_settings() if settings is None else {**DEFAULTS, **settings}
    points = _read_points_binary(Path(points_path))
    centers = _camera_centers(Path(points_path).parent) if settings.get("enabled", True) else {}
    if len(points["xyz"]) < int(settings["min_points"]):
        return points["xyz"], points["rgb"]
    keep = sparse_keep_mask(points, centers, settings)
    if keep.mean() < float(settings["min_keep_ratio"]):
        return points["xyz"], points["rgb"]
    return points["xyz"][keep], points["rgb"][keep]
