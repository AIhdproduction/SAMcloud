"""Explicit gravity and optional wall-direction alignment for indoor point clouds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from samcloud.dxf.cancel import CancelToken, ProgressCallback, check_cancel, report_progress
from samcloud.dxf.config import AlignmentConfig, DxfExportConfig, load_dxf_config


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """A local rotation that maps estimated gravity to positive Z."""

    rotation: np.ndarray
    tilt_degrees: float
    yaw_degrees: float
    method: str
    inlier_count: int

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("Alignment rotation must be a 3 by 3 matrix")
        object.__setattr__(self, "rotation", rotation)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable alignment state for project persistence."""
        data = asdict(self)
        data["rotation"] = self.rotation.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentResult":
        """Restore a serialized alignment state."""
        return cls(
            np.asarray(data["rotation"], dtype=np.float64),
            float(data["tilt_degrees"]),
            float(data["yaw_degrees"]),
            str(data["method"]),
            int(data["inlier_count"]),
        )


def estimate_alignment(
    xyz: np.ndarray,
    classification: np.ndarray,
    *,
    config: DxfExportConfig | AlignmentConfig | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> AlignmentResult:
    """Estimate gravity from floor points, falling back to horizontal normal evidence."""
    points = _validate_cloud(xyz, classification)
    settings = _alignment_settings(config)
    floor_points = points[np.asarray(classification) == 102]
    if len(floor_points) >= settings.min_floor_points:
        report_progress(progress, 0.1, "Fitting floor plane")
        normal, inliers = _ransac_plane(floor_points, settings, cancel)
        method = "floor_ransac"
    else:
        report_progress(progress, 0.1, "Estimating surface normals")
        normal, inliers = _dominant_horizontal_normal(points, settings, cancel)
        method = "normal_fallback"
    check_cancel(cancel)
    if normal[2] < 0:
        normal = -normal
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    gravity_rotation = _rotation_between(normal, np.array([0.0, 0.0, 1.0]))
    yaw = 0.0
    rotation = gravity_rotation
    if settings.align_wall_yaw:
        walls = points[np.asarray(classification) == 100]
        if len(walls) >= settings.normal_neighbors:
            yaw = _wall_yaw(walls, gravity_rotation, settings, cancel)
            rotation = _rotation_z(yaw) @ gravity_rotation
    report_progress(progress, 1.0, "Gravity alignment complete")
    return AlignmentResult(rotation, tilt, float(np.degrees(yaw)), method, int(inliers))


def _ransac_plane(points: np.ndarray, settings: AlignmentConfig, cancel: CancelToken | None) -> tuple[np.ndarray, int]:
    sampled = _sample_points(points, settings.normal_sample_size, seed=20261109)
    rng = np.random.default_rng(20261109)
    best_mask: np.ndarray | None = None
    best_normal: np.ndarray | None = None
    for index in range(settings.ransac_iterations):
        if index % 32 == 0:
            check_cancel(cancel)
        sample = sampled[rng.choice(len(sampled), size=3, replace=False)]
        normal = _cross3(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            continue
        normal /= length
        mask = np.abs((sampled - sample[0]) @ normal) <= settings.ransac_distance
        if best_mask is None or int(mask.sum()) > int(best_mask.sum()):
            best_mask, best_normal = mask, normal
    if best_mask is None or best_normal is None or int(best_mask.sum()) < 3:
        raise ValueError("Could not fit a plane to floor-class points")
    inliers = sampled[best_mask]
    refined = _plane_normal(inliers)
    return refined, int(best_mask.sum())


def _dominant_horizontal_normal(points: np.ndarray, settings: AlignmentConfig, cancel: CancelToken | None) -> tuple[np.ndarray, int]:
    sampled = _sample_points(points, settings.normal_sample_size, seed=20261110)
    if len(sampled) < settings.normal_neighbors:
        raise ValueError("Too few points for fallback gravity alignment")
    tree = cKDTree(sampled)
    _, indices = tree.query(sampled, k=min(settings.normal_neighbors, len(sampled)))
    normals = np.empty_like(sampled)
    for index, neighbors in enumerate(np.asarray(indices)):
        if index % 256 == 0:
            check_cancel(cancel)
        normals[index] = _plane_normal(sampled[neighbors])
    horizontal = normals[np.abs(normals[:, 2]) >= 0.85]
    if len(horizontal) < 3:
        raise ValueError("Could not find horizontal surfaces for fallback gravity alignment")
    for normal in horizontal:
        if normal[2] < 0:
            normal *= -1
    normal = np.mean(horizontal, axis=0)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    return normal, len(horizontal)


def _wall_yaw(walls: np.ndarray, gravity_rotation: np.ndarray, settings: AlignmentConfig, cancel: CancelToken | None) -> float:
    aligned = walls @ gravity_rotation.T
    sampled = _sample_points(aligned, settings.normal_sample_size, seed=20261111)
    tree = cKDTree(sampled)
    _, indices = tree.query(sampled, k=min(settings.normal_neighbors, len(sampled)))
    azimuths: list[float] = []
    for index, neighbors in enumerate(np.asarray(indices)):
        if index % 256 == 0:
            check_cancel(cancel)
        normal = _plane_normal(sampled[neighbors])
        if abs(float(normal[2])) <= 0.35:
            azimuths.append(float(np.arctan2(normal[1], normal[0])))
    if not azimuths:
        return 0.0
    angles = np.asarray(azimuths)
    folded = 4.0 * angles
    orientation = 0.25 * float(np.arctan2(np.sin(folded).mean(), np.cos(folded).mean()))
    return -orientation


def _plane_normal(points: np.ndarray) -> np.ndarray:
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    normal = right[-1]
    return normal / np.linalg.norm(normal)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    start = np.asarray(source, dtype=np.float64) / np.linalg.norm(source)
    end = np.asarray(target, dtype=np.float64) / np.linalg.norm(target)
    cross = _cross3(start, end)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(start, end), -1.0, 1.0))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array(((0.0, -cross[2], cross[1]), (cross[2], 0.0, -cross[0]), (-cross[1], cross[0], 0.0)))
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def _cross3(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the three-dimensional vector product without NumPy 2D promotion."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("Cross-product operands must be three-dimensional vectors")
    return np.array(
        (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ),
        dtype=np.float64,
    )


def _sample_points(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(points) <= limit:
        return np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return np.asarray(points[rng.choice(len(points), size=limit, replace=False)], dtype=np.float64)


def _validate_cloud(xyz: np.ndarray, classification: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float64)
    classes = np.asarray(classification)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(classes):
        raise ValueError("XYZ points must have shape (N, 3) and match classification length")
    if len(points) < 3:
        raise ValueError("At least three points are required for alignment")
    return points


def _alignment_settings(config: DxfExportConfig | AlignmentConfig | None) -> AlignmentConfig:
    if config is None:
        return load_dxf_config().alignment
    return config.alignment if isinstance(config, DxfExportConfig) else config
