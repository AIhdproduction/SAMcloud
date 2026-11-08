"""Orient horizon-levelled panorama reconstructions to gravity.

The numeric helpers in this module deliberately operate on NumPy arrays only.
COLMAP access and file I/O are isolated in :func:`align_sparse_model`, so the
orientation estimate can be tested without a COLMAP executable or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np


# The central ray of the rendered ``up`` cubemap face.  The renderer uses
# image coordinates with y down; see cubemap_rig_from_camera_rotations().
UP_FACE_AXIS_RIG = np.array([0.0, -1.0, 0.0])


class GravityAlignmentError(ValueError):
    """Raised when a panorama model cannot yield a reliable gravity axis."""


@dataclass(frozen=True)
class PlaneFit:
    """One horizontal RANSAC plane in the aligned model frame."""

    normal: np.ndarray
    offset: float
    inlier_count: int
    point_count: int
    angle_to_z_deg: float

    @property
    def z_at_origin(self) -> float:
        """Return the plane height at x=y=0 for an upward-facing normal."""
        if abs(float(self.normal[2])) < 1e-12:
            raise GravityAlignmentError("Horizontal plane normal has no Z component")
        return float(-self.offset / self.normal[2])


@dataclass(frozen=True)
class GravityEstimate:
    """Robust gravity estimate derived from registered rig poses."""

    up_world: np.ndarray
    rotation: np.ndarray
    input_count: int
    retained_count: int
    spread_deg: float
    max_deviation_deg: float


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return one finite non-zero vector with unit length."""
    value = np.asarray(vector, dtype=float).reshape(3)
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 1e-12:
        raise GravityAlignmentError("Expected a finite non-zero 3D vector")
    return value / length


def rig_up_vectors(rig_from_world_rotations: Iterable[np.ndarray]) -> np.ndarray:
    """Return the up-face optical axis in world coordinates for every rig pose.

    ``rig_from_world`` maps world coordinates into the rig.  Consequently its
    transpose maps the known rig-local up-face optical axis back to world.
    """
    vectors = []
    for rotation in rig_from_world_rotations:
        matrix = np.asarray(rotation, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            continue
        vectors.append(unit_vector(matrix.T @ UP_FACE_AXIS_RIG))
    if not vectors:
        raise GravityAlignmentError("No registered rig poses with valid rotations were found")
    return np.asarray(vectors, dtype=float)


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Return the unsigned angle between two vectors in degrees."""
    cosine = float(np.clip(np.dot(unit_vector(first), unit_vector(second)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a proper rotation mapping ``source`` onto ``target``."""
    first, second = unit_vector(source), unit_vector(target)
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0:
            return np.eye(3)
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        axis = unit_vector(np.cross(first, basis))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def estimate_gravity(rig_from_world_rotations: Iterable[np.ndarray]) -> GravityEstimate:
    """Robustly average the panorama up axis and return its world rotation."""
    vectors = rig_up_vectors(rig_from_world_rotations)
    provisional = unit_vector(vectors.mean(axis=0))
    deviations = np.asarray([angle_degrees(vector, provisional) for vector in vectors])
    median = float(np.median(deviations))
    mad = float(np.median(np.abs(deviations - median)))
    # Horizon-levelled input should be tightly clustered.  This protects the
    # average from an occasional failed or incorrectly associated rig pose.
    threshold = max(3.0, median + max(0.25, 3.0 * 1.4826 * mad))
    retained = vectors[deviations <= threshold]
    if not len(retained):
        raise GravityAlignmentError("All rig up axes were rejected as outliers")
    up_world = unit_vector(retained.mean(axis=0))
    final_deviations = np.asarray([angle_degrees(vector, up_world) for vector in retained])
    return GravityEstimate(
        up_world=up_world,
        rotation=rotation_from_vectors(up_world, np.array([0.0, 0.0, 1.0])),
        input_count=len(vectors),
        retained_count=len(retained),
        spread_deg=float(np.sqrt(np.mean(final_deviations**2))),
        max_deviation_deg=float(final_deviations.max(initial=0.0)),
    )


def largest_horizontal_plane(
    points: np.ndarray, *, max_angle_deg: float = 15.0, iterations: int = 600,
) -> PlaneFit | None:
    """Fit the largest approximately horizontal sparse-point plane with RANSAC."""
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < 3:
        return None
    if len(values) > 12000:
        values = values[np.linspace(0, len(values) - 1, 12000, dtype=int)]
    extent = float(np.linalg.norm(np.ptp(values, axis=0)))
    threshold = max(extent * 0.004, 1e-6)
    horizontal_limit = math.cos(math.radians(max_angle_deg))
    generator = np.random.default_rng(20261106)
    best_mask: np.ndarray | None = None
    best_normal: np.ndarray | None = None
    for _ in range(iterations):
        sample = values[generator.choice(len(values), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        try:
            normal = unit_vector(normal)
        except GravityAlignmentError:
            continue
        if abs(float(normal[2])) < horizontal_limit:
            continue
        distances = np.abs((values - sample[0]) @ normal)
        mask = distances <= threshold
        if best_mask is None or int(mask.sum()) > int(best_mask.sum()):
            best_mask, best_normal = mask, normal
    minimum_inliers = max(6, int(math.ceil(len(values) * 0.005)))
    if best_mask is None or best_normal is None or int(best_mask.sum()) < minimum_inliers:
        return None
    inliers = values[best_mask]
    centroid = inliers.mean(axis=0)
    _singular, _values, right = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = unit_vector(right[-1])
    if normal[2] < 0:
        normal = -normal
    return PlaneFit(
        normal=normal,
        offset=float(-normal @ centroid),
        inlier_count=len(inliers),
        point_count=len(values),
        angle_to_z_deg=angle_degrees(normal, np.array([0.0, 0.0, 1.0])),
    )


def gravity_transform(
    rig_from_world_rotations: Iterable[np.ndarray], points: np.ndarray | None = None,
    camera_centres: np.ndarray | None = None, *, translate_floor: bool = True,
) -> tuple[GravityEstimate, np.ndarray, PlaneFit | None, bool]:
    """Return gravity rotation, translation, plane check, and floor status.

    The optional translation is deliberately vertical only.  It puts a largest
    horizontal plane at z=0 only when that plane lies below the median rig
    centre, avoiding a ceiling being mistaken for the floor.
    """
    estimate = estimate_gravity(rig_from_world_rotations)
    rotation = estimate.rotation
    aligned_points = None if points is None else (rotation @ np.asarray(points, dtype=float).reshape(-1, 3).T).T
    plane = largest_horizontal_plane(aligned_points) if aligned_points is not None else None
    translation = np.zeros(3)
    floor_applied = False
    if translate_floor and plane is not None and camera_centres is not None:
        centres = np.asarray(camera_centres, dtype=float).reshape(-1, 3)
        centres = centres[np.all(np.isfinite(centres), axis=1)]
        if len(centres):
            aligned_centres = (rotation @ centres.T).T
            plane_z = plane.z_at_origin
            if plane_z < float(np.median(aligned_centres[:, 2])):
                translation[2] = -plane_z
                floor_applied = True
    return estimate, translation, plane, floor_applied


def _frame_pose_arrays(reconstruction) -> tuple[list[np.ndarray], np.ndarray]:
    rotations: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    for frame in reconstruction.frames.values():
        if not frame.has_pose() or frame.rig_from_world is None:
            continue
        pose = frame.rig_from_world
        rotation = np.asarray(pose.rotation.matrix(), dtype=float)
        translation = np.asarray(pose.translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            continue
        rotations.append(rotation)
        centres.append(-rotation.T @ translation)
    if not rotations:
        raise GravityAlignmentError("Sparse model has no registered rig frames")
    return rotations, np.asarray(centres, dtype=float)


def report_dict(
    estimate: GravityEstimate, translation: np.ndarray, plane: PlaneFit | None, floor_applied: bool,
) -> dict[str, object]:
    """Return JSON-ready alignment metadata without NumPy scalar values."""
    aligned_up = estimate.rotation @ estimate.up_world
    result: dict[str, object] = {
        "scale": "arbitrary (no reference)",
        "frame_count": estimate.input_count,
        "retained_frame_count": estimate.retained_count,
        "up_world_before": estimate.up_world.tolist(),
        "up_world_after": aligned_up.tolist(),
        "rotation": estimate.rotation.tolist(),
        "translation": np.asarray(translation, dtype=float).tolist(),
        "up_spread_deg_rms": estimate.spread_deg,
        "up_max_deviation_deg": estimate.max_deviation_deg,
        "up_face_is_positive_z": bool(angle_degrees(aligned_up, [0.0, 0.0, 1.0]) < 1e-6),
        "floor_translation_applied": floor_applied,
        "floor_below_camera": floor_applied,
    }
    if plane is None:
        result["largest_horizontal_plane"] = None
    else:
        result["largest_horizontal_plane"] = {
            "normal": plane.normal.tolist(),
            "z_at_origin": plane.z_at_origin + float(translation[2]),
            "inlier_count": plane.inlier_count,
            "point_count": plane.point_count,
            "angle_to_aligned_z_deg": plane.angle_to_z_deg,
            "warning": plane.angle_to_z_deg > 3.0,
        }
    return result


def align_sparse_model(
    source_model: Path, destination_model: Path, *, translate_floor: bool = True,
) -> dict[str, object]:
    """Write an aligned COLMAP model and return its gravity report."""
    import pycolmap

    source_model, destination_model = Path(source_model), Path(destination_model)
    reconstruction = pycolmap.Reconstruction(source_model)
    rotations, centres = _frame_pose_arrays(reconstruction)
    points = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=float)
    estimate, translation, plane, floor_applied = gravity_transform(
        rotations, points, centres, translate_floor=translate_floor,
    )
    destination_model.mkdir(parents=True, exist_ok=True)
    reconstruction.transform(pycolmap.Sim3d(np.column_stack((estimate.rotation, translation))))
    reconstruction.write(destination_model)
    report = report_dict(estimate, translation, plane, floor_applied)
    report["source_model"] = str(source_model)
    report["aligned_model"] = str(destination_model)
    return report
