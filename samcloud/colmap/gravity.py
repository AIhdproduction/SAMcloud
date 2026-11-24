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
    up: np.ndarray | None = None,
) -> PlaneFit | None:
    """Fit the largest approximately horizontal sparse-point plane with RANSAC.

    ``up`` is the direction "horizontal" is measured against (default +Z). With a custom ``up`` the
    returned normal points along it and ``angle_to_z_deg`` is the angle to that direction.
    """
    reference = np.array([0.0, 0.0, 1.0]) if up is None else unit_vector(up)
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
        if abs(float(normal @ reference)) < horizontal_limit:
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
    if normal @ reference < 0:
        normal = -normal
    return PlaneFit(
        normal=normal,
        offset=float(-normal @ centroid),
        inlier_count=len(inliers),
        point_count=len(values),
        angle_to_z_deg=angle_degrees(normal, reference),
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


# --------------------------------------------------------------------------- handheld camera captures

# COLMAP camera frame: x right, y down, z forward; "up" in the image is the negative y axis.
CAMERA_UP_AXIS = np.array([0.0, -1.0, 0.0])
CAMERA_OUTLIER_MIN_DEG = 25.0  # handheld pitch varies, much wider than the levelled 360 rig
CAMERA_OUTLIER_MAX_DEG = 60.0
CAMERA_PLANE_MAX_ANGLE_DEG = 10.0  # also the largest accepted disagreement with the camera axes
CAMERA_SURFACE_NORMAL_DEG = 20.0  # local normals further from the coarse up are walls/clutter, not floor or ceiling
CAMERA_SURFACE_NEIGHBOURS = 16
CAMERA_SURFACE_MAX_FLATNESS = 0.05  # smallest / sum of the neighbourhood covariance eigenvalues
CAMERA_SURFACE_MAX_POINTS = 40000
CAMERA_PLANE_MIN_INLIERS = 200
CAMERA_PLANE_MIN_FRACTION = 0.05


def camera_up_vectors(cam_from_world_rotations: Iterable[np.ndarray]) -> np.ndarray:
    """World-space "up" (negative camera y axis) of every registered camera."""
    vectors = []
    for rotation in cam_from_world_rotations:
        matrix = np.asarray(rotation, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            continue
        vectors.append(unit_vector(matrix.T @ CAMERA_UP_AXIS))
    if not vectors:
        raise GravityAlignmentError("No registered camera poses with valid rotations were found")
    return np.asarray(vectors, dtype=float)


def estimate_camera_up(cam_from_world_rotations: Iterable[np.ndarray]) -> GravityEstimate:
    """Coarse gravity from handheld cameras: robust mean of the per-image up vectors."""
    vectors = camera_up_vectors(cam_from_world_rotations)
    if len(vectors) < 3:
        raise GravityAlignmentError("At least three registered cameras are needed to estimate gravity")
    provisional = unit_vector(vectors.mean(axis=0))
    for _ in range(3):  # re-centre on the inliers so a few flipped cameras do not drag the mean
        deviations = np.asarray([angle_degrees(vector, provisional) for vector in vectors])
        median = float(np.median(deviations))
        mad = float(np.median(np.abs(deviations - median)))
        threshold = min(CAMERA_OUTLIER_MAX_DEG, max(CAMERA_OUTLIER_MIN_DEG, median + 3.0 * 1.4826 * mad))
        retained = vectors[deviations <= threshold]
        if not len(retained):
            raise GravityAlignmentError("All camera up axes were rejected as outliers")
        provisional = unit_vector(retained.mean(axis=0))
    final = np.asarray([angle_degrees(vector, provisional) for vector in retained])
    return GravityEstimate(
        up_world=provisional,
        rotation=rotation_from_vectors(provisional, np.array([0.0, 0.0, 1.0])),
        input_count=len(vectors),
        retained_count=len(retained),
        spread_deg=float(np.sqrt(np.mean(final**2))),
        max_deviation_deg=float(final.max(initial=0.0)),
    )


def _horizontal_surface_points(
    points: np.ndarray, up: np.ndarray, *, max_angle_deg: float = CAMERA_SURFACE_NORMAL_DEG,
    neighbours: int = CAMERA_SURFACE_NEIGHBOURS, max_flatness: float = CAMERA_SURFACE_MAX_FLATNESS,
    max_points: int = CAMERA_SURFACE_MAX_POINTS,
) -> np.ndarray:
    """Keep sparse points on locally flat patches whose normal is near ``up`` (floor, ceiling, tabletops).

    Wall points and scattered clutter are dropped, so a RANSAC band cutting through textured walls
    cannot win against the real floor.
    """
    from scipy.spatial import cKDTree

    reference = unit_vector(up)
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) > max_points:
        values = values[np.linspace(0, len(values) - 1, max_points, dtype=int)]
    k = min(neighbours, len(values))
    if k < 4:
        return values[:0]
    _distance, index = cKDTree(values).query(values, k=k)
    kept = np.zeros(len(values), dtype=bool)
    limit = math.cos(math.radians(max_angle_deg))
    for start in range(0, len(values), 20000):
        block = values[index[start:start + 20000]]
        block = block - block.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", block, block)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        total = np.maximum(eigenvalues.sum(axis=1), 1e-18)
        flat = eigenvalues[:, 0] / total <= max_flatness
        aligned = np.abs(eigenvectors[:, :, 0] @ reference) >= limit
        kept[start:start + 20000] = flat & aligned
    return values[kept]


def camera_gravity_transform(
    cam_from_world_rotations: Iterable[np.ndarray], points: np.ndarray | None = None,
    camera_centres: np.ndarray | None = None, *, translate_floor: bool = True,
    min_plane_inliers: int = CAMERA_PLANE_MIN_INLIERS, min_plane_fraction: float = CAMERA_PLANE_MIN_FRACTION,
    max_plane_angle_deg: float = CAMERA_PLANE_MAX_ANGLE_DEG,
) -> tuple[GravityEstimate, np.ndarray, PlaneFit | None, bool, dict[str, object]]:
    """Rotation, translation, aligned-frame plane check, floor status and extras for a camera model.

    The coarse up comes from the camera poses; a large sparse-point plane within ``max_plane_angle_deg``
    of it (floor or ceiling) refines the axis. Otherwise the coarse up is kept and a warning is added.
    """
    rotations = [np.asarray(rotation, dtype=float) for rotation in cam_from_world_rotations]
    coarse = estimate_camera_up(rotations)
    values = None if points is None else np.asarray(points, dtype=float).reshape(-1, 3)
    refined = None
    warnings: list[str] = []
    candidates = 0
    if values is not None and len(values):
        surface = _horizontal_surface_points(values, coarse.up_world)
        candidates = len(surface)
        refined = largest_horizontal_plane(surface, max_angle_deg=max_plane_angle_deg, up=coarse.up_world)
    if refined is not None and (
        refined.inlier_count < min_plane_inliers
        or refined.inlier_count < min_plane_fraction * max(refined.point_count, 1)
    ):
        warnings.append(
            f"Largest horizontal plane is too small ({refined.inlier_count} of {refined.point_count} points); "
            "the camera up axes are used without refinement."
        )
        refined = None
    elif refined is None:
        warnings.append("No horizontal plane found in the sparse points; the camera up axes are used without refinement.")
    plane_height_below_cameras: float | None = None
    if refined is not None:
        disagreement = angle_degrees(refined.normal, coarse.up_world)
        centres_all = None if camera_centres is None else np.asarray(camera_centres, dtype=float).reshape(-1, 3)
        if centres_all is not None:
            centres_all = centres_all[np.all(np.isfinite(centres_all), axis=1)]
            if len(centres_all):
                plane_height = float(-refined.offset * (refined.normal @ coarse.up_world))
                plane_height_below_cameras = float(np.median(centres_all @ coarse.up_world) - plane_height)
        if disagreement > max_plane_angle_deg:
            warnings.append(
                f"The horizontal plane disagreed with the camera axes by {disagreement:.1f} degrees; "
                "the camera up axes are used without refinement."
            )
            refined = None
    up_world = refined.normal if refined is not None else coarse.up_world
    rotation = rotation_from_vectors(up_world, np.array([0.0, 0.0, 1.0]))
    spread = np.asarray([angle_degrees(vector, up_world) for vector in camera_up_vectors(rotations)])
    estimate = GravityEstimate(
        up_world=up_world, rotation=rotation, input_count=coarse.input_count,
        retained_count=coarse.retained_count, spread_deg=float(np.sqrt(np.mean(spread**2))),
        max_deviation_deg=float(spread.max(initial=0.0)),
    )
    aligned_points = None if values is None else (rotation @ values.T).T
    plane = largest_horizontal_plane(aligned_points) if aligned_points is not None and len(aligned_points) else None
    translation = np.zeros(3)
    floor_applied = False
    if translate_floor and plane is not None and camera_centres is not None:
        centres = np.asarray(camera_centres, dtype=float).reshape(-1, 3)
        centres = centres[np.all(np.isfinite(centres), axis=1)]
        if len(centres) and plane.z_at_origin < float(np.median((rotation @ centres.T).T[:, 2])):
            translation[2] = -plane.z_at_origin
            floor_applied = True
    extras: dict[str, object] = {
        "method": "camera_up_plane" if refined is not None else "camera_up_only",
        "image_count": coarse.input_count,
        "coarse_up_world": coarse.up_world.tolist(),
        "final_up_world": up_world.tolist(),
        "coarse_to_final_angle_deg": angle_degrees(coarse.up_world, up_world),
        "tilt_corrected_deg": angle_degrees(up_world, [0.0, 0.0, 1.0]),
        "plane_inlier_count": 0 if refined is None else refined.inlier_count,
        "plane_point_count": 0 if refined is None else refined.point_count,
        "plane_height_below_cameras": plane_height_below_cameras,
        "horizontal_candidate_count": candidates,
        "warnings": warnings,
    }
    return estimate, translation, plane, floor_applied, extras


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


def align_camera_model(
    source_model: Path, destination_model: Path, *, translate_floor: bool = True,
    min_plane_inliers: int = CAMERA_PLANE_MIN_INLIERS, min_plane_fraction: float = CAMERA_PLANE_MIN_FRACTION,
) -> dict[str, object]:
    """Level a handheld camera model (floor horizontal, +Z up), write it and return the report."""
    import pycolmap

    source_model, destination_model = Path(source_model), Path(destination_model)
    reconstruction = pycolmap.Reconstruction(source_model)
    rotations: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        pose = image.cam_from_world() if callable(getattr(image, "cam_from_world", None)) else image.cam_from_world
        rotation = np.asarray(pose.rotation.matrix(), dtype=float)
        translation = np.asarray(pose.translation, dtype=float)
        rotations.append(rotation)
        centres.append(-rotation.T @ translation)
    if not rotations:
        raise GravityAlignmentError("Sparse model has no registered images")
    points = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=float)
    estimate, translation, plane, floor_applied, extras = camera_gravity_transform(
        rotations, points, np.asarray(centres), translate_floor=translate_floor,
        min_plane_inliers=min_plane_inliers, min_plane_fraction=min_plane_fraction,
    )
    destination_model.mkdir(parents=True, exist_ok=True)
    reconstruction.transform(pycolmap.Sim3d(np.column_stack((estimate.rotation, translation))))
    reconstruction.write(destination_model)
    report = report_dict(estimate, translation, plane, floor_applied)
    report.update(extras)
    if source_model.name.endswith("_tagscale"):
        report["scale"] = "apriltag"
    report["source_model"] = str(source_model)
    report["aligned_model"] = str(destination_model)
    return report
