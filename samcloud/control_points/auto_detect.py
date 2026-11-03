"""Automatic target prediction, matching and conservative merging.

The functions in this module deliberately keep the image measurements in the
distorted, original-camera frame.  This makes target detection independent of
COLMAP's dense image resampling and lets bundle adjustment use the full camera
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from samcloud.control_points.targets import TargetDetection, detect_target, load_settings


@dataclass(frozen=True)
class PredictedObservation:
    image: str
    u: float
    v: float
    radius_px: float


def project_point(image: Any, camera: Any, xyz: np.ndarray, *, margin_px: float = 20.0) -> tuple[float, float] | None:
    """Project one model point through a distorted pycolmap camera."""
    camera_point = np.asarray(image.cam_from_world() * np.asarray(xyz, dtype=float), dtype=float)
    if camera_point.shape != (3,) or camera_point[2] <= 0:
        return None
    projected = camera.img_from_cam(camera_point)
    if projected is None:
        return None
    u, v = map(float, projected)
    if not (-margin_px <= u <= float(camera.width) + margin_px and -margin_px <= v <= float(camera.height) + margin_px):
        return None
    return u, v


def search_radius_px(
    image: Any, camera: Any, xyz: np.ndarray, prior_std_xy_m: float, prior_std_z_m: float,
    minimum: float = 40.0, maximum: float = 800.0,
) -> float:
    """Estimate a search radius from three standard deviations of the GPS prior."""
    centre = project_point(image, camera, xyz, margin_px=float("inf"))
    if centre is None:
        return float(minimum)
    displacements = []
    for delta in (np.array([3 * prior_std_xy_m, 0, 0]), np.array([0, 3 * prior_std_xy_m, 0]), np.array([0, 0, 3 * prior_std_z_m])):
        shifted = project_point(image, camera, np.asarray(xyz, dtype=float) + delta, margin_px=float("inf"))
        if shifted is not None:
            displacements.append(float(np.hypot(shifted[0] - centre[0], shifted[1] - centre[1])))
    return float(np.clip(max(displacements, default=minimum), minimum, maximum))


def predict_observations(
    reconstruction: Any, xyz: np.ndarray, prior_std_xy_m: float, prior_std_z_m: float,
    settings: dict[str, Any], *, margin_px: float = 20.0,
) -> list[PredictedObservation]:
    """Predict a GCP in every registered original image."""
    predicted = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        camera = reconstruction.camera(image.camera_id)
        uv = project_point(image, camera, xyz, margin_px=margin_px)
        if uv is not None:
            predicted.append(PredictedObservation(
                image.name, *uv,
                search_radius_px(image, camera, xyz, prior_std_xy_m, prior_std_z_m,
                                 float(settings.get("search_radius_min_px", 40)), float(settings.get("search_radius_max_px", 800))),
            ))
    return predicted


def robust_prediction_correction(predicted_xyz: np.ndarray, triangulated_xyz: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return a robust similarity correction from predictions to detections."""
    source, target = np.asarray(predicted_xyz, float), np.asarray(triangulated_xyz, float)
    if len(source) != len(target) or not len(source):
        raise ValueError("Prediction and triangulation counts must match and be non-empty")
    if len(source) < 3:
        return 1.0, np.eye(3), np.median(target - source, axis=0)
    scale, rotation, translation = _similarity(source, target)
    residuals = np.linalg.norm((scale * (rotation @ source.T)).T + translation - target, axis=1)
    threshold = 3.0 * max(float(np.median(residuals)), 1e-9)
    keep = residuals <= threshold
    return _similarity(source[keep], target[keep]) if keep.sum() >= 3 else (1.0, np.eye(3), np.median(target - source, axis=0))


def assign_detection(
    candidate: TargetDetection, predictions: dict[str, tuple[float, float, float]], point_name: str,
) -> bool:
    """Accept only a detection that is unambiguously closest to its point."""
    own = predictions[point_name]
    own_distance = float(np.hypot(candidate.u - own[0], candidate.v - own[1]))
    if own_distance > own[2]:
        return False
    rivals = [float(np.hypot(candidate.u - u, candidate.v - v)) for name, (u, v, radius) in predictions.items()
              if name != point_name and float(np.hypot(candidate.u - u, candidate.v - v)) <= radius]
    return not rivals or min(rivals) >= 2.0 * own_distance


def detect_stage(
    image_paths: dict[str, Path], predictions: dict[str, list[PredictedObservation]], *, radius_override: float | None,
    target_type: str, settings: dict[str, Any], existing: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Detect targets in predicted windows while preserving manual observations."""
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in predictions}
    stats = {"windows_searched": 0, "found": 0, "rejected": 0, "ambiguous": 0}
    manual = existing or {}
    by_image: dict[str, dict[str, tuple[float, float, float]]] = {}
    for name, items in predictions.items():
        for item in items:
            by_image.setdefault(item.image, {})[name] = (item.u, item.v, radius_override or item.radius_px)
    for image_name, point_predictions in by_image.items():
        image = cv2.imread(str(image_paths[image_name]), cv2.IMREAD_UNCHANGED) if image_name in image_paths else None
        if image is None:
            stats["rejected"] += len(point_predictions)
            continue
        for name, (u, v, radius) in point_predictions.items():
            if any(observation.get("image") == image_name and observation.get("source") == "manual" for observation in manual.get(name, {}).get("observations", [])):
                continue
            stats["windows_searched"] += 1
            found = detect_target(image, (u, v), radius, target_type, settings=load_settings(settings))
            if found is None:
                stats["rejected"] += 1
                continue
            if not assign_detection(found, point_predictions, name):
                stats["ambiguous"] += 1
                continue
            results[name].append({"image": image_name, "u": found.u, "v": found.v, "score": found.score, "source": "auto", "enabled": True})
            stats["found"] += 1
    return results, stats


def triangulate_observations(reconstruction: Any, observations: list[dict[str, Any]], *, inlier_px: float = 3.0) -> tuple[np.ndarray, list[float]]:
    """Robustly triangulate distorted-image observations with pair RANSAC."""
    usable = []
    for observation in observations:
        image = reconstruction.find_image_with_name(str(observation["image"]))
        if image is None:
            continue
        camera = reconstruction.camera(image.camera_id)
        normalized = camera.cam_from_img(np.asarray([observation["u"], observation["v"]], dtype=float))
        if normalized is not None:
            usable.append((image, camera, np.asarray(normalized, dtype=float)))
    if len(usable) < 2:
        raise ValueError("at least two registered observations are required")
    candidates: list[np.ndarray] = []
    for first in range(len(usable)):
        for second in range(first + 1, len(usable)):
            rows = []
            for image, _, normalized in (usable[first], usable[second]):
                matrix = np.asarray(image.cam_from_world().matrix(), dtype=float)
                rows.extend((normalized[0] * matrix[2] - matrix[0], normalized[1] * matrix[2] - matrix[1]))
            _, _, right_t = np.linalg.svd(np.asarray(rows))
            homogeneous = right_t[-1]
            if abs(homogeneous[3]) > 1e-12:
                candidates.append(homogeneous[:3] / homogeneous[3])
    if not candidates:
        raise ValueError("triangulation failed")
    best = min(candidates, key=lambda xyz: sum(_reprojection_error(image, camera, xyz, normalized) > inlier_px for image, camera, normalized in usable))
    errors = [_reprojection_error(image, camera, best, normalized) for image, camera, normalized in usable]
    inliers = [view for view, error in zip(usable, errors) if error <= inlier_px]
    if len(inliers) < 2:
        raise ValueError("no triangulation pair has sufficient inliers")
    best = _refine_least_squares(best, inliers)
    errors = [_reprojection_error(image, camera, best, normalized) for image, camera, normalized in usable]
    return best, errors


def automatic_detection(
    reconstruction: Any,
    references_model: dict[str, np.ndarray],
    roles: dict[str, str],
    image_paths: dict[str, Path],
    settings: dict[str, Any],
    prior_std_xy_m: float,
    prior_std_z_m: float,
    *,
    existing: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Perform the two detection stages and retain all reviewable outcomes."""
    existing = existing or {}
    first_predictions = {name: predict_observations(reconstruction, xyz, prior_std_xy_m, prior_std_z_m, settings) for name, xyz in references_model.items()}
    stage_one, statistics = detect_stage(image_paths, first_predictions, radius_override=None, target_type=str(settings.get("detection", {}).get("target_type", "cross")), settings=settings, existing=existing)
    predicted, triangulated = [], []
    for name, observations in stage_one.items():
        try:
            xyz, _ = triangulate_observations(reconstruction, observations)
        except ValueError:
            continue
        predicted.append(references_model[name]); triangulated.append(xyz)
    correction = robust_prediction_correction(np.asarray(predicted), np.asarray(triangulated)) if predicted else (1.0, np.eye(3), np.zeros(3))
    scale, rotation, translation = correction
    corrected = {name: scale * rotation @ xyz + translation for name, xyz in references_model.items()}
    second_predictions = {name: predict_observations(reconstruction, xyz, prior_std_xy_m, prior_std_z_m, settings) for name, xyz in corrected.items()}
    stage_two, second_statistics = detect_stage(image_paths, second_predictions, radius_override=float(settings.get("stage2_radius_px", 25)), target_type=str(settings.get("detection", {}).get("target_type", "cross")), settings=settings, existing=existing)
    for key, value in second_statistics.items():
        statistics[key] = statistics.get(key, 0) + value
    results: dict[str, dict[str, Any]] = {}
    for name, reference in references_model.items():
        old = existing.get(name, {})
        manual = [dict(item) for item in old.get("observations", []) if item.get("source") == "manual"]
        automatic = stage_two.get(name, [])
        observations = manual + [item for item in automatic if item["image"] not in {manual_item["image"] for manual_item in manual}]
        status, xyz = "not_found", None
        try:
            xyz, errors = triangulate_observations(reconstruction, observations)
            for observation, error in zip(observations, errors, strict=False):
                observation["reprojection_px"] = float(error)
            tolerance = max(float(settings.get("merge_tolerance_m", 0.5)), 1.0) if roles.get(name) == "control" else float(settings.get("merge_tolerance_m", 0.5))
            status = "accepted" if np.linalg.norm(xyz - corrected[name]) <= tolerance else "needs_review"
        except ValueError:
            pass
        results[name] = {
            "role": roles.get(name, old.get("role", "pass")),
            "enabled": not bool(old.get("disabled_by_reviewer", False)),
            "status": "disabled" if old.get("disabled_by_reviewer", False) else status,
            "disabled_by_reviewer": bool(old.get("disabled_by_reviewer", False)),
            "triangulated_model_xyz": None if xyz is None else [float(value) for value in xyz],
            "observations": observations,
        }
        if results[name]["role"] == "control" and xyz is None and results[name]["status"] != "disabled":
            results[name]["enabled"] = False
            results[name]["status"] = "not_found"
            results[name]["reason"] = "kein Target erkannt – bitte manuell setzen"
    return results, statistics


def populate_undistorted_coordinates(
    distorted_reconstruction: Any, undistorted_reconstruction: Any, detections: dict[str, dict[str, Any]],
) -> None:
    """Convert original pixels to dense-image pixels through both camera models."""
    for entry in detections.values():
        for observation in entry.get("observations", []):
            distorted_image = distorted_reconstruction.find_image_with_name(str(observation["image"]))
            undistorted_image = undistorted_reconstruction.find_image_with_name(str(observation["image"]))
            if distorted_image is None or undistorted_image is None:
                continue
            distorted_camera = distorted_reconstruction.camera(distorted_image.camera_id)
            undistorted_camera = undistorted_reconstruction.camera(undistorted_image.camera_id)
            normalized = distorted_camera.cam_from_img(np.asarray([observation["u"], observation["v"]], dtype=float))
            if normalized is None:
                continue
            projected = undistorted_camera.img_from_cam(np.asarray([normalized[0], normalized[1], 1.0], dtype=float))
            if projected is None:
                continue
            observation["u_undist"], observation["v_undist"] = map(float, projected)


def _refine_least_squares(xyz: np.ndarray, views: list[tuple[Any, Any, np.ndarray]], iterations: int = 10) -> np.ndarray:
    """Gauss-Newton refinement of a 3D point over all inlier views (pixel reprojection error)."""
    point = np.asarray(xyz, dtype=float).copy()
    for _ in range(iterations):
        rows, residuals = [], []
        for image, camera, normalized in views:
            observed = camera.img_from_cam(np.array([normalized[0], normalized[1], 1.0]))
            projected = camera.img_from_cam(np.asarray(image.cam_from_world() * point, dtype=float))
            if observed is None or projected is None:
                continue
            residuals.extend(np.asarray(projected, dtype=float) - np.asarray(observed, dtype=float))
            jacobian = np.zeros((2, 3))
            for axis in range(3):
                step = np.zeros(3)
                step[axis] = 1e-4
                shifted = camera.img_from_cam(np.asarray(image.cam_from_world() * (point + step), dtype=float))
                if shifted is None:
                    return point
                jacobian[:, axis] = (np.asarray(shifted, dtype=float) - np.asarray(projected, dtype=float)) / 1e-4
            rows.extend(jacobian)
        if len(rows) < 4:
            return point
        delta, *_ = np.linalg.lstsq(np.asarray(rows), -np.asarray(residuals), rcond=None)
        point += delta
        if np.linalg.norm(delta) < 1e-7:
            break
    return point


def _reprojection_error(image: Any, camera: Any, xyz: np.ndarray, normalized: np.ndarray) -> float:
    projected = camera.img_from_cam(np.asarray(image.cam_from_world() * xyz, dtype=float))
    observed = camera.img_from_cam(np.array([normalized[0], normalized[1], 1.0]))
    return float("inf") if projected is None or observed is None else float(np.linalg.norm(np.asarray(projected) - np.asarray(observed)))


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    centered_source, centered_target = source - source_mean, target - target_mean
    left, singular, right_t = np.linalg.svd(centered_target.T @ centered_source / len(source))
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right_t
    scale = float(np.trace(np.diag(singular) @ correction) / np.sum(centered_source ** 2) * len(source))
    return scale, rotation, target_mean - scale * rotation @ source_mean
