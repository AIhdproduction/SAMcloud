"""Ground-control bundle adjustment, validation and accuracy reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import math

import numpy as np

from samcloud.control_points.auto_detect import project_point, triangulate_observations


def pose_priors_in_model_frame(
    reconstruction: Any, priors: list[Any], *, fallback_std_xy_m: float = 3.0, fallback_std_z_m: float = 5.0,
    robust: bool = True,
) -> list[Any]:
    """Express WGS84 GPS pose priors in the reconstruction's metric frame for pose-prior BA."""
    import pycolmap
    from samcloud.control_points.georeference import enu_rotation_at_ecef
    from samcloud.pointcloud.georeference import _transform_strict, fit_similarity, strict_transformer

    registered = set(reconstruction.reg_image_ids())
    to_ecef = strict_transformer("EPSG:4979", "EPSG:4978")
    pairs: list[tuple[Any, np.ndarray, np.ndarray]] = []
    for prior in priors:
        image_id = int(prior.corr_data_id.id)
        if image_id not in registered or not prior.has_position():
            continue
        position = np.asarray(prior.position, dtype=float)
        if prior.coordinate_system == pycolmap.PosePriorCoordinateSystem.WGS84:
            x, y, z = _transform_strict(to_ecef, [position[1]], [position[0]], [position[2]])
            ecef = np.array([float(x[0]), float(y[0]), float(z[0])])
        elif np.linalg.norm(position) > 1e6:
            ecef = position
        else:
            return list(priors)
        pairs.append((prior, ecef, np.asarray(reconstruction.image(image_id).projection_center(), dtype=float)))
    if len(pairs) < 3:
        raise ValueError("At least three registered images with GPS priors are required")
    source = np.asarray([item[1] for item in pairs])
    target = np.asarray([item[2] for item in pairs])
    scale, rotation, translation, _ = fit_similarity(source, target)
    if robust:
        residuals = np.linalg.norm((scale * (rotation @ source.T)).T + translation - target, axis=1)
        keep = residuals <= max(0.05, 3.0 * float(np.median(residuals)))
        if keep.sum() >= 3 and not keep.all():
            scale, rotation, translation, _ = fit_similarity(source[keep], target[keep])
    residuals = np.linalg.norm((scale * (rotation @ source.T)).T + translation - target, axis=1)
    print(f"GPS prior frame: {len(pairs)} priors, residual mean {residuals.mean():.3f} m, max {residuals.max():.3f} m")
    converted = []
    for prior, ecef, _centre in pairs:
        covariance = np.asarray(prior.position_covariance, dtype=float)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            covariance = np.diag([fallback_std_xy_m ** 2, fallback_std_xy_m ** 2, fallback_std_z_m ** 2])
        enu = enu_rotation_at_ecef(ecef)
        covariance_ecef = enu.T @ covariance @ enu
        covariance_model = scale ** 2 * rotation @ covariance_ecef @ rotation.T
        converted.append(pycolmap.PosePrior(
            pose_prior_id=prior.pose_prior_id,
            corr_data_id=prior.corr_data_id,
            position=scale * rotation @ ecef + translation,
            position_covariance=covariance_model,
            coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
        ))
    return converted


def add_pass_points_to_reconstruction(reconstruction: Any, detections: dict[str, dict[str, object]], references: dict[str, np.ndarray]) -> list[int]:
    """Add enabled pass targets only; control targets never enter BA."""
    import pycolmap

    image_ids = {reconstruction.image(image_id).name: image_id for image_id in reconstruction.reg_image_ids()}
    point_ids: list[int] = []
    for name, entry in detections.items():
        if entry.get("role") != "pass" or not entry.get("enabled", True) or name not in references:
            continue
        track = pycolmap.Track()
        for observation in entry.get("observations", []):
            image_id = image_ids.get(str(observation["image"]))
            if image_id is None or not observation.get("enabled", True):
                continue
            image = reconstruction.image(image_id)
            image.points2D.append(pycolmap.Point2D(np.asarray([observation["u"], observation["v"]], dtype=float)))
            track.add_element(image.image_id, image.num_points2D() - 1)
        if track.length() >= 2:
            point_ids.append(reconstruction.add_point3D(np.asarray(references[name], dtype=float), track))
    return point_ids


def run_gcp_bundle_adjustment(
    model_dir: Path, output_dir: Path, detections: dict[str, dict[str, object]], references: dict[str, np.ndarray],
    *, pose_priors: list[Any] | None = None, prior_stddev: float = 3.0, second_pass: bool = True,
) -> dict[str, Any]:
    """Adjust with fixed pass points, filter ordinary ties, then adjust again."""
    import pycolmap

    reconstruction = pycolmap.Reconstruction(model_dir)
    before = camera_intrinsics(reconstruction)
    point_ids = _solve(reconstruction, detections, references, pose_priors, prior_stddev)
    if second_pass:
        _filter_tie_points(reconstruction, set(point_ids))
        # The marker tracks are rebuilt for the second solve, preventing their
        # observations from being treated as ordinary filtered tie points.
        for point_id in point_ids:
            if reconstruction.exists_point3D(point_id):
                reconstruction.delete_point3D(point_id)
        point_ids = _solve(reconstruction, detections, references, pose_priors, prior_stddev)
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(output_dir)
    return {"pass_point_ids": point_ids, "intrinsics_before": before, "intrinsics": camera_intrinsics(reconstruction)}


def _solve(reconstruction: Any, detections: dict[str, dict[str, object]], references: dict[str, np.ndarray], pose_priors: list[Any] | None, prior_stddev: float) -> list[int]:
    import pycolmap

    point_ids = add_pass_points_to_reconstruction(reconstruction, detections, references)
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        config.add_image(image_id)
    for point_id in point_ids:
        config.add_constant_point(point_id)
    options = pycolmap.BundleAdjustmentOptions(refine_focal_length=True, refine_principal_point=True, refine_extra_params=True)
    # Marker observations must keep full weight: ordinary tie points are already filtered to ~1-2 px,
    # and a robust loss would down-weight exactly the marker residuals the adjustment has to remove.
    options.ceres.loss_function_type = pycolmap.LossFunctionType.TRIVIAL
    if pose_priors:
        prior_options = pycolmap.PosePriorBundleAdjustmentOptions(prior_position_fallback_stddev=float(prior_stddev))
        prior_options.ceres.prior_position_loss_function_type = pycolmap.LossFunctionType.CAUCHY
        prior_options.ceres.prior_position_loss_scale = float(prior_stddev)
        pose_priors = pose_priors_in_model_frame(reconstruction, pose_priors, fallback_std_xy_m=float(prior_stddev), fallback_std_z_m=float(prior_stddev))
        pycolmap.create_pose_prior_bundle_adjuster(options, prior_options, config, pose_priors, reconstruction).solve()
    else:
        pycolmap.create_default_bundle_adjuster(options, config, reconstruction).solve()
    return point_ids


def _filter_tie_points(reconstruction: Any, protected: set[int]) -> None:
    """Apply the required 1 px / three-view tie-point filter via pycolmap."""
    reconstruction.update_point_3d_errors()
    for point_id in list(reconstruction.point3D_ids()):
        if point_id in protected:
            continue
        point = reconstruction.point3D(point_id)
        if point.track.length() < 3 or (point.has_error() and point.error > 1.0):
            reconstruction.delete_point3D(point_id)


def point_metrics(reconstruction: Any, detections: dict[str, dict[str, object]], references: dict[str, np.ndarray]) -> dict[str, dict[str, float | list[float] | None]]:
    """Triangulate each marker and calculate actual final-model errors."""
    metrics: dict[str, dict[str, float | list[float] | None]] = {}
    for name, entry in detections.items():
        observations = [item for item in entry.get("observations", []) if item.get("enabled", True)]
        if entry.get("role") == "control" and len(observations) < 2:
            metrics[name] = {
                "xyz": None,
                "error_x_m": None,
                "error_y_m": None,
                "error_z_m": None,
                "error_3d_m": None,
                "projections": float(len(observations)),
                "reprojection_rms_px": None,
            }
            continue
        try:
            xyz, _ = triangulate_observations(reconstruction, observations)
        except ValueError:
            xyz = None
        if xyz is not None:
            entry["triangulated_model_xyz"] = np.asarray(xyz, dtype=float).tolist()
        residuals = []
        for item in entry.get("observations", []):
            image = reconstruction.find_image_with_name(str(item["image"]))
            projected = image.project_point(xyz) if image is not None and xyz is not None else None
            value = float("inf") if projected is None else float(np.linalg.norm(np.asarray(projected) - [item["u"], item["v"]]))
            item["reprojection_px"] = value
            if item.get("enabled", True):
                residuals.append(value)
        reference = np.asarray(references[name], dtype=float)
        vector = np.asarray(xyz, dtype=float) - reference if xyz is not None and name in references else np.full(3, np.nan)
        metrics[name] = {"xyz": None if xyz is None else np.asarray(xyz, dtype=float).tolist(), "error_x_m": float(vector[0]), "error_y_m": float(vector[1]), "error_z_m": float(vector[2]), "error_3d_m": float(np.linalg.norm(vector)), "projections": float(len(observations)), "reprojection_rms_px": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("inf")}
    return metrics


def populate_review_projections(
    reconstruction: Any, detections: dict[str, dict[str, object]], references: dict[str, np.ndarray],
) -> None:
    """Store final-model predictions and per-observation reprojections."""
    for name, entry in detections.items():
        reference = references.get(name)
        active_obs = [item for item in entry.get("observations", []) if item.get("enabled", True)]
        if len(active_obs) >= 2:
            try:
                tri_xyz, _ = triangulate_observations(reconstruction, active_obs)
                triangulated = [float(v) for v in tri_xyz]
                entry["triangulated_model_xyz"] = triangulated
            except ValueError:
                triangulated = entry.get("triangulated_model_xyz")
        else:
            triangulated = entry.get("triangulated_model_xyz")
        prediction_xyz = reference if reference is not None else triangulated
        reprojected_xyz = reference if entry.get("role") == "pass" else triangulated
        predictions: list[dict[str, float | str]] = []
        for image_id in sorted(reconstruction.reg_image_ids()):
            image = reconstruction.image(image_id)
            camera = reconstruction.camera(image.camera_id)
            if prediction_xyz is not None:
                predicted = project_point(image, camera, np.asarray(prediction_xyz, dtype=float), margin_px=0.0)
                if predicted is not None:
                    predictions.append({"image": str(image.name), "u": float(predicted[0]), "v": float(predicted[1])})
            observation = next((item for item in entry.get("observations", []) if item.get("image") == image.name), None)
            if observation is None or reprojected_xyz is None:
                continue
            reprojected = project_point(image, camera, np.asarray(reprojected_xyz, dtype=float), margin_px=float("inf"))
            if reprojected is None:
                continue
            observation["reprojected_u"], observation["reprojected_v"] = map(float, reprojected)
            observation["reprojection_px"] = float(np.hypot(
                reprojected[0] - float(observation["u"]), reprojected[1] - float(observation["v"]),
            ))
        entry["predictions"] = predictions


def quality_gate(reconstruction: Any, detections: dict[str, dict[str, object]], references: dict[str, np.ndarray], settings: dict[str, Any]) -> tuple[bool, list[str], dict[str, dict[str, float | list[float] | None]]]:
    """Disable poor image clicks and gate on measured 3D pass/check errors."""
    max_rms, min_projections = float(settings.get("max_marker_reprojection_px", 1.0)), int(settings.get("min_projections", 3))
    factor = float(settings.get("observation_outlier_factor", 3.0))
    # First pass uses the fixed reference point for reliable outlier removal.
    for name, entry in detections.items():
        manually_disabled = bool(entry.get("disabled_by_reviewer", False))
        reference = np.asarray(references.get(name, entry.get("triangulated_model_xyz") or [0, 0, 0]), dtype=float)
        residuals = []
        for item in entry.get("observations", []):
            image = reconstruction.find_image_with_name(str(item["image"]))
            projected = image.project_point(reference) if image is not None else None
            error = float("inf") if projected is None else float(np.linalg.norm(np.asarray(projected) - [item["u"], item["v"]]))
            item["reprojection_px"] = error
            if item.get("enabled", True):
                residuals.append((item, error))
        # A single bad click must not inflate the threshold that is intended to
        # identify it.  The median is stable until at least half of a target's
        # observations are corrupted, while the subsequent RMS gate still
        # rejects uniformly bad targets.
        baseline = float(np.median([value for _, value in residuals])) if residuals else float("inf")
        for item, error in residuals:
            if error > factor * max(baseline, 1e-9):
                item["enabled"] = False
        active = [value for item, value in residuals if item.get("enabled", True)]
        if manually_disabled:
            entry["enabled"] = False
            entry["status"] = "disabled"
            entry["reason"] = "disabled by reviewer"
            continue
        if entry.get("role") == "control" and len(active) < 2:
            entry["enabled"] = False
            entry["status"] = "not_found"
            entry["reason"] = "kein Target erkannt – bitte manuell setzen"
            continue
        required_projections = 2 if entry.get("role") == "control" else min_projections
        # Agisoft-style marker pixel error: residuals to the marker's estimated (triangulated) position;
        # the deviation from the surveyed reference is judged separately in metres below.
        active_items = [item for item, _ in residuals if item.get("enabled", True)]
        try:
            estimated, estimated_errors = triangulate_observations(reconstruction, active_items)
            marker_rms = float(np.sqrt(np.mean(np.square(estimated_errors))))
        except ValueError:
            marker_rms = float("inf")
        entry["marker_rms_px"] = marker_rms
        entry["enabled"] = len(active) >= required_projections and marker_rms <= max_rms
        entry["status"] = "accepted" if entry["enabled"] else "needs_review"
        if not entry["enabled"]:
            entry["reason"] = "too few projections" if len(active) < required_projections else "marker reprojection RMS is too high"
    details = point_metrics(reconstruction, detections, references)
    enabled_pass = [name for name, entry in detections.items() if entry.get("role") == "pass" and entry.get("enabled")]
    reasons = []
    if len(enabled_pass) < int(settings.get("min_points", 3)):
        reasons.append(f"fewer than {settings.get('min_points', 3)} good pass points")
    pass_errors = [float(details[name]["error_3d_m"]) for name in enabled_pass if np.isfinite(float(details[name]["error_3d_m"]))]
    limit = float(settings.get("max_residual_warning_m", 0.1))
    if pass_errors and float(np.sqrt(np.mean(np.square(pass_errors)))) > limit:
        reasons.append("pass-point 3D RMS exceeds the warning threshold")
    bad_controls = [name for name, entry in detections.items() if entry.get("role") == "control" and details[name]["error_3d_m"] is not None and np.isfinite(float(details[name]["error_3d_m"])) and float(details[name]["error_3d_m"]) > 3.0 * limit]
    if bad_controls:
        reasons.append("control-point 3D error exceeds the blunder threshold: " + ", ".join(sorted(bad_controls)))
    return not reasons, reasons, details


def leave_one_out_errors(
    model_dir: Path, output_root: Path, detections: dict[str, dict[str, object]], references: dict[str, np.ndarray],
    *, pose_priors: list[Any] | None = None, prior_stddev: float = 3.0,
) -> dict[str, float | None]:
    """Measure each pass point after re-adjusting without that marker."""
    enabled = [name for name, entry in detections.items() if entry.get("role") == "pass" and entry.get("enabled")]
    if len(enabled) < 4:
        return {name: None for name in detections}
    result: dict[str, float | None] = {name: None for name in detections}
    import pycolmap
    for name in enabled:
        reduced = copy.deepcopy(detections)
        reduced[name]["enabled"] = False
        output = output_root / f"leave_one_out_{name}"
        run_gcp_bundle_adjustment(model_dir, output, reduced, references, pose_priors=pose_priors, prior_stddev=prior_stddev, second_pass=False)
        metrics = point_metrics(pycolmap.Reconstruction(output), {name: detections[name]}, {name: references[name]})
        result[name] = float(metrics[name]["error_3d_m"])
    return result


def build_control_point_report(detections: dict[str, dict[str, object]], gate: tuple[bool, list[str], dict[str, dict[str, float | list[float] | None]]], intrinsics_before: dict[str, Any], intrinsics_after: dict[str, Any], *, leave_one_out: dict[str, float | None] | None = None) -> dict[str, Any]:
    """Build the UI-compatible report from final triangulated measurements."""
    passed, reasons, details = gate
    points = []
    for name, entry in detections.items():
        metrics = details[name]
        error_values = [metrics.get(key) for key in ("error_x_m", "error_y_m", "error_z_m")]
        error_x, error_y, error_z = (None if value is None else float(value) for value in error_values)
        metric_error = metrics.get("error_3d_m")
        error_3d = None if metric_error is None else float(metric_error)
        points.append({"name": name, "role": entry.get("role"), "enabled": bool(entry.get("enabled", False)), "status": entry.get("status", "needs_review"), "projections": int(metrics.get("projections", 0)), "reprojection_rms_px": metrics.get("reprojection_rms_px"), "error_x_m": error_x, "error_y_m": error_y, "error_z_m": error_z, "error_3d_m": error_3d, "residual_m": error_3d, "leave_one_out_residual_m": (leave_one_out or {}).get(name), "observations": entry.get("observations", [])})
    return {"gate": {"passed": passed, "reasons": reasons}, "points": points, "totals": _totals(points), "camera_groups": {"before": intrinsics_before, "after": intrinsics_after}}


def camera_intrinsics(reconstruction: Any) -> dict[str, list[float]]:
    return {str(camera_id): [float(value) for value in reconstruction.camera(camera_id).params] for camera_id in reconstruction.cameras}


def _totals(points: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role in ("pass", "control"):
        valid_points = []
        for point in points:
            if point.get("role") != role:
                continue
            if not bool(point.get("enabled", True)):
                continue
            error_3d = point.get("error_3d_m")
            if error_3d is None:
                continue
            try:
                val = float(error_3d)
                if not math.isfinite(val):
                    continue
            except (TypeError, ValueError):
                continue
            valid_points.append(point)

        if not valid_points:
            result[role] = {
                "rms_x_m": None,
                "rms_y_m": None,
                "rms_z_m": None,
                "rms_xy_m": None,
                "rms_3d_m": None,
                "mean_reprojection_px": None,
                "count": 0,
            }
            continue

        def _to_float(v: Any) -> float:
            try:
                f = float(v)
                return f if math.isfinite(f) else 0.0
            except (TypeError, ValueError):
                return 0.0

        xs = np.asarray([_to_float(p.get("error_x_m")) for p in valid_points], dtype=float)
        ys = np.asarray([_to_float(p.get("error_y_m")) for p in valid_points], dtype=float)
        zs = np.asarray([_to_float(p.get("error_z_m")) for p in valid_points], dtype=float)
        xy = np.sqrt(xs ** 2 + ys ** 2)
        xyz = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)

        rms = lambda vals: float(np.sqrt(np.mean(np.square(vals))))

        finite_reprojections = []
        for p in valid_points:
            rep = p.get("reprojection_rms_px")
            if rep is not None:
                try:
                    rep_val = float(rep)
                    if math.isfinite(rep_val):
                        finite_reprojections.append(rep_val)
                except (TypeError, ValueError):
                    pass

        mean_reprojection = float(np.mean(finite_reprojections)) if finite_reprojections else None

        result[role] = {
            "rms_x_m": rms(xs),
            "rms_y_m": rms(ys),
            "rms_z_m": rms(zs),
            "rms_xy_m": rms(xy),
            "rms_3d_m": rms(xyz),
            "mean_reprojection_px": mean_reprojection,
            "count": len(valid_points),
        }
    return result
