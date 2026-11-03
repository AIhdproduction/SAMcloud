"""Real pycolmap regression tests for the GCP adjustment workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from samcloud.control_points.adjustment import _totals, build_control_point_report, leave_one_out_errors, point_metrics, quality_gate, run_gcp_bundle_adjustment
from synthetic_scene import clone_detections, make_scene


SETTINGS = {"max_marker_reprojection_px": 1.0, "min_projections": 3, "min_points": 3, "observation_outlier_factor": 3.0, "max_residual_warning_m": 0.1}


def _adjust(scene, destination: Path, detections=None, references=None, **kwargs):
    return run_gcp_bundle_adjustment(scene.model_dir, destination, clone_detections(scene.detections) if detections is None else detections, scene.references if references is None else references, **kwargs)


def test_gcp_ba_corrects_doming_and_scale(tmp_path: Path) -> None:
    scene = make_scene(tmp_path, perturbed=True)
    before = point_metrics(pycolmap.Reconstruction(scene.model_dir), scene.detections, scene.references)
    result = _adjust(scene, tmp_path / "adjusted")
    metrics = point_metrics(pycolmap.Reconstruction(tmp_path / "adjusted"), scene.detections, scene.references)
    assert before["CP1"]["error_3d_m"] > 0.05
    assert max(float(metrics[name]["error_3d_m"]) for name in ("P1", "P2", "P3", "P4")) < 0.02
    assert max(float(metrics[name]["error_3d_m"]) for name in ("CP1", "CP2")) < 0.05
    assert abs(result["intrinsics"]["1"][4] + 0.055) < 0.002


def test_control_points_do_not_influence_adjustment(tmp_path: Path) -> None:
    scene = make_scene(tmp_path, perturbed=True)
    first, second = clone_detections(scene.detections), clone_detections(scene.detections)
    shifted = {name: value.copy() for name, value in scene.references.items()}; shifted["CP1"] += [5.0, 0.0, 0.0]
    _adjust(scene, tmp_path / "first", first); _adjust(scene, tmp_path / "second", second, shifted)
    first_model, second_model = pycolmap.Reconstruction(tmp_path / "first"), pycolmap.Reconstruction(tmp_path / "second")
    for image_id in first_model.reg_image_ids():
        assert np.max(np.abs(first_model.image(image_id).cam_from_world().matrix() - second_model.image(image_id).cam_from_world().matrix())) < 1e-9
    first_error = point_metrics(first_model, first, scene.references)["CP1"]["error_3d_m"]
    second_error = point_metrics(second_model, second, shifted)["CP1"]["error_3d_m"]
    assert 4.94 < float(second_error) - float(first_error) < 5.06


def test_gate_disables_blunder_observation_and_passes(tmp_path: Path) -> None:
    scene = make_scene(tmp_path); detections = clone_detections(scene.detections)
    detections["P2"]["observations"][0]["u"] += 5.0
    passed, reasons, _ = quality_gate(pycolmap.Reconstruction(scene.model_dir), detections, scene.references, SETTINGS)
    assert not detections["P2"]["observations"][0]["enabled"]
    assert detections["P2"]["enabled"]
    assert passed and reasons == []


def test_gate_fails_with_too_few_good_pass_points(tmp_path: Path) -> None:
    scene = make_scene(tmp_path); detections = clone_detections(scene.detections)
    for name in ("P1", "P2"):
        for observation in detections[name]["observations"]:
            observation["u"] += 8.0
    passed, reasons, _ = quality_gate(pycolmap.Reconstruction(scene.model_dir), detections, scene.references, SETTINGS)
    assert not passed and any("fewer than 3 good pass points" in reason for reason in reasons)
    assert not detections["P1"]["enabled"] and not detections["P2"]["enabled"]
    assert detections["P1"]["reason"] and detections["P2"]["reason"]


def test_undetected_control_point_is_reported_without_failing_gate(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    detections = clone_detections(scene.detections)
    detections["CP1"]["observations"] = []

    gate = quality_gate(pycolmap.Reconstruction(scene.model_dir), detections, scene.references, SETTINGS)
    report = build_control_point_report(detections, gate, {}, {})
    control = next(point for point in report["points"] if point["name"] == "CP1")

    assert gate[0]
    assert control["status"] == "not_found" and not control["enabled"]
    assert control["error_x_m"] is None and control["error_y_m"] is None
    assert control["error_z_m"] is None and control["error_3d_m"] is None
    assert detections["CP1"]["reason"] == "kein Target erkannt – bitte manuell setzen"


def test_undetected_pass_point_still_counts_against_minimum(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    detections = clone_detections(scene.detections)
    detections["P1"]["observations"] = []

    passed, reasons, _ = quality_gate(
        pycolmap.Reconstruction(scene.model_dir), detections, scene.references, {**SETTINGS, "min_points": 4},
    )

    assert not passed
    assert any("fewer than 4 good pass points" in reason for reason in reasons)


def test_report_errors_and_rms_totals_are_real(tmp_path: Path) -> None:
    scene = make_scene(tmp_path); references = {name: value.copy() for name, value in scene.references.items()}; references["P3"] += [0.03, -0.04, 0.0]
    detections = clone_detections(scene.detections); gate = quality_gate(pycolmap.Reconstruction(scene.model_dir), detections, references, SETTINGS)
    report = build_control_point_report(detections, gate, {}, {})
    point = next(value for value in report["points"] if value["name"] == "P3")
    assert abs(abs(float(point["error_x_m"])) - 0.03) < 0.005 and abs(abs(float(point["error_y_m"])) - 0.04) < 0.005
    errors = [float(value["error_3d_m"]) for value in report["points"] if value["role"] == "pass"]
    assert np.isclose(report["totals"]["pass"]["rms_3d_m"], np.sqrt(np.mean(np.square(errors))), atol=1e-12)


def test_rerun_from_pristine_is_idempotent(tmp_path: Path) -> None:
    scene = make_scene(tmp_path, perturbed=True); reports = []
    for index in range(2):
        detections = clone_detections(scene.detections); result = _adjust(scene, tmp_path / f"run_{index}", detections)
        gate = quality_gate(pycolmap.Reconstruction(tmp_path / f"run_{index}"), detections, scene.references, SETTINGS)
        reports.append(build_control_point_report(detections, gate, result["intrinsics_before"], result["intrinsics"]))
    _assert_close(reports[0], reports[1])


def test_pose_priors_reach_pose_prior_adjuster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene = make_scene(tmp_path)
    reconstruction = pycolmap.Reconstruction(scene.model_dir)

    offset = np.array([1.0e7, 0.0, 0.0])
    priors = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        data_id = getattr(image, "data_id", None)
        if data_id is None:
            if hasattr(pycolmap, "CorrDataId"):
                try:
                    data_id = pycolmap.CorrDataId(image.image_id)
                except TypeError:
                    data_id = pycolmap.CorrDataId(id=image.image_id)
            else:
                data_id = getattr(image, "corr_data_id", None)

        prior = pycolmap.PosePrior(
            corr_data_id=data_id,
            position=image.projection_center() + offset,
            position_covariance=np.diag([0.01 ** 2] * 3),
            coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
        )
        priors.append(prior)

    calls: list[tuple] = []
    real = pycolmap.create_pose_prior_bundle_adjuster

    def wrapped(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(pycolmap, "create_pose_prior_bundle_adjuster", wrapped)
    _adjust(scene, tmp_path / "with_priors", pose_priors=priors, second_pass=False)

    assert len(calls) == 1
    passed_priors = calls[0][3]
    assert len(passed_priors) == len(priors)
    for prior in passed_priors:
        assert prior.coordinate_system == pycolmap.PosePriorCoordinateSystem.CARTESIAN
        image_id = int(prior.corr_data_id.id)
        expected_centre = reconstruction.image(image_id).projection_center()
        assert np.linalg.norm(np.asarray(prior.position, dtype=float) - expected_centre) < 0.01

    monkeypatch.setattr(pycolmap, "create_pose_prior_bundle_adjuster", lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected prior adjuster")))
    _adjust(scene, tmp_path / "without_priors", pose_priors=None, second_pass=False)


def test_leave_one_out_errors(tmp_path: Path) -> None:
    scene = make_scene(tmp_path, perturbed=True); references = {name: value.copy() for name, value in scene.references.items()}; references["P4"] += [0.30, 0.0, 0.0]
    detections = clone_detections(scene.detections); _adjust(scene, tmp_path / "full", detections, references)
    normal = point_metrics(pycolmap.Reconstruction(tmp_path / "full"), detections, references)["P4"]["error_3d_m"]
    errors = leave_one_out_errors(scene.model_dir, tmp_path / "loo", detections, references)
    assert errors["P4"] is not None and float(errors["P4"]) > float(normal) + 0.10


def _assert_close(first, second) -> None:
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first: _assert_close(first[key], second[key])
    elif isinstance(first, list):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True): _assert_close(left, right)
    elif isinstance(first, float):
        assert np.isclose(first, second, atol=1e-9, rtol=0.0) or (np.isnan(first) and np.isnan(second))
    else: assert first == second


def test_wgs84_priors_are_converted_into_model_frame(tmp_path: Path) -> None:
    from samcloud.control_points.adjustment import pose_priors_in_model_frame
    from samcloud.control_points.georeference import enu_rotation_at_ecef
    from samcloud.pointcloud.georeference import _transform_strict, strict_transformer

    scene = make_scene(tmp_path)
    reconstruction = pycolmap.Reconstruction(scene.model_dir)

    origin_lat, origin_lon, origin_h = 47.1345, 8.7555, 953.8
    to_ecef = strict_transformer("EPSG:4979", "EPSG:4978")
    origin_x, origin_y, origin_z = _transform_strict(to_ecef, [origin_lon], [origin_lat], [origin_h])
    origin = np.array([float(origin_x[0]), float(origin_y[0]), float(origin_z[0])])
    E = enu_rotation_at_ecef(origin)

    to_wgs84 = strict_transformer("EPSG:4978", "EPSG:4979")
    priors: list[Any] = []
    camera_centres: dict[int, np.ndarray] = {}

    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        c = np.asarray(image.projection_center(), dtype=float)
        camera_centres[image_id] = c
        ecef = origin + E.T @ c
        lon, lat, h = _transform_strict(to_wgs84, [ecef[0]], [ecef[1]], [ecef[2]])

        data_id = getattr(image, "data_id", None)
        if data_id is None:
            if hasattr(pycolmap, "CorrDataId"):
                try:
                    data_id = pycolmap.CorrDataId(image.image_id)
                except TypeError:
                    data_id = pycolmap.CorrDataId(id=image.image_id)
            else:
                data_id = getattr(image, "corr_data_id", None)

        prior = pycolmap.PosePrior(
            corr_data_id=data_id,
            position=np.array([float(lat[0]), float(lon[0]), float(h[0])]),
            position_covariance=np.diag([0.01 ** 2, 0.01 ** 2, 0.03 ** 2]),
            coordinate_system=pycolmap.PosePriorCoordinateSystem.WGS84,
        )
        priors.append(prior)

    converted = pose_priors_in_model_frame(reconstruction, priors)
    assert len(converted) == len(priors)
    for prior in converted:
        assert prior.coordinate_system == pycolmap.PosePriorCoordinateSystem.CARTESIAN
        image_id = int(prior.corr_data_id.id)
        distance = float(np.linalg.norm(np.asarray(prior.position, dtype=float) - camera_centres[image_id]))
        assert distance < 0.001


def test_gcp_ba_uses_full_weight_marker_observations(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    output_dir = tmp_path / "adjusted"
    references_shifted = {
        name: (ref - np.array([0.0, 0.0, 0.15]) if scene.detections[name]["role"] == "pass" else ref.copy())
        for name, ref in scene.references.items()
    }
    _adjust(scene, output_dir, references=references_shifted, pose_priors=None)
    adjusted = pycolmap.Reconstruction(output_dir)
    metrics = point_metrics(adjusted, scene.detections, references_shifted)
    for name in ("P1", "P2", "P3", "P4"):
        assert abs(float(metrics[name]["error_z_m"])) < 0.02


def test_gate_uses_estimated_marker_position_for_pixel_error(tmp_path: Path) -> None:
    from samcloud.control_points.auto_detect import project_point

    scene = make_scene(tmp_path)
    reconstruction = pycolmap.Reconstruction(scene.model_dir)
    detections = clone_detections(scene.detections)

    shifted_pt = scene.references["P1"] + np.array([0.0, 0.0, 0.012])
    new_obs = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        camera = reconstruction.camera(image.camera_id)
        uv = project_point(image, camera, shifted_pt, margin_px=0.0)
        if uv is not None:
            new_obs.append({"image": image.name, "u": float(uv[0]), "v": float(uv[1]), "enabled": True, "source": "auto"})
    detections["P1"]["observations"] = new_obs

    settings = {**SETTINGS, "max_marker_reprojection_px": 1.5}
    passed, reasons, details = quality_gate(reconstruction, detections, scene.references, settings)

    entry = detections["P1"]
    assert entry["enabled"]
    assert entry["marker_rms_px"] < 0.2
    assert abs(float(details["P1"]["error_3d_m"]) - 0.012) <= 0.002


def test_automatic_disable_is_not_treated_as_reviewer_disable(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    detections = clone_detections(scene.detections)
    detections["P1"]["enabled"] = False
    detections["P1"]["status"] = "needs_review"
    detections["P1"].pop("disabled_by_reviewer", None)

    reconstruction = pycolmap.Reconstruction(scene.model_dir)
    passed, reasons, details = quality_gate(reconstruction, detections, scene.references, SETTINGS)

    assert detections["P1"]["enabled"] is True
    assert detections["P1"]["status"] == "accepted"

    detections["P1"]["disabled_by_reviewer"] = True
    passed, reasons, details = quality_gate(reconstruction, detections, scene.references, SETTINGS)

    assert detections["P1"]["enabled"] is False
    assert detections["P1"]["status"] == "disabled"
    assert detections["P1"]["reason"] == "disabled by reviewer"


def test_totals_ignore_disabled_points_and_use_none() -> None:
    points = [
        {
            "name": "P1",
            "role": "pass",
            "enabled": True,
            "error_x_m": 0.01,
            "error_y_m": 0.0,
            "error_z_m": 0.0,
            "error_3d_m": 0.01,
        },
        {
            "name": "P2",
            "role": "pass",
            "enabled": False,
            "error_x_m": 0.5,
            "error_y_m": 0.0,
            "error_z_m": 0.0,
            "error_3d_m": 0.5,
        },
        {
            "name": "C1",
            "role": "control",
            "enabled": True,
            "error_x_m": None,
            "error_y_m": None,
            "error_z_m": None,
            "error_3d_m": None,
        },
    ]
    totals = _totals(points)

    assert totals["pass"]["count"] == 1
    assert totals["pass"]["rms_x_m"] == pytest.approx(0.01)
    assert totals["pass"]["rms_y_m"] == pytest.approx(0.0)
    assert totals["pass"]["rms_z_m"] == pytest.approx(0.0)
    assert totals["pass"]["rms_xy_m"] == pytest.approx(0.01)
    assert totals["pass"]["rms_3d_m"] == pytest.approx(0.01)

    assert totals["control"]["count"] == 0
    assert totals["control"]["rms_x_m"] is None
    assert totals["control"]["rms_y_m"] is None
    assert totals["control"]["rms_z_m"] is None
    assert totals["control"]["rms_xy_m"] is None
    assert totals["control"]["rms_3d_m"] is None
    assert totals["control"]["mean_reprojection_px"] is None


