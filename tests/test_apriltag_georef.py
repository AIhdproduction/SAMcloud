"""AprilTag georeferencing from surveyed tag centres on a synthetic, arbitrarily placed room model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
import pytest

from samcloud.control_points.apriltags import (
    TagGeorefUnavailable,
    save_tag_detections,
    tags_are_non_collinear,
    triangulate_tags,
)
from samcloud.control_points.georeference import coordinates_to_ecef
from samcloud.control_points.repository import load_detections, save_detections, write_apriltag_coordinates
from samcloud.engine import pipeline
from test_apriltag_pipeline import _full_cache, _images_for
from test_apriltag_scale import TAG_WALLS, TRUE_SCALE, _build_reconstruction, _observations

CRS = "EPSG:2056+5728"
ORIGIN = np.array([2600100.0, 1200050.0, 450.0])
YAW = np.deg2rad(23.0)
SETTINGS = {"max_marker_reprojection_px": 1.0, "min_projections": 2, "min_points": 3,
            "min_observations_per_point": 2, "observation_outlier_factor": 3.0,
            "max_residual_warning_m": 0.1, "max_gate_iterations": 2}


def _surveyed(tag_id: int, rng: np.random.Generator) -> np.ndarray:
    """LV95 E/N plus LN02 height of a tag centre, with 1 mm survey noise."""
    room = TAG_WALLS[tag_id][0]
    east_north = np.array([[np.cos(YAW), -np.sin(YAW)], [np.sin(YAW), np.cos(YAW)]]) @ room[:2]
    return ORIGIN + np.array([east_north[0], east_north[1], room[2]]) + rng.normal(0.0, 0.001, 3)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _world(root: Path, tag_ids: tuple[int, ...] = (11, 12, 13, 14)):
    """Tags seen in a room model that was moved by an arbitrary rigid transform (scale stays 1/TRUE_SCALE)."""
    reconstruction, _ = _build_reconstruction(root)
    detections = _observations(reconstruction, {tag_id: 0.150 for tag_id in tag_ids})
    rng = np.random.default_rng(5)
    reconstruction.transform(pycolmap.Sim3d(1.0, pycolmap.Rotation3d(_random_rotation(rng)), rng.uniform(-3.0, 3.0, 3)))
    model_dir = root / "world_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(model_dir)
    return reconstruction, model_dir, detections


def _write_csv(control_points: Path, rows: dict[int, tuple[np.ndarray, str]]) -> None:
    write_apriltag_coordinates(pd.DataFrame([
        {"id": tag_id, "x": xyz[0], "y": xyz[1], "z": xyz[2], "role": role} for tag_id, (xyz, role) in rows.items()
    ]), control_points / "apriltags.csv")


def _local_from_survey(georeference: dict, xyz: np.ndarray) -> np.ndarray:
    ecef = coordinates_to_ecef(np.asarray([xyz], dtype=float), CRS)[0]
    return np.asarray(georeference["rotation"]).T @ (ecef - np.asarray(georeference["translation"]))


def _run_control_points(tmp_path: Path, roles: dict[int, str], surveyed: dict[int, np.ndarray]):
    reconstruction, model_dir, detections = _world(tmp_path)
    control_points = tmp_path / "control_points"
    _write_csv(control_points, {tag_id: (surveyed[tag_id], role) for tag_id, role in roles.items()})
    geometries = triangulate_tags(reconstruction, detections)
    sparse = tmp_path / "products" / "sparse"
    sparse.mkdir(parents=True)
    shutil.copytree(model_dir, sparse / "pre_gcp")
    report_path = tmp_path / "reports" / "control_points_report.json"
    result = pipeline._apriltag_control_points(
        sparse / "pre_gcp", sparse / "pre_gcp_txt", tmp_path / "images", control_points, CRS, SETTINGS, report_path,
        geometries, detections,
    )
    return result, json.loads(report_path.read_text(encoding="utf-8")), tmp_path / "products"


def test_four_pass_tags_georeference_the_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    surveyed = {tag_id: _surveyed(tag_id, rng) for tag_id in (11, 12, 13, 14)}

    model, report, products = _run_control_points(tmp_path, {t: "pass" for t in surveyed}, surveyed)

    assert model == products / "sparse" / "gcp_adjusted"
    assert report["gate"]["passed"] is True
    assert report["totals"]["pass"]["count"] == 4
    assert report["totals"]["pass"]["rms_3d_m"] < 0.005
    georeference = json.loads((products / "georeference.json").read_text(encoding="utf-8"))
    assert georeference["source_frame"] == "gcp_local_enu"
    assert georeference["method"] == "gcp_bundle_adjustment"
    assert georeference["control_points"]["crs"] == CRS
    detections = load_detections(tmp_path / "control_points" / "detections.json")
    assert sorted(detections) == ["TAG11", "TAG12", "TAG13", "TAG14"]
    assert all(entry["role"] == "pass" and entry["enabled"] for entry in detections.values())
    assert all(item["source"] == "apriltag" for entry in detections.values() for item in entry["observations"])
    assert (products / "sparse" / "final_model.txt").read_text(encoding="utf-8").strip() == "gcp_adjusted"


def test_control_tag_is_checked_but_not_used_in_adjustment(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    surveyed = {tag_id: _surveyed(tag_id, rng) for tag_id in (11, 12, 13, 14)}
    surveyed[14] = surveyed[14] + np.array([0.0, 0.0, 0.006])  # a 6 mm survey offset must stay visible

    model, report, _ = _run_control_points(tmp_path, {11: "pass", 12: "pass", 13: "pass", 14: "control"}, surveyed)

    assert model is not None
    points = {point["name"]: point for point in report["points"]}
    assert points["TAG14"]["role"] == "control"
    assert 0.003 < points["TAG14"]["error_3d_m"] < 0.01
    assert report["totals"]["pass"]["count"] == 3
    assert report["totals"]["control"]["count"] == 1


def test_collinear_tags_are_unavailable_and_step_falls_back_to_scale(tmp_path: Path) -> None:
    reconstruction, model_dir, detections = _world(tmp_path, (11, 12, 13))
    control_points, report_dir = tmp_path / "control_points", tmp_path / "reports"
    line = {tag_id: ORIGIN + np.array([float(index), 0.5 * index, 0.0]) for index, tag_id in enumerate((11, 12, 13))}
    _write_csv(control_points, {tag_id: (xyz, "pass") for tag_id, xyz in line.items()})
    assert not tags_are_non_collinear(np.asarray(list(line.values())))
    geometries = triangulate_tags(reconstruction, detections)
    with pytest.raises(TagGeorefUnavailable):
        pipeline._apriltag_control_points(
            tmp_path / "p" / "pre_gcp", tmp_path / "p" / "pre_gcp_txt", tmp_path / "images", control_points, CRS, SETTINGS,
            report_dir / "control_points_report.json", geometries, detections,
        )

    images_dir = tmp_path / "images"
    _images_for(reconstruction, images_dir)
    save_tag_detections(control_points / "apriltag_detections.json", _full_cache(reconstruction, detections))
    outcome = pipeline._apriltag_step(
        model_dir, tmp_path / "products" / "sparse", images_dir, control_points, report_dir,
        outlier_pct=3.0, crs=CRS, settings=SETTINGS,
    )

    assert outcome.georeferenced is False and outcome.gate_failed is False
    assert outcome.report["mode"] == "scale"
    assert outcome.report["scale_factor"] == pytest.approx(TRUE_SCALE, rel=0.005)
    assert any("Weniger als 3 eingemessene AprilTags" in warning for warning in outcome.report["warnings"])
    assert not (tmp_path / "products" / "georeference.json").exists()


def test_two_surveyed_tags_fall_back_to_scale(tmp_path: Path) -> None:
    reconstruction, model_dir, detections = _world(tmp_path, (11, 12, 13))
    rng = np.random.default_rng(3)
    control_points, report_dir, images_dir = tmp_path / "control_points", tmp_path / "reports", tmp_path / "images"
    _write_csv(control_points, {11: (_surveyed(11, rng), "pass"), 12: (_surveyed(12, rng), "pass")})
    _images_for(reconstruction, images_dir)
    save_tag_detections(control_points / "apriltag_detections.json", _full_cache(reconstruction, detections))

    outcome = pipeline._apriltag_step(
        model_dir, tmp_path / "products" / "sparse", images_dir, control_points, report_dir,
        outlier_pct=3.0, crs=CRS, settings=SETTINGS,
    )

    assert outcome.georeferenced is False
    assert outcome.report["mode"] == "scale"
    assert outcome.model.name == "0_tagscale"
    assert any("nur Maßstab" in warning for warning in outcome.report["warnings"])
    stored = json.loads((report_dir / "apriltag_report.json").read_text(encoding="utf-8"))
    assert stored["mode"] == "scale"


def test_georef_report_holds_tag_geometry_in_final_frame(tmp_path: Path) -> None:
    reconstruction, model_dir, detections = _world(tmp_path)
    rng = np.random.default_rng(12)
    surveyed = {tag_id: _surveyed(tag_id, rng) for tag_id in (11, 12, 13, 14)}
    control_points, report_dir, images_dir = tmp_path / "control_points", tmp_path / "reports", tmp_path / "images"
    _write_csv(control_points, {tag_id: (xyz, "pass") for tag_id, xyz in surveyed.items()})
    _images_for(reconstruction, images_dir)
    save_tag_detections(control_points / "apriltag_detections.json", _full_cache(reconstruction, detections))
    sparse = tmp_path / "products" / "sparse"

    outcome = pipeline._apriltag_step(
        model_dir, sparse, images_dir, control_points, report_dir, outlier_pct=3.0, crs=CRS, settings=SETTINGS,
    )

    assert outcome.georeferenced is True and outcome.gate_failed is False
    assert outcome.model == sparse / "gcp_adjusted"
    assert (sparse / "pre_gcp" / "images.bin").is_file() and (sparse / "pre_gcp_txt" / "images.txt").is_file()
    stored = json.loads((report_dir / "apriltag_report.json").read_text(encoding="utf-8"))
    assert stored["mode"] == "georef"
    georeference = json.loads((tmp_path / "products" / "georeference.json").read_text(encoding="utf-8"))
    assert len(stored["tags"]) == 4
    for tag in stored["tags"]:
        expected = _local_from_survey(georeference, surveyed[tag["id"]])
        assert np.linalg.norm(np.asarray(tag["center_xyz"]) - expected) < 0.005
        assert np.asarray(tag["corners_xyz"]).shape == (4, 3)
        assert tag["error_3d_m"] is not None and tag["error_3d_m"] < 0.005
        assert tag["size_mm_after"] == pytest.approx(150.0, abs=3.0)
    assert stored["scale_applied"] == pytest.approx(TRUE_SCALE, rel=0.01)


def test_apriltag_observation_source_round_trips_through_detections(tmp_path: Path) -> None:
    entry = {
        "role": "pass", "enabled": True, "status": "accepted", "triangulated_model_xyz": [1.0, 2.0, 3.0], "predictions": [],
        "observations": [
            {"image": "a.jpg", "u": 10.5, "v": 20.25, "score": 1.0, "source": "apriltag", "enabled": True},
            {"image": "b.jpg", "u": 11.5, "v": 21.25, "score": 1.0, "source": "apriltag", "enabled": True},
        ],
    }
    path = tmp_path / "detections.json"

    save_detections(path, {"TAG7": entry})
    loaded = load_detections(path)

    assert [item["source"] for item in loaded["TAG7"]["observations"]] == ["apriltag", "apriltag"]
    assert loaded["TAG7"]["observations"][0]["u"] == pytest.approx(10.5)
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["TAG7"]["observations"][0]["source"] = "unknown"
    with pytest.raises(ValueError):
        save_detections(path, bad)


def test_resume_reseeds_from_reviewed_detections_on_pre_gcp(tmp_path: Path) -> None:
    """The optimise/resume path re-triangulates tag centres from detections.json instead of using GPS."""
    rng = np.random.default_rng(14)
    surveyed = {tag_id: _surveyed(tag_id, rng) for tag_id in (11, 12, 13, 14)}
    first_model, _, products = _run_control_points(tmp_path, {t: "pass" for t in surveyed}, surveyed)
    assert first_model is not None
    control_points = tmp_path / "control_points"
    reviewed = load_detections(control_points / "detections.json")
    reviewed["TAG14"]["disabled_by_reviewer"] = True
    save_detections(control_points / "detections.json", reviewed)
    sparse = products / "sparse"

    again = pipeline._rerun_control_points(
        sparse / "pre_gcp", sparse / "pre_gcp_txt", tmp_path / "images", control_points, CRS,
        products / "gps_ref.txt", SETTINGS, tmp_path / "reports" / "again.json", rtk=False,
    )

    assert again == sparse / "gcp_adjusted"
    report = json.loads((tmp_path / "reports" / "again.json").read_text(encoding="utf-8"))
    points = {point["name"]: point for point in report["points"]}
    assert points["TAG14"]["enabled"] is False
    assert report["totals"]["pass"]["count"] == 3


LOCAL_ORIGIN = np.array([1000.0, 2000.0, 100.0])


def _surveyed_local(tag_id: int, rng: np.random.Generator) -> np.ndarray:
    """Tag centre in a local survey system (metres, rotated 31 degrees, far from the origin) with 1 mm noise."""
    room = TAG_WALLS[tag_id][0]
    yaw = np.deg2rad(31.0)
    xy = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ room[:2]
    return LOCAL_ORIGIN + np.array([xy[0], xy[1], room[2]]) + rng.normal(0.0, 0.001, 3)


def _run_local_control_points(tmp_path: Path, roles: dict[int, str], surveyed: dict[int, np.ndarray]):
    reconstruction, model_dir, detections = _world(tmp_path)
    control_points = tmp_path / "control_points"
    _write_csv(control_points, {tag_id: (surveyed[tag_id], role) for tag_id, role in roles.items()})
    geometries = triangulate_tags(reconstruction, detections)
    sparse = tmp_path / "products" / "sparse"
    sparse.mkdir(parents=True)
    shutil.copytree(model_dir, sparse / "pre_gcp")
    report_path = tmp_path / "reports" / "control_points_report.json"
    result = pipeline._apriltag_control_points(
        sparse / "pre_gcp", sparse / "pre_gcp_txt", tmp_path / "images", control_points, "LOCAL", SETTINGS, report_path,
        geometries, detections,
    )
    return result, json.loads(report_path.read_text(encoding="utf-8")), tmp_path / "products", detections


def test_local_tag_coordinates_place_model_exactly_in_local_frame(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    surveyed = {tag_id: _surveyed_local(tag_id, rng) for tag_id in (11, 12, 13, 14)}

    model, report, products, detections = _run_local_control_points(tmp_path, {t: "pass" for t in surveyed}, surveyed)

    assert model == products / "sparse" / "gcp_adjusted"
    assert report["gate"]["passed"] is True
    assert report["totals"]["pass"]["rms_3d_m"] < 0.005
    assert report["crs"] == "LOCAL" and report["source_frame"] == "local_tags" and report["method"] == "local_tags"
    assert not (products / "georeference.json").exists()
    assert pipeline._is_georeferenced(products) is True
    final = triangulate_tags(pycolmap.Reconstruction(model), detections)
    for tag_id, xyz in surveyed.items():
        assert np.linalg.norm(final[tag_id].center_xyz - xyz) < 0.005  # absolute local coordinates, no offset


def test_local_control_tag_is_checked_in_local_frame(tmp_path: Path) -> None:
    rng = np.random.default_rng(22)
    surveyed = {tag_id: _surveyed_local(tag_id, rng) for tag_id in (11, 12, 13, 14)}
    surveyed[14] = surveyed[14] + np.array([0.0, 0.0, 0.006])

    model, report, _, _ = _run_local_control_points(
        tmp_path, {11: "pass", 12: "pass", 13: "pass", 14: "control"}, surveyed,
    )

    assert model is not None
    points = {point["name"]: point for point in report["points"]}
    assert points["TAG14"]["role"] == "control"
    assert 0.003 < points["TAG14"]["error_3d_m"] < 0.010
    assert report["totals"]["control"]["count"] == 1


def test_local_crs_is_not_accepted_for_the_gps_automatic_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_control_point_pipeline import _write_points
    from synthetic_scene import make_scene

    products = tmp_path / "products"
    scene = make_scene(products / "sparse")
    control_points, model_txt = tmp_path / "control_points", products / "sparse_txt"
    control_points.mkdir()
    model_txt.mkdir()
    _write_points(control_points, scene.references)
    gps_ref = products / "gps_ref.txt"
    gps_ref.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(pipeline, "fit_enu_to_ecef", lambda *_args: {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [6371000.0, 0.0, 0.0]})
    report_path = tmp_path / "reports" / "control_points_report.json"

    result = pipeline._automatic_control_points(
        scene.model_dir, model_txt, tmp_path / "images", control_points, "LOCAL", gps_ref, SETTINGS, report_path, rtk=False,
    )

    assert result is None
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gate"]
    assert gate["passed"] is False and "ohne GPS" in gate["reasons"][0]
    assert pipeline._is_georeferenced(tmp_path / "products") is False
