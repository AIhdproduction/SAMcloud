"""End-to-end control-point pipeline tests using real pycolmap models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pycolmap
import pytest
import yaml

from samcloud.app import processing
from samcloud.app.pages import settings as settings_page
from samcloud.control_points.repository import save_detections, write_control_points
from samcloud.engine import pipeline
from samcloud.engine.run_summary import RunSummary
from synthetic_scene import clone_detections, make_scene


SETTINGS = {"max_marker_reprojection_px": 1.0, "min_projections": 3, "min_points": 3,
            "min_observations_per_point": 2, "observation_outlier_factor": 3.0,
            "max_residual_warning_m": 0.1, "max_gate_iterations": 1}


def _write_points(directory: Path, references: dict[str, np.ndarray]) -> None:
    rows = [{"name": name, "x": xyz[0], "y": xyz[1], "z": xyz[2], "role": "pass" if name.startswith("P") else "control"} for name, xyz in references.items()]
    write_control_points(pd.DataFrame(rows), directory / "control_points.csv")


def _write_text_model(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cameras.txt").write_text("1 PINHOLE 10 10 5 5 5 5\n", encoding="utf-8")
    (directory / "images.txt").write_text("1 1 0 0 0 0 0 0 1 image.jpg\n\n", encoding="utf-8")
    (directory / "points3D.txt").write_text("1 0 0 1 255 255 255 0 1 0\n", encoding="utf-8")
    (directory / "rigs.txt").write_text("1 1 CAMERA 1\n", encoding="utf-8")
    (directory / "frames.txt").write_text("1 1 1 0 0 0 0 0 0 1 CAMERA 1 1\n", encoding="utf-8")


def _fake_run(commands: list[list[str]], cache_states: list[bool] | None = None):
    def run(command, **_kwargs):
        commands.append(command)
        action = command[1]
        if action == "mapper":
            _write_text_model(Path(command[command.index("--output_path") + 1]) / "0")
        elif action == "image_undistorter":
            output = Path(command[command.index("--output_path") + 1])
            (output / "images").mkdir(parents=True, exist_ok=True)
            _write_text_model(output / "sparse")
        elif action == "model_converter":
            output = Path(command[command.index("--output_path") + 1])
            assert output.is_dir(), "COLMAP requires a pre-existing model_converter output directory"
            _write_text_model(output)
        elif action == "patch_match_stereo":
            workspace = Path(command[command.index("--workspace_path") + 1])
            if cache_states is not None:
                cache_states.append(
                    not (workspace / "stereo" / "depth_maps" / "stale.bin").exists()
                    and not (workspace / "stereo" / "normal_maps" / "stale.bin").exists()
                )
            (workspace / "stereo" / "depth_maps").mkdir(parents=True, exist_ok=True)
        elif action == "stereo_fusion":
            Path(command[command.index("--output_path") + 1]).write_bytes(b"ply")
    return run


def test_after_alignment_stops_after_undistortion_and_text_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "images"; images.mkdir(); (images / "image.jpg").write_bytes(b"image")
    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run", _fake_run(commands))
    monkeypatch.setattr(pipeline, "find_colmap_exe", lambda _: "colmap")
    monkeypatch.setattr(sys, "argv", ["pipeline", "--capture-type", "camera", "--images", str(images), "--outdir", str(tmp_path / "products"), "--control-points-mode", "after_alignment", "--skip-export"])
    pipeline.main()
    actions = [command[1] for command in commands]
    assert "image_undistorter" in actions and "model_converter" in actions
    assert "patch_match_stereo" not in actions


def _resume_args(control_points: Path) -> SimpleNamespace:
    return SimpleNamespace(control_points_dir=str(control_points), control_points_crs="EPSG:4979", colmap_bin="colmap", skip_export=True)


def test_resume_uses_pristine_backup_clears_cache_and_never_repeats_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outdir, dense, control_points = tmp_path / "products", tmp_path / "products" / "dense", tmp_path / "control_points"
    _write_text_model(dense / "sparse"); _write_text_model(dense / "sparse_txt")
    (dense / "images").mkdir(); (dense / "stereo" / "depth_maps").mkdir(parents=True); (dense / "stereo" / "normal_maps").mkdir()
    control_points.mkdir()
    (dense / "stereo" / "depth_maps" / "stale.bin").write_bytes(b"stale")
    (dense / "stereo" / "normal_maps" / "stale.bin").write_bytes(b"stale")
    commands: list[list[str]] = []; received: list[Path] = []; transformed: list[Path] = []; cache_states: list[bool] = []
    monkeypatch.setattr(pipeline, "run", _fake_run(commands, cache_states)); monkeypatch.setattr(pipeline, "find_colmap_exe", lambda _: "colmap")
    def georeference(model, *_args, **_kwargs):
        received.append(Path(model)); return ({"scale": 1., "rotation": np.eye(3).tolist(), "translation": [0., 0., 0.], "pairs": 3, "residual_m": {"mean": 0., "p95": 0., "max": 0.}}, {"points": []}, (1., np.eye(3), np.zeros(3)))
    def transform(source, destination, *_args): transformed.append(Path(source)); _write_text_model(Path(destination))
    monkeypatch.setattr(pipeline, "georeference_from_control_points", georeference)
    monkeypatch.setattr(pipeline, "transform_text_model", transform)
    monkeypatch.setattr(pipeline, "_classify_and_finish", lambda *_args: None)
    summary = RunSummary(outdir, {})
    for index in range(2):
        pipeline._resume_after_control_points(_resume_args(control_points), outdir, tmp_path / "reports", summary, None, {}, 10, 1, 0, outdir / "classified.ply")
        if index == 0:
            (dense / "stereo" / "depth_maps" / "stale.bin").write_bytes(b"stale")
            (dense / "stereo" / "normal_maps").mkdir(parents=True, exist_ok=True)
            (dense / "stereo" / "normal_maps" / "stale.bin").write_bytes(b"stale")
    pristine = dense / "sparse_txt_pre_gcp"
    assert received == [pristine, pristine]
    assert transformed == [pristine, pristine]
    assert cache_states == [True, True]
    actions = [command[1] for command in commands]
    assert "patch_match_stereo" in actions and "stereo_fusion" in actions
    assert not {"feature_extractor", "exhaustive_matcher", "mapper", "image_undistorter"}.intersection(actions)


def test_resume_rejects_insufficient_control_points_before_dense_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outdir, dense, control_points = tmp_path / "products", tmp_path / "products" / "dense", tmp_path / "control_points"
    _write_text_model(dense / "sparse"); _write_text_model(dense / "sparse_txt"); (dense / "images").mkdir(); control_points.mkdir()
    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run", _fake_run(commands))
    monkeypatch.setattr(pipeline, "georeference_from_control_points", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("At least 3 usable control points are required; found 2")))
    with pytest.raises(ValueError, match="At least 3 usable"):
        pipeline._resume_after_control_points(_resume_args(control_points), outdir, tmp_path / "reports", RunSummary(outdir, {}), None, {}, 10, 1, 0, outdir / "classified.ply")
    assert commands == []


def test_automatic_mode_is_accepted_by_pipeline_argparse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "images"; images.mkdir(); (images / "image.jpg").write_bytes(b"image")
    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run", _fake_run(commands))
    monkeypatch.setattr(pipeline, "find_colmap_exe", lambda _: "colmap")
    monkeypatch.setattr(sys, "argv", ["pipeline", "--capture-type", "camera", "--images", str(images), "--outdir", str(tmp_path / "products"), "--control-points-mode", "automatic", "--geometry-refinement", "0", "--skip-export"])
    pipeline.main()
    assert any(command[1] == "mapper" for command in commands)


def test_settings_offer_automatic_control_point_mode() -> None:
    assert settings_page.CONTROL_POINTS_MODE_OPTIONS["automatic"] == "Automatic (detect targets)"


def test_automatic_control_points_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    products = tmp_path / "products"; scene = make_scene(products / "sparse")
    control_points, images, model_txt = tmp_path / "control_points", tmp_path / "images", products / "sparse_txt"
    control_points.mkdir(); images.mkdir(); model_txt.mkdir(); _write_points(control_points, scene.references)
    gps_ref = products / "gps_ref.txt"; gps_ref.write_text("synthetic\n", encoding="utf-8")
    origin = np.array([6_371_000.0, 0.0, 0.0])
    monkeypatch.setattr(pipeline, "fit_enu_to_ecef", lambda *_args: {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": origin.tolist()})
    monkeypatch.setattr(pipeline, "coordinates_to_ecef", lambda values, _crs: np.asarray(values, dtype=float) + origin)
    monkeypatch.setattr(pipeline, "enu_rotation_at_ecef", lambda _ecef: np.eye(3))
    monkeypatch.setattr(pipeline, "automatic_detection", lambda *_args, **_kwargs: (clone_detections(scene.detections), {"found": 40, "rejected": 0}))
    result = pipeline._automatic_control_points(scene.model_dir, model_txt, images, control_points, "LOCAL", gps_ref, SETTINGS, tmp_path / "reports" / "control_points_report.json", rtk=True)
    report = json.loads((tmp_path / "reports" / "control_points_report.json").read_text())
    georeference = json.loads((products / "georeference.json").read_text())
    assert result is not None and result.is_dir()
    assert (control_points / "detections.json").is_file()
    assert report["gate"]["passed"] and max(point["error_3d_m"] for point in report["points"]) < 0.05
    assert georeference["source_frame"] == "gcp_local_enu"


def test_continue_after_control_points_uses_detections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    products = tmp_path / "products"
    sparse_pre_gcp = products / "sparse" / "pre_gcp"
    sparse_pre_gcp.mkdir(parents=True)
    sparse_pre_gcp_txt = products / "sparse" / "pre_gcp_txt"
    sparse_pre_gcp_txt.mkdir(parents=True)
    (products / "image_list.txt").write_text("image.jpg\n", encoding="utf-8")
    control_points = tmp_path / "control_points"
    control_points.mkdir(parents=True)
    save_detections(control_points / "detections.json", {})
    images = tmp_path / "images"
    images.mkdir()

    recorded_automatic_models: list[Path] = []
    gcp_adjusted = products / "sparse" / "gcp_adjusted"

    def fake_automatic_control_points(model_path, *_args, **_kwargs):
        recorded_automatic_models.append(Path(model_path))
        return gcp_adjusted

    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    def forbidden_bundle_adjustment(*_args, **_kwargs):
        raise AssertionError("run_gcp_bundle_adjustment was unexpectedly called")

    monkeypatch.setattr(pipeline, "_automatic_control_points", fake_automatic_control_points)
    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(pipeline, "_prepare_detections_for_resume", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "_clear_stereo_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "_classify_and_finish", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "run_gcp_bundle_adjustment", forbidden_bundle_adjustment)

    args = argparse.Namespace(
        images=str(images),
        colmap_bin="colmap",
        control_points_dir=str(control_points),
        control_points_crs="LOCAL",
        rtk=False,
    )
    summary = RunSummary(products, {})
    pipeline._resume_after_control_points(
        args, products, tmp_path / "reports", summary, None, {}, 10, 1, 0, products / "classified.ply",
    )

    assert recorded_automatic_models == [sparse_pre_gcp]
    assert len(commands) == 4
    first_cmd = commands[0]
    assert first_cmd[1] == "image_undistorter"
    input_path_idx = first_cmd.index("--input_path")
    assert Path(first_cmd[input_path_idx + 1]) == gcp_adjusted
    actions = [cmd[1] for cmd in commands]
    assert actions == ["image_undistorter", "patch_match_stereo", "stereo_fusion", "model_converter"]


def test_resume_automatic_uses_pristine_distorted_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_continue_after_control_points_uses_detections(tmp_path, monkeypatch)


def test_image_path_map_resolves_subfolders(tmp_path: Path) -> None:
    images = tmp_path / "images"
    flight_a = images / "flight_a"
    flight_b = images / "flight_b"
    flight_a.mkdir(parents=True)
    flight_b.mkdir(parents=True)
    file_ax = flight_a / "x.jpg"
    file_bx = flight_b / "x.jpg"
    file_by = flight_b / "y.jpg"
    file_ax.write_bytes(b"ax")
    file_bx.write_bytes(b"bx")
    file_by.write_bytes(b"by")

    mapping = pipeline._image_path_map(images)
    assert mapping["flight_a/x.jpg"] == file_ax
    assert mapping["flight_b/x.jpg"] == file_bx
    assert mapping["flight_b/y.jpg"] == file_by
    assert mapping["y.jpg"] == file_by
    assert "x.jpg" not in mapping


def test_clear_stereo_cache_recreates_image_subfolders(tmp_path: Path) -> None:
    dense = tmp_path / "dense"
    flight_a = dense / "images" / "flight_a"
    flight_b = dense / "images" / "flight_b"
    flight_a.mkdir(parents=True)
    flight_b.mkdir(parents=True)
    (flight_a / "x.jpg").write_bytes(b"")
    (flight_b / "y.jpg").write_bytes(b"")

    old_bin = dense / "stereo" / "depth_maps" / "old.bin"
    old_bin.parent.mkdir(parents=True)
    old_bin.write_bytes(b"old")

    pipeline._clear_stereo_cache(dense)

    assert not old_bin.exists()
    assert (dense / "stereo" / "depth_maps" / "flight_a").is_dir()
    assert (dense / "stereo" / "depth_maps" / "flight_b").is_dir()
    assert (dense / "stereo" / "normal_maps" / "flight_a").is_dir()
    assert (dense / "stereo" / "normal_maps" / "flight_b").is_dir()
