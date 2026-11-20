"""Coordinate-aware control-point defaults and safe automatic fallbacks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from samcloud.app import processing
from samcloud.control_points.repository import write_control_points
from samcloud.core.project_manager import ProjectManager
from samcloud.engine import pipeline
from samcloud.engine.run_summary import RunSummary


def _write_points(directory: Path, count: int = 3) -> None:
    write_control_points(
        pd.DataFrame([
            {"name": f"P{index}", "x": float(index), "y": float(index + 1), "z": float(index + 2), "role": "pass"}
            for index in range(count)
        ]),
        directory / "control_points.csv",
    )


def _project() -> dict:
    return {
        "acquisition": {"type": "camera", "sources": {}},
        "settings": {"workflow": {}, "classification": {}, "coordinate_systems": {}, "refinement": {}},
    }


def test_auto_mode_with_three_surveyed_points_resolves_to_automatic(tmp_path: Path) -> None:
    control_points = tmp_path / "control_points"
    _write_points(control_points)

    assert pipeline._resolve_control_points_mode("auto", control_points) == "automatic"
    command = object.__new__(processing.ProcessingManager)._pipeline_command(tmp_path, _project(), "auto")
    assert command[command.index("--control-points-mode") + 1] == "auto"


@pytest.mark.parametrize("count", [0, 2])
def test_auto_mode_without_three_points_matches_none_pipeline_args(tmp_path: Path, count: int) -> None:
    control_points = tmp_path / "control_points"
    control_points.mkdir()
    if count:
        _write_points(control_points, count)
    mode = pipeline._resolve_control_points_mode("auto", control_points)
    manager = object.__new__(processing.ProcessingManager)

    assert mode == "none"
    assert manager._pipeline_command(tmp_path, _project(), mode) == manager._pipeline_command(tmp_path, _project(), "none")


def test_explicit_none_is_never_replaced_by_auto_detection(tmp_path: Path) -> None:
    control_points = tmp_path / "control_points"
    _write_points(control_points)

    assert pipeline._resolve_control_points_mode("none", control_points) == "none"


def test_new_and_legacy_projects_default_to_auto(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    project_directory = manager.create_project("Coordinates", "D:/images")
    project = manager.load(project_directory)

    assert project["workflow"]["control_points_mode"] == "auto"
    project["workflow"].pop("control_points_mode")
    manager.save(project_directory, project)

    assert processing._project_control_points_mode(manager.load(project_directory)) == "auto"


def test_quick_height_correction_preserves_the_estimate_details(tmp_path: Path) -> None:
    checkpoint_report = tmp_path / "checkpoints_pre_gcp.json"
    checkpoint_report.write_text(json.dumps({
        "points": [
            {"name": "P1", "status": "ok", "dH": 0.02},
            {"name": "P2", "status": "ok", "dH": 0.03},
            {"name": "P3", "status": "ok", "dH": 0.04},
            {"name": "P4", "status": "not_found"},
        ],
    }), encoding="utf-8")

    correction = pipeline._quick_height_correction(checkpoint_report)

    assert correction is not None
    assert correction["height_offset_m"] == pytest.approx(-0.03)
    assert [point["name"] for point in correction["points"]] == ["P1", "P2", "P3"]


def test_automatic_path_evaluates_checkpoints_before_bundle_adjustment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    control_points = tmp_path / "control_points"
    images = tmp_path / "images"
    reports = tmp_path / "reports"
    control_points.mkdir()
    images.mkdir()
    _write_points(control_points)
    gps_ref = tmp_path / "gps_ref.txt"
    gps_ref.write_text("reference\n", encoding="utf-8")
    events: list[str] = []

    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace(Reconstruction=lambda _path: object()))
    monkeypatch.setattr(pipeline, "ensure_text_model", lambda *_args: None)
    monkeypatch.setattr(pipeline, "fit_enu_to_ecef", lambda *_args: {
        "scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0],
    })
    monkeypatch.setattr(pipeline, "coordinates_to_ecef", lambda values, _crs: np.asarray(values, dtype=float))

    def evaluate(*_args, **_kwargs):
        events.append("checkpoints")
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "checkpoints_pre_gcp.json").write_text(json.dumps({"totals": {"n_ok": 3}}), encoding="utf-8")
        (reports / "checkpoints_pre_gcp.md").write_text("# pre_gcp\n", encoding="utf-8")
        return {"totals": {"n_ok": 3}}

    def finish(*_args, **kwargs):
        events.append("bundle_adjustment")
        assert kwargs["quick_correction"] == {"height_offset_m": 0.02}
        return tmp_path / "gcp_adjusted"

    monkeypatch.setattr(pipeline, "_evaluate_pre_gcp_checkpoints", evaluate)
    monkeypatch.setattr(pipeline, "_quick_height_correction", lambda _path: {"height_offset_m": 0.02})
    monkeypatch.setattr(pipeline, "automatic_detection", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(pipeline, "_finish_control_point_adjustment", finish)
    summary = RunSummary(tmp_path / "products", {})

    pipeline._automatic_control_points(
        tmp_path / "model", tmp_path / "model_txt", images, control_points, "EPSG:4978", gps_ref,
        {}, reports / "control_points_report.json", rtk=False, summary=summary,
    )

    assert events == ["checkpoints", "bundle_adjustment"]
    assert (reports / "checkpoints_pre_gcp.json").is_file()
    assert (reports / "checkpoints_pre_gcp.md").is_file()
    assert summary.settings["quick_correction"] == {"height_offset_m": 0.02}


@pytest.mark.parametrize("gate_passed, expected_calls", [(False, 1), (True, 0)])
def test_height_shift_is_only_applied_after_a_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_passed: bool, expected_calls: int,
) -> None:
    model = tmp_path / "project" / "products" / "sparse" / "pre_gcp"
    control_points = tmp_path / "control_points"
    control_points.mkdir()
    applied: list[float] = []

    class Reconstruction:
        def __init__(self, _path: Path) -> None:
            pass

        def transform(self, _transform) -> None:
            pass

        def write(self, _path: Path) -> None:
            pass

    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace(Reconstruction=Reconstruction, Sim3d=lambda value: value))
    monkeypatch.setitem(
        sys.modules,
        "samcloud.drone.height_correction",
        SimpleNamespace(apply_height_shift=lambda _project, shift: applied.append(float(shift["height_offset_m"]))),
    )
    monkeypatch.setattr(pipeline, "camera_intrinsics", lambda _reconstruction: {})
    monkeypatch.setattr(pipeline, "run_gcp_bundle_adjustment", lambda *_args, **_kwargs: {"intrinsics": {}})
    monkeypatch.setattr(pipeline, "quality_gate", lambda *_args, **_kwargs: (gate_passed, [], {}))
    monkeypatch.setattr(pipeline, "populate_review_projections", lambda *_args: None)
    monkeypatch.setattr(pipeline, "save_detections", lambda *_args: None)
    monkeypatch.setattr(pipeline, "build_control_point_report", lambda *_args, **_kwargs: {"gate": {"passed": gate_passed}, "points": []})
    monkeypatch.setattr(pipeline, "fit_similarity", lambda *_args: (1.0, np.eye(3), np.zeros(3), {"max": 0.0}))
    monkeypatch.setattr(pipeline, "enu_rotation_at_ecef", lambda _ecef: np.eye(3))
    monkeypatch.setattr(pipeline, "leave_one_out_errors", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(pipeline, "_write_final_model", lambda *_args: None)
    monkeypatch.setattr(pipeline, "save_georeference", lambda *_args: None)
    detections = {
        f"P{index}": {"enabled": True, "triangulated_model_xyz": [float(index), 0.0, 0.0]}
        for index in range(3)
    }
    references = {name: np.asarray(entry["triangulated_model_xyz"]) for name, entry in detections.items()}
    ecef = {name: value.copy() for name, value in references.items()}
    report_path = tmp_path / "reports" / "control_points_report.json"

    result = pipeline._finish_control_point_adjustment(
        model, control_points, "EPSG:4978", {"max_gate_iterations": 1}, report_path,
        detections, {}, references, ecef, list(detections), rtk=False, use_pose_priors=False,
        quick_correction={"height_offset_m": 0.04}, summary=RunSummary(tmp_path / "project" / "products", {}),
    )

    assert (result is None) is (not gate_passed)
    assert applied == [0.04] * expected_calls
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["quick_correction"]["applied"] is (not gate_passed)
