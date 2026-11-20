"""Outdoor environment model integration in the automatic final export stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from samcloud.engine import pipeline


def _write_completed_run(project: Path) -> Path:
    reports = project / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metrics = {
        "status": "Completed", "run_id": "20990101_000000Z", "started_at": "2999-01-01T00:00:00+00:00",
        "total_seconds": 100.0, "stage_seconds": {}, "settings": {},
    }
    for name in ("run_metrics.json", "run_metrics_20990101_000000Z.json"):
        (reports / name).write_text(json.dumps(metrics), encoding="utf-8")
    for name in ("run_summary.txt", "run_summary_20990101_000000Z.txt"):
        (reports / name).write_text("Run summary\n", encoding="utf-8")
    return reports / "progress.json"


def _request(project: Path, **changes: Any) -> dict[str, Any]:
    return {
        "enabled": True, "project_dir": project, "progress_file": None, "started": 0.0,
        "capture_type": "drone_gps", "class_set": "outdoor", "rerun_environment": False,
        **changes,
    }


def test_outdoor_stage_runs_environment_and_records_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    progress = _write_completed_run(project)
    progress.write_text(json.dumps({"percent": 100, "stage": "Complete"}), encoding="utf-8")
    calls: list[tuple[Path, dict[str, Any]]] = []

    def fake_run(project_dir: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append((project_dir, kwargs))
        return {"out_dir": str(project / "products" / "environment"), "steps": {
            "terrain": {"status": "done"}, "export": {"status": "done"},
        }}

    from samcloud.drone import run_environment as environment_module

    monkeypatch.setattr(environment_module, "run_environment", fake_run)
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", _request(project, progress_file=progress))

    pipeline.run_auto_dxf_stage()

    assert calls == [(project, {"resume": True})]
    metrics = json.loads((project / "reports" / "run_metrics.json").read_text(encoding="utf-8"))
    assert metrics["stage_seconds"]["Umgebungsmodell"] >= 0
    assert metrics["settings"]["environment_model"]["steps"] == {"terrain": "done", "export": "done"}
    summary = (project / "reports" / "run_summary.txt").read_text(encoding="utf-8")
    assert "Umgebungsmodell" in summary and "terrain: done" in summary and "export: done" in summary
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert payload["stage"] == "Umgebungsmodell" and payload["environment_model"]["status"] == "completed"


def test_environment_failure_is_warned_and_does_not_propagate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    _write_completed_run(project)

    def failing_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("terrain unavailable")

    from samcloud.drone import run_environment as environment_module

    monkeypatch.setattr(environment_module, "run_environment", failing_run)
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", _request(project))

    pipeline.run_auto_dxf_stage()

    metrics = json.loads((project / "reports" / "run_metrics.json").read_text(encoding="utf-8"))
    assert metrics["settings"]["environment_model"]["status"] == "failed"
    summary = (project / "reports" / "run_summary.txt").read_text(encoding="utf-8")
    assert "WARNING [Umgebungsmodell] terrain unavailable" in summary


def test_reclassification_forces_environment_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    _write_completed_run(project)
    calls: list[dict[str, Any]] = []

    def fake_run(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"steps": {"semantics": {"status": "done"}}}

    from samcloud.drone import run_environment as environment_module

    monkeypatch.setattr(environment_module, "run_environment", fake_run)
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", _request(project, rerun_environment=True))

    pipeline.run_auto_dxf_stage()

    assert calls == [{"steps": None, "resume": False}]


def test_indoor_stage_keeps_existing_dxf_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    _write_completed_run(project)
    dxf_calls: list[Path] = []

    from samcloud.engine import auto_dxf
    from samcloud.drone import run_environment as environment_module

    monkeypatch.setattr(auto_dxf, "run_after_pipeline", lambda project_dir, **_kwargs: dxf_calls.append(project_dir))
    monkeypatch.setattr(environment_module, "run_environment", lambda *_args, **_kwargs: pytest.fail("environment must not run"))
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", _request(project, capture_type="camera", class_set="indoor"))

    pipeline.run_auto_dxf_stage()

    assert dxf_calls == [project]
