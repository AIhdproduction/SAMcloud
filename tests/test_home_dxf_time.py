from __future__ import annotations

import json
from pathlib import Path

import pytest

from samcloud.app.dxf_time import dxf_ifc_created, dxf_ifc_seconds
from samcloud.app.pages import home


def _state(project: Path) -> None:
    env = project / "products" / "environment"
    env.mkdir(parents=True)
    steps = {"load": {"status": "done", "seconds": 100.5}, "terrain": {"status": "done", "seconds": 50},
             "roofs": {"status": "failed", "seconds": 999}, "x": {"status": "done"}}
    (env / "state.json").write_text(json.dumps({"steps": steps}), encoding="utf-8")


def _dxf(project: Path) -> None:
    (project / "products" / "dxf" / "export").mkdir(parents=True, exist_ok=True)
    (project / "products" / "dxf" / "export" / "p_clean.dxf").write_text("dxf", encoding="utf-8")


def _pipeline(project: Path) -> None:
    (project / "reports").mkdir(parents=True, exist_ok=True)
    (project / "reports" / "run_metrics_1.json").write_text(json.dumps({
        "status": "Completed", "started_at": "2026-01-01", "total_seconds": 1000.0,
        "stage_seconds": {"Sparse reconstruction": 400.0, "Dense stereo": 600.0}}), encoding="utf-8")


def test_outdoor_steps_summed_with_exported_dxf(tmp_path: Path) -> None:
    _state(tmp_path)
    _dxf(tmp_path)
    assert dxf_ifc_seconds(tmp_path) == pytest.approx(150.5)
    assert dxf_ifc_created(tmp_path)


def test_without_dxf_is_none_and_pending(tmp_path: Path) -> None:
    _state(tmp_path)
    assert dxf_ifc_seconds(tmp_path) is None
    assert not dxf_ifc_created(tmp_path)
    assert dxf_ifc_seconds(None) is None
    _pipeline(tmp_path)
    summary = home._processing_time_summary(tmp_path)
    assert summary["dxf_pending"] is True and "DXF/IFC-Export" not in summary["groups"]


def test_no_double_counting_with_pipeline_timing(tmp_path: Path) -> None:
    _pipeline(tmp_path)
    _state(tmp_path)
    _dxf(tmp_path)
    summary = home._processing_time_summary(tmp_path)
    assert summary["groups"]["DXF/IFC-Export"] == pytest.approx(150.5)
    assert "Übrige Schritte" not in summary["groups"]
    assert summary["total_seconds"] == pytest.approx(1150.5)
    assert summary["dxf_pending"] is False


def test_indoor_metrics_take_precedence(tmp_path: Path) -> None:
    _state(tmp_path)
    _dxf(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "dxf_run_metrics.json").write_text(json.dumps({"status": "completed", "total_seconds": 125}), encoding="utf-8")
    assert dxf_ifc_seconds(tmp_path) == 125.0
