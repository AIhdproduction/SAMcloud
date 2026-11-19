"""Home workflow panel: DXF/IFC step status, run-time record and the Total line."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from samcloud.app import i18n
from samcloud.app.pages import home
from samcloud.dxf import jobs
from samcloud.dxf.cancel import CancelToken, Cancelled


LOGGER = logging.getLogger("samcloud.tests.home_dxf_ifc")


def _workdir(name: str) -> Path:
    root = Path(os.environ["SAMCLOUD_TEST_WORKDIR"]) / "home_dxf_ifc" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _configure_outdoor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "load_state", lambda _project: SimpleNamespace(mode_override=None, segment_source="combined", levels=[]))
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="outdoor"))
    monkeypatch.setattr(jobs, "preview_status", lambda *_a: "ok")
    monkeypatch.setattr(jobs, "workflow_step_status", lambda *_a: {"ortho": "missing", "segment": "missing", "clean_plan": "missing"})


def _read_metrics(project: Path) -> dict:
    return json.loads((project / "reports" / "dxf_run_metrics.json").read_text(encoding="utf-8"))


def test_run_all_writes_metrics_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("metrics_ok")
    _configure_outdoor(monkeypatch)
    for name in ("run_ortho", "run_segment", "run_clean_plan", "run_export"):
        monkeypatch.setattr(jobs, name, lambda *_a, **_k: None)
    jobs.run_all(project, project / "cloud.las", None, lambda *_a: None, CancelToken(), LOGGER, mode="outdoor")
    data = _read_metrics(project)
    assert data["status"] == "completed"
    assert data["total_seconds"] >= 0
    assert list(data["step_seconds"]) == ["Build orthophoto", "Segment orthophoto", "Generate clean plan", "Export DXF"]
    assert data["started_at"] <= data["finished_at"]
    assert not list((project / "reports").glob(".dxf_run_metrics*"))


def test_run_all_writes_metrics_on_failure_and_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("metrics_failed")
    _configure_outdoor(monkeypatch)
    monkeypatch.setattr(jobs, "run_ortho", lambda *_a, **_k: None)

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("segment exploded")

    monkeypatch.setattr(jobs, "run_segment", boom)
    with pytest.raises(RuntimeError):
        jobs.run_all(project, project / "cloud.las", None, lambda *_a: None, CancelToken(), LOGGER, mode="outdoor")
    data = _read_metrics(project)
    assert data["status"] == "failed"
    assert set(data["step_seconds"]) == {"Build orthophoto", "Segment orthophoto"}

    cancel = CancelToken()

    def stop(*_a: object, **_k: object) -> None:
        cancel.cancel()

    monkeypatch.setattr(jobs, "run_segment", stop)
    with pytest.raises(Cancelled):
        jobs.run_all(project, project / "cloud.las", None, lambda *_a: None, cancel, LOGGER, mode="outdoor")
    assert _read_metrics(project)["status"] == "cancelled"


def test_run_all_failure_before_steps_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("metrics_early_failure")
    monkeypatch.setattr(jobs, "load_state", lambda _project: (_ for _ in ()).throw(ValueError("no state")))
    with pytest.raises(ValueError):
        jobs.run_all(project, project / "cloud.las", None, lambda *_a: None, CancelToken(), LOGGER, mode="outdoor")
    assert _read_metrics(project)["status"] == "failed"


def _photogrammetry_metrics(project: Path) -> None:
    reports = project / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "run_metrics_20261115_081835Z.json").write_text(json.dumps({
        "status": "Completed",
        "started_at": "2026-11-15T08:18:35Z",
        "total_seconds": 1000,
        "stage_seconds": {"Sparse reconstruction": 400, "Dense stereo": 500},
    }), encoding="utf-8")


def _dxf_metrics(project: Path, status: str = "completed", seconds: float = 125.0) -> None:
    (project / "reports").mkdir(parents=True, exist_ok=True)
    (project / "reports" / "dxf_run_metrics.json").write_text(json.dumps({"status": status, "total_seconds": seconds}), encoding="utf-8")


def _outputs(project: Path) -> None:
    (project / "products" / "dxf" / "export").mkdir(parents=True, exist_ok=True)
    (project / "products" / "dxf" / "export" / "p_level_1_clean.dxf").write_text("dxf", encoding="utf-8")
    (project / "products" / "bim" / "level_1").mkdir(parents=True, exist_ok=True)
    (project / "products" / "bim" / "level_1" / "p_level_1.ifc").write_text("ifc", encoding="utf-8")


def test_summary_includes_dxf_ifc_group_and_total_is_sum(tmp_path: Path) -> None:
    _photogrammetry_metrics(tmp_path)
    _dxf_metrics(tmp_path, seconds=125.0)
    summary = home._processing_time_summary(tmp_path)
    assert summary is not None
    assert summary["groups"]["DXF/IFC-Export"] == 125.0
    assert summary["total_seconds"] == 1125.0
    assert summary["total_seconds"] == pytest.approx(sum(summary["groups"].values()))


def test_summary_without_or_with_failed_dxf_metrics_has_no_dxf_line(tmp_path: Path) -> None:
    _photogrammetry_metrics(tmp_path)
    summary = home._processing_time_summary(tmp_path)
    assert summary is not None and "DXF/IFC-Export" not in summary["groups"]
    assert summary["total_seconds"] == 1000.0
    _dxf_metrics(tmp_path, status="failed")
    summary = home._processing_time_summary(tmp_path)
    assert "DXF/IFC-Export" not in summary["groups"] and summary["total_seconds"] == 1000.0
    (tmp_path / "reports" / "dxf_run_metrics.json").write_text("{broken", encoding="utf-8")
    assert home._processing_time_summary(tmp_path)["total_seconds"] == 1000.0


def test_step_never_run_is_ready(tmp_path: Path) -> None:
    assert home._dxf_ifc_step(tmp_path) == {"status": "ready", "percent": 0.0}
    assert home._dxf_ifc_step(None)["status"] == "ready"


def test_step_completed_when_outputs_and_metrics_exist(tmp_path: Path) -> None:
    _outputs(tmp_path)
    _dxf_metrics(tmp_path)
    assert home._dxf_ifc_step(tmp_path) == {"status": "completed", "percent": 100.0}


def test_step_outputs_without_metrics_is_completed_without_time_line(tmp_path: Path) -> None:
    _photogrammetry_metrics(tmp_path)
    _outputs(tmp_path)
    assert home._dxf_ifc_step(tmp_path)["status"] == "completed"
    summary = home._processing_time_summary(tmp_path)
    assert "DXF/IFC-Export" not in summary["groups"]
    assert summary["total_seconds"] == 1000.0


def test_step_only_dxf_output_is_not_completed(tmp_path: Path) -> None:
    (tmp_path / "products" / "dxf" / "export").mkdir(parents=True)
    (tmp_path / "products" / "dxf" / "export" / "p_1_clean.dxf").write_text("dxf", encoding="utf-8")
    assert home._dxf_ifc_step(tmp_path)["status"] == "ready"


def test_step_failed_run(tmp_path: Path) -> None:
    _outputs(tmp_path)
    _dxf_metrics(tmp_path, status="failed")
    assert home._dxf_ifc_step(tmp_path)["status"] == "failed"


def test_step_running_reports_progress(tmp_path: Path) -> None:
    status = tmp_path / "products" / "dxf" / "job_status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"action": "run_all", "state": "running", "fraction": 0.4}), encoding="utf-8")
    assert home._dxf_ifc_step(tmp_path) == {"status": "running", "percent": pytest.approx(40.0)}
    status.write_text(json.dumps({"action": "export", "state": "running", "fraction": 0.4}), encoding="utf-8")
    assert home._dxf_ifc_step(tmp_path)["status"] == "ready"
    old = status.stat().st_mtime - 7200
    status.write_text(json.dumps({"action": "run_all", "state": "running", "fraction": 0.4}), encoding="utf-8")
    os.utime(status, (old, old))
    assert home._dxf_ifc_step(tmp_path)["status"] == "ready"


@pytest.mark.parametrize("key", [
    "home.workflow.open_dxf", "home.workflow.total", "home.time_group.dxf_ifc", "workflow.step.dxf",
])
def test_locale_keys_present_in_both_languages(key: str) -> None:
    german, english = i18n.catalogue("de"), i18n.catalogue("en")
    assert key in german and key in english
    assert "IFC" in german[key] + english[key] or key == "home.workflow.total"
