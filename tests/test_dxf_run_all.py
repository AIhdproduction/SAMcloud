"""Regression tests for the one-click DXF workflow jobs."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from samcloud.dxf import jobs
from samcloud.dxf.cancel import CancelToken, Cancelled


LOGGER = logging.getLogger("samcloud.tests.dxf_run_all")


def _state() -> SimpleNamespace:
    return SimpleNamespace(mode_override=None, segment_source="combined", levels=[])


def _workdir(name: str) -> Path:
    root = Path(os.environ["SAMCLOUD_TEST_WORKDIR"])
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_outdoor(monkeypatch: pytest.MonkeyPatch, status: dict[str, str] | None = None) -> None:
    monkeypatch.setattr(jobs, "load_state", lambda _project: _state())
    monkeypatch.setattr(jobs, "read_las_info", lambda *_args, **_kwargs: SimpleNamespace(mode="outdoor"))
    monkeypatch.setattr(jobs, "preview_status", lambda *_args: "ok")
    monkeypatch.setattr(jobs, "workflow_step_status", lambda *_args: status or {"ortho": "missing", "segment": "missing", "clean_plan": "missing"})


def test_run_all_outdoor_orders_steps_and_skips_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = _workdir("outdoor")
    _configure_outdoor(monkeypatch)
    calls: list[tuple[str, dict[str, object]]] = []

    def record(name: str):
        def run(*_args: object, **kwargs: object) -> None:
            calls.append((name, kwargs))
        return run

    monkeypatch.setattr(jobs, "run_ortho", record("ortho"))
    monkeypatch.setattr(jobs, "run_segment", record("segment"))
    monkeypatch.setattr(jobs, "run_clean_plan", record("clean_plan"))
    monkeypatch.setattr(jobs, "run_export", record("export"))
    jobs.run_all(tmp_path, tmp_path / "cloud.las", None, lambda *_args: None, CancelToken(), LOGGER, mode="outdoor")
    assert [name for name, _kwargs in calls] == ["ortho", "segment", "clean_plan", "export"]
    assert calls[-1][1]["variant"] == "ortho"

    calls.clear()
    export_dir = tmp_path / "products" / "dxf" / "export"
    clean = tmp_path / "products" / "dxf" / "site" / "ortho" / "result.json"
    clean.parent.mkdir(parents=True, exist_ok=True)
    clean.write_text("{}", encoding="utf-8")
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / f"{tmp_path.name}_site_clean.dxf").write_text("dxf", encoding="utf-8")
    _configure_outdoor(monkeypatch, {"ortho": "ok", "segment": "ok", "clean_plan": "ok"})
    jobs.run_all(tmp_path, tmp_path / "cloud.las", None, lambda *_args: None, CancelToken(), LOGGER, mode="outdoor")
    assert calls == []
    jobs.run_all(tmp_path, tmp_path / "cloud.las", None, lambda *_args: None, CancelToken(), LOGGER, mode="outdoor", force=True)
    assert [name for name, _kwargs in calls] == ["ortho", "segment", "clean_plan", "export"]


def test_run_all_indoor_requires_level(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = _workdir("indoor_requires_level")
    monkeypatch.setattr(jobs, "load_state", lambda _project: _state())
    monkeypatch.setattr(jobs, "read_las_info", lambda *_args, **_kwargs: SimpleNamespace(mode="indoor"))
    with pytest.raises(ValueError, match="requires an explicit level ID"):
        jobs.run_all(tmp_path, tmp_path / "cloud.las", None, lambda *_args: None, CancelToken(), LOGGER, mode="indoor")


def test_run_all_progress_is_monotonic_and_prefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = _workdir("progress")
    _configure_outdoor(monkeypatch)

    def report_step(*args: object, **_kwargs: object) -> None:
        progress = args[3]
        progress(0.25, "working")
        progress(0.75, "finishing")

    monkeypatch.setattr(jobs, "run_ortho", report_step)
    monkeypatch.setattr(jobs, "run_segment", report_step)
    monkeypatch.setattr(jobs, "run_clean_plan", report_step)
    monkeypatch.setattr(jobs, "run_export", report_step)
    updates: list[tuple[float, str]] = []
    jobs.run_all(tmp_path, tmp_path / "cloud.las", None, lambda fraction, stage: updates.append((fraction, stage)), CancelToken(), LOGGER, mode="outdoor")
    assert [fraction for fraction, _stage in updates] == sorted(fraction for fraction, _stage in updates)
    assert all(stage.startswith("Step ") for _fraction, stage in updates)


def test_run_all_cancel_in_second_step_stops_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = _workdir("cancel")
    _configure_outdoor(monkeypatch)
    calls: list[str] = []
    cancel = CancelToken()
    monkeypatch.setattr(jobs, "run_ortho", lambda *_args, **_kwargs: calls.append("ortho"))

    def stop(*_args: object, **_kwargs: object) -> None:
        calls.append("segment")
        cancel.cancel()

    monkeypatch.setattr(jobs, "run_segment", stop)
    monkeypatch.setattr(jobs, "run_clean_plan", lambda *_args, **_kwargs: calls.append("clean_plan"))
    monkeypatch.setattr(jobs, "run_export", lambda *_args, **_kwargs: calls.append("export"))
    with pytest.raises(Cancelled):
        jobs.run_all(tmp_path, tmp_path / "cloud.las", None, lambda *_args: None, cancel, LOGGER, mode="outdoor")
    assert calls == ["ortho", "segment"]


def test_cancelled_run_all_writes_cancelled_status(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("cancelled_status")
    statuses: list[str] = []
    monkeypatch.setattr(jobs, "write_job_status", lambda _path, _action, state, *_args, **_kwargs: statuses.append(state))
    monkeypatch.setattr(jobs, "run_all", lambda *_args, **_kwargs: (_ for _ in ()).throw(Cancelled("cancelled in step 2")))
    monkeypatch.setattr(sys, "argv", [
        "jobs", "run_all", "--project", str(project), "--las", str(project / "cloud.las"), "--mode", "outdoor",
    ])
    with pytest.raises(SystemExit, match="2"):
        jobs.main()
    assert statuses[-1] == "cancelled"


def test_prepare_levels_chains_preview_alignment_and_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = _workdir("prepare_levels")
    calls: list[str] = []
    monkeypatch.setattr(jobs, "load_state", lambda _project: _state())
    monkeypatch.setattr(jobs, "read_las_info", lambda *_args, **_kwargs: SimpleNamespace(mode="indoor"))
    monkeypatch.setattr(jobs, "preview_status", lambda *_args: "missing")
    for name in ("run_preview", "run_align", "run_levels"):
        monkeypatch.setattr(jobs, name, lambda *_args, _name=name, **_kwargs: calls.append(_name))
    jobs.run_prepare_levels(tmp_path, tmp_path / "cloud.las", lambda *_args: None, CancelToken(), LOGGER)
    assert calls == ["run_preview", "run_align", "run_levels"]
