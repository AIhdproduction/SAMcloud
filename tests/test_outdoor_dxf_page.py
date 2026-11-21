"""Outdoor DXF/IFC page: environment job files, command line and compute-card status lines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from samcloud.app.dxf_controller import DxfPageController
from samcloud.drone import env_job


def _project(root: Path, name: str, *, drone: bool, class_set: str | None = None) -> Path:
    project = (root / name).resolve()
    project.mkdir(parents=True)
    class_set = class_set or ("outdoor" if drone else "indoor")
    document: dict = {"project": {"name": name}, "settings": {"classification": {"class_set": class_set}}, "workflow": {}}
    if drone:
        document["acquisition"] = {"type": "drone_gps"}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return project


def _write_state(project: Path, steps: dict) -> None:
    out = project / "products" / "environment"
    out.mkdir(parents=True, exist_ok=True)
    (out / "state.json").write_text(json.dumps({"steps": steps}), encoding="utf-8")


_MIXED = {
    "load": {"status": "done", "seconds": 12.34},
    "semantics": {"status": "skipped", "reason": "resume: completed checkpoint", "seconds": 3.0},
    "terrain": {"status": "done", "seconds": 45.0},
    "tin": {"status": "failed", "seconds": 7.5, "error": "boom " + "x" * 400 + "\nsecond line", "traceback": "..."},
    "buildings": {"status": "skipped", "reason": "dependency unavailable: tin", "seconds": 0},
}


def _write_qa(project: Path, failed: tuple[str, ...] = ("2_area", "9_names")) -> None:
    rules = [{"rule": f"{i}_r", "ok": True} for i in range(3)] + [{"rule": name, "ok": False} for name in failed]
    (project / "products" / "environment" / "qa_cad_report.json").write_text(json.dumps(rules), encoding="utf-8")


def test_environment_job_status_parses_state_json(tmp_path: Path) -> None:
    project = _project(tmp_path, "Mixed", drone=True)
    _write_state(project, _MIXED)
    _write_qa(project)
    status = env_job.environment_job_status(project)
    by_name = {item["name"]: item for item in status["steps"]}
    assert [item["name"] for item in status["steps"]] == list(env_job.ENVIRONMENT_STEPS)
    assert by_name["load"]["status"] == "done" and by_name["load"]["seconds"] == pytest.approx(12.34)
    assert by_name["tin"]["status"] == "failed"
    assert by_name["tin"]["error"].startswith("boom xxx") and "\n" not in by_name["tin"]["error"] and len(by_name["tin"]["error"]) <= 140
    assert by_name["buildings"]["error"] == "dependency unavailable: tin"
    assert by_name["vegetation"]["status"] == "pending" and by_name["report"]["status"] == "pending"
    assert status["state"] == "idle" and status["running"] is False
    assert status["qa"] == {"passed": 3, "total": 5, "failed": ["2_area", "9_names"]}
    assert 0 < status["progress"] < 1


def test_environment_job_status_marks_active_step_while_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, "Running", drone=True)
    _write_state(project, {"load": {"status": "done", "seconds": 1.0}})
    env_job.job_status_path(project).write_text(json.dumps({"state": "running", "pid": 4242, "started_at": "2000-01-01T00:00:00+00:00", "rerun_all": False}), encoding="utf-8")
    monkeypatch.setattr(env_job, "_pid_alive", lambda pid: pid == 4242)
    status = env_job.environment_job_status(project)
    assert status["running"] is True and status["state"] == "running"
    assert [item["status"] for item in status["steps"]][:3] == ["done", "running", "pending"]
    # worker vanished without the watcher noting it: reported as failed (interrupted), never as running
    monkeypatch.setattr(env_job, "_pid_alive", lambda pid: False)
    status = env_job.environment_job_status(project)
    assert status["running"] is False and status["state"] == "failed" and status["error"] == "interrupted"


class _FakeProcess:
    pid = 4711

    def wait(self) -> int:
        return 0


def test_start_environment_job_command_and_single_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, "Start", drone=True)
    commands: list[list[str]] = []

    def fake_popen(command, **_kwargs):
        commands.append(list(command))
        return _FakeProcess()

    class _NoThread:
        def __init__(self, *_a, **_k) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(env_job.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(env_job.threading, "Thread", _NoThread)
    monkeypatch.setattr(env_job, "_pid_alive", lambda pid: pid == 4711)

    record = env_job.start_environment_job(project, rerun_all=False)
    assert commands[0][:4] == [sys.executable, "-m", "samcloud.drone.run_environment", "--project"]
    assert commands[0][4] == str(project) and "--no-resume" not in commands[0]
    assert record["state"] == "running" and json.loads(env_job.job_status_path(project).read_text(encoding="utf-8"))["pid"] == 4711
    assert env_job.is_environment_job_running(project)

    with pytest.raises(RuntimeError):  # second start while the first pid is alive
        env_job.start_environment_job(project, rerun_all=True)
    assert len(commands) == 1

    monkeypatch.setattr(env_job, "_pid_alive", lambda pid: False)  # first worker is gone: a new start is allowed
    env_job.start_environment_job(project, rerun_all=True)
    assert len(commands) == 2 and commands[1][-1] == "--no-resume"


def test_watcher_records_exit(tmp_path: Path) -> None:
    project = _project(tmp_path, "Watch", drone=True)
    env_job.job_status_path(project).parent.mkdir(parents=True, exist_ok=True)
    env_job.job_status_path(project).write_text(json.dumps({"state": "running", "pid": 1, "started_at": "t0"}), encoding="utf-8")
    env_job._watch(project, _FakeProcess(), "t0")  # type: ignore[arg-type]
    record = json.loads(env_job.job_status_path(project).read_text(encoding="utf-8"))
    assert record["state"] == "finished" and record["returncode"] == 0


def test_compute_card_rows_outdoor_environment_and_indoor_classic(tmp_path: Path) -> None:
    outdoor = _project(tmp_path, "Drone", drone=True)
    _write_state(outdoor, _MIXED)
    _write_qa(outdoor, failed=("2_area",))
    controller = DxfPageController(outdoor)
    assert controller.mode == "outdoor" and controller.uses_environment_chain
    card = controller.compute_card_rows(None)
    assert card["kind"] == "environment"
    assert [row[0] for row in card["rows"]] == list(env_job.ENVIRONMENT_STEPS)
    rows = {name: (status, detail) for name, status, detail in card["rows"]}
    assert rows["load"][0] == "done" and "12.3" in rows["load"][1]
    assert rows["tin"][0] == "failed" and "boom" in rows["tin"][1]
    assert card["qa"] == {"passed": 3, "total": 4, "failed": ["2_area"]}

    indoor = _project(tmp_path, "Indoor", drone=False)
    legacy = DxfPageController(indoor)
    assert legacy.mode == "indoor" and not legacy.uses_environment_chain
    classic = legacy.compute_card_rows(None)
    assert classic["kind"] == "classic"
    assert [key for key, _status in classic["rows"]] == ["ortho", "segmentation", "clean_plan", "refine", "export", "ifc"]

    plain = DxfPageController(_project(tmp_path, "PlainOutdoor", drone=False, class_set="outdoor"))
    assert plain.mode == "outdoor" and not plain.uses_environment_chain
    assert [key for key, _status in plain.compute_card_rows(None)["rows"]] == ["ortho", "segmentation", "clean_plan", "export", "ifc"]


def test_controller_start_refuses_running_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, "Busy", drone=True)
    env_job.job_status_path(project).parent.mkdir(parents=True, exist_ok=True)
    env_job.job_status_path(project).write_text(json.dumps({"state": "running", "pid": 99, "started_at": "t"}), encoding="utf-8")
    monkeypatch.setattr(env_job, "_pid_alive", lambda pid: True)
    with pytest.raises(ValueError):
        DxfPageController(project).start_environment_job(False)
