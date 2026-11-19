"""IFC export job wiring: freshness, run_all integration, cancel/failure handling, CLI and controller."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from samcloud.app.dxf_controller import DxfPageController
from samcloud.bim import jobs as bim_jobs
from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene
from samcloud.dxf import jobs
from samcloud.dxf.cancel import CancelToken, Cancelled

LOGGER = logging.getLogger("samcloud.tests.bim_jobs")
LEVEL = "level_1"


def _workdir(name: str) -> Path:
    path = Path(os.environ["SAMCLOUD_TEST_WORKDIR"]) / "bim_jobs" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _plan_model(project: Path, content: str = '{"schema_version": 1}') -> Path:
    path = bim_jobs.plan_model_file(project, LEVEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _install_fake_export(monkeypatch: pytest.MonkeyPatch, calls: list[str] | None = None, after: Any = None) -> None:
    """Replace export_level by a stand-in that writes a real IFC (the viewer export stays real)."""

    def export_level(project_root: Path, level_key: str, *, progress=None, cancel=None) -> Any:
        if calls is not None:
            calls.append("export_level")
        directory = Path(project_root) / "products" / "bim" / level_key
        directory.mkdir(parents=True, exist_ok=True)
        ifc = directory / f"{Path(project_root).name}_{level_key}.ifc"
        write_ifc(build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE)), ifc, run_id="job-test")
        if progress:
            progress(0.5, "writing")
            progress(1.0, "done")
        if cancel:
            cancel()
        if after:
            after()
        return SimpleNamespace(ifc_path=ifc, qa_json_path=directory / "qa.json", qa_html_path=directory / "qa.html",
                               counts={"IfcWall": 4}, ids_failures=0, express_findings=0, warnings=[])

    monkeypatch.setattr(bim_jobs, "_export_level", export_level)


# ----------------------------------------------------------------------- freshness


def test_status_lifecycle_missing_fresh_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("status")
    plan = _plan_model(project)
    assert bim_jobs.ifc_export_status(project, LEVEL) == "missing"
    _install_fake_export(monkeypatch)
    updates: list[tuple[float, str]] = []
    manifest = bim_jobs.run_ifc_export(project, LEVEL, lambda f, s: updates.append((f, s)))
    directory = project / "products" / "bim" / LEVEL
    assert (directory / "model.glb").is_file() and (directory / "props.json").is_file()
    assert manifest["plan_model_sha256"] == bim_jobs.plan_model_sha256(project, LEVEL)
    assert json.loads((directory / "export.json").read_text(encoding="utf-8")) == manifest
    assert manifest["viewer_counts"]["IfcWall"] == 4 and manifest["ids_failures"] == 0
    fractions = [fraction for fraction, _ in updates]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert bim_jobs.ifc_export_status(project, LEVEL) == "fresh"

    plan.write_text('{"schema_version": 1, "changed": true}', encoding="utf-8")
    assert bim_jobs.ifc_export_status(project, LEVEL) == "stale"
    plan.write_text('{"schema_version": 1}', encoding="utf-8")
    assert bim_jobs.ifc_export_status(project, LEVEL) == "fresh"
    (directory / "model.glb").unlink()
    assert bim_jobs.ifc_export_status(project, LEVEL) == "missing"


def test_missing_plan_model_fails_before_export(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("no_plan")
    calls: list[str] = []
    _install_fake_export(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="plan_model.json not found"):
        bim_jobs.run_ifc_export(project, LEVEL)
    assert calls == []


def test_cancel_between_stages_writes_no_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("cancel_between")
    _plan_model(project)
    state = {"cancelled": False}
    _install_fake_export(monkeypatch, after=lambda: state.update(cancelled=True))
    with pytest.raises(Cancelled):
        bim_jobs.run_ifc_export(project, LEVEL, cancel=lambda: state["cancelled"])
    directory = project / "products" / "bim" / LEVEL
    assert not (directory / "export.json").exists() and not (directory / "model.glb").exists()
    assert bim_jobs.ifc_export_status(project, LEVEL) == "missing"


def test_cancel_inside_export_level_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("cancel_inside")
    _plan_model(project)
    token = {"cancelled": False}

    def export_level(_root: Path, _key: str, *, progress=None, cancel=None) -> Any:
        token["cancelled"] = True
        cancel()
        raise AssertionError("cancel() must raise")

    monkeypatch.setattr(bim_jobs, "_export_level", export_level)
    with pytest.raises(Cancelled):
        bim_jobs.run_ifc_export(project, LEVEL, cancel=lambda: token["cancelled"])


# ----------------------------------------------------------------------- run_all


def _indoor_state() -> SimpleNamespace:
    return SimpleNamespace(
        mode_override=None, segment_source="combined",
        levels=[{"id": LEVEL, "name": "1", "floor_z": 0.0, "ceiling_z": 2.5, "checked": True}],
    )


def _configure(monkeypatch: pytest.MonkeyPatch, mode: str, status: str = "missing") -> list[str]:
    calls: list[str] = []
    state = _indoor_state() if mode == "indoor" else SimpleNamespace(mode_override=None, segment_source="combined", levels=[])
    monkeypatch.setattr(jobs, "load_state", lambda _project: state)
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode=mode))
    monkeypatch.setattr(jobs, "preview_status", lambda *_a: "ok")
    monkeypatch.setattr(jobs, "workflow_step_status", lambda *_a: {"ortho": status, "segment": status, "clean_plan": status})
    for name, label in (("run_ortho", "ortho"), ("run_segment", "segment"), ("run_clean_plan", "clean_plan"), ("run_refine_openings", "refine"), ("run_export", "export"), ("run_export_ifc", "export_ifc")):
        monkeypatch.setattr(jobs, name, lambda *_a, _label=label, **_k: calls.append(_label))
    monkeypatch.setattr(jobs, "refine_openings_status", lambda *_a: "fresh" if status == "ok" else "missing")
    return calls


def test_run_all_indoor_ends_with_export_ifc(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("run_all_indoor")
    calls = _configure(monkeypatch, "indoor")
    stages: list[str] = []
    jobs.run_all(project, project / "c.las", LEVEL, lambda _f, stage: stages.append(stage), CancelToken(), LOGGER, mode="indoor")
    assert calls == ["ortho", "segment", "clean_plan", "refine", "export", "export_ifc"]
    assert any(stage.startswith("Step 6/6: Export IFC") for stage in stages)


def test_run_all_outdoor_skips_ifc(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    project = _workdir("run_all_outdoor")
    calls = _configure(monkeypatch, "outdoor")
    outdoor_logger = logging.getLogger("samcloud.tests.bim_jobs.outdoor")
    with caplog.at_level(logging.INFO, logger=outdoor_logger.name):
        jobs.run_all(project, project / "c.las", None, lambda *_a: None, CancelToken(), outdoor_logger, mode="outdoor")
    assert calls == ["ortho", "segment", "clean_plan", "export"]
    assert "IFC export: indoor only for now" in caplog.text


def test_run_all_skips_fresh_ifc_but_force_reruns(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("run_all_fresh")
    calls = _configure(monkeypatch, "indoor", status="ok")
    # make the DXF step fresh so only the IFC step decides
    clean = project / "products" / "dxf" / LEVEL / "ortho" / "result.json"
    clean.parent.mkdir(parents=True)
    clean.write_text("{}", encoding="utf-8")
    export = project / "products" / "dxf" / "export" / f"{jobs.sanitize_dxf_name(project.name)}_1_clean.dxf"
    export.parent.mkdir(parents=True)
    export.write_text("dxf", encoding="utf-8")
    plan = _plan_model(project)
    directory = project / "products" / "bim" / LEVEL
    directory.mkdir(parents=True)
    for name in ("m.ifc", "model.glb", "props.json"):
        (directory / name).write_bytes(b"x")
    manifest = {"plan_model_sha256": bim_jobs.plan_model_sha256(project, LEVEL), "ifc": "m.ifc", "glb": "model.glb", "props": "props.json"}
    (directory / "export.json").write_text(json.dumps(manifest), encoding="utf-8")

    jobs.run_all(project, project / "c.las", LEVEL, lambda *_a: None, CancelToken(), LOGGER, mode="indoor")
    assert calls == []
    plan.write_text('{"schema_version": 1, "x": 1}', encoding="utf-8")
    jobs.run_all(project, project / "c.las", LEVEL, lambda *_a: None, CancelToken(), LOGGER, mode="indoor")
    assert calls == ["export_ifc"]
    calls.clear()
    plan.write_text('{"schema_version": 1}', encoding="utf-8")
    jobs.run_all(project, project / "c.las", LEVEL, lambda *_a: None, CancelToken(), LOGGER, mode="indoor", force=True)
    assert calls == ["ortho", "segment", "clean_plan", "refine", "export", "export_ifc"]


def test_run_all_cancel_before_ifc_step(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("run_all_cancel")
    calls = _configure(monkeypatch, "indoor")
    token = CancelToken()
    monkeypatch.setattr(jobs, "run_export", lambda *_a, **_k: (calls.append("export"), token.cancel()))
    with pytest.raises(Cancelled):
        jobs.run_all(project, project / "c.las", LEVEL, lambda *_a: None, token, LOGGER, mode="indoor")
    assert "export_ifc" not in calls


# ----------------------------------------------------------------------- job process / CLI


def _run_main(monkeypatch: pytest.MonkeyPatch, project: Path, action: str = "export_ifc") -> None:
    monkeypatch.setattr(jobs, "load_state", lambda _project: _indoor_state())
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="indoor"))
    monkeypatch.setattr(sys, "argv", ["jobs", action, "--project", str(project), "--las", str(project / "c.las"), "--level", LEVEL, "--mode", "indoor"])
    jobs.main()


def test_export_ifc_job_success_writes_done_status(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("job_ok")
    _plan_model(project)
    _install_fake_export(monkeypatch)
    _run_main(monkeypatch, project)
    status = json.loads((project / "products" / "dxf" / "job_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "done" and status["action"] == "export_ifc"
    assert bim_jobs.ifc_export_status(project, LEVEL) == "fresh"


def test_export_failure_marks_job_failed_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("job_fail")
    _plan_model(project)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("IDS validation exploded")

    monkeypatch.setattr(bim_jobs, "_export_level", boom)
    with pytest.raises(SystemExit) as raised:
        _run_main(monkeypatch, project)
    assert raised.value.code == 1
    status = json.loads((project / "products" / "dxf" / "job_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed" and "IDS validation exploded" in status["message"]
    assert bim_jobs.ifc_export_status(project, LEVEL) == "missing"


def test_export_ifc_job_rejects_outdoor(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("job_outdoor")
    monkeypatch.setattr(jobs, "load_state", lambda _project: SimpleNamespace(mode_override=None, segment_source="combined", levels=[]))
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="outdoor"))
    monkeypatch.setattr(sys, "argv", ["jobs", "export_ifc", "--project", str(project), "--las", str(project / "c.las"), "--mode", "outdoor"])
    with pytest.raises(SystemExit):
        jobs.main()
    status = json.loads((project / "products" / "dxf" / "job_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed" and "indoor only" in status["message"]


def test_cli_exit_codes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    project = _workdir("cli")
    assert bim_jobs.main(["export", "--project", str(project), "--level", LEVEL]) == 1
    assert "plan_model.json not found" in capsys.readouterr().err
    _plan_model(project)
    _install_fake_export(monkeypatch)
    assert bim_jobs.main(["export", "--project", str(project), "--level", LEVEL]) == 0
    out = capsys.readouterr().out
    assert "[100%]" in out and '"plan_model_sha256"' in out
    assert not (project / "products" / "dxf" / "job_status.json").exists()

    def cancelled(*_a: Any, **_k: Any) -> Any:
        raise Cancelled("stop")

    monkeypatch.setattr(bim_jobs, "_export_level", cancelled)
    assert bim_jobs.main(["export", "--project", str(project), "--level", LEVEL]) == 2


# ----------------------------------------------------------------------- controller


class _FakeRunner:
    def __init__(self) -> None:
        self.started: list[tuple[str, dict[str, Any]]] = []

    def start(self, action: str, **kwargs: Any) -> str:
        self.started.append((action, kwargs))
        return "started"


def test_controller_accepts_export_ifc_and_reports_status(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("controller")
    las = project / "c.las"
    las.write_bytes(b"x")
    monkeypatch.setattr(DxfPageController, "blocked_message", property(lambda _self: None))
    monkeypatch.setattr(DxfPageController, "profile_required", property(lambda _self: False))
    runner = _FakeRunner()
    controller = DxfPageController(project, runner=runner, las_path=las)  # type: ignore[arg-type]
    assert controller.start_job("export_ifc", LEVEL, force=True) == "started"
    action, kwargs = runner.started[0]
    assert action == "export_ifc" and kwargs["level"] == LEVEL and kwargs["force"] is True
    with pytest.raises(ValueError):
        controller.start_job("export_nonsense", LEVEL)

    assert controller.ifc_export_status(LEVEL) == "missing"
    assert controller.ifc_export_status(None) == "missing"
    _plan_model(project)
    _install_fake_export(monkeypatch)
    bim_jobs.run_ifc_export(project, LEVEL)
    assert controller.ifc_export_status(LEVEL) == "fresh"
    bim_jobs.plan_model_file(project, LEVEL).write_text("{}", encoding="utf-8")
    assert controller.ifc_export_status(LEVEL) == "stale"
