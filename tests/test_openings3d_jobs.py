"""run_all / CLI wiring of the "Refine openings" step: order, freshness, single re-plan, failure and cancel handling."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from samcloud.bim.openings3d.model import OpeningsResult, save_openings_3d
from samcloud.bim.openings3d.pipeline import RefineOutcome
from samcloud.dxf import jobs
from samcloud.dxf.cancel import CancelToken, Cancelled

LOGGER = logging.getLogger("samcloud.tests.openings3d_jobs")
LEVEL = "level_1"


def _workdir(name: str) -> Path:
    path = Path(os.environ["SAMCLOUD_TEST_WORKDIR"]) / f"openings3d_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state() -> SimpleNamespace:
    return SimpleNamespace(mode_override=None, segment_source="combined", levels=[{"id": LEVEL, "name": "1", "floor_z": 0.0, "ceiling_z": 2.5, "checked": True}])


def _configure(monkeypatch: pytest.MonkeyPatch, mode: str = "indoor", status: str = "missing", refine_status: str = "missing") -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(jobs, "load_state", lambda _project: _state())
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode=mode))
    monkeypatch.setattr(jobs, "preview_status", lambda *_a: "ok")
    monkeypatch.setattr(jobs, "workflow_step_status", lambda *_a: {"ortho": status, "segment": status, "clean_plan": status})
    monkeypatch.setattr(jobs, "refine_openings_status", lambda *_a: refine_status)
    for name, label in (("run_ortho", "ortho"), ("run_segment", "segment"), ("run_clean_plan", "clean_plan"), ("run_refine_openings", "refine"), ("run_export", "export"), ("run_export_ifc", "export_ifc")):
        monkeypatch.setattr(jobs, name, lambda *_a, _label=label, **_k: calls.append(_label))
    return calls


def test_run_all_indoor_runs_refine_between_plan_and_dxf_export(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("order")
    calls = _configure(monkeypatch)
    stages: list[str] = []
    jobs.run_all(project, project / "c.las", LEVEL, lambda _f, stage: stages.append(stage), CancelToken(), LOGGER, mode="indoor")
    assert calls == ["ortho", "segment", "clean_plan", "refine", "export", "export_ifc"]
    names = [stage.split(": ", 1)[1].split(" – ")[0] for stage in stages if stage.startswith("Step ")]
    ordered = list(dict.fromkeys(names))
    assert ordered == ["Build orthophoto", "Segment orthophoto", "Generate clean plan", "Refine openings", "Export DXF", "Export IFC"]
    assert any(stage.startswith("Step 4/6: Refine openings") for stage in stages)


def test_run_all_outdoor_has_no_refine_step(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("outdoor")
    calls = _configure(monkeypatch, mode="outdoor")
    jobs.run_all(project, project / "c.las", None, lambda *_a: None, CancelToken(), LOGGER, mode="outdoor")
    assert "refine" not in calls


def test_fresh_refinement_is_skipped_and_force_reruns_it(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("fresh")
    calls = _configure(monkeypatch, status="ok", refine_status="fresh")
    clean = project / "products" / "dxf" / LEVEL / "ortho" / "result.json"
    clean.parent.mkdir(parents=True, exist_ok=True)
    clean.write_text("{}", encoding="utf-8")
    export = project / "products" / "dxf" / "export" / f"{jobs.sanitize_dxf_name(project.name)}_1_clean.dxf"
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_text("dxf", encoding="utf-8")
    monkeypatch.setattr(jobs, "ifc_export_status", lambda *_a: "fresh")
    jobs.run_all(project, project / "c.las", LEVEL, lambda *_a: None, CancelToken(), LOGGER, mode="indoor")
    assert calls == []
    jobs.run_all(project, project / "c.las", LEVEL, lambda *_a: None, CancelToken(), LOGGER, mode="indoor", force=True)
    assert calls == ["ortho", "segment", "clean_plan", "refine", "export", "export_ifc"]


def test_stale_refinement_runs_while_the_rest_is_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("stale")
    calls = _configure(monkeypatch, status="ok", refine_status="stale")
    jobs.run_all(project, project / "c.las", LEVEL, lambda *_a: None, CancelToken(), LOGGER, mode="indoor")
    assert calls[:1] == ["refine"]


def test_cancel_inside_refine_stops_the_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("cancel")
    calls = _configure(monkeypatch)
    token = CancelToken()
    monkeypatch.setattr(jobs, "run_refine_openings", lambda *_a, **_k: (calls.append("refine"), token.cancel()))
    with pytest.raises(Cancelled):
        jobs.run_all(project, project / "c.las", LEVEL, lambda *_a: None, token, LOGGER, mode="indoor")
    assert calls[-1] == "refine" and "export" not in calls and "export_ifc" not in calls


# ---------------------------------------------------------------- run_refine_openings: plan -> refine -> plan once


@pytest.fixture()
def refine_env(monkeypatch: pytest.MonkeyPatch):
    """Real run_refine_openings with a fake refinement and a fake clean plan that records which openings file it used."""
    project = _workdir("refine_env_" + os.urandom(3).hex())
    directory = project / "products" / "dxf" / LEVEL / "ortho"
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "plan_model.json"
    plan_path.write_text(json.dumps({"schema_version": 1, "source": {}}), encoding="utf-8")
    log: dict[str, object] = {"refine": 0, "plan": 0, "outcome": RefineOutcome("refined", "full", 3, True)}
    monkeypatch.setattr(jobs, "load_state", lambda _project: _state())
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="indoor"))

    def refine(root, level_key, **kwargs):
        log["refine"] += 1
        outcome = log["outcome"]
        if outcome.status == "refined":
            save_openings_3d(OpeningsResult(openings=[], inputs={"n": log["refine"]}), directory)
        return outcome

    def plan(project_dir, las, level_id, progress, cancel, logger, **kwargs):
        from samcloud.bim.openings3d.model import openings_3d_sha256

        log["plan"] += 1
        progress(0.5, "planning")
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        sha = openings_3d_sha256(directory)
        data["source"] = {"openings_3d": {"sha256": sha, "applied": True}} if sha else {}
        plan_path.write_text(json.dumps(data), encoding="utf-8")

    import samcloud.bim.openings3d as package

    monkeypatch.setattr(package, "refine_level_openings", refine)
    monkeypatch.setattr(jobs, "run_clean_plan", plan)
    return project, directory, log


def _run(project: Path, **kwargs) -> list[tuple[float, str]]:
    updates: list[tuple[float, str]] = []
    jobs.run_refine_openings(project, project / "c.las", LEVEL, lambda f, s: updates.append((f, s)), CancelToken(), LOGGER, mode="indoor", **kwargs)
    return updates


def test_changed_openings_regenerate_the_plan_exactly_once_and_then_settle(refine_env) -> None:
    project, directory, log = refine_env
    updates = _run(project)
    assert (log["refine"], log["plan"]) == (1, 1)
    assert [f for f, _ in updates] == sorted(f for f, _ in updates) and all(0.0 <= f <= 1.0 for f, _ in updates)
    # a second run (what a repeated run_all would do): the refinement is fresh, the plan already holds it
    log["outcome"] = RefineOutcome("fresh", "full", 3, False)
    _run(project)
    assert (log["refine"], log["plan"]) == (2, 1)  # no further plan pass: no loop


def test_failed_refinement_keeps_the_2d_plan(refine_env, caplog: pytest.LogCaptureFixture) -> None:
    project, _directory, log = refine_env
    log["outcome"] = RefineOutcome("failed", "", 0, False, message="boom")
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        _run(project)
    assert (log["refine"], log["plan"]) == (1, 0)
    assert "keeps the 2D openings" in caplog.text


def test_force_is_passed_to_the_refinement(refine_env, monkeypatch: pytest.MonkeyPatch) -> None:
    project, _directory, log = refine_env
    seen: dict[str, object] = {}
    import samcloud.bim.openings3d as package

    original = package.refine_level_openings

    def spy(root, level_key, **kwargs):
        seen.update(kwargs)
        return original(root, level_key, **kwargs)

    monkeypatch.setattr(package, "refine_level_openings", spy)
    _run(project, force=True)
    assert seen["force"] is True and seen["point_source"] == project / "c.las"


def test_refine_openings_is_indoor_only(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("indoor_only")
    monkeypatch.setattr(jobs, "load_state", lambda _project: _state())
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="outdoor"))
    with pytest.raises(ValueError, match="indoor only"):
        jobs.run_refine_openings(project, project / "c.las", None, lambda *_a: None, CancelToken(), LOGGER, mode="outdoor")


def test_cli_action_refine_openings_writes_done_status(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("cli")
    monkeypatch.setattr(jobs, "load_state", lambda _project: _state())
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="indoor"))
    ran: list[bool] = []
    monkeypatch.setattr(jobs, "run_refine_openings", lambda *_a, **kwargs: ran.append(kwargs["force"]))
    monkeypatch.setattr(sys, "argv", ["jobs", "refine_openings", "--project", str(project), "--las", str(project / "c.las"), "--level", LEVEL, "--mode", "indoor", "--force"])
    jobs.main()
    status = json.loads((project / "products" / "dxf" / "job_status.json").read_text(encoding="utf-8"))
    assert ran == [True]
    assert status["state"] == "done" and status["action"] == "refine_openings"


def test_cli_cancel_writes_cancelled_status(monkeypatch: pytest.MonkeyPatch) -> None:
    project = _workdir("cli_cancel")
    monkeypatch.setattr(jobs, "load_state", lambda _project: _state())
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="indoor"))

    def stop(*_a, **_k):
        raise Cancelled("stopped")

    monkeypatch.setattr(jobs, "run_refine_openings", stop)
    monkeypatch.setattr(sys, "argv", ["jobs", "refine_openings", "--project", str(project), "--las", str(project / "c.las"), "--level", LEVEL, "--mode", "indoor"])
    with pytest.raises(SystemExit) as exit_info:
        jobs.main()
    assert exit_info.value.code == 2
    assert json.loads((project / "products" / "dxf" / "job_status.json").read_text(encoding="utf-8"))["state"] == "cancelled"
