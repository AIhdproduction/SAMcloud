"""Automatic final pipeline stage "DXF/IFC export" (workflow step 8): flag, hook, level selection, accounting."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from conftest import use_language
from samcloud.app import i18n
from samcloud.app.dxf_controller import DxfPageController
from samcloud.app.pages.home import _processing_time_summary
from samcloud.app.processing import ProcessingManager, display_progress_text
from samcloud.dxf import job_runner as job_runner_module
from samcloud.dxf.jobs import dxf_run_metrics_path, write_dxf_run_metrics
from samcloud.dxf.state import load_state, save_state
from samcloud.engine import auto_dxf, pipeline


# ------------------------------------------------------------------------------------------ setting -> flag


def _command(workflow: dict[str, Any]) -> list[str]:
    project = {
        "settings": {"workflow": workflow, "classification": {"class_set": "indoor"}, "coordinate_systems": {}},
        "acquisition": {"type": "camera", "sources": {"image_directory": "D:/images"}, "georeferencing": {}},
    }
    return ProcessingManager._pipeline_command(object(), Path("D:/demo"), project, "none")


@pytest.mark.parametrize(
    ("workflow", "expected"),
    [({}, "1"), ({"auto_dxf": True}, "1"), ({"auto_dxf": False}, "0")],
    ids=["missing key is ON", "enabled", "disabled"],
)
def test_pipeline_command_passes_auto_dxf_flag(workflow: dict[str, Any], expected: str) -> None:
    command = _command(workflow)
    assert command[command.index("--auto-dxf") + 1] == expected


def test_auto_dxf_enabled_defaults_to_on() -> None:
    assert auto_dxf.auto_dxf_enabled({}) is True
    assert auto_dxf.auto_dxf_enabled({"workflow": {}}) is True
    assert auto_dxf.auto_dxf_enabled({"workflow": {"auto_dxf": False}}) is False
    assert auto_dxf.auto_dxf_enabled(None) is True


def test_pipeline_parses_auto_dxf_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outdir = tmp_path / "proj" / "products"
    # --frame-interval 0 makes main() stop right after parsing, which is all this test needs.
    for flag, expected in (("1", True), ("0", False)):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--outdir", str(outdir), "--auto-dxf", flag, "--frame-interval", "0"])
        with pytest.raises(SystemExit):
            pipeline.main()
        assert pipeline._AUTO_DXF_REQUEST["enabled"] is expected
        assert pipeline._AUTO_DXF_REQUEST["project_dir"] == (tmp_path / "proj").resolve()


# ------------------------------------------------------------------------------------------ hook behaviour


def _write_run_metrics(project: Path, status: str, started_at: str = "2999-01-01T00:00:00+00:00", total: float = 100.0) -> Path:
    reports = project / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "run_id": "20990101_000000Z", "started_at": started_at, "total_seconds": total, "stage_seconds": {}}
    (reports / "run_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    (reports / "run_summary.txt").write_text("Run summary\n\nWarnings\n--------\n- existing\n", encoding="utf-8")
    return reports


def _request(project: Path, enabled: bool = True) -> dict[str, Any]:
    return {"enabled": enabled, "project_dir": project, "progress_file": None, "started": 0.0}


@pytest.fixture
def stage_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []

    def fake_run(project_dir: Path | str, *, progress_file: Any = None) -> dict[str, Any]:
        calls.append(Path(project_dir))
        return {"status": "completed", "reason": "", "total_seconds": 0.0}

    monkeypatch.setattr(auto_dxf, "run_auto_dxf", fake_run)
    return calls


def test_stage_runs_after_a_completed_run_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage_calls: list[Path]) -> None:
    _write_run_metrics(tmp_path, "Completed")
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", _request(tmp_path))
    pipeline.run_auto_dxf_stage()
    assert stage_calls == [tmp_path]


@pytest.mark.parametrize("status", ["Failed", "Waiting for control points"])
def test_stage_is_not_called_when_the_run_did_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage_calls: list[Path], status: str,
) -> None:
    _write_run_metrics(tmp_path, status)
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", _request(tmp_path))
    pipeline.run_auto_dxf_stage()
    assert stage_calls == []


def test_stage_is_not_called_without_a_fresh_run_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage_calls: list[Path]) -> None:
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", _request(tmp_path))
    pipeline.run_auto_dxf_stage()  # no run_metrics.json at all
    _write_run_metrics(tmp_path, "Completed", started_at="2000-01-01T00:00:00+00:00")  # left over from an older run
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", {**_request(tmp_path), "started": 1_000_000_000.0 * 2})
    pipeline.run_auto_dxf_stage()
    assert stage_calls == []


def test_stage_is_not_called_when_disabled_or_never_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage_calls: list[Path]) -> None:
    _write_run_metrics(tmp_path, "Completed")
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", _request(tmp_path, enabled=False))
    pipeline.run_auto_dxf_stage()
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", None)
    pipeline.run_auto_dxf_stage()
    assert stage_calls == []


def test_cli_runs_the_stage_after_main_only_when_main_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(pipeline, "main", lambda: order.append("main"))
    monkeypatch.setattr(pipeline, "run_auto_dxf_stage", lambda: order.append("stage"))
    pipeline.run_cli()
    assert order == ["main", "stage"]

    def failing_main() -> None:
        order.append("main")
        raise RuntimeError("pipeline failed")

    order.clear()
    monkeypatch.setattr(pipeline, "main", failing_main)
    with pytest.raises(RuntimeError):
        pipeline.run_cli()
    assert order == ["main"]


def test_stage_crash_never_breaks_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_run_metrics(tmp_path, "Completed")
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", _request(tmp_path))

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(auto_dxf, "run_after_pipeline", boom)
    pipeline.run_auto_dxf_stage()  # must not raise


# ------------------------------------------------------------------------------------------ the stage itself


class _FakeProcess:
    def __init__(self, code: int) -> None:
        self._code = code

    def wait(self) -> int:
        return self._code


class _FakeJobs:
    """Replaces DxfJobRunner.start: records the calls and mimics what the real job leaves behind."""

    def __init__(self, project: Path, fail_levels: tuple[str | None, ...] = (), detect_levels: tuple[str, ...] = ()) -> None:
        self.project = project
        self.fail_levels = fail_levels
        self.detect_levels = detect_levels
        self.calls: list[dict[str, Any]] = []
        self.progress_seen: list[dict[str, Any]] = []

    def start(self, runner: Any, action: str, las: Any, level: str | None = None, mode: str | None = None,
              source: str | None = None, variant: str | None = None, force: bool = False) -> _FakeProcess:
        self.calls.append({"action": action, "las": Path(las), "level": level, "mode": mode, "source": source, "variant": variant, "force": force})
        progress = self.project / "reports" / "progress.json"
        if progress.is_file():
            self.progress_seen.append(json.loads(progress.read_text(encoding="utf-8")))
        if action == "prepare_levels":
            state = load_state(self.project)
            state.levels = [{"id": item, "name": item, "floor_z": 0.0, "ceiling_z": 2.5, "checked": True} for item in self.detect_levels]
            save_state(self.project, state)
            return _FakeProcess(0 if self.detect_levels else 1)
        failed = level in self.fail_levels
        write_dxf_run_metrics(self.project, {
            "action": "run_all", "level": level, "status": "failed" if failed else "completed",
            "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:01:00+00:00",
            "total_seconds": 10.0, "step_seconds": {"Build orthophoto": 4.0, "Export DXF": 6.0},
        })
        return _FakeProcess(1 if failed else 0)


def _project(tmp_path: Path, *, mode: str, levels: list[dict[str, Any]] | None = None) -> Path:
    project = tmp_path / "proj"
    (project / "products" / "classified").mkdir(parents=True)
    (project / "products" / "classified" / "classified.ply").write_bytes(b"ply")
    _write_run_metrics(project, "Completed")
    state = load_state(project)
    state.mode_override = mode
    state.profile_name = "sam3_indoor" if mode == "indoor" else "sam3_outdoor"
    state.levels = levels or []
    state.segment_source = "combined"
    save_state(project, state)
    return project


def _install(monkeypatch: pytest.MonkeyPatch, jobs: _FakeJobs) -> None:
    monkeypatch.setattr(job_runner_module.DxfJobRunner, "start", lambda self, *args, **kwargs: jobs.start(self, *args, **kwargs))


def _levels(*ids: str, unchecked: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    return [{"id": item, "name": item, "floor_z": float(index), "ceiling_z": index + 2.5, "checked": item not in unchecked}
            for index, item in enumerate(ids)]


def test_level_selection_matches_the_page(tmp_path: Path) -> None:
    indoor = DxfPageController(_project(tmp_path / "a", mode="indoor", levels=_levels("level_1", "level_2", "level_3", unchecked=("level_2",))))
    assert auto_dxf.select_levels(indoor) == [str(level["id"]) for level in indoor.checked_levels()] == ["level_1", "level_3"]
    assert auto_dxf.select_levels(indoor) == indoor.review_targets()
    outdoor = DxfPageController(_project(tmp_path / "b", mode="outdoor", levels=_levels("level_1")))
    assert auto_dxf.select_levels(outdoor) == [None] == outdoor.review_targets()


def test_indoor_runs_run_all_per_level_like_the_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, mode="indoor", levels=_levels("level_1", "level_2"))
    jobs = _FakeJobs(project)
    _install(monkeypatch, jobs)
    result = auto_dxf.run_auto_dxf(project)
    assert result["status"] == "completed" and result["levels"] == ["level_1", "level_2"]
    assert [(call["action"], call["level"]) for call in jobs.calls] == [("run_all", "level_1"), ("run_all", "level_2")]
    for call in jobs.calls:  # the page: --las classified.ply --mode --source combined --variant ortho
        assert call["las"] == (project / "products" / "classified" / "classified.ply").resolve()
        assert (call["mode"], call["source"], call["variant"], call["force"]) == ("indoor", "combined", "ortho", True)
    combined = json.loads(dxf_run_metrics_path(project).read_text(encoding="utf-8"))
    assert combined["status"] == "completed" and combined["total_seconds"] == 20.0
    assert combined["step_seconds"] == {"Build orthophoto": 8.0, "Export DXF": 12.0}


def test_indoor_without_stored_levels_detects_storeys_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, mode="indoor")
    jobs = _FakeJobs(project, detect_levels=("level_1",))
    _install(monkeypatch, jobs)
    result = auto_dxf.run_auto_dxf(project)
    assert [(call["action"], call["level"]) for call in jobs.calls] == [("prepare_levels", None), ("run_all", "level_1")]
    assert result["status"] == "completed"


def test_indoor_without_detectable_storeys_fails_with_a_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, mode="indoor")
    jobs = _FakeJobs(project)
    _install(monkeypatch, jobs)
    result = auto_dxf.run_auto_dxf(project)
    assert result["status"] == "failed" and "storeys" in result["reason"]
    assert [call["action"] for call in jobs.calls] == ["prepare_levels"]


def test_outdoor_runs_a_single_run_all_without_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, mode="outdoor")
    jobs = _FakeJobs(project)
    _install(monkeypatch, jobs)
    result = auto_dxf.run_auto_dxf(project)
    assert result["status"] == "completed" and result["mode"] == "outdoor"
    assert [(call["action"], call["level"], call["mode"]) for call in jobs.calls] == [("run_all", None, "outdoor")]


def test_unknown_profile_is_skipped_with_a_recorded_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, mode="outdoor")
    state = load_state(project)
    state.profile_name = None
    save_state(project, state)
    jobs = _FakeJobs(project)
    _install(monkeypatch, jobs)
    monkeypatch.setattr(DxfPageController, "profile", property(lambda self: None))
    result = auto_dxf.run_auto_dxf(project)
    assert result["status"] == "skipped" and "profile" in result["reason"]
    assert jobs.calls == []
    assert "skipped" in (project / "reports" / "run_summary.txt").read_text(encoding="utf-8")


def test_failure_keeps_the_run_completed_records_failed_metrics_and_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, mode="indoor", levels=_levels("level_1", "level_2"))
    metrics_before = (project / "reports" / "run_metrics.json").read_text(encoding="utf-8")
    jobs = _FakeJobs(project, fail_levels=("level_2",))
    _install(monkeypatch, jobs)

    result = auto_dxf.run_after_pipeline(project)

    assert result is not None and result["status"] == "failed" and result["failed_levels"] == ["level_2"]
    assert (project / "reports" / "run_metrics.json").read_text(encoding="utf-8") == metrics_before, "processing status stays Completed"
    assert json.loads(dxf_run_metrics_path(project).read_text(encoding="utf-8"))["status"] == "failed"
    summary = (project / "reports" / "run_summary.txt").read_text(encoding="utf-8")
    assert "- existing" in summary and summary.count("WARNING [auto DXF/IFC]") == 1 and "level_2" in summary
    assert json.loads((project / "reports" / "auto_dxf.json").read_text(encoding="utf-8"))["status"] == "failed"

    # A later successful run replaces the warning instead of stacking another one.
    jobs.fail_levels = ()
    assert auto_dxf.run_auto_dxf(project)["status"] == "completed"
    assert "WARNING [auto DXF/IFC]" not in (project / "reports" / "run_summary.txt").read_text(encoding="utf-8")


def test_unexpected_error_inside_the_stage_is_recorded_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, mode="outdoor")

    def boom(self: Any, *args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(job_runner_module.DxfJobRunner, "start", boom)
    result = auto_dxf.run_auto_dxf(project)
    assert result["status"] == "failed" and "disk full" in result["reason"]
    assert "WARNING [auto DXF/IFC]" in (project / "reports" / "run_summary.txt").read_text(encoding="utf-8")


def test_run_metrics_total_excludes_the_auto_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, mode="outdoor")
    reports = project / "reports"
    run = {"status": "Completed", "run_id": "20990101_000000Z", "started_at": "2026-11-17T10:00:00+00:00", "total_seconds": 1000.0,
           "stage_seconds": {"Sparse reconstruction": 400.0, "Dense stereo": 500.0}}
    (reports / "run_metrics.json").write_text(json.dumps(run), encoding="utf-8")
    (reports / "run_metrics_20990101_000000Z.json").write_text(json.dumps(run), encoding="utf-8")
    jobs = _FakeJobs(project)
    _install(monkeypatch, jobs)

    auto_dxf.run_auto_dxf(project)

    for name in ("run_metrics.json", "run_metrics_20990101_000000Z.json"):
        assert json.loads((reports / name).read_text(encoding="utf-8"))["total_seconds"] == 1000.0
    summary = _processing_time_summary(project)
    assert summary is not None
    assert summary["groups"]["DXF/IFC-Export"] == 10.0
    assert summary["total_seconds"] == 1010.0, "pipeline 1000 s + DXF/IFC 10 s, counted once"
    assert "Übrige Schritte" not in summary["groups"] or summary["groups"]["Übrige Schritte"] == 100.0


def test_progress_json_gets_the_dxf_ifc_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, mode="indoor", levels=_levels("level_1", "level_2"))
    progress_file = project / "reports" / "progress.json"
    progress_file.write_text(json.dumps({"percent": 100.0, "stage": "Complete", "stages": [{"key": "classification", "status": "done"}]}), encoding="utf-8")
    jobs = _FakeJobs(project)
    _install(monkeypatch, jobs)

    auto_dxf.run_auto_dxf(project, progress_file=progress_file)

    assert [(item["stage"], item["detail"]) for item in jobs.progress_seen] == [
        ("DXF/IFC export", "level_1 (1/2)"), ("DXF/IFC export", "level_2 (2/2)"),
    ]
    assert all(item["percent"] == 100.0 and item["stages"] for item in jobs.progress_seen), "steps 1-6 stay complete"
    final = json.loads(progress_file.read_text(encoding="utf-8"))
    assert (final["stage"], final["percent"]) == ("Complete", 100.0)


def test_standalone_main_returns_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    project = _project(tmp_path, mode="outdoor")
    jobs = _FakeJobs(project)
    _install(monkeypatch, jobs)
    assert auto_dxf.main(["--project", str(project)]) == 0
    jobs.fail_levels = (None,)
    assert auto_dxf.main(["--project", str(project)]) == 1
    capsys.readouterr()


# ------------------------------------------------------------------------------------------ texts


@pytest.mark.parametrize("language", ["de", "en"])
def test_locale_keys_exist(monkeypatch: pytest.MonkeyPatch, language: str) -> None:
    catalogue = i18n.catalogue(language)
    for key in ("workflow.stage.dxf_ifc", "projects.create.auto_dxf", "settings.processing.auto_dxf"):
        assert catalogue.get(key), f"{key} missing in {language}"
    if language == "de":
        assert "automatisch" in catalogue["projects.create.auto_dxf"]
        use_language(monkeypatch, "de")
        i18n.reload_translations()
        assert display_progress_text(auto_dxf.STAGE_TEXT) == "DXF/IFC-Export"
