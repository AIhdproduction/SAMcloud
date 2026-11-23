"""Workflow panel: live step 1 (image preparation) and the AprilTag status of the control-point step.

All projects, progress files and reports are synthetic. The panel is rendered once; the refresh timer
callback is driven by hand, so every check runs without a page reload.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from nicegui import ui
from nicegui.testing.user_simulation import user_simulation

from samcloud.app.components import workflow_display
from samcloud.app.pages import home
from samcloud.app.state import state as app_state

_CALLBACKS: list[Any] = []
_CLIENT: dict[str, Any] = {}


def _route() -> None:
    _CLIENT["client"] = ui.context.client
    home._render_workflow(app_state.active_project())


# NiceGUI's test reset drops sys.modules entries of page modules outside "tests."; keep samcloud loaded.
_route.__module__ = "tests.workflow_live_steps_route"


def _now_iso(offset: float = 0.0) -> str:
    return datetime.fromtimestamp(time.time() + offset, tz=timezone.utc).isoformat()


def _project(
    root: Path,
    workflow: dict,
    *,
    processing: dict | None = None,
    apriltags: bool = False,
    mode: str | None = None,
) -> tuple[Path, dict]:
    directory = root / "Synthetic"
    directory.mkdir(parents=True, exist_ok=True)
    document: dict = {
        "project": {"name": "Synthetic", "updated_at": ""},
        "workflow": dict(workflow),
        "processing": processing or {},
        "settings": {"apriltags": {"enabled": apriltags}},
    }
    if mode is not None:
        document["workflow"]["control_points_mode"] = mode
    _save(directory, document)
    return directory, document


def _save(directory: Path, document: dict) -> None:
    (directory / "project.samcloud.yml").write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def _fresh_run(**extra: Any) -> dict:
    """Processing block of a full run that was just started by the app."""
    return {
        "started_at": _now_iso(), "completed_at": "", "last_error": "",
        "progress": {"percent": 0, "stage": "Preparing images", "detail": ""}, **extra,
    }


_RUNNING = {"alignment": "running", "control_points": "not_started", "dense_cloud": "queued",
            "classification": "queued", "refinement": "queued", "export": "not_started"}
_FINISHED = {"alignment": "completed", "control_points": "skipped", "dense_cloud": "completed",
             "classification": "completed", "refinement": "completed", "export": "not_started"}


def _prepare_progress(status: str, percent: float, detail: str = "", *, started_at: float | None = None) -> dict:
    prepared = status == "done"
    return {
        "percent": 0.0 if not prepared else 4.0, "stage": "Preparing images" if not prepared else "Extracting features",
        "detail": detail, "started_at": started_at if started_at is not None else time.time(), "updated_at": time.time(),
        "stage_key": "prepare_images" if not prepared else "feature_extraction",
        "stage_percent": percent, "stage_eta_seconds": 125.0 if not prepared else 60.0, "eta_seconds": 900.0,
        "stages": [
            {"key": "prepare_images", "status": status, "percent": percent, "expected_seconds": 300},
            {"key": "feature_extraction", "status": "running" if prepared else "pending",
             "percent": 10.0 if prepared else 0.0, "expected_seconds": 600},
            {"key": "matching", "status": "pending", "percent": 0.0, "expected_seconds": 600},
        ],
    }


def _steps() -> list[dict]:
    """Rendered workflow steps: classes, label, status text and the bar row (value, text) if any."""
    result = []
    for element in list(_CLIENT["client"].elements.values()):
        if "sam-workflow-step" not in element.classes:
            continue
        children = list(element.default_slot.children)
        labels = [child for child in children if isinstance(child, ui.label)]
        bar = None
        for row in (child for child in children if isinstance(child, ui.row)):
            inner = list(row.default_slot.children)
            progress = next((c for c in inner if isinstance(c, ui.linear_progress)), None)
            text = next((c for c in inner if isinstance(c, ui.label)), None)
            if progress is not None and text is not None:
                bar = (progress.value, text.text.replace(" ", " "))
        result.append({
            "label": labels[1].text if len(labels) > 1 else "", "status": labels[2].text if len(labels) > 2 else "",
            "active": "is-active" in element.classes, "complete": "is-complete" in element.classes, "bar": bar,
        })
    return result


def _refresh() -> None:
    callback = next(cb for cb in reversed(_CALLBACKS) if getattr(cb, "__name__", "") == "refresh_workflow")
    callback()


@pytest.fixture
def panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Open a user simulation; ``await show(directory)`` renders the panel for that project."""
    _CALLBACKS.clear()
    monkeypatch.setattr(ui, "timer", lambda _interval, callback, **_kw: _CALLBACKS.append(callback))
    workflow_display._REPORT_CACHE.clear()
    home._PROGRESS_FILE_CACHE.clear()
    return tmp_path


async def _show(user: Any, directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_state, "active_project_directory", directory)
    ui.page("/w")(_route)
    await user.open("/w")


# ------------------------------------------------------------------------------------ step 1

@pytest.mark.anyio
async def test_step_one_appears_live_from_start_without_reload(panel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, _ = _project(panel, _RUNNING, processing=_fresh_run())
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        step = _steps()[0]
        assert step["status"] == "running" and step["active"]
        assert step["bar"] == (0.0, "0%")  # no progress.json yet: running, 0 %, no invented time
        assert _steps()[1]["status"] == "queued" and not _steps()[1]["active"]

        _write_json(directory / "reports" / "progress.json", _prepare_progress("running", 40, "Frames extrahieren 264/463"))
        _refresh()
        step = _steps()[0]
        assert step["status"] == "running" and step["active"] and not step["complete"]
        value, text = step["bar"]
        assert value == pytest.approx(0.4)
        assert text.startswith("40% · Frames extrahieren 264/463 · ") and "2m 05s" in text
        assert len(_steps()) == 8

        _write_json(directory / "reports" / "progress.json",
                    _prepare_progress("running", 83, "Würfelflächen 1500/2778"))
        _refresh()
        assert "Würfelflächen 1500/2778" in _steps()[0]["bar"][1]

        _write_json(directory / "reports" / "progress.json", _prepare_progress("done", 100))
        _refresh()
        first, second = _steps()[0], _steps()[1]
        assert first["status"] == "completed" and first["complete"] and not first["active"]
        assert first["bar"] == (1.0, "100%")
        assert second["status"] == "running" and second["active"]


@pytest.mark.anyio
async def test_step_one_ignores_progress_file_of_an_earlier_run(panel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, _ = _project(panel, _RUNNING, processing=_fresh_run())
    _write_json(directory / "reports" / "progress.json",
                _prepare_progress("done", 100, started_at=time.time() - 86400))
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        step = _steps()[0]
        assert step["status"] == "running" and step["bar"] == (0.0, "0%")
        assert not step["complete"]
        assert _steps()[1]["status"] == "queued"


@pytest.mark.anyio
async def test_resume_run_keeps_step_one_unchanged(panel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = {**_RUNNING, "alignment": "completed", "control_points": "completed", "dense_cloud": "running"}
    processing = {"started_at": _now_iso(), "progress": {"percent": 57, "stage": "Applying control points", "detail": ""}}
    directory, _ = _project(panel, workflow, processing=processing)
    _write_json(directory / "reports" / "progress.json", {
        "percent": 60.0, "stage": "Tiefenkarten", "stage_key": "dense_stereo", "started_at": time.time(),
        "stages": [{"key": "dense_stereo", "status": "running", "percent": 10.0, "expected_seconds": 100}],
    })
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        step = _steps()[0]
        assert step["bar"] is None and step["status"] == "ready"
        _refresh()
        assert _steps()[0]["bar"] is None and _steps()[0]["status"] == "ready"


@pytest.mark.anyio
async def test_reclassify_and_old_progress_show_no_false_bar(panel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = {**_FINISHED, "classification": "running", "refinement": "queued"}
    processing = {"started_at": _now_iso(), "progress": {"percent": 87, "stage": "Reclassifying", "detail": ""}}
    directory, _ = _project(panel, workflow, processing=processing)
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        assert _steps()[0]["bar"] is None and _steps()[0]["status"] == "ready"
        assert _steps()[4]["status"] == "running"

    old_directory = panel / "old"
    old_directory.mkdir()
    directory, _ = _project(old_directory, _RUNNING, processing={
        "started_at": _now_iso(), "progress": {"percent": 20, "stage": "Aligning cameras", "detail": ""}})
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        assert _steps()[0]["bar"] is None and _steps()[0]["status"] == "ready"


def test_prepare_pending_only_for_fresh_full_runs() -> None:
    fresh = {"workflow": {"alignment": "running"}}
    assert home._prepare_pending(fresh, {"percent": 0, "stage": "Preparing images"})
    assert not home._prepare_pending(fresh, {"percent": 10, "stage": "Aligning cameras"})
    assert not home._prepare_pending(fresh, {"stage": "Preparing images", "stages": [{"key": "mapping"}]})
    assert not home._prepare_pending({"workflow": {"classification": "running"}}, {"stage": "Preparing images"})
    assert not home._prepare_pending(None, {})


# ------------------------------------------------------------------------------------ step 3

def _report(mode: str, count: int = 8, *, scale_applied: float | None = 1.1, rejected: int = 0) -> dict:
    tags = [{"id": index, "status": "rejected" if index < rejected else "ok"} for index in range(count)]
    return {"mode": mode, "scale_applied": scale_applied, "warnings": [], "tags": tags}


def _write_report(directory: Path, report: Any, *, age: float = 0.0) -> Path:
    path = directory / "reports" / "apriltag_report.json"
    _write_json(path, report)
    if age:
        old = time.time() - age
        os.utime(path, (old, old))
    return path


def _control_points(directory: Path) -> dict:
    return next(step for step in _steps() if step["label"] == "Add control points")


_SKIPPED = "skipped (no control points)"
_AFTER_ALIGNMENT = {**_RUNNING, "alignment": "completed", "dense_cloud": "running"}
_AFTER_PROCESSING = {"started_at": _now_iso(), "progress": {"percent": 70, "stage": "Building dense cloud", "detail": ""}}


@pytest.mark.anyio
@pytest.mark.parametrize(("report", "text", "complete", "active"), [
    (_report("scale"), "8 AprilTags found · scale set", True, False),
    (_report("georef", scale_applied=1.0), "8 AprilTags found · georeferenced", True, False),
    (_report("none", scale_applied=None), "8 AprilTags found · no scale", False, False),
    (_report("scale", scale_applied=None), "8 AprilTags found · no scale", False, False),
    (_report("scale", count=1), "1 AprilTag found · scale set", True, False),
    (_report("scale", rejected=2), "8 AprilTags found · scale set (2 rejected)", True, False),
    (_report("scale", count=3, rejected=1), "3 AprilTags found · scale set (1 rejected)", True, False),
    (_report("scale", count=0), _SKIPPED, False, False),
])
async def test_step_three_found_states(
    panel: Path, monkeypatch: pytest.MonkeyPatch, report: dict, text: str, complete: bool, active: bool,
) -> None:
    directory, _ = _project(panel, _AFTER_ALIGNMENT, processing=_AFTER_PROCESSING, apriltags=True)
    _write_report(directory, report)
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        step = _control_points(directory)
        assert step["status"] == text
        assert step["complete"] is complete and step["active"] is active


@pytest.mark.anyio
async def test_step_three_pending_then_found_without_reload(panel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, document = _project(panel, _RUNNING, processing=_fresh_run(), apriltags=True)
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        step = _control_points(directory)
        assert step["status"] == "AprilTag search pending" and not step["active"] and not step["complete"]

        _refresh()  # the search itself is not shown: the text stays until the report exists
        assert _control_points(directory)["status"] == "AprilTag search pending"

        _write_report(directory, _report("scale"))
        document["processing"]["progress"] = {"percent": 57, "stage": "Preparing dense reconstruction", "detail": ""}
        _save(directory, document)
        _refresh()
        step = _control_points(directory)
        assert step["status"] == "8 AprilTags found · scale set" and step["complete"] and not step["active"]


@pytest.mark.anyio
async def test_step_three_stale_report_of_earlier_run_is_ignored_while_aligning(
    panel: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, document = _project(panel, _RUNNING, processing=_fresh_run(), apriltags=True)
    _write_report(directory, _report("scale"), age=86400)
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        assert _control_points(directory)["status"] == "AprilTag search pending"
        # a finished project keeps the report whatever its age (also after reclassify / refine)
        document["workflow"] = dict(_FINISHED)
        _save(directory, document)
        workflow_display._REPORT_CACHE.clear()
    finished_directory = panel / "finished"
    finished_directory.mkdir()
    directory, _ = _project(finished_directory, _FINISHED, processing={"started_at": _now_iso()}, apriltags=True)
    _write_report(directory, _report("scale"), age=86400)
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        assert _control_points(directory)["status"] == "8 AprilTags found · scale set"


@pytest.mark.anyio
@pytest.mark.parametrize("content", ["", "{\"mode\": \"scale\", \"tags\": [", "[1, 2]", "{\"mode\": \"scale\"}", "null"])
async def test_step_three_broken_report_never_raises(panel: Path, monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    directory, document = _project(panel, _RUNNING, processing=_fresh_run(), apriltags=True)
    path = _write_report(directory, content)
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        assert _control_points(directory)["status"] == "AprilTag search pending"
        _refresh()
        assert _control_points(directory)["status"] == "AprilTag search pending"
        # the writer finishes: the next refresh picks the report up
        path.write_text(json.dumps(_report("scale")), encoding="utf-8")
        document["workflow"] = dict(_AFTER_ALIGNMENT)
        document["processing"] = dict(_AFTER_PROCESSING)
        _save(directory, document)
        _refresh()
        assert _control_points(directory)["status"] == "8 AprilTags found · scale set"


@pytest.mark.anyio
async def test_step_three_keeps_skipped_text_without_tags(panel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # AprilTags off, even with a leftover report
    directory, _ = _project(panel, _FINISHED, processing=_AFTER_PROCESSING, apriltags=False)
    _write_report(directory, _report("scale"))
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        assert _control_points(directory)["status"] == _SKIPPED
    # AprilTags on, finished project, search ended without a report
    second = panel / "second"
    second.mkdir()
    directory, _ = _project(second, _FINISHED, processing=_AFTER_PROCESSING, apriltags=True)
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        assert _control_points(directory)["status"] == _SKIPPED
    # AprilTags on, run already past the search, no report
    third = panel / "third"
    third.mkdir()
    directory, _ = _project(third, {**_RUNNING, "alignment": "running"},
                            processing={**_fresh_run(), "progress": {"percent": 70, "stage": "Building dense cloud"}},
                            apriltags=True)
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        assert _control_points(directory)["status"] == _SKIPPED


@pytest.mark.anyio
async def test_step_three_with_control_points_is_unchanged(panel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, _ = _project(panel, {**_AFTER_ALIGNMENT, "control_points": "completed"},
                            processing=_AFTER_PROCESSING, apriltags=True)
    _write_report(directory, _report("scale"))
    (directory / "control_points").mkdir()
    (directory / "control_points" / "control_points.csv").write_text("id,x,y,z\n", encoding="utf-8")
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        step = _control_points(directory)
        assert step["status"] == "completed" and step["complete"]
        assert "AprilTag" not in step["status"]

    waiting = panel / "waiting"
    waiting.mkdir()
    directory, _ = _project(waiting, {**_RUNNING, "alignment": "completed", "control_points": "waiting"},
                            processing={"started_at": _now_iso(),
                                        "progress": {"percent": 57, "stage": "Waiting for control points"}},
                            apriltags=True)
    _write_report(directory, _report("scale"))
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        assert _control_points(directory)["status"] == "waiting"


def test_apriltag_report_is_cached_by_file_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workflow_display._REPORT_CACHE.clear()
    path = _write_report(tmp_path, _report("scale"))
    first, _ = workflow_display.load_apriltag_report(path)
    reads: list[int] = []
    original = Path.read_text

    def counting(self: Path, *args: Any, **kwargs: Any) -> str:
        reads.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting)
    second, _ = workflow_display.load_apriltag_report(path)
    assert second is first and not reads
    assert workflow_display.load_apriltag_report(tmp_path / "missing.json") == (None, None)


@pytest.mark.anyio
async def test_step_three_german_texts(panel: Path, monkeypatch: pytest.MonkeyPatch, german_ui: None) -> None:
    directory, document = _project(panel, _RUNNING, processing=_fresh_run(), apriltags=True)
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        step = next(step for step in _steps() if step["label"] == _steps()[2]["label"])
        assert step["status"] == "AprilTag-Suche folgt"
        _refresh()
        assert _steps()[2]["status"] == "AprilTag-Suche folgt"
        _write_report(directory, _report("scale", rejected=1))
        document["workflow"] = dict(_AFTER_ALIGNMENT)
        document["processing"] = dict(_AFTER_PROCESSING)
        _save(directory, document)
        _refresh()
        assert _steps()[2]["status"] == "8 AprilTags gefunden · Massstab gesetzt (1 verworfen)"


def test_home_locales_have_identical_apriltag_keys() -> None:
    locales = Path(home.__file__).resolve().parents[1] / "locales"
    german = json.loads((locales / "de" / "home.json").read_text(encoding="utf-8"))["apriltag"]
    english = json.loads((locales / "en" / "home.json").read_text(encoding="utf-8"))["apriltag"]
    assert set(german) == set(english) and len(german) == 7


def test_phase_text_never_splits_a_value() -> None:
    text = home._nobreak_parts("40% · Frames extrahieren 264/463 · 9 min 33 s")
    assert text.split(" · ") == ["40%", "Frames extrahieren 264/463", "9 min 33 s"]
    assert text.replace(" ", " ") == "40% · Frames extrahieren 264/463 · 9 min 33 s"
    assert home._nobreak_parts("") == ""


@pytest.mark.anyio
async def test_active_step_has_inner_right_margin_and_nonbreaking_row(panel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, _ = _project(panel, _RUNNING, processing=_fresh_run())
    _write_json(directory / "reports" / "progress.json", _prepare_progress("running", 40, "Frames extrahieren 264/463"))
    async with user_simulation(root=lambda: None) as user:
        await _show(user, directory, monkeypatch)
        labels = [e for e in _CLIENT["client"].elements.values() if isinstance(e, ui.label) and "264/463" in e.text]
        assert labels and "Frames extrahieren 264/463" in labels[0].text
        assert labels[0]._style.get("overflow-wrap") == "anywhere"
    assert ".sam-workflow-step.is-active { padding-right:8px; }" in Path(home.__file__).read_text(encoding="utf-8")
