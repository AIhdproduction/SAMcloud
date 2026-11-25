"""AI assistant panel on the /dxf page: gating, shared 2D/3D selection, chat, preview, accept / undo / re-export, errors."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import struct
from typing import Any, AsyncIterator

import pytest
import yaml
from nicegui import ElementFilter, ui
from nicegui.testing.user import User
from nicegui.testing.user_simulation import user_simulation

from assistant_synthetic import FLOOR_Z, LEVEL_ID, TABLE_BOX, make_plan, make_project
from samcloud.app.components import assistant_panel as panel_module
from samcloud.app.pages import dxf as dxf_page
from samcloud.app.state import state as app_state
from samcloud.assistant import agent as agent_module
from samcloud.assistant import credentials as ai_creds
from samcloud.assistant import settings as ai_settings
from samcloud.assistant.agent import TurnResult
from samcloud.assistant.paths import assistant_dir
from samcloud.assistant.selection import global_id_for
from samcloud.assistant.service import AssistantService
from samcloud.dxf.job_runner import DxfJobRunner
from samcloud.dxf.state import load_state, save_state
from test_assistant_agent import FakeProvider, call, say

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

DUMMY_KEY = "sk-test-DUMMY-0000"
AMBER, RED, GREEN = "#ffb74d", "#ef6b73", "#4bc692"


# ------------------------------------------------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the real user's AI settings / credentials."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _enable_assistant(full_control: bool | None = False) -> None:
    """Enabled OpenAI assistant; proposal + accept by default (full control is tested in test_assistant_full_control*)."""
    settings = ai_settings.AssistantSettings(enabled=True, provider="openai", model="fake-model", cost_limit_request_usd=0.0,
                                             cost_limit_project_day_usd=0.0, privacy_ack_cloud=True, full_control=full_control)
    ai_settings.save_settings(settings)
    ai_creds.save_ai_credentials({"openai_api_key": DUMMY_KEY})


def _project(tmp_path: Path, *, ifc: bool = True, variant: str = "ortho") -> Path:
    """Synthetic indoor project (plan_model + clean-plan result + classified cloud) with one level and optionally the IFC."""
    project = make_project(tmp_path)
    document = {"project": {"name": "assistant_synthetic"}, "workflow": {},
                "settings": {"classification": {"class_set": "indoor", "model_profile": "sam3_indoor"}}}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    state = load_state(project)
    state.levels = [{"id": LEVEL_ID, "name": "EG", "floor_z": FLOOR_Z, "ceiling_z": 2.8, "point_count": 1000, "confidence": 1.0,
                     "low_confidence": False, "checked": True}]
    state.plan_variant = variant
    save_state(project, state)
    if ifc:
        from samcloud.bim.jobs import run_ifc_export

        run_ifc_export(project, LEVEL_ID)
    AssistantService(project, LEVEL_ID).discard()
    return project


def _record_jobs(monkeypatch: pytest.MonkeyPatch, *, end_state: str = "done",
                 start_error: str | None = None) -> list[tuple[str, str | None]]:
    """Record DxfJobRunner.start (no subprocess); a recorded job counts as ended (``end_state``) right away.
    ``start_error`` makes every start raise RuntimeError with that text (nothing recorded)."""
    calls: list[tuple[str, str | None]] = []

    def start(self: DxfJobRunner, action: str, las: Any, level: str | None = None, **_kwargs: Any) -> None:
        if start_error is not None:
            raise RuntimeError(start_error)
        calls.append((str(action), level))

    def status(self: DxfJobRunner) -> dict[str, Any]:
        action = calls[-1][0] if calls else ""
        return {"action": action, "state": end_state if calls else "idle", "fraction": 1.0 if calls else 0.0, "stage": "",
                "message": "", "started": "", "finished": None, "pid": None}

    monkeypatch.setattr(DxfJobRunner, "start", start)
    monkeypatch.setattr(DxfJobRunner, "status", status)
    monkeypatch.setattr(DxfJobRunner, "is_running", lambda self: False)
    return calls


@pytest.fixture
def jobs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None]]:
    """Recorded runner jobs that end "done" right away."""
    return _record_jobs(monkeypatch)


def _dxf_route() -> None:
    dxf_page.render()


_dxf_route.__module__ = "tests.assistant_panel_route"


@asynccontextmanager
async def _dxf_user(project: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[User]:
    monkeypatch.setattr(app_state, "active_project_directory", project)
    monkeypatch.setattr(dxf_page, "_RUNNERS", {})
    async with user_simulation(root=lambda: None) as user:
        ui.page("/dxf")(_dxf_route)
        await user.open("/dxf")
        yield user
        await asyncio.sleep(0.3)  # let the outbox flush before the simulation resets the app


def _element(user: User, marker: str) -> Any:
    elements = user.find(marker=marker).elements
    assert len(elements) == 1, f"expected one element marked {marker!r}, found {len(elements)}"
    return next(iter(elements))


def _has(user: User, marker: str) -> bool:
    """Non-raising lookup (User.find raises when nothing matches)."""
    with user:
        return any(True for _element in ElementFilter(marker=marker))


def _set(user: User, marker: str, value: Any) -> None:
    with user:
        _element(user, marker).value = value


def _emit(element: Any, event_type: str, args: Any) -> None:
    """Deliver a browser event (as the Vue component would) to every Python listener of an element."""
    listeners = [listener for listener in element._event_listeners.values() if listener.type == event_type]
    assert listeners, f"no {event_type} listener registered"
    for listener in listeners:
        element._handle_event({"listener_id": listener.id, "args": args})


def _pick(user: User, x: float, y: float, ctrl: bool = False) -> None:
    with user:
        _emit(_element(user, "result-canvas"), "pick", {"x": x, "y": y, "ctrl": ctrl})


async def _chips(user: User, prefix: str = "assistant-chip-") -> list[str]:
    """Texts of the selection chips marked <prefix>0, <prefix>1, ... in order (after pending refreshes ran)."""
    await asyncio.sleep(0.05)  # refreshable.refresh() outside an await runs on the next loop iteration
    with user:
        found = {}
        for chip in ElementFilter(kind=ui.chip):
            for marker in chip._markers:
                if marker.startswith(prefix) and marker[len(prefix):].isdigit():
                    found[int(marker[len(prefix):])] = chip.text
    return [found[index] for index in sorted(found)]


def _no_error_logs(caplog: pytest.LogCaptureFixture) -> None:
    errors = [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
    assert not errors, errors


async def _eventually(check: Any, timeout: float = 6.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if check():
            return
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.1)


def _use(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> FakeProvider:
    monkeypatch.setattr(agent_module, "get_provider", lambda name, settings, creds: provider)
    return provider


async def _ask(user: User, text: str, expect: str) -> None:
    _set(user, "assistant-input", text)
    user.find(marker="assistant-send").click()
    await user.should_see(expect, retries=60)


ASSISTANT_PROPS = ("pick-enabled", "pick-floor", "selection-overlay", "preview-overlay", "selected-many", "highlight", "boxes")


# ------------------------------------------------------------------------------------------------- 1. gating

@pytest.mark.anyio
async def test_assistant_off_page_and_viewer_props_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            caplog: pytest.LogCaptureFixture, jobs: list) -> None:
    project = _project(tmp_path)

    def no_credentials() -> dict[str, str]:
        raise AssertionError("credentials must not be read while the assistant is off")

    monkeypatch.setattr(ai_creds, "load_ai_credentials", no_credentials)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="result-canvas")
        await user.should_see(marker="ifc-viewer")
        await user.should_not_see(marker="assistant-panel")
        await user.should_not_see(marker="assistant-panel-ifc")
        canvas_off = dict(_element(user, "result-canvas").props)
        viewer_off = dict(_element(user, "ifc-viewer").props)
        canvas_listeners = sorted(listener.type for listener in _element(user, "result-canvas")._event_listeners.values())
        viewer_listeners = sorted(listener.type for listener in _element(user, "ifc-viewer")._event_listeners.values())
    assert not any(name in canvas_off or name in viewer_off for name in ASSISTANT_PROPS)
    assert "pick" not in canvas_listeners and viewer_listeners.count("pick") == 1
    assert jobs == []

    # the same page with the assistant module patched out renders identical viewer props and listeners
    monkeypatch.setattr(dxf_page, "create_assistant_panel", lambda page, level_id: None)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="ifc-viewer")
        assert dict(_element(user, "result-canvas").props) == canvas_off
        assert dict(_element(user, "ifc-viewer").props) == viewer_off
        assert sorted(listener.type for listener in _element(user, "result-canvas")._event_listeners.values()) == canvas_listeners
        assert sorted(listener.type for listener in _element(user, "ifc-viewer")._event_listeners.values()) == viewer_listeners
    assert not assistant_dir(project, LEVEL_ID).exists()  # nothing written while off
    _no_error_logs(caplog)


@pytest.mark.anyio
async def test_enabled_but_not_configured_shows_no_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    project = _project(tmp_path, ifc=False)
    ai_settings.save_settings(ai_settings.AssistantSettings(enabled=True, provider="openai", model="fake-model"))  # no key
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="result-canvas")
        await user.should_not_see(marker="assistant-panel")
        assert "pick-enabled" not in _element(user, "result-canvas").props


# ------------------------------------------------------------------------------------------------- 2. selection

@pytest.mark.anyio
async def test_shared_selection_2d_and_3d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
                                          jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path)
    plan = make_plan()
    door_gid = global_id_for(plan, "opening", "O001")
    wall_gid = global_id_for(plan, "wall", "W001")
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        await user.should_not_see(marker="assistant-panel-ifc")  # one chat card between DXF and IFC, no strip
        canvas = _element(user, "result-canvas")
        viewer = _element(user, "ifc-viewer")
        assert canvas.props["pick-enabled"] is True and viewer.props["pick-enabled"] is True
        assert viewer.props["pick-floor"] == pytest.approx(FLOOR_Z)  # identity frame: model Y of the floor = floor height

        _pick(user, 4.01, 2.48)  # the door in the partition
        await user.should_see("Door O001 (0.90 m)")
        _pick(user, 6.03, 0.02, ctrl=True)  # Ctrl+click adds the south wall
        assert await _chips(user) == ["Door O001 (0.90 m)", "Wall W001"]
        assert len(canvas.props["selection-overlay"]) == 2 and all(len(item["ring"]) >= 4 for item in canvas.props["selection-overlay"])
        assert viewer.props["selected-many"] == [door_gid, wall_gid]

        _pick(user, 6.03, 0.02, ctrl=True)  # Ctrl+click on a selected element toggles it off
        assert await _chips(user) == ["Door O001 (0.90 m)"]
        _pick(user, 6.03, 0.02, ctrl=True)
        _set(user, "assistant-chip-0", False)  # the x of the door chip
        assert await _chips(user) == ["Wall W001"] and viewer.props["selected-many"] == [wall_gid]

        _pick(user, 8.71, 5.69)  # empty space outside the flat: point selection replaces the selection
        assert await _chips(user) == ["Point x 8.71 / y 5.69"]
        assert canvas.props["selection-overlay"] == [{"point": [8.71, 5.69]}] and viewer.props["selected-many"] == []

        with user:  # 3D pick of the door GlobalId selects the plan element (shared selection)
            _emit(viewer, "pick", {"global_id": door_gid, "ifc_class": "IfcDoor", "ctrl": False, "point": [4.0, 1.0, -2.5]})
        assert await _chips(user) == ["Door O001 (0.90 m)"]
        assert len(canvas.props["selection-overlay"]) == 1 and viewer.props["selected-many"] == [door_gid]
        with user:  # Ctrl+click on the floor in 3D adds a point (model frame -> plan metres: y = -z)
            _emit(viewer, "pick", {"global_id": None, "ifc_class": "", "ctrl": True, "point": [6.5, FLOOR_Z, -3.25]})
        assert await _chips(user) == ["Door O001 (0.90 m)", "Point x 6.50 / y 3.25"]
        user.find(marker="assistant-chip-clear").click()
        assert await _chips(user) == [] and viewer.props["selected-many"] == []
        await user.should_see(marker="ifc-props")  # the property panel still works next to the assistant
    _no_error_logs(caplog)


# ------------------------------------------------------------------------------------------------- 3. ask / accept / undo

@pytest.mark.anyio
async def test_prompt_preview_accept_undo_and_reexport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       caplog: pytest.LogCaptureFixture, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path)
    door_gid = global_id_for(make_plan(), "opening", "O001")
    provider = _use(monkeypatch, FakeProvider([call("set_opening", target="O001", width_m=1.0), say("Tür O001 ist jetzt 1.00 m breit.")]))
    service = AssistantService(project, LEVEL_ID)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        _pick(user, 4.01, 2.48)
        await _ask(user, "mach diese Tür 1 m breit", "Tür O001 ist jetzt 1.00 m breit.")
        assert len(provider.requests) == 2
        await user.should_see(marker="assistant-summary")
        await user.should_see("Width 0.90 m → 1.00 m")
        assert "2 000 in / 400 out" in _element(user, "assistant-usage").text
        assert _element(user, "assistant-input").value == ""  # the sent draft is cleared
        canvas = _element(user, "result-canvas")
        viewer = _element(user, "ifc-viewer")
        overlay = canvas.props["preview-overlay"]
        assert [item["change"] for item in overlay] == ["modified"] and overlay[0]["before"] and overlay[0]["after"]
        assert viewer.props["highlight"] == {door_gid: AMBER}
        assert len(viewer.props["boxes"]) == 1 and viewer.props["boxes"][0]["color"] == AMBER
        box = viewer.props["boxes"][0]
        assert box["min"][1] == pytest.approx(FLOOR_Z) and box["max"][1] == pytest.approx(FLOOR_Z + 2.05)  # sill .. head
        assert jobs == []  # nothing exported before accept

        user.find(marker="assistant-accept").click()
        edits = service.edits()
        assert [edit["status"] for edit in edits] == ["active"]
        assert jobs[0] == ("export", LEVEL_ID)
        await _eventually(lambda: ("export_ifc", LEVEL_ID) in jobs, timeout=4.0)  # IFC follows when the DXF export is done
        assert _element(user, "result-canvas") is canvas  # no page rebuild while the next export step runs
        assert service.pending() is None
        await _eventually(lambda: _has(user, "result-canvas") and _element(user, "result-canvas") is not canvas)  # once after the last
        assert load_state(project).plan_variant == "ortho"
        await _eventually(lambda: _has(user, "result-canvas") and _element(user, "result-canvas").props.get("preview-overlay") == [])
        await user.should_see(marker=f"assistant-undo-{edits[0]['id']}")
        assert await _chips(user) == ["Door O001 (1.00 m)"]  # the chip follows the edited plan

        before = len(jobs)
        user.find(marker=f"assistant-undo-{edits[0]['id']}").click()
        assert [edit["status"] for edit in service.edits()] == ["undone"]
        await _eventually(lambda: len(jobs) >= before + 1)
        assert jobs[before] == ("export", LEVEL_ID)
        await user.should_see("undone")
        assert load_state(project).plan_variant == "ortho"  # accept / undo never switch the plan variant
    _no_error_logs(caplog)


@pytest.mark.anyio
async def test_confirmation_dialog_cancel_and_confirm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([call("delete_opening", target="O002"), say("Fenster O002 entfernt.")]))
    service = AssistantService(project, LEVEL_ID)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        await _ask(user, "dieses Fenster gibt es nicht", "Fenster O002 entfernt.")
        overlay = _element(user, "result-canvas").props["preview-overlay"]
        assert [item["change"] for item in overlay] == ["removed"] and overlay[0]["after"] is None
        user.find(marker="assistant-accept").click()
        await user.should_see(marker="assistant-confirm-dialog")
        await user.should_see("Deleted: Window O002")
        user.find(marker="assistant-confirm-cancel").click()
        assert service.edits() == [] and service.pending() is not None and jobs == []
        user.find(marker="assistant-accept").click()
        user.find(marker="assistant-confirm-ok").click()
        assert [edit["status"] for edit in service.edits()] == ["active"]
        assert jobs and jobs[0] == ("export", LEVEL_ID)


@pytest.mark.anyio
async def test_accept_disabled_while_a_job_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Raum heisst jetzt Gast.")]))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Raum heisst jetzt Gast.")
        accept = _element(user, "assistant-accept")
        assert accept.enabled
        running = {"value": True}
        monkeypatch.setattr(DxfJobRunner, "is_running", lambda self: running["value"])
        await _eventually(lambda: not accept.enabled)
        assert _element(user, "assistant-locked").visible
        running["value"] = False
        await _eventually(lambda: not _has(user, "assistant-accept") or _element(user, "assistant-accept").enabled)


# ------------------------------------------------------------------------------------------------- 4. errors + manual

@pytest.mark.anyio
@pytest.mark.parametrize("kind,title", [("unreachable", "Server not reachable"), ("cost_limit", "Cost limit reached")])
async def test_errors_are_readable_and_page_keeps_working(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list,
                                                          caplog: pytest.LogCaptureFixture, kind: str, title: str) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    answers = [TurnResult(error=f"Problem of kind {kind}.", error_kind=kind, reply=f"Problem of kind {kind}."),
               TurnResult(reply="Second answer works.")]
    monkeypatch.setattr(AssistantService, "ask", lambda self, prompt, selection, allow_manual=False: answers.pop(0))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "irgendwas", f"Problem of kind {kind}.")
        await user.should_see(title, marker="assistant-error")
        await user.should_not_see(marker="assistant-accept")
        await _ask(user, "nochmal", "Second answer works.")
        await user.should_not_see(marker="assistant-error")
        _pick(user, 4.01, 2.48)
        assert await _chips(user) == ["Door O001 (0.90 m)"]
    _no_error_logs(caplog)


@pytest.mark.anyio
async def test_model_without_tools_shows_the_no_tools_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([say("x")], tools=False))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "mach was", "cannot call tools")
        await user.should_see("Model without tool support", marker="assistant-error")


@pytest.mark.anyio
async def test_unexpected_exception_never_shows_a_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)

    def broken(self: AssistantService, prompt: str, selection: list, allow_manual: bool = False) -> TurnResult:
        raise RuntimeError(f"secret {DUMMY_KEY} in a traceback")

    monkeypatch.setattr(AssistantService, "ask", broken)
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "kaputt", "RuntimeError")
        error = _element(user, "assistant-error")
        assert DUMMY_KEY not in error.text if hasattr(error, "text") else True
        await user.should_not_see(DUMMY_KEY)


@pytest.mark.anyio
async def test_asks_manual_reasks_with_allow_manual_and_same_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                       jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    seen: list[tuple[str, list[dict[str, Any]], bool]] = []

    def ask(self: AssistantService, prompt: str, selection: list, allow_manual: bool = False) -> TurnResult:
        seen.append((prompt, [dict(item) for item in selection], allow_manual))
        if not allow_manual:
            return TurnResult(reply="Dort ist nichts zu sehen.", asks_manual=True)
        return TurnResult(reply="Manuell gesetzt.")

    monkeypatch.setattr(AssistantService, "ask", ask)
    async with _dxf_user(project, monkeypatch) as user:
        _pick(user, 7.02, 4.01)
        await _ask(user, "setz hier einen Stuhl", "Dort ist nichts zu sehen.")
        await user.should_see(marker="assistant-manual")
        user.find(marker="assistant-manual-yes").click()
        await user.should_see("Manuell gesetzt.", retries=60)
    assert [(prompt, allow) for prompt, _selection, allow in seen] == [("setz hier einen Stuhl", False), ("setz hier einen Stuhl", True)]
    assert seen[0][1] == seen[1][1] and seen[0][1][0]["id"] == "R002"


# ------------------------------------------------------------------------------------------------- 5. history / chat

@pytest.mark.anyio
async def test_stale_edit_is_listed_with_reason_and_removable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([call("rename_room", target="R001", name="Wohnzimmer"), say("ok")]))
    service = AssistantService(project, LEVEL_ID)
    service.ask("umbenennen", [])
    edit_id = service.accept()
    edits_file = assistant_dir(project, LEVEL_ID) / "edits.json"
    data = json.loads(edits_file.read_text(encoding="utf-8"))
    data["edits"][0]["status"] = "stale"
    data["edits"][0]["stale_reason"] = "Room R001 no longer exists."
    edits_file.write_text(json.dumps(data), encoding="utf-8")
    (assistant_dir(project, LEVEL_ID) / "pipeline_report.json").write_text(json.dumps({
        "ifc_export": {"level": LEVEL_ID, "step": "ifc_export", "applied": [], "stale": [], "changed": False,
                       "error": "ValueError: broken plan"}}), encoding="utf-8")
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see("No longer applies: Room R001 no longer exists.", marker=f"assistant-stale-{edit_id}")
        await user.should_see("ValueError: broken plan", marker="assistant-pipeline-error")
        assert _element(user, f"assistant-undo-{edit_id}").text == "Remove"
        user.find(marker=f"assistant-undo-{edit_id}").click()
        assert [edit["status"] for edit in service.edits()] == ["undone"]


@pytest.mark.anyio
async def test_chat_history_shown_and_cleared_without_touching_edits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                     jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Erledigt: Gast.")]))
    service = AssistantService(project, LEVEL_ID)
    service.ask("Raum heisst Gast", [])
    service.accept()
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see("Raum heisst Gast", marker="assistant-chat")
        await user.should_see("Erledigt: Gast.", marker="assistant-chat")
        user.find(marker="assistant-clear-chat").click()
        await user.should_not_see("Erledigt: Gast.")
    assert service.chat() == [] and [edit["status"] for edit in service.edits()] == ["active"]


@pytest.mark.anyio
async def test_german_texts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, german_ui: None, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see("KI-Assistent")
        _pick(user, 8.71, 5.69)
        assert await _chips(user) == ["Punkt x 8.71 / y 5.69"]
        await user.should_see("Senden")


# ------------------------------------------------------------------------------------------------- 6. helpers / frames

def _glb_bounds(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + length])
    bounds = {}
    for node in document["nodes"]:
        lows, highs = [], []
        for primitive in document["meshes"][node["mesh"]]["primitives"]:
            accessor = document["accessors"][primitive["attributes"]["POSITION"]]
            lows.append(accessor["min"])
            highs.append(accessor["max"])
        bounds[node["name"]] = ([min(v[i] for v in lows) for i in range(3)], [max(v[i] for v in highs) for i in range(3)])
    return bounds


def test_plan_to_model_frame_matches_the_exported_glb(tmp_path: Path) -> None:
    """The panel's plan <-> model conversion reproduces where the IFC export put the table and the door."""
    project = _project(tmp_path)
    props = json.loads((project / "products" / "bim" / LEVEL_ID / "props.json").read_text(encoding="utf-8"))
    offset = panel_module._origin_offset(props)
    bounds = _glb_bounds(project / "products" / "bim" / LEVEL_ID / "model.glb")
    plan = make_plan()
    table = plan.furniture[0]
    expected = panel_module.model_box(table.footprint, table.z_min, table.z_max, offset, GREEN)
    low, high = bounds[global_id_for(plan, "furniture", "F001")]
    assert low == pytest.approx(expected["min"], abs=2e-3) and high == pytest.approx(expected["max"], abs=2e-3)
    x0, x1, y0, y1 = TABLE_BOX
    centre = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
    model = panel_module.plan_to_model(centre[0], centre[1], FLOOR_Z, offset)
    assert panel_module.model_to_plan(model, offset) == pytest.approx(centre)
    door_low, door_high = bounds[global_id_for(plan, "opening", "O001")]
    assert panel_module.model_to_plan([(door_low[0] + door_high[0]) / 2, 0.0, (door_low[2] + door_high[2]) / 2], offset) == pytest.approx(
        [4.0, 2.5], abs=0.02)


def test_preview_entries_carry_height_ranges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = make_project(tmp_path, with_cloud=False)
    _use(monkeypatch, FakeProvider([call("set_opening", target="O001", width_m=1.0), say("ok")]))
    service = AssistantService(project, LEVEL_ID)
    service.discard()
    service.ask("breiter", [])
    moved = service.preview()["moved"][0]
    assert moved["z_min"] == pytest.approx(FLOOR_Z) and moved["z_max"] == pytest.approx(FLOOR_Z + 2.05)
    service.discard()
    _use(monkeypatch, FakeProvider([call("rename_room", target="R001", name="Wohnzimmer"), say("ok")]))
    service.ask("umbenennen", [])
    room = service.preview()["moved"][0]
    assert room["kind"] == "room" and room["z_min"] is None and room["z_max"] is None
    service.discard()


def test_outlines_and_refresh_selections(tmp_path: Path) -> None:
    _enable_assistant()
    project = make_project(tmp_path, with_cloud=False)
    service = AssistantService(project, LEVEL_ID)
    door = service.hit_test(4.0, 2.5)
    point = {"type": "point", "x": 8.7, "y": 5.7}
    gone = {"type": "element", "kind": "furniture", "id": "F999", "label": "x"}
    outlines = service.outlines_for([door, point, gone])
    assert outlines[0]["id"] == "O001" and len(outlines[0]["ring"]) == 4 and outlines[1] == {"point": [8.7, 5.7]} and len(outlines) == 2
    assert service.refresh_selections([door, point, gone]) == [door, point]


def test_usage_and_change_texts() -> None:
    result = TurnResult(provider="openai", cost_usd=0.0042)
    result.usage.input_tokens, result.usage.output_tokens = 1234, 210
    assert panel_module.usage_text(result) == "1 234 in / 210 out · ≈ $0.004"
    local = TurnResult(provider="ollama")
    local.usage.input_tokens, local.usage.output_tokens = 900, 50
    assert panel_module.usage_text(local) == "900 in / 50 out · local"
    assert panel_module.usage_text(TurnResult()) is None
    unknown = TurnResult(provider="openrouter")
    unknown.usage.input_tokens = 10
    assert panel_module.usage_text(unknown).endswith("cost unknown")
    entry = {"element_id": "O001", "kind": "opening", "change": "modified",
             "before": {"type": "door", "width_m": 0.9, "center": [4.0, 2.5]}, "after": {"type": "door", "width_m": 1.0, "center": [4.0, 2.5]}}
    assert panel_module.change_title(entry) == "Door O001"
    assert panel_module.change_details(entry) == ["Width 0.90 m → 1.00 m"]
    table = {"element_id": "F002", "kind": "furniture", "change": "added", "before": None,
             "after": {"type": "furniture", "label": "Stuhl", "size_m": [0.45, 0.5], "center": [5.5, 2.3]}}
    assert panel_module.change_title(table) == "Furniture F002 (Stuhl)"
    assert panel_module.change_details(table) == ["Size 0.45 / 0.50 m", "Position (5.50 / 2.30)"]
    assert panel_module.error_title("no_tools") == "Model without tool support"
    assert panel_module.error_title("something_new") == "Not possible"



# ------------------------------------------------------------------------------------------------- 7. follow-up review

@pytest.mark.anyio
async def test_non_ortho_view_shows_a_hint_instead_of_the_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    project = _project(tmp_path, variant="pointcloud")
    async with _dxf_user(project, monkeypatch) as user:  # assistant off: neither panel nor hint
        await user.should_see(marker="result-canvas")
        await user.should_not_see(marker="assistant-variant-hint")
        await user.should_not_see(marker="assistant-panel")
    _enable_assistant()
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see("The AI assistant works on the ortho plan", marker="assistant-variant-hint")
        await user.should_not_see(marker="assistant-panel")
        await user.should_not_see(marker="assistant-panel-ifc")
        assert "pick-enabled" not in _element(user, "result-canvas").props
        assert "pick-enabled" not in _element(user, "ifc-viewer").props
        assert load_state(project).plan_variant == "pointcloud"  # nothing switched by just opening the page
        user.find(marker="assistant-variant-switch").click()
        await user.should_see(marker="assistant-panel")
        await user.should_not_see(marker="assistant-variant-hint")
        assert load_state(project).plan_variant == "ortho"
        assert _element(user, "result-canvas").props["pick-enabled"] is True


@pytest.mark.anyio
async def test_outdoor_project_shows_the_panel_only_with_its_ortho_plan_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                            jobs: list) -> None:
    from samcloud.bim.plan_model import save_plan_model
    from samcloud.ortho.plan import clean_plan_cache_dir
    from test_dxf_gui import _result_project

    project = _result_project(tmp_path, "OutdoorAssistant")
    state = load_state(project)
    state.plan_variant = "ortho"
    save_state(project, state)
    _enable_assistant()
    async with _dxf_user(project, monkeypatch) as user:  # no ortho plan_model for the site: no panel, no hint
        await user.should_see(marker="result-canvas")
        await user.should_not_see(marker="assistant-panel")
        await user.should_not_see(marker="assistant-variant-hint")
        assert "pick-enabled" not in _element(user, "result-canvas").props
    save_plan_model(make_plan(), clean_plan_cache_dir(project, None))
    AssistantService(project, "site").discard()
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        assert _element(user, "result-canvas").props["pick-enabled"] is True


@pytest.mark.anyio
async def test_export_start_error_clears_the_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    calls = _record_jobs(monkeypatch, start_error="runner refused")
    _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Raum heisst jetzt Gast.")]))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Raum heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()
        await user.should_see("The export could not be started: runner refused")
        await asyncio.sleep(1.3)  # a poll later nothing else is started
        assert calls == []
        await user.should_not_see(marker="assistant-export-status")  # hidden: no queued step left
    assert [edit["status"] for edit in AssistantService(project, LEVEL_ID).edits()] == ["active"]


@pytest.mark.anyio
async def test_export_job_not_done_clears_the_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    calls = _record_jobs(monkeypatch, end_state="failed")
    _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Raum heisst jetzt Gast.")]))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Raum heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()
        assert calls == [("export", LEVEL_ID)]
        await _eventually(lambda: user.notify.contains("did not finish"))
        await asyncio.sleep(1.3)
        assert calls == [("export", LEVEL_ID)]  # export_ifc is never started after a failed DXF export
        await user.should_not_see(marker="assistant-export-status")


@pytest.mark.anyio
@pytest.mark.parametrize("key,text", [("assistant.history.outdated", "The plan changed since the proposal was made."),
                                      ("assistant.history.ambiguous", "no longer matches the recomputed plan unambiguously")])
async def test_outdated_or_ambiguous_accept_discards_the_proposal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list,
                                                                  key: str, text: str) -> None:
    from samcloud.assistant.history import HistoryError

    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Raum heisst jetzt Gast.")]))

    def refuse(self: AssistantService, *, confirmed: bool = False) -> str:
        raise HistoryError("refused", key=key)

    monkeypatch.setattr(AssistantService, "accept", refuse)
    service = AssistantService(project, LEVEL_ID)
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Raum heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()
        await user.should_see(text, marker="assistant-accept-notes")
        await user.should_see("Please ask again.", marker="assistant-accept-notes")
        await user.should_not_see(marker="assistant-accept")
        assert service.pending() is None and jobs == [] and service.edits() == []
        assert _element(user, "result-canvas").props["preview-overlay"] == []


@pytest.mark.anyio
async def test_rebase_notes_are_shown_after_accept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    from samcloud.assistant.history import AcceptReport

    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Raum heisst jetzt Gast.")]))

    def rebased_accept(self: AssistantService, *, confirmed: bool = False) -> str:
        self.last_accept = AcceptReport("E0001", {"R002": "R001"}, ["Applied to the recomputed plan."])
        self.discard()
        return "E0001"

    monkeypatch.setattr(AssistantService, "accept", rebased_accept)
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Raum heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()
        await user.should_see("recomputed plan (R002 → R001)", marker="assistant-accept-notes")
        await user.should_see("Applied to the recomputed plan.", marker="assistant-accept-notes")
        assert jobs and jobs[0] == ("export", LEVEL_ID)


@pytest.mark.anyio
async def test_pipeline_report_error_is_shown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    folder = assistant_dir(project, LEVEL_ID)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pipeline_report.json").write_text(json.dumps({
        "clean_plan": {"level": LEVEL_ID, "step": "clean_plan", "applied": [], "stale": [], "changed": False, "error": None},
        "dxf_export": {"level": LEVEL_ID, "step": "dxf_export", "applied": [], "stale": [], "changed": False,
                       "error": "KeyError: 'W009'"}}), encoding="utf-8")
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see("Step dxf_export: the changes could not be re-applied (KeyError: 'W009').",
                              marker="assistant-pipeline-error")
        assert len(user.find(marker="assistant-pipeline-error").elements) == 1  # steps without an error stay silent


@pytest.mark.anyio
async def test_unknown_3d_pick_clears_selection_and_properties(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path)
    door_gid = global_id_for(make_plan(), "opening", "O001")
    async with _dxf_user(project, monkeypatch) as user:
        viewer = _element(user, "ifc-viewer")
        with user:
            _emit(viewer, "pick", {"global_id": door_gid, "ifc_class": "IfcDoor", "ctrl": False, "point": [4.0, 1.0, -2.5]})
        assert await _chips(user) == ["Door O001 (0.90 m)"] and viewer.props["selected-many"] == [door_gid]
        await user.should_not_see("No properties found.")
        with user:  # an id that is neither in the plan nor in props.json, and no hit point
            _emit(viewer, "pick", {"global_id": "0000000000000000000000", "ifc_class": "", "ctrl": False, "point": None})
        assert await _chips(user) == [] and viewer.props["selected-many"] == []
        await user.should_see("No properties found.", marker="ifc-props")  # no stale door properties
        assert user.notify.contains("not part of the plan")


@pytest.mark.anyio
async def test_canvas_pick_error_notifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)

    def broken(self: AssistantService, x: float, y: float) -> dict[str, Any]:
        raise ValueError("broken plan")

    monkeypatch.setattr(AssistantService, "hit_test", broken)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        _pick(user, 4.01, 2.48)
        await user.should_see("The clicked position could not be evaluated.")
        assert await _chips(user) == []


# ------------------------------------------------------------------------------------------------- 8. browser review fixes

@pytest.mark.anyio
async def test_pick_at_a_drawn_label_selects_its_room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    """The canvas world frame is the payload frame (non-zero frame origin, background bounds not starting at 0): a pick
    at the position where the payload draws a room label selects that room, and its outline lands on the room."""
    from samcloud.app.dxf_controller import DxfPageController

    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    payload = DxfPageController(project).plan_payload(LEVEL_ID, "ortho")
    assert payload["background_bounds"][0] != 0 and payload["background_bounds"][1] != 0
    label = next(e for e in payload["layers"]["A-ANNO"]["entities"] if str(e.get("text", "")).startswith("Schlafen"))
    room = next(r for r in make_plan().rooms if r.name == "Schlafen")
    async with _dxf_user(project, monkeypatch) as user:
        _pick(user, float(label["point"][0]), float(label["point"][1]))
        assert await _chips(user) == ["Room Schlafen"]
        ring = _element(user, "result-canvas").props["selection-overlay"][0]["ring"]
        assert sorted(map(tuple, ring)) == pytest.approx(sorted(map(tuple, room.polygon["exterior"])), abs=1e-3)


@pytest.mark.anyio
async def test_pending_proposal_is_labelled_as_not_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Vorschlag: Raum R002 heisst nach Übernehmen Gast.")]))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Vorschlag: Raum R002 heisst nach Übernehmen Gast.")
        await user.should_see("Proposal – not accepted yet", marker="assistant-pending")
        await user.should_see("Proposal – not accepted yet", marker="assistant-pending-chat")
        user.find(marker="assistant-accept").click()
        await user.should_not_see(marker="assistant-pending")
        await user.should_not_see(marker="assistant-pending-chat")


def test_system_prompts_ask_for_proposal_wording() -> None:
    from samcloud.assistant.prompts import system_prompt

    assert "Vorschlag:" in system_prompt("de") and "Übernehmen wird" in system_prompt("de") and "nie als erledigt" in system_prompt("de")
    assert "Proposal:" in system_prompt("en") and "never as done" in system_prompt("en")


# ------------------------------------------------------------------------------------------------- 9. turns without a proposal

@pytest.mark.anyio
async def test_turn_without_proposal_says_nothing_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([say("Der Raum R006 ist ein Büro mit 13.1 m².")]))  # no edit tool called
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Was ist das für ein Raum?", "Der Raum R006 ist ein Büro mit 13.1 m².")
        await user.should_see("No change proposed – the plan was not changed.", marker="assistant-no-proposal")
        await user.should_not_see(marker="assistant-no-proposal-claim")
        await user.should_not_see(marker="assistant-accept")


@pytest.mark.anyio
async def test_reply_claiming_a_change_without_proposal_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list,
                                                                   german_ui: None) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    reply = "Ich habe den Raum R006 in 'Arbeitszimmer' umbenannt. Vorschlag: übernehmen."
    _use(monkeypatch, FakeProvider([say(reply)]))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Benenne diesen Raum in Arbeitszimmer um.", reply)  # the model's text is shown unchanged
        await user.should_see("Kein Änderungsvorschlag – am Plan wurde nichts geändert.", marker="assistant-no-proposal")
        await user.should_see("Die KI hat keine Änderung ausgeführt. Bitte erneut fragen oder das Element genauer auswählen.",
                              marker="assistant-no-proposal-claim")
        await user.should_not_see(marker="assistant-accept")
    assert AssistantService(project, LEVEL_ID).chat()[-1]["text"] == reply


@pytest.mark.anyio
async def test_real_proposal_shows_no_nothing_changed_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Vorschlag: R002 wird nach Übernehmen umbenannt.")]))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Vorschlag: R002 wird nach Übernehmen umbenannt.")
        await user.should_see(marker="assistant-accept")
        await user.should_not_see(marker="assistant-no-proposal")
        await user.should_not_see(marker="assistant-no-proposal-claim")


def test_claims_change_detects_de_and_en_claims() -> None:
    assert panel_module.claims_change("Ich habe den Raum umbenannt.") and panel_module.claims_change("I renamed the room.")
    assert panel_module.claims_change("Die Tür wurde Geändert") and panel_module.claims_change("Window O002 was deleted")
    assert not panel_module.claims_change("Der Raum ist ein Büro.") and not panel_module.claims_change("Which door do you mean?")
    assert not panel_module.claims_change(None)

