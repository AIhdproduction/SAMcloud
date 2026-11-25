"""AI assistant service API for the UI: availability, ask / preview / accept / undo, confirmation, chat.json, no key leaks."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from assistant_synthetic import EMPTY_BOX, LEVEL_ID, make_project, renumbered_recompute
from samcloud.assistant import agent as agent_module
from samcloud.assistant import credentials as ai_creds
from samcloud.assistant import history as history_module
from samcloud.assistant import settings as ai_settings
from samcloud.assistant.history import HistoryError
from samcloud.assistant.paths import assistant_dir, plan_model_path
from samcloud.assistant.selection import global_id_for
from samcloud.assistant.service import AssistantService
from samcloud.bim.plan_model import load_plan_model, save_plan_model
from test_assistant_agent import FakeProvider, call, say

DUMMY_KEY = "sk-test-DUMMY-0000"
DOOR = {"type": "element", "kind": "opening", "id": "O001", "label": "door O001", "subkind": "door", "width_m": 0.8}


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Private user config with an enabled OpenAI assistant (dummy key, no cost limits)."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    settings = ai_settings.AssistantSettings(enabled=True, provider="openai", model="fake-model", cost_limit_request_usd=0.0,
                                             cost_limit_project_day_usd=0.0, privacy_ack_cloud=True,
                                             full_control=False)  # proposal + accept (full control: test_assistant_full_control*)
    ai_settings.save_settings(settings)
    ai_creds.save_ai_credentials({"openai_api_key": DUMMY_KEY})
    return tmp_path


@pytest.fixture
def project(tmp_path: Path, config: Path) -> Path:
    return make_project(tmp_path)


def use(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> FakeProvider:
    monkeypatch.setattr(agent_module, "get_provider", lambda name, settings, creds: provider)
    return provider


@pytest.fixture
def service(project: Path) -> AssistantService:
    svc = AssistantService(project, LEVEL_ID)
    svc.discard()
    yield svc
    svc.discard()


def test_availability_reasons(tmp_path: Path, config: Path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    svc = AssistantService(project, LEVEL_ID)
    assert svc.is_available() == (True, None)
    assert AssistantService(tmp_path / "nothing", LEVEL_ID).is_available() == (False, "assistant.agent.no_plan")
    settings = ai_settings.load_settings()
    settings.model = ""
    ai_settings.save_settings(settings)
    assert svc.is_available() == (False, "assistant.agent.no_model")
    settings.enabled = False
    ai_settings.save_settings(settings)
    assert svc.is_available() == (False, "assistant.agent.disabled")


def test_disabled_assistant_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    project = make_project(tmp_path, with_cloud=False)
    provider = use(monkeypatch, FakeProvider([say("x")]))
    svc = AssistantService(project, LEVEL_ID)
    result = svc.ask("mach die Tür breiter", [])
    assert result.error_kind == "unavailable" and result.proposal is None
    assert provider.requests == []
    assert not assistant_dir(project, LEVEL_ID).exists()
    assert not (project / "products" / "assistant").exists()


def test_door_width_accept_and_undo(project: Path, service: AssistantService, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_file = plan_model_path(project, LEVEL_ID)
    plan = load_plan_model(plan_file)
    next(o for o in plan.openings if o.id == "O001").width_m = 0.8
    save_plan_model(plan, plan_file.parent)
    original = plan_file.read_bytes()
    provider = use(monkeypatch, FakeProvider([call("set_opening", target="O001", width_m=0.9), say("Tür O001: 0.80 m -> 0.90 m.")]))

    result = service.ask("mach diese Tür 90 cm breit", [DOOR])
    assert result.error is None and result.proposal is not None and result.qa is not None and result.qa.ok
    assert [(e["element_id"], e["before"]["width_m"], e["after"]["width_m"]) for e in result.summary] == [("O001", 0.8, 0.9)]
    assert service.pending() is result
    assert plan_file.read_bytes() == original  # nothing written before accept
    assert len(provider.requests) == 2

    preview = service.preview()
    assert preview["added"] == [] and preview["removed"] == []
    moved = preview["moved"][0]
    assert moved["id"] == "O001" and moved["change"] == "modified"
    assert moved["global_id"] == global_id_for(plan, "opening", "O001")
    assert len(moved["before"]) == 4 and len(moved["after"]) == 4 and moved["before"] != moved["after"]

    edit_id = service.accept()
    assert service.pending() is None and service.preview() == {"added": [], "removed": [], "moved": []}
    edited = load_plan_model(plan_file)
    assert next(o for o in edited.openings if o.id == "O001").width_m == pytest.approx(0.9)
    edits = service.edits()
    assert [e["id"] for e in edits] == [edit_id] and edits[0]["status"] == "active"
    assert edits[0]["prompt"] == "mach diese Tür 90 cm breit" and edits[0]["provider"] == "openai"
    assert edits[0]["summary"] == [{"element_id": "O001", "kind": "opening", "change": "modified"}]
    assert edits[0]["tools"] == ["set_opening"] and "signatures" not in edits[0]

    report = service.undo(edit_id)
    assert report["edit_id"] == edit_id and report["changed"] is True and report["stale"] == []
    restored = load_plan_model(plan_file)
    assert next(o for o in restored.openings if o.id == "O001").width_m == pytest.approx(0.8)
    assert [e["status"] for e in service.edits()] == ["undone"]
    assert service.stale() == []
    with pytest.raises(HistoryError) as raised:
        service.undo(edit_id)
    assert raised.value.key == "assistant.history.not_active"

    chat = service.chat()
    assert [m["role"] for m in chat] == ["user", "assistant"]
    assert chat[0]["meta"]["selection"] == ["Door O001 (0.80 m)"]
    assert chat[1]["meta"]["tools"][0]["tool"] == "set_opening" and chat[1]["meta"]["changes"] == 1


def test_confirmation_for_deletions(project: Path, service: AssistantService, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeProvider([call("delete_opening", target="O002"), say("Fenster gelöscht.")]))
    result = service.ask("dieses Fenster gibt es nicht", [{"type": "element", "kind": "opening", "id": "O002"}])
    assert result.needs_confirmation is True
    assert service.preview()["removed"][0]["id"] == "O002" and service.preview()["removed"][0]["after"] is None
    with pytest.raises(HistoryError) as raised:
        service.accept()
    assert raised.value.key == "assistant.agent.confirm_required"
    assert not history_module.History(project, LEVEL_ID).exists()
    service.accept(confirmed=True)
    assert all(o.id != "O002" for o in load_plan_model(plan_model_path(project, LEVEL_ID)).openings)


def test_accept_without_pending_and_discard(project: Path, service: AssistantService, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(HistoryError) as raised:
        service.accept()
    assert raised.value.key == "assistant.agent.no_pending"
    use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Umbenannt.")]))
    service.ask("Raum heisst Gast", [])
    assert AssistantService(project, LEVEL_ID).pending() is not None  # shared per project level
    service.discard()
    assert service.pending() is None
    with pytest.raises(HistoryError):
        service.accept()


def test_manual_add_preview_is_green(project: Path, service: AssistantService, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeProvider([call("add_furniture", box=EMPTY_BOX, kind="chair", manual=True), say("Gesetzt.")]))
    result = service.ask("trotzdem einen Stuhl setzen", [{"type": "point", "x": 7.0, "y": 4.0}], allow_manual=True)
    assert result.proposal is not None
    added = service.preview()["added"]
    assert len(added) == 1 and added[0]["kind"] == "furniture" and added[0]["before"] is None and len(added[0]["after"]) == 4


def test_accept_and_undo_refuse_while_the_history_is_locked(project: Path, service: AssistantService,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeProvider([call("rename_room", target="R001", name="Wohnzimmer"), say("ok")]))
    service.ask("umbenennen", [])
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with history_module._locked(assistant_dir(project, LEVEL_ID)):
            held.set()
            release.wait(10)

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert held.wait(10)
        with pytest.raises(HistoryError) as raised:
            service.accept()
        assert raised.value.key == "assistant.history.locked"
        with pytest.raises(HistoryError) as raised:
            service.undo("E-unknown")
        assert raised.value.key == "assistant.history.locked"
    finally:
        release.set()
        worker.join(10)
    assert service.pending() is not None
    assert service.accept()


def test_selection_helpers(project: Path, service: AssistantService) -> None:
    plan = service.plan()
    door = service.hit_test(4.0, 2.5)
    assert door["kind"] == "opening" and door["id"] == "O001"
    assert service.chip_label(door) == "Door O001 (0.90 m)"
    assert service.hit_test(6.0, 3.0)["id"] == "R002"
    gid = global_id_for(plan, "opening", "O001")
    picked = service.selection_from_global_id(gid, None)
    assert picked["type"] == "element" and picked["id"] == "O001" and picked["subkind"] == "door"
    assert service.selection_from_global_id("not-a-guid", [1.2345, 2.0]) == {"type": "point", "x": 1.234, "y": 2.0}
    assert service.selection_from_global_id("not-a-guid", None)["type"] == "unknown"
    ids = service.global_ids_for([door, {"type": "point", "x": 1, "y": 1}, {"type": "element", "kind": "wall", "id": "W005"}])
    assert ids == [gid, global_id_for(plan, "wall", "W005")]


def test_keys_never_reach_chat_or_edits(project: Path, service: AssistantService, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = use(monkeypatch, FakeProvider([
        call("rename_room", target="R002", name="Gast"),
        say(f"Erledigt. (Key {DUMMY_KEY} bitte nie teilen.)"),
    ]))
    result = service.ask(f"Raum umbenennen, mein Key ist {DUMMY_KEY}", [])
    assert result.proposal is not None
    assert all(DUMMY_KEY not in m.content for request in provider.requests for m in request)
    service.accept()
    directory = assistant_dir(project, LEVEL_ID)
    chat_file, edits_file = directory / "chat.json", directory / "edits.json"
    assert chat_file.is_file() and edits_file.is_file()
    for path in directory.rglob("*"):
        if path.is_file():
            assert DUMMY_KEY.encode() not in path.read_bytes(), path.name
    chat = json.loads(chat_file.read_text(encoding="utf-8"))
    assert "***" in chat["messages"][0]["text"]
    # chat.json is never part of what a share packs (products/dxf, products/bim)
    relative = chat_file.relative_to(project).parts
    assert relative[:2] == ("products", "assistant") and "dxf" not in relative and "bim" not in relative
    service.clear_chat()
    assert service.chat() == [] and chat_file.is_file()


def test_failed_turn_keeps_previous_pending_cleared(project: Path, service: AssistantService,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("ok")]))
    service.ask("umbenennen", [])
    assert service.pending() is not None
    use(monkeypatch, FakeProvider([say("Ich weiss nicht, was gemeint ist.")]))
    result = service.ask("hmm", [])
    assert result.proposal is None and service.pending() is None
    assert [m["role"] for m in service.chat()] == ["user", "assistant", "user", "assistant"]


def test_busy_while_another_question_runs(project: Path, service: AssistantService, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("ok")]))
    first = service.ask("umbenennen", [])
    assert service.pending() is first
    provider = use(monkeypatch, FakeProvider([say("x")]))
    lock = service._ask_lock()
    assert lock.acquire(blocking=False)
    try:
        result = AssistantService(project, LEVEL_ID).ask("noch etwas", [])
    finally:
        lock.release()
    assert result.error_kind == "busy" and result.proposal is None
    assert provider.requests == [] and service.pending() is first
    assert [m["role"] for m in service.chat()] == ["user", "assistant"]


def test_projects_do_not_share_pending(tmp_path: Path, config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = AssistantService(make_project(tmp_path / "a", with_cloud=False), LEVEL_ID)
    second = AssistantService(make_project(tmp_path / "b", with_cloud=False), LEVEL_ID)
    try:
        use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("ok")]))
        assert first.ask("umbenennen", []).proposal is not None
        assert first.pending() is not None and second.pending() is None and second.preview()["moved"] == []
        with pytest.raises(HistoryError):
            second.accept()
        use(monkeypatch, FakeProvider([call("rename_room", target="R001", name="Stube"), say("ok")]))
        assert second.ask("umbenennen", []).proposal is not None
        assert first.pending() is not second.pending()
        first.accept()
        assert second.pending() is not None
        assert [r.name for r in second.plan().rooms] == ["Wohnen", "Schlafen"]
    finally:
        first.discard()
        second.discard()


def test_pending_after_a_recompute_is_rebased_by_signature(project: Path, service: AssistantService,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("ok")]))
    assert service.ask("umbenennen", []).proposal is not None
    plan_file = plan_model_path(project, LEVEL_ID)
    # the pipeline recomputes the plan while the proposal waits: new ids (R002 -> R001), millimetre shifts
    save_plan_model(renumbered_recompute(load_plan_model(plan_file)), plan_file.parent)
    edit_id = service.accept()
    assert edit_id and service.pending() is None
    assert service.last_accept.rebased == {"R002": "R001"}
    assert service.last_accept_report.edit_id == edit_id
    assert any("R002 -> R001" in note for note in service.last_accept_report.notes)
    rooms = {r.id: r.name for r in load_plan_model(plan_file).rooms}
    assert rooms == {"R001": "Gast", "R002": "Wohnen"}


def test_pending_whose_target_vanished_is_outdated(project: Path, service: AssistantService,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("ok")]))
    assert service.ask("umbenennen", []).proposal is not None
    plan_file = plan_model_path(project, LEVEL_ID)
    plan = load_plan_model(plan_file)
    plan.rooms = [r for r in plan.rooms if r.id != "R002"]
    save_plan_model(plan, plan_file.parent)
    with pytest.raises(HistoryError) as raised:
        service.accept()
    assert raised.value.key == "assistant.history.outdated"
    assert service.pending() is not None  # kept; the UI discards it and asks again
    assert [r.name for r in load_plan_model(plan_file).rooms] == ["Wohnen"]


def test_lock_error_from_history_is_authoritative(project: Path, service: AssistantService,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """The fast pre-check can miss a writer that starts right after it; History.accept's own lock error wins."""
    from samcloud.assistant import service as service_module

    use(monkeypatch, FakeProvider([call("rename_room", target="R001", name="Stube"), say("ok")]))
    service.ask("umbenennen", [])
    monkeypatch.setattr(service_module, "history_locked", lambda project, level_id: False)
    monkeypatch.setattr(history_module, "LOCK_TIMEOUT_S", 0.3)
    directory = assistant_dir(project, LEVEL_ID)
    directory.mkdir(parents=True, exist_ok=True)
    # another process holds the OS file lock (simulated by a second handle on the lock file)
    with open(directory / history_module.LOCK_FILENAME, "a+b") as handle:
        assert history_module._try_lock(handle)
        try:
            with pytest.raises(HistoryError) as raised:
                service.accept()
            assert raised.value.key == "assistant.history.locked"
        finally:
            history_module._unlock(handle)
    assert service.pending() is not None
    assert service.accept()
