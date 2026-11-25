"""Full control in the UI: the settings card switch (provider defaults, explicit choice kept across provider changes) and
the /dxf panel (applied directly: no accept buttons, "Applied -- Undo" line, automatic exports, undo).  NiceGUI user
simulation, fake providers, no network."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from assistant_synthetic import LEVEL_ID
from samcloud.assistant.paths import plan_model_path
from samcloud.assistant.service import AssistantService
from samcloud.bim.plan_model import load_plan_model
from test_assistant_agent import FakeProvider, call, say
from test_assistant_panel import _ask, _dxf_user, _element, _enable_assistant, _eventually, _has, _project, _use, jobs  # noqa: F401
from test_assistant_settings_ui import _config_home, _markers_present, _settings_route
from test_apriltag_gui import _page_user, _set

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


# --------------------------------------------------------------------------------------------------- settings card

@pytest.mark.anyio
async def test_settings_switch_follows_the_provider_until_chosen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _config_home(tmp_path, monkeypatch)
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        switch = _element(user, "assistant-full-control")
        assert switch.value is False  # Ollama: off by default
        assert "Ollama" in _element(user, "assistant-full-control-default").text
        assert ": off." in _element(user, "assistant-full-control-default").text
        assert _element(user, "assistant-full-control-hint").text
        _set(user, "assistant-provider", "openai")
        await asyncio.sleep(0.1)
        assert switch.value is True  # cloud provider: on by default
        assert ": on." in _element(user, "assistant-full-control-default").text
        assert _markers_present(user, "assistant-full-control-reset") == 0
        _set(user, "assistant-full-control", False)  # explicit choice
        await asyncio.sleep(0.1)
        assert _markers_present(user, "assistant-full-control-reset") == 1
        _set(user, "assistant-provider", "anthropic")
        await asyncio.sleep(0.1)
        assert switch.value is False  # the explicit choice is not overwritten by the provider change
        user.find(marker="assistant-save").click()
        await asyncio.sleep(0.1)
    saved = yaml.safe_load((home / "ai_settings.yaml").read_text(encoding="utf-8"))
    assert saved["full_control"] is False and saved["provider"] == "anthropic"

    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        switch = _element(user, "assistant-full-control")
        assert switch.value is False
        _set(user, "assistant-provider", "ollama")
        await asyncio.sleep(0.1)
        assert switch.value is False
        user.find(marker="assistant-full-control-reset").click()  # back to the provider default
        await asyncio.sleep(0.1)
        assert switch.value is False and _markers_present(user, "assistant-full-control-reset") == 0
        _set(user, "assistant-provider", "openrouter")
        await asyncio.sleep(0.1)
        assert switch.value is True
        user.find(marker="assistant-save").click()
        await asyncio.sleep(0.1)
    assert yaml.safe_load((home / "ai_settings.yaml").read_text(encoding="utf-8"))["full_control"] is None


# ------------------------------------------------------------------------------------------------------------ panel

@pytest.mark.anyio
async def test_panel_applies_directly_exports_and_undoes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:  # noqa: F811
    _enable_assistant(full_control=None)  # OpenAI: full control by default
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Raum R002 wird zu Gast.")]))
    service = AssistantService(project, LEVEL_ID)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-full-mode")
        await _ask(user, "Raum umbenennen", "Raum R002 wird zu Gast.")
        await user.should_see(marker="assistant-applied")
        await user.should_see("Applied: 1 elements")
        assert not _has(user, "assistant-accept") and not _has(user, "assistant-discard")
        edits = service.edits()
        assert [edit["status"] for edit in edits] == ["active"]
        assert load_plan_model(plan_model_path(project, LEVEL_ID)).rooms[1].name == "Gast"
        assert jobs and jobs[0] == ("export", LEVEL_ID)  # the automatic update started like after an accept
        await _eventually(lambda: _has(user, "assistant-applied-undo"))
        user.find(marker="assistant-applied-undo").click()
        await _eventually(lambda: [edit["status"] for edit in service.edits()] == ["undone"])
        assert load_plan_model(plan_model_path(project, LEVEL_ID)).rooms[1].name == "Schlafen"
        await _eventually(lambda: not _has(user, "assistant-applied"))


@pytest.mark.anyio
async def test_panel_without_full_control_keeps_accept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: list) -> None:  # noqa: F811
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Vorschlag: Gast.")]))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Vorschlag: Gast.")
        await user.should_see(marker="assistant-accept")
        assert not _has(user, "assistant-full-mode") and not _has(user, "assistant-applied")
        assert jobs == []
    AssistantService(project, LEVEL_ID).discard()
