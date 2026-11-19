"""The "create DXF/IFC automatically" switch on the project-creation page and the settings page."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from nicegui import ui
from nicegui.testing.user_interaction import UserInteraction

from test_apriltag_gui import _element, _page_user, _projects_route
from test_i18n_settings import _element as _settings_element
from test_i18n_settings import _make_project, _saved_project, _settings_user, _use_config
from samcloud.app.state import state as app_state

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.anyio
@pytest.mark.parametrize("switched_off", [False, True])
async def test_projects_page_switch_defaults_on_and_is_passed_on_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, switched_off: bool,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create_project(self: Any, name: str, image_directory: str, **kwargs: Any) -> Path:
        calls.append(kwargs)
        raise ValueError("stop after capture")

    monkeypatch.setattr(type(app_state.projects), "create_project", fake_create_project)
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        await user.should_see(marker="auto-dxf-switch")
        switch = _element(user, "auto-dxf-switch")
        assert switch.value is True and switch.text == "Create DXF/IFC automatically after processing"
        if switched_off:
            with user:
                switch.value = False
        with user:
            form_button = max(user.find("Create project", kind=ui.button).elements, key=lambda e: e.id)
        UserInteraction(user, {form_button}, "Create project").click()
        await asyncio.sleep(0.3)
    assert calls[0]["auto_dxf"] is (not switched_off)


@pytest.mark.anyio
async def test_projects_page_switch_is_german(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, german_ui: None) -> None:
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert _element(user, "auto-dxf-switch").text == "DXF/IFC nach der Berechnung automatisch erstellen"


@pytest.mark.anyio
async def test_settings_page_switch_reads_and_saves_the_project_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "en")
    project = _make_project(tmp_path)
    assert _saved_project(project)["settings"]["workflow"]["auto_dxf"] is True, "new projects start with the switch ON"
    async with _settings_user(project, monkeypatch) as user:
        switch = _settings_element(user, "auto-dxf-switch")
        assert switch.value is True
        with user:
            switch.value = False
        user.find("Save settings").click()
        await asyncio.sleep(0.3)
    assert _saved_project(project)["settings"]["workflow"]["auto_dxf"] is False
