"""Settings page language selector, translated option mappings and draft preservation across reloads."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import yaml
from nicegui import app, ui
from nicegui.testing.user import User
from nicegui.testing.user_simulation import user_simulation

from samcloud.app import i18n
from samcloud.app.pages import projects as projects_page
from samcloud.app.pages import settings as settings_page
from samcloud.app.state import state as app_state
from samcloud.core.project_manager import ProjectManager

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _settings_route() -> None:
    settings_page.render()


# NiceGUI's test reset drops sys.modules entries of page modules outside "tests."; keep samcloud loaded.
_settings_route.__module__ = "tests.i18n_settings_route"


def _use_config(monkeypatch: pytest.MonkeyPatch, value: Any) -> None:
    monkeypatch.setattr(i18n, "_read_config_language", lambda: value)
    i18n.reload_translations()


def _make_project(root: Path) -> Path:
    images = root / "images"
    images.mkdir()
    return ProjectManager(root / "projects").create_project("lang", str(images), capture_type="camera")


def _record_reloads(user: User, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any] | None]:
    """Record the stored draft at each simulated reload, then let the user simulation reload the page."""
    calls: list[dict[str, Any] | None] = []
    real_reload = user.navigate.reload

    def recording_reload() -> None:
        draft = app.storage.tab.get(settings_page.DRAFT_STORAGE_KEY)
        calls.append(dict(draft) if isinstance(draft, dict) else None)
        real_reload()

    monkeypatch.setattr(user.navigate, "reload", recording_reload)
    return calls


@asynccontextmanager
async def _settings_user(project: Path | None, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[User]:
    monkeypatch.setattr(app_state, "active_project_directory", project)
    async with user_simulation(root=lambda: None) as user:
        ui.page("/settings")(_settings_route)
        await user.open("/settings")
        await asyncio.sleep(0.1)  # let the draft-restore timer run
        yield user


def _element(user: User, marker: str) -> Any:
    elements = user.find(marker=marker).elements
    assert len(elements) == 1, f"expected one element marked {marker!r}, found {len(elements)}"
    return next(iter(elements))


def _number_by_label(user: User, label: str) -> Any:
    with user:
        numbers = [e for e in user.find(kind=ui.number).elements if e.props.get("label") == label]
    assert len(numbers) == 1
    return numbers[0]


def _saved_project(project: Path) -> dict[str, Any]:
    return yaml.safe_load((project / "project.samcloud.yml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- option mappings


def _translated_mappings() -> dict[str, projects_page.TranslatedOptions]:
    found = {}
    for module in (projects_page, settings_page):
        for name, value in vars(module).items():
            if isinstance(value, projects_page.TranslatedOptions):
                found[f"{module.__name__}.{name}"] = value
    return found


def test_translated_option_keys_exist_in_both_languages() -> None:
    mappings = _translated_mappings()
    assert {"samcloud.app.pages.projects.STORAGE_OPTIONS", "samcloud.app.pages.settings.CONTROL_POINTS_MODE_OPTIONS"} <= set(mappings)
    german, english = i18n.catalogue("de"), i18n.catalogue("en")
    missing = [
        f"{name}: {key}"
        for name, mapping in mappings.items()
        for key in mapping.label_keys.values()
        if key not in german or key not in english
    ]
    assert not missing, "\n".join(missing)


def test_translated_options_keep_values_and_follow_language(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    assert list(settings_page.CONTROL_POINTS_MODE_OPTIONS) == ["none", "after_alignment", "automatic"]
    assert settings_page.CONTROL_POINTS_MODE_OPTIONS["automatic"] == "Automatisch (Targets erkennen)"
    assert list(settings_page.TILE_SIZE_OPTIONS) == [768, 1008, 1280]
    assert settings_page.STORAGE_OPTIONS is projects_page.STORAGE_OPTIONS
    _use_config(monkeypatch, "en")
    assert settings_page.CONTROL_POINTS_MODE_OPTIONS["automatic"] == "Automatic (detect targets)"
    assert dict(projects_page.CLASS_SET_OPTIONS) == {"outdoor": "Outdoor", "indoor": "Indoor"}


def test_draft_applies_only_to_its_own_project() -> None:
    class Control:
        def __init__(self, value: Any) -> None:
            self.value = value

    controls = {"min_votes": Control(1), "use_gpu": Control(True), "missing": None}
    settings_page._apply_draft(controls, {"project": "B", "values": {"min_votes": 9}}, "A")
    assert controls["min_votes"].value == 1
    settings_page._apply_draft(controls, {"project": "A", "values": {"min_votes": 9, "missing": 1, "unknown": 2}}, "A")
    assert controls["min_votes"].value == 9 and controls["use_gpu"].value is True
    settings_page._apply_draft(controls, None, "A")
    assert settings_page._snapshot_draft(controls) == {"min_votes": 9, "use_gpu": True}


# --------------------------------------------------------------------------- rendered page


@pytest.mark.anyio
async def test_settings_page_renders_in_german(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    project = _make_project(tmp_path)
    async with _settings_user(project, monkeypatch) as user:
        await user.should_see("Sprache / Language")
        await user.should_see("Einstellungen speichern")
        await user.should_see("Klassifizierung")
        await user.should_see("Standard aus config/defaults.yml: Deutsch")
        await user.should_not_see("Save settings")
        await user.should_not_see("Classification")
        language = _element(user, "settings-language")
        assert language.value == "de"
        assert language.options == {"de": "Deutsch", "en": "English"}
        assert not _element(user, "settings-language-reset").enabled, "nothing stored yet, nothing to reset"
        with user:
            control_points = [e for e in user.find(kind=ui.select).elements if e.props.get("label") == "Passpunkte"]
        assert len(control_points) == 1
        assert control_points[0].options["automatic"] == "Automatisch (Targets erkennen)"


@pytest.mark.anyio
async def test_language_selector_without_project(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "en")
    async with _settings_user(None, monkeypatch) as user:
        await user.should_see("Open a project before editing settings.")
        assert _element(user, "settings-language").value == "en"


@pytest.mark.anyio
async def test_language_change_is_stored_and_keeps_unsaved_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    project = _make_project(tmp_path)
    saved_before = _saved_project(project)
    async with _settings_user(project, monkeypatch) as user:
        reloads = _record_reloads(user, monkeypatch)
        min_votes = _number_by_label(user, "Mindestanzahl übereinstimmender Bilder pro Punkt")
        with user:
            min_votes.value = 7  # unsaved edit
            _element(user, "settings-language").value = "en"
        await asyncio.sleep(0.3)  # simulated reload + draft-restore timer

        assert len(reloads) == 1
        assert reloads[0] is not None and reloads[0]["values"]["min_votes"] == 7, "the unsaved value is kept as a draft"
        assert reloads[0]["project"] == str(project)
        assert i18n.user_language() == "en"
        assert i18n.current_language() == "en"
        assert json.loads(i18n.PREFERENCES_PATH.read_text(encoding="utf-8")) == {"ui_language": "en"}

        # The reloaded page is English and shows the unsaved value again.
        await user.should_see("Save settings")
        await user.should_not_see("Einstellungen speichern")
        assert _element(user, "settings-language").value == "en"
        assert _number_by_label(user, "Minimum agreeing images per point").value == 7
        with user:
            assert settings_page.DRAFT_STORAGE_KEY not in app.storage.tab, "the draft is consumed once"
        assert _element(user, "settings-language-reset").enabled
    assert _saved_project(project) == saved_before, "changing the language never saves the project"


@pytest.mark.anyio
async def test_reset_returns_to_config_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    i18n.set_language("en")
    project = _make_project(tmp_path)
    async with _settings_user(project, monkeypatch) as user:
        reloads = _record_reloads(user, monkeypatch)
        await user.should_see("Save settings")
        user.find(marker="settings-language-reset").click()
        await asyncio.sleep(0.3)
        assert len(reloads) == 1
        assert i18n.user_language() is None
        assert i18n.current_language() == "de"
        assert json.loads(i18n.PREFERENCES_PATH.read_text(encoding="utf-8")) == {}
        await user.should_see("Einstellungen speichern")
        assert _element(user, "settings-language").value == "de"
        assert not _element(user, "settings-language-reset").enabled


@pytest.mark.anyio
async def test_save_settings_unchanged_by_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    project = _make_project(tmp_path)
    async with _settings_user(project, monkeypatch) as user:
        min_votes = _number_by_label(user, "Mindestanzahl übereinstimmender Bilder pro Punkt")
        with user:
            min_votes.value = 3
        user.find("Einstellungen speichern").click()
        await asyncio.sleep(0.1)
        await user.should_see("Projekteinstellungen gespeichert")
    saved = _saved_project(project)
    assert saved["settings"]["classification"]["min_votes"] == 3
    assert saved["workflow"]["control_points_mode"] == "none"
    assert set(saved["settings"]["storage"]) == set(projects_page.STORAGE_OPTIONS)
