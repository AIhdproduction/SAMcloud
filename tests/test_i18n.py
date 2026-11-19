"""Tests for the central UI translation module (samcloud.app.i18n)."""

from __future__ import annotations

import json
import logging
import re
import string
from pathlib import Path
from typing import Any

import pytest
from nicegui.testing.user_simulation import user_simulation

from samcloud.app import i18n, layout
from samcloud.app.state import state as app_state


APP_DIRECTORY = Path(i18n.__file__).resolve().parent
LOCALES = i18n.LOCALES_DIRECTORY


def _flatten(prefix: str, value: Any, target: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}", item, target)
    else:
        target[prefix] = value


def _namespace_keys(language: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((LOCALES / language).glob("*.json")):
        flat: dict[str, Any] = {}
        _flatten(path.stem, json.loads(path.read_text(encoding="utf-8")), flat)
        result[path.stem] = flat
    return result


def _placeholders(text: str) -> set[str]:
    names = set()
    for _, field, _, _ in string.Formatter().parse(text):
        if field is not None:
            names.add(re.split(r"[.\[!:]", field, maxsplit=1)[0])
    return names


def _use_config(monkeypatch: pytest.MonkeyPatch, value: Any) -> None:
    monkeypatch.setattr(i18n, "_read_config_language", lambda: value)
    i18n.reload_translations()


def _write_preferences(data: dict[str, Any]) -> Path:
    path = i18n.PREFERENCES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    i18n.reload_translations()
    return path


# --------------------------------------------------------------------------- catalogues


def test_de_and_en_catalogues_have_identical_key_sets() -> None:
    german = _namespace_keys("de")
    english = _namespace_keys("en")
    assert {"common", "nav", "workflow", "classes"} <= set(english)
    assert set(german) == set(english), f"namespace files differ: de-only={set(german) - set(english)}, en-only={set(english) - set(german)}"
    problems = []
    for namespace in sorted(english):
        de_keys, en_keys = set(german[namespace]), set(english[namespace])
        if de_keys != en_keys:
            problems.append(f"{namespace}: missing in en={sorted(de_keys - en_keys)}, missing in de={sorted(en_keys - de_keys)}")
    assert not problems, "\n".join(problems)


def test_all_translation_values_are_non_empty_strings() -> None:
    for language in i18n.SUPPORTED_LANGUAGES:
        for namespace, flat in _namespace_keys(language).items():
            bad = [key for key, value in flat.items() if not isinstance(value, str) or not value.strip()]
            assert not bad, f"{language}/{namespace}.json has empty or non-text values: {bad}"


def test_placeholders_match_between_languages() -> None:
    german = _namespace_keys("de")
    english = _namespace_keys("en")
    mismatches = []
    for namespace, flat in english.items():
        for key, en_text in flat.items():
            de_text = german.get(namespace, {}).get(key)
            if not isinstance(de_text, str) or not isinstance(en_text, str):
                continue
            if _placeholders(de_text) != _placeholders(en_text):
                mismatches.append(f"{key}: de={sorted(_placeholders(de_text))} en={sorted(_placeholders(en_text))}")
    assert not mismatches, "\n".join(mismatches)


# --------------------------------------------------------------------------- language resolution


def test_default_language_without_preference_or_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, None)
    assert i18n.current_language() == "de"
    assert i18n.config_language() is None
    assert i18n.user_language() is None


def test_config_language_used_without_user_preference(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "en")
    assert i18n.current_language() == "en"
    _use_config(monkeypatch, "de")
    assert i18n.current_language() == "de"


def test_user_preference_overrides_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    _write_preferences({"ui_language": "en"})
    assert i18n.user_language() == "en"
    assert i18n.current_language() == "en"


def test_invalid_config_value_falls_back_to_default(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    i18n._warned.clear()
    _use_config(monkeypatch, "fr")
    with caplog.at_level(logging.WARNING, logger=i18n.__name__):
        assert i18n.current_language() == "de"
        assert i18n.config_language() is None
        assert i18n.config_language() is None
    assert sum("'fr'" in record.getMessage() for record in caplog.records) == 1, "an invalid value is logged exactly once"


def test_invalid_stored_value_falls_back_to_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "en")
    _write_preferences({"ui_language": "klingon"})
    assert i18n.user_language() is None
    assert i18n.current_language() == "en"


def test_broken_preferences_file_falls_back_to_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "en")
    path = i18n.PREFERENCES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    i18n.reload_translations()
    assert i18n.current_language() == "en"


@pytest.mark.parametrize(("raw", "expected"), [("de-CH", "de"), ("DE", "de"), ("en_US", "en"), (" EN ", "en"), ("fr", None), ("", None), (None, None), (3, None)])
def test_normalize_language(raw: Any, expected: str | None) -> None:
    assert i18n.normalize_language(raw) == expected


def test_regional_config_and_preference_values_normalise(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "en_US")
    assert i18n.current_language() == "en"
    _write_preferences({"ui_language": "de-CH"})
    assert i18n.current_language() == "de"


def test_real_config_loader_reads_defaults_yml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import samcloud.core.configuration as configuration

    monkeypatch.undo()  # drop the conftest config override for this test only
    monkeypatch.setattr(i18n, "PREFERENCES_PATH", tmp_path / ".samcloud" / "user_preferences.json")
    monkeypatch.setattr(configuration, "CONFIG_DIRECTORY", tmp_path)
    (tmp_path / "defaults.yml").write_text("application:\n  language: en\n", encoding="utf-8")
    i18n.reload_translations()
    assert i18n.config_language() == "en"
    (tmp_path / "defaults.yml").write_text("application: [unclosed\n", encoding="utf-8")
    assert i18n.config_language() is None
    (tmp_path / "defaults.yml").unlink()
    assert i18n.config_language() is None
    (tmp_path / "defaults.yml").write_text("application:\n  projects_directory: projects\n", encoding="utf-8")
    assert i18n.config_language() is None


# --------------------------------------------------------------------------- persistence


def test_set_language_writes_atomically_and_keeps_other_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    path = _write_preferences({"theme": "dark", "ui_language": "de", "recent": [1, 2]})
    replaced: list[tuple[str, str]] = []
    real_replace = i18n.os.replace

    def tracking_replace(source: Any, target: Any) -> None:
        replaced.append((str(source), str(target)))
        real_replace(source, target)

    monkeypatch.setattr(i18n.os, "replace", tracking_replace)
    assert i18n.set_language("EN") == "en"
    assert len(replaced) == 1
    source, target = replaced[0]
    assert Path(target) == path
    assert Path(source).parent == path.parent, "the temp file must live in the same directory"
    assert not Path(source).exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"theme": "dark", "ui_language": "en", "recent": [1, 2]}
    assert i18n.current_language() == "en"
    assert sorted(item.name for item in path.parent.iterdir()) == [path.name], "no temp files are left behind"


def test_set_language_creates_directory_and_rejects_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    path = i18n.PREFERENCES_PATH
    assert not path.parent.exists()
    assert i18n.set_language("en") == "en"
    assert json.loads(path.read_text(encoding="utf-8")) == {"ui_language": "en"}
    with pytest.raises(ValueError):
        i18n.set_language("fr")
    assert json.loads(path.read_text(encoding="utf-8")) == {"ui_language": "en"}
    assert i18n.current_language() == "en"


def test_clear_user_language_removes_only_language_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    path = _write_preferences({"theme": "dark", "ui_language": "en"})
    assert i18n.current_language() == "en"
    assert i18n.clear_user_language() == "de"
    assert path.exists(), "the preferences file is rewritten, never deleted"
    assert json.loads(path.read_text(encoding="utf-8")) == {"theme": "dark"}
    assert i18n.user_language() is None
    assert i18n.current_language() == "de"


def test_clear_user_language_without_file_creates_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "en")
    assert i18n.clear_user_language() == "en"
    assert not i18n.PREFERENCES_PATH.exists()


# --------------------------------------------------------------------------- lookup


def _with_catalogues(monkeypatch: pytest.MonkeyPatch, language: str, de: dict[str, str], en: dict[str, str]) -> None:
    _use_config(monkeypatch, language)
    monkeypatch.setattr(i18n, "_catalogues", {"de": de, "en": en})


def test_t_fallback_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_catalogues(
        monkeypatch, "de",
        de={"demo.both": "Beide"},
        en={"demo.both": "Both", "demo.only_en": "English only"},
    )
    assert i18n.t("demo.both") == "Beide"
    assert i18n.t("demo.only_en") == "English only"
    assert i18n.t("demo.nowhere") == "demo.nowhere"


def test_t_formats_placeholders_and_tolerates_missing_params(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_catalogues(
        monkeypatch, "en",
        de={"demo.points": "{count} Punkte in {name}"},
        en={"demo.points": "{count} points in {name}", "demo.brace": "Value {0} and {"},
    )
    assert i18n.t("demo.points", count=3, name="Room") == "3 points in Room"
    assert i18n.t("demo.points", count=3) == "{count} points in {name}"
    assert i18n.t("demo.points") == "{count} points in {name}"
    assert i18n.t("demo.brace", value=1) == "Value {0} and {"


def test_translate_or(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_catalogues(
        monkeypatch, "de",
        de={"classes.wall": "Wand"},
        en={"classes.wall": "Wall", "classes.floor": "Floor", "demo.n": "{n} items"},
    )
    assert i18n.translate_or("classes.wall", "wall") == "Wand"
    assert i18n.translate_or("classes.floor", "floor") == "Floor"
    assert i18n.translate_or("classes.garden gnome", "garden gnome") == "garden gnome"
    assert i18n.translate_or("demo.n", "x", n=2) == "2 items"


def test_real_catalogues_translate_shared_namespaces(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    assert i18n.t("nav.overview") == "Übersicht"
    assert i18n.t("common.close") == "Schliessen"
    assert i18n.translate_or("classes.wall", "wall") == "Wand"
    assert i18n.translate_or("workflow.status.completed", "completed") == "Abgeschlossen"
    _use_config(monkeypatch, "en")
    assert i18n.t("nav.overview") == "Overview"
    assert i18n.translate_or("workflow.step.dxf", "dxf") == "DXF/IFC export"


def test_class_keys_cover_configured_class_names() -> None:
    config = json.loads((Path(i18n.PROJECT_ROOT) / "config" / "classes.json").read_text(encoding="utf-8"))
    english = i18n.catalogue("en")
    names = {entry.split(",")[0].strip() for class_set in ("indoor", "outdoor") for entry in config[class_set]}
    missing = sorted(name for name in names if f"classes.{name}" not in english)
    assert not missing, f"classes.json names without a classes.* translation: {missing}"


# --------------------------------------------------------------------------- static key scan


_LITERAL_KEY = re.compile(r"""(?<![\w.])(?:t|translate_or)\(\s*(?P<quote>["'])(?P<key>[A-Za-z0-9_]+(?:\.[^"'{}\n]+)+)(?P=quote)""")


def test_literal_keys_in_app_sources_exist_in_english_catalogue() -> None:
    english = i18n.catalogue("en")
    missing = []
    for path in sorted(APP_DIRECTORY.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in _LITERAL_KEY.finditer(source):
            key = match.group("key")
            if key not in english:
                line = source.count("\n", 0, match.start()) + 1
                missing.append(f"{path.relative_to(APP_DIRECTORY)}:{line}: {key}")
    nav_missing = [key for key, _, _ in layout._NAVIGATION if key not in english]
    assert not missing, "keys used in samcloud/app but missing in locales/en:\n" + "\n".join(missing)
    assert not nav_missing, f"navigation keys missing in locales/en: {nav_missing}"


def test_literal_key_scan_skips_f_strings() -> None:
    sample = 't("nav.overview") + t(f"classes.{name}") + translate_or("workflow.step.dxf", x) + self.t("x.y")'
    assert [match.group("key") for match in _LITERAL_KEY.finditer(sample)] == ["nav.overview", "workflow.step.dxf"]


# --------------------------------------------------------------------------- sidebar (NiceGUI user simulation)


def _shell_route() -> None:
    layout.render_page("I18n", lambda: None, current_path="/")


# NiceGUI's test reset drops sys.modules entries of page modules outside "tests."; keep samcloud loaded.
_shell_route.__module__ = "tests.i18n_shell_route"


def _tech_footer_content(user: Any) -> str:
    elements = list(user.find(marker="sidebar-tech").elements)
    assert len(elements) == 1
    return elements[0].content


@pytest.mark.anyio
async def test_sidebar_german(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "de")
    monkeypatch.setattr(app_state, "active_project_directory", None)
    async with user_simulation(root=lambda: None) as user:
        from nicegui import ui

        ui.page("/i18n-shell")(_shell_route)
        await user.open("/i18n-shell")
        await user.should_see("Übersicht", marker="nav-overview")
        await user.should_see("Projekte", marker="nav-projects")
        await user.should_not_see("Overview")
        footer = _tech_footer_content(user)
        for name in ("COLMAP", "SAM3", "FlurSeg", "Depth Anything"):
            assert f'<span class="sam-tech">{name}</span>' in footer
        assert footer.count('<span class="sam-tech-sep">|</span>') == 3


@pytest.mark.anyio
async def test_sidebar_english(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, "en")
    monkeypatch.setattr(app_state, "active_project_directory", None)
    async with user_simulation(root=lambda: None) as user:
        from nicegui import ui

        ui.page("/i18n-shell")(_shell_route)
        await user.open("/i18n-shell")
        await user.should_see("Overview", marker="nav-overview")
        await user.should_not_see("Übersicht")
        assert "Depth Anything" in _tech_footer_content(user)


# --------------------------------------------------------------------------- display-time translation of stored texts

@pytest.mark.parametrize(("language", "expected"), [("en", "Only 2 AprilTag(s) found – no check possible"), ("de", "Nur 2 AprilTag(s) gefunden – keine Kontrolle möglich")])
def test_apriltag_report_warnings_are_translated_for_display(monkeypatch: pytest.MonkeyPatch, language: str, expected: str) -> None:
    from samcloud.app.pages.control_points import apriltag_warning_text
    from samcloud.control_points.apriltags import WARNING_GEOREF_UNAVAILABLE, WARNING_NO_CONTROL, WARNING_NO_TAGS

    _use_config(monkeypatch, language)
    assert apriltag_warning_text(WARNING_NO_CONTROL.format(n=2)) == expected
    assert apriltag_warning_text(WARNING_NO_TAGS) == i18n.t("control_points.apriltags.warning_no_tags")
    assert apriltag_warning_text(WARNING_GEOREF_UNAVAILABLE) == i18n.t("control_points.apriltags.warning_georef_unavailable")
    assert apriltag_warning_text("AprilTag-Ausreißer ignoriert: 3, 7").endswith(": 3, 7")
    assert apriltag_warning_text("unknown free text") == "unknown free text"


def test_progress_texts_are_translated_for_display_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app.pages import home
    from samcloud.app.processing import display_progress_text

    _use_config(monkeypatch, "de")
    assert display_progress_text("Exporting LAS") == i18n.t("processing.stage.exporting_las") != "Exporting LAS"
    assert display_progress_text("Pipeline ended with exit code 3. See reports/pipeline.log.") == i18n.t("processing.error.pipeline_exit", code="3")
    assert display_progress_text("some engine detail") == "some engine detail"
    assert home._stage_text({"stage": "Complete"}) == i18n.t("home.stage.complete")
    assert home._stage_text({"stage": "Some new stage"}) == "Some new stage"


def test_viewer_server_messages_are_translated_for_display(monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app import pointcloud_viewer, potree_viewer

    _use_config(monkeypatch, "de")
    strings = potree_viewer._viewer_strings()
    assert strings["server_messages"]["Potree or PotreeConverter is not installed"] == i18n.t("viewer.server.not_installed")
    assert "{details}" in strings["server_prefixes"]["PotreeConverter failed: "]
    script = potree_viewer._script(potree_viewer._viewer_config("dense", "Dense", []))
    # The status logic still compares against the raw (English) server text; only the display is translated.
    assert 'String(state.message).includes("not been prepared")' in script
    assert "serverText(state.message)" in script
    assert "server_messages" in pointcloud_viewer._script("viewer", "/api/pointcloud/sparse")
