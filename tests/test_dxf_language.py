"""DXF texts in the app language: rule tables, export (run_export / CLI / runner) and the browser payload."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ezdxf
import numpy as np
import pytest

from samcloud.dxf import job_runner, jobs
from samcloud.dxf.cancel import CancelToken
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus, vector_payload
from samcloud.dxf.text_language import (
    has_words,
    table_collisions,
    translate_note,
    translate_result,
    translate_text,
)
from samcloud.dxf.writer import export_plan, outdoor_symbol_insert

LOGGER = logging.getLogger("samcloud.tests.dxf_language")

GERMAN_WORDS = (
    "Bad", "Küche", "Schlafen", "Büro", "Zimmer", "Wohnen", "Essen", "Flur", "Möbel", "Einbau", "Tür", "Durchgang", "Fenster",
    "Treppe", "Gebüsch", "Dach", "Traufe", "KRONE", "STAMM", "HOEHE", "nicht gemessen", "Schacht", "Baum", "DK", "St.", "Raum",
    "Sessel", "Bett", "Tisch", "Stuhl", "Schrank", "Regal", "Spüle", "Herd", "Lavabo", "Dusche", "Badewanne", "KS", "HS",
    "Garderobe", "Kücheninsel", "Kochfeld", "Boden", "Gebäude", "Wasser", "Fahrzeug", "Gehweg", "Massstab", "Variante", "Versatz",
    "unbekanntes", "First", "Spiegel", "Pflanze", "Kissen", "Decke",
)
ENGLISH_WORDS = (
    "Window", "Door", "Stairs", "Tree", "Crown", "Trunk", "Roof", "Ridge", "Eaves", "Shrub", "Room", "Kitchen", "Furniture",
    "Passage", "CROWN", "TRUNK", "HEIGHT", "not measured", "unknown profile", "scale", "variant", "Manhole", "Cooktop", "Sink",
    "Mirror", "Built-in", "Bathroom", "Bedroom", "Living", "Ground", "Building", "Water", "Vehicle", "Sidewalk",
)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![A-Za-zÄÖÜäöüß]){re.escape(word)}(?![A-Za-zÄÖÜäöüß])", text) is not None


def _make_result(layers: dict[str, list[dict[str, Any]]], *, mode: str = "indoor", level_name: str = "EG",
                 project_name: str = "LangProject", variant: str = "ortho", blocks: list[str] | None = None) -> DxfPlanResult:
    return DxfPlanResult(
        project_name=project_name, level_name=level_name, mode=mode, profile_name=None, scale_factor=1.0,
        frame=Frame(np.zeros(3, dtype=np.float64), 1.0, np.eye(3, dtype=np.float64)), settings_fingerprint="fp-lang",
        layers=layers, statuses={name: LayerStatus(name, "ok", None, len(entities)) for name, entities in layers.items()},
        layer_colors={name: 7 for name in layers}, background=BackgroundImage(None, None, (0.013, -0.021, 12.37, 9.81)),
        blocks=list(blocks or []), variant=variant,
    )


def _text(text: str, x: float = 1.013, y: float = 2.027) -> dict[str, Any]:
    return {"type": "text", "text": text, "point": [x, y], "height": 0.12, "rotation": 0.0}


def _furniture(label: str, **extra: Any) -> dict[str, Any]:
    return {"type": "insert", "name": "FURNITURE", "point": [3.017, 4.031], "rotation": 0.0, "xscale": 0.81, "yscale": 0.43,
            "attributes": {"LABEL": label, **extra}}


def _indoor_result() -> DxfPlanResult:
    """Every indoor label literal, produced by the real functions where they are cheap to call."""
    from samcloud.ortho.plan import FINE_LABEL_GERMAN_NAMES, _FIXTURE_KIND_GERMAN, _furniture_label_text
    from samcloud.ortho.vector.builtin_fronts import BUILTIN_LABELS
    from samcloud.ortho.vector.indoor import Opening, opening_entities
    from samcloud.ortho.vector.kitchen_layout import tall_unit_entities
    from samcloud.ortho.vector.labels import class_display_name
    from samcloud.ortho.vector.niche import NICHE_LABELS
    from samcloud.ortho.vector.plan_polish import _GENERIC_FURNITURE
    from samcloud.ortho.vector.room_names import classify_room, room_label_text
    from shapely.geometry import box

    room_names = {classify_room(tags, area) for tags, area in (
        ({"bathtub"}, 6.1), ({"toilet"}, 1.7), ({"sink_wc"}, 4.3), ({"kitchen", "living"}, 30.2), ({"kitchen", "dining"}, 18.3),
        ({"kitchen"}, 9.4), ({"bed"}, 14.2), ({"office"}, 11.3), ({"wardrobe"}, 10.7), ({"living"}, 22.6), ({"dining"}, 12.9))}
    room_names.discard(None)
    room_names |= {"Flur", "Zimmer"}
    annotations = [_text(room_label_text(name, 12.34 + index * 1.01)) for index, name in enumerate(sorted(room_names))]
    annotations.append(_text("Room 3 (12.4 m²)"))
    displays = sorted(set(FINE_LABEL_GERMAN_NAMES.values()) | set(_FIXTURE_KIND_GERMAN.values()) | {"Möbel", class_display_name("mirror")})
    furniture = [_furniture(name, INSTANCE="7") for name in displays]
    furniture += [_text(_furniture_label_text(name, "sofa", 0.7613)) for name in displays]
    furniture.append(_text(_furniture_label_text("Stuhl", "chair", 0.45)))
    furniture += [_furniture(raw) for raw in ("cooktop", "stove", "sink", "kitchen sink", "Spüle", "Herd")]
    furniture.append({"type": "insert", "name": "FURNITURE", "point": [1.0, 1.0], "rotation": 12.5, "xscale": 1.2, "yscale": 0.6,
                      "attributes": {"class": "117", "height": "0.800", "instance": 3, "label": "sofa"}})
    builtins = [_text("Kücheninsel"), _text("Küche"), _text("Küche h=0.91"), _text("Einbau h=2.13"), _text(f"{_GENERIC_FURNITURE} h=0.52")]
    builtins += [_text(name) for name in sorted(set(BUILTIN_LABELS.values()) | set(NICHE_LABELS.values()) | {"Einbau"})]
    builtins += tall_unit_entities(box(0.0, 0.0, 0.61, 0.62), "KS", 1.0) + tall_unit_entities(box(1.0, 0.0, 1.61, 0.62), "HS", 1.0)
    openings = opening_entities([
        Opening("DOOR", (2.013, 0.007), 0.8123, 0.0, 0.12, leaf_detected=False, head_m=2.0517),
        Opening("passage", (4.011, 0.009), 0.9031, 0.0, 0.12),
        Opening("passage", (6.017, 0.013), 1.1047, 0.0, 0.12, head_m=2.1033),
        Opening("DOOR", (8.021, 0.011), 0.8519, 0.0, 0.12, leaf_detected=True, hinge_side="left", swing_side=1),
        Opening("WINDOW", (10.03, 0.017), 1.2031, 0.0, 0.25),
    ])
    openings.append({"type": "text", "text": "Stairs", "point": [5.0, 5.0], "height": 0.12, "rotation": 30.0, "layer_role": "label"})
    return _make_result({"A-ANNO": annotations, "A-FURN": furniture, "A-EINB": builtins, "A-DOOR": openings})


def _outdoor_result() -> DxfPlanResult:
    from samcloud.ortho.vector.buildings import roof_height_entities
    from samcloud.ortho.vector.labels import class_display_name
    from shapely.geometry import box

    texts = [_text(f"DK {4.83:.1f} m  H {6.47:.1f} m"), _text(f"St. Ø {0.347:.2f} m"), _text(f"Gebüsch h={1.23:.1f} m")]
    texts += roof_height_entities(box(0.0, 0.0, 10.0, 8.0), 431.257, 431.1) + roof_height_entities(box(0.0, 0.0, 10.0, 8.0), 436.21, 431.07)
    texts += [_text(class_display_name(name)) for name in ("Ground", "Road Surface", "Sealed terrain", "Side walk", "Building",
                                                            "Low Vegetation", "Vehicle", "Worksite equipment", "Water", "HCS construction material")]
    texts += [_text("+431.26"), _text("432")]  # spot height and contour level: numbers stay, never reported
    symbols = [
        {"type": "insert", "name": "TREE", "point": [4.0, 4.0], "rotation": 0.0, "xscale": 4.8, "yscale": 4.8,
         "attributes": {"LABEL": f"Tree d={4.83:.1f} h={6.47:.1f}"}},
        outdoor_symbol_insert("TREE", (7.013, 2.021, 431.0), {"HEIGHT": 8.4, "TRUNK_DIA": 0.35}, crown_radius=2.5),
        outdoor_symbol_insert("TREE", (9.011, 3.017, 431.0), {"HEIGHT": 6.0}),
        outdoor_symbol_insert("manhole", (3.0, 4.0, 431.256), {"Z_DECKEL": 431.256, "TYP": "Schacht"}),
    ]
    return _make_result({"S-ANNO": texts, "S-TREE": symbols}, mode="outdoor", level_name="site", blocks=[])


def _all_texts(result: DxfPlanResult) -> list[str]:
    from samcloud.dxf.text_language import TEXT_ATTRIBUTE_TAGS

    texts: list[str] = []
    for entities in result.layers.values():
        for entity in entities:
            if entity.get("type") == "text":
                texts.append(str(entity["text"]))
            elif entity.get("type") == "insert":
                texts.extend(str(value) for tag, value in (entity.get("attributes") or {}).items() if tag in TEXT_ATTRIBUTE_TAGS and isinstance(value, str))
    return texts


# --------------------------------------------------------------------------- 1. table completeness


def test_tables_have_no_conflicting_source_spellings() -> None:
    assert table_collisions() == []


@pytest.mark.parametrize("factory", [_indoor_result, _outdoor_result])
def test_english_leaves_no_german_and_numbers_unchanged(factory: Any) -> None:
    source = factory()
    translated, missing = translate_result(source, "en")
    assert missing == []
    before, after = _all_texts(source), _all_texts(translated)
    assert len(before) == len(after) and len(after) > 10
    for original, text in zip(before, after, strict=True):
        assert not re.search(r"[äöüÄÖÜß]", text), (original, text)
        assert not [word for word in GERMAN_WORDS if _contains_word(text, word)], (original, text)
        assert _NUMBER.findall(original) == _NUMBER.findall(text), (original, text)


@pytest.mark.parametrize("factory", [_indoor_result, _outdoor_result])
def test_german_leaves_no_english_and_round_trips(factory: Any) -> None:
    source = factory()
    english, _ = translate_result(source, "en")
    german, missing = translate_result(english, "de")
    assert missing == []
    direct, _ = translate_result(source, "de")
    assert _all_texts(german) == _all_texts(direct)
    for original, text in zip(_all_texts(source), _all_texts(direct), strict=True):
        assert not [word for word in ENGLISH_WORDS if _contains_word(text, word)], (original, text)
        assert _NUMBER.findall(original) == _NUMBER.findall(text), (original, text)


def test_translate_result_never_mutates_the_input() -> None:
    source = _indoor_result()
    snapshot = json.dumps(source.layers, sort_keys=True, default=str)
    translate_result(source, "en")
    assert json.dumps(source.layers, sort_keys=True, default=str) == snapshot


def test_known_examples_and_unknown_texts() -> None:
    assert translate_text("Küche  12.4 m²", "en") == "Kitchen  12.4 m²"
    assert translate_text("Wohnen / Küche  20.13 m²", "en") == "Living / Kitchen  20.13 m²"
    assert translate_text("Tür 0.81 m H 2.05", "en") == "Door 0.81 m H 2.05"
    assert translate_text("Window 1.20 m", "de") == "Fenster 1.20 m"
    assert translate_text("Door 0.85 m", "de") == "Tür 0.85 m"
    assert translate_text("Stairs", "de") == "Treppe"
    assert translate_text("STAMM=nicht gemessen", "en") == "TRUNK=not measured"
    assert translate_text("KRONE=n/a", "en") == "CROWN=n/a"
    assert translate_text("KS", "en") == "RF" and translate_text("HS", "en") == "TU"
    assert translate_text("Gizmo 3000", "en") == "Gizmo 3000"
    translated, missing = translate_result(_make_result({"A-ANNO": [_text("Gizmo 3000"), _text("412.35")]}), "de")
    assert missing == ["Gizmo 3000"] and _all_texts(translated) == ["Gizmo 3000", "412.35"]
    assert not has_words("+412.35") and not has_words("12.4 m²") and has_words("Gizmo 3000")
    with pytest.raises(ValueError):
        translate_text("Bad", "fr")


def test_note_line_is_translated_part_by_part() -> None:
    english = "LangProject | EG | unknown profile | scale 1 | 2026-11-24 | variant ortho | local origin offset E 2600000.000 N 1200000.000 Z 0.000 (true = written + offset)"
    german = translate_note(english, "de")
    assert german == ("LangProject | EG | unbekanntes Profil | Massstab 1 | 2026-11-24 | Variante ortho | lokaler Ursprung Versatz "
                      "E 2600000.000 N 1200000.000 Z 0.000 (wahr = geschrieben + Versatz)")
    assert translate_note(german, "en") == english
    assert translate_note("Haus | OG | sam3_indoor | scale 0.998 | 2026-11-24", "de") == "Haus | OG | sam3_indoor | Massstab 0.998 | 2026-11-24"


# --------------------------------------------------------------------------- 2./3. run_export


def _patch_export_inputs(monkeypatch: pytest.MonkeyPatch, result: DxfPlanResult, project: Path) -> None:
    level = {"id": "lvl_eg", "name": result.level_name, "floor_z": 0.013, "ceiling_z": 2.61, "checked": True}
    state = SimpleNamespace(mode_override="indoor", levels=[level], view={"coordinates": "local"}, segment_source="combined")
    cache = project / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (project / "products" / "dxf").mkdir(parents=True, exist_ok=True)
    (cache / "result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(jobs, "load_state", lambda _project: state)
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="indoor", frame=result.frame))
    monkeypatch.setattr(jobs, "plan_settings_for", lambda *_a, **_k: None)
    monkeypatch.setattr(jobs, "cache_directory", lambda *_a, **_k: cache)
    monkeypatch.setattr(jobs, "load_cached_result", lambda _cache: result)
    monkeypatch.setattr(jobs, "is_stale", lambda *_a, **_k: False)


def _stable_dxf(path: Path) -> str:
    """DXF text without the volatile parts (GUIDs, ezdxf save stamps, header timestamps)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\{[0-9A-Fa-f-]{36}\}", "{GUID}", text)
    text = re.sub(r"\d+\.\d+\.\d+ @ \S+", "EZDXF-STAMP", text)
    return re.sub(r"(\$TD[A-Z]+\n\s*\d+\n)[^\n]*", r"\1TIME", text)


def _dxf_texts(path: Path) -> list[str]:
    document = ezdxf.readfile(str(path))
    texts = [str(entity.dxf.text) for entity in document.modelspace().query("TEXT")]
    for insert in document.modelspace().query("INSERT"):
        texts.extend(str(attrib.dxf.text) for attrib in insert.attribs if attrib.dxf.tag in {"LABEL", "label", "TYP", "KRONE", "STAMM", "HOEHE"})
    return texts


def test_run_export_without_language_is_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "LangProject"
    result = _indoor_result()
    _patch_export_inputs(monkeypatch, result, project)
    jobs.run_export(project, project / "cloud.las", "lvl_eg", lambda *_a: None, CancelToken(), LOGGER, mode="indoor")
    exported = project / "products" / "dxf" / "export" / "LangProject_EG.dxf"
    reference = export_plan(result, tmp_path / "reference", coordinates="local")  # the unmodified code path
    assert _stable_dxf(exported) == _stable_dxf(reference)
    manifest = json.loads(jobs.export_manifest_path(project).read_text(encoding="utf-8"))
    assert "language" not in manifest and "untranslated" not in manifest
    assert jobs.exported_file_language(project, exported) is None


@pytest.mark.parametrize("language", ["en", "de"])
def test_run_export_writes_the_requested_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str) -> None:
    project = tmp_path / "LangProject"
    result = _indoor_result()
    _patch_export_inputs(monkeypatch, result, project)
    jobs.run_export(project, project / "cloud.las", "lvl_eg", lambda *_a: None, CancelToken(), LOGGER, mode="indoor", language=language)
    exported = project / "products" / "dxf" / "export" / "LangProject_EG.dxf"
    texts = _dxf_texts(exported)
    banned = GERMAN_WORDS if language == "en" else ENGLISH_WORDS
    assert not [(text, word) for text in texts for word in banned if _contains_word(text, word)]
    assert ("Kitchen  " if language == "en" else "Küche  ") in "\n".join(texts)
    note = next(text for text in texts if text.startswith("LangProject | EG | "))
    assert ("unknown profile | scale 1" if language == "en" else "unbekanntes Profil | Massstab 1") in note
    manifest = json.loads(jobs.export_manifest_path(project).read_text(encoding="utf-8"))
    assert manifest["language"] == language and manifest["untranslated"] == []
    assert manifest["file_languages"] == {"LangProject_EG.dxf": language} and manifest["level"] == "lvl_eg"
    assert jobs.exported_file_language(project, exported) == language
    # The payload shows exactly the exported texts (X-NOTE and attributes are DXF-only).
    payload = vector_payload(result, language=language)
    payload_texts = sorted(entity["text"] for layer in payload["layers"].values() for entity in layer["entities"] if entity["type"] == "text")
    exported_texts = sorted(str(entity.dxf.text) for entity in ezdxf.readfile(str(exported)).modelspace().query("TEXT") if entity.dxf.layer != "X-NOTE")
    assert payload_texts == exported_texts


def test_run_export_rejects_unknown_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "LangProject"
    _patch_export_inputs(monkeypatch, _indoor_result(), project)
    with pytest.raises(ValueError):
        jobs.run_export(project, project / "cloud.las", "lvl_eg", lambda *_a: None, CancelToken(), LOGGER, mode="indoor", language="fr")


def test_manifest_keeps_languages_of_earlier_files(tmp_path: Path) -> None:
    export_dir = tmp_path / "products" / "dxf" / "export"
    export_dir.mkdir(parents=True)
    first, second = export_dir / "P_EG_clean.dxf", export_dir / "P_OG_clean.dxf"
    jobs.write_export_manifest(export_dir, [first], "local", language="de", untranslated=[], variant="ortho", level="eg")
    jobs.write_export_manifest(export_dir, [second], "local", language="en", untranslated=["X"], variant="ortho", level="og")
    assert jobs.exported_file_language(tmp_path, first) == "de"
    assert jobs.exported_file_language(tmp_path, second) == "en"
    jobs.write_export_manifest(export_dir, [second], "local")  # an export without language: unknown again
    assert jobs.exported_file_language(tmp_path, second) is None


def test_payload_without_language_keeps_cached_texts() -> None:
    result = _make_result({"A-ANNO": [_text("Küche  12.4 m²"), _text("Stairs")]})
    texts = [entity["text"] for entity in vector_payload(result)["layers"]["A-ANNO"]["entities"]]
    assert texts == ["Küche  12.4 m²", "Stairs"]
    texts = [entity["text"] for entity in vector_payload(result, language="de")["layers"]["A-ANNO"]["entities"]]
    assert texts == ["Küche  12.4 m²", "Treppe"]


# --------------------------------------------------------------------------- 4. CLI and runner


def test_cli_accepts_lang_and_passes_it_to_run_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(jobs, "run_export", lambda *_a, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(jobs, "setup_job_logging", lambda _path: LOGGER)
    project = tmp_path / "cli_project"
    project.mkdir()
    monkeypatch.setattr(sys, "argv", ["jobs", "export", "--project", str(project), "--las", str(project / "c.las"), "--variant", "ortho", "--lang", "en"])
    jobs.main()
    assert calls == [{"mode": None, "variant": "ortho", "language": "en"}]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["jobs", "export", "--project", str(project), "--las", str(project / "c.las")])
    jobs.main()
    assert calls == [{"mode": None, "variant": "pointcloud"}]


class _FakePopen:
    commands: list[list[str]] = []

    def __init__(self, command: list[str], **_kwargs: Any) -> None:
        _FakePopen.commands.append(list(command))

    def poll(self) -> int:
        return 0


@pytest.mark.parametrize(("action", "expected"), [("export", ["--lang", "en"]), ("run_all", ["--lang", "en"]), ("preview", [])])
def test_runner_passes_the_ui_language_for_export_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str, expected: list[str]) -> None:
    _FakePopen.commands = []
    monkeypatch.setattr(job_runner.subprocess, "Popen", _FakePopen)
    runner = job_runner.DxfJobRunner(tmp_path / "runner_project")
    runner.start(action, tmp_path / "cloud.las", level="lvl", variant="ortho" if action == "export" else None)
    command = _FakePopen.commands[-1]
    found = command[command.index("--lang"): command.index("--lang") + 2] if "--lang" in command else []
    assert found == expected  # the autouse fixture runs the tests with the English UI


def test_runner_uses_the_german_ui_and_an_explicit_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, german_ui: None) -> None:
    _FakePopen.commands = []
    monkeypatch.setattr(job_runner.subprocess, "Popen", _FakePopen)
    runner = job_runner.DxfJobRunner(tmp_path / "runner_project_de")
    runner.start("export", tmp_path / "cloud.las")
    assert _FakePopen.commands[-1][-4:-2] == ["--lang", "de"]
    runner._process = None
    runner.status_file.unlink()
    runner.start("export", tmp_path / "cloud.las", language="en")
    assert "en" == _FakePopen.commands[-1][_FakePopen.commands[-1].index("--lang") + 1]


# --------------------------------------------------------------------------- run_all freshness


def test_run_all_redoes_only_the_export_for_another_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(jobs, "load_state", lambda _project: SimpleNamespace(mode_override=None, segment_source="combined", levels=[]))
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="outdoor"))
    monkeypatch.setattr(jobs, "preview_status", lambda *_a: "ok")
    monkeypatch.setattr(jobs, "workflow_step_status", lambda *_a: {"ortho": "ok", "segment": "ok", "clean_plan": "ok"})
    for name in ("run_ortho", "run_segment", "run_clean_plan", "run_export"):
        monkeypatch.setattr(jobs, name, lambda *_a, _name=name, **kwargs: calls.append((_name, kwargs)))
    project = tmp_path / "AllProject"
    export_dir = project / "products" / "dxf" / "export"
    clean = project / "products" / "dxf" / "site" / "ortho" / "result.json"
    clean.parent.mkdir(parents=True)
    clean.write_text("{}", encoding="utf-8")
    export_dir.mkdir(parents=True)
    exported = export_dir / "AllProject_site_clean.dxf"
    exported.write_text("dxf", encoding="utf-8")
    jobs.write_export_manifest(export_dir, [exported], "local", language="de", untranslated=[], variant="ortho", level=None)
    jobs.run_all(project, project / "cloud.las", None, lambda *_a: None, CancelToken(), LOGGER, mode="outdoor", language="de")
    assert calls == []
    jobs.run_all(project, project / "cloud.las", None, lambda *_a: None, CancelToken(), LOGGER, mode="outdoor", language="en")
    assert [name for name, _kwargs in calls] == ["run_export"] and calls[0][1]["language"] == "en"
    calls.clear()
    jobs.run_all(project, project / "cloud.las", None, lambda *_a: None, CancelToken(), LOGGER, mode="outdoor")
    assert calls == []  # without a language the old freshness rule applies


# --------------------------------------------------------------------------- GUI controller: export redone after a language switch


class _FakeRunner:
    def __init__(self) -> None:
        self.started: list[tuple[str, dict[str, Any]]] = []

    def is_running(self) -> bool:
        return False

    def start(self, action: str, **kwargs: Any) -> None:
        self.started.append((action, kwargs))


def _controller_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str | None) -> Path:
    from samcloud.app import dxf_controller

    project = tmp_path / "CtrlProject"
    export_dir = project / "products" / "dxf" / "export"
    export_dir.mkdir(parents=True)
    exported = export_dir / "CtrlProject_site_clean.dxf"
    exported.write_text("dxf", encoding="utf-8")
    las = project / "cloud.las"
    las.write_bytes(b"las")
    if recorded is None:
        jobs.write_export_manifest(export_dir, [exported], "local")
    else:
        jobs.write_export_manifest(export_dir, [exported], "local", language=recorded, untranslated=[], variant="ortho", level=None)
    cls = dxf_controller.DxfPageController
    monkeypatch.setattr(cls, "blocked_message", property(lambda self: None))
    monkeypatch.setattr(cls, "profile_required", property(lambda self: False))
    monkeypatch.setattr(cls, "uses_environment_chain", property(lambda self: False))
    monkeypatch.setattr(cls, "mode", property(lambda self: "outdoor"))
    monkeypatch.setattr(cls, "las_path", property(lambda self: las))
    monkeypatch.setattr(cls, "plan_status", lambda self, level_id, variant="pointcloud": "ok")
    return project


@pytest.mark.parametrize(("recorded", "expected"), [("de", True), ("en", False), (None, False)])
def test_controller_redoes_only_the_export_after_a_language_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str | None, expected: bool) -> None:
    from samcloud.app.dxf_controller import DxfPageController

    project = _controller_project(tmp_path, monkeypatch, recorded)
    runner = _FakeRunner()
    controller = DxfPageController(project, runner=runner)  # the page passes its runner; the UI language is English
    assert runner.started == []  # constructing the controller never starts a job
    assert controller.sync_export_language() is expected  # the page calls this once (DxfPage.build)
    if expected:
        assert runner.started == [("export", {"las": project / "cloud.las", "level": None, "mode": "outdoor", "variant": "ortho", "language": "en"})]
    else:
        assert runner.started == []
    assert controller.export_language_outdated(None) is expected
    route_runners: list[_FakeRunner] = []
    monkeypatch.setattr("samcloud.app.dxf_controller.DxfJobRunner", lambda _project: route_runners.append(_FakeRunner()) or route_runners[-1])
    DxfPageController(project)  # HTTP routes build controllers without a runner: they never start jobs
    assert len(route_runners) == 1 and route_runners[0].started == []


# --------------------------------------------------------------------------- existing DXF files without a plan cache


def _file_texts(path: Path) -> list[str]:
    """Every TEXT, MTEXT and attribute text of the modelspace, in file order."""
    document = ezdxf.readfile(str(path))
    texts: list[str] = []
    for entity in document.modelspace():
        if entity.dxftype() == "TEXT":
            texts.append(str(entity.dxf.text))
        elif entity.dxftype() == "MTEXT":
            texts.append(entity.plain_text())
        elif entity.dxftype() == "INSERT":
            texts.extend(f"{attrib.dxf.tag}={attrib.dxf.text}" for attrib in entity.attribs)
    return texts


def _german_outdoor_dxf(directory: Path, name: str = "Drone_Site_clean.dxf") -> Path:
    from samcloud.dxf.text_language import translate_document_notes
    from samcloud.dxf.writer import build_dxf_document

    directory.mkdir(parents=True, exist_ok=True)
    document = build_dxf_document(translate_result(_outdoor_result(), "de")[0])
    translate_document_notes(document, "de")
    document.modelspace().add_mtext("Gebüsch h=0.8 m", dxfattribs={"layer": "S-ANNO", "char_height": 0.3}).set_location((2.013, 3.017))
    target = directory / name
    document.saveas(target)
    return target


def test_relabel_round_trip_restores_the_texts(tmp_path: Path) -> None:
    from samcloud.dxf.text_language import dxf_file_language, language_sidecar_path, relabel_dxf_file

    path = _german_outdoor_dxf(tmp_path / "export")
    original = _file_texts(path)
    assert any("KRONE=" in text for text in original) and "Gebüsch h=0.8 m" in original
    assert dxf_file_language(path) is None
    assert relabel_dxf_file(path, "en") == []
    english = _file_texts(path)
    assert dxf_file_language(path) == "en" and language_sidecar_path(path).is_file()
    labels = [text.split("=", 1)[1] if text.split("=", 1)[0] in {"LABEL", "TYP", "KRONE", "STAMM", "HOEHE"} else text for text in english]
    assert not [(text, word) for text in labels for word in GERMAN_WORDS if _contains_word(text, word)]
    assert "Shrub h=0.8 m" in english and any("unknown profile" in text for text in english)
    assert [_NUMBER.findall(text) for text in english] == [_NUMBER.findall(text) for text in original]
    assert relabel_dxf_file(path, "de") == []
    assert _file_texts(path) == original
    assert dxf_file_language(path) == "de"
    # A rewritten file (new drone run) makes the sidecar stale: the language is unknown again.
    _german_outdoor_dxf(tmp_path / "export")
    assert dxf_file_language(path) is None


def test_relabel_without_changes_keeps_the_file_bytes(tmp_path: Path) -> None:
    from samcloud.dxf.text_language import dxf_file_language, relabel_dxf_file

    path = _german_outdoor_dxf(tmp_path / "export")
    before = path.read_bytes()
    relabel_dxf_file(path, "de")
    assert path.read_bytes() == before and dxf_file_language(path) == "de"
    assert jobs.exported_file_language(tmp_path, path) == "de"


def test_export_plan_post_build_runs_before_saving(tmp_path: Path) -> None:
    result = _make_result({"A-ANNO": [_text("Küche  12.4 m²")]})
    seen: list[Any] = []

    def mark(document: Any) -> None:
        seen.append(document)
        document.modelspace().add_text("MARKER", dxfattribs={"layer": "X-NOTE"})

    path = export_plan(result, tmp_path / "hooked", post_build=mark)
    assert len(seen) == 1 and "MARKER" in _file_texts(path)
    plain = export_plan(result, tmp_path / "plain")
    assert "MARKER" not in _file_texts(plain)
    assert _stable_dxf(plain) == _stable_dxf(export_plan(result, tmp_path / "plain_again", post_build=None))


def test_controller_relabels_an_uncached_drone_dxf_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app.dxf_controller import DxfPageController
    from samcloud.dxf.text_language import dxf_file_language

    project = _controller_project(tmp_path, monkeypatch, None)
    from samcloud.dxf.text_language import relabel_dxf_file

    drone = _german_outdoor_dxf(project / "products" / "dxf" / "export", "CtrlProject_site_clean.dxf")
    before = drone.read_bytes()
    DxfPageController(project, runner=_FakeRunner()).sync_export_language()
    assert drone.read_bytes() == before  # language never recorded: the page leaves the file alone
    relabel_dxf_file(drone, "de")  # e.g. a share in the German UI recorded it
    runner = _FakeRunner()
    controller = DxfPageController(project, runner=runner)
    assert dxf_file_language(drone) == "de"  # constructing the controller changes nothing
    controller.sync_export_language()
    assert runner.started == []  # no export job: the file has no clean plan cache
    assert dxf_file_language(drone) == "en" and "Shrub h=0.8 m" in _file_texts(drone)
    assert controller.export_language_outdated(None) is False


def test_review_payload_passes_the_language_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app import dxf_review_canvas

    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(dxf_review_canvas.DxfPageController, "__init__", lambda self, *_a, **_k: None)
    monkeypatch.setattr(dxf_review_canvas.DxfPageController, "plan_payload", lambda self, level, variant, language=None: seen.append((level, variant, language)) or {})
    dxf_review_canvas.review_payload(tmp_path, "site", "ortho", language="de")
    dxf_review_canvas.review_payload(tmp_path, "lvl_1")
    assert seen == [(None, "ortho", "de"), ("lvl_1", "pointcloud", None)]
