"""Translate the human-readable texts of a DXF plan result into the UI language (``de`` or ``en``).

The pipeline bakes its labels into the cached :class:`~samcloud.dxf.result.DxfPlanResult` (mostly German,
a few English ones such as ``Window 0.90 m`` or ``Stairs``) and pipeline logic keys on those German names,
so the cache never changes. Translation happens only when a result is exported or sent to the viewer:

* :func:`translate_text` maps one text with deterministic rule tables,
* :func:`translate_result` returns a NEW result with translated ``text`` entities and insert attribute values,
* :func:`translate_note` / :func:`translate_document_notes` handle the X-NOTE header line of the DXF writer,
* :func:`export_plan_localized` is :func:`samcloud.dxf.writer.export_plan` with the translation applied,
* :func:`relabel_dxf_file` translates an existing DXF file in place (no plan cache, e.g. the drone DXF) and records
  the language in ``<file>.lang.json`` (:func:`dxf_file_language`).

The tables live in ``samcloud/app/locales/<lang>/dxf_labels.json`` (identical keys in both languages) and are
read directly, without NiceGUI or the i18n module, so the DXF job subprocess stays light:

* ``names``: canonical key -> display name (rooms, furniture, fixtures, built-ins, abbreviations, outdoor areas).
  The reverse lookup accepts the German AND the English spelling (plus the raw English fine labels in
  :data:`SOURCE_ALIASES`), so a text is translated whichever language was baked.
* ``patterns``: composite texts with ``str.format`` placeholders. ``{name}`` must be a known name, ``{value}`` is a
  known name or a number-like token that is kept, every other placeholder is a number token kept as the exact
  original substring (numbers are never reformatted).
* ``note``: the parts of the X-NOTE header line.

Texts that match no rule stay unchanged and are reported; pure numbers (spot heights, contour levels) are neither
translated nor reported.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import re
import string
import tempfile
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from samcloud.dxf.result import DxfPlanResult

LOGGER = logging.getLogger(__name__)

SUPPORTED_DXF_LANGUAGES = ("de", "en")
LOCALES_DIRECTORY = Path(__file__).resolve().parents[1] / "app" / "locales"
LABELS_FILE = "dxf_labels.json"
NOTE_LAYER = "X-NOTE"
NOTE_SEPARATOR = " | "

# Insert attributes whose values are words (all other attributes are numbers, codes or instance ids).
TEXT_ATTRIBUTE_TAGS = frozenset({"LABEL", "label", "TYP", "KRONE", "STAMM", "HOEHE"})

# Most specific patterns first; ``height_label`` ("{name} h={height}") is the catch-all and comes last.
PATTERN_ORDER: tuple[str, ...] = (
    "room_area", "room_generic", "door_head", "door", "passage_head", "passage", "window", "tree_label",
    "tree_crown_text", "tree_trunk_text", "shrub_text", "roof_flat", "roof_ridge", "tree_crown_attr",
    "tree_trunk_attr", "tree_height_attr", "height_label",
)

# Raw source spellings (English fine labels, Cyclone class names) that are neither the de nor the en display name.
SOURCE_ALIASES: dict[str, str] = {
    "couch": "sofa", "mattress": "bed", "wardrobe": "cabinet", "tv stand": "sideboard", "television": "tv",
    "flat screen tv": "tv", "pillow": "cushion", "duvet": "blanket", "fridge": "refrigerator", "kitchen counter": "kitchen",
    "kitchen cabinet": "kitchen", "kitchen sink": "sink", "toilet": "wc", "wash basin": "washbasin", "bath tub": "bathtub",
    "shower tray": "shower", "computer monitor": "monitor", "computer screen": "monitor", "flat screen": "screen",
    "ceiling lamp": "ceiling_light", "chandelier": "pendant_light", "ceiling spotlight": "spotlight",
    "track light": "spotlight", "downlight": "spotlight", "sprinkler": "sprinkler", "ceiling projector": "projector",
    "table (desk)": "desk", "side walk": "sidewalk", "low vegetation": "low_vegetation",
    "hcs construction material": "construction_material", "road surface": "road_surface",
}

_WORD = re.compile(r"[A-Za-zÄÖÜäöüß]{2,}")
_UNIT_WORDS = frozenset({"mm", "cm", "km"})
_NUMBER = r"\S+"


def normalize_dxf_language(language: object) -> str | None:
    """Return ``"de"`` / ``"en"`` for values such as ``"DE"`` or ``"en-US"``; None for None or unsupported values."""
    if not isinstance(language, str):
        return None
    code = language.strip().lower().replace("_", "-").split("-", 1)[0]
    return code if code in SUPPORTED_DXF_LANGUAGES else None


def current_dxf_language() -> str | None:
    """The UI language DXF texts are written in ("de" / "en"); None when the i18n module is unavailable."""
    try:
        from samcloud.app.i18n import current_language

        return normalize_dxf_language(current_language())
    except Exception:  # noqa: BLE001 - no UI language: callers keep the cached texts
        return None


def _require_language(language: str) -> str:
    code = normalize_dxf_language(language)
    if code is None:
        raise ValueError(f"Unsupported DXF text language: {language!r}")
    return code


@lru_cache(maxsize=4)
def _raw_tables(language: str) -> dict[str, dict[str, str]]:
    path = LOCALES_DIRECTORY / language / LABELS_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    return {section: {str(key): str(value) for key, value in dict(data.get(section, {})).items()} for section in ("names", "patterns", "note")}


class _Tables:
    """Name lookup and compiled patterns of both languages."""

    def __init__(self) -> None:
        self.names = {language: _raw_tables(language)["names"] for language in SUPPORTED_DXF_LANGUAGES}
        self.patterns = {language: _raw_tables(language)["patterns"] for language in SUPPORTED_DXF_LANGUAGES}
        self.note = {language: _raw_tables(language)["note"] for language in SUPPORTED_DXF_LANGUAGES}
        self.exact: dict[str, str] = {}
        self.folded: dict[str, str] = {}
        for language in SUPPORTED_DXF_LANGUAGES:
            for key, text in self.names[language].items():
                self._register(text, key)
        for alias, key in SOURCE_ALIASES.items():
            if key in self.names["en"]:
                self._register(alias, key)
        for key in self.names["en"]:  # the fine label spelling of a canonical key, e.g. "coffee table"
            self._register(key.replace("_", " "), key)
        self.compiled: list[tuple[str, re.Pattern[str]]] = []
        for key in PATTERN_ORDER:
            for language in SUPPORTED_DXF_LANGUAGES:
                template = self.patterns[language].get(key)
                if template:
                    self.compiled.append((key, _compile(template)))
        self.note_compiled = {key: [_compile(self.note[language][key]) for language in SUPPORTED_DXF_LANGUAGES] for key in ("scale", "variant", "offset")}

    def _register(self, text: str, key: str) -> None:
        self.exact.setdefault(text, key)
        self.folded.setdefault(text.casefold(), key)

    def name_key(self, text: str) -> str | None:
        key = self.exact.get(text)
        if key is None:
            key = self.folded.get(text.strip().casefold())
        return key


def _compile(template: str) -> re.Pattern[str]:
    """Regex for a ``str.format`` template: ``{name}`` / ``{value}`` match lazily, other fields one number token."""
    parts: list[str] = []
    for literal, field, _spec, _conversion in string.Formatter().parse(template):
        parts.append(re.escape(literal))
        if field is None:
            continue
        parts.append(f"(?P<{field}>.+?)" if field in {"name", "value"} else f"(?P<{field}>{_NUMBER})")
    return re.compile("".join(parts) + r"\Z", re.DOTALL)


@lru_cache(maxsize=1)
def _tables() -> _Tables:
    return _Tables()


def reload_tables() -> None:
    """Drop the cached rule tables (after editing the locale files)."""
    _raw_tables.cache_clear()
    _tables.cache_clear()


def _is_number_like(text: str) -> bool:
    stripped = text.strip()
    return stripped == "n/a" or bool(re.search(r"\d", stripped))


def has_words(text: str) -> bool:
    """True when ``text`` contains a word (two or more letters other than a unit), i.e. is not a pure number."""
    cleaned = str(text).replace("m²", " ").replace("n/a", " ")
    return any(word.lower() not in _UNIT_WORDS for word in _WORD.findall(cleaned))


def _translate_name(text: str, language: str) -> str | None:
    tables = _tables()
    key = tables.name_key(text)
    if key is None:
        return None
    return tables.names[language].get(key)


def _translate_known(text: str, language: str) -> str | None:
    """Translation of ``text`` by the name table or a pattern; None when no rule matches."""
    named = _translate_name(text, language)
    if named is not None:
        return named
    tables = _tables()
    for key, pattern in tables.compiled:
        match = pattern.match(text)
        if match is None:
            continue
        values: dict[str, str] = {}
        ok = True
        for field, captured in match.groupdict().items():
            if field == "name":
                translated = _translate_name(captured, language)
                if translated is None:
                    ok = False
                    break
                values[field] = translated
            elif field == "value":
                translated = _translate_name(captured, language)
                if translated is None and not _is_number_like(captured):
                    ok = False
                    break
                values[field] = captured if translated is None else translated
            else:
                values[field] = captured
        if ok:
            return tables.patterns[language][key].format(**values)
    return None


def translate_text(text: str, lang: str) -> str:
    """Translate one DXF label text into ``lang`` ("de" or "en"); unknown texts are returned unchanged."""
    language = _require_language(lang)
    source = str(text)
    translated = _translate_known(source, language)
    return source if translated is None else translated


def _translate_reporting(text: str, language: str, missing: dict[str, None]) -> str:
    translated = _translate_known(text, language)
    if translated is not None:
        return translated
    if has_words(text):
        missing.setdefault(text, None)
    return text


def translate_result(result: DxfPlanResult, lang: str) -> tuple[DxfPlanResult, list[str]]:
    """Return a NEW result with translated ``text`` entities and insert attribute values, and the unmatched texts.

    The input result (often memoised by the GUI) is never mutated: changed entities are shallow copies.
    Only the attribute tags in :data:`TEXT_ATTRIBUTE_TAGS` are translated; numbers and codes stay as they are.
    """
    language = _require_language(lang)
    missing: dict[str, None] = {}
    layers: dict[str, list[dict[str, Any]]] = {}
    for layer, entities in result.layers.items():
        output: list[dict[str, Any]] = []
        for entity in entities:
            kind = entity.get("type")
            if kind == "text" and isinstance(entity.get("text"), str):
                translated = _translate_reporting(entity["text"], language, missing)
                output.append(entity if translated == entity["text"] else {**entity, "text": translated})
            elif kind == "insert" and isinstance(entity.get("attributes"), dict):
                attributes = dict(entity["attributes"])
                changed = False
                for tag, value in entity["attributes"].items():
                    if tag in TEXT_ATTRIBUTE_TAGS and isinstance(value, str) and value:
                        translated = _translate_reporting(value, language, missing)
                        if translated != value:
                            attributes[tag] = translated
                            changed = True
                output.append({**entity, "attributes": attributes} if changed else entity)
            else:
                output.append(entity)
        layers[layer] = output
    return replace(result, layers=layers), list(missing)


def localize_result(result: Any, lang: str | None) -> Any:
    """``translate_result(result, lang)[0]``; returns ``result`` unchanged for ``lang=None`` or a non-plan object."""
    if lang is None or not hasattr(result, "layers"):
        return result
    return translate_result(result, lang)[0]


def translate_note(text: str, lang: str) -> str:
    """Translate the X-NOTE header line ("project | level | profile | scale ... | date ...") part by part.

    Project and level names, the profile name and the date are user data and stay unchanged.
    """
    language = _require_language(lang)
    tables = _tables()
    note = tables.note[language]
    parts: list[str] = []
    for part in str(text).split(NOTE_SEPARATOR):
        if part in {table["unknown_profile"] for table in tables.note.values()}:
            parts.append(note["unknown_profile"])
            continue
        for key, patterns in tables.note_compiled.items():
            match = next((found for found in (pattern.match(part) for pattern in patterns) if found is not None), None)
            if match is not None:
                parts.append(note[key].format(**match.groupdict()))
                break
        else:
            parts.append(part)
    return NOTE_SEPARATOR.join(parts)


def translate_document_notes(document: Any, lang: str) -> None:
    """Translate every TEXT on the X-NOTE layer of an ezdxf document in place (no-op for objects without a modelspace)."""
    if not callable(getattr(document, "modelspace", None)):
        return
    for entity in document.modelspace().query(f'TEXT[layer=="{NOTE_LAYER}"]'):
        entity.dxf.text = translate_note(str(entity.dxf.text), lang)


def export_plan_localized(result: DxfPlanResult, out_dir: Path, lang: str, coordinates: str = "local") -> tuple[Path, list[str]]:
    """:func:`samcloud.dxf.writer.export_plan` with every text in ``lang``; returns (path, untranslated texts).

    The translated result goes through ``export_plan`` itself (same file name, atomic write and re-read audit); its
    ``post_build`` hook translates the X-NOTE line the writer adds.
    """
    from samcloud.dxf.writer import export_plan

    language = _require_language(lang)
    translated, missing = translate_result(result, language)
    destination = export_plan(translated, out_dir, coordinates=coordinates,
                              post_build=lambda document: translate_document_notes(document, language))
    return destination, missing


# --------------------------------------------------------------------------- existing DXF files (no plan cache)


def language_sidecar_path(path: Path | str) -> Path:
    """``<file>.lang.json``: the text language of a DXF file that was relabelled in place."""
    target = Path(path)
    return target.with_name(f"{target.name}.lang.json")


def dxf_file_language(path: Path | str) -> str | None:
    """Language recorded by :func:`relabel_dxf_file` for ``path``; None when unknown or the file changed since."""
    target = Path(path)
    try:
        data = json.loads(language_sidecar_path(target).read_text(encoding="utf-8"))
        stat = target.stat()
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("mtime_ns") != stat.st_mtime_ns or data.get("size") != stat.st_size:
        return None  # the file was rewritten (new export, new drone run): the sidecar no longer describes it
    return normalize_dxf_language(data.get("language"))


def record_dxf_language(path: Path | str, lang: str, untranslated: list[str] | None = None) -> Path:
    """Record that the DXF file ``path`` (as it is now) holds texts in ``lang``; returns the sidecar path.

    For writers of DXF files that are not exports (e.g. the drone chain's ``outdoor.dxf``, already translated).
    """
    target = Path(path)
    _write_sidecar(target, _require_language(lang), list(untranslated or []))
    return language_sidecar_path(target)


def copy_dxf_language(source: Path | str, target: Path | str) -> str | None:
    """Give the copy ``target`` the recorded language of ``source`` (None and nothing written when it has none)."""
    language = dxf_file_language(source)
    if language is None:
        return None
    try:
        untranslated = list(json.loads(language_sidecar_path(source).read_text(encoding="utf-8")).get("untranslated") or [])
    except (OSError, ValueError, AttributeError):
        untranslated = []
    record_dxf_language(target, language, untranslated)
    return language


def _write_sidecar(path: Path, language: str, missing: list[str]) -> None:
    stat = path.stat()
    sidecar = language_sidecar_path(path)
    temporary = sidecar.with_name(f".{sidecar.name}.tmp")
    payload = {"language": language, "untranslated": missing, "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, sidecar)


def _relabel_value(value: str, language: str, missing: dict[str, None], *, note: bool = False) -> str:
    if not value:
        return value
    return translate_note(value, language) if note else _translate_reporting(value, language, missing)


def _relabel_entity(entity: Any, language: str, missing: dict[str, None]) -> bool:
    """Translate one TEXT / MTEXT / ATTRIB / INSERT (its text attributes) in place; True when something changed."""
    kind = entity.dxftype()
    changed = False
    if kind == "TEXT":
        text = str(entity.dxf.text)
        translated = _relabel_value(text, language, missing, note=entity.dxf.layer == NOTE_LAYER)
        if translated != text:
            entity.dxf.text = translated
            changed = True
    elif kind == "MTEXT":
        plain = entity.plain_text()
        translated = _relabel_value(plain, language, missing)
        if translated != plain:
            entity.text = translated  # a known label carries no inline formatting worth keeping
            changed = True
    elif kind in {"INSERT", "ATTRIB"}:
        attribs = list(entity.attribs) if kind == "INSERT" else [entity]
        for attrib in attribs:
            if attrib.dxf.tag in TEXT_ATTRIBUTE_TAGS:
                value = str(attrib.dxf.text)
                translated = _relabel_value(value, language, missing)
                if translated != value:
                    attrib.dxf.text = translated
                    changed = True
    return changed


def relabel_dxf_file(path: Path | str, lang: str) -> list[str]:
    """Translate the texts of an existing DXF file in place (TEXT, MTEXT, insert attributes, X-NOTE line).

    For DXF files without a plan cache to export from (e.g. the drone ``*_Site_clean.dxf``). The rule tables accept
    both spellings, so de -> en -> de restores the texts. The file is rewritten (atomically, re-read and audited) only
    when a text changed; the language is recorded in ``<file>.lang.json`` (see :func:`dxf_file_language`).
    Returns the texts without a translation rule.
    """
    import ezdxf

    from samcloud.dxf.writer import _audit

    language = _require_language(lang)
    target = Path(path)
    document = ezdxf.readfile(str(target))
    missing: dict[str, None] = {}
    changed = False
    for entity in document.modelspace():
        changed = _relabel_entity(entity, language, missing) or changed
    if changed:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            document.saveas(temporary)
            _audit(ezdxf.readfile(temporary))
            os.replace(temporary, target)
        except Exception as error:
            if temporary.exists():
                temporary.unlink()
            raise RuntimeError(f"DXF relabel failed: {error}") from error
    _write_sidecar(target, language, list(missing))
    return list(missing)


def table_collisions() -> list[str]:
    """Source spellings that map to canonical keys with different translations (must stay empty)."""
    tables = _tables()
    seen: dict[str, set[tuple[str, str]]] = {}
    sources: Iterable[tuple[str, str]] = [(text, key) for language in SUPPORTED_DXF_LANGUAGES for key, text in tables.names[language].items()]
    for text, key in list(sources) + list(SOURCE_ALIASES.items()):
        seen.setdefault(text.casefold(), set()).add((tables.names["de"][key], tables.names["en"][key]))
    return sorted(text for text, outputs in seen.items() if len(outputs) > 1)
