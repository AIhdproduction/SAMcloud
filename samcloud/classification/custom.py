"""Per-project SAM3 class customisation (edited prompts and added classes).

A project with customised classes gets its own ``config/classes.json`` that every pipeline step,
the LAS export and the viewers read instead of the global ``config/classes.json``. Standard classes
keep their ids and names (step 6 and the geometry rules address them by name); own classes are
appended, so existing ids never shift. FlurSeg has a fixed class set and ignores customisation.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from typing import Any

from samcloud.core.paths import CONFIG_DIRECTORY
from samcloud.classification.profiles import load_cyclone_codes

PROJECT_CLASSES_RELATIVE = Path("config") / "classes.json"
CUSTOM_LAS_CODE_START = 82
CUSTOM_CLASS_LIMIT = 255
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def global_classes_path() -> Path:
    return CONFIG_DIRECTORY / "classes.json"


def project_classes_path(project_directory: Path | str | None) -> Path:
    """The project's own classes.json when it exists, otherwise the global one."""
    if project_directory:
        candidate = Path(project_directory) / PROJECT_CLASSES_RELATIVE
        if candidate.is_file():
            return candidate
    return global_classes_path()


def migrate_classes_config(config: dict) -> dict:
    """Return an in-memory Cyclone-safe migration of a legacy project class configuration."""
    config = deepcopy(config)
    global_config = json.loads(global_classes_path().read_text(encoding="utf-8"))
    custom_codes = [code for code, entry in load_cyclone_codes().items() if entry.get("custom") and code != 82]
    for class_set in ("outdoor", "indoor"):
        codes = config.get("las_codes", {}).get(class_set)
        entries = config.get(class_set)
        if not isinstance(codes, list) or not isinstance(entries, list):
            continue
        global_codes = global_config["las_codes"][class_set]
        standard_codes = {
            class_name(entry): global_codes[index]
            for index, entry in enumerate(global_config[class_set], start=1)
        }
        migrated = [global_codes[0]]
        used = {global_codes[0]}
        for index, entry in enumerate(entries, start=1):
            name = class_name(entry)
            if name in standard_codes:
                migrated.append(standard_codes[name])
                used.add(standard_codes[name])
                continue
            current = codes[index] if index < len(codes) else None
            if current not in custom_codes or current in used:
                current = next((code for code in custom_codes if code not in used), None)
                if current is None:
                    raise ValueError("Keine freien LAS-Klassencodes mehr")
            migrated.append(current)
            used.add(current)
        config.setdefault("las_codes", {})[class_set] = migrated
    return config


def load_project_classes(project_directory: Path | str | None) -> dict:
    """Load effective project classes and migrate legacy codes in memory only."""
    return migrate_classes_config(json.loads(project_classes_path(project_directory).read_text(encoding="utf-8")))


def _migrate_custom_codes(path: Path) -> dict:
    """Compatibility wrapper returning the in-memory migration without writing it."""
    return migrate_classes_config(json.loads(path.read_text(encoding="utf-8")))


def _atomic_write_classes(path: Path, config: dict) -> None:
    """Persist an explicitly rebuilt project class configuration atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def class_name(entry: str) -> str:
    """Class name of a ``"name, prompt, prompt"`` class-set entry."""
    return entry.split(",", 1)[0].strip()


def standard_classes(class_set: str, config: dict | None = None) -> list[dict[str, Any]]:
    """Standard classes of a class set: name, German label and prompts (for the class editor)."""
    config = config if config is not None else json.loads(global_classes_path().read_text(encoding="utf-8"))
    entries = config[class_set]
    prompts = config.get("prompts", {}).get(class_set) or [
        [part.strip() for part in entry.split(",")[1:] if part.strip()] for entry in entries
    ]
    labels = config.get("labels_de", {}).get(class_set) or []
    return [
        {"name": class_name(entry), "label": labels[index + 1] if index + 1 < len(labels) else class_name(entry),
         "prompts": list(prompts[index])}
        for index, entry in enumerate(entries)
    ]


def _clean_prompts(values: Any, owner: str) -> list[str]:
    if isinstance(values, str):
        values = values.split(",")
    prompts = []
    for value in values or []:
        prompt = str(value).strip()
        if prompt and prompt not in prompts:
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"Klasse '{owner}' braucht mindestens einen Prompt")
    return prompts


def normalize_customization(customization: dict | None, class_set: str, config: dict | None = None) -> dict:
    """Validate a customisation; returns ``{"prompts": {...}, "added": [...]}`` with only real changes."""
    config = config if config is not None else json.loads(global_classes_path().read_text(encoding="utf-8"))
    if class_set not in {"outdoor", "indoor"}:
        raise ValueError(f"Unsupported class set: {class_set}")
    customization = customization or {}
    defaults = {item["name"]: item for item in standard_classes(class_set, config)}
    prompts: dict[str, list[str]] = {}
    for name, values in (customization.get("prompts") or {}).items():
        if name not in defaults:
            raise ValueError(f"Unbekannte Standardklasse: {name}")
        cleaned = _clean_prompts(values, name)
        if cleaned != defaults[name]["prompts"]:
            prompts[name] = cleaned
    added: list[dict[str, Any]] = []
    names = {name.lower() for name in defaults}
    for item in customization.get("added") or []:
        name = " ".join(str(item.get("name", "")).strip().lower().split())
        if not name:
            raise ValueError("Eigene Klasse braucht einen Namen")
        if "," in name:
            raise ValueError(f"Klassenname darf kein Komma enthalten: {name}")
        if name in names:
            raise ValueError(f"Klasse existiert bereits: {name}")
        names.add(name)
        color = str(item.get("color") or "#FF00FF").upper()
        if not _HEX_COLOR.match(color):
            raise ValueError(f"Ungültige Farbe für '{name}': {color}")
        added.append({
            "name": name,
            "label": str(item.get("label") or "").strip() or name.title(),
            "prompts": _clean_prompts(item.get("prompts") or [name], name),
            "color": color,
        })
    if len(defaults) + len(added) > CUSTOM_CLASS_LIMIT:
        raise ValueError(f"Höchstens {CUSTOM_CLASS_LIMIT} Klassen möglich")
    return {"prompts": prompts, "added": added}


def is_customized(customization: dict | None) -> bool:
    return bool(customization and (customization.get("prompts") or customization.get("added")))


def build_custom_classes(class_set: str, customization: dict | None, config: dict | None = None) -> dict:
    """Return a full classes.json with the customisation applied to ``class_set``."""
    base = migrate_classes_config(config if config is not None else json.loads(global_classes_path().read_text(encoding="utf-8")))
    custom = normalize_customization(customization, class_set, base)
    standard = standard_classes(class_set, base)
    prompts = [custom["prompts"].get(item["name"], item["prompts"]) for item in standard]
    entries = [", ".join([item["name"], *values]) for item, values in zip(standard, prompts)]
    las_codes = list(base.get("las_codes", {}).get(class_set) or [])
    colors = list(base.get("colors", {}).get(class_set) or [])
    labels = list(base.get("labels_de", {}).get(class_set) or [])
    used_codes = set(las_codes)
    custom_codes = [code for code, entry in load_cyclone_codes().items() if entry.get("custom") and code != 82]
    available_codes = [code for code in custom_codes if code not in used_codes]
    for item in custom["added"]:
        entries.append(", ".join([item["name"], *item["prompts"]]))
        prompts.append(item["prompts"])
        if not available_codes:
            raise ValueError("Keine freien LAS-Klassencodes mehr")
        next_code = available_codes.pop(0)
        used_codes.add(next_code)
        if las_codes:
            las_codes.append(next_code)
        if colors:
            colors.append(item["color"])
        if labels:
            labels.append(item["label"])
    base[class_set] = entries
    base.setdefault("prompts", {})[class_set] = prompts
    if las_codes:
        base["las_codes"][class_set] = las_codes
    if colors:
        base["colors"][class_set] = colors
    if labels:
        base["labels_de"][class_set] = labels
    base["source"] = f"{base.get('source', 'config/classes.json')} + project customisation"
    return base


def write_project_classes(project_directory: Path, class_set: str, customization: dict | None) -> Path | None:
    """Write the project's classes.json for a customisation; returns None when nothing is customised."""
    if not is_customized(customization):
        return None
    path = Path(project_directory) / PROJECT_CLASSES_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_classes(path, build_custom_classes(class_set, customization))
    return path
