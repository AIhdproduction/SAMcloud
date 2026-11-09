"""Per-project SAM3 class customisation (edited prompts and added classes).

A project with customised classes gets its own ``config/classes.json`` that every pipeline step,
the LAS export and the viewers read instead of the global ``config/classes.json``. Standard classes
keep their ids and names (step 6 and the geometry rules address them by name); own classes are
appended, so existing ids never shift. FlurSeg has a fixed class set and ignores customisation.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from samcloud.core.paths import CONFIG_DIRECTORY

PROJECT_CLASSES_RELATIVE = Path("config") / "classes.json"
CUSTOM_LAS_CODE_START = 64  # first LAS 1.4 user-definable class code
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
    base = deepcopy(config if config is not None else json.loads(global_classes_path().read_text(encoding="utf-8")))
    custom = normalize_customization(customization, class_set, base)
    standard = standard_classes(class_set, base)
    prompts = [custom["prompts"].get(item["name"], item["prompts"]) for item in standard]
    entries = [", ".join([item["name"], *values]) for item, values in zip(standard, prompts)]
    las_codes = list(base.get("las_codes", {}).get(class_set) or [])
    colors = list(base.get("colors", {}).get(class_set) or [])
    labels = list(base.get("labels_de", {}).get(class_set) or [])
    used_codes = set(las_codes)
    next_code = CUSTOM_LAS_CODE_START
    for item in custom["added"]:
        entries.append(", ".join([item["name"], *item["prompts"]]))
        prompts.append(item["prompts"])
        while next_code in used_codes:
            next_code += 1
        if next_code > 255:
            raise ValueError("Keine freien LAS-Klassencodes mehr")
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
    path.write_text(json.dumps(build_custom_classes(class_set, customization), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path
