"""Build config/classes.json from a Nomivox class preset (presets/default.json).

Indoor uses the Nomivox classes with scene ``indoor`` or ``both``, outdoor
those with ``outdoor`` or ``both``, each in preset order. Every class keeps
its Nomivox LAS code, colour, German label and its full prompt list; SAM3
receives each prompt separately and a class scores the best of its prompts
(long comma-joined prompt strings made SAM3's presence score collapse).

Usage::

    python -m samcloud.tools.import_nomivox_classes D:\\Nomivox\\presets\\default.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from samcloud.core.paths import CONFIG_DIRECTORY

SCENES = {"indoor": ("indoor", "both"), "outdoor": ("outdoor", "both")}


def build_classes(preset: dict, source: str) -> dict:
    """Convert a Nomivox preset into SAMcloud's classes.json structure."""
    special = {item["id"]: item for item in preset.get("special", [])}
    unclassified = special.get("unclassified", {"code": preset.get("unclassified_code", 1), "color": "#000000"})
    result: dict = {
        "schema_version": 2,
        "source": f"{source} ({preset.get('preset_name', 'Nomivox preset')}, imported {date.today().isoformat()})",
        "unknown_class_id": 0,
        "unknown_class_name": "Unknown/noise",
        "prompts": {},
        "las_codes": {},
        "colors": {},
        "labels_de": {},
    }
    for class_set, scenes in SCENES.items():
        classes = [item for item in preset["classes"] if item.get("enabled", True) and item.get("scene") in scenes]
        names = [item["id"].replace("_", " ") for item in classes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate class ids in {class_set}")
        result[class_set] = [", ".join([name, *item["prompts"]]) for name, item in zip(names, classes)]
        result["prompts"][class_set] = [list(item["prompts"]) for item in classes]
        result["las_codes"][class_set] = [int(unclassified["code"]), *(int(item["code"]) for item in classes)]
        result["colors"][class_set] = [unclassified.get("color", "#000000"), *(item["color"] for item in classes)]
        result["labels_de"][class_set] = ["Unklassifiziert", *(item["name_de"] for item in classes)]
        if len(classes) > 254:
            raise ValueError("class ids are stored as uint8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("preset", type=Path)
    parser.add_argument("--out", type=Path, default=CONFIG_DIRECTORY / "classes.json")
    args = parser.parse_args()
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    classes = build_classes(preset, str(args.preset))
    args.out.write_text(json.dumps(classes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for class_set in SCENES:
        print(f"{class_set}: {len(classes[class_set])} classes, "
              f"{sum(len(p) for p in classes['prompts'][class_set])} prompts")


if __name__ == "__main__":
    main()
