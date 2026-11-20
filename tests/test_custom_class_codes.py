"""LAS code allocation: 82 Road marking, 83 Path, user classes only at codes >= 200."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from samcloud.classification.custom import (
    CUSTOM_LAS_CODE_START,
    build_custom_classes,
    migrate_classes_config,
)
from samcloud.classification.profiles import load_cyclone_codes
from samcloud.classification.service import class_las_codes

GLOBAL = json.loads(Path("config/classes.json").read_text(encoding="utf-8"))


def _added(count: int, prefix: str = "extra") -> dict:
    return {"added": [{"name": f"{prefix} {index}", "prompts": [f"{prefix} {index}"]} for index in range(count)]}


def test_new_custom_class_gets_code_from_200():
    config = build_custom_classes("outdoor", _added(1))
    assert CUSTOM_LAS_CODE_START == 200
    assert class_las_codes("outdoor", config)[-1] >= 200


def test_many_custom_classes_never_below_200_or_82_83():
    config = build_custom_classes("outdoor", _added(20))
    codes = class_las_codes("outdoor", config)
    new = codes[len(GLOBAL["outdoor"]) + 1:]
    assert len(new) == 20 and len(set(new)) == 20
    assert all(code >= 200 and code not in (82, 83) for code in new)


def test_existing_project_codes_are_not_renumbered():
    config = deepcopy(GLOBAL)
    base = len(config["outdoor"])
    config["outdoor"] += ["old path, old path", "old thing, old thing"]
    config["las_codes"]["outdoor"] = list(config["las_codes"]["outdoor"]) + [83, 120]
    migrated = migrate_classes_config(config)
    assert migrated["las_codes"]["outdoor"][base + 1:] == [83, 120]
    assert migrated["las_codes"]["outdoor"][:base + 1] == GLOBAL["las_codes"]["outdoor"][:base + 1]


def test_migration_assigns_new_custom_class_from_200():
    config = deepcopy(GLOBAL)
    config["outdoor"] += ["old path, old path", "fresh, fresh"]
    config["las_codes"]["outdoor"] = list(config["las_codes"]["outdoor"]) + [83]
    migrated = migrate_classes_config(config)
    assert migrated["las_codes"]["outdoor"][-2] == 83
    assert migrated["las_codes"]["outdoor"][-1] >= 200


def test_cyclone_codes_82_road_marking_83_path():
    registry = load_cyclone_codes()
    assert registry[82]["name"] == "Road marking" and not registry[82].get("custom")
    assert registry[83]["name"] == "Path" and not registry[83].get("custom")


def test_limit_error_when_slots_run_out():
    slots = [code for code, entry in load_cyclone_codes().items()
             if entry.get("custom") and code >= CUSTOM_LAS_CODE_START]
    used = set(GLOBAL["las_codes"]["outdoor"])
    free = [code for code in slots if code not in used]
    assert free
    build_custom_classes("outdoor", _added(len(free)))
    with pytest.raises(ValueError, match="Keine freien LAS-Klassencodes"):
        build_custom_classes("outdoor", _added(len(free) + 1))
