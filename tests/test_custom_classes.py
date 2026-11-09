"""Per-project SAM3 class customisation: edited prompts, own classes, project classes.json."""

import json
from pathlib import Path

import pytest

from samcloud.app.processing import ProcessingManager
from samcloud.classification.custom import (
    CUSTOM_LAS_CODE_START,
    build_custom_classes,
    global_classes_path,
    normalize_customization,
    project_classes_path,
    standard_classes,
)
from samcloud.classification.service import class_default_colors, class_las_codes
from samcloud.core.project_manager import ProjectManager
from samcloud.engine import pipeline

GLOBAL = json.loads(Path("config/classes.json").read_text(encoding="utf-8"))
SOLAR = {"name": "Solar Panel", "label": "Solaranlage", "prompts": "solar panel, photovoltaic module",
         "color": "#ffaa00"}


def test_standard_classes_match_global_config():
    items = standard_classes("outdoor")
    assert [item["name"] for item in items] == [entry.split(",")[0].strip() for entry in GLOBAL["outdoor"]]
    assert items[0]["prompts"] == GLOBAL["prompts"]["outdoor"][0]
    assert items[0]["label"] == GLOBAL["labels_de"]["outdoor"][1]


@pytest.mark.parametrize("class_set", ["outdoor", "indoor"])
def test_build_keeps_ids_and_appends_own_classes(class_set):
    first = standard_classes(class_set)[0]["name"]
    config = build_custom_classes(class_set, {"prompts": {first: "dirt, gravel ,dirt"}, "added": [SOLAR]})
    count = len(GLOBAL[class_set])
    assert len(config[class_set]) == count + 1
    assert config[class_set][0] == f"{first}, dirt, gravel"
    assert config["prompts"][class_set][0] == ["dirt", "gravel"]
    assert config[class_set][1:count] == GLOBAL[class_set][1:count]
    assert config[class_set][-1] == "solar panel, solar panel, photovoltaic module"
    assert config["prompts"][class_set][-1] == ["solar panel", "photovoltaic module"]
    codes = class_las_codes(class_set, config)
    assert len(codes) == count + 2 and codes[-1] >= CUSTOM_LAS_CODE_START
    assert codes[-1] not in codes[:-1]
    assert class_default_colors(class_set, config)[-1] == "#FFAA00"
    assert config["labels_de"][class_set][-1] == "Solaranlage"
    other = "indoor" if class_set == "outdoor" else "outdoor"
    assert config[other] == GLOBAL[other]


def test_two_own_classes_get_distinct_free_las_codes():
    config = build_custom_classes("outdoor", {"added": [SOLAR, {"name": "bench", "prompts": ["bench"]}]})
    codes = class_las_codes("outdoor", config)
    assert len(set(codes[-2:])) == 2 and not set(codes[-2:]) & set(codes[:-2])


def test_normalize_drops_unchanged_prompts_and_defaults_label_and_prompts():
    tree = next(item for item in standard_classes("outdoor") if item["name"] == "tree")
    custom = normalize_customization({"prompts": {"tree": tree["prompts"]}, "added": [{"name": "bench"}]}, "outdoor")
    assert custom == {"prompts": {}, "added": [{"name": "bench", "label": "Bench", "prompts": ["bench"],
                                                "color": "#FF00FF"}]}


@pytest.mark.parametrize("customization,message", [
    ({"prompts": {"nonexistent": ["x"]}}, "Unbekannte"),
    ({"prompts": {"tree": " , "}}, "mindestens einen Prompt"),
    ({"added": [{"name": "Tree"}]}, "existiert bereits"),
    ({"added": [{"name": "bench"}, {"name": "bench"}]}, "existiert bereits"),
    ({"added": [{"name": "a, b"}]}, "Komma"),
    ({"added": [{"name": ""}]}, "Namen"),
    ({"added": [{"name": "bench", "color": "red"}]}, "Farbe"),
])
def test_normalize_rejects_invalid_customization(customization, message):
    with pytest.raises(ValueError, match=message):
        normalize_customization(customization, "outdoor")


def test_project_classes_path_falls_back_to_global(tmp_path):
    assert project_classes_path(None) == global_classes_path()
    assert project_classes_path(tmp_path) == global_classes_path()


def test_create_project_with_sam3_customisation_writes_project_classes(tmp_path):
    manager = ProjectManager(tmp_path / "projects")
    directory = manager.create_project("Site", "D:/images", custom_classes={"added": [SOLAR]})
    project = manager.load(directory)
    assert project["settings"]["classification"]["custom_classes"]["added"][0]["name"] == "solar panel"
    path = project_classes_path(directory)
    assert path == directory / "config" / "classes.json"
    assert json.loads(path.read_text(encoding="utf-8"))["outdoor"][-1].startswith("solar panel,")


def test_create_project_indoor_accepts_customisation(tmp_path):
    manager = ProjectManager(tmp_path / "projects")
    directory = manager.create_project("Room", "D:/images", capture_type="camera", class_set="indoor",
                                       custom_classes={"added": [{"name": "radiator"}]})
    assert json.loads(project_classes_path(directory).read_text(encoding="utf-8"))["indoor"][-1].startswith("radiator")


def test_create_project_flurseg_ignores_customisation(tmp_path):
    manager = ProjectManager(tmp_path / "projects")
    directory = manager.create_project("Fast", "D:/images", classification_engine="flurseg",
                                       custom_classes={"added": [SOLAR]})
    assert "custom_classes" not in manager.load(directory)["settings"]["classification"]
    assert project_classes_path(directory) == global_classes_path()


def test_create_project_without_customisation_uses_global_classes(tmp_path):
    manager = ProjectManager(tmp_path / "projects")
    directory = manager.create_project("Plain", "D:/images", custom_classes={"prompts": {}, "added": []})
    assert project_classes_path(directory) == global_classes_path()


def test_create_project_rejects_invalid_customisation_without_creating_folder(tmp_path):
    manager = ProjectManager(tmp_path / "projects")
    with pytest.raises(ValueError):
        manager.create_project("Bad", "D:/images", custom_classes={"added": [{"name": "tree"}]})
    assert not (tmp_path / "projects" / "bad").exists()


def _command(directory):
    manager = object.__new__(ProcessingManager)
    return manager._pipeline_command(directory, {
        "settings": {
            "workflow": {"use_gpu": True},
            "classification": {"class_set": "outdoor"},
            "coordinate_systems": {"working_crs": "LOCAL", "export_crs": "EPSG:2056"},
        },
        "acquisition": {"type": "camera", "sources": {"image_directory": "D:/images"}},
        "paths": {"image_directory": "D:/images"},
    }, "none")


def test_pipeline_command_passes_project_classes(tmp_path):
    command = _command(tmp_path)
    assert Path(command[command.index("--classes-json") + 1]) == global_classes_path()
    own = tmp_path / "config" / "classes.json"
    own.parent.mkdir()
    own.write_text("{}", encoding="utf-8")
    command = _command(tmp_path)
    assert Path(command[command.index("--classes-json") + 1]) == own


def test_pipeline_cli_accepts_classes_json(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(pipeline, "_reclassify_project", lambda args, *rest: seen.update(path=args.classes_json))
    monkeypatch.setattr("sys.argv", ["pipeline", "--outdir", str(tmp_path / "products"), "--reclassify",
                                     "--classes-json", str(tmp_path / "c.json")])
    pipeline.main()
    assert seen == {"path": str(tmp_path / "c.json")}
