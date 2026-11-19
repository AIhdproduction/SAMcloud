"""Tests for samcloud.bim.ui_data: BIM artefact paths, props caching, IFC freshness and category mapping."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from samcloud.bim import ui_data


def _write(path: Path, content: str = "x", mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_paths_follow_the_file_contract(tmp_path: Path) -> None:
    assert ui_data.bim_dir(tmp_path, "1") == tmp_path / "products" / "bim" / "1"
    assert ui_data.glb_path(tmp_path, "1") is None
    assert ui_data.find_ifc(tmp_path, "1") is None
    folder = tmp_path / "products" / "bim" / "1"
    glb = _write(folder / "model.glb")
    ifc = _write(folder / "Home_1.ifc")
    assert ui_data.glb_path(tmp_path, "1") == glb
    assert ui_data.find_ifc(tmp_path, "1") == ifc
    assert ui_data.glb_path(tmp_path, "2") is None


def test_level_key_cannot_escape_the_bim_folder(tmp_path: Path) -> None:
    for key in ("..", "a/b", "a" + chr(92) + "b", ""):
        with pytest.raises(ValueError):
            ui_data.bim_dir(tmp_path, key)


def test_find_ifc_prefers_props_ifc_file_over_newest(tmp_path: Path) -> None:
    folder = tmp_path / "products" / "bim" / "site"
    wanted = _write(folder / "wanted.ifc", mtime=1_000_000)
    _write(folder / "newer.ifc", mtime=2_000_000)
    _write(folder / "props.json", json.dumps({"ifc_file": "wanted.ifc", "elements": {}}))
    assert ui_data.find_ifc(tmp_path, "site") == wanted


def test_props_cached_by_mtime_and_reloaded_on_change(tmp_path: Path) -> None:
    path = _write(tmp_path / "products" / "bim" / "1" / "props.json", json.dumps({"classes": {"IfcWall": 1}}), mtime=1_000_000)
    first = ui_data.props(tmp_path, "1")
    assert first["classes"] == {"IfcWall": 1}
    assert ui_data.props(tmp_path, "1") is first  # same object: served from the cache
    path.write_text(json.dumps({"classes": {"IfcWall": 5}}), encoding="utf-8")
    os.utime(path, (2_000_000, 2_000_000))
    second = ui_data.props(tmp_path, "1")
    assert second is not first and second["classes"] == {"IfcWall": 5}


def test_props_and_qa_tolerate_missing_and_broken_files(tmp_path: Path) -> None:
    assert ui_data.props(tmp_path, "1") == {}
    assert ui_data.qa(tmp_path, "1") == {}
    _write(tmp_path / "products" / "bim" / "1" / "props.json", "{not json")
    _write(tmp_path / "products" / "bim" / "1" / "qa.json", "[1, 2]")
    assert ui_data.props(tmp_path, "1") == {}
    assert ui_data.qa(tmp_path, "1") == {}
    _write(tmp_path / "products" / "bim" / "1" / "qa.json", json.dumps({"ids_failures": 3, "elements": 9}))
    assert ui_data.qa(tmp_path, "1")["ids_failures"] == 3


def _export(root: Path, level_key: str, sha: str) -> None:
    folder = ui_data.bim_dir(root, level_key)
    for name in ("Home.ifc", "model.glb", "props.json"):
        _write(folder / name)
    _write(folder / "export.json", json.dumps({"ifc": "Home.ifc", "glb": "model.glb", "props": "props.json", "plan_model_sha256": sha}))


@pytest.mark.parametrize("level_key", ["1", "site"])
def test_ifc_status_missing_fresh_stale(tmp_path: Path, level_key: str) -> None:
    import hashlib

    assert ui_data.ifc_status(tmp_path, level_key) == "missing"
    model = _write(ui_data.plan_model_file(tmp_path, level_key), "{}")
    assert model.parent.name == "ortho" and model.parent.parent.name == level_key
    assert ui_data.ifc_status(tmp_path, level_key) == "missing"  # a plan model alone is not an export
    _export(tmp_path, level_key, hashlib.sha256(b"{}").hexdigest())
    assert ui_data.ifc_status(tmp_path, level_key) == "fresh"
    model.write_text('{"changed": 1}', encoding="utf-8")
    assert ui_data.ifc_status(tmp_path, level_key) == "stale"
    _export(tmp_path, level_key, "0" * 64)
    assert ui_data.ifc_status(tmp_path, level_key) == "stale"
    (ui_data.bim_dir(tmp_path, level_key) / "model.glb").unlink()
    assert ui_data.ifc_status(tmp_path, level_key) == "missing"


def test_ifc_status_without_manifest_is_missing(tmp_path: Path) -> None:
    _write(ui_data.bim_dir(tmp_path, "1") / "Home.ifc")
    assert ui_data.ifc_status(tmp_path, "1") == "missing"


@pytest.mark.parametrize(
    ("ifc_class", "kind", "expected"),
    [
        ("IfcWall", "SOLIDWALL", "walls"), ("IfcSlab", "FLOOR", "floors"), ("IfcSlab", "BASESLAB", "floors"),
        ("IfcSlab", "ROOF", "ceilings"), ("IfcSlab", "USERDEFINED", "ceilings"), ("IfcCovering", "CEILING", "ceilings"),
        ("IfcSpace", None, "spaces"), ("IfcDoor", "DOOR", "doors"), ("IfcWindow", None, "windows"),
        ("IfcOpeningElement", None, "openings"), ("IfcFurniture", None, "furniture"),
        ("IfcSanitaryTerminal", None, "sanitary"), ("IfcElectricAppliance", None, "appliances"),
        ("IfcSystemFurnitureElement", None, "builtins"), ("IfcBuildingElementProxy", None, "other"), ("IfcMystery", None, "other"),
    ],
)
def test_category_of(ifc_class: str, kind: str | None, expected: str) -> None:
    assert ui_data.category_of(ifc_class, kind) == expected


def test_category_index_counts_in_display_order() -> None:
    data = {"elements": {
        "a": {"ifc_class": "IfcWall"}, "b": {"ifc_class": "IfcWall"}, "c": {"ifc_class": "IfcSlab", "predefined_type": "ROOF"},
        "d": {"ifc_class": "IfcSpace"}, "e": "broken",
    }}
    by_node, counts = ui_data.category_index(data)
    assert by_node == {"a": "walls", "b": "walls", "c": "ceilings", "d": "spaces"}
    assert list(counts.items()) == [("walls", 2), ("ceilings", 1), ("spaces", 1)]


WINDOW_INFO = {
    "ifc_class": "IfcWindow", "name": "Window O008",
    "psets": {
        "Lining": {"LiningDepth": 0.06},
        "Panel 1": {}, "Panel 2": {}, "Panel 3": {},
        "Pset_WindowCommon": {"IsExternal": True},
        "Qto_WindowBaseQuantities": {"Width": 1.9, "Height": 1.52},
        "SAMcloud_Scan": {"InstanceId": 4},
        "SAMcloud_Frame": {"UnitsPerMetre": 1.0},
        "Raumlot_Quality": {"Status": "unsure"},
        "Raumlot_Location": {"StoreyElevation": 2.0, "BBoxMinZ": 2.2, "BBoxMaxZ": 3.72},
        "Raumlot_User": {"Glazed": False},
    },
}


def test_ceilings_are_hidden_by_default_next_to_spaces_and_openings() -> None:
    assert set(ui_data.HIDDEN_BY_DEFAULT) == {"spaces", "openings", "ceilings"}
    assert ui_data.category_of("IfcCovering", "CEILING") == "ceilings"
    assert ui_data.category_of("IfcSlab", "ROOF") == "ceilings"


def test_default_psets_hide_raumlot_and_internal_sets() -> None:
    names = [name for name, _ in ui_data.visible_psets(WINDOW_INFO)]
    assert names == ["Pset_WindowCommon", "Qto_WindowBaseQuantities", "SAMcloud_Scan"]
    everything = [name for name, _ in ui_data.visible_psets(WINDOW_INFO, show_all=True)]
    assert "Raumlot_Quality" in everything and "Lining" in everything and len(everything) == len(WINDOW_INFO["psets"])
    assert ui_data.visible_psets({}) == [] and ui_data.visible_psets({"psets": None}) == []


def test_window_details_have_sill_head_width_mullions_and_glazed() -> None:
    details = ui_data.opening_details(WINDOW_INFO)
    assert details is not None
    assert details["width"] == pytest.approx(1.9) and details["height"] == pytest.approx(1.52)
    assert details["sill"] == pytest.approx(0.2) and details["head"] == pytest.approx(1.72)
    assert details["mullions"] == 2  # three panels -> two mullions when no explicit count exists
    assert "glazed" not in details  # glazing is a door property, never shown as "no" on windows
    glazed_window = {**WINDOW_INFO, "psets": {**WINDOW_INFO["psets"], "Raumlot_User": {"Glazed": True}}}
    assert ui_data.opening_details(glazed_window)["glazed"] is True


def test_explicit_mullion_count_and_missing_data_and_other_classes() -> None:
    info = {"ifc_class": "IfcWindow", "psets": {"Raumlot_User": {"MullionCount": 3.0}}}
    assert ui_data.opening_details(info) == {"mullions": 3}
    door = {"ifc_class": "IfcDoor", "psets": {"Qto_DoorBaseQuantities": {"Width": 0.9, "Height": 2.05},
                                              "Raumlot_Location": {"StoreyElevation": 1.0, "BBoxMinZ": 0.9, "BBoxMaxZ": 3.05},
                                              "Raumlot_User": {"Glazed": True}}}
    details = ui_data.opening_details(door)
    assert details is not None and details["sill"] == 0.0 and details["head"] == pytest.approx(2.05)
    assert "mullions" not in details and details["glazed"] is True
    assert ui_data.opening_details({"ifc_class": "IfcWall", "psets": {}}) is None
    assert ui_data.opening_details({}) is None
