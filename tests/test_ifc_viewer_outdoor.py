"""Outdoor IFC viewer: category mapping, default visibility, colours and the viewer config handed to the JS."""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import Client, ui

from samcloud.app.ifc_viewer import IfcViewer
from samcloud.app.i18n import t
from samcloud.app.pages.dxf import ifc_category_label
from samcloud.bim import ui_data

REAL_PROPS = Path(__file__).resolve().parents[1] / "projects" / "drohnenrekonstruktion_3d_test" / "products" / "bim" / "site" / "props.json"


def _el(ifc_class: str, name: str, obj: str | None = None, kind: str | None = None) -> dict:
    return {"ifc_class": ifc_class, "name": name, "object_type": obj, "predefined_type": kind}


def _outdoor_props() -> dict:
    rows = (
        [_el("IfcWall", f"Building 1 facade {i}", "Wall 200", "STANDARD") for i in range(5)]
        + [_el("IfcWall", "Building 1 wall", "EXTERIOR", "STANDARD")]
        + [_el("IfcSlab", "Building 1 roof", None, "ROOF")] * 2
        + [_el("IfcWall", f"Mauer {i}", "Mauer", "STANDARD") for i in range(3)]
        + [_el("IfcRailing", "Zaun", "Zaun")] * 2
        + [_el("IfcGeographicElement", "Baum", "Baum", "USERDEFINED")] * 4
        + [_el("IfcGeographicElement", "Busch", "Busch", "USERDEFINED")] * 2
        + [_el("IfcGeographicElement", "Terrain", None, "TERRAIN")]
        + [_el("IfcCivilElement", "Curb", "Randstein")] * 3
        + [_el("IfcCivilElement", "Pavement", "Trottoir")] * 2
        + [_el("IfcCivilElement", "Pavement", "Strasse")] * 2
        + [_el("IfcDistributionChamberElement", "Manhole", "Schacht", "MANHOLE")]
        + [_el("IfcAnnotation", "Marking")]
        + [_el("IfcBuildingElementProxy", "Thing")]
    )
    return {"elements": {f"g{i}": info for i, info in enumerate(rows)}}


def test_outdoor_category_mapping_and_counts() -> None:
    by_node, counts = ui_data.category_index(_outdoor_props())
    assert counts == {
        "terrain": 1, "pavements": 2, "sidewalks": 2, "curbs": 3, "buildings": 6, "roofs": 2, "site_walls": 3,
        "fences": 2, "trees": 4, "shrubs": 2, "manholes": 1, "markings": 1, "other": 1,
    }
    assert list(counts) == [name for name in ui_data.OUTDOOR_CATEGORY_ORDER if name in counts]
    assert by_node["g0"] == "buildings" and by_node["g6"] == "roofs"
    assert len(by_node) == sum(counts.values())


def test_real_outdoor_props_map_every_element_to_a_known_category() -> None:
    if not REAL_PROPS.is_file():
        return
    data = json.loads(REAL_PROPS.read_text(encoding="utf-8"))
    assert ui_data.is_outdoor("site", data)
    by_node, counts = ui_data.category_index(data)
    assert sum(counts.values()) == len(data["elements"])
    assert set(counts) <= set(ui_data.OUTDOOR_CATEGORY_ORDER)
    assert counts.get("other", 0) == 0


def test_indoor_props_keep_indoor_categories_and_hidden_defaults() -> None:
    data = {"elements": {
        "a": {"ifc_class": "IfcWall"}, "b": {"ifc_class": "IfcSlab", "predefined_type": "ROOF"},
        "c": {"ifc_class": "IfcSpace"}, "d": {"ifc_class": "IfcDoor"}, "e": {"ifc_class": "IfcOpeningElement"},
    }}
    assert not ui_data.is_outdoor("1", data)
    config = ui_data.viewer_config("1", data)
    assert not config["outdoor"] and config["colors"] == {}
    assert list(config["counts"]) == ["walls", "ceilings", "doors", "spaces", "openings"]
    assert config["hidden"] == {"ceilings", "spaces", "openings"}


def test_outdoor_categories_all_visible_by_default_including_roofs() -> None:
    config = ui_data.viewer_config("site", _outdoor_props())
    assert config["outdoor"] is True
    assert config["hidden"] == set()
    assert "roofs" in config["counts"]
    # classes alone are enough to detect outdoor, even for an unusual level key
    assert ui_data.viewer_config("9", _outdoor_props())["outdoor"] is True


def test_outdoor_viewer_config_has_distinct_colours_per_category() -> None:
    config = ui_data.viewer_config("site", _outdoor_props())
    colours = config["colors"]
    assert set(colours) == set(config["counts"])
    assert all(len(value) == 7 and value.startswith("#") for value in colours.values())
    assert len({value for key, value in colours.items() if key != "other"}) == len(colours) - 1
    assert colours["roofs"] != colours["buildings"] and colours["trees"] != colours["shrubs"]


def test_viewer_component_carries_category_colours_and_js_supports_them() -> None:
    # Own client context: other tests (user_simulation) leave NiceGUI without a default slot stack.
    with Client(ui.page("/_ifc_viewer_props_test")).content:
        viewer = IfcViewer()
    assert viewer._props["category-colors"] == {}
    viewer.set_category_colors({"roofs": "#c8643c"})
    assert viewer._props["category-colors"] == {"roofs": "#c8643c"}
    js = (Path(__file__).resolve().parents[1] / "samcloud" / "app" / "ifc_viewer.js").read_text(encoding="utf-8")
    assert "categoryColors" in js and "applyColors" in js and "fit(keepAngle)" in js


def test_category_labels_are_translated_in_both_languages() -> None:
    for name in ui_data.OUTDOOR_CATEGORY_ORDER:
        label = ifc_category_label(name, True)
        assert label and not label.startswith("dxf.ifc_outdoor.") and not label.startswith("ifc.category.")
    assert ifc_category_label("walls", False) == t("ifc.category.walls")
    for lang in ("de", "en"):
        path = Path(__file__).resolve().parents[1] / "samcloud" / "app" / "locales" / lang / "dxf.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = set(data["ifc_outdoor"])
        assert keys == set(ui_data.OUTDOOR_CATEGORY_ORDER) - {"other"}
