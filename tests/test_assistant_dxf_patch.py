"""DXF patch layer: only edited elements are redrawn, every other entity stays the very same dict in the same order."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from assistant_synthetic import EMPTY_BOX, make_plan, make_result
from samcloud.assistant.dxf_patch import apply_dxf_patch
from samcloud.assistant.edits import EditOp, Proposal, diff_plans
from samcloud.assistant.evidence import MANUAL_LAYER


def _patched(*ops: EditOp):
    base = make_plan()
    proposal = Proposal(base)
    for op in ops:
        proposal.apply(op)
    result = make_result(base)
    return base, proposal, result, apply_dxf_patch(result, base, proposal.plan, proposal.changes)


def _untouched(result, patched, layers) -> None:
    for name in layers:
        assert len(patched.layers[name]) == len(result.layers[name]), name
        assert all(a is b for a, b in zip(result.layers[name], patched.layers[name])), name


def _wall_mass(result) -> Polygon:
    return unary_union([Polygon(e["points"], e.get("holes") or []) for e in result.layers["A-WALL"]])


def test_no_changes_returns_the_same_object() -> None:
    base = make_plan()
    result = make_result(base)
    assert apply_dxf_patch(result, base, base, []) is result
    assert apply_dxf_patch(result, base, base, diff_plans(base, base)) is result


def test_window_width_redraws_only_the_window() -> None:
    _base, _proposal, result, patched = _patched(EditOp("set_opening", "O002", {"width_m": 1.5}))
    insert = next(e for e in patched.layers["A-WIND"] if e["type"] == "insert")
    assert insert["attributes"]["LABEL"] == "Window 1.50 m"
    assert insert["attributes"]["BRH"] == "0.90" and insert["attributes"]["STH"] == "2.10"
    assert insert["xscale"] == pytest.approx(1.5) and insert["point"] == pytest.approx([1.25, 0.0])
    _untouched(result, patched, ["A-DOOR", "A-WALL", "A-WALL-PATT", "A-FURN", "A-ANNO", "A-AREA"])
    assert patched.layers is not result.layers and result.layers["A-WIND"][0]["xscale"] == pytest.approx(1.2)  # input untouched


def test_door_shift_redraws_door_symbol_and_jambs() -> None:
    _base, _proposal, result, patched = _patched(EditOp("set_opening", "O001", {"shift_m": 0.5, "width_m": 1.0}))
    centres = {tuple(e["_opening_center"]) for e in patched.layers["A-DOOR"]}
    assert centres == {(4.0, 3.0)}
    insert = next(e for e in patched.layers["A-DOOR"] if e["type"] == "insert")
    assert insert["name"] == "DOOR" and insert["attributes"] == {"LABEL": "Door 1.00 m", "H": "2.05"}
    assert sum(1 for e in patched.layers["A-DOOR"] if e.get("_jamb")) == 2
    _untouched(result, patched, ["A-WIND", "A-WALL", "A-FURN", "A-ANNO"])


def test_move_wall_patches_walls_door_rooms_and_keeps_the_rest() -> None:
    _base, proposal, result, patched = _patched(EditOp("move_wall", "W005", {"offset_m": 0.5}))
    mass = _wall_mass(patched)
    assert mass.contains(Point(4.5, 1.0)) and not mass.contains(Point(4.0, 1.0))
    assert not mass.contains(Point(4.5, 2.5))  # the moved door void stays open
    assert mass.contains(Point(0.0, 2.0)) and mass.contains(Point(8.0, 2.0))  # exterior walls untouched
    hatch = unary_union([Polygon(e["points"], e.get("holes") or []) for e in patched.layers["A-WALL-PATT"]])
    assert hatch.symmetric_difference(mass).area < 1e-9
    assert {tuple(e["_opening_center"]) for e in patched.layers["A-DOOR"]} == {(4.5, 2.5)}
    labels = {e["_room_name"]: e["text"] for e in patched.layers["A-ANNO"] if "_room_name" in e}
    assert labels == {"Wohnen": "Wohnen  20.0 m²", "Schlafen": "Schlafen  15.3 m²"}
    areas = sorted(Polygon(e["points"]).area for e in patched.layers["A-AREA"])
    assert areas == pytest.approx(sorted([4.25 * 4.7, 3.25 * 4.7]))
    _untouched(result, patched, ["A-WIND", "A-FURN"])
    furniture_texts = [e for e in patched.layers["A-ANNO"] if "_furn_id" in e]
    assert furniture_texts[0] is next(e for e in result.layers["A-ANNO"] if "_furn_id" in e)


def test_thicker_wall_band_in_dxf() -> None:
    _base, _proposal, _result, patched = _patched(EditOp("set_wall_thickness", "W005", {"thickness_m": 0.4}))
    mass = _wall_mass(patched)
    assert mass.contains(Point(3.85, 1.0)) and mass.contains(Point(4.15, 1.0))
    assert not mass.contains(Point(3.75, 1.0))
    jambs = [e for e in patched.layers["A-DOOR"] if e.get("_jamb")]
    lengths = [math.dist(e["start"], e["end"]) for e in jambs]
    assert lengths == pytest.approx([0.4, 0.4])


def test_furniture_move_rotate_and_delete() -> None:
    _base, _proposal, result, patched = _patched(EditOp("move_furniture", "F001", {"dx_m": 1.0, "dy_m": 2.0}))
    outline = next(e for e in patched.layers["A-FURN"] if e["type"] == "polyline")
    assert Polygon(outline["points"]).centroid.coords[0] == pytest.approx((2.6, 3.4))
    assert outline["_furn_id"] == 1  # the pipeline entity, moved
    text = next(e for e in patched.layers["A-ANNO"] if e.get("_furn_id") == 1)
    assert text["point"] == pytest.approx([2.6, 3.4]) and text["text"] == "Tisch h=0.75"
    _untouched(result, patched, ["A-WALL", "A-DOOR", "A-WIND", "A-AREA"])

    _base, _proposal, result, patched = _patched(EditOp("rotate_furniture", "F001", {"angle_deg": 90.0}))
    outline = next(e for e in patched.layers["A-FURN"] if e["type"] == "polyline")
    xs = [p[0] for p in outline["points"]]
    assert max(xs) - min(xs) == pytest.approx(0.8)

    _base, _proposal, result, patched = _patched(EditOp("delete_furniture", "F001", {}))
    assert patched.layers["A-FURN"] == []
    assert [e["text"] for e in patched.layers["A-ANNO"]] == [e["text"] for e in result.layers["A-ANNO"] if "_furn_id" not in e]


def test_class_change_relabels() -> None:
    _base, _proposal, _result, patched = _patched(EditOp("set_furniture_class", "F001", {"kind": "desk"}))
    text = next(e for e in patched.layers["A-ANNO"] if e.get("_furn_id") == 1)
    assert text["text"] == "Schreibtisch h=0.75"


def test_manual_item_on_its_own_layer() -> None:
    _base, _proposal, result, patched = _patched(EditOp("add_furniture", None, {"box": EMPTY_BOX, "kind": "chair", "manual": True}))
    assert MANUAL_LAYER not in result.layers
    entities = patched.layers[MANUAL_LAYER]
    assert len(entities) == 1 and entities[0]["closed"]
    assert Polygon(entities[0]["points"]).centroid.coords[0] == pytest.approx((7.0, 4.0))
    assert patched.layer_colors[MANUAL_LAYER] == 6 and patched.layer_linetypes[MANUAL_LAYER] == "DASHED"
    assert patched.statuses[MANUAL_LAYER].entity_count == 1
    assert any(e.get("text") == "Stuhl" for e in patched.layers["A-ANNO"])  # a chair is labelled with its name only
    _untouched(result, patched, ["A-WALL", "A-DOOR", "A-WIND", "A-FURN", "A-AREA"])


def test_rename_room_and_dxf_document_builds(tmp_path) -> None:
    from samcloud.dxf.writer import build_dxf_document

    _base, _proposal, result, patched = _patched(
        EditOp("rename_room", "R002", {"name": "Gast"}),
        EditOp("add_furniture", None, {"box": EMPTY_BOX, "kind": "table", "manual": True}),
        EditOp("move_wall", "W005", {"offset_m": -0.2}),
    )
    labels = [e["text"] for e in patched.layers["A-ANNO"] if "_room_name" in e]
    assert labels == ["Wohnen  16.7 m²", "Gast  18.6 m²"]
    document = build_dxf_document(patched)
    names = {entity.dxf.layer for entity in document.modelspace()}
    assert MANUAL_LAYER in names and "A-WALL" in names
    assert MANUAL_LAYER in [layer.dxf.name for layer in document.layers]
    plain = build_dxf_document(result)
    assert MANUAL_LAYER not in [layer.dxf.name for layer in plain.layers]


def test_assistant_layer_style_changes_no_fingerprint(tmp_path) -> None:
    """``assistant_layers`` in config/dxf_layers.yaml is invisible to every cache fingerprint and settings hash.

    The fingerprints (clean plan ``clean_plan_fingerprint``, point-cloud DXF ``settings_fingerprint`` / ``_settings_snapshot``)
    hash only the parsed ``layers`` records of ``load_layer_config``; ortho/build.py and ortho/config.py read only ``layers``.
    """
    import yaml

    from samcloud.core.paths import CONFIG_DIRECTORY
    from samcloud.dxf.layers import load_layer_config
    from samcloud.dxf.writer import _load_extra_layers

    source = CONFIG_DIRECTORY / "dxf_layers.yaml"
    text = source.read_text(encoding="utf-8")
    assert "assistant_layers:" in text
    without = tmp_path / "dxf_layers.yaml"
    without.write_text(text[: text.index("# AI assistant")], encoding="utf-8")
    assert load_layer_config(source) == load_layer_config(without)
    full, stripped = yaml.safe_load(text), yaml.safe_load(without.read_text(encoding="utf-8"))
    assert {k: v for k, v in full.items() if k != "assistant_layers"} == stripped
    assert _load_extra_layers(str(source)) == _load_extra_layers(str(without))
    assert MANUAL_LAYER not in {layer.name for layer in load_layer_config(source).layers}


def test_real_fingerprint_functions_ignore_the_assistant_layers(tmp_path, monkeypatch) -> None:
    """The clean-plan fingerprint and the point-cloud DXF settings fingerprint are equal with and without the block."""
    from types import SimpleNamespace

    import numpy as np

    import samcloud.ortho.plan as ortho_plan
    from dxf_synthetic import write_classified_las
    from samcloud.core.paths import CONFIG_DIRECTORY
    from samcloud.dxf.layers import load_layer_config
    from samcloud.dxf.pipeline import PlanSettings, settings_fingerprint

    source = CONFIG_DIRECTORY / "dxf_layers.yaml"
    text = source.read_text(encoding="utf-8")
    without = tmp_path / "dxf_layers.yaml"
    without.write_text(text[: text.index("# AI assistant")], encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(ortho_plan, "load_segment_result", lambda *args, **kwargs: SimpleNamespace(fingerprint="synthetic-segment"))
    hashes = []
    for path in (source, without):
        monkeypatch.setattr(ortho_plan, "load_layer_config", lambda path=path: load_layer_config(path))
        hashes.append(ortho_plan.clean_plan_fingerprint(project, SimpleNamespace(id="level_1", name="EG", floor_z=0.0, ceiling_z=2.6)))
    assert hashes[0] == hashes[1]

    rng = np.random.default_rng(3)
    xyz = rng.uniform(0.013, 4.987, size=(400, 3))
    las = write_classified_las(tmp_path / "cloud.las", xyz, np.full(len(xyz), 1, np.uint8), mode="indoor")
    settings = [settings_fingerprint(las, "indoor", None, PlanSettings(layer_config=load_layer_config(path)), 0.05, load_layer_config(path))
                for path in (source, without)]
    assert settings[0] == settings[1]
