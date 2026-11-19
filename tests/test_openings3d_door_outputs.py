"""Outputs of the door classification: plan opening fields, DXF door symbol orientation, IFC door operation / passages."""

from __future__ import annotations

import math

import ifcopenshell
import numpy as np
import pytest

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model.elements import DoorOperation
from samcloud.bim.openings3d.apply import to_plan_opening
from samcloud.bim.openings3d.model import RefinedOpening
from samcloud.bim.plan_model import PlanOpening
from samcloud.bim.scene_builder import build_scene, door_hand
from samcloud.ortho.vector.indoor import Opening, opening_entities
from test_bim_scene_builder import make_plan


def _refined(kind: str = "door", **kwargs) -> RefinedOpening:
    values = dict(
        id="C001", kind=kind, center=[3.0, 0.0], width_m=0.9, axis_angle=0.0, wall_thickness_m=0.25, host_wall_id="W001", sill_m=0.0, head_m=2.05,
        height_source="points", evidence="synthetic", confidence=0.8,
    )
    values.update(kwargs)
    return RefinedOpening(**values)


# ---------------------------------------------------------------- plan opening fields


@pytest.mark.parametrize("upm", [1.0, 0.8124])
def test_refined_door_with_leaf_carries_hinge_swing_and_angle_into_the_plan_opening(upm: float) -> None:
    refined = _refined(hinge_side="right", swing_side=-1, leaf_angle_deg=83.0, leaf_detected=True)
    opening = to_plan_opening(refined, upm, (3.0 * upm, 0.0), 0.0)
    assert (opening.kind, opening.hinge_side, opening.swing_side, opening.leaf_angle_deg, opening.leaf_detected, opening.leaf_mode) == ("DOOR", "right", -1, 83.0, True, "open")
    assert opening.width == pytest.approx(0.9 * upm)


def test_refined_passage_and_door_without_leaf_carry_no_leaf() -> None:
    passage = to_plan_opening(_refined("passage", head_m=2.45), 1.0, (3.0, 0.0), 0.0)
    assert passage.kind == "passage" and passage.hinge_side is None and passage.swing_side is None and not passage.leaf_detected and passage.head_m == pytest.approx(2.45)
    door = to_plan_opening(_refined(), 1.0, (3.0, 0.0), 0.0)
    assert door.kind == "DOOR" and door.leaf_detected is None and door.hinge_side is None  # unmeasured: the 2D leaf detection still runs
    leaf_on_passage = to_plan_opening(_refined("passage", hinge_side="left", swing_side=1, leaf_detected=True), 1.0, (3.0, 0.0), 0.0)
    assert leaf_on_passage.hinge_side is None and not leaf_on_passage.leaf_detected  # a passage never carries a leaf


# ---------------------------------------------------------------- DXF door symbol


def _door_symbol(angle: float, hinge_side: str, swing_side: int, centre=(1.0, 0.5), width: float = 0.9) -> tuple[dict, np.ndarray, np.ndarray]:
    opening = Opening("DOOR", centre, width, angle, 0.25, hinge_side=hinge_side, swing_side=swing_side, leaf_detected=True, leaf_angle_deg=80.0)
    entities = [e for e in opening_entities([opening]) if e.get("type") == "insert"]
    assert len(entities) == 1
    insert = entities[0]
    rotation = math.radians(float(insert["rotation"]))
    closed_leaf = np.array([math.cos(rotation), math.sin(rotation)]) * float(insert["xscale"])
    open_leaf = np.array([-math.sin(rotation), math.cos(rotation)]) * float(insert["yscale"])
    return insert, closed_leaf, open_leaf


@pytest.mark.parametrize("angle", [0.0, math.radians(33.0), math.pi / 2, math.radians(147.0)])
@pytest.mark.parametrize(("hinge_side", "swing_side"), [("left", 1), ("left", -1), ("right", 1), ("right", -1)])
def test_dxf_door_symbol_sits_on_the_measured_hinge_and_swings_to_the_measured_side(angle: float, hinge_side: str, swing_side: int) -> None:
    centre = np.array([1.0, 0.5])
    axis = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-axis[1], axis[0]])
    insert, closed_leaf, open_leaf = _door_symbol(angle, hinge_side, swing_side)
    sign = -1.0 if hinge_side == "left" else 1.0
    assert np.asarray(insert["point"]) == pytest.approx(centre + sign * axis * 0.45)  # the hinge: at the -axis end for "left"
    assert closed_leaf == pytest.approx(-sign * axis * 0.9)  # the closed leaf spans the opening, from the hinge to the other jamb
    assert open_leaf == pytest.approx(swing_side * normal * 0.9)  # the open leaf stands on the measured swing side
    assert insert["_hinge_side"] == hinge_side and insert["_swing_side"] == swing_side


def test_dxf_passage_and_leafless_door_have_no_leaf_symbol_and_unknown_swing_keeps_the_default() -> None:
    passage = Opening("passage", (1.0, 0.0), 0.8, 0.0, 0.25, head_m=2.45, height_source="points", sill_m=0.0)
    entities = opening_entities([passage])
    assert not [e for e in entities if e.get("type") == "insert"] and len([e for e in entities if e.get("_jamb")]) == 2
    assert any("Durchgang" in e["text"] for e in entities if e.get("type") == "text")
    leafless = Opening("DOOR", (1.0, 0.0), 0.8, 0.0, 0.25, leaf_detected=False)
    assert not [e for e in opening_entities([leafless]) if e.get("type") == "insert"]
    unknown = Opening("DOOR", (1.0, 0.0), 0.8, 0.0, 0.25)  # hinge and swing unknown: left hinge, +normal swing (the existing default)
    insert = next(e for e in opening_entities([unknown]) if e.get("type") == "insert")
    assert insert["_hinge_side"] == "left" and insert["_swing_side"] == 1


# ---------------------------------------------------------------- IFC


@pytest.mark.parametrize("angle_deg", [0.0, 33.0, 90.0, 147.0])
@pytest.mark.parametrize(("hinge_side", "swing_side"), [("left", 1), ("left", -1), ("right", 1), ("right", -1)])
def test_door_hand_is_the_hinge_side_seen_from_the_side_the_door_opens_towards(angle_deg: float, hinge_side: str, swing_side: int) -> None:
    angle = math.radians(angle_deg)
    axis = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-axis[1], axis[0]])
    centre = np.zeros(2)
    hinge = centre + (-axis if hinge_side == "left" else axis) * 0.45
    facing = -swing_side * normal  # the observer stands on the swing side and looks at the door
    observer_left = np.array([-facing[1], facing[0]])
    expected = DoorOperation.SINGLE_SWING_LEFT if float((hinge - centre) @ observer_left) > 0 else DoorOperation.SINGLE_SWING_RIGHT
    operation, side = door_hand(hinge_side, swing_side)
    assert operation == expected
    assert side.value == ("left" if expected == DoorOperation.SINGLE_SWING_LEFT else "right")


def test_door_hand_without_measurement_keeps_the_defaults() -> None:
    assert door_hand(None, None)[0] == DoorOperation.SINGLE_SWING_RIGHT
    assert door_hand(None, 1)[0] == DoorOperation.SINGLE_SWING_RIGHT
    assert door_hand("left", None)[0] == DoorOperation.SINGLE_SWING_LEFT


def test_ifc_door_operation_type_follows_hinge_and_swing_and_passages_are_plain_openings(tmp_path) -> None:
    plan = make_plan()
    plan.openings = [
        PlanOpening("O001", "door", [4.0, 2.5], 0.9, math.pi / 2, 0.2, host_wall_id="W005", head_m=2.05, height_source="points", hinge_side="left", swing_side=-1, leaf_angle_deg=82.0, leaf_detected=True, confidence=0.9),
        PlanOpening("O002", "door", [4.0, 1.2], 0.9, math.pi / 2, 0.2, host_wall_id="W005", head_m=2.05, height_source="points", hinge_side="right", swing_side=-1, leaf_angle_deg=82.0, leaf_detected=True, confidence=0.9),
        PlanOpening("O003", "door", [1.5, 0.0], 0.9, 0.0, 0.3, host_wall_id="W001", head_m=2.05, height_source="points", confidence=0.9),
        PlanOpening("O004", "passage", [3.5, 0.0], 0.8, 0.0, 0.3, host_wall_id="W001", head_m=2.45, height_source="points", confidence=0.9),
    ]
    scene = build_scene(plan)
    path = tmp_path / "doors.ifc"
    write_ifc(scene, path)
    file = ifcopenshell.open(str(path))
    doors = {door.Name: door for door in file.by_type("IfcDoor")}
    assert doors["Door O001"].OperationType == "SINGLE_SWING_LEFT"  # hinge at -axis, swings to -normal: hinge on the left of the observer
    assert doors["Door O002"].OperationType == "SINGLE_SWING_RIGHT"
    assert doors["Door O003"].OperationType == "SINGLE_SWING_RIGHT"  # unmeasured: the default
    assert "Door O004" not in doors and len(doors) == 3  # a passage has no IfcDoor ...
    openings = {opening.Name for opening in file.by_type("IfcOpeningElement")}
    assert "Passage opening O004" in openings and "Door opening O001" in openings  # ... only its opening in the wall
    types = {t.Name: t.OperationType for t in file.by_type("IfcDoorType")}
    assert "SINGLE_SWING_LEFT" in types.values() and "SINGLE_SWING_RIGHT" in types.values()


# ---------------------------------------------------------------- UI property data


def test_ui_opening_details_show_hinge_swing_and_leaf_angle_of_a_door_only() -> None:
    from samcloud.bim.ui_data import opening_details

    psets = {"Qto_DoorBaseQuantities": {"Width": 0.9, "Height": 2.05}, "Raumlot_User": {"Glazed": False, "HingeSide": "right", "SwingNormalSide": "-normal", "LeafOpeningAngleDeg": 82.0}}
    details = opening_details({"ifc_class": "IfcDoor", "psets": psets})
    assert details["hinge"] == "right" and details["swing"] == "-normal" and details["leaf_angle"] == 82.0
    plain = opening_details({"ifc_class": "IfcDoor", "psets": {"Raumlot_User": {"Glazed": False}}})
    assert "hinge" not in plain and "swing" not in plain and "leaf_angle" not in plain
    window = opening_details({"ifc_class": "IfcWindow", "psets": {"Raumlot_User": {"HingeSide": "left", "MullionCount": 1.0}}})
    assert "hinge" not in window and window["mullions"] == 1
