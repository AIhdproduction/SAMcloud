"""AI assistant selections: hit-testing, chip labels, read tools and the GlobalId mapping against a written IFC."""

from __future__ import annotations

import json

import ifcopenshell
import pytest

from assistant_synthetic import make_plan
from samcloud.assistant.selection import chip_label, element_details, element_list, global_id_for, global_id_table, hit_test, plan_id_for_global_id


@pytest.fixture(scope="module")
def ifc_file(tmp_path_factory: pytest.TempPathFactory):
    from samcloud.bim.core.ifc import write_ifc
    from samcloud.bim.scene_builder import build_scene

    path = tmp_path_factory.mktemp("assistant_ifc") / "plan.ifc"
    write_ifc(build_scene(make_plan()), path)
    return ifcopenshell.open(str(path))


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (4.02, 2.6, ("opening", "O001")),  # in the door
        (2.0, 0.05, ("opening", "O002")),  # in the window
        (1.5, 1.4, ("furniture", "F001")),  # on the table
        (4.05, 4.0, ("wall", "W005")),  # on the partition
        (8.1, 2.0, ("wall", "W002")),  # on the exterior wall
        (6.0, 3.0, ("room", "R002")),  # inside the east room
        (2.5, 3.5, ("room", "R001")),
    ],
)
def test_hit_test_elements(x: float, y: float, expected: tuple[str, str]) -> None:
    hit = hit_test(make_plan(), x, y)
    assert hit["type"] == "element" and (hit["kind"], hit["id"]) == expected


def test_hit_test_point_and_tolerance() -> None:
    plan = make_plan()
    assert hit_test(plan, 12.0, 9.0) == {"type": "point", "x": 12.0, "y": 9.0}
    assert hit_test(plan, 2.32, 1.4)["id"] == "F001"  # 12 cm beside the table: within 0.15 m
    assert hit_test(plan, 2.32, 1.4, tolerance_m=0.05)["kind"] == "room"


def test_chip_labels() -> None:
    plan = make_plan()
    door = hit_test(plan, 4.0, 2.5)
    assert chip_label(door, "en") == "Door O001 (0.90 m)"
    assert chip_label(door, "de") == "Tür O001 (0.90 m)"
    assert chip_label(hit_test(plan, 1.5, 1.4), "de") == "Tisch F001"
    assert chip_label(hit_test(plan, 6.0, 3.0), "en") == "Room Schlafen"
    assert chip_label({"type": "point", "x": 1.234, "y": 5.0}, "en") == "Point x 1.23 / y 5.00"


def test_element_list_and_details() -> None:
    plan = make_plan()
    everything = element_list(plan)
    json.dumps(everything)
    assert {row["id"] for row in everything} == {"W001", "W002", "W003", "W004", "W005", "O001", "O002", "R001", "R002", "F001"}
    doors = element_list(plan, "door")
    assert [row["id"] for row in doors] == ["O001"] and doors[0]["width_m"] == pytest.approx(0.9)
    in_room = element_list(plan, {"room": "R001", "kind": ["furniture", "window"]})
    assert {row["id"] for row in in_room} == {"F001", "O002"}
    assert element_list(plan, {"label": "tisch"})[0]["id"] == "F001"
    wall = element_details(plan, "W005")
    assert wall["openings"] == ["O001"] and set(wall["rooms"]) == {"R001", "R002"} and wall["thickness_m"] == pytest.approx(0.2)
    assert set(wall["joins"]) == {"W001", "W003"}
    room = element_details(plan, "R001")
    assert room["furniture"] == ["F001"] and "O002" in room["openings"]
    assert element_details(plan, "X999") == {"id": "X999", "found": False}


def test_global_ids_round_trip_with_the_written_ifc(ifc_file) -> None:
    plan = make_plan()
    table = global_id_table(plan)
    checked = 0
    for ifc_class, kind in (("IfcWall", "wall"), ("IfcDoor", "opening"), ("IfcWindow", "opening"), ("IfcSpace", "room"),
                            ("IfcFurniture", "furniture"), ("IfcOpeningElement", "opening")):
        for entity in ifc_file.by_type(ifc_class):
            found = plan_id_for_global_id(plan, entity.GlobalId)
            assert found is not None, (ifc_class, entity.Name)
            assert found[0] == kind
            assert entity.GlobalId in table
            checked += 1
    assert checked >= 5 + 1 + 1 + 2 + 1 + 2
    door = ifc_file.by_guid(global_id_for(plan, "opening", "O001"))
    assert door.is_a("IfcDoor") and door.Name == "Door O001"
    assert ifc_file.by_guid(global_id_for(plan, "opening_element", "O001")).is_a("IfcOpeningElement")
    assert ifc_file.by_guid(global_id_for(plan, "wall", "W005")).is_a("IfcWall")
    assert ifc_file.by_guid(global_id_for(plan, "room", "R002")).Name == "Schlafen"
    assert ifc_file.by_guid(global_id_for(plan, "furniture", "F001")).Name == "Tisch 01"
    assert global_id_for(plan, "wall", "W999") is None
    assert plan_id_for_global_id(plan, "0000000000000000000000") is None
