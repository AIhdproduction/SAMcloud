"""Description and ObjectType on every exported element (MASTERPROMPT 10.1 point 2)."""

from __future__ import annotations

from pathlib import Path

import ifcopenshell
import pytest

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.ifc.identity import door_description, space_description, window_description
from samcloud.bim.core.model import CeilingIfcClass, ElementKind
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene

from .conftest import Export, validation_statements

EXPORTED_CLASSES = (
    "IfcWall",
    "IfcSlab",
    "IfcCovering",
    "IfcOpeningElement",
    "IfcWindow",
    "IfcDoor",
    "IfcSpace",
)


def test_every_element_has_description_and_object_type(layout_export: Export) -> None:
    checked = 0
    for ifc_class in EXPORTED_CLASSES:
        for entity in layout_export.file.by_type(ifc_class, include_subtypes=False):
            assert isinstance(entity.Description, str) and entity.Description.strip()
            assert isinstance(entity.ObjectType, str) and entity.ObjectType.strip()
            checked += 1
    assert checked >= 7


def test_generated_texts_for_one_wall_and_one_window(layout_export: Export) -> None:
    wall = next(w for w in layout_export.file.by_type("IfcWall") if w.Name == "Exterior wall south")
    assert wall.Description == "Exterior wall, 0.30 m thick, 2.80 m high"
    assert wall.ObjectType == "Exterior wall 300"
    window = layout_export.file.by_type("IfcWindow")[0]
    assert window.Description == "Window 1.20 x 1.40 m, sill 0.90 m"
    assert window.ObjectType == window.IsTypedBy[0].RelatingType.Name


def test_other_kinds_texts(layout_export: Export) -> None:
    file = layout_export.file
    assert file.by_type("IfcDoor")[0].Description == "Door 0.90 x 2.10 m, single swing left"
    assert file.by_type("IfcSlab")[0].Description == "Floor slab, 0.25 m"
    opening = file.by_type("IfcOpeningElement")[0]
    assert opening.ObjectType == "Opening"
    assert opening.Description == "Opening 1.20 x 1.40 m in Exterior wall south"
    space = file.by_type("IfcSpace")[0]
    assert space.ObjectType == "Space"
    assert space.Description == "Room 001 Room 1"


def test_ceiling_object_type_stays_ceiling(layout_export: Export) -> None:
    ceilings = [
        s for s in layout_export.file.by_type("IfcSlab") if s.PredefinedType == "USERDEFINED"
    ]
    assert ceilings
    for ceiling in ceilings:
        assert ceiling.ObjectType == "CEILING"
        assert ceiling.Description == "Ceiling, 0.20 m"


def test_covering_ceiling_has_type_name_as_object_type(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    scene.settings.ceiling_ifc_class = CeilingIfcClass.COVERING
    write_ifc(scene, tmp_path / "c.ifc")
    file = ifcopenshell.open(str(tmp_path / "c.ifc"))
    covering = file.by_type("IfcCovering")[0]
    assert covering.ObjectType == covering.IsTypedBy[0].RelatingType.Name
    assert covering.Description == "Ceiling, 0.20 m"
    assert validation_statements(file) == []


def test_scene_description_wins_over_generated_text(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    wall = next(e for e in scene.elements.values() if e.kind == ElementKind.WALL)
    wall.description = "Custom wall text"
    write_ifc(scene, tmp_path / "d.ifc")
    file = ifcopenshell.open(str(tmp_path / "d.ifc"))
    match = [w for w in file.by_type("IfcWall") if w.Tag == wall.id]
    assert match[0].Description == "Custom wall text"


def test_export_is_still_valid(layout_export: Export) -> None:
    assert validation_statements(layout_export.file) == []


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (window_description(1.2, 1.4, 0.9), "Window 1.20 x 1.40 m, sill 0.90 m"),
        (door_description(0.9, 2.1, "single_swing_left"), "Door 0.90 x 2.10 m, single swing left"),
        (space_description("", "Hall"), "Room Hall"),
        (space_description("002", ""), "Room 002"),
    ],
)
def test_description_formats(actual: str, expected: str) -> None:
    assert actual == expected
