"""Structure of the exported IFC: validation, counts, units, types and relations."""

from __future__ import annotations

import ifcopenshell.util.element as element_util
import ifcopenshell.util.unit
import pytest

from samcloud.bim.core.model import (
    DoorElement,
    ElementKind,
    JoinEnd,
    SpaceElement,
    WallJoinElement,
    WindowElement,
)

from .conftest import Export, entity_by_tag, validation_statements

CONNECTION = {JoinEnd.START: "ATSTART", JoinEnd.END: "ATEND", JoinEnd.PATH: "ATPATH"}
PANEL_OPERATION = {"side_hung_left": "SIDEHUNGLEFTHAND"}


def test_validates_without_findings(layout_export: Export) -> None:
    assert validation_statements(layout_export.file) == []


def test_validates_without_crs(rectangle_export: Export) -> None:
    assert validation_statements(rectangle_export.file) == []


def test_counts_per_class(layout_export: Export) -> None:
    scene = layout_export.scene
    counts = layout_export.result.counts
    f = layout_export.file
    expected = {
        "IfcWall": len(scene.elements_of(ElementKind.WALL)),
        "IfcSlab": len(scene.elements_of(ElementKind.SLAB))
        + len(scene.elements_of(ElementKind.CEILING)),
        "IfcOpeningElement": len(scene.elements_of(ElementKind.OPENING)),
        "IfcWindow": len(scene.elements_of(ElementKind.WINDOW)),
        "IfcDoor": len(scene.elements_of(ElementKind.DOOR)),
        "IfcSpace": len(scene.elements_of(ElementKind.SPACE)),
        "IfcRelConnectsPathElements": len(scene.elements_of(ElementKind.WALL_JOIN)),
        "IfcProject": 1,
        "IfcSite": 1,
        "IfcBuilding": 1,
        "IfcBuildingStorey": 1,
    }
    for ifc_class, number in expected.items():
        assert counts[ifc_class] == number, ifc_class
        assert len(f.by_type(ifc_class, include_subtypes=False)) == number, ifc_class
    assert counts["IfcRelVoidsElement"] == expected["IfcOpeningElement"]
    assert counts["IfcRelFillsElement"] == expected["IfcWindow"] + expected["IfcDoor"]
    assert layout_export.result.warnings == []
    assert not layout_export.result.incomplete
    assert layout_export.result.flagged_elements == {}


def test_units_are_si_metre(layout_export: Export) -> None:
    f = layout_export.file
    assert ifcopenshell.util.unit.calculate_unit_scale(f) == 1.0
    units = {(unit.UnitType, unit.Name, unit.Prefix) for unit in f.by_type("IfcSIUnit")}
    assert ("LENGTHUNIT", "METRE", None) in units
    assert ("AREAUNIT", "SQUARE_METRE", None) in units
    assert ("VOLUMEUNIT", "CUBIC_METRE", None) in units
    assert ("PLANEANGLEUNIT", "RADIAN", None) in units
    assert not any(unit.Prefix for unit in f.by_type("IfcSIUnit"))


def test_contexts(layout_export: Export) -> None:
    contexts = {
        (c.ContextType, c.ContextIdentifier, getattr(c, "TargetView", None))
        for c in layout_export.file.by_type("IfcGeometricRepresentationContext")
    }
    assert ("Model", None, None) in contexts
    assert ("Model", "Body", "MODEL_VIEW") in contexts
    assert ("Model", "Axis", "GRAPH_VIEW") in contexts
    assert ("Plan", None, None) in contexts


def test_spatial_tree_and_storey(layout_export: Export) -> None:
    f = layout_export.file
    storey_scene = next(iter(layout_export.scene.storeys.values()))
    (storey,) = f.by_type("IfcBuildingStorey")
    assert storey.Name == storey_scene.name
    assert storey.Elevation == storey_scene.elevation
    building = element_util.get_aggregate(storey)
    assert building.is_a("IfcBuilding")
    site = element_util.get_aggregate(building)
    assert site.is_a("IfcSite")
    assert element_util.get_aggregate(site).is_a("IfcProject")
    for wall in f.by_type("IfcWall"):
        assert element_util.get_container(wall) == storey
    for space in f.by_type("IfcSpace"):
        assert element_util.get_aggregate(space) == storey


def test_every_wall_has_type_and_layer_set(layout_export: Export) -> None:
    walls = layout_export.file.by_type("IfcWall")
    assert walls
    for wall in walls:
        wall_type = element_util.get_type(wall)
        assert wall_type is not None and wall_type.is_a("IfcWallType")
        assert wall_type.PredefinedType == "STANDARD"
        assert wall.PredefinedType == "STANDARD"
        type_material = element_util.get_material(wall_type)
        assert type_material is not None and type_material.is_a("IfcMaterialLayerSet")
        usage = element_util.get_material(wall)
        assert usage is not None and usage.is_a("IfcMaterialLayerSetUsage")
        assert usage.ForLayerSet == type_material
        assert usage.LayerSetDirection == "AXIS2"
        assert usage.DirectionSense == "POSITIVE"


def test_types_are_deduplicated(layout_export: Export) -> None:
    f = layout_export.file
    thicknesses = {
        e.params.thickness
        for e in layout_export.scene.elements.values()
        if e.kind == ElementKind.WALL and hasattr(e.params, "thickness")
    }
    assert len(f.by_type("IfcWallType")) == len(thicknesses)
    assert len(f.by_type("IfcWindowType")) == 1
    assert len(f.by_type("IfcDoorType")) == 1


def test_wall_joins(layout_export: Export) -> None:
    f = layout_export.file
    expected = set()
    for element in layout_export.scene.elements.values():
        if not isinstance(element, WallJoinElement):
            continue
        params = element.params
        expected.add(
            (
                entity_by_tag(f, params.wall_a).id(),
                entity_by_tag(f, params.wall_b).id(),
                CONNECTION[params.end_a],
                CONNECTION[params.end_b],
            )
        )
    found = {
        (
            r.RelatingElement.id(),
            r.RelatedElement.id(),
            r.RelatingConnectionType,
            r.RelatedConnectionType,
        )
        for r in f.by_type("IfcRelConnectsPathElements")
    }
    assert found == expected
    expected_path_joins = sum(
        1
        for element in layout_export.scene.elements.values()
        if isinstance(element, WallJoinElement)
        and JoinEnd.PATH in (element.params.end_a, element.params.end_b)
    )
    assert len([key for key in found if "ATPATH" in key]) == expected_path_joins


def test_openings_void_host_wall_and_fillings_fill_openings(layout_export: Export) -> None:
    f = layout_export.file
    for scene_opening in layout_export.scene.elements_of(ElementKind.OPENING):
        opening = entity_by_tag(f, scene_opening.id)
        host = entity_by_tag(f, scene_opening.params.host_wall_id)  # type: ignore[attr-defined]
        assert [r.RelatingBuildingElement for r in opening.VoidsElements] == [host]
        assert not opening.ContainedInStructure
        fillers = [r.RelatedBuildingElement for r in opening.HasFillings]
        assert len(fillers) == 1
        assert fillers[0].is_a() in ("IfcWindow", "IfcDoor")
        assert element_util.get_container(fillers[0]) is not None
    # every window and door fills exactly one opening
    for filler in (*f.by_type("IfcWindow"), *f.by_type("IfcDoor")):
        assert len(filler.FillsVoids) == 1


def test_windows_and_doors_attributes_and_types(layout_export: Export) -> None:
    f = layout_export.file
    scene = layout_export.scene
    for scene_element in scene.elements.values():
        if isinstance(scene_element, WindowElement):
            window = entity_by_tag(f, scene_element.id)
            geometry = layout_export.geometry.windows[scene_element.id]
            assert window.OverallWidth == pytest.approx(geometry.width)
            assert window.OverallHeight == pytest.approx(geometry.height)
            assert window.PredefinedType == "WINDOW"
            assert window.PartitioningType == "SINGLE_PANEL"
            window_type = element_util.get_type(window)
            assert window_type is not None and window_type.PartitioningType == "SINGLE_PANEL"
            assert {p.is_a() for p in window_type.HasPropertySets} == {
                "IfcWindowLiningProperties",
                "IfcWindowPanelProperties",
            }
            panel = next(
                p for p in window_type.HasPropertySets if p.is_a().endswith("PanelProperties")
            )
            assert panel.OperationType == PANEL_OPERATION[scene_element.params.operation.value]
        elif isinstance(scene_element, DoorElement):
            door = entity_by_tag(f, scene_element.id)
            geometry = layout_export.geometry.doors[scene_element.id]
            assert door.OverallWidth == pytest.approx(geometry.width)
            assert door.OverallHeight == pytest.approx(geometry.height)
            assert door.PredefinedType == "DOOR"
            assert door.OperationType == "SINGLE_SWING_LEFT"
            door_type = element_util.get_type(door)
            assert door_type is not None and door_type.OperationType == "SINGLE_SWING_LEFT"
            assert {p.is_a() for p in door_type.HasPropertySets} == {
                "IfcDoorLiningProperties",
                "IfcDoorPanelProperties",
            }


def test_spaces_have_names_bodies_and_boundaries(layout_export: Export) -> None:
    f = layout_export.file
    scene_spaces = [e for e in layout_export.scene.elements.values() if isinstance(e, SpaceElement)]
    assert len(f.by_type("IfcSpace")) == len(scene_spaces)
    for scene_space in scene_spaces:
        space = next(s for s in f.by_type("IfcSpace") if s.Name == scene_space.params.number)
        assert space.LongName == scene_space.params.name
        assert space.PredefinedType == "INTERNAL"
        assert space.Representation is not None
        geometry = layout_export.geometry.spaces[scene_space.id]
        boundaries = list(space.BoundedBy)
        assert len(boundaries) == (
            len(geometry.bounding_wall_ids) + len(geometry.floor_ids) + len(geometry.ceiling_ids)
        )
        wall_boundaries = [b for b in boundaries if b.RelatedBuildingElement.is_a("IfcWall")]
        assert len(wall_boundaries) == len(geometry.bounding_wall_ids)
        assert {b.RelatedBuildingElement.is_a() for b in boundaries} == {"IfcWall", "IfcSlab"}
        for boundary in boundaries:
            assert boundary.is_a() == "IfcRelSpaceBoundary"
            assert boundary.PhysicalOrVirtualBoundary == "PHYSICAL"
            plane = boundary.ConnectionGeometry.SurfaceOnRelatingElement
            assert plane.is_a("IfcCurveBoundedPlane")
