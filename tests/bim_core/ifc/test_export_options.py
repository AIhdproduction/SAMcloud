"""Georeferencing, determinism, ceiling class switch, tilted ceilings and incomplete exports."""

from __future__ import annotations

import math
from pathlib import Path

import ifcopenshell
import ifcopenshell.util.element as element_util
import ifcopenshell.util.geolocation
import numpy as np
import pytest

from samcloud.bim.core.model import (
    CeilingElement,
    CeilingIfcClass,
    CrsInfo,
    ElementKind,
    FurnitureElement,
    Tristate,
    WallElement,
    WallFunction,
)
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene

from .conftest import FIXED_TIME, Export, entity_by_tag, export_scene, validation_statements
from .test_geometry_fidelity import qto, world_box
from .test_properties import psets_of


def test_georeferencing_round_trip(layout_export: Export) -> None:
    f = layout_export.file
    result = layout_export.result
    assert result.referenced
    (conversion,) = f.by_type("IfcMapConversion")
    (crs,) = f.by_type("IfcProjectedCRS")
    offset = layout_export.scene.settings.origin_offset
    assert conversion.TargetCRS == crs
    assert conversion.SourceCRS.is_a("IfcGeometricRepresentationContext")
    assert crs.Name == "EPSG:2056"
    assert crs.VerticalDatum == "LN02"
    assert crs.MapUnit.Name == "METRE"
    assert (conversion.Eastings, conversion.Northings, conversion.OrthogonalHeight) == offset
    assert (conversion.XAxisAbscissa, conversion.XAxisOrdinate, conversion.Scale) == (1.0, 0.0, 1.0)
    # local + map conversion reproduces the absolute coordinates of every wall corner
    for scene_wall in layout_export.scene.elements_of(ElementKind.WALL):
        geometry = layout_export.geometry.walls[scene_wall.id]
        low, high = world_box(entity_by_tag(f, scene_wall.id))
        for local, source in (
            (low, (min(p[0] for p in geometry.footprint), min(p[1] for p in geometry.footprint))),
            (high, (max(p[0] for p in geometry.footprint), max(p[1] for p in geometry.footprint))),
        ):
            east, north, height = ifcopenshell.util.geolocation.auto_xyz2enh(
                f, local[0], local[1], local[2]
            )
            assert east == pytest.approx(source[0] + offset[0], abs=1e-6)
            assert north == pytest.approx(source[1] + offset[1], abs=1e-6)
            assert height == pytest.approx(local[2] + offset[2], abs=1e-6)
            back = ifcopenshell.util.geolocation.auto_enh2xyz(f, east, north, height)
            assert back == pytest.approx(local, abs=1e-6)
    # the Raumlot_Location coordinates agree with the conversion
    location = psets_of(entity_by_tag(f, layout_export.scene.elements_of(ElementKind.WALL)[0].id))[
        "Raumlot_Location"
    ]
    low, high = world_box(entity_by_tag(f, layout_export.scene.elements_of(ElementKind.WALL)[0].id))
    enh = ifcopenshell.util.geolocation.auto_xyz2enh(f, low[0], low[1], low[2])
    assert location["BBoxMinX"] == pytest.approx(enh[0], abs=1e-6)
    assert location["BBoxMinZ"] == pytest.approx(enh[2], abs=1e-6)


def test_header_marks_unreferenced_and_has_injected_timestamp(
    rectangle_export: Export, layout_export: Export
) -> None:
    header = rectangle_export.file.header
    text = " ".join(header.file_description.description)
    assert "unreferenced" in text
    assert header.file_name.time_stamp == FIXED_TIME.isoformat(timespec="seconds")
    assert header.file_name.name == "model.ifc"
    assert "unreferenced" not in " ".join(layout_export.file.header.file_description.description)
    assert "EPSG:2056" in " ".join(layout_export.file.header.file_description.description)


def test_unreferenced_export_has_no_map_conversion(rectangle_export: Export) -> None:
    f = rectangle_export.file
    assert not rectangle_export.result.referenced
    assert not f.by_type("IfcMapConversion")
    assert not f.by_type("IfcProjectedCRS")
    offset = rectangle_export.scene.settings.origin_offset
    for scene_wall in rectangle_export.scene.elements_of(ElementKind.WALL):
        location = psets_of(entity_by_tag(f, scene_wall.id))["Raumlot_Location"]
        assert location["CrsName"] == "unreferenced"
        assert location["HeightSystem"] == "unreferenced"
        geometry = rectangle_export.geometry.walls[scene_wall.id]
        # coordinates of the source cloud (local + origin offset), labelled unreferenced
        assert location["BBoxMinX"] == pytest.approx(
            min(p[0] for p in geometry.footprint) + offset[0], abs=1e-6
        )
    assert validation_statements(f) == []


def test_crs_with_only_wkt_is_referenced(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    scene.settings.crs = CrsInfo(wkt='PROJCS["demo"]', name="Demo CRS", source="user")
    export = export_scene(scene, tmp_path / "model.ifc")
    (crs,) = export.file.by_type("IfcProjectedCRS")
    assert crs.Name == "Demo CRS"
    assert crs.Description == 'PROJCS["demo"]'
    assert export.result.referenced
    assert validation_statements(export.file) == []


def test_crs_without_code_is_unreferenced_with_warning(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    scene.settings.crs = CrsInfo(name="mystery", source="none")
    export = export_scene(scene, tmp_path / "model.ifc")
    assert not export.result.referenced
    assert not export.file.by_type("IfcMapConversion")
    assert any("unreferenced" in w for w in export.result.warnings)


def test_export_is_deterministic(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))
    scene.settings.crs = CrsInfo(epsg=2056, vertical_datum="LN02", source="input")
    first = export_scene(scene, tmp_path / "a.ifc", run_id="r")
    second = export_scene(scene, tmp_path / "b.ifc", run_id="r")
    text_a = first.path.read_text(encoding="utf-8").replace("a.ifc", "x.ifc")
    text_b = second.path.read_text(encoding="utf-8").replace("b.ifc", "x.ifc")
    assert text_a == text_b


def test_global_ids_stable_and_unique(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))
    first = export_scene(scene, tmp_path / "a.ifc")
    second = export_scene(scene, tmp_path / "b.ifc")

    def ids(export: Export) -> list[tuple[str, str]]:
        return sorted((e.is_a(), e.GlobalId) for e in export.file.by_type("IfcRoot"))

    assert ids(first) == ids(second)
    guids = [guid for _cls, guid in ids(first)]
    assert len(guids) == len(set(guids))
    assert all(len(guid) == 22 for guid in guids)


def test_global_ids_survive_adding_an_element(tmp_path: Path) -> None:
    """Scene element ids keep their GlobalId when other elements are removed."""
    scene = build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))
    first = export_scene(scene, tmp_path / "a.ifc")
    partition = next(e for e in scene.elements.values() if e.tag == "partition")
    smaller = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    second = export_scene(smaller, tmp_path / "b.ifc")
    for element_id in (e.id for e in smaller.elements_of(ElementKind.WALL)):
        assert (
            entity_by_tag(first.file, element_id).GlobalId
            == entity_by_tag(second.file, element_id).GlobalId
        )
    assert partition.id not in {e.Tag for e in second.file.by_type("IfcElement")}


def test_ceiling_as_covering(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    scene.settings.ceiling_ifc_class = CeilingIfcClass.COVERING
    export = export_scene(scene, tmp_path / "model.ifc")
    assert validation_statements(export.file) == []
    f = export.file
    (ceiling,) = f.by_type("IfcCovering")
    assert ceiling.PredefinedType == "CEILING"
    assert len(f.by_type("IfcSlab")) == 1
    ceiling_type = element_util.get_type(ceiling)
    assert ceiling_type.is_a("IfcCoveringType") and ceiling_type.PredefinedType == "CEILING"
    geometry = export.geometry.ceilings[ceiling.Tag]
    values = qto(ceiling, "Qto_CoveringBaseQuantities")
    assert values == pytest.approx(
        {
            "Width": geometry.thickness,
            "GrossArea": geometry.gross_area,
            "NetArea": geometry.net_area,
        }
    )
    assert "Pset_CoveringCommon" in psets_of(ceiling)
    assert "Qto_SlabBaseQuantities" not in psets_of(ceiling)
    low, high = world_box(ceiling)
    assert low[2] == pytest.approx(geometry.bottom_z, abs=1e-6)
    assert high[2] == pytest.approx(geometry.top_z, abs=1e-6)
    space = f.by_type("IfcSpace")[0]
    assert {b.RelatedBuildingElement.is_a() for b in space.BoundedBy} == {
        "IfcWall",
        "IfcSlab",
        "IfcCovering",
    }


def test_ceiling_as_slab_is_default(rectangle_export: Export) -> None:
    f = rectangle_export.file
    assert not f.by_type("IfcCovering")
    ceiling_type = next(t for t in f.by_type("IfcSlabType") if t.PredefinedType == "USERDEFINED")
    assert ceiling_type.ElementType == "CEILING"
    assert "Pset_SlabCommon" in psets_of(
        next(s for s in f.by_type("IfcSlab") if s.PredefinedType == "USERDEFINED")
    )


def test_tilted_ceiling_follows_the_plane(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    ceiling = next(e for e in scene.elements.values() if isinstance(e, CeilingElement))
    ceiling.params.plane_normal = (0.1, 0.05, 1.0)
    ceiling.params.plane_offset = 3.0
    export = export_scene(scene, tmp_path / "model.ifc")
    assert validation_statements(export.file) == []
    geometry = export.geometry.ceilings[ceiling.id]
    assert len(set(geometry.outline_z)) > 1
    product = entity_by_tag(export.file, ceiling.id)
    low, high = world_box(product)
    assert low[2] == pytest.approx(min(geometry.outline_z), abs=1e-6)
    assert high[2] == pytest.approx(max(geometry.outline_z) + geometry.thickness, abs=1e-6)
    from .test_geometry_fidelity import shape_vertices

    vertices = shape_vertices(product, world=True)
    for x, y, z in vertices:
        # every vertex lies on the underside plane or exactly one thickness above it
        under = (3.0 - 0.1 * x - 0.05 * y) / 1.0
        assert z == pytest.approx(under, abs=1e-6) or z == pytest.approx(
            under + geometry.thickness, abs=1e-6
        )
    pitch = psets_of(product)["Pset_SlabCommon"]["PitchAngle"]
    assert pitch == pytest.approx(math.acos(1.0 / math.sqrt(0.1**2 + 0.05**2 + 1.0)))
    assert qto(product, "Qto_SlabBaseQuantities")["NetArea"] == pytest.approx(geometry.net_area)
    assert np.isfinite(vertices).all()


def test_missing_data_flags_elements_and_marks_export_incomplete(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    wall = next(e for e in scene.elements.values() if isinstance(e, WallElement))
    wall.params.function = WallFunction.UNKNOWN
    wall.params.load_bearing = Tristate.TRUE
    export = export_scene(scene, tmp_path / "model.ifc")
    result = export.result
    assert result.incomplete
    assert set(result.flagged_elements) == {wall.id}
    assert any("IsExternal" in w for w in result.warnings)
    product = entity_by_tag(export.file, wall.id)
    quality = psets_of(product)["Raumlot_Quality"]
    assert quality["Incomplete"] is True
    assert "IsExternal" in str(quality["MissingAttributes"])
    common = psets_of(product)["Pset_WallCommon"]
    assert "IsExternal" not in common
    assert common["LoadBearing"] is True
    assert "incomplete" in " ".join(export.file.header.file_description.description)
    assert validation_statements(export.file) == []


def test_unsupported_kind_is_reported_not_exported(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    furniture = FurnitureElement(name="Chair")
    scene.add(furniture)
    export = export_scene(scene, tmp_path / "model.ifc")
    assert export.result.incomplete
    assert furniture.id in export.result.flagged_elements
    assert not export.file.by_type("IfcFurniture")


def test_scene_without_storey_exports_to_building_and_flags(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    for element in scene.elements.values():
        element.storey_id = None
    scene.storeys.clear()
    export = export_scene(scene, tmp_path / "model.ifc")
    assert export.result.incomplete
    assert not export.file.by_type("IfcBuildingStorey")
    wall = export.file.by_type("IfcWall")[0]
    assert element_util.get_container(wall).is_a("IfcBuilding")
    location = psets_of(wall)["Raumlot_Location"]
    assert location["StoreyName"] == "not applicable"
    assert location["StoreyElevation"] == "not applicable"
    assert validation_statements(export.file) == []


def test_run_id_and_backend_default_to_not_applicable(rectangle_export: Export) -> None:
    wall = rectangle_export.file.by_type("IfcWall")[0]
    quality = psets_of(wall)["Raumlot_Quality"]
    assert quality["RunId"] == "not applicable"
    assert quality["AiBackend"] == "not applicable"
    assert quality["AiModelId"] == "not applicable"


def test_result_paths_and_file_written(rectangle_export: Export) -> None:
    assert Path(rectangle_export.result.path) == rectangle_export.path
    assert rectangle_export.path.is_file()
    assert isinstance(rectangle_export.file, ifcopenshell.file)
