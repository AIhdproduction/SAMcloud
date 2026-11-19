"""Property sets, quantities, identity and the Raumlot psets."""

from __future__ import annotations

import ifcopenshell.util.element as element_util
import ifcopenshell.util.pset
import pytest

from samcloud.bim.core.model import (
    DoorElement,
    ElementKind,
    PropertyValue,
    SpaceElement,
    WallElement,
    WindowElement,
)
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene

from .conftest import Export, entity_by_tag, export_scene

PRODUCT_CLASSES = ("IfcWall", "IfcSlab", "IfcOpeningElement", "IfcWindow", "IfcDoor", "IfcSpace")
NOT_APPLICABLE = "not applicable"


def all_products(export: Export) -> list[ifcopenshell.entity_instance]:
    return [e for cls in PRODUCT_CLASSES for e in export.file.by_type(cls)]


def psets_of(element: ifcopenshell.entity_instance) -> dict[str, dict[str, object]]:
    return {
        name: {k: v for k, v in values.items() if k != "id"}
        for name, values in element_util.get_psets(element, should_inherit=False).items()
    }


def test_every_element_has_quality_and_location(layout_export: Export) -> None:
    products = all_products(layout_export)
    assert len(products) == len(
        [e for e in layout_export.scene.elements.values() if e.kind != ElementKind.WALL_JOIN]
    )
    for product in products:
        sets = psets_of(product)
        assert "Raumlot_Quality" in sets, product.Name
        assert "Raumlot_Location" in sets, product.Name


def test_quality_values(layout_export: Export) -> None:
    for scene_element in layout_export.scene.elements.values():
        if scene_element.kind == ElementKind.WALL_JOIN:
            continue
        quality = psets_of(_product(layout_export, scene_element.id))["Raumlot_Quality"]
        assert quality["ElementId"] == scene_element.id
        assert quality["Provenance"] == "script"
        assert quality["RunId"] == "run-1"
        assert quality["AiBackend"] == "openrouter"
        assert quality["AiModelId"] == "test-model"
        assert quality["ToleranceProfile"] == "tls"
        assert quality["DataSource"] == "slam"
        assert quality["Locked"] is False
        assert quality["Status"] == "unsure"
        assert quality["Incomplete"] is False
        assert "MissingAttributes" not in quality
        origins = str(quality["ParameterOrigins"])
        for name in scene_element.param_origin:
            assert f"{name}=template" in origins
            assert quality[f"Origin_{name}"] == "template"


def _product(export: Export, element_id: str) -> ifcopenshell.entity_instance:
    element = export.scene.elements[element_id]
    if isinstance(element, SpaceElement):
        return next(s for s in export.file.by_type("IfcSpace") if s.Name == element.params.number)
    return entity_by_tag(export.file, element_id)


def test_sill_height_and_window_extras_in_quality(layout_export: Export) -> None:
    for element in layout_export.scene.elements.values():
        if isinstance(element, WindowElement):
            quality = psets_of(_product(layout_export, element.id))["Raumlot_Quality"]
            assert quality["SillHeight"] == pytest.approx(0.9)
            assert quality["Operation"] == "side_hung_left"
            assert quality["SillProjection"] == 0.0
        elif isinstance(element, DoorElement):
            quality = psets_of(_product(layout_export, element.id))["Raumlot_Quality"]
            assert quality["SillHeight"] == 0.0
            assert quality["SwingSide"] == "left"


def test_location_values_with_crs(layout_export: Export) -> None:
    scene = layout_export.scene
    offset = scene.settings.origin_offset
    storey = next(iter(scene.storeys.values()))
    for scene_element in scene.elements.values():
        if scene_element.kind == ElementKind.WALL_JOIN:
            continue
        location = psets_of(_product(layout_export, scene_element.id))["Raumlot_Location"]
        assert location["SiteName"] == "Site"
        assert location["BuildingName"] == "Building"
        assert location["StoreyName"] == storey.name
        assert location["StoreyElevation"] == pytest.approx(storey.elevation)
        assert location["CrsName"] == "EPSG:2056"
        assert location["HeightSystem"] == "LN02"
        assert location["RoomAssignment"] == "ok"
        geometry = layout_export.geometry.geometry_of(scene_element.id)
        assert geometry is not None
        for axis, index in (("X", 0), ("Y", 1), ("Z", 2)):
            low = location[f"BBoxMin{axis}"]
            high = location[f"BBoxMax{axis}"]
            reference = location[f"Reference{axis}"]
            assert isinstance(low, float) and isinstance(high, float)
            assert low <= high
            # map coordinates are local coordinates plus the origin offset
            assert low > offset[index] - 50.0 and high < offset[index] + 50.0
            assert reference == pytest.approx((low + high) / 2.0) or axis == "Z"
    # wall bounding box in map coordinates reproduces the footprint plus origin offset
    for scene_wall in scene.elements_of(ElementKind.WALL):
        geometry = layout_export.geometry.walls[scene_wall.id]
        location = psets_of(entity_by_tag(layout_export.file, scene_wall.id))["Raumlot_Location"]
        assert location["BBoxMinX"] == pytest.approx(
            min(p[0] for p in geometry.footprint) + offset[0], abs=1e-6
        )
        assert location["BBoxMaxY"] == pytest.approx(
            max(p[1] for p in geometry.footprint) + offset[1], abs=1e-6
        )
        assert location["BBoxMinZ"] == pytest.approx(geometry.base_z + offset[2], abs=1e-6)
        assert location["BBoxMaxZ"] == pytest.approx(geometry.top_z + offset[2], abs=1e-6)


def test_space_names_in_location(layout_export: Export) -> None:
    spaces = [e for e in layout_export.scene.elements.values() if isinstance(e, SpaceElement)]
    for scene_element in layout_export.scene.elements.values():
        if scene_element.kind != ElementKind.WALL:
            continue
        geometry = layout_export.geometry.walls[scene_element.id]
        expected = [
            s
            for s in spaces
            if scene_element.id in layout_export.geometry.spaces[s.id].bounding_wall_ids
            or s.id in scene_element.space_ids
        ]
        location = psets_of(entity_by_tag(layout_export.file, scene_element.id))["Raumlot_Location"]
        assert geometry is not None
        assert location["SpaceNames"] == ", ".join(s.params.name for s in expected)
        assert location["SpaceNumbers"] == ", ".join(s.params.number for s in expected)
        assert expected


def test_identity_attributes(layout_export: Export) -> None:
    import uuid

    import ifcopenshell.guid

    for scene_element in layout_export.scene.elements.values():
        if scene_element.kind == ElementKind.WALL_JOIN:
            continue
        product = _product(layout_export, scene_element.id)
        assert product.GlobalId == ifcopenshell.guid.compress(uuid.UUID(scene_element.id).hex)
        assert product.Name
        if product.is_a("IfcElement"):
            assert product.Tag == scene_element.id
    wall = layout_export.file.by_type("IfcWall")[0]
    assert wall.Name.startswith("Exterior wall")


def test_every_standard_property_name_exists_in_the_template(layout_export: Export) -> None:
    template = ifcopenshell.util.pset.get_template("IFC4")
    checked = 0
    for product in all_products(layout_export):
        for name, values in psets_of(product).items():
            if name.startswith("Raumlot_"):
                continue
            definition = template.get_by_name(name)
            assert definition is not None, name
            known = {p.Name for p in definition.HasPropertyTemplates}
            assert set(values) <= known, (name, set(values) - known)
            checked += len(values)
    assert checked > 50


def test_standard_psets_are_applicable_to_their_class(layout_export: Export) -> None:
    query = ifcopenshell.util.pset.PsetQto("IFC4")
    for product in all_products(layout_export):
        applicable = set(query.get_applicable_names(product.is_a(), product.PredefinedType or ""))
        for name in psets_of(product):
            if not name.startswith("Raumlot_"):
                assert name in applicable, (product.is_a(), name)


def test_user_properties_go_to_standard_or_user_pset(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    wall = next(e for e in scene.elements.values() if isinstance(e, WallElement))
    wall.properties = {
        "FireRating": PropertyValue(value="EI30", value_type="text", pset="Pset_WallCommon"),
        "Colour": PropertyValue(value="white", value_type="text"),
        "Cost": PropertyValue(value=12.5, value_type="number", pset="Acme_Costs"),
        "Checked": PropertyValue(value=True, value_type="boolean", pset="Pset_DoorCommon"),
    }
    path = tmp_path_factory.mktemp("user") / "model.ifc"
    export = export_scene(scene, path)
    sets = psets_of(entity_by_tag(export.file, wall.id))
    assert sets["Pset_WallCommon"]["FireRating"] == "EI30"
    assert sets["Raumlot_User"] == {"Colour": "white", "Cost": 12.5, "Checked": True}
    assert "Pset_DoorCommon" not in sets
    assert sets["Raumlot_Quality"]["PropertyOrigin_Colour"] == "manual"


def test_quality_metrics_are_written_when_present(tmp_path_factory: pytest.TempPathFactory) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    wall = next(e for e in scene.elements.values() if isinstance(e, WallElement))
    wall.metrics.deviation_rms = 0.003
    wall.metrics.deviation_p95 = 0.007
    wall.metrics.deviation_max = 0.012
    wall.metrics.coverage = 0.8
    wall.metrics.confidence = 0.9
    wall.metrics.point_count = 1234
    wall.locked = True
    path = tmp_path_factory.mktemp("metrics") / "model.ifc"
    export = export_scene(scene, path)
    quality = psets_of(entity_by_tag(export.file, wall.id))["Raumlot_Quality"]
    assert quality["DeviationRms"] == pytest.approx(0.003)
    assert quality["DeviationP95"] == pytest.approx(0.007)
    assert quality["DeviationMax"] == pytest.approx(0.012)
    assert quality["Coverage"] == pytest.approx(0.8)
    assert quality["Confidence"] == pytest.approx(0.9)
    assert quality["PointCount"] == 1234
    assert quality["Locked"] is True
    other = next(
        e for e in export.scene.elements.values() if isinstance(e, WallElement) and e.id != wall.id
    )
    assert "DeviationRms" not in psets_of(entity_by_tag(export.file, other.id))["Raumlot_Quality"]
