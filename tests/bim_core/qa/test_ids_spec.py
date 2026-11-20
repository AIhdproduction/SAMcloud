from __future__ import annotations

from pathlib import Path

from ifctester import ids

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.qa import IDS_PATH, build_ids, write_ids
from samcloud.bim.core.qa import validate_ifc
from samcloud.bim.core.qa.ids_spec import SPECS
from samcloud.bim.core.synth.room import RoomSpec, build_room_scene

EXPECTED_SPECIFICATIONS = [
    "IfcWall",
    "IfcSlab floor",
    "IfcSlab ceiling",
    "IfcCovering ceiling",
    "IfcOpeningElement",
    "IfcWindow",
    "IfcDoor",
    "IfcSpace",
    "IfcStair",
    "IfcFurniture",
    "IfcElectricAppliance",
    "IfcSanitaryTerminal",
    "IfcAudioVisualAppliance",
    "IfcLightFixture",
    "IfcSpaceHeater",
    "IfcBuildingElementProxy",
    "IfcBuildingStorey",
]


def test_checked_in_ids_equals_generated(tmp_path: Path) -> None:
    generated = write_ids(tmp_path / "generated.ids")

    assert IDS_PATH.read_bytes() == generated.read_bytes()


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = write_ids(tmp_path / "a.ids").read_bytes()
    second = write_ids(tmp_path / "b.ids").read_bytes()

    assert first == second


def test_one_specification_per_element_type() -> None:
    assert [spec.name for spec in SPECS] == EXPECTED_SPECIFICATIONS
    document = ids.open(str(IDS_PATH), validate=True)

    assert [spec.name for spec in document.specifications] == EXPECTED_SPECIFICATIONS
    assert all(spec.ifcVersion == ["IFC4"] for spec in document.specifications)


def test_requirements_cover_mandatory_attributes() -> None:
    names = {spec.name: spec for spec in build_ids().specifications}
    wall = names["IfcWall"]
    labels = {
        (getattr(facet, "propertySet", None), getattr(facet, "baseName", None))
        for facet in wall.requirements
    }

    assert ("Qto_WallBaseQuantities", "Width") in labels
    assert ("Pset_WallCommon", "IsExternal") in labels
    assert ("Raumlot_Quality", "ParameterOrigins") in labels
    assert ("Raumlot_Location", "BBoxMaxZ") in labels
    assert any(type(facet).__name__ == "PartOf" for facet in wall.requirements)
    assert any(getattr(facet, "name", None) == "Tag" for facet in wall.requirements)


def test_description_and_object_type_are_required_everywhere_they_are_written() -> None:
    for spec in build_ids().specifications:
        if spec.name == "IfcBuildingStorey":
            continue
        required = {getattr(f, "name", None) for f in spec.requirements}
        assert {"Description", "ObjectType"} <= required, spec.name


def test_indoor_fixture_ids_result_is_unchanged(tmp_path: Path) -> None:
    result = write_ifc(build_room_scene(RoomSpec()), tmp_path / "indoor.ifc")
    report = validate_ifc(result.path, reports_dir=tmp_path / "reports")

    assert report.ids_failures == []
    applicable = {
        item["name"]: item["applicable"]
        for item in report.ids_specifications
        if item["applicable"]
    }
    assert applicable == {
        "IfcWall": 4,
        "IfcSlab floor": 1,
        "IfcSlab ceiling": 1,
        "IfcOpeningElement": 3,
        "IfcWindow": 2,
        "IfcDoor": 1,
        "IfcSpace": 1,
        "IfcBuildingStorey": 1,
    }
