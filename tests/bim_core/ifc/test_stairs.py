"""IFC export of stairs: structure, property sets, validation, IDS and re-import (WP5.3)."""

from __future__ import annotations

from pathlib import Path

import ifcopenshell.util.element as element_util
import pytest

from samcloud.bim.core.model import ElementKind
from samcloud.bim.core.qa import validate_ifc
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, StairSpec, build_room_scene

from .conftest import Export, export_scene, validation_statements
from .test_properties import psets_of


def _export(tmp_path: Path, spec: StairSpec, layout: RoomLayout = RoomLayout.RECTANGLE) -> Export:
    scene = build_room_scene(RoomSpec(layout=layout, two_rooms_size=(10.0, 5.0), stair=spec))
    return export_scene(scene, tmp_path / "stair.ifc", run_id="run-1")


@pytest.fixture(scope="module")
def single(tmp_path_factory: pytest.TempPathFactory) -> Export:
    return _export(tmp_path_factory.mktemp("stair_single"), StairSpec())


@pytest.fixture(scope="module")
def double(tmp_path_factory: pytest.TempPathFactory) -> Export:
    spec = StairSpec(step_count=16, flights=((8, 1.0), (8, 0.0)), start=(0.7, 3.0))
    return _export(tmp_path_factory.mktemp("stair_double"), spec, RoomLayout.TWO_ROOMS)


def test_export_is_valid_and_complete(single: Export, double: Export) -> None:
    for export in (single, double):
        assert validation_statements(export.file) == []
        assert not export.result.incomplete
        assert export.result.warnings == []


def test_stair_aggregates_its_flight_and_is_contained_in_the_storey(single: Export) -> None:
    (stair,) = single.file.by_type("IfcStair")
    assert stair.PredefinedType == "STRAIGHT_RUN_STAIR"
    flights = [p for p in element_util.get_parts(stair) if p.is_a("IfcStairFlight")]
    assert len(flights) == 1
    assert flights[0].NumberOfRisers == 14
    assert flights[0].NumberOfTreads == 13
    assert flights[0].RiserHeight == pytest.approx(0.18)
    assert element_util.get_container(stair).is_a("IfcBuildingStorey")
    assert not flights[0].ContainedInStructure
    assert element_util.get_aggregate(flights[0]) == stair
    stair_type = element_util.get_type(stair)
    assert stair_type is not None and stair_type.is_a("IfcStairType")
    assert stair.Description and stair.ObjectType == stair_type.Name


def test_property_sets_equal_the_parameters(single: Export) -> None:
    (stair,) = single.file.by_type("IfcStair")
    params = next(e for e in single.scene.elements.values() if e.kind == ElementKind.STAIR).params
    psets = psets_of(stair)
    common = psets["Pset_StairCommon"]
    assert common["NumberOfRiser"] == params.step_count
    assert common["NumberOfTreads"] == params.step_count - 1
    assert common["RiserHeight"] == pytest.approx(params.riser_height)
    assert common["TreadLength"] == pytest.approx(params.tread_length)
    quality = psets["Raumlot_Quality"]
    assert quality["StairWidth"] == pytest.approx(params.width)
    assert quality["TotalRise"] == pytest.approx(14 * 0.18)
    assert "Raumlot_Location" in psets
    (flight,) = [p for p in element_util.get_parts(stair) if p.is_a("IfcStairFlight")]
    flight_sets = psets_of(flight)
    assert flight_sets["Pset_StairFlightCommon"]["NumberOfRiser"] == 14
    qto = flight_sets["Qto_StairFlightBaseQuantities"]
    assert qto["Length"] == pytest.approx(13 * 0.28)
    assert qto["GrossVolume"] > 0


def test_flight_body_is_an_extrusion_with_the_expected_box(single: Export) -> None:
    import ifcopenshell.util.shape
    from ifcopenshell import geom

    (stair,) = single.file.by_type("IfcStair")
    (flight,) = [p for p in element_util.get_parts(stair) if p.is_a("IfcStairFlight")]
    shape = geom.create_shape(geom.settings(), flight)
    verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
    zs = verts[:, 2]
    assert zs.min() == pytest.approx(0.0, abs=1e-6)
    assert zs.max() == pytest.approx(13 * 0.18, abs=1e-6)


def test_two_flights_and_a_landing(double: Export) -> None:
    (stair,) = double.file.by_type("IfcStair")
    assert stair.PredefinedType == "TWO_STRAIGHT_RUN_STAIR"
    parts = element_util.get_parts(stair)
    assert sum(1 for p in parts if p.is_a("IfcStairFlight")) == 2
    landings = [p for p in parts if p.is_a("IfcSlab")]
    assert [p.PredefinedType for p in landings] == ["LANDING"]
    assert double.result.counts["IfcStairFlight"] == 2


def test_ids_and_reimport_pass(single: Export, double: Export) -> None:
    for export in (single, double):
        report = validate_ifc(
            export.path,
            scene=export.scene,
            export_result=export.result,
            reports_dir=export.path.parent,
        )
        assert report.express_findings == []
        stair_spec = next(s for s in report.ids_specifications if s["name"] == "IfcStair")
        assert stair_spec["applicable"] == 1 and stair_spec["applicable_failed"] == 0
        assert report.reimport_mismatches == []
        assert report.ok


def test_reimport_detects_a_changed_riser_count(single: Export) -> None:
    from samcloud.bim.core.geometry import compute_scene_geometry
    from samcloud.bim.core.qa.validate import reimport_check

    scene = single.scene.model_copy(deep=True)
    stair = next(e for e in scene.elements.values() if e.kind == ElementKind.STAIR)
    stair.params.riser_height = 0.17
    mismatches = reimport_check(single.file, scene, compute_scene_geometry(scene))
    assert any("RiserHeight" in m.message for m in mismatches)


def test_non_parametric_stair_is_flagged_in_quality(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(stair=StairSpec()))
    stair = next(e for e in scene.elements.values() if e.kind == ElementKind.STAIR)
    stair.non_parametric = True
    stair.notes.append("simplified solid")
    export = export_scene(scene, tmp_path / "np.ifc")
    (ifc_stair,) = export.file.by_type("IfcStair")
    quality = psets_of(ifc_stair)["Raumlot_Quality"]
    assert quality["NonParametric"] is True
    assert "simplified" in quality["Notes"]
    assert validation_statements(export.file) == []
