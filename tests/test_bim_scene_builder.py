"""Scene building and IFC export from a synthetic BuildingPlan (two rooms, door, window, furniture)."""

from __future__ import annotations

import json
from pathlib import Path

import ifcopenshell
import ifcopenshell.api.pset
import ifcopenshell.geom
import ifcopenshell.util.element as element_util
import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from samcloud.bim.core.geometry import compute_scene_geometry
from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import ElementKind
from samcloud.bim.core.qa.validate import validate_ifc
from samcloud.bim.export import BimExportResult, export_level
from samcloud.bim.plan_model import (
    BuildingPlan,
    PlanFurniture,
    PlanOpening,
    PlanRoom,
    PlanWall,
    save_plan_model,
)
from samcloud.bim.scene_builder import build_scene

FLOOR_Z = 0.5
CEILING_Z = 3.1


def _wall(wall_id: str, a: tuple[float, float], b: tuple[float, float], thickness: float, kind: str) -> PlanWall:
    return PlanWall(id=wall_id, centreline=[list(a), list(b)], thickness_m=thickness, kind=kind)


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def make_plan(units_per_metre: float = 1.0) -> BuildingPlan:
    """8 x 5 m flat: exterior walls 0.30 m, partition at x=4 (0.20 m), two rooms."""
    u = units_per_metre

    def pts(points: list[list[float]]) -> list[list[float]]:
        return [[x * u, y * u] for x, y in points]

    walls = [
        _wall("W001", (0, 0), (8, 0), 0.30, "exterior"),
        _wall("W002", (8, 0), (8, 5), 0.30, "exterior"),
        _wall("W003", (8, 5), (0, 5), 0.30, "exterior"),
        _wall("W004", (0, 5), (0, 0), 0.30, "exterior"),
        _wall("W005", (4, 0), (4, 5), 0.20, "interior"),
        _wall("W006", (4, 0.1), (4.0, 4.9), 0.20, "interior"),  # stacked duplicate of W005
    ]
    for wall in walls:
        wall.centreline = pts(wall.centreline)
    rooms = [
        PlanRoom("R1", "Living", {"exterior": pts(_rect(0.15, 0.15, 3.9, 4.85)), "holes": []}, 3.75 * 4.7, [2 * u, 2.5 * u]),
        PlanRoom("R2", "Bedroom", {"exterior": pts(_rect(4.1, 0.15, 7.85, 4.85)), "holes": []}, 3.75 * 4.7, [6 * u, 2.5 * u]),
    ]
    openings = [
        PlanOpening("O001", "door", [4.0 * u, 2.5 * u], 0.9, 1.5707963267948966, 0.2, host_wall_id="W005", sill_m=0.0, head_m=2.05),
        PlanOpening("O002", "window", [2.0 * u, 0.0], 1.2, 0.0, 0.3, host_wall_id="W001", sill_m=0.9, head_m=2.1, mullion_count=2),
    ]

    def z(value: float) -> float:
        return (FLOOR_Z + value) * u

    furniture = [
        PlanFurniture("F001", 7, "Sofa", "sofa", None, "A-FURN", pts(_rect(1.0, 1.0, 2.8, 1.9)), z_min=z(0.0), z_max=z(0.85), z_source="instance_bbox", confidence=0.9),
        PlanFurniture("F002", None, "WC", "WC", "WC_WALL", "P-SANI", pts(_rect(6.0, 0.4, 6.4, 0.9)), default_height_m=0.4),
        PlanFurniture("F003", None, "upper_cabinet", "upper_cabinet", "upper_cabinet", "A-OBEN", pts(_rect(5.0, 4.4, 6.5, 4.8)), default_z_m=[1.45, 2.15]),
        PlanFurniture("F004", 9, "Objekt", "Objekt", None, "A-FURN", pts(_rect(1.0, 3.0, 1.5, 3.4)), default_height_m=0.5),
    ]
    return BuildingPlan(
        project_name="synthetic", level_id="level_1", level_name="1", units_per_metre=units_per_metre,
        frame={"origin": [1.0, 2.0, 3.0], "scale_factor": 1.0, "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        floor_z=FLOOR_Z * u, ceiling_z=CEILING_Z * u, walls=walls, rooms=rooms, openings=openings, furniture=furniture,
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    scene = build_scene(make_plan())
    path = tmp_path_factory.mktemp("ifc") / "synthetic.ifc"
    result = write_ifc(scene, path)
    return scene, result, ifcopenshell.open(str(path)), path


def _bbox(entity) -> np.ndarray:
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    verts = np.array(ifcopenshell.geom.create_shape(settings, entity).geometry.verts).reshape(-1, 3)
    return np.array([verts.min(axis=0), verts.max(axis=0)])


def test_duplicate_wall_piece_is_removed(built) -> None:
    scene, _, file, _ = built
    report = scene.derived["samcloud_build"]

    assert len(file.by_type("IfcWall")) == 5
    assert report["walls_in"] == 6
    assert report["duplicates_merged"] == 1
    assert report["walls_out"] == 5


def test_wall_ends_meet_without_overlap_or_gap(built) -> None:
    scene, *_ = built
    geometry = compute_scene_geometry(scene)
    footprints = [Polygon(g.footprint) for g in geometry.walls.values()]

    for i, a in enumerate(footprints):
        for b in footprints[i + 1:]:
            assert a.intersection(b).area <= 0.05 * min(a.area, b.area)
    union = unary_union(footprints)
    assert union.geom_type == "Polygon"
    assert len(union.interiors) == 2  # closed ring: one hole per room
    assert scene.derived["samcloud_build"]["wall_overlaps"] == []
    assert scene.derived["samcloud_build"]["joins_l"] == 4
    assert scene.derived["samcloud_build"]["joins_t"] == 2


def test_two_spaces_with_plan_areas(built) -> None:
    _, _, file, _ = built
    spaces = {s.Name: s for s in file.by_type("IfcSpace")}

    assert sorted(spaces) == ["Bedroom", "Living"]
    for space in spaces.values():
        quantities = element_util.get_pset(space, "Qto_SpaceBaseQuantities")
        assert quantities["NetFloorArea"] == pytest.approx(3.75 * 4.7, rel=0.01)
        assert quantities["Height"] == pytest.approx(CEILING_Z - FLOOR_Z)
        assert space.LongName == space.Name
    box = _bbox(spaces["Living"])
    assert box[0][2] == pytest.approx(FLOOR_Z)
    assert box[1][2] == pytest.approx(CEILING_Z, abs=1e-3)


def test_ceilings_end_at_the_wall_faces(built) -> None:
    scene, _, file, _ = built
    geometry = compute_scene_geometry(scene)
    walls = unary_union([Polygon(g.footprint) for g in geometry.walls.values()])

    assert len(file.by_type("IfcCovering")) == 2
    for ceiling in geometry.ceilings.values():
        shell = Polygon(ceiling.outline)
        assert shell.intersection(walls).area <= 1e-4  # 1 cm2
        assert shell.area == pytest.approx(3.75 * 4.7, rel=0.01)
        assert ceiling.elevation == pytest.approx(CEILING_Z)


def test_floor_slab_covers_rooms_and_walls(built) -> None:
    scene, *_ = built
    slabs = scene.elements_of(ElementKind.SLAB)

    assert len(slabs) == 1
    assert Polygon(slabs[0].params.outline).area == pytest.approx(8.3 * 5.3, rel=0.01)  # outer wall faces
    assert slabs[0].params.top_elevation == pytest.approx(FLOOR_Z)
    assert slabs[0].params.thickness == pytest.approx(0.20)


def test_opening_positions_along_the_wall(built) -> None:
    _, _, file, _ = built
    openings = {o.Name: o for o in file.by_type("IfcOpeningElement")}

    door = _bbox(openings["Door opening O001"])
    assert (door[0][1] + door[1][1]) / 2.0 == pytest.approx(2.5, abs=1e-3)
    assert door[1][1] - door[0][1] == pytest.approx(0.9, abs=1e-3)
    assert door[0][2] == pytest.approx(FLOOR_Z, abs=1e-3)
    assert door[1][2] == pytest.approx(FLOOR_Z + 2.05, abs=1e-3)
    window = _bbox(openings["Window opening O002"])
    assert (window[0][0] + window[1][0]) / 2.0 == pytest.approx(2.0, abs=1e-3)
    assert window[1][0] - window[0][0] == pytest.approx(1.2, abs=1e-3)
    assert window[0][2] == pytest.approx(FLOOR_Z + 0.9, abs=1e-3)
    assert window[1][2] == pytest.approx(FLOOR_Z + 2.1, abs=1e-3)
    assert len(file.by_type("IfcDoor")) == 1 and len(file.by_type("IfcWindow")) == 1


def test_window_mullion_count_is_stored(built) -> None:
    _, _, file, _ = built
    window = file.by_type("IfcWindow")[0]

    assert element_util.get_pset(window, "Raumlot_User")["MullionCount"] == 2


def test_furniture_classes_and_types(built) -> None:
    _, _, file, _ = built
    by_name = {
        e.Name: e
        for e in [*file.by_type("IfcFurniture"), *file.by_type("IfcSanitaryTerminal"), *file.by_type("IfcBuildingElementProxy")]
    }

    sofa = by_name["Sofa 01"]
    assert sofa.is_a() == "IfcFurniture" and sofa.PredefinedType == "SOFA"
    assert element_util.get_type(sofa).is_a() == "IfcFurnitureType"
    assert element_util.get_type(sofa).PredefinedType == "SOFA"
    wc = by_name["WC 01"]
    assert wc.is_a() == "IfcSanitaryTerminal" and wc.PredefinedType == "TOILETPAN"
    cabinet = by_name["upper_cabinet 01"]
    assert (cabinet.is_a(), cabinet.PredefinedType, cabinet.ObjectType) == ("IfcFurniture", "USERDEFINED", "WallCabinet")
    assert element_util.get_type(cabinet).ElementType == "WallCabinet"
    unknown = by_name["Objekt 01"]
    assert (unknown.is_a(), unknown.PredefinedType, unknown.ObjectType) == ("IfcBuildingElementProxy", "NOTDEFINED", "Objekt")


def test_furniture_z_ranges(built) -> None:
    _, _, file, _ = built
    by_name = {e.Name: e for e in file.by_type("IfcProduct") if e.Name}

    sofa = _bbox(by_name["Sofa 01"])
    assert (sofa[0][2], sofa[1][2]) == pytest.approx((FLOOR_Z, FLOOR_Z + 0.85), abs=1e-6)  # instance z
    assert sofa[0][0] == pytest.approx(1.0, abs=1e-6) and sofa[1][0] == pytest.approx(2.8, abs=1e-6)
    wc = _bbox(by_name["WC 01"])
    assert (wc[0][2], wc[1][2]) == pytest.approx((FLOOR_Z, FLOOR_Z + 0.4), abs=1e-6)  # default height
    cabinet = _bbox(by_name["upper_cabinet 01"])
    assert (cabinet[0][2], cabinet[1][2]) == pytest.approx((FLOOR_Z + 1.45, FLOOR_Z + 2.15), abs=1e-6)  # hanging


def test_furniture_scan_pset_style_and_room_reference(built) -> None:
    _, _, file, _ = built
    sofa = next(e for e in file.by_type("IfcFurniture") if e.Name == "Sofa 01")

    scan = element_util.get_pset(sofa, "SAMcloud_Scan")
    assert scan["InstanceId"] == 7 and scan["ZSource"] == "instance_bbox" and scan["Label"] == "Sofa"
    assert scan["Confidence"] == pytest.approx(0.9)
    assert scan["Height"] == pytest.approx(0.85)
    assert scan["Length"] == pytest.approx(1.8) and scan["Width"] == pytest.approx(0.9)
    assert element_util.get_container(sofa).is_a("IfcBuildingStorey")
    assert [r.RelatingStructure.Name for r in sofa.ReferencedInStructures] == ["Living"]
    wc = file.by_type("IfcSanitaryTerminal")[0]
    assert [r.RelatingStructure.Name for r in wc.ReferencedInStructures] == ["Bedroom"]
    representations = sofa.Representation.Representations
    assert {rep.RepresentationIdentifier for rep in representations} == {"Body", "Box", "FootPrint"}
    body = next(rep for rep in representations if rep.RepresentationIdentifier == "Body")
    assert all(item.StyledByItem for item in body.Items), "body must carry a surface style"


def test_project_frame_pset(built) -> None:
    _, _, file, _ = built
    frame = element_util.get_pset(file.by_type("IfcProject")[0], "SAMcloud_Frame")

    assert frame["UnitsPerMetre"] == 1.0 and frame["LevelId"] == "level_1"
    assert json.loads(frame["Origin"]) == [1.0, 2.0, 3.0]


def test_export_is_valid_and_deterministic(built, tmp_path: Path) -> None:
    scene, result, file, path = built
    report = validate_ifc(path, scene=scene, export_result=result, reports_dir=tmp_path)

    assert report.express_findings == []
    assert report.ids_failures == []
    assert report.reimport_checked and report.reimport_mismatches == []
    other = write_ifc(build_scene(make_plan()), tmp_path / "again.ifc")
    ids_first = sorted(e.GlobalId for e in file.by_type("IfcRoot"))
    ids_second = sorted(e.GlobalId for e in ifcopenshell.open(other.path).by_type("IfcRoot"))
    assert ids_first == ids_second


def test_ids_catches_missing_furniture_data(built, tmp_path: Path) -> None:
    _, _, _, path = built
    file = ifcopenshell.open(str(path))
    sofa = next(e for e in file.by_type("IfcFurniture") if e.Name == "Sofa 01")
    pset = element_util.get_psets(sofa)["SAMcloud_Scan"]
    ifcopenshell.api.pset.edit_pset(file, pset=file.by_id(pset["id"]), properties={"ZSource": None})
    broken = tmp_path / "broken.ifc"
    file.write(str(broken))

    failures = validate_ifc(broken, reports_dir=tmp_path).ids_failures

    assert [f.name for f in failures] == ["Sofa 01"]
    assert "ZSource" in failures[0].requirement


def test_units_per_metre_is_applied() -> None:
    scene = build_scene(make_plan(units_per_metre=2.0))
    walls = scene.elements_of(ElementKind.WALL)
    storey = next(iter(scene.storeys.values()))

    assert storey.elevation == pytest.approx(FLOOR_Z) and storey.height == pytest.approx(CEILING_Z - FLOOR_Z)
    assert max(w.params.end[0] for w in walls) == pytest.approx(8.0)
    areas = [Polygon(s.params.outline).area for s in scene.elements_of(ElementKind.SPACE)]
    assert areas == pytest.approx([3.75 * 4.7, 3.75 * 4.7], rel=1e-6)


def test_single_line_wall_gets_the_default_thickness() -> None:
    plan = make_plan()
    plan.walls[4].thickness_m = 0.0
    scene = build_scene(plan)

    partition = next(w for w in scene.elements_of(ElementKind.WALL) if w.name == "W005")
    assert partition.params.thickness == pytest.approx(0.12)
    assert any("single line" in w for w in scene.derived["samcloud_build"]["warnings"])
    assert scene.derived["samcloud_build"]["single_line_default_thickness"] == 1


def test_opening_in_a_wall_gap_bridges_the_gap() -> None:
    plan = make_plan()
    # partition split at the door: two jamb pieces with a 0.9 m gap, the door sits in the gap
    plan.walls = [w for w in plan.walls if w.id not in ("W005", "W006")]
    plan.walls += [
        _wall("W005", (4, 0), (4, 2.05), 0.20, "interior"),
        _wall("W006", (4, 2.95), (4, 5), 0.20, "interior"),
    ]
    scene = build_scene(plan)

    partitions = [
        w
        for w in scene.elements_of(ElementKind.WALL)
        if w.params.start[0] == pytest.approx(4.0) and w.params.end[0] == pytest.approx(4.0)
    ]
    assert len(partitions) == 1
    assert abs(partitions[0].params.end[1] - partitions[0].params.start[1]) == pytest.approx(5.0)
    assert scene.derived["samcloud_build"]["bridged"] == 1
    assert scene.derived["samcloud_build"]["openings_hosted"] == 2


def test_opening_without_host_is_skipped_with_warning() -> None:
    plan = make_plan()
    plan.openings.append(PlanOpening("O003", "door", [30.0, 30.0], 0.9, 0.0, 0.2, host_wall_id=None))
    scene = build_scene(plan)

    assert scene.derived["samcloud_build"]["openings_skipped"] == 1
    assert any("O003" in w and "no host" in w for w in scene.derived["samcloud_build"]["warnings"])
    assert len(scene.elements_of(ElementKind.DOOR)) == 1


def test_crossing_walls_are_split_into_two_t_joins() -> None:
    plan = make_plan()
    plan.walls.append(_wall("W007", (2, -1), (2, 3), 0.10, "interior"))  # crosses the south wall
    scene = build_scene(plan)

    assert scene.derived["samcloud_build"]["split_crossings"] == 1
    assert scene.derived["samcloud_build"]["wall_overlaps"] == []


def test_export_level_writes_files_and_qa(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    save_plan_model(make_plan(), project / "products" / "dxf" / "level_1" / "ortho")
    calls: list[tuple[float, str]] = []

    result = export_level(project, "level_1", progress=lambda f, s: calls.append((f, s)))

    assert isinstance(result, BimExportResult)
    assert result.ifc_path == project / "products" / "bim" / "level_1" / "synthetic_level_1.ifc"
    assert result.ifc_path.is_file() and result.qa_html_path.is_file()
    assert result.ids_failures == 0 and result.express_findings == 0
    assert result.counts["IfcWall"] == 5 and result.counts["IfcSpace"] == 2
    qa = json.loads(result.qa_json_path.read_text(encoding="utf-8"))
    assert set(qa) >= {"ids_failures", "express_findings", "reimport_ok", "elements", "counts", "warnings", "summary"}
    assert qa["reimport_ok"] is True and qa["wall_overlap_count"] == 0
    assert qa["consistency"]["spaces"] == {"plan_rooms": 2, "ifc_spaces": 2}
    assert qa["consistency"]["furniture"] == {"plan": 4, "ifc": 4}
    fractions = [f for f, _ in calls]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0


def test_export_level_cancel_raises_between_stages(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    save_plan_model(make_plan(), project / "products" / "dxf" / "level_1" / "ortho")
    state = {"n": 0}

    def cancel() -> None:
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        export_level(project, "level_1", cancel=cancel)
    assert not list((project / "products" / "bim" / "level_1").glob("*.ifc"))


def test_export_level_without_plan_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_level(tmp_path, "level_1")
