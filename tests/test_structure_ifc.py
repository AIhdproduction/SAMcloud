"""Measured structure of a BuildingPlan in the IFC: wall heights, per-room / sloped ceilings, IfcColumn, IfcBeam, IfcStair."""

from __future__ import annotations

import math
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element_util
import numpy as np
import pytest

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.qa.validate import validate_ifc
from samcloud.bim.plan_model import BuildingPlan, PlanOpening, PlanRoom, PlanWall
from samcloud.bim.scene_builder import build_scene

U = 2.0  # local units per metre (non-trivial on purpose)
FLOOR = 0.4  # level floor in metres
STOREY = 2.6


def _pts(points):
    return [[x * U, y * U] for x, y in points]


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _z(metres: float) -> float:
    return (FLOOR + metres) * U


def make_plan() -> BuildingPlan:
    """10 x 5 m: room R1 (flat measured ceiling 2.45) | half-height wall | room R2 (sloped ceiling 2.2 -> 3.0)."""
    walls = [
        PlanWall("W001", _pts([(0, 0), (10, 0)]), 0.30, "exterior", height_m=2.45, height_source="room_ceiling"),
        PlanWall("W002", _pts([(10, 0), (10, 5)]), 0.30, "exterior", height_m=3.0, height_source="room_ceiling"),
        PlanWall("W003", _pts([(10, 5), (0, 5)]), 0.30, "exterior", height_m=3.0, height_source="room_ceiling"),
        PlanWall("W004", _pts([(0, 5), (0, 0)]), 0.30, "exterior", height_m=2.45, height_source="room_ceiling"),
        PlanWall("W005", _pts([(5, 0), (5, 5)]), 0.15, "interior", height_m=1.10, height_source="points"),  # knee wall
    ]
    slope = 0.8 / 4.7  # rises 0.8 m across R2 (y direction)
    rooms = [
        PlanRoom("R1", "Wohnen", {"exterior": _pts(_rect(0.15, 0.15, 4.925, 4.85)), "holes": []}, 4.775 * 4.7, [2 * U, 2 * U],
                 floor_z=_z(0.0), ceiling_z=_z(2.45), ceiling_source="points", floor_source="points"),
        PlanRoom("R2", "Atelier", {"exterior": _pts(_rect(5.075, 0.15, 9.85, 4.85)), "holes": []}, 4.775 * 4.7, [7 * U, 2 * U],
                 floor_z=_z(0.0), ceiling_z=_z(2.6), ceiling_source="plane", floor_source="points",
                 # local: z = a x + b y + c  with a = 0, b = slope, c chosen so z(y = 0.15 m) = 2.2 m above the floor
                 ceiling_plane=[0.0, slope, _z(2.2) - slope * 0.15 * U]),
    ]
    openings = [PlanOpening("O001", "door", [5.0 * U, 2.5 * U], 0.9, math.pi / 2, 0.15, host_wall_id="W005", sill_m=0.0, head_m=1.0)]
    column = {"id": "C001", "shape": "rect", "center": [2.5 * U, 2.5 * U], "footprint": _pts(_rect(2.35, 2.35, 2.65, 2.65)),
              "width_m": 0.3, "depth_m": 0.3, "z_min": _z(0.0), "z_max": _z(2.45), "residual_m": 0.004, "confidence": 0.9}
    round_column = {"id": "C002", "shape": "circle", "center": [7.5 * U, 2.5 * U], "diameter_m": 0.36, "width_m": 0.36, "depth_m": 0.36,
                    "footprint": _pts([(7.5 + 0.18 * math.cos(a), 2.5 + 0.18 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)]),
                    "z_min": _z(0.0), "z_max": _z(2.6), "residual_m": 0.003, "confidence": 0.8}
    beam = {"id": "B001", "axis": _pts([(1.0, 0.15), (1.0, 4.85)]), "width_m": 0.24, "depth_m": 0.30, "footprint": _pts(_rect(0.88, 0.15, 1.12, 4.85)),
            "z_min": _z(2.15), "z_max": _z(2.45), "confidence": 0.7}
    stair = {"id": "ST001", "start": [6.0 * U, 1.0 * U], "direction_deg": 90.0, "width_m": 1.0, "riser_count": 14, "riser_height_m": 0.18,
             "tread_depth_m": 0.27, "from_z": _z(0.0), "to_z": _z(2.52), "footprint": _pts(_rect(5.5, 1.0, 6.5, 4.51)), "confidence": 0.8}
    return BuildingPlan(
        project_name="structure", level_id="level_1", level_name="EG", units_per_metre=U,
        frame={"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()},
        floor_z=_z(0.0), ceiling_z=_z(STOREY), walls=walls, rooms=rooms, openings=openings,
        columns=[column, round_column], beams=[beam], stairs=[stair],
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    scene = build_scene(make_plan())
    path = tmp_path_factory.mktemp("ifc") / "structure.ifc"
    result = write_ifc(scene, path)
    return scene, result, ifcopenshell.open(str(path)), path


def _bbox(entity) -> np.ndarray:
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    verts = np.array(ifcopenshell.geom.create_shape(settings, entity).geometry.verts).reshape(-1, 3)
    return np.array([verts.min(axis=0), verts.max(axis=0)])


def test_wall_heights_follow_the_plan(built) -> None:
    _, _, file, _ = built
    heights = {wall.Name: _bbox(wall)[1, 2] - _bbox(wall)[0, 2] for wall in file.by_type("IfcWall")}
    assert heights["W005"] == pytest.approx(1.10, abs=1e-3)
    assert heights["W001"] == pytest.approx(2.45, abs=1e-3)
    assert heights["W002"] == pytest.approx(3.0, abs=1e-3)


def test_wall_height_origin_is_measured(built) -> None:
    scene, _, _, _ = built
    origins = {element.name: element.param_origin["height"].value for element in scene.elements.values() if element.kind.value == "wall"}
    assert origins["W005"] == "measured"


def test_room_ceilings_at_their_measured_height_and_slope(built) -> None:
    _, _, file, _ = built
    coverings = {item.Name: _bbox(item) for item in file.by_type("IfcCovering")}
    flat = coverings["Ceiling Wohnen"]
    assert flat[0, 2] == pytest.approx(FLOOR + 2.45, abs=1e-3)
    sloped = coverings["Ceiling Atelier"]
    assert sloped[0, 2] == pytest.approx(FLOOR + 2.2, abs=0.01)
    assert sloped[1, 2] - sloped[0, 2] > 0.75  # spans the 0.8 m rise


def test_space_heights_and_properties(built) -> None:
    _, _, file, _ = built
    spaces = {space.LongName or space.Name: space for space in file.by_type("IfcSpace")}
    space = next(item for item in spaces.values() if "Wohnen" in str(item.Name) or "Wohnen" in str(item.LongName))
    props = element_util.get_psets(space)
    values = {key: value for pset in props.values() for key, value in pset.items()}
    assert values.get("MeasuredClearHeight") == pytest.approx(2.45, abs=1e-3)
    assert values.get("CeilingHeightSource") == "points"


def test_columns_beams_and_stair_classes(built) -> None:
    _, result, file, _ = built
    columns = file.by_type("IfcColumn")
    assert len(columns) == 2
    assert {item.PredefinedType for item in columns} == {"COLUMN"}
    assert all(element_util.get_type(item).is_a("IfcColumnType") for item in columns)
    rect = next(item for item in columns if item.Name == "Column C001")
    box = _bbox(rect)
    assert box[1, 2] - box[0, 2] == pytest.approx(2.45, abs=1e-3)
    assert box[1, 0] - box[0, 0] == pytest.approx(0.30, abs=1e-3)
    beams = file.by_type("IfcBeam")
    assert len(beams) == 1 and beams[0].PredefinedType == "BEAM"
    beam_box = _bbox(beams[0])
    assert beam_box[0, 2] == pytest.approx(FLOOR + 2.15, abs=1e-3) and beam_box[1, 2] == pytest.approx(FLOOR + 2.45, abs=1e-3)
    stairs = file.by_type("IfcStair")
    assert len(stairs) == 1
    flights = file.by_type("IfcStairFlight")
    assert len(flights) == 1
    assert len(file.by_type("IfcFurniture")) == 0  # structure never becomes furniture
    assert result.counts.get("IfcColumn") == 2


def test_structure_export_is_valid(built, tmp_path: Path) -> None:
    scene, result, _, path = built
    report = validate_ifc(path, scene=scene, export_result=result, reports_dir=tmp_path)
    assert report.express_findings == []
    assert report.ids_failures == []
    assert report.reimport_checked and report.reimport_mismatches == []


def test_plan_without_structure_keeps_storey_heights() -> None:
    plan = make_plan()
    for wall in plan.walls:
        wall.height_m, wall.height_source = None, "storey"
    for room in plan.rooms:
        room.ceiling_z, room.ceiling_plane, room.ceiling_source = None, None, "level"
    plan.columns, plan.beams, plan.stairs = [], [], []
    scene = build_scene(plan)
    walls = [element for element in scene.elements.values() if element.kind.value == "wall"]
    assert all(element.params.height == pytest.approx(STOREY) for element in walls)
    ceilings = [element for element in scene.elements.values() if element.kind.value == "ceiling"]
    assert all(element.params.elevation == pytest.approx(FLOOR + STOREY) and element.params.plane_normal is None for element in ceilings)


def test_old_plan_model_without_structure_keys_loads() -> None:
    data = make_plan().to_dict()
    for key in ("columns", "beams", "stairs"):
        data.pop(key)
    for wall in data["walls"]:
        wall.pop("height_source")
    for room in data["rooms"]:
        for key in ("ceiling_plane", "ceiling_source", "floor_source"):
            room.pop(key)
    plan = BuildingPlan.from_dict(data)
    assert plan.columns == [] and plan.walls[0].height_source == "storey" and plan.rooms[0].ceiling_source == "level"


def test_columns_and_beams_are_own_scene_kinds_not_furniture(built) -> None:
    scene, _, file, _ = built
    kinds = [element.kind.value for element in scene.elements.values()]
    assert kinds.count("column") == 2 and kinds.count("beam") == 1 and kinds.count("stair") == 1
    assert "furniture" not in kinds
    assert len(file.by_type("IfcCircleProfileDef")) == 1  # the round column keeps a true circle profile
    assert len(file.by_type("IfcRectangleProfileDef")) >= 2
    psets = element_util.get_psets(next(item for item in file.by_type("IfcColumn") if item.Name == "Column C001"))
    assert psets["Qto_ColumnBaseQuantities"]["Length"] == pytest.approx(2.45, abs=1e-3)
    assert "Raumlot_Quality" in psets


def test_qa_lists_structure_separately_from_furniture(built, tmp_path: Path) -> None:
    import time

    from samcloud.bim.export import _qa_payload

    scene, result, _, path = built
    report = validate_ifc(path, scene=scene, export_result=result, reports_dir=tmp_path)
    qa = _qa_payload(make_plan(), scene, result, report, time.perf_counter(), path)
    assert "0 furniture items, 2 columns, 1 beams, 1 stairs" in qa["summary"]
    assert qa["consistency"]["columns"] == {"plan": 2, "ifc": 2}
    assert qa["consistency"]["furniture"]["ifc"] == 0


def test_viewer_categories_put_structure_in_its_own_group() -> None:
    from samcloud.bim import ui_data

    assert ui_data.category_of("IfcColumn") == ui_data.category_of("IfcBeam") == ui_data.category_of("IfcStairFlight") == "structure"
    assert "structure" in ui_data.CATEGORY_ORDER


def test_gable_ceiling_is_two_sloped_coverings(tmp_path: Path) -> None:
    plan = make_plan()
    slope = math.tan(math.radians(20.0))
    ridge_y = 2.5
    # local z = a x + b y + c ; plane rising towards the ridge from both sides, 2.2 m at the eaves (y = 0.15 / 4.85)
    rising = [0.0, slope, _z(2.2) - slope * 0.15 * U]
    falling = [0.0, -slope, _z(2.2) + slope * 4.85 * U]
    room = plan.rooms[1]
    room.ceiling_source, room.ceiling_planes, room.ceiling_envelope, room.ceiling_plane = "planes", [rising, falling], "min", rising
    room.ceiling_z = _z(2.2 + slope * (ridge_y - 0.15))
    plan.beams, plan.stairs = [], []
    scene = build_scene(plan)
    path = tmp_path / "gable.ifc"
    result = write_ifc(scene, path)
    file = ifcopenshell.open(str(path))
    pieces = [item for item in file.by_type("IfcCovering") if str(item.Name).startswith("Ceiling Atelier")]
    assert len(pieces) == 2
    boxes = [_bbox(item) for item in pieces]
    top = FLOOR + 2.2 + slope * (ridge_y - 0.15)
    for box in boxes:
        assert box[0, 2] == pytest.approx(FLOOR + 2.2, abs=0.02)  # eaves
        assert box[1, 2] == pytest.approx(top + 0.10 / math.cos(math.radians(20.0)), abs=0.03)  # ridge (+ ceiling thickness)
    ys = sorted(((box[0, 1] + box[1, 1]) / 2.0) for box in boxes)
    assert ys[0] < ridge_y < ys[1]
    report = validate_ifc(path, scene=scene, export_result=result, reports_dir=tmp_path)
    assert report.express_findings == [] and report.ids_failures == [] and report.reimport_mismatches == []
