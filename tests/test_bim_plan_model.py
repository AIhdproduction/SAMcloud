"""WP2: structured BuildingPlan sidecar next to the clean-plan result."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

import test_ortho_plan as scene_tests
from samcloud.bim.plan_model import (
    BuildingPlan, PlanFurniture, PlanOpening, PlanRoom, PlanWall, build_plan_model, load_bim_export_config,
    load_plan_model, plan_model_path, save_plan_model,
)
from samcloud.dxf.las_source import Frame
from samcloud.dxf.levels import Level
from samcloud.dxf.pipeline import load_cached_result
from samcloud.ortho import plan as plan_module
from samcloud.ortho.plan import clean_plan_cache_dir, generate_clean_plan


@pytest.fixture(scope="module")
def work_dir(request: pytest.FixtureRequest) -> Path:
    root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if root:
        path = Path(root) / f"bim_plan_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("bim_plan")


@pytest.fixture(scope="module")
def generated(work_dir: Path):
    """Run the real indoor clean plan once; capture the DoubleWallResult it builds."""
    captured: list = []
    original = plan_module.build_double_walls

    def spy(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append(value)
        return value

    project_dir = work_dir / "indoor"
    level, _room = scene_tests._build_indoor_wall_face_scene(project_dir, units_per_metre=0.8)
    plan_module.build_double_walls = spy
    try:
        result = generate_clean_plan(project_dir, level=level, mode="indoor")
    finally:
        plan_module.build_double_walls = original
    assert len(captured) == 1
    return project_dir, level, result, captured[0], load_plan_model(clean_plan_cache_dir(project_dir, level))


def test_round_trip_to_dict_and_file(tmp_path: Path) -> None:
    plan = BuildingPlan(
        "p", "level_1", "1", 0.8, {"origin": [1.0, 2.0, 3.0], "scale_factor": 0.8, "rotation": np.eye(3).tolist()}, 0.0, 2.0,
        walls=[PlanWall("W001", [[0, 0], [4, 0]], 0.2, "exterior", "measured", [[0, 0], [4, 0]], [[0, 0.2], [4, 0.2]], body_id=0)],
        rooms=[PlanRoom("R001", "Bad", {"exterior": [[0, 0], [1, 0], [1, 1]], "holes": []}, 0.5, [0.7, 0.3])],
        openings=[PlanOpening("O001", "door", [2.0, 0.0], 0.9, 0.0, 0.2, "W001", "left", 1, 90.0, True, evidence="gap")],
        furniture=[PlanFurniture("F001", 7, "Bed", "bed", "BED_DOUBLE", "A-FURN", [[0, 0], [1, 0], [1, 2]], 0.0, 0.5, "instance_bbox", 0.9)],
        wall_bodies=[{"exterior": [[0, 0], [4, 0], [4, 0.2]], "holes": []}], source={"result_fingerprint": "abc"},
    )
    assert BuildingPlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan
    path = save_plan_model(plan, tmp_path / "ortho")
    assert path == plan_model_path(tmp_path / "ortho") and path.is_file()
    assert not list((tmp_path / "ortho").glob(".*.tmp"))
    assert load_plan_model(tmp_path / "ortho") == plan
    bad = plan.to_dict() | {"schema_version": 99}
    with pytest.raises(ValueError):
        BuildingPlan.from_dict(bad)


def test_config_defaults_loaded() -> None:
    config = load_bim_export_config()
    assert config["defaults"]["window_sill_m"] == 0.9 and config["defaults"]["upper_cabinet_z_m"] == [1.45, 2.15]
    assert config["furniture_height_by_kind_m"]["BED_DOUBLE"] == 0.5 and config["furniture_height_by_label_m"]["wardrobe"] == 2.1


def test_walls_match_double_wall_pieces(generated) -> None:
    _project, _level, result, double, plan = generated
    assert len(double.pieces) > 0
    assert len(plan.walls) == len(double.pieces)
    for wall, piece in zip(plan.walls, double.pieces, strict=True):
        assert wall.thickness_m == pytest.approx(piece["thickness"])
        assert wall.kind == piece["kind"]
        mid = (np.asarray(piece["face"]) + np.asarray(piece["face_b"])) / 2.0
        assert np.allclose(wall.centreline, mid)
    assert len(plan.wall_bodies) == len(double.closed_polygons)
    assert all(wall.body_id is not None for wall in plan.walls)
    assert plan.source["result_fingerprint"] == result.settings_fingerprint
    assert plan.frame["scale_factor"] == result.frame.scale_factor and np.allclose(plan.frame["rotation"], result.frame.rotation)


def test_openings_match_dxf_symbols(generated) -> None:
    _project, _level, result, _double, plan = generated
    upm = plan.units_per_metre
    symbols = [entity for layer in ("A-DOOR", "A-WIND") for entity in result.layers[layer] if entity.get("_opening_center") is not None]
    assert symbols
    inserts = [entity for entity in symbols if entity.get("type") == "insert"]
    assert inserts
    for entity in inserts:
        matches = [item for item in plan.openings if np.linalg.norm(np.asarray(item.center) - np.asarray(entity["_opening_center"][:2])) <= 1e-6]
        assert len(matches) == 1
        assert matches[0].width_m * upm == pytest.approx(float(entity["xscale"]), abs=1e-9)
        assert matches[0].kind == ("window" if entity["name"] == "WINDOW" else "door")
    distinct = {tuple(np.round(entity["_opening_center"][:2], 9)) for entity in symbols}
    assert len(plan.openings) == len(distinct)
    assert {item.kind for item in plan.openings} >= {"door", "window"}
    window = next(item for item in plan.openings if item.kind == "window")
    assert (window.sill_m, window.head_m) == (0.9, 2.1)
    host_ids = {wall.id for wall in plan.walls}
    hosted = [item for item in plan.openings if item.host_wall_id is not None]
    assert hosted and all(item.host_wall_id in host_ids for item in hosted)


def test_rooms_match_area_polylines(generated) -> None:
    _project, _level, result, _double, plan = generated
    areas = sorted(Polygon(entity["points"], entity.get("holes") or None).area / plan.units_per_metre**2 for entity in result.layers["A-AREA"])
    assert areas and len(plan.rooms) == len(areas)
    for expected, actual in zip(areas, sorted(room.area_m2 for room in plan.rooms), strict=True):
        assert actual == pytest.approx(expected, rel=0.01)
    assert all(room.name for room in plan.rooms)


def test_furniture_footprints_exist_for_dxf_inserts(generated) -> None:
    _project, level, result, _double, plan = generated
    dxf_count = sum(
        1 for layer in ("A-FURN", "A-EINB", "A-OBEN", "P-SANI") for entity in result.layers.get(layer, [])
        if entity.get("type") == "insert" or (entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points", [])) >= 3)
    )
    assert dxf_count > 0 and len(plan.furniture) == dxf_count
    assert plan.floor_z == level.floor_z and plan.ceiling_z == level.ceiling_z
    assert all(item.default_height_m is not None or item.default_z_m is not None for item in plan.furniture)


def test_instance_z_range_uses_frame_transform(tmp_path: Path) -> None:
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    frame = Frame(np.asarray([10.0, 20.0, 5.0]), 2.0, rotation)
    instances = tmp_path / "instances.json"
    instances.write_text(json.dumps({"schema_version": 1, "instances": [
        {"id": 4, "bbox_min": [11.0, 21.0, 5.5], "bbox_max": [12.0, 22.0, 6.5], "confidence": 0.8},
    ]}), encoding="utf-8")
    layers = {
        "A-FURN": [
            {"type": "insert", "name": "BED_DOUBLE", "point": [1.0, 1.0], "rotation": 0.0, "xscale": 1.0, "yscale": 1.0,
             "attributes": {"LABEL": "Bed", "INSTANCE": "4"}, "_catalogue": {"kind": "BED_DOUBLE", "layer": "A-FURN", "wall_snapped": False}},
            {"type": "polyline", "points": [[0, 0], [1, 0], [1, 1], [0, 1]], "closed": True, "holes": [], "_label": "Table", "_semantic_label": "table", "_instance_id": 4},
            {"type": "polyline", "points": [[3, 0], [4, 0], [4, 1], [3, 1]], "closed": True, "holes": [], "_label": "Chair", "_semantic_label": "chair"},
        ],
        "A-OBEN": [{"type": "polyline", "points": [[0, 3], [1, 3], [1, 4]], "closed": True, "holes": [], "_upper_cabinet": True}],
    }
    level = Level("level_1", "1", 1.0, 6.0, 0, 1.0, False)
    plan = build_plan_model(project_name="x", level=level, frame=frame, units_per_metre=1.0, layers=layers, openings=[], rooms=[], instances_path=instances)
    by_label = {item.label: item for item in plan.furniture}
    expected = (np.asarray([5.5, 6.5]) - 5.0) * 2.0
    assert by_label["Bed"].z_source == "instance_bbox" and by_label["Bed"].instance_id == 4
    assert (by_label["Bed"].z_min, by_label["Bed"].z_max) == pytest.approx(tuple(expected))
    assert (by_label["Table"].z_min, by_label["Table"].z_max) == pytest.approx(tuple(expected))
    assert by_label["Bed"].confidence == pytest.approx(0.8)
    assert by_label["Chair"].z_min is None and by_label["Chair"].z_source == "default" and by_label["Chair"].default_height_m == 0.9
    assert by_label["Bed"].default_height_m == 0.5
    upper = next(item for item in plan.furniture if item.layer == "A-OBEN")
    assert upper.default_z_m == [1.45, 2.15] and upper.default_height_m is None


def test_opening_host_wall_found_for_gap_between_pieces() -> None:
    pieces = [
        {"face": [[0, 0], [1.0, 0]], "face_b": [[0, 0.2], [1.0, 0.2]], "thickness": 0.2, "kind": "interior", "evidence": "measured"},
        {"face": [[2.0, 0], [4.0, 0]], "face_b": [[2.0, 0.2], [4.0, 0.2]], "thickness": 0.2, "kind": "exterior", "evidence": "measured"},
    ]

    class _Opening:
        kind, center, width, axis_angle, wall_thickness = "DOOR", (1.5, 0.1), 1.0, 0.0, 0.2
        evidence, leaf_angle_deg, hinge_side, swing_side, leaf_detected = "gap", None, None, None, None

    class _Perpendicular(_Opening):
        axis_angle = np.pi / 2

    level = Level("level_1", "1", 0.0, 2.5, 0, 1.0, False)
    frame = Frame(np.zeros(3), 1.0, np.eye(3))
    plan = build_plan_model(project_name="x", level=level, frame=frame, units_per_metre=1.0, layers={}, openings=[_Opening(), _Perpendicular()], rooms=[], wall_pieces=pieces)
    assert plan.openings[0].host_wall_id in {"W001", "W002"}
    assert plan.openings[1].host_wall_id is None  # axis not parallel to any wall


def test_clean_plan_stale_when_plan_model_missing(generated) -> None:
    from samcloud.app.dxf_controller import DxfPageController

    project_dir, level, result, _double, _plan = generated
    controller = DxfPageController(project_dir)
    controller.state.levels = [{"id": level.id, "name": level.name, "floor_z": level.floor_z, "ceiling_z": level.ceiling_z}]
    cached = clean_plan_cache_dir(project_dir, level)
    assert load_cached_result(cached).settings_fingerprint == result.settings_fingerprint
    assert controller.plan_status(level.id, "ortho") == "ok"
    sidecar = plan_model_path(cached)
    backup = sidecar.with_name("plan_model.json.bak")
    sidecar.rename(backup)
    try:
        assert controller.plan_status(level.id, "ortho") == "stale"
    finally:
        backup.rename(sidecar)
    assert controller.plan_status(level.id, "ortho") == "ok"


def test_plan_model_failure_does_not_break_dxf_result(work_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("plan model exploded")

    monkeypatch.setattr(plan_module, "build_plan_model", boom)
    project_dir = work_dir / "indoor_fail"
    level, _room = scene_tests._build_indoor_wall_face_scene(project_dir, units_per_metre=0.8)
    with caplog.at_level("WARNING", logger=plan_module.__name__):
        result = generate_clean_plan(project_dir, level=level, mode="indoor")
    cache_dir = clean_plan_cache_dir(project_dir, level)
    assert (cache_dir / "result.json").is_file()
    assert not plan_model_path(cache_dir).exists()
    assert load_cached_result(cache_dir).settings_fingerprint == result.settings_fingerprint
    assert any("plan_model.json sidecar could not be written" in record.getMessage() and record.exc_info for record in caplog.records)


def test_single_line_walls_fallback_when_no_wall_pieces() -> None:
    layers = {"A-WALL": [
        {"type": "line", "start": [0.0, 0.0], "end": [4.0, 0.0]},
        {"type": "polyline", "points": [[4.0, 0.0], [4.0, 3.0], [1.0, 3.0]], "closed": False},
        {"type": "polyline", "points": [[0, 0], [1, 0], [1, 1]], "closed": True},
        {"type": "circle", "center": [0.0, 0.0], "radius": 1.0},
    ]}
    frame = Frame(np.zeros(3), 1.0, np.eye(3))
    level = Level("level_1", "1", 0.0, 2.5, 0, 1.0, False)
    plan = build_plan_model(project_name="x", level=level, frame=frame, units_per_metre=1.0, layers=layers, openings=[], rooms=[], wall_pieces=None)
    assert [wall.id for wall in plan.walls] == ["W001", "W002", "W003"]
    assert all(wall.thickness_m == 0.0 for wall in plan.walls)
    assert [wall.centreline for wall in plan.walls] == [[[0.0, 0.0], [4.0, 0.0]], [[4.0, 0.0], [4.0, 3.0]], [[4.0, 3.0], [1.0, 3.0]]]
    assert plan.wall_bodies == []
