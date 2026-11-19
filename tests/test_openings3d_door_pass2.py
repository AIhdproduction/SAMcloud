"""Second clean-plan pass with classified doors: leaf symbol, passage, dropped candidate (closed wall), new hosted door, determinism."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

import test_ortho_plan as scene_tests
from samcloud.bim.openings3d import pipeline
from samcloud.bim.openings3d.config import load_openings_config
from samcloud.bim.openings3d.model import OpeningsResult, RefinedOpening, save_openings_3d
from samcloud.bim.openings3d.walls import wall_lines
from samcloud.bim.plan_model import BuildingPlan, load_plan_model, plan_model_path
from samcloud.ortho.plan import clean_plan_cache_dir, generate_clean_plan

UNITS = (1.0, 0.8124)


@dataclass
class Run:
    plan1: BuildingPlan
    plan2: BuildingPlan
    layers: dict
    upm: float
    plan_bytes: bytes
    again_bytes: bytes


def _refined(index: int, kind: str, centre, width_m: float, angle: float, thickness_m: float, host: str | None, **extra) -> RefinedOpening:
    door_like = kind in ("door", "passage")
    return RefinedOpening(
        id=f"C{index:03d}", kind=kind, center=list(centre), width_m=width_m, axis_angle=angle, wall_thickness_m=thickness_m, host_wall_id=host, sill_m=0.0 if door_like else 1.05,
        head_m=2.1 if kind == "door" else (2.45 if kind == "passage" else 2.0), height_source="points", evidence="synthetic classification", confidence=0.8, **extra,
    )


def _run(upm: float, mode: str, tmp_root: Path) -> Run:
    project = tmp_root / f"door_pass2_{mode}_{upm}_{uuid.uuid4().hex[:6]}" / "indoor"
    level, _room = scene_tests._build_indoor_wall_face_scene(project, units_per_metre=upm)
    generate_clean_plan(project, level=level, mode="indoor")
    directory = clean_plan_cache_dir(project, level)
    plan1 = load_plan_model(directory)
    refined: list[RefinedOpening] = []
    closed: list[dict] = []
    for index, opening in enumerate(plan1.openings, start=1):
        if opening.kind == "door":
            if mode == "none":  # classified as no opening: dropped, its span is handed to the wall stage
                closed.append({"center": list(opening.center), "width_m": opening.width_m, "axis_angle": opening.axis_angle, "wall_thickness_m": opening.wall_thickness_m, "host_wall_id": opening.host_wall_id, "line": "", "reason": "test"})
                continue
            kind = "passage" if mode == "passage" else "door"
            extra = dict(hinge_side="right", swing_side=1, leaf_angle_deg=80.0, leaf_detected=True) if mode in ("leaf", "entrance") else {}
            refined.append(_refined(index, kind, opening.center, opening.width_m, opening.axis_angle, opening.wall_thickness_m, opening.host_wall_id, **extra))
        else:
            refined.append(_refined(index, opening.kind, opening.center, opening.width_m, opening.axis_angle, opening.wall_thickness_m, opening.host_wall_id, mullion_count=0))
    if mode == "entrance":  # a door the 2D plan did not know: the wall scan found its frame on the left wall
        refined.append(_refined(len(refined) + 1, "door", (0.2 * upm, 2.2 * upm), 0.9, math.pi / 2, 0.25, None))
    config = load_openings_config()
    inputs = pipeline.current_inputs(plan1, wall_lines(plan1, config), project, level.id, config, pipeline.resolve_point_source(project, level.id))
    save_openings_3d(OpeningsResult(openings=refined, closed=closed, inputs=inputs, mode="full"), directory)
    second = generate_clean_plan(project, level=level, mode="indoor")
    plan2 = load_plan_model(directory)
    first_bytes = plan_model_path(directory).read_bytes()
    generate_clean_plan(project, level=level, mode="indoor")  # same inputs again
    return Run(plan1, plan2, second.layers, upm, first_bytes, plan_model_path(directory).read_bytes())


@pytest.fixture(scope="module")
def tmp_root(request: pytest.FixtureRequest) -> Path:
    root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    return Path(root) if root else request.getfixturevalue("tmp_path_factory").mktemp("door_pass2")


@pytest.fixture(scope="module", params=UNITS)
def runs(request: pytest.FixtureRequest, tmp_root: Path) -> dict[str, Run]:
    return {mode: _run(float(request.param), mode, tmp_root) for mode in ("leaf", "passage", "none", "entrance")}


def _inserts(layers: dict) -> list[dict]:
    return [entity for entity in layers["A-DOOR"] if entity.get("type") == "insert"]


def _jambs(layers: dict) -> list[dict]:
    return [entity for entity in layers["A-DOOR"] if entity.get("_jamb")]


def test_door_with_measured_leaf_keeps_hinge_swing_and_is_drawn_on_them(runs: dict[str, Run]) -> None:
    run = runs["leaf"]
    door = next(o for o in run.plan2.openings if o.kind == "door" and abs(o.center[1] - 0.2 * run.upm) < 0.1 * run.upm)
    assert (door.hinge_side, door.swing_side, door.leaf_detected, door.leaf_angle_deg) == ("right", 1, True, 80.0)
    inserts = _inserts(run.layers)
    assert len(inserts) == 1
    insert = inserts[0]
    axis = np.array([math.cos(door.axis_angle), math.sin(door.axis_angle)])
    normal = np.array([-axis[1], axis[0]])
    centre = np.asarray(door.center) / 1.0
    rotation = math.radians(float(insert["rotation"]))
    open_leaf = np.array([-math.sin(rotation), math.cos(rotation)]) * float(insert["yscale"])
    along = float((np.asarray(insert["point"][:2]) - centre) @ axis)
    assert along == pytest.approx(door.width_m * run.upm / 2.0, abs=0.02 * run.upm)  # hinge at the +axis end (right)
    assert float(open_leaf @ normal) > 0  # swings to the +normal side (swing_side +1)
    assert len(_jambs(run.layers)) == 2  # the opening ends stay marked next to the symbol


def test_passage_stays_an_opening_without_leaf_or_arc(runs: dict[str, Run]) -> None:
    run = runs["passage"]
    passages = [o for o in run.plan2.openings if o.kind == "passage"]
    assert len(passages) == 1 and passages[0].head_m == pytest.approx(2.45) and not passages[0].leaf_detected and passages[0].hinge_side is None
    assert not _inserts(run.layers) and len(_jambs(run.layers)) == 2
    assert not [o for o in run.plan2.openings if o.kind == "door"]


def test_dropped_candidate_closes_the_wall(runs: dict[str, Run]) -> None:
    run = runs["none"]
    assert [o.kind for o in run.plan2.openings] == ["window"]
    assert not run.layers["A-DOOR"]
    union = unary_union([Polygon(body["exterior"], body.get("holes") or []) for body in run.plan2.wall_bodies])
    span = [Point(x * run.upm, 0.05 * run.upm) for x in np.linspace(2.4, 3.2, 9)]  # inside the wall body (face at y = 0.2 m, body towards -y); where the 2D plan had the door (its gap in the wall)
    assert all(union.contains(point) for point in span), "the former door gap is not closed by wall"
    thickness = [union.intersection(LineString([(x * run.upm, 0.2 * run.upm), (x * run.upm, -0.5 * run.upm)])).length for x in (1.0, 2.8, 4.2)]
    assert max(thickness) - min(thickness) < 0.02 * run.upm  # the closed span carries the host wall's thickness: one wall, nothing doubled
    outline = unary_union([Polygon(body["exterior"]) for body in run.plan2.wall_bodies]).envelope.buffer(0.3 * run.upm)
    assert all(outline.contains(Point(wall.centreline[0])) and outline.contains(Point(wall.centreline[1])) for wall in run.plan2.walls)  # no stray wall outside the building
    leaf, passage = runs["leaf"], runs["passage"]
    leaf_union = unary_union([Polygon(body["exterior"], body.get("holes") or []) for body in leaf.plan2.wall_bodies])
    assert not any(leaf_union.contains(point) for point in span)  # with the door the same span is open: the closure is the classification's doing
    assert passage.plan2.openings[0].kind == "passage"


def test_new_door_from_the_scan_is_added_and_hosted_on_its_wall(runs: dict[str, Run]) -> None:
    run = runs["entrance"]
    left_wall = [o for o in run.plan2.openings if o.kind == "door" and abs(o.center[0] - 0.2 * run.upm) < 0.1 * run.upm]
    assert len(left_wall) == 1, [o.to_dict() for o in run.plan2.openings]
    door = left_wall[0]
    assert door.host_wall_id is not None and door.width_m == pytest.approx(0.9)
    assert any(wall.id == door.host_wall_id for wall in run.plan2.walls)
    assert len(run.plan2.openings) == len(run.plan1.openings) + 1


def test_second_pass_is_deterministic(runs: dict[str, Run]) -> None:
    for mode, run in runs.items():
        assert hashlib.sha256(run.plan_bytes).hexdigest() == hashlib.sha256(run.again_bytes).hexdigest(), mode


# ---------------------------------------------------------------- wall stage: closing a dropped candidate's span


def _wall_with_gap(upm: float, rooms: bool, spans: list[dict]):
    import test_ortho_double_walls as walls

    faces, room_polygons, free, valid, grid = walls._two_rooms(upm)
    split = []
    for face in faces:  # the bottom face of room A (y = 0, x 0..4 m) with a 0.9 m gap at 1.5..2.4 m
        (x0, y0), (x1, y1) = face["points"]
        if abs(y0) < 1e-9 and abs(y1) < 1e-9 and abs(x0) < 1e-9 and abs(x1 - 4.0 * upm) < 1e-9:
            split += [walls._face(0.0, 0.0, 1.5 * upm, 0.0), walls._face(2.4 * upm, 0.0, 4.0 * upm, 0.0)]
        else:
            split.append(face)
    from samcloud.ortho.vector.double_walls import build_double_walls

    result = build_double_walls(split, [], room_polygons if rooms else [], free, valid, grid, upm, walls.SETTINGS, close_spans=spans)
    body = unary_union(result.polygons)
    return result, [body.contains(Point(x * upm, -0.05 * upm)) for x in np.linspace(1.6, 2.3, 8)]


@pytest.mark.parametrize("upm", UNITS)
def test_wall_stage_closes_the_span_of_a_dropped_candidate(upm: float) -> None:
    span = {"center": [1.95 * upm, 0.0], "axis_angle": 0.0, "width": 0.9 * upm}
    _result, open_gap = _wall_with_gap(upm, rooms=False, spans=[])
    assert not any(open_gap)  # the measured faces leave the gap open
    result, closed = _wall_with_gap(upm, rooms=False, spans=[span])
    assert all(closed) and result.stats["double_wall_spans_closed"] == 1
    assert any(piece["evidence"] == "door-fill" for piece in result.pieces)


@pytest.mark.parametrize("upm", UNITS)
def test_wall_stage_does_not_double_a_span_the_walls_already_cover(upm: float) -> None:
    span = {"center": [1.95 * upm, 0.0], "axis_angle": 0.0, "width": 0.9 * upm}
    result, covered = _wall_with_gap(upm, rooms=True, spans=[span])  # the room edge wall already closes this gap
    assert all(covered) and "double_wall_spans_closed" not in result.stats
    far = {"center": [20.0 * upm, 20.0 * upm], "axis_angle": 0.0, "width": 0.9 * upm}
    result, _ = _wall_with_gap(upm, rooms=False, spans=[far])  # no wall near the span: nothing to continue
    assert "double_wall_spans_closed" not in result.stats


# ---------------------------------------------------------------- host wall beside a door, closure keeps one wall


@pytest.mark.parametrize("upm", UNITS)
def test_host_side_spans_continue_the_original_wall_beside_a_door(upm: float) -> None:
    from openings3d_doors_synth import make_wall_plan
    from samcloud.bim.openings3d.apply import host_side_spans

    config = load_openings_config()
    plan = make_wall_plan(upm)  # one 4.4 m wall piece, 0.25 m thick, along +x at y = 0
    line = wall_lines(plan, config)[0]
    door = _refined(1, "door", (2.45 * upm, 0.0), 0.9, 0.0, 0.25, "W001", line=line.id, u0=2.0, u1=2.9)
    spans = host_side_spans([door], plan, config)
    assert len(spans) == 2
    left, right = sorted(spans, key=lambda item: item["center"][0])
    assert left["width"] == pytest.approx(1.5 * upm, abs=1e-3) and left["center"][0] == pytest.approx(1.25 * upm, abs=1e-3)  # reach 1.5 m of the original piece beside the door
    assert right["width"] == pytest.approx(1.5 * upm, abs=1e-3) and right["center"][0] == pytest.approx(3.65 * upm, abs=1e-3)
    assert all(item["thickness"] == pytest.approx(0.25 * upm, abs=1e-3) and item["axis_angle"] == pytest.approx(0.0) and abs(item["center"][1]) < 1e-6 for item in spans)
    window = _refined(2, "window", (1.0 * upm, 0.0), 0.8, 0.0, 0.25, "W001", line=line.id, u0=0.6, u1=1.4)
    assert host_side_spans([window], plan, config) == []  # only doors and passages


@pytest.mark.parametrize("upm", UNITS)
def test_door_in_an_unmeasured_wall_keeps_the_host_wall_on_both_sides(upm: float) -> None:
    import test_ortho_double_walls as walls
    from samcloud.ortho.vector.double_walls import build_double_walls
    from samcloud.ortho.vector.indoor import Opening

    faces, rooms, free, valid, grid = walls._two_rooms(upm)
    kept = []
    for face in faces:  # room A's bottom wall (y = 0, x 0..4 m) measured only at both ends: the stretch around the door is unmeasured
        (x0, y0), (x1, y1) = face["points"]
        if abs(y0) < 1e-9 and abs(y1) < 1e-9 and abs(x0) < 1e-9 and abs(x1 - 4.0 * upm) < 1e-9:
            kept += [walls._face(0.0, 0.0, 0.8 * upm, 0.0), walls._face(3.2 * upm, 0.0, 4.0 * upm, 0.0)]
        else:
            kept.append(face)
    door = Opening("DOOR", (2.0 * upm, 0.0), 0.9 * upm, 0.0, 0.3 * upm)
    host = {"center": [1.175 * upm, 0.0], "axis_angle": 0.0, "width": 0.75 * upm, "thickness": 0.3 * upm}  # the original wall beside the door, 0.8 .. 1.55 m
    host_right = {"center": [2.825 * upm, 0.0], "axis_angle": 0.0, "width": 0.75 * upm, "thickness": 0.3 * upm}  # 2.45 .. 3.2 m
    probes = lambda xs: [(x * upm, -0.05 * upm) for x in xs]  # noqa: E731

    def covered(result, xs):
        body = unary_union(result.polygons)
        return [body.contains(Point(point)) for point in probes(xs)]

    without = build_double_walls(kept, [door], [], free, valid, grid, upm, walls.SETTINGS)
    assert not any(covered(without, [1.0, 1.4, 2.6, 3.0]))  # nothing continues the host wall by itself
    with_fill = build_double_walls(kept, [door], [], free, valid, grid, upm, walls.SETTINGS, close_spans=[host, host_right])
    assert all(covered(with_fill, [1.0, 1.4, 2.6, 3.0]))  # the host wall stands on both sides of the door ...
    assert not any(covered(with_fill, [1.7, 2.0, 2.3]))  # ... and the door span itself stays open
    bodies = unary_union(with_fill.polygons)
    thickness = [bodies.intersection(LineString([(x * upm, 0.3 * upm), (x * upm, -0.6 * upm)])).length for x in (0.4, 1.0, 3.0, 3.6)]
    assert max(thickness) - min(thickness) < 0.02 * upm  # one wall of the measured thickness, no thicker doubled piece


@pytest.mark.parametrize("upm", UNITS)
def test_host_side_spans_leave_other_openings_on_the_line_open(upm: float) -> None:
    from openings3d_doors_synth import make_wall_plan
    from samcloud.bim.openings3d.apply import host_side_spans

    config = load_openings_config()
    plan = make_wall_plan(upm)
    line = wall_lines(plan, config)[0].id
    door = _refined(1, "door", (2.45 * upm, 0.0), 0.9, 0.0, 0.25, "W001", line=line, u0=2.0, u1=2.9)
    window = _refined(2, "window", (1.2 * upm, 0.0), 0.8, 0.0, 0.25, "W001", line=line, u0=0.8, u1=1.6)  # left of the door
    passage = _refined(3, "passage", (3.5 * upm, 0.0), 0.6, 0.0, 0.25, "W001", line=line, u0=3.2, u1=3.8)  # right of it
    spans = host_side_spans([door, window, passage], plan, config)
    covered = []
    for span in spans:  # span intervals along the line, metres
        lo = span["center"][0] / upm - span["width"] / upm / 2.0
        covered.append((round(lo, 2), round(lo + span["width"] / upm, 2)))
    for u0, u1 in covered:  # no filler over the window (0.8..1.6) or the passage (3.2..3.8)
        assert u1 <= 0.8 + 0.06 or u0 >= 1.6 - 0.06 or (u0 >= 1.6 - 0.06 and u1 <= 3.2 + 0.06) or u0 >= 3.8 - 0.06 or u1 <= 3.2 + 0.06, (u0, u1)
        assert not (u0 < 1.5 and u1 > 0.9) and not (u0 < 3.7 and u1 > 3.3)
    assert any(abs(u0 - 1.6) < 0.1 and abs(u1 - 2.0) < 0.1 for u0, u1 in covered)  # the stretch between window and door stays a host stretch
