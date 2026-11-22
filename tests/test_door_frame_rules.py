"""Door rules of the 3D refinement: the FRAME decides where a door is, a leaf is never an opening or a wall.

1. a seed door is measured on the wall line running in its own direction and a line that ends at the gap is continued over it
2. a candidate without casing that lies along the open leaf of a door on another wall is that leaf, no opening
3. a candidate with frame evidence is never lost to the duplicate filter (restored, at worst without a leaf)
plus the wall stage: a wall run along a measured leaf is dropped, free ends across a small step are joined, steps are straightened.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openings3d_doors_synth import HEIGHT, Cloud, collect
from samcloud.bim.openings3d.classify import classify_openings, drop_leaf_shadows, restore_framed_duplicates
from samcloud.bim.openings3d.config import load_openings_config
from samcloud.bim.openings3d.detect import _line_of_point
from samcloud.bim.openings3d.model import RefinedOpening
from samcloud.bim.openings3d.walls import wall_lines
from samcloud.bim.plan_model import BuildingPlan, PlanOpening, PlanWall

UNITS = pytest.mark.parametrize("upm", [1.0, 0.9891])


def _plan(walls, openings, upm):
    frame = {"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}
    return BuildingPlan("synthetic", "level_1", "Level 1", upm, frame, 0.0, HEIGHT * upm, walls=walls, openings=openings)


def _wall(identifier, p0, p1, upm, thickness=0.2):
    return PlanWall(id=identifier, centreline=[[p0[0] * upm, p0[1] * upm], [p1[0] * upm, p1[1] * upm]], thickness_m=thickness, kind="interior")


def _seed_opening(upm, center, width, axis):
    return PlanOpening(id="O001", kind="door", center=[center[0] * upm, center[1] * upm], width_m=width, axis_angle=axis, wall_thickness_m=0.2)


@UNITS
def test_wall_line_continues_over_a_door_gap_at_the_end_of_a_wall(upm):
    """A vertical wall ends at y = 3.0 (the jamb); the door gap runs on down to the crossing horizontal wall at y = 2.2."""
    config = load_openings_config()
    walls = [_wall("W001", (4.6, 3.0), (4.6, 6.0), upm), _wall("W002", (0.0, 2.2), (4.6, 2.2), upm), _wall("W003", (4.6, 2.2), (9.0, 2.2), upm)]
    seed = _seed_opening(upm, (4.65, 2.6), 0.85, math.pi / 2)
    plain = {line.id: line for line in wall_lines(_plan(walls, [], upm), config)}
    extended = {line.id: line for line in wall_lines(_plan(walls, [seed], upm), config)}
    vertical_plain = next(line for line in plain.values() if abs(line.d[1]) > 0.9)
    vertical = next(line for line in extended.values() if abs(line.d[1]) > 0.9)
    assert vertical.length_m > vertical_plain.length_m + 0.3
    lowest = min(vertical.a[1], vertical.a[1] + vertical.d[1] * vertical.length) / upm
    assert lowest <= 2.6 - 0.85 / 2 + 0.02  # the line reaches the far edge of the gap
    assert vertical.pad_start_m > 0.0 or vertical.pad_end_m > 0.0  # the continuation is an artificial end
    # the horizontal walls are untouched
    assert {line.length_m for line in extended.values() if abs(line.d[0]) > 0.9} == {line.length_m for line in plain.values() if abs(line.d[0]) > 0.9}


@UNITS
def test_a_seed_is_measured_on_the_line_of_its_own_direction(upm):
    """The seed (vertical gap) lies 0.09 m from a horizontal wall: it must not be hosted on that wall, whose leaf lies along it."""
    config = load_openings_config()
    walls = [_wall("W001", (4.6, 3.0), (4.6, 6.0), upm), _wall("W002", (0.0, 2.5), (9.0, 2.5), upm)]
    seed = _seed_opening(upm, (4.7, 2.59), 0.85, math.pi / 2)
    plan = _plan(walls, [seed], upm)
    evidence = {line.id: SimpleNamespace(line=line) for line in wall_lines(plan, config)}
    found = _line_of_point(evidence, list(seed.center), plan, config, seed.axis_angle)
    assert found is not None
    assert abs(evidence[found[1]].line.d[1]) > 0.9  # the vertical line (extended over the gap), never the horizontal one
    horizontal_only = {key: value for key, value in evidence.items() if abs(value.line.d[0]) > 0.9}
    assert _line_of_point(horizontal_only, list(seed.center), plan, config, seed.axis_angle) is None
    assert _line_of_point(horizontal_only, list(seed.center), plan, config) is not None  # without the direction the old behaviour stays available


def _record(upm, line, u0, u1, **changes):
    base = dict(
        id="", kind="door", center=[(u0 + u1) / 2 * upm, 0.0], width_m=round(u1 - u0, 3), axis_angle=0.0, wall_thickness_m=0.25, host_wall_id="W001", sill_m=0.0, head_m=2.0,
        height_source="points", evidence="seed", confidence=0.8, line=line, u0=u0, u1=u1,
    )
    base.update(changes)
    return RefinedOpening(**base)


def _door_with_leaf(upm, line="L1"):
    leaf = {"hinge": [4.0, 0.2], "free": [5.2, 0.45], "angle_deg": 80.0, "length_m": 1.2, "points": 5000, "z_min": 0.5, "z_max": 1.8, "hinge_offset_m": 0.1, "hinge_side": "left", "swing_side": -1}
    return _record(upm, line, 1.0, 1.9, kind="door", frame_jambs=["left", "right"], head_frame=True, leaf_detected=True, leaf=leaf, hinge_side="left", swing_side=-1)


@UNITS
def test_candidate_lying_along_an_open_leaf_is_that_leaf(upm):
    config = load_openings_config()
    door = _door_with_leaf(upm)
    # closed-leaf candidate on another wall line, 0.17 m from the leaf segment, no casing strip at all
    shadow = _record(upm, "L2", 0.0, 0.95, center=[4.6 * upm, 0.35 * upm], evidence="seed; door: closed leaf (door cells 0.54, head strip 0.61)")
    kept, dropped = drop_leaf_shadows([door, shadow], config, upm)
    assert [item.line for item in kept] == ["L1"]
    assert len(dropped) == 1 and "open leaf" in dropped[0]["reason"] and dropped[0]["line"] == "L2"


@UNITS
def test_candidate_with_frame_evidence_is_never_taken_for_a_leaf(upm):
    config = load_openings_config()
    door = _door_with_leaf(upm)
    framed = _record(upm, "L2", 0.0, 0.95, center=[4.6 * upm, 0.35 * upm], frame_jambs=["right"], head_frame=True, evidence="seed; door: frame (right jamb + head strip)")
    kept, dropped = drop_leaf_shadows([door, framed], config, upm)
    assert len(kept) == 2 and dropped == []


@UNITS
def test_candidate_far_from_the_leaf_or_on_the_same_wall_stays(upm):
    config = load_openings_config()
    door = _door_with_leaf(upm)
    far = _record(upm, "L2", 0.0, 0.95, center=[8.0 * upm, 0.0], evidence="seed; door: closed leaf")
    same_line = _record(upm, "L1", 3.0, 3.9, center=[4.6 * upm, 0.35 * upm], evidence="seed; door: closed leaf")
    kept, dropped = drop_leaf_shadows([door, far, same_line], config, upm)
    assert len(kept) == 3 and dropped == []


@UNITS
def test_framed_duplicate_is_restored_but_not_where_another_opening_stands(tmp_path: Path, upm: float):
    cloud = Cloud(seed=11)
    cloud.room_shell()
    cloud.wall_with_openings([(1.5, 2.4, 2.05), (3.2, 4.0, 2.05)])
    cloud.frame(1.5, 2.4, 2.05)
    cloud.frame(3.2, 4.0, 2.05)
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    line = lines[0]
    by_line = {line.id: line}
    lost = _record(upm, line.id, 1.5, 2.4, confidence=0.3)  # outranked by a window rectangle, rejected as duplicate
    covered = _record(upm, line.id, 3.2, 4.0, confidence=0.3)
    other = _record(upm, line.id, 3.3, 3.9, confidence=0.9)
    restored = restore_framed_duplicates([other], [lost, covered], by_line, evidence, plan, config)
    assert [(item.u0, item.u1) for item in restored] == [(1.5, 2.4)]
    assert restored[0].frame_jambs and restored[0].head_frame and restored[0].kind == "door"
    assert "restored" in restored[0].evidence


@UNITS
def test_unframed_duplicate_is_not_restored(tmp_path: Path, upm: float):
    cloud = Cloud(seed=12)
    cloud.room_shell()
    cloud.wall_with_openings([(1.5, 2.4, HEIGHT)])
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    lost = _record(upm, lines[0].id, 1.5, 2.4)
    assert restore_framed_duplicates([], [lost], {lines[0].id: lines[0]}, evidence, plan, config) == []


@UNITS
def test_classify_keeps_a_framed_door_that_the_dedupe_dropped(tmp_path: Path, upm: float):
    cloud = Cloud(seed=13)
    cloud.room_shell()
    cloud.wall_with_openings([(1.5, 2.4, 2.05)])
    cloud.frame(1.5, 2.4, 2.05)
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    lost = _record(upm, lines[0].id, 1.5, 2.4, confidence=0.3, plan_seed="O001")
    kept, _dropped, _summary = classify_openings([], lines, evidence, plan, config, duplicates=[lost])
    assert [item.kind for item in kept] == ["door"]
    assert kept[0].frame_jambs and kept[0].head_frame


@UNITS
def test_door_host_pieces_are_no_wall_lines_but_closing_pieces_are(upm):
    """Pieces the wall stage builds around openings must not turn a plan gap into a measurable wall (feedback into the refinement)."""
    config = load_openings_config()
    measured = _wall("W001", (0.0, 0.0), (6.0, 0.0), upm)
    fill = _wall("W002", (3.0, 1.0), (3.0, 2.0), upm)
    fill.evidence = "door-host"
    closure = _wall("W003", (1.0, 2.0), (1.0, 3.0), upm)
    closure.evidence = "closure"
    lines = wall_lines(_plan([measured, fill, closure], [], upm), config)
    assert [line.pieces for line in lines] == [("W001",), ("W003",)]  # closing pieces join measured walls and stay wall lines; door-host pieces do not
