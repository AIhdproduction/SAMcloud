"""Door invariants (samcloud.bim.invariants): host wall, frame/lintel backing, no wall body inside a door span."""

from __future__ import annotations

import math

import pytest

from samcloud.bim.invariants import door_backing, plan_violations, refinement_violations
from samcloud.bim.plan_model import BuildingPlan, PlanOpening, PlanWall


def _opening(**changes):
    base = dict(
        id="O001", kind="door", center=[2.0, 0.0], width_m=0.9, axis_angle=0.0, wall_thickness_m=0.2, host_wall_id="W001", hinge_side=None, swing_side=None,
        leaf_angle_deg=None, leaf_detected=None, sill_m=0.0, head_m=2.05, height_source="points", mullion_count=0, evidence="backing=frame; seed O003", confidence=0.9,
    )
    base.update(changes)
    return PlanOpening(**base)


def _plan(openings, extra_walls=(), upm=1.0):
    walls = [PlanWall(id="W001", centreline=[[0.0, 0.0], [4.0 * upm, 0.0]], thickness_m=0.2, kind="interior"), *extra_walls]
    frame = {"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    return BuildingPlan("synthetic", "level_1", "Level 1", upm, frame, 0.0, 2.5 * upm, walls=walls, openings=openings)


def _wall(identifier, p0, p1, upm, thickness=0.2):
    return PlanWall(id=identifier, centreline=[[p0[0] * upm, p0[1] * upm], [p1[0] * upm, p1[1] * upm]], thickness_m=thickness, kind="interior")


@pytest.mark.parametrize("upm", [1.0, 0.9891])
def test_host_wall_covering_the_span_is_fine_and_a_t_junction_at_the_jamb_too(upm):
    # the host wall runs through the span (the IFC voids it); a crossing wall that meets the jamb covers only a sliver
    crossing = _wall("W002", (1.45, 0.0), (1.45, 3.0), upm)
    assert plan_violations(_plan([_opening(center=[2.0 * upm, 0.0])], [crossing], upm)) == []


@pytest.mark.parametrize("upm", [1.0, 0.9891])
def test_other_wall_standing_in_the_door_span_is_a_violation(upm):
    # a second wall runs along the door span (e.g. the open leaf or a wall closing the gap): "a door is never a wall"
    closing = _wall("W002", (1.4, 0.0), (2.6, 0.0), upm)
    found = plan_violations(_plan([_opening(center=[2.0 * upm, 0.0])], [closing], upm))
    assert [item["rule"] for item in found] == ["wall_in_door"]


def test_door_without_host_wall_is_flagged():
    found = plan_violations(_plan([_opening(host_wall_id=None)]))
    assert "door_has_host" in [item["rule"] for item in found]


def test_door_with_unknown_host_is_flagged():
    found = plan_violations(_plan([_opening(host_wall_id="W099")]))
    assert "door_has_host" in [item["rule"] for item in found]


def test_measured_door_needs_frame_or_lintel_backing():
    found = plan_violations(_plan([_opening(evidence="3d:points; leaf only")]))
    assert [item["rule"] for item in found] == ["door_has_frame_or_lintel"]
    assert plan_violations(_plan([_opening(evidence="3d:points; backing=lintel; seed O001")])) == []


def test_two_dimensional_doors_are_not_judged_for_backing():
    assert plan_violations(_plan([_opening(evidence="gap", confidence=None)])) == []


def test_refined_door_backing_labels():
    assert door_backing({"frame_jambs": ["left"], "head_frame": False}) == "frame"
    assert door_backing({"evidence": "x; door: closed leaf (door cells 0.54)"}) == "closed leaf"
    assert door_backing({"evidence": "door: lintel (headroom 0.5 m)"}) == "lintel"
    assert door_backing({"evidence": "plan gap only (host is a short stub)"}) == "plan gap"
    assert door_backing({"evidence": "seed O001; door frac 0.8"}) is None


def test_refinement_violations_flag_unhosted_and_unbacked_doors():
    good = {"id": "C001", "kind": "door", "host_wall_id": "W001", "frame_jambs": ["left", "right"], "head_frame": True, "evidence": ""}
    no_host = {"id": "C002", "kind": "door", "host_wall_id": None, "frame_jambs": [], "head_frame": False, "evidence": "lintel (headroom 0.5 m)"}
    leaf_only = {"id": "C003", "kind": "door", "host_wall_id": "W002", "frame_jambs": [], "head_frame": False, "evidence": "seed O002; leaf hinge left"}
    window = {"id": "C004", "kind": "window", "host_wall_id": None, "evidence": ""}
    found = refinement_violations([good, no_host, leaf_only, window])
    assert sorted((item["opening"], item["rule"]) for item in found) == [("C002", "door_has_host"), ("C003", "door_has_frame_or_lintel")]


def test_angle_is_respected_for_vertical_walls():
    vertical = _wall("W001", (0.0, 0.0), (0.0, 4.0), 1.0)
    plan = _plan([_opening(center=[0.0, 2.0], axis_angle=math.pi / 2)])
    plan.walls[0] = vertical
    assert plan_violations(plan) == []
