"""Rules for openings that share a wall gap: casings are used once, weak evidence needs a plan door, windows absorb doors.

Classification cases run the real streaming point pass and the real classification on a seeded, noisy synthetic cloud at
units_per_metre 1.0 and 0.8124; the detection cases use the noisy wall-plane evidence of ``openings3d_synth``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from openings3d_doors_synth import HEIGHT, Cloud, collect
from openings3d_synth import Scene, build_evidence, make_plan, seed_door
from samcloud.bim.openings3d.classify import classify_features, classify_openings
from samcloud.bim.openings3d.config import load_openings_config
from samcloud.bim.openings3d.detect import detect_openings, rooms_on_both_sides
from samcloud.bim.openings3d.frames import measure_opening
from samcloud.bim.openings3d.model import RefinedOpening
from samcloud.bim.openings3d.walls import wall_lines
from samcloud.bim.plan_model import PlanRoom

UNITS = pytest.mark.parametrize("upm", [1.0, 0.8124])


def _seed(upm: float, line_id: str, u0: float, u1: float, kind: str = "door", plan_seed: str | None = "O001") -> RefinedOpening:
    return RefinedOpening(
        id="", kind=kind, center=[(u0 + u1) / 2 * upm, 0.0], width_m=round(u1 - u0, 3), axis_angle=0.0, wall_thickness_m=0.25, host_wall_id="W001", sill_m=0.0, head_m=2.0,
        height_source="default", evidence="seed", confidence=0.7, line=line_id, u0=u0, u1=u1, plan_seed=plan_seed,
    )


@UNITS
def test_a_door_and_a_passage_in_one_wall_gap_never_overlap_and_the_jamb_is_used_once(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=21)
    cloud.room_shell()
    cloud.wall_with_openings([(1.0, 1.85, HEIGHT), (2.0, 2.9, 2.05)])  # a passage up to the ceiling, 15 cm of casing wall, a door under a lintel
    cloud.frame(2.0, 2.9, 2.05)  # the casing stands at the door only
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    # the plan seeds overlap: the passage seed reaches 13 cm into the door, the casing of the door lies inside the passage seed's right edge
    seeds = [_seed(upm, lines[0].id, 1.0, 2.08, "passage", "O008"), _seed(upm, lines[0].id, 1.95, 2.9, "door", "O003")]
    # the passage seed alone would take the door's left jamb strip as its own frame ...
    shared = measure_opening(evidence.grids[lines[0].id], 1.0, 2.08, HEIGHT, config)
    assert "right" in shared.frame_jambs
    kept, dropped, _summary = classify_openings(seeds, lines, evidence, plan, config)
    assert [item.kind for item in sorted(kept, key=lambda item: item.u0)] == ["passage", "door"], [item.to_dict() for item in kept]
    passage, door = sorted(kept, key=lambda item: item.u0)
    assert passage.u1 <= door.u0 + 0.01 and not dropped  # trimmed at the stronger opening: no overlap
    assert door.frame_jambs and door.head_frame
    assert not passage.frame_jambs  # the casing belongs to the door
    assert passage.width_m == pytest.approx(passage.u1 - passage.u0, abs=0.001)
    assert abs(passage.center[0] / upm - (passage.u0 + passage.u1) / 2) < 0.06  # centre follows the trimmed span


@UNITS
def test_a_trimmed_opening_that_is_too_narrow_is_dropped_and_closed(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=22)
    cloud.room_shell()
    cloud.wall_with_openings([(1.5, 2.4, 2.05)])
    cloud.frame(1.5, 2.4, 2.05)
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    seeds = [_seed(upm, lines[0].id, 1.5, 2.4, "door", "O001"), _seed(upm, lines[0].id, 1.0, 1.9, "door", "O002")]  # the second one is mostly inside the first
    kept, dropped, _summary = classify_openings(seeds, lines, evidence, plan, config)
    assert [(item.u0, item.u1) for item in kept] == [(1.5, 2.4)]
    assert len(dropped) == 1 and "too narrow" in dropped[0]["reason"] and dropped[0]["u"] == [1.0, 1.9]


@UNITS
def test_a_gap_without_any_point_up_to_the_ceiling_is_a_passage(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=23)
    cloud.box(0.0, 4.4, -0.4, 0.4, 0.0, 0.01, 102, density=300.0)  # floor only: the ceiling was not measured above the wall
    cloud.wall_with_openings([(1.6, 2.5, HEIGHT)])
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    features = measure_opening(evidence.grids[lines[0].id], 1.6, 2.5, HEIGHT, config)
    assert features.head_m is None and features.free_fraction > 0.9
    assert classify_features(features, config)[0] == "passage"
    off = load_openings_config(overrides={"classify": {"passage_unmeasured_head_open": False}})
    assert classify_features(features, off, seeded=False)[0] == "none"


@UNITS
def test_door_class_points_without_head_strip_are_no_closed_leaf_and_an_unseeded_gap_is_no_door(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=24)
    cloud.room_shell()
    cloud.wall_with_openings([(1.6, 2.8, HEIGHT)])  # a 1.2 m gap up to the ceiling
    cloud.box(1.75, 2.6, -0.02, 0.02, 0.05, 1.9, 103, density=2500.0)  # the open leaf of the neighbouring door lies in it (no head strip)
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    features = measure_opening(evidence.grids[lines[0].id], 1.6, 2.8, HEIGHT, config)
    assert features.door_fraction >= 0.4 and features.head_cov < 0.5
    assert not features.head_frame
    assert classify_features(features, config, seeded=False)[0] == "none"  # nothing but door-class cells: no casing, no head, no lintel
    assert classify_features(features, config, seeded=True)[0] == "door"  # a plan door there keeps the safe default
    kept, dropped, _summary = classify_openings([_seed(upm, lines[0].id, 1.6, 2.8, "door", None)], lines, evidence, plan, config)
    assert kept == [] and len(dropped) == 1


@UNITS
def test_a_closed_leaf_under_a_head_strip_is_still_a_door(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=25)
    cloud.room_shell()
    cloud.wall_with_openings([(2.0, 3.0, 2.05)])
    cloud.frame(2.0, 3.0, 2.05, jambs=(), head_strip=True)
    cloud.closed_leaf(2.0, 3.0)
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    features = measure_opening(evidence.grids[lines[0].id], 2.0, 3.0, HEIGHT, config)
    assert features.door_fraction > 0.6 and features.head_cov >= 0.5
    kind, reason = classify_features(features, config, seeded=False)
    assert kind == "door"


# --------------------------------------------------------------------------
# detection: windows with a room on both sides, doors inside a window
# --------------------------------------------------------------------------

def _room(identifier: str, ring: list[list[float]]) -> PlanRoom:
    return PlanRoom(identifier, "Zimmer", {"exterior": ring, "holes": []}, 0.0, ring[0])


def test_a_weak_window_with_a_room_on_both_sides_is_no_window_even_on_a_line_called_exterior() -> None:
    rect = (1.0, 1.5, 0.25, 1.9)
    scene = Scene(kind="exterior", holes=[rect], windows=[rect])
    evidence, plan, config = build_evidence(scene)
    both = [_room("R1", [[-1.0, 0.3], [3.0, 0.3], [3.0, 2.0], [-1.0, 2.0]]), _room("R2", [[-1.0, -2.0], [3.0, -2.0], [3.0, -0.3], [-1.0, -0.3]])]
    outside = [both[0]]  # a room on one side only: a real exterior window
    for rooms, expected in ((both, 0), (outside, 1), ([], 1)):
        plan.rooms = list(rooms)
        detection = detect_openings({evidence.line.id: evidence}, [], plan, config)
        assert len([o for o in detection.openings if o.kind == "window"]) == expected, (len(rooms), [o.to_dict() for o in detection.openings])
    plan.rooms = list(both)
    assert rooms_on_both_sides(evidence.line, 1.0, 1.5, [__import__("shapely.geometry", fromlist=["Polygon"]).Polygon(r.polygon["exterior"]) for r in both], 0.3, 1.0)
    assert not rooms_on_both_sides(evidence.line, 1.0, 1.5, [], 0.3, 1.0)


def test_a_door_gap_inside_a_window_on_an_exterior_wall_is_part_of_the_window_but_a_floor_level_door_stays() -> None:
    window = (1.0, 2.9, 0.65, 2.05)  # sill 0.65 m across the whole span, so the wall below the window is a parapet
    gap_in_window = (2.0, 2.9, 0.0, 2.05)
    scene = Scene(kind="exterior", holes=[window], windows=[window])
    evidence, plan, config = build_evidence(scene)
    detection = detect_openings({evidence.line.id: evidence}, [seed_door("O003", 2.45, 0.9)], plan, config)
    assert [o.kind for o in detection.openings] == ["window"], [o.to_dict() for o in detection.openings]
    assert detection.openings[0].width_m == pytest.approx(1.9, abs=0.1)
    assert any("part of the window" in str(item.get("reason")) for item in detection.rejected)
    # the door gap reaches the floor beside the window (a door next to a window, not inside it): both stay
    window2 = (1.0, 2.0, 0.65, 2.05)
    scene = Scene(kind="exterior", holes=[window2, gap_in_window], windows=[window2], doors=[gap_in_window])
    evidence, plan, config = build_evidence(scene)
    detection = detect_openings({evidence.line.id: evidence}, [seed_door("O003", 2.45, 0.9)], plan, config)
    assert sorted(o.kind for o in detection.openings) == ["door", "window"], [o.to_dict() for o in detection.openings]
