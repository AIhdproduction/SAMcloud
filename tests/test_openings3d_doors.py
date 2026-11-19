"""Door frames, head / headroom, open leaves, door / passage / none classification and the entrance scan.

Every case runs the real streaming point pass and the real classification on a seeded, noisy synthetic point cloud
at units_per_metre 1.0 and 0.8124 (surface samples at random positions, none on a grid boundary).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openings3d_doors_synth import HEIGHT, Cloud, collect
from samcloud.bim.openings3d.classify import assign_leaves, classify_features, classify_openings, scan_entrances
from samcloud.bim.openings3d.frames import measure_head, measure_opening
from samcloud.bim.openings3d.leaves import find_leaf
from samcloud.bim.openings3d.model import RefinedOpening

UNITS = pytest.mark.parametrize("upm", [1.0, 0.8124])


def _seed(upm: float, line_id: str, u0: float, u1: float, kind: str = "door") -> RefinedOpening:
    return RefinedOpening(
        id="", kind=kind, center=[(u0 + u1) / 2 * upm, 0.0], width_m=round(u1 - u0, 3), axis_angle=0.0, wall_thickness_m=0.25, host_wall_id="W001", sill_m=0.0, head_m=2.0,
        height_source="default", evidence="seed", confidence=0.7, line=line_id, u0=u0, u1=u1,
    )


def _door_wall(cloud: Cloud, u0: float, u1: float, head: float = 2.05, **frame) -> None:
    cloud.wall_with_openings([(u0, u1, head)])
    cloud.frame(u0, u1, head, **frame)


@UNITS
def test_frame_head_and_headroom_of_a_door(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=1)
    cloud.room_shell()
    cloud.wall_with_openings([(1.5, 2.4, 2.05)])
    cloud.frame(1.5, 2.4, 2.05)
    evidence, lines, _plan, config = collect(cloud, tmp_path, upm)
    features = measure_opening(evidence.grids[lines[0].id], 1.5, 2.4, HEIGHT, config)
    assert features.frame_jambs == ["left", "right"] and features.head_frame and not features.flat_leaf_sides
    assert features.head_m == pytest.approx(2.05, abs=0.11)
    assert features.headroom_m == pytest.approx(0.45, abs=0.11)
    assert features.head_peak_m is not None and 1.85 <= features.head_peak_m <= 2.3
    assert features.side_wall_cov["left"] > 0.9 and features.side_wall_cov["right"] > 0.9
    assert classify_features(features, config)[0] == "door"


@UNITS
def test_a_single_jamb_without_head_strip_is_no_frame(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=2)
    cloud.room_shell()
    cloud.wall_with_openings([(1.5, 2.4, HEIGHT)])  # up to the ceiling
    cloud.frame(1.5, 2.4, 2.05, jambs=("left",), head_strip=False)
    evidence, lines, _plan, config = collect(cloud, tmp_path, upm)
    features = measure_opening(evidence.grids[lines[0].id], 1.5, 2.4, HEIGHT, config)
    assert features.frame_jambs == ["left"] and not features.head_frame
    assert features.headroom_m is not None and features.headroom_m < 0.15
    assert features.head_m == pytest.approx(HEIGHT - 0.04, abs=0.1)
    assert classify_features(features, config)[0] == "passage"  # a lone jamb strip is not a casing


@UNITS
def test_head_measurement_ignores_the_closed_leaf(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=3)
    cloud.room_shell()
    cloud.wall_with_openings([(2.0, 3.0, 2.1)])
    cloud.frame(2.0, 3.0, 2.1)
    cloud.closed_leaf(2.0, 3.0, top=2.0)
    evidence, lines, _plan, config = collect(cloud, tmp_path, upm)
    grids = evidence.grids[lines[0].id]
    features = measure_opening(grids, 2.0, 3.0, HEIGHT, config)
    assert features.door_fraction > 0.6
    assert measure_head(grids["any"], 2.0, 3.0, config) is not None
    assert features.head_m == pytest.approx(2.1, abs=0.15)  # the lintel above the leaf, not the leaf top
    assert classify_features(features, config)[0] == "door"


@UNITS
def test_open_leaf_gives_hinge_swing_and_angle(tmp_path: Path, upm: float) -> None:
    for hinge_side, hinge_u, toward, swing in (("left", 1.5, 1, 1), ("right", 2.4, -1, -1), ("left", 1.5, 1, -1), ("right", 2.4, -1, 1)):
        cloud = Cloud(seed=4)
        cloud.room_shell()
        _door_wall(cloud, 1.5, 2.4)
        cloud.open_leaf(hinge_u + (0.03 if hinge_side == "right" else -0.03), 80.0, swing=swing, toward=toward)
        evidence, _lines, _plan, config = collect(cloud, tmp_path, upm)
        leaf = find_leaf(evidence.door_points, (1.95, 0.0), 0.0, 0.9, config)
        assert leaf is not None, (hinge_side, swing)
        assert leaf.hinge_side == hinge_side and leaf.swing_side == swing
        assert leaf.angle_deg == pytest.approx(80.0, abs=9.0)
        assert 0.7 <= leaf.length_m <= 1.3 and leaf.z_min < 0.75 and leaf.z_max > 1.65
        assert leaf.hinge_offset_m <= 0.07  # the jamb stands this close to the leaf plane


@UNITS
def test_closed_leaf_is_not_an_open_leaf(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=6)
    cloud.room_shell()
    _door_wall(cloud, 1.5, 2.4)
    cloud.closed_leaf(1.5, 2.4)
    evidence, _lines, _plan, config = collect(cloud, tmp_path, upm)
    assert find_leaf(evidence.door_points, (1.95, 0.0), 0.0, 0.9, config) is None


@UNITS
def test_door_passage_and_no_opening_are_told_apart(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=7)
    cloud.room_shell(5.0)
    cloud.wall_with_openings([(0.4, 1.3, 2.05), (1.9, 2.7, HEIGHT)], length=3.0)  # door (lintel + frame), passage (to the ceiling)
    cloud.frame(0.4, 1.3, 2.05)
    # nothing at all (no wall, no points) from 3.0 m on: a seed there is not on an observed wall
    evidence, lines, plan, config = collect(cloud, tmp_path, upm, length=5.0)
    line = lines[0].id
    seeds = [_seed(upm, line, 0.4, 1.3), _seed(upm, line, 1.9, 2.7), _seed(upm, line, 4.0, 4.9)]
    kept, dropped, summary = classify_openings(seeds, lines, evidence, plan, config)
    by_kind = {round(item.u0, 1): item for item in kept}
    assert by_kind[0.4].kind == "door" and by_kind[0.4].classification == "door"
    assert by_kind[0.4].frame_jambs == ["left", "right"] and by_kind[0.4].head_frame
    assert by_kind[0.4].headroom_m == pytest.approx(0.45, abs=0.12)
    assert by_kind[1.9].kind == "passage" and not by_kind[1.9].frame_jambs
    assert by_kind[1.9].headroom_m is not None and by_kind[1.9].headroom_m < 0.2
    assert by_kind[1.9].head_m == pytest.approx(HEIGHT - 0.04, abs=0.12)  # measured, not the default 2.0
    assert len(kept) == 2 and len(dropped) == 1
    assert "not on an observed wall" in dropped[0]["reason"]
    assert summary["doors"] == 1 and summary["passages"] == 1 and summary["dropped"] == 1


@UNITS
def test_furniture_in_the_wall_band_is_no_passage(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=8)
    cloud.room_shell()
    cloud.wall_with_openings([(1.6, 2.5, HEIGHT)])  # a gap to the ceiling ...
    cloud.furniture(1.6, 2.5)  # ... filled by a wardrobe standing in the wall band
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    kept, dropped, _summary = classify_openings([_seed(upm, lines[0].id, 1.6, 2.5)], lines, evidence, plan, config)
    assert kept == [] and len(dropped) == 1
    features = dropped[0]["features"]
    assert features["blocked_fraction"] > 0.5 and features["free_fraction"] < 0.3
    assert dropped[0]["was"] == "door" and "no door evidence" in dropped[0]["reason"]


@UNITS
def test_leaf_of_a_neighbouring_door_creates_no_door_and_no_leaf_on_a_passage(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=9)
    cloud.room_shell()
    cloud.wall_with_openings([(0.4, 1.3, 2.05), (1.45, 2.3, HEIGHT)])
    cloud.frame(0.4, 1.3, 2.05)
    cloud.open_leaf(1.33, 85.0, swing=1, toward=-1)  # leaf of the first door, hinged 3 cm beyond its right jamb, within the hinge zone of the passage
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    seeds = [_seed(upm, lines[0].id, 0.4, 1.3), _seed(upm, lines[0].id, 1.45, 2.3)]
    kept, dropped, _summary = classify_openings(seeds, lines, evidence, plan, config)
    door = next(item for item in kept if item.u0 < 1.0)
    passage = next(item for item in kept if item.u0 > 1.0)
    assert door.kind == "door" and door.leaf_detected and door.hinge_side == "right" and door.swing_side == 1
    assert passage.kind == "passage" and not passage.leaf_detected and passage.hinge_side is None and passage.swing_side is None
    assert not dropped


@UNITS
def test_an_open_leaf_alone_never_makes_a_door(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=10)
    cloud.room_shell()
    cloud.wall_with_openings([(2.0, 2.9, HEIGHT)])
    cloud.open_leaf(2.03, 75.0, swing=-1, toward=-1)  # a leaf stands at the jamb, but there is no frame and no lintel
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    assert find_leaf(evidence.door_points, (2.45, 0.0), 0.0, 0.9, config) is not None  # the leaf is real ...
    kept, _dropped, _summary = classify_openings([_seed(upm, lines[0].id, 2.0, 2.9)], lines, evidence, plan, config)
    assert [item.kind for item in kept] == ["passage"]  # ... and still makes no door
    assert not kept[0].leaf_detected and kept[0].hinge_side is None


@UNITS
def test_assign_leaves_gives_the_leaf_to_the_door_whose_jamb_carries_the_hinge(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=11)
    cloud.room_shell()
    cloud.wall_with_openings([(0.4, 1.3, 2.05), (1.5, 2.4, 2.05)])
    cloud.frame(0.4, 1.3, 2.05)
    cloud.frame(1.5, 2.4, 2.05)
    cloud.open_leaf(1.35, 80.0, swing=1, toward=-1)  # between the two doors: 5 cm from the first door's jamb, 15 cm from the second's
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    seeds = [_seed(upm, lines[0].id, 0.4, 1.3), _seed(upm, lines[0].id, 1.5, 2.4)]
    kept, _dropped, _summary = classify_openings(seeds, lines, evidence, plan, config)
    assert [item.kind for item in kept] == ["door", "door"]
    assert [bool(item.leaf_detected) for item in kept] == [True, False]
    assert find_leaf(evidence.door_points, (1.95, 0.0), 0.0, 0.9, config) is not None  # the second door finds the same leaf on its own ...
    assert kept[0].hinge_side == "right" and kept[0].swing_side == 1  # ... and loses it to the door whose jamb is closer to the hinge
    assert assign_leaves(kept, None, config) == kept  # no door points: nothing changes


@UNITS
def test_exterior_door_is_added_from_a_frame_in_a_wall_without_opening(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=12)
    cloud.room_shell()
    cloud.wall_with_openings([(2.0, 3.0, 2.05)])
    cloud.frame(2.0, 3.0, 2.05)
    cloud.closed_leaf(2.0, 3.0)
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    kept, dropped, summary = classify_openings([], lines, evidence, plan, config)
    assert len(kept) == 1 and summary["added_by_scan"] == 1 and not dropped
    door = kept[0]
    assert door.kind == "door" and door.host_wall_id == "W001" and door.line == lines[0].id
    assert door.width_m == pytest.approx(1.0, abs=0.2) and abs(door.center[0] / upm - 2.5) < 0.2
    assert door.frame_jambs and door.evidence.startswith("door frame found by the wall scan")


@UNITS
def test_no_door_is_added_without_a_frame(tmp_path: Path, upm: float) -> None:
    for name, build in (
        ("empty gap", lambda c: c.wall_with_openings([(2.0, 3.0, 2.05)])),
        ("closed leaf without casing", lambda c: (c.wall_with_openings([(2.0, 3.0, 2.05)]), c.closed_leaf(2.0, 3.0))),
        ("flat open leaf against the wall", lambda c: (c.wall_with_openings([(2.0, 3.0, 2.05)]), c.box(3.1, 4.0, -0.03, 0.03, 0.05, 2.0, 103, density=2500.0))),
        ("single jamb", lambda c: (c.wall_with_openings([(2.0, 3.0, 2.05)]), c.frame(2.0, 3.0, 2.05, jambs=("left",), head_strip=False))),
    ):
        cloud = Cloud(seed=13)
        cloud.room_shell()
        build(cloud)
        evidence, lines, plan, config = collect(cloud, tmp_path, upm)
        added, report = scan_entrances(lines, evidence, plan, [], config)
        assert added == [], (name, report)


@UNITS
def test_scan_never_overlaps_an_existing_opening(tmp_path: Path, upm: float) -> None:
    cloud = Cloud(seed=14)
    cloud.room_shell()
    _door_wall(cloud, 2.0, 3.0)
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    seeds = [_seed(upm, lines[0].id, 2.0, 3.0)]
    kept, _dropped, summary = classify_openings(seeds, lines, evidence, plan, config)
    assert len(kept) == 1 and summary["added_by_scan"] == 0
    assert any(item["result"].startswith("skipped") for item in summary["scan"])


@UNITS
def test_classification_is_deterministic(tmp_path: Path, upm: float) -> None:
    results = []
    for _ in range(2):
        cloud = Cloud(seed=15)
        cloud.room_shell()
        cloud.wall_with_openings([(0.4, 1.3, 2.05), (1.9, 2.7, HEIGHT)])
        cloud.frame(0.4, 1.3, 2.05)
        cloud.open_leaf(1.33, 85.0, swing=1, toward=-1)
        evidence, lines, plan, config = collect(cloud, tmp_path, upm)
        kept, dropped, summary = classify_openings([_seed(upm, lines[0].id, 0.4, 1.3), _seed(upm, lines[0].id, 1.9, 2.7)], lines, evidence, plan, config)
        results.append(([item.to_dict() for item in kept], dropped, summary))
    assert results[0] == results[1]


@UNITS
def test_scan_skips_a_span_where_the_plan_has_an_opening_the_refinement_did_not_measure(tmp_path: Path, upm: float) -> None:
    from samcloud.bim.plan_model import PlanOpening

    cloud = Cloud(seed=16)
    cloud.room_shell()
    _door_wall(cloud, 2.0, 3.0)
    cloud.closed_leaf(2.0, 3.0)
    evidence, lines, plan, config = collect(cloud, tmp_path, upm)
    added, _report = scan_entrances(lines, evidence, plan, [], config)
    assert len(added) == 1  # without a plan opening the frame is found ...
    plan.openings = [PlanOpening("O001", "door", [2.45 * upm, 0.05 * upm], 0.9, 0.0, 0.25, host_wall_id="W001")]
    added, report = scan_entrances(lines, evidence, plan, [], config)
    assert added == [] and any(item["result"].startswith("skipped") for item in report)  # ... with one it is that opening, not a new door
