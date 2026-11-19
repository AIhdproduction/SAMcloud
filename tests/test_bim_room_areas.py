"""QA: IfcSpace area must match the plan room (per room and in total)."""

from __future__ import annotations

import pytest

import test_bim_scene_builder as scene_tests
from samcloud.bim.export import check_room_areas
from samcloud.bim.scene_builder import build_scene

UPMS = (1.0, 0.8124)


@pytest.mark.parametrize("upm", UPMS)
def test_consistent_plan_has_matching_space_areas(upm: float) -> None:
    scene = build_scene(scene_tests.make_plan(upm))
    check = check_room_areas(scene.derived["samcloud_build"])
    assert check["ok"], check["errors"]
    assert len(check["rooms"]) == 2
    for room in check["rooms"]:
        assert room["ifc_m2"] == pytest.approx(room["plan_m2"], rel=0.03)


@pytest.mark.parametrize("upm", UPMS)
def test_room_cut_in_pieces_by_a_wall_keeps_every_piece_as_a_space(upm: float) -> None:
    """One plan room that covers both sides of the partition: the wall clipping splits it; both pieces are written (no area is dropped)."""
    plan = scene_tests.make_plan(upm)
    wide = scene_tests._rect(0.15, 0.15, 7.85, 4.85)
    plan.rooms[0].polygon = {"exterior": [[x * upm, y * upm] for x, y in wide], "holes": []}
    plan.rooms[0].area_m2 = 7.7 * 4.7
    plan.rooms = plan.rooms[:1]
    plan.openings = [o for o in plan.openings if o.kind != "door"]  # no doorway across the partition: the wall really splits the room
    scene = build_scene(plan)
    build = scene.derived["samcloud_build"]
    spaces = [e for e in scene.elements.values() if e.kind.name == "SPACE"]
    assert len(spaces) == 2 and {e.tag for e in spaces} == {"R1", "R1.1"} and {e.name for e in spaces} == {"Living"}
    check = check_room_areas(build)
    assert check["ok"], check["errors"]
    assert check["rooms"][0]["ifc_m2"] == pytest.approx(36.19 - 0.2 * 4.7, rel=0.02)  # the partition wall is the only loss
    assert any("2 pieces" in warning for warning in build["warnings"])


def test_dropped_room_counts_as_zero_area() -> None:
    check = check_room_areas({"room_areas": [{"id": "R1", "name": "A", "plan_m2": 10.0, "ifc_m2": 10.1}, {"id": "R2", "name": "B", "plan_m2": 5.0, "ifc_m2": 0.0}]})
    assert not check["ok"] and [e["room"] for e in check["errors"]] == ["R2", "total"]
    assert check_room_areas({"room_areas": [{"id": "R1", "name": "A", "plan_m2": 3.0, "ifc_m2": 2.8}]})["ok"]  # 0.2 m2 < the 0.25 m2 allowance


@pytest.mark.parametrize("upm", UPMS)
def test_room_reaching_through_a_doorway_stays_one_space(upm: float) -> None:
    """With the door in the partition the doorway is no wall: the room that reaches across it keeps its whole area."""
    plan = scene_tests.make_plan(upm)
    wide = scene_tests._rect(0.15, 0.15, 7.85, 4.85)
    plan.rooms[0].polygon = {"exterior": [[x * upm, y * upm] for x, y in wide], "holes": []}
    plan.rooms[0].area_m2 = 7.7 * 4.7
    plan.rooms = plan.rooms[:1]
    check = check_room_areas(build_scene(plan).derived["samcloud_build"])
    assert check["rooms"][0]["ifc_m2"] > 0.9 * 36.0  # only the partition wall itself is lost (7.7 x 4.7 = 36.2 m2, partition about 1 m2)
