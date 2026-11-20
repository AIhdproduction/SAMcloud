"""Kitchen built-ins never stand on a door (span, casing, open leaf) and niches / upper cabinets only exist in the kitchen."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, box

from samcloud.ortho.vector.door_zones import DoorZoneSettings, door_keep_clear_zones, drop_misplaced_builtins
from samcloud.ortho.vector.indoor import Opening

UNITS = pytest.mark.parametrize("upm", [1.0, 0.9891])


def _poly(x0, y0, x1, y1, upm, **flags):
    return {"type": "polyline", "points": [[x0 * upm, y0 * upm], [x1 * upm, y0 * upm], [x1 * upm, y1 * upm], [x0 * upm, y1 * upm]], "closed": True, "holes": [], **flags}


def _door(upm, centre, width=0.85, axis=math.pi / 2, thickness=0.15):
    return Opening("DOOR", (centre[0] * upm, centre[1] * upm), width * upm, axis, thickness * upm)


@UNITS
def test_upper_cabinet_and_niche_on_the_door_casing_are_dropped(upm):
    door = _door(upm, (4.70, 12.83))  # vertical wall at x = 4.70
    layers = {
        "A-OBEN": [_poly(4.77, 12.65, 4.92, 13.52, upm, _upper_cabinet=True)],
        "A-EINB": [_poly(4.53, 12.37, 4.79, 13.52, upm, _einb_kind="niche"), _poly(8.0, 10.0, 8.6, 10.5, upm, _einb_kind="builtin_front")],
    }
    leaf = LineString([(4.62 * upm, 12.4 * upm), (5.8 * upm, 12.6 * upm)])
    dropped = drop_misplaced_builtins(layers, [door], [leaf], None, upm)
    assert dropped["builtins_dropped_at_door"] == 2
    assert layers["A-OBEN"] == []
    assert [item["_einb_kind"] for item in layers["A-EINB"]] == ["builtin_front"]  # a wardrobe front far from any door stays


@UNITS
def test_kitchen_run_next_to_a_door_casing_is_not_a_door_part(upm):
    door = _door(upm, (0.0, 2.0), axis=0.0)
    run = _poly(-3.0, 1.0, -0.3, 2.0, upm, _einb_kind="fine_kitchen")  # counter ends 0.3 m before the door span
    layers = {"A-EINB": [run], "A-OBEN": []}
    assert drop_misplaced_builtins(layers, [door], [], None, upm)["builtins_dropped_at_door"] == 0
    overlapping = _poly(-1.0, 1.9, 1.0, 2.4, upm, _einb_kind="fine_kitchen")
    layers = {"A-EINB": [overlapping], "A-OBEN": []}
    assert drop_misplaced_builtins(layers, [door], [], None, upm)["builtins_dropped_at_door"] == 0  # kitchen runs / islands are exempt from the door test


@UNITS
def test_niche_and_upper_cabinet_outside_the_kitchen_room_are_dropped(upm):
    kitchen = box(0.0 * upm, 0.0, 5.0 * upm, 4.0 * upm)
    layers = {
        "A-OBEN": [_poly(1.0, 3.5, 2.5, 3.9, upm, _upper_cabinet=True), _poly(6.0, 1.0, 6.4, 2.0, upm, _upper_cabinet=True)],
        "A-EINB": [_poly(0.2, 0.2, 1.0, 0.6, upm, _einb_kind="niche"), _poly(7.0, 1.0, 7.6, 2.0, upm, _einb_kind="niche"), _poly(7.0, 3.0, 7.6, 3.6, upm, _einb_kind="builtin_front")],
    }
    dropped = drop_misplaced_builtins(layers, [], [], kitchen, upm)
    assert dropped["builtins_dropped_outside_kitchen"] == 2
    assert len(layers["A-OBEN"]) == 1 and len(layers["A-EINB"]) == 2
    assert {item.get("_einb_kind") for item in layers["A-EINB"]} == {"niche", "builtin_front"}


@UNITS
def test_without_kitchen_evidence_nothing_is_dropped_for_its_place(upm):
    layers = {"A-OBEN": [_poly(1.0, 1.0, 2.0, 1.3, upm, _upper_cabinet=True)], "A-EINB": [_poly(3.0, 1.0, 3.6, 2.0, upm, _einb_kind="niche")]}
    dropped = drop_misplaced_builtins(layers, [], [], None, upm)
    assert dropped == {"builtins_dropped_at_door": 0, "builtins_dropped_outside_kitchen": 0}


@UNITS
def test_zones_cover_the_span_and_the_leaf(upm):
    door = _door(upm, (0.0, 0.0), axis=0.0)
    leaf = LineString([(-0.4 * upm, 0.1 * upm), (-0.4 * upm, 1.2 * upm)])
    zones = door_keep_clear_zones([door, Opening("WINDOW", (9.0, 9.0), 1.0, 0.0, 0.2)], [leaf], upm, DoorZoneSettings())
    assert len(zones) == 2  # windows have no zone
    assert zones[0].contains(box(-0.3 * upm, -0.05 * upm, 0.3 * upm, 0.05 * upm)) and zones[1].distance(leaf) == 0.0


def test_wall_settings_follow_the_applied_refinement():
    from samcloud.ortho.plan import _wall_settings_for_pass
    from samcloud.ortho.vector.double_walls import DoubleWallSettings

    base = DoubleWallSettings(double_line=True)
    assert _wall_settings_for_pass(base, {}).refined_openings is False
    assert _wall_settings_for_pass(base, {"sha256": "abc", "applied": False}).refined_openings is False
    final = _wall_settings_for_pass(base, {"sha256": "abc", "applied": True})
    assert final.refined_openings is True and final.door_spans is True


@UNITS
def test_upper_cabinet_behind_the_counter_front_stays_in_the_kitchen_zone(upm):
    # The kitchen room outline stops at the counter front; the unit above the run lies behind it (within the margin).
    kitchen = box(0.0 * upm, 0.0, 5.0 * upm, 4.0 * upm)
    layers = {"A-OBEN": [_poly(1.0, 4.1, 3.0, 4.45, upm, _upper_cabinet=True), _poly(1.0, 5.6, 3.0, 5.95, upm, _upper_cabinet=True)], "A-EINB": []}
    dropped = drop_misplaced_builtins(layers, [], [], kitchen, upm)
    assert dropped["builtins_dropped_outside_kitchen"] == 1
    assert len(layers["A-OBEN"]) == 1
