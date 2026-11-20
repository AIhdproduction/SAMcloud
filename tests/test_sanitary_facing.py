"""Sanitary blocks (WC, washbasin) have their back on the nearest final wall face and face the room."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from samcloud.bim.plan_model import PlanFurniture, PlanWall, _assign_back_walls
from samcloud.ortho.vector.sanitary_facing import SanitaryFacingSettings, orient_sanitary_inserts
from samcloud.ortho.vector.symbols import CATALOGUE

UPM = 0.989  # plan units per metre (as in a real project)
SETTINGS = SanitaryFacingSettings()


def _walls(*boxes_m):
    return unary_union([box(*(value * UPM for value in item)) for item in boxes_m])


def _room():
    """Room 3.0 x 2.0 m, walls 0.25 m thick; inner faces y = 0 (south), y = 2 (north), x = 0 (west), x = 3 (east)."""
    return _walls((-0.25, -0.25, 3.25, 0.0), (-0.25, 2.0, 3.25, 2.25), (-0.25, 0.0, 0.0, 2.0), (3.0, 0.0, 3.25, 2.0))


def _insert(kind, centre_m, facing_deg, *, jitter=(0.0, 0.0), tilt_deg=0.0):
    """A catalogue insert whose front looks along ``facing_deg``; ``centre_m`` is the centre of its footprint."""
    entry = CATALOGUE[kind][0]
    facing = math.radians(facing_deg + tilt_deg)
    y_axis = np.array([math.cos(facing), math.sin(facing)])
    x_axis = np.array([y_axis[1], -y_axis[0]])
    centre = (np.array(centre_m) + np.array(jitter)) * UPM
    origin = centre - y_axis * entry.depth_m * UPM / 2.0
    return {
        "type": "insert", "name": entry.block, "point": [float(origin[0]), float(origin[1])],
        "rotation": math.degrees(math.atan2(x_axis[1], x_axis[0])) % 360.0, "xscale": UPM, "yscale": UPM,
        "attributes": {}, "_catalogue": {"kind": kind, "width_m": entry.width_m, "depth_m": entry.depth_m, "layer": "P-SANI", "wall_snapped": False},
    }


def _footprint(entity):
    catalogue = entity["_catalogue"]
    w, d = catalogue["width_m"] * UPM, catalogue["depth_m"] * UPM
    r = math.radians(entity["rotation"])
    x_axis, y_axis = np.array([math.cos(r), math.sin(r)]), np.array([-math.sin(r), math.cos(r)])
    o = np.array(entity["point"])
    return Polygon([o - x_axis * w / 2, o + x_axis * w / 2, o + x_axis * w / 2 + y_axis * d, o - x_axis * w / 2 + y_axis * d])


def _run(entity, walls=None):
    report = orient_sanitary_inserts({"P-SANI": [entity]}, _room() if walls is None else walls, UPM, SETTINGS)
    return entity, report


def _same_angle(a, b, tolerance=1.0):
    return abs(((a - b + 180.0) % 360.0) - 180.0) < tolerance


def _assert_back_on(entity, walls, facing_expected):
    assert _same_angle(entity["_facing_deg"], facing_expected)
    polygon = _footprint(entity)
    assert polygon.intersection(walls).area < 1e-4  # no overlap into the wall
    assert polygon.distance(walls) < 0.004 * UPM  # no gap


def test_reversed_wc_is_turned_to_back_on_the_wall():
    # Reversed: the bowl front looks at the north wall while the back stands 0.4 m in front of it (the v12 bathroom case).
    walls = _room()
    entity, report = _run(_insert("WC_WALL", (1.5, 1.72), 90.0, jitter=(0.012, -0.015)), walls)
    _assert_back_on(entity, walls, 270.0)
    assert report["sanitary_facing_flipped"] == 1


def test_correct_wc_stays_and_is_snapped_flush():
    walls = _room()
    entity, report = _run(_insert("WC_WALL", (1.5, 1.68), 270.0, jitter=(-0.01, 0.01)), walls)
    _assert_back_on(entity, walls, 270.0)
    assert report["sanitary_facing_flipped"] == 0 and report["sanitary_facing_checked"] == 1


def test_wc_back_overlapping_the_wall_is_moved_out():
    walls = _room()
    entity, _ = _run(_insert("WC_WALL", (0.9, 0.22), 90.0), walls)  # back 0.06 m inside the south wall body
    _assert_back_on(entity, walls, 90.0)


@pytest.mark.parametrize(("x", "back_facing", "wrong"), [(0.25, 0.0, 180.0), (2.75, 180.0, 0.0)])
def test_washbasin_on_east_and_west_wall(x, back_facing, wrong):
    walls = _room()
    entity, _ = _run(_insert("WASHBASIN", (x, 1.0), wrong, jitter=(0.01, 0.02)), walls)  # reversed: front on the wall
    _assert_back_on(entity, walls, back_facing)


def test_slightly_rotated_footprint_is_squared_onto_the_wall():
    walls = _room()
    entity, _ = _run(_insert("WC_WALL", (1.2, 0.30), 90.0, tilt_deg=4.0, jitter=(0.01, -0.01)), walls)
    _assert_back_on(entity, walls, 90.0)
    assert abs(((entity["rotation"] + 45.0) % 90.0) - 45.0) < 0.5  # axis-aligned like the wall


def test_corner_basin_keeps_the_wall_its_back_edge_is_parallel_to():
    walls = _room()
    # South-west corner: back on the south wall (long back edge parallel to it), the left end touching the west wall.
    entity, _ = _run(_insert("WASHBASIN", (0.40, 0.25), 90.0, jitter=(0.004, 0.011)), walls)
    _assert_back_on(entity, walls, 90.0)


def test_corner_basin_reversed_still_back_on_south_wall():
    walls = _room()
    entity, _ = _run(_insert("WASHBASIN", (0.40, 0.25), 270.0, jitter=(-0.003, 0.012)), walls)
    _assert_back_on(entity, walls, 90.0)


def test_wc_with_its_long_side_on_the_wall_is_turned_perpendicular():
    walls = _room()
    entity, report = _run(_insert("WC_WALL", (1.0, 0.20), 0.0, jitter=(0.0, 0.01)), walls)  # 0.56 m long axis along the south wall
    assert report["sanitary_facing_checked"] == 1
    polygon = _footprint(entity)
    x0, y0, x1, y1 = polygon.bounds
    assert polygon.distance(walls) < 0.02
    assert (y1 - y0) > (x1 - x0)  # long axis perpendicular to the wall


def test_free_standing_item_is_untouched():
    basin = _insert("WASHBASIN", (1.5, 1.0), 90.0)
    before = (list(basin["point"]), basin["rotation"])
    entity, report = _run(basin)
    assert (entity["point"], entity["rotation"]) == before and "_facing_deg" not in entity
    assert len(report["sanitary_facing_free"]) == 1


def test_bathtub_is_left_alone():
    tub = _insert("BATHTUB", (1.5, 0.45), 270.0)
    before = (list(tub["point"]), tub["rotation"])
    entity, report = _run(tub)
    assert (entity["point"], entity["rotation"]) == before and report["sanitary_facing_checked"] == 0


def test_disabled_setting_changes_nothing():
    wc = _insert("WC_WALL", (1.5, 1.72), 90.0)
    before = (list(wc["point"]), wc["rotation"])
    report = orient_sanitary_inserts({"P-SANI": [wc]}, _room(), UPM, SanitaryFacingSettings(enabled=False))
    assert (wc["point"], wc["rotation"]) == before and report["sanitary_facing_checked"] == 0


def test_wall_face_lines_work_as_walls_too():
    line = LineString([(0.0, 2.0 * UPM), (3.0 * UPM, 2.0 * UPM)])
    entity, _ = _run(_insert("WC_WALL", (1.5, 1.72), 90.0), line)
    assert _same_angle(entity["_facing_deg"], 270.0)
    assert _footprint(entity).distance(line) < 0.004 * UPM


def test_back_wall_id_follows_the_facing():
    north = PlanWall(id="W001", centreline=[[0.0, 2.125 * UPM], [3.0 * UPM, 2.125 * UPM]], thickness_m=0.25, kind="interior", evidence="x",
                     face_a=[[0.0, 2.0 * UPM], [3.0 * UPM, 2.0 * UPM]], face_b=[[0.0, 2.25 * UPM], [3.0 * UPM, 2.25 * UPM]])
    south = PlanWall(id="W002", centreline=[[0.0, -0.125 * UPM], [3.0 * UPM, -0.125 * UPM]], thickness_m=0.25, kind="interior", evidence="x",
                     face_a=[[0.0, 0.0], [3.0 * UPM, 0.0]], face_b=[[0.0, -0.25 * UPM], [3.0 * UPM, -0.25 * UPM]])
    wc = _insert("WC_WALL", (1.5, 1.72), 270.0)
    _run(wc)
    ring = [[float(x), float(y)] for x, y in list(_footprint(wc).exterior.coords)[:-1]]
    item = PlanFurniture(id="F001", instance_id=None, label="WC", semantic_label="WC", catalogue_kind="WC_WALL", layer="P-SANI", footprint=ring, facing_deg=wc["_facing_deg"])
    _assign_back_walls([item], [south, north], UPM)
    assert item.back_wall_id == "W001"


def test_reversed_wc_overlapping_deep_into_the_wall_uses_the_room_side_face():
    # Front edge 0.16 m inside the 0.25 m north wall: the far wall face (y 2.25) is nearer than the room face (y 2.0 is 0.16 away).
    walls = _room()
    entity, _ = _run(_insert("WC_WALL", (1.5, 1.88), 90.0, jitter=(0.01, 0.008)), walls)
    _assert_back_on(entity, walls, 270.0)
    assert _footprint(entity).bounds[3] <= 2.0 * UPM + 0.004


# Real geometry of the v12 bathroom (plan units; wall faces measured, reversed WC / washbasin as the first placement produced them).
_BAD_WALLS = Polygon([(4.71, 14.34), (6.67, 14.36), (7.55, 14.36), (7.55, 14.65), (6.67, 14.64), (4.71, 14.63)])
_GAESTE_WALLS = Polygon([(-1.82, 15.25), (1.18, 15.27), (1.17, 15.55), (-1.82, 15.53)])


def _real_insert(kind, point, rotation):
    entry = CATALOGUE[kind][0]
    return {"type": "insert", "name": entry.block, "point": list(point), "rotation": rotation, "xscale": UPM, "yscale": UPM, "attributes": {},
            "_catalogue": {"kind": kind, "width_m": entry.width_m, "depth_m": entry.depth_m, "layer": "P-SANI", "wall_snapped": False}}


@pytest.mark.parametrize(("kind", "point"), [("WC_WALL", (6.17, 13.96)), ("WASHBASIN", (5.39, 13.96))])
def test_real_v12_bad_reversed_blocks_are_turned_onto_the_north_wall(kind, point):
    entity, _ = _run(_real_insert(kind, point, 0.4), _BAD_WALLS)
    assert _same_angle(entity["_facing_deg"], 270.4, 1.5)
    polygon = _footprint(entity)
    assert polygon.intersection(_BAD_WALLS).area < 1e-3 and polygon.distance(_BAD_WALLS) < 0.01
    assert polygon.bounds[3] < 14.37  # no longer reaches into the wall body


@pytest.mark.parametrize(("kind", "point"), [("WC_WALL", (0.44, 15.23)), ("WASHBASIN", (-0.38, 15.22))])
def test_real_v12_gaeste_wc_blocks_keep_facing_the_room(kind, point):
    entity, report = _run(_real_insert(kind, point, 180.4), _GAESTE_WALLS)
    assert _same_angle(entity["_facing_deg"], 270.4, 1.5) and report["sanitary_facing_flipped"] == 0
    assert _footprint(entity).distance(_GAESTE_WALLS) < 0.01
