"""Opening, window and door geometry."""

from __future__ import annotations

import pytest

from samcloud.bim.core.geometry import CORNER_ORDER, compute_scene_geometry
from samcloud.bim.core.model import (
    DoorElement,
    DoorOperation,
    DoorParams,
    Side,
    WallElement,
    WindowElement,
    WindowOperation,
    WindowParams,
)

from .conftest import add_opening, room_scene


def _south_wall(scene):  # type: ignore[no-untyped-def]
    return next(
        e
        for e in scene.elements.values()
        if isinstance(e, WallElement) and e.params.start == (0.0, 0.0)
    )


def test_opening_box_net_area_and_volume() -> None:
    scene = room_scene()
    wall = _south_wall(scene)
    opening = add_opening(scene, wall, position=2.0, width=1.0, height=2.0, sill=0.5)
    geometry = compute_scene_geometry(scene)
    og = geometry.openings[opening.id]
    # south wall: d = +x, n = +y, center justified, left face at y = +0.1, base z = 0.1
    assert og.origin == pytest.approx((1.5, 0.1, 0.6))
    assert og.u_axis == (1.0, 0.0, 0.0)
    assert og.w_axis == (-0.0, -1.0, 0.0)
    assert len(og.corners) == len(CORNER_ORDER) == 8
    assert og.corners[0] == pytest.approx((1.5, 0.1, 0.6))
    assert og.corners[1] == pytest.approx((2.5, 0.1, 0.6))
    assert og.corners[2] == pytest.approx((2.5, -0.1, 0.6))
    assert og.corners[4] == pytest.approx((1.5, 0.1, 2.6))
    assert og.corners[6] == pytest.approx((2.5, -0.1, 2.6))
    assert og.quantities["Width"] == pytest.approx(1.0)
    assert og.quantities["Height"] == pytest.approx(2.0)
    assert og.quantities["Area"] == pytest.approx(2.0)
    assert og.issues == ()
    wg = geometry.walls[wall.id]
    assert wg.gross_side_area == pytest.approx(12.5)
    assert wg.net_side_area == pytest.approx(10.5)
    assert wg.net_volume == pytest.approx(wg.gross_volume - 2.0 * 0.2)
    assert wg.opening_cuts[0].area == pytest.approx(2.0)


def test_overlapping_openings_are_counted_once() -> None:
    scene = room_scene()
    wall = _south_wall(scene)
    add_opening(scene, wall, position=2.0, width=1.0, height=1.0, sill=0.5)
    add_opening(scene, wall, position=2.5, width=1.0, height=1.0, sill=0.5)
    wg = compute_scene_geometry(scene).walls[wall.id]
    assert wg.net_side_area == pytest.approx(12.5 - 1.5)


def test_position_is_measured_from_untrimmed_start_with_left_justification() -> None:
    scene = room_scene()
    wall = _south_wall(scene)
    wall.params.justification = type(wall.params.justification)("left")
    opening = add_opening(scene, wall, position=1.0, width=0.5, height=1.0)
    og = compute_scene_geometry(scene).openings[opening.id]
    # left justification: axis is the left face, so the box origin lies on the axis
    assert og.origin[:2] == pytest.approx((0.75, 0.0))
    assert og.corners[2][:2] == pytest.approx((1.25, -0.2))


def test_opening_outside_wall_gives_issue() -> None:
    scene = room_scene()
    wall = _south_wall(scene)
    start = add_opening(scene, wall, position=0.3, width=1.0)
    end = add_opening(scene, wall, position=4.8, width=1.0)
    high = add_opening(scene, wall, position=2.5, width=1.0, height=2.0, sill=0.6)
    geometry = compute_scene_geometry(scene)
    assert "before the wall start" in geometry.openings[start.id].issues[0]
    assert "after the wall end" in geometry.openings[end.id].issues[0]
    assert "above the wall top" in geometry.openings[high.id].issues[0]
    assert start.id in geometry.issues and end.id in geometry.issues
    assert high.id in geometry.issues


def test_opening_in_trimmed_part_gives_issue() -> None:
    scene = room_scene()
    wall = _south_wall(scene)
    # the mitre cut leaves the axis untrimmed at 0, so use a T-trimmed wall instead
    from samcloud.bim.core.model import JoinEnd, JoinType

    from .conftest import make_join, make_wall

    path = make_wall((0.0, 3.0), (6.0, 3.0), 0.3)
    stub = make_wall((2.0, 0.0), (2.0, 3.0), 0.2)
    scene.add(path)
    scene.add(stub)
    scene.add(make_join(stub, JoinEnd.END, path, JoinEnd.PATH, JoinType.T))
    opening = add_opening(scene, stub, position=2.85, width=0.2, height=1.0)
    og = compute_scene_geometry(scene).openings[opening.id]
    assert og.issues == ("opening lies in a trimmed part of the wall (join cut)",)
    del wall


def test_missing_host_wall_is_reported() -> None:
    scene = room_scene()
    wall = _south_wall(scene)
    opening = add_opening(scene, wall, position=2.0)
    opening.params.host_wall_id = "missing"
    geometry = compute_scene_geometry(scene)
    assert opening.id not in geometry.openings
    assert "not found" in geometry.issues[opening.id][0]


def test_window_and_door_geometry_pass_through() -> None:
    scene = room_scene()
    wall = _south_wall(scene)
    opening_w = add_opening(scene, wall, position=1.2, width=1.0, height=1.2, sill=0.9)
    opening_d = add_opening(scene, wall, position=3.5, width=0.9, height=2.1)
    window = WindowElement(
        params=WindowParams(opening_id=opening_w.id, operation=WindowOperation.TILT_AND_TURN)
    )
    door = DoorElement(
        params=DoorParams(
            opening_id=opening_d.id,
            operation=DoorOperation.SINGLE_SWING_LEFT,
            swing_side=Side.LEFT,
        )
    )
    dangling = DoorElement(params=DoorParams(opening_id="nope"))
    for element in (window, door, dangling):
        scene.add(element)
    geometry = compute_scene_geometry(scene)
    wg = geometry.windows[window.id]
    dg = geometry.doors[door.id]
    assert wg.quantities == pytest.approx(
        {"Width": 1.0, "Height": 1.2, "Area": 1.2, "SillHeight": 0.9}
    )
    assert wg.operation == WindowOperation.TILT_AND_TURN
    assert wg.swing_side is None
    assert dg.quantities["Area"] == pytest.approx(0.9 * 2.1)
    assert dg.operation == DoorOperation.SINGLE_SWING_LEFT
    assert dg.swing_side == Side.LEFT
    assert dg.corners == geometry.openings[opening_d.id].corners
    assert "no geometry" in geometry.issues[dangling.id][0]
