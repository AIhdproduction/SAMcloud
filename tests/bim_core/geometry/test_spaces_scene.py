"""Spaces, scene-level geometry, determinism and large-coordinate precision."""

from __future__ import annotations

import pytest

from samcloud.bim.core.geometry import (
    compute_scene_geometry,
    derive_storey_rooms,
)
from samcloud.bim.core.model import Scene, SpaceElement, SpaceParams, WallElement
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene

from .conftest import add_opening, room_scene


def test_space_polygon_area_height_volume() -> None:
    scene = room_scene()
    geometry = compute_scene_geometry(scene)
    space = next(iter(geometry.spaces.values()))
    assert space.matched_by == "single_room"
    assert space.gross_area == pytest.approx(4.8 * 3.8, abs=1e-9)
    assert space.net_area == pytest.approx(18.24, abs=1e-9)
    assert space.perimeter == pytest.approx(17.2, abs=1e-9)
    # floor top 0.1, ceiling underside 2.9
    assert space.floor_z == pytest.approx(0.1)
    assert space.ceiling_z == pytest.approx(2.9)
    assert space.height == pytest.approx(2.8)
    assert space.height_source == "ceiling"
    q = space.quantities
    assert q["Height"] == pytest.approx(2.8)
    assert q["GrossFloorArea"] == pytest.approx(18.24, abs=1e-9)
    assert q["NetFloorArea"] == pytest.approx(18.24, abs=1e-9)
    assert q["GrossPerimeter"] == pytest.approx(17.2, abs=1e-9)
    assert q["GrossVolume"] == pytest.approx(18.24 * 2.8, abs=1e-8)
    assert q["NetVolume"] == pytest.approx(18.24 * 2.8, abs=1e-8)
    assert len(space.bounding_wall_ids) == 4
    assert len(space.floor_ids) == 1
    assert len(space.ceiling_ids) == 1
    assert space.issues == ()


def test_space_height_fallbacks() -> None:
    scene = room_scene()
    for element in list(scene.elements.values()):
        if element.kind.value == "ceiling":
            del scene.elements[element.id]
    space = next(iter(compute_scene_geometry(scene).spaces.values()))
    assert space.height_source == "storey_height"
    assert space.height == pytest.approx(2.7)
    assert "storey_height" in space.issues[0]
    for storey in scene.storeys.values():
        storey.height = None
    space = next(iter(compute_scene_geometry(scene).spaces.values()))
    assert space.height_source == "wall_height"
    assert space.height == pytest.approx(2.5)


def test_derive_storey_rooms_lists_polygons_with_bounding_elements() -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))
    geometry = compute_scene_geometry(scene)
    storey_id = next(iter(scene.storeys))
    rooms = derive_storey_rooms(
        storey_id,
        list(geometry.walls.values()),
        list(geometry.slabs.values()),
        list(geometry.ceilings.values()),
        geometry.origin,
    )
    assert len(rooms) == 2
    # exterior 0.30, partition 0.12: each room is (4 - 0.15 - 0.06) x (5 - 0.30)
    for room in rooms:
        assert room.net_area == pytest.approx(3.79 * 4.7, abs=1e-9)
        assert len(room.bounding_wall_ids) == 4
        assert len(room.floor_ids) == 1
        assert len(room.ceiling_ids) == 1
    assert rooms[0].representative_point[0] < rooms[1].representative_point[0]
    partition = next(
        e.id for e in scene.elements.values() if isinstance(e, WallElement) and e.tag == "partition"
    )
    assert all(partition in room.bounding_wall_ids for room in rooms)


def _strip_assignment_hints(scene: Scene) -> list[SpaceElement]:
    """Remove seeds and wall space_ids so that only the ordering fallback is left."""
    spaces = [e for e in scene.elements.values() if isinstance(e, SpaceElement)]
    for space in spaces:
        space.params.seed = None
    for element in scene.elements.values():
        element.space_ids = []
    return spaces


def test_synthetic_two_rooms_use_their_seed_without_issues() -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))
    spaces = [e for e in scene.elements.values() if isinstance(e, SpaceElement)]
    geometry = compute_scene_geometry(scene)
    assert all(s.id not in geometry.issues for s in spaces)
    assert all(geometry.spaces[s.id].matched_by == "seed" for s in spaces)
    assert geometry.spaces[spaces[0].id].exterior[0][0] < 4.0  # room 1 is the west room
    assert geometry.spaces[spaces[1].id].exterior[0][0] > 4.0


def test_two_rooms_without_hints_is_flagged_ambiguous_and_seed_resolves_it() -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))
    spaces = _strip_assignment_hints(scene)
    geometry = compute_scene_geometry(scene)
    assert all("ambiguous" in geometry.issues[s.id][0] for s in spaces)
    assert geometry.spaces[spaces[0].id].exterior[0][0] < 4.0  # room 1 is the west room
    # explicit seeds in the project frame: west room gets room 2 on purpose
    seeds = {spaces[0].id: (6.0, 2.5), spaces[1].id: (2.0, 2.5)}
    seeded = compute_scene_geometry(scene, seeds=seeds)
    assert all(s.id not in seeded.issues for s in spaces)
    assert seeded.spaces[spaces[0].id].matched_by == "seed"
    assert seeded.spaces[spaces[0].id].exterior[0][0] > 4.0
    bad = compute_scene_geometry(scene, seeds={spaces[0].id: (50.0, 50.0)})
    assert "seed lies in no room polygon" in bad.issues[spaces[0].id][0]


def test_space_param_seed_is_used_and_explicit_mapping_wins() -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))
    spaces = _strip_assignment_hints(scene)
    spaces[0].params.seed = (6.0, 2.5)
    spaces[1].params.seed = (2.0, 2.5)
    geometry = compute_scene_geometry(scene)
    assert all(s.id not in geometry.issues for s in spaces)
    assert geometry.spaces[spaces[0].id].matched_by == "seed"
    assert geometry.spaces[spaces[0].id].exterior[0][0] > 4.0
    override = compute_scene_geometry(scene, seeds={spaces[0].id: (2.0, 2.5)})
    assert override.spaces[spaces[0].id].exterior[0][0] < 4.0
    bad = compute_scene_geometry(scene, seeds={spaces[0].id: (50.0, 50.0)})
    assert "seed lies in no room polygon" in bad.issues[spaces[0].id][0]


def test_space_by_bounding_walls() -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))
    spaces = _strip_assignment_hints(scene)
    east_walls = [
        e
        for e in scene.elements.values()
        if isinstance(e, WallElement) and e.tag in ("exterior_east", "partition")
    ]
    for wall in east_walls:
        wall.space_ids = [spaces[0].id]
    geometry = compute_scene_geometry(scene)
    assert geometry.spaces[spaces[0].id].matched_by == "bounding_walls"
    assert geometry.spaces[spaces[0].id].exterior[0][0] > 4.0
    # second space has no information and one free polygon -> single_room
    assert geometry.spaces[spaces[1].id].matched_by == "single_room"


def test_more_spaces_than_rooms_reports_missing_geometry() -> None:
    scene = room_scene()
    extra = SpaceElement(params=SpaceParams(name="Extra", number="2"), storey_id=None)
    extra.storey_id = next(iter(scene.storeys))
    scene.add(extra)
    geometry = compute_scene_geometry(scene)
    assert extra.id not in geometry.spaces
    assert "no free room polygon" in geometry.issues[extra.id][0]


def test_scene_quantities_and_issues_cover_all_elements() -> None:
    scene = room_scene()
    wall = next(e for e in scene.elements.values() if isinstance(e, WallElement))
    add_opening(scene, wall, position=2.0)
    geometry = compute_scene_geometry(scene)
    quantities = geometry.quantities
    expected = {e.id for e in scene.elements.values() if e.kind.value != "wall_join"}
    assert set(quantities) == expected
    assert geometry.quantities_of("unknown") == {}
    assert geometry.geometry_of(wall.id) is geometry.walls[wall.id]
    assert set(quantities[wall.id]) == {
        "Length",
        "Width",
        "Height",
        "GrossSideArea",
        "NetSideArea",
        "GrossVolume",
        "NetVolume",
        "GrossFootprintArea",
    }


def test_synth_room_rectangle_values() -> None:
    scene = build_room_scene()
    geometry = compute_scene_geometry(scene)
    assert not geometry.issues
    space = next(iter(geometry.spaces.values()))
    assert space.net_area == pytest.approx(4.7 * 3.7, abs=1e-9)
    assert space.height == pytest.approx(2.8)


def test_determinism() -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))
    assert compute_scene_geometry(scene) == compute_scene_geometry(scene)


def test_large_coordinates_keep_precision() -> None:
    ox, oy = 2_600_000.0, 1_200_000.0
    local = compute_scene_geometry(room_scene())
    big_scene = room_scene(ox, oy)
    big = compute_scene_geometry(big_scene)
    # the local scene has the same ids only structurally: compare by element order
    for (_, wl), (_, wb) in zip(local.walls.items(), big.walls.items(), strict=True):
        for key, value in wl.quantities.items():
            assert wb.quantities[key] == pytest.approx(value, abs=1e-9, rel=0)
        for (xl, yl), (xb, yb) in zip(wl.footprint, wb.footprint, strict=True):
            assert xb - ox == pytest.approx(xl, abs=1e-9, rel=0)
            assert yb - oy == pytest.approx(yl, abs=1e-9, rel=0)
    sl = next(iter(local.spaces.values()))
    sb = next(iter(big.spaces.values()))
    for key, value in sl.quantities.items():
        assert sb.quantities[key] == pytest.approx(value, abs=1e-9, rel=0)
    assert sb.net_area == pytest.approx(18.24, abs=1e-9)
    kl = next(iter(local.slabs.values()))
    kb = next(iter(big.slabs.values()))
    assert kb.quantities == pytest.approx(kl.quantities, abs=1e-9, rel=0)
