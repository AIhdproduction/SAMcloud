"""Rooms go from door to door: door lines, passages, furniture, coverage (synthetic, scale-parametrised)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from samcloud.ortho.vector.door_rooms import DoorRoomSettings, door_bounded_rooms, door_line, opening_slab
from samcloud.ortho.vector.plan_polish import PlanPolishSettings, polygon_mask, rooms_from_wall_cells
from samcloud.ortho.vector.raster_grid import RasterGrid

UPMS = (1.0, 0.8124)


def _s(geometry, upm: float):
    return affinity.scale(geometry, xfact=upm, yfact=upm, origin=(0.0, 0.0))


def _opening(kind: str, centre: tuple[float, float], angle: float, width: float, thickness: float, upm: float):
    return SimpleNamespace(kind=kind, center=(centre[0] * upm, centre[1] * upm), axis_angle=angle, width=width * upm, wall_thickness=thickness * upm)


def _grid(upm: float, extent: float = 12.0, pixel: float = 0.04) -> RasterGrid:
    size = int(round(extent * upm / (pixel * upm)))
    return RasterGrid(min_x=-1.0 * upm, max_y=7.0 * upm, pixel_size=pixel * upm, width=size, height=size)


def _floor(rooms_m: list[Polygon], holes_m: list[Polygon], upm: float, grid: RasterGrid, margin: float = 0.15) -> np.ndarray:
    """Free-floor mask: the rooms eroded by ``margin`` from every wall, minus furniture (inflated, as the real mask keeps clear of it)."""
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for room in rooms_m:
        mask |= polygon_mask(_s(room.buffer(-margin), upm), grid)
    for hole in holes_m:
        mask &= ~polygon_mask(_s(hole.buffer(0.2), upm), grid)
    return mask


def _furniture_mask(holes_m: list[Polygon], upm: float, grid: RasterGrid) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for hole in holes_m:
        mask |= polygon_mask(_s(hole, upm), grid)
    return mask


def _flat_with_hall(kind: str | tuple[str, str], upm: float):
    """9 x 6 m interior, partitions at x = 3 and x = 6 (0.2 thick) with a 0.9 m opening each at y = 3.

    Left room and right room are wall-enclosed cells (cores, ragged by a furniture notch); the middle hall is no core.
    """
    ring = unary_union([box(-0.2, -0.2, 9.2, 6.2).difference(box(0.0, 0.0, 9.0, 6.0))])
    partitions = [
        box(2.9, 0.0, 3.1, 2.55), box(2.9, 3.45, 3.1, 6.0),
        box(5.9, 0.0, 6.1, 2.55), box(5.9, 3.45, 6.1, 6.0),
    ]
    walls = [_s(item, upm) for item in [ring, *partitions]]
    kinds = (kind, kind) if isinstance(kind, str) else kind
    openings = [_opening(kinds[0], (3.0, 3.0), math.pi / 2, 0.9, 0.2, upm), _opening(kinds[1], (6.0, 3.0), math.pi / 2, 0.9, 0.2, upm)]
    left_core = box(0.05, 0.05, 2.9, 3.0)  # ragged: the cell reaches 6.0 in y, the measured room only 3.0
    right_core = box(6.1, 0.05, 8.95, 6.0).difference(box(8.0, 4.5, 8.95, 6.0))  # furniture notch
    cores = [_s(left_core, upm), _s(right_core, upm)]
    return walls, openings, cores


def _interior_area_m2(with_reveal: bool) -> float:
    area = 9.0 * 6.0 - 2 * (0.2 * 5.1)
    return area + (2 * 0.9 * 0.2 if with_reveal else 0.0)


def _areas(rooms: list[Polygon], upm: float) -> list[float]:
    return sorted(room.area / (upm * upm) for room in rooms)


@pytest.mark.parametrize("upm", UPMS)
def test_doors_split_at_the_door_line_and_every_square_metre_has_one_room(upm: float) -> None:
    walls, openings, cores = _flat_with_hall("DOOR", upm)
    grid = _grid(upm)
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6), box(6.1, 0, 9, 6)], [], upm, grid)
    rooms, stats = door_bounded_rooms(cores, walls, openings, floor, grid, upm)
    assert len(rooms) == 3  # left, right and the hall between the two doors
    areas = _areas(rooms, upm)
    # the whole interior (door reveals included) is covered exactly once
    assert sum(areas) == pytest.approx(_interior_area_m2(True), rel=0.02)
    union = unary_union(rooms)
    assert sum(room.area for room in rooms) == pytest.approx(union.area, rel=1e-6)  # no overlap
    # left room reaches the full wall cell (6 m high) although only 3 m of ceiling was measured
    left = min(rooms, key=lambda room: room.centroid.x)
    assert left.bounds[3] == pytest.approx(6.0 * upm, abs=0.06 * upm)
    assert stats["door_room_new"] == 1


@pytest.mark.parametrize("upm", UPMS)
def test_each_room_gets_half_of_the_door_reveal(upm: float) -> None:
    walls, openings, cores = _flat_with_hall("DOOR", upm)
    grid = _grid(upm)
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6), box(6.1, 0, 9, 6)], [], upm, grid)
    rooms, _ = door_bounded_rooms(cores, walls, openings, floor, grid, upm)
    left = min(rooms, key=lambda room: room.centroid.x)
    hall = [room for room in rooms if 3.0 * upm < room.centroid.x < 6.0 * upm][0]
    # the door line x = 3.0: the left room ends at the line in the doorway, the hall starts at it
    doorway = box(2.5 * upm, 2.6 * upm, 3.5 * upm, 3.4 * upm)
    assert left.intersection(doorway).bounds[2] == pytest.approx(3.0 * upm, abs=0.02 * upm)
    assert hall.intersection(doorway).bounds[0] == pytest.approx(3.0 * upm, abs=0.02 * upm)
    reveal = opening_slab(openings[0])
    assert left.intersection(reveal).area == pytest.approx(0.5 * reveal.area, rel=0.05)
    assert hall.intersection(reveal).area == pytest.approx(0.5 * reveal.area, rel=0.05)
    line = door_line(openings[0])
    assert line.length == pytest.approx(0.9 * upm)
    assert line.centroid.x == pytest.approx(3.0 * upm)


@pytest.mark.parametrize("upm", UPMS)
def test_passages_do_not_split(upm: float) -> None:
    grid = _grid(upm)
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6), box(6.1, 0, 9, 6)], [], upm, grid)
    # a passage between the left room and the hall, a door between the hall and the right room
    walls, openings, cores = _flat_with_hall(("passage", "DOOR"), upm)
    rooms, stats = door_bounded_rooms(cores, walls, openings, floor, grid, upm)
    assert stats["door_room_new"] == 0  # the hall is open to the left room: it joins it, never a room of its own
    assert len(rooms) == 2
    assert sorted(_areas(rooms, upm))[1] == pytest.approx(17.4 + 16.8 + 0.18 + 0.09, rel=0.02)  # left + hall + passage reveal + half the door reveal
    left = min(rooms, key=lambda room: room.centroid.x)
    assert left.contains(Point(4.5 * upm, 3.0 * upm))  # hall + left room are one room across the passage
    assert not left.intersects(Point(7.5 * upm, 3.0 * upm))
    # passages at both partitions: everything is open, one room
    walls, openings, cores = _flat_with_hall("passage", upm)
    rooms, _ = door_bounded_rooms(cores, walls, openings, floor, grid, upm)
    assert len(rooms) == 1
    assert sum(_areas(rooms, upm)) == pytest.approx(_interior_area_m2(True), rel=0.02)


@pytest.mark.parametrize("upm", UPMS)
def test_furniture_never_carves_or_splits_a_room(upm: float) -> None:
    walls, openings, cores = _flat_with_hall("DOOR", upm)
    grid = _grid(upm)
    holes = [box(3.6, 0.3, 5.0, 2.2), box(3.5, 4.0, 5.5, 5.8), box(7.0, 1.0, 8.0, 2.0)]  # a bed, a sofa and a table
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6), box(6.1, 0, 9, 6)], holes, upm, grid)
    rooms, _ = door_bounded_rooms(cores, walls, openings, floor, grid, upm, furniture=_furniture_mask(holes, upm, grid))
    assert len(rooms) == 3
    assert all(not room.interiors for room in rooms)  # no hole where furniture stands
    assert sum(_areas(rooms, upm)) == pytest.approx(_interior_area_m2(True), rel=0.03)
    hall = [room for room in rooms if 3.0 * upm < room.centroid.x < 6.0 * upm][0]
    for hole in holes[:2]:
        assert hall.contains(_s(hole, upm).buffer(-0.01 * upm))


@pytest.mark.parametrize("upm", UPMS)
def test_exterior_walls_without_gaps_keep_rooms_inside(upm: float) -> None:
    walls, openings, cores = _flat_with_hall("DOOR", upm)
    grid = _grid(upm)
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6), box(6.1, 0, 9, 6)], [], upm, grid)
    rooms, _ = door_bounded_rooms(cores, walls, openings, floor, grid, upm)
    inside = _s(box(0.0, 0.0, 9.0, 6.0), upm).buffer(0.01 * upm)
    for room in rooms:
        assert room.difference(inside).area < 1e-3 * upm * upm
    wall_union = unary_union(walls)
    assert unary_union(rooms).intersection(wall_union).area < 1e-3 * upm * upm


@pytest.mark.parametrize("upm", UPMS)
def test_window_span_is_a_wall_not_a_floor(upm: float) -> None:
    walls, openings, cores = _flat_with_hall("DOOR", upm)
    window = _opening("WINDOW", (1.5, -0.1), 0.0, 1.2, 0.2, upm)
    ring_gap = _s(box(0.9, -0.2, 2.1, 0.0), upm)  # the window span is cut out of the exterior wall body
    walls = [item.difference(ring_gap) if i == 0 else item for i, item in enumerate(walls)]
    grid = _grid(upm)
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6), box(6.1, 0, 9, 6)], [], upm, grid)
    rooms, _ = door_bounded_rooms(cores, walls, [*openings, window], floor, grid, upm)
    assert unary_union(rooms).intersection(opening_slab(window)).area < 1e-3 * upm * upm


def test_disabled_returns_input_rooms() -> None:
    walls, openings, cores = _flat_with_hall("DOOR", 1.0)
    rooms, stats = door_bounded_rooms(cores, walls, openings, None, None, 1.0, DoorRoomSettings(enabled=False))
    assert rooms == cores and stats["door_room_new"] == 0


@pytest.mark.parametrize("upm", UPMS)
def test_a_furniture_blob_seen_through_a_window_is_not_interior(upm: float) -> None:
    walls, openings, cores = _flat_with_hall("DOOR", upm)
    grid = _grid(upm)
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6), box(6.1, 0, 9, 6)], [], upm, grid)
    outside = [box(1.0, 6.9, 3.0, 7.0)]  # a parked object far beyond the north wall
    inside = [box(3.5, 0.2, 4.5, 1.2)]
    rooms, _ = door_bounded_rooms(cores, walls, openings, floor, grid, upm, furniture=_furniture_mask(outside + inside, upm, grid))
    union = unary_union(rooms)
    assert not union.intersects(_s(outside[0], upm))
    assert union.contains(_s(inside[0], upm).buffer(-0.01 * upm))


@pytest.mark.parametrize("upm", UPMS)
def test_floor_does_not_grow_into_the_void_where_a_wall_is_missing(upm: float) -> None:
    walls, openings, cores = _flat_with_hall("DOOR", upm)
    gap = _s(box(3.2, 5.9, 5.8, 6.3), upm)  # the north wall above the hall is not measured
    walls = [item.difference(gap) for item in walls]
    grid = _grid(upm)
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6), box(6.1, 0, 9, 6)], [], upm, grid)
    rooms, _ = door_bounded_rooms(cores, walls, openings, floor, grid, upm)
    hall = [room for room in rooms if 3.0 * upm < room.centroid.x < 6.0 * upm][0]
    top = hall.intersection(LineString([(4.5 * upm, 0.0), (4.5 * upm, 7.0 * upm)])).bounds[3]
    assert top <= 5.87 * upm  # mid-gap, far from any wall: the floor stays at its evidence (free-floor margin 0.15 m)
    assert hall.bounds[3] <= (5.85 + 0.30 + 0.02) * upm  # beside the wall ends it grows at most floor_grow_m


@pytest.mark.parametrize("upm", UPMS)
def test_a_wall_cell_with_two_measured_rooms_is_one_room_without_a_door(upm: float) -> None:
    """The shipped config sets room_split_max_m = 0: only walls and door lines split, a ceiling-room step does not."""
    import yaml
    from pathlib import Path

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "ortho_plan.yaml").read_text(encoding="utf-8"))
    polish = PlanPolishSettings(room_split_max_m=float(config["plan_polish"]["room_split_max_m"]))
    assert polish.room_split_max_m == 0.0
    ring = box(-0.2, -0.2, 9.2, 6.2).difference(box(0.0, 0.0, 9.0, 6.0))
    walls = [_s(ring, upm)]
    kitchen, living = _s(box(0.05, 0.05, 4.5, 5.95), upm), _s(box(4.5, 0.05, 8.95, 5.95), upm)  # two measured ceiling rooms, no wall between
    rooms, _ = rooms_from_wall_cells([kitchen, living], walls, [], None, None, upm, 0.0, polish)
    assert len(rooms) == 1 and rooms[0].area / (upm * upm) == pytest.approx(54.0, rel=0.01)
    # a partition with a door opening: two cells, and the door line keeps them two rooms
    partition = [_s(box(4.4, 0.0, 4.6, 2.55), upm), _s(box(4.4, 3.45, 4.6, 6.0), upm)]
    door = _opening("DOOR", (4.5, 3.0), math.pi / 2, 0.9, 0.2, upm)
    barrier = opening_slab(door).buffer(0.01 * upm, join_style=2)  # the real cut shapes reach into the walls
    cells, _ = rooms_from_wall_cells([kitchen, living], walls + partition, [barrier], None, None, upm, 0.0, polish)
    assert len(cells) == 2
    done, stats = door_bounded_rooms(cells, walls + partition, [door], _floor([box(0, 0, 9, 6)], [], upm, _grid(upm)), _grid(upm), upm)
    assert len(done) == 2 and sum(room.area for room in done) / (upm * upm) == pytest.approx(9.0 * 6.0 - 0.2 * 5.1 + 0.18, rel=0.02)


def _two_measured_rooms(upm: float, gap_m: float | None):
    """9 x 6 m interior; two measured ceiling rooms (left, right) on either side of x = 4.5.  ``gap_m``: length of the
    opening in a 0.2 m partition (None: no partition at all); the rest of the partition is wall."""
    ring = box(-0.2, -0.2, 9.2, 6.2).difference(box(0.0, 0.0, 9.0, 6.0))
    bodies = [ring]
    if gap_m is not None:
        low, high = 3.0 - gap_m / 2.0, 3.0 + gap_m / 2.0
        bodies += [box(4.4, 0.0, 4.6, low), box(4.4, high, 4.6, 6.0)]
    cores = [box(0.05, 0.05, 4.4 if gap_m is not None else 4.5, 5.95), box(4.6 if gap_m is not None else 4.5, 0.05, 8.95, 5.95)]
    return [_s(item, upm) for item in bodies], [_s(item, upm) for item in cores]


@pytest.mark.parametrize("upm", UPMS)
def test_rooms_open_to_each_other_along_a_wide_boundary_are_one_room(upm: float) -> None:
    grid = _grid(upm)
    floor = _floor([box(0, 0, 9, 6)], [], upm, grid)
    walls, cores = _two_measured_rooms(upm, None)  # no wall at all between the two measured rooms
    rooms, stats = door_bounded_rooms(cores, walls, [], floor, grid, upm)
    assert len(rooms) == 1 and stats["door_room_open_merged"] == 1
    assert rooms[0].area / (upm * upm) == pytest.approx(54.0, rel=0.02)
    walls, cores = _two_measured_rooms(upm, 2.5)  # a 2.5 m wide opening in the partition, no door: open
    rooms, stats = door_bounded_rooms(cores, walls, [], floor, grid, upm)
    assert len(rooms) == 1


@pytest.mark.parametrize("upm", UPMS)
def test_a_short_open_stretch_is_a_wall_gap_and_a_door_keeps_the_rooms_apart(upm: float) -> None:
    grid = _grid(upm)
    floor = _floor([box(0, 0, 9, 6)], [], upm, grid)
    walls, cores = _two_measured_rooms(upm, 0.6)  # 0.6 m gap: a missing wall piece, narrower than any door
    rooms, stats = door_bounded_rooms(cores, walls, [], floor, grid, upm)
    assert len(rooms) == 2 and stats["door_room_open_merged"] == 0
    walls, cores = _two_measured_rooms(upm, 1.4)  # wide opening, but a door stands in it: the door line separates
    door = _opening("DOOR", (4.5, 3.0), math.pi / 2, 1.4, 0.2, upm)
    rooms, stats = door_bounded_rooms(cores, walls, [door], floor, grid, upm)
    assert len(rooms) == 2 and stats["door_room_open_merged"] == 0
    assert sum(room.area for room in rooms) / (upm * upm) == pytest.approx(9.0 * 6.0 - 0.2 * 4.6 + 0.28, rel=0.02)


@pytest.mark.parametrize("upm", UPMS)
def test_unmeasured_ground_outside_the_wall_line_never_becomes_room_floor(upm: float) -> None:
    walls, openings, cores = _flat_with_hall("DOOR", upm)
    gap = _s(box(3.2, 5.9, 5.8, 6.3), upm)  # north wall above the hall not measured
    walls = [item.difference(gap) for item in walls]
    grid = _grid(upm)
    # floor classified outside the building (seen through the gap): reachable by the free floor, but nothing measured around it
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6.6), box(6.1, 0, 9, 6)], [], upm, grid)
    flooded, _ = door_bounded_rooms(cores, walls, openings, floor, grid, upm)
    hall_before = [room for room in flooded if 3.0 * upm < room.centroid.x < 6.0 * upm][0]
    assert hall_before.bounds[3] > 6.3 * upm  # without measured support the hall floods out of the gap
    support = polygon_mask(_s(box(-0.3, -0.3, 9.3, 6.05), upm), grid)  # points exist inside the building and the wall tops only
    rooms, _ = door_bounded_rooms(cores, walls, openings, floor, grid, upm, support=support)
    hall = [room for room in rooms if 3.0 * upm < room.centroid.x < 6.0 * upm][0]
    assert hall.bounds[3] <= 6.0 * upm + 0.35 * upm  # bounded by the measured ground
    assert hall.area < hall_before.area - 0.5 * upm * upm
    # supported interior is untouched: same areas for the side rooms
    left = min(rooms, key=lambda room: room.centroid.x)
    assert left.area / (upm * upm) == pytest.approx(min(flooded, key=lambda room: room.centroid.x).area / (upm * upm), rel=0.01)


def _piece(x0: float, y0: float, x1: float, y1: float, thickness: float, upm: float, kind: str = "exterior") -> dict:
    """A wall piece in the shape of ``DoubleWallResult.pieces``: inner face (x0, y0)->(x1, y1), outer face ``thickness`` further out (+y)."""
    return {"kind": kind, "face": [[x0 * upm, y0 * upm], [x1 * upm, y1 * upm]], "face_b": [[x0 * upm, (y0 + thickness) * upm], [x1 * upm, (y1 + thickness) * upm]]}


@pytest.mark.parametrize("upm", UPMS)
def test_missing_exterior_wall_stretch_is_bridged_along_the_wall_line(upm: float) -> None:
    walls, openings, cores = _flat_with_hall("DOOR", upm)
    gap = _s(box(3.2, 5.9, 5.8, 6.3), upm)  # 2.6 m of the north wall above the hall not measured
    walls = [item.difference(gap) for item in walls]
    pieces = [_piece(-0.2, 6.0, 3.2, 6.0, 0.2, upm), _piece(5.8, 6.0, 9.2, 6.0, 0.2, upm)]
    grid = _grid(upm)
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6), box(6.1, 0, 9, 6)], [], upm, grid)
    plain, _ = door_bounded_rooms(cores, walls, openings, floor, grid, upm)
    bridged, stats = door_bounded_rooms(cores, walls, openings, floor, grid, upm, wall_pieces=pieces)
    hall_plain = [room for room in plain if 3.0 * upm < room.centroid.x < 6.0 * upm][0]
    hall = [room for room in bridged if 3.0 * upm < room.centroid.x < 6.0 * upm][0]
    assert stats["door_room_exterior_gaps"] >= 1
    assert hall.bounds[3] <= 6.0 * upm + 0.02 * upm  # the bridge continues the wall: the hall ends at the wall line
    assert hall_plain.bounds[3] > 6.05 * upm  # without the bridge the growth leaves through the gap
    # a gap that holds a door span is an opening, not a missing wall: it stays open
    door = _opening("DOOR", (4.5, 6.1), 0.0, 1.0, 0.2, upm)
    _, stats_door = door_bounded_rooms(cores, walls, [*openings, door], floor, grid, upm, wall_pieces=pieces)
    assert stats_door["door_room_exterior_gaps"] == 0 < stats["door_room_exterior_gaps"]


@pytest.mark.parametrize("upm", UPMS)
def test_a_measured_room_across_a_doorway_is_cut_at_the_door_line(upm: float) -> None:
    ring = box(-0.2, -0.2, 9.2, 6.2).difference(box(0.0, 0.0, 9.0, 6.0))
    partitions = [box(4.4, 0.0, 4.6, 2.55), box(4.4, 3.45, 4.6, 6.0)]
    walls = [_s(item, upm) for item in (ring, *partitions)]
    door = _opening("DOOR", (4.5, 3.0), math.pi / 2, 0.9, 0.2, upm)
    interior = box(0.05, 0.05, 8.95, 5.95).difference(unary_union(partitions))  # one ceiling room reaching across the doorway
    grid = _grid(upm)
    floor = _floor([box(0, 0, 9, 6)], [], upm, grid)
    rooms, stats = door_bounded_rooms([_s(interior, upm)], walls, [door], floor, grid, upm)
    assert len(rooms) == 2 and stats["door_room_cores_cut"] >= 1
    left, right = sorted(rooms, key=lambda room: room.centroid.x)
    assert left.bounds[2] == pytest.approx(4.5 * upm, abs=0.02 * upm) or left.bounds[2] <= 4.6 * upm
    assert left.area / (upm * upm) == pytest.approx(right.area / (upm * upm), rel=0.05)
    # the same room without the door (a passage) stays one room
    passage = _opening("passage", (4.5, 3.0), math.pi / 2, 0.9, 0.2, upm)
    one, _ = door_bounded_rooms([_s(interior, upm)], walls, [passage], floor, grid, upm)
    assert len(one) == 1


@pytest.mark.parametrize("upm", UPMS)
def test_a_measured_room_is_cut_back_to_the_measured_ground(upm: float) -> None:
    walls, openings, cores = _flat_with_hall("DOOR", upm)
    grid = _grid(upm)
    floor = _floor([box(0, 0, 2.9, 6), box(3.1, 0, 5.9, 6), box(6.1, 0, 9, 6)], [], upm, grid)
    gap = _s(box(6.2, 5.9, 8.8, 6.3), upm)  # north wall above the right room not measured
    walls = [item.difference(gap) for item in walls]
    spill = [cores[0], _s(box(6.1, 0.05, 8.95, 7.2), upm)]  # the right ceiling room reaches 1.2 m beyond the building (unmeasured)
    support = polygon_mask(_s(box(-0.3, -0.3, 9.3, 6.05), upm), grid)
    plain, _ = door_bounded_rooms(spill, walls, openings, floor, grid, upm)
    cut, _ = door_bounded_rooms(spill, walls, openings, floor, grid, upm, support=support)
    right_plain = max(plain, key=lambda room: room.centroid.x)
    right = max(cut, key=lambda room: room.centroid.x)
    assert right_plain.bounds[3] >= 7.0 * upm  # without support the room keeps its spill
    assert right.bounds[3] <= 6.0 * upm + 0.35 * upm
    assert right.area < right_plain.area - 2.0 * upm * upm




@pytest.mark.parametrize("upm", UPMS)
def test_a_passage_opening_between_two_wall_cells_makes_them_one_room_a_door_keeps_them_two(upm: float) -> None:
    grid = _grid(upm)
    floor = _floor([box(0, 0, 9, 6)], [], upm, grid)
    for kind, expected in (("passage", 1), ("DOOR", 2)):
        walls, cores = _two_measured_rooms(upm, 0.9)  # a 0.9 m opening in the partition (shorter than the 1.0 m open-boundary merge)
        opening = _opening(kind, (4.5, 3.0), math.pi / 2, 0.9, 0.2, upm)
        rooms, _ = door_bounded_rooms(cores, walls, [opening], floor, grid, upm)
        assert len(rooms) == expected, kind
    # no opening recorded at all: a 0.9 m gap is a missing wall piece, not an opening: two rooms
    walls, cores = _two_measured_rooms(upm, 0.9)
    assert len(door_bounded_rooms(cores, walls, [], floor, grid, upm)[0]) == 2


def _flat_with_closet(upm: float):
    """9 x 6 m interior, two measured cells (left / right of x = 4.5) with NO wall between them, and a closet in the top middle
    (x 4.0 .. 5.0, y 4.6 .. 6.0) whose door (0.8 m, in its bottom wall at y = 4.5) straddles the open boundary x = 4.5."""
    ring = box(-0.2, -0.2, 9.2, 6.2).difference(box(0.0, 0.0, 9.0, 6.0))
    closet_walls = [box(3.9, 4.4, 4.1, 4.6), box(4.9, 4.4, 5.1, 4.6), box(3.9, 4.6, 4.0, 6.0), box(5.0, 4.6, 5.1, 6.0)]
    walls = [_s(item, upm) for item in [ring, *closet_walls]]
    left = box(0.05, 0.05, 4.5, 5.95).difference(box(3.9, 4.4, 4.5, 6.0))
    right = box(4.5, 0.05, 8.95, 5.95).difference(box(4.5, 4.4, 5.1, 6.0))
    closet = box(4.0, 4.6, 5.0, 5.95)
    door = _opening("DOOR", (4.5, 4.5), 0.0, 0.8, 0.2, upm)
    return walls, [_s(left, upm), _s(right, upm), _s(closet, upm)], door


@pytest.mark.parametrize("upm", UPMS)
def test_a_door_of_a_third_room_on_the_open_boundary_does_not_keep_two_open_rooms_apart(upm: float) -> None:
    """Only a door that connects the two rooms (one at each side of it) separates them; the closet door touches both from one side."""
    grid = _grid(upm)
    walls, cores, door = _flat_with_closet(upm)
    floor = _floor([box(0, 0, 9, 6)], [], upm, grid)
    rooms, stats = door_bounded_rooms(cores, walls, [door], floor, grid, upm)
    assert stats["door_room_open_merged"] == 1
    assert len(rooms) == 2, [room.bounds for room in rooms]  # one open room (left + right) and the closet
    closet = min(rooms, key=lambda room: room.area)
    big = max(rooms, key=lambda room: room.area)
    assert big.contains(Point(2.0 * upm, 2.0 * upm)) and big.contains(Point(7.0 * upm, 2.0 * upm))
    assert closet.contains(Point(4.5 * upm, 5.3 * upm)) and not closet.intersects(Point(2.0 * upm, 2.0 * upm))


@pytest.mark.parametrize("upm", UPMS)
def test_a_door_between_two_rooms_still_separates_them_when_a_third_door_touches_the_boundary(upm: float) -> None:
    grid = _grid(upm)
    walls, cores, door = _flat_with_closet(upm)
    # a partition at x = 4.5 below the closet with its own door: the open boundary is shorter than any door and carries a door
    partition = [_s(box(4.4, 0.0, 4.6, 1.55), upm), _s(box(4.4, 2.45, 4.6, 4.4), upm)]
    between = _opening("DOOR", (4.5, 2.0), math.pi / 2, 0.9, 0.2, upm)
    floor = _floor([box(0, 0, 9, 6)], [], upm, grid)
    rooms, stats = door_bounded_rooms(cores, walls + partition, [door, between], floor, grid, upm)
    assert stats["door_room_open_merged"] == 0
    assert len(rooms) == 3


@pytest.mark.parametrize("upm", UPMS)
def test_floor_hidden_beside_a_wall_still_belongs_to_the_room_up_to_the_wall_face(upm: float) -> None:
    """Furniture hides the floor along a partition (no measured support for 0.4 m): both rooms still reach the wall faces."""
    grid = _grid(upm)
    ring = box(-0.2, -0.2, 9.2, 6.2).difference(box(0.0, 0.0, 9.0, 6.0))
    partition = [box(4.4, 0.0, 4.6, 2.55), box(4.4, 3.45, 4.6, 6.0)]
    walls = [_s(item, upm) for item in [ring, *partition]]
    door = _opening("DOOR", (4.5, 3.0), math.pi / 2, 0.9, 0.2, upm)
    cores = [_s(box(0.05, 0.05, 4.0, 5.95), upm), _s(box(5.0, 0.05, 8.95, 5.95), upm)]  # the measured rooms stop 0.4 m before the partition
    floor = _floor([box(0, 0, 4.0, 6), box(5.0, 0, 9, 6)], [], upm, grid)
    support = _floor([box(0, 0, 3.8, 6), box(5.2, 0, 9, 6)], [], upm, grid)  # measured ground ends 0.2 m before the rooms do (furniture hides the strip)
    rooms, _ = door_bounded_rooms(cores, walls, [door], floor, grid, upm, support=support)
    assert len(rooms) == 2
    left = min(rooms, key=lambda room: room.centroid.x)
    right = max(rooms, key=lambda room: room.centroid.x)
    assert left.bounds[2] >= 4.4 * upm - 0.03 * upm and right.bounds[0] <= 4.6 * upm + 0.03 * upm
    assert left.area / (upm * upm) == pytest.approx(4.4 * 5.9, rel=0.05)
    assert right.area / (upm * upm) == pytest.approx(4.4 * 5.9, rel=0.05)


@pytest.mark.parametrize("upm", UPMS)
def test_a_corner_hidden_behind_a_tall_unit_is_closed_to_the_wall_faces(upm: float) -> None:
    """A 0.8 x 0.8 m corner pocket (no measured floor, no support) bounded by two walls and one room belongs to that room."""
    grid = _grid(upm)
    ring = box(-0.2, -0.2, 9.2, 6.2).difference(box(0.0, 0.0, 9.0, 6.0))
    walls = [_s(ring, upm)]
    corner = box(8.2, 0.0, 9.0, 0.8)
    core = _s(box(0.05, 0.05, 8.95, 5.95).difference(corner), upm)
    floor = _floor([box(0, 0, 9, 6).difference(corner)], [], upm, grid)
    support = _floor([box(0, 0, 9, 6).difference(corner.buffer(0.1, join_style=2))], [], upm, grid)
    settings = DoorRoomSettings()
    rooms, stats = door_bounded_rooms([core], walls, [], floor, grid, upm, settings, support=support)
    assert len(rooms) == 1
    assert rooms[0].area / (upm * upm) == pytest.approx(54.0, rel=0.01)
    assert rooms[0].contains(Point(8.9 * upm, 0.1 * upm))  # the pocket itself
