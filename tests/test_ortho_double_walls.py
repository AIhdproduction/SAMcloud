"""Double-line indoor walls (samcloud.ortho.vector.double_walls) on synthetic plans."""

from __future__ import annotations

import math
from dataclasses import replace

import ezdxf
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from samcloud.ortho.vector.double_walls import (
    DoubleWallSettings,
    WallEvidence,
    WallRun,
    _Face,
    _neighbour_slab,
    _wall_bodies,
    bound_rooms_by_walls,
    build_double_walls,
    update_opening_symbols,
)
from samcloud.ortho.vector.indoor import Opening, opening_entities
from samcloud.ortho.vector.raster_grid import RasterGrid

SETTINGS = DoubleWallSettings(double_line=True)


def _grid(bounds: tuple[float, float, float, float], pixel: float = 0.02) -> RasterGrid:
    min_x, min_y, max_x, max_y = bounds
    return RasterGrid(min_x=min_x, max_y=max_y, pixel_size=pixel, width=int(math.ceil((max_x - min_x) / pixel)), height=int(math.ceil((max_y - min_y) / pixel)))


def _mask(grid: RasterGrid, polygons: list[Polygon]) -> np.ndarray:
    rows, cols = np.mgrid[0:grid.height, 0:grid.width]
    xs = grid.min_x + (cols + 0.5) * grid.pixel_size
    ys = grid.max_y - (rows + 0.5) * grid.pixel_size
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for polygon in polygons:
        x0, y0, x1, y1 = polygon.bounds
        mask |= (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return mask


def _face(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"type": "polyline", "points": [[x0, y0], [x1, y1]], "closed": False, "_source": "measured"}


def _rect_faces(x0: float, y0: float, x1: float, y1: float) -> list[dict]:
    return [_face(x0, y0, x1, y0), _face(x1, y0, x1, y1), _face(x1, y1, x0, y1), _face(x0, y1, x0, y0)]


def _scaled(value: float, upm: float) -> float:
    return value * upm


def _two_rooms(upm: float = 1.0, wall: float = 0.15, occlude_b: bool = False):
    """Room A [0,4]x[0,3] and room B [4+wall,8]x[0,3]: one interior wall, exterior walls around."""
    s = upm
    room_a = box(0.0, 0.0, 4.0 * s, 3.0 * s)
    room_b = box((4.0 + wall) * s, 0.0, 8.0 * s, 3.0 * s)
    grid = _grid((-1.0 * s, -1.0 * s, 9.0 * s, 4.0 * s), 0.02 * s)
    free = _mask(grid, [room_a, room_b])
    # Measured data reaches 5 cm into the walls only; outside the building is void.
    valid = _mask(grid, [room_a.buffer(0.05 * s, join_style=2), room_b.buffer(0.05 * s, join_style=2)])
    faces = _rect_faces(0.0, 0.0, 4.0 * s, 3.0 * s)
    b_faces = _rect_faces((4.0 + wall) * s, 0.0, 8.0 * s, 3.0 * s)
    if occlude_b:
        # B's face on the shared wall is seen only in its lower half (furniture in front of the rest).
        b_faces[3] = _face((4.0 + wall) * s, 1.4 * s, (4.0 + wall) * s, 0.0)
    return faces + b_faces, [room_a, room_b], free, valid, grid


def _body(result) -> Polygon:
    return unary_union(result.polygons)


def test_shared_wall_gets_measured_thickness_and_two_faces() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    result = build_double_walls(entities, [], rooms, free, valid, grid, 1.0, SETTINGS)
    body = _body(result)
    assert all(entity.get("closed") and entity.get("_double_wall") for entity in result.entities)
    # The shared wall is solid between both measured faces, and only there.
    for y in (0.3, 1.5, 2.7):
        assert body.contains(Point(4.075, y))
        assert not body.contains(Point(3.95, y))
        assert not body.contains(Point(4.20, y))
    interior = [piece for piece in result.pieces if piece["kind"] == "interior"]
    assert interior and all(abs(piece["thickness"] - 0.15) <= 0.01 for piece in interior)
    assert result.stats["double_wall_measured_pairs"] >= 1
    # Both rooms are holes of the wall body bounded by the inner faces.
    holes = [Polygon(ring) for polygon in result.polygons for ring in polygon.interiors]
    assert len(holes) == 2
    assert any(hole.symmetric_difference(rooms[0]).area <= 0.02 for hole in holes)
    assert any(hole.symmetric_difference(rooms[1]).area <= 0.02 for hole in holes)


def test_exterior_walls_get_default_thickness_outwards() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    result = build_double_walls(entities, [], rooms, free, valid, grid, 1.0, replace(SETTINGS, exterior_thickness_m=0.30))
    body = _body(result)
    assert body.contains(Point(2.0, -0.15)) and not body.contains(Point(2.0, -0.35))
    assert body.contains(Point(-0.15, 1.5)) and not body.contains(Point(-0.35, 1.5))
    assert body.contains(Point(8.15, 1.5)) and not body.contains(Point(8.35, 1.5))
    exterior = [piece for piece in result.pieces if piece["kind"] == "exterior"]
    assert exterior and all(abs(piece["thickness"] - 0.30) <= 1e-6 for piece in exterior)
    assert result.stats["double_wall_exterior_measured"] is False


def test_l_corner_is_filled_and_outline_closed() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    result = build_double_walls(entities, [], rooms, free, valid, grid, 1.0, SETTINGS)
    body = _body(result)
    # Outer corner square of two exterior walls meeting at an L.
    for point in ((-0.15, -0.15), (8.15, 3.15), (-0.15, 3.15), (8.15, -0.15)):
        assert body.contains(Point(point))
    # One closed wall body (T joints of the interior wall with both exterior walls merge it).
    assert len(result.polygons) == 1
    outer = Polygon(result.polygons[0].exterior)
    assert abs(outer.area - (8.6 * 3.6)) <= 0.05


def test_occluded_face_takes_partner_thickness_along_whole_wall() -> None:
    entities, rooms, free, valid, grid = _two_rooms(occlude_b=True)
    # Furniture against the wall in room B: no free floor in the upper part next to the wall.
    free[_mask(grid, [box(4.15, 1.4, 4.75, 3.0)])] = False
    result = build_double_walls(entities, [], rooms, free, valid, grid, 1.0, SETTINGS)
    body = _body(result)
    for y in (0.5, 2.0, 2.8):
        assert body.contains(Point(4.075, y))
        assert not body.contains(Point(4.22, y))
    shared = [piece for piece in result.pieces if piece["kind"] == "interior" and abs(piece["face"][0][0] - 4.0) < 0.01]
    assert shared and all(abs(piece["thickness"] - 0.15) <= 0.01 for piece in shared)


def test_t_joint_has_no_gap_or_overshoot() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    result = build_double_walls(entities, [], rooms, free, valid, grid, 1.0, SETTINGS)
    body = _body(result)
    # Where the interior wall meets the exterior wall, the wall material is continuous ...
    assert body.contains(Point(4.075, -0.10)) and body.contains(Point(4.075, 3.10))
    # ... and nothing sticks out beyond the exterior face.
    assert not body.contains(Point(4.075, -0.35)) and not body.contains(Point(4.075, 3.35))
    # The inner faces meet exactly: the room holes are rectangles (4 corners each).
    for polygon in result.polygons:
        for ring in polygon.interiors:
            assert len(ring.coords) - 1 == 4


def test_door_cuts_both_faces_with_closed_jambs() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    door = Opening("DOOR", (4.0, 1.5), 0.9, math.pi / 2.0, 0.04, leaf_detected=False)
    # The face of room A is already cut at the door (as the plan does); room B's face is not.
    entities = [entity for entity in entities if not (entity["points"][0][0] == 4.0 and entity["points"][1][0] == 4.0)]
    entities += [_face(4.0, 0.0, 4.0, 1.05), _face(4.0, 1.95, 4.0, 3.0)]
    result = build_double_walls(entities, [door], rooms, free, valid, grid, 1.0, SETTINGS)
    body = _body(result)
    assert not body.contains(Point(4.075, 1.5)) and not body.intersects(LineString([(3.9, 1.5), (4.3, 1.5)]))
    assert body.contains(Point(4.075, 0.9)) and body.contains(Point(4.075, 2.1))
    low, high = result.opening_slabs[0]
    assert abs((high - low) - 0.15) <= 0.01
    # Jambs: one cap line per side of the door spanning the full thickness.
    jambs = [line for line in result.caps if abs(abs(line.coords[0][1] - 1.5) - 0.45) <= 0.01 and abs(line.coords[0][1] - line.coords[1][1]) <= 1e-6]
    assert len(jambs) == 2 and all(abs(line.length - 0.15) <= 0.01 for line in jambs)
    kind, span = result.opening_spans[0]
    assert kind == "DOOR" and abs(span.length - 0.9) <= 1e-6 and abs(span.coords[0][0] - 4.075) <= 0.01


def test_door_symbols_follow_the_measured_thickness() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    door = Opening("DOOR", (4.0, 1.5), 0.9, math.pi / 2.0, 0.04, leaf_detected=False)
    window = Opening("WINDOW", (2.0, 3.0), 1.2, 0.0, 0.25)
    result = build_double_walls(entities, [door, window], rooms, free, valid, grid, 1.0, SETTINGS)
    layers = {"A-DOOR": [], "A-WIND": []}
    for entity in opening_entities([door, window], free, grid=grid):
        layers["A-WIND" if entity.get("name") == "WINDOW" else "A-DOOR"].append(entity)
    assert update_opening_symbols(layers, [door, window], result.opening_slabs, 1.0) >= 3
    jambs = [entity for entity in layers["A-DOOR"] if entity.get("_jamb")]
    assert len(jambs) == 2
    for entity in jambs:
        xs = sorted((entity["start"][0], entity["end"][0]))
        assert abs(xs[0] - 4.0) <= 0.01 and abs(xs[1] - 4.15) <= 0.01
    window_insert = layers["A-WIND"][0]
    assert abs(float(window_insert["yscale"]) - 0.30) <= 0.01
    assert abs(float(window_insert["point"][1]) - 3.0) <= 0.01


@pytest.mark.parametrize("upm", [0.5, 2.5])
def test_unscaled_units_scale_every_distance(upm: float) -> None:
    entities, rooms, free, valid, grid = _two_rooms(upm=upm)
    result = build_double_walls(entities, [], rooms, free, valid, grid, upm, SETTINGS)
    body = _body(result)
    assert body.contains(Point(4.075 * upm, 1.5 * upm)) and not body.contains(Point(4.2 * upm, 1.5 * upm))
    assert body.contains(Point(-0.15 * upm, 1.5 * upm)) and not body.contains(Point(-0.35 * upm, 1.5 * upm))
    interior = [piece for piece in result.pieces if piece["kind"] == "interior"]
    assert interior and all(abs(piece["thickness"] - 0.15) <= 0.01 for piece in interior)


def test_thick_and_light_walls_are_learned_as_classes() -> None:
    # Three rooms in a row: a 0.25 m wall and a 0.10 m wall.
    room_a, room_b, room_c = box(0, 0, 3, 3), box(3.25, 0, 6, 3), box(6.10, 0, 9, 3)
    grid = _grid((-1.0, -1.0, 10.0, 4.0))
    free = _mask(grid, [room_a, room_b, room_c])
    valid = _mask(grid, [room.buffer(0.05, join_style=2) for room in (room_a, room_b, room_c)])
    entities = _rect_faces(0, 0, 3, 3) + _rect_faces(3.25, 0, 6, 3) + _rect_faces(6.10, 0, 9, 3)
    result = build_double_walls(entities, [], [room_a, room_b, room_c], free, valid, grid, 1.0, SETTINGS)
    classes = [value for value, _count in result.stats["double_wall_interior_classes_m"]]
    assert any(abs(value - 0.25) <= 0.01 for value in classes)
    assert any(abs(value - 0.10) <= 0.01 for value in classes)
    body = _body(result)
    assert body.contains(Point(3.12, 1.5)) and body.contains(Point(6.05, 1.5))
    assert not body.contains(Point(3.3, 1.5)) and not body.contains(Point(6.15, 1.5))


def test_isolated_short_face_is_not_a_wall() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    stub = _face(2.0, 1.0, 2.0, 1.6)
    result = build_double_walls(entities + [stub], [], rooms, free, valid, grid, 1.0, SETTINGS)
    assert result.stats["double_wall_isolated_faces"] == 1
    assert not _body(result).contains(Point(2.0, 1.3))
    assert all(entity.get("closed") for entity in result.entities)
    kept = build_double_walls(entities + [stub], [], rooms, free, valid, grid, 1.0, replace(SETTINGS, keep_isolated_faces=True))
    assert any(not entity.get("closed") and entity["points"][0] == [2.0, 1.0] for entity in kept.entities)


def test_wall_hidden_behind_kitchen_units_is_closed_at_the_data_end() -> None:
    # One room [0,4]x[0,3]; along the north side kitchen units (measured, not floor) reach
    # 0.30 m beyond the ceiling room edge, and no wall face was measured there.
    room = box(0.0, 0.0, 4.0, 3.0)
    grid = _grid((-1.0, -1.0, 5.0, 4.5))
    free = _mask(grid, [box(0.0, 0.0, 4.0, 2.4)])
    valid = _mask(grid, [box(-0.05, -0.05, 4.05, 3.30)])
    # The side walls are measured up to the hidden wall; only a 1 m piece of the north face is seen.
    entities = [_face(0.0, 0.0, 4.0, 0.0), _face(4.0, 0.0, 4.0, 3.3), _face(0.0, 3.3, 0.0, 0.0), _face(0.0, 3.3, 1.0, 3.3)]
    result = build_double_walls(entities, [], [room], free, valid, grid, 1.0, SETTINGS)
    assert result.stats["double_wall_room_edge_pieces"] >= 1
    body = _body(result)
    # The new wall continues the measured north face piece at y = 3.3 outwards.
    assert body.contains(Point(2.5, 3.45)) and not body.contains(Point(2.5, 3.2))
    # And the building outline is closed: one wall body enclosing the room.
    assert len(result.polygons) == 1 and len(result.polygons[0].interiors) == 1


def test_rooms_grow_to_inner_faces_and_ceiling_fragments_merge() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    result = build_double_walls(entities, [], rooms, free, valid, grid, 1.0, SETTINGS)
    # Ceiling rooms stop 8 cm short of the faces; room B is split by a ceiling hole into a big
    # part and a small fragment with no wall between them.
    shrunk_a = box(0.08, 0.08, 3.92, 2.92)
    big_b = box(4.23, 0.08, 7.0, 2.92)
    small_b = box(7.25, 0.08, 7.92, 2.92)
    bounded, stats = bound_rooms_by_walls([shrunk_a, big_b, small_b], result.closed_polygons, 1.0, SETTINGS)
    assert stats["rooms_merged"] == 1 and len(bounded) == 2
    areas = sorted(polygon.area for polygon in bounded)
    assert abs(areas[0] - 3.85 * 3.0) <= 0.15 and abs(areas[1] - 4.0 * 3.0) <= 0.15
    walls = unary_union(result.closed_polygons)
    assert all(polygon.intersection(walls).area <= 0.01 for polygon in bounded)


def test_hatch_entity_is_written_and_audited(tmp_path) -> None:
    from samcloud.dxf.las_source import Frame
    from samcloud.dxf.result import BackgroundImage, DxfPlanResult
    from samcloud.dxf.writer import build_dxf_document

    entities, rooms, free, valid, grid = _two_rooms()
    result = build_double_walls(entities, [], rooms, free, valid, grid, 1.0, SETTINGS)
    plan = DxfPlanResult(
        project_name="p", level_name="l", mode="indoor", profile_name=None, scale_factor=1.0,
        frame=Frame(origin=np.zeros(3), scale_factor=1.0, rotation=np.eye(3)), settings_fingerprint="x",
        layers={"A-WALL": result.entities, "A-WALL-PATT": result.hatches}, statuses={}, layer_colors={"A-WALL": 7, "A-WALL-PATT": 253},
        background=BackgroundImage(None, None, (-1.0, -1.0, 9.0, 4.0)), layer_styles={"A-WALL-PATT": {"hatch_pattern": "SOLID"}},
    )
    document = build_dxf_document(plan)
    hatches = document.modelspace().query('HATCH[layer=="A-WALL-PATT"]')
    assert len(hatches) == len(result.polygons) >= 1
    assert len(hatches[0].paths) == 1 + len(result.polygons[0].interiors)
    walls = document.modelspace().query('LWPOLYLINE[layer=="A-WALL"]')
    assert all(entity.closed for entity in walls)
    path = tmp_path / "walls.dxf"
    document.saveas(path)
    assert not ezdxf.readfile(path).audit().has_errors


def test_double_line_setting_is_loaded_and_fingerprinted(tmp_path) -> None:
    from samcloud.ortho.plan import load_clean_plan_settings

    config = tmp_path / "ortho_plan.yaml"
    config.write_text("clean_plan:\n  segment_source: combined\nwalls:\n  double_line: true\n  exterior_thickness_m: 0.36\n", encoding="utf-8")
    settings = load_clean_plan_settings(config)
    assert settings.walls.double_line is True and abs(settings.walls.exterior_thickness_m - 0.36) <= 1e-9
    default = load_clean_plan_settings(tmp_path / "missing.yaml")
    assert default.walls.double_line is True
    config.write_text("clean_plan:\n  segment_source: combined\nwalls:\n  double_line: false\n", encoding="utf-8")
    assert load_clean_plan_settings(config).walls.double_line is False


def test_unnamed_room_with_entrance_and_doors_is_a_hall() -> None:
    from samcloud.ortho.vector.room_names import RoomName, RoomNameSettings, name_halls

    hall = box(0.0, 0.0, 3.0, 2.0)
    living = box(3.2, 0.0, 8.0, 4.0)
    store = box(0.0, 2.2, 3.0, 4.0)
    decisions = [RoomName(None, 6.0, True, frozenset()), RoomName("Wohnen", 19.2, True, frozenset({"living"})), RoomName(None, 5.4, True, frozenset())]
    # Entrance door in the south wall of the hall (outside below), a door into the living room.
    doors = [((1.5, -0.1), 0.0), ((3.1, 1.0), math.pi / 2.0)]
    named = name_halls([hall, living, store], decisions, doors, 1.0, RoomNameSettings())
    assert named[0].name == "Flur"
    assert named[1].name == "Wohnen"
    # One interior door only and no entrance: stays unnamed.
    assert named[2].name is None
    # Without the entrance door a single interior door is not enough either.
    assert name_halls([hall], [decisions[0]], [((3.1, 1.0), math.pi / 2.0)], 1.0)[0].name is None


def test_table_split_into_instances_becomes_one_axis_rectangle() -> None:
    from samcloud.ortho.vector.footprints import ObjectInstance, refine_dining_instances

    grid = _grid((0.0, 0.0, 4.0, 4.0))
    hm = np.full((grid.height, grid.width), 0.75, dtype=np.float32)
    classes = np.full((grid.height, grid.width), 117, dtype=np.uint8)
    # One 1.6 x 0.9 table top seen as three instances (two halves and a corner piece), all labelled table.
    pieces = [box(1.0, 1.0, 1.8, 1.9), box(1.84, 1.0, 2.6, 1.6), box(1.84, 1.66, 2.6, 1.9)]
    instances = [ObjectInstance(_mask(grid, [piece]), "table", 117, True, 0.75, 1.0, instance_id=index + 1) for index, piece in enumerate(pieces)]
    stats: dict = {}
    result = refine_dining_instances(instances, classes, grid, hm, 0.0, 1.0, stats)
    tables = [item for item in result if item.label == "table"]
    assert stats["tables_merged"] == 2 and len(tables) == 1
    polygon = tables[0].polygon
    assert polygon is not None
    x0, y0, x1, y1 = polygon.bounds
    assert abs(x0 - 1.0) <= 0.05 and abs(x1 - 2.6) <= 0.05 and abs(y0 - 1.0) <= 0.05 and abs(y1 - 1.9) <= 0.05
    assert abs(polygon.area - polygon.minimum_rotated_rectangle.area) <= 1e-6


def test_furniture_between_two_faces_is_not_a_measured_wall_pair() -> None:
    entities, rooms, free, valid, grid = _two_rooms(wall=0.35)
    plain = build_double_walls(entities, [], rooms, free, valid, grid, 1.0, SETTINGS)
    assert plain.stats["double_wall_measured_pairs"] >= 1
    furniture = _mask(grid, [box(4.0, 0.0, 4.35, 3.0)])
    result = build_double_walls(entities, [], rooms, free, valid, grid, 1.0, SETTINGS, furniture=furniture)
    assert result.stats["double_wall_measured_pairs"] == 0


def _evidence(grid: RasterGrid, **masks) -> "WallEvidence":
    from samcloud.ortho.vector.double_walls import WallEvidence

    return WallEvidence(bins=16, storey_height=2.4, **masks)


def _full_wall_bits(grid: RasterGrid, polygons: list[Polygon]) -> np.ndarray:
    bits = np.zeros((grid.height, grid.width), dtype=np.uint32)
    bits[_mask(grid, polygons)] = np.uint32(0xFFFF)
    return bits


def test_door_detected_as_open_leaf_becomes_the_opening_in_the_wall() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    # The shared wall (x 4.00..4.15) is open between y 1.0 and 1.9 at sill and mid height;
    # the door detector reported the open leaf instead: perpendicular to the wall, hinge at (4.0, 1.9).
    solid = _mask(grid, [box(3.97, 0.0, 4.18, 1.0), box(3.97, 1.9, 4.18, 3.0)])
    entities = [entity for entity in entities if not (entity["points"][0][0] == 4.0 and entity["points"][1][0] == 4.0)]
    entities += [_face(4.0, 0.0, 4.0, 1.0), _face(4.0, 1.9, 4.0, 3.0)]
    leaf = Opening("DOOR", (3.55, 1.9), 0.9, 0.0, 0.04, leaf_detected=True)
    evidence = _evidence(grid, section=solid, low=solid, floor=free)
    result = build_double_walls(entities, [leaf], rooms, free, valid, grid, 1.0, SETTINGS, evidence=evidence)
    doors = [item for item in result.openings if item.kind == "DOOR"]
    assert len(doors) == 1
    door = doors[0]
    assert abs(abs(math.cos(door.axis_angle))) <= 0.05  # along the wall (vertical)
    assert 3.95 <= door.center[0] <= 4.2 and abs(door.center[1] - 1.45) <= 0.08 and abs(door.width - 0.9) <= 0.08
    assert any(old is leaf for old, _new in result.openings_changed)
    body = _body(result)
    assert not body.contains(Point(4.075, 1.45)) and body.contains(Point(4.075, 0.5))


def test_doorway_without_detection_is_found_in_the_wall_body() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    solid = _mask(grid, [box(3.97, 0.0, 4.18, 1.2), box(3.97, 2.0, 4.18, 3.0)])
    floor = free | _mask(grid, [box(4.0, 1.2, 4.15, 2.0)])
    evidence = _evidence(grid, section=solid, low=solid, floor=floor)
    result = build_double_walls(entities, [], rooms, free, valid, grid, 1.0, SETTINGS, evidence=evidence)
    assert result.stats["double_wall_doors_detected"] == 1
    door = result.openings[0]
    assert abs(door.center[1] - 1.6) <= 0.06 and abs(door.width - 0.8) <= 0.08
    assert not _body(result).contains(Point(4.075, 1.6))


def test_door_span_that_is_solid_wall_is_dropped() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    solid = _mask(grid, [box(3.97, 0.0, 4.18, 3.0)])
    fake = Opening("DOOR", (4.0, 1.5), 0.8, math.pi / 2.0, 0.15, evidence="face-gap", leaf_detected=False)
    evidence = _evidence(grid, section=solid, low=solid, floor=free)
    result = build_double_walls(entities, [fake], rooms, free, valid, grid, 1.0, SETTINGS, evidence=evidence)
    assert result.stats["double_wall_doors_rejected"] == 1 and not result.openings
    assert _body(result).contains(Point(4.075, 1.5))


def test_counter_front_is_not_a_wall_but_a_curtained_wall_is() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    # A kitchen counter front in room A at y = 2.4 (data only up to 0.9 m, counter behind it).
    counter = box(1.0, 2.4, 2.5, 3.0)
    front = _face(1.0, 2.4, 2.5, 2.4)
    all_bits = np.zeros((grid.height, grid.width), dtype=np.uint32)
    all_bits[_mask(grid, [counter.buffer(0.02)])] = np.uint32(0b111111)  # up to ~0.9 m
    wall_bits = _full_wall_bits(grid, [box(-0.05, -0.05, 8.05, 0.05), box(3.95, 0.0, 4.20, 3.0)])
    # The south wall of room B is hidden by a curtain: no wall points, a floor-to-ceiling curtain surface.
    all_bits[_mask(grid, [box(4.3, 2.92, 7.9, 2.96)])] = np.uint32(0xFFFF)
    furniture = _mask(grid, [counter])
    evidence = _evidence(grid, wall_bits=wall_bits, all_bits=all_bits)
    result = build_double_walls(entities + [front], [], rooms, free, valid, grid, 1.0, SETTINGS, furniture=furniture, evidence=evidence)
    assert result.stats["double_wall_faces_low"] == 1
    assert not _body(result).intersects(LineString([(1.2, 2.39), (2.3, 2.39)]))
    assert _body(result).contains(Point(6.0, 3.15))


def test_mirror_phantom_face_without_camera_view_is_dropped() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    # A phantom face 1 m beyond the east exterior wall: cameras are only inside the rooms.
    phantom = _face(9.0, 0.5, 9.0, 2.5)
    cameras = np.asarray([[2.0, 1.5], [6.0, 1.5], [1.0, 2.5]])
    evidence = _evidence(grid, cameras=cameras)
    result = build_double_walls(entities + [phantom], [], rooms, free, valid, grid, 1.0, SETTINGS, evidence=evidence)
    assert result.stats["double_wall_faces_unseen"] == 1
    assert not _body(result).intersects(LineString([(9.0, 0.6), (9.0, 2.4)]))
    assert _body(result).contains(Point(8.15, 1.5))


def test_window_overlapping_a_door_keeps_only_its_side_part() -> None:
    entities, rooms, free, valid, grid = _two_rooms()
    door = Opening("DOOR", (1.0, 0.0), 0.8, 0.0, 0.3, leaf_detected=False)
    window = Opening("WINDOW", (1.6, 0.0), 1.6, 0.0, 0.3)
    result = build_double_walls(entities, [door, window], rooms, free, valid, grid, 1.0, SETTINGS, evidence=_evidence(grid))
    windows = [item for item in result.openings if item.kind == "WINDOW"]
    assert len(windows) == 1 and abs(windows[0].width - 1.0) <= 1e-6 and abs(windows[0].center[0] - 1.9) <= 1e-6


# --------------------------------------------------------------------------
# tall built-in kitchen units taken as walls
# --------------------------------------------------------------------------

def _units(grid: RasterGrid, front: list[Polygon], fridge: list[Polygon] = (), low: list[Polygon] = ()) -> "TallUnitEvidence":
    from samcloud.ortho.vector.double_walls import TallUnitEvidence

    return TallUnitEvidence(front=_mask(grid, list(front)), fridge=_mask(grid, list(fridge)), low=_mask(grid, list(low)))


def _kitchen_row_scene():
    """Kitchen [0,4.53]x[0,3] with a row of tall units (x 3.9..4.53, y 0..2.0) along its east wall; the real wall
    (x 4.53..4.68) is seen above the units from the kitchen and from room B [4.68,7]x[0,3].  The plan measured the
    unit fronts at x 3.9 as a wall face."""
    grid = _grid((-1.0, -1.0, 8.0, 4.0))
    kitchen, unit, room_b = box(0.0, 0.0, 4.53, 3.0), box(3.9, 0.0, 4.53, 2.0), box(4.68, 0.0, 7.0, 3.0)
    free = _mask(grid, [kitchen, room_b]) & ~_mask(grid, [unit])
    valid = _mask(grid, [box(-0.4, -0.4, 7.4, 3.4)]) & ~_mask(grid, [box(3.93, 0.0, 4.53, 1.99), box(4.54, -0.4, 4.67, 3.4)])
    entities = [
        _face(0.0, 0.0, 3.9, 0.0), _face(0.0, 0.0, 0.0, 3.0), _face(0.0, 3.0, 4.53, 3.0),
        _face(3.9, 0.0, 3.9, 2.0),  # unit fronts measured as a wall face
        _face(4.53, 2.0, 4.53, 3.0),  # the real wall, seen above the units
        _face(4.68, 0.0, 4.68, 3.0), _face(4.68, 0.0, 7.0, 0.0), _face(7.0, 0.0, 7.0, 3.0), _face(4.68, 3.0, 7.0, 3.0),
    ]
    return grid, entities, free, valid


def test_tall_kitchen_unit_front_is_not_a_wall_and_the_wall_behind_stays() -> None:
    grid, entities, free, valid = _kitchen_row_scene()
    without = build_double_walls(entities, [], [], free, valid, grid, 1.0, SETTINGS)
    assert _body(without).contains(Point(3.95, 1.0)) and not without.tall_units  # old behaviour: the fronts are a wall
    units = _units(grid, [box(3.87, 0.0, 3.93, 2.0)], fridge=[box(3.87, 0.0, 3.93, 0.57)])
    result = build_double_walls(entities, [], [], free, valid, grid, 1.0, SETTINGS, tall_units=units)
    body = _body(result)
    for y in (0.3, 1.0, 1.7):
        assert not body.contains(Point(3.95, y)) and not body.contains(Point(4.3, y))  # no wall through the units
        assert body.contains(Point(4.6, y))  # the real wall behind them stays
    assert body.contains(Point(4.6, 2.5)) and body.contains(Point(-0.05, 1.5))
    shapes = [shape for shape, _ in result.tall_units]
    labels = [label for _, label in result.tall_units]
    assert labels == ["KS", "HS"] and result.stats["tall_units"] == 1
    union = unary_union(shapes)
    x0, y0, x1, y1 = union.bounds
    assert abs(x0 - 3.9) <= 0.04 and abs(x1 - 4.53) <= 0.04 and y0 <= 0.05 and abs(y1 - 2.0) <= 0.06
    fridge = shapes[0].bounds
    assert fridge[1] <= 0.05 and 0.55 <= fridge[3] - fridge[1] <= 0.75  # the fridge module at the end holding the fridge
    assert union.intersection(body).area <= 0.01  # the units never overlap a wall body


def test_low_surfaces_behind_a_front_keep_it_a_wall() -> None:
    """A counter seen behind the 'front' (low surfaces in the footprint) is no floor-to-ceiling unit."""
    grid, entities, free, valid = _kitchen_row_scene()
    valid = valid | _mask(grid, [box(3.93, 0.0, 4.53, 1.99)])
    units = _units(grid, [box(3.87, 0.0, 3.93, 2.0)], low=[box(3.95, 0.0, 4.50, 1.99)])
    result = build_double_walls(entities, [], [], free, valid, grid, 1.0, SETTINGS, tall_units=units)
    assert not result.tall_units and _body(result).contains(Point(3.95, 1.0))
    # Disabled by the setting: same as without evidence.
    units = _units(grid, [box(3.87, 0.0, 3.93, 2.0)])
    off = build_double_walls(entities, [], [], free, valid, grid, 1.0, replace(SETTINGS, tall_units=False), tall_units=units)
    assert not off.tall_units and _body(off).contains(Point(3.95, 1.0))


def test_corner_tall_unit_continues_the_hidden_wall_behind_it() -> None:
    """Kitchen [0,4]x[0,3]: a tall corner block (x 2.7..4.0, front at y 2.35) hides the north wall behind it; the
    plan took its front as the wall.  The unit becomes a built-in and the north wall continues behind it."""
    grid = _grid((-1.0, -1.0, 5.0, 4.0))
    kitchen, unit = box(0.0, 0.0, 4.0, 3.0), box(2.7, 2.35, 4.0, 3.0)
    free = _mask(grid, [kitchen]) & ~_mask(grid, [unit, box(0.0, 2.4, 2.7, 3.0)])  # a counter run along the north wall
    valid = _mask(grid, [box(-0.4, -0.4, 4.4, 3.0)]) & ~_mask(grid, [box(2.72, 2.38, 4.0, 3.0)])
    entities = [
        _face(0.0, 0.0, 4.0, 0.0), _face(0.0, 0.0, 0.0, 3.0), _face(4.0, 0.0, 4.0, 3.0),
        _face(0.0, 3.0, 2.6, 3.0),  # north wall behind the counter
        _face(2.7, 2.35, 4.0, 2.35),  # tall unit front measured as a wall face
    ]
    without = build_double_walls(entities, [], [], free, valid, grid, 1.0, SETTINGS)
    assert _body(without).contains(Point(3.3, 2.45)) and not _body(without).contains(Point(3.3, 3.15))
    units = _units(grid, [box(2.7, 2.33, 3.98, 2.38)], low=[box(0.0, 2.4, 2.7, 3.0)])
    result = build_double_walls(entities, [], [], free, valid, grid, 1.0, SETTINGS, tall_units=units)
    body = _body(result)
    assert len(result.tall_units) == 1 and result.tall_units[0][1] == "HS"
    assert result.stats["tall_unit_walls_continued"] == 1
    x0, y0, x1, y1 = result.tall_units[0][0].bounds
    assert abs(x0 - 2.7) <= 0.04 and abs(x1 - 4.0) <= 0.04 and abs(y0 - 2.35) <= 0.04 and abs(y1 - 3.0) <= 0.04
    for x in (2.8, 3.3, 3.9):
        assert not body.contains(Point(x, 2.45)) and not body.contains(Point(x, 2.8))  # no wall through the block
        assert body.contains(Point(x, 3.1))  # the north wall continues behind it
    assert body.contains(Point(4.1, 3.1)) and body.contains(Point(1.0, 3.1))  # closed corner, wall still there
    assert body.geom_type == "Polygon"  # one closed body around the kitchen corner


# --- isolated / stub / free-standing face rules (wall diagnosis v10) -----------------------------------------------

def _faces_of(entities: list[dict]):
    from samcloud.ortho.vector.double_walls import _collect_faces

    faces, _ = _collect_faces(entities, 1.0, SETTINGS.min_face_m)
    return faces


def _split(entities: list[dict], rooms=(), settings=SETTINGS):
    from samcloud.ortho.vector.double_walls import _split_isolated

    kept, isolated = _split_isolated(_faces_of(entities), 1.0, settings, list(rooms))
    return {tuple(round(float(v), 2) for v in (*f.start, *f.end)) for f in kept}, {tuple(round(float(v), 2) for v in (*f.start, *f.end)) for f in isolated}


def test_stub_with_a_free_end_is_dropped_but_a_jamb_between_two_walls_stays() -> None:
    wall_a, wall_b = _face(0.0, 0.0, 3.0, 0.0), _face(0.0, 0.5, 3.0, 0.5)
    attached_one_end = _face(1.0, 0.0, 1.0, -0.5)  # 0.5 m, touches wall A only; the free end is 0.5 m from anything
    jamb = _face(2.0, 0.5, 2.0, 0.0)  # 0.5 m between the two walls: both ends near
    kept, isolated = _split([wall_a, wall_b, attached_one_end, jamb])
    assert (1.0, 0.0, 1.0, -0.5) in isolated and (2.0, 0.5, 2.0, 0.0) in kept
    # rule off (stub_min_length_m = 0): the old behaviour (one near end is enough) keeps it
    kept_old, isolated_old = _split([wall_a, wall_b, attached_one_end, jamb], settings=replace(SETTINGS, stub_min_length_m=0.0))
    assert (1.0, 0.0, 1.0, -0.5) in kept_old and not isolated_old


def test_stub_end_must_touch_a_wall_not_just_lie_within_the_joint_reach() -> None:
    # A 0.48 m face beside a wall (0.2 m from it at its lower end) is a free object edge, not a return: it touches a wall only
    # at the top. Within joint_reach (0.35 m) the old rule kept it; a jamb touches within 0.10 m at both ends.
    wall = _face(0.0, 0.0, 4.0, 0.0)
    long_wall = _face(-0.2, -0.5, -0.2, 3.0)  # parallel wall 0.2 m left of the stub's lower end
    stub = _face(0.0, 0.0, 0.0, -0.48)
    # lower end (0,-0.48) is 0.2 m from long_wall: near by joint reach, not touching
    kept, isolated = _split([wall, long_wall, stub])
    assert (0.0, 0.0, 0.0, -0.48) in isolated
    kept_old, isolated_old = _split([wall, long_wall, stub], settings=replace(SETTINGS, stub_end_reach_m=0.0))
    assert (0.0, 0.0, 0.0, -0.48) in kept_old and not isolated_old
    wall_below = _face(0.0, -0.48, 4.0, -0.48)
    jamb = _face(2.0, 0.0, 2.0, -0.48)  # touches a wall at both ends
    kept_touch, isolated_touch = _split([wall, wall_below, jamb])
    assert (2.0, 0.0, 2.0, -0.48) in kept_touch and not isolated_touch


def test_face_of_stub_length_or_more_keeps_the_single_end_rule() -> None:
    wall = _face(0.0, 0.0, 3.0, 0.0)
    return_wall = _face(1.0, 0.0, 1.0, -0.8)  # 0.8 m (>= 0.6), attached at one end
    kept, isolated = _split([wall, return_wall])
    assert (1.0, 0.0, 1.0, -0.8) in kept and not isolated


def test_free_standing_face_under_2_5_m_is_dropped_unless_a_room_is_near() -> None:
    wall = _face(0.0, 0.0, 6.0, 0.0)
    free = _face(0.0, 3.0, 2.0, 3.0)  # 2.0 m, far from every wall end
    kept, isolated = _split([wall, free])
    assert (0.0, 3.0, 2.0, 3.0) in isolated
    room_near = box(0.0, 2.0, 4.0, 2.6)  # ceiling room 0.4 m from the face: it borders a room
    kept_near, isolated_near = _split([wall, free], rooms=[room_near])
    assert (0.0, 3.0, 2.0, 3.0) in kept_near and not isolated_near
    room_far = box(0.0, 1.0, 4.0, 2.0)  # 1.0 m away: no longer protects it
    assert (0.0, 3.0, 2.0, 3.0) in _split([wall, free], rooms=[room_far])[1]
    long_free = _face(0.0, 3.0, 2.6, 3.0)  # >= 2.5 m: kept
    assert (0.0, 3.0, 2.6, 3.0) in _split([wall, long_free])[0]
    attached = _face(0.0, 3.0, 2.0, 3.0), _face(0.0, 3.0, 0.0, 0.2)  # touching a wall end: not free at both ends
    assert (0.0, 3.0, 2.0, 3.0) in _split([wall, *attached])[0]
    assert (0.0, 3.0, 2.0, 3.0) in _split([wall, free], settings=replace(SETTINGS, isolated_free_max_length_m=0.0))[0]  # rule off


def test_stub_and_free_rules_spare_builtin_and_niche_faces() -> None:
    wall = _face(0.0, 0.0, 6.0, 0.0)
    front = {**_face(0.0, 3.0, 2.0, 3.0), "_source": "builtin_front"}
    tiny = {**_face(5.0, 3.0, 5.0, 3.5), "_source": "niche_panel"}
    kept, isolated = _split([wall, front, tiny])
    assert (0.0, 3.0, 2.0, 3.0) in kept
    assert (5.0, 3.0, 5.0, 3.5) in isolated  # < 1.2 m with no near end is still the old isolated rule


def test_room_edge_body_thickness_is_capped_at_max_thickness() -> None:
    room = box(0.0, 0.0, 4.0, 3.0)
    grid = _grid((-1.0, -1.0, 5.0, 5.5))
    free = _mask(grid, [box(0.0, 0.0, 4.0, 2.4)])
    valid = _mask(grid, [box(-0.05, -0.05, 4.05, 3.30)])
    base = [_face(0.0, 0.0, 4.0, 0.0), _face(4.0, 0.0, 4.0, 3.3), _face(0.0, 3.3, 0.0, 0.0), _face(0.0, 3.3, 1.0, 3.3)]
    behind = [_face(0.5, 4.0, 3.5, 4.0), _face(0.5, 4.2, 3.5, 4.2)]  # a wall 0.7 m behind the hidden north wall: mass in between is ~0.71 m
    result = build_double_walls(base + behind, [], [room], free, valid, grid, 1.0, SETTINGS)
    edge = [piece for piece in result.pieces if piece.get("source") == "room_edge"]
    assert len(edge) == 1
    assert edge[0]["thickness"] == pytest.approx(SETTINGS.max_thickness_m)
    narrower = build_double_walls(base + [_face(0.5, 3.6, 3.5, 3.6), _face(0.5, 3.8, 3.5, 3.8)], [], [room], free, valid, grid, 1.0, SETTINGS)
    assert [piece["thickness"] for piece in narrower.pieces if piece.get("source") == "room_edge"] == [pytest.approx(0.3)]  # below the cap: unchanged


def _hidden_north_wall_scene():
    room = box(0.0, 0.0, 4.0, 3.0)
    grid = _grid((-1.0, -1.0, 5.0, 4.5))
    free = _mask(grid, [box(0.0, 0.0, 4.0, 2.4)])
    valid = _mask(grid, [box(-0.05, -0.05, 4.05, 3.30)])
    entities = [_face(0.0, 0.0, 4.0, 0.0), _face(4.0, 0.0, 4.0, 3.3), _face(0.0, 3.3, 0.0, 0.0), _face(0.0, 3.3, 1.0, 3.3)]
    return room, grid, free, valid, entities


def test_room_edge_invents_no_wall_in_front_of_a_window_span() -> None:
    # The measured window marks the wall line 0.5 m beyond the room edge's hidden wall: no outside block may be
    # built in the window span (it stayed as a loose wall block beside the cut), the walls beside the window stay.
    room, grid, free, valid, entities = _hidden_north_wall_scene()
    window = Opening("WINDOW", (2.0, 3.8), 1.6, 0.0, 0.3)
    plain = build_double_walls(entities, [], [room], free, valid, grid, 1.0, SETTINGS)
    assert _body(plain).contains(Point(2.0, 3.45))  # without an opening the hidden wall is closed over the full side
    result = build_double_walls(entities, [window], [room], free, valid, grid, 1.0, SETTINGS)
    body = _body(result)
    assert not body.contains(Point(2.0, 3.45)) and not body.contains(Point(1.3, 3.45)) and not body.contains(Point(2.7, 3.45))
    assert body.contains(Point(0.7, 3.45)) and body.contains(Point(3.3, 3.45))
    off = build_double_walls(entities, [window], [room], free, valid, grid, 1.0, replace(SETTINGS, room_edge_opening_reach_m=0.0))
    assert _body(off).contains(Point(2.0, 3.45))  # rule off: the old behaviour


def test_room_edge_drops_a_stub_left_beside_a_window_span() -> None:
    # Window span x 1.13..3.87 leaves 0.13 m beside it at both ends, far below room_edge_min_length (0.3 m): no stub block.
    room, grid, free, valid, entities = _hidden_north_wall_scene()
    window = Opening("WINDOW", (2.5, 3.8), 2.7, 0.0, 0.3)
    result = build_double_walls(entities, [window], [room], free, valid, grid, 1.0, SETTINGS)
    body = _body(result)
    assert not body.contains(Point(1.08, 3.55)) and not body.contains(Point(3.93, 3.55)) and not body.contains(Point(2.5, 3.55))
    off = build_double_walls(entities, [window], [room], free, valid, grid, 1.0, replace(SETTINGS, room_edge_opening_reach_m=0.0))
    assert _body(off).contains(Point(2.5, 3.55)) and _body(off).contains(Point(3.93, 3.55))


def test_room_edge_ignores_openings_that_are_not_parallel_or_far() -> None:
    room, grid, free, valid, entities = _hidden_north_wall_scene()
    across = Opening("DOOR", (2.0, 3.8), 0.9, math.pi / 2.0, 0.3, leaf_detected=False)  # perpendicular to the room edge
    far = Opening("WINDOW", (2.0, 6.0), 1.6, 0.0, 0.3)  # 2.7 m beyond the edge
    for opening in (across, far):
        result = build_double_walls(entities, [opening], [room], free, valid, grid, 1.0, SETTINGS)
        assert _body(result).contains(Point(2.0, 3.45)), opening


def _vertical_run(index: int, x: float, y0: float, y1: float, side: int, thickness: float = 0.25) -> tuple[_Face, WallRun]:
    face = _Face(index, np.asarray((x, y0)), np.asarray((x, y1)), {"_source": "measured"}, side=side)
    return face, WallRun(index, 0.0, abs(y1 - y0), thickness, "interior", "measured")


def test_hostless_door_filling_a_short_wall_segment_takes_the_slab_of_the_collinear_run() -> None:
    # The door fills the whole wall segment between y=2.0 and y=3.0: no run covers it, so it used to cut nothing and the room leaked.
    face, run = _vertical_run(0, 0.0, 0.0, 2.0, side=1)  # solid on -x: slab x in [-0.25, 0]
    door = Opening("DOOR", (0.0, 2.6), 0.8, math.pi / 2.0, 0.25, leaf_detected=False)
    centre, axis = np.asarray(door.center), np.asarray((0.0, 1.0))
    low, high = _neighbour_slab(centre, axis, 0.4, [run], {0: face}, 1.0, SETTINGS)
    assert (low, high) == (pytest.approx(0.0), pytest.approx(0.25))  # offsets along the opening normal (-1, 0): x in [-0.25, 0]
    _bands, slabs, cuts, _jambs = _wall_bodies([run], [face], [door], 1.0, SETTINGS)
    assert slabs == [(pytest.approx(0.0), pytest.approx(0.25))]
    assert cuts and unary_union(cuts).contains(Point(-0.12, 2.6))  # the gap is closed over the wall thickness


def test_hostless_door_slab_comes_from_the_nearest_run_across_and_along() -> None:
    near, near_run = _vertical_run(0, 0.0, 0.0, 2.0, side=1)  # on the opening line, ends 0.2 m before the span
    other, other_run = _vertical_run(1, 0.46, 3.0, 5.0, side=-1)  # 0.46 m beside the line, starts right after the span
    centre, axis = np.asarray((0.0, 2.6)), np.asarray((0.0, 1.0))
    result = _neighbour_slab(centre, axis, 0.4, [other_run, near_run], {0: near, 1: other}, 1.0, SETTINGS)
    assert result == (pytest.approx(0.0), pytest.approx(0.25))
    assert _neighbour_slab(centre, axis, 0.4, [other_run], {1: other}, 1.0, SETTINGS) is not None  # alone it qualifies: order does not matter


def test_hostless_door_without_a_collinear_run_still_cuts_nothing() -> None:
    face, run = _vertical_run(0, 0.0, 0.0, 2.0, side=1)
    centre, axis = np.asarray((0.0, 4.0)), np.asarray((0.0, 1.0))  # 1.6 m beyond the run end
    assert _neighbour_slab(centre, axis, 0.4, [run], {0: face}, 1.0, SETTINGS) is None
    across = np.asarray((1.0, 0.0))  # perpendicular door axis: the run is not a host
    assert _neighbour_slab(np.asarray((0.0, 2.2)), across, 0.4, [run], {0: face}, 1.0, SETTINGS) is None
    door = Opening("DOOR", (0.0, 4.0), 0.8, math.pi / 2.0, 0.25, leaf_detected=False)
    _bands, slabs, cuts, _jambs = _wall_bodies([run], [face], [door], 1.0, SETTINGS)
    assert slabs == [(0.0, 0.0)] and not cuts


def test_hostless_window_next_to_a_run_still_cuts_nothing() -> None:
    face, run = _vertical_run(0, 0.0, 0.0, 2.0, side=1)
    window = Opening("WINDOW", (0.0, 2.6), 0.8, math.pi / 2.0, 0.25)
    _bands, slabs, cuts, _jambs = _wall_bodies([run], [face], [window], 1.0, SETTINGS)
    assert slabs == [(0.0, 0.0)] and not cuts


# --------------------------------------------------------------------------
# cabinet fronts taken as walls
# --------------------------------------------------------------------------

def _cabinet_front_scene(upm: float = 1.0, wall_points: bool = False, cabinet_labels: bool = True):
    """Room A [0,4]x[0,3] with a run of kitchen cabinets along its north wall measured as a wall face at y = 2.3
    (cabinet pixels y 2.3..3.0).  ``wall_points``: wall-class points of full height at the face (a real wall)."""
    from samcloud.ortho.vector.double_walls import TallUnitEvidence

    s = upm
    entities, rooms, free, valid, grid = _two_rooms(upm)
    free = free & ~_mask(grid, [box(0.9 * s, 2.3 * s, 3.1 * s, 3.0 * s)])
    cabinet = _mask(grid, [box(1.0 * s, 2.3 * s, 3.0 * s, 3.0 * s)]) if cabinet_labels else np.zeros((grid.height, grid.width), dtype=bool)
    zeros = np.zeros_like(cabinet)
    units = TallUnitEvidence(front=zeros, fridge=zeros, low=zeros, cabinet=cabinet)
    wall_solid = [box(-0.05 * s, -0.05 * s, 8.05 * s, 0.05 * s), box(3.95 * s, 0.0, 4.20 * s, 3.0 * s)]
    if wall_points:
        wall_solid.append(box(0.95 * s, 2.28 * s, 3.05 * s, 2.45 * s))
    all_bits = np.zeros((grid.height, grid.width), dtype=np.uint32)
    all_bits[_mask(grid, [box(1.0 * s, 2.3 * s, 3.0 * s, 2.7 * s)])] = np.uint32(0b111111)
    evidence = WallEvidence(bins=16, storey_height=2.4 * s, wall_bits=_full_wall_bits(grid, wall_solid), all_bits=all_bits)
    return entities + [_face(1.0 * s, 2.3 * s, 3.0 * s, 2.3 * s)], rooms, free, valid, grid, units, evidence


@pytest.mark.parametrize("upm", (1.0, 0.8124))
def test_wall_run_on_cabinet_labels_without_wall_points_is_dropped(upm: float) -> None:
    entities, rooms, free, valid, grid, units, evidence = _cabinet_front_scene(upm)
    settings = replace(SETTINGS, face_test=False, visibility_test=False)
    probe = Point(2.0 * upm, 2.34 * upm)
    off = build_double_walls(entities, [], rooms, free, valid, grid, upm, replace(settings, cabinet_walls=False), evidence=evidence, tall_units=units)
    assert _body(off).contains(probe)  # old behaviour: the cabinet front is a wall
    result = build_double_walls(entities, [], rooms, free, valid, grid, upm, settings, evidence=evidence, tall_units=units)
    assert result.stats["cabinet_wall_runs"] == 1
    assert not _body(result).contains(probe)
    assert _body(result).contains(Point(-0.15 * upm, 1.5 * upm))  # the exterior walls stay
    assert result.stats["cabinet_wall_m"] == pytest.approx(2.0, abs=0.25)


@pytest.mark.parametrize("upm", (1.0, 0.8124))
def test_real_wall_behind_cabinets_stays_a_wall(upm: float) -> None:
    entities, rooms, free, valid, grid, units, evidence = _cabinet_front_scene(upm, wall_points=True)
    settings = replace(SETTINGS, face_test=False, visibility_test=False)
    result = build_double_walls(entities, [], rooms, free, valid, grid, upm, settings, evidence=evidence, tall_units=units)
    assert result.stats.get("cabinet_wall_runs", 0) == 0
    assert _body(result).contains(Point(2.0 * upm, 2.34 * upm))


@pytest.mark.parametrize("upm", (1.0, 0.8124))
def test_wall_run_without_cabinet_labels_stays_a_wall(upm: float) -> None:
    entities, rooms, free, valid, grid, units, evidence = _cabinet_front_scene(upm, cabinet_labels=False)
    settings = replace(SETTINGS, face_test=False, visibility_test=False)
    result = build_double_walls(entities, [], rooms, free, valid, grid, upm, settings, evidence=evidence, tall_units=units)
    assert result.stats.get("cabinet_wall_runs", 0) == 0
    assert _body(result).contains(Point(2.0 * upm, 2.34 * upm))


# --------------------------------------------------------------------------
# cabinet columns whose side panels were taken as a wall
# --------------------------------------------------------------------------

def _column_scene(upm: float = 1.0, wall_points: bool | float = False, cabinet_labels: bool = True, length: float = 1.6, far_floor: bool = True):
    """Room A [0,4]x[0,3] with a measured wall face at x = 2.0 running from the north wall down ``length`` metres.  Kitchen
    cabinets stand east of it (x 2.3..2.9, flank only: the face body itself is not on a label); a built-in unit outline
    covers x 2.0..2.9.  ``wall_points``: wall-class points of full height along the face (a real partition)."""
    from samcloud.ortho.vector.double_walls import TallUnitEvidence

    s = upm
    entities, rooms, free, valid, grid = _two_rooms(upm)
    y0 = 3.0 - length
    free = free & ~_mask(grid, [box(1.95 * s, y0 * s, 2.95 * s, 3.0 * s)])
    if not far_floor:  # nothing measured west of the face: it is the outer wall behind the cabinets
        free = free & ~_mask(grid, [box(0.5 * s, y0 * s, 1.95 * s, 3.0 * s)])
    cabinet = _mask(grid, [box(2.3 * s, y0 * s, 2.9 * s, 3.0 * s)]) if cabinet_labels else np.zeros((grid.height, grid.width), dtype=bool)
    zeros = np.zeros_like(cabinet)
    units = TallUnitEvidence(front=zeros, fridge=zeros, low=zeros, cabinet=cabinet)
    solid = [box(-0.05 * s, -0.05 * s, 8.05 * s, 0.05 * s), box(3.95 * s, 0.0, 4.20 * s, 3.0 * s), box(-0.05 * s, 2.95 * s, 8.05 * s, 3.05 * s), box(-0.05 * s, 0.0, 0.05 * s, 3.0 * s), box(7.95 * s, 0.0, 8.05 * s, 3.0 * s)]
    if wall_points is True:
        solid.append(box(1.9 * s, y0 * s, 2.15 * s, 3.0 * s))
    elif wall_points:  # a weakly measured real wall: wall points over only the given share of the run (at its far end)
        solid.append(box(1.9 * s, (3.0 - float(wall_points) * length) * s, 2.15 * s, 3.0 * s))
    evidence = WallEvidence(bins=16, storey_height=2.4 * s, wall_bits=_full_wall_bits(grid, solid), all_bits=np.zeros((grid.height, grid.width), dtype=np.uint32))
    builtin = [box(2.0 * s, y0 * s, 2.9 * s, 3.0 * s)]
    return entities + [_face(2.0 * s, 3.0 * s, 2.0 * s, y0 * s)], rooms, free, valid, grid, units, evidence, builtin


@pytest.mark.parametrize("upm", (1.0, 0.8124))
def test_unsupported_wall_stretch_beside_kitchen_cabinets_is_a_cabinet_column(upm: float) -> None:
    entities, rooms, free, valid, grid, units, evidence, builtin = _column_scene(upm)
    settings = replace(SETTINGS, face_test=False, visibility_test=False)
    probe = box(1.7 * upm, 1.5 * upm, 2.3 * upm, 2.5 * upm)  # the face body, whichever side the solid side is taken
    off = build_double_walls(entities, [], rooms, free, valid, grid, upm, replace(settings, cabinet_walls=False), evidence=evidence, tall_units=units, builtins=builtin)
    assert _body(off).intersects(probe)  # old behaviour: the column's side panel is a wall
    result = build_double_walls(entities, [], rooms, free, valid, grid, upm, settings, evidence=evidence, tall_units=units, builtins=builtin)
    assert result.stats["cabinet_column_stretches"] == 1 and result.stats["cabinet_column_m"] == pytest.approx(1.6, abs=0.3)
    assert not _body(result).intersects(probe)
    assert _body(result).contains(Point(-0.15 * upm, 1.5 * upm))  # exterior walls stay
    assert len(result.tall_units) == 1 and result.tall_units[0][1] == "HS"
    x0, y0, x1, y1 = result.tall_units[0][0].bounds
    assert x0 == pytest.approx(2.0 * upm, abs=0.1 * upm) and x1 == pytest.approx(2.9 * upm, abs=0.1 * upm)


@pytest.mark.parametrize("upm", (1.0, 0.8124))
def test_cabinet_column_rule_keeps_real_walls_and_short_or_unlabelled_stretches(upm: float) -> None:
    settings = replace(SETTINGS, face_test=False, visibility_test=False)
    probe = box(1.7 * upm, 1.5 * upm, 2.3 * upm, 2.5 * upm)
    for kwargs in ({"wall_points": True}, {"wall_points": 0.3}, {"cabinet_labels": False}, {"length": 0.5}, {"far_floor": False}):
        entities, rooms, free, valid, grid, units, evidence, builtin = _column_scene(upm, **kwargs)
        result = build_double_walls(entities, [], rooms, free, valid, grid, upm, settings, evidence=evidence, tall_units=units, builtins=builtin)
        assert result.stats.get("cabinet_column_stretches", 0) == 0, kwargs
        if "length" not in kwargs:
            assert _body(result).intersects(probe), kwargs


@pytest.mark.parametrize("upm", (1.0, 0.8124))
def test_short_remnant_beside_a_removed_cabinet_column_leaves_no_wall_stub(upm: float) -> None:
    """The column stretch is removed; the 0.2 m of the run beyond it would be an island stub in the room: it goes too."""
    entities, rooms, free, valid, grid, units, evidence, builtin = _column_scene(upm, length=1.8, wall_points=False)
    # wall points over the last 0.2 m at the far end of the face keep that piece 'supported'
    solid = _mask(grid, [box(1.9 * upm, 2.8 * upm, 2.15 * upm, 3.0 * upm)])
    bits = evidence.wall_bits.copy()
    bits[solid] = np.uint32(0xFFFF)
    evidence = WallEvidence(bins=16, storey_height=2.4 * upm, wall_bits=bits, all_bits=evidence.all_bits)
    settings = replace(SETTINGS, face_test=False, visibility_test=False)
    result = build_double_walls(entities, [], rooms, free, valid, grid, upm, settings, evidence=evidence, tall_units=units, builtins=builtin)
    assert result.stats["cabinet_column_stretches"] == 1
    stub = box(1.7 * upm, 2.75 * upm, 2.3 * upm, 2.98 * upm)
    assert not _body(result).intersects(stub)  # no 0.2 m stub left behind
    keep = build_double_walls(entities, [], rooms, free, valid, grid, upm, replace(settings, cabinet_remnant_min_m=0.1), evidence=evidence, tall_units=units, builtins=builtin)
    assert _body(keep).intersects(stub)  # with the old limit the stub stays (this is what the rule removes)


@pytest.mark.parametrize("upm", [1.0, 0.8124])
def test_a_small_floor_cell_between_two_doors_is_kept_as_a_hall_a_dead_end_cell_is_not(upm: float) -> None:
    """A 1.4 m2 cell with two doors (a bit of corridor between doors) is floor of the flat: kept as "Flur"; with one door it is no hall."""
    from samcloud.ortho.vector.room_names import RoomName, RoomNameSettings, name_halls

    cell = affinity.scale(box(3.1, 12.6, 4.6, 13.55), upm, upm, origin=(0, 0))  # 1.5 x 0.95 m
    area = cell.area / (upm * upm)
    assert 1.2 < area < 1.5
    decisions = [RoomName(None, area, False, frozenset())]  # below the 2 m2 room minimum: dropped without the hall rule
    two = [((3.1 * upm, 13.0 * upm), math.pi / 2.0), ((4.6 * upm, 13.1 * upm), math.pi / 2.0)]
    assert name_halls([cell], decisions, two, upm, RoomNameSettings())[0].keep is True
    assert name_halls([cell], decisions, two[:1], upm, RoomNameSettings())[0].keep is False
    tiny = affinity.scale(box(0.0, 0.0, 0.6, 0.6), upm, upm, origin=(0, 0))  # 0.36 m2 niche: never a room
    assert name_halls([tiny], [RoomName(None, 0.36, False, frozenset())], [((0.0, 0.3 * upm), math.pi / 2.0), ((0.6 * upm, 0.3 * upm), math.pi / 2.0)], upm, RoomNameSettings())[0].keep is False


def test_a_rejected_span_is_not_filled_through_a_kept_passage() -> None:
    """A span the refinement classified as no opening continues its host wall only up to the band of a kept passage; the part
    beyond the passage (across it from the host) is not filled."""
    from samcloud.ortho.vector.double_walls import _close_spans

    def setup():
        face = _Face(0, np.asarray((4.402, 0.0)), np.asarray((4.398, 3.503)), {"type": "polyline", "points": []}, side=-1)
        return [face], [WallRun(0, 0.0, face.length, 0.203, "interior", "face")]

    span = {"center": (4.5, 3.913), "axis_angle": math.pi / 2, "width": 0.81}
    passage = Opening("passage", (5.52, 3.957), 2.43, 0.004, 0.207)
    faces, runs = setup()
    assert _close_spans([span], runs, faces, 1.0) == 1  # without kept openings: filled across the whole span
    fill = [run for run in runs if run.evidence == "door-fill"]
    assert len(fill) == 1 and faces[fill[0].face].end[1] > 4.3
    faces, runs = setup()
    _close_spans([span], runs, faces, 1.0, keep_open=[passage])
    fills = [faces[run.face] for run in runs if run.evidence == "door-fill"]
    assert fills, "the part on the host's side is still filled"
    for face in fills:
        top = max(face.start[1], face.end[1])
        assert top <= 3.957 - 0.5 * 0.207 + 0.01  # stops at the passage slab
    assert not any(min(face.start[1], face.end[1]) > 3.9 for face in fills)  # nothing beyond the passage
    # a window bridge is never cut
    faces, runs = setup()
    _close_spans([{**span, "bridge": True}], runs, faces, 1.0, keep_open=[passage])
    assert max(max(faces[run.face].start[1], faces[run.face].end[1]) for run in runs if run.evidence == "door-fill") > 4.3


def test_a_rejected_span_beyond_a_door_in_the_same_wall_line_is_still_filled() -> None:
    """A door in the filler's own wall line only interrupts it: the stub between the door and the corner is still closed."""
    from samcloud.ortho.vector.double_walls import _close_spans

    face = _Face(0, np.asarray((0.003, 0.0)), np.asarray((-0.002, 3.207)), {"type": "polyline", "points": []}, side=1)
    runs = [WallRun(0, 0.0, face.length, 0.304, "exterior", "exterior")]
    door = Opening("DOOR", (-0.148, 3.618), 0.82, math.pi / 2 + 0.003, 0.304)
    span = {"center": (-0.149, 4.233), "axis_angle": math.pi / 2, "width": 0.41}
    faces = [face]
    assert _close_spans([span], runs, faces, 1.0, keep_open=[door]) == 1
    fills = [faces[run.face] for run in runs if run.evidence == "door-fill"]
    assert len(fills) == 1
    ends = sorted((fills[0].start[1], fills[0].end[1]))
    assert ends[0] == pytest.approx(4.028, abs=0.01) and ends[1] == pytest.approx(4.458, abs=0.01)


def test_a_rejected_span_ending_at_the_wall_of_an_overshooting_door_is_filled_up_to_its_slab() -> None:
    """The door in the crossing wall is detected a few cm wider than the room, past the stub: the stub is still closed."""
    from samcloud.ortho.vector.double_walls import _close_spans

    face = _Face(0, np.asarray((2.351, 2.06)), np.asarray((2.393, 5.237)), {"type": "polyline", "points": []}, side=1)
    runs = [WallRun(0, 0.0, face.length, 0.192, "interior", "face")]
    door = Opening("DOOR", (1.752, 6.541), 1.413, math.pi - 0.021, 0.303)
    span = {"center": (2.33, 5.823), "axis_angle": 1.557, "width": 1.15}
    faces = [face]
    assert _close_spans([span], runs, faces, 1.0, keep_open=[door]) == 1
    fills = [faces[run.face] for run in runs if run.evidence == "door-fill"]
    assert len(fills) == 1
    top = max(fills[0].start[1], fills[0].end[1])
    assert 6.541 - 0.303 / 2.0 - 0.03 <= top <= 6.541 - 0.303 / 2.0 + 0.03



def test_two_windows_of_one_facade_are_joined_by_the_pier_between_them_unless_a_door_stands_there() -> None:
    """The pier between two windows of one exterior wall line is wall (1.48 m here, longer than the 1.0 m window bridge)."""
    from samcloud.ortho.vector.double_walls import _close_spans

    def setup():
        face = _Face(0, np.asarray((0.0, 0.003)), np.asarray((2.497, 0.001)), {"type": "polyline", "points": []}, side=1)
        return [face], [WallRun(0, 0.0, face.length, 0.21, "exterior", "exterior")]

    spans = [
        {"center": (3.384, 0.104), "axis_angle": 0.001, "width": 1.81, "bridge": True},
        {"center": (6.603, 0.106), "axis_angle": 0.001, "width": 1.66, "bridge": True},
    ]
    faces, runs = setup()
    _close_spans(list(spans), runs, faces, 1.0, keep_open=[])
    fills = sorted((min(faces[r.face].start[0], faces[r.face].end[0]), max(faces[r.face].start[0], faces[r.face].end[0])) for r in runs if r.evidence == "door-fill")
    assert len(fills) == 2
    assert fills[1][0] <= fills[0][1] + 0.01  # the second window bridge reaches the first: no gap in the facade
    # a door in the pier keeps it open
    door = Opening("DOOR", (4.997, 0.105), 0.9, 0.001, 0.21)
    faces, runs = setup()
    _close_spans(list(spans), runs, faces, 1.0, keep_open=[door])
    fills = sorted((min(faces[r.face].start[0], faces[r.face].end[0]), max(faces[r.face].start[0], faces[r.face].end[0])) for r in runs if r.evidence == "door-fill")
    assert fills[1][0] > fills[0][1] + 0.5


def test_a_door_beside_the_end_of_a_wall_is_hosted_in_that_walls_band() -> None:
    """A door detected on the outer face, 9 cm off the centreline of the wall it continues: the door-host piece takes the band of
    that wall (no 9 cm step between the wall and the door's host)."""
    from samcloud.ortho.vector.double_walls import _enforce_door_spans, _run_centre

    face = _Face(0, np.asarray((0.0, -0.105)), np.asarray((3.012, -0.098)), {"type": "polyline", "points": []}, side=1)
    runs = [WallRun(0, 0.0, face.length, 0.21, "exterior", "exterior")]
    faces = [face]
    door = Opening("DOOR", (3.751, 0.093), 0.92, 0.002, 0.21)
    _enforce_door_spans(runs, faces, [door], 1.0, SETTINGS)
    host = [r for r in runs if r.evidence == "door-host"]
    assert len(host) == 1
    centre = _run_centre(host[0], faces[host[0].face], host[0].start + 0.3)
    wall_line_y = 0.0 + (centre[0] / 3.012) * 0.007  # the wall's centreline continued
    assert abs(centre[1] - wall_line_y) <= 0.01
