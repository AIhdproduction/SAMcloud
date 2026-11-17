"""Double-line indoor walls (samcloud.ortho.vector.double_walls) on synthetic plans."""

from __future__ import annotations

import math
from dataclasses import replace

import ezdxf
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from samcloud.ortho.vector.double_walls import (
    DoubleWallSettings,
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
