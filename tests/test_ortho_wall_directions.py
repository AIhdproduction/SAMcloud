"""Indoor walls off the dominant axis pair: extraction, joints, thin walls, rooms (synthetic data only)."""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import uuid

import ezdxf
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from samcloud.dxf.writer import export_plan
from samcloud.ortho.plan import CleanPlanSettings, _indoor_wall_axis, generate_clean_plan, load_clean_plan_settings
from samcloud.ortho.vector.double_walls import DoubleWallSettings, build_double_walls
from samcloud.ortho.vector.raster_grid import RasterGrid
from samcloud.ortho.vector.wall_directions import DirectionSettings, add_direction_faces, candidate_directions, trim_rooms_to_faces
from samcloud.ortho.vector.wall_lines import WallFace, extract_axis_wall_faces, face_entities

from indoor_walls_synthetic import SCENES, evaluate_plan, write_scene

DIRECTIONS = DirectionSettings()
TOP = 2.36


@pytest.fixture
def work_dir(request: pytest.FixtureRequest) -> Path:
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"wall_directions_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("wall_directions")


def _face_points(segments: list[tuple[tuple[float, float], tuple[float, float]]], *, noise: float = 0.008, seed: int = 7, top: float = TOP, bottom: float = 0.30, skip: list[tuple[int, float, float]] = ()) -> np.ndarray:
    """x, y, z_above_floor samples of vertical wall faces (2 cm along, 6 cm up) with Gaussian noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for index, (first, second) in enumerate(segments):
        a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        length = float(np.linalg.norm(b - a))
        values = np.arange(0.01, length, 0.02)
        for skip_index, low, high in skip:
            if skip_index == index:
                values = values[(values < low) | (values > high)]
        xy = a + values[:, None] * (b - a) / length
        for z in np.arange(bottom, top, 0.06):
            rows.append(np.column_stack((xy, np.full(len(xy), z))))
    points = np.vstack(rows)
    points[:, :2] += rng.normal(0.0, noise, (len(points), 2))
    return points


def _ring_segments(polygon: Polygon) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    coords = list(polygon.exterior.coords)
    return list(zip(coords[:-1], coords[1:], strict=True))


def _grid_for(points: np.ndarray) -> RasterGrid:
    low, high = points[:, :2].min(axis=0) - 0.5, points[:, :2].max(axis=0) + 0.5
    return RasterGrid(float(low[0]), float(high[1]), 0.02, int(math.ceil((high[0] - low[0]) / 0.02)), int(math.ceil((high[1] - low[1]) / 0.02)))


def _angle(line: LineString) -> float:
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0


def _angle_error(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def _match(faces: list[WallFace], truth: LineString) -> WallFace | None:
    """Face parallel (< 1 deg) to ``truth`` whose midpoint lies within 3 cm of the truth line."""
    truth_angle = _angle(truth)
    for face in faces:
        if face.line is None or _angle_error(_angle(face.line), truth_angle) >= 1.0:
            continue
        middle = face.line.interpolate(0.5, normalized=True)
        if truth.distance(middle) < 0.03 and face.length >= 0.5 * truth.length:
            return face
    return None


# --- extraction ------------------------------------------------------------------------------------------------


def test_manhattan_room_is_bit_identical_with_the_direction_pass() -> None:
    room = Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 3.0), (3.0, 3.0), (3.0, 6.0), (0.0, 6.0)])
    angle = math.radians(17.0)
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    points = _face_points(_ring_segments(room))
    points[:, :2] = points[:, :2] @ rotation.T
    grid = _grid_for(points)
    plain = extract_axis_wall_faces(points, grid, units_per_metre=1.0)
    extended = extract_axis_wall_faces(points, grid, units_per_metre=1.0, extra_directions=DIRECTIONS)
    assert len(plain) == len(extended)
    for first, second in zip(plain, extended, strict=True):
        assert np.array_equal(np.asarray(first.line.coords), np.asarray(second.line.coords))
        assert second.axis_snapped and second.facing is None


@pytest.mark.parametrize("noise", [0.005, 0.015])
def test_slanted_30_and_45_degree_walls_are_measured(noise: float) -> None:
    room = Polygon([(0.0, 0.0), (7.0, 0.0), (7.0, 2.0), (5.0, 4.0), (4.0 * math.tan(math.radians(30.0)), 4.0)])
    points = _face_points(_ring_segments(room), noise=noise, skip=[(0, 2.0, 2.6)])  # partial occlusion
    faces = extract_axis_wall_faces(points, _grid_for(points), units_per_metre=1.0, extra_directions=DIRECTIONS)
    for first, second in _ring_segments(room):
        truth = LineString((first, second))
        face = _match(faces, truth)
        assert face is not None, f"missing face {truth}"
        assert _angle_error(_angle(face.line), _angle(truth)) < 1.0
        for end in (Point(face.line.coords[0]), Point(face.line.coords[-1])):
            assert truth.distance(end) < 0.03
        axis_like = min(_angle_error(_angle(truth), 0.0), _angle_error(_angle(truth), 90.0)) < 1.0
        assert face.axis_snapped == axis_like
    # Corners are closed as line intersections, also at the acute 60 degree corner.
    for vertex in list(room.exterior.coords)[:-1]:
        assert min(min(Point(face.line.coords[0]).distance(Point(vertex)), Point(face.line.coords[-1]).distance(Point(vertex))) for face in faces) < 0.03


def test_trapezoid_room_has_four_faces_no_duplicates() -> None:
    room = Polygon([(0.0, 0.0), (6.0, 0.0), (5.0, 4.0), (1.0, 4.0)])
    points = _face_points(_ring_segments(room), noise=0.010)
    faces = extract_axis_wall_faces(points, _grid_for(points), units_per_metre=1.0, extra_directions=DIRECTIONS)
    assert len(faces) == 4
    for first, second in _ring_segments(room):
        assert _match(faces, LineString((first, second))) is not None


def test_slanted_partition_replaces_axis_stair_steps() -> None:
    outline = box(0.0, 0.0, 8.0, 5.0)
    left = Polygon([(0.0, 0.0), (3.44, 0.0), (4.44, 5.0), (0.0, 5.0)])
    right = Polygon([(3.56, 0.0), (8.0, 0.0), (8.0, 5.0), (4.56, 5.0)])
    points = _face_points([*_ring_segments(left), *_ring_segments(right)], noise=0.008)
    plain = extract_axis_wall_faces(points, _grid_for(points), units_per_metre=1.0)
    faces = extract_axis_wall_faces(points, _grid_for(points), units_per_metre=1.0, extra_directions=DIRECTIONS)
    inside = Polygon([(3.3, 0.3), (3.7, 0.3), (4.7, 4.7), (4.3, 4.7)])
    # Projection binning on the axis grid leaves short stair-step faces along the slanted wall ...
    assert any(face.axis_snapped and inside.contains(face.line.interpolate(0.5, normalized=True)) for face in plain)
    # ... which give way to the two measured faces of the partition.
    assert not any(face.axis_snapped and inside.contains(face.line.interpolate(0.5, normalized=True)) for face in faces)
    for truth in (LineString(((3.44, 0.0), (4.44, 5.0))), LineString(((3.56, 0.0), (4.56, 5.0)))):
        assert _match(faces, truth) is not None
    assert outline.area > 0.0


def test_thin_slanted_wall_keeps_both_faces_with_opposite_facing() -> None:
    room_a = Polygon([(0.0, 0.0), (5.95, 0.0), (8.28, 5.0), (0.0, 5.0)])
    offset = 0.10 / math.cos(math.radians(25.0))
    room_b = Polygon([(5.95 + offset, 0.0), (11.0, 0.0), (11.0, 5.0), (8.28 + offset, 5.0)])
    points = _face_points([*_ring_segments(room_a), *_ring_segments(room_b)], noise=0.008)
    faces = extract_axis_wall_faces(points, _grid_for(points), units_per_metre=1.0, extra_directions=DIRECTIONS)
    slanted = [face for face in faces if not face.axis_snapped]
    assert len(slanted) == 2
    first, second = slanted
    normal = np.asarray(first.facing)
    separation = float(np.dot(np.asarray(second.line.coords[0]) - np.asarray(first.line.coords[0]), normal))
    assert abs(abs(separation) - 0.10) < 0.02
    assert float(np.dot(np.asarray(first.facing), np.asarray(second.facing))) < -0.99
    assert all("_facing" in entity for entity in face_entities(slanted))


def test_open_door_leaf_is_not_a_slanted_wall() -> None:
    room = box(0.0, 0.0, 6.0, 4.0)
    # A door gap in the bottom wall, an open leaf at 35 degrees from its hinge (2.00 m high only).
    segments = _ring_segments(room)
    points = _face_points(segments, skip=[(0, 2.0, 2.9)])
    hinge = np.asarray((2.9, 0.0))
    tip = hinge + 0.85 * np.asarray((-math.cos(math.radians(35.0)), math.sin(math.radians(35.0))))
    leaf = _face_points([(tuple(hinge), tuple(tip))], top=2.0, bottom=0.05, seed=3)
    lintel = _face_points([((2.0, 0.0), (2.9, 0.0))], bottom=2.06, seed=4)
    evidence = np.vstack((points, leaf, lintel))
    plain = extract_axis_wall_faces(evidence, _grid_for(evidence), units_per_metre=1.0)
    faces = extract_axis_wall_faces(evidence, _grid_for(evidence), units_per_metre=1.0, extra_directions=DIRECTIONS)
    assert len(faces) == len(plain)
    assert all(face.axis_snapped for face in faces)


def test_candidate_directions_skip_axis_directions() -> None:
    room = box(0.0, 0.0, 6.0, 4.0)
    points = _face_points(_ring_segments(room))
    assert candidate_directions(points[:, :2], points[:, 2], 0.0, 1.0, DIRECTIONS) == []
    slanted = _face_points([((0.0, 0.0), (3.0, 3.0 * math.tan(math.radians(30.0))))])
    directions = candidate_directions(slanted[:, :2], slanted[:, 2], 0.0, 1.0, DIRECTIONS)
    assert directions and abs(math.degrees(directions[0]) - 30.0) < 0.5


def test_disabled_pass_and_short_isolated_pieces_change_nothing() -> None:
    room = box(0.0, 0.0, 6.0, 4.0)
    stub = ((2.0, 1.5), (2.0 + 0.6 * math.cos(math.radians(30.0)), 1.5 + 0.6 * math.sin(math.radians(30.0))))
    points = _face_points([*_ring_segments(room), stub])
    grid = _grid_for(points)
    plain = extract_axis_wall_faces(points, grid, units_per_metre=1.0)
    assert len(extract_axis_wall_faces(points, grid, units_per_metre=1.0, extra_directions=replace(DIRECTIONS, enabled=False))) == len(plain)
    # A free-standing 0.6 m slanted piece (no joined ends, shorter than a wall) is not a wall.
    assert len(add_direction_faces(plain, points, 0.0, grid, 1.0, DIRECTIONS)) == len(plain)


def test_dominant_axis_ignores_long_slanted_faces() -> None:
    lines = [LineString(((0.0, 0.0), (8.0, 8.0 * math.tan(math.radians(25.0))))), LineString(((0.0, 0.0), (4.0, 0.0)))]
    assert abs(_indoor_wall_axis(lines, [False, True])) < 1e-9
    assert abs(math.degrees(_indoor_wall_axis(lines)) - 25.0) < 1e-6


def test_direction_settings_load_from_config() -> None:
    settings = load_clean_plan_settings()
    assert isinstance(settings.indoor_wall_directions, DirectionSettings)
    assert CleanPlanSettings().indoor_wall_directions == DirectionSettings()
    assert DirectionSettings.from_mapping({"min_length_m": "1.5", "enabled": False}) == replace(DirectionSettings(), min_length_m=1.5, enabled=False)
    assert DirectionSettings.from_mapping(False).enabled is False


# --- rooms and wall bodies -----------------------------------------------------------------------------------------


def test_stair_step_room_teeth_beyond_a_slanted_face_are_trimmed() -> None:
    stairs = Polygon([(0, 0), (7, 0), (7, 2.3), (6.6, 2.3), (6.6, 2.7), (6.2, 2.7), (6.2, 3.1), (5.8, 3.1), (5.8, 3.5), (5.4, 3.5), (5.4, 4.0), (0, 4.0)])
    face = WallFace("line", np.empty((0, 2)), line=LineString(((7.0, 2.0), (5.0, 4.0))), axis_snapped=False)
    trimmed = trim_rooms_to_faces([stairs], [face], 1.0)[0]
    truth = Polygon([(0, 0), (7, 0), (7, 2.0), (5.0, 4.0), (0, 4.0)])
    assert trimmed.symmetric_difference(truth).area < 0.25
    assert trimmed.area < stairs.area
    axis_face = WallFace("line", np.empty((0, 2)), line=LineString(((0.0, 0.0), (7.0, 0.0))), axis_snapped=True)
    unchanged = [stairs]
    assert trim_rooms_to_faces(unchanged, [axis_face], 1.0) is unchanged


def _mask(grid: RasterGrid, polygons: list[Polygon]) -> np.ndarray:
    import shapely

    rows, cols = np.mgrid[0:grid.height, 0:grid.width]
    xs, ys = grid.min_x + (cols + 0.5) * grid.pixel_size, grid.max_y - (rows + 0.5) * grid.pixel_size
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for polygon in polygons:
        mask |= shapely.contains_xy(polygon, xs, ys)
    return mask


def _entity(first: tuple[float, float], second: tuple[float, float], facing: tuple[float, float] | None = None) -> dict:
    entity = {"type": "polyline", "points": [list(first), list(second)], "closed": False, "_source": "measured"}
    if facing is not None:
        entity["_facing"] = list(facing)
    return entity


def test_oblique_exterior_corner_gets_a_mitre_joint() -> None:
    room = Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (4.0, 4.0), (0.0, 4.0)])
    coords = list(room.exterior.coords)
    entities = [_entity(first, second) for first, second in zip(coords[:-1], coords[1:], strict=True)]
    grid = RasterGrid(-1.0, 5.0, 0.02, 400, 300)
    free = _mask(grid, [room])
    valid = _mask(grid, [room.buffer(0.05, join_style=2)])
    result = build_double_walls(entities, [], [room], free, valid, grid, 1.0, replace(DoubleWallSettings(double_line=True), exterior_thickness_m=0.30, unmeasured_thickness_m=0.0))
    from shapely.ops import unary_union

    body = unary_union(result.polygons)
    outer = room.buffer(0.30, join_style=2)
    # The wall body is the 0.30 m shell, including both oblique (135 degree) corners, without notches or spikes.
    assert body.symmetric_difference(outer.difference(room)).area < 0.05
    assert len(result.polygons) == 1


def test_thin_wall_pair_with_measured_facing_gets_its_thickness_although_floor_closing_covers_it() -> None:
    direction = np.asarray((math.cos(math.radians(65.0)), math.sin(math.radians(65.0))))
    normal = np.asarray((-direction[1], direction[0]))
    start = np.asarray((3.0, 0.0))
    first = (tuple(start), tuple(start + direction * 5.0))
    second = (tuple(start - normal * 0.10), tuple(start - normal * 0.10 + direction * 5.0))
    room = box(-1.0, -0.5, 8.0, 5.0)
    grid = RasterGrid(-1.5, 5.5, 0.02, 500, 320)
    free = _mask(grid, [room])  # the closed floor mask has swallowed the thin wall
    valid = free.copy()
    entities = [_entity(*first, facing=tuple(normal)), _entity(*second, facing=tuple(-normal))]
    result = build_double_walls(entities, [], [], free, valid, grid, 1.0, replace(DoubleWallSettings(double_line=True), keep_isolated_faces=True))
    thicknesses = [piece["thickness"] for piece in result.pieces]
    assert thicknesses and all(abs(value - 0.10) < 0.01 for value in thicknesses)
    # Without the measured facing both faces stay undecided light partitions.
    plain = build_double_walls([_entity(*first), _entity(*second)], [], [], free, valid, grid, 1.0, replace(DoubleWallSettings(double_line=True), keep_isolated_faces=True))
    assert not any(abs(piece["thickness"] - 0.10) < 0.01 and piece.get("evidence") == "face" for piece in plain.pieces)


# --- full indoor clean plan on synthetic scans ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rectangle_17", "slanted_30_45", "trapezoid", "l_room", "slanted_partition", "thin_and_thick", "thin_axis_partition"])
def test_clean_plan_walls_and_rooms_on_synthetic_scans(name: str, work_dir: Path) -> None:
    scene = SCENES[name]()
    project = work_dir / name
    level = write_scene(scene, project)
    settings = load_clean_plan_settings()
    settings = replace(settings, indoor_wall_directions=replace(settings.indoor_wall_directions, enabled=True))
    if name in {"rectangle_17", "thin_axis_partition"}:
        # Both faces of a thin AXIS wall need the optional side-aware suppression.
        settings = replace(settings, indoor_wall_directions=replace(settings.indoor_wall_directions, axis_thin_wall_faces=True))
    result = generate_clean_plan(project, level=level, mode="indoor", settings=settings)
    metrics = evaluate_plan(scene, result)
    assert metrics["missing_faces"] == 0, metrics
    assert metrics["angle_error_max"] < 1.0, metrics
    assert metrics["face_error_p90_max"] < 0.03, metrics
    assert metrics["thickness_error_max"] < 0.03, metrics
    assert metrics["room_count"] == len(scene.rooms), metrics
    assert metrics["room_iou_min"] >= 0.95 and metrics["rooms_valid"], metrics
    assert metrics["wall_bodies"] == 1, metrics
    assert metrics["stray_wall_area"] < 0.05, metrics
    # Slanted walls reach the DXF unchanged (closed double-line bodies at their measured angle) and the file audits clean.
    document = ezdxf.readfile(export_plan(result, project / "dxf", coordinates="local"))
    assert not document.audit().has_errors
    assert len(document.modelspace().query('LWPOLYLINE[layer=="A-WALL"]')) + len(document.modelspace().query('POLYLINE[layer=="A-WALL"]')) >= 1


def test_thin_axis_wall_without_the_option_keeps_todays_learned_thickness(work_dir: Path) -> None:
    scene = SCENES["thin_axis_partition"]()
    project = work_dir / "thin_axis_default"
    level = write_scene(scene, project)
    legacy = load_clean_plan_settings()  # the scene has no measured thickness; its walls are the 0.30 m default of old
    legacy = replace(legacy, walls=replace(legacy.walls, unmeasured_thickness_m=0.0))
    metrics = evaluate_plan(scene, generate_clean_plan(project, level=level, mode="indoor", settings=legacy))
    # Default: one face of the 0.10 m wall survives the 0.15 m suppression (the option above measures both).
    assert metrics["room_count"] == 2 and metrics["thickness_error_max"] <= 0.03


# --- round 2: unscanned edges, unscanned corners, exterior thickness from reveals ------------------------------------


def _bits_raster(grid: RasterGrid, polygons: list[Polygon], bins: tuple[int, ...]) -> np.ndarray:
    value = 0
    for item in bins:
        value |= 1 << item
    return np.where(_mask(grid, polygons), value, 0).astype(np.int64)


def test_ceiling_piece_on_the_edge_of_unscanned_space_is_dropped() -> None:
    from samcloud.ortho.vector.wall_directions import drop_unobserved_edge_pieces

    grid = RasterGrid(-1.0, 4.0, 0.02, 300, 250)
    observed = _mask(grid, [box(-1.0, -1.0, 2.0, 4.0)])  # nothing measured for x > 2
    edge = WallFace("line", np.empty((0, 2)), line=LineString(((2.0, 0.0), (2.0, 3.0))), axis_snapped=True, source="ceiling")
    window_wall = WallFace("line", np.empty((0, 2)), line=LineString(((0.0, 3.0), (2.0, 3.0))), axis_snapped=True, source="ceiling")
    # The window wall has a lintel / frame strip (points high up); the scan edge has nothing vertical.
    bitmask = _bits_raster(grid, [box(0.0, 2.97, 2.0, 3.03)], (12, 13, 14))
    observed &= ~_mask(grid, [box(-1.0, 3.0, 2.0, 4.0)])
    kept, dropped = drop_unobserved_edge_pieces([edge, window_wall], bitmask, observed, grid, 1.0)
    assert dropped == 1 and kept == [window_wall]
    # Observed space beyond the piece: it stays even without vertical points (ordinary ceiling completion).
    kept, dropped = drop_unobserved_edge_pieces([edge], bitmask, _mask(grid, [box(-1.0, -1.0, 5.0, 4.0)]), grid, 1.0)
    assert dropped == 0


def test_walls_meet_in_an_unscanned_corner_but_not_in_observed_space() -> None:
    from samcloud.ortho.vector.wall_directions import close_corners_through_unobserved

    grid = RasterGrid(-1.0, 6.0, 0.02, 400, 350)
    # Both measured walls end where the unscanned square begins.
    bottom = WallFace("line", np.empty((0, 2)), line=LineString(((5.0, 0.0), (0.82, 0.0))))
    left = WallFace("line", np.empty((0, 2)), line=LineString(((0.0, 5.0), (0.0, 0.82))))
    unobserved = _mask(grid, [box(-0.8, -0.8, 0.8, 0.8)])
    observed = _mask(grid, [box(-1.0, -1.0, 6.0, 6.0)]) & ~unobserved
    faces, closed = close_corners_through_unobserved([bottom, left], observed, grid, 1.0)
    assert closed == 1
    assert Point(faces[0].line.coords[-1]).distance(Point(0.0, 0.0)) < 1e-6 and Point(faces[1].line.coords[-1]).distance(Point(0.0, 0.0)) < 1e-6
    assert bottom.line.coords[-1] == (0.82, 0.0)  # the caller's faces are not modified
    faces, closed = close_corners_through_unobserved([bottom, left], _mask(grid, [box(-1.0, -1.0, 6.0, 6.0)]), grid, 1.0)
    assert closed == 0 and faces[0] is bottom


def test_unscanned_corner_scene_has_no_false_wall_and_a_full_room(work_dir: Path) -> None:
    scene = SCENES["unscanned_corner"]()
    project = work_dir / "unscanned_corner"
    level = write_scene(scene, project)
    settings = load_clean_plan_settings()
    # The unscanned-pocket rule for hidden-wall room edges is an option (off by default, see config).
    settings = replace(settings, walls=replace(settings.walls, room_edge_unscanned_pocket=True))
    metrics = evaluate_plan(scene, generate_clean_plan(project, level=level, mode="indoor", settings=settings))
    assert metrics["stray_wall_area"] < 0.05, metrics
    assert metrics["room_iou_min"] >= 0.99 and metrics["room_count"] == 1, metrics
    assert metrics["wall_bodies"] == 1, metrics


def test_exterior_thickness_from_window_reveals_without_outer_face() -> None:
    from samcloud.ortho.vector.double_walls import WallEvidence
    from samcloud.ortho.vector.indoor import Opening

    room = box(0.0, 0.0, 6.0, 4.0)
    coords = list(room.exterior.coords)
    entities = [_entity(first, second) for first, second in zip(coords[:-1], coords[1:], strict=True)]
    entities[1] = _entity((6.0, 0.0), (6.0, 1.1))
    entities.append(_entity((6.0, 2.9), (6.0, 4.0)))
    grid = RasterGrid(-1.0, 5.0, 0.02, 400, 300)
    free = _mask(grid, [room])
    valid = _mask(grid, [room.buffer(0.05, join_style=2)])
    # Window 1.8 m wide in the right wall, frame 0.08 m deep in the middle of a 0.40 m wall:
    # the reveal (wall points) reaches 0.16 m into the wall at both jambs.
    jambs = [box(6.0, 1.08, 6.16, 1.12), box(6.0, 2.88, 6.16, 2.92)]
    wall_bits = _bits_raster(grid, [*jambs, room.exterior.buffer(0.02)], tuple(range(2, 15)))
    evidence = WallEvidence(wall_bits=wall_bits, all_bits=wall_bits, bins=16, storey_height=2.5, section=None, low=None, floor=free, door=None, cameras=None, depth_points=None)
    window = Opening("WINDOW", (6.0, 2.0), 1.8, math.pi / 2.0, 0.04)
    settings = replace(DoubleWallSettings(double_line=True), face_test=False, visibility_test=False, exterior_from_reveals=True, unmeasured_thickness_m=0.0)
    result = build_double_walls(entities, [window], [room], free, valid, grid, 1.0, settings, evidence=evidence)
    assert result.stats.get("double_wall_exterior_source") == "window-reveal"
    assert abs(result.stats["double_wall_exterior_thickness_m"] - 0.40) <= 0.03
    exterior = [piece for piece in result.pieces if piece["kind"] == "exterior"]
    assert exterior and all(abs(piece["thickness"] - result.stats["double_wall_exterior_thickness_m"]) < 1e-6 for piece in exterior), [(p["evidence"], p["thickness"]) for p in exterior]
    assert any(piece["evidence"] == "exterior-reveal" for piece in exterior)
    plain = build_double_walls(entities, [window], [room], free, valid, grid, 1.0, replace(settings, exterior_from_reveals=False), evidence=evidence)
    assert abs(plain.stats["double_wall_exterior_thickness_m"] - 0.30) <= 1e-6


def test_shallow_short_slant_within_scan_smear_is_not_a_new_direction() -> None:
    # v10 (unfused cloud, wall faces smeared 85-115 mm): false 6-11 degree faces of 0.5-1.95 m appeared, whose lateral
    # deviation (length x sin(angle)) stayed below 0.35 m.  A face must deviate by >= min_slant_m (0.40 m) to count.
    room = box(0.0, 0.0, 6.0, 4.0)
    shallow = ((1.0, 1.0), (1.0 + 1.8 * math.cos(math.radians(8.0)), 1.0 + 1.8 * math.sin(math.radians(8.0))))
    points = _face_points([*_ring_segments(room), shallow])
    grid = _grid_for(points)
    faces = extract_axis_wall_faces(points, grid, units_per_metre=1.0, extra_directions=DIRECTIONS)
    assert all(face.axis_snapped for face in faces)
    # The same evidence with the rule relaxed does produce the shallow face: the rule is what rejects it.
    relaxed = extract_axis_wall_faces(points, grid, units_per_metre=1.0, extra_directions=replace(DIRECTIONS, min_slant_m=0.0))
    assert any(not face.axis_snapped for face in relaxed)
