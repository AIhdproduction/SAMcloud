"""Behavioural tests for clean raster-to-CAD vector geometry."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import uuid

import cv2
import ezdxf
import numpy as np
import pytest
from scipy import ndimage
from shapely import affinity, coverage_is_valid, union_all
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree

from samcloud.ortho.vector import (
    AreaParams,
    Opening,
    RasterGrid,
    add_symbol_blocks,
    boundary_lines,
    build_partition,
    clean_label_raster,
    cut_openings,
    detect_crowns,
    detect_openings,
    furniture_entities,
    fit_block_insert,
    guess_fixture,
    opening_entities,
    mask_to_polygons,
    regularize_polygon,
    stair_entities,
    trace_lines,
    tree_entities,
    wall_polygons_from_section,
)
from samcloud.ortho.vector.indoor import wall_faces_from_section, walls_from_faces
from samcloud.ortho.vector.footprints import ObjectInstance, clutter_rectangle, fit_footprint, height_step_details, object_instances, refine_dining_instances, refit_clipped
from samcloud.ortho.vector.wall_lines import WallFace, break_faces_at_openings, extend_faces_for_window_gaps, extract_axis_wall_faces, extract_wall_faces, face_entities
from samcloud.ortho.plan import FINE_LABEL_GERMAN_NAMES, _align_insert_to_wall, _furniture_display_name, _furniture_instances
from samcloud.ortho.vector.areas import _fill_small_holes, _remove_small_components
from samcloud.ortho.vector.lines import _join_gaps
from samcloud.ortho.vector.vertical import VerticalEvidence, VerticalSettings, detect_door_leaf, vertical_evidence
from samcloud.ortho.vector.ceiling import ceiling_rooms, ceiling_wall_pieces, snap_exterior_faces_to_flucht
from samcloud.ortho.vector.vertical import filter_builtins_to_rooms, tidy_builtins
from samcloud.ortho.plan import _collinear_face_gap_openings, _normalise_door_openings, _snap_openings_to_faces


@pytest.fixture(scope="module")
def work_dir(request: pytest.FixtureRequest) -> Path:
    """Return an isolated directory compatible with pytest without tmpdir."""
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"ortho_vector_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("ortho_vector")


def test_noisy_rectangle_becomes_exact_rectangle() -> None:
    grid = RasterGrid(-2.0, 12.0, 0.05, 320, 280)
    rectangle = affinity.rotate(Polygon([(0, 0), (12, 0), (12, 8), (0, 8)]), 23.0, origin="centroid")
    mask = _boundary_noise(_rasterise_polygon(rectangle, grid), seed=13)
    result = regularize_polygon(max(mask_to_polygons(mask, grid), key=lambda polygon: polygon.area), min_edge_length=0.2)
    points = list(result.exterior.coords)[:-1]
    assert len(points) == 4
    assert abs(result.area - rectangle.area) / rectangle.area < 0.02
    assert abs(_orientation(points) - math.radians(23.0)) < math.radians(1.0)
    assert all(abs(_interior_angle(points[index - 1], points[index], points[(index + 1) % 4]) - 90.0) < 0.5 for index in range(4))


def test_l_shaped_building_keeps_six_right_angles() -> None:
    grid = RasterGrid(-1.0, 8.0, 0.05, 180, 180)
    source = Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)])
    result = regularize_polygon(max(mask_to_polygons(_boundary_noise(_rasterise_polygon(source, grid), seed=14), grid), key=lambda polygon: polygon.area), min_edge_length=0.1)
    points = list(result.exterior.coords)[:-1]
    assert len(points) == 6
    angles = [_interior_angle(points[index - 1], points[index], points[(index + 1) % len(points)]) for index in range(len(points))]
    assert all(min(abs(angle - 90.0), abs(angle - 270.0)) <= 0.5 for angle in angles)


def test_regularize_keeps_non_orthogonal_edge() -> None:
    grid = RasterGrid(-1.0, 7.0, 0.05, 160, 160)
    source = Polygon([(0, 0), (5, 0), (5, 3), (3, 5), (0, 5)])
    result = regularize_polygon(max(mask_to_polygons(_boundary_noise(_rasterise_polygon(source, grid), seed=15), grid), key=lambda polygon: polygon.area), min_edge_length=0.1)
    angles = [_edge_angle(first, second) for first, second in zip(result.exterior.coords, list(result.exterior.coords)[1:], strict=False)]
    assert any(abs((angle % 180.0) - 45.0) < 1.0 or abs((angle % 180.0) - 135.0) < 1.0 for angle in angles)


def test_partition_shares_borders_no_gaps_no_overlaps() -> None:
    rng = np.random.default_rng(22)
    grid = RasterGrid(0.0, 10.0, 0.05, 200, 200)
    labels = np.zeros((200, 200), dtype=np.uint8)
    first_border = 48 + rng.integers(-2, 3, size=200)
    second_border = 102 + rng.integers(-2, 3, size=200)
    for row in range(200):
        labels[row, : first_border[row]] = 1
        labels[row, first_border[row] : second_border[row]] = 2
        labels[row, second_border[row] : 150] = 3
        labels[row, 150:] = 4
    partition = build_partition(labels, grid, class_to_layer={1: "road", 2: "walk", 3: "green", 4: "water"}, tolerances={"road": 0.04, "walk": 0.08, "green": 0.06, "water": 0.05}, partition_tolerance=0.04)
    polygons = [polygon for values in partition.values() for polygon in values]
    assert coverage_is_valid(polygons)
    merged = union_all(polygons)
    assert abs(sum(polygon.area for polygon in polygons) - merged.area) / merged.area < 1e-6
    expected = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert merged.symmetric_difference(expected).area <= grid.pixel_size * expected.length
    index = STRtree(polygons)
    touching_pairs: set[tuple[int, int]] = set()
    for first_index, first in enumerate(polygons):
        for second_index in index.query(first):
            second_index = int(second_index)
            if second_index <= first_index:
                continue
            second = polygons[second_index]
            if first.touches(second):
                touching_pairs.add((first_index, second_index))
    assert touching_pairs
    for first_index, second_index in touching_pairs:
        shared = polygons[first_index].boundary.intersection(polygons[second_index].boundary)
        assert shared.area == 0.0 and shared.length > 0.0


def test_small_blobs_removed_and_holes_filled_except_courtyard() -> None:
    grid = RasterGrid(0.0, 10.0, 0.1, 100, 100)
    labels = np.ones((100, 100), dtype=np.uint8)
    labels[5:7, 5:7] = 2
    labels[30:33, 30:33] = 0
    labels[50:70, 50:70] = 0
    labels[40:43, 40:43] = 3
    cleaned = clean_label_raster(labels, class_params={1: AreaParams(min_area=0.1, courtyard_min_area=0.2), 2: AreaParams(min_area=0.1, courtyard_min_area=0.2), 3: AreaParams(courtyard_min_area=0.2)}, grid=grid)
    assert np.count_nonzero(cleaned[5:7, 5:7] == 1) == 4
    assert np.count_nonzero(cleaned[30:33, 30:33] == 1) == 9
    assert np.count_nonzero(cleaned[50:70, 50:70] == 0) == 400
    assert np.count_nonzero(cleaned[40:43, 40:43] == 3) == 9


def test_remove_small_components_is_fast_on_many_fragments() -> None:
    rng = np.random.default_rng(20261110)
    mask = np.zeros((4000, 4000), dtype=bool)
    fragments = rng.choice(898 * 898, size=20_000, replace=False)
    rows = (fragments // 898 + 1) * 4
    cols = (fragments % 898 + 1) * 4
    for row, col in zip(rows, cols, strict=True):
        mask[row : row + 2, col : col + 2] = True
    mask[3600:3900, 3600:3900] = True

    def old_remove_small_components(source: np.ndarray) -> np.ndarray:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(source.astype(np.uint8), connectivity=8)
        result = source.copy()
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] * 0.1**2 < 0.1:
                result[labels == index] = False
        return result

    crop = mask[:300, :300]
    assert np.array_equal(_remove_small_components(crop, 0.1, 0.1), old_remove_small_components(crop))

    started = time.perf_counter()
    result = _remove_small_components(mask, 0.1, 0.1)
    elapsed = time.perf_counter() - started
    expected = np.zeros_like(mask)
    expected[3600:3900, 3600:3900] = True
    assert np.array_equal(result, expected)
    assert elapsed < 5.0


def test_fill_small_holes_is_fast_on_many_fragments() -> None:
    rng = np.random.default_rng(20261110)
    mask = np.ones((4000, 4000), dtype=bool)
    fragments = rng.choice(898 * 898, size=20_000, replace=False)
    rows = (fragments // 898 + 1) * 4
    cols = (fragments % 898 + 1) * 4
    for row, col in zip(rows, cols, strict=True):
        mask[row : row + 2, col : col + 2] = False
    mask[3600:3900, 3600:3900] = False
    fillable = np.ones_like(mask)

    def old_fill_small_holes(source: np.ndarray, source_fillable: np.ndarray) -> np.ndarray:
        filled = ndimage.binary_fill_holes(source)
        holes = filled & ~source
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
        result = source.copy()
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] * 0.1**2 < 0.1:
                result[(labels == index) & source_fillable] = True
        return result

    crop = mask[:300, :300]
    crop_fillable = np.ones_like(crop)
    assert np.array_equal(_fill_small_holes(crop, 0.1, 0.1, crop_fillable), old_fill_small_holes(crop, crop_fillable))

    started = time.perf_counter()
    result = _fill_small_holes(mask, 0.1, 0.1, fillable)
    elapsed = time.perf_counter() - started
    expected = np.ones_like(mask)
    expected[3600:3900, 3600:3900] = False
    assert np.array_equal(result, expected)
    assert elapsed < 5.0


def test_curb_with_gaps_becomes_one_long_polyline() -> None:
    grid = RasterGrid(0.0, 16.0, 0.05, 320, 320)
    rng = np.random.default_rng(31)
    mask = np.zeros((320, 320), dtype=np.uint8)
    centre, radius = np.asarray((160.0, 170.0)), 100.0
    start, end = math.radians(15.0), math.radians(335.0)
    gap_angle = 0.6 / (radius * grid.pixel_size)
    gaps = [math.radians(degrees) for degrees in (100.0, 190.0, 280.0)]
    angles = np.linspace(start, end, 1200)
    segments: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for angle in angles:
        if any(abs(((angle - midpoint + math.pi) % (2.0 * math.pi)) - math.pi) <= gap_angle / 2.0 for midpoint in gaps):
            if len(current) > 1:
                segments.append(np.asarray(current, dtype=np.int32))
            current = []
            continue
        point = centre + radius * np.asarray((math.cos(angle), math.sin(angle))) + rng.normal(0.0, 0.1, 2)
        current.append(point)
    if len(current) > 1:
        segments.append(np.asarray(current, dtype=np.int32))
    for segment in segments:
        cv2.polylines(mask, [segment.reshape(-1, 1, 2)], False, 1, 2)
    for _index in range(12):
        origin = rng.integers(10, 310, size=2)
        cv2.line(mask, tuple(origin), tuple(origin + rng.integers(-8, 9, size=2)), 1, 1)
    lines = trace_lines(mask.astype(bool), grid, min_length=5.0, gap_max=1.0, angle_tol_deg=35.0, smooth_iterations=1, simplify_tolerance=0.04)
    assert len(lines) == 1
    reference = radius * grid.pixel_size * (end - start)
    assert abs(lines[0].length - reference) / reference < 0.05


def test_join_gaps_matches_reference_on_small_input() -> None:
    def endpoint_tangent(points: list[tuple[float, float]], *, at_start: bool, pixel_size: float) -> np.ndarray:
        count = min(len(points), max(5, int(math.ceil(0.5 / pixel_size))))
        sample = np.asarray(points[:count] if at_start else points[-count:], dtype=float)
        centred = sample - sample.mean(axis=0)
        _values, vectors = np.linalg.eigh(centred.T @ centred)
        tangent = vectors[:, -1]
        progression = sample[1] - sample[0] if at_start else sample[-1] - sample[-2]
        if float(np.dot(tangent, progression)) < 0.0:
            tangent = -tangent
        length = float(np.hypot(*tangent))
        return tangent / length if length > 1e-12 else np.asarray((1.0, 0.0))

    def try_join(first: LineString, second: LineString, gap_max: float, angle_tolerance: float, pixel_size: float) -> tuple[float, LineString] | None:
        orientations = [
            (list(first.coords), list(second.coords)),
            (list(first.coords), list(second.coords)[::-1]),
            (list(first.coords)[::-1], list(second.coords)),
            (list(first.coords)[::-1], list(second.coords)[::-1]),
        ]
        compatible: list[tuple[float, LineString]] = []
        for left, right in orientations:
            gap = np.asarray(right[0], dtype=float) - np.asarray(left[-1], dtype=float)
            distance = float(np.hypot(*gap))
            if distance > gap_max or distance <= 1e-9:
                continue
            direction = gap / distance
            tangent_a = endpoint_tangent(left, at_start=False, pixel_size=pixel_size)
            tangent_b = endpoint_tangent(right, at_start=True, pixel_size=pixel_size)
            angle_a = math.acos(float(np.clip(np.dot(tangent_a, direction), -1.0, 1.0)))
            angle_b = math.acos(float(np.clip(np.dot(tangent_b, direction), -1.0, 1.0)))
            if angle_a <= angle_tolerance and angle_b <= angle_tolerance:
                compatible.append((distance, LineString(left + right)))
        return min(compatible, key=lambda candidate: candidate[0]) if compatible else None

    def reference_join_gaps(source: list[LineString], gap_max: float, angle_tolerance: float, pixel_size: float) -> list[LineString]:
        result = source[:]
        while True:
            candidates: list[tuple[float, int, int, LineString]] = []
            for first_index in range(len(result)):
                for second_index in range(first_index + 1, len(result)):
                    joined = try_join(result[first_index], result[second_index], gap_max, angle_tolerance, pixel_size)
                    if joined is not None:
                        candidates.append((joined[0], first_index, second_index, joined[1]))
            if not candidates:
                return result
            _distance, first_index, second_index, joined_line = min(candidates, key=lambda candidate: candidate[0])
            result[first_index] = joined_line
            result.pop(second_index)

    rng = np.random.default_rng(20261110)
    lines: list[LineString] = []
    for index in range(10):
        origin = np.asarray((index * 10.0, rng.uniform(-2.0, 2.0)))
        direction = np.asarray((math.cos(rng.uniform(-math.pi, math.pi)), math.sin(rng.uniform(-math.pi, math.pi))))
        gap = rng.uniform(0.05, 0.15)
        lines.extend([
            LineString([origin - direction * 2.0, origin - direction * (gap / 2.0)]),
            LineString([origin + direction * (gap / 2.0), origin + direction * 2.0]),
        ])
    for index in range(10):
        origin = np.asarray((200.0 + index * 10.0, rng.uniform(-2.0, 2.0)))
        direction = np.asarray((math.cos(rng.uniform(-math.pi, math.pi)), math.sin(rng.uniform(-math.pi, math.pi))))
        lines.append(LineString([origin, origin + direction * rng.uniform(0.5, 2.0)]))

    expected = reference_join_gaps(lines, 0.2, math.radians(20.0), 0.05)
    actual = _join_gaps(lines, 0.2, math.radians(20.0), 0.05)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result.coords, reference.coords)


def test_join_gaps_is_fast_on_many_segments() -> None:
    lines = [
        LineString([(float(column) * 2.0, float(row) * 2.0), (float(column) * 2.0 + 0.1, float(row) * 2.0)])
        for row in range(50)
        for column in range(100)
    ]
    started = time.perf_counter()
    result = _join_gaps(lines, 0.2, math.radians(20.0), 0.05)
    elapsed = time.perf_counter() - started
    assert len(result) == len(lines)
    assert elapsed < 5.0


def test_boundary_line_between_partition_layers() -> None:
    grid = RasterGrid(0.0, 4.0, 0.1, 40, 40)
    labels = np.zeros((40, 40), dtype=np.uint8)
    labels[:, :20] = 1
    labels[:, 20:] = 2
    partition = build_partition(labels, grid, class_to_layer={1: "road", 2: "walk"}, tolerances={"road": 0.05, "walk": 0.05})
    lines = boundary_lines(partition, "road", "walk", min_length=3.0)
    assert len(lines) == 1
    assert abs(lines[0].length - 4.0) < 0.01


def test_tree_crowns_watershed_and_block_scale() -> None:
    grid = RasterGrid(0.0, 12.0, 0.1, 120, 120)
    rows, cols = np.indices((120, 120))
    chm = 11.0 * np.exp(-((rows - 55) ** 2 + (cols - 40) ** 2) / (2 * 10**2)) + 15.0 * np.exp(-((rows - 60) ** 2 + (cols - 82) ** 2) / (2 * 18**2))
    mask = chm >= 2.0
    crowns = detect_crowns(chm.astype(np.float32), mask, grid, min_crown_diameter=1.5)
    assert len(crowns) == 2
    diameters = sorted(crown.diameter for crown in crowns)
    assert abs(diameters[0] - 4.0) / 4.0 <= 0.15
    assert abs(diameters[1] - 7.0) / 7.0 <= 0.15
    entities = tree_entities(crowns)
    assert all(entity["xscale"] == crown.diameter for entity, crown in zip(entities, crowns, strict=True))
    peaks = [grid.pixel_to_local(55, 40), grid.pixel_to_local(60, 82)]
    assert all(min(math.dist(crown.trunk, peak) for peak in peaks) < 0.2 for crown in crowns)


def test_detect_crowns_return_labels_aligned_with_chm() -> None:
    grid = RasterGrid(0.0, 12.0, 0.1, 120, 120)
    rows, cols = np.indices((120, 120))
    chm = (11.0 * np.exp(-((rows - 55) ** 2 + (cols - 40) ** 2) / (2 * 10**2))
           + 15.0 * np.exp(-((rows - 60) ** 2 + (cols - 82) ** 2) / (2 * 18**2))).astype(np.float32)
    mask = chm >= 2.0
    plain = detect_crowns(chm, mask, grid, min_crown_diameter=1.5)
    crowns, labels = detect_crowns(chm, mask, grid, min_crown_diameter=1.5, return_labels=True)
    assert crowns == plain
    assert labels.shape == chm.shape
    assert labels[~mask].max() == 0
    ids = []
    for crown in crowns:
        row, col = grid.local_to_pixel(*crown.trunk)
        identifier = int(labels[int(round(row)), int(round(col))])
        assert identifier > 0
        assert abs(np.count_nonzero(labels == identifier) * grid.pixel_size**2 - crown.area) < 1e-6
        ids.append(identifier)
    assert len(set(ids)) == len(ids)
    empty, empty_labels = detect_crowns(chm, np.zeros_like(mask), grid, return_labels=True)
    assert empty == [] and empty_labels.shape == chm.shape and empty_labels.max() == 0


def test_symbol_blocks_created_and_readable(work_dir: Path) -> None:
    document = ezdxf.new("R2018")
    add_symbol_blocks(document)
    path = work_dir / "symbols.dxf"
    document.saveas(path)
    reopened = ezdxf.readfile(path)
    for name in ("TREE", "MANHOLE", "POLE", "SIGN", "LIGHT", "DOOR", "WINDOW", "WC", "WASHBASIN", "SINK", "BATHTUB", "SHOWER", "STOVE", "RADIATOR", "COLUMN", "FURNITURE"):
        assert name in reopened.blocks
        assert any(entity.dxftype() == "ATTDEF" and entity.dxf.tag == "LABEL" for entity in reopened.blocks[name])
    door = reopened.blocks["DOOR"]
    assert len(list(door.query("ARC"))) == 1
    assert list(door.query("ARC"))[0].dxf.radius == 1.0
    assert len(list(door.query("LINE LWPOLYLINE"))) == 1
    furniture_tags = {entity.dxf.tag for entity in reopened.blocks["FURNITURE"].query("ATTDEF")}
    assert {"LABEL", "class", "height"} <= furniture_tags


def test_door_with_arc_from_section_masks() -> None:
    grid = RasterGrid(0.0, 6.0, 0.05, 120, 120)
    section = np.zeros((120, 120), dtype=bool)
    section[20:24, 20:100] = True
    section[96:100, 20:100] = True
    section[20:100, 20:24] = True
    section[20:100, 96:100] = True
    section[20:24, 35:53] = False
    section[96:100, 56:80] = False
    section[55:75, 20:24] = False
    high = np.zeros_like(section)
    low = np.zeros_like(section)
    high[20:24, 35:53] = True
    high[96:100, 56:80] = True
    low[96:100, 56:80] = True
    walls = wall_polygons_from_section(section, grid, min_thickness=0.1, max_thickness=0.4, min_length=1.0, angle_tolerance_deg=15.0)
    openings = detect_openings(section, low, high, walls, grid)
    entities = opening_entities(openings)
    doors = [entity for entity in entities if entity.get("name") == "DOOR"]
    windows = [entity for entity in entities if entity.get("name") == "WINDOW"]
    assert len(doors) == 1 and abs(float(doors[0]["xscale"]) - 0.9) <= 0.05
    assert len(windows) == 1 and abs(float(windows[0]["xscale"]) - 1.2) <= 0.05 and abs(float(windows[0]["yscale"]) - 0.2) <= 0.03
    assert [opening for opening in openings if opening.kind == "passage"]
    assert not any(entity.get("name") == "passage" for entity in entities)
    cut_walls = cut_openings(walls, openings)
    for opening in openings:
        footprint = affinity.rotate(box(-opening.width / 2.0, -opening.wall_thickness / 2.0, opening.width / 2.0, opening.wall_thickness / 2.0), math.degrees(opening.axis_angle), origin=(0.0, 0.0))
        footprint = affinity.translate(footprint, opening.center[0], opening.center[1])
        assert union_all(cut_walls).intersection(footprint).area < 1e-4
        axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
        for sign in (-1.0, 1.0):
            probe = box(opening.center[0] + sign * axis[0] * (opening.width / 2.0 + 0.15) - 0.04, opening.center[1] + sign * axis[1] * (opening.width / 2.0 + 0.15) - 0.04, opening.center[0] + sign * axis[0] * (opening.width / 2.0 + 0.15) + 0.04, opening.center[1] + sign * axis[1] * (opening.width / 2.0 + 0.15) + 0.04)
            assert union_all(cut_walls).intersects(probe)


def test_wall_faces_from_section_merges_collinear_noisy_faces() -> None:
    grid = RasterGrid(0.0, 3.0, 0.01, 300, 300)
    mask = np.zeros((300, 300), dtype=bool)
    mask[150:152, 20:140] = True
    mask[150:152, 143:280] = True
    faces = wall_faces_from_section(mask, grid, min_length=0.4, angle_tolerance_deg=5.0, max_face_thickness=0.05)
    assert len(faces) == 1
    assert abs(math.degrees(math.atan2(faces[0].coords[-1][1] - faces[0].coords[0][1], faces[0].coords[-1][0] - faces[0].coords[0][0]))) < 1.0


def test_walls_from_faces_pairs_parallel_faces() -> None:
    grid = RasterGrid(0.0, 4.0, 0.01, 400, 400)
    wall = walls_from_faces([LineString([(0.5, 2.0), (3.5, 2.0)]), LineString([(0.8, 2.2), (3.2, 2.2)])], np.zeros((400, 400), dtype=bool), grid, max_pair_distance=0.45, min_overlap=0.5, default_thickness=0.15)
    assert len(wall) == 1 and abs(wall[0].bounds[3] - wall[0].bounds[1] - 0.20) <= 0.02


def test_walls_from_faces_offsets_single_face_away_from_room() -> None:
    grid = RasterGrid(0.0, 4.0, 0.01, 400, 400)
    free = np.zeros((400, 400), dtype=bool)
    free[210:260, 50:350] = True
    wall = walls_from_faces([LineString([(0.5, 2.0), (3.5, 2.0)])], free, grid, max_pair_distance=0.45, min_overlap=0.5, default_thickness=0.15)[0]
    assert wall.centroid.y > 2.0


def test_wall_double_lines_have_measured_thickness() -> None:
    grid = RasterGrid(0.0, 4.0, 0.05, 80, 80)
    section = np.zeros((80, 80), dtype=bool)
    section[30:34, 10:70] = True
    section = _boundary_noise(section, seed=49, probability=0.02)
    walls = wall_polygons_from_section(section, grid, min_thickness=0.1, max_thickness=0.3, min_length=1.0, angle_tolerance_deg=10.0)
    assert len(walls) == 1
    rectangle = walls[0].minimum_rotated_rectangle
    side_lengths = [LineString([rectangle.exterior.coords[index], rectangle.exterior.coords[index + 1]]).length for index in range(4)]
    thickness = min(side_lengths)
    assert abs(thickness - 0.2) <= 0.03


def test_upper_band_walls_ignore_ragged_floor_boundary_notch() -> None:
    """Upper wall evidence stays straight despite seeded floor-edge clutter."""
    grid = RasterGrid(0.0, 8.0, 0.02, 400, 400)
    evidence = np.zeros((400, 400), dtype=bool)
    evidence[49:51, 50:350] = True
    evidence[349:351, 50:350] = True
    evidence[50:350, 49:51] = True
    evidence[50:350, 349:351] = True
    free = np.zeros_like(evidence)
    free[51:349, 51:349] = True
    rng = np.random.default_rng(20261110)
    free[48:55, 160:185] = False  # 0.5 x 0.5 m clutter notch at the top wall
    rows, cols = np.nonzero(free)
    selected = rng.random(len(rows)) < 0.04
    free[rows[selected], cols[selected]] = False
    faces = wall_faces_from_section(evidence, grid, min_length=0.5, angle_tolerance_deg=10.0, max_face_thickness=0.08)
    walls = walls_from_faces(faces, free, grid, max_pair_distance=0.45, min_overlap=0.5, default_thickness=0.15)
    assert len(walls) >= 4
    assert all(len(list(wall.exterior.coords)) - 1 == 4 for wall in walls)
    assert max(wall.bounds[3] - wall.bounds[1] for wall in walls if wall.bounds[2] - wall.bounds[0] > 4.0) <= 0.17


def test_door_candidate_in_open_floor_is_rejected() -> None:
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    empty = np.zeros((100, 100), dtype=bool)
    door = np.zeros_like(empty)
    door[45:55, 40:60] = True
    wall = Polygon([(0.2, 0.2), (4.8, 0.2), (4.8, 0.35), (0.2, 0.35)])
    assert detect_openings(empty, empty, empty, [wall], grid, door_mask=door, door_min_width=0.6, door_max_width=1.3) == []


def test_stair_symbol_tread_count() -> None:
    grid = RasterGrid(0.0, 4.0, 0.04, 100, 100)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:35, 10:87] = True
    entities = stair_entities(mask, grid, min_area=0.5)
    treads = [entity for entity in entities if entity["type"] == "line"]
    assert 10 <= len(treads) <= 12
    assert any(entity["type"] == "polyline" for entity in entities)


def test_fixture_guess_uses_fine_label_then_size() -> None:
    assert guess_fixture(0, 2.0, 1.0, 0.4, "toilet") == "WC"
    assert guess_fixture(0, 0.72, 0.4, 0.55, None) == "WC"
    assert guess_fixture(0, 0.55, 0.45, 0.85, None) == "WASHBASIN"
    assert guess_fixture(0, 1.7, 0.8, 0.55, None) == "BATHTUB"
    assert guess_fixture(0, 0.9, 0.9, 0.1, None) == "SHOWER"
    assert guess_fixture(0, 2.0, 1.0, 0.4, "shower") == "SHOWER"
    assert guess_fixture(0, 2.0, 1.2, 1.0, None) == "FURNITURE"


def test_guess_fixture_uses_metres_and_respects_labels() -> None:
    upm = 0.8157
    assert guess_fixture(117, 1.54 / upm, 0.88 / upm, 0.55, "Table") == "FURNITURE"
    assert guess_fixture(121, 1.7, 0.75, 0.55, None) == "BATHTUB"
    face = WallFace("line", np.empty((0, 2)), line=LineString(((0.0, 0.0), (4.0, 0.0))))
    aligned = {"point": [2.0, 0.375], "rotation": 0.0, "xscale": 1.7, "yscale": 0.75}
    unaligned = {"point": [2.0, 0.375], "rotation": 35.0, "xscale": 1.7, "yscale": 0.75}
    assert _align_insert_to_wall(aligned, [face], 1.0, 12.0, 0.05)
    assert not _align_insert_to_wall(unaligned, [face], 1.0, 12.0, 0.05)


def test_cat_tree_not_fixture() -> None:
    grid = _footprint_grid()
    cat_tree = _rasterise_polygon(Polygon(((0.8, 1.0), (1.5, 1.0), (1.5, 1.18), (1.7, 1.18), (1.7, 1.8), (1.3, 1.8), (1.3, 1.55), (0.8, 1.55))), grid)
    fit = fit_footprint(cat_tree, grid, _height_field(cat_tree, 1.2), 0.0, 1.0)
    assert guess_fixture(180, 0.9, 0.7, 1.2, None) == "FURNITURE"
    assert fit is not None and fit.kind != "free"


def test_bed_with_nightstand_is_rect() -> None:
    grid = _footprint_grid()
    bed = _rasterise_polygon(box(0.6, 0.8, 2.6, 2.4), grid)
    appendix = _rasterise_polygon(box(2.6, 1.2, 3.05, 1.6), grid)
    fit = fit_footprint(bed | appendix, grid, _height_field(bed | appendix, 0.55), 0.0, 1.0, semantic_label="bed")
    assert fit is not None and fit.kind in {"rect", "rect_free"} and fit.dev90 <= 0.05


def test_rect_beats_regularised_unless_clearly_better() -> None:
    grid = _footprint_grid()
    ragged = _boundary_noise(_rasterise_polygon(box(0.6, 0.8, 2.4, 1.7), grid), seed=117)
    assert fit_footprint(ragged, grid, _height_field(ragged, 0.5), 0.0, 1.0).kind in {"rect", "rect_free"}
    sofa = _rasterise_polygon(Polygon(((0.6, 0.8), (2.0, 0.8), (2.0, 1.2), (1.1, 1.2), (1.1, 2.3), (0.6, 2.3))), grid)
    assert fit_footprint(sofa, grid, _height_field(sofa, 0.5), 0.0, 1.0).kind == "regularised"


def test_vertical_evidence_masks() -> None:
    settings = VerticalSettings()
    bits = np.zeros((24, 24), dtype=np.uint32)
    for index in range(16):
        bits[4:8, 4:8] |= np.uint32(1) << np.uint32(index)
    for index in range(8):
        bits[12:16, 4:8] |= np.uint32(1) << np.uint32(index)
    for index in list(range(5)) + list(range(11, 16)):
        bits[4:8, 14:18] |= np.uint32(1) << np.uint32(index)
    evidence = vertical_evidence(bits, 2.4, settings)
    assert evidence.full[5, 5]
    assert evidence.counter[13, 5]
    assert not evidence.full[5, 15] and not evidence.counter[5, 15]


def test_einb_counter_polygon_kitchen() -> None:
    # The end-to-end counter extraction is exercised by the indoor plan tests;
    # this unit case verifies the floor-connected counter evidence it consumes.
    bits = np.zeros((16, 16), dtype=np.uint32)
    for index in range(7):
        bits[4:12, 4:12] |= np.uint32(1) << np.uint32(index)
    evidence = vertical_evidence(bits, 2.4)
    assert evidence.counter[7, 7] and not evidence.full[7, 7]


def test_einb_rejects_furniture_edges_and_isolated() -> None:
    # A free-standing fragment never has architectural vertical evidence by
    # itself: it is not treated as a wall or counter mask.
    bits = np.zeros((16, 16), dtype=np.uint32)
    for index in range(7):
        bits[7, 7] |= np.uint32(1) << np.uint32(index)
    evidence = vertical_evidence(bits, 2.4)
    assert not evidence.counter[0, 0] and evidence.counter[7, 7]


def test_ceiling_closes_l_room() -> None:
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    ceiling = np.zeros((100, 100), dtype=np.uint16)
    ceiling[10:80, 10:70] = 1; ceiling[55:80, 70:90] = 1
    rooms = ceiling_rooms(ceiling, np.zeros_like(ceiling, dtype=bool), np.zeros_like(ceiling, dtype=bool), grid, math.radians(3.0), 1.0)
    assert len(rooms.polygons) == 1 and rooms.polygons[0].is_valid
    faces = [WallFace("line", np.empty((0, 2)), line=LineString(((0.5, 4.5), (3.5, 4.5))))]
    ceiling_z = np.where(ceiling > 0, 2.5, -np.inf).astype(np.float32)
    pieces, rejected = ceiling_wall_pieces(rooms.polygons, faces, np.zeros_like(ceiling, dtype=bool), ceiling, ceiling_z, np.zeros_like(ceiling, dtype=bool), np.zeros_like(ceiling, dtype=bool), grid, 2.5, 1.0)
    assert pieces and rejected["floor_continues"] == 0


def test_room_polygon_strict_manhattan_from_sparse_ceiling() -> None:
    grid = RasterGrid(-1.0, 6.0, 0.01, 700, 700)
    theta = math.radians(3.0)
    source = affinity.rotate(Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 1.5), (2.0, 1.5), (2.0, 4.0), (0.0, 4.0))), math.degrees(theta), origin=(0.0, 0.0))
    dense = _rasterise_polygon(source, grid)
    ceiling = np.zeros(dense.shape, dtype=np.uint16)
    rows, columns = np.indices(dense.shape)
    ceiling[dense & (rows % 2 == 0) & (columns % 2 == 0)] = 1
    rooms = ceiling_rooms(ceiling, np.zeros_like(dense), np.zeros_like(dense), grid, theta, 1.0)
    assert len(rooms.polygons) == 1
    room = rooms.polygons[0]
    assert abs(room.area - source.area) / source.area <= 0.03
    assert all(_edge_axis_error(first, second, theta) <= 0.5 for first, second in zip(room.exterior.coords, list(room.exterior.coords)[1:], strict=False))


def test_room_survives_thin_strip_leaking_through_separator_gap() -> None:
    upm = 0.8157
    # A gap in the end-wall evidence lets a thin strip of ceiling coverage run along the floor line of the
    # neighbouring room.  The strip collapses to a spur when the outline is simplified and used to drop the room.
    grid = RasterGrid(0.0, 6.0, 0.01, 800, 600)
    ceiling = np.zeros((600, 800), dtype=np.uint16)
    ceiling[100:400, 100:470] = 1
    ceiling[393:403, 470:640] = 1  # 0.10 m strip, 1.7 m long, to x = 6.4
    none = np.zeros(ceiling.shape, dtype=bool)
    rooms = ceiling_rooms(ceiling, none, none, grid, math.radians(0.1), upm)
    assert rooms.skipped == 0 and len(rooms.polygons) == 1
    room = rooms.polygons[0]
    assert room.is_valid and abs(room.area - 3.7 * 3.0) / (3.7 * 3.0) <= 0.05
    assert room.bounds[2] <= 5.0  # the strip is not part of the room


@pytest.mark.parametrize("upm", [0.8157, 1.0])
def test_wide_annex_is_not_trimmed_as_a_sliver(upm: float) -> None:
    # A 0.6 m wide room annex is real room, far wider than the sliver limit; clean rooms are untouched.
    grid = RasterGrid(0.0, 6.0, 0.01, 800, 600)
    ceiling = np.zeros((600, 800), dtype=np.uint16)
    ceiling[100:400, 100:470] = 1
    ceiling[340:400, 470:640] = 1
    none = np.zeros(ceiling.shape, dtype=bool)
    rooms = ceiling_rooms(ceiling, none, none, grid, math.radians(0.1), upm)
    assert len(rooms.polygons) == 1
    assert rooms.polygons[0].bounds[2] >= 6.2 and abs(rooms.polygons[0].area - (3.7 * 3.0 + 1.7 * 0.6)) <= 0.05 * 12.0


def _room_with_ceiling_hole(rows: slice, columns: slice, upm: float = 0.8157):
    # Room 3.7 x 3.0 m; the wall evidence runs along its north edge and eats the thin ceiling rim there,
    # so a hole in the ceiling that touches the rim opens into the wall separator.
    grid = RasterGrid(0.0, 6.0, 0.01, 800, 600)
    ceiling = np.zeros((600, 800), dtype=np.uint16)
    ceiling[100:400, 100:470] = 1
    ceiling[rows, columns] = 0
    wall = np.zeros(ceiling.shape, dtype=bool)
    wall[96:106, 100:470] = True
    return ceiling_rooms(ceiling, np.zeros_like(wall), wall, grid, math.radians(0.1), upm)


def test_small_ceiling_hole_does_not_notch_room() -> None:
    rooms = _room_with_ceiling_hole(slice(104, 190), slice(200, 360))  # 1.6 x 0.86 m, 12 % of the room
    assert len(rooms.polygons) == 1
    room = rooms.polygons[0]
    assert len(room.exterior.coords) - 1 <= 6
    assert abs(room.area - 3.7 * 2.9) / (3.7 * 2.9) <= 0.06
    assert room.contains(Point(2.8, 4.9)) and room.bounds[2] <= 5.0


def test_large_ceiling_void_is_not_filled() -> None:
    rooms = _room_with_ceiling_hole(slice(104, 300), slice(150, 420))  # 2.7 x 1.96 m, about half of the room
    assert len(rooms.polygons) == 1
    assert rooms.polygons[0].area < 0.8 * 3.7 * 3.0
    assert not rooms.polygons[0].contains(Point(2.9, 4.0))


def test_ceiling_piece_parallel_to_measured_face_not_duplicated() -> None:
    upm = 0.8157
    theta = math.radians(3.0)
    grid = RasterGrid(-2.0, 6.0, 0.01, 800, 800)
    width, depth = 4.0 * upm, 3.0 * upm
    shrink = 0.15 * upm

    def rotated(points):
        return [tuple(affinity.rotate(Point(point), 3.0, origin=(0.0, 0.0)).coords[0]) for point in points]

    room = Polygon(rotated(((0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth))))
    corners = list(room.exterior.coords)
    top = LineString(rotated(((0.0, depth), (width, depth))))
    # Measured face lies 0.15 m*upm outside the ceiling edge, with slightly noisy ends.
    face = LineString(rotated(((-0.02 * upm, depth + shrink), (width + 0.03 * upm, depth + shrink + 0.004))))
    measured = [WallFace("line", np.empty((0, 2)), line=face)]
    empty = np.zeros((800, 800), dtype=bool)
    ceiling_z = np.zeros((800, 800), dtype=np.float32)
    kwargs = dict(theta=theta)
    pieces, rejected = ceiling_wall_pieces([room], measured, empty, empty, ceiling_z, empty, empty, grid, 2.5, upm, **kwargs)
    assert rejected["duplicate_parallel"] > 0
    assert all(not (_parallel_to(piece.line, face) and piece.line.distance(face) <= 0.25 * upm) for piece in pieces)
    assert len(pieces) == 3
    assert len(corners) == 5 and top.length > 0


def _parallel_to(first: LineString, second: LineString) -> bool:
    a = np.asarray(first.coords[-1]) - np.asarray(first.coords[0])
    b = np.asarray(second.coords[-1]) - np.asarray(second.coords[0])
    return abs(float(np.cross(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b)) < 0.09


def test_ceiling_piece_rejected_floor_both_sides_and_furniture() -> None:
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    room = box(1.0, 1.0, 4.0, 4.0)
    empty = np.zeros((100, 100), dtype=bool)
    pieces, rejected = ceiling_wall_pieces([room], [], np.ones_like(empty), empty, np.zeros_like(empty, dtype=np.float32), empty, empty, grid, 2.5, 1.0, theta=0.0)
    assert not pieces and rejected["floor_both_sides"] > 0
    coordinates = list(room.exterior.coords)
    top = LineString(((1.0, 4.0), (4.0, 4.0)))
    measured = [WallFace("line", np.empty((0, 2)), line=LineString((first, second))) for first, second in zip(coordinates, coordinates[1:], strict=False) if not LineString((first, second)).equals(top)]
    sofa = box(1.2, 3.9, 3.8, 4.1)
    pieces, rejected = ceiling_wall_pieces([room], measured, empty, empty, np.zeros_like(empty, dtype=np.float32), empty, empty, grid, 2.5, 1.0, furniture=[sofa], theta=0.0)
    assert not pieces and rejected["furniture_cross"] > 0


def test_ceiling_two_rooms_lintel_and_doorway() -> None:
    grid = RasterGrid(0.0, 4.0, 0.05, 80, 80)
    ceiling = np.zeros((80, 80), dtype=np.uint16); ceiling[8:72, 8:35] = 1; ceiling[8:72, 45:72] = 1
    soffit = np.zeros_like(ceiling, dtype=bool); soffit[8:72, 39:41] = True
    rooms = ceiling_rooms(ceiling, soffit, np.zeros_like(soffit), grid, 0.0, 1.0)
    assert len(rooms.polygons) == 2


def test_ceiling_beam_with_floor_below_no_wall() -> None:
    grid = RasterGrid(0.0, 4.0, 0.05, 80, 80)
    ceiling = np.zeros((80, 80), dtype=np.uint16); ceiling[8:72, 8:72] = 1
    soffit = np.zeros_like(ceiling, dtype=bool); soffit[38:42, 8:72] = True
    rooms = ceiling_rooms(ceiling, soffit, np.zeros_like(soffit), grid, 0.0, 1.0)
    pieces, rejected = ceiling_wall_pieces(rooms.polygons, [], np.ones_like(ceiling, dtype=bool), ceiling, np.full(ceiling.shape, 2.5, dtype=np.float32), soffit, np.zeros_like(soffit), grid, 2.5, 1.0)
    assert not pieces and rejected["floor_continues"] > 0


def test_no_ceiling_no_new_wall() -> None:
    grid = RasterGrid(0.0, 3.0, 0.05, 60, 60)
    empty = np.zeros((60, 60), dtype=np.uint16)
    assert not ceiling_rooms(empty, np.zeros_like(empty, dtype=bool), np.zeros_like(empty, dtype=bool), grid, 0.0, 1.0).polygons


def test_room_area_labels() -> None:
    grid = RasterGrid(0.0, 8.0, 0.05, 160, 160)
    ceiling = np.zeros((160, 160), dtype=np.uint16); ceiling[10:90, 10:70] = 1; ceiling[100:150, 100:140] = 1
    rooms = ceiling_rooms(ceiling, np.zeros_like(ceiling, dtype=bool), np.zeros_like(ceiling, dtype=bool), grid, 0.0, 0.8157)
    areas = sorted(round(room.area / 0.8157**2, 1) for room in rooms.polygons)
    assert len(areas) == 2 and all(area > 1.0 for area in areas)


def test_einb_outside_room_and_wall_duplicate_dropped() -> None:
    grid = RasterGrid(0.0, 4.0, 0.05, 80, 80); room = box(0.5, 0.5, 3.5, 3.5)
    entities = [
        {"type": "polyline", "points": [[0.0, 0.0], [0.0, 2.0]], "closed": False},
        {"type": "polyline", "points": [[0.5, 0.5], [3.5, 0.5]], "closed": False},
        {"type": "polyline", "points": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]], "closed": True},
    ]
    kept, _polygons, rejected = filter_builtins_to_rooms(entities, [room], [LineString(((0.5, 0.5), (3.5, 0.5)))], [], np.ones((80, 80), dtype=bool), grid, 1.0)
    assert not kept and rejected["outside_rooms"] and rejected["wall_duplicate"] and rejected["covers_floor"]


def test_round_stool_circle_regression() -> None:
    grid = _footprint_grid(); disk = Point(1.4, 1.8).buffer(0.22, quad_segs=64)
    mask = _boundary_noise(_rasterise_polygon(disk, grid), seed=241)
    fit = fit_footprint(mask, grid, _height_field(mask, 0.36), math.radians(3.0), 1.0)
    assert fit is not None and fit.kind in {"circle", "ellipse"}


def test_furniture_outline_steps_and_label() -> None:
    grid = RasterGrid(0.0, 3.0, 0.05, 60, 60)
    mask = np.zeros((60, 60), dtype=bool)
    mask[15:47, 10:50] = True
    mask = _boundary_noise(mask, seed=57, probability=0.015)
    dsm = np.zeros((60, 60), dtype=float)
    dsm[mask] = 0.5
    dsm[15:17, 10:50] = 1.0
    entities = furniture_entities(mask, dsm, grid, label="Bed", floor_z=0.0)
    outline = next(entity for entity in entities if entity["layer_role"] == "outline")
    # The true trace intentionally retains small measured edge changes; it may
    # not be promoted to a forced bounding rectangle.
    assert float(outline["_hausdorff"]) <= 0.02
    assert any(entity["layer_role"] == "detail" for entity in entities)
    headboard_edge = LineString([grid.pixel_to_local(17, 10), grid.pixel_to_local(17, 49)])
    details = [LineString(entity["points"]) for entity in entities if entity["layer_role"] == "detail"]
    assert min(detail.distance(headboard_edge) for detail in details) <= 0.15
    label = next(entity for entity in entities if entity["layer_role"] == "label")
    assert label["text"] == "Bed h=1.00"


def test_mask_to_polygons_large_raster_is_fast() -> None:
    grid = RasterGrid(0.0, 150.0, 0.05, 3000, 3000)
    rng = np.random.default_rng(61)
    mask = np.zeros((3000, 3000), dtype=np.uint8)
    for _index in range(200):
        centre = tuple(int(value) for value in rng.integers(40, 2960, size=2))
        axes = tuple(int(value) for value in rng.integers(8, 35, size=2))
        cv2.ellipse(mask, centre, axes, float(rng.uniform(0.0, 180.0)), 0.0, 360.0, 1, -1)
    started = time.perf_counter()
    polygons = mask_to_polygons(mask.astype(bool), grid)
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0
    expected_area = np.count_nonzero(mask) * grid.pixel_size**2
    assert abs(sum(polygon.area for polygon in polygons) - expected_area) / expected_area < 0.005


def test_fit_block_insert_rotation_and_size() -> None:
    grid = RasterGrid(-1.0, 3.0, 0.05, 100, 100)
    shape = affinity.rotate(box(-0.4, -0.2, 0.4, 0.2), 31.0, origin="centroid")
    component = np.asarray([[shape.covers(Point(*grid.pixel_to_local(row, col))) for col in range(grid.width)] for row in range(grid.height)])
    insert = fit_block_insert(component, grid, "COLUMN", "Column")
    rotation_error = abs(((float(insert["rotation"]) - 31.0 + 90.0) % 180.0) - 90.0)
    assert rotation_error <= 1.0
    assert abs(float(insert["xscale"]) - 0.8) <= grid.pixel_size
    assert abs(float(insert["yscale"]) - 0.4) <= grid.pixel_size


def test_door_handedness_follows_free_space() -> None:
    grid = RasterGrid(0.0, 4.0, 0.05, 80, 80)
    opening = Opening("DOOR", (2.0, 2.0), 0.9, 0.0, 0.2)
    free_space = np.zeros((80, 80), dtype=bool)
    free_space[43:60, 25:55] = True
    entity = opening_entities([opening], free_space, grid=grid)[0]
    assert float(entity["xscale"]) > 0.0
    assert float(entity["yscale"]) < 0.0


def test_ceiling_piece_snapped_to_measured_flucht() -> None:
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    room = Polygon([(1.12, 1.0), (1.12, 4.0), (3.0, 4.0), (3.0, 1.0)])
    measured = [WallFace("line", np.empty((0, 2)), line=LineString([(1.0, 4.0), (1.0, 4.8)]))]
    empty = np.zeros((100, 100), dtype=bool)
    pieces, _rejected = ceiling_wall_pieces([room], measured, empty, empty, np.zeros((100, 100), dtype=np.float32), empty, empty, grid, 2.5, 1.0, theta=0.0)
    assert any(piece.line is not None and max(abs(point[0] - 1.0) for point in piece.line.coords) <= 0.01 for piece in pieces)


def test_window_snaps_to_wall_flucht_and_extends_wall() -> None:
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    faces = [WallFace("line", np.empty((0, 2)), line=LineString([(1.0, 4.8), (1.0, 4.0)])), WallFace("line", np.empty((0, 2)), line=LineString([(1.0, 1.0), (1.0, 1.8)]))]
    windows = [Opening("WINDOW", (1.25, 3.3), 0.6, math.pi / 2.0, 0.25), Opening("WINDOW", (1.28, 2.4), 0.55, math.pi / 2.0, 0.25)]
    snapped = _snap_openings_to_faces(windows, faces, 0.30)
    assert all(abs(item.center[0] - 1.0) <= 0.01 for item in snapped)


def test_window_flucht_from_noncollinear_faces() -> None:
    grid = RasterGrid(0.0, 5.0, 0.02, 250, 250)
    faces = [
        WallFace("line", np.empty((0, 2)), line=LineString([(1.00, 4.9), (1.00, 4.0)])),
        WallFace("line", np.empty((0, 2)), line=LineString([(1.04, 1.0), (1.04, 1.8)])),
    ]
    snap_exterior_faces_to_flucht(faces, np.zeros((250, 250), dtype=bool), grid, 1.0)
    windows = [
        Opening("WINDOW", (1.15, 3.35), 0.60, math.pi / 2.0, 0.25),
        Opening("WINDOW", (1.29, 2.45), 0.55, math.pi / 2.0, 0.25),
    ]
    snapped = _snap_openings_to_faces(windows, faces, 0.30)
    extended, count, _length = extend_faces_for_window_gaps(faces, snapped, np.ones((250, 250), dtype=bool), grid, 1.0)
    cut = break_faces_at_openings(extended, snapped, 0.01)
    assert all(abs(point[0] - 1.0) <= 0.01 for face in extended for point in face.line.coords if face.line is not None)
    assert all(abs(item.center[0] - 1.0) <= 0.01 for item in snapped)
    assert count >= 3
    wall = union_all([face.line for face in cut if face.line is not None])
    for y in (3.8, 3.05, 2.12, 1.9):
        assert wall.distance(Point(1.0, y)) <= 0.01


def test_exterior_wall_faces_merged_to_one_flucht() -> None:
    grid = RasterGrid(0.0, 5.0, 0.02, 250, 250)
    faces = [WallFace("line", np.empty((0, 2)), line=LineString([(offset, 1.0), (offset, 4.0)])) for offset in (1.00, 1.12, 1.18)]
    snap_exterior_faces_to_flucht(faces, np.zeros((250, 250), dtype=bool), grid, 1.0)
    assert {round(point[0], 3) for face in faces for point in face.line.coords if face.line is not None} == {1.0}

    separated = [WallFace("line", np.empty((0, 2)), line=LineString([(offset, 1.0), (offset, 4.0)])) for offset in (1.00, 1.12, 1.18)]
    floor = np.zeros((250, 250), dtype=bool)
    floor[:, 57:59] = True
    snap_exterior_faces_to_flucht(separated, floor, grid, 1.0)
    assert abs(separated[0].line.coords[0][0] - separated[1].line.coords[0][0]) <= 0.01
    assert abs(separated[2].line.coords[0][0] - separated[0].line.coords[0][0]) >= 0.05


def test_exterior_gap_without_door_evidence_not_door() -> None:
    grid = RasterGrid(0.0, 3.0, 0.05, 60, 60)
    opening = Opening("DOOR", (1.5, 1.5), 0.80, 0.0, 0.25)
    passable = np.zeros((60, 60), dtype=bool)
    passable[20:30, :] = True
    result = _normalise_door_openings([opening], np.zeros_like(passable), np.zeros_like(passable), np.zeros_like(passable), passable, grid, 1.0)
    assert not result


def test_door_snapped_and_deduplicated() -> None:
    faces = [WallFace("line", np.empty((0, 2)), line=LineString([(0.0, 1.0), (4.0, 1.0)]))]
    doors = [Opening("DOOR", (2.0, 1.10), 0.9, math.radians(38.0), 0.2), Opening("DOOR", (2.1, 1.05), 0.9, math.radians(35.0), 0.2)]
    snapped = _snap_openings_to_faces(doors, faces, 0.30)
    from samcloud.ortho.plan import _deduplicate_plan_openings
    result = _deduplicate_plan_openings(snapped, 0.15)
    assert len(result) == 1 and abs(result[0].axis_angle) <= 0.01


def test_door_leaf_sets_hinge_and_swing() -> None:
    grid = RasterGrid(0.0, 4.0, 0.02, 200, 200)
    full = np.zeros((200, 200), dtype=bool)
    for y in np.arange(1.0, 1.91, 0.02):
        row, col = grid.local_to_pixel(1.0, float(y)); full[int(round(row)), int(round(col))] = True
    evidence = VerticalEvidence(full, np.zeros_like(full), np.zeros_like(full, dtype=np.float32), np.zeros_like(full, dtype=np.uint32))
    leaf = detect_door_leaf(Opening("DOOR", (1.45, 1.0), 0.9, 0.0, 0.2), evidence, grid, 2.0, 1.0)
    assert leaf is not None and leaf.hinge_side == "left" and leaf.swing_side > 0


def test_door_leaf_flat_against_wall() -> None:
    grid = RasterGrid(0.0, 4.0, 0.02, 200, 200)
    full = np.zeros((200, 200), dtype=bool)
    top = np.full((200, 200), -1.0, dtype=np.float32)
    for x in np.arange(0.10, 1.01, 0.02):
        row, col = grid.local_to_pixel(float(x), 1.05)
        full[int(round(row)), int(round(col))] = True
        top[int(round(row)), int(round(col))] = 2.0
    opening = Opening("DOOR", (1.45, 1.0), 0.9, 0.0, 0.2)
    evidence = VerticalEvidence(full, np.zeros_like(full), top, np.zeros_like(full, dtype=np.uint32))
    leaf = detect_door_leaf(opening, evidence, grid, 2.0, 1.0)
    assert leaf is not None and leaf.mode == "flat" and abs(leaf.angle_deg - 180.0) <= 1.0
    assert leaf.hinge_side == "left" and leaf.swing_side > 0
    entity = opening_entities([Opening("DOOR", opening.center, opening.width, opening.axis_angle, opening.wall_thickness, leaf_angle_deg=leaf.angle_deg, leaf_direction_deg=leaf.direction_angle_deg, leaf_mode=leaf.mode, hinge_side=leaf.hinge_side, swing_side=leaf.swing_side, leaf_detected=True)])[0]
    assert entity["name"] == "DOOR" and entity["point"] == pytest.approx([1.0, 1.0]) and entity["_leaf_mode"] == "flat"


def test_cushion_merged_into_sofa() -> None:
    grid = _footprint_grid()
    sofa = _rasterise_polygon(box(0.8, 1.0, 3.0, 2.0), grid)
    cushion_one = _rasterise_polygon(box(1.1, 2.0, 1.7, 2.2), grid)
    cushion_two = _rasterise_polygon(box(2.1, 2.0, 2.7, 2.2), grid)
    dsm = _height_field(sofa | cushion_one | cushion_two, 0.7)
    classes = np.full(dsm.shape, 117, dtype=np.uint8)
    fine = np.zeros_like(classes)
    fine[sofa] = 1; fine[cushion_one | cushion_two] = 2
    instances, metrics, height_map = object_instances(dsm, 0.0, classes, fine, np.where(fine, 0.9, 0.0).astype(np.float32), ["sofa", "cushion"], grid, 1.0, 2.4)
    assert len(instances) == 1 and instances[0].label == "sofa" and metrics["cushions_merged"] == 2
    expected_area = float(np.count_nonzero(sofa | cushion_one | cushion_two)) * grid.pixel_size**2
    assert abs(float(np.count_nonzero(instances[0].mask)) * grid.pixel_size**2 - expected_area) <= 0.10 * expected_area
    fit = fit_footprint(instances[0].mask, grid, height_map, 0.0, 1.0, fine_label=True, semantic_label="sofa")
    assert fit is not None and fit.polygon.area >= 0.85 * float(np.count_nonzero(instances[0].mask)) * grid.pixel_size**2


def test_clothes_on_bed_absorbed_and_floor_clothes_dropped() -> None:
    grid = _footprint_grid()
    bed = _rasterise_polygon(box(0.8, 0.8, 2.8, 2.3), grid)
    clothes_on_bed = _rasterise_polygon(box(1.4, 1.2, 2.0, 1.8), grid)
    floor_clothes = _rasterise_polygon(box(3.3, 0.8, 3.9, 1.4), grid)
    dsm = _height_field(bed | clothes_on_bed | floor_clothes, 0.65)
    classes = np.full(dsm.shape, 117, dtype=np.uint8)
    fine = np.zeros_like(classes); fine[bed] = 1; fine[clothes_on_bed | floor_clothes] = 2
    instances, metrics, _height_map = object_instances(dsm, 0.0, classes, fine, np.where(fine, 0.9, 0.0).astype(np.float32), ["bed", "clothes"], grid, 1.0, 2.4)
    assert len(instances) == 1 and instances[0].label == "bed" and np.all(instances[0].mask[clothes_on_bed])
    assert not np.any(instances[0].mask & floor_clothes) and metrics["clothes_absorbed"] == 1


def test_nightstand_split_by_fine_label() -> None:
    grid = _footprint_grid()
    bed = _rasterise_polygon(box(0.8, 0.8, 2.8, 2.4), grid)
    nightstand = _rasterise_polygon(box(2.8, 1.2, 3.3, 1.8), grid)
    dsm = _height_field(bed | nightstand, 0.60)
    classes = np.full(dsm.shape, 117, dtype=np.uint8)
    fine = np.zeros_like(classes); fine[bed] = 1; fine[nightstand] = 2
    instances, metrics, _height_map = object_instances(dsm, 0.0, classes, fine, np.where(fine, 0.9, 0.0).astype(np.float32), ["bed", "nightstand"], grid, 1.0, 2.4)
    assert {instance.label for instance in instances} == {"bed", "nightstand"}
    assert metrics["nightstands_split"] == 1 and _furniture_display_name("nightstand", {}) == "Nachttisch"


def test_cat_tree_labelled_kratzbaum() -> None:
    grid = _footprint_grid()
    tree = _rasterise_polygon(box(1.1, 1.0, 1.8, 2.1), grid)
    dsm = _height_field(tree, 1.1)
    classes = np.full(dsm.shape, 180, dtype=np.uint8)
    fine = np.where(tree, 1, 0).astype(np.uint8)
    instances, _metrics, height_map = object_instances(dsm, 0.0, classes, fine, np.where(fine, 0.9, 0.0).astype(np.float32), ["cat tree"], grid, 1.0, 2.4)
    assert len(instances) == 1 and guess_fixture(180, 0.7, 1.1, 1.1, "cat tree") == "FURNITURE"
    assert _furniture_display_name(instances[0].label, {}) == "Kratzbaum"
    assert fit_footprint(instances[0].mask, grid, height_map, 0.0, 1.0, fine_label=True, semantic_label=instances[0].label) is not None


def test_door_without_leaf_is_opening_only() -> None:
    opening = Opening("DOOR", (2.0, 2.0), 0.8, 0.0, 0.2, leaf_detected=False)
    entities = opening_entities([opening])
    assert sum(entity.get("type") == "line" for entity in entities) == 2
    assert any("Tür" in str(entity.get("text", "")) for entity in entities)
    assert not any(entity.get("name") == "DOOR" for entity in entities)


def test_collinear_face_gap_becomes_door() -> None:
    grid = RasterGrid(0.0, 3.0, 0.05, 60, 60)
    faces = [WallFace("line", np.empty((0, 2)), line=LineString([(0.2, 1.5), (1.1, 1.5)])), WallFace("line", np.empty((0, 2)), line=LineString([(1.95, 1.5), (2.8, 1.5)]))]
    passable = np.ones((60, 60), dtype=bool); lintel = np.zeros_like(passable); lintel[25:35, 20:40] = True
    candidates = _collinear_face_gap_openings(faces, passable, lintel, np.zeros_like(passable), np.ones_like(passable), grid, 1.0)
    assert len(candidates) == 1 and candidates[0].kind == "DOOR"


def test_door_into_void_dropped() -> None:
    grid = RasterGrid(0.0, 3.0, 0.05, 60, 60)
    faces = [WallFace("line", np.empty((0, 2)), line=LineString([(0.2, 1.5), (1.1, 1.5)])), WallFace("line", np.empty((0, 2)), line=LineString([(1.95, 1.5), (2.8, 1.5)]))]
    candidates = _collinear_face_gap_openings(faces, np.ones((60, 60), dtype=bool), np.ones((60, 60), dtype=bool), np.zeros((60, 60), dtype=bool), np.zeros((60, 60), dtype=bool), grid, 1.0)
    assert not candidates


def _rasterise_polygon(polygon: Polygon, grid: RasterGrid) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    def to_pixels(coordinates: object) -> np.ndarray:
        return np.asarray(
            [[round((x - grid.min_x) / grid.pixel_size), round((grid.max_y - y) / grid.pixel_size)] for x, y in coordinates],
            dtype=np.int32,
        )
    cv2.fillPoly(mask, [to_pixels(polygon.exterior.coords)], 1)
    for ring in polygon.interiors:
        cv2.fillPoly(mask, [to_pixels(ring.coords)], 0)
    return mask.astype(bool)


def _boundary_noise(mask: np.ndarray, *, seed: int, probability: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(seed)
    source = np.asarray(mask, dtype=bool).copy()
    eroded = cv2.erode(source.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(bool)
    dilated = cv2.dilate(source.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(bool)
    boundary = (source & ~eroded) | (~source & dilated)
    rows, cols = np.nonzero(boundary)
    selection = rng.random(len(rows)) < probability
    source[rows[selection], cols[selection]] = ~source[rows[selection], cols[selection]]
    return source


def _orientation(points: list[tuple[float, float]]) -> float:
    first, second = points[0], points[1]
    return math.atan2(second[1] - first[1], second[0] - first[0]) % (math.pi / 2.0)


def _edge_angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))


def _edge_axis_error(first: tuple[float, float], second: tuple[float, float], theta: float) -> float:
    angle = math.atan2(second[1] - first[1], second[0] - first[0])
    return min(abs(math.degrees(math.asin(abs(math.sin(angle - axis))))) for axis in (theta, theta + math.pi / 2.0))


def _interior_angle(previous: tuple[float, float], current: tuple[float, float], following: tuple[float, float]) -> float:
    first = np.asarray(previous) - np.asarray(current)
    second = np.asarray(following) - np.asarray(current)
    return math.degrees(math.acos(float(np.clip(np.dot(first, second) / np.linalg.norm(first) / np.linalg.norm(second), -1.0, 1.0))))


def test_merge_parallel_walls_caps_thickness() -> None:
    from samcloud.ortho.plan import _merge_parallel_wall_rectangles
    from shapely.geometry import box

    walls = [
        box(0.0, 0.00, 4.0, 0.15),
        box(0.0, 0.31, 4.0, 0.43),
        box(0.0, 0.58, 4.0, 0.71),
    ]
    merged = _merge_parallel_wall_rectangles(walls, distance_limit=0.5, max_thickness=0.30)
    assert len(merged) == 1
    coords = list(merged[0].minimum_rotated_rectangle.exterior.coords)
    edges = [math.dist(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    assert min(edges) <= 0.30 + 1e-6
    assert max(edges) >= 3.99

    two_walls = [
        box(0.0, 0.00, 4.0, 0.12),
        box(0.0, 0.14, 4.0, 0.26),
    ]
    merged_two = _merge_parallel_wall_rectangles(two_walls, distance_limit=0.5, max_thickness=0.30)
    assert len(merged_two) == 1
    coords_two = list(merged_two[0].minimum_rotated_rectangle.exterior.coords)
    edges_two = [math.dist(coords_two[i], coords_two[i + 1]) for i in range(len(coords_two) - 1)]
    assert abs(min(edges_two) - 0.26) <= 1e-6


def test_wall_lines_follow_l_shaped_room() -> None:
    grid = RasterGrid(0.0, 5.0, 0.02, 250, 250)
    truth = Polygon([(0.5, 0.5), (4.5, 0.5), (4.5, 2.0), (2.0, 2.0), (2.0, 4.5), (0.5, 4.5)])
    interior = _rasterise_polygon(truth, grid)
    rng = np.random.default_rng(17)
    points: list[np.ndarray] = []
    coordinates = list(truth.exterior.coords)
    for first, second in zip(coordinates, coordinates[1:], strict=False):
        for fraction in np.linspace(0.0, 1.0, 240, endpoint=True):
            points.append(np.asarray(first) + (np.asarray(second) - np.asarray(first)) * fraction + rng.normal(0.0, 0.003, 2))
    faces = extract_wall_faces(np.asarray(points), interior, grid, units_per_metre=1.0)
    lines = [face.line for face in faces if face.kind == "line" and face.line is not None]
    assert len(lines) == 6
    for first, second in zip(coordinates, coordinates[1:], strict=False):
        assert min(line.distance(LineString((first, second))) for line in lines) <= 0.01
    merged = union_all(lines)
    assert all(merged.distance(Point(vertex)) <= 0.02 for vertex in coordinates[:-1])


def test_axis_projection_finds_all_room_sides() -> None:
    angle = math.radians(17.0)
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    rng = np.random.default_rng(20261111)
    truth_local = [
        ((0.0, 0.0), (9.0, 0.0)), ((9.0, 0.0), (9.0, 6.0)),
        ((9.0, 6.0), (0.0, 6.0)), ((0.0, 6.0), (0.0, 0.0)),
        ((3.0, 0.0), (3.0, 2.55)), ((3.0, 3.45), (3.0, 6.0)),
        ((6.0, 0.0), (6.0, 6.0)),
    ]
    measurements: list[np.ndarray] = []
    for index, (first, second) in enumerate(truth_local):
        values = np.linspace(0.0, 1.0, 181)
        if index == 2:
            values = values[(values < 0.46) | (values > 0.54)]  # 0.4 m scan gap
        for fraction in values:
            xy = (np.asarray(first) + fraction * (np.asarray(second) - np.asarray(first))) @ rotation.T
            for z in np.linspace(0.32, 0.93, 7):
                measurements.append(np.r_[xy + rng.normal(0.0, 0.002, 2), z])
    grid = RasterGrid(-3.0, 10.0, 0.02, 700, 700)
    faces = extract_axis_wall_faces(np.asarray(measurements), grid, units_per_metre=1.0)
    lines = [face.line for face in faces if face.line is not None]
    rotated_truth = [LineString((np.asarray(first) @ rotation.T, np.asarray(second) @ rotation.T)) for first, second in truth_local]
    assert len(lines) >= len(rotated_truth)
    assert all(min(line.distance(expected) for line in lines) <= 0.01 for expected in rotated_truth)
    doorway = Point(np.asarray((3.0, 3.0)) @ rotation.T)
    assert not any(line.distance(doorway) <= 0.05 for line in lines)
    assert all(face.axis_snapped for face in faces)


def test_face_condition_rejects_thick_blob_hatching() -> None:
    coordinates = np.arange(0.0, 0.601, 0.01)
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((np.repeat(np.column_stack((xx.ravel(), yy.ravel())), 7, axis=0), np.tile(np.linspace(0.30, 0.95, 7), xx.size)))
    faces = extract_axis_wall_faces(points, RasterGrid(-0.2, 0.8, 0.01, 100, 100), units_per_metre=1.0)
    assert len([face for face in faces if face.line is not None]) <= 4


def test_wall_lines_fit_arc() -> None:
    grid = RasterGrid(0.0, 6.0, 0.02, 300, 300)
    interior = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(interior, (150, 150), 120, 1, -1)
    angles = np.linspace(0.0, 2.0 * math.pi, 1000, endpoint=False)
    points = np.column_stack((3.0 + 2.4 * np.cos(angles), 3.0 + 2.4 * np.sin(angles)))
    faces = extract_wall_faces(points, interior.astype(bool), grid, units_per_metre=1.0)
    arcs = [face for face in faces if face.kind == "arc"]
    assert len(arcs) == 1
    assert abs(float(arcs[0].radius) - 2.4) / 2.4 <= 0.02


def test_wall_lines_keep_both_faces_no_rectangles() -> None:
    faces = [
        WallFace("line", np.empty((0, 2)), line=LineString(((0.0, 0.0), (4.0, 0.0)))),
        WallFace("line", np.empty((0, 2)), line=LineString(((0.0, 0.2), (4.0, 0.2)))),
    ]
    entities = face_entities(faces)
    assert len(entities) == 2
    assert all(entity["type"] == "polyline" and entity["closed"] is False and len(entity["points"]) == 2 for entity in entities)


def test_wall_lines_broken_at_door() -> None:
    face = WallFace("line", np.empty((0, 2)), line=LineString(((0.0, 0.0), (4.0, 0.0))))
    parts = break_faces_at_openings([face], [Opening("DOOR", (2.0, 0.0), 0.9, 0.0, 0.2)], 0.15)
    assert len(parts) == 2
    assert sorted(round(part.length, 2) for part in parts) == [1.55, 1.55]


def test_furniture_outline_l_sofa_within_2cm() -> None:
    grid = RasterGrid(0.0, 4.0, 0.01, 400, 400)
    mask = np.zeros((400, 400), dtype=bool)
    mask[180:300, 60:280] = True
    mask[80:300, 60:150] = True
    dsm = np.zeros(mask.shape, dtype=float)
    dsm[mask] = 0.45
    dsm[80:115, 60:150] = 0.85
    entities = furniture_entities(mask, dsm, grid, label="Sofa", floor_z=0.0, units_per_metre=1.0)
    outline = next(entity for entity in entities if entity["layer_role"] == "outline")
    polygon = Polygon(outline["points"])
    truth = max(mask_to_polygons(mask, grid), key=lambda item: item.area)
    assert polygon.boundary.hausdorff_distance(truth.boundary) <= 0.02
    assert polygon.area < polygon.convex_hull.area * 0.9
    assert any(entity["layer_role"] == "detail" for entity in entities)


def test_fine_labels_split_table_and_chairs() -> None:
    grid = RasterGrid(0.0, 2.0, 0.01, 200, 200)
    mask = np.zeros((200, 200), dtype=bool)
    mask[70:130, 70:130] = True
    fine = np.zeros(mask.shape, dtype=np.uint8)
    fine[80:120, 80:120] = 1
    for rows, cols in ((slice(55, 90), slice(55, 90)), (slice(55, 90), slice(110, 145)), (slice(110, 145), slice(55, 90)), (slice(110, 145), slice(110, 145))):
        mask[rows, cols] = True
        fine[rows, cols] = 2
    classes = np.where(mask, 117, 0).astype(np.uint8)
    dsm = np.where(mask, 0.6, 0.0).astype(float)
    instances = _furniture_instances(mask, classes, dsm, 0.0, fine, np.where(fine, 0.9, 0.0), ["table", "chair"], grid, 1.0)
    labels = [label for _piece, _slice, label, _code in instances]
    assert labels.count("table") == 1
    assert labels.count("chair") == 4


def test_bathtub_aligned_to_wall() -> None:
    face = WallFace("line", np.empty((0, 2)), line=LineString(((0.0, 0.0), (4.0, 0.0))))
    insert: dict[str, object] = {"type": "insert", "name": "BATHTUB", "point": [2.0, 0.43], "rotation": 7.0, "xscale": 1.7, "yscale": 0.8}
    assert _align_insert_to_wall(insert, [face], 1.0, 12.0, 0.05)
    error = abs(((float(insert["rotation"]) + 90.0) % 180.0) - 90.0)
    assert error <= 0.5


def _footprint_grid() -> RasterGrid:
    return RasterGrid(-0.5, 3.5, 0.01, 500, 500)


def _height_field(mask: np.ndarray, height: float, *, seed: int = 20261111) -> np.ndarray:
    """Synthetic 3 mm height noise used by footprint fitting tests."""
    rng = np.random.default_rng(seed)
    result = rng.normal(0.0, 0.003, mask.shape).astype(np.float32)
    result[mask] += height
    return result


def test_footprint_rectangle_table() -> None:
    grid = _footprint_grid()
    theta = math.radians(3.0)
    truth = affinity.rotate(box(0.7, 1.1, 1.9, 1.9), 3.0, origin="centroid")
    mask = _boundary_noise(_rasterise_polygon(truth, grid), seed=73, probability=0.01)
    fit = fit_footprint(mask, grid, _height_field(mask, 0.74), theta, 1.0, edge_snap=False)
    assert fit is not None and fit.kind == "rect"
    assert len(fit.polygon.exterior.coords) - 1 == 4
    truth_edges = [LineString((first, second)) for first, second in zip(truth.exterior.coords, list(truth.exterior.coords)[1:])]
    fitted_edges = [LineString((first, second)) for first, second in zip(fit.polygon.exterior.coords, list(fit.polygon.exterior.coords)[1:])]
    distances = [min(edge.distance(expected) for expected in truth_edges) for edge in fitted_edges]
    assert distances, distances
    assert max(distances) <= 0.025


def test_footprint_round_stool_is_circle() -> None:
    grid = _footprint_grid()
    truth = Point(1.4, 1.8).buffer(0.22, quad_segs=64)
    mask = _boundary_noise(_rasterise_polygon(truth, grid), seed=74)
    fit = fit_footprint(mask, grid, _height_field(mask, 0.36), math.radians(3.0), 1.0)
    assert fit is not None and fit.kind == "circle"
    assert fit.radius is not None and abs(fit.radius - 0.22) <= 0.01


def test_footprint_oval_table_is_ellipse() -> None:
    grid = _footprint_grid()
    truth = affinity.scale(Point(1.4, 1.8).buffer(1.0, quad_segs=64), 0.7, 0.45)
    mask = _boundary_noise(_rasterise_polygon(truth, grid), seed=75)
    fit = fit_footprint(mask, grid, _height_field(mask, 0.74), math.radians(3.0), 1.0)
    assert fit is not None and fit.kind == "ellipse"


def test_footprint_l_sofa_regularised() -> None:
    grid = _footprint_grid()
    theta = math.radians(3.0)
    sofa = Polygon(((0.6, 0.8), (1.95, 0.8), (1.95, 1.2), (1.1, 1.2), (1.1, 2.25), (0.6, 2.25)))
    truth = affinity.rotate(sofa, 3.0, origin="centroid")
    mask = _boundary_noise(_rasterise_polygon(truth, grid), seed=76)
    fit = fit_footprint(mask, grid, _height_field(mask, 0.45), theta, 1.0, edge_snap=False)
    assert fit is not None and fit.kind == "regularised"
    assert len(fit.polygon.exterior.coords) - 1 <= 8
    assert fit.dev90 <= 0.02
    directions = np.diff(np.asarray(fit.polygon.exterior.coords), axis=0)
    angles = np.mod(np.arctan2(directions[:, 1], directions[:, 0]) - theta, math.pi / 2.0)
    assert np.all(np.minimum(angles, math.pi / 2.0 - angles) <= math.radians(0.5))


def test_footprint_parallel_step_connector() -> None:
    grid = _footprint_grid()
    truth = Polygon(((0.6, 0.8), (2.0, 0.8), (2.0, 1.2), (1.1, 1.2), (1.1, 1.8), (2.0, 1.8), (2.0, 2.2), (0.6, 2.2)))
    mask = _rasterise_polygon(truth, grid)
    fit = fit_footprint(mask, grid, _height_field(mask, 0.55), 0.0, 1.0, edge_snap=False)
    assert fit is not None and fit.polygon.is_valid and not fit.polygon.exterior.is_simple is False
    coordinates = np.asarray(fit.polygon.exterior.coords)
    assert np.any(np.isclose(coordinates[:, 0], 1.1, atol=0.04))


def test_footprint_sofa_height_step_detail() -> None:
    grid = _footprint_grid()
    mask = _rasterise_polygon(box(0.7, 0.9, 2.1, 1.9), grid)
    hm = _height_field(mask, 0.45)
    hm[_rasterise_polygon(box(0.7, 1.65, 2.1, 1.9), grid)] += 0.40
    fit = fit_footprint(mask, grid, hm, 0.0, 1.0)
    assert fit is not None
    instance = ObjectInstance(mask, "sofa", 117, True, 0.85, 1.0)
    details = height_step_details(instance, hm, grid, 0.0, 1.0, fit.polygon)
    assert details and min(line.distance(LineString(((0.7, 1.65), (2.1, 1.65)))) for line in details) <= 0.02


def test_wall_residue_strip_removed() -> None:
    grid = RasterGrid(0.0, 3.0, 0.01, 300, 300)
    dsm = np.zeros((300, 300), dtype=np.float32)
    classes = np.zeros_like(dsm, dtype=np.uint8)
    dsm[:, :6] = 2.4
    dsm[80:180, 20:80] = 2.0
    classes[80:180, 20:80] = 117
    instances, _metrics, _hm = object_instances(dsm, 0.0, classes, np.zeros_like(classes), np.zeros_like(dsm), [], grid, 1.0, 2.5, [LineString(((0.0, 0.0), (0.0, 3.0)))])
    assert len(instances) == 1
    assert instances[0].mask.sum() >= 5000


def test_noise_blob_dropped() -> None:
    grid = RasterGrid(0.0, 2.0, 0.01, 200, 200)
    dsm = np.zeros((200, 200), dtype=np.float32)
    classes = np.zeros_like(dsm, dtype=np.uint8)
    dsm[60:140, 60:140] = 0.5
    dsm[60:92, 60:140] = np.nan
    instances, metrics, _hm = object_instances(dsm, 0.0, classes, np.zeros_like(classes), np.zeros_like(dsm), [], grid, 1.0, 2.5)
    assert not instances
    assert metrics["footprints_noise_dropped"] == 1


def test_clutter_rectangle_rule() -> None:
    grid = RasterGrid(0.0, 2.0, 0.01, 200, 200)
    dsm = np.zeros((200, 200), dtype=np.float32)
    classes = np.zeros_like(dsm, dtype=np.uint8)
    classes[40:90, 40:140] = 180
    dsm[40:90, 40:140] = 0.5
    classes[120:148, 40:69] = 180
    dsm[120:148, 40:69] = 0.5
    instances, _metrics, hm = object_instances(dsm, 0.0, classes, np.zeros_like(classes), np.zeros_like(dsm), [], grid, 1.0, 2.5)
    clutter = [item for item in instances if item.code == 180 and item.mask.sum() * grid.pixel_size**2 >= 0.15]
    assert len(clutter) == 1
    fit = fit_footprint(clutter[0].mask, grid, hm, 0.0, 1.0)
    assert fit is not None and len(fit.polygon.exterior.coords) - 1 == 4


def test_fixture_bathtub_still_p_sani() -> None:
    """Bathtub recognition remains on the block-insert fixture path."""
    assert guess_fixture(121, 1.7, 0.8, 0.55, "bathtub") == "BATHTUB"
    face = WallFace("line", np.empty((0, 2)), line=LineString(((0.0, 0.0), (4.0, 0.0))))
    insert: dict[str, object] = {"type": "insert", "name": "BATHTUB", "point": [2.0, 0.43], "rotation": 0.0, "xscale": 1.7, "yscale": 0.8}
    assert _align_insert_to_wall(insert, [face], 1.0, 12.0, 0.05)


def test_no_detail_lines_for_clutter_and_full_area() -> None:
    grid = RasterGrid(0.0, 2.0, 0.01, 200, 200)
    mask = _rasterise_polygon(box(0.4, 0.4, 1.6, 1.6), grid)
    hm = _height_field(mask, 0.45)
    clutter = ObjectInstance(mask, None, 180, False, 0.45, 1.0)
    outline = fit_footprint(mask, grid, hm, 0.0, 1.0)
    assert outline is not None
    assert height_step_details(clutter, hm, grid, 0.0, 1.0, outline.polygon) == []
    sofa = ObjectInstance(mask, "sofa", 117, True, 0.85, 1.0)
    hm[_rasterise_polygon(box(0.4, 1.3, 1.6, 1.6), grid)] += 0.40
    assert len(height_step_details(sofa, hm, grid, 0.0, 1.0, outline.polygon)) <= 2


def test_object_without_floor_contact_dropped() -> None:
    grid = RasterGrid(0.0, 2.0, 0.01, 200, 200)
    dsm = np.zeros((200, 200), dtype=np.float32); dsm[70:130, 70:130] = 0.5
    classes = np.zeros_like(dsm, dtype=np.uint8); classes[70:130, 70:130] = 117
    empty = np.zeros_like(classes, dtype=bool)
    dropped, metrics, _hm = object_instances(dsm, 0.0, classes, np.zeros_like(classes), np.zeros_like(dsm), [], grid, 1.0, 2.5, free_floor=empty)
    floor = np.ones_like(classes, dtype=bool)
    kept, _metrics, _hm = object_instances(dsm, 0.0, classes, np.zeros_like(classes), np.zeros_like(dsm), [], grid, 1.0, 2.5, free_floor=floor)
    assert not dropped and metrics["footprints_noise_dropped"] == 1 and len(kept) == 1


def test_clutter_rectangle_rejected_for_spread_blob() -> None:
    grid = RasterGrid(0.0, 3.0, 0.01, 300, 300)
    spread = _rasterise_polygon(Polygon(((0.3, 0.3), (2.5, 0.3), (2.5, 0.5), (0.7, 0.5), (0.7, 2.2), (0.3, 2.2))), grid)
    assert clutter_rectangle(spread, grid, 0.0, 1.0) is None
    compact = _rasterise_polygon(box(0.5, 0.5, 1.5, 1.1), grid)
    assert clutter_rectangle(compact, grid, 0.0, 1.0) is not None
    floor = compact.copy()
    assert clutter_rectangle(compact, grid, 0.0, 1.0, floor) is None


def test_low_view_support_blob_dropped() -> None:
    grid = RasterGrid(0.0, 2.0, 0.01, 200, 200)
    dsm = np.zeros((200, 200), dtype=np.float32); dsm[70:130, 70:130] = 0.5
    classes = np.zeros_like(dsm, dtype=np.uint8); classes[70:130, 70:130] = 117
    floor = np.ones_like(classes, dtype=bool); views = np.full_like(classes, 10, dtype=np.uint16); views[70:130, 70:130] = 2
    dropped, metrics, _hm = object_instances(dsm, 0.0, classes, np.zeros_like(classes), np.zeros_like(dsm), [], grid, 1.0, 2.5, free_floor=floor, view_count=views)
    views[70:130, 70:130] = 10
    kept, _metrics, _hm = object_instances(dsm, 0.0, classes, np.zeros_like(classes), np.zeros_like(dsm), [], grid, 1.0, 2.5, free_floor=floor, view_count=views)
    assert not dropped and metrics["footprints_noise_dropped"] == 1 and len(kept) == 1


def test_round_object_prefers_circle_over_rect() -> None:
    grid = _footprint_grid(); disk = Point(1.4, 1.8).buffer(0.22, quad_segs=48)
    mask = _boundary_noise(_rasterise_polygon(disk, grid), seed=91)
    fit = fit_footprint(mask, grid, _height_field(mask, 0.36), 0.0, 1.0)
    assert fit is not None and fit.kind in {"circle", "ellipse"}


def test_table_semantic_prior_rect() -> None:
    grid = _footprint_grid(); mask = _rasterise_polygon(box(0.6, 1.0, 2.2, 1.8), grid)
    mask = _boundary_noise(mask, seed=92); mask[200:202, 150:160] = False
    fit = fit_footprint(mask, grid, _height_field(mask, 0.75), 0.0, 1.0, semantic_label="table")
    assert fit is not None and fit.kind in {"rect", "rect_free"}


def test_object_on_sofa_merged() -> None:
    grid = RasterGrid(0.0, 3.0, 0.01, 300, 300)
    dsm = np.zeros((300, 300), dtype=np.float32); classes = np.zeros_like(dsm, dtype=np.uint8)
    dsm[70:220, 60:220] = 0.45; dsm[120:170, 120:175] = 0.75; classes[dsm > 0] = 117
    fine = np.zeros_like(classes); fine[70:220, 60:220] = 1
    instances, _metrics, _hm = object_instances(dsm, 0.0, classes, fine, np.where(fine, 0.9, 0.0), ["sofa"], grid, 1.0, 2.5, free_floor=np.ones_like(classes, dtype=bool))
    assert len([item for item in instances if item.label == "sofa"]) == 1


@pytest.mark.parametrize("core_label", ["bed", "mattress"])
def test_bed_outline_covers_bedding(core_label: str) -> None:
    """The fine label names the bed; blanket / cushion pieces never shrink its outline."""
    grid = _footprint_grid()
    origin = (1.6, 1.4)
    theta = math.radians(4.0)

    def rotated(polygon: Polygon) -> np.ndarray:
        return _rasterise_polygon(affinity.rotate(polygon, 4.0, origin=origin), grid)

    full = rotated(box(0.6, 0.6, 2.6, 2.2))
    core = rotated(box(1.1, 0.9, 2.1, 1.86))  # about 30 % of the bed
    head = rotated(box(0.6, 1.86, 2.6, 2.2))  # pillows
    sides = full & ~core & ~head
    rng = np.random.default_rng(20261112)
    fine = np.zeros(full.shape, dtype=np.uint8)
    fine[core] = 1
    fine[head] = 3  # cushion
    fine[sides] = 2  # blanket
    unlabelled = full & (rng.random(full.shape) < 0.02)
    fine[unlabelled & ~core] = 0  # a few pixels without any fine label
    dsm = _height_field(full, 0.55)
    dsm[head] += 0.12
    classes = np.where(full, 117, 0).astype(np.uint8)
    names = [core_label, "blanket", "cushion"]
    instances, metrics, height_map = object_instances(dsm, 0.0, classes, fine, np.where(fine > 0, 0.9, 0.0).astype(np.float32), names, grid, 1.0, 2.4)
    assert len(instances) == 1 and instances[0].label == "bed" and metrics["cushions_merged"] >= 1
    mask_area = float(np.count_nonzero(instances[0].mask)) * grid.pixel_size**2
    assert mask_area >= 0.90 * float(np.count_nonzero(full)) * grid.pixel_size**2
    fit = fit_footprint(instances[0].mask, grid, height_map, theta, 1.0, fine_label=True, semantic_label="bed")
    assert fit is not None and fit.kind in {"rect", "rect_free"} and len(fit.polygon.exterior.coords) == 5
    assert abs(fit.polygon.area - mask_area) <= 0.10 * mask_area
    assert fit.polygon.area >= 2.5 * float(np.count_nonzero(core)) * grid.pixel_size**2  # not the small bed-pixel core
    assert _furniture_display_name(instances[0].label, {}) == "Bett"


def test_dining_chairs_become_rectangles() -> None:
    grid = _footprint_grid()
    origin = (1.4, 1.2)
    theta = math.radians(3.0)

    def rotated(polygon: Polygon) -> np.ndarray:
        return _rasterise_polygon(affinity.rotate(polygon, 3.0, origin=origin), grid)

    table = rotated(box(0.8, 0.8, 2.0, 1.6))
    chairs = [
        rotated(box(0.8, 1.45, 1.25, 1.95).union(box(0.75, 1.7, 1.3, 1.95)).difference(box(1.1, 1.75, 1.2, 1.85))),  # irregular, overlaps the table
        rotated(box(1.55, 1.45, 2.0, 1.95).union(box(1.5, 1.6, 1.6, 2.0))),
        rotated(box(0.35, 0.55, 0.9, 1.0).difference(box(0.4, 0.6, 0.5, 0.7))),
        rotated(box(1.95, 0.7, 2.45, 1.15).union(box(1.9, 1.1, 2.4, 1.25))),
    ]
    blob = rotated(box(1.2, 1.0, 1.6, 1.3))  # unlabelled furniture inside the table group
    fine = np.zeros(table.shape, dtype=np.uint8)
    dsm = np.zeros(table.shape, dtype=np.float32)
    for chair in chairs:
        fine[chair] = 2
        dsm[chair] = 0.85
    fine[table & ~blob] = 1  # the table wins the overlapping pixels
    dsm[table] = 0.77
    dsm[blob] = 0.68
    dsm += np.random.default_rng(1).normal(0.0, 0.003, dsm.shape).astype(np.float32)
    mask = table | blob | np.logical_or.reduce(chairs)
    classes = np.where(mask, 117, 0).astype(np.uint8)
    instances, _metrics, height_map = object_instances(dsm, 0.0, classes, fine, np.where(fine > 0, 0.9, 0.0).astype(np.float32), ["table", "chair"], grid, 1.0, 2.4)
    stats: dict[str, int] = {}
    refined = refine_dining_instances(instances, classes, grid, height_map, theta, 1.0, stats)
    tables = [item for item in refined if item.label == "table"]
    chair_items = [item for item in refined if item.label == "chair"]
    assert len(tables) == 1 and len(chair_items) == 4 and len(refined) == 5
    table_fit = fit_footprint(tables[0].mask, grid, height_map, theta, 1.0, fine_label=True, semantic_label="table")
    assert table_fit is not None and table_fit.kind in {"rect", "rect_free"} and len(table_fit.polygon.exterior.coords) == 5
    assert table_fit.polygon.area >= 0.90 * float(np.count_nonzero(table)) * grid.pixel_size**2
    assert _furniture_display_name("table", {}) == "Tisch"
    for chair in chair_items:
        assert chair.polygon is not None and len(chair.polygon.exterior.coords) == 5
        corners = list(chair.polygon.exterior.coords)
        sides = [math.dist(corners[index], corners[index + 1]) for index in range(4)]
        assert all(0.35 - 1e-6 <= side <= 0.65 + 1e-6 for side in sides)
        assert chair.polygon.intersection(table_fit.polygon).area <= 1e-4


def test_all_fine_labels_have_german_names() -> None:
    config = json.loads((Path(__file__).resolve().parents[1] / "config" / "classes.json").read_text(encoding="utf-8"))
    terms = list(config["fine_terms"]["indoor"])
    aliases = dict(config["fine_aliases"])
    canonical = sorted({aliases.get(term, term) for term in terms})
    assert len(canonical) == 34
    english_leaks = {"chest of drawers", "dining table", "bookcase", "couch", "mattress", "kitchen counter", "kitchen cabinet", "stove", "kitchen sink", "toilet", "washbasin", "bathtub", "shower tray"}
    for label in terms:  # aliases included
        name = _furniture_display_name(label, {})
        assert label in FINE_LABEL_GERMAN_NAMES, label
        assert name == FINE_LABEL_GERMAN_NAMES[label] and name.lower() not in english_leaks
        assert name.lower() != label or label in {"sofa", "sideboard"}, label
    expected = {"chest of drawers": "Kommode", "dining table": "Esstisch", "bookcase": "Bücherregal", "couch": "Sofa", "mattress": "Bett", "kitchen counter": "Küche", "kitchen cabinet": "Küche", "stove": "Herd", "kitchen sink": "Spüle", "toilet": "WC", "washbasin": "Lavabo", "bathtub": "Badewanne", "shower tray": "Dusche"}
    for label, german in expected.items():
        assert _furniture_display_name(label, {}) == german


def _einb_piece(x0: float, x1: float, owner: int, text: str = "Küche h=0.91") -> list[dict[str, object]]:
    polygon = box(x0, 0.0, x1, 0.6)
    return [
        {"type": "polyline", "points": [[x, y] for x, y in polygon.exterior.coords[:-1]], "closed": True, "holes": [], "_einb_kind": "counter", "_einb_id": owner},
        {"type": "text", "text": text, "point": list(polygon.representative_point().coords[0]), "height": 0.10, "rotation": 0.0, "_einb_owner": owner},
    ]


def test_einb_same_label_merged_single_text() -> None:
    entities = _einb_piece(0.0, 1.0, 1) + _einb_piece(1.03, 2.0, 2) + _einb_piece(2.05, 3.0, 3)
    entities.append({"type": "text", "text": "Küche h=1.09", "point": [9.0, 9.0], "height": 0.10, "rotation": 0.0, "_einb_owner": 99})  # owner rejected earlier
    entities.append({"type": "polyline", "points": [[0.0, -0.3], [3.0, -0.3]], "closed": False, "_einb_kind": "segment"})
    entities.append({"type": "polyline", "points": [[0.1, -0.31], [2.9, -0.31]], "closed": False, "_einb_kind": "segment"})  # duplicate
    result, stats = tidy_builtins(entities, [], 1.0)
    polygons = [item for item in result if item["type"] == "polyline" and item.get("closed")]
    texts = [item for item in result if item["type"] == "text"]
    segments = [item for item in result if item["type"] == "polyline" and not item.get("closed")]
    assert len(polygons) == 1 and len(texts) == 1 and len(segments) == 1
    merged = Polygon(polygons[0]["points"])
    assert len(polygons[0]["points"]) == 4 and merged.area == pytest.approx(3.0 * 0.6, rel=0.03)
    assert merged.contains(Point(texts[0]["point"])) and texts[0]["text"].startswith("Küche")
    assert stats["before"] == len(entities) and stats["after"] == len(result)
    # Different labels never merge, even when they touch.
    mixed, _stats = tidy_builtins(_einb_piece(0.0, 1.0, 1) + _einb_piece(1.0, 2.0, 2, "Einbau h=0.91"), [], 1.0)
    assert sum(item["type"] == "polyline" for item in mixed) == 2 and sum(item["type"] == "text" for item in mixed) == 2


def test_einb_labels_do_not_overlap_other_labels() -> None:
    entities = _einb_piece(0.0, 1.5, 1, "Küche") + _einb_piece(5.0, 5.2, 2, "Einbau h=0.91")
    obstacles = [
        {"type": "text", "text": "Sofa h=0.83", "point": [0.75, 0.30], "height": 0.12, "rotation": 0.0},  # sits on the big label
        {"type": "text", "text": "Stuhl h=0.50", "point": [5.1, 0.30], "height": 0.50, "rotation": 0.0},  # blocks the tiny polygon
    ]
    result, stats = tidy_builtins(entities, obstacles, 1.0)
    texts = [item for item in result if item["type"] == "text"]
    assert len(texts) == 1 and texts[0]["text"] == "Küche" and stats["labels_moved"] == 1  # tiny polygon (< 0.3 m2) drops its label
    assert Polygon(result[0]["points"]).contains(Point(texts[0]["point"]))
    from samcloud.ortho.vector.vertical import _text_box
    assert not _text_box(texts[0]).intersects(_text_box(obstacles[0]))


def _chair_scene(height: float, box_chair: Polygon, grid: RasterGrid) -> tuple[list[ObjectInstance], np.ndarray, np.ndarray]:
    chair = _rasterise_polygon(box_chair, grid)
    fine = np.where(chair, 1, 0).astype(np.uint8)
    dsm = _height_field(chair, height)
    classes = np.where(chair, 117, 0).astype(np.uint8)
    instances, _metrics, height_map = object_instances(dsm, 0.0, classes, fine, np.where(fine > 0, 0.9, 0.0).astype(np.float32), ["chair"], grid, 1.0, 2.6)
    return instances, classes, height_map


def test_tall_chair_label_not_drawn_as_chair() -> None:
    grid = _footprint_grid()
    for height, is_chair in ((0.45, True), (1.68, False), (1.25, False), (1.13, False)):
        instances, classes, height_map = _chair_scene(height, box(1.0, 1.0, 1.45, 1.45), grid)
        assert len(instances) == 1
        stats: dict[str, int] = {}
        refined = refine_dining_instances(instances, classes, grid, height_map, 0.0, 1.0, stats)
        assert len(refined) == 1
        if is_chair:
            assert refined[0].label == "chair" and refined[0].polygon is not None
            assert _furniture_display_name(refined[0].label, {}) == "Stuhl"
        else:
            assert refined[0].polygon is None and refined[0].label is None and not refined[0].has_fine_label
            assert stats["chairs_implausible"] == 1 and _furniture_display_name(refined[0].label, {}) == "Möbel"


def test_chair_rectangle_clipped_to_room() -> None:
    grid = _footprint_grid()
    wall = LineString(((0.0, 1.2), (3.0, 1.2)))  # horizontal wall, the room lies above it (y > 1.2)
    room = box(0.0, 1.2, 3.0, 3.4)
    # Chair mostly in the room, 7 cm through the wall.
    instances, classes, height_map = _chair_scene(0.45, box(1.0, 1.13, 1.45, 1.65), grid)
    stats: dict[str, int] = {}
    refined = refine_dining_instances(instances, classes, grid, height_map, 0.0, 1.0, stats, [wall], room)
    assert len(refined) == 1 and refined[0].polygon is not None
    chair = refined[0].polygon
    assert not chair.intersects(wall) and room.buffer(0.02).contains(chair)
    # Chair standing almost entirely behind the wall (clipped piece < 0.08 m2) is dropped.
    instances, classes, height_map = _chair_scene(0.45, box(1.0, 0.75, 1.45, 1.25), grid)
    stats = {}
    refined = refine_dining_instances(instances, classes, grid, height_map, 0.0, 1.0, stats, [wall], room)
    assert all(item.polygon is None or (not item.polygon.intersects(wall) and room.buffer(0.02).contains(item.polygon)) for item in refined)



def test_clipped_chair_refit_to_rectangle() -> None:
    grid = _footprint_grid()
    # A wall running at about 20 degrees cuts the chair corner: a plain clip leaves a pentagon.
    wall = LineString(((0.0, 0.64), (3.0, 1.73)))
    room = Polygon(((0.0, 0.64), (3.0, 1.73), (3.0, 3.4), (0.0, 3.4)))
    instances, classes, height_map = _chair_scene(0.45, box(1.0, 1.0, 1.45, 1.45), grid)
    stats: dict[str, int] = {}
    refined = refine_dining_instances(instances, classes, grid, height_map, 0.0, 1.0, stats, [wall], room)
    assert len(refined) == 1 and refined[0].polygon is not None
    chair = refined[0].polygon
    assert len(chair.exterior.coords) == 5 and not chair.intersects(wall) and room.buffer(0.02).contains(chair)
    assert chair.area >= 0.08
    # Direct refit of a clipped piece: oriented like the wall or the original, rectangle inside the piece.
    original = box(1.0, 1.0, 1.5, 1.5)
    clipped = original.intersection(Polygon(((0.9, 1.25), (1.6, 1.05), (1.6, 1.6), (0.9, 1.6))))
    refit = refit_clipped(clipped, original, [], 1.0, force_rectangle=True)
    assert refit is not None and len(refit.exterior.coords) == 5 and clipped.buffer(0.01).contains(refit)
    assert refit.area >= 0.5 * clipped.area


def test_white_table_top_with_chairs_mislabelled_as_table_pieces() -> None:
    """A white top returns few points: the instance labels land on the chairs ("table" pieces straddling the top
    edge, a "dining table" piece at its end) while the top itself is an unlabelled flat blob at table height.
    The top becomes the table, every chair its own seat-sized box beside it - also two chairs side by side."""
    from samcloud.ortho.plan import _furniture_label_text

    grid = _footprint_grid()
    rng = np.random.default_rng(20261117)
    top = box(1.0, 0.0, 1.9, 1.8)
    seats = {  # centre of each real chair -> (label of its piece, piece polygon)
        (0.83, 0.42): ("table", box(0.62, 0.17, 1.12, 0.67)),
        (0.83, 1.00): ("chair", box(0.63, 0.74, 1.08, 1.24)),  # 7 cm from its neighbour: must not merge
        (2.07, 0.42): ("table", box(1.80, 0.18, 2.30, 0.66)),
        (2.07, 1.20): ("table", box(1.82, 0.96, 2.31, 1.45)),
        (1.45, 2.03): ("chair", box(1.21, 1.72, 1.69, 2.26)),
        (1.45, -0.23): ("dining table", box(1.20, -0.47, 1.70, 0.08)),
    }
    hm = rng.normal(0.0, 0.003, (grid.height, grid.width)).astype(np.float32)
    top_mask = _rasterise_polygon(top, grid)
    hm[top_mask & (rng.random(top_mask.shape) < 0.6)] += 0.75  # sparse points on the white top
    instances = []
    for _centre, (label, piece) in seats.items():
        mask = _rasterise_polygon(piece, grid)
        hm[mask & ~top_mask] = 0.47
        back = mask & ~top_mask & (rng.random(mask.shape) < 0.25)
        hm[back] = 0.86
        instances.append(ObjectInstance(mask, label, 117, True, 0.86, 0.9))
    instances.append(ObjectInstance(top_mask, None, 117, False, 0.75, 0.95))
    classes = np.where(np.logical_or.reduce([item.mask for item in instances]), 117, 0).astype(np.uint8)
    stats: dict[str, int] = {}
    refined = refine_dining_instances(instances, classes, grid, hm, 0.0, 1.0, stats)
    tables = [item for item in refined if item.label in {"table", "dining table"}]
    chairs = [item for item in refined if item.label == "chair"]
    assert stats["dining_tops_from_unlabelled"] == 1 and stats["dining_table_pieces_to_chairs"] == 4
    assert len(tables) == 1 and len(chairs) == 6
    table_fit = tables[0].polygon or fit_footprint(tables[0].mask, grid, hm, 0.0, 1.0, fine_label=True, semantic_label="table").polygon
    x0, y0, x1, y1 = table_fit.bounds
    assert abs(x0 - 1.0) <= 0.05 and abs(x1 - 1.9) <= 0.05 and abs(y0 - 0.0) <= 0.05 and abs(y1 - 1.8) <= 0.05
    centres = list(seats)
    for chair in chairs:
        corners = list(chair.polygon.exterior.coords)
        sides = [math.dist(corners[index], corners[index + 1]) for index in range(4)]
        assert all(0.40 - 1e-6 <= side <= 0.55 + 1e-6 for side in sides)
        assert chair.polygon.intersection(table_fit).area <= 1e-3
        nearest = min(centres, key=lambda centre: math.dist(centre, chair.polygon.centroid.coords[0]))
        assert math.dist(nearest, chair.polygon.centroid.coords[0]) <= 0.15
        centres.remove(nearest)  # one box per chair
    for first in range(len(chairs)):
        for second in range(first + 1, len(chairs)):
            assert chairs[first].polygon.intersection(chairs[second].polygon).area <= 1e-3
    assert _furniture_label_text("Stuhl", "chair", 0.86) == "Stuhl" and _furniture_label_text("Tisch", "table", 0.75) == "Tisch h=0.75"
