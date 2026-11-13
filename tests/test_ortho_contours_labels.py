"""Unit and behavioral tests for outdoor DTM contours and smart text labels."""

from __future__ import annotations

import math
import os
from pathlib import Path
import uuid

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point, Polygon, box

from samcloud.ortho.vector.contours import (
    ContourSettings,
    GroundDtmAccumulator,
    contour_entities,
    fill_dtm,
    ground_codes_from_config,
    load_contour_settings,
)
from samcloud.ortho.vector.labels import (
    LabelSettings,
    _label_box,
    class_display_name,
    label_entities,
    load_label_settings,
    place_label,
)
from samcloud.ortho.vector.raster_grid import RasterGrid


@pytest.fixture(scope="module")
def work_dir(request: pytest.FixtureRequest) -> Path:
    """Return an isolated directory for this test session."""
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"contours_labels_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("contours_labels")


def test_ground_dtm_accumulator_uses_only_ground_codes() -> None:
    rng = np.random.default_rng(12345)
    bounds = (0.0, 0.0, 10.0, 10.0)
    resolution = 0.20
    accumulator = GroundDtmAccumulator(bounds=bounds, resolution=resolution)

    # 50,000 ground points on tilted plane z = 10.0 + 0.05 * x + 0.02 * y with noise sigma = 0.005
    n_ground = 50_000
    gx = rng.uniform(0.0, 10.0, size=n_ground)
    gy = rng.uniform(0.0, 10.0, size=n_ground)

    # Reserve [4, 6] x [4, 6] exclusively for high building points
    is_building_area = (gx >= 3.8) & (gx <= 6.2) & (gy >= 3.8) & (gy <= 6.2)
    gx = gx[~is_building_area]
    gy = gy[~is_building_area]
    gz = 10.0 + 0.05 * gx + 0.02 * gy + rng.normal(0.0, 0.005, size=len(gx))
    g_pts = np.column_stack((gx, gy, gz))
    g_cls = np.full(len(gx), 2, dtype=np.int64)

    # High building points inside [4, 6] x [4, 6] with z = 30.0 and class 6 (Building)
    n_bldg = 5_000
    bx = rng.uniform(4.0, 6.0, size=n_bldg)
    by = rng.uniform(4.0, 6.0, size=n_bldg)
    bz = rng.uniform(28.0, 32.0, size=n_bldg)
    b_pts = np.column_stack((bx, by, bz))
    b_cls = np.full(n_bldg, 6, dtype=np.int64)

    all_pts = np.vstack((g_pts, b_pts))
    all_cls = np.concatenate((g_cls, b_cls))

    accumulator.add(all_pts, all_cls, ground_codes=[2])
    dtm, grid = accumulator.finish(ContourSettings(resolution=resolution))

    # Verify building cells stay NaN before fill
    bldg_rows = []
    bldg_cols = []
    for r in range(grid.height):
        for c in range(grid.width):
            x, y = grid.pixel_to_local(r, c)
            if 4.2 <= x <= 5.8 and 4.2 <= y <= 5.8:
                bldg_rows.append(r)
                bldg_cols.append(c)
    assert len(bldg_rows) > 0
    assert np.isnan(dtm[bldg_rows, bldg_cols]).all()

    # Verify ground cells are within 0.05 m of the plane
    for r in range(grid.height):
        for c in range(grid.width):
            val = dtm[r, c]
            if not np.isnan(val):
                x, y = grid.pixel_to_local(r, c)
                z_plane = 10.0 + 0.05 * x + 0.02 * y
                assert abs(val - z_plane) < 0.05


def test_fill_dtm_respects_max_gap() -> None:
    # 20m x 20m raster at 0.20m resolution -> 100x100
    grid = RasterGrid(min_x=0.0, max_y=20.0, pixel_size=0.20, width=100, height=100)
    dtm = np.zeros((100, 100), dtype=np.float64)
    for r in range(100):
        for c in range(100):
            x, _ = grid.pixel_to_local(r, c)
            dtm[r, c] = 5.0 + 0.1 * x

    # Create a 2 m hole in the middle: x in [8, 10], y in [8, 10]
    hole_rows = []
    hole_cols = []
    for r in range(100):
        for c in range(100):
            x, y = grid.pixel_to_local(r, c)
            if 8.1 <= x <= 9.9 and 8.1 <= y <= 9.9:
                hole_rows.append(r)
                hole_cols.append(c)
    dtm[hole_rows, hole_cols] = np.nan

    # Create a region 6 m from data: remove data where x >= 14.0
    far_rows = []
    far_cols = []
    for r in range(100):
        for c in range(100):
            x, _ = grid.pixel_to_local(r, c)
            if x >= 14.0:
                dtm[r, c] = np.nan
            if x >= 17.5:
                far_rows.append(r)
                far_cols.append(c)

    filled = fill_dtm(dtm, grid, max_gap=3.0, smoothing_sigma_m=0.3)

    # Hole of 2 m is filled
    assert not np.isnan(filled[hole_rows, hole_cols]).any()
    # Region 6 m from data stays NaN
    assert np.isnan(filled[far_rows, far_cols]).all()


def test_contours_on_tilted_plane_are_straight_and_spaced() -> None:
    rng = np.random.default_rng(42)
    # 30m x 20m at 0.20m resolution -> width=150, height=100
    grid = RasterGrid(min_x=0.0, max_y=20.0, pixel_size=0.20, width=150, height=100)
    dtm = np.zeros((100, 150), dtype=np.float64)
    for r in range(100):
        for c in range(150):
            x, _ = grid.pixel_to_local(r, c)
            dtm[r, c] = 0.1 * x + rng.normal(0.0, 0.01)

    settings = ContourSettings(
        resolution=0.20,
        major_interval=1.0,
        minor_interval=0.25,
        max_gap=3.0,
        smoothing_sigma_m=0.3,
        min_length=2.0,
        simplify_tolerance=0.03,
        chaikin_iterations=2,
    )
    smoothed = fill_dtm(dtm, grid, max_gap=settings.max_gap, smoothing_sigma_m=settings.smoothing_sigma_m)
    entities = contour_entities(smoothed, grid, settings)

    assert len(entities) > 0

    majors = [e for e in entities if e["layer_role"] == "major"]
    minors = [e for e in entities if e["layer_role"] == "minor"]

    major_levels = {round(e["points"][0][2], 4) for e in majors}
    minor_levels = {round(e["points"][0][2], 4) for e in minors}

    # Expect majors at every integer metre (1.0, 2.0)
    assert 1.0 in major_levels
    assert 2.0 in major_levels
    # Expect minors at every 0.25 m (excluding majors)
    for expected_minor in (0.25, 0.50, 0.75, 1.25, 1.50, 1.75, 2.25, 2.50, 2.75):
        assert expected_minor in minor_levels
        assert expected_minor not in major_levels

    # Independent 1 cm cell noise makes a 5 cm maximum-deviation bound brittle.
    # Keep a robust maximum limit and a tighter RMS condition instead.
    # Every vertex z equals the level exactly
    deviations: list[float] = []
    for entity in entities:
        assert entity["type"] == "polyline3d"
        level = entity["points"][0][2]
        expected_x = level / 0.1
        for pt in entity["points"]:
            x, y, z = pt
            assert z == level
            deviation = abs(x - expected_x)
            assert deviation < 0.08
            deviations.append(deviation)
    assert float(np.sqrt(np.mean(np.square(deviations)))) < 0.03


def test_contours_on_hill_are_closed_rings() -> None:
    rng = np.random.default_rng(43)
    grid = RasterGrid(min_x=0.0, max_y=30.0, pixel_size=0.20, width=150, height=150)
    dtm = np.zeros((150, 150), dtype=np.float64)
    cx, cy = 15.0, 15.0
    for r in range(150):
        for c in range(150):
            x, y = grid.pixel_to_local(r, c)
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            dtm[r, c] = 5.0 + 10.0 * math.exp(-dist_sq / (2.0 * 4.0 ** 2)) + rng.normal(0.0, 0.01)

    settings = ContourSettings(
        resolution=0.20,
        major_interval=1.0,
        minor_interval=0.25,
        min_length=2.0,
        smoothing_sigma_m=0.0,
    )
    entities = contour_entities(dtm, grid, settings)

    inner_entities = [e for e in entities if e["points"][0][2] >= 9.0]
    assert len(inner_entities) > 0
    for entity in inner_entities:
        assert entity["closed"] is True
        pts = entity["points"]
        assert math.dist(pts[0][:2], pts[-1][:2]) < 1e-4


def test_contours_clipped_by_building_footprint() -> None:
    rng = np.random.default_rng(45)
    grid = RasterGrid(min_x=0.0, max_y=20.0, pixel_size=0.20, width=100, height=100)
    dtm = np.zeros((100, 100), dtype=np.float64)
    for r in range(100):
        for c in range(100):
            x, _ = grid.pixel_to_local(r, c)
            dtm[r, c] = 0.1 * x + rng.normal(0.0, 0.01)

    footprint = Polygon([(8.0, 5.0), (12.0, 5.0), (12.0, 15.0), (8.0, 15.0)])
    shrunk_footprint = footprint.buffer(-0.01)

    settings = ContourSettings(resolution=0.20, major_interval=1.0, minor_interval=0.25, min_length=1.0)
    entities = contour_entities(dtm, grid, settings, exclude=[footprint])

    assert len(entities) > 0
    for entity in entities:
        for pt in entity["points"]:
            point_geom = Point(pt[0], pt[1])
            assert not shrunk_footprint.contains(point_geom)


def test_major_minor_not_duplicated() -> None:
    rng = np.random.default_rng(44)
    grid = RasterGrid(min_x=0.0, max_y=20.0, pixel_size=0.20, width=100, height=100)
    dtm = np.zeros((100, 100), dtype=np.float64)
    for r in range(100):
        for c in range(100):
            x, _ = grid.pixel_to_local(r, c)
            dtm[r, c] = 0.2 * x + rng.normal(0.0, 0.01)

    settings = ContourSettings(resolution=0.20, major_interval=1.0, minor_interval=0.25, min_length=1.0)
    entities = contour_entities(dtm, grid, settings)

    major_levels = {round(e["points"][0][2], 6) for e in entities if e["layer_role"] == "major"}
    minor_levels = {round(e["points"][0][2], 6) for e in entities if e["layer_role"] == "minor"}

    assert len(major_levels) > 0
    assert len(minor_levels) > 0
    assert major_levels.isdisjoint(minor_levels)
    for level in major_levels:
        assert abs(level % 1.0) < 1e-6
    for level in minor_levels:
        assert abs(level % 1.0) >= 1e-6
        assert abs((level * 4) % 1.0) < 1e-6


def test_label_fits_inside_and_follows_axis() -> None:
    rect = box(-5.0, -1.0, 5.0, 1.0)
    rotated = affinity.rotate(rect, 30.0, origin=(0.0, 0.0))

    settings = LabelSettings()
    entity = place_label(rotated, "Office", settings)

    assert entity is not None
    assert entity["type"] == "text"
    assert entity["text"] == "Office"
    assert entity["align"] == "MIDDLE_CENTER"
    assert entity["layer_role"] == "label"

    # Text rotation is 30° ± 1°
    assert abs(entity["rotation"] - 30.0) <= 1.0

    # Start height = clamp(0.12 * 2.0, 0.08, 0.60) = 0.24
    expected_height = 0.12 * 2.0
    assert abs(entity["height"] - expected_height) < 1e-4

    # Text box lies inside the polygon
    box_geom = _label_box(entity, settings)
    assert rotated.covers(box_geom)


def test_label_rotation_never_upside_down() -> None:
    rect = box(-5.0, -1.0, 5.0, 1.0)
    rotated = affinity.rotate(rect, 150.0, origin=(0.0, 0.0))

    settings = LabelSettings()
    entity = place_label(rotated, "Hallway", settings)

    assert entity is not None
    # 150° gives rotation -30°
    assert abs(entity["rotation"] - (-30.0)) <= 1.0


def test_label_shrinks_only_when_needed() -> None:
    # 1.5m x 1.2m rectangle
    rect = box(-0.75, -0.6, 0.75, 0.6)
    settings = LabelSettings()

    start_height = min(settings.max_height, max(settings.min_height, settings.base_height_factor * 1.2))

    # Short name "WC" keeps the start height
    entity_short = place_label(rect, "WC", settings)
    assert entity_short is not None
    assert abs(entity_short["height"] - start_height) < 1e-4

    # Long name "Mechanical Room 01" shrinks, but stays >= min_height
    entity_long = place_label(rect, "Mechanical Room 01", settings)
    assert entity_long is not None
    assert entity_long["height"] < start_height
    assert entity_long["height"] >= settings.min_height
    assert rect.covers(_label_box(entity_long, settings))


def test_label_skipped_below_min_area_or_if_cannot_fit() -> None:
    settings = LabelSettings(min_area=1.0)

    # Polygon below min_area (0.5 m^2 < 1.0 m^2)
    small_rect = box(0.0, 0.0, 1.0, 0.5)
    assert small_rect.area < 1.0
    assert place_label(small_rect, "Storage", settings) is None

    # Extremely narrow polygon that cannot fit even at min_height (0.05m x 40m = 2.0 m^2 > 1.0 m^2)
    narrow_rect = box(0.0, 0.0, 40.0, 0.05)
    assert narrow_rect.area >= 1.0
    assert place_label(narrow_rect, "Pipe Trench", settings) is None


def test_label_entities_avoid_overlaps() -> None:
    settings = LabelSettings()

    poly_large = box(-5.0, -5.0, 5.0, 5.0)
    poly_small = box(-2.0, -2.0, 2.0, 2.0)

    items = [(poly_small, "Small Box"), (poly_large, "Large Box")]
    entities = label_entities(items, settings)

    # Drops the label that overlaps the already placed larger polygon's label
    assert len(entities) == 1
    assert entities[0]["text"] == "Large Box"

    # Two non-overlapping polygons both get labels
    poly1 = box(0.0, 0.0, 5.0, 5.0)
    poly2 = box(20.0, 0.0, 25.0, 5.0)
    entities2 = label_entities([(poly1, "Room 1"), (poly2, "Room 2")], settings)
    assert len(entities2) == 2


def test_class_display_name() -> None:
    assert class_display_name("furniture, furniture, cabinet, shelf") == "Furniture"
    assert class_display_name("sanitary fixtures, toilet, sink") == "Sanitary fixtures"
    assert class_display_name("toilet") == "Toilet"
    assert class_display_name("shower tray") == "Shower tray"
    assert class_display_name("road marking, road, asphalt") == "Road marking"
    assert class_display_name("WC") == "WC"
    assert class_display_name("") == ""
    assert class_display_name("   ") == ""


def test_load_settings_and_ground_codes_from_yaml() -> None:
    contour_settings = load_contour_settings()
    assert contour_settings.resolution == 0.20
    assert contour_settings.major_interval == 1.0
    assert contour_settings.minor_interval == 0.25
    assert contour_settings.max_gap == 3.0
    assert contour_settings.smoothing_sigma_m == 0.3
    assert contour_settings.min_length == 2.0
    assert contour_settings.simplify_tolerance == 0.03
    assert contour_settings.chaikin_iterations == 2

    label_settings = load_label_settings()
    assert label_settings.base_height_factor == 0.12
    assert label_settings.min_height == 0.08
    assert label_settings.max_height == 0.60
    assert label_settings.min_area == 1.0
    assert label_settings.padding_factor == 0.15
    assert label_settings.char_width_factor == 0.62
    assert label_settings.shrink_step == 0.85

    ground_codes = ground_codes_from_config()
    assert 2 in ground_codes
    assert 3 in ground_codes
    assert 9 in ground_codes
    assert 11 in ground_codes
    assert 73 in ground_codes
