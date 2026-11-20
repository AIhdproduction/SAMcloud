"""Depth evidence in the indoor plan: query helpers, wall faces / thickness, furniture, upper cabinets, tight footprints.

All fixtures are synthetic and noisy (no exact ties, voxels not on cell borders, realistic 1.5 cm voxels and 0.05 m free-space cells).
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from samcloud.bim.depth_evidence import DepthEvidence
from samcloud.bim.depth_evidence import plan_use
from samcloud.bim.depth_evidence.plan_use import PlanDepth, open_plan_depth, plan_depth_fingerprint
from samcloud.dxf.las_source import Frame
from samcloud.ortho.vector.double_walls import (
    DoubleWallSettings,
    WallEvidence,
    WallRun,
    _Face,
    _depth_extend_faces,
    _depth_thickness,
)
from samcloud.ortho.vector.indoor import Opening

SETTINGS = DoubleWallSettings(double_line=True)
UNITS = pytest.mark.parametrize("upm", [1.0, 0.9891])


# --------------------------------------------------------------------------
# a synthetic depth product on disk
# --------------------------------------------------------------------------

def _write_product(project, xyz, class_id, source, n_views, confidence, gap, free_cells=None, free_through=None):
    product = project / "products" / "depth"
    product.mkdir(parents=True, exist_ok=True)
    count = len(xyz)
    np.savez(
        product / "depth_points.npz", xyz=np.asarray(xyz, np.float32), class_id=np.asarray(class_id, np.uint8),
        confidence=np.asarray(confidence, np.float32), n_views=np.asarray(n_views, np.uint16), source=np.asarray(source, np.uint8),
        gap_fill=np.asarray(gap, bool),
    )
    cells = np.zeros((0, 3), np.int32) if free_cells is None else np.asarray(free_cells, np.int32)
    through = np.full(len(cells), 4, np.uint16) if free_through is None else np.asarray(free_through, np.uint16)
    np.savez(
        product / "free_space.npz", origin=np.zeros(3, np.float32), voxel_m=np.float32(0.05), indices=cells, through=through,
        ending=np.zeros(len(cells), np.uint16),
    )
    return product


def _frame(scale: float = 1.0) -> Frame:
    return Frame(origin=np.zeros(3), scale_factor=scale)


@pytest.fixture(autouse=True)
def _clear_cache():
    plan_use._CACHE.clear()
    yield
    plan_use._CACHE.clear()


def _rng(seed: int = 3) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_no_product_or_switch_off_means_no_depth(tmp_path) -> None:
    assert open_plan_depth(tmp_path, _frame(), 0.0, 1.0) is None
    assert plan_depth_fingerprint(tmp_path) is None
    rng = _rng()
    xyz = rng.uniform(0, 2, (50, 3))
    _write_product(tmp_path, xyz, np.full(50, 2), np.full(50, 1), np.full(50, 3), np.full(50, 0.9), np.ones(50, bool))
    assert open_plan_depth(tmp_path, _frame(), 0.0, 1.0) is not None
    assert plan_depth_fingerprint(tmp_path) is not None
    assert open_plan_depth(tmp_path, _frame(), 0.0, 1.0, {"use_in_plan": False}) is None
    assert plan_depth_fingerprint(tmp_path, {"use_in_plan": False}) is None


def test_fingerprint_changes_with_settings(tmp_path) -> None:
    xyz = _rng().uniform(0, 2, (20, 3))
    _write_product(tmp_path, xyz, np.full(20, 2), np.full(20, 1), np.full(20, 3), np.full(20, 0.9), np.ones(20, bool))
    assert plan_depth_fingerprint(tmp_path) != plan_depth_fingerprint(tmp_path, {"plan_wall_min_views": 3})


def test_select_and_points_filter_source_views_confidence_and_gap(tmp_path) -> None:
    xyz = np.array([[0.1, 0.1, 1.0], [0.2, 0.1, 1.0], [0.3, 0.1, 1.0], [0.4, 0.1, 1.0], [0.5, 0.1, 1.0]]) + _rng().normal(0, 0.002, (5, 3))
    _write_product(
        tmp_path, xyz, [2, 2, 2, 17, 2], source=[1, 2, 3, 1, 1], n_views=[3, 5, 1, 4, 2], confidence=[0.9, 0.45, 0.9, 0.9, 0.9],
        gap=[True, True, True, True, False],
    )
    evidence = DepthEvidence.open(tmp_path, _frame())
    ring = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    points, classes, _confidence, views, source = evidence.points(ring, (0, 2), classes=(2,), gap_only=True, min_views=2, colmap_only=True, with_source=True)
    assert len(points) == 1 and views[0] == 3 and source[0] == 1 and classes[0] == 2
    assert int(evidence.select((2,), gap_only=True, min_views=2).sum()) == 2  # the Depth-Anything-only voxel (source 2) is allowed without colmap_only
    assert int(evidence.select(min_confidence=0.5).sum()) == 4


def test_wall_and_furniture_subsets_use_the_colmap_rule(tmp_path) -> None:
    rng = _rng(5)
    wall = np.column_stack((rng.uniform(0, 2, 200), 0.02 * rng.standard_normal(200), rng.uniform(0.4, 2.0, 200)))
    sofa = np.column_stack((rng.uniform(3, 4, 100), rng.uniform(0, 1, 100), 0.45 + 0.01 * rng.standard_normal(100)))
    xyz = np.vstack((wall, sofa))
    cls = np.r_[np.full(200, 2), np.full(100, 17)]
    # half of the wall voxels are Depth-Anything-only, half COLMAP; one view only for some
    source = np.r_[np.where(np.arange(200) % 2 == 0, 1, 2), np.full(100, 1)]
    views = np.r_[np.where(np.arange(200) % 5 == 0, 1, 3), np.full(100, 3)]
    _write_product(tmp_path, xyz, cls, source, views, np.full(300, 0.9), np.ones(300, bool))
    depth = open_plan_depth(tmp_path, _frame(), 0.0, 1.0)
    wall_points = depth.wall_points(0.3, 2.2)
    expected = int(np.sum((source[:200] == 1) & (views[:200] >= 2)))
    assert len(wall_points) == expected
    furniture, classes = depth.furniture_points(colmap_only=True)
    assert len(furniture) == 100 and set(classes) == {17}


def test_floor_is_subtracted_and_scale_applied(tmp_path) -> None:
    xyz = np.array([[1.0, 1.0, 1.0 + 0.5] for _ in range(30)]) + _rng().normal(0, 0.001, (30, 3))
    _write_product(tmp_path, xyz, np.full(30, 2), np.full(30, 1), np.full(30, 4), np.full(30, 0.9), np.ones(30, bool))
    depth = open_plan_depth(tmp_path, _frame(2.0), 0.4, 2.0)
    points = depth.wall_points(0.0, 10.0)
    assert points[:, 2].mean() == pytest.approx(2 * 1.5 - 0.4, abs=0.01)  # scaled by 2 and the floor 0.4 removed
    assert points[:, 0].mean() == pytest.approx(2.0, abs=0.01)


def test_free_share_counts_seen_empty_volume_and_ignores_solid_regions(tmp_path) -> None:
    # free cells (0.05 m) fill the box x 0..1, y 0..1, z 0.05..1.0, nothing in x 1..2
    cells = np.array([(i, j, k) for i in range(20) for j in range(20) for k in range(1, 20)])
    xyz = _rng().uniform(0, 1, (10, 3))
    _write_product(tmp_path, xyz, np.full(10, 2), np.full(10, 1), np.full(10, 4), np.full(10, 0.9), np.ones(10, bool), free_cells=cells)
    depth = open_plan_depth(tmp_path, _frame(), 0.0, 1.0)
    empty = box(0.1, 0.1, 0.9, 0.9)
    solid = box(1.2, 0.1, 1.8, 0.9)
    assert depth.free_share(empty, 0.1, 0.9) >= 0.8
    assert depth.is_seen_empty(empty, 0.1, 0.9)
    assert depth.free_share(solid, 0.1, 0.9) == 0.0
    assert not depth.is_seen_empty(solid, 0.1, 0.9)
    half = box(0.5, 0.1, 1.5, 0.9)  # half of it seen empty
    assert 0.3 < depth.free_share(half, 0.1, 0.9) < 0.7


def test_top_height_map_uses_robust_top_of_colmap_voxels(tmp_path) -> None:
    from samcloud.ortho.vector.raster_grid import RasterGrid

    rng = _rng(11)
    count = 4000
    xy = rng.uniform(0.0, 1.0, (count, 2))
    z = 0.45 + 0.01 * rng.standard_normal(count)
    stray = np.array([[0.505, 0.505, 1.9]])  # one stray voxel in a cell: ignored
    xyz = np.vstack((np.column_stack((xy, z)), stray))
    cls = np.full(len(xyz), 17)
    _write_product(tmp_path, xyz, cls, np.full(len(xyz), 1), np.full(len(xyz), 3), np.full(len(xyz), 0.9), np.ones(len(xyz), bool))
    depth = open_plan_depth(tmp_path, _frame(), 0.0, 1.0)
    grid = RasterGrid(0.0, 1.0, 0.05, 20, 20)
    top = depth.top_height_map(grid, z_low=0.05, z_high=2.3)
    cell = top[int((1.0 - 0.505) / 0.05), int(0.505 / 0.05)]
    assert np.isfinite(top).sum() >= 300
    assert cell == pytest.approx(0.47, abs=0.05)  # p95 of the cell, not the 1.9 m stray voxel
    assert np.nanmax(top) < 0.6


# --------------------------------------------------------------------------
# walls: face continuation and thickness from depth voxels
# --------------------------------------------------------------------------

def _wall_voxels(rng, a0, a1, y, upm, density=60, height=(0.35, 2.1), noise=0.006):
    count = int(max(1.0, (a1 - a0)) * density * 8)
    along = rng.uniform(a0, a1, count) * upm
    across = (y + noise * rng.standard_normal(count)) * upm
    z = rng.uniform(*height, count) * upm
    return np.column_stack((along, across, z))


def _face(upm, start, end, side=1):
    return _Face(0, np.asarray(start, float) * upm, np.asarray(end, float) * upm, {"type": "polyline", "points": [], "closed": False}, side=side)


@UNITS
def test_unseen_stretch_on_the_face_line_extends_the_face(upm) -> None:
    rng = _rng(21)
    face = _face(upm, (0.0, 0.0), (2.0, 0.0))
    voxels = np.vstack((_wall_voxels(rng, 2.0, 2.4, 0.0, upm), _wall_voxels(rng, -0.3, 0.0, 0.0, upm)))
    extended = _depth_extend_faces([face], voxels, [], upm, SETTINGS, WallEvidence(storey_height=2.5 * upm))
    assert extended == 2
    assert face.end[0] / upm == pytest.approx(2.4, abs=0.08)
    assert face.start[0] / upm == pytest.approx(-0.3, abs=0.08)


@UNITS
def test_voxels_off_the_line_or_beyond_the_reach_do_not_extend(upm) -> None:
    rng = _rng(22)
    face = _face(upm, (0.0, 0.0), (2.0, 0.0))
    off_line = _wall_voxels(rng, 2.0, 2.5, 0.12, upm)  # a parallel surface 12 cm away is another face
    far = _wall_voxels(rng, 2.9, 3.4, 0.0, upm)  # a gap of 0.9 m is more than the extension reach
    _depth_extend_faces([face], np.vstack((off_line, far)), [], upm, SETTINGS, WallEvidence(storey_height=2.5 * upm))
    assert face.end[0] / upm == pytest.approx(2.0, abs=1e-6)


@UNITS
def test_extension_stops_before_an_opening_span(upm) -> None:
    rng = _rng(23)
    face = _face(upm, (0.0, 0.0), (2.0, 0.0))
    door = Opening("DOOR", (2.45 * upm, 0.0), 0.9 * upm, 0.0, 0.2 * upm)  # jamb at x = 2.0 .. 2.9
    voxels = _wall_voxels(rng, 2.0, 2.5, 0.0, upm)
    _depth_extend_faces([face], voxels, [door], upm, SETTINGS, WallEvidence(storey_height=2.5 * upm))
    assert face.end[0] / upm == pytest.approx(2.0, abs=1e-6)


@UNITS
def test_extension_stops_at_walkable_floor_but_crosses_unseen_wall(upm) -> None:
    from samcloud.ortho.vector.raster_grid import RasterGrid

    rng = _rng(25)
    grid = RasterGrid(0.0, 1.0 * upm, 0.05 * upm, int(4.0 / 0.05), int(2.0 / 0.05))
    free = np.zeros((grid.height, grid.width), dtype=bool)
    free[10:31, int(2.25 / 0.05):int(2.6 / 0.05)] = True  # walkable floor in a 0.35 m gap starting 0.25 m behind the face end
    voxels = _wall_voxels(rng, 2.0, 2.7, 0.0, upm)  # wall-class voxels on the line also across the gap (labelled through the door)
    face = _face(upm, (0.0, 0.0), (2.0, 0.0))
    _depth_extend_faces([face], voxels, [], upm, SETTINGS, WallEvidence(storey_height=2.5 * upm), free, grid)
    assert face.end[0] / upm == pytest.approx(2.23, abs=0.07)  # stops before the floor cell, does not enter the gap
    open_face = _face(upm, (0.0, 0.0), (2.0, 0.0))
    _depth_extend_faces([open_face], voxels[voxels[:, 0] < 2.4 * upm], [], upm, SETTINGS, WallEvidence(storey_height=2.5 * upm), np.zeros_like(free), grid)
    assert open_face.end[0] / upm == pytest.approx(2.4, abs=0.08)  # no floor: an unseen stretch of wall is closed


@UNITS
def test_lamp_height_voxels_do_not_make_a_surface(upm) -> None:
    rng = _rng(24)
    face = _face(upm, (0.0, 0.0), (2.0, 0.0))
    voxels = _wall_voxels(rng, 2.0, 2.4, 0.0, upm, height=(1.00, 1.15))  # a single height band only
    _depth_extend_faces([face], voxels, [], upm, SETTINGS, WallEvidence(storey_height=2.5 * upm))
    assert face.end[0] / upm == pytest.approx(2.0, abs=1e-6)


def _run(face, upm, thickness=0.2, evidence="learned"):
    return WallRun(face.index, 0.0, face.length, thickness * upm, "interior", evidence)


@UNITS
def test_depth_measures_the_thickness_where_the_cloud_saw_one_side(upm) -> None:
    rng = _rng(31)
    face = _face(upm, (0.0, 0.0), (3.0, 0.0))  # solid normal = left (+y)
    far_face = _wall_voxels(rng, 0.0, 3.0, 0.19, upm)  # the unseen second face, 19 cm behind
    run = _run(face, upm, evidence="learned")
    assert _depth_thickness([run], [face], far_face, upm, SETTINGS, WallEvidence(storey_height=2.5 * upm)) == 1
    assert run.evidence == "depth"
    assert run.thickness / upm == pytest.approx(0.19, abs=0.012)


@UNITS
def test_measured_thickness_and_short_or_noisy_cover_are_kept(upm) -> None:
    rng = _rng(32)
    face = _face(upm, (0.0, 0.0), (3.0, 0.0))
    far_face = _wall_voxels(rng, 0.0, 3.0, 0.19, upm)
    run = _run(face, upm, evidence="face")
    _depth_thickness([run], [face], far_face, upm, SETTINGS, WallEvidence(storey_height=2.5 * upm))
    assert run.evidence == "face" and run.thickness / upm == pytest.approx(0.2)
    exterior = _run(face, upm, thickness=0.288, evidence="exterior")  # an exterior default stays: its outer face is mostly not seen
    _depth_thickness([exterior], [face], far_face, upm, SETTINGS, WallEvidence(storey_height=2.5 * upm))
    assert exterior.evidence == "exterior" and exterior.thickness / upm == pytest.approx(0.288)
    short = _run(face, upm)
    only_short = _wall_voxels(rng, 0.0, 0.2, 0.19, upm)
    _depth_thickness([short], [face], only_short, upm, SETTINGS, WallEvidence(storey_height=2.5 * upm))
    assert short.evidence == "learned"
    scattered = np.column_stack((rng.uniform(0, 3, 400) * upm, rng.uniform(0.06, 0.45, 400) * upm, rng.uniform(0.35, 2.1, 400) * upm))
    other = _run(face, upm)
    _depth_thickness([other], [face], scattered, upm, SETTINGS, WallEvidence(storey_height=2.5 * upm))
    assert other.evidence == "learned"


@UNITS
def test_step_is_not_invented_the_two_faces_of_one_wall_stay_one_wall(upm) -> None:
    # the opposite faces of a 19 cm wall are NOT a step: only a face on the SAME side offset >= 6 cm would be one
    rng = _rng(33)
    face = _face(upm, (0.0, 0.0), (3.0, 0.0))
    both_faces = np.vstack((_wall_voxels(rng, 0.0, 3.0, 0.19, upm), _wall_voxels(rng, 0.0, 3.0, -0.0, upm)))
    run = _run(face, upm)
    _depth_thickness([run], [face], both_faces, upm, SETTINGS, WallEvidence(storey_height=2.5 * upm))
    assert run.thickness / upm == pytest.approx(0.19, abs=0.012)  # the far face of the same wall, the near face (offset 0) is the face itself


# --------------------------------------------------------------------------
# upper cabinets from points
# --------------------------------------------------------------------------

def _box_points(rng, x0, x1, y0, y1, z0, z1, count):
    return np.column_stack((rng.uniform(x0, x1, count), rng.uniform(y0, y1, count), rng.uniform(z0, z1, count)))


def _kitchen_points(rng, upm):
    base = _box_points(rng, 0.0, 3.0, 0.0, 0.6, 0.0, 0.9, 6000)
    upper = _box_points(rng, 0.2, 1.4, 0.02, 0.35, 1.4, 2.2, 3000)  # 1.2 m wide, 0.33 deep, z 1.4..2.2
    return np.vstack((base, upper)) * upm


@UNITS
def test_upper_cabinet_without_a_base_run_is_found_from_points(upm) -> None:
    from samcloud.ortho.vector.upper_points import UpperPointSettings, detect_upper_cabinets_from_points

    rng = _rng(41)
    # no base cabinet below: only the hanging unit, 0.35 deep, on the wall y = 0
    upper = _box_points(rng, 0.2, 1.4, 0.03, 0.36, 1.38, 2.18, 3000) * upm
    wall = LineString([(0.0, 0.0), (4.0 * upm, 0.0)])
    found = detect_upper_cabinets_from_points(upper, [wall], box(-1, -1, 5 * upm, 2 * upm), [], upm, UpperPointSettings())
    assert len(found) == 1
    polygon, (z0, z1) = found[0]
    minx, miny, maxx, maxy = polygon.bounds
    assert (maxx - minx) / upm == pytest.approx(1.2, abs=0.12)
    assert miny / upm == pytest.approx(0.0, abs=0.02)  # reaches back to the wall
    assert (maxy - miny) / upm == pytest.approx(0.36, abs=0.06)
    assert z0 == pytest.approx(1.40, abs=0.08) and z1 == pytest.approx(2.15, abs=0.08)


@UNITS
def test_upper_cabinet_above_a_base_run_and_a_tall_unit_is_not_a_hanging_unit(upm) -> None:
    from samcloud.ortho.vector.upper_points import detect_upper_cabinets_from_points

    rng = _rng(42)
    points = _kitchen_points(rng, upm)
    tall = _box_points(rng, 2.0, 2.6, 0.0, 0.6, 0.0, 2.2, 5000) * upm  # floor-to-top unit: evidence in the gap band as well
    wall = LineString([(0.0, 0.0), (4.0 * upm, 0.0)])
    found = detect_upper_cabinets_from_points(np.vstack((points, tall)), [wall], None, [], upm)
    assert len(found) == 1
    minx, _miny, maxx, _maxy = found[0][0].bounds
    assert maxx / upm < 1.7 and minx / upm > 0.0  # only the hanging unit at x 0.2..1.4, not the tall unit at 2.0..2.6


@UNITS
def test_duplicate_zone_wall_distance_and_height_rules(upm) -> None:
    from samcloud.ortho.vector.upper_points import detect_upper_cabinets_from_points

    rng = _rng(43)
    points = _kitchen_points(rng, upm)
    wall = LineString([(0.0, 0.0), (4.0 * upm, 0.0)])
    existing = box(0.2 * upm, 0.0, 1.4 * upm, 0.35 * upm)
    assert detect_upper_cabinets_from_points(points, [wall], None, [existing], upm) == []  # the run-based detection already has it
    assert detect_upper_cabinets_from_points(points, [wall], box(5 * upm, 5 * upm, 6 * upm, 6 * upm), [], upm) == []  # outside the kitchen zone
    far_wall = LineString([(0.0, 3.0 * upm), (4.0 * upm, 3.0 * upm)])
    assert detect_upper_cabinets_from_points(points, [far_wall], None, [], upm) == []  # no wall within reach
    shelf = _box_points(rng, 0.2, 1.4, 0.02, 0.35, 1.9, 2.05, 3000) * upm  # a thin shelf edge: z span < 0.40
    assert detect_upper_cabinets_from_points(shelf, [wall], None, [], upm) == []


@UNITS
def test_measured_z_band_is_robust_against_stray_points(upm) -> None:
    from samcloud.ortho.vector.upper_points import measured_z_band

    rng = _rng(44)
    points = np.vstack((_box_points(rng, 0.2, 1.4, 0.02, 0.35, 1.35, 2.17, 2000) * upm, np.array([[0.5 * upm, 0.2 * upm, 2.9 * upm]])))
    band = measured_z_band(points, box(0.2 * upm, 0.0, 1.4 * upm, 0.36 * upm), upm)
    assert band is not None
    assert band[0] == pytest.approx(1.40, abs=0.06) and band[1] == pytest.approx(2.13, abs=0.06)  # not the loose 1.14..2.31 of an instance bbox
    assert measured_z_band(points[:5], box(0, 0, 1, 1), upm) is None


def test_upper_cabinet_z_from_measured_band_in_the_plan_model() -> None:
    from samcloud.bim.plan_model import _furniture, load_bim_export_config

    layers = {"A-OBEN": [{"type": "polyline", "closed": True, "points": [[0, 0], [1.2, 0], [1.2, 0.35], [0, 0.35]], "_upper_cabinet": True, "_z_range_m": [1.38, 2.16], "_z_source": "points"}]}
    items = _furniture(layers, {}, load_bim_export_config(), [], 0.1, 1.0)
    assert len(items) == 1
    assert items[0].z_source == "points"
    assert items[0].z_min == pytest.approx(0.1 + 1.38) and items[0].z_max == pytest.approx(0.1 + 2.16)


# --------------------------------------------------------------------------
# tight footprints
# --------------------------------------------------------------------------

@UNITS
def test_tight_rectangle_ignores_stray_points_and_follows_the_wall_axis(upm) -> None:
    from samcloud.bim.furniture_footprint import tight_rectangle

    rng = _rng(51)
    angle = np.radians(20.0)
    a, b = rng.uniform(0.0, 0.7, 3000), rng.uniform(0.0, 0.6, 3000)
    points = np.column_stack((a * np.cos(angle) - b * np.sin(angle), a * np.sin(angle) + b * np.cos(angle))) * upm
    strays = np.array([[3.0, 3.0], [-2.0, 1.0], [0.4, 4.0]]) * upm
    rectangle = tight_rectangle(np.vstack((points, strays)), angle, upm)
    assert rectangle is not None
    assert rectangle.area / upm ** 2 == pytest.approx(0.7 * 0.6, rel=0.12)


@UNITS
def test_thin_slab_is_completed_towards_the_wall_behind_it(upm) -> None:
    from samcloud.bim.furniture_footprint import TightFootprintSettings, tight_footprint

    rng = _rng(52)
    # a fridge seen from the front only: x 0..0.64, y 0.55..0.68 (13 cm thick); the wall face is at y = 1.20 behind it
    slab = _box_points(rng, 0.0, 0.64, 0.55, 0.68, 0.2, 2.2, 4000) * upm
    wall = LineString([(-1.0 * upm, 1.20 * upm), (3.0 * upm, 1.20 * upm)])
    wall_union = box(-1.0 * upm, 1.20 * upm, 3.0 * upm, 1.45 * upm)
    footprint = tight_footprint(slab[:, :2], [wall], wall_union, upm, TightFootprintSettings())
    assert footprint is not None
    minx, miny, maxx, maxy = footprint.bounds
    assert (maxx - minx) / upm == pytest.approx(0.64, abs=0.08)
    assert miny / upm == pytest.approx(0.55, abs=0.06)
    assert maxy / upm == pytest.approx(1.20, abs=0.06)  # depth now 0.65, from the points to the wall


@UNITS
def test_slab_without_a_wall_in_reach_stays_thin_and_stray_depth_is_capped(upm) -> None:
    from samcloud.bim.furniture_footprint import TightFootprintSettings, tight_footprint

    rng = _rng(53)
    slab = _box_points(rng, 0.0, 0.64, 0.55, 0.68, 0.2, 2.2, 4000) * upm
    far_wall = LineString([(-1.0 * upm, 3.0 * upm), (3.0 * upm, 3.0 * upm)])
    footprint = tight_footprint(slab[:, :2], [far_wall], box(-1 * upm, 3 * upm, 3 * upm, 3.3 * upm), upm, TightFootprintSettings())
    assert footprint is not None and (footprint.bounds[3] - footprint.bounds[1]) / upm < 0.2


@UNITS
def test_instance_point_sets_add_only_nearby_depth_voxels(upm) -> None:
    from samcloud.bim.furniture_footprint import instance_point_sets

    rng = _rng(54)
    cloud = _box_points(rng, 0.0, 0.6, 0.0, 0.6, 0.9, 2.1, 6000) * upm
    ids = np.full(len(cloud), 7, np.uint32)
    near = np.array([[0.62, 0.3, 1.5], [0.30, 0.64, 1.2]]) * upm  # within 8 cm of the instance
    far = np.array([[1.5, 1.5, 1.5], [0.3, 0.3, 3.5]]) * upm
    sets = instance_point_sets(cloud, ids, np.vstack((near, far)), upm)
    assert set(sets) == {7}
    assert len(sets[7]) == len(cloud) + 2
    assert instance_point_sets(cloud[:20], ids[:20], None, upm) == {}  # fewer than min_points


# --------------------------------------------------------------------------
# furniture rules of the plan model
# --------------------------------------------------------------------------

class _FakeFootprints:
    def __init__(self, footprint, settings=None):
        from samcloud.bim.furniture_footprint import TightFootprintSettings

        self._footprint = footprint
        self.settings = settings or TightFootprintSettings()

    def has(self, instance_id):
        return True

    def footprint(self, instance_id):
        return self._footprint


def _plain_item(footprint, instance_id=33, z=(0.0, 2.3), layer="A-FURN", label="Kühlschrank"):
    from samcloud.bim.plan_model import PlanFurniture

    return PlanFurniture(
        id="F001", instance_id=instance_id, label=label, semantic_label="refrigerator", catalogue_kind=None, layer=layer,
        footprint=[[float(x), float(y)] for x, y in footprint.exterior.coords][:-1], z_min=z[0], z_max=z[1], z_source="instance_bbox",
    )


@UNITS
def test_thin_standing_slab_takes_the_measured_depth(upm) -> None:
    from samcloud.bim.plan_model import _complete_thin_slabs

    thin = box(0.0, 0.55 * upm, 0.64 * upm, 0.68 * upm)
    tight = box(-0.02 * upm, 0.55 * upm, 0.66 * upm, 1.20 * upm)
    item = _plain_item(thin, z=(0.0, 2.3 * upm))
    report: dict = {}
    _complete_thin_slabs([item], _FakeFootprints(tight), upm, 0.0, report)
    assert Polygon(item.footprint).area / upm ** 2 == pytest.approx(0.68 * 0.65, rel=0.05)
    assert report["thin_slabs_completed"][0]["depth_m"] == pytest.approx(0.65, abs=0.05)


@UNITS
def test_complete_thin_slabs_leaves_deep_low_or_partless_items_alone(upm) -> None:
    from samcloud.bim.plan_model import _complete_thin_slabs

    deep = box(0.0, 0.0, 0.64 * upm, 0.60 * upm)
    tight = box(0.0, 0.0, 0.64 * upm, 0.70 * upm)
    item = _plain_item(deep, z=(0.0, 2.3 * upm))
    _complete_thin_slabs([item], _FakeFootprints(tight), upm, 0.0, None)
    assert Polygon(item.footprint).area == pytest.approx(deep.area)  # not thin
    low = _plain_item(box(0.0, 0.55 * upm, 0.64 * upm, 0.68 * upm), z=(0.0, 0.5 * upm))  # a low slab (a rug) is not completed
    _complete_thin_slabs([low], _FakeFootprints(tight), upm, 0.0, None)
    assert Polygon(low.footprint).area == pytest.approx(0.64 * 0.13 * upm ** 2, rel=0.01)


class _FakeDepth:
    """Stub with the PlanDepth surface the furniture rules use."""

    def __init__(self, shares, floor_points):
        self.settings = {"plan_floor_contact_z_m": 0.20, "plan_empty_share": 0.80}
        self._shares = shares
        self._floor = floor_points

    def furniture_points(self, colmap_only=False, **_):
        return self._floor, np.full(len(self._floor), 17)

    def free_share(self, polygon, z_low, z_high):
        for key, value in self._shares.items():
            if polygon.centroid.x < key:
                return value
        return 0.0

    @staticmethod
    def points_inside(polygon, points, z_low=None, z_high=None):
        return PlanDepth.points_inside(polygon, points, z_low, z_high)


@UNITS
def test_box_in_seen_empty_space_is_rejected_and_floor_contact_is_confirmed(upm) -> None:
    from samcloud.bim.plan_model import _apply_depth_rules

    rng = _rng(61)
    solid = _plain_item(box(0.0, 0.0, 0.8 * upm, 0.6 * upm), z=(0.0, 0.8 * upm), label="Regal")
    ghost = _plain_item(box(3.0 * upm, 0.0, 3.8 * upm, 0.6 * upm), z=(0.0, 0.8 * upm), label="Phantom")
    ghost.id = "F002"
    floor = np.column_stack((rng.uniform(0.0, 0.8, 200) * upm, rng.uniform(0.0, 0.6, 200) * upm, rng.uniform(0.0, 0.15, 200) * upm))
    depth = _FakeDepth({1.0: 0.05, 9.0: 0.93}, floor)
    report: dict = {}
    kept = _apply_depth_rules([solid, ghost], depth, upm, 0.0, report)
    assert [item.id for item in kept] == ["F001"]
    assert kept[0].floor_contact is True
    assert report["rejected_in_free_space"][0]["id"] == "F002"


@UNITS
def test_upper_cabinet_never_gets_floor_contact_from_the_base_run_below(upm) -> None:
    from samcloud.bim.plan_model import _apply_depth_rules

    rng = _rng(62)
    upper = _plain_item(box(0.0, 0.0, 1.2 * upm, 0.35 * upm), z=(1.4 * upm, 2.2 * upm), layer="A-OBEN", label="upper_cabinet")
    floor = np.column_stack((rng.uniform(0.0, 1.2, 300) * upm, rng.uniform(0.0, 0.35, 300) * upm, rng.uniform(0.0, 0.15, 300) * upm))  # the base run below
    kept = _apply_depth_rules([upper], _FakeDepth({9.0: 0.0}, floor), upm, 0.0, None)
    assert kept[0].floor_contact is None


def test_floor_contact_keeps_a_standing_shelf_on_the_floor_in_the_scene() -> None:
    from test_bim_scene_builder import make_plan
    from samcloud.bim.plan_model import PlanFurniture
    from samcloud.bim.scene_builder import build_scene

    plan = make_plan()
    wall = plan.walls[0]
    (x0, y0), (x1, y1) = wall.centreline[0], wall.centreline[1]
    midpoint = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    ring = [[midpoint[0] - 0.4, midpoint[1] + 0.12], [midpoint[0] + 0.4, midpoint[1] + 0.12], [midpoint[0] + 0.4, midpoint[1] + 0.42], [midpoint[0] - 0.4, midpoint[1] + 0.42]]
    zs = (plan.floor_z + 0.9, plan.floor_z + 1.9)

    def shelf(contact):
        return PlanFurniture(id="F900", instance_id=None, label="Regal", semantic_label="shelf", catalogue_kind=None, layer="A-FURN", footprint=ring,
                             z_min=zs[0], z_max=zs[1], z_source="instance_bbox", floor_contact=contact)

    def z_bottom(contact):
        plan.furniture = [shelf(contact)]
        scene = build_scene(plan)
        element = next(item for item in scene.elements.values() if item.name.startswith("Regal"))
        return element.params.model_extra["z_bottom"]

    standing = z_bottom(True)  # floor-level depth voxels: the shelf stands on the floor
    assert z_bottom(None) > standing + 0.5  # without the evidence a shelf next to a wall keeps its measured, suspended z origin
    assert z_bottom(False) > standing + 0.5


@UNITS
def test_upper_cabinet_without_a_wall_in_the_final_plan_is_dropped(upm) -> None:
    from samcloud.bim.plan_model import PlanWall, _drop_unmounted_uppers

    wall = PlanWall(id="W001", centreline=[[0.0, 0.0], [4.0 * upm, 0.0]], thickness_m=0.2, kind="interior", evidence="face",
                    face_a=[[0.0, 0.1 * upm], [4.0 * upm, 0.1 * upm]], face_b=[[0.0, -0.1 * upm], [4.0 * upm, -0.1 * upm]])
    mounted = _plain_item(box(0.2 * upm, 0.1 * upm, 1.4 * upm, 0.45 * upm), instance_id=None, layer="A-OBEN", z=(1.4 * upm, 2.2 * upm), label="upper_cabinet")
    floating = _plain_item(box(0.2 * upm, 0.9 * upm, 1.4 * upm, 1.25 * upm), instance_id=None, layer="A-OBEN", z=(1.4 * upm, 2.2 * upm), label="upper_cabinet")
    floating.id = "F002"
    base = _plain_item(box(0.2 * upm, 2.0 * upm, 1.4 * upm, 2.6 * upm), instance_id=None, layer="A-FURN", label="Regal")  # not an upper cabinet: untouched
    base.id = "F003"
    report: dict = {}
    kept = _drop_unmounted_uppers([mounted, floating, base], [wall], upm, 0.30, report)
    assert [item.id for item in kept] == ["F001", "F003"]
    assert report["uppers_dropped_no_wall"] == [{"id": "F002", "label": "upper_cabinet"}]
    assert _drop_unmounted_uppers([floating], [], upm, 0.30, None) == [floating]  # no wall at all: nothing to judge against


@UNITS
def test_point_upper_cabinet_always_reaches_back_to_the_wall_and_far_evidence_is_no_wall_unit(upm) -> None:
    from samcloud.ortho.vector.upper_points import detect_upper_cabinets_from_points

    rng = _rng(71)
    wall = LineString([(0.0, 0.0), (4.0 * upm, 0.0)])
    stand_off = _box_points(rng, 0.2, 1.4, 0.40, 0.70, 1.4, 2.2, 3000) * upm  # evidence 0.40 m in front of the wall: a shelf on a stand
    assert detect_upper_cabinets_from_points(stand_off, [wall], None, [], upm) == []
    front_only = _box_points(rng, 0.2, 1.4, 0.18, 0.36, 1.4, 2.2, 3000) * upm  # only the front seen, 0.18 m off the wall
    found = detect_upper_cabinets_from_points(front_only, [wall], None, [], upm)
    assert len(found) == 1 and found[0][0].bounds[1] / upm == pytest.approx(0.0, abs=1e-6)
