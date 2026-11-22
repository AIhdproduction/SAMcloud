"""Measured kitchen fronts: diagonal corner units at any angle, recessed bases, upper-cabinet ends, kept rotations.

Synthetic scan points only (seeded noise, samples jittered off the cell borders); every case runs at two scales.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import ezdxf
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from samcloud.bim.furniture_recon.pipeline import _kitchen_solids, _level_abutting_tops
from samcloud.bim.furniture_recon.settings import ReconSettings
from samcloud.bim.plan_model import _furniture, load_bim_export_config
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
from samcloud.dxf.writer import export_plan
from samcloud.ortho.vector.kitchen_fronts import KitchenFrontSettings, load_kitchen_points
from samcloud.ortho.vector.kitchen_layout import _block_footprint, counter_lines, kitchen_cad_layout, kitchen_scan_bounds
from samcloud.ortho.vector.measured_rotation import clear_point_rotation, clear_rotation
from samcloud.ortho.vector.plan_polish import PlanPolishSettings, regularize_furniture

SCALES = ((1.0, 0.0), (0.8124, 0.0), (0.9891, 0.213))  # units per metre, wall axis (radians)
WALLS = box(-0.25, -0.25, 5.25, 4.25).difference(box(0.0, 0.0, 5.0, 4.0))
ROOM = box(0.0, 0.0, 5.0, 4.0)
# an L in the corner x=0 / y=4: the north leg reaches y 3.413, the west leg x 0.587
NORTH = box(0.0, 3.413, 3.113, 4.0)
WEST = box(0.0, 0.917, 0.587, 4.0)


def _plan(geometry, upm: float, theta: float):
    return affinity.rotate(affinity.scale(geometry, upm, upm, origin=(0.0, 0.0)), theta, origin=(0.0, 0.0), use_radians=True)


def _xy(points: np.ndarray, upm: float, theta: float) -> np.ndarray:
    cos, sin = math.cos(theta), math.sin(theta)
    values = np.asarray(points, dtype=float).reshape(-1, 2) * upm
    return values @ np.asarray(((cos, sin), (-sin, cos)))


def _metres(points, upm: float, theta: float) -> np.ndarray:
    cos, sin = math.cos(theta), math.sin(theta)
    return np.asarray(points, dtype=float).reshape(-1, 2) @ np.asarray(((cos, -sin), (sin, cos))) / upm


def _ring(polygon: Polygon) -> list[list[float]]:
    return [list(point) for point in list(polygon.exterior.coords)[:-1]]


def _face(rng, first, second, low: float = 0.09, high: float = 0.865, noise: float = 0.008, density: float = 2600.0) -> np.ndarray:
    """Scanned vertical face between two plan points (metres): (x, y, z)."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    vector = second - first
    length = float(np.hypot(*vector))
    count = int(density * length * (high - low))
    normal = np.asarray((-vector[1], vector[0])) / length
    xy = first + rng.random(count)[:, None] * vector + rng.normal(0.0, noise, count)[:, None] * normal
    return np.column_stack((xy, rng.uniform(low, high, count)))


def _top(rng, area: Polygon, height: float = 0.895, density: float = 1500.0) -> np.ndarray:
    x0, y0, x1, y1 = area.bounds
    count = int(density * area.area)
    return np.column_stack((rng.uniform(x0, x1, count), rng.uniform(y0, y1, count), rng.normal(height, 0.004, count)))


def _diagonal_scene(rng, angle: float, north: float = 3.413, west: float = 0.587, cut: float = 0.463):
    """Cabinet fronts with a diagonal across the inner corner; ``angle`` against the north front."""
    drop = cut * math.tan(math.radians(angle))
    first, second = (west + cut, north), (west, north - drop)
    points = np.vstack((
        _face(rng, first, (3.113, north)), _face(rng, (west, 0.917), second), _face(rng, second, first),
        _top(rng, box(0.7, 3.5, 3.0, 3.97)),
    ))
    return points, np.asarray(first), np.asarray(second)


def _layout(points, upm: float, theta: float, legs=(NORTH, WEST), chamfer: Polygon | None = None, appliances=None, uppers=(), settings=None):
    shapes = [_plan(leg, upm, theta) for leg in legs]
    parts = shapes + ([_plan(chamfer, upm, theta)] if chamfer is not None else [])
    run = unary_union(parts).buffer(1e-7).buffer(-1e-7)
    entity = {"type": "polyline", "points": _ring(run), "closed": True, "holes": [], "_einb_kind": "fine_kitchen", "_run_legs": [_ring(shape) for shape in shapes]}
    if chamfer is not None:
        entity["_chamfers"] = [_ring(_plan(chamfer, upm, theta))]
    layers = {
        "A-EINB": [entity], "A-FURN": [], "A-ANNO": [], "A-FURN-DETL": [],
        "A-OBEN": [{"type": "polyline", "points": _ring(_plan(shape, upm, theta)), "closed": True, "_upper_cabinet": True, "_z_range_m": [1.40, 2.18]} for shape in uppers],
    }
    walls, room = _plan(WALLS, upm, theta), _plan(ROOM, upm, theta)
    scan = None if points is None else np.column_stack((_xy(points[:, :2], upm, theta), points[:, 2] * upm))
    labels = {kind: _xy(values, upm, theta) for kind, values in (appliances or {}).items()}
    stats = kitchen_cad_layout(
        layers, [room], walls, theta, upm, None, labels, scan_points=scan, front_settings=settings or KitchenFrontSettings(),
    )
    return layers, walls, stats, next(item for item in layers["A-EINB"] if item.get("_run_solids"))


def _solids(run: dict, kind: str) -> list[Polygon]:
    return [Polygon(ring) for leg in run["_run_solids"] for ring in leg[kind]]


def _assert_abutting_and_identical(run: dict, upm: float) -> None:
    """Base and worktop cover the 2D counter exactly once."""
    counter = Polygon(run["points"])
    for kind in ("base", "worktop"):
        pieces = _solids(run, kind)
        for index, piece in enumerate(pieces):
            assert piece.is_valid and all(piece.intersection(other).area < 1e-5 * upm * upm for other in pieces[index + 1:]), kind
        if kind == "worktop":
            assert unary_union(pieces).symmetric_difference(counter).area < 0.004 * upm * upm


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize("angle", [45.0, 30.0, 58.0])
def test_diagonal_corner_front_is_measured_at_any_angle(upm: float, theta: float, angle: float) -> None:
    rng = np.random.default_rng(20261122 + int(angle))
    points, first, second = _diagonal_scene(rng, angle)
    layers, walls, stats, run = _layout(points, upm, theta)
    assert len(stats["kitchen_corner_fronts"]) == 1
    assert stats["kitchen_corner_fronts"][0]["angle_deg"] == pytest.approx(angle, abs=1.5)
    ring = _metres(run["points"], upm, theta)
    for expected in (first, second):
        assert np.min(np.hypot(*(ring - expected).T)) <= 0.02, expected
    _assert_abutting_and_identical(run, upm)
    # the counter lines of the 2D DXF carry the same diagonal
    assert counter_lines(layers, walls, upm) == 1
    drawn = unary_union([LineString(_metres(item["points"], upm, theta)) for item in layers["A-EINB"] if item.get("_counter_front")])
    assert drawn.buffer(0.02).covers(LineString([first, second]))


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_fronts_beside_a_measured_diagonal_take_their_measured_offset(upm: float, theta: float) -> None:
    rng = np.random.default_rng(7)
    points, first, second = _diagonal_scene(rng, 45.0, north=3.4427, west=0.6221)
    _, _, stats, run = _layout(points, upm, theta)
    assert len(stats["kitchen_corner_fronts"]) == 1
    north, west = (_metres(ring, upm, theta) for ring in run["_run_legs"])
    assert north[:, 1].min() == pytest.approx(3.4427, abs=0.008) and north[:, 1].max() == pytest.approx(4.0, abs=1e-6)
    assert west[:, 0].max() == pytest.approx(0.6221, abs=0.008) and west[:, 0].min() == pytest.approx(0.0, abs=1e-6)
    ring = _metres(run["points"], upm, theta)
    for expected in (first, second):
        assert np.min(np.hypot(*(ring - expected).T)) <= 0.02, expected
    _assert_abutting_and_identical(run, upm)


def _square_scene(rng, hidden: float = 0.0) -> np.ndarray:
    return np.vstack((_face(rng, (0.587 + hidden, 3.413), (3.113, 3.413)), _face(rng, (0.587, 0.917), (0.587, 3.413 - hidden))))


def _noisy_corner(rng) -> np.ndarray:
    return np.vstack((_square_scene(rng, 0.45), _face(rng, (0.587, 2.963), (1.037, 3.413), noise=0.07)))


def _low_object(rng) -> np.ndarray:
    """A box standing diagonally in the corner, ending below the worktop."""
    return np.vstack((_square_scene(rng, 0.45), _face(rng, (0.587, 2.963), (1.037, 3.413), high=0.48)))


def _round_bin(rng) -> np.ndarray:
    turn = rng.uniform(-0.5 * math.pi, 0.0, 1300)
    radius = rng.normal(0.42, 0.006, 1300)
    arc = np.column_stack((0.587 + radius * np.cos(turn), 3.413 + radius * np.sin(turn), rng.uniform(0.09, 0.865, 1300)))
    return np.vstack((_square_scene(rng, 0.43), arc))


def _object_before_square_corner(rng) -> np.ndarray:
    """Both straight fronts are seen into the corner behind a slanted board leaning there."""
    return np.vstack((_square_scene(rng), _face(rng, (0.587, 2.963), (1.037, 3.413))))


def _slanted_object_without_fronts(rng) -> np.ndarray:
    return _face(rng, (0.587, 2.963), (1.037, 3.413))


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize("scene", [
    _square_scene, lambda rng: _square_scene(rng, 0.45), _noisy_corner, _low_object, _round_bin, _object_before_square_corner,
    _slanted_object_without_fronts,
], ids=["square", "occluded", "noisy", "low_object", "round_bin", "object_before", "no_fronts"])
def test_corner_without_clear_evidence_stays_square(upm: float, theta: float, scene) -> None:
    _, _, stats, run = _layout(scene(np.random.default_rng(31)), upm, theta)
    assert stats["kitchen_corner_fronts"] == []
    assert Polygon(run["points"]).area / upm**2 == pytest.approx(NORTH.union(WEST).area, abs=0.003)
    assert len(_metres(run["points"], upm, theta)) == 6
    _assert_abutting_and_identical(run, upm)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_raster_chamfer_is_dropped_where_the_scan_shows_a_square_corner(upm: float, theta: float) -> None:
    chamfer = Polygon([(0.587, 3.413), (0.587, 2.983), (1.017, 3.413)])
    _, _, _, run = _layout(_square_scene(np.random.default_rng(3)), upm, theta, chamfer=chamfer)
    assert run["_chamfers"] == []
    assert Polygon(run["points"]).area / upm**2 == pytest.approx(NORTH.union(WEST).area, abs=0.003)
    # without scan points nothing can judge it: it stays, in 2D and in the solids alike
    _, _, _, kept = _layout(None, upm, theta, chamfer=chamfer)
    assert len(kept["_chamfers"]) == 1
    assert Polygon(kept["points"]).area / upm**2 == pytest.approx(NORTH.union(WEST).area + chamfer.area, abs=0.003)
    _assert_abutting_and_identical(kept, upm)


def test_measurement_can_be_switched_off() -> None:
    points, _, _ = _diagonal_scene(np.random.default_rng(5), 45.0)
    _, _, stats, run = _layout(points, 1.0, 0.0, settings=KitchenFrontSettings(enabled=False))
    assert stats["kitchen_corner_fronts"] == [] and len(run["points"]) == 6


def _rim(rng, centre, width: float, depth: float, angle: float, count: int = 140) -> np.ndarray:
    """Label points on the rim of a sink."""
    rectangle = affinity.rotate(box(-width / 2, -depth / 2, width / 2, depth / 2), angle, origin=(0.0, 0.0))
    rectangle = affinity.translate(rectangle, *centre)
    points = np.asarray([rectangle.exterior.interpolate(value, normalized=True).coords[0] for value in (np.arange(count) + 0.37) / count])
    return points + rng.normal(0.0, 0.003, points.shape)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize("angle", [45.0, 35.0])
def test_sink_in_the_diagonal_unit_is_parallel_to_it_in_2d_and_3d(upm: float, theta: float, angle: float) -> None:
    rng = np.random.default_rng(11)
    points, first, second = _diagonal_scene(rng, angle, cut=0.50)
    direction = (first - second) / np.hypot(*(first - second))
    inward = np.asarray((-direction[1], direction[0]))
    centre = (first + second) / 2.0 + 0.255 * inward
    layers, _, stats, run = _layout(points, upm, theta, appliances={"KITCHEN_SINK": _rim(rng, centre, 0.497, 0.405, angle)})
    assert stats["kitchen_appliances_mounted"] == ["KITCHEN_SINK"]
    sink = next(item for item in layers["A-EINB"] if item.get("type") == "insert")
    assert sink["_catalogue"]["kind"] == "KITCHEN_SINK" and sink["name"] == "KITCHEN_SINK_500X400"
    assert sink["_catalogue"]["measured_rotation"] and sink["_catalogue"]["corner"]
    difference = (sink["rotation"] - math.degrees(theta) - angle) % 180.0
    assert min(difference, 180.0 - difference) <= 2.0
    footprint = _block_footprint(sink, upm)
    assert Polygon(run["points"]).buffer(1e-3 * upm).covers(footprint)
    assert math.dist(_metres(footprint.centroid.coords[0], upm, theta)[0], centre) <= 0.025

    items = _furniture(layers, {}, load_bim_export_config(), [], 0.0, upm)
    bases = [Polygon(item.footprint) for item in items if item.catalogue_kind == "kitchen_base"]
    tops = [Polygon(item.footprint) for item in items if item.catalogue_kind == "worktop"]
    assert len(bases) == len(tops) == 2
    assert bases[0].intersection(bases[1]).area < 1e-5 and tops[0].intersection(tops[1]).area < 1e-5
    assert unary_union(tops).symmetric_difference(Polygon(run["points"])).area < 0.004 * upm * upm
    model_sink = Polygon(next(item.footprint for item in items if item.catalogue_kind == "KITCHEN_SINK"))
    assert model_sink.symmetric_difference(footprint).area < 1e-4, "the measured rotation is not squared"
    assert unary_union(tops).buffer(2e-3 * upm).covers(model_sink)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize("clear", [True, False])
def test_insert_on_a_straight_leg_keeps_only_a_clearly_measured_rotation(upm: float, theta: float, clear: bool) -> None:
    rng = np.random.default_rng(13)
    if clear:
        label = _rim(rng, (1.913, 3.711), 0.60, 0.40, 9.0)
    else:
        label = np.column_stack((rng.normal(1.913, 0.07, 160), rng.normal(3.711, 0.05, 160)))
    layers, _, _, run = _layout(_square_scene(rng), upm, theta, appliances={"KITCHEN_SINK": label})
    sink = next(item for item in layers["A-EINB"] if item.get("type") == "insert")
    turned = (sink["rotation"] - math.degrees(theta) + 45.0) % 90.0 - 45.0
    assert abs(turned - (9.0 if clear else 0.0)) <= 1.5
    if clear:
        assert sink["_catalogue"]["measured_rotation"]
        front = np.asarray((-math.sin(math.radians(sink["rotation"])), math.cos(math.radians(sink["rotation"]))))
        assert float(front @ (math.sin(theta), -math.cos(theta))) > 0.9, "+Y looks into the room"
    assert Polygon(run["points"]).buffer(1e-3 * upm).covers(_block_footprint(sink, upm))


def test_insert_footprint_of_a_basin_survives_the_dxf(tmp_path_factory: pytest.TempPathFactory) -> None:
    rng = np.random.default_rng(17)
    points, first, second = _diagonal_scene(rng, 45.0)
    centre = (first + second) / 2.0 + 0.26 * np.asarray((-math.sqrt(0.5), math.sqrt(0.5)))
    layers, walls, _, _ = _layout(points, 1.0, 0.0, appliances={"KITCHEN_SINK": _rim(rng, centre, 0.50, 0.40, 45.0)})
    counter_lines(layers, walls, 1.0)
    result = DxfPlanResult(
        project_name="Kit", level_name="EG", mode="indoor", profile_name="test", scale_factor=1.0, frame=Frame(np.zeros(3), 1.0, np.eye(3)),
        settings_fingerprint="x", layers=layers, statuses={name: LayerStatus(name, "ok", None, len(items)) for name, items in layers.items()},
        layer_colors={name: 7 for name in layers}, background=BackgroundImage(None, None, (0.0, 0.0, 5.0, 4.0)),
        layer_linetypes={}, layer_styles={}, blocks=["KITCHEN_SINK_500X400"], variant="ortho",
    )
    root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    folder = Path(root) / f"fronts_{uuid.uuid4().hex[:8]}" if root else tmp_path_factory.mktemp("fronts")
    folder.mkdir(parents=True, exist_ok=True)
    document = ezdxf.readfile(export_plan(result, folder, coordinates="local"))
    assert not document.audit().errors
    reference = next(item for item in document.modelspace() if item.dxftype() == "INSERT")
    assert reference.dxf.name.endswith("500X400") and reference.dxf.rotation % 90.0 == pytest.approx(45.0, abs=2.0)


def _peninsula_scene(rng, *, top: bool = True, hidden: bool = False) -> np.ndarray:
    """West leg + peninsula (front y 1.487): a cabinet to x 1.313, then a knee space whose back stands 0.24 m in."""
    parts = [
        _face(rng, (0.587, 1.487), (0.587, 3.95)), _face(rng, (0.587, 1.487), (1.313, 1.487)),
        _face(rng, (1.313, 1.247), (1.913, 1.247), high=0.74),
    ]
    if top:
        parts.append(_top(rng, box(0.62, 0.93, 1.91, 1.48)))
    if hidden:
        parts.append(_face(rng, (1.35, 1.69), (1.90, 1.69), high=0.75))  # something standing in front of it
    return np.vstack(parts)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_base_steps_back_under_the_worktop(upm: float, theta: float) -> None:
    peninsula = box(0.0, 0.917, 1.913, 1.487)
    _, _, stats, run = _layout(_peninsula_scene(np.random.default_rng(19)), upm, theta, legs=(WEST, peninsula))
    assert len(stats["kitchen_base_recesses"]) == 1
    recess = stats["kitchen_base_recesses"][0]
    assert recess["depth_m"] == pytest.approx(0.24, abs=0.025) and recess["length_m"] == pytest.approx(0.60, abs=0.06)
    base, worktop = unary_union(_solids(run, "base")), unary_union(_solids(run, "worktop"))
    assert (worktop.area - base.area) / upm**2 == pytest.approx(0.24 * 0.60, abs=0.025)
    assert worktop.buffer(1e-6).covers(base)
    _assert_abutting_and_identical(run, upm)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize("case", [{"top": False}, {"hidden": True}], ids=["no_worktop", "hidden_front"])
def test_no_recess_without_evidence(upm: float, theta: float, case: dict) -> None:
    peninsula = box(0.0, 0.917, 1.913, 1.487)
    _, _, stats, run = _layout(_peninsula_scene(np.random.default_rng(23), **case), upm, theta, legs=(WEST, peninsula))
    assert stats["kitchen_base_recesses"] == []
    assert unary_union(_solids(run, "base")).symmetric_difference(unary_union(_solids(run, "worktop"))).area < 1e-6


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize("seen", [True, False])
def test_upper_cabinet_end_is_cut_only_where_the_wall_behind_is_seen(upm: float, theta: float, seen: bool) -> None:
    rng = np.random.default_rng(29)
    parts = [_square_scene(rng), _face(rng, (1.133, 3.653), (2.913, 3.653), low=1.40, high=2.18)]
    if seen:
        parts.append(_face(rng, (0.713, 3.992), (1.10, 3.992), low=1.40, high=2.25, noise=0.004))
    layers, _, stats, _ = _layout(np.vstack(parts), upm, theta, uppers=(box(0.713, 3.653, 2.913, 4.0),))
    unit = _metres(next(item for item in layers["A-OBEN"] if item.get("_upper_cabinet"))["points"], upm, theta)
    assert unit[:, 0].min() == pytest.approx(1.133 if seen else 0.713, abs=0.035)
    assert unit[:, 0].max() == pytest.approx(2.913, abs=1e-3)
    assert len(stats["kitchen_uppers_trimmed"]) == int(seen)


@pytest.mark.parametrize("upm", [1.0, 0.8124])
def test_rotation_is_kept_only_with_clear_support(upm: float) -> None:
    def turned(width: float, depth: float, angle: float) -> Polygon:
        return affinity.scale(affinity.rotate(box(-width / 2, -depth / 2, width / 2, depth / 2), angle, origin=(0.0, 0.0)), upm, upm, origin=(0.0, 0.0))

    assert clear_rotation(turned(1.40, 0.70, 11.0), 0.0, upm) is not None
    assert clear_rotation(turned(1.40, 0.70, 2.0), 0.0, upm) is None, "within the noise of a traced outline"
    assert clear_rotation(turned(0.71, 0.70, 20.0), 0.0, upm) is None, "a square has no direction"
    notched = turned(1.40, 0.70, 11.0).difference(affinity.translate(turned(0.60, 0.50, 11.0), 0.40 * upm, 0.30 * upm))
    assert clear_rotation(notched, 0.0, upm) is None, "residual too large"
    rng = np.random.default_rng(37)
    half = _rim(rng, (0.0, 0.0), 0.60, 0.40, 12.0, 120)
    assert clear_point_rotation(half * upm, 0.0, upm) is not None
    assert clear_point_rotation(half[:20] * upm, 0.0, upm) is None, "too few points"


@pytest.mark.parametrize("upm", [1.0, 0.8124])
@pytest.mark.parametrize(("angle", "kept"), [(11.0, True), (2.0, False)])
def test_free_standing_furniture_keeps_a_clearly_measured_rotation(upm: float, angle: float, kept: bool) -> None:
    table = affinity.rotate(box(1.713, 1.513, 3.113, 2.213), angle, origin="centroid")
    table = affinity.scale(table, upm, upm, origin=(0.0, 0.0))
    layers = {"A-FURN": [{"type": "polyline", "points": _ring(table), "closed": True, "holes": [], "_label": "Tisch", "_furn_id": 1}], "A-ANNO": []}
    regularize_furniture(layers, [_plan(ROOM, upm, 0.0)], _plan(WALLS, upm, 0.0), 0.0, upm, PlanPolishSettings())
    ring = np.asarray(layers["A-FURN"][0]["points"])
    edge = max((ring[(index + 1) % len(ring)] - ring[index] for index in range(len(ring))), key=lambda vector: float(np.hypot(*vector)))
    measured = abs((math.degrees(math.atan2(edge[1], edge[0])) + 45.0) % 90.0 - 45.0)
    assert measured == pytest.approx(angle if kept else 0.0, abs=0.6)


def test_worktop_over_a_recessed_base_gets_the_measured_top_of_its_run() -> None:
    top, base = box(0.0, 0.0, 2.0, 0.6), box(0.0, 0.0, 2.0, 0.6).difference(box(1.4, 0.35, 2.0, 0.6))
    item = SimpleNamespace(catalogue_kind="worktop", default_z_m=[0.86, 0.90])
    worktop = SimpleNamespace(family="worktop", key=("top",), polygon=top, item=item)
    cabinet = SimpleNamespace(family="kitchen_base", key=("base",), polygon=base, item=item)
    model, solids = _kitchen_solids(worktop, top, {("base",): 0.915}, [cabinet, worktop], ReconSettings())
    assert model == "kitchen" and solids[0].z1 == pytest.approx(0.915)


def test_scan_points_are_optional_and_cut_to_the_kitchen() -> None:
    frame = Frame(np.zeros(3), 1.0, np.eye(3))
    assert load_kitchen_points(None, frame, 0.0, (0.0, 0.0, 1.0, 1.0)).shape == (0, 3)
    assert load_kitchen_points("cloud.laz", frame, 0.0, (0.0, 0.0, 1.0, 1.0)).shape == (0, 3)
    assert kitchen_scan_bounds({"A-EINB": []}, 1.0) is None
    run = {"type": "polyline", "points": _ring(NORTH), "closed": True, "_run_legs": [_ring(NORTH)]}
    assert kitchen_scan_bounds({"A-EINB": [run]}, 1.0) == pytest.approx((-0.6, 2.813, 3.713, 4.6))


def test_abutting_worktops_share_one_level_and_a_real_step_stays() -> None:
    works = [
        SimpleNamespace(key=("long",), polygon=box(0.0, 0.0, 2.8, 0.57)),
        SimpleNamespace(key=("short",), polygon=box(0.0, -1.9, 0.62, 0.0)),
        SimpleNamespace(key=("bar",), polygon=box(0.62, -1.9, 1.9, -1.4)),
        SimpleNamespace(key=("other",), polygon=box(4.0, 0.0, 5.0, 0.6)),
    ]
    tops = {("long",): 0.897, ("short",): 0.881, ("bar",): 1.101, ("other",): 0.885}
    counts = {("long",): 22000, ("short",): 4100, ("bar",): 900, ("other",): 700}
    _level_abutting_tops(works, tops, counts, 0.04, 1.0)
    assert tops == {("long",): 0.897, ("short",): 0.897, ("bar",): 1.101, ("other",): 0.885}


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_untouched_leg_keeps_exactly_its_fitted_rectangle(upm: float, theta: float) -> None:
    _, _, _, run = _layout(None, upm, theta, legs=(NORTH,))
    assert run["_run_solids"][0]["worktop"] == [run["_run_legs"][0]]
    assert run["_run_solids"][0]["base"] == [run["_run_legs"][0]]
