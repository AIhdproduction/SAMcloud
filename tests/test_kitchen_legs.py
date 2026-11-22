"""A kitchen leg the label raster lost, established from its scanned front; what is no leg stays none.

Synthetic scan points only (seeded noise, samples off the cell borders); every case runs at three scales / axes.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from samcloud.ortho.plan import _kitchen_label_points, load_clean_plan_settings
from samcloud.ortho.vector.kitchen_fronts import KitchenFrontSettings, ScanPoints, _density_ridge
from samcloud.ortho.vector.kitchen_layout import kitchen_cad_layout, kitchen_scan_bounds
from samcloud.ortho.vector.kitchen_legs import FrontLegSettings, recover_front_legs
from samcloud.ortho.vector.plan_polish import _LegBox
from samcloud.ortho.vector.raster_grid import RasterGrid
from test_kitchen_fronts import ROOM, SCALES, WALLS, _assert_abutting_and_identical, _face, _metres, _plan, _ring, _top, _xy

# the known leg on the north wall; like a raster run that lost the second leg it ends before the corner
NORTH = box(0.64, 3.413, 3.113, 4.0)


def _labels(rng, area: Polygon, density: float = 500.0) -> np.ndarray:
    x0, y0, x1, y1 = area.bounds
    count = int(density * area.area)
    return np.column_stack((rng.uniform(x0, x1, count), rng.uniform(y0, y1, count)))


def _known(rng) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return [_top(rng, box(0.9, 3.43, 3.0, 3.97))], [_labels(rng, box(0.75, 3.43, 3.1, 3.8))]


def _l_scene(rng, angle: float | None = 45.0, *, label_depth: float = 0.25, density: float = 2600.0, noise: float = 0.008, west: float = 0.587):
    """North run plus the lost west leg: its front, worktop and only a thin strip of labels (why the raster lost it)."""
    points, labels = _known(rng)
    face = lambda first, second: _face(rng, first, second, density=density, noise=noise)  # noqa: E731
    if angle is None:
        points += [face((west, 3.413), (3.113, 3.413)), face((west, 1.117), (west, 3.413))]
        first = second = np.asarray((west, 3.413))
    else:
        cut = 0.463
        first, second = np.asarray((west + cut, 3.413)), np.asarray((west, 3.413 - cut * math.tan(math.radians(angle))))
        points += [face(first, (3.113, 3.413)), face((west, 1.117), second), face(second, first)]
    points.append(_top(rng, box(west - 0.33, 1.13, west - 0.02, 3.8)))
    labels.append(_labels(rng, box(west - label_depth, 1.117, west, 3.6)))
    return np.vstack(points), np.vstack(labels), first, second


def _layout(points, labels, upm: float, theta: float, *, walls=WALLS, known=(NORTH,), extra=(), leg_settings=FrontLegSettings()):
    shapes = [_plan(leg, upm, theta) for leg in known]
    run = {"type": "polyline", "points": _ring(unary_union(shapes)), "closed": True, "holes": [], "_einb_kind": "fine_kitchen",
           "_kitchen_run": "straight", "_run_legs": [_ring(shape) for shape in shapes]}
    layers = {"A-EINB": [run, *extra], "A-FURN": [], "A-ANNO": [], "A-FURN-DETL": [], "A-OBEN": []}
    scan = np.column_stack((_xy(points[:, :2], upm, theta), points[:, 2] * upm))
    stats = kitchen_cad_layout(
        layers, [_plan(ROOM, upm, theta)], _plan(walls, upm, theta), theta, upm, None, {}, scan_points=scan,
        kitchen_labels=None if labels is None else _xy(labels, upm, theta), leg_settings=leg_settings,
    )
    return layers, stats, next(item for item in layers["A-EINB"] if item.get("_run_solids"))


def _bounds(ring, upm: float, theta: float) -> tuple[float, float, float, float]:
    local = _metres(ring, upm, theta)
    return float(local[:, 0].min()), float(local[:, 1].min()), float(local[:, 0].max()), float(local[:, 1].max())


# --------------------------------------------------------------------------------------------- legs that are there

@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize("angle", [45.0, 32.0, 56.0])
def test_lost_leg_and_its_diagonal_come_back_from_the_front(upm: float, theta: float, angle: float) -> None:
    points, labels, first, second = _l_scene(np.random.default_rng(41), angle)
    _, stats, run = _layout(points, labels, upm, theta)
    assert len(stats["kitchen_legs_from_fronts"]) == 1
    support = stats["kitchen_legs_from_fronts"][0]
    assert support["joined"] == "diagonal" and support["behind"] == "wall"
    assert support["depth_m"] == pytest.approx(0.587, abs=0.01)
    north, west = (_bounds(ring, upm, theta) for ring in run["_run_legs"])
    assert west == pytest.approx((0.0, 1.117, 0.587, 4.0), abs=0.05)
    assert west[0] == pytest.approx(0.0, abs=1e-6) and west[3] == pytest.approx(4.0, abs=1e-6)  # flush on both walls
    assert north[0] == pytest.approx(0.0, abs=1e-6), "the known leg reaches into the corner"
    assert run["_kitchen_run"] == "L"
    assert len(stats["kitchen_corner_fronts"]) == 1
    assert stats["kitchen_corner_fronts"][0]["angle_deg"] == pytest.approx(angle, abs=1.5)
    ring = _metres(run["points"], upm, theta)
    for expected in (first, second):
        assert np.min(np.hypot(*(ring - expected).T)) <= 0.025, expected
    _assert_abutting_and_identical(run, upm)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_lost_leg_at_a_square_corner(upm: float, theta: float) -> None:
    points, labels, _, _ = _l_scene(np.random.default_rng(43), None)
    _, stats, run = _layout(points, labels, upm, theta)
    assert [item["joined"] for item in stats["kitchen_legs_from_fronts"]] == ["fronts meet"]
    assert stats["kitchen_corner_fronts"] == []
    assert Polygon(run["points"]).area / upm**2 == pytest.approx(3.113 * 0.587 + 0.587 * (3.413 - 1.117), abs=0.05)
    _assert_abutting_and_identical(run, upm)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_known_leg_that_overshoots_or_stops_short_still_finds_the_front(upm: float, theta: float) -> None:
    points, labels, _, _ = _l_scene(np.random.default_rng(47))
    for start in (0.0, 0.31, 1.063):
        _, stats, run = _layout(points, labels, upm, theta, known=(box(start, 3.413, 3.113, 4.0),))
        assert len(stats["kitchen_legs_from_fronts"]) == 1, start
        assert _bounds(run["_run_legs"][0], upm, theta)[0] == pytest.approx(0.0, abs=1e-6), start


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_u_with_a_peninsula_is_found_leg_after_leg(upm: float, theta: float) -> None:
    rng = np.random.default_rng(53)
    points, labels = _known(rng)
    points += [
        _face(rng, (1.05, 3.413), (3.113, 3.413)), _face(rng, (0.587, 1.617), (0.587, 2.95)), _face(rng, (0.587, 2.95), (1.05, 3.413)),
        _face(rng, (0.587, 1.617), (1.613, 1.617)),  # peninsula front
        _face(rng, (0.60, 0.963), (1.613, 0.963), noise=0.012),  # its back, seen from the other side of the room
        _top(rng, box(0.25, 1.0, 0.57, 3.8)), _top(rng, box(0.6, 1.0, 1.60, 1.60)),
    ]
    labels += [_labels(rng, box(0.35, 1.617, 0.587, 3.6)), _labels(rng, box(0.587, 1.30, 1.613, 1.617))]
    _, stats, run = _layout(np.vstack(points), np.vstack(labels), upm, theta)
    assert [item["behind"] for item in stats["kitchen_legs_from_fronts"]] == ["wall", "back face"]
    peninsula = _bounds(run["_run_legs"][2], upm, theta)
    assert peninsula == pytest.approx((0.0, 0.963, 1.613, 1.617), abs=0.05)
    assert run["_kitchen_run"] == "U"
    _assert_abutting_and_identical(run, upm)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_recovered_leg_ends_at_a_flush_tall_unit(upm: float, theta: float) -> None:
    rng = np.random.default_rng(59)
    points, labels, _, _ = _l_scene(rng)
    tower = np.vstack((_face(rng, (0.587, 0.457), (0.587, 1.117)), _face(rng, (0.587, 0.457), (0.587, 1.117), low=0.865, high=2.10)))
    labels = np.vstack((labels, _labels(rng, box(0.3, 0.457, 0.587, 1.117))))
    _, stats, run = _layout(np.vstack((points, tower)), labels, upm, theta)
    assert len(stats["kitchen_legs_from_fronts"]) == 1
    assert _bounds(run["_run_legs"][1], upm, theta)[1] == pytest.approx(1.117, abs=0.06), "never into the tower"


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_dense_noisy_scan_keeps_leg_and_diagonal(upm: float, theta: float) -> None:
    rng = np.random.default_rng(61)
    points, labels, _, _ = _l_scene(rng, 45.0, density=26000.0, noise=0.019)
    _, stats, _ = _layout(points, labels, upm, theta)
    assert len(stats["kitchen_legs_from_fronts"]) == 1
    assert stats["kitchen_corner_fronts"][0]["angle_deg"] == pytest.approx(45.0, abs=2.0)


def test_density_ridge_pulls_a_slid_line_back_onto_the_face() -> None:
    rng = np.random.default_rng(67)
    normal, distance = np.asarray((math.cos(math.radians(44.0)), math.sin(math.radians(44.0)))), 0.327
    along = rng.uniform(-distance / normal[0] * normal[1], distance / normal[1] * normal[0], 9000)
    across = rng.normal(0.0, 0.012, 9000)
    strays = rng.random(9000) < 0.15
    across[strays] = rng.uniform(-0.02, 0.08, int(strays.sum()))
    xy = (distance + across)[:, None] * normal + along[:, None] * np.asarray((-normal[1], normal[0]))
    slid = math.radians(41.0)
    found = _density_ridge(xy[:, 0], xy[:, 1], np.asarray((math.cos(slid), math.sin(slid))), distance + 0.018, KitchenFrontSettings(), 1.0)
    assert found is not None
    assert math.degrees(math.atan2(found[0][1], found[0][0])) == pytest.approx(44.0, abs=0.5)
    assert found[1] == pytest.approx(distance, abs=0.005)


# --------------------------------------------------------------------------------------------- what is no leg

def _second_wall(rng, *, height: float = 0.865, top: float = 0.895, start: float = 2.95, stop: float = 1.117, front: float = 0.587):
    points, labels = _known(rng)
    points += [
        _face(rng, (front, stop), (front, start), high=height), _face(rng, (front + 0.463, 3.413), (3.113, 3.413)),
        _top(rng, box(front - 0.33, stop, front - 0.02, start), height=top),
    ]
    return points, labels


def _table(rng):
    """A table along the second wall, wrongly labelled kitchen: top at 0.75, only legs below it."""
    points, labels = _known(rng)
    points += [_face(rng, (1.05, 3.413), (3.113, 3.413)), _top(rng, box(0.05, 1.3, 0.79, 2.7), height=0.75)]
    points += [_face(rng, (0.75, y), (0.75, y + 0.05), high=0.73, density=9000.0) for y in (1.33, 2.62)]
    labels.append(_labels(rng, box(0.05, 1.3, 0.8, 2.7)))
    return np.vstack(points), np.vstack(labels)


def _sideboard(rng):
    """A worktop-high sideboard right at the corner, closed by a diagonal face even, but no kitchen label on it."""
    points, labels = _second_wall(rng)
    points.append(_face(rng, (0.587, 2.95), (1.05, 3.413)))
    return np.vstack(points), np.vstack(labels)


def _low_sideboard(rng):
    """Wrongly labelled kitchen, but only 0.74 high."""
    points, labels = _second_wall(rng, height=0.72, top=0.74)
    points.append(_face(rng, (0.587, 2.95), (1.05, 3.413), high=0.72))
    labels.append(_labels(rng, box(0.3, 1.117, 0.587, 2.95)))
    return np.vstack(points), np.vstack(labels)


def _separate_sideboard(rng):
    """Same height, wrongly labelled, along the second wall, but standing apart from the run."""
    points, labels = _second_wall(rng, start=2.60)
    labels.append(_labels(rng, box(0.3, 1.117, 0.587, 2.60)))
    return np.vstack(points), np.vstack(labels)


def _tall_row(rng):
    points, labels = _second_wall(rng)
    points += [_face(rng, (0.587, 2.95), (1.05, 3.413)), _face(rng, (0.587, 1.117), (0.587, 2.95), low=0.865, high=2.10)]
    labels.append(_labels(rng, box(0.3, 1.117, 0.587, 2.95)))
    return np.vstack(points), np.vstack(labels)


def _short_stub(rng):
    points, labels = _second_wall(rng, stop=2.517)
    points.append(_face(rng, (0.587, 2.95), (1.05, 3.413)))
    labels.append(_labels(rng, box(0.3, 2.517, 0.587, 2.95)))
    return np.vstack(points), np.vstack(labels)


def _against_the_wall(rng):
    """A shallow shelf: the wall is only 0.30 m behind its front."""
    points, labels = _second_wall(rng, front=0.30)
    points.append(_face(rng, (0.30, 2.95), (0.763, 3.413)))
    labels.append(_labels(rng, box(0.05, 1.117, 0.30, 2.95)))
    return np.vstack(points), np.vstack(labels)


def _island_without_a_back(rng):
    """Free in the room with no scanned back: nothing says how deep it would be."""
    points, labels = _second_wall(rng, front=1.387)
    points.append(_face(rng, (1.387, 2.95), (1.85, 3.413)))
    labels.append(_labels(rng, box(1.1, 1.117, 1.387, 2.95)))
    return np.vstack(points), np.vstack(labels)


NEGATIVES = [
    (_table, None), (_sideboard, "labels"), (_low_sideboard, "worktop"), (_separate_sideboard, "not joined"), (_tall_row, "tall"),
    (_short_stub, None), (_against_the_wall, "wall too close"), (_island_without_a_back, "nothing behind"),
]


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize(("scene", "reason"), NEGATIVES, ids=[scene.__name__.strip("_") for scene, _ in NEGATIVES])
def test_what_is_no_kitchen_leg_stays_none(upm: float, theta: float, scene, reason: str | None) -> None:
    points, labels = scene(np.random.default_rng(71))
    _, stats, run = _layout(points, labels, upm, theta)
    assert stats["kitchen_legs_from_fronts"] == []
    assert len(run["_run_legs"]) == 1 and run["_kitchen_run"] == "straight"
    assert stats["kitchen_corner_fronts"] == []
    if reason is not None:
        # the test that was meant to stop it did (wall frame: theta plays no part here)
        wall_frame = lambda shape: affinity.scale(shape, upm, upm, origin=(0.0, 0.0))  # noqa: E731
        leg = _LegBox.of(wall_frame(NORTH), 0.0)
        leg.back = "v1"
        trace: list[dict] = []
        scan = ScanPoints(points[:, :2] * upm, points[:, 2])
        assert recover_front_legs([leg], scan, labels * upm, wall_frame(WALLS), wall_frame(ROOM), KitchenFrontSettings(), upm, 0.90, trace=trace) == []
        assert reason in {item["reason"] for item in trace}, trace


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_a_perfect_front_without_labels_worktop_or_switch_is_no_leg(upm: float, theta: float) -> None:
    rng = np.random.default_rng(73)
    points, labels, _, _ = _l_scene(rng)
    assert _layout(points, None, upm, theta)[1]["kitchen_legs_from_fronts"] == [], "no labels"
    assert _layout(points, labels, upm, theta, leg_settings=FrontLegSettings(enabled=False))[1]["kitchen_legs_from_fronts"] == []
    west_top = (points[:, 2] > 0.875) & (points[:, 0] < 0.60)
    assert _layout(points[~west_top], labels, upm, theta)[1]["kitchen_legs_from_fronts"] == [], "no worktop behind the front"


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_known_second_leg_is_not_found_again(upm: float, theta: float) -> None:
    points, labels, _, _ = _l_scene(np.random.default_rng(79))
    known = (box(0.0, 3.413, 3.113, 4.0), box(0.0, 1.117, 0.587, 4.0))
    _, stats, run = _layout(points, labels, upm, theta, known=known)
    assert stats["kitchen_legs_from_fronts"] == [] and len(run["_run_legs"]) == 2


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_segment_that_retraces_the_recovered_leg_is_dropped(upm: float, theta: float) -> None:
    points, labels, _, _ = _l_scene(np.random.default_rng(83))
    segment = lambda first, second: {  # noqa: E731
        "type": "polyline", "points": _xy(np.asarray((first, second)), upm, theta).tolist(), "closed": False, "_einb_kind": "segment",
    }
    on_leg, elsewhere = segment((0.60, 1.3), (0.60, 2.7)), segment((4.2, 1.0), (4.2, 2.0))
    layers, _, _ = _layout(points, labels, upm, theta, extra=(on_leg, elsewhere))
    segments = [item for item in layers["A-EINB"] if item.get("_einb_kind") == "segment"]
    assert segments == [elsewhere]


def test_scan_bounds_and_label_points_reach_the_lost_leg() -> None:
    grid = RasterGrid(0.0, 4.0, 0.02, 250, 200)
    mask = np.zeros((200, 250), dtype=bool)
    mask[100:150, 5:25] = True
    labels = _kitchen_label_points([(SimpleNamespace(mask=mask), "kitchen counter", "Küche")], grid)
    assert labels is not None and len(labels) == mask.sum()
    assert labels.min(axis=0) == pytest.approx((0.11, 1.01)) and labels.max(axis=0) == pytest.approx((0.49, 1.99))
    run = {"type": "polyline", "points": _ring(NORTH), "closed": True, "_run_legs": [_ring(NORTH)]}
    x0, y0, _, _ = kitchen_scan_bounds({"A-EINB": [run]}, 1.0, labels)
    assert x0 <= 0.0 and y0 <= 0.5
    assert _kitchen_label_points([], grid) is None


def test_leg_settings_come_from_the_plan_config() -> None:
    settings = load_clean_plan_settings().indoor_kitchen_front_legs
    assert isinstance(settings, FrontLegSettings) and settings.enabled
    assert settings.min_label_share >= 0.5 and settings.min_depth_m < settings.max_depth_m
