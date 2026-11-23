"""Diagonal kitchen corner fronts with something in front of them, the straight front beside them, and the run shape.

Synthetic scan points only (seeded noise, samples jittered off the cell borders); every case runs at the three scales of
``test_kitchen_fronts``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import box

from samcloud.ortho.vector.kitchen_fronts import KitchenFrontSettings, _rear_envelope
from samcloud.ortho.vector.kitchen_layout import _run_shape
from samcloud.ortho.vector.plan_polish import _LegBox
from test_kitchen_fronts import NORTH, SCALES, WEST, _diagonal_scene, _face, _layout, _metres, _round_bin, _top


def _towel(rng, first, second, start: float = 0.25, stop: float = 0.70, bulge: float = 0.035, low: float = 0.24, high: float = 0.71) -> np.ndarray:
    """A cloth hanging in front of the middle of a front (``start``..``stop`` of its length): up to ``bulge`` into the room."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    vector = second - first
    length = float(np.hypot(*vector))
    normal = np.asarray((-vector[1], vector[0])) / length
    if float(normal @ (np.asarray((2.5, 2.0)) - first)) < 0.0:
        normal = -normal  # into the room
    count = int(2600.0 * length * (stop - start) * (high - low))
    t = rng.uniform(start, stop, count)
    sag = np.sin((t - start) / (stop - start) * math.pi)  # hangs off its middle
    offset = 0.012 + bulge * sag + rng.normal(0.0, 0.005, count)
    xy = first + t[:, None] * vector + offset[:, None] * normal
    return np.column_stack((xy, rng.uniform(low, high, count)))


def _hidden_behind(points: np.ndarray, first, second, start: float, stop: float, low: float, high: float) -> np.ndarray:
    """Drop the face points the cloth covers (seen from the room the cloth hides the front behind it)."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    vector = second - first
    length = float(np.hypot(*vector))
    t = (points[:, :2] - first) @ vector / length**2
    across = np.abs((points[:, :2] - first) @ np.asarray((-vector[1], vector[0])) / length)
    covered = (t >= start) & (t <= stop) & (across <= 0.03) & (points[:, 2] >= low) & (points[:, 2] <= high)
    return points[~covered]


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize("angle", [45.0, 41.0])
def test_diagonal_with_a_cloth_hanging_in_front_is_still_measured(upm: float, theta: float, angle: float) -> None:
    """A cloth over the middle of the corner unit, from below its top drawer to near the plinth (it hides the door)."""
    rng = np.random.default_rng(20261123 + int(angle))
    points, first, second = _diagonal_scene(rng, angle)
    points = np.vstack((_hidden_behind(points, first, second, 0.25, 0.75, 0.22, 0.72), _towel(rng, first, second, 0.25, 0.75, 0.045, 0.22, 0.72)))
    _, _, stats, run = _layout(points, upm, theta)
    assert len(stats["kitchen_corner_fronts"]) == 1
    assert stats["kitchen_corner_fronts"][0]["angle_deg"] == pytest.approx(angle, abs=1.5)
    ring = _metres(run["points"], upm, theta)
    for expected in (first, second):
        assert np.min(np.hypot(*(ring - expected).T)) <= 0.025, expected


def _door_ajar(rng, first, second, hinge_at_first: bool, opened_deg: float, low: float = 0.22, high: float = 0.72) -> np.ndarray:
    """The door of the corner unit (between its plinth and its top drawer) standing ajar by ``opened_deg`` into the room."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    hinge, free = (first, second) if hinge_at_first else (second, first)
    hinge = hinge + 0.03 * (free - hinge) / float(np.hypot(*(free - hinge)))
    vector = free - hinge
    corner = np.asarray((0.587, 3.413))
    for sign in (1.0, -1.0):  # the turn that moves the free edge away from the corner, into the room
        cos, sin = math.cos(sign * math.radians(opened_deg)), math.sin(sign * math.radians(opened_deg))
        moved = np.asarray((vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos))
        if float(np.hypot(*(hinge + moved - corner))) > float(np.hypot(*(free - corner))):
            break
    leaf = 0.94 * moved
    return _face(rng, hinge + 0.012 * leaf / float(np.hypot(*leaf)), hinge + leaf, low, high)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize(("angle", "hinge_at_first", "opened"), [(45.0, True, 7.0), (43.0, False, 9.0), (50.0, True, 6.0)])
def test_diagonal_with_a_door_ajar_is_fitted_on_the_unit_not_the_door(upm: float, theta: float, angle: float, hinge_at_first: bool, opened: float) -> None:
    """Plinth and top drawer stay on the unit's plane; the door between them stands ajar (as on a real sink unit)."""
    rng = np.random.default_rng(20261124 + int(angle) + int(opened))
    points, first, second = _diagonal_scene(rng, angle)
    points = np.vstack((_hidden_behind(points, first, second, 0.0, 1.0, 0.22, 0.72), _door_ajar(rng, first, second, hinge_at_first, opened)))
    _, _, stats, run = _layout(points, upm, theta)
    assert len(stats["kitchen_corner_fronts"]) == 1
    assert stats["kitchen_corner_fronts"][0]["angle_deg"] == pytest.approx(angle, abs=1.0)
    ring = _metres(run["points"], upm, theta)
    for expected in (first, second):
        assert np.min(np.hypot(*(ring - expected).T)) <= 0.02, expected


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_unit_without_anything_in_front_keeps_the_common_fit(upm: float, theta: float) -> None:
    """All height bands agree: the band check leaves the fit alone."""
    from samcloud.ortho.vector import kitchen_fronts

    points, _, _ = _diagonal_scene(np.random.default_rng(77), 45.0)
    _, _, plain, _ = _layout(points, upm, theta)
    calls: list[object] = []
    original = kitchen_fronts._rear_band_line

    def spy(*args):
        calls.append(original(*args))
        return calls[-1]

    kitchen_fronts._rear_band_line = spy
    try:
        _, _, again, _ = _layout(points, upm, theta)
    finally:
        kitchen_fronts._rear_band_line = original
    assert calls and all(item is None for item in calls)
    assert again["kitchen_corner_fronts"] == plain["kitchen_corner_fronts"]


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_round_bin_in_front_of_a_square_corner_stays_square_with_a_cloth_beside_it(upm: float, theta: float) -> None:
    rng = np.random.default_rng(41)
    points = np.vstack((_round_bin(rng), _towel(rng, (0.587, 2.0), (0.587, 2.9))))
    _, _, stats, _ = _layout(points, upm, theta)
    assert stats["kitchen_corner_fronts"] == []


def test_rear_envelope_keeps_the_rear_most_sample_per_station() -> None:
    along = np.asarray([0.011, 0.013, 0.012, 0.052, 0.054, 0.093])
    across = np.asarray([0.031, -0.004, 0.022, 0.003, -0.011, 0.007])
    middle, rear = _rear_envelope(along, across, 0.04)
    assert rear.tolist() == pytest.approx([-0.004, -0.011, 0.007])
    assert middle.tolist() == pytest.approx([0.012, 0.053, 0.093])
    assert all(len(item) == 0 for item in _rear_envelope(np.zeros(0), np.zeros(0), 0.04))


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize("seen_from", [2.25, 0.917])
def test_front_beside_the_diagonal_is_judged_next_to_it(upm: float, theta: float, seen_from: float) -> None:
    """The west front is seen only from ``seen_from`` up (a peninsula or an open end below it), 4.7 cm off the fitted edge."""
    rng = np.random.default_rng(5)
    west, north, cut = 0.6337, 3.413, 0.463
    first, second = (west + cut, north), (west, north - cut)
    points = np.vstack((
        _face(rng, first, (3.113, north)), _face(rng, (west, seen_from), second), _face(rng, second, first),
        _top(rng, box(0.7, 3.5, 3.0, 3.97)),
    ))
    _, _, stats, run = _layout(points, upm, theta)
    assert len(stats["kitchen_corner_fronts"]) == 1
    west_leg = _metres(run["_run_legs"][1], upm, theta)
    assert west_leg[:, 0].max() == pytest.approx(west, abs=0.008)
    ring = _metres(run["points"], upm, theta)
    for expected in (first, second):
        assert np.min(np.hypot(*(ring - expected).T)) <= 0.02, expected


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_front_seen_nowhere_near_the_diagonal_keeps_the_fitted_edge(upm: float, theta: float) -> None:
    """Only a short piece of the west front far from the corner is seen: the leg keeps its fitted edge."""
    rng = np.random.default_rng(6)
    north, cut = 3.413, 0.463
    first, second = (0.587 + cut, north), (0.587, north - cut)
    points = np.vstack((
        _face(rng, first, (3.113, north)), _face(rng, (0.6337, 0.95), (0.6337, 1.25)), _face(rng, second, first),
        _top(rng, box(0.7, 3.5, 3.0, 3.97)),
    ))
    _, _, _, run = _layout(points, upm, theta)
    west_leg = _metres(run["_run_legs"][1], upm, theta)
    assert west_leg[:, 0].max() == pytest.approx(WEST.bounds[2], abs=1e-6)


def test_near_front_setting_is_physical() -> None:
    settings = KitchenFrontSettings()
    assert 0.6 <= settings.near_front_m <= 1.5


@pytest.mark.parametrize("upm", [1.0, 0.8124])
def test_run_shape_follows_the_final_legs(upm: float) -> None:
    def leg(shape):
        x0, y0, x1, y1 = (value * upm for value in shape.bounds)
        return _LegBox.of(box(x0, y0, x1, y1), 0.0)

    assert _run_shape([leg(NORTH), leg(WEST)], upm) == "L"
    assert _run_shape([leg(NORTH)], upm) == "straight"
    peninsula = box(0.587, 0.917, 2.3, 1.487)
    assert _run_shape([leg(NORTH), leg(WEST), leg(peninsula)], upm) == "U"
    apart = box(1.65, 3.413, 3.113, 4.0)  # the north leg starts 1.06 m beyond the west leg's front
    assert _run_shape([leg(apart), leg(box(0.0, 0.917, 0.587, 3.413))], upm) == "multi"
    near = box(0.607, 3.413, 3.113, 4.0)  # a 2 cm joint is one run
    assert _run_shape([leg(near), leg(box(0.0, 0.917, 0.587, 4.0))], upm) == "L"


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_layout_reports_the_shape_of_the_final_run(upm: float, theta: float) -> None:
    rng = np.random.default_rng(9)
    points, _, _ = _diagonal_scene(rng, 45.0)
    _, _, stats, run = _layout(points, upm, theta)
    assert stats["kitchen_run_shape"] == "L" and run["_kitchen_run"] == "L"
