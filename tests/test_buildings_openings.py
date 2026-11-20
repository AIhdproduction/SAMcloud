import numpy as np

from samcloud.buildings.openings import detect_building_openings
from samcloud.drone.contracts import BuildingModel, PointSet, WINDOW, DOOR


def _model():
    vertices = np.array([[0, 0, 0], [10, 0, 0], [10, 6, 0], [0, 6, 0], [0, 0, 3], [10, 0, 3], [10, 6, 3], [0, 6, 3]], float)
    faces = np.array([[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6], [3, 7, 6], [3, 0, 7], [0, 4, 7]])
    return BuildingModel(11, "2.2", vertices, faces, np.ones(len(faces), np.uint8), "ok")


def _patch(u0, u1, z0, z1, label, views=6, seed=3):
    rng = np.random.default_rng(seed)
    us = np.arange(u0 + 0.02, u1 - 0.01, 0.04)
    zs = np.arange(z0 + 0.02, z1 - 0.01, 0.04)
    u, z = np.meshgrid(us, zs)
    return np.column_stack((u.ravel(), rng.normal(0.0, 0.01, u.size), z.ravel())), np.full(u.size, label, np.uint16), np.full(u.size, views, np.uint16)


def _points(*patches):
    xyz, labels, views = zip(*patches)
    return PointSet(np.concatenate(xyz), np.concatenate(labels), view_count=np.concatenate(views))


def _detect(points):
    return detect_building_openings(points, _model(), [], dtm_sampler=lambda x, y: 0.0)


def test_two_windows_and_door_are_measured_on_one_facade():
    ps = _points(_patch(1, 2, 1, 2, WINDOW), _patch(3, 4, 1.1, 2.1, WINDOW), _patch(6, 7, 0, 2.1, DOOR))
    openings, qa = _detect(ps)

    assert len(openings) == 3
    assert sorted(item.kind for item in openings) == ["door", "window", "window"]
    assert not qa
    assert all(abs(item.u0 - target) <= 0.05 for item, target in zip(sorted(openings, key=lambda item: item.u0), (1, 3, 6)))


def test_floating_door_is_redeclared_as_window():
    openings, qa = _detect(_points(_patch(2, 3, 0.9, 2.2, DOOR)))

    assert not qa
    assert len(openings) == 1
    assert openings[0].kind == "window"
    assert openings[0].status == "warning"
    assert "floating door" in openings[0].reason


def test_low_floating_door_is_rejected():
    openings, qa = _detect(_points(_patch(2, 3, 0.3, 2.2, DOOR)))

    assert not openings
    assert any("floating door" in item.reason for item in qa)


def test_low_window_sill_is_rejected():
    openings, qa = _detect(_points(_patch(2, 3, 0.2, 1.3, WINDOW)))

    assert not openings
    assert any("window sill" in item.reason for item in qa)


def test_valid_door_and_window_are_kept():
    openings, qa = _detect(_points(_patch(2, 3, 0.05, 2.1, DOOR), _patch(4, 5, 0.9, 1.9, WINDOW)))

    assert len(openings) == 2
    assert not qa
    assert all(item.status == "ok" for item in openings)


def test_minimum_views_rejects_weak_window():
    openings, qa = _detect(_points(_patch(2, 3, 0.9, 1.9, WINDOW, views=2)))

    assert not openings
    assert any("only 2 views" in item.reason for item in qa)


def test_outside_and_tiny_openings_are_rejected():
    outside = _patch(9.5, 10.4, 0.9, 1.9, WINDOW)
    tiny = _patch(4, 4.25, 0.9, 1.2, WINDOW)
    openings, qa = _detect(_points(outside, tiny))

    assert not openings
    assert any("area" in item.reason or "outside facade" in item.reason for item in qa)
