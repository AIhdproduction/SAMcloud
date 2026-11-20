import math
import time

import numpy as np

from samcloud.buildings.roof_planes import detect_roof_planes, plane_ids
from samcloud.drone.contracts import PointSet, ROOF_FLAT, ROOF_INCLINED


def _pointset(xyz, normals, code=ROOF_INCLINED):
    return PointSet(np.asarray(xyz), np.full(len(xyz), code, dtype=np.uint16), normals=np.asarray(normals))


def _gable(seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.linspace(0, 8, 28), np.linspace(0, 5, 25))
    left = np.c_[x.ravel(), y.ravel(), 6 + 0.6 * x.ravel()]
    right = np.c_[10 - x.ravel(), y.ravel(), 6 + 0.6 * x.ravel()]
    points = np.r_[left, right] + rng.normal(0, 0.01, (len(left) * 2, 3))
    normals = np.r_[np.tile([-0.6, 0, 1], (len(left), 1)), np.tile([0.6, 0, 1], (len(right), 1))]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return _pointset(points, normals)


def test_gable_detects_two_accurate_planes():
    planes = detect_roof_planes(_gable(), k=16, epsilon=0.08, min_points=200, normal_angle_deg=10)
    assert len(planes) == 2
    assert all(abs(plane.slope_deg - math.degrees(math.atan(0.6))) < 1 for plane in planes)
    assert all(abs(abs(plane.normal[0]) - 0.6 / math.sqrt(1.36)) < 0.02 for plane in planes)


def test_hip_detects_four_planes():
    rng = np.random.default_rng(11)
    u, v = np.meshgrid(np.linspace(0.05, 3.95, 24), np.linspace(0.05, 3.95, 24))
    u, v = u.ravel(), v.ravel()
    templates = [
        (np.c_[u, v], np.array([-0.7, -0.7, 1.])), (np.c_[8-u, v], np.array([0.7, -0.7, 1.])),
        (np.c_[u, 8-v], np.array([-0.7, 0.7, 1.])), (np.c_[8-u, 8-v], np.array([0.7, 0.7, 1.])),
    ]
    points, normals = [], []
    for xy, normal in templates:
        z = 10 - 0.7 * np.abs(xy[:, 0] - 4) - 0.7 * np.abs(xy[:, 1] - 4)
        points.append(np.c_[xy, z] + rng.normal(0, 0.01, (len(xy), 3)))
        normal /= np.linalg.norm(normal)
        normals.append(np.tile(normal, (len(xy), 1)))
    planes = detect_roof_planes(_pointset(np.vstack(points), np.vstack(normals)), k=12, epsilon=0.08, min_points=120, normal_angle_deg=10)
    assert len(planes) >= 4


def test_flat_annex_and_unassigned_ids():
    x, y = np.meshgrid(np.linspace(0, 6, 24), np.linspace(0, 6, 24))
    main = np.c_[x.ravel(), y.ravel(), np.full(x.size, 6.0)]
    xa, ya = np.meshgrid(np.linspace(6.2, 9, 16), np.linspace(0, 4, 16))
    annex = np.c_[xa.ravel(), ya.ravel(), np.full(xa.size, 7.5)]
    points = np.r_[main, annex]
    ps = _pointset(points, np.tile([0., 0., 1.], (len(points), 1)), ROOF_FLAT)
    planes = detect_roof_planes(ps, k=12, epsilon=0.08, min_points=100, normal_angle_deg=10)
    assert len(planes) == 2 and all(plane.slope_deg < 1 for plane in planes)
    assert (plane_ids(ps, planes) >= 0).all()


def test_twenty_degree_ridge_is_not_merged():
    ps = _gable(19)
    slope = math.tan(math.radians(10))
    ps.xyz[:, 2] = 6 + slope * (5 - np.abs(ps.xyz[:, 0] - 5))
    ps.normals[: len(ps) // 2] = [-slope, 0, 1]
    ps.normals[len(ps) // 2:] = [slope, 0, 1]
    ps.normals /= np.linalg.norm(ps.normals, axis=1)[:, None]
    assert len(detect_roof_planes(ps, k=16, epsilon=0.08, min_points=200, normal_angle_deg=10)) == 2


def test_three_hundred_thousand_points_finish_within_twenty_seconds():
    rng = np.random.default_rng(23)
    xy = rng.uniform((0, 0), (100, 100), (300_000, 2))
    points = np.c_[xy, 8 + rng.normal(0, 0.01, len(xy))]
    ps = _pointset(points, np.tile([0., 0., 1.], (len(points), 1)), ROOF_FLAT)
    started = time.perf_counter()
    planes = detect_roof_planes(ps, k=16, epsilon=0.08, min_points=200, normal_angle_deg=10)
    assert len(planes) == 1
    assert time.perf_counter() - started < 20
