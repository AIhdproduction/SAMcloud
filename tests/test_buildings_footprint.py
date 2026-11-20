import numpy as np
from shapely.geometry import Polygon

from samcloud.buildings.footprint import detect_facade_planes, footprint_from_facades
from samcloud.drone.contracts import PointSet


def _facade_cloud(vertices, seed=3):
    rng = np.random.default_rng(seed)
    points, normals = [], []
    for first, second in zip(vertices, vertices[1:] + vertices[:1], strict=False):
        t, z = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 6, 12))
        xy = np.asarray(first) + t.ravel()[:, None] * (np.asarray(second) - np.asarray(first))
        cloud = np.c_[xy, z.ravel()] + rng.normal(0, 0.01, (len(xy), 3))
        edge = np.asarray(second) - np.asarray(first)
        normal = np.array([-edge[1], edge[0], 0.])
        normal /= np.linalg.norm(normal)
        points.append(cloud)
        normals.append(np.tile(normal, (len(cloud), 1)))
    return PointSet(np.vstack(points), np.full(sum(map(len, points)), 6, dtype=np.uint16), normals=np.vstack(normals))


def test_l_footprint_from_six_facades():
    truth = Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 8), (0, 8)])
    facades = detect_facade_planes(_facade_cloud(list(truth.exterior.coords)[:-1]), distance=0.05, min_points=100, max_planes=8)
    footprint, source = footprint_from_facades(facades)
    assert source == "facades"
    assert footprint.intersection(truth).area / footprint.union(truth).area > 0.95


def test_missing_facades_fall_back_to_outline():
    truth = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
    facades = detect_facade_planes(_facade_cloud([(0, 0), (10, 0)]), distance=0.05, min_points=100, max_planes=8)
    footprint, source = footprint_from_facades(facades, np.asarray(truth.exterior.coords))
    assert source in {"mixed", "roof_outline"}
    assert footprint.intersection(truth).area / footprint.union(truth).area > 0.9
