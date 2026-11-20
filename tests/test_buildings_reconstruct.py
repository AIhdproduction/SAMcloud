import time

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from samcloud.buildings.extrude import extrude_lod22
from samcloud.buildings.footprint import detect_facade_planes, footprint_from_facades
from samcloud.buildings.labeling import label_cells
from samcloud.buildings.partition import partition_footprint, plane_lines
from samcloud.buildings.reconstruct import _roof_subset, reconstruct_building
from samcloud.buildings.roof_planes import detect_roof_planes
from samcloud.buildings.validate import validate_model
from samcloud.drone.contracts import BUILDING, PointSet


def test_garbage_cloud_returns_failed_model_without_exception():
    rng = np.random.default_rng(12)
    points = PointSet(rng.normal(size=(40, 3)), np.full(40, 6, dtype=np.uint16))
    model = reconstruct_building(points, 41, dtm_sampler=lambda _x, _y: 0.)
    assert model.status == "failed"
    assert model.reason


def test_timeout_configuration_falls_back_or_fails_cleanly():
    rng = np.random.default_rng(3)
    xyz = rng.uniform(0, 4, (80, 3))
    points = PointSet(xyz, np.full(80, 145, dtype=np.uint16), normals=np.tile([0., 0., 1.], (80, 1)))
    config = {"buildings": {"ransac_distance_m": .05, "region_growing": {"k": 8, "epsilon_m": .08, "min_points": 20, "normal_angle_deg": 10}, "label_pruning_buffer_m": 2.5, "complexity_factor": .7, "max_seconds_per_building": .01}}
    model = reconstruct_building(points, 42, dtm_sampler=lambda _x, _y: 0., config=config)
    assert model.status in {"lod1_fallback", "failed"}


def _config(*, timeout=10.0):
    return {"buildings": {"ransac_distance_m": 0.05, "region_growing": {"k": 12, "epsilon_m": 0.08, "min_points": 100, "normal_angle_deg": 10}, "label_pruning_buffer_m": 2.5, "complexity_factor": 0.7, "max_seconds_per_building": timeout}}


def _roof_points(rng, polygon, height, normal, *, density=34, sigma=0.01):
    min_x, min_y, max_x, max_y = polygon.bounds
    xx, yy = np.meshgrid(np.linspace(min_x + 0.04, max_x - 0.04, density), np.linspace(min_y + 0.04, max_y - 0.04, density))
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    xy = xy[np.array([polygon.covers(Point(point)) for point in xy])]
    xyz = np.column_stack((xy, height(xy[:, 0], xy[:, 1])))
    return xyz + rng.normal(0.0, sigma, xyz.shape), np.tile(np.asarray(normal, dtype=float) / np.linalg.norm(normal), (len(xyz), 1))


def _facade_points(rng, footprint, *, eave=6.0, sigma=0.01):
    points, normals = [], []
    ring = np.asarray(footprint.exterior.coords, dtype=float)
    for start, end in zip(ring[:-1], ring[1:], strict=False):
        edge = end - start
        length = float(np.linalg.norm(edge))
        count = max(18, int(np.ceil(length * 5)))
        fractions, heights = np.meshgrid(np.linspace(0.03, 0.97, count), np.linspace(0.08, eave - 0.08, 24))
        xy = start + fractions.ravel()[:, None] * edge
        xyz = np.column_stack((xy, heights.ravel()))
        points.append(xyz + rng.normal(0.0, sigma, xyz.shape))
        normals.append(np.tile(np.array((-edge[1], edge[0], 0.0)) / length, (len(xyz), 1)))
    return np.vstack(points), np.vstack(normals)


def _point_set(seed, roof_parts, footprint, *, eave=6.0):
    rng = np.random.default_rng(seed)
    point_groups, normal_groups = [], []
    for polygon, height, normal in roof_parts:
        points, normals = _roof_points(rng, polygon, height, normal)
        point_groups.append(points)
        normal_groups.append(normals)
    points, normals = _facade_points(rng, footprint, eave=eave)
    point_groups.append(points)
    normal_groups.append(normals)
    xyz, surface_normals = np.vstack(point_groups), np.vstack(normal_groups)
    return PointSet(xyz, np.full(len(xyz), BUILDING, dtype=np.uint16), normals=surface_normals)


def _gable_cloud(seed=41):
    footprint = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
    return _point_set(seed, [(Polygon([(0, 0), (5, 0), (5, 8), (0, 8)]), lambda x, _y: 6.0 + 0.6 * x, [-0.6, 0, 1]), (Polygon([(5, 0), (10, 0), (10, 8), (5, 8)]), lambda x, _y: 12.0 - 0.6 * x, [0.6, 0, 1])], footprint)


def _hip_cloud(seed=42):
    footprint = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
    return _point_set(seed, [(Polygon([(0, 0), (4, 4), (0, 8)]), lambda x, _y: 6.0 + 0.75 * x, [-0.75, 0, 1]), (Polygon([(12, 0), (12, 8), (8, 4)]), lambda x, _y: 15.0 - 0.75 * x, [0.75, 0, 1]), (Polygon([(0, 0), (12, 0), (8, 4), (4, 4)]), lambda _x, y: 6.0 + 0.75 * y, [0, -0.75, 1]), (Polygon([(0, 8), (4, 4), (8, 4), (12, 8)]), lambda _x, y: 12.0 - 0.75 * y, [0, 0.75, 1])], footprint)


def _flat_annex_cloud(seed=43):
    main = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    annex = Polygon([(10, 0), (14, 0), (14, 5), (10, 5)])
    return _point_set(seed, [(main, lambda x, _y: np.full_like(x, 6.0), [0, 0, 1]), (annex, lambda x, _y: np.full_like(x, 7.5), [0, 0, 1])], unary_union((main, annex)), eave=7.5)


def _l_gable_cloud(seed=44):
    main = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
    wing = Polygon([(10, 0), (14, 0), (14, 5), (10, 5)])
    footprint = unary_union((main, wing))
    cloud = _point_set(seed, [(Polygon([(0, 0), (5, 0), (5, 8), (0, 8)]), lambda x, _y: 6.0 + 0.6 * x, [-0.6, 0, 1]), (Polygon([(5, 0), (10, 0), (10, 8), (5, 8)]), lambda x, _y: 12.0 - 0.6 * x, [0.6, 0, 1]), (Polygon([(10, 0), (14, 0), (14, 2.5), (10, 2.5)]), lambda _x, y: 6.0 + 0.6 * y, [0, -0.6, 1]), (Polygon([(10, 2.5), (14, 2.5), (14, 5), (10, 5)]), lambda _x, y: 9.0 - 0.6 * y, [0, 0.6, 1])], footprint)
    return cloud, footprint


def _assert_watertight(model):
    assert trimesh.Trimesh(vertices=model.vertices, faces=model.faces, process=False).is_watertight


def _projected_iou(model, expected):
    actual = unary_union([Polygon(model.vertices[face, :2]) for face in model.faces[model.face_roles == 2]])
    return actual.intersection(expected).area / actual.union(expected).area


def test_gable_house_reconstructs_lod22_with_accurate_roof_and_heights():
    started = time.perf_counter()
    model = reconstruct_building(_gable_cloud(), 51, dtm_sampler=lambda _x, _y: 0.0, config=_config())
    elapsed = time.perf_counter() - started
    assert model.status == "ok"
    assert model.lod == "2.2"
    _assert_watertight(model)
    assert model.rmse < 0.03
    assert abs(model.heights["ridge"] - 9.0) < 0.05
    assert abs(model.heights["eave"] - 6.0) < 0.05
    assert elapsed < 20.0


def test_hip_roof_has_four_labels_is_watertight_and_accurate():
    cloud, config = _hip_cloud(), _config()
    started = time.perf_counter()
    planes = detect_roof_planes(cloud, k=12, epsilon=0.08, min_points=100, normal_angle_deg=10)
    facades = detect_facade_planes(cloud, distance=0.05, min_points=100, max_planes=16)
    footprint, _source = footprint_from_facades(facades, cloud.xyz[:, :2])
    partition = partition_footprint(footprint, plane_lines(planes, footprint))
    roof_xyz, label_planes = _roof_subset(cloud, planes)
    labels = label_cells(partition, label_planes, roof_xyz, buffer=2.5, complexity_factor=0.7, time_limit=10.0)
    vertices, faces, _roles = extrude_lod22(partition, labels.labels, label_planes, footprint, 0.0)
    report = validate_model(vertices, faces, roof_xyz, footprint)
    model = reconstruct_building(cloud, 52, dtm_sampler=lambda _x, _y: 0.0, config=config)
    elapsed = time.perf_counter() - started
    assert len(set(labels.labels) - {0}) == 4
    assert trimesh.Trimesh(vertices=vertices, faces=faces, process=False).is_watertight
    assert report["rmse"] < 0.03
    assert model.status == "ok"
    assert model.rmse < 0.03
    assert elapsed < 20.0


def test_flat_roof_annex_reconstructs_step_wall_and_both_heights():
    cloud, config = _flat_annex_cloud(), _config()
    started = time.perf_counter()
    model = reconstruct_building(cloud, 53, dtm_sampler=lambda x, _y: 0.02 * x, config=config)
    elapsed = time.perf_counter() - started
    assert model.status == "ok"
    _assert_watertight(model)
    assert model.rmse < 0.03
    planes = detect_roof_planes(cloud, k=12, epsilon=0.08, min_points=100, normal_angle_deg=10)
    facades = detect_facade_planes(cloud, distance=0.05, min_points=100, max_planes=16)
    footprint, _source = footprint_from_facades(facades, cloud.xyz[:, :2])
    partition = partition_footprint(footprint, plane_lines(planes, footprint))
    roof_xyz, label_planes = _roof_subset(cloud, planes)
    labels = label_cells(partition, label_planes, roof_xyz, buffer=2.5, complexity_factor=0.7, time_limit=10.0)
    annex = Polygon([(10, 0), (14, 0), (14, 5), (10, 5)])
    annex_label = 1 + int(np.argmax([-plane.d / plane.normal[2] for plane in label_planes]))
    labelled_annex_area = sum(cell.intersection(annex).area for cell, label in zip(partition.cells, labels.labels, strict=False) if label == annex_label)
    assert labelled_annex_area >= 0.95 * annex.area
    walls = model.vertices[model.faces[model.face_roles == 1]]
    step_faces = walls[np.isclose(walls[:, :, 0], 10.0, atol=0.08).all(axis=1)]
    assert len(step_faces)
    step_faces = step_faces[(np.ptp(step_faces[:, :, 2], axis=1) > 1.3) & (np.ptp(step_faces[:, :, 2], axis=1) < 1.7)]
    assert len(step_faces)
    step_y = step_faces[:, :, 1]
    assert (step_y >= -0.1).all() and (step_y <= 5.1).all()
    assert step_y.min() <= 0.1 and step_y.max() >= 4.9
    assert abs(model.heights["eave"] - 6.0) < 0.05
    assert abs(model.heights["ridge"] - 7.5) < 0.05
    assert elapsed < 20.0


def test_l_shaped_gables_preserve_footprint_or_return_reasoned_fallback():
    cloud, expected = _l_gable_cloud()
    started = time.perf_counter()
    model = reconstruct_building(cloud, 54, dtm_sampler=lambda _x, _y: 0.0, config=_config())
    elapsed = time.perf_counter() - started
    assert model.status in {"ok", "lod1_fallback"}
    if model.status == "lod1_fallback":
        assert model.reason
    assert model.rmse < 0.03
    assert _projected_iou(model, expected) > 0.95
    assert elapsed < 20.0


def test_timeout_on_gable_returns_watertight_lod1_fallback():
    model = reconstruct_building(_gable_cloud(), 55, dtm_sampler=lambda _x, _y: 0.0, config=_config(timeout=0.01))
    assert model.status == "lod1_fallback"
    assert model.lod in {"1.3", "1.2"}
    assert "timed out" in model.reason
    _assert_watertight(model)


def test_gable_face_roles_have_expected_normals():
    model = reconstruct_building(_gable_cloud(), 56, dtm_sampler=lambda _x, _y: 0.0, config=_config())
    normals = np.cross(model.vertices[model.faces[:, 1]] - model.vertices[model.faces[:, 0]], model.vertices[model.faces[:, 2]] - model.vertices[model.faces[:, 0]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    assert (normals[model.face_roles == 0, 2] < 0).all()
    assert (normals[model.face_roles == 2, 2] > 0).all()
    assert (np.abs(normals[model.face_roles == 1, 2]) < 0.1).all()
