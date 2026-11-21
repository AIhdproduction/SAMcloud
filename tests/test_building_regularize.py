"""Regression tests for conservative LoD2 building regularisation."""

from __future__ import annotations

import math

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon

from samcloud.buildings.reconstruct import reconstruct_building
from samcloud.buildings.regularize import footprint_iou, regularize_footprint, regularize_roof_planes
from samcloud.drone.contracts import BUILDING, PointSet, RoofPlane


def _config(**regularize):
    return {
        "buildings": {
            "ransac_distance_m": 0.05,
            "region_growing": {"k": 12, "epsilon_m": 0.08, "min_points": 80, "normal_angle_deg": 10},
            "label_pruning_buffer_m": 2.5,
            "complexity_factor": 0.7,
            "max_seconds_per_building": 10.0,
            "regularize": regularize,
        }
    }


def _plane(gradient, intercept, outline, indices):
    normal = np.array((-gradient[0], -gradient[1], 1.0), dtype=float)
    normal /= np.linalg.norm(normal)
    return RoofPlane(normal, -intercept / np.linalg.norm(np.array((-gradient[0], -gradient[1], 1.0))), np.asarray(indices), 0.01, outline=outline)


def _gable_cloud(seed=101, skew=0.0):
    rng = np.random.default_rng(seed)
    footprint = Polygon([(0, 0), (12, 0), (12, 6), (0, 6)])
    roof, normals = [], []
    for x_range, gradient in [((0.05, 5.95), np.array((skew, 0.6))), ((6.05, 11.95), np.array((-skew, -0.6)))]:
        x, y = np.meshgrid(np.linspace(*x_range, 25), np.linspace(0.05, 5.95, 25))
        z = 6.0 + x * gradient[0] + y * gradient[1] if gradient[1] > 0 else 6.0 + x * gradient[0] + (6.0 - y) * -gradient[1]
        roof.append(np.c_[x.ravel(), y.ravel(), z.ravel()] + rng.normal(0, 0.01, (x.size, 3)))
        normal = np.array((-gradient[0], -gradient[1], 1.0)); normal /= np.linalg.norm(normal)
        normals.append(np.tile(normal, (x.size, 1)))
    walls, wall_normals = [], []
    for start, end in zip(list(footprint.exterior.coords)[:-1], list(footprint.exterior.coords)[1:], strict=False):
        start, end = np.asarray(start), np.asarray(end)
        t, z = np.meshgrid(np.linspace(0.02, 0.98, 24), np.linspace(0.05, 5.95, 20))
        xy = start + t.ravel()[:, None] * (end - start)
        walls.append(np.c_[xy, z.ravel()] + rng.normal(0, 0.01, (len(xy), 3)))
        edge = end - start; normal = np.array((-edge[1], edge[0], 0.0)); normal /= np.linalg.norm(normal)
        wall_normals.append(np.tile(normal, (len(xy), 1)))
    return PointSet(np.vstack([*roof, *walls]), np.full(sum(len(item) for item in [*roof, *walls]), BUILDING, dtype=np.uint16), normals=np.vstack([*normals, *wall_normals]))


def test_noisy_l_footprint_removes_jogs_and_keeps_iou():
    noisy = Polygon([(0, 0), (10, 0.16), (10.03, 4), (4.18, 4.02), (4, 4.18), (4.04, 8), (-0.13, 8), (0.12, 7.78)])
    regular, report = regularize_footprint(noisy)
    assert len(regular.exterior.coords) - 1 == 6
    assert report["iou"] >= 0.95
    assert footprint_iou(noisy, regular) >= 0.95


def test_long_oblique_wall_is_preserved():
    footprint = Polygon([(0, 0), (12, 0), (12, 4), (8, 8), (0, 8)])
    regular, _report = regularize_footprint(footprint)
    vectors = np.diff(np.asarray(regular.exterior.coords), axis=0)
    angles = np.mod(np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])), 180.0)
    assert np.any(np.isclose(angles, 45.0, atol=3.0) | np.isclose(angles, 135.0, atol=3.0))


def test_gable_planes_become_symmetric_with_horizontal_ridge():
    footprint = Polygon([(0, 0), (12, 0), (12, 6), (0, 6)])
    planes = [
        _plane((0.03, 0.70), 6.0, Polygon([(0, 0), (6, 0), (6, 6), (0, 6)]), []),
        _plane((-0.02, -0.75), 10.2, Polygon([(6, 0), (12, 0), (12, 6), (6, 6)]), []),
    ]
    result = regularize_roof_planes(planes, footprint)
    gradients = [-plane.normal[:2] / plane.normal[2] for plane in result]
    assert np.allclose(np.abs(gradients[0]), np.abs(gradients[1]), atol=1e-6)
    assert abs(gradients[0][0]) < 1e-8 and abs(gradients[1][0]) < 1e-8
    assert gradients[0][1] * gradients[1][1] < 0


def test_hip_roof_retains_four_axis_aligned_planes():
    footprint = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
    planes = [
        _plane((0.72, 0.02), 6.0, footprint, []), _plane((-0.70, -0.03), 14.4, footprint, []),
        _plane((0.02, 0.71), 6.0, footprint, []), _plane((-0.03, -0.74), 11.9, footprint, []),
    ]
    result = regularize_roof_planes(planes, footprint)
    assert len(result) == 4
    gradients = np.asarray([-plane.normal[:2] / plane.normal[2] for plane in result])
    assert np.all(np.minimum(np.abs(gradients[:, 0]), np.abs(gradients[:, 1])) < 1e-8)


def test_fidelity_guard_retains_original_roof_model():
    model = reconstruct_building(_gable_cloud(skew=0.35), 91, dtm_sampler=lambda _x, _y: 0.0, config=_config(angle_tol_deg=90, max_rmse_increase_m=0.0))
    assert model.status in {"ok", "lod1_fallback"}
    assert not model.heights["regularization"]["regularized"]
    assert "guard" in model.heights["regularization"]["reason"]


def test_regularized_gable_model_is_watertight():
    model = reconstruct_building(_gable_cloud(), 92, dtm_sampler=lambda _x, _y: 0.0, config=_config())
    assert model.status == "ok"
    assert trimesh.Trimesh(vertices=model.vertices, faces=model.faces, process=False).is_watertight
    assert model.heights["regularization"]["roof_rmse_before"] is not None
