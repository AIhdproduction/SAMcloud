"""Fast synthetic tests for geometry-based classification refinement."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from samcloud.engine.classify import extract_top2_confidences
from samcloud.engine.geometry.features import compute_voxel_features
from samcloud.engine.geometry.refine import (
    apply_flat_roof_rule,
    class_name_mapping,
    evaluate_rules,
    neighbour_reassign,
)
from samcloud.engine.geometry.rules import passes_for_class
from samcloud.engine.geometry.terrain import build_dtm, height_above_ground, morphological_opening
from samcloud.app.processing import ProcessingManager


def _grid(x0, x1, y0, y1, step=1.0):
    x, y = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.column_stack((x.ravel(), y.ravel()))


def test_dtm_follows_slope_and_removes_flat_roof_plateau():
    xy = _grid(0, 100, 0, 100)
    terrain_z = 0.08 * xy[:, 0] + 0.03 * xy[:, 1]
    terrain = np.column_stack((xy, terrain_z))
    roof_xy = _grid(40, 60, 40, 60)
    roof = np.column_stack((roof_xy, 0.08 * roof_xy[:, 0] + 0.03 * roof_xy[:, 1] + 6.0))
    # Repeated observations meet min_points and one low outlier remains below the 5% rank.
    roof_footprint = ((xy[:, 0] >= 40) & (xy[:, 0] < 60) & (xy[:, 1] >= 40) & (xy[:, 1] < 60))
    terrain_only = terrain[~roof_footprint]
    xyz = np.vstack((np.repeat(terrain_only, 6, axis=0), np.repeat(roof, 6, axis=0), [[5, 5, -100]]))
    mask = np.ones(len(xyz), dtype=bool)
    dtm, origin, cell_size = build_dtm(xyz, mask, 1.0, 5, 5, 40.0)
    hag = height_above_ground(np.vstack((terrain_only, roof)), dtm, origin, cell_size)
    assert abs(np.median(hag[:len(terrain_only)])) < 0.5
    assert abs(np.median(hag[len(terrain_only):]) - 6.0) < 0.35
    assert abs(dtm[5, 5] - (0.08 * 5 + 0.03 * 5)) < 0.4


def test_morphological_opening_removes_edge_plateau_and_preserves_sloped_plane():
    """Interior slope tolerance is 0.01 m; truncated windows bias the outer edge by 2.20 m."""
    y, x = np.mgrid[0:80, 0:80]
    slope = 0.08 * x + 0.03 * y
    flat_edge_roof = np.zeros_like(slope)
    flat_edge_roof[25:45, :20] = 6.0
    assert np.max(morphological_opening(flat_edge_roof, 40)[25:45, :20]) < 0.01

    opened_slope = morphological_opening(slope, 40)
    assert np.max(np.abs(opened_slope[20:-20, 20:-20] - slope[20:-20, 20:-20])) < 0.01
    assert np.isclose(np.max(np.abs(opened_slope - slope)), 2.20)


def test_voxel_features_recognise_horizontal_vertical_and_sparse_voxels():
    horizontal_xy = _grid(0, 0.5, 0, 0.5, 0.1)
    horizontal = np.column_stack((horizontal_xy, np.zeros(len(horizontal_xy))))
    vertical_yz = _grid(1, 1.5, 0, 0.5, 0.1)
    vertical = np.column_stack((np.ones(len(vertical_yz)), vertical_yz[:, 0], vertical_yz[:, 1]))
    sparse = np.array([[3.0, 3.0, 3.0]])
    features, _, _ = compute_voxel_features(np.vstack((horizontal, vertical, sparse)).astype(np.float32), 1.0, 6)
    assert np.nanmedian(features["normal_z"][:len(horizontal)]) > 0.99
    assert np.nanmedian(features["normal_z"][len(horizontal):-1]) < 0.01
    assert np.isnan(features["normal_z"][-1])


def test_flat_roof_rule_converts_only_elevated_planar_source_points():
    classes = np.array([1, 1, 2], dtype=np.uint8)
    features = {
        "hag": np.array([6.0, 0.1, 6.0]),
        "normal_z": np.array([0.99, 0.99, 0.99]),
        "planarity": np.array([0.9, 0.9, 0.9]),
        "roughness": np.zeros(3),
    }
    names = {"ground": 1, "building": 2}
    result, count = apply_flat_roof_rule(classes, features, names, {
        "flat_roof": {"source_classes": ["ground"], "min_hag": 2.5, "min_normal_z": 0.9, "min_planarity": 0.3}
    })
    assert count == 1
    assert result.tolist() == [2, 1, 2]


def test_flat_roof_rule_requires_finite_geometric_evidence():
    classes = np.array([1, 1, 1], dtype=np.uint8)
    features = {
        "hag": np.array([6.0, np.nan, 6.0]),
        "normal_z": np.array([np.nan, 0.99, 0.99]),
        "planarity": np.array([0.9, 0.9, np.nan]),
        "roughness": np.zeros(3),
    }
    result, count = apply_flat_roof_rule(classes, features, {"ground": 1, "building": 2}, {
        "flat_roof": {"source_classes": ["ground"], "min_hag": 2.5, "min_normal_z": 0.9, "min_planarity": 0.3}
    })
    assert count == 0
    assert result.tolist() == [1, 1, 1]


def test_confidence_extraction_uses_mean_probabilities_and_zero_views():
    scores = np.array([[1.6, 0.6, 0.2], [0.0, 0.0, 0.0], [0.1, 0.8, 0.4]], dtype=np.float32)
    views = np.array([2, 0, 1], dtype=np.uint16)
    labels = np.array([1, 0, 2], dtype=np.uint8)
    first, first_conf, second, second_conf = extract_top2_confidences(scores, views, labels, chunk_size=2)
    assert first.tolist() == [1, 0, 2]
    assert np.allclose(first_conf, [0.8, 0.0, 0.8], atol=0.001)
    assert second.tolist() == [2, 0, 3]
    assert np.allclose(second_conf, [0.3, 0.0, 0.4], atol=0.001)


def test_flat_roof_confidence_respects_ambiguous_band_but_not_strong_zone():
    classes = np.array([1, 1, 1], dtype=np.uint8)
    features = {
        "hag": np.array([5.5, 3.0, 3.0]), "normal_z": np.full(3, 0.99),
        "planarity": np.full(3, 0.9), "confidence": np.array([0.95, 0.9, 0.3]),
    }
    result, count = apply_flat_roof_rule(classes, features, {"ground": 1, "building": 2}, {
        "flat_roof": {"source_classes": ["ground"], "min_hag": 2.5, "strong_min_hag": 5.0,
                      "ambiguous_max_confidence": 0.6, "min_normal_z": 0.9, "min_planarity": 0.3}
    })
    assert count == 2
    assert result.tolist() == [2, 1, 2]


def test_confident_geometry_failure_is_protected_during_refinement():
    xyz = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [0.2, 0.2, 0.0]] * 3)
    classes = np.ones(len(xyz), dtype=np.uint8)
    scores = {"top1_class": classes, "top1_conf": np.full(len(xyz), 0.95),
              "top2_class": np.zeros(len(xyz), dtype=np.uint8), "top2_conf": np.zeros(len(xyz)),
              "view_count": np.ones(len(xyz), dtype=np.uint16)}
    config = {"terrain": {"cell_size": 1.0, "percentile": 5, "min_points": 3,
                            "opening_window_m": 1, "ground_classes": ["ground"]},
              "features": {"voxel_size": 1.0, "min_points": 3}, "support": {"cell_size": 1.0, "percentile": 5, "min_points": 3, "window_m": 1},
              "confidence": {"protect_above": 0.85, "dtm_min_confidence": 0.5, "second_best_min": 0.2},
              "neighbors": {"iterations": 1}, "flat_roof": {}, "rules": {"ground": {"max_hag": -1.0}}}
    from samcloud.engine.geometry.refine import refine_classification
    refined, _ = refine_classification(xyz, classes, config, {1: "ground"}, {"ground": 1}, scores)
    assert refined.tolist() == classes.tolist()


def test_reassignment_prefers_second_best_then_neighbours_then_unknown():
    classes = np.array([1, 1, 2, 1], dtype=np.uint8)
    features = {"hag": np.array([3.0, 3.0, 3.0, 3.0]), "normal_z": np.ones(4),
                "planarity": np.ones(4), "roughness": np.zeros(4), "hls": np.full(4, np.nan)}
    voxels = np.array([[0, 0, 0], [2, 0, 0], [2, 0, 0], [20, 0, 0]], dtype=np.int64)
    result, _, unknown, detail = neighbour_reassign(
        classes, np.array([True, True, False, True]), np.array([False, False, True, False]), features, voxels,
        {1: "ground", 2: "tree"}, {"ground": {"max_hag": 0.5}, "tree": {"min_hag": 2.0}},
        {"voxel_radius": 0, "min_share": 0.05}, np.array([2, 0, 0, 0], dtype=np.uint8),
        np.array([0.5, 0.0, 0.0, 0.0]), {"second_best_min": 0.2}, return_details=True,
    )
    assert result.tolist() == [2, 2, 2, 0]
    assert detail["second_best"] == 1
    assert detail["neighbours"] == 1
    assert unknown == 1


def test_low_vegetation_uses_local_support_and_tall_terrain_can_become_tree():
    roof_features = {"hag": np.array([5.0]), "hls": np.array([0.2])}
    assert passes_for_class("low vegetation", roof_features, {"low vegetation": {
        "max_hag": 2.0, "max_height_above_support": 2.0
    }}).item()
    terrain_features = {"hag": np.array([4.0]), "hls": np.array([4.0])}
    assert not passes_for_class("low vegetation", terrain_features, {"low vegetation": {
        "max_hag": 2.0, "max_height_above_support": 2.0
    }}).item()


def test_fence_uses_local_support_for_roof_railings():
    config = {"fence": {"max_hag": 3.0, "max_height_above_support": 3.0}}
    roof_features = {"hag": np.array([12.0]), "hls": np.array([1.2])}
    tall_features = {"hag": np.array([12.0]), "hls": np.array([4.0])}
    assert passes_for_class("fence", roof_features, config).item()
    assert not passes_for_class("fence", tall_features, config).item()


def test_curb_requires_adjacent_road_or_sidewalk_and_power_line_rejects_planar_patch():
    curb_cfg = {"curb": {"max_hag": 0.6, "min_adjacent_points": 3}}
    features = {"hag": np.array([0.3, 0.3]), "adjacent_road_sidewalk": np.array([3, 0])}
    assert passes_for_class("curb", features, curb_cfg).tolist() == [True, False]
    line_cfg = {"overhead power line": {"min_hag": 3.0, "max_planarity": 0.4}}
    line_features = {"hag": np.array([4.0, 4.0, 4.0]), "planarity": np.array([0.9, 0.1, np.nan])}
    assert passes_for_class("overhead power line", line_features, line_cfg).tolist() == [False, True, True]


def test_curb_adjacency_uses_only_road_and_sidewalk_points_that_pass_rules():
    classes = np.array([1, 2, 2, 2, 1], dtype=np.uint8)
    features = {"hag": np.full(5, 0.3), "normal_z": np.ones(5), "planarity": np.ones(5),
                "roughness": np.zeros(5), "hls": np.full(5, np.nan)}
    voxels = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0]], dtype=np.int64)
    names = {1: "curb", 2: "road"}
    config = {"curb": {"max_hag": 0.6, "adjacent_classes": ["road"], "min_adjacent_points": 3,
                       "adjacent_voxel_radius": 2},
              "road": {"max_hag": 0.5, "min_normal_z": 0.85, "max_roughness": 0.15}}
    passed = evaluate_rules(classes, features, names, config, voxels, {"curb": 1, "road": 2})
    assert passed.tolist() == [True, True, True, True, False]


def test_reassignment_to_curb_requires_lazy_local_road_adjacency():
    classes = np.array([1, 2, 2, 2, 1], dtype=np.uint8)
    features = {"hag": np.full(5, 0.3), "normal_z": np.ones(5), "planarity": np.ones(5),
                "roughness": np.zeros(5), "hls": np.full(5, np.nan)}
    voxels = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0]], dtype=np.int64)
    names = {1: "ground", 2: "road", 3: "curb"}
    config = {"ground": {"max_hag": 0.1},
              "road": {"max_hag": 0.5, "min_normal_z": 0.85, "max_roughness": 0.15},
              "curb": {"max_hag": 0.6, "adjacent_classes": ["road"], "min_adjacent_points": 3,
                       "adjacent_voxel_radius": 2}}
    refined, _, unknown = neighbour_reassign(
        classes, np.array([True, False, False, False, True]),
        np.array([False, True, True, True, False]), features, voxels, names, config,
        {"voxel_radius": 0, "min_share": 0.05}, np.array([3, 0, 0, 0, 3], dtype=np.uint8),
        np.array([0.5, 0.0, 0.0, 0.0, 0.5]), {"second_best_min": 0.2}, batch_size=2,
    )
    assert refined.tolist() == [3, 2, 2, 2, 0]
    assert unknown == 1


def test_low_confidence_ground_does_not_shape_dtm():
    xy = _grid(0, 10, 0, 10)
    ground = np.column_stack((xy, np.zeros(len(xy))))
    roof_xy = _grid(3, 7, 3, 7)
    roof = np.column_stack((roof_xy, np.full(len(roof_xy), 6.0)))
    xyz = np.vstack((np.repeat(ground, 4, axis=0), np.repeat(roof, 4, axis=0)))
    confidence = np.r_[np.full(len(ground) * 4, 0.9), np.full(len(roof) * 4, 0.1)]
    filtered, origin, cell_size = build_dtm(xyz, confidence >= 0.5, 1.0, 5, 3, 3.0)
    hag = height_above_ground(roof, filtered, origin, cell_size)
    assert np.median(hag) > 5.0


def test_neighbour_control_reassigns_and_marks_isolated_point_unknown():
    class_ids = np.array([1, 2, 2, 2, 1], dtype=np.uint8)  # road, low vegetation, and isolated road
    features = {
        "hag": np.array([1.0, 1.0, 1.0, 1.0, 4.0]),
        "normal_z": np.ones(5), "planarity": np.ones(5), "roughness": np.zeros(5),
    }
    names = {1: "road", 2: "low vegetation", 3: "utility pole"}
    config = {"road": {"max_hag": 0.5}, "low vegetation": {"max_hag": 1.5}}
    passed = evaluate_rules(class_ids, features, names, config)
    voxels = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 10, 10]], dtype=np.int64)
    refined, targets, unknown = neighbour_reassign(
        class_ids, ~passed, passed, features, voxels, names, config,
        {"voxel_radius": 1, "min_share": 0.05}, batch_size=10,
    )
    assert refined.tolist() == [2, 2, 2, 2, 0]
    assert targets[2] == 1
    assert unknown == 1


def test_class_mapping_does_not_depend_on_id_order_and_unruled_classes_pass():
    class_data = json.dumps({"outdoor": ["building, house", "ground, soil", "utility pole, pole"]})
    with patch.object(Path, "read_text", return_value=class_data):
        id_to_name, name_to_id = class_name_mapping(Path("classes.json"), "outdoor")
    assert name_to_id == {"building": 1, "ground": 2, "utility pole": 3}
    features = {"hag": np.array([99.0]), "normal_z": np.array([0.0]), "planarity": np.array([0.0]), "roughness": np.array([0.0])}
    assert evaluate_rules(np.array([3], dtype=np.uint8), features, id_to_name, {}).item()


def test_pipeline_command_forwards_geometry_refinement_setting():
    manager = object.__new__(ProcessingManager)
    command = manager._pipeline_command(Path("D:/demo"), {
        "settings": {
            "workflow": {"use_gpu": False},
            "classification": {"class_set": "outdoor", "geometry_refinement": True},
            "coordinate_systems": {"working_crs": "LOCAL", "export_crs": "EPSG:2056"},
        },
        "acquisition": {"type": "camera", "sources": {"image_directory": "D:/images"}},
        "paths": {"image_directory": "D:/images"},
    }, "none")
    position = command.index("--geometry-refinement")
    assert command[position + 1] == "1"


def test_pipeline_command_enables_geometry_refinement_by_default():
    manager = object.__new__(ProcessingManager)
    command = manager._pipeline_command(Path("D:/demo"), {
        "settings": {
            "workflow": {"use_gpu": False},
            "classification": {"class_set": "outdoor"},
            "coordinate_systems": {"working_crs": "LOCAL", "export_crs": "EPSG:2056"},
        },
        "acquisition": {"type": "camera", "sources": {"image_directory": "D:/images"}},
        "paths": {"image_directory": "D:/images"},
    }, "none")
    position = command.index("--geometry-refinement")
    assert command[position + 1] == "1"
