"""Tests for progressive morphological filter, surface rules, and neighbourhood smoothing."""

from __future__ import annotations

import numpy as np

from samcloud.engine.geometry.refine import refine_classification
from samcloud.engine.geometry.surface import apply_surface_rule, neighbourhood_majority
from samcloud.engine.geometry.terrain import build_dtm, height_above_ground


def test_pmf_keeps_hill_and_removes_building():
    rng = np.random.default_rng(2024)
    x_steps = np.arange(0.137, 60.0, 0.4)
    y_steps = np.arange(0.284, 60.0, 0.4)
    xx, yy = np.meshgrid(x_steps, y_steps)
    x = xx.ravel()
    y = yy.ravel()

    # Mound with 30 % slope: 3 m cone, 20 m base diameter
    z = np.maximum(0.0, 3.0 - 0.3 * np.hypot(x - 30.137, y - 40.284))

    noise = rng.uniform(-0.02, 0.02, size=len(x))
    z_terrain = z + noise

    # 10x10x6 m building block and 1.2x1.2x1.5 m bush (low bushes are handled by the roughness rule)
    building_footprint = (x >= 5.137) & (x <= 15.137) & (y >= 5.284) & (y <= 15.284)
    bush_footprint = (x >= 48.137) & (x <= 49.337) & (y >= 10.284) & (y <= 11.484)

    terrain_pts_mask = ~(building_footprint | bush_footprint)
    xyz_terrain = np.column_stack([x[terrain_pts_mask], y[terrain_pts_mask], z_terrain[terrain_pts_mask]])

    b_x, b_y = x[building_footprint], y[building_footprint]
    b_z = z_terrain[building_footprint] + 6.0
    xyz_building = np.column_stack([b_x, b_y, b_z])

    bush_x, bush_y = x[bush_footprint], y[bush_footprint]
    bush_z = z_terrain[bush_footprint] + 1.5
    xyz_bush = np.column_stack([bush_x, bush_y, bush_z])

    xyz_all = np.vstack([xyz_terrain, xyz_building, xyz_bush])
    ground_mask = np.ones(len(xyz_all), dtype=bool)

    pmf_opts = {
        "windows_m": (1, 3, 6, 12, 24, 40),
        "slope": 0.35,
        "initial_threshold_m": 0.15,
        "max_threshold_m": 3.0,
    }
    dtm_pmf, origin_pmf, cell_pmf = build_dtm(
        xyz_all, ground_mask, cell_size=0.5, percentile=5.0, min_points=1,
        opening_window_m=40.0, method="pmf", pmf=pmf_opts,
    )

    hill_terrain_pts = xyz_terrain[np.hypot(xyz_terrain[:, 0] - 30.137, xyz_terrain[:, 1] - 40.284) < 9.0]
    hag_hill_pmf = height_above_ground(hill_terrain_pts, dtm_pmf, origin_pmf, cell_pmf)
    assert abs(np.median(hag_hill_pmf)) < 0.05

    hag_building = height_above_ground(xyz_building, dtm_pmf, origin_pmf, cell_pmf)
    assert np.median(hag_building) > 5.0

    hag_bush = height_above_ground(xyz_bush, dtm_pmf, origin_pmf, cell_pmf)
    assert np.median(hag_bush) > 0.5

    # Old opening method with 40 m window flattens the hill (demonstrating the bug)
    dtm_opening, origin_op, cell_op = build_dtm(
        xyz_all, ground_mask, cell_size=0.5, percentile=5.0, min_points=1,
        opening_window_m=40.0, method="opening",
    )
    hag_hill_opening = height_above_ground(hill_terrain_pts, dtm_opening, origin_op, cell_op)
    assert np.median(hag_hill_opening) > 0.5


def test_surface_rule_lawn_bush_soil():
    rng = np.random.default_rng(999)

    def make_patch(x0: float, y0: float, z0: float, z_noise_amp: float) -> np.ndarray:
        gx = np.arange(x0, x0 + 2.0, 0.1)
        gy = np.arange(y0, y0 + 2.0, 0.1)
        xx, yy = np.meshgrid(gx, gy)
        px = xx.ravel()
        py = yy.ravel()
        pz = z0 + rng.uniform(-z_noise_amp, z_noise_amp, size=len(px))
        return np.column_stack([px, py, pz])

    p_lawn = make_patch(0.137, 0.284, 0.0, 0.002)
    p_bush = make_patch(5.137, 0.284, 0.8, 0.08)
    p_soil = make_patch(10.137, 0.284, 0.5, 0.08)

    n_pts = len(p_lawn)
    xyz = np.vstack([p_lawn, p_bush, p_soil])
    lawn_idx = slice(0, n_pts)
    bush_idx = slice(n_pts, 2 * n_pts)
    soil_idx = slice(2 * n_pts, 3 * n_pts)

    class_ids = np.zeros(len(xyz), dtype=np.uint8)
    class_ids[lawn_idx] = 2  # low vegetation
    class_ids[bush_idx] = 1  # ground
    class_ids[soil_idx] = 2  # low vegetation

    rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    rgb[lawn_idx] = [40, 200, 40]   # green lawn
    rgb[bush_idx] = [30, 160, 40]   # green bush
    rgb[soil_idx] = [140, 90, 50]   # brown soil

    hag = np.zeros(len(xyz), dtype=np.float64)
    hag[lawn_idx] = 0.05
    hag[bush_idx] = 0.8
    hag[soil_idx] = 0.5

    name_to_id = {"ground": 1, "low vegetation": 2, "sidewalk": 3}
    config = {
        "enabled": True,
        "classes": ["ground", "low vegetation", "sidewalk"],
        "cell_size": 0.5,
        "min_points": 10,
        "max_hag": 2.5,
        "rough_vegetation_m": 0.025,
        "smooth_ground_m": 0.025,
        "lawn_max_hag": 0.25,
        "green_index_min": 0.05,
    }

    out, stats = apply_surface_rule(class_ids, xyz, rgb, hag, name_to_id, config)

    assert np.all(out[lawn_idx] == 1)
    assert np.all(out[bush_idx] == 2)
    assert np.all(out[soil_idx] == 1)
    assert stats["to_low_vegetation"] == n_pts
    assert stats["to_ground"] == 2 * n_pts


def test_neighbourhood_majority_removes_island_and_keeps_pole():
    rng = np.random.default_rng(555)

    # Road plane: 2x2 m, step 0.1 m
    gx = np.arange(0.137, 2.137, 0.1)
    gy = np.arange(0.284, 2.284, 0.1)
    xx, yy = np.meshgrid(gx, gy)
    road_x = xx.ravel()
    road_y = yy.ravel()
    road_z = rng.uniform(-0.005, 0.005, size=len(road_x))
    road_pts = np.column_stack([road_x, road_y, road_z])
    road_classes = np.full(len(road_pts), 1, dtype=np.int64)  # class 1: road

    # 6-point "ground" island in the middle of the road plane
    island_mask = (
        (road_x >= 1.0) & (road_x <= 1.25) &
        (road_y >= 1.0) & (road_y <= 1.25)
    )
    island_indices = np.flatnonzero(island_mask)[:6]
    road_classes[island_indices] = 2  # class 2: ground island

    # Vertical utility pole (class 3) in a building-labelled neighbourhood (class 4)
    pole_z = np.linspace(0.1, 2.5, 12)
    pole_pts = np.column_stack([
        np.full(len(pole_z), 10.137),
        np.full(len(pole_z), 10.284),
        pole_z,
    ])
    pole_classes = np.full(len(pole_pts), 3, dtype=np.int64)  # class 3: utility pole

    # Surrounding building points
    b_offsets = [(-0.15, -0.15), (-0.15, 0.15), (0.15, -0.15), (0.15, 0.15)]
    b_pts_list = []
    for dx, dy in b_offsets:
        b_pts_list.append(np.column_stack([
            np.full(len(pole_z), 10.137 + dx),
            np.full(len(pole_z), 10.284 + dy),
            pole_z,
        ]))
    building_pts = np.vstack(b_pts_list)
    building_classes = np.full(len(building_pts), 4, dtype=np.int64)  # class 4: building

    xyz = np.vstack([road_pts, pole_pts, building_pts])
    classes = np.concatenate([road_classes, pole_classes, building_classes])

    pole_start = len(road_pts)
    pole_slice = slice(pole_start, pole_start + len(pole_pts))

    result = neighbourhood_majority(
        xyz, classes, n_classes=5, voxel=0.3, iterations=2,
        own_max_share=0.25, majority_min_share=0.6, protected=(3,),
    )

    # Road island points should be converted to road (class 1)
    assert np.all(result[island_indices] == 1)
    # Utility pole points are protected and unchanged (class 3)
    assert np.all(result[pole_slice] == 3)


def test_refine_summary_reports_new_steps():
    rng = np.random.default_rng(123)
    gx = np.arange(0.137, 5.0, 0.1)
    gy = np.arange(0.284, 5.0, 0.1)
    xx, yy = np.meshgrid(gx, gy)
    x = xx.ravel()
    y = yy.ravel()
    z = rng.uniform(-0.01, 0.01, size=len(x))
    xyz = np.column_stack([x, y, z])
    class_ids = np.ones(len(xyz), dtype=np.uint8)
    rgb = np.full((len(xyz), 3), 120, dtype=np.uint8)

    id_to_name = {1: "ground", 2: "low vegetation", 3: "building"}
    name_to_id = {"ground": 1, "low vegetation": 2, "building": 3}
    config = {
        "terrain": {
            "cell_size": 0.5,
            "percentile": 5,
            "min_points": 2,
            "opening_window_m": 5.0,
            "ground_classes": ["ground"],
            "method": "pmf",
            "pmf": {
                "windows_m": [1, 3, 5],
                "slope": 0.35,
                "initial_threshold_m": 0.15,
                "max_threshold_m": 3.0,
            },
        },
        "features": {"voxel_size": 0.5, "min_points": 2},
        "support": {"cell_size": 1.0, "percentile": 5, "min_points": 2, "window_m": 2},
        "confidence": {"protect_above": 0.85, "dtm_min_confidence": 0.5, "second_best_min": 0.2},
        "neighbors": {"iterations": 1, "voxel_radius": 1, "min_share": 0.05},
        "flat_roof": {},
        "rules": {"ground": {"max_hag": 1.0}},
        "surface": {
            "enabled": True,
            "classes": ["ground", "low vegetation"],
            "cell_size": 0.5,
            "min_points": 2,
            "max_hag": 2.5,
            "rough_vegetation_m": 0.025,
            "smooth_ground_m": 0.025,
            "lawn_max_hag": 0.25,
            "green_index_min": 0.05,
        },
        "smoothing": {
            "enabled": True,
            "voxel_size": 0.3,
            "iterations": 1,
            "own_max_share": 0.25,
            "majority_min_share": 0.6,
            "protected_classes": [],
        },
    }

    refined, summary = refine_classification(
        xyz, class_ids, config, id_to_name, name_to_id, scores=None, rgb=rgb
    )
    assert "surface_rule" in summary
    assert "smoothed_points" in summary
    assert summary["dtm_method"] == "pmf"
    assert isinstance(summary["surface_rule"], dict)
    assert "to_low_vegetation" in summary["surface_rule"]
    assert "to_ground" in summary["surface_rule"]
    assert isinstance(summary["smoothed_points"], int)
