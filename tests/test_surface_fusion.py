"""TSDF surface fusion: one point layer from all depth maps, prior only where it agrees with COLMAP."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

torch = pytest.importorskip("torch")

from samcloud.colmap.plane_fill import write_colmap_array  # noqa: E402
from samcloud.colmap.surface_fusion import (  # noqa: E402
    DEFAULTS,
    consolidate_classified_cloud,
    dedupe_first,
    fit_prior_locally,
    snap_to_planes,
    sparse_point_mask,
)
from test_plane_fill import HOLE, SIZE, WALL_Y, _views  # noqa: E402

BIASES = (-0.015, -0.005, 0.006, 0.014)  # per-view scale error of the prior
SETTINGS = {"max_prior_depth": 6.0, "max_incidence_deg": 89.0, "edge_tolerance": 0.5, "min_neighbors": 0,
            "plane_classes": [], "bounds_percentile": 0.0, "voxel_m": 0.02}


def _true_depth(view) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.mgrid[0:SIZE, 0:SIZE].astype(float) + 0.5
    rays = np.stack(((u - view.cx) / view.fx, (v - view.cy) / view.fy, np.ones_like(u)), axis=-1) @ view.rotation
    t = (WALL_Y - view.center[1]) / rays[..., 1]
    hit = view.center + rays * t[..., None]
    in_hole = (hit[..., 0] > HOLE[0]) & (hit[..., 0] < HOLE[1]) & (hit[..., 2] > HOLE[2]) & (hit[..., 2] < HOLE[3])
    return t.astype(np.float32), in_hole


def _workspace(tmp_path: Path, views, prior_factor) -> tuple[Path, Path, dict]:
    dense = tmp_path / "dense"
    (dense / "stereo" / "depth_maps").mkdir(parents=True)
    priors = {}
    for view, bias in zip(views, BIASES):
        depth, hole = _true_depth(view)
        geometric = np.where(hole, 0.0, depth * (1 + np.random.default_rng(view.index).normal(0, 0.002, depth.shape)))
        write_colmap_array(dense / "stereo" / "depth_maps" / f"{view.name}.geometric.bin", geometric.astype(np.float32))
        priors[view.name] = prior_factor(depth, bias, view).astype(np.float32)
    rng = np.random.default_rng(1)
    seed = np.column_stack((rng.uniform(-2.5, 2.5, 4000), np.full(4000, WALL_Y), rng.uniform(0.0, 2.8, 4000)))
    data = np.zeros(len(seed), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"),
                                      ("blue", "u1"), ("scalar_class", "u1")])
    data["x"], data["y"], data["z"] = seed.T
    ply = dense / "classified.ply"
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(ply))
    return dense, ply, priors


def _read(ply: Path) -> np.ndarray:
    v = PlyData.read(str(ply), mmap=False)["vertex"].data
    return np.column_stack((v["x"], v["y"], v["z"])).astype(float)


def test_extract_gives_one_thin_layer_and_fills_the_hole_with_the_fitted_prior(tmp_path: Path) -> None:
    views = _views()
    dense, ply, priors = _workspace(tmp_path, views, lambda depth, bias, view: depth * (1 + bias))
    report = consolidate_classified_cloud(dense, ply, views, priors.get, SETTINGS)
    xyz = _read(ply)
    assert report["mode"] == "extract" and report["final_points"] == len(xyz) > 1000
    wall = np.abs(xyz[:, 1] - WALL_Y) < 0.2
    assert wall.mean() > 0.97
    assert np.std(xyz[wall, 1]) < 0.006  # one layer, not one copy per camera
    in_hole = wall & (xyz[:, 0] > HOLE[0] + 0.1) & (xyz[:, 0] < HOLE[1] - 0.1) & (xyz[:, 2] > HOLE[2] + 0.1) & (xyz[:, 2] < HOLE[3] - 0.1)
    assert in_hole.sum() > 50  # textureless patch filled from the prior


def test_disagreeing_prior_is_not_used(tmp_path: Path) -> None:
    views = _views()

    def noisy(depth, bias, view):  # 10 % random error: never agrees with COLMAP
        return depth * (1 + np.random.default_rng(view.index + 7).normal(0, 0.1, depth.shape))

    dense, ply, priors = _workspace(tmp_path, views, noisy)
    consolidate_classified_cloud(dense, ply, views, priors.get, SETTINGS)
    xyz = _read(ply)
    deep_hole = (xyz[:, 0] > HOLE[0] + 0.25) & (xyz[:, 0] < HOLE[1] - 0.25) & (xyz[:, 2] > HOLE[2] + 0.25) & (xyz[:, 2] < HOLE[3] - 0.25)
    assert deep_hole.sum() == 0


def test_local_fit_corrects_a_tilted_prior() -> None:
    h = w = 160
    depth = torch.full((h, w), 3.0)
    tilt = 1.0 + torch.linspace(-0.04, 0.04, w)[None, :].expand(h, w)  # +-4 % across the image
    prior = depth * tilt
    geometric = depth.clone()
    geometric[60:100, 60:100] = 0  # gap
    fitted, usable = fit_prior_locally(prior, geometric, DEFAULTS)
    gap = fitted[60:100, 60:100]
    assert bool(usable[60:100, 60:100].all())
    assert float(((gap - 3.0).abs() / 3.0).max()) < 0.012


def test_planes_put_two_ceiling_heights_onto_one() -> None:
    rng = np.random.default_rng(2)
    n = 40000
    xy = rng.uniform(0, 4, (n, 2))
    z = np.where(rng.random(n) < 0.5, 2.400, 2.428) + rng.normal(0, 0.003, n)  # two layers 2.8 cm apart
    points = np.column_stack((xy, z))
    labels = np.full(n, 3, np.uint8)
    out, report = snap_to_planes(points, labels, {"ceiling": 3}, {**DEFAULTS, "plane_min_points": 5000})
    assert report["ceiling"]["snapped_points"] > 0.97 * n
    assert np.std(out[:, 2]) < 0.002


def test_walls_only_snap_to_vertical_planes() -> None:
    rng = np.random.default_rng(3)
    n = 30000
    wall = np.column_stack((rng.uniform(0, 4, n), 2.0 + rng.normal(0, 0.01, n), rng.uniform(0, 2.5, n)))
    floor = np.column_stack((rng.uniform(0, 4, n), rng.uniform(0, 1.5, n), rng.normal(0, 0.003, n)))  # mislabelled, away from the wall
    points = np.vstack((wall, floor))
    labels = np.full(len(points), 2, np.uint8)  # all labelled wall
    out, _ = snap_to_planes(points, labels, {"wall": 2}, {**DEFAULTS, "plane_min_points": 5000})
    assert np.std(out[:n, 1]) < 0.002
    assert np.allclose(out[n:, 2], floor[:, 2])  # the horizontal part is not a wall plane


def test_dedupe_keeps_first_point_per_voxel() -> None:
    points = np.array([[0.001, 0.0, 0.0], [0.004, 0.002, 0.0], [0.5, 0.5, 0.5], [0.503, 0.5, 0.5]])
    keep = np.array([True, True, True, False])
    assert dedupe_first(points, keep, 0.01).tolist() == [0, 2]


def test_sparse_haze_is_removed_but_surface_stays() -> None:
    rng = np.random.default_rng(8)
    gx, gz = np.meshgrid(np.arange(0, 1.0, 0.01), np.arange(0, 1.0, 0.01))
    wall = np.column_stack((gx.ravel() + rng.normal(0, 0.001, gx.size), np.full(gx.size, 2.003), gz.ravel() + 0.0037))
    haze = rng.uniform([0, 1.2, 0], [1, 1.9, 1], (200, 3))
    keep = sparse_point_mask(np.vstack((wall, haze)), {"min_neighbors": 30, "neighbor_cell_m": 0.03, "neighbor_passes": 2})
    assert keep[: len(wall)].mean() > 0.97
    assert keep[len(wall):].mean() < 0.05


def test_ceiling_levels_closer_than_merge_distance_become_one_height() -> None:
    rng = np.random.default_rng(5)
    n = 30000
    xy = rng.uniform(0, 4, (n, 2))
    z = np.select([rng.random(n) < 0.6, rng.random(n) < 0.5], [2.40, 2.52], 2.61) + rng.normal(0, 0.003, n)
    points = np.column_stack((xy, z))
    out, report = snap_to_planes(points, np.full(n, 3, np.uint8), {"ceiling": 3},
                                 {**DEFAULTS, "plane_min_points": 2000, "plane_merge_m": {"ceiling": 0.25}})
    assert report["ceiling"]["levels"] == [pytest.approx(2.40, abs=0.01)]
    assert np.std(out[:, 2]) < 0.003


def test_visibility_removes_points_the_cameras_look_through(tmp_path: Path) -> None:
    from samcloud.colmap.surface_fusion import visibility_mask

    views = _views()
    dense, _, _ = _workspace(tmp_path, views, lambda depth, bias, view: depth)
    rng = np.random.default_rng(4)
    wall = np.column_stack((rng.uniform(-1.0, 1.0, 500), np.full(500, WALL_Y), rng.uniform(1.0, 2.0, 500)))
    wall = wall[~((wall[:, 0] > HOLE[0]) & (wall[:, 0] < HOLE[1]) & (wall[:, 2] > HOLE[2]) & (wall[:, 2] < HOLE[3]))]
    ghost = wall.copy()
    ghost[:, 1] = WALL_Y - 0.43  # a false wall 43 cm in front of the real one
    keep, stats = visibility_mask(np.vstack((wall, ghost)), views, dense / "stereo", DEFAULTS)
    assert keep[: len(wall)].mean() > 0.97
    assert keep[len(wall):].mean() < 0.05
    assert stats["removed_visibility"] >= 0.95 * len(ghost)


def test_slanted_wall_planes_are_rejected_and_same_side_copies_merge() -> None:
    rng = np.random.default_rng(6)
    n = 30000
    main = np.column_stack((rng.uniform(0, 5, n), 3.0 + rng.normal(0, 0.003, n), rng.uniform(0, 2.4, n)))
    copy = np.column_stack((rng.uniform(1, 4, n // 2), 3.0 - 0.12 + rng.normal(0, 0.003, n // 2), rng.uniform(0, 2.4, n // 2)))
    t = rng.uniform(0, 3, n // 2)  # a slanted band (20 deg) that does not exist
    slant = np.column_stack((1 + t * math.cos(math.radians(20)), 1 + t * math.sin(math.radians(20)) + rng.normal(0, 0.003, n // 2),
                             rng.uniform(0, 2.4, n // 2)))
    points = np.vstack((main, copy, slant))
    normals = np.tile([0.0, -1.0, 0.0], (len(points), 1)).astype(np.float32)  # all seen from the room side (y < 3)
    labels = np.full(len(points), 2, np.uint8)
    settings = {**DEFAULTS, "plane_min_points": 5000}
    out, report = snap_to_planes(points, labels, {"wall": 2}, settings, normals=normals)
    assert np.std(out[n:n + n // 2, 1] - 3.0) < 0.005  # the copy now lies on the real wall
    for wall in report["wall"]["walls"]:
        nx, ny = abs(wall["normal"][0]), abs(wall["normal"][1])
        assert min(nx, ny) < math.sin(math.radians(4.5))  # only the building's main directions
    assert np.allclose(out[-(n // 2):], slant)  # the slanted band is not snapped to anything


def test_interior_wall_faces_facing_opposite_ways_stay_apart() -> None:
    rng = np.random.default_rng(9)
    n = 20000
    a = np.column_stack((rng.uniform(0, 4, n), 3.0 + rng.normal(0, 0.002, n), rng.uniform(0, 2.4, n)))
    b = np.column_stack((rng.uniform(0, 4, n), 3.12 + rng.normal(0, 0.002, n), rng.uniform(0, 2.4, n)))
    normals = np.vstack((np.tile([0, -1.0, 0], (n, 1)), np.tile([0, 1.0, 0], (n, 1)))).astype(np.float32)
    out, _ = snap_to_planes(np.vstack((a, b)), np.full(2 * n, 2, np.uint8), {"wall": 2},
                            {**DEFAULTS, "plane_min_points": 5000}, normals=normals)
    assert abs(np.median(out[n:, 1]) - np.median(out[:n, 1]) - 0.12) < 0.01


def test_class_coherence_relabels_stray_wall_points_on_the_sofa() -> None:
    from samcloud.colmap.surface_fusion import class_coherence

    rng = np.random.default_rng(12)
    sofa = rng.uniform([0, 0, 0], [1.0, 0.8, 0.5], (20000, 3))
    labels = np.full(len(sofa), 5, np.uint8)  # furniture
    labels[:600] = 2  # scattered wall labels inside the sofa
    socket_wall = np.column_stack((rng.uniform(3, 5, 20000), np.full(20000, 2.0), rng.uniform(0, 2, 20000)))
    wall_labels = np.full(len(socket_wall), 2, np.uint8)
    on_socket = (np.abs(socket_wall[:, 0] - 4) < 0.05) & (np.abs(socket_wall[:, 2] - 0.3) < 0.05)
    wall_labels[on_socket] = 14  # electrical: small class, not touched
    floating = rng.uniform([8, 8, 1], [8.03, 8.03, 1.03], (5, 3))
    points = np.vstack((sofa, socket_wall, floating))
    lab = np.concatenate((labels, wall_labels, np.full(5, 2, np.uint8)))
    ids = {"wall": 2, "ceiling": 3, "floor": 4, "furniture": 5, "electrical": 14}
    out, keep, stats = class_coherence(points, lab, ids, DEFAULTS)
    assert (out[:600] == 5).mean() > 0.95
    assert (out[600:20000] == 5).all()
    assert (out[20000:40000][on_socket] == 14).all()
    assert not keep[-5:].any() and keep[:-5].all()


def test_points_above_a_single_ceiling_are_removed() -> None:
    from samcloud.colmap.surface_fusion import levels_mask

    points = np.array([[0, 0, 1.0], [0, 0, 2.43], [0, 0, 2.6], [0, 0, -0.1]])
    report = {"ceiling": {"levels": [2.40]}, "floor": {"levels": [0.0]}}
    assert levels_mask(points, report, DEFAULTS).tolist() == [True, True, False, False]
    two_levels = {"ceiling": {"levels": [2.40, 2.9]}, "floor": {"levels": [0.0]}}
    assert levels_mask(points, two_levels, DEFAULTS).tolist() == [True, True, True, False]


def test_a_ceiling_level_with_few_points_does_not_count_as_a_second_storey() -> None:
    from samcloud.colmap.surface_fusion import levels_mask

    points = np.array([[0, 0, 1.0], [0, 0, 1.93], [0, 0, 1.67]])
    report = {"ceiling": {"levels": [1.666, 1.93], "level_points": {"1.666": 540000, "1.930": 7700}}}
    assert levels_mask(points, report, DEFAULTS).tolist() == [True, False, True]
    real = {"ceiling": {"levels": [1.666, 1.93], "level_points": {"1.666": 540000, "1.930": 90000}}}
    assert levels_mask(points, real, DEFAULTS).all()


def test_a_wall_plane_behind_the_real_wall_that_no_camera_sees_is_removed(tmp_path: Path) -> None:
    views = _views()
    dense, ply, priors = _workspace(tmp_path, views, lambda depth, bias, view: depth)
    rng = np.random.default_rng(21)
    n = 12000
    real = np.column_stack((rng.uniform(-1.2, 1.2, n), WALL_Y + rng.normal(0, 0.002, n), rng.uniform(0.2, 0.6, n)))
    ghost = real.copy()
    ghost[:, 1] += 0.31  # same wall again, 31 cm behind: hidden from every camera
    pts = np.vstack((real, ghost))
    data = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"),
                                     ("blue", "u1"), ("scalar_class", "u1")])
    data["x"], data["y"], data["z"] = pts.T
    data["scalar_class"] = 2
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(ply))
    settings = {**SETTINGS, "mode": "snap", "plane_classes": ["wall"], "plane_min_points": 2000, "class_coherence": False,
                "dedupe_m": 0.0}
    report = consolidate_classified_cloud(dense, ply, views, priors.get, settings, class_names=["clutter", "wall"])
    xyz = _read(ply)
    assert report["removed_unseen_walls"]["points"] > 0.9 * n
    assert (np.abs(xyz[:, 1] - WALL_Y) < 0.05).sum() > 0.9 * n
    assert (xyz[:, 1] > WALL_Y + 0.2).sum() < 0.02 * n


def test_points_hidden_behind_the_wall_from_every_camera_are_removed(tmp_path: Path) -> None:
    from samcloud.colmap.surface_fusion import visibility_mask

    views = _views()
    dense, _, priors = _workspace(tmp_path, views, lambda depth, bias, view: depth)
    rng = np.random.default_rng(31)
    wall = np.column_stack((rng.uniform(-1.0, 1.0, 400), np.full(400, WALL_Y), rng.uniform(0.2, 0.6, 400)))
    behind = rng.uniform([-0.8, WALL_Y + 0.2, 0.2], [0.8, WALL_Y + 0.6, 0.45], (300, 3))  # sight lines miss the patch
    settings = {**DEFAULTS, "point_hidden_views": 3}  # only 4 synthetic cameras
    keep, stats, seen = visibility_mask(np.vstack((wall, behind)), views, dense / "stereo", settings, count_seen=True,
                                   prior_reader=priors.get)
    assert keep[:400].mean() > 0.97 and seen[:400].min() >= 2
    assert keep[400:].mean() < 0.03
    assert stats["removed_hidden"] >= 0.97 * 300


def test_haze_filter_on_the_gpu_matches_numpy() -> None:
    rng = np.random.default_rng(17)
    pts = np.vstack((rng.uniform(0, 1, (20000, 3)) * [1, 1, 0.02], rng.uniform(-1, 2, (300, 3))))
    settings = {"min_neighbors": 30, "neighbor_cell_m": 0.03, "neighbor_passes": 2}
    assert (sparse_point_mask(pts, settings) == sparse_point_mask(pts, settings, device="cpu")).all()
