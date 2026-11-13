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


# ----------------------------------------------------------------------------- hidden_filter (points no camera can see)


def _scaled(views, scale: float):
    from samcloud.colmap.plane_fill import View

    return [View(v.index, v.name, v.rotation, v.translation * scale, v.fx, v.fy, v.cx, v.cy, v.width, v.height) for v in views]


def _rays(view):
    v, u = np.mgrid[0:SIZE, 0:SIZE].astype(float) + 0.5
    return np.stack(((u - view.cx) / view.fx, (v - view.cy) / view.fy, np.ones_like(u)), axis=-1) @ view.rotation


def _write_maps(tmp_path: Path, views, depth_of) -> Path:
    dense = tmp_path / "dense"
    (dense / "stereo" / "depth_maps").mkdir(parents=True)
    for view in views:
        depth = depth_of(view) * (1 + np.random.default_rng(view.index + 50).normal(0, 0.002, (SIZE, SIZE)))
        write_colmap_array(dense / "stereo" / "depth_maps" / f"{view.name}.geometric.bin", depth.astype(np.float32))
    return dense


def _hidden_settings(scale: float) -> dict:
    # the older "seen < 2 and hidden" rule is switched off so only the new rule can remove anything
    return {**DEFAULTS, "units_per_metre": scale, "point_hidden_views": 999,
            "hidden_filter": {**DEFAULTS["hidden_filter"], "enabled": True, "min_closed_views": 2}}


@pytest.mark.parametrize("scale", [1.0, 0.8157])
def test_floor_behind_wall_removed(tmp_path: Path, scale: float) -> None:
    from samcloud.colmap.surface_fusion import visibility_mask

    views = _scaled(_views(), scale)
    wall_y = WALL_Y * scale

    def depth_of(view):  # every camera measures the wall plane everywhere
        return ((wall_y - view.center[1]) / _rays(view)[..., 1]).astype(np.float32)

    dense = _write_maps(tmp_path, views, depth_of)
    rng = np.random.default_rng(41)
    wall = np.column_stack((rng.uniform(-1.0, 1.0, 500) * scale, wall_y + rng.normal(0, 0.002, 500),
                            rng.uniform(0.2, 0.8, 500) * scale))
    floor_behind = np.column_stack((rng.uniform(-0.9, 0.9, 400) * scale, wall_y + rng.uniform(0.25, 1.2, 400) * scale,
                                    rng.normal(0.0, 0.003, 400)))  # beyond the wall, seen by no camera
    evidence: dict = {}
    keep, stats, _ = visibility_mask(np.vstack((wall, floor_behind)), views, dense / "stereo", _hidden_settings(scale),
                                     count_seen=True, evidence=evidence)
    assert keep[:500].mean() > 0.97
    assert keep[500:].mean() < 0.03
    assert stats["removed_unconfirmed"] >= 0.97 * 400 and stats["removed_hidden"] == 0
    assert (evidence["window_closed"][500:] >= 2).mean() > 0.97  # the wall is the COLMAP surface in front, in 2+ views


def test_textureless_ceiling_without_evidence_kept(tmp_path: Path) -> None:
    from samcloud.colmap.surface_fusion import visibility_mask

    views = _scaled(_views(), 0.8157)

    def depth_of(view):  # no COLMAP depth in the upper half of every image: a textureless ceiling
        depth = ((WALL_Y * 0.8157 - view.center[1]) / _rays(view)[..., 1]).astype(np.float32)
        depth[: int(SIZE * 0.45)] = 0.0
        return depth

    dense = _write_maps(tmp_path, views, depth_of)
    rng = np.random.default_rng(43)
    ceiling = np.column_stack((rng.uniform(-0.8, 0.8, 600), rng.uniform(1.8, 2.6, 600), 2.2 + rng.normal(0, 0.003, 600))) * 0.8157
    evidence: dict = {}
    keep, stats, _ = visibility_mask(ceiling, views, dense / "stereo", _hidden_settings(0.8157), count_seen=True, evidence=evidence)
    no_evidence = evidence["window_closed"] + evidence["window_open"] + evidence["support"] + evidence["through"] == 0
    assert no_evidence.mean() > 0.97
    assert keep.mean() > 0.99 and stats["removed_unconfirmed"] == 0


def test_room_seen_through_open_door_kept(tmp_path: Path) -> None:
    from samcloud.colmap.surface_fusion import visibility_mask

    views = _views()
    back_y = WALL_Y + 1.8
    hole = np.asarray(HOLE)

    def depth_of(view):  # wall with a door opening; through it the cameras measure the back wall of the next room
        rays = _rays(view)
        t_wall = (WALL_Y - view.center[1]) / rays[..., 1]
        hit = view.center + rays * t_wall[..., None]
        in_hole = (hit[..., 0] > hole[0]) & (hit[..., 0] < hole[1]) & (hit[..., 2] > hole[2]) & (hit[..., 2] < hole[3])
        return np.where(in_hole, (back_y - view.center[1]) / rays[..., 1], t_wall).astype(np.float32)

    dense = _write_maps(tmp_path, views, depth_of)
    rng = np.random.default_rng(47)
    room = np.column_stack((rng.uniform(-3.0, 3.0, 6000), back_y + rng.normal(0, 0.003, 6000), rng.uniform(-0.3, 2.9, 6000)))
    # a point is directly visible when at least two cameras look at it through the opening
    through = np.zeros(len(room), dtype=int)
    for view in views:
        ray = room - view.center
        crossing = (WALL_Y - view.center[1]) / ray[:, 1]
        at_wall = view.center + ray * crossing[:, None]
        through += ((at_wall[:, 0] > hole[0] + 0.05) & (at_wall[:, 0] < hole[1] - 0.05)
                    & (at_wall[:, 2] > hole[2] + 0.05) & (at_wall[:, 2] < hole[3] - 0.05)).astype(int)
    visible, shut = room[through >= 2], room[through == 0]
    assert len(visible) > 150 and len(shut) > 150
    keep, stats, _ = visibility_mask(np.vstack((visible, shut)), views, dense / "stereo", _hidden_settings(1.0), count_seen=True)
    assert keep[: len(visible)].mean() > 0.97  # directly visible through the door: stays
    assert keep[len(visible):].mean() < 0.40  # the same room behind the closed part of the wall: gone


# ----------------------------------------------------------------------------- raymarch_filter (every supporting view looks through a wall)


def _evidence_for(points, views, dense, settings):
    from samcloud.colmap.surface_fusion import visibility_mask

    evidence: dict = {}
    visibility_mask(points, views, dense / "stereo", settings, count_seen=True, evidence=evidence)
    return evidence


def test_raymarch_removes_floor_behind_wall_seen_by_grazing_view(tmp_path: Path) -> None:
    from samcloud.colmap.surface_fusion import raymarch_mask

    scale = 0.8157
    views = _scaled(_views(), scale)
    wall_y = WALL_Y * scale

    def depth_of(view):  # COLMAP "measures" the floor plane z = 0 everywhere, also where the wall hides it
        rays = _rays(view)
        t = np.where(rays[..., 2] < -1e-3, (0.0 - view.center[2]) / rays[..., 2], 0.0)
        return np.where((t > 0) & (t < 12), t, 0.0).astype(np.float32)

    dense = _write_maps(tmp_path, views, depth_of)
    rng = np.random.default_rng(61)
    n = 60_000
    wall = np.column_stack((rng.uniform(-3.0, 3.0, n) * scale, wall_y + rng.normal(0, 0.003, n), rng.uniform(-0.05, 2.9, n) * scale))
    floor_front = np.column_stack((rng.uniform(-0.8, 0.8, 300) * scale, rng.uniform(1.5, 3.4, 300) * scale, rng.normal(0, 0.003, 300)))
    floor_behind = np.column_stack((rng.uniform(-0.8, 0.8, 300) * scale, wall_y + rng.uniform(0.3, 1.4, 300) * scale,
                                    rng.normal(0, 0.003, 300)))
    points = np.vstack((wall, floor_front, floor_behind))
    labels = np.concatenate((np.full(n, 1), np.zeros(600))).astype(np.uint8)
    settings = {**DEFAULTS, "units_per_metre": scale}
    evidence = _evidence_for(points, views, dense, settings)
    assert (evidence["support"][n + 300:] >= 3).mean() > 0.9  # COLMAP supports the hidden floor in several views
    remove, stats = raymarch_mask(points, labels, {"wall": 1}, views, dense / "stereo", settings, evidence)
    assert remove[n + 300:].mean() > 0.9  # ... but every one of those views looks through the wall
    assert remove[n:n + 300].mean() < 0.05  # the floor in front of the wall stays
    assert remove[:n].mean() < 0.02


def test_raymarch_keeps_room_through_door(tmp_path: Path) -> None:
    from samcloud.colmap.surface_fusion import raymarch_mask

    views = _views()
    back_y = WALL_Y + 1.8
    hole = np.asarray(HOLE)

    def depth_of(view):
        rays = _rays(view)
        t_wall = (WALL_Y - view.center[1]) / rays[..., 1]
        hit = view.center + rays * t_wall[..., None]
        in_hole = (hit[..., 0] > hole[0]) & (hit[..., 0] < hole[1]) & (hit[..., 2] > hole[2]) & (hit[..., 2] < hole[3])
        return np.where(in_hole, (back_y - view.center[1]) / rays[..., 1], t_wall).astype(np.float32)

    dense = _write_maps(tmp_path, views, depth_of)
    rng = np.random.default_rng(63)
    x, z = rng.uniform(-3.0, 3.0, 70_000), rng.uniform(-0.05, 2.9, 70_000)
    closed_wall = np.column_stack((x, WALL_Y + rng.normal(0, 0.003, x.shape), z))
    in_opening = (x > hole[0]) & (x < hole[1]) & (z > hole[2]) & (z < hole[3])
    wall_with_door = closed_wall[~in_opening]
    room = np.column_stack((rng.uniform(-0.6, 0.7, 3000), back_y + rng.normal(0, 0.003, 3000), rng.uniform(0.8, 1.7, 3000)))
    through = np.zeros(len(room), dtype=int)
    for view in views:
        ray = room - view.center
        at_wall = view.center + ray * ((WALL_Y - view.center[1]) / ray[:, 1])[:, None]
        through += ((at_wall[:, 0] > hole[0] + 0.05) & (at_wall[:, 0] < hole[1] - 0.05)
                    & (at_wall[:, 2] > hole[2] + 0.05) & (at_wall[:, 2] < hole[3] - 0.05)).astype(int)
    room = room[through >= 2]
    assert len(room) > 100
    settings = {**DEFAULTS}
    for wall, expect_removed in ((wall_with_door, False), (closed_wall, True)):  # open door vs. a wall without the opening
        points = np.vstack((wall, room))
        labels = np.concatenate((np.full(len(wall), 1), np.zeros(len(room)))).astype(np.uint8)
        evidence = _evidence_for(points, views, dense, settings)
        remove, _ = raymarch_mask(points, labels, {"wall": 1}, views, dense / "stereo", settings, evidence)
        rate = remove[len(wall):].mean()
        assert (rate > 0.9) if expect_removed else (rate < 0.03)


# ----------------------------------------------------------------------------- tall furniture as occluder (round 3)

FLOOR_ID, FURNITURE_ID, WALL_ID = 1, 2, 3
CLASS_IDS = {"floor": FLOOR_ID, "furniture": FURNITURE_ID, "wall": WALL_ID}


def _floor_plane_maps(tmp_path: Path, views) -> Path:
    def depth_of(view):  # COLMAP "measures" the floor plane z = 0 everywhere, also where furniture hides it
        rays = _rays(view)
        t = np.where(rays[..., 2] < -1e-3, (0.0 - view.center[2]) / rays[..., 2], 0.0)
        return np.where((t > 0) & (t < 12), t, 0.0).astype(np.float32)

    return _write_maps(tmp_path, views, depth_of)


def _furniture_block(rng: np.random.Generator, scale: float, front_y: float, depth: float, height_m: float, count: int,
                     half_width: float = 1.2) -> np.ndarray:
    """Front face, top and both sides of a box standing on the floor (surface points with a few mm of noise)."""
    n_front, n_top, n_side = int(count * 0.7), int(count * 0.2), int(count * 0.05)
    top = height_m * scale
    front = np.column_stack((rng.uniform(-half_width, half_width, n_front) * scale, front_y * scale + rng.normal(0, 0.003, n_front),
                             rng.uniform(0.0, top, n_front)))
    roof = np.column_stack((rng.uniform(-half_width, half_width, n_top) * scale, rng.uniform(front_y, front_y + depth, n_top) * scale,
                            top + rng.normal(0, 0.003, n_top)))
    sides = [np.column_stack((np.full(n_side, sign * half_width * scale) + rng.normal(0, 0.003, n_side),
                              rng.uniform(front_y, front_y + depth, n_side) * scale, rng.uniform(0.0, top, n_side)))
             for sign in (-1.0, 1.0)]
    return np.vstack((front, roof, *sides))


def _furniture_scene(tmp_path: Path, scale: float, height_m: float, front_y: float, behind: tuple[float, float], seed: int):
    views = _scaled(_views(), scale)
    dense = _floor_plane_maps(tmp_path, views)
    rng = np.random.default_rng(seed)
    block = _furniture_block(rng, scale, front_y, 0.5, height_m, 24_000)
    floor_x = rng.uniform(-3.0, 3.0, 4000) * scale
    floor_front = np.column_stack((floor_x, rng.uniform(0.4, 3.9, 4000) * scale, rng.normal(0, 0.003, 4000)))
    floor_front = floor_front[(floor_front[:, 1] < front_y * scale) | (np.abs(floor_front[:, 0]) > 1.3 * scale)]
    behind_pts = np.column_stack((rng.uniform(-0.5, 0.5, 500) * scale, rng.uniform(*behind, 500) * scale, rng.normal(0, 0.003, 500)))
    points = np.vstack((block, floor_front, behind_pts))
    labels = np.concatenate((np.full(len(block), FURNITURE_ID), np.full(len(floor_front), FLOOR_ID), np.zeros(len(behind_pts)))).astype(np.uint8)
    settings = {**DEFAULTS, "units_per_metre": scale}
    evidence = _evidence_for(points, views, dense, settings)
    return views, dense, points, labels, settings, evidence, (len(block), len(floor_front), len(behind_pts))


@pytest.mark.parametrize("scale", [1.0, 0.8157])
def test_raymarch_tall_furniture_occludes_points_behind(tmp_path: Path, scale: float) -> None:
    from samcloud.colmap.surface_fusion import raymarch_mask

    views, dense, points, labels, settings, evidence, (n_block, n_floor, n_behind) = _furniture_scene(
        tmp_path, scale, height_m=2.0, front_y=2.0, behind=(2.6, 3.5), seed=71)
    behind = slice(n_block + n_floor, None)
    assert (evidence["support"][behind] >= 3).mean() > 0.9  # COLMAP supports the hidden points through the wardrobe
    remove, stats = raymarch_mask(points, labels, CLASS_IDS, views, dense / "stereo", settings, evidence)
    assert stats["raymarch_tall_furniture_points"] > 0.75 * n_block  # front and sides are the vertical body; the roof is not
    assert remove[behind].mean() > 0.9  # behind the 2 m wardrobe nobody could see them
    assert remove[:n_block].mean() < 0.01  # the wardrobe itself stays (front, top and sides)
    assert remove[n_block:n_block + n_floor].mean() < 0.02  # real floor in front stays
    off, _ = raymarch_mask(points, labels, CLASS_IDS, views, dense / "stereo",
                           {**settings, "raymarch_filter": {"tall_furniture_enabled": False}}, evidence)
    assert off[behind].mean() < 0.1  # without the rule the furniture is permeable (round 2 behaviour)


@pytest.mark.parametrize("scale", [1.0, 0.8157])
def test_raymarch_low_furniture_does_not_occlude(tmp_path: Path, scale: float) -> None:
    from samcloud.colmap.surface_fusion import raymarch_mask

    # a 0.75 m table close to the cameras: floor far behind is visible over it, floor right behind it is outside the
    # line of sight, but low furniture must stay permeable either way
    views, dense, points, labels, settings, evidence, (n_block, n_floor, n_behind) = _furniture_scene(
        tmp_path, scale, height_m=0.75, front_y=0.7, behind=(2.2, 3.9), seed=73)
    behind = slice(n_block + n_floor, None)
    assert (evidence["support"][behind] >= 3).mean() > 0.9
    remove, stats = raymarch_mask(points, labels, CLASS_IDS, views, dense / "stereo", settings, evidence)
    assert stats["raymarch_tall_furniture_points"] == 0
    assert remove[behind].mean() < 0.05
    assert remove[:n_block].mean() < 0.01


def test_tall_furniture_ignores_floating_cabinet_and_thin_pole() -> None:
    from samcloud.colmap.surface_fusion import tall_furniture_members

    scale = 0.8157
    rng = np.random.default_rng(79)
    floor = np.column_stack((rng.uniform(-3, 3, 4000) * scale, rng.uniform(0, 4, 4000) * scale, rng.normal(0, 0.003, 4000)))
    # upper kitchen cabinet: 1.4 m wide, hangs between 1.3 m and 2.1 m above the floor, nothing below it
    upper = np.column_stack((rng.uniform(-0.7, 0.7, 6000) * scale, 3.0 * scale + rng.normal(0, 0.003, 6000), rng.uniform(1.3, 2.1, 6000) * scale))
    pole = np.column_stack((1.9 * scale + rng.normal(0, 0.01, 800), 1.0 * scale + rng.normal(0, 0.01, 800), rng.uniform(0.0, 1.8, 800) * scale))
    sparse = np.column_stack((rng.uniform(-1.0, 1.0, 120) * scale, 0.5 * scale + rng.normal(0, 0.003, 120), rng.uniform(0.0, 1.9, 120) * scale))
    wardrobe = np.column_stack((rng.uniform(-0.9, 0.9, 9000) * scale + 0.013, 2.0 * scale + rng.normal(0, 0.003, 9000), rng.uniform(0.0, 2.0, 9000) * scale))
    points = np.vstack((floor, upper, pole, sparse, wardrobe))
    labels = np.concatenate((np.full(len(floor), FLOOR_ID), np.full(len(upper) + len(pole) + len(sparse) + len(wardrobe), FURNITURE_ID))).astype(np.uint8)
    members = tall_furniture_members(points, labels, CLASS_IDS, scale, {**DEFAULTS["raymarch_filter"]})
    parts = np.cumsum([0, len(upper), len(pole), len(sparse), len(wardrobe)]) + len(floor)
    rates = [members[parts[i]:parts[i + 1]].mean() for i in range(4)]
    assert rates[0] < 0.05 and rates[1] < 0.05 and rates[2] < 0.05  # floating cabinet, thin pole, sparse object
    assert rates[3] > 0.95  # real wardrobe


def test_hidden_filter_disabled_by_default(tmp_path: Path) -> None:
    import yaml
    from samcloud.colmap.surface_fusion import visibility_mask

    assert DEFAULTS["hidden_filter"]["enabled"] is False
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "colmap.yml").read_text(encoding="utf-8"))
    assert config["dense_reconstruction"]["surface_fusion"]["hidden_filter"]["enabled"] is False
    scale = 0.8157
    views = _scaled(_views(), scale)
    wall_y = WALL_Y * scale
    dense = _write_maps(tmp_path, views, lambda view: ((wall_y - view.center[1]) / _rays(view)[..., 1]).astype(np.float32))
    rng = np.random.default_rng(83)
    wall = np.column_stack((rng.uniform(-1.0, 1.0, 500) * scale, wall_y + rng.normal(0, 0.002, 500), rng.uniform(0.2, 0.8, 500) * scale))
    behind = np.column_stack((rng.uniform(-0.9, 0.9, 400) * scale, wall_y + rng.uniform(0.25, 1.2, 400) * scale, rng.normal(0.0, 0.003, 400)))
    defaults_only = {**DEFAULTS, "units_per_metre": scale, "point_hidden_views": 999, "hidden_filter": {"min_closed_views": 2}}
    keep, stats, _ = visibility_mask(np.vstack((wall, behind)), views, dense / "stereo", defaults_only, count_seen=True)
    assert stats["removed_unconfirmed"] == 0 and keep[500:].mean() > 0.97  # the default settings leave the rule off
    enabled = {**defaults_only, "hidden_filter": {"enabled": True, "min_closed_views": 2}}
    keep_on, stats_on, _ = visibility_mask(np.vstack((wall, behind)), views, dense / "stereo", enabled, count_seen=True)
    assert keep_on[500:].mean() < 0.03 and stats_on["removed_unconfirmed"] > 300  # explicit opt-in still works


# ----------------------------------------------------------------------------- holes in tall bodies / buried points (round 4)


def _holey_wall_scene(tmp_path: Path, scale: float, slot: float, seed: int):
    """A 2 m wall (class wall) with a full-height slot of width ``slot`` in the middle; floor points behind it that exactly
    one camera sees through the slot while the other three look at the wall."""
    views = _scaled(_views(), scale)
    dense = _floor_plane_maps(tmp_path, views)
    rng = np.random.default_rng(seed)
    wall_y, n = 2.0, 40_000
    x = rng.uniform(-1.8, 1.8, n)
    x = x[np.abs(x - 0.013) > slot / 2]  # slot centre deliberately off the voxel grid
    wall = np.column_stack((x * scale, wall_y * scale + rng.normal(0, 0.003, len(x)), rng.uniform(-0.02, 2.0, len(x)) * scale))
    floor_front = np.column_stack((rng.uniform(-3.0, 3.0, 3000) * scale, rng.uniform(0.4, 1.9, 3000) * scale, rng.normal(0, 0.003, 3000)))
    cand = np.column_stack((rng.uniform(-1.5, 1.5, 60_000), rng.uniform(2.5, 3.6, 60_000), np.zeros(60_000)))
    crossing = []
    for view in views:
        center = view.center / scale
        t = (wall_y - center[1]) / (cand[:, 1] - center[1])
        crossing.append(center[0] + t * (cand[:, 0] - center[0]))
    crossing = np.array(crossing)  # (cameras, points): where each camera's line of sight pierces the wall plane
    through = np.abs(crossing - 0.013) < slot / 2 - 0.12  # clear of the dilated wall edges
    return views, dense, wall, floor_front, cand, through, scale


def _run_slot(tmp_path, scale, slot, seed, select, **config):
    from samcloud.colmap.surface_fusion import raymarch_mask

    views, dense, wall, floor_front, cand, through, scale = _holey_wall_scene(tmp_path, scale, slot, seed)
    behind = cand[select(through)][:300] * scale
    points = np.vstack((wall, floor_front, behind))
    labels = np.concatenate((np.full(len(wall), WALL_ID), np.full(len(floor_front), FLOOR_ID), np.zeros(len(behind)))).astype(np.uint8)
    settings = {**DEFAULTS, "units_per_metre": scale, "raymarch_filter": config}
    evidence = _evidence_for(points, views, dense, settings)
    remove, stats = raymarch_mask(points, labels, CLASS_IDS, views, dense / "stereo", settings, evidence)
    return remove, len(wall), len(floor_front), len(behind), stats


@pytest.mark.parametrize("scale", [1.0, 0.8157])
def test_narrow_gap_in_tall_wall_is_closed(tmp_path: Path, scale: float) -> None:
    only_one = lambda through: (through.sum(0) == 1) & through[0]  # noqa: E731 - exactly camera 0 looks through the 0.44 m slot
    removed, n_wall, n_floor, n_behind, stats = _run_slot(tmp_path, scale, 0.44, 91, only_one)
    assert n_behind > 100 and stats["raymarch_closed_columns"] > 0
    assert removed[n_wall + n_floor:].mean() > 0.9  # one lucky line of sight through a 0.44 m slot is not a view of the room
    assert removed[:n_wall].mean() < 0.01 and removed[n_wall:n_wall + n_floor].mean() < 0.02
    open_, *_ = _run_slot(tmp_path / "off", scale, 0.44, 91, only_one, close_gaps_enabled=False)
    assert open_[n_wall + n_floor:].mean() < 0.3  # without closing the slot leaves them visible (round 3 behaviour)


@pytest.mark.parametrize("scale", [1.0, 0.8157])
def test_wide_opening_is_not_closed(tmp_path: Path, scale: float) -> None:
    two_through = lambda through: through.sum(0) >= 2  # noqa: E731 - two cameras look through a 1.1 m doorway
    removed, n_wall, n_floor, n_behind, stats = _run_slot(tmp_path, scale, 1.1, 93, two_through)
    assert n_behind > 100
    assert removed[n_wall + n_floor:].mean() < 0.05  # a real doorway (wider than close_gap_m) keeps what is seen through it
    assert removed[:n_wall].mean() < 0.01


def test_buried_point_removed_and_surface_point_kept(tmp_path: Path) -> None:
    from samcloud.colmap.surface_fusion import raymarch_mask

    scale = 0.8157
    views = _scaled(_views(), scale)
    dense = _floor_plane_maps(tmp_path, views)
    rng = np.random.default_rng(97)
    layers = [np.column_stack((rng.uniform(-1.8, 1.8, 9000) * scale, (2.0 + 0.11 * k) * scale + rng.normal(0, 0.003, 9000),
                               rng.uniform(-0.02, 2.0, 9000) * scale)) for k in range(5)]  # a 0.44 m thick solid mass
    wall = np.vstack(layers)
    floor_front = np.column_stack((rng.uniform(-3.0, 3.0, 3000) * scale, rng.uniform(0.4, 1.7, 3000) * scale, rng.normal(0, 0.003, 3000)))
    buried = np.column_stack((rng.uniform(-1.0, 1.0, 300) * scale, rng.uniform(2.2, 2.35, 300) * scale, rng.normal(0, 0.003, 300)))
    surface = np.column_stack((rng.uniform(-1.0, 1.0, 300) * scale, rng.uniform(1.75, 1.85, 300) * scale, rng.normal(0, 0.003, 300)))
    points = np.vstack((wall, floor_front, buried, surface))
    labels = np.concatenate((np.full(len(wall), WALL_ID), np.full(len(floor_front), FLOOR_ID), np.zeros(600))).astype(np.uint8)
    settings = {**DEFAULTS, "units_per_metre": scale, "raymarch_filter": {"bury_depth_m": 0.25}}
    evidence = _evidence_for(points, views, dense, settings)
    n = len(wall) + len(floor_front)
    remove, _ = raymarch_mask(points, labels, CLASS_IDS, views, dense / "stereo", settings, evidence)
    assert (evidence["support"][n:n + 300] >= 3).mean() > 0.9
    assert remove[n:n + 300].mean() > 0.9  # deep inside the solid mass, every line of sight leaves through it
    assert remove[n + 300:].mean() < 0.05  # floor right in front of the mass stays
    assert remove[:n].mean() < 0.02
    default, _ = raymarch_mask(points, labels, CLASS_IDS, views, dense / "stereo", {**settings, "raymarch_filter": {}}, evidence)
    assert default[n:n + 300].mean() < 0.1  # the rule is opt-in (bury_depth_m defaults to 0)
