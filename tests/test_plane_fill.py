"""Depth-verified plane filling, dense capture profiles and vis index mapping."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from samcloud.colmap.alignment import dense_profile, patch_match_arguments, stereo_fusion_command
from samcloud.colmap.plane_fill import (
    View,
    append_to_fused,
    fill_planes,
    read_colmap_array,
    write_colmap_array,
)
from samcloud.engine.classify import build_image_to_points_index

WALL_Y = 4.03  # wall plane y = WALL_Y, deliberately not on a grid boundary
HOLE = (-0.71, 0.83, 0.62, 1.94)  # x0, x1, z0, z1 of the textureless patch
SIZE = 160
FOCAL = 120.0


def _look_at(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    return rotation, -rotation @ center


def _views() -> list[View]:
    rng = np.random.default_rng(3)
    views = []
    for index, x in enumerate((-1.37, -0.41, 0.53, 1.29)):
        center = np.array([x, 0.11 + rng.normal(0, 0.01), 1.27 + rng.normal(0, 0.01)])
        rotation, translation = _look_at(center, np.array([0.05, WALL_Y, 1.3]))
        views.append(View(index, f"cam{index}.jpg", rotation, translation, FOCAL, FOCAL, SIZE / 2, SIZE / 2, SIZE, SIZE))
    return views


def _wall_points(rng: np.random.Generator, hole: bool = True) -> np.ndarray:
    x = rng.uniform(-3.0, 3.0, 60_000)
    z = rng.uniform(-0.05, 2.9, 60_000)
    keep = np.ones_like(x, dtype=bool)
    if hole:
        keep = ~((x > HOLE[0]) & (x < HOLE[1]) & (z > HOLE[2]) & (z < HOLE[3]))
    y = WALL_Y + rng.normal(0, 0.002, x.shape)
    wall = np.column_stack((x, y, z))[keep]
    floor_x = rng.uniform(-3.0, 3.0, 30_000)
    floor_y = rng.uniform(0.0, WALL_Y, 30_000)
    floor = np.column_stack((floor_x, floor_y, rng.normal(0.0, 0.002, floor_x.shape)))
    return np.vstack((wall, floor))


def _depth_for(view: View, through_hole: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Photometric depth sees the plane everywhere; geometric is empty in the hole.

    With ``through_hole`` the geometric map reliably sees a surface that far
    behind the wall inside the hole (an open door).
    """
    v, u = np.mgrid[0:SIZE, 0:SIZE].astype(float) + 0.5
    rays_cam = np.stack(((u - view.cx) / view.fx, (v - view.cy) / view.fy, np.ones_like(u)), axis=-1)
    rays_world = rays_cam @ view.rotation
    center = view.center
    t = (WALL_Y - center[1]) / rays_world[..., 1]
    hit = center + rays_world * t[..., None]
    depth = t.astype(np.float32)  # camera z equals ray parameter for unit-z rays
    in_hole = (hit[..., 0] > HOLE[0]) & (hit[..., 0] < HOLE[1]) & (hit[..., 2] > HOLE[2]) & (hit[..., 2] < HOLE[3])
    photometric = depth * (1 + np.random.default_rng(view.index).normal(0, 0.003, depth.shape)).astype(np.float32)
    geometric = depth.copy()
    geometric[in_hole] = 0.0 if through_hole is None else depth[in_hole] * (1.0 + through_hole)
    if through_hole is not None:
        photometric[in_hole] = geometric[in_hole]
    return photometric, geometric


def _workspace(tmp_path: Path, through_hole: float | None = None) -> tuple[Path, list[View]]:
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    (stereo / "normal_maps").mkdir()
    views = _views()
    for view in views:
        photometric, geometric = _depth_for(view, through_hole)
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.photometric.bin", photometric)
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", geometric)
        normal_cam = view.rotation @ np.array([0.0, -1.0, 0.0])
        write_colmap_array(
            stereo / "normal_maps" / f"{view.name}.photometric.bin",
            np.broadcast_to(normal_cam.astype(np.float32), (SIZE, SIZE, 3)),
        )
    return stereo, views


def _grey(_path: Path) -> np.ndarray:
    return np.full((SIZE, SIZE, 3), 231, dtype=np.uint8)


SETTINGS = {"min_plane_share": 0.05, "max_planes": 4, "ransac_points": 20_000, "ransac_iterations": 200}


def test_colmap_array_round_trip(tmp_path: Path) -> None:
    depth = np.arange(12, dtype=np.float32).reshape(3, 4) + 0.25
    write_colmap_array(tmp_path / "d.bin", depth)
    assert np.array_equal(read_colmap_array(tmp_path / "d.bin"), depth)
    normals = np.random.default_rng(1).normal(size=(3, 4, 3)).astype(np.float32)
    write_colmap_array(tmp_path / "n.bin", normals)
    assert np.array_equal(read_colmap_array(tmp_path / "n.bin"), normals)


def test_fill_adds_only_depth_verified_points_inside_the_hole(tmp_path: Path) -> None:
    stereo, views = _workspace(tmp_path)
    points = _wall_points(np.random.default_rng(7))

    result = fill_planes(points, views, stereo, tmp_path / "images", SETTINGS, image_reader=_grey)

    assert result.report["added_points"] > 50
    assert np.all(np.abs(result.xyz[:, 1] - WALL_Y) < 0.02)
    margin = 0.15
    assert np.all((result.xyz[:, 0] > HOLE[0] - margin) & (result.xyz[:, 0] < HOLE[1] + margin))
    assert np.all((result.xyz[:, 2] > HOLE[2] - margin) & (result.xyz[:, 2] < HOLE[3] + margin))
    assert np.all(result.rgb == 231)
    assert all(len(views_) >= 3 for views_ in result.visibility)
    assert all(set(views_.tolist()) <= {0, 1, 2, 3} for views_ in result.visibility)


def test_fill_keeps_open_doors_open(tmp_path: Path) -> None:
    stereo, views = _workspace(tmp_path, through_hole=0.6)
    points = _wall_points(np.random.default_rng(7))

    result = fill_planes(points, views, stereo, tmp_path / "images", SETTINGS, image_reader=_grey)

    assert result.report["added_points"] == 0
    assert result.report["rejected_free_space"] > 0


def test_fill_requires_enough_supporting_views(tmp_path: Path) -> None:
    stereo, views = _workspace(tmp_path)
    points = _wall_points(np.random.default_rng(7))

    result = fill_planes(points, views[:2], stereo, tmp_path / "images", SETTINGS, image_reader=_grey)

    assert result.report["added_points"] == 0


def test_fill_without_holes_adds_nothing(tmp_path: Path) -> None:
    stereo, views = _workspace(tmp_path)
    points = _wall_points(np.random.default_rng(7), hole=False)

    result = fill_planes(points, views, stereo, tmp_path / "images", SETTINGS, image_reader=_grey)

    assert result.report["added_points"] == 0


def test_append_to_fused_extends_ply_and_vis(tmp_path: Path) -> None:
    stereo, views = _workspace(tmp_path)
    points = _wall_points(np.random.default_rng(7))
    result = fill_planes(points, views, stereo, tmp_path / "images", SETTINGS, image_reader=_grey)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertex = np.zeros(2, dtype=dtype)
    ply_path, vis_path = tmp_path / "fused.ply", tmp_path / "fused.ply.vis"
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(ply_path))
    with open(vis_path, "wb") as handle:
        handle.write(struct.pack("<Q", 2))
        for ids in ([0, 1], [2]):
            handle.write(struct.pack("<I", len(ids)) + np.asarray(ids, "<u4").tobytes())

    append_to_fused(ply_path, vis_path, result)

    merged = PlyData.read(str(ply_path))["vertex"].data
    assert len(merged) == 2 + len(result.xyz)
    from samcloud.engine.ply_io import read_vis_file
    vis = read_vis_file(vis_path)
    assert len(vis) == len(merged)
    assert vis[0].tolist() == [0, 1] and vis[1].tolist() == [2]
    assert vis[2].tolist() == result.visibility[0].tolist()


def _write_fused(ply_path: Path, vis_path: Path, points: np.ndarray) -> None:
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertex = np.zeros(len(points), dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(ply_path))
    with open(vis_path, "wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        record = struct.pack("<I", 2) + np.asarray([0, 1], "<u4").tobytes()
        handle.write(record * len(points))


def test_fill_dense_workspace_appends_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end on disk: fused.ply is read and rewritten in place (Windows Errno 22 regression)."""
    import cv2

    import samcloud.colmap.plane_fill as plane_fill
    from samcloud.engine.ply_io import read_vis_file

    dense = tmp_path / "dense"
    dense.mkdir()
    stereo, views = _workspace(dense)
    (dense / "images").mkdir()
    for view in views:
        cv2.imwrite(str(dense / "images" / view.name), np.full((SIZE, SIZE, 3), 231, np.uint8))
    points = _wall_points(np.random.default_rng(7))
    _write_fused(dense / "fused.ply", dense / "fused.ply.vis", points)
    monkeypatch.setattr(plane_fill, "load_views", lambda _sparse: views)

    report = plane_fill.fill_dense_workspace(dense, SETTINGS)

    assert stereo == dense / "stereo"
    assert report["added_points"] > 50
    assert report["original_points"] == len(points)
    merged = PlyData.read(str(dense / "fused.ply"), mmap=False)["vertex"].data
    assert len(merged) == len(points) + report["added_points"]
    assert len(read_vis_file(dense / "fused.ply.vis")) == len(merged)
    assert (dense / "plane_fill.json").is_file()


def test_dense_profiles_keep_capture_types_apart() -> None:
    config = {
        "patch_match": {"window_radius": 6, "filter_min_ncc": 0.07},
        "fusion": {"min_num_pixels": 3},
        "plane_fill": {"enabled": True},
        "capture_profiles": {
            "drone_gps": {"patch_match": {"window_radius": 5}, "plane_fill": {"enabled": False}},
            "panorama_360": {"fusion": {"min_num_pixels": 4}},
        },
    }
    drone = dense_profile(config, "drone_gps")
    pano = dense_profile(config, "panorama_360")
    camera = dense_profile(config, "camera")

    assert drone["patch_match"] == {"window_radius": 5, "filter_min_ncc": 0.07}
    assert drone["plane_fill"]["enabled"] is False
    assert pano["fusion"]["min_num_pixels"] == 4 and pano["patch_match"]["window_radius"] == 6
    assert camera["fusion"]["min_num_pixels"] == 3 and camera["plane_fill"]["enabled"] is True
    assert config["patch_match"]["window_radius"] == 6  # base config is never mutated


def test_patch_match_arguments_and_fusion_command() -> None:
    assert patch_match_arguments({"window_radius": 6, "filter_min_ncc": 0.07, "unknown": 1}) == [
        "--PatchMatchStereo.window_radius", "6", "--PatchMatchStereo.filter_min_ncc", "0.07",
    ]
    command = stereo_fusion_command("colmap", Path("dense"), Path("dense/fused.ply"), {"min_num_pixels": 3})
    assert command[command.index("--StereoFusion.min_num_pixels") + 1] == "3"


def test_shipped_config_has_profiles_for_every_capture_type() -> None:
    from samcloud.core.configuration import load_global_config

    dense = load_global_config("colmap.yml")["dense_reconstruction"]
    assert set(dense["capture_profiles"]) == {"drone_gps", "camera", "panorama_360"}
    for capture_type in ("drone_gps", "camera", "panorama_360"):
        profile = dense_profile(dense, capture_type)
        assert profile["plane_fill"]["min_support_views"] >= 3
        assert profile["plane_fill"]["max_free_space_violations"] == 0


def test_vis_indices_follow_the_mvs_image_order() -> None:
    """360 rigs: COLMAP orders views frame by frame (1, 226, 451, ...), not by image id."""
    vis = [np.array([0, 2], dtype=np.uint32), np.array([1], dtype=np.uint32), np.array([5], dtype=np.uint32)]
    rig_order = [1, 226, 451]

    index = build_image_to_points_index(vis, rig_order)

    assert dict(index) == {1: [0], 451: [0], 226: [1]}


def test_mvs_image_order_reads_registration_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys
    import types

    from samcloud.engine.classify import mvs_image_order

    fake = types.SimpleNamespace(Reconstruction=lambda _path: types.SimpleNamespace(reg_image_ids=lambda: [1, 226, 2, 227]))
    monkeypatch.setitem(sys.modules, "pycolmap", fake)
    assert mvs_image_order(tmp_path, [227, 226, 2, 1]) == [1, 226, 2, 227]

    def broken(_path):
        raise RuntimeError("no model")

    monkeypatch.setitem(sys.modules, "pycolmap", types.SimpleNamespace(Reconstruction=broken))
    assert mvs_image_order(tmp_path, [3, 1, 2]) == [1, 2, 3]


def test_load_views_uses_registration_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys
    import types

    import samcloud.colmap.plane_fill as plane_fill

    pose = types.SimpleNamespace(rotation=types.SimpleNamespace(matrix=lambda: np.eye(3)), translation=np.zeros(3))
    camera = types.SimpleNamespace(params=[100.0, 100.0, 50.0, 40.0], width=100, height=80)
    images = {i: types.SimpleNamespace(name=f"img{i}.jpg", camera_id=1, cam_from_world=lambda: pose) for i in (1, 2, 226)}
    model = types.SimpleNamespace(images=images, cameras={1: camera}, reg_image_ids=lambda: [1, 226, 2])
    monkeypatch.setitem(sys.modules, "pycolmap", types.SimpleNamespace(Reconstruction=lambda _p: model))

    views = plane_fill.load_views(tmp_path)

    assert [(view.index, view.name) for view in views] == [(0, "img1.jpg"), (1, "img226.jpg"), (2, "img2.jpg")]


# --- class-confirmed fill (indoor) ------------------------------------------

INDOOR = ["floor, ground", "ceiling, soffit", "wall, interior wall", "door, doorway"]
FLOOR, WALL, DOOR = 1, 3, 4


def _labels_for(view: View, hole_label: int) -> np.ndarray:
    """SAM label raster: wall everywhere on the wall plane, ``hole_label`` inside the hole."""
    v, u = np.mgrid[0:SIZE, 0:SIZE].astype(float) + 0.5
    rays = np.stack(((u - view.cx) / view.fx, (v - view.cy) / view.fy, np.ones_like(u)), axis=-1) @ view.rotation
    t = (WALL_Y - view.center[1]) / rays[..., 1]
    hit = view.center + rays * t[..., None]
    in_hole = (hit[..., 0] > HOLE[0]) & (hit[..., 0] < HOLE[1]) & (hit[..., 2] > HOLE[2]) & (hit[..., 2] < HOLE[3])
    labels = np.full((SIZE, SIZE), WALL, dtype=np.uint8)
    labels[in_hole] = hole_label
    return labels


def _classified_points(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    points = _wall_points(rng)
    classes = np.where(np.abs(points[:, 1] - WALL_Y) < 0.01, WALL, FLOOR).astype(np.uint8)
    return points, classes


def _class_fill(tmp_path: Path, hole_label: int, classes_override=None, spacing=None):
    from samcloud.colmap.plane_fill import fill_class_ids, fill_classified_planes

    stereo, views = _workspace(tmp_path)
    labels = {view.name: _labels_for(view, hole_label) for view in views}
    points, classes = _classified_points(np.random.default_rng(7))
    if classes_override is not None:
        classes = classes_override(classes)
    settings = {**SETTINGS, "fill_spacing_m": spacing}
    return fill_classified_planes(
        points, classes, fill_class_ids(INDOOR, ["floor", "ceiling", "wall"]), views, stereo,
        tmp_path / "images", settings, label_reader=labels.get, image_reader=_grey,
    )


def test_class_fill_adds_wall_points_confirmed_by_depth_and_sam(tmp_path: Path) -> None:
    result = _class_fill(tmp_path, hole_label=WALL)

    assert result.report["added_points"] > 50
    assert np.all(result.classes == WALL)
    assert result.report["added_per_class"]["wall"] == len(result.xyz)
    assert np.all(np.abs(result.xyz[:, 1] - WALL_Y) < 0.02)
    margin = 0.15
    assert np.all((result.xyz[:, 0] > HOLE[0] - margin) & (result.xyz[:, 0] < HOLE[1] + margin))
    assert np.all((result.xyz[:, 2] > HOLE[2] - margin) & (result.xyz[:, 2] < HOLE[3] + margin))


def test_class_fill_rejects_depth_matches_that_sam_calls_something_else(tmp_path: Path) -> None:
    """A closed door is flat and depth-consistent but SAM says door: never filled as wall."""
    result = _class_fill(tmp_path, hole_label=DOOR)
    wall_result = _class_fill(tmp_path / "wall", hole_label=WALL)

    inset = 0.04  # grid cells straddling the door edge may still sit on real wall
    inside = (
        (result.xyz[:, 0] > HOLE[0] + inset) & (result.xyz[:, 0] < HOLE[1] - inset)
        & (result.xyz[:, 2] > HOLE[2] + inset) & (result.xyz[:, 2] < HOLE[3] - inset)
    )
    assert not np.any(inside)
    assert result.report["added_points"] < 0.1 * wall_result.report["added_points"]
    assert result.report["rejected_class"] > 0


def test_class_fill_only_uses_floor_ceiling_and_wall_planes(tmp_path: Path) -> None:
    result = _class_fill(tmp_path, hole_label=WALL, classes_override=lambda c: np.where(c == WALL, DOOR, c))

    assert result.report["planes"]["wall"] == 0
    assert np.all(result.classes != WALL) and np.all(result.classes != DOOR)


def test_class_fill_spacing_gives_a_dense_regular_grid(tmp_path: Path) -> None:
    coarse = _class_fill(tmp_path / "a", hole_label=WALL)
    dense = _class_fill(tmp_path / "b", hole_label=WALL, spacing=0.01)

    assert len(dense.xyz) > 4 * len(coarse.xyz)
    xz = dense.xyz[:, [0, 2]]
    sample = xz[np.random.default_rng(1).choice(len(xz), min(400, len(xz)), replace=False)]
    nearest = np.sort(np.linalg.norm(sample[:, None, :] - xz[None, :, :], axis=2), axis=1)[:, 1]
    assert np.median(nearest) == pytest.approx(0.01, abs=0.004)


def test_fill_classified_cloud_rewrites_ply_and_scores(tmp_path: Path) -> None:
    import cv2

    from samcloud.colmap.plane_fill import fill_classified_cloud
    from samcloud.engine.ply_io import read_classified_ply, write_classified_ply

    dense = tmp_path / "dense"
    dense.mkdir()
    stereo, views = _workspace(dense)
    labels_dir = dense / "sam_labels"
    labels_dir.mkdir()
    for view in views:
        cv2.imwrite(str(labels_dir / f"{view.name}.png"), _labels_for(view, WALL))
    points, classes = _classified_points(np.random.default_rng(7))
    ply = tmp_path / "classified_raw.ply"
    write_classified_ply(ply, points.astype(np.float32), None, classes)
    scores = tmp_path / "classified_scores.npz"
    n = len(points)
    np.savez(scores, top1_class=classes, top1_conf=np.full(n, 0.7, np.float16), top2_class=np.zeros(n, np.uint8),
             top2_conf=np.zeros(n, np.float16), view_count=np.full(n, 2, np.uint16))

    report = fill_classified_cloud(dense, ply, labels_dir, INDOOR, SETTINGS, scores_path=scores,
                                   views=views, image_reader=_grey)

    xyz, _rgb, written = read_classified_ply(ply)
    assert report["added_points"] > 50 and report["scores_extended"] is True
    assert len(xyz) == n + report["added_points"]
    assert np.all(written[n:] == WALL) and np.array_equal(written[:n], classes)
    with np.load(scores) as archive:
        assert all(len(archive[name]) == len(xyz) for name in archive.files)
        assert np.all(archive["top1_class"][n:] == WALL)
    assert (dense / "plane_fill.json").is_file()


def test_fill_classified_cloud_skips_without_label_rasters(tmp_path: Path) -> None:
    from samcloud.colmap.plane_fill import fill_classified_cloud
    from samcloud.engine.ply_io import write_classified_ply

    dense = tmp_path / "dense"
    dense.mkdir()
    _workspace(dense)
    points, classes = _classified_points(np.random.default_rng(7))
    ply = tmp_path / "classified_raw.ply"
    write_classified_ply(ply, points.astype(np.float32), None, classes)

    report = fill_classified_cloud(dense, ply, dense / "sam_labels", INDOOR, SETTINGS, views=_views())

    assert report["added_points"] == 0 and "label" in report["reason"]


def test_outdoor_class_set_never_fills() -> None:
    from samcloud.colmap.plane_fill import fill_class_ids
    from samcloud.core.configuration import load_global_config  # noqa: F401  (config must load)
    import json as _json
    from samcloud.core.paths import CONFIG_DIRECTORY

    outdoor = _json.loads((CONFIG_DIRECTORY / "classes.json").read_text(encoding="utf-8"))["outdoor"]
    assert fill_class_ids(outdoor, ["floor", "ceiling", "wall"]) == {}


@pytest.mark.parametrize(("class_set", "expected"), [("indoor", True), ("outdoor", False)])
def test_pipeline_fills_only_indoor_and_only_with_depth_anything(class_set: str, expected: bool) -> None:
    from argparse import Namespace

    from samcloud.engine.pipeline import _depth_fill_settings, _labels_arguments, _plane_fill_settings

    from samcloud.engine.pipeline import _cleaning_settings, _surface_fusion_settings

    for capture_type in ("drone_gps", "camera", "panorama_360"):
        # surface fusion off: the classic Depth-Anything point fill runs (indoor only)
        args = Namespace(class_set=class_set, capture_type=capture_type, depth_model="large", surface_fusion="0")
        assert (_depth_fill_settings(args) is not None) is expected
        assert bool(_labels_arguments(args, Path("dense"))) is expected
        assert _plane_fill_settings(args) is None  # grid fill is switched off in colmap.yml
        off = Namespace(class_set=class_set, capture_type=capture_type, depth_model="off", surface_fusion="0")
        assert _depth_fill_settings(off) is None
        no_fill = Namespace(class_set=class_set, capture_type=capture_type, depth_model="large", depth_fill="0",
                            surface_fusion="0")
        assert _depth_fill_settings(no_fill) is None
        # default: surface fusion runs indoors; in extract mode it replaces fill and cleaning
        from samcloud.engine.pipeline import _fusion_rebuilds_cloud

        default = Namespace(class_set=class_set, capture_type=capture_type, depth_model="large")
        assert (_surface_fusion_settings(default) is not None) is expected
        replaced = _fusion_rebuilds_cloud(default)
        assert (_depth_fill_settings(default) is None) is (replaced or not expected)
        # indoors the fusion's GPU visibility check replaces the CPU point cleaning; outdoors cleaning stays
        assert (_cleaning_settings(default) is None) is (replaced or expected)


# --- rule A: SAM + plane + free space ----------------------------------------

RULE_A = {**SETTINGS, "rule": "sam_plane_free_space", "fill_spacing_m": 0.01, "min_support_views": 3}


def _inside_hole(xyz: np.ndarray, inset: float = 0.05) -> np.ndarray:
    return (
        (xyz[:, 0] > HOLE[0] + inset) & (xyz[:, 0] < HOLE[1] - inset)
        & (xyz[:, 2] > HOLE[2] + inset) & (xyz[:, 2] < HOLE[3] - inset)
    )


def _rule_a(tmp_path: Path, hole_label: int, through_hole: float | None = None):
    from samcloud.colmap.plane_fill import fill_by_class_rule, fill_class_ids

    stereo, views = _workspace(tmp_path, through_hole=through_hole)
    labels = {view.name: _labels_for(view, hole_label) for view in views}
    points, classes = _classified_points(np.random.default_rng(7))
    return fill_by_class_rule(
        points, classes, fill_class_ids(INDOOR, ["floor", "ceiling", "wall"]), views, stereo,
        tmp_path / "images", RULE_A, label_reader=labels.get, image_reader=_grey,
    )


def test_rule_a_fills_a_white_wall_without_depth_on_a_1cm_grid(tmp_path: Path) -> None:
    """The hole has no depth at all (white wall); SAM wall in all views fills it."""
    result = _rule_a(tmp_path, hole_label=WALL)

    inside = _inside_hole(result.xyz)
    hole_area = (HOLE[1] - HOLE[0] - 0.1) * (HOLE[3] - HOLE[2] - 0.1)
    assert np.count_nonzero(inside) > 0.8 * hole_area / 0.01 ** 2
    assert np.all(np.abs(result.xyz[:, 1] - WALL_Y) < 0.05)
    assert np.all(result.classes == WALL)
    assert all(len(views) >= 3 for views in result.visibility)


def test_rule_a_keeps_a_closed_door_out_of_the_wall(tmp_path: Path) -> None:
    result = _rule_a(tmp_path, hole_label=DOOR)

    assert not np.any(_inside_hole(result.xyz))


def test_rule_a_keeps_an_open_doorway_open(tmp_path: Path) -> None:
    """Cameras measure the room behind the doorway: free space, nothing is added there."""
    result = _rule_a(tmp_path, hole_label=WALL, through_hole=0.6)

    assert not np.any(_inside_hole(result.xyz))
    assert result.report["rejected_free_space"] > 0


def test_fill_classified_cloud_uses_rule_a_when_configured(tmp_path: Path) -> None:
    import cv2

    from samcloud.colmap.plane_fill import fill_classified_cloud
    from samcloud.engine.ply_io import read_classified_ply, write_classified_ply

    dense = tmp_path / "dense"
    dense.mkdir()
    _stereo, views = _workspace(dense)
    (dense / "sam_labels").mkdir()
    for view in views:
        cv2.imwrite(str(dense / "sam_labels" / f"{view.name}.png"), _labels_for(view, WALL))
    points, classes = _classified_points(np.random.default_rng(7))
    ply = tmp_path / "classified_raw.ply"
    write_classified_ply(ply, points.astype(np.float32), None, classes)

    report = fill_classified_cloud(dense, ply, dense / "sam_labels", INDOOR, RULE_A, views=views, image_reader=_grey)

    xyz, _rgb, written = read_classified_ply(ply)
    assert report["rule"] == "sam_plane_free_space" and report["added_points"] > 1000
    assert len(xyz) == len(points) + report["added_points"]
    assert np.all(written[len(points):] == WALL)


# --- free-space cleaning ------------------------------------------------------


def _cleaning_workspace(tmp_path: Path):
    """Geometric depth sees the full wall in every view (no hole)."""
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    views = _views()
    for view in views:
        v, u = np.mgrid[0:SIZE, 0:SIZE].astype(float) + 0.5
        rays = np.stack(((u - view.cx) / view.fx, (v - view.cy) / view.fy, np.ones_like(u)), axis=-1) @ view.rotation
        depth = ((WALL_Y - view.center[1]) / rays[..., 1]).astype(np.float32)
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", depth)
    return views


def test_cleaning_removes_points_floating_in_front_of_a_measured_wall(tmp_path: Path) -> None:
    from samcloud.colmap.point_cleaning import clean_dense_workspace
    from samcloud.engine.ply_io import read_vis_file

    views = _cleaning_workspace(tmp_path)
    rng = np.random.default_rng(5)
    wall = np.column_stack((rng.uniform(-0.8, 0.9, 3000), WALL_Y + rng.normal(0, 0.002, 3000), rng.uniform(0.7, 1.9, 3000)))
    floaters = np.column_stack((rng.uniform(-0.5, 0.6, 40), rng.uniform(1.6, 2.9, 40), rng.uniform(1.0, 1.6, 40)))
    points = np.vstack((wall, floaters))
    _write_fused(tmp_path / "fused.ply", tmp_path / "fused.ply.vis", points)

    report = clean_dense_workspace(tmp_path, views=views)

    merged = PlyData.read(str(tmp_path / "fused.ply"), mmap=False)["vertex"].data
    kept = np.column_stack((merged["x"], merged["y"], merged["z"]))
    assert report["removed_points"] == 40
    assert np.all(np.abs(kept[:, 1] - WALL_Y) < 0.02) and len(kept) == len(wall)
    assert len(read_vis_file(tmp_path / "fused.ply.vis")) == len(kept)
    assert (tmp_path / "point_cleaning.json").is_file()


def test_cleaning_keeps_points_that_no_depth_map_contradicts(tmp_path: Path) -> None:
    from samcloud.colmap.point_cleaning import free_space_counts, floating_mask, DEFAULTS

    views = _cleaning_workspace(tmp_path)
    behind_wall = np.array([[0.1, WALL_Y + 0.5, 1.2]])  # occluded, never "in front"
    free, support = free_space_counts(behind_wall, views, tmp_path / "stereo", DEFAULTS)

    assert free[0] == 0 and not floating_mask(free, support, DEFAULTS)[0]


def test_point_cleaning_profile_exists_for_every_capture_type() -> None:
    from samcloud.core.configuration import load_global_config

    dense = load_global_config("colmap.yml")["dense_reconstruction"]
    for capture_type in ("drone_gps", "camera", "panorama_360"):
        assert dense_profile(dense, capture_type)["point_cleaning"]["enabled"] is True
