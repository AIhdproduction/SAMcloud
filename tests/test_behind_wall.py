"""Synthetic coverage for solid-wall ray blocking in indoor surface fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from samcloud.colmap.behind_wall import behind_wall_mask, build_wall_grids
from samcloud.colmap.plane_fill import View, write_colmap_array


IDS = {"wall": 2, "floor": 4, "door": 5, "window": 6}
ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def _view(index: int, centre: np.ndarray) -> View:
    return View(index, f"wall_{index}.png", ROTATION, -ROTATION @ centre, 300.0, 300.0, 240.0, 180.0, 480, 360)


def _scene(kind: str = "mirror", scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[View], np.ndarray]:
    rng = np.random.default_rng(20261112)
    x, z = np.meshgrid(np.arange(-2.0, 2.01, 0.05), np.arange(0.0, 2.41, 0.05))
    wall = np.column_stack((x.ravel(), np.full(x.size, 3.0), z.ravel()))
    hole = (np.abs(wall[:, 0]) < 0.52) & (wall[:, 2] > 0.62) & (wall[:, 2] < 1.76)
    if kind == "door":
        hole = (wall[:, 0] > -0.55) & (wall[:, 0] < 0.55) & (wall[:, 2] < 2.05)
    labels = np.full(len(wall), IDS["wall"], np.uint8)
    if kind in {"mirror", "door", "window", "niche"}:
        wall = wall[~hole]
        labels = labels[~hole]
    if kind == "window":
        wx, wz = np.meshgrid(np.arange(-0.48, 0.49, 0.05), np.arange(0.68, 1.71, 0.05))
        window = np.column_stack((wx.ravel(), np.full(wx.size, 3.0), wz.ravel()))
        wall, labels = np.vstack((wall, window)), np.concatenate((labels, np.full(len(window), IDS["window"], np.uint8)))
    wall += rng.normal(0.0, 0.003, wall.shape)
    phantom_y = rng.uniform(4.5, 6.0, 700)
    phantom = np.column_stack((rng.uniform(-0.4, 0.4, len(phantom_y)), phantom_y, rng.uniform(0.75, 1.65, len(phantom_y))))
    if kind == "door":
        phantom[:, 0] = rng.uniform(-0.35, 0.35, len(phantom))
        phantom[:, 2] = rng.uniform(0.2, 1.8, len(phantom))
    if kind == "niche":
        phantom[:, 1] = rng.uniform(3.3, 3.9, len(phantom))
    phantom += rng.normal(0.0, 0.003, phantom.shape)
    real = np.column_stack((rng.uniform(-1.8, 1.8, 600), rng.uniform(0.7, 2.2, 600), rng.uniform(0.1, 2.2, 600)))
    real += rng.normal(0.0, 0.003, real.shape)
    points = np.vstack((wall, phantom, real)) * scale
    point_labels = np.concatenate((labels, np.full(len(phantom) + len(real), 17, np.uint8)))
    wall_index = np.full(len(points), -1, np.int64)
    wall_index[:len(wall)] = 0
    centres = [np.array((value, 0.0, 1.2)) * scale for value in (-1.4, -0.8, -0.25, 0.3, 0.9, 1.5)]
    if kind == "window":
        centres = [np.array((value, 0.0, 1.2)) * scale for value in (-0.18, -0.1, -0.03, 0.04, 0.11, 0.18)]
    if kind == "door":
        centres.extend([np.array((-0.3, 4.1, 1.1)) * scale, np.array((0.35, 4.1, 1.35)) * scale])
    views = [_view(index, centre) for index, centre in enumerate(centres)]
    return points, point_labels, wall_index, views, np.arange(len(wall), len(wall) + len(phantom))


def _write_depths(stereo: Path, views: list[View], points: np.ndarray) -> None:
    (stereo / "depth_maps").mkdir(parents=True)
    for view in views:
        depth = np.zeros((360, 480), np.float32)
        local = points @ view.rotation.T + view.translation
        u = ((view.fx * local[:, 0] / local[:, 2] + view.cx) * depth.shape[1] / view.width).astype(int)
        v = ((view.fy * local[:, 1] / local[:, 2] + view.cy) * depth.shape[0] / view.height).astype(int)
        for px, py, value in zip(u, v, local[:, 2]):
            if 0 <= px < depth.shape[1] and 0 <= py < depth.shape[0]:
                depth[py, px] = value
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", depth)


def _run(tmp_path: Path, kind: str = "mirror", scale: float = 1.0, enabled: bool = True):
    points, labels, wall_index, views, target = _scene(kind, scale)
    _write_depths(tmp_path / "stereo", views, points)
    report = {"wall": {"walls": [{"normal": [0.0, 1.0, 0.0], "offset": -3.0 * scale}]}}
    settings = {"behind_wall_filter": {"enabled": enabled, "keep_depth_m": 1.0, "cell_m": 0.05,
                "floor_band_m": 0.15, "extent_margin_m": 0.25, "blocked_share": 0.9, "min_support_views": 2},
                "visibility_margin_m": 0.03, "visibility_margin_rel": 0.01, "max_geometric_depth": 10.0,
                "units_per_metre": scale}
    keep, stats = behind_wall_mask(points, labels, wall_index, report, views, tmp_path / "stereo", settings, class_ids=IDS)
    return keep, stats, target, len(points) - 600


def test_phantom_room_behind_mirror_wall_removed(tmp_path: Path) -> None:
    keep, _stats, phantom, real_start = _run(tmp_path)
    assert (~keep[phantom]).mean() >= 0.95
    assert (~keep[real_start:]).mean() < 0.005


def test_real_room_through_door_kept(tmp_path: Path) -> None:
    keep, _stats, phantom, _real_start = _run(tmp_path, "door")
    assert keep[phantom].all()


def test_wardrobe_niche_within_one_metre_kept(tmp_path: Path) -> None:
    keep, _stats, phantom, _real_start = _run(tmp_path, "niche")
    assert keep[phantom].all()


def test_points_behind_window_kept(tmp_path: Path) -> None:
    keep, _stats, phantom, _real_start = _run(tmp_path, "window")
    assert keep[phantom].all()


def test_scaled_units_behind_wall(tmp_path: Path) -> None:
    keep, _stats, phantom, _real_start = _run(tmp_path, scale=0.8157)
    assert (~keep[phantom]).mean() >= 0.95


def test_behind_wall_disabled_is_noop(tmp_path: Path) -> None:
    keep, stats, _phantom, _real_start = _run(tmp_path, enabled=False)
    assert keep.all()
    assert stats["behind_wall_removed"] == 0


def test_behind_wall_disabled_by_default() -> None:
    from samcloud.colmap.behind_wall import DEFAULTS as BEHIND_WALL_DEFAULTS
    from samcloud.colmap.surface_fusion import DEFAULTS as FUSION_DEFAULTS

    assert not BEHIND_WALL_DEFAULTS["enabled"]
    assert not FUSION_DEFAULTS["behind_wall_filter"]["enabled"]


def _mirror_wardrobe_scene(scale: float = 1.0, reflected: bool = True):
    rng = np.random.default_rng(20261112)
    x, z = np.meshgrid(np.arange(-2.5, 2.51, 0.05), np.arange(0.0, 2.01, 0.05))
    wall = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    real = np.column_stack((rng.uniform(-2.2, 2.2, 6500), rng.uniform(-2.7, -0.05, 6500),
                            rng.uniform(0.0, 2.0, 6500)))
    chosen = rng.choice(len(real), int(len(real) * 0.60), replace=False)
    if reflected:
        phantom = real[chosen].copy()
        phantom[:, 1] = 0.30 - phantom[:, 1]
        phantom += rng.normal(0.0, 0.003, phantom.shape)
    else:
        phantom = np.column_stack((rng.uniform(-2.2, 2.2, len(chosen)), rng.uniform(0.25, 3.0, len(chosen)),
                                   rng.uniform(0.0, 2.0, len(chosen))))
    points = np.vstack((wall, real, phantom)) * scale
    labels = np.concatenate((np.full(len(wall), IDS["wall"], np.uint8),
                             np.full(len(real) + len(phantom), 17, np.uint8)))
    wall_index = np.full(len(points), -1, np.int64)
    wall_index[:len(wall)] = 0
    views = [_view(index, np.array((x0, -1.4, 1.2)) * scale) for index, x0 in enumerate((-1.5, -0.5, 0.4, 1.3))]
    report = {"wall": {"walls": [{"normal": [0.0, 1.0, 0.0], "offset": 0.0, "extent": [-2.5 * scale, 2.5 * scale]}]}}
    settings = {"behind_wall_filter": {"enabled": True, "keep_depth_m": 1.0, "cell_m": 0.05,
                "floor_band_m": 0.15, "extent_margin_m": 0.25, "blocked_share": 0.9, "min_support_views": 2,
                "mirror_image": {"enabled": True, "search_front_m": 1.2, "search_back_m": 0.2, "step_m": 0.01,
                                 "match_m": 0.04, "min_score": 0.35, "min_points": 2000}},
                "visibility_margin_m": 0.03, "visibility_margin_rel": 0.01, "units_per_metre": scale}
    return points, labels, wall_index, views, report, settings, slice(len(wall), len(wall) + len(real)), slice(len(wall) + len(real), len(points))


def test_mirror_image_behind_wardrobe_removed(tmp_path: Path) -> None:
    points, labels, wall_index, views, report, settings, real, phantom = _mirror_wardrobe_scene()
    keep, stats = behind_wall_mask(points, labels, wall_index, report, views, tmp_path / "missing", settings, class_ids=IDS)
    assert (~keep[phantom]).mean() >= 0.90
    assert (~keep[real]).mean() < 0.005
    found = stats["mirror_image_planes"]
    assert found and abs(found[0]["offset_d"] + 0.15) <= 0.02


def test_real_wardrobe_interior_not_mirror(tmp_path: Path) -> None:
    points, labels, wall_index, views, report, settings, real, interior = _mirror_wardrobe_scene(reflected=False)
    keep, stats = behind_wall_mask(points, labels, wall_index, report, views, tmp_path / "missing", settings, class_ids=IDS)
    assert keep[interior].all()
    assert not stats["mirror_image_planes"]


def test_real_room_through_door_with_door_points_kept(tmp_path: Path) -> None:
    points, labels, wall_index, views, phantom = _scene("door")
    z = np.linspace(0.0, 2.0, 30)
    jambs = np.vstack((np.column_stack((np.full(len(z), -0.55), np.full(len(z), 3.0), z)),
                       np.column_stack((np.full(len(z), 0.55), np.full(len(z), 3.0), z))))
    points = np.vstack((points, jambs))
    labels = np.concatenate((labels, np.full(len(jambs), IDS["door"], np.uint8)))
    wall_index = np.concatenate((wall_index, np.full(len(jambs), -1, np.int64)))
    _write_depths(tmp_path / "stereo", views, points)
    report = {"wall": {"walls": [{"normal": [0.0, 1.0, 0.0], "offset": -3.0}]}}
    settings = {"behind_wall_filter": {"enabled": True, "keep_depth_m": 1.0, "cell_m": 0.05,
                "floor_band_m": 0.15, "extent_margin_m": 0.25, "blocked_share": 0.9, "min_support_views": 2},
                "visibility_margin_m": 0.03, "visibility_margin_rel": 0.01, "max_geometric_depth": 10.0,
                "units_per_metre": 1.0}
    keep, _stats = behind_wall_mask(points, labels, wall_index, report, views, tmp_path / "stereo", settings, class_ids=IDS)
    assert keep[phantom].all()


def test_floor_hole_without_door_evidence_is_solid() -> None:
    x, z = np.meshgrid(np.arange(-1.5, 1.51, 0.05), np.arange(0.0, 2.21, 0.05))
    wall = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    hole = (np.abs(wall[:, 0]) < 0.40) & (wall[:, 2] < 2.0)
    wall = wall[~hole]
    labels = np.full(len(wall), IDS["wall"], np.uint8)
    wall_index = np.zeros(len(wall), np.int64)
    report = {"wall": {"walls": [{"normal": [0.0, 1.0, 0.0], "offset": 0.0, "extent": [-1.5, 1.5]}]}}
    settings = {"cell_m": 0.05, "floor_band_m": 0.15, "extent_margin_m": 0.25, "plane_snap_m": 0.05}
    grid = build_wall_grids(wall, labels, wall_index, report, settings, IDS)[0]
    assert grid.solid[int((1.0 - grid.z0) / grid.cell), int((-grid.u0) / grid.cell)]


def test_mirror_image_scaled_units(tmp_path: Path) -> None:
    scale = 0.8157
    points, labels, wall_index, views, report, settings, real, phantom = _mirror_wardrobe_scene(scale)
    keep, stats = behind_wall_mask(points, labels, wall_index, report, views, tmp_path / "missing", settings, class_ids=IDS)
    assert (~keep[phantom]).mean() >= 0.90
    assert (~keep[real]).mean() < 0.005
    assert abs(stats["mirror_image_planes"][0]["offset_d"] + 0.15 * scale) <= 0.02 * scale


def _tall_furniture_scene(height: float, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[View], dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    """A door-sized wall gap is only blocked by the semantic furniture box."""
    rng = np.random.default_rng(20261113)
    x, z = np.meshgrid(np.arange(-2.0, 2.01, 0.05), np.arange(0.0, 2.41, 0.05))
    wall = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    hole = (np.abs(wall[:, 0]) <= 1.05) & (wall[:, 2] <= 2.10)
    wall = wall[~hole]
    door_z = np.arange(0.0, 2.11, 0.05)
    door = np.vstack((np.column_stack((np.full(len(door_z), -1.02), np.zeros(len(door_z)), door_z)),
                      np.column_stack((np.full(len(door_z), 1.02), np.zeros(len(door_z)), door_z))))
    z_side = np.arange(0.0, height + 0.001, 0.05)
    y_side = np.arange(0.08, 0.49, 0.05)
    left = np.column_stack((np.full(z_side.size * y_side.size, -1.0), np.tile(y_side, z_side.size),
                            np.repeat(z_side, y_side.size)))
    right = left.copy(); right[:, 0] = 1.0
    x_top, y_top = np.meshgrid(np.arange(-1.0, 1.01, 0.05), y_side)
    top = np.column_stack((x_top.ravel(), y_top.ravel(), np.full(x_top.size, height)))
    wardrobe = np.vstack((left, right, top))
    floor_x, floor_y = np.meshgrid(np.arange(-2.2, 2.21, 0.10), np.arange(-1.5, 3.01, 0.10))
    floor = np.column_stack((floor_x.ravel(), floor_y.ravel(), np.zeros(floor_x.size)))
    phantom = np.column_stack((rng.uniform(-0.85, 0.85, 1800), rng.uniform(1.25, 3.0, 1800),
                               rng.uniform(0.15, min(1.85, height - 0.05), 1800)))
    real = np.column_stack((rng.uniform(-1.8, 1.8, 1400), rng.uniform(-2.5, -0.25, 1400), rng.uniform(0.1, 2.1, 1400)))
    points = np.vstack((wall, door, wardrobe, floor, phantom, real))
    labels = np.concatenate((np.full(len(wall), IDS["wall"], np.uint8), np.full(len(door), IDS["door"], np.uint8), np.full(len(wardrobe), 17, np.uint8),
                             np.full(len(floor), IDS["floor"], np.uint8), np.full(len(phantom) + len(real), 17, np.uint8)))
    fine = np.zeros(len(points), np.uint8)
    fine[len(wall) + len(door):len(wall) + len(door) + len(wardrobe)] = 1
    wall_index = np.full(len(points), -1, np.int64); wall_index[:len(wall)] = 0
    views = [_view(index, np.array((value, -2.2, 1.15))) for index, value in enumerate((-0.7, -0.2, 0.3, 0.8))]
    report = {"wall": {"walls": [{"normal": [0.0, 1.0, 0.0], "offset": 0.0, "extent": [-2.0, 2.0]}]}}
    settings = {"behind_wall_filter": {"enabled": True, "keep_depth_m": 1.0, "cell_m": 0.05,
                "floor_band_m": 0.15, "extent_margin_m": 0.25, "blocked_share": 0.9, "min_support_views": 2,
                "mirror_image": {"enabled": False}}, "visibility_margin_m": 0.03, "visibility_margin_rel": 0.01,
                "max_geometric_depth": 10.0, "units_per_metre": 1.0}
    phantom_rows = np.arange(len(wall) + len(door) + len(wardrobe) + len(floor), len(wall) + len(door) + len(wardrobe) + len(floor) + len(phantom))
    real_rows = np.arange(phantom_rows[-1] + 1, len(points))
    return points, labels, fine, views, report, phantom_rows, real_rows, wall_index, settings


def test_wardrobe_box_blocks_reflection(tmp_path: Path) -> None:
    points, labels, fine, views, report, phantom, real, wall_index, settings = _tall_furniture_scene(2.0, "wardrobe")
    _write_depths(tmp_path / "stereo", views, points)
    keep, stats = behind_wall_mask(points, labels, wall_index, report, views, tmp_path / "stereo", settings,
                                   class_ids=IDS, fine_terms=fine, fine_labels=["wardrobe"])
    wardrobe = fine == 1
    assert (~keep[phantom]).mean() >= 0.90
    assert keep[wardrobe].all()
    assert (~keep[real]).mean() < 0.005
    assert stats["furniture_occluders"]["count"] >= 1
    assert any(item["label"] == "wardrobe" and item["size"][2] >= 2.0 for item in stats["furniture_occluders"]["boxes"])


def test_short_furniture_not_occluder(tmp_path: Path) -> None:
    points, labels, fine, views, report, phantom, _real, wall_index, settings = _tall_furniture_scene(0.8, "chest of drawers")
    _write_depths(tmp_path / "stereo", views, points)
    keep, stats = behind_wall_mask(points, labels, wall_index, report, views, tmp_path / "stereo", settings,
                                   class_ids=IDS, fine_terms=fine, fine_labels=["chest of drawers"])
    assert keep[phantom].all()
    assert stats["furniture_occluders"]["count"] == 0


def test_mirror_image_not_accepted_at_wall_plane(tmp_path: Path) -> None:
    rng = np.random.default_rng(20261114)
    x, z = np.meshgrid(np.arange(-2.0, 2.01, 0.04), np.arange(0.0, 2.01, 0.04))
    wall = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    repeats = int(np.ceil(3000 / len(wall)))
    candidate = np.tile(wall, (repeats, 1))[:3000].copy()
    candidate[:, 1] = rng.uniform(0.02, 0.10, len(candidate))
    candidate += rng.normal(0.0, 0.002, candidate.shape)
    points = np.vstack((wall, candidate))
    labels = np.concatenate((np.full(len(wall), IDS["wall"], np.uint8), np.full(len(candidate), 17, np.uint8)))
    wall_index = np.full(len(points), -1, np.int64); wall_index[:len(wall)] = 0
    views = [_view(index, np.array((value, -1.5, 1.0))) for index, value in enumerate((-0.8, -0.2, 0.4, 1.0))]
    report = {"wall": {"walls": [{"normal": [0.0, 1.0, 0.0], "offset": 0.0, "extent": [-2.0, 2.0]}]}}
    settings = {"behind_wall_filter": {"enabled": True, "keep_depth_m": 1.0, "cell_m": 0.05,
                "floor_band_m": 0.15, "extent_margin_m": 0.25, "blocked_share": 0.9, "min_support_views": 2,
                "mirror_image": {"enabled": True, "search_front_m": 1.2, "search_back_m": 0.2, "step_m": 0.01,
                                 "match_m": 0.04, "min_score": 0.35, "min_points": 2000}}, "units_per_metre": 1.0}
    keep, stats = behind_wall_mask(points, labels, wall_index, report, views, tmp_path / "missing", settings, class_ids=IDS)
    assert keep.all()
    assert not stats["mirror_image_planes"]
