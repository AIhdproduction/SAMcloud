"""Display-only clean-up of the sparse tie-point cloud."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from samcloud.viewer.sparse_filter import (
    DEFAULTS,
    _read_points_binary,
    load_clean_sparse_points,
    neighbor_counts,
    sparse_keep_mask,
    triangulation_angles,
)


def _write_points(path: Path, points: list[tuple[tuple[float, float, float], float, list[int]]]) -> None:
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(points)))
        for index, (xyz, error, image_ids) in enumerate(points):
            file.write(struct.pack("<QdddBBBd", index + 1, *xyz, 10, 20, 30, error))
            file.write(struct.pack("<Q", len(image_ids)))
            for point2d, image_id in enumerate(image_ids):
                file.write(struct.pack("<ii", image_id, point2d))


def _wall(rng: np.random.Generator, count: int = 400) -> np.ndarray:
    """Noisy points on a 4 m x 2.5 m wall at y = 3 (no exact grid positions)."""
    return np.column_stack((rng.uniform(-2, 2, count), 3 + rng.normal(0, 0.004, count), rng.uniform(0.0, 2.5, count)))


def _centers() -> dict[int, np.ndarray]:
    return {i + 1: np.array([-1.5 + 0.37 * i, 0.013 * i, 1.4]) for i in range(9)}


def _points(xyz: np.ndarray, *, track: int = 4, error: float = 0.6) -> dict[str, np.ndarray]:
    owner = np.repeat(np.arange(len(xyz)), track)
    image_id = np.tile(np.arange(1, track + 1) * 2, len(xyz))  # images 2, 4, 6, 8: wide baseline
    return {
        "xyz": xyz, "rgb": np.zeros((len(xyz), 3), np.uint8), "error": np.full(len(xyz), error),
        "track": np.full(len(xyz), track), "owner": owner, "image_id": image_id,
    }


def test_binary_reader_matches_colmap_layout(tmp_path: Path) -> None:
    path = tmp_path / "points3D.bin"
    _write_points(path, [((1.0, 2.0, 3.0), 0.5, [1, 2, 3]), ((4.0, 5.0, 6.0), 2.5, [7])])
    points = _read_points_binary(path)
    assert points["xyz"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert points["track"].tolist() == [3, 1]
    assert points["owner"].tolist() == [0, 0, 0, 1]
    assert points["image_id"].tolist() == [1, 2, 3, 7]
    assert points["error"].tolist() == [0.5, 2.5]


def test_triangulation_angle_separates_wide_and_narrow_rays() -> None:
    xyz = np.array([[0.0, 3.0, 1.0], [0.0, 30.0, 1.0]])
    centers = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    angles = triangulation_angles(xyz, np.array([0, 0, 1, 1]), centers)
    assert 30 < angles[0] < 40  # 2 * atan(1/3) = 36.9 deg
    assert angles[1] < 4


def test_keeps_wall_and_drops_two_view_far_and_isolated_points() -> None:
    rng = np.random.default_rng(7)
    wall = _wall(rng)
    points = _points(wall)
    keep = sparse_keep_mask(points, _centers(), dict(DEFAULTS))
    assert keep.mean() > 0.97

    stray = np.array([[0.3, 1.7, 0.9], [9.1, 41.3, 1.2], [-0.4, 2.2, 2.1]])
    xyz = np.vstack((wall, stray))
    points = _points(xyz)
    points["track"][-1] = 2  # two-view point
    points["error"][-3] = 3.1  # bad reprojection
    keep = sparse_keep_mask(points, _centers(), dict(DEFAULTS))
    assert not keep[-3:].any()
    assert keep[:-3].mean() > 0.97


def test_narrow_baseline_points_are_removed() -> None:
    rng = np.random.default_rng(3)
    xyz = _wall(rng)
    points = _points(xyz)
    centers = {i: np.array([0.001 * i, 0.0, 1.4]) for i in range(1, 10)}  # nearly one position
    keep = sparse_keep_mask(points, centers, dict(DEFAULTS))
    assert not keep.any()


def test_disabled_filter_keeps_everything() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]])
    points = _points(xyz, track=2, error=5.0)
    assert sparse_keep_mask(points, _centers(), {**DEFAULTS, "enabled": False}).all()


def test_neighbor_counts_flag_isolated_point() -> None:
    rng = np.random.default_rng(11)
    xyz = np.vstack((_wall(rng), [[0.1, 9.7, 1.3]]))
    counts = neighbor_counts(xyz)
    assert counts[-1] == 0
    assert np.median(counts[:-1]) >= 4


def test_load_without_camera_model_still_filters_track_and_error(tmp_path: Path) -> None:
    path = tmp_path / "points3D.bin"
    rng = np.random.default_rng(5)
    wall = _wall(rng, 200)
    rows = [(tuple(p), 0.4, [1, 2, 3]) for p in wall]
    rows.append(((0.2, 3.0, 1.0), 0.4, [1, 2]))  # only two views
    rows.append(((0.3, 3.0, 1.1), 4.0, [1, 2, 3]))  # bad error
    _write_points(path, rows)
    xyz, rgb = load_clean_sparse_points(path, {"min_neighbors": 0, "min_points": 0})
    assert len(xyz) == 200
    assert rgb.shape == (200, 3)


def test_tiny_cloud_and_overly_strict_filter_fall_back_to_raw(tmp_path: Path) -> None:
    path = tmp_path / "points3D.bin"
    _write_points(path, [((1.0, 2.0, 3.0), 0.4, [1, 2])])
    assert len(load_clean_sparse_points(path)[0]) == 1  # below min_points
    rng = np.random.default_rng(9)
    _write_points(path, [(tuple(p), 0.4, [1, 2]) for p in _wall(rng, 300)])  # all two-view
    xyz, _ = load_clean_sparse_points(path, {"min_points": 0})
    assert len(xyz) == 300  # filter would drop everything -> shown unfiltered
