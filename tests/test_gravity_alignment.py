"""Synthetic coverage for panorama gravity alignment and its maintenance CLI."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from samcloud.colmap.gravity import (
    UP_FACE_AXIS_RIG,
    estimate_gravity,
    gravity_transform,
    report_dict,
    rotation_from_vectors,
)
from samcloud.engine import pipeline
from samcloud.tools import align_gravity


def _horizon_levelled_rig_rotation() -> np.ndarray:
    """Return a world-to-rig rotation for a rig whose world up is +Z."""
    return rotation_from_vectors(np.array([0.0, 0.0, 1.0]), UP_FACE_AXIS_RIG)


def _write_ply(path: Path) -> None:
    data = np.array(
        [(1.0, 2.0, 3.0, 1.0, 0.0, 0.0), (4.0, 5.0, 6.0, 0.0, 1.0, 0.0)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def test_gravity_estimate_rejects_one_bad_rig_and_maps_up_face_to_positive_z() -> None:
    rotations = [_horizon_levelled_rig_rotation() for _ in range(9)] + [np.eye(3)]

    estimate = estimate_gravity(rotations)

    assert estimate.input_count == 10
    assert estimate.retained_count == 9
    assert estimate.spread_deg < 1e-8
    assert np.allclose(estimate.up_world, [0.0, 0.0, 1.0])
    assert np.allclose(estimate.rotation @ estimate.up_world, [0.0, 0.0, 1.0])


def test_gravity_transform_reports_horizontal_floor_below_camera() -> None:
    floor = np.array([[x, y, 0.0] for x in (-2.0, 0.0, 2.0) for y in (-2.0, 0.0, 2.0)])
    estimate, translation, plane, floor_applied = gravity_transform(
        [_horizon_levelled_rig_rotation()] * 6,
        floor,
        np.array([[0.0, 0.0, 1.5], [1.0, 0.0, 1.4]]),
    )

    assert plane is not None
    assert floor_applied is True
    assert plane.angle_to_z_deg < 1e-6
    report = report_dict(estimate, translation, plane, floor_applied)
    assert report["scale"] == "arbitrary (no reference)"
    assert report["up_face_is_positive_z"] is True
    assert report["floor_below_camera"] is True
    assert report["largest_horizontal_plane"]["warning"] is False


def test_cli_rotates_synthetic_ply_and_keeps_gzip_storage_copy(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    source_model = project / "products" / "sparse" / "0"
    source_model.mkdir(parents=True)
    (source_model / "points3D.bin").write_bytes(b"points")
    fused = project / "products" / "dense" / "fused.ply"
    _write_ply(fused)
    original = fused.read_bytes()
    with fused.open("rb") as source, gzip.open(fused.with_name("fused.ply.gz"), "wb") as target:
        target.write(source.read())

    def fake_align(source: Path, destination: Path) -> dict[str, object]:
        assert source == source_model
        destination.mkdir(parents=True)
        (destination / "points3D.bin").write_bytes(b"aligned")
        return {
            "aligned_model": str(destination), "frame_count": 2, "up_spread_deg_rms": 0.0,
            "rotation": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            "translation": [10.0, 20.0, 30.0],
        }

    monkeypatch.setattr(align_gravity, "align_sparse_model", fake_align)
    report = align_gravity.align_project(project)

    assert report["rotated_point_clouds"] == ["products/dense/fused.ply"]
    assert (project / "products" / "sparse" / "final_model.txt").read_text(encoding="utf-8") == "0_gravity\n"
    assert fused.with_name("fused.pre_gravity.ply").read_bytes() == original
    backup_gzip = fused.with_name("fused.pre_gravity.ply.gz")
    assert backup_gzip.is_file()
    with gzip.open(backup_gzip, "rb") as source:
        assert source.read() == original
    vertex = PlyData.read(str(fused))["vertex"].data
    assert np.allclose([vertex["x"][0], vertex["y"][0], vertex["z"][0]], [8.0, 21.0, 33.0])
    assert np.allclose([vertex["nx"][0], vertex["ny"][0], vertex["nz"][0]], [0.0, 1.0, 0.0])
    with gzip.open(fused.with_name("fused.ply.gz"), "rb") as source:
        assert source.read() == fused.read_bytes()


def test_pipeline_hook_selects_gravity_model_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    sparse = tmp_path / "products" / "sparse"
    source = sparse / "0_ba_2"
    source.mkdir(parents=True)

    def fake_align(source_model: Path, destination: Path) -> dict[str, object]:
        assert source_model == source
        destination.mkdir(parents=True)
        return {
            "frame_count": 7,
            "up_spread_deg_rms": 0.25,
            "largest_horizontal_plane": {"warning": True, "angle_to_aligned_z_deg": 4.0},
        }

    monkeypatch.setattr(pipeline, "align_sparse_model", fake_align)
    aligned, report = pipeline._align_panorama_gravity(sparse, source, tmp_path / "reports")

    assert aligned == sparse / "0_gravity"
    assert report["frame_count"] == 7
    assert (tmp_path / "reports" / "gravity_alignment.json").is_file()
