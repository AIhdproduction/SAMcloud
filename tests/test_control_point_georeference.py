"""Synthetic tests for GCP triangulation and similarity diagnostics."""

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from samcloud.control_points.georeference import _add_leave_one_out, coordinates_to_ecef, enu_rotation_at_ecef, triangulate_point
from samcloud.engine.colmap_model import Camera, Image
from samcloud.engine.colmap_model import load_model, qvec2rotmat
from samcloud.control_points.model_transform import transform_text_model
from samcloud.pointcloud.georeference import enu_to_target, fit_similarity
from samcloud.app.processing import ProcessingManager


def _scratch_directory_or_skip() -> tempfile.TemporaryDirectory[str]:
    """Return a writable system-temp directory or skip in restricted sandboxes."""
    directory = tempfile.TemporaryDirectory(prefix="samcloud-gcp-", ignore_cleanup_errors=True)
    try:
        (Path(directory.name) / "write_probe").mkdir()
    except PermissionError:
        try:
            directory.cleanup()
        except PermissionError:
            pass
        pytest.skip("A writable system temporary directory is unavailable")
    return directory


def _model():
    camera = Camera(1, "PINHOLE", 1000, 1000, [1000.0, 1000.0, 500.0, 500.0])
    images = {
        1: Image(1, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 1, "left.jpg"),
        2: Image(2, np.array([1.0, 0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 1, "right.jpg"),
        3: Image(3, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), 1, "top.jpg"),
    }
    return {1: camera}, images


def test_triangulation_recovers_synthetic_pinhole_point() -> None:
    point, errors = triangulate_point([
        {"image": "left.jpg", "u": 500.0, "v": 500.0},
        {"image": "right.jpg", "u": 400.0, "v": 500.0},
    ], *_model(), min_triangulation_angle_deg=0.1)
    assert np.allclose(point, [0.0, 0.0, 10.0])
    assert max(item["reprojection_error_px"] for item in errors) < 1e-8


def test_triangulation_rejects_bad_observation() -> None:
    with pytest.raises(ValueError, match="reprojection"):
        triangulate_point([
            {"image": "left.jpg", "u": 500.0, "v": 500.0},
            {"image": "right.jpg", "u": 800.0, "v": 500.0},
            {"image": "top.jpg", "u": 500.0, "v": 400.0},
        ], *_model(), max_reprojection_px=0.01, min_triangulation_angle_deg=0.1)


def test_similarity_and_leave_one_out_identify_outlier() -> None:
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 1.]])
    destination = 2.0 * source + np.array([10., 20., 30.])
    scale, rotation, translation, _ = fit_similarity(source, destination)
    assert scale == pytest.approx(2.0)
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, [10., 20., 30.])
    destination[-1] += [1., 0., 0.]
    details = [{"name": str(index)} for index in range(4)]
    _add_leave_one_out(details, source, destination)
    assert details[-1]["leave_one_out_residual_m"] > details[0]["leave_one_out_residual_m"]


def test_continue_command_uses_resume_flag_and_project_crs() -> None:
    command = ProcessingManager._pipeline_command(
        object(), Path("D:/demo"), {
            "settings": {"workflow": {}, "classification": {}, "coordinate_systems": {}, "control_points": {"crs": "EPSG:4979"}},
            "acquisition": {"type": "camera", "sources": {"image_directory": "D:/images"}, "georeferencing": {}},
        }, "none", resume_after_control_points=True,
    )
    assert "--resume-after-control-points" in command
    assert command[command.index("--control-points-crs") + 1] == "EPSG:4979"
    assert command[command.index("--control-points-dir") + 1] == "D:\\demo\\control_points"


def test_lv95_ln02_georeference_round_trip_or_skips_without_grid() -> None:
    coordinate = np.array([[2600000.0, 1200000.0, 500.0]])
    try:
        ecef = coordinates_to_ecef(coordinate, "EPSG:2056+5728")
    except ValueError as error:
        pytest.skip(f"CHGeo2004 grid is unavailable: {error}")
    rotation = enu_rotation_at_ecef(ecef[0])
    georeference = {"scale": 1.0, "rotation": rotation.T.tolist(), "translation": ecef[0].tolist(), "pairs": 3, "residual_m": {"mean": 0.0, "p95": 0.0, "max": 0.0}}
    actual = enu_to_target(np.zeros((1, 3)), georeference, "EPSG:2056")[0]
    assert np.allclose(actual, coordinate[0], atol=0.001)


def test_transform_text_model_keeps_rigs_and_transforms_frames() -> None:
    with _scratch_directory_or_skip() as temporary:
        source, destination = Path(temporary) / "source", Path(temporary) / "destination"
        source.mkdir()
        (source / "cameras.txt").write_text("1 PINHOLE 100 100 10 10 50 50\n", encoding="utf-8")
        (source / "images.txt").write_text("# header\n1 1 0 0 0 0 0 0 1 one.jpg\n\n2 1 0 0 0 -1 0 0 1 two.jpg\n\n", encoding="utf-8")
        (source / "points3D.txt").write_text("1 1 2 3 1 2 3 0.1 1 0\n", encoding="utf-8")
        (source / "rigs.txt").write_text(
            "1 1 CAMERA 1\n2 2 CAMERA 1 CAMERA 2 1 1 0 0 0 1 2 3\n",
            encoding="utf-8",
        )
        (source / "frames.txt").write_text("1 1 1 0 0 0 0 0 0 1 CAMERA 1 1\n", encoding="utf-8")
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        transform_text_model(source, destination, 2.0, rotation, np.array([10., 20., 30.]))
        cameras, images = load_model(destination)
        assert cameras[1].width == 100
        assert len(images) == 2
        centre = -(qvec2rotmat(images[1].qvec).T @ images[1].tvec)
        assert np.allclose(centre, [10., 20., 30.])
        rigs = (destination / "rigs.txt").read_text(encoding="utf-8")
        assert "1 1 CAMERA 1" in rigs
        assert "2 4 6" in rigs
        assert len((destination / "frames.txt").read_text(encoding="utf-8").split()) == 13


def test_colmap_round_trip_for_real_text_model() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "projects" / "test_hslu2" / "products" / "dense" / "sparse_txt"
    colmap = root / "colmap" / "COLMAP.bat"
    if not source.is_dir() or not colmap.exists():
        pytest.skip("COLMAP or projects/test_hslu2 dense sparse text model is unavailable")
    angle = math.radians(30.0)
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.], [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    scale, translation = 2.5, np.array([100., -50., 10.])
    with _scratch_directory_or_skip() as temporary:
        work = Path(temporary); transformed, binary, round_trip = work / "transformed", work / "binary", work / "round_trip"
        transform_text_model(source, transformed, scale, rotation, translation)
        binary.mkdir()
        subprocess.run([str(colmap), "model_converter", "--input_path", str(transformed), "--output_path", str(binary), "--output_type", "BIN"], check=True)
        round_trip.mkdir()
        subprocess.run([str(colmap), "model_converter", "--input_path", str(binary), "--output_path", str(round_trip), "--output_type", "TXT"], check=True)
        _, before = load_model(source); _, after = load_model(round_trip)
        for identifier in list(before)[:5]:
            centre = -(qvec2rotmat(before[identifier].qvec).T @ before[identifier].tvec)
            actual = -(qvec2rotmat(after[identifier].qvec).T @ after[identifier].tvec)
            assert np.allclose(actual, scale * rotation @ centre + translation, rtol=1e-6, atol=1e-6)
        for original, converted in zip(_points3d(source)[:5], _points3d(round_trip)[:5], strict=True):
            assert original[0] == converted[0]
            assert np.allclose(converted[1], scale * rotation @ original[1] + translation, rtol=1e-6, atol=1e-6)


def _points3d(model: Path) -> list[tuple[int, np.ndarray]]:
    return [(int(fields[0]), np.asarray(fields[1:4], dtype=float)) for line in (model / "points3D.txt").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#") for fields in [line.split()]]
