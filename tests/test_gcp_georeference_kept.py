"""Regression coverage for retaining and restoring GCP georeferencing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from samcloud.control_points.georeference import coordinates_to_ecef, gcp_georeference_from_ecef
from samcloud.control_points.repository import read_control_points
from samcloud.engine import pipeline
from samcloud.pointcloud.georeference import (
    enu_to_target,
    gcp_georeference_for_cloud_frame,
    georeference_is_control_point_based,
    restore_gcp_georeference,
    save_georeference,
)


CRS = "EPSG:2056+5728"
POINTS = """name,x,y,z,role
P1,2600000.0,1200000.0,500.0,pass
P2,2600012.0,1200003.0,501.0,pass
P3,2600004.0,1200014.0,502.0,pass
CP1,2600008.0,1200008.0,503.0,control
"""
GCP_CENTRES = np.array([
    [2.1, -4.2, 3.7],
    [11.4, 1.8, -2.5],
    [-3.2, 8.6, 4.9],
    [7.5, 6.1, 10.3],
])


def _control_points(path: Path) -> Path:
    path.write_text(POINTS, encoding="utf-8")
    return path


def _camera_georeference() -> dict:
    return {
        "source_frame": "colmap_enu", "method": "similarity_from_camera_centres", "scale": 1.0,
        "rotation": np.eye(3).tolist(), "translation": [1.0, 2.0, 3.0], "pairs": 3,
        "residual_m": {"mean": 0.0, "p95": 0.0, "max": 0.0},
    }


def _write_model(path: Path, centres: np.ndarray) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "cameras.txt").write_text("1 PINHOLE 1000 800 800 800 500 400\n", encoding="utf-8")
    lines = []
    for index, centre in enumerate(centres, start=1):
        lines.extend((
            f"{index} 1 0 0 0 {-centre[0]:.15g} {-centre[1]:.15g} {-centre[2]:.15g} 1 image_{index}.jpg\n",
            "\n",
        ))
    (path / "images.txt").write_text("".join(lines), encoding="utf-8")
    return path


def _cloud_centres(gcp_centres: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    scale = 1.0007
    rotation = np.array([
        [0.93629336, -0.31299183, -0.15934508],
        [0.28962948, 0.94470249, -0.153792],
        [0.19866933, 0.0978434, 0.97517033],
    ])
    translation = np.array([4.3, -2.7, 1.6])
    return ((rotation.T @ ((gcp_centres - translation) / scale).T).T, scale, rotation, translation)


def _products_with_models(tmp_path: Path) -> tuple[Path, np.ndarray, float, np.ndarray, np.ndarray]:
    products = tmp_path / "products"
    cloud, scale, rotation, translation = _cloud_centres(GCP_CENTRES)
    _write_model(products / "dense" / "sparse", cloud)
    _write_model(products / "sparse" / "gcp_adjusted", GCP_CENTRES)
    return products, cloud, scale, rotation, translation


def _gcp_georeference(csv_path: Path) -> dict:
    points = read_control_points(csv_path)
    enabled = points.loc[points["role"] == "pass", ["x", "y", "z"]].to_numpy(dtype=float)
    return gcp_georeference_from_ecef(
        coordinates_to_ecef(enabled, CRS), CRS, {"mean": 0.0, "p95": 0.0, "max": 0.0},
    )


def test_control_point_georeference_detection(tmp_path: Path) -> None:
    path = tmp_path / "georeference.json"
    for value in (
        {**_camera_georeference(), "source_frame": "gcp_local_enu"},
        {**_camera_georeference(), "method": "gcp_bundle_adjustment"},
        {**_camera_georeference(), "method": "similarity_from_control_points"},
    ):
        save_georeference(path, value)
        assert georeference_is_control_point_based(path)
    save_georeference(path, _camera_georeference())
    assert not georeference_is_control_point_based(path)
    assert not georeference_is_control_point_based(tmp_path / "missing.json")


def test_camera_centre_refit_keeps_gcp_georeference_and_writes_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "georeference.json"
    gcp = {**_camera_georeference(), "source_frame": "gcp_local_enu", "method": "gcp_bundle_adjustment"}
    save_georeference(path, gcp)
    monkeypatch.setattr(pipeline, "fit_enu_to_ecef", lambda *_args: (_ for _ in ()).throw(AssertionError("must not fit")))

    assert pipeline._write_camera_centre_georeference(tmp_path, tmp_path / "gps_ref.txt", path) is None
    assert json.loads(path.read_text(encoding="utf-8")) == gcp
    assert "keeping control-point georeference" in capsys.readouterr().out

    camera = _camera_georeference()
    monkeypatch.setattr(pipeline, "fit_enu_to_ecef", lambda *_args: camera)
    new_path = tmp_path / "camera.json"
    assert pipeline._write_camera_centre_georeference(tmp_path, tmp_path / "gps_ref.txt", new_path) == camera
    assert json.loads(new_path.read_text(encoding="utf-8")) == camera


def test_restore_gcp_georeference_uses_shared_calculation_and_requires_model(tmp_path: Path) -> None:
    products, _, _, _, _ = _products_with_models(tmp_path)
    csv_path = _control_points(tmp_path / "control_points.csv")
    original = _camera_georeference()
    save_georeference(products / "georeference.json", original)

    try:
        restored = restore_gcp_georeference(products, csv_path, CRS)
        expected = gcp_georeference_for_cloud_frame(
            _gcp_georeference(csv_path), products / "dense" / "sparse", products / "sparse" / "gcp_adjusted",
        )
    except ValueError as error:
        if "grid" in str(error).lower():
            pytest.skip(str(error))
        raise

    assert np.allclose(restored["rotation"], expected["rotation"])
    assert np.allclose(restored["translation"], expected["translation"])
    assert json.loads((products / "georeference.camera_fit_backup.json").read_text(encoding="utf-8")) == original

    with pytest.raises(ValueError, match="GCP-adjusted sparse model is missing"):
        restore_gcp_georeference(tmp_path / "missing_products", csv_path, CRS)


def test_restored_gcp_georeference_round_trips_lv95_within_one_millimetre(tmp_path: Path) -> None:
    products, _, scale, rotation, translation = _products_with_models(tmp_path)
    csv_path = _control_points(tmp_path / "control_points.csv")
    try:
        georeference = restore_gcp_georeference(products, csv_path, CRS)
        points = read_control_points(csv_path)
        coordinates = points.loc[points["role"] == "pass", ["x", "y", "z"]].to_numpy(dtype=float)
        base = _gcp_georeference(csv_path)
    except ValueError as error:
        if "grid" in str(error).lower():
            pytest.skip(str(error))
        raise

    ecef = coordinates_to_ecef(coordinates, CRS)
    gcp_rotation = np.asarray(base["rotation"], dtype=float)
    gcp_translation = np.asarray(base["translation"], dtype=float)
    gcp_enu = (gcp_rotation.T @ (ecef - gcp_translation).T).T
    cloud = (rotation.T @ ((gcp_enu - translation) / scale).T).T
    assert np.allclose(enu_to_target(cloud, georeference, "EPSG:2056"), coordinates, atol=0.001)


def test_restore_gcp_out_writes_only_destination(tmp_path: Path) -> None:
    products, _, _, _, _ = _products_with_models(tmp_path)
    csv_path = _control_points(tmp_path / "control_points.csv")
    original = _camera_georeference()
    source = products / "georeference.json"
    out = tmp_path / "frame_check" / "georeference.json"
    save_georeference(source, original)
    try:
        restore_gcp_georeference(products, csv_path, CRS, out)
    except ValueError as error:
        if "grid" in str(error).lower():
            pytest.skip(str(error))
        raise
    assert json.loads(source.read_text(encoding="utf-8")) == original
    assert out.is_file()
    assert not (products / "georeference.camera_fit_backup.json").exists()


def test_pipeline_guard_composes_gcp_georeference_for_different_cloud_frame(tmp_path: Path) -> None:
    products, cloud, scale, rotation, translation = _products_with_models(tmp_path)
    csv_path = _control_points(tmp_path / "control_points.csv")
    try:
        base = _gcp_georeference(csv_path)
    except ValueError as error:
        if "grid" in str(error).lower():
            pytest.skip(str(error))
        raise
    path = products / "georeference.json"
    save_georeference(path, base)

    actual = pipeline._write_camera_centre_georeference(products / "dense" / "sparse", tmp_path / "gps_ref.txt", path)

    assert actual is not None
    assert actual["method"] == "gcp_bundle_adjustment"
    assert np.isclose(actual["scale"], float(base["scale"]) * scale)
    assert np.allclose(actual["rotation"], np.asarray(base["rotation"]) @ rotation)
    expected_translation = np.asarray(base["rotation"]) @ translation + np.asarray(base["translation"])
    assert np.allclose(actual["translation"], expected_translation)
    point = cloud[0]
    expected_ecef = np.asarray(base["rotation"]) @ GCP_CENTRES[0] + np.asarray(base["translation"])
    assert np.allclose(
        float(actual["scale"]) * np.asarray(actual["rotation"]) @ point + np.asarray(actual["translation"]),
        expected_ecef,
    )
    repeated = pipeline._write_camera_centre_georeference(products / "dense" / "sparse", tmp_path / "gps_ref.txt", path)
    assert repeated is not None
    assert np.allclose(repeated["rotation"], actual["rotation"])
    assert np.allclose(repeated["translation"], actual["translation"])
