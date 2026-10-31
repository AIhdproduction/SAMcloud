"""Synthetic coverage for GPS-derived COLMAP ENU georeferencing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import warnings

import laspy
import numpy as np
import pytest
from pyproj import datadir

from samcloud.app.processing import ProcessingManager
from samcloud.app.pages.settings import _export_crs_options
from samcloud.core.configuration import load_export_crs_presets
from samcloud.engine.export_las import PLY_DTYPE, export_las
from samcloud.pointcloud.georeference import (
    enu_to_target,
    export_height_system,
    fit_enu_to_ecef,
    fit_similarity,
    require_export_grids,
    save_georeference,
    strict_transformer,
)
import samcloud.core.configuration as configuration_module
import samcloud.pointcloud.georeference as georeference_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LN02_GRID = "ch_swisstopo_chgeo2004_ETRS89_LN02.tif"
LN02_HEIGHT_SYSTEM = "LN02 (swisstopo CHGeo2004)"


def test_fit_similarity_recovers_known_transform_and_rejects_reflection() -> None:
    source = np.array([
        [-2.0, 1.0, 0.5], [3.0, -1.0, 2.0], [1.5, 4.0, -3.0],
        [-4.0, -2.0, 1.0], [0.0, 2.0, 5.0],
    ])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scale = 1.7
    translation = np.array([12.0, -8.0, 4.5])
    destination = (scale * (rotation @ source.T)).T + translation

    actual_scale, actual_rotation, actual_translation, residuals = fit_similarity(source, destination)

    assert actual_scale == pytest.approx(scale)
    assert actual_rotation == pytest.approx(rotation)
    assert actual_translation == pytest.approx(translation)
    assert residuals["max"] < 1e-12

    reflected = destination.copy()
    reflected[:, 0] *= -1
    _, reflection_safe_rotation, _, _ = fit_similarity(source, reflected)
    assert np.linalg.det(reflection_safe_rotation) == pytest.approx(1.0)


def test_enu_to_target_matches_pyproj_with_synthetic_camera_model(tmp_path: Path) -> None:
    _require_ln02_grid()
    model_dir = tmp_path / "sparse_txt"
    model_dir.mkdir()
    (model_dir / "cameras.txt").write_text("1 PINHOLE 100 100 50 50 50 50\n", encoding="utf-8")
    latitude, longitude, altitude = 47.0133, 8.3063, 530.0
    origin_ecef = _ecef_from_geodetic(longitude, latitude, altitude)
    enu_to_ecef_rotation = _enu_to_ecef_rotation(latitude, longitude)
    centres = np.array([
        [-12.0, -8.0, 4.0], [-8.0, 11.0, -2.0], [-3.0, 5.0, 7.0],
        [0.0, 0.0, 0.0], [4.0, -6.0, 3.0], [9.0, 7.0, -4.0],
        [12.0, -3.0, 6.0], [-5.0, -10.0, -5.0], [7.0, 13.0, 1.0], [15.0, 2.0, -3.0],
    ])
    gps_lines = []
    image_lines = ["# Image list with two lines of data per image:\n"]
    ecef_to_geodetic = strict_transformer("EPSG:4978", "EPSG:4979")
    for index, centre in enumerate(centres, start=1):
        name = f"image_{index:02d}.jpg"
        ecef = origin_ecef + enu_to_ecef_rotation @ centre
        lon, lat, height = ecef_to_geodetic.transform(*ecef.tolist(), errcheck=True)
        gps_lines.append(f"{name} {lat:.12f} {lon:.12f} {height:.9f}\n")
        image_lines.extend([
            f"{index} 1 0 0 0 {-centre[0]} {-centre[1]} {-centre[2]} 1 {name}\n",
            "\n",
        ])
    (model_dir / "images.txt").write_text("".join(image_lines), encoding="utf-8")
    gps_ref = tmp_path / "gps_ref.txt"
    gps_ref.write_text("".join(gps_lines), encoding="utf-8")

    georeference = fit_enu_to_ecef(model_dir, gps_ref)
    point = np.array([[3.25, -4.5, 2.75]])
    actual = enu_to_target(point, georeference, "EPSG:2056")
    point_ecef = origin_ecef + enu_to_ecef_rotation @ point[0]
    point_lon, point_lat, point_height = ecef_to_geodetic.transform(*point_ecef.tolist(), errcheck=True)
    expected = strict_transformer("EPSG:4979", "EPSG:2056+5728").transform(
        float(point_lon), float(point_lat), float(point_height), errcheck=True
    )

    assert georeference["pairs"] == len(centres)
    assert georeference["residual_m"]["max"] < 1e-4
    assert actual[0] == pytest.approx(expected, abs=0.001)
    assert actual[0, 2] < point_height - 40.0
    assert export_height_system("EPSG:2056") == LN02_HEIGHT_SYSTEM


def test_enu_to_target_single_point_emits_no_deprecation_warning() -> None:
    points = np.array([[4.125, -3.875, 2.625]], dtype=float)
    georeference = _synthetic_georeference(points)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        transformed = enu_to_target(points, georeference, "EPSG:4326")

    assert transformed.shape == (1, 3)
    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]


def test_export_las_transforms_local_enu_with_georeference(tmp_path: Path) -> None:
    _require_ln02_grid()
    points = np.array([[0.0, 0.0, 0.0], [4.0, -3.0, 2.0]], dtype=float)
    georeference = _synthetic_georeference(points)
    georeference_path = tmp_path / "georeference.json"
    save_georeference(georeference_path, georeference)
    input_path = _write_ply(tmp_path / "classified.ply", points)
    output_path = tmp_path / "classified.las"

    export_las(
        input_path, output_path, PROJECT_ROOT / "config" / "classes.json", 1,
        "LOCAL", "EPSG:2056", "outdoor", georeference_path,
    )

    las = laspy.read(output_path)
    expected = enu_to_target(points, georeference, "EPSG:2056")
    assert np.column_stack((las.x, las.y, las.z)) == pytest.approx(expected, abs=0.001)
    assert las.header.offsets[0] > 2_000_000
    assert [crs.to_epsg() for crs in las.header.parse_crs().sub_crs_list] == [2056, 5728]


def test_export_las_keeps_local_coordinates_and_requires_georeference(tmp_path: Path) -> None:
    points = np.array([[1.25, -2.5, 3.75]], dtype=float)
    input_path = _write_ply(tmp_path / "classified.ply", points)
    local_output = tmp_path / "local.las"
    export_las(
        input_path, local_output, PROJECT_ROOT / "config" / "classes.json", 100,
        "LOCAL", "LOCAL", "outdoor",
    )
    local_las = laspy.read(local_output)
    assert np.column_stack((local_las.x, local_las.y, local_las.z)) == pytest.approx(points, abs=0.001)

    with pytest.raises(
        ValueError,
        match="must be exported as LOCAL; georeferencing via control points is not available yet",
    ):
        export_las(
            input_path, tmp_path / "missing.las", PROJECT_ROOT / "config" / "classes.json", 100,
            "LOCAL", "EPSG:2056", "outdoor",
        )


def test_wgs84_las_export_preserves_coordinates_below_one_centimetre(tmp_path: Path) -> None:
    points = np.array([[4.125, -3.875, 2.625]], dtype=float)
    georeference = _synthetic_georeference(points)
    georeference_path = tmp_path / "georeference.json"
    save_georeference(georeference_path, georeference)
    input_path = _write_ply(tmp_path / "classified.ply", points)
    output_path = tmp_path / "classified.las"

    export_las(
        input_path, output_path, PROJECT_ROOT / "config" / "classes.json", 100,
        "LOCAL", "EPSG:4326", "outdoor", georeference_path,
    )

    actual = laspy.read(output_path)
    expected = enu_to_target(points, georeference, "EPSG:4326")[0]
    horizontal_error_m = np.hypot(
        (actual.x[0] - expected[0]) * 111_320 * np.cos(np.deg2rad(expected[1])),
        (actual.y[0] - expected[1]) * 111_320,
    )
    assert horizontal_error_m < 0.01
    assert actual.z[0] == pytest.approx(expected[2], abs=0.001)


def test_export_crs_loader_rejects_invalid_entry(monkeypatch) -> None:
    monkeypatch.setattr(
        configuration_module,
        "load_global_config",
        lambda _: {"presets": {"BROKEN": {"label": "Broken", "target": "not a CRS"}}},
    )

    with pytest.raises(ValueError, match="Invalid export CRS preset 'BROKEN'.*target"):
        load_export_crs_presets()


def test_export_crs_settings_options_come_from_configuration() -> None:
    presets = load_export_crs_presets()

    assert _export_crs_options() == {key: preset["label"] for key, preset in presets.items()}


def test_missing_configured_grid_raises_a_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(georeference_module.datadir, "get_data_dir", lambda: "Z:/missing-proj-data")
    monkeypatch.setattr(georeference_module.datadir, "get_user_data_dir", lambda: "")

    with pytest.raises(ValueError, match=r"Export CRS EPSG:2056 needs PROJ grid .*config/export_crs.yml"):
        require_export_grids("EPSG:2056")


def test_missing_grid_raises_a_clear_error(monkeypatch) -> None:
    class MissingGridGroup:
        best_available = False
        transformers: list[object] = []
        unavailable_operations: list[object] = []

    monkeypatch.setattr(georeference_module, "TransformerGroup", lambda *args, **kwargs: MissingGridGroup())

    with pytest.raises(ValueError, match="No non-ballpark coordinate operation"):
        strict_transformer("EPSG:4979", "EPSG:2056+5728")


def test_processing_adds_existing_georeference_to_export_command(tmp_path: Path, monkeypatch) -> None:
    project_directory = tmp_path / "project"
    products = project_directory / "products"
    classified = products / "classified" / "classified.ply"
    classified.parent.mkdir(parents=True)
    classified.write_bytes(b"ply")
    georeference = products / "georeference.json"
    georeference.write_text("{}", encoding="utf-8")
    project = {
        "settings": {"classification": {"class_set": "outdoor"}, "coordinate_systems": {
            "working_crs": "LOCAL", "export_crs": "EPSG:2056",
        }},
    }

    class Projects:
        def load(self, _: Path) -> dict:
            return project

        def save(self, _: Path, __: dict) -> None:
            pass

    captured: list[str] = []
    manager = ProcessingManager(SimpleNamespace(projects=Projects()))

    class Process:
        def poll(self):
            return None

    def launch(command: list[str], _: Path) -> Process:
        captured.extend(command)
        return Process()

    class NoThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(manager, "_launch", launch)
    monkeypatch.setattr("samcloud.app.processing.Thread", NoThread)
    manager.export_las(project_directory)

    assert "--georeference" in captured
    assert str(georeference) in captured


def _ecef_from_geodetic(longitude: float, latitude: float, altitude: float) -> np.ndarray:
    transformer = strict_transformer("EPSG:4979", "EPSG:4978")
    return np.asarray(transformer.transform(float(longitude), float(latitude), float(altitude), errcheck=True), dtype=float)


def _enu_to_ecef_rotation(latitude: float, longitude: float) -> np.ndarray:
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)
    return np.array([
        [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
        [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
        [0.0, np.cos(lat), np.sin(lat)],
    ])


def _synthetic_georeference(points: np.ndarray) -> dict:
    latitude, longitude, altitude = 47.0133, 8.3063, 530.0
    origin_ecef = _ecef_from_geodetic(longitude, latitude, altitude)
    rotation = _enu_to_ecef_rotation(latitude, longitude)
    return {
        "source_frame": "colmap_enu",
        "method": "similarity_from_camera_centres",
        "scale": 1.0,
        "rotation": rotation.tolist(),
        "translation": origin_ecef.tolist(),
        "pairs": 10,
        "residual_m": {"mean": 0.0, "p95": 0.0, "max": 0.0},
        "height_reference": "synthetic",
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def _write_ply(path: Path, xyz: np.ndarray) -> Path:
    records = np.zeros(len(xyz), dtype=PLY_DTYPE)
    records["x"], records["y"], records["z"] = xyz.T
    records["red"] = 10
    records["green"] = 20
    records["blue"] = 30
    records["scalar_class"] = 1
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(records)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar scalar_class\nend_header\n"
    ).encode("ascii")
    path.write_bytes(header + records.tobytes())
    return path


def _require_ln02_grid() -> None:
    grid = Path(datadir.get_data_dir()) / LN02_GRID
    if not grid.is_file():
        pytest.skip(f"LN02 test requires the PROJ grid: {grid}")
