"""Recover a COLMAP ENU frame from aligned camera centres and GPS priors."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS, Transformer, datadir
from pyproj.exceptions import ProjError
from pyproj.transformer import TransformerGroup

from samcloud.core.configuration import load_export_crs_presets
from samcloud.engine.colmap_model import load_model, qvec2rotmat


HEIGHT_REFERENCE = "GPS altitude as recorded in the image EXIF (treated as ellipsoidal height)"
def camera_centres(model_dir: Path) -> dict[str, np.ndarray]:
    """Return COLMAP camera centres in the model's world coordinate frame."""
    model = Path(model_dir)
    if (model / "images.txt").is_file():
        _, images = load_model(model)
        return {
            image.name: -(qvec2rotmat(image.qvec).T @ image.tvec)
            for image in images.values()
        }
    try:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(model))
    except Exception as error:  # noqa: BLE001 - add the model location to COLMAP's opaque error
        raise ValueError(f"Could not read COLMAP model for camera centres: {model}") from error
    return {
        image.name: np.asarray(image.projection_center(), dtype=float)
        for image in reconstruction.images.values()
    }


def read_gps_ref(path: Path) -> dict[str, tuple[float, float, float]]:
    """Read COLMAP ``ref_images`` records as image name -> latitude, longitude, height."""
    references: dict[str, tuple[float, float, float]] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name, latitude, longitude, altitude = line.rsplit(maxsplit=3)
            references[name] = (float(latitude), float(longitude), float(altitude))
        except ValueError as error:
            raise ValueError(f"Invalid GPS reference on line {line_number}: {raw_line!r}") from error
    if not references:
        raise ValueError(f"GPS reference file contains no image positions: {path}")
    return references


def fit_similarity(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    """Fit a reflection-safe Umeyama similarity from ``src`` to ``dst``."""
    source = _points_array(src, "source")
    destination = _points_array(dst, "destination")
    if len(source) != len(destination):
        raise ValueError("Source and destination point counts must match")
    _require_non_collinear(source, destination)

    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    covariance = destination_centered.T @ source_centered / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    source_variance = np.sum(source_centered * source_centered) / len(source)
    scale = float(np.trace(np.diag(singular_values) @ correction) / source_variance)
    translation = destination_mean - scale * (rotation @ source_mean)

    residuals = np.linalg.norm((scale * (rotation @ source.T)).T + translation - destination, axis=1)
    residual_m = {
        "mean": float(residuals.mean()),
        "p95": float(np.percentile(residuals, 95)),
        "max": float(residuals.max()),
    }
    return scale, rotation, translation, residual_m


def fit_enu_to_ecef(
    model_dir: Path, gps_ref: Path, height_reference: str | None = None,
) -> dict[str, Any]:
    """Estimate the COLMAP ENU-to-ECEF similarity from matched camera positions.

    ``height_reference`` describes the origin of the heights in ``gps_ref`` and is stored
    in the result; it defaults to ``HEIGHT_REFERENCE``.
    """
    centres = camera_centres(model_dir)
    gps_positions = read_gps_ref(gps_ref)
    names = sorted(set(centres).intersection(gps_positions))
    if len(names) < 3:
        raise ValueError(
            "At least three camera centres with matching GPS references are required "
            "to create a georeference"
        )

    source = np.asarray([centres[name] for name in names], dtype=float)
    latitude, longitude, altitude = np.asarray([gps_positions[name] for name in names], dtype=float).T
    to_ecef = strict_transformer("EPSG:4979", "EPSG:4978")
    x, y, z = _transform_strict(to_ecef, longitude, latitude, altitude)
    destination = np.column_stack((x, y, z))
    scale, rotation, translation, residual_m = fit_similarity(source, destination)
    return {
        "source_frame": "colmap_enu",
        "method": "similarity_from_camera_centres",
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "pairs": len(names),
        "residual_m": residual_m,
        "height_reference": height_reference if height_reference else HEIGHT_REFERENCE,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_georeference(path: Path, georeference: dict[str, Any]) -> None:
    """Write a georeference JSON file for reuse by LAS export."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(georeference, indent=2) + "\n", encoding="utf-8")


def georeference_is_control_point_based(path: Path) -> bool:
    """Return whether an existing georeference was produced from control points."""
    source = Path(path)
    if not source.is_file():
        return False
    try:
        georeference = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        georeference.get("source_frame") == "gcp_local_enu"
        or georeference.get("method") in {"gcp_bundle_adjustment", "similarity_from_control_points"}
    )


def fit_model_frame_transform(
    source_model: Path,
    destination_model: Path,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float], int]:
    """Fit the similarity that maps camera centres from one COLMAP model frame to another."""
    source_centres = camera_centres(source_model)
    destination_centres = camera_centres(destination_model)
    names = sorted(set(source_centres).intersection(destination_centres))
    if len(names) < 3:
        raise ValueError(
            "At least three matching camera centres are required to relate the cloud and GCP frames"
        )
    scale, rotation, translation, residual_m = fit_similarity(
        np.asarray([source_centres[name] for name in names], dtype=float),
        np.asarray([destination_centres[name] for name in names], dtype=float),
    )
    return scale, rotation, translation, residual_m, len(names)


def gcp_georeference_for_cloud_frame(
    gcp_georeference: dict[str, Any],
    cloud_model: Path,
    gcp_model: Path,
) -> dict[str, Any]:
    """Compose a GCP-local georeference with the actual dense-cloud model frame.

    The dense cloud is expressed in ``cloud_model``'s frame, while the passed GCP
    solution is expressed in ``gcp_model``'s local ENU frame. Matching camera
    centres provide the similarity between both frames without using GPS priors.
    """
    scale, rotation, translation, residual_m, pairs = fit_model_frame_transform(cloud_model, gcp_model)
    stored_local = gcp_georeference.get("gcp_local_georeference")
    local_georeference = stored_local if isinstance(stored_local, dict) else gcp_georeference
    gcp_scale = float(local_georeference["scale"])
    gcp_rotation = np.asarray(local_georeference["rotation"], dtype=float)
    gcp_translation = np.asarray(local_georeference["translation"], dtype=float)
    if gcp_rotation.shape != (3, 3) or gcp_translation.shape != (3,):
        raise ValueError("GCP georeference has an invalid similarity transform")

    result = copy.deepcopy(local_georeference)
    result["source_frame"] = "colmap_enu"
    result["scale"] = gcp_scale * scale
    result["rotation"] = (gcp_rotation @ rotation).tolist()
    result["translation"] = (gcp_scale * (gcp_rotation @ translation) + gcp_translation).tolist()
    result["cloud_frame_transform"] = {
        "method": "similarity_from_camera_centres",
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "pairs": pairs,
        "residual_m": residual_m,
    }
    result["gcp_local_georeference"] = copy.deepcopy(local_georeference)
    return result


def restore_gcp_georeference(
    products_dir: Path,
    control_points_csv: Path,
    control_points_crs: str = "EPSG:2056+5728",
    out: Path | None = None,
) -> dict[str, Any]:
    """Restore a completed GCP project's dense-cloud georeference.

    Run with ``python -m samcloud.pointcloud.georeference restore-gcp --products <dir>
    --control-points <csv> [--crs ...] [--out <path>]``. Without ``--out``, the
    camera-centre metadata is copied to ``georeference.camera_fit_backup.json``
    before the repaired metadata is written. With ``--out``, only that output file
    is written, so an existing completed project remains untouched.
    """
    from samcloud.control_points.georeference import coordinates_to_ecef, gcp_georeference_from_ecef
    from samcloud.control_points.repository import read_control_points

    products = Path(products_dir)
    sparse = products / "sparse"
    gcp_model = sparse / "gcp_adjusted"
    cloud_model = products / "dense" / "sparse"
    if not gcp_model.is_dir():
        raise ValueError("GCP-adjusted sparse model is missing; refusing to restore georeference")
    if not cloud_model.is_dir():
        raise ValueError("Dense sparse model is missing; cannot determine the cloud coordinate frame")

    points = read_control_points(Path(control_points_csv))
    enabled = points.loc[points["role"] == "pass", ["x", "y", "z"]]
    if enabled.empty:
        raise ValueError("No enabled pass control points with coordinates were found")
    ecef = coordinates_to_ecef(enabled.to_numpy(dtype=float), control_points_crs)
    residual_m = {"mean": 0.0, "p95": 0.0, "max": 0.0}
    gcp_georeference = gcp_georeference_from_ecef(ecef, control_points_crs, residual_m)
    georeference = gcp_georeference_for_cloud_frame(gcp_georeference, cloud_model, gcp_model)
    destination = Path(out) if out is not None else products / "georeference.json"
    if out is None and destination.is_file():
        shutil.copy2(destination, products / "georeference.camera_fit_backup.json")
    save_georeference(destination, georeference)
    return georeference


def ensure_project_georeference(project_directory: Path) -> Path | None:
    """Return products/georeference.json, creating it from the GPS reference for projects processed before it existed.

    Shared by the LAS export and the DXF point source so both place a LOCAL GPS-aligned cloud identically.
    """
    from samcloud.core.storage import ensure_text_model

    products = Path(project_directory) / "products"
    georeference = products / "georeference.json"
    if georeference.is_file():
        return georeference
    gps_ref = products / "gps_ref.txt"
    model_dir = products / "dense" / "sparse_txt"
    binary_model = products / "dense" / "sparse"
    if not gps_ref.is_file() or not binary_model.is_dir():
        return None
    ensure_text_model(binary_model, model_dir)
    try:
        save_georeference(georeference, fit_enu_to_ecef(model_dir, gps_ref))
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not create GPS georeference for LAS export: {error}") from error
    return georeference


def load_georeference(path: Path) -> dict[str, Any]:
    """Load and minimally validate a saved ENU-to-ECEF georeference."""
    source = Path(path)
    try:
        georeference = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Georeference file is not valid JSON: {source}") from error
    required = {"scale", "rotation", "translation", "pairs", "residual_m"}
    missing = sorted(required.difference(georeference))
    if missing:
        raise ValueError(f"Georeference file is missing required fields: {', '.join(missing)}")
    rotation = np.asarray(georeference["rotation"], dtype=float)
    translation = np.asarray(georeference["translation"], dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Georeference rotation must be 3x3 and translation must have three values")
    if (
        not np.isfinite(float(georeference["scale"]))
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
    ):
        raise ValueError("Georeference contains non-finite transformation values")
    if "height_offset_m" in georeference and not np.isfinite(float(georeference["height_offset_m"])):
        raise ValueError("Georeference height offset must be finite")
    return georeference


def enu_to_target(points_enu: np.ndarray, georeference: dict[str, Any], target_crs: str) -> np.ndarray:
    """Transform one ENU chunk to a target CRS, applying its optional target-height offset.

    ``height_offset_m`` is zero when absent.  A checkpoint-derived correction is stored
    as ``-mean(dH)`` because checkpoint residuals use ``dH = measured - surveyed``.
    """
    points = _points_array(points_enu, "ENU")
    scale = float(georeference["scale"])
    rotation = np.asarray(georeference["rotation"], dtype=float)
    translation = np.asarray(georeference["translation"], dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Georeference rotation must be 3x3 and translation must have three values")
    ecef = (scale * (rotation @ points.T)).T + translation
    to_geodetic = strict_transformer("EPSG:4978", "EPSG:4979")
    longitude, latitude, height = _transform_strict(
        to_geodetic, ecef[:, 0], ecef[:, 1], ecef[:, 2]
    )

    target_definition = export_target_crs(target_crs)
    target = CRS.from_user_input(target_definition)
    if target.is_geographic:
        return np.column_stack((longitude, latitude, height))
    transformer = strict_transformer("EPSG:4979", target_definition)
    x, y, transformed_height = _transform_strict(transformer, longitude, latitude, height)
    height_offset_m = float(georeference.get("height_offset_m", 0.0))
    return np.column_stack((x, y, transformed_height + height_offset_m))


def export_target_crs(target_crs: str) -> str:
    """Return the configured PROJ target for an export CRS key."""
    return _export_crs_preset(target_crs)["target"]


def export_height_system(target_crs: str) -> str:
    """Describe the vertical coordinate written for an export target."""
    return str(_export_crs_preset(target_crs).get("height", "not specified"))


def require_export_grids(target_crs: str) -> None:
    """Ensure every configured PROJ grid is installed before an LAS export."""
    preset = _export_crs_preset(target_crs)
    data_directories = [Path(directory) for directory in datadir.get_data_dir().split(os.pathsep)]
    user_data_directory = datadir.get_user_data_dir()
    if user_data_directory:
        data_directories.extend(Path(directory) for directory in user_data_directory.split(os.pathsep))
    install_directory = data_directories[0]
    for grid in preset["required_grids"]:
        if not any((directory / grid).is_file() for directory in data_directories):
            raise ValueError(
                f"Export CRS {target_crs} needs PROJ grid {grid}. Install it into "
                f"{install_directory} (see config/export_crs.yml)."
            )


def format_residuals(georeference: dict[str, Any]) -> str:
    """Return compact residual information suitable for CLI and log output."""
    residuals = georeference["residual_m"]
    return (
        f"pairs={georeference['pairs']}, mean={float(residuals['mean']):.3f} m, "
        f"p95={float(residuals['p95']):.3f} m, max={float(residuals['max']):.3f} m"
    )


def strict_transformer(source_crs: str | CRS, target_crs: str | CRS) -> Transformer:
    """Create a non-ballpark transformer and report unavailable required grids clearly."""
    source = CRS.from_user_input(source_crs)
    target = CRS.from_user_input(target_crs)
    group = TransformerGroup(source, target, always_xy=True, allow_ballpark=False)
    if not group.best_available or not group.transformers:
        raise ValueError(_grid_error(source, target, group))
    try:
        return Transformer.from_crs(
            source,
            target,
            always_xy=True,
            allow_ballpark=False,
            only_best=True,
        )
    except ProjError as error:
        raise ValueError(_grid_error(source, target, group)) from error


def _transform_strict(
    transformer: Transformer,
    x: float | np.ndarray,
    y: float | np.ndarray,
    z: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform coordinates without accepting PROJ errors or infinite fallback values."""
    coordinates = tuple(
        np.ascontiguousarray(np.atleast_1d(values), dtype=np.float64) for values in (x, y, z)
    )
    try:
        if len(coordinates[0]) == 1:
            result = transformer.transform(
                *(float(values[0]) for values in coordinates),
                errcheck=True,
            )
            result = tuple(np.asarray([value], dtype=np.float64) for value in result)
        else:
            result = transformer.transform(*coordinates, errcheck=True)
    except ProjError as error:
        raise ValueError("Coordinate transformation failed. A required PROJ grid may be unavailable.") from error
    if not all(np.all(np.isfinite(values)) for values in result):
        raise ValueError("Coordinate transformation returned invalid values. A required PROJ grid may be unavailable.")
    return tuple(np.asarray(values) for values in result)


def _grid_error(source: CRS, target: CRS, group: TransformerGroup) -> str:
    names = {
        grid.short_name
        for operation in group.unavailable_operations
        for grid in operation.grids
        if not grid.available
    }
    grids = ", ".join(sorted(names)) if names else "the required grid files"
    return (
        f"No non-ballpark coordinate operation is available for {source.to_string()} -> "
        f"{target.to_string()}. Install required PROJ grid: {grids}."
    )


def _export_crs_preset(target_crs: str) -> dict[str, Any]:
    key = str(target_crs).strip()
    try:
        return load_export_crs_presets()[key]
    except KeyError as error:
        raise ValueError(
            f"Unknown export CRS '{target_crs}'. Add it to config/export_crs.yml."
        ) from error


def _points_array(points: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{label} points must be an Nx3 array")
    if not len(array):
        raise ValueError(f"{label} points must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} points contain non-finite values")
    return array


def _require_non_collinear(source: np.ndarray, destination: np.ndarray) -> None:
    if len(source) < 3:
        raise ValueError("At least three non-collinear point pairs are required")
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise ValueError("At least three non-collinear source point pairs are required")
    if np.linalg.matrix_rank(destination - destination.mean(axis=0)) < 2:
        raise ValueError("At least three non-collinear destination point pairs are required")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "restore-gcp":
        parser = argparse.ArgumentParser(description="Restore GCP georeferencing for a completed project")
        parser.add_argument("restore-gcp")
        parser.add_argument("--products", required=True, type=Path, help="project products directory")
        parser.add_argument("--control-points", required=True, type=Path, help="surveyed control-point CSV")
        parser.add_argument("--crs", default="EPSG:2056+5728", help="control-point CRS")
        parser.add_argument("--out", type=Path, help="write repaired metadata here without changing products")
        args = parser.parse_args()
        restore_gcp_georeference(args.products, args.control_points, args.crs, args.out)
        print(f"Restored GCP georeference in {args.out if args.out else args.products}")
        return

    parser = argparse.ArgumentParser(description="Estimate COLMAP ENU georeferencing from GPS camera centres")
    parser.add_argument("--model-dir", required=True, type=Path, help="COLMAP TXT model directory")
    parser.add_argument("--gps-ref", required=True, type=Path, help="GPS reference file used by model_aligner")
    parser.add_argument("--out", required=True, type=Path, help="destination georeference JSON file")
    args = parser.parse_args()

    georeference = fit_enu_to_ecef(args.model_dir, args.gps_ref)
    save_georeference(args.out, georeference)
    print(f"Wrote georeference to {args.out}")
    print(f"Georeference residuals: {format_residuals(georeference)}")


if __name__ == "__main__":
    main()
