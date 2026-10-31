"""Recover a COLMAP ENU frame from aligned camera centres and GPS priors."""

from __future__ import annotations

import argparse
import json
import os
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
    _, images = load_model(Path(model_dir))
    return {
        image.name: -(qvec2rotmat(image.qvec).T @ image.tvec)
        for image in images.values()
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


def fit_enu_to_ecef(model_dir: Path, gps_ref: Path) -> dict[str, Any]:
    """Estimate the COLMAP ENU-to-ECEF similarity from matched camera positions."""
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
        "height_reference": HEIGHT_REFERENCE,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_georeference(path: Path, georeference: dict[str, Any]) -> None:
    """Write a georeference JSON file for reuse by LAS export."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(georeference, indent=2) + "\n", encoding="utf-8")


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
    return georeference


def enu_to_target(points_enu: np.ndarray, georeference: dict[str, Any], target_crs: str) -> np.ndarray:
    """Transform one point-cloud chunk from COLMAP ENU to the requested target CRS."""
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
    return np.column_stack((x, y, transformed_height))


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
