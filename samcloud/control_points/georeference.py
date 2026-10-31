"""Ground-control-point triangulation and local ENU georeferencing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from samcloud.control_points.repository import load_observations, read_control_points
from samcloud.engine.colmap_model import Camera, Image, load_model, qvec2rotmat
from samcloud.pointcloud.georeference import (
    _transform_strict,
    fit_similarity,
    strict_transformer,
)


def triangulate_point(
    observations: list[dict[str, float | str]],
    cameras: dict[int, Camera],
    images: dict[int, Image],
    *,
    max_reprojection_px: float = 3.0,
    min_triangulation_angle_deg: float = 2.0,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    """Triangulate an undistorted PINHOLE point and enforce useful geometry."""
    by_name = {image.name: image for image in images.values()}
    views = []
    for observation in observations:
        image = by_name.get(str(observation["image"]))
        if image is None:
            raise ValueError(f"Observation references an image outside the model: {observation['image']}")
        camera = cameras[image.camera_id]
        fx, fy, cx, cy = camera.intrinsics()
        u, v = float(observation["u"]), float(observation["v"])
        rotation = qvec2rotmat(image.qvec)
        projection = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]) @ np.column_stack((rotation, image.tvec))
        views.append((image, projection, (fx, fy, cx, cy), u, v))
    if len(views) < 2:
        raise ValueError("A control point needs observations in at least two images")

    matrix = np.asarray([row for _, projection, _, u, v in views for row in (u * projection[2] - projection[0], v * projection[2] - projection[1])])
    _, _, right_t = np.linalg.svd(matrix)
    homogeneous = right_t[-1]
    if abs(homogeneous[3]) < 1e-12:
        raise ValueError("Control-point triangulation is degenerate")
    point = homogeneous[:3] / homogeneous[3]
    point = _refine_point(point, views)
    errors = _reprojection_errors(point, views)
    rms = float(np.sqrt(np.mean([entry["reprojection_error_px"] ** 2 for entry in errors])))
    if rms > max_reprojection_px:
        raise ValueError(f"Control-point reprojection RMS is {rms:.2f} px (maximum {max_reprojection_px:.2f} px)")
    if _minimum_triangulation_angle(point, [view[0] for view in views]) < min_triangulation_angle_deg:
        raise ValueError("Control-point triangulation angle is too small")
    return point, errors


def georeference_from_control_points(
    model_dir: Path,
    control_points_csv: Path,
    observations_path: Path,
    crs: str,
    settings: dict[str, Any],
    *,
    gps_residuals: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[float, np.ndarray, np.ndarray]]:
    """Fit model coordinates to GCPs and return ENU export metadata and report."""
    cameras, images = load_model(model_dir)
    points = read_control_points(control_points_csv).set_index("name")
    observations = load_observations(observations_path)
    usable: list[tuple[str, np.ndarray, np.ndarray, list[dict[str, float | str]]]] = []
    rejected: dict[str, str] = {}
    for name, row in points.iterrows():
        clicks = observations.get(str(name), [])
        if len(clicks) < int(settings["min_observations_per_point"]):
            rejected[str(name)] = "needs at least two observations"
            continue
        try:
            model_point, reprojections = triangulate_point(
                clicks, cameras, images,
                max_reprojection_px=float(settings["max_reprojection_px"]),
                min_triangulation_angle_deg=float(settings["min_triangulation_angle_deg"]),
            )
        except ValueError as error:
            rejected[str(name)] = str(error)
            continue
        usable.append((str(name), model_point, row[["x", "y", "z"]].to_numpy(dtype=float), reprojections))
    if len(usable) < int(settings["min_points"]):
        raise ValueError(f"At least {settings['min_points']} usable control points are required; found {len(usable)}")

    names = [item[0] for item in usable]
    target_coordinates = np.asarray([item[2] for item in usable])
    ecef = coordinates_to_ecef(target_coordinates, crs)
    centroid_ecef = ecef.mean(axis=0)
    enu_rotation = enu_rotation_at_ecef(centroid_ecef)
    target_enu = (enu_rotation @ (ecef - centroid_ecef).T).T
    model_points = np.asarray([item[1] for item in usable])
    scale, rotation, translation, residual_summary = fit_similarity(model_points, target_enu)
    transformed = (scale * (rotation @ model_points.T)).T + translation
    residual_vectors = transformed - target_enu
    details = []
    for index, item in enumerate(usable):
        vector = residual_vectors[index]
        details.append({
            "name": item[0], "observations": len(observations[item[0]]),
            "reprojection": item[3], "residual_m": float(np.linalg.norm(vector)),
            "horizontal_residual_m": float(np.linalg.norm(vector[:2])),
            "vertical_residual_m": float(abs(vector[2])),
        })
    _add_leave_one_out(details, model_points, target_enu)
    report = {"method": "similarity_from_control_points", "crs": crs, "usable_points": len(usable),
              "rejected_points": rejected, "residual_m": residual_summary, "points": details,
              "gps_camera_centre_residual_m": gps_residuals}
    georeference = {
        "source_frame": "gcp_local_enu", "method": "similarity_from_control_points", "scale": 1.0,
        "rotation": enu_rotation.T.tolist(), "translation": centroid_ecef.tolist(), "pairs": len(usable),
        "residual_m": residual_summary, "created_at": datetime.now(timezone.utc).isoformat(),
        "control_points": {"crs": crs, "origin_ecef": centroid_ecef.tolist(), "details": details},
    }
    return georeference, report, (scale, rotation, translation)


def coordinates_to_ecef(coordinates: np.ndarray, crs: str) -> np.ndarray:
    """Transform GCP coordinates to ECEF without accepting ballpark operations."""
    transformer = strict_transformer(crs, "EPSG:4978")
    x, y, z = _transform_strict(transformer, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    return np.column_stack((x, y, z))


def enu_rotation_at_ecef(ecef: np.ndarray) -> np.ndarray:
    """Return the ECEF-to-ENU rotation at an ECEF position on WGS84."""
    to_geodetic = strict_transformer("EPSG:4978", "EPSG:4979")
    longitude, latitude, _ = _transform_strict(to_geodetic, ecef[0], ecef[1], ecef[2])
    lon, lat = np.deg2rad(float(longitude[0])), np.deg2rad(float(latitude[0]))
    return np.array([[-np.sin(lon), np.cos(lon), 0.0],
                     [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                     [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])


def save_control_point_report(path: Path, report: dict[str, Any]) -> None:
    """Save a human- and UI-readable control-point residual report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _refine_point(point: np.ndarray, views: list[tuple[Image, np.ndarray, tuple[float, float, float, float], float, float]]) -> np.ndarray:
    """Refine DLT output with a compact Gauss-Newton reprojection solve."""
    current = point.copy()
    for _ in range(8):
        residuals, jacobian = [], []
        for image, _, intrinsics, u, v in views:
            fx, fy, cx, cy = intrinsics
            camera_point = qvec2rotmat(image.qvec) @ current + image.tvec
            x, y, z = camera_point
            if z <= 1e-9:
                return current
            residuals.extend((fx * x / z + cx - u, fy * y / z + cy - v))
            derivative = np.array([[fx / z, 0.0, -fx * x / z ** 2], [0.0, fy / z, -fy * y / z ** 2]]) @ qvec2rotmat(image.qvec)
            jacobian.extend(derivative)
        delta, *_ = np.linalg.lstsq(np.asarray(jacobian), -np.asarray(residuals), rcond=None)
        current += delta
        if np.linalg.norm(delta) < 1e-8:
            break
    return current


def _reprojection_errors(point: np.ndarray, views: list[tuple[Image, np.ndarray, tuple[float, float, float, float], float, float]]) -> list[dict[str, float | str]]:
    results = []
    for image, _, intrinsics, u, v in views:
        fx, fy, cx, cy = intrinsics
        x, y, z = qvec2rotmat(image.qvec) @ point + image.tvec
        error = float(np.hypot(fx * x / z + cx - u, fy * y / z + cy - v))
        results.append({"image": image.name, "reprojection_error_px": error})
    return results


def _minimum_triangulation_angle(point: np.ndarray, images: list[Image]) -> float:
    centres = [-(qvec2rotmat(image.qvec).T @ image.tvec) for image in images]
    rays = [point - centre for centre in centres]
    angles = []
    for index, first in enumerate(rays):
        for second in rays[index + 1:]:
            cosine = np.clip(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cosine)))
    return float(max(angles))


def _add_leave_one_out(details: list[dict[str, Any]], source: np.ndarray, destination: np.ndarray) -> None:
    if len(details) < 4:
        return
    for index, detail in enumerate(details):
        remaining = np.arange(len(details)) != index
        scale, rotation, translation, _ = fit_similarity(source[remaining], destination[remaining])
        predicted = scale * rotation @ source[index] + translation
        detail["leave_one_out_residual_m"] = float(np.linalg.norm(predicted - destination[index]))
