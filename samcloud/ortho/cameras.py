"""COLMAP camera loading and differentiable projection of original photographs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pycolmap
import torch


@dataclass(frozen=True, slots=True)
class OrthoCamera:
    name: str
    image_path: Path | None
    rotation: np.ndarray
    translation: np.ndarray
    model_name: str
    params: np.ndarray
    width: int
    height: int
    centre: np.ndarray

    @property
    def R(self) -> np.ndarray:
        """World-to-camera rotation matrix using COLMAP's conventional name."""
        return self.rotation

    @property
    def t(self) -> np.ndarray:
        """World-to-camera translation using COLMAP's conventional name."""
        return self.translation


@lru_cache(maxsize=128)
def _project_image_roots(resolved_project_dir: str) -> tuple[Path, ...]:
    """Return ordered, de-duplicated candidate root directories for project images."""
    from samcloud.core.configuration import load_yaml

    project_dir = Path(resolved_project_dir)
    project_file = project_dir / "project.samcloud.yml"

    document: dict[str, Any] = {}
    if project_file.is_file():
        try:
            loaded = load_yaml(project_file)
            if isinstance(loaded, dict):
                document = loaded
        except Exception:
            document = {}

    def _safe_get(data: Any, *keys: str) -> Any:
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    candidates: list[Any] = [
        _safe_get(document, "paths", "image_directory"),
        _safe_get(document, "acquisition", "sources", "image_directory"),
        _safe_get(document, "settings", "paths", "image_directory"),
        _safe_get(document, "settings", "acquisition", "sources", "image_directory"),
        _safe_get(document, "settings", "acquisition", "image_directory"),
    ]

    roots: list[Path] = []
    seen: set[Path] = set()

    for item in candidates:
        if isinstance(item, (str, Path)):
            text = str(item).strip()
            if text:
                path = Path(text)
                resolved = path.resolve() if path.is_absolute() else (project_dir / path).resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    roots.append(resolved)

    for fallback in (project_dir / "inputs" / "cubemap", project_dir / "inputs" / "images", project_dir / "inputs"):
        resolved_fallback = fallback.resolve()
        if resolved_fallback not in seen:
            seen.add(resolved_fallback)
            roots.append(resolved_fallback)

    return tuple(roots)


def resolve_image_path(project_dir: Path, name: str) -> Path | None:
    """Resolve a COLMAP relative image name without importing application code."""
    resolved_dir_str = str(Path(project_dir).resolve())
    for root in _project_image_roots(resolved_dir_str):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def load_cameras(model_dir: Path, project_dir: Path) -> list[OrthoCamera]:
    """Load registered cameras; ``Image.cam_from_world()`` composes rig frame and sensor poses in pycolmap 4.2.1."""
    reconstruction = pycolmap.Reconstruction(model_dir)
    cameras: list[OrthoCamera] = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        camera = image.camera
        pose = image.cam_from_world()
        rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64).copy()
        translation = np.asarray(pose.translation, dtype=np.float64).copy()
        centre = -rotation.T @ translation
        cameras.append(OrthoCamera(image.name, resolve_image_path(project_dir, image.name), rotation, translation, str(camera.model_name), np.asarray(camera.params, dtype=np.float64).copy(), int(camera.width), int(camera.height), centre))
    return cameras


def scaled(camera: OrthoCamera, factor: float) -> OrthoCamera:
    """Scale focal lengths and principal point while retaining all distortion coefficients."""
    if factor <= 0:
        raise ValueError("Camera scale factor must be positive")
    params = camera.params.copy()
    count = 1 if camera.model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"} else 2
    params[:count + 2] *= factor
    return replace(camera, params=params, width=max(1, round(camera.width * factor)), height=max(1, round(camera.height * factor)))


@lru_cache(maxsize=128)
def _max_undistorted_radius_impl(key: tuple[str, tuple[float, ...], int, int]) -> float:
    model_name, params, width, height = key
    if model_name in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return float("inf")
    if model_name not in {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"}:
        raise ValueError(f"Unsupported COLMAP camera model: {model_name}")

    params_arr = np.asarray(params, dtype=np.float64)
    try:
        colmap_cam = pycolmap.Camera(model=model_name, width=int(width), height=int(height), params=params_arr)
    except TypeError:
        colmap_cam = pycolmap.Camera(camera_id=1, model=model_name, width=int(width), height=int(height), params=params_arr)

    xs = np.linspace(0.0, float(width), 64)
    ys = np.linspace(0.0, float(height), 64)
    edge_points = np.vstack([
        np.column_stack((xs, np.zeros(64, dtype=np.float64))),
        np.column_stack((xs, np.full(64, float(height), dtype=np.float64))),
        np.column_stack((np.zeros(64, dtype=np.float64), ys)),
        np.column_stack((np.full(64, float(width), dtype=np.float64), ys)),
    ])

    try:
        undistorted = colmap_cam.cam_from_img(edge_points)
    except Exception:
        undistorted = []
        for pt in edge_points:
            try:
                res = colmap_cam.cam_from_img(pt)
                if res is not None:
                    undistorted.append(res)
            except Exception:
                pass

    if isinstance(undistorted, list):
        cleaned = []
        for row in undistorted:
            if row is not None:
                arr = np.asarray(row, dtype=np.float64).reshape(-1)
                if len(arr) == 3:
                    cleaned.append([arr[0] / arr[2], arr[1] / arr[2]])
                elif len(arr) >= 2:
                    cleaned.append([arr[0], arr[1]])
        pts = np.asarray(cleaned, dtype=np.float64) if cleaned else np.empty((0, 2), dtype=np.float64)
    else:
        pts = np.asarray(undistorted, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 3:
            pts = pts[:, :2] / pts[:, 2:3]
        elif pts.ndim == 2 and pts.shape[1] >= 2:
            pts = pts[:, :2]

    if len(pts) > 0:
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]

    if len(pts) > 0:
        r_corner = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    else:
        r_corner = 0.0

    if r_corner <= 0:
        return float("inf")

    r = np.linspace(0.0, 4.0 * r_corner, 20001)
    r2 = r * r

    if model_name == "SIMPLE_RADIAL":
        k = float(params[3])
        r_d = r * (1.0 + k * r2)
    elif model_name == "RADIAL":
        k1, k2 = float(params[3]), float(params[4])
        r_d = r * (1.0 + k1 * r2 + k2 * (r2 * r2))
    elif model_name == "OPENCV":
        k1, k2 = float(params[4]), float(params[5])
        r_d = r * (1.0 + k1 * r2 + k2 * (r2 * r2))
    elif model_name == "FULL_OPENCV":
        numerator = 1.0 + float(params[4]) * r2 + float(params[5]) * (r2 * r2) + float(params[8]) * (r2**3)
        denominator = 1.0 + float(params[9]) * r2 + float(params[10]) * (r2 * r2) + float(params[11]) * (r2**3)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_d = r * (numerator / denominator)
    else:
        r_d = r

    deriv = np.gradient(r_d, r)
    non_pos = np.where((r > 0) & ((deriv <= 0) | (~np.isfinite(deriv))))[0]
    if len(non_pos) > 0:
        r_mono = float(r[non_pos[0]])
    else:
        r_mono = float("inf")

    return float(min(1.05 * r_corner, 0.98 * r_mono))


def max_undistorted_radius(camera: OrthoCamera) -> float:
    """Return the maximum safe undistorted normalized radius before radial distortion fold-back."""
    key = (camera.model_name, tuple(camera.params), camera.width, camera.height)
    return _max_undistorted_radius_impl(key)


def project_points(camera: OrthoCamera, xyz_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project COLMAP-world points using COLMAP perspective camera-model formulas."""
    if xyz_world.ndim != 2 or xyz_world.shape[1] != 3:
        raise ValueError("World points must have shape (N, 3)")
    matrix = torch.as_tensor(camera.rotation, dtype=xyz_world.dtype, device=xyz_world.device)
    translation = torch.as_tensor(camera.translation, dtype=xyz_world.dtype, device=xyz_world.device)
    points = xyz_world @ matrix.T + translation
    depth = points[:, 2]
    z = torch.where(depth.abs() > torch.finfo(points.dtype).eps, depth, torch.ones_like(depth))
    x, y = points[:, 0] / z, points[:, 1] / z
    x_undist, y_undist = x, y
    p = torch.as_tensor(camera.params, dtype=xyz_world.dtype, device=xyz_world.device)
    name = camera.model_name
    if name == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif name == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif name in {"SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = p[0]; cx, cy = p[1], p[2]
        radius = x * x + y * y
        radial = 1.0 + p[3] * radius + (p[4] * radius * radius if name == "RADIAL" else 0.0)
        x, y = x * radial, y * radial
    elif name in {"OPENCV", "FULL_OPENCV"}:
        fx, fy, cx, cy = p[:4]
        radius = x * x + y * y
        if name == "OPENCV":
            radial = 1.0 + p[4] * radius + p[5] * radius * radius
            p1, p2 = p[6], p[7]
        else:
            numerator = 1.0 + p[4] * radius + p[5] * radius * radius + p[8] * radius**3
            denominator = 1.0 + p[9] * radius + p[10] * radius * radius + p[11] * radius**3
            radial = numerator / denominator
            p1, p2 = p[6], p[7]
        xy = x * y
        x, y = x * radial + 2.0 * p1 * xy + p2 * (radius + 2.0 * x * x), y * radial + p1 * (radius + 2.0 * y * y) + 2.0 * p2 * xy
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {name}")
    u, v = fx * x + cx, fy * y + cy
    valid = (depth > 0) & torch.isfinite(u) & torch.isfinite(v) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    limit = max_undistorted_radius(camera)
    if limit < float("inf"):
        valid = valid & ((x_undist * x_undist + y_undist * y_undist) <= (limit * limit))
    return u, v, depth, valid
