"""COLMAP ENU -> LV95-aligned local frame for the drone outdoor chain.

The reconstructed cloud lives in the COLMAP ENU frame (rotated against the LV95 grid by the meridian
convergence, ~1 degree in central Switzerland, with a slightly different scale).  ``LocalFrame`` is
translation-only, so the cloud is transformed ONCE into ``local = LV95(E, N, H_LN02) - origin``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from samcloud.drone.contracts import LocalFrame
from samcloud.pointcloud.georeference import (
    enu_to_target,
    ensure_project_georeference,
    load_georeference,
    require_export_grids,
)

LOGGER = logging.getLogger(__name__)

TARGET_CRS = "EPSG:2056"  # export_crs.yml key -> "EPSG:2056+5728" (LV95 + LN02)


def project_georeference(project_dir: Path | str) -> dict:
    """Load (creating from the GPS reference if needed) the project's ENU->ECEF georeference."""
    path = ensure_project_georeference(Path(project_dir))
    if path is None:
        raise FileNotFoundError(f"no georeference.json and no GPS reference under {project_dir}/products")
    require_export_grids(TARGET_CRS)
    return load_georeference(path)


def enu_to_lv95(xyz_enu: np.ndarray, georef: dict, *, chunk: int = 2_000_000) -> np.ndarray:
    """COLMAP ENU (N,3) -> LV95 E, N, H(LN02), float64, processed in chunks."""
    xyz = np.asarray(xyz_enu, dtype=np.float64).reshape(-1, 3)
    out = np.empty_like(xyz)
    step = max(int(chunk), 1)
    for start in range(0, len(xyz), step):
        out[start:start + step] = enu_to_target(xyz[start:start + step], georef, TARGET_CRS)
    return out


def make_local_frame(lv95_xyz: np.ndarray) -> LocalFrame:
    """Frame whose origin is the data centroid rounded down (E/N to 100 m, H to 1 m)."""
    centroid = np.asarray(lv95_xyz, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    east = float(np.floor(centroid[0] / 100.0) * 100.0)
    north = float(np.floor(centroid[1] / 100.0) * 100.0)
    height = float(np.floor(centroid[2]))
    return LocalFrame(origin_lv95=(east, north, height), epsg=2056, vertical_datum="LN02")


def to_local(lv95_xyz: np.ndarray, frame: LocalFrame) -> np.ndarray:
    """LV95 -> local."""
    return frame.from_lv95(lv95_xyz)


def to_lv95(local_xyz: np.ndarray, frame: LocalFrame) -> np.ndarray:
    """Local -> LV95."""
    return frame.to_lv95(local_xyz)


def local_jacobian(center_enu: np.ndarray, georef: dict, *, step: float = 50.0) -> np.ndarray:
    """3x3 linear map ENU -> LV95 axes at ``center_enu`` (central finite differences)."""
    center = np.asarray(center_enu, dtype=np.float64).reshape(3)
    offsets = np.vstack([center + sign * step * axis for axis in np.eye(3) for sign in (1.0, -1.0)])
    mapped = enu_to_lv95(offsets, georef)
    return ((mapped[0::2] - mapped[1::2]) / (2.0 * step)).T  # column i = d lv95 / d enu_i


def transform_normals(normals: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    """Rotate/scale normals with the inverse-transpose of the Jacobian and renormalise."""
    mapped = np.asarray(normals, dtype=np.float64) @ np.linalg.inv(jacobian)
    norm = np.linalg.norm(mapped, axis=1, keepdims=True)
    return mapped / np.where(norm > 0, norm, 1.0)


def _bearing_deg(vec: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(vec[0], vec[1])))


def frame_report(xyz_enu: np.ndarray, georef: dict) -> dict:
    """Rotation/scale difference between ENU and LV95 over the data extent.

    Compares the ENU bbox diagonal and the east/north axes (at the centroid) with their LV95 images:
    ``bearing_diff_deg`` (positive = LV95 bearing larger), ``scale`` (LV95 length / ENU length),
    ``max_offset_m`` (translation-only error at the farthest bbox corner relative to the centroid).
    """
    xyz = np.asarray(xyz_enu, dtype=np.float64).reshape(-1, 3)
    lo, hi, centroid = xyz.min(axis=0), xyz.max(axis=0), xyz.mean(axis=0)
    jac = local_jacobian(centroid, georef)
    east, north = jac[:, 0], jac[:, 1]
    diag_enu = hi - lo
    diag_lv = jac @ diag_enu
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    lin = (corners - centroid) @ jac.T
    exact = enu_to_lv95(corners, georef) - enu_to_lv95(centroid[None], georef)[0]
    return {
        "bearing_diff_deg": _bearing_deg(east) - 90.0,
        "north_axis_bearing_diff_deg": _bearing_deg(north),
        "scale_east": float(np.linalg.norm(east)),
        "scale_north": float(np.linalg.norm(north)),
        "diagonal_enu_m": float(np.linalg.norm(diag_enu[:2])),
        "diagonal_lv95_m": float(np.linalg.norm(diag_lv[:2])),
        "diagonal_bearing_diff_deg": _bearing_deg(diag_lv) - _bearing_deg(diag_enu),
        "max_offset_m": float(np.linalg.norm(exact[:, :2] - (corners - centroid)[:, :2], axis=1).max()),
        "max_nonlinearity_m": float(np.linalg.norm(exact - lin, axis=1).max()),
    }
