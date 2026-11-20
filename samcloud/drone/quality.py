"""Photogrammetry quality metrics to compare reconstruction runs.

Every metric is measured from the files in a project's ``products`` directory;
nothing is estimated or invented.  A metric whose inputs are missing is reported
as ``{"available": False, "reason": ...}`` instead of raising.

Product layout (see ``samcloud/engine/pipeline.py``):

* ``sparse/<model>``             COLMAP sparse model (``resolve_sparse_model`` picks it)
* ``image_list.txt``             input images, one name per line
* ``gps_ref.txt``                ``<image_name> <lat> <lon> <alt>`` (WGS84, RTK/GPS reference);
                                 the model is aligned to the COLMAP ENU frame (``model_aligner``)
* ``dense/fused.ply[.gz]``       dense fused cloud (density source)
* ``classified/classified.ply``  classified cloud with ``scalar_class`` (plane noise, facades)

Subsampling (deterministic stride, never random): the convex hull uses at most
``HULL_MAX_POINTS`` points, plane patches at most ``PLANE_MAX_POINTS_PER_CLASS``
points per class and normals at most ``NORMAL_MAX_POINTS``.  Density counts use
every point (chunked).  The applied strides are written to the report.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from samcloud.drone.contracts import BUILDING, GROUND, ROAD
from samcloud.drone.io import load_classified_pointset

LOGGER = logging.getLogger(__name__)

CLASS_CODES = {"ground": GROUND, "road": ROAD, "building": BUILDING}  # LAS codes (contracts)
HULL_MAX_POINTS = 2_000_000
PLANE_MAX_POINTS_PER_CLASS = 3_000_000
NORMAL_MAX_POINTS = 2_000_000
MAX_GRID_CELLS = 30_000_000
CHUNK = 5_000_000
PATCH_RADIUS_M = 0.5
PATCH_MIN_POINTS = 50
PATCH_MAX_POINTS = 3000
MAX_PATCHES_PER_CLASS = 300
RANSAC_THRESHOLD_M = 0.05
RANSAC_ITERATIONS = 64
MIN_INLIER_RATIO = 0.6
ROOF_NZ = 0.5
FACADE_NZ = 0.3
SEED = 12345


def unavailable(reason: str) -> dict[str, Any]:
    """Return the standard block for a metric that could not be measured."""
    return {"available": False, "reason": reason}


def stride_subsample(count: int, max_points: int) -> tuple[np.ndarray, int]:
    """Return deterministic stride indices (at most ``max_points``) and the stride factor."""
    step = max(1, -(-count // max(1, max_points)))
    return np.arange(0, count, step, dtype=np.int64), step


# --------------------------------------------------------------------------- sparse

def compute_sparse_metrics(reconstruction: Any, input_images: int | None) -> dict[str, Any]:
    """Registration, reprojection error and track length of a COLMAP reconstruction.

    ``reconstruction`` needs ``reg_image_ids()`` and ``points3D`` (each point with
    ``error`` and ``track.length()``), as pycolmap provides.  Reprojection statistics
    are over 3D points (each point's mean error).
    """
    registered = len(list(reconstruction.reg_image_ids()))
    errors, tracks = [], []
    for point in reconstruction.points3D.values():
        errors.append(float(point.error))
        tracks.append(int(point.track.length()))
    block: dict[str, Any] = {"available": True, "registered_images": registered, "input_images": input_images}
    block["registered_ratio"] = (registered / input_images) if input_images else None
    block["num_points3D"] = len(errors)
    block["num_points3d"] = len(errors)
    if errors:
        err, trk = np.asarray(errors), np.asarray(tracks, dtype=float)
        block.update(
            reprojection_error_mean_px=float(err.mean()), reprojection_error_median_px=float(np.median(err)),
            track_length_mean=float(trk.mean()), track_length_median=float(np.median(trk)),
        )
    else:
        block.update(reprojection_error_mean_px=None, reprojection_error_median_px=None,
                     track_length_mean=None, track_length_median=None)
    return block


def _read_input_count(products: Path) -> int | None:
    path = products / "image_list.txt"
    if not path.is_file():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_reconstruction(products: Path) -> tuple[Any | None, str | None]:
    from samcloud.core.sparse_model import resolve_sparse_model

    sparse = products / "sparse"
    if not sparse.is_dir():
        return None, "products/sparse not found"
    model = resolve_sparse_model(sparse)
    if model is None:
        return None, "no sparse model with points found under products/sparse"
    try:
        import pycolmap

        return pycolmap.Reconstruction(str(model)), None
    except Exception as exc:  # unreadable model must not crash the report
        return None, f"sparse model unreadable: {exc}"


# --------------------------------------------------------------------------- RTK

def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity ``target ~ s * R @ source + t`` (Umeyama 1991)."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    xs, xt = source - mu_s, target - mu_t
    cov = xt.T @ xs / len(source)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    var = (xs ** 2).sum() / len(source)
    scale = float(np.trace(np.diag(d) @ sign) / var) if var > 0 else 1.0
    return scale, rotation, mu_t - scale * rotation @ mu_s


def _axis_stats(residual: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, axis in enumerate("ENU"):
        col = residual[:, i]
        out[axis] = {"mean": float(col.mean()), "rms": float(np.sqrt((col ** 2).mean())),
                     "max": float(np.abs(col).max())}
    return out


def geodetic_to_local_enu(llh: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
    """Convert (lat, lon, alt) WGS84 rows to ENU metres around ``origin`` (lat, lon, alt).

    Without ``origin`` the centroid of the rows is used.  COLMAP ``model_aligner`` uses the
    first GPS coordinate as ENU origin, so RTK residuals pass that row explicitly.
    """
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x, y, z = tr.transform(llh[:, 1], llh[:, 0], llh[:, 2])
    ecef = np.column_stack((x, y, z))
    if origin is None:
        centre = ecef.mean(axis=0)
        lat0, lon0 = np.radians(llh[:, 0].mean()), np.radians(llh[:, 1].mean())
    else:
        ox, oy, oz = tr.transform(float(origin[1]), float(origin[0]), float(origin[2]))
        centre = np.array([ox, oy, oz])
        lat0, lon0 = np.radians(float(origin[0])), np.radians(float(origin[1]))
    rot = np.array([
        [-np.sin(lon0), np.cos(lon0), 0.0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
    ])
    return (ecef - centre) @ rot.T


def compute_rtk_residuals(centres: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    """Camera centres versus reference positions, per axis E/N/U.

    Reference rows that look like WGS84 (lat, lon, alt within a few degrees) are converted to
    local ENU around the FIRST matched reference row (dict order = ``gps_ref.txt`` order), which is the
    frame COLMAP's ``model_aligner --alignment_type enu`` produces ("Using the first GPS coordinate
    as the ENU origin").  ``raw`` is the direct difference ``model - reference``
    in that frame; ``after_similarity`` is after an Umeyama similarity fit (isolates GPS noise
    from frame offset/scale).  Residuals are signed ``model - reference`` in metres.
    """
    ordered = [n for n in reference if n in centres]
    names = sorted(ordered)
    if len(names) < 3:
        return unavailable(f"fewer than 3 camera centres match gps_ref ({len(names)})")
    model = np.asarray([centres[n] for n in names], dtype=float)
    ref = np.asarray([reference[n] for n in names], dtype=float)
    geodetic = bool(np.all(np.abs(ref[:, 0]) <= 90) and np.all(np.abs(ref[:, 1]) <= 180)
                    and np.ptp(ref[:, 0]) < 5 and np.ptp(ref[:, 1]) < 5)
    if geodetic:
        ref = geodetic_to_local_enu(ref, origin=np.asarray(reference[ordered[0]], dtype=float))
    block: dict[str, Any] = {"available": True, "matched_images": len(names),
                             "reference_frame": "wgs84_local_enu" if geodetic else "as_given",
                             "raw": _axis_stats(model - ref)}
    scale, rotation, translation = umeyama(model, ref)
    fitted = scale * model @ rotation.T + translation
    fit = _axis_stats(fitted - ref)
    fit["scale"] = scale
    fit["rms_3d"] = float(np.sqrt(((fitted - ref) ** 2).sum(axis=1).mean()))
    block["after_similarity"] = fit
    return block


def _read_gps_ref(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4:
            try:
                result[parts[0]] = np.asarray([float(v) for v in parts[1:]])
            except ValueError:
                continue
    return result


# --------------------------------------------------------------------------- cloud IO

def _ensure_plain(path: Path, workdir: Path) -> Path:
    if path.suffix != ".gz":
        return path
    target = workdir / path.stem
    with gzip.open(path, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read ``(xyz float64, scalar_class or None)`` from a PLY via open3d."""
    import open3d as o3d

    cloud = o3d.t.io.read_point_cloud(str(path))
    xyz = cloud.point.positions.numpy().astype(np.float64)
    classes = None
    if "scalar_class" in cloud.point:
        classes = cloud.point["scalar_class"].numpy().reshape(-1).astype(np.uint8)
    return xyz, classes


# --------------------------------------------------------------------------- density

def compute_density(xyz: np.ndarray, *, cell: float = 1.0) -> dict[str, Any]:
    """Points per m^2 on a 1 m XY grid plus the gap share inside the convex hull.

    Counting uses all points.  The footprint is the convex hull of at most ``HULL_MAX_POINTS``
    points (deterministic stride); a cell belongs to it when its centre lies in the hull or it
    holds points.  ``gap_share`` = empty cells / footprint cells.  Density statistics are over
    occupied cells.  A convex hull overestimates concave footprints, so gap_share is an upper bound.
    """
    import shapely
    from shapely.geometry import MultiPoint

    n = len(xyz)
    if n < 10:
        return unavailable("fewer than 10 points")
    lo = np.floor(xyz[:, :2].min(axis=0) / cell) * cell
    hi = xyz[:, :2].max(axis=0)
    nx, ny = (np.floor((hi - lo) / cell).astype(np.int64) + 1)
    if nx * ny > MAX_GRID_CELLS:
        return unavailable(f"grid too large ({nx}x{ny} cells > {MAX_GRID_CELLS})")
    counts = np.zeros(int(nx * ny), dtype=np.int64)
    for start in range(0, n, CHUNK):
        ij = np.floor((xyz[start:start + CHUNK, :2] - lo) / cell).astype(np.int64)
        ij[:, 0] = np.clip(ij[:, 0], 0, nx - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, ny - 1)
        counts += np.bincount(ij[:, 0] * ny + ij[:, 1], minlength=counts.size)
    idx, step = stride_subsample(n, HULL_MAX_POINTS)
    hull = MultiPoint(xyz[idx, :2]).convex_hull
    gx, gy = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * cell, lo[1] + (np.arange(ny) + 0.5) * cell, indexing="ij")
    inside = shapely.contains_xy(hull, gx.ravel(), gy.ravel())
    footprint = inside | (counts > 0)
    occupied = counts[counts > 0]
    total = int(footprint.sum())
    empty = int((footprint & (counts == 0)).sum())
    area = cell * cell
    return {
        "available": True, "num_points": int(n), "cell_m": cell,
        "points_per_m2_median": float(np.median(occupied) / area),
        "points_per_m2_p10": float(np.percentile(occupied, 10) / area),
        "points_per_m2_p90": float(np.percentile(occupied, 90) / area),
        "footprint_cells": total, "empty_cells": empty, "gap_share": empty / total if total else None,
        "footprint": "convex hull of points (upper bound for gaps)", "hull_subsample_stride": step,
    }


# --------------------------------------------------------------------------- planes

def fit_plane_ransac(points: np.ndarray, rng: np.random.Generator, *, threshold: float = RANSAC_THRESHOLD_M,
                     iterations: int = RANSAC_ITERATIONS) -> tuple[np.ndarray, float, float] | None:
    """RANSAC plane + least-squares refit on inliers -> (unit normal, rms_distance, inlier_ratio)."""
    n = len(points)
    if n < 3:
        return None
    tri = rng.integers(0, n, size=(iterations, 3))
    p0, p1, p2 = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(normals, axis=1)
    ok = norms > 1e-12
    if not ok.any():
        return None
    normals = normals[ok] / norms[ok, None]
    offsets = -(normals * p0[ok]).sum(axis=1)
    dist = np.abs(points @ normals.T + offsets)
    best = int(np.argmax((dist < threshold).sum(axis=0)))
    inliers = dist[:, best] < threshold
    normal = normals[best]
    for _ in range(3):  # refit + re-evaluate
        if inliers.sum() < 3:
            return None
        sel = points[inliers]
        centroid = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel - centroid, full_matrices=False)
        normal = vt[2]
        inliers = np.abs((points - centroid) @ normal) < threshold
    sel = points[inliers]
    if len(sel) < 3:
        return None
    centroid = sel.mean(axis=0)
    d = (sel - centroid) @ normal
    return normal, float(np.sqrt((d ** 2).mean())), float(len(sel) / n)


def _patch_stats(points: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    voxel = 2.0 * PATCH_RADIUS_M
    keys = np.floor((points - points.min(axis=0)) / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first = np.sort(first)
    if len(first) > MAX_PATCHES_PER_CLASS:
        first = np.sort(rng.choice(first, MAX_PATCHES_PER_CLASS, replace=False))
    tree = cKDTree(points)
    balls = tree.query_ball_point(points[first], PATCH_RADIUS_M)
    rms_all, rms_roof, rms_facade, rejected = [], [], [], 0
    for ball in balls:
        if len(ball) < PATCH_MIN_POINTS:
            rejected += 1
            continue
        if len(ball) > PATCH_MAX_POINTS:
            ball = ball[:: -(-len(ball) // PATCH_MAX_POINTS)]
        fit = fit_plane_ransac(points[ball], rng)
        if fit is None or fit[2] < MIN_INLIER_RATIO:
            rejected += 1
            continue
        normal, rms, _ = fit
        rms_all.append(rms)
        if abs(normal[2]) > ROOF_NZ:
            rms_roof.append(rms)
        elif abs(normal[2]) < FACADE_NZ:
            rms_facade.append(rms)

    def summ(values: list[float]) -> dict[str, Any]:
        return {"patches": len(values), "rms_m_median": float(np.median(values)) if values else None}

    return {**summ(rms_all), "rejected_patches": rejected, "roof": summ(rms_roof), "facade": summ(rms_facade)}


def compute_plane_noise(xyz: np.ndarray, classes: np.ndarray,
                        class_codes: dict[str, int] | None = None) -> dict[str, Any]:
    """Median RANSAC plane RMS (m) over 0.5 m patches (>=50 points) for building/road/ground.

    Per class at most ``PLANE_MAX_POINTS_PER_CLASS`` points (stride) are used; the stride is
    reported.  Inlier threshold 5 cm, patches with <60 % inliers are rejected.  Building
    patches are additionally split by fitted normal into roof (|nz|>0.5) and facade (|nz|<0.3).
    """
    ids = class_codes or CLASS_CODES
    rng = np.random.default_rng(SEED)
    result: dict[str, Any] = {"available": True, "patch_radius_m": PATCH_RADIUS_M,
                              "ransac_threshold_m": RANSAC_THRESHOLD_M, "classes": {}}
    any_ok = False
    for name, cid in ids.items():
        pts = xyz[classes == cid]
        if len(pts) < PATCH_MIN_POINTS:
            result["classes"][name] = unavailable(f"only {len(pts)} points of class {cid}")
            continue
        idx, step = stride_subsample(len(pts), PLANE_MAX_POINTS_PER_CLASS)
        stats = _patch_stats(pts[idx], rng)
        stats.update(available=stats["patches"] > 0, subsample_stride=step, num_points=int(len(pts)))
        if not stats["available"]:
            stats["reason"] = "no patch with >=50 points and a planar fit"
        any_ok = any_ok or stats["available"]
        result["classes"][name] = stats
    if not any_ok:
        return unavailable("no class produced a valid plane patch")
    return result


# --------------------------------------------------------------------------- facades

def compute_facade_coverage(xyz: np.ndarray, classes: np.ndarray,
                            building_code: int = BUILDING) -> dict[str, Any]:
    """Share of building points with |nz| < 0.3 (normals: open3d KNN 16, stride subsample)."""
    import open3d as o3d

    pts = xyz[classes == building_code]
    if len(pts) < 20:
        return unavailable(f"only {len(pts)} building points")
    idx, step = stride_subsample(len(pts), NORMAL_MAX_POINTS)
    sub = pts[idx]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub - sub.mean(axis=0)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
    nz = np.abs(np.asarray(pc.normals)[:, 2])
    return {"available": True, "building_points": int(len(pts)), "facade_share": float((nz < FACADE_NZ).mean()),
            "roof_share": float((nz > ROOF_NZ).mean()), "subsample_stride": step}


# --------------------------------------------------------------------------- driver

def compute_quality(products_dir: Path | str, *, run_name: str,
                    classes_ply: Path | str | None = None) -> dict[str, Any]:
    """Measure all quality blocks for one run; missing inputs yield ``unavailable`` blocks."""
    products = Path(products_dir)
    metrics: dict[str, Any] = {"run": run_name, "products_dir": str(products)}

    recon, reason = _load_reconstruction(products)
    if recon is None:
        metrics["sparse"] = unavailable(reason or "no sparse model")
        metrics["rtk"] = unavailable(reason or "no sparse model")
    else:
        metrics["sparse"] = compute_sparse_metrics(recon, _read_input_count(products))
        gps_ref = products / "gps_ref.txt"
        if not gps_ref.is_file():
            metrics["rtk"] = unavailable("products/gps_ref.txt not found")
        else:
            centres = {im.name: np.asarray(im.projection_center(), dtype=float)
                       for im in recon.images.values() if im.has_pose}
            metrics["rtk"] = compute_rtk_residuals(centres, _read_gps_ref(gps_ref))

    classified = Path(classes_ply) if classes_ply else products / "classified" / "classified.ply"
    with tempfile.TemporaryDirectory(prefix="quality_") as tmp:
        dense_source = None
        for candidate in (products / "dense" / "fused.ply", products / "dense" / "fused.ply.gz", classified):
            if candidate.is_file():
                dense_source = candidate
                break
        if dense_source is None:
            metrics["density"] = unavailable("no fused.ply or classified.ply found")
        else:
            try:
                xyz, _ = load_cloud(_ensure_plain(dense_source, Path(tmp)))
                metrics["density"] = compute_density(xyz)
                metrics["density"]["source"] = dense_source.name
            except Exception as exc:
                metrics["density"] = unavailable(f"cloud unreadable: {exc}")
        if not classified.is_file():
            metrics["plane_noise"] = unavailable(f"{classified.name} not found")
            metrics["facade_coverage"] = unavailable(f"{classified.name} not found")
        else:
            try:
                pointset = load_classified_pointset(products, ply_path=classified, load_scores=False)
                xyz, classes = pointset.xyz, pointset.classes
                metrics["plane_noise"] = compute_plane_noise(xyz, classes)
                metrics["facade_coverage"] = compute_facade_coverage(xyz, classes)
            except Exception as exc:
                metrics["plane_noise"] = unavailable(f"classified cloud unusable: {exc}")
                metrics["facade_coverage"] = unavailable(f"classified cloud unusable: {exc}")
    return metrics


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_quality_report(metrics: dict[str, Any], out_dir: Path | str) -> Path:
    """Write ``quality_<run>.json`` into ``out_dir`` and return its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"quality_{metrics.get('run', 'run')}.json"
    path.write_text(json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _get(metrics: dict[str, Any], *keys: str) -> Any:
    node: Any = metrics
    for key in keys:
        if not isinstance(node, dict) or node.get("available") is False or key not in node:
            return None
        node = node[key]
    return node


def _fmt(value: Any, digits: int = 3, scale: float = 1.0) -> str:
    return "n/a" if value is None else f"{value * scale:.{digits}f}"


_ROWS: list[tuple[str, tuple[str, ...], int, float]] = [
    ("Registered ratio", ("sparse", "registered_ratio"), 3, 1.0),
    ("Registered images", ("sparse", "registered_images"), 0, 1.0),
    ("Sparse 3D points", ("sparse", "num_points3D"), 0, 1.0),
    ("Reproj. error mean [px]", ("sparse", "reprojection_error_mean_px"), 3, 1.0),
    ("Reproj. error median [px]", ("sparse", "reprojection_error_median_px"), 3, 1.0),
    ("Track length mean", ("sparse", "track_length_mean"), 2, 1.0),
    ("Track length median", ("sparse", "track_length_median"), 1, 1.0),
    ("RTK RMS E raw [cm]", ("rtk", "raw", "E", "rms"), 1, 100.0),
    ("RTK RMS N raw [cm]", ("rtk", "raw", "N", "rms"), 1, 100.0),
    ("RTK RMS U raw [cm]", ("rtk", "raw", "U", "rms"), 1, 100.0),
    ("RTK RMS E fit [cm]", ("rtk", "after_similarity", "E", "rms"), 1, 100.0),
    ("RTK RMS N fit [cm]", ("rtk", "after_similarity", "N", "rms"), 1, 100.0),
    ("RTK RMS U fit [cm]", ("rtk", "after_similarity", "U", "rms"), 1, 100.0),
    ("Density median [pts/m2]", ("density", "points_per_m2_median"), 1, 1.0),
    ("Density p10 [pts/m2]", ("density", "points_per_m2_p10"), 1, 1.0),
    ("Density p90 [pts/m2]", ("density", "points_per_m2_p90"), 1, 1.0),
    ("Gap share [%]", ("density", "gap_share"), 2, 100.0),
    ("Plane RMS building [cm]", ("plane_noise", "classes", "building", "rms_m_median"), 2, 100.0),
    ("Plane RMS roof [cm]", ("plane_noise", "classes", "building", "roof", "rms_m_median"), 2, 100.0),
    ("Plane RMS facade [cm]", ("plane_noise", "classes", "building", "facade", "rms_m_median"), 2, 100.0),
    ("Plane RMS road [cm]", ("plane_noise", "classes", "road", "rms_m_median"), 2, 100.0),
    ("Plane RMS ground [cm]", ("plane_noise", "classes", "ground", "rms_m_median"), 2, 100.0),
    ("Facade share [%]", ("facade_coverage", "facade_share"), 1, 100.0),
]


def compare_runs(runs: Iterable[dict[str, Any]]) -> str:
    """Markdown table with the main metrics of several runs side by side."""
    runs = list(runs)
    header = "| Metric | " + " | ".join(str(r.get("run", "?")) for r in runs) + " |"
    lines = [header, "|" + "---|" * (len(runs) + 1)]
    for label, keys, digits, scale in _ROWS:
        cells = [_fmt(_get(r, *keys), digits, scale) for r in runs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: measure one run and/or compare existing quality JSON files."""
    parser = argparse.ArgumentParser(description="Photogrammetry quality metrics")
    parser.add_argument("--products", type=Path)
    parser.add_argument("--run", default="run")
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--classes-ply", type=Path)
    parser.add_argument("--compare", nargs="*", type=Path, default=[])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    runs: list[dict[str, Any]] = []
    if args.products:
        metrics = compute_quality(args.products, run_name=args.run, classes_ply=args.classes_ply)
        LOGGER.info("wrote %s", write_quality_report(metrics, args.out))
        runs.append(metrics)
    runs.extend(json.loads(p.read_text(encoding="utf-8")) for p in args.compare)
    if len(runs) > 1 or args.compare:
        table = compare_runs(runs)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "quality_compare.md").write_text(table, encoding="utf-8")
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
