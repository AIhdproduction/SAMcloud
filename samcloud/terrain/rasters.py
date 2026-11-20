"""DTM / DSM rasters with validity band, GeoTIFF + hillshade export.

Raster convention: arrays are ``(rows, cols)`` and north-up, i.e. row 0 is the
northern edge (``ymax`` of the bounds) and column 0 the western edge (``xmin``).
Bounds are local metric ``(xmin, ymin, xmax, ymax)``.

Validity bands (uint8): 0 not determinable (NaN), 1 measured, 2 interpolated /
hole-filled.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator

from samcloud.drone.contracts import (
    CURB, GROUND, LOW_VEGETATION, MANHOLE, PATH, ROAD, ROAD_MARKING, SEALED_TERRAIN,
    SIDEWALK, LocalFrame, PointSet, load_drone_config,
)
from samcloud.terrain.csf import cloth_simulation_filter

logger = logging.getLogger(__name__)

GROUND_CANDIDATE_CLASSES: tuple[int, ...] = (
    GROUND, ROAD, SIDEWALK, PATH, LOW_VEGETATION, CURB, MANHOLE, ROAD_MARKING, SEALED_TERRAIN,
)
VALIDITY_NONE, VALIDITY_MEASURED, VALIDITY_FILLED = 0, 1, 2
_MIN_POINTS_PER_CELL = 4
_MAX_HOLE_CELLS = 2


# ----------------------------------------------------------------------------------------------
# ground filtering
# ----------------------------------------------------------------------------------------------

def _pmf_mask(xyz: np.ndarray, *, cell_size: float = 0.5, threshold_m: float = 0.15,
              percentile: float = 3.0, **pmf: float) -> np.ndarray:
    """Ground mask via the existing progressive morphological filter (raster based)."""
    from samcloud.engine.geometry.terrain import build_dtm, sample_dtm

    all_points = np.ones(len(xyz), dtype=bool)
    dtm, origin, cell = build_dtm(xyz, all_points, cell_size, percentile, 1, 0.0, method="pmf", pmf=pmf)
    xy = xyz[:, :2]
    dz = xyz[:, 2] - sample_dtm(dtm, origin, cell, xy)
    gy, gx = np.gradient(dtm, cell)
    slope = np.hypot(gx, gy)
    ix = np.clip(np.floor((xy[:, 0] - origin[0]) / cell).astype(np.int64), 0, dtm.shape[1] - 1)
    iy = np.clip(np.floor((xy[:, 1] - origin[1]) / cell).astype(np.int64), 0, dtm.shape[0] - 1)
    tolerance = threshold_m + np.minimum(slope[iy, ix], 1.7) * cell
    return np.abs(dz) <= tolerance


def ground_mask(ps: PointSet, method: str = "pmf", **kw) -> np.ndarray:
    """Boolean ground mask over all points of ``ps``.

    Only points of the classes ground/road/sidewalk/path/low vegetation/curb/manhole/
    marking/sealed terrain are candidates; buildings, trees, vehicles etc. are never ground.
    ``method`` is ``"pmf"`` or ``"csf"``; ``kw`` goes to the respective filter.
    """
    if method not in ("pmf", "csf"):
        raise ValueError(f"unknown ground filter {method!r} (expected 'pmf' or 'csf')")
    mask = np.zeros(len(ps), dtype=bool)
    candidates = np.flatnonzero(np.isin(ps.classes, GROUND_CANDIDATE_CLASSES))
    if len(candidates) == 0:
        return mask
    xyz = np.asarray(ps.xyz, dtype=np.float64)[candidates]
    if method == "csf":
        mask[candidates] = cloth_simulation_filter(xyz, **kw)
    else:
        mask[candidates] = _pmf_mask(xyz, **kw)
    return mask


# ----------------------------------------------------------------------------------------------
# gridding helpers
# ----------------------------------------------------------------------------------------------

def _grid_shape(bounds: tuple[float, float, float, float], resolution: float) -> tuple[int, int]:
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    xmin, ymin, xmax, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("bounds must have positive extent")
    return max(1, int(np.ceil((ymax - ymin) / resolution - 1e-9))), max(1, int(np.ceil((xmax - xmin) / resolution - 1e-9)))


def _cell_of(xyz: np.ndarray, bounds, resolution: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Flat cell index (row-major, north-up) per point and an in-bounds mask."""
    xmin, _ymin, _xmax, ymax = bounds
    col = np.floor((xyz[:, 0] - xmin) / resolution).astype(np.int64)
    row = np.floor((ymax - xyz[:, 1]) / resolution).astype(np.int64)
    inside = (col >= 0) & (col < shape[1]) & (row >= 0) & (row < shape[0])
    return row * shape[1] + col, inside


def _grouped_rank_values(flat: np.ndarray, z: np.ndarray, rank_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort points by (cell, z); return unique cells, group starts and counts plus sorted z."""
    order = np.lexsort((z, flat))
    flat_sorted = flat[order]
    z_sorted = z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat_sorted)) + 1]
    counts = np.diff(np.r_[starts, len(flat_sorted)])
    return flat_sorted[starts], rank_fn(starts, counts, z_sorted), counts


def _cell_median(flat: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def rank(starts, counts, z_sorted):
        low = z_sorted[starts + (counts - 1) // 2]
        high = z_sorted[starts + counts // 2]
        return 0.5 * (low + high)
    cells, values, _ = _grouped_rank_values(flat, z, rank)
    return cells, values


def _cell_percentile(flat: np.ndarray, z: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    def rank(starts, counts, z_sorted):
        idx = np.rint(percentile / 100.0 * (counts - 1)).astype(np.int64)
        return z_sorted[starts + idx]
    cells, values, _ = _grouped_rank_values(flat, z, rank)
    return cells, values


# ----------------------------------------------------------------------------------------------
# rasters
# ----------------------------------------------------------------------------------------------

def build_dtm_raster(
    xyz_ground: np.ndarray, *, resolution: float,
    bounds: tuple[float, float, float, float], max_interp_distance: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise ground points to ``(dtm, validity, dist)``.

    Cell value = median ground Z (validity 1).  Empty cells inside the convex hull of the
    measured cells and within ``max_interp_distance`` of a measured cell are linearly
    interpolated (validity 2); all others are NaN (validity 0).  ``dist`` is the distance in
    metres to the nearest measured cell (0 on measured cells, inf if nothing is measured).
    """
    shape = _grid_shape(bounds, resolution)
    pts = np.asarray(xyz_ground, dtype=np.float64).reshape(-1, 3)
    dtm = np.full(shape, np.nan, dtype=np.float32)
    validity = np.zeros(shape, dtype=np.uint8)
    dist = np.full(shape, np.inf, dtype=np.float32)
    if len(pts) == 0:
        return dtm, validity, dist
    flat, inside = _cell_of(pts, bounds, resolution, shape)
    if not inside.any():
        return dtm, validity, dist
    cells, values = _cell_median(flat[inside], pts[inside, 2])
    dtm.ravel()[cells] = values
    validity.ravel()[cells] = VALIDITY_MEASURED
    measured = validity == VALIDITY_MEASURED
    dist = (ndimage.distance_transform_edt(~measured) * resolution).astype(np.float32)

    todo = (~measured) & (dist <= max_interp_distance)
    if todo.any() and measured.sum() >= 3:
        rows, cols = np.nonzero(measured)
        src = np.column_stack([cols, rows]).astype(np.float64)
        trow, tcol = np.nonzero(todo)
        try:
            interp = LinearNDInterpolator(src, dtm[rows, cols].astype(np.float64))
            est = interp(np.column_stack([tcol, trow]).astype(np.float64))
        except Exception as exc:  # degenerate geometry (collinear cells)
            logger.warning("DTM interpolation skipped: %s", exc)
            est = np.full(len(trow), np.nan)
        ok = np.isfinite(est)
        dtm[trow[ok], tcol[ok]] = est[ok]
        validity[trow[ok], tcol[ok]] = VALIDITY_FILLED
    return dtm, validity, dist


def build_dsm_raster(
    xyz: np.ndarray, *, resolution: float,
    bounds: tuple[float, float, float, float], percentile: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise all points to ``(dsm, validity)`` using a high percentile (99 = near max) per cell.

    Holes of at most 2 connected empty cells are filled with the mean of their measured
    neighbours (validity 2).
    """
    shape = _grid_shape(bounds, resolution)
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    dsm = np.full(shape, np.nan, dtype=np.float32)
    validity = np.zeros(shape, dtype=np.uint8)
    if len(pts) == 0:
        return dsm, validity
    flat, inside = _cell_of(pts, bounds, resolution, shape)
    if not inside.any():
        return dsm, validity
    cells, values = _cell_percentile(flat[inside], pts[inside, 2], percentile)
    dsm.ravel()[cells] = values
    validity.ravel()[cells] = VALIDITY_MEASURED

    empty = validity == VALIDITY_NONE
    labels, count = ndimage.label(empty, structure=np.ones((3, 3), dtype=int))
    if count:
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        small = (sizes[labels] <= _MAX_HOLE_CELLS) & empty
        if small.any():
            kernel = np.ones((3, 3))
            known = validity == VALIDITY_MEASURED
            total = ndimage.convolve(np.where(known, dsm, 0.0).astype(np.float64), kernel, mode="constant")
            number = ndimage.convolve(known.astype(np.float64), kernel, mode="constant")
            fill = small & (number > 0)
            dsm[fill] = (total[fill] / number[fill]).astype(np.float32)
            validity[fill] = VALIDITY_FILLED
    return dsm, validity


def _density_resolution(xyz: np.ndarray, target: float) -> float:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return float(target)
    ij = np.floor(pts[:, :2]).astype(np.int64)
    _, counts = np.unique(ij, axis=0, return_counts=True)
    density = float(np.median(counts))                      # points per m2 (1 m cells)
    if density <= 0:
        return float(target)
    needed = float(np.sqrt(_MIN_POINTS_PER_CELL / density))
    return float(max(target, np.ceil(needed * 100.0) / 100.0))


def choose_resolution(xyz: np.ndarray, target_dsm: float = 0.10, target_dtm: float = 0.25) -> tuple[float, float]:
    """Return ``(dsm_resolution, dtm_resolution)`` in metres.

    A cell must hold at least 4 points for the median occupied 1 m cell, and the result is
    never finer than the target.  Pass the cloud the raster will be built from.
    """
    return _density_resolution(xyz, target_dsm), _density_resolution(xyz, target_dtm)


# ----------------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------------

def _crs_for(frame: LocalFrame):
    from rasterio.crs import CRS

    if frame.epsg == 2056 and frame.vertical_datum.upper() == "LN02":
        try:
            from pyproj import CRS as ProjCRS

            return CRS.from_wkt(ProjCRS.from_user_input("EPSG:2056+5728").to_wkt()), "compound"
        except Exception as exc:
            logger.warning("compound CRS EPSG:2056+5728 not accepted (%s), using EPSG:2056", exc)
    return CRS.from_epsg(frame.epsg), "horizontal"


def write_geotiff(
    path: str | Path, array: np.ndarray, *, bounds_local: tuple[float, float, float, float],
    resolution: float, frame: LocalFrame, nodata: float | None = -9999,
    add_frame_height: bool = True,
) -> Path:
    """Write a north-up single-band GeoTIFF in LV95 (EPSG:2056, LN02 heights).

    The upper-left corner is ``frame.to_lv95((xmin, ymax, 0))``.  CRS is the compound
    EPSG:2056+5728 when GDAL accepts it, otherwise EPSG:2056; the tag ``VERTICAL_DATUM``
    (``LN02``) is always written.  Float arrays are height rasters: with ``add_frame_height``
    the frame's LN02 origin height is added to the local values.  They become float32 (NaN ->
    ``nodata``); integer
    arrays (validity) are written as uint8 without nodata - validity is a separate file.
    LZW compressed, tiled.
    """
    import rasterio
    from rasterio.transform import from_origin

    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError("array must be 2-D")
    xmin, _ymin, _xmax, ymax = bounds_local
    east, north, _ = frame.to_lv95(np.array([xmin, ymax, 0.0]))
    transform = from_origin(float(east), float(north), resolution, resolution)
    crs, crs_kind = _crs_for(frame)
    is_float = np.issubdtype(arr.dtype, np.floating)
    if is_float:
        shifted = arr.astype(np.float64) + (frame.origin_lv95[2] if add_frame_height else 0.0)
        data = np.where(np.isfinite(arr), shifted, nodata).astype(np.float32)
        dtype, node = "float32", nodata
    else:
        data = arr.astype(np.uint8)
        dtype, node = "uint8", None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=dtype,
                   crs=crs, transform=transform, nodata=node, compress="lzw")
    if data.shape[0] >= 16 and data.shape[1] >= 16:
        profile.update(tiled=True, blockxsize=16, blockysize=16)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(VERTICAL_DATUM=frame.vertical_datum, HORIZONTAL_EPSG=str(frame.epsg), CRS_KIND=crs_kind)
    return out


def hillshade_png(dtm: np.ndarray, resolution: float, path: str | Path,
                  azimuth: float = 315.0, altitude: float = 45.0) -> Path:
    """Write a Horn hillshade of ``dtm`` (north-up) as RGBA PNG; NaN cells are transparent."""
    from PIL import Image

    z = np.asarray(dtm, dtype=np.float64)
    valid = np.isfinite(z)
    filled = z.copy()
    if valid.any():
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = z[idx[0], idx[1]]
    else:
        filled[:] = 0.0
    p = np.pad(filled, 1, mode="edge")
    dzdx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])) / (8 * resolution)
    dzdy = ((p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]) - (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])) / (8 * resolution)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az = np.deg2rad(360.0 - azimuth + 90.0)
    alt = np.deg2rad(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    gray = np.clip(shade, 0.0, 1.0) * 255.0
    rgba = np.zeros(z.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = gray.astype(np.uint8)[..., None]
    rgba[..., 3] = np.where(valid, 255, 0)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out)
    return out


def _snapped_bounds(xyz: np.ndarray, resolution: float) -> tuple[float, float, float, float]:
    lo = xyz[:, :2].min(axis=0)
    hi = xyz[:, :2].max(axis=0)
    xmin, ymin = np.floor(lo / resolution) * resolution
    xmax, ymax = np.ceil(hi / resolution) * resolution
    if xmax <= xmin:
        xmax = xmin + resolution
    if ymax <= ymin:
        ymax = ymin + resolution
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _validity_shares(validity: np.ndarray) -> dict:
    total = max(1, validity.size)
    return {"measured": float((validity == 1).sum() / total),
            "interpolated": float((validity == 2).sum() / total),
            "undetermined": float((validity == 0).sum() / total)}


def export_terrain(ps: PointSet, out_dir: str | Path, frame: LocalFrame, config: dict | None = None) -> dict:
    """Ground filter -> DTM/DSM -> GeoTIFFs, hillshade and ``terrain_report.json`` in ``out_dir``.

    Files: ``dtm.tif``, ``dtm_validity.tif``, ``dsm.tif``, ``dsm_validity.tif``,
    ``hillshade.png``, ``terrain_report.json``.  The filter comes from
    ``terrain.ground_filter`` (``pmf``/``csf``/``auto``; auto compares both and picks the best).
    Returns the report dict plus a ``files`` mapping.
    """
    from samcloud.terrain.compare import compare_ground_filters, pick_best

    cfg = config if config is not None else load_drone_config()
    tcfg = cfg.get("terrain", {})
    csf_cfg = tcfg.get("csf", {})
    csf_kw = {
        "cloth_resolution": float(csf_cfg.get("cloth_resolution_m", 0.5)),
        "rigidness": int(csf_cfg.get("rigidness", 2)),
        "class_threshold": float(csf_cfg.get("class_threshold_m", 0.15)),
        "iterations": int(csf_cfg.get("iterations", 500)),
        "time_step": float(csf_cfg.get("time_step", 0.65)),
    }
    if len(ps) == 0:
        raise ValueError("PointSet is empty")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(ps.xyz, dtype=np.float64)
    timings: dict[str, float] = {}

    def timed(name, fn):
        start = time.perf_counter()
        result = fn()
        timings[name] = round(time.perf_counter() - start, 3)
        return result

    requested = str(tcfg.get("ground_filter", "auto")).lower()
    filter_info: dict = {"requested": requested}
    if requested == "auto":
        masks = {"pmf": timed("ground_pmf", lambda: ground_mask(ps, "pmf")),
                 "csf": timed("ground_csf", lambda: ground_mask(ps, "csf", **csf_kw))}
        metrics = compare_ground_filters(ps, masks)
        used, reason = pick_best(metrics)
        mask = masks[used]
        filter_info.update(metrics=metrics, reason=reason)
    elif requested in ("pmf", "csf"):
        used = requested
        mask = timed(f"ground_{used}", lambda: ground_mask(ps, used, **(csf_kw if used == "csf" else {})))
    else:
        raise ValueError(f"terrain.ground_filter must be auto|pmf|csf, got {requested!r}")
    filter_info["used"] = used
    ground = xyz[mask]
    if len(ground) == 0:
        raise ValueError("ground filter found no ground points; cannot build a DTM")

    dsm_target = float(tcfg.get("dsm_resolution_m", 0.10))
    dtm_target = float(tcfg.get("dtm_resolution_m", 0.25))
    dsm_res = choose_resolution(xyz, dsm_target, dtm_target)[0]
    dtm_res = choose_resolution(ground, dsm_target, dtm_target)[1]
    dtm_bounds = _snapped_bounds(xyz, dtm_res)
    dsm_bounds = _snapped_bounds(xyz, dsm_res)

    dtm, dtm_valid, _dist = timed("dtm", lambda: build_dtm_raster(ground, resolution=dtm_res, bounds=dtm_bounds))
    dsm, dsm_valid = timed("dsm", lambda: build_dsm_raster(xyz, resolution=dsm_res, bounds=dsm_bounds))

    files = {"dtm": out / "dtm.tif", "dtm_validity": out / "dtm_validity.tif",
             "dsm": out / "dsm.tif", "dsm_validity": out / "dsm_validity.tif",
             "hillshade": out / "hillshade.png", "report": out / "terrain_report.json"}

    def write_all():
        write_geotiff(files["dtm"], dtm, bounds_local=dtm_bounds, resolution=dtm_res, frame=frame)
        write_geotiff(files["dtm_validity"], dtm_valid, bounds_local=dtm_bounds, resolution=dtm_res, frame=frame)
        write_geotiff(files["dsm"], dsm, bounds_local=dsm_bounds, resolution=dsm_res, frame=frame)
        write_geotiff(files["dsm_validity"], dsm_valid, bounds_local=dsm_bounds, resolution=dsm_res, frame=frame)
        hillshade_png(dtm, dtm_res, files["hillshade"])

    timed("write", write_all)
    report = {
        "filter": filter_info,
        "ground_points": int(mask.sum()),
        "total_points": int(len(ps)),
        "dtm": {"resolution_m": dtm_res, "bounds_local": list(dtm_bounds), "shape": list(dtm.shape),
                "validity": _validity_shares(dtm_valid)},
        "dsm": {"resolution_m": dsm_res, "bounds_local": list(dsm_bounds), "shape": list(dsm.shape),
                "validity": _validity_shares(dsm_valid)},
        "crs": {"epsg": frame.epsg, "vertical_datum": frame.vertical_datum,
                "validity_band": "separate uint8 GeoTIFF: 0 none, 1 measured, 2 interpolated"},
        "timings_s": timings,
    }
    files["report"].write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    report["files"] = {k: str(v) for k, v in files.items()}
    return report
