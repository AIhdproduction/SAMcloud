"""Terrain builder: labelled TIN -> ``TerrainElement`` (without pavement) + DXF TIN, contours, spot heights.

The terrain element is the labelled TIN *without* road / sidewalk / path triangles (lawn stays part
of the terrain, it is a classification only). Because pavement and terrain come from one TIN they
share the boundary vertices, so there is no gap and no overlap.

DXF: the full TIN as 3DFACE on S-TOPO-TIN (the writer freezes that layer), contours (SPLINE) on
S-TOPO-MAJR / S-TOPO-MINR restricted to the measured part of the DTM, spot heights on S-ELEV.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import shapely
from scipy import ndimage

from samcloud.bim.core.model import TerrainElement, TerrainParams
from samcloud.bim.core.model.enums import ParamOrigin, Status
from samcloud.drone import contracts as c

LOGGER = logging.getLogger(__name__)

PAVEMENT_LABELS = ("road", "sidewalk", "path")
TIN_LAYER = "S-TOPO-TIN"
MAJOR_LAYER = "S-TOPO-MAJR"
MINOR_LAYER = "S-TOPO-MINR"
SPOT_LAYER = "S-ELEV"
MEASURED = 1  # validity code of measured DTM cells
MIN_TRIANGLE_AREA_M2 = 1e-6
MIN_TRIANGLE_ANGLE_DEG = 1.0
AUTO_NOISE_MAX_M = 0.05
AUTO_SLOPE_MAX = 0.10


# ---------------------------------------------------------------------------------- TIN helpers


def triangle_quality(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3D area (m2) and minimum angle (deg) per triangle."""
    a, b, cc = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1)
    la, lb, lc = np.linalg.norm(b - cc, axis=1), np.linalg.norm(a - cc, axis=1), np.linalg.norm(a - b, axis=1)
    angles = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for opp, s1, s2 in ((la, lb, lc), (lb, la, lc), (lc, la, lb)):
            angles.append(np.degrees(np.arccos(np.clip((s1**2 + s2**2 - opp**2) / (2 * s1 * s2), -1.0, 1.0))))
    return area, np.nan_to_num(np.min(np.stack(angles), axis=0), nan=0.0)


def _orient_ccw(vertices: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a, b, cc = (vertices[tris[:, i], :2] for i in range(3))
    area2 = (b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0])
    out = tris.copy()
    out[area2 < 0] = tris[area2 < 0][:, [0, 2, 1]]
    return out


class TinSampler:
    """Linear height interpolation on a triangle mesh (NaN outside)."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self._tree = shapely.STRtree(shapely.polygons(self.vertices[self.triangles][:, :, :2])) if len(self.triangles) else None

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Heights at ``xy`` (N,2); NaN where no triangle covers the point."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        z = np.full(len(xy), np.nan)
        if self._tree is None or len(xy) == 0:
            return z
        pidx, tidx = self._tree.query(shapely.points(xy), predicate="intersects")
        _, first = np.unique(pidx, return_index=True)
        pidx, tidx = pidx[first], tidx[first]
        tri = self.vertices[self.triangles[tidx]]
        p = xy[pidx]
        v0, v1, v2 = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
        den = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
        ok = np.abs(den) > 1e-18
        w0 = np.where(ok, ((v1[:, 1] - v2[:, 1]) * (p[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (p[:, 1] - v2[:, 1])) / np.where(ok, den, 1), 0)
        w1 = np.where(ok, ((v2[:, 1] - v0[:, 1]) * (p[:, 0] - v2[:, 0]) + (v0[:, 0] - v2[:, 0]) * (p[:, 1] - v2[:, 1])) / np.where(ok, den, 1), 0)
        w2 = 1.0 - w0 - w1
        zz = w0 * tri[:, 0, 2] + w1 * tri[:, 1, 2] + w2 * tri[:, 2, 2]
        z[pidx[ok]] = zz[ok]
        return z


# ---------------------------------------------------------------------------------- contour interval


def choose_minor_interval(
    dtm: np.ndarray, validity: np.ndarray, resolution: float, *, config: dict | None = None
) -> tuple[float, dict]:
    """Minor contour interval from ``dxf.contour_minor_m`` (number, or ``"auto"``).

    ``auto``: 0.2 m if the DTM noise (RMS of the residual to a 1 m box-smoothed DTM on measured
    cells, validity == 1, evaluated where the 1 m window is completely measured) is < 0.05 m and the
    median slope is < 10 %, else 0.5 m. Returns ``(interval, info)``; ``info`` reports noise, slope
    and the reason so the choice can be logged and written to the element evidence. Deterministic.
    """
    cfg = config if config is not None else c.load_drone_config()
    setting = cfg.get("dxf", {}).get("contour_minor_m", "auto")
    if not (isinstance(setting, str) and setting.lower() == "auto"):
        value = float(setting)
        return value, {"mode": "configured", "minor_interval_m": value, "reason": "dxf.contour_minor_m is a fixed value"}
    res = float(resolution)
    z = np.where(np.asarray(validity) == MEASURED, np.asarray(dtm, dtype=np.float64), np.nan)
    valid = np.isfinite(z)
    info: dict = {"mode": "auto", "measured_cells": int(valid.sum())}
    if valid.sum() < 9:
        info.update(minor_interval_m=0.5, noise_rms_m=None, median_slope=None, reason="too few measured DTM cells")
        return 0.5, info
    size = max(3, int(round(1.0 / res)))
    w = valid.astype(np.float64)
    num = ndimage.uniform_filter(np.nan_to_num(z), size=size, mode="constant")
    den = ndimage.uniform_filter(w, size=size, mode="constant")
    smooth = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    full = valid & (den > 1.0 - 1e-9)
    use = full if full.sum() >= 9 else valid
    noise = float(np.sqrt(np.mean((z[use] - smooth[use]) ** 2)))
    gy, gx = np.gradient(smooth, res)
    slope = np.hypot(gx, gy)
    finite = np.isfinite(slope) & valid
    median_slope = float(np.median(slope[finite])) if finite.any() else None
    if median_slope is not None and noise < AUTO_NOISE_MAX_M and median_slope < AUTO_SLOPE_MAX:
        interval, reason = 0.2, f"noise {noise:.3f} m < {AUTO_NOISE_MAX_M} m and median slope {median_slope * 100:.1f} % < {AUTO_SLOPE_MAX * 100:.0f} %"
    else:
        slope_txt = "unknown" if median_slope is None else f"{median_slope * 100:.1f} %"
        interval, reason = 0.5, f"noise {noise:.3f} m or median slope {slope_txt} above the 0.2 m limits"
    info.update(minor_interval_m=interval, noise_rms_m=round(noise, 5),
                median_slope=None if median_slope is None else round(median_slope, 5), reason=reason)
    LOGGER.info("contour minor interval %.1f m (%s)", interval, reason)
    return interval, info


# ---------------------------------------------------------------------------------- builder


def build_terrain_elements(
    tin: c.LabeledTin,
    *,
    dtm: np.ndarray | None = None,
    validity: np.ndarray | None = None,
    resolution: float | None = None,
    config: dict | None = None,
) -> tuple[list[TerrainElement], list[c.QAEntry]]:
    """``TerrainElement`` from the TIN without road/sidewalk/path triangles (lawn stays).

    Degenerate triangles (area < 1e-6 m2 or minimum angle < 1 deg, the limits of qa_cad rule 10) are
    dropped and reported as ``QAEntry``; more than 5 % dropped sets the status to WARNING. When a
    DTM raster (north-up, ``validity`` 1 = measured) is given the contour interval choice is stored
    in the element evidence (``contour_minor_m``, ``dtm_noise_rms_m``, ``dtm_median_slope``).
    """
    cfg = config if config is not None else c.load_drone_config()
    qa: list[c.QAEntry] = []
    terrain = tin.without(*PAVEMENT_LABELS)
    vertices = np.asarray(terrain.vertices, dtype=np.float64)
    triangles = np.asarray(terrain.triangles, dtype=np.int64)
    if len(triangles) == 0:
        qa.append(c.QAEntry("terrain", "terrain", "TIN has no non-pavement triangles"))
        return [], qa
    area, min_angle = triangle_quality(vertices, triangles)
    bad = (area < MIN_TRIANGLE_AREA_M2) | (min_angle < MIN_TRIANGLE_ANGLE_DEG)
    if bad.any():
        qa.append(c.QAEntry("terrain", "terrain", f"{int(bad.sum())} degenerate terrain triangle(s) dropped",
                            {"dropped": int(bad.sum()), "total": int(len(triangles))}))
    tris = _orient_ccw(vertices, triangles[~bad])
    if len(tris) == 0:
        qa.append(c.QAEntry("terrain", "terrain", "all terrain triangles degenerate"))
        return [], qa
    used, inverse = np.unique(tris.ravel(), return_inverse=True)
    verts = vertices[used]
    tri_idx = inverse.reshape(tris.shape)
    evidence: dict = {
        "triangle_count": int(len(tri_idx)), "dropped_degenerate": int(bad.sum()),
        "lawn_triangles": int((np.asarray(terrain.labels)[~bad] == c.TIN_LABELS["lawn"]).sum()),
        "pavement_triangles_removed": int(len(tin.triangles) - len(terrain.triangles)),
        "area_m2": round(float(_xy_area(verts, tri_idx)), 3),
        "z_range_m": round(float(verts[:, 2].max() - verts[:, 2].min()), 3),
    }
    if dtm is not None and validity is not None and resolution is not None:
        interval, info = choose_minor_interval(dtm, validity, resolution, config=cfg)
        evidence["contour_minor_m"] = interval
        evidence["contour_choice"] = info.get("reason", "")
        if info.get("noise_rms_m") is not None:
            evidence["dtm_noise_rms_m"] = info["noise_rms_m"]
        if info.get("median_slope") is not None:
            evidence["dtm_median_slope"] = info["median_slope"]
    status = Status.WARNING if bad.sum() > 0.05 * len(triangles) else Status.OK
    element = TerrainElement(
        id="terrain-0", tag="terrain",
        params=TerrainParams(vertices=[tuple(map(float, v)) for v in verts],
                             triangles=[tuple(int(i) for i in t) for t in tri_idx]),
        param_origin={"vertices": ParamOrigin.FITTED, "triangles": ParamOrigin.FITTED},
        evidence=evidence, status=status,
    )
    element.metrics.point_count = int(len(verts))
    return [element], qa


def _xy_area(vertices: np.ndarray, tris: np.ndarray) -> float:
    a, b, cc = (vertices[tris[:, i], :2] for i in range(3))
    return float(0.5 * np.abs((b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0])).sum())


def category_counts(elements: Iterable) -> dict[str, int]:
    """``{"terrain": n}`` for ``qa_cad.count_consistency``."""
    return {"terrain": sum(1 for e in elements if isinstance(e, TerrainElement))}


# ---------------------------------------------------------------------------------- DXF


def _tin_faces(vertices: np.ndarray, triangles: np.ndarray, frame: c.LocalFrame) -> list[dict]:
    pts = frame.to_lv95(vertices[triangles])
    return [{"type": "3dface", "points": tri.tolist()} for tri in pts]


def _contours(
    dtm: np.ndarray, validity: np.ndarray, bounds: tuple[float, float, float, float], resolution: float,
    frame: c.LocalFrame, cfg: dict,
) -> tuple[dict[str, list[dict]], dict]:
    from samcloud.dxf.writer import contours_as_splines
    from samcloud.ortho.vector.contours import ContourSettings, contour_entities
    from samcloud.ortho.vector.raster_grid import RasterGrid

    dcfg = cfg.get("dxf", {})
    minor, info = choose_minor_interval(dtm, validity, resolution, config=cfg)
    major = float(dcfg.get("contour_major_m", 1.0))
    z = np.where(np.asarray(validity) == MEASURED, np.asarray(dtm, dtype=np.float64), np.nan)
    xmin, _ymin, _xmax, ymax = bounds
    grid = RasterGrid(min_x=float(xmin), max_y=float(ymax), pixel_size=float(resolution), width=z.shape[1], height=z.shape[0])
    settings = ContourSettings(resolution=float(resolution), major_interval=major, minor_interval=float(minor),
                               smoothing_sigma_m=0.0)
    raw = contour_entities(z, grid, settings, z_offset=float(frame.origin_lv95[2]))
    out: dict[str, list[dict]] = {}
    for ent in raw:
        pts = frame.to_lv95(np.asarray(ent["points"], dtype=np.float64))
        ent = {**ent, "points": pts.tolist()}
        layer = MAJOR_LAYER if ent.get("layer_role") == "major" else MINOR_LAYER
        out.setdefault(layer, []).append(ent)
    if bool(dcfg.get("spline_contours", True)):
        out = {layer: contours_as_splines(ents) for layer, ents in out.items()}
    info = {**info, "major_interval_m": major, "contour_count": int(len(raw))}
    return out, info


def _spots(
    sampler: TinSampler, frame: c.LocalFrame, spacing: float, curbs: Sequence, manholes: Sequence,
) -> list[dict]:
    from samcloud.ortho.vector.spots import SpotSettings, ground_grid_points, select_spots, spot_entities

    v = sampler.vertices
    bounds = (float(v[:, 0].min()), float(v[:, 1].min()), float(v[:, 0].max()), float(v[:, 1].max()))
    on_tin = lambda x, y: bool(np.isfinite(sampler.sample(np.array([[x, y]]))[0]))  # noqa: E731
    curb_xy = [(float(p[0]), float(p[1])) for a in curbs for p in np.asarray(getattr(a, "points", a))]
    mh_xy = [(float(m[0]), float(m[1])) for m in manholes]
    settings = SpotSettings(grid_spacing_m=float(spacing))
    groups = [("manhole", mh_xy), ("curb", curb_xy), ("grid", ground_grid_points(bounds, spacing, on_tin))]
    spots = select_spots(groups, settings)
    if not spots:
        return []
    heights = sampler.sample(np.array([[s.x, s.y] for s in spots]))
    absolute = heights + float(frame.origin_lv95[2])
    entities = spot_entities(spots, heights.tolist(), absolute.tolist(), settings)
    shift = np.asarray(frame.origin_lv95, dtype=np.float64)
    for ent in entities:
        for key in ("start", "end", "point"):
            if key in ent:
                ent[key] = (np.asarray(ent[key], dtype=np.float64) + shift).tolist()
    return entities


def terrain_dxf_entities(
    elements: Iterable,
    frame: c.LocalFrame,
    *,
    tin: c.LabeledTin | None = None,
    dtm: np.ndarray | None = None,
    validity: np.ndarray | None = None,
    dtm_bounds: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    curbs: Sequence = (),
    manholes: Sequence = (),
    config: dict | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Like :func:`to_dxf_entities` but also returns the contour report (interval choice, counts)."""
    cfg = config if config is not None else c.load_drone_config()
    terrain = [e for e in elements if isinstance(e, TerrainElement)]
    if tin is not None:
        vertices, triangles = np.asarray(tin.vertices, dtype=np.float64), np.asarray(tin.triangles, dtype=np.int64)
    elif terrain:
        vertices = np.asarray(terrain[0].params.vertices, dtype=np.float64)
        triangles = np.asarray(terrain[0].params.triangles, dtype=np.int64)
    else:
        return {}, {}
    out: dict[str, list[dict]] = {}
    if len(triangles):
        out[TIN_LAYER] = _tin_faces(vertices, triangles, frame)
    info: dict = {}
    if dtm is not None and validity is not None and dtm_bounds is not None and resolution is not None:
        contours, info = _contours(dtm, validity, dtm_bounds, resolution, frame, cfg)
        out.update(contours)
    if len(triangles):
        spots = _spots(TinSampler(vertices, triangles), frame, float(cfg.get("dxf", {}).get("spot_spacing_m", 10.0)), curbs, manholes)
        if spots:
            out[SPOT_LAYER] = spots
    return out, info


def to_dxf_entities(elements: Iterable, frame: c.LocalFrame, **kwargs) -> dict[str, list[dict]]:
    """DXF entity dicts per layer, coordinates already in LV95.

    * ``S-TOPO-TIN``: 3DFACE per triangle of the full TIN (``tin=``; default the terrain element mesh).
    * ``S-TOPO-MAJR`` / ``S-TOPO-MINR``: SPLINE contours (``dxf.spline_contours``) from the DTM raster
      (``dtm=``, ``validity=``, ``dtm_bounds=(xmin, ymin, xmax, ymax)``, ``resolution=``), measured cells only.
    * ``S-ELEV``: spot crosses + height text on a thinned grid (``dxf.spot_spacing_m``, default 10 m), at manholes
      (``manholes=`` xy list) and at curb break points (``curbs=`` CurbAxis list).
    """
    return terrain_dxf_entities(elements, frame, **kwargs)[0]
