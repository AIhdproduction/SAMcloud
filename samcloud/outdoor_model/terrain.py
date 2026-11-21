"""Terrain builder: labelled TIN -> ``TerrainElement`` (without pavement) + DXF TIN, contours, spot heights.

The terrain element is the labelled TIN *without* road / sidewalk / path triangles. Lawn triangles stay in
the element mesh (the DXF TIN is complete) but their indices are listed in ``evidence["lawn_triangle_indices"]``:
the IFC export moves them to the lawn surface elements. Because pavement and terrain come from one TIN they
share the boundary vertices, so there is no gap and no overlap.

DXF: the full TIN as 3DFACE on S-TOPO-TIN (the writer freezes that layer), contours (SPLINE) on
S-TOPO-MAJR / S-TOPO-MINR restricted to the measured part of the DTM, height labels of the major contours on
S-TOPO-TEXT (``dxf.contour_label_spacing_m``). Spot heights (S-ELEV) are only written with ``dxf.spot_heights``.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence

import numpy as np
import shapely
from scipy import ndimage

from samcloud.bim.core.model import TerrainElement, TerrainParams
from samcloud.bim.core.model.enums import ParamOrigin, Status
from samcloud.drone import contracts as c

LOGGER = logging.getLogger(__name__)

PAVEMENT_LABELS = ("road", "sidewalk", "path")
LAWN_TRIANGLE_KEY = "lawn_triangle_indices"
TIN_LAYER = "S-TOPO-TIN"
MAJOR_LAYER = "S-TOPO-MAJR"
MINOR_LAYER = "S-TOPO-MINR"
LABEL_LAYER = "S-TOPO-TEXT"
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
        # Indices (into this element's triangles) of the lawn-labelled triangles: the IFC writer hands
        # exactly these over to the lawn surface elements (samcloud/bim/core/ifc/outdoor.py).
        LAWN_TRIANGLE_KEY: [int(i) for i in np.flatnonzero(np.asarray(terrain.labels)[~bad] == c.TIN_LABELS["lawn"])],
        "step_triangles": int((np.asarray(terrain.labels)[~bad] == c.TIN_LABELS["step"]).sum()),
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
    frame: c.LocalFrame, cfg: dict, footprints: Sequence = (),
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
    clipped = {"before": int(sum(len(v) for v in out.values())), "after": None}
    if len(footprints):
        from shapely import affinity
        from shapely.ops import unary_union

        gap = float(dcfg.get("contour_building_gap_m", 0.10))
        forbidden = affinity.translate(unary_union(list(footprints)).buffer(gap), float(frame.origin_lv95[0]), float(frame.origin_lv95[1]))
        out = {layer: clip_contours(ents, forbidden) for layer, ents in out.items()}
        clipped["after"] = int(sum(len(v) for v in out.values()))
    labels = contour_labels(out.get(MAJOR_LAYER, []), spacing_m=float(dcfg.get("contour_label_spacing_m", 10.0)),
                            text_height_m=float(dcfg.get("contour_label_height_m", 0.25)))
    if labels:
        out[LABEL_LAYER] = labels
    if bool(dcfg.get("spline_contours", True)):
        out = {layer: contours_as_splines(ents) for layer, ents in out.items()}
    info = {**info, "major_interval_m": major, "contour_count": int(len(raw)), "contour_labels": int(len(labels)), "contour_pieces": clipped}
    return out, info


def building_footprints(elements: Iterable) -> list:
    """Plan footprint (shapely geometry, local frame) of every ``BuildingElement``: the union of its projected triangles
    (walls and roof, so eaves count)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from samcloud.bim.core.model import BuildingElement

    out = []
    for element in elements:
        if not isinstance(element, BuildingElement) or not element.params.triangles:
            continue
        vertices = np.asarray(element.params.vertices, dtype=float)
        triangles = np.asarray(element.params.triangles, dtype=np.int64)
        polygons = [Polygon(tri) for tri in vertices[triangles][:, :, :2] if abs(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-9]
        if polygons:
            out.append(unary_union(polygons))
    return out


def clip_contours(entities: Iterable[dict], forbidden, *, min_length_m: float = 0.5) -> list[dict]:
    """Cut contour polylines (``points`` (N,3)) at ``forbidden`` (shapely geometry, same frame): the parts inside are
    removed, parts shorter than ``min_length_m`` are dropped. The elevation of every piece stays that of the contour."""
    from shapely.geometry import LineString

    if forbidden is None or forbidden.is_empty:
        return list(entities)
    import shapely

    shapely.prepare(forbidden)
    out: list[dict] = []
    for entity in entities:
        points = np.asarray(entity["points"], dtype=float)
        if len(points) < 2 or not shapely.intersects(forbidden, LineString(points[:, :2])):
            out.append(entity)
            continue
        z = float(np.median(points[:, 2]))
        rest = LineString(points[:, :2]).difference(forbidden)
        pieces = [rest] if rest.geom_type == "LineString" else [g for g in getattr(rest, "geoms", ()) if g.geom_type == "LineString"]
        for piece in pieces:
            if piece.is_empty or piece.length < min_length_m:
                continue
            xy = np.asarray(piece.coords, dtype=float)[:, :2]
            out.append({**{k: v for k, v in entity.items() if k != "points"}, "points": np.column_stack((xy, np.full(len(xy), z))).tolist(), "closed": False})
    return out


def contour_labels(
    entities: Iterable[dict], *, spacing_m: float = 10.0, text_height_m: float = 0.25, layer_role: str = "contour_label",
) -> list[dict]:
    """Height labels (TEXT) along major contour polylines.

    Every ``polyline3d`` / ``spline`` with ``layer_role == "major"`` and an ``elevation`` gets a label each
    ``spacing_m`` metres along the line (``round(length / spacing_m)`` labels spread evenly, at least one; a line
    shorter than the spacing gets one label in the middle). The text is the absolute elevation with one decimal
    ("432.0"), rotated along the line and never upside down (angle in (-90, 90]). Lines too short to carry the text
    stay unlabelled. Coordinates are used as given (already LV95 in the DXF chain).
    """
    labels: list[dict] = []
    spacing = max(float(spacing_m), 0.5)
    for entity in entities:
        if entity.get("layer_role") != "major" or "elevation" not in entity:
            continue
        key = "points" if "points" in entity else "fit_points" if "fit_points" in entity else None
        if key is None:
            continue
        points = np.asarray(entity[key], dtype=float)
        if points.ndim != 2 or len(points) < 2:
            continue
        closed = bool(entity.get("closed")) or bool(np.allclose(points[0], points[-1]) and len(points) > 3)
        coords = points if not closed or np.allclose(points[0], points[-1]) else np.vstack((points, points[:1]))
        steps = np.linalg.norm(np.diff(coords[:, :2], axis=0), axis=1)
        length = float(steps.sum())
        text = f"{float(entity['elevation']):.1f}"
        width = 0.7 * float(text_height_m) * len(text)
        if length < 1.5 * width:
            continue
        cumulative = np.r_[0.0, np.cumsum(steps)]
        count = max(1, int(round(length / spacing)))
        for index in range(count):
            arc = (index + 0.5) * length / count
            centre = _point_on(coords, cumulative, arc)
            before = _point_on(coords, cumulative, max(0.0, arc - 0.5 * width))
            after = _point_on(coords, cumulative, min(length, arc + 0.5 * width))
            tangent = after[:2] - before[:2]
            angle = math.degrees(math.atan2(tangent[1], tangent[0])) if np.hypot(*tangent) > 1e-9 else 0.0
            if angle > 90.0:
                angle -= 180.0
            elif angle <= -90.0:
                angle += 180.0
            labels.append({"type": "text", "text": text, "point": [float(centre[0]), float(centre[1]), float(centre[2])],
                           "height": float(text_height_m), "rotation": float(angle), "align": "MIDDLE_CENTER",
                           "layer_role": layer_role, "elevation": float(entity["elevation"])})
    return labels


def _point_on(coords: np.ndarray, cumulative: np.ndarray, arc: float) -> np.ndarray:
    index = int(np.clip(np.searchsorted(cumulative, arc, side="right") - 1, 0, len(coords) - 2))
    span = cumulative[index + 1] - cumulative[index]
    fraction = 0.0 if span <= 1e-12 else (arc - cumulative[index]) / span
    return coords[index] * (1.0 - fraction) + coords[index + 1] * fraction


def _spots(
    sampler: TinSampler, frame: c.LocalFrame, spacing: float, curbs: Sequence, manholes: Sequence,
    validity: np.ndarray | None = None, dtm_bounds: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
) -> list[dict]:
    from samcloud.ortho.vector.spots import SpotSettings, ground_grid_points, select_spots, spot_entities

    v = sampler.vertices
    bounds = (float(v[:, 0].min()), float(v[:, 1].min()), float(v[:, 0].max()), float(v[:, 1].max()))
    def measured(x: float, y: float) -> bool:
        if validity is None or dtm_bounds is None or resolution is None:
            return True
        xmin, _ymin, _xmax, ymax = dtm_bounds
        col = int(np.floor((x - xmin) / resolution))
        row = int(np.floor((ymax - y) / resolution))
        return 0 <= row < validity.shape[0] and 0 <= col < validity.shape[1] and validity[row, col] == MEASURED

    on_tin = lambda x, y: measured(x, y) and bool(np.isfinite(sampler.sample(np.array([[x, y]]))[0]))  # noqa: E731
    curb_xy = [(float(p[0]), float(p[1])) for a in curbs for p in np.asarray(getattr(a, "points", a))]
    mh_xy = [(float(m[0]), float(m[1])) for m in manholes]
    settings = SpotSettings(grid_spacing_m=float(spacing))
    groups = [("manhole", [p for p in mh_xy if on_tin(*p)]), ("curb", [p for p in curb_xy if on_tin(*p)]),
              ("grid", ground_grid_points(bounds, spacing, on_tin))]
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
    footprints: Sequence = (),
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
        contours, info = _contours(dtm, validity, dtm_bounds, resolution, frame, cfg, footprints)
        out.update(contours)
    if len(triangles) and bool(cfg.get("dxf", {}).get("spot_heights", False)):
        spots = _spots(TinSampler(vertices, triangles), frame, float(cfg.get("dxf", {}).get("spot_spacing_m", 10.0)),
                       curbs, manholes, validity, dtm_bounds, resolution)
        if spots:
            out[SPOT_LAYER] = spots
    return out, info


def to_dxf_entities(elements: Iterable, frame: c.LocalFrame, **kwargs) -> dict[str, list[dict]]:
    """DXF entity dicts per layer, coordinates already in LV95.

    * ``S-TOPO-TIN``: 3DFACE per triangle of the full TIN (``tin=``; default the terrain element mesh).
    * ``S-TOPO-MAJR`` / ``S-TOPO-MINR``: SPLINE contours (``dxf.spline_contours``) from the DTM raster
      (``dtm=``, ``validity=``, ``dtm_bounds=(xmin, ymin, xmax, ymax)``, ``resolution=``), measured cells only.
    * Contours are cut at ``footprints=`` (building footprints, local frame, see :func:`building_footprints`) buffered by
      ``dxf.contour_building_gap_m`` (0.10 m); pieces under 0.5 m are dropped and labels sit on the remaining pieces only.
    * ``S-TOPO-TEXT``: height label every ``dxf.contour_label_spacing_m`` (default 10 m) along each major contour.
    * ``S-ELEV``: only with ``dxf.spot_heights: true`` (default off): spot crosses + height text on a thinned grid
      (``dxf.spot_spacing_m``), at manholes (``manholes=`` xy list) and at curb break points (``curbs=`` CurbAxis list).
    """
    return terrain_dxf_entities(elements, frame, **kwargs)[0]
