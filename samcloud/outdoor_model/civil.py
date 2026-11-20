"""Civil-engineering objects of the outdoor model: curbs, road markings and manholes.

* ``CurbElement`` per ``CurbAxis`` (the IFC writer chooses extrusion vs. sweep from ``ifc.curb_sweep``);
  DXF ``S-CURB`` polyline3d of the thinned axis.
* ``MarkingElement`` (IfcAnnotation "Markierung") per marking feature (class 82); DXF ``S-MARK`` polylines.
* ``ManholeElement`` (IfcDistributionChamberElement MANHOLE) per class-75 cluster; DXF ``S-MANH`` INSERT
  ``MANHOLE`` with ``Z_DECKEL`` (2 decimals) and ``TYP``.

:func:`marking_features` and :func:`manhole_detections` derive the instances from the classified
point set (local metric frame). Coordinates stay local; :func:`to_dxf_entities` applies ``frame.to_lv95``.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, shape
from shapely.validation import make_valid

from samcloud.bim.core.model import CurbElement, CurbParams, ManholeElement, ManholeParams, MarkingElement, MarkingParams
from samcloud.bim.core.model.enums import ParamOrigin, Status
from samcloud.drone import contracts as c
from samcloud.dxf.writer import outdoor_symbol_insert
from samcloud.environment.curbs import douglas_peucker_3d, split_straight_and_arcs

LOGGER = logging.getLogger(__name__)

CURB_LAYER = "S-CURB"
MARK_LAYER = "S-MARK"
MANHOLE_TYPE = "Schacht"
MANHOLE_BLOCK_RADIUS_M = 0.3
_STATUS_MAP = {"ok": Status.OK, "warning": Status.WARNING, "failed": Status.FAIL, "rejected": Status.FAIL}


# ---------------------------------------------------------------------------------- detections


@dataclass
class MarkingFeature:
    """One road marking: a centre line (``kind="line"``) or an outline polygon (``kind="polygon"``)."""

    marking_id: int
    kind: Literal["line", "polygon"]
    points: np.ndarray                # (M,3) local, open polyline or closed ring without repeated end point
    closed: bool
    width: float | None = None        # line width (m), None for polygons
    evidence: c.Evidence = field(default_factory=c.Evidence)
    status: c.Status = "ok"
    reason: str = ""


@dataclass
class ManholeDetection:
    """One manhole cover: centre (x, y, Z of the cover), diameter, evidence."""

    manhole_id: int
    center: np.ndarray                # (3,) local; z = median Z of the cluster (Z_DECKEL)
    diameter: float
    evidence: c.Evidence = field(default_factory=c.Evidence)
    status: c.Status = "ok"
    reason: str = ""


def _components(points: np.ndarray, eps: float) -> list[np.ndarray]:
    n = len(points)
    if n == 0:
        return []
    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    count, labels = csgraph.connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(count + 1))
    return [order[bounds[i]: bounds[i + 1]] for i in range(count)]


def _pca_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit axes (major, minor) of the XY scatter."""
    centred = xy - xy.mean(axis=0)
    vals, vecs = np.linalg.eigh(centred.T @ centred)
    major = vecs[:, int(np.argmax(vals))]
    return major, np.array([-major[1], major[0]])


def manhole_detections(
    ps: c.PointSet,
    *,
    eps: float = 0.15,
    min_points: int = 30,
    diameter_range: tuple[float, float] = (0.4, 1.2),
    min_roundness: float = 0.7,
) -> list[ManholeDetection]:
    """Manhole covers from MANHOLE (75) points.

    Points are clustered (eps-neighbour components). A cluster is a manhole when its PCA extents
    give a diameter ``(major + minor) / 2`` inside ``diameter_range`` (0.4-1.2 m) and a roundness
    ``minor / major`` >= ``min_roundness``. The centre is the middle of the extents, ``Z_DECKEL`` is the
    median Z of the cluster. Other clusters are returned with ``status="rejected"`` and a reason.
    """
    xyz = np.asarray(ps.xyz, dtype=np.float64)[ps.classes == c.MANHOLE]
    out: list[ManholeDetection] = []
    for idx in _components(xyz, eps):
        if len(idx) < min_points:
            continue
        pts = xyz[idx]
        xy = pts[:, :2]
        major, minor = _pca_axes(xy)
        u, v = xy @ major, xy @ minor
        ext_major, ext_minor = float(np.ptp(u)), float(np.ptp(v))
        centre_xy = major * (u.min() + u.max()) / 2.0 + minor * (v.min() + v.max()) / 2.0
        diameter = (ext_major + ext_minor) / 2.0
        roundness = ext_minor / ext_major if ext_major > 0 else 0.0
        status, reason = "ok", ""
        if not diameter_range[0] <= diameter <= diameter_range[1]:
            status, reason = "rejected", f"diameter {diameter:.2f} m outside {diameter_range[0]}-{diameter_range[1]} m"
        elif roundness < min_roundness:
            status, reason = "rejected", f"not round (minor/major {roundness:.2f} < {min_roundness})"
        out.append(ManholeDetection(
            manhole_id=len(out), center=np.array([centre_xy[0], centre_xy[1], float(np.median(pts[:, 2]))]),
            diameter=float(diameter),
            evidence=c.Evidence(points=int(len(idx)), notes={"roundness": round(roundness, 3)}),
            status=status, reason=reason,
        ))
    return out


def _disk(radius_cells: int) -> np.ndarray:
    r = max(int(radius_cells), 1)
    yy, xx = np.mgrid[-r: r + 1, -r: r + 1]
    return (xx * xx + yy * yy) <= r * r


def _line_centerline(pts: np.ndarray, major: np.ndarray, minor: np.ndarray, step: float, dp_tol: float) -> np.ndarray | None:
    u = pts[:, :2] @ major
    bins = np.floor((u - u.min()) / step).astype(np.int64)
    rows = []
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum() < 3:
            continue
        cell = pts[sel]
        rows.append([cell[:, 0].mean(), cell[:, 1].mean(), float(np.median(cell[:, 2]))])
    if len(rows) < 2:
        return None
    return douglas_peucker_3d(np.asarray(rows), dp_tol)


def marking_features(
    ps: c.PointSet,
    *,
    resolution: float = 0.04,
    close_radius: float = 0.08,
    min_points: int = 20,
    min_area: float = 0.02,
    line_max_width: float = 0.30,
    step: float = 0.10,
    dp_tolerance: float = 0.01,
) -> list[MarkingFeature]:
    """Road markings from ROAD_MARKING (82) points.

    The points are rasterised (``resolution``), closed (``close_radius``) and split into connected
    components. A component whose PCA width is <= ``line_max_width`` and that is at least three times as
    long as wide is a *line*: centre line sampled every ``step`` m along the major axis, thinned with
    ``dp_tolerance`` (straight or gently curved marks; Z = cluster median per bin). Every other component
    is a *polygon*: vectorised mask outline simplified to ``dp_tolerance`` (Z of the nearest cluster
    point). Components below ``min_points`` points or ``min_area`` m2 are returned as ``rejected``.
    """
    sel = ps.classes == c.ROAD_MARKING
    xyz = np.asarray(ps.xyz, dtype=np.float64)[sel]
    if len(xyz) == 0:
        return []
    x0 = math.floor(xyz[:, 0].min() / resolution) * resolution - 2 * close_radius
    y0 = math.floor(xyz[:, 1].min() / resolution) * resolution - 2 * close_radius
    nx = int(math.floor((xyz[:, 0].max() - x0) / resolution)) + 1 + int(2 * close_radius / resolution)
    ny = int(math.floor((xyz[:, 1].max() - y0) / resolution)) + 1 + int(2 * close_radius / resolution)
    col = np.floor((xyz[:, 0] - x0) / resolution).astype(np.int64)
    row = np.floor((xyz[:, 1] - y0) / resolution).astype(np.int64)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[row, col] = True
    rc = max(int(round(close_radius / resolution)), 1)
    closed = ndimage.binary_closing(np.pad(mask, rc + 1), structure=_disk(rc))[rc + 1: -(rc + 1), rc + 1: -(rc + 1)]
    labels, count = ndimage.label(closed, structure=np.ones((3, 3), dtype=int))
    point_label = labels[row, col]
    transform = Affine(resolution, 0.0, x0, 0.0, resolution, y0)
    out: list[MarkingFeature] = []
    for k in range(1, count + 1):
        pts = xyz[point_label == k]
        n = len(pts)
        if n < min_points:
            if n >= 3:
                out.append(MarkingFeature(len(out), "polygon", np.zeros((0, 3)), True,
                                          evidence=c.Evidence(points=n), status="rejected",
                                          reason=f"only {n} points (< {min_points})"))
            continue
        major, minor = _pca_axes(pts[:, :2])
        length = float(np.ptp(pts[:, :2] @ major))
        width = float(np.ptp(pts[:, :2] @ minor))
        evidence = c.Evidence(points=n)
        if width <= line_max_width and length >= 3.0 * max(width, 1e-6):
            line = _line_centerline(pts, major, minor, step, dp_tolerance)
            if line is None:
                out.append(MarkingFeature(len(out), "line", np.zeros((0, 3)), False, evidence=evidence,
                                          status="rejected", reason="centre line has fewer than two bins"))
            else:
                out.append(MarkingFeature(len(out), "line", line, False, width=width, evidence=evidence))
            continue
        polys = [shape(g) for g, val in features.shapes((labels == k).astype(np.uint8), mask=labels == k, transform=transform) if val]
        poly = make_valid(max(polys, key=lambda p: p.area)) if polys else None
        if poly is not None and poly.geom_type != "Polygon":
            parts = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
            poly = max(parts, key=lambda p: p.area) if parts else None
        if poly is not None:
            poly = Polygon(poly.exterior).buffer(-resolution / 2.0, join_style="mitre", mitre_limit=2.0)
            if poly.geom_type != "Polygon" and hasattr(poly, "geoms") and len(poly.geoms):
                poly = max(poly.geoms, key=lambda p: p.area)
        if poly is None or poly.is_empty or poly.area < min_area:
            out.append(MarkingFeature(len(out), "polygon", np.zeros((0, 3)), True, evidence=evidence,
                                      status="rejected", reason=f"area below {min_area} m2"))
            continue
        poly = poly.simplify(dp_tolerance, preserve_topology=True)
        ring = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]
        _, nearest = cKDTree(pts[:, :2]).query(ring)
        out.append(MarkingFeature(len(out), "polygon", np.column_stack([ring, pts[nearest, 2]]), True, evidence=evidence))
    return out


# ---------------------------------------------------------------------------------- builder


def _status(raw: str) -> Status:
    return _STATUS_MAP.get(raw, Status.UNSURE)


def _curb_cfg(cfg: dict) -> dict:
    return cfg.get("curb", {})


def build_civil_elements(
    curbs: Sequence[c.CurbAxis],
    markings: Sequence[MarkingFeature] = (),
    manholes: Sequence[ManholeDetection] = (),
    *,
    config: dict | None = None,
) -> tuple[list, list[c.QAEntry]]:
    """Curb, marking and manhole elements; below-threshold / rejected instances become ``QAEntry``.

    Curbs use ``curb.section_width_m`` / ``section_height_m`` (origin ASSUMED) and ``ifc.curb_sweep``
    (passed to the element; the IFC writer decides extrusion vs. sweep). The axis is split with
    ``split_straight_and_arcs`` for the evidence (``straight_parts``, ``arc_parts``, ``parts``). Axes
    shorter than the optional ``curb.min_length_m`` (default 0.3 m) are not emitted. Manholes get the type
    "Schacht" (ASSUMED, the cover type is not observable).
    """
    cfg = config if config is not None else c.load_drone_config()
    cc = _curb_cfg(cfg)
    width, height = float(cc.get("section_width_m", 0.12)), float(cc.get("section_height_m", 0.25))
    sweep = str(cfg.get("ifc", {}).get("curb_sweep", "auto"))
    sweep = sweep if sweep in ("auto", "extrude_only") else "auto"
    min_len = float(cc.get("curb_min_length_m", cc.get("min_length_m", 0.3)))
    elements: list = []
    qa: list[c.QAEntry] = []

    for axis in curbs:
        eid = f"curb-{axis.curb_id}"
        pts = np.asarray(axis.points, dtype=np.float64)
        if axis.status == "rejected":
            qa.append(c.QAEntry("curb", eid, f"rejected by detection: {axis.reason}"))
            continue
        if len(pts) < 2:
            qa.append(c.QAEntry("curb", eid, "axis has fewer than two points"))
            continue
        length = float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum())
        if length < min_len:
            qa.append(c.QAEntry("curb", eid, f"axis length {length:.2f} m below {min_len} m", {"length_m": length}))
            continue
        parts = split_straight_and_arcs(
            pts, arc_max_residual=float(cc.get("arc_fit_max_residual_m", 0.01)),
            arc_max_radius=float(cc.get("arc_max_radius_m", 50.0)), arc_min_points=int(cc.get("arc_min_points", 4)))
        kinds = [k for k, _ in parts]
        el = CurbElement(
            id=eid, tag=str(axis.curb_id),
            params=CurbParams(axis=[tuple(map(float, p)) for p in pts], width=width, height=height, curb_sweep=sweep),
            param_origin={"axis": ParamOrigin.MEASURED, "width": ParamOrigin.ASSUMED, "height": ParamOrigin.ASSUMED,
                          "curb_sweep": ParamOrigin.ASSUMED},
            source_instance_ids=[str(axis.curb_id)],
            evidence={"length_m": round(length, 3), "vertices": int(len(pts)), "straight_parts": kinds.count("straight"),
                      "arc_parts": kinds.count("arc"),
                      "parts": ",".join(f"{k}[{s.start}:{s.stop}]" for k, s in parts)},
            status=_status(axis.status), notes=[axis.reason] if axis.reason else [],
        )
        el.metrics.point_count = int(axis.evidence.points) or None
        elements.append(el)

    for mk in markings:
        eid = f"marking-{mk.marking_id}"
        if mk.status == "rejected" or len(mk.points) < (3 if mk.closed else 2):
            qa.append(c.QAEntry("marking", eid, f"rejected: {mk.reason or 'too few points'}", {"points": mk.evidence.points}))
            continue
        el = MarkingElement(
            id=eid, tag=str(mk.marking_id),
            params=MarkingParams(points=[tuple(map(float, p)) for p in mk.points], closed=bool(mk.closed)),
            param_origin={"points": ParamOrigin.FITTED, "closed": ParamOrigin.MEASURED},
            source_instance_ids=[str(mk.marking_id)],
            evidence={"kind": mk.kind, **({"width_m": round(mk.width, 3)} if mk.width is not None else {})},
            status=_status(mk.status), notes=[mk.reason] if mk.reason else [],
        )
        el.metrics.point_count = int(mk.evidence.points) or None
        elements.append(el)

    for mh in manholes:
        eid = f"manhole-{mh.manhole_id}"
        if mh.status == "rejected":
            qa.append(c.QAEntry("manhole", eid, f"rejected: {mh.reason}", {"points": mh.evidence.points, "diameter_m": mh.diameter}))
            continue
        el = ManholeElement(
            id=eid, tag=str(mh.manhole_id),
            params=ManholeParams(center=tuple(map(float, mh.center)), diameter=float(mh.diameter), type=MANHOLE_TYPE),
            param_origin={"center": ParamOrigin.MEASURED, "diameter": ParamOrigin.FITTED, "type": ParamOrigin.ASSUMED},
            source_instance_ids=[str(mh.manhole_id)],
            evidence={"z_deckel_local": round(float(mh.center[2]), 3), **mh.evidence.notes},
            status=_status(mh.status), notes=[mh.reason] if mh.reason else [],
        )
        el.metrics.point_count = int(mh.evidence.points) or None
        elements.append(el)
    return elements, qa


def category_counts(elements: Iterable) -> dict[str, int]:
    """Counts ``curb`` / ``marking`` / ``manhole`` for ``qa_cad.count_consistency``."""
    counts = {"curb": 0, "marking": 0, "manhole": 0}
    for e in elements:
        if isinstance(e, CurbElement):
            counts["curb"] += 1
        elif isinstance(e, MarkingElement):
            counts["marking"] += 1
        elif isinstance(e, ManholeElement):
            counts["manhole"] += 1
    return counts


# ---------------------------------------------------------------------------------- DXF


def thin_collinear_xy(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Drop interior vertices within ``tolerance`` (XY) of the chord of their neighbours.

    Smallest deviation first, repeated until every remaining interior vertex deviates by more than
    ``tolerance`` (the criterion of qa_cad rule 5). End points are kept; Z of dropped vertices is
    implicitly interpolated along the remaining chord.
    """
    pts = np.asarray(points, dtype=np.float64)
    limit = tolerance * 1.001 + 1e-9
    while len(pts) > 2:
        a, b, cc = pts[:-2, :2], pts[1:-1, :2], pts[2:, :2]
        chord = cc - a
        norm = np.hypot(chord[:, 0], chord[:, 1])
        rel = b - a
        cross = np.abs(chord[:, 0] * rel[:, 1] - chord[:, 1] * rel[:, 0])
        dist = np.where(norm < 1e-12, np.hypot(rel[:, 0], rel[:, 1]), cross / np.maximum(norm, 1e-12))
        k = int(np.argmin(dist))
        if dist[k] > limit:
            break
        pts = np.delete(pts, k + 1, axis=0)
    return pts


def to_dxf_entities(elements: Iterable, frame: c.LocalFrame, *, config: dict | None = None) -> dict[str, list[dict]]:
    """DXF entities per layer in LV95: ``S-CURB`` / ``S-MARK`` polyline3d and ``S-MANH`` MANHOLE inserts.

    The curb polyline is the axis re-thinned in plan with ``curb.dp_tolerance_m`` (qa_cad rule 5: no
    collinear vertices). Manhole inserts carry ``Z_DECKEL`` (absolute LN02 height, 2 decimals) and ``TYP``
    and are scaled to the measured diameter.
    """
    cfg = config if config is not None else c.load_drone_config()
    tol = float(_curb_cfg(cfg).get("dp_tolerance_m", 0.015))
    out: dict[str, list[dict]] = {}
    for e in elements:
        if isinstance(e, CurbElement):
            pts = thin_collinear_xy(np.asarray(e.params.axis, dtype=np.float64), tol)
            out.setdefault(CURB_LAYER, []).append({"type": "polyline3d", "points": frame.to_lv95(pts).tolist(), "closed": False})
        elif isinstance(e, MarkingElement):
            pts = np.asarray([p if len(p) == 3 else (p[0], p[1], 0.0) for p in e.params.points], dtype=np.float64)
            out.setdefault(MARK_LAYER, []).append({"type": "polyline3d", "points": frame.to_lv95(pts).tolist(), "closed": bool(e.params.closed)})
        elif isinstance(e, ManholeElement):
            xyz = frame.to_lv95(np.asarray(e.params.center, dtype=np.float64))
            ins = outdoor_symbol_insert("manhole", xyz.tolist(), {"Z_DECKEL": float(xyz[2]), "TYP": e.params.type or MANHOLE_TYPE})
            scale = float(e.params.diameter) / (2.0 * MANHOLE_BLOCK_RADIUS_M)
            ins["xscale"] = ins["yscale"] = scale
            out.setdefault(ins["layer"], []).append(ins)
    return out
