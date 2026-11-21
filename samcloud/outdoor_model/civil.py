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
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import shapely
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon, shape
from shapely.validation import make_valid

from samcloud.bim.core.model import CurbElement, CurbParams, FenceElement, ManholeElement, SiteWallElement, ManholeParams, MarkingElement, MarkingParams
from samcloud.bim.core.model.enums import ParamOrigin, Status
from samcloud.drone import contracts as c
from samcloud.dxf.writer import outdoor_symbol_insert
from samcloud.environment.curbs import douglas_peucker_3d, split_straight_and_arcs

LOGGER = logging.getLogger(__name__)

CURB_LAYER = "S-CURB"
GUTTER_LAYER = "S-GUTR"
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
    surface: Callable[[np.ndarray], np.ndarray] | None = None,
    pavement: object | None = None,
    outline: object | None = None,
) -> tuple[list, list[c.QAEntry]]:
    """Curb, marking and manhole elements; below-threshold / rejected instances become ``QAEntry``.

    Curbs use ``curb.section_width_m`` / ``section_height_m`` (origin ASSUMED) and ``ifc.curb_sweep``
    (passed to the element; the IFC writer decides extrusion vs. sweep). The axis is split with
    ``split_straight_and_arcs`` for the evidence (``straight_parts``, ``arc_parts``, ``parts``). Axes
    shorter than the optional ``curb.min_length_m`` (default 0.3 m) are not emitted. Manholes get the type
    "Schacht" (ASSUMED, the cover type is not observable).

    With ``surface`` (the final TIN height sampler) every curb is snapped to it (:func:`snap_curb_to_surface`): top =
    upper-side surface, foot / gutter top = road-side surface; ``pavement`` (shapely geometry of the road and sidewalk
    polygons) decides where a gutter stone exists at all.
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
        # the element carries the thinned axis (qa_cad rule 5), so IFC body and DXF line use identical vertices
        keep = thin_collinear_indices(pts, float(cc.get("dp_tolerance_m", 0.015)))
        pts = pts[keep]
        if surface is not None and outline is not None:  # curb axis within terrain.step.curb_snap_m of a pavement outline lies ON it (TIN step line)
            snap_m = float(cfg.get("terrain", {}).get("step", {}).get("curb_snap_m", 0.15))
            on_line = np.asarray(shapely.distance(shapely.points(pts[:, :2]), outline), dtype=float) <= snap_m
            if on_line.any():
                pts = pts.copy()
                pts[on_line, :2] = shapely.get_coordinates(shapely.shortest_line(shapely.points(pts[on_line, :2]), outline))[1::2]
        axis_foot = None if axis.foot_z is None or len(axis.foot_z) != len(axis.points) else np.asarray(axis.foot_z, dtype=np.float64)[keep]
        parts = split_straight_and_arcs(
            pts, arc_max_residual=float(cc.get("arc_fit_max_residual_m", 0.01)),
            arc_max_radius=float(cc.get("arc_max_radius_m", 50.0)), arc_min_points=int(cc.get("arc_min_points", 4)))
        kinds = [k for k, _ in parts]
        foot = None
        if axis_foot is not None:
            foot = [float(v) for v in np.minimum(axis_foot, pts[:, 2])]
        notes_in = axis.evidence.notes
        measured_foot = foot is not None and notes_in.get("foot") != "assumed"
        gutter_w = float(axis.gutter_width) if foot is not None and axis.road_side else 0.0
        road_side = int(axis.road_side) if foot is not None else 0
        snapped = snap_curb_to_surface(pts, road_side, gutter_w, cc, surface, pavement, outline, cfg.get("terrain", {}).get("step", {})) if surface is not None else None
        if snapped is not None:
            pts = np.column_stack([pts[:, :2], snapped["top"]])
            foot, measured_foot, road_side = [float(v) for v in snapped["foot"]], True, int(snapped["side"])
            gutter_w = (gutter_w if gutter_w > 0 else float(cc.get("gutter_width_m", 0.30))) if snapped["gutter"] else 0.0
            notes_in = {**notes_in, "reveal_m": round(snapped["reveal"], 3), "foot": "measured",
                        "gutter": ("measured" if axis.gutter_measured else "assumed") if gutter_w > 0 else
                                  ("flush" if snapped["flush"] else ("no pavement" if not snapped["on_pavement"] else ("no step" if not snapped["step_built"] else "none")))}
        section = {}
        if foot is not None:
            section = {"road_side": road_side, "flush": bool(axis.flush) if snapped is None else bool(snapped["flush"]),
                       "flush_stations": int(axis.flush_stations), "surface": "tin" if snapped is not None else "points",
                       "reveal_m": notes_in.get("reveal_m"), "gutter_width_m": round(gutter_w, 3),
                       "gutter": notes_in.get("gutter", "none")}
        el = CurbElement(
            id=eid, tag=str(axis.curb_id),
            params=CurbParams(axis=[tuple(map(float, p)) for p in pts], width=width, height=height, curb_sweep=sweep,
                              foot_z=foot, road_side=road_side, gutter_width=gutter_w,
                              gutter_thickness=float(cc.get("gutter_thickness_m", 0.12)), embed=float(cc.get("embed_m", 0.15)),
                              gutter_axis=[tuple(map(float, q)) for q in snapped["gutter_axis"]] if snapped is not None and gutter_w > 0 else None,
                              gutter_z_outer=[float(v) for v in snapped["gutter_z_outer"]] if snapped is not None and gutter_w > 0 else None),
            param_origin={"axis": ParamOrigin.MEASURED, "width": ParamOrigin.ASSUMED, "height": ParamOrigin.ASSUMED,
                          "curb_sweep": ParamOrigin.ASSUMED,
                          **({"foot_z": ParamOrigin.MEASURED if measured_foot else ParamOrigin.ASSUMED} if foot is not None else {}),
                          **({"gutter_width": ParamOrigin.MEASURED if axis.gutter_measured else ParamOrigin.ASSUMED} if gutter_w > 0 else {})},
            source_instance_ids=[str(axis.curb_id)],
            evidence={"length_m": round(length, 3), "vertices": int(len(pts)), "straight_parts": kinds.count("straight"),
                      "arc_parts": kinds.count("arc"),
                      "parts": ",".join(f"{k}[{s.start}:{s.stop}]" for k, s in parts), **section},
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
    """Counts ``curb`` / ``gutter`` / ``marking`` / ``manhole`` / ``fence`` / ``site_wall`` for ``qa_cad.count_consistency``.

    ``gutter`` counts the curbs that carry a gutter stone (written as its own IFC product and ``S-GUTR`` line).
    """
    counts = {"curb": 0, "gutter": 0, "marking": 0, "manhole": 0, "fence": 0, "site_wall": 0}
    for e in elements:
        if isinstance(e, CurbElement):
            counts["curb"] += 1
            counts["gutter"] += int(gutter_outer_edge(e) is not None)
        elif isinstance(e, MarkingElement):
            counts["marking"] += 1
        elif isinstance(e, ManholeElement):
            counts["manhole"] += 1
        elif isinstance(e, FenceElement):
            counts["fence"] += 1
        elif isinstance(e, SiteWallElement):
            counts["site_wall"] += 1
    return counts


# ---------------------------------------------------------------------------------- DXF


def thin_collinear_indices(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices of the vertices kept by :func:`thin_collinear_xy` (ascending, end points always kept)."""
    pts = np.asarray(points, dtype=np.float64)
    idx = np.arange(len(pts))
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
        idx = np.delete(idx, k + 1)
    return idx


def thin_collinear_xy(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Drop interior vertices within ``tolerance`` (XY) of the chord of their neighbours.

    Smallest deviation first, repeated until every remaining interior vertex deviates by more than
    ``tolerance`` (the criterion of qa_cad rule 5). End points are kept; Z of dropped vertices is
    implicitly interpolated along the remaining chord.
    """
    pts = np.asarray(points, dtype=np.float64)
    return pts[thin_collinear_indices(pts, tolerance)]


def curbs_off_the_fit(elements: Iterable, fitted: Iterable, config: dict | None = None) -> tuple[set[int], list[CurbElement]]:
    """Curb elements the fitted-line DXF export (``pavement.curb_dxf_entities``) does not describe one to one.

    A fitted curb is described exactly when exactly one curb element carries its id and that element's axis is the
    thinned fitted axis. Curbs split or shortened by the boundary clip (ids ``curb-N-pK``), and elements without a
    fitted curb, are not: returns ``(fitted ids to leave out of the fitted export, elements to export from their axis)``,
    so every curb element gets exactly one ``S-CURB`` entity.
    """
    tol = float(_curb_cfg(config if config is not None else c.load_drone_config()).get("dp_tolerance_m", 0.015))
    by_id: dict[int, list[CurbElement]] = {}
    for e in elements:
        match = re.match(r"curb-(\d+)(?:-p\d+)?$", getattr(e, "id", "")) if isinstance(e, CurbElement) else None
        if match:
            by_id.setdefault(int(match.group(1)), []).append(e)
    fitted_ids = {int(f.curb_id) for f in fitted}
    exclude: set[int] = set()
    extra: list[CurbElement] = []
    for f in fitted:
        els = by_id.get(int(f.curb_id), [])
        if len(els) == 1:
            ref = thin_collinear_xy(np.asarray(f.dense_xyz, dtype=np.float64), tol)
            axis = np.asarray(els[0].params.axis, dtype=np.float64)
            if len(ref) == len(axis) and np.allclose(ref[:, :2], axis[:, :2], atol=1e-6):
                continue
        if els:
            exclude.add(int(f.curb_id))
            extra += els
    for cid, els in by_id.items():
        if cid not in fitted_ids:
            extra += els
    return exclude, extra


def tin_surface_sampler(tin) -> Callable[[np.ndarray], np.ndarray]:
    """Height sampler ``xy (N,2) -> z (N,)`` of a labelled TIN by its own triangles (NaN outside).

    Unlike a Delaunay re-triangulation of the vertices this keeps the steps at curbs: a point lies on the triangle
    that really covers it (the first one when several overlap on an edge).
    """
    v = np.asarray(tin.vertices, dtype=np.float64)
    t = np.asarray(tin.triangles, dtype=np.int64)
    tri = v[t]
    tree = shapely.STRtree(shapely.polygons(np.concatenate([tri[:, :, :2], tri[:, :1, :2]], axis=1)))

    def sample(xy) -> np.ndarray:
        q = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = np.full(len(q), np.nan)
        qi, ti = tree.query(shapely.points(q), predicate="intersects")
        if len(qi) == 0:
            return out
        a, b, d = tri[ti, 0], tri[ti, 1], tri[ti, 2]
        d1, d2 = b[:, :2] - a[:, :2], d[:, :2] - a[:, :2]
        den = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
        rel = q[qi] - a[:, :2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (rel[:, 0] * d2[:, 1] - d2[:, 0] * rel[:, 1]) / den
            w = (d1[:, 0] * rel[:, 1] - rel[:, 0] * d1[:, 1]) / den
        z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + w * (d[:, 2] - a[:, 2])
        ok = np.isfinite(z)
        out[qi[ok][::-1]] = z[ok][::-1]  # first hit wins
        return out

    return sample


def _station_normals(xy: np.ndarray) -> np.ndarray:
    """Left unit normals per vertex of a plan polyline."""
    seg = np.diff(xy, axis=0)
    seg = seg / np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-12)[:, None]
    tang = np.vstack([seg[:1], (seg[:-1] + seg[1:]), seg[-1:]])
    tang = tang / np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-12)[:, None]
    return np.column_stack([-tang[:, 1], tang[:, 0]])


def snap_curb_to_surface(
    pts: np.ndarray, road_side: int, gutter_w: float, cc: dict, surface: Callable[[np.ndarray], np.ndarray],
    pavement: object | None = None, outline: object | None = None, steps: dict | None = None,
) -> dict | None:
    """Curb top, foot and a gutter draped on the final terrain surface (TIN).

    The TIN has its step top edge on the pavement outline near the curb (``terrain.step``: curb axis within
    ``curb_snap_m`` of an outline, reveal between ``min_jump_m`` and ``curb_max_jump_m``). The curb top is the surface
    (upper side at a step) exactly on that step line (the outline point nearest to the axis, else the axis); the road
    side is the lower side. The gutter strip is sampled per station at its curb-side edge (just beyond the step) and at
    its outer edge, so its top coincides with the TIN everywhere. A gutter exists only where W1's step exists, the road
    side really is pavement (road/sidewalk polygon on the strip) and the curb is not flush.
    Returns ``None`` when the surface does not cover the curb.
    """
    steps = steps or {}
    snap = float(steps.get("curb_snap_m", 0.15))
    min_jump, max_jump = float(steps.get("min_jump_m", 0.04)), float(steps.get("curb_max_jump_m", 0.5))
    xy = pts[:, :2]
    hw = float(cc.get("section_width_m", 0.12)) / 2.0
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    stations = np.unique(np.round(np.r_[cum, np.arange(0.0, cum[-1], float(cc.get("gutter_station_m", 0.5)))], 6))
    base = LineString(xy)
    sxy = shapely.get_coordinates(shapely.line_interpolate_point(base, stations))
    snrm = _station_normals(sxy)
    near = np.zeros(len(sxy), dtype=bool)
    line_xy = sxy.copy()
    if outline is not None:
        near = np.asarray(shapely.distance(shapely.points(sxy), outline), dtype=float) <= snap
        if near.any():
            seg = shapely.get_coordinates(shapely.shortest_line(shapely.points(sxy[near]), outline))
            line_xy[near] = seg[1::2]  # the step line: the outline point nearest to the curb axis
    probe = float(cc.get("surface_probe_m", 0.06))
    zl, zr = surface(sxy + snrm * (hw + probe)), surface(sxy - snrm * (hw + probe))
    if not (np.isfinite(zl).any() or np.isfinite(zr).any()):
        return None
    ml = float(np.nanmedian(zl)) if np.isfinite(zl).any() else np.nan
    mr = float(np.nanmedian(zr)) if np.isfinite(zr).any() else np.nan
    if np.isfinite(ml) and np.isfinite(mr) and abs(ml - mr) > 0.005:
        side = 1 if ml < mr else -1
    elif road_side:
        side = int(road_side)
    else:
        side = 1
    z_axis = surface(line_xy)  # on a step line the sampler returns the upper side; elsewhere the plain surface under the curb line
    top_d = np.where(np.isfinite(z_axis), z_axis, np.interp(stations, cum, pts[:, 2]))
    gw = gutter_w if gutter_w > 0 else float(cc.get("gutter_width_m", 0.30))
    # curb-side edge of the gutter: beyond the curb face AND beyond the step line (which may lie up to snap off the axis)
    shift = np.einsum("ij,ij->i", line_xy - sxy, side * snrm)
    inner_off = np.maximum(hw, shift + 0.03)
    z_in = surface(sxy + side * snrm * inner_off[:, None])
    z_out = surface(sxy + side * snrm * (hw + gw))
    fill = lambda z, alt: np.where(np.isfinite(z), z, alt)  # noqa: E731
    z_low = surface(sxy + side * snrm * (hw + probe))
    base_low = np.where(np.isfinite(z_low), z_low, top_d - float(cc.get("default_reveal_m", 0.12)))
    z_in, z_out = fill(z_in, base_low), fill(z_out, fill(z_in, base_low))
    z_in, z_out = np.minimum(z_in, top_d), np.minimum(z_out, top_d)
    reveal_d = top_d - z_in
    flush = float(np.median(reveal_d)) < float(cc.get("flush_max_m", 0.03))
    step_built = bool(near.any() and float(np.mean(near)) >= 0.5 and float(np.mean((reveal_d >= min_jump) & (reveal_d <= max_jump))) >= 0.5) if outline is not None else True
    on_pavement = True
    if pavement is not None:
        reach = float(cc.get("gutter_pavement_reach_m", 0.3))
        hit = np.zeros(len(sxy), dtype=bool)
        for offset in (0.05, reach / 2.0, reach):
            hit |= shapely.distance(shapely.points(sxy + side * snrm * (hw + offset)), pavement) <= 0.02
        on_pavement = float(np.mean(hit)) >= 0.5
    plausible = float(np.median(reveal_d)) <= float(cc.get("max_reveal_m", 0.35))
    at_vertices = lambda z: np.interp(cum, stations, z)  # noqa: E731
    return {"top": at_vertices(top_d), "foot": at_vertices(z_in), "side": side, "flush": flush,
            "gutter": bool(on_pavement and plausible and not flush and step_built), "on_pavement": bool(on_pavement),
            "step_built": step_built, "reveal": float(np.median(reveal_d)),
            "gutter_axis": np.column_stack([sxy, z_in]), "gutter_z_outer": z_out}


def apply_curb_heights(fitted: Iterable, elements: Iterable) -> list:
    """Fitted curbs with their heights replaced by those of the matching curb element (one element per fitted curb).

    The fitted export keeps the line/arc geometry; its z follows the element axis (surface-snapped curb top) so that
    DXF ``S-CURB`` and the IFC body agree. Curbs without a matching single element stay unchanged.
    """
    from dataclasses import replace
    by_id: dict[int, list[CurbElement]] = {}
    for e in elements:
        match = re.match(r"curb-(\d+)$", getattr(e, "id", "")) if isinstance(e, CurbElement) else None
        if match:
            by_id.setdefault(int(match.group(1)), []).append(e)
    out = []
    for f in fitted:
        els = by_id.get(int(f.curb_id), [])
        if len(els) != 1:
            out.append(f)
            continue
        axis = np.asarray(els[0].params.axis, dtype=np.float64)
        line = LineString(axis[:, :2])
        cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(axis[:, :2], axis=0).T))]

        def height(p: np.ndarray, _line=line, _cum=cum, _axis=axis) -> np.ndarray:
            return np.interp(shapely.line_locate_point(_line, shapely.points(np.asarray(p, dtype=np.float64)[:, :2])), _cum, _axis[:, 2])

        dense = np.asarray(f.dense_xyz, dtype=np.float64).copy()
        dense[:, 2] = height(dense)
        out.append(replace(f, dense_xyz=dense, z=height(np.asarray(f.line.pts, dtype=np.float64))))
    return out


def gutter_outer_edge(curb: CurbElement, tolerance: float = 0.015) -> np.ndarray | None:
    """Outer edge of the gutter stone (Wasserstein) as (M,3) polyline at foot height; None without gutter.

    The edge is the axis offset to the road side by half the curb width plus the gutter width (mitre joins).
    """
    p = curb.params
    if not p.gutter_width or not p.road_side:
        return None
    if p.gutter_axis is not None and len(p.gutter_axis) >= 2:
        axis = np.asarray(p.gutter_axis, dtype=np.float64)
        foot = np.asarray(p.gutter_z_outer, dtype=np.float64) if p.gutter_z_outer is not None and len(p.gutter_z_outer) == len(axis) else axis[:, 2]
    elif p.foot_z is not None and len(p.foot_z) == len(p.axis):
        axis = np.asarray(p.axis, dtype=np.float64)
        foot = np.asarray(p.foot_z, dtype=np.float64)
    else:
        return None
    base = LineString(axis[:, :2])
    off = base.offset_curve(float(p.road_side) * (p.width / 2.0 + p.gutter_width), join_style="mitre")
    if off.is_empty:
        return None
    if off.geom_type != "LineString":
        off = max(off.geoms, key=lambda g: g.length)
    xy = np.asarray(off.coords, dtype=np.float64)
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(axis[:, :2], axis=0).T))]
    z = np.interp(shapely.line_locate_point(base, shapely.points(xy)), cum, foot)
    return thin_collinear_xy(np.column_stack([xy, z]), tolerance)


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
            edge = gutter_outer_edge(e, tol)
            if edge is not None:
                out.setdefault(GUTTER_LAYER, []).append({"type": "polyline3d", "points": frame.to_lv95(edge).tolist(), "closed": False})
        elif isinstance(e, MarkingElement):
            pts = np.asarray([p if len(p) == 3 else (p[0], p[1], 0.0) for p in e.params.points], dtype=np.float64)
            out.setdefault(MARK_LAYER, []).append({"type": "polyline3d", "points": frame.to_lv95(pts).tolist(), "closed": bool(e.params.closed)})
        elif isinstance(e, ManholeElement):
            xyz = frame.to_lv95(np.asarray(e.params.center, dtype=np.float64))
            ins = outdoor_symbol_insert("manhole", xyz.tolist(), {"Z_DECKEL": float(xyz[2]), "TYP": e.params.type or MANHOLE_TYPE},
                                        crown_radius=float(e.params.diameter) / 2.0)  # the writer scales the symbol to the measured radius
            out.setdefault(ins["layer"], []).append(ins)
    return out
