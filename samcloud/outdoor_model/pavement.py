"""Pavement builder: road / sidewalk / path polygons + labelled TIN -> ``PavementElement`` + DXF.

Order of work (curb = master breakline):

1. :func:`pavement_topology` snaps the pavement polygon boundaries onto the curb axes
   (search radius ``pavement.snap_search_m``, default 0.15 m; this is a *search* radius, the
   residual target stays ``pavement.max_gap_overlap_cm2``), resolves overlaps by priority
   (sidewalk > road > path > lawn) and closes narrow gaps.
2. The caller builds the labelled TIN from the returned polygons (``result.regions()``), so
   terrain and pavement share the polygon boundary vertices.
3. :func:`build_pavement_elements` takes, per polygon, the TIN triangles of its label whose
   centroid lies inside it as the top surface; the IFC writer closes the solid with the
   configured thickness.

Everything stays in the local metric frame; :func:`to_dxf_entities` applies ``frame.to_lv95``.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from samcloud.bim.core.model import PavementElement, PavementParams
from samcloud.bim.core.model.enums import ParamOrigin, Status
from samcloud.drone import contracts as c
from samcloud.environment import topology

LOGGER = logging.getLogger(__name__)

KIND_LAYERS: dict[str, str] = {"road": "S-PAVE", "sidewalk": "S-WALK", "path": "S-PATH"}
#: Priority of the pavement kinds when polygons overlap (highest first); lawn follows.
_PRIORITY_KINDS = ("sidewalk", "road", "path")
#: Z range up to which a pavement outline is written as a flat elevated LWPOLYLINE.
FLAT_Z_RANGE_M = 0.30
_STATUS_MAP = {"ok": Status.OK, "warning": Status.WARNING, "failed": Status.FAIL, "rejected": Status.FAIL}


@dataclass
class PavementTopology:
    """Result of :func:`pavement_topology`."""

    polygons: list[c.PavementPolygon]
    lawns: list[Polygon]
    report: dict[str, dict]
    snapped_ids: set[tuple[str, int]] = field(default_factory=set)

    def regions(self) -> dict[str, list[Polygon]]:
        """Polygons per TIN region name (``road``, ``sidewalk``, ``path``, ``lawn``) for ``build_labeled_tin``."""
        out: dict[str, list[Polygon]] = {k: [] for k in (*c.PAVEMENT_KINDS, "lawn")}
        for p in self.polygons:
            out[p.kind].append(p.polygon)
        out["lawn"] = list(self.lawns)
        return out

    def worst_cm2(self) -> float:
        """Largest gap + overlap (cm2) over all neighbouring pairs."""
        return max((r["overlap_cm2"] + r["gap_cm2"] for r in self.report.values()), default=0.0)


def _pave_cfg(config: dict | None) -> dict:
    cfg = config if config is not None else c.load_drone_config()
    return cfg


def _explode(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _explode(g)]
    return []


def pavement_topology(
    polygons: Sequence[c.PavementPolygon],
    lawn_polygons: Iterable[Polygon] = (),
    curbs: Sequence[c.CurbAxis] = (),
    *,
    config: dict | None = None,
) -> PavementTopology:
    """Snap pavement polygons to the curb axes and resolve overlaps/gaps by priority.

    The snapping search radius is ``pavement.snap_search_m`` (default 0.15 m). Overlaps are
    removed in the order sidewalk > road > path > lawn; gaps narrower than the search radius that
    touch two groups are given to the neighbour with the longest shared edge. Polygon identity
    (``pavement_id``) is kept: every resulting part inherits the id of the original polygon it
    overlaps most. The pairwise gap/overlap report is in ``result.report``.
    """
    cfg = _pave_cfg(config)
    pav = cfg.get("pavement", {})
    search = float(pav.get("snap_search_m", 0.15))
    originals = [p for p in polygons if p.status != "rejected" and not p.polygon.is_empty]
    lawns_in = [make_valid(p) for p in lawn_polygons]
    lawns_in = [q for g in lawns_in for q in _explode(g)]
    axes = [a for a in curbs if a.status != "rejected" and len(a.points) >= 2]

    snapped_items = topology.snap_to_breakline(originals, axes, tolerance=search) if axes else list(originals)
    by_kind: dict[str, list[c.PavementPolygon]] = {k: [] for k in c.PAVEMENT_KINDS}
    for item in snapped_items:
        by_kind[item.kind].append(item)
    groups = [[p.polygon for p in by_kind[k]] for k in _PRIORITY_KINDS] + [lawns_in]
    gap_tol = float(pav.get("gap_tolerance_m", max(search, 0.05)))
    resolved = topology.resolve_overlaps(groups, gap_tolerance=gap_tol)

    out: list[c.PavementPolygon] = []
    snapped: set[tuple[str, int]] = set()
    orig_by_kind = {k: [p for p in originals if p.kind == k] for k in c.PAVEMENT_KINDS}
    for kind, parts in zip(_PRIORITY_KINDS, resolved[:3]):
        pool = orig_by_kind[kind]
        for part in parts:
            if part.area <= 1e-9:
                continue
            src = max(pool, key=lambda p: p.polygon.intersection(part).area, default=None)
            if src is None or src.polygon.intersection(part).area <= 0.0:
                src = pool[0] if pool else None
            if src is None:
                continue
            changed = part.symmetric_difference(src.polygon).area > 1e-6
            if changed:
                snapped.add((kind, src.pavement_id))
            out.append(c.PavementPolygon(
                pavement_id=src.pavement_id, kind=kind, polygon=part,
                evidence=src.evidence, status=src.status, reason=src.reason,
            ))
    lawns = resolved[3]
    named = {k: [p.polygon for p in out if p.kind == k] for k in c.PAVEMENT_KINDS}
    named["lawn"] = lawns
    named = {k: v for k, v in named.items() if v}
    report = {f"{a}|{b}": r for (a, b), r in topology.pairwise_report(named, tolerance=0.05).items()}
    return PavementTopology(polygons=out, lawns=lawns, report=report, snapped_ids=snapped)


# ---------------------------------------------------------------------------------- mesh helpers


def _ccw(vertices: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Orient triangles counter-clockwise in XY (upward normal)."""
    a, b, cc = (vertices[tris[:, i], :2] for i in range(3))
    area2 = (b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0])
    out = tris.copy()
    flip = area2 < 0
    out[flip] = tris[flip][:, [0, 2, 1]]
    return out


def _tin_triangles_inside(tin: c.LabeledTin, kind: str, polygon: Polygon, centroids: np.ndarray) -> np.ndarray:
    """Indices of TIN triangles with label ``kind`` whose centroid lies inside ``polygon``."""
    cand = np.flatnonzero(np.asarray(tin.labels) == c.TIN_LABELS[kind])
    if len(cand) == 0:
        return cand
    inside = shapely.contains_xy(polygon, centroids[cand, 0], centroids[cand, 1])
    return cand[inside]


def _compact_mesh(vertices: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used, inverse = np.unique(tris.ravel(), return_inverse=True)
    return vertices[used], inverse.reshape(tris.shape)


def _tri_area(vertices: np.ndarray, tris: np.ndarray) -> float:
    a, b, cc = (vertices[tris[:, i], :2] for i in range(3))
    return float(0.5 * np.abs((b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0])).sum())


def build_pavement_elements(
    polygons: Sequence[c.PavementPolygon],
    lawn_polygons: Iterable[Polygon],
    curbs: Sequence[c.CurbAxis],
    tin: c.LabeledTin,
    *,
    config: dict | None = None,
) -> tuple[list[PavementElement], list[c.QAEntry]]:
    """One ``PavementElement`` per (snapped) polygon; top surface = TIN triangles of its label inside it.

    ``polygons`` / ``lawn_polygons`` / ``curbs`` are passed through :func:`pavement_topology`
    (idempotent when the TIN was already built from ``topology.regions()``). Thickness is
    ``pavement.min_thickness_m``. Polygons without TIN triangles, with a rejected status or an
    area below the optional ``pavement.min_area_m2`` become ``QAEntry`` instead of elements.
    A mismatch > 2 % between polygon area and its triangle area sets the status to WARNING.
    """
    cfg = _pave_cfg(config)
    pav = cfg.get("pavement", {})
    thickness = float(pav.get("min_thickness_m", 0.10))
    min_area = float(pav.get("min_area_m2", 0.0))
    limit_cm2 = float(pav.get("max_gap_overlap_cm2", 1.0))
    qa: list[c.QAEntry] = []
    for p in polygons:
        if p.status == "rejected":
            qa.append(c.QAEntry("pavement", f"{p.kind}-{p.pavement_id}", f"rejected by detection: {p.reason}"))
    topo = pavement_topology(polygons, lawn_polygons, curbs, config=cfg)
    if topo.worst_cm2() >= limit_cm2:
        worst = max(topo.report.items(), key=lambda kv: kv[1]["overlap_cm2"] + kv[1]["gap_cm2"])
        qa.append(c.QAEntry("pavement_topology", worst[0], f"gap + overlap {topo.worst_cm2():.2f} cm2 >= {limit_cm2} cm2",
                            {"overlap_cm2": worst[1]["overlap_cm2"], "gap_cm2": worst[1]["gap_cm2"]}))

    vertices = np.asarray(tin.vertices, dtype=np.float64)
    triangles = np.asarray(tin.triangles, dtype=np.int64)
    centroids = vertices[triangles].mean(axis=1) if len(triangles) else np.zeros((0, 3))
    elements: list[PavementElement] = []
    part_counter: dict[tuple[str, int], int] = {}
    for p in topo.polygons:
        key = (p.kind, p.pavement_id)
        part = part_counter.get(key, 0)
        part_counter[key] = part + 1
        eid = f"pavement-{p.kind}-{p.pavement_id}" + (f"-{part}" if part else "")
        if p.polygon.area < min_area:
            qa.append(c.QAEntry("pavement", eid, f"area {p.polygon.area:.2f} m2 below pavement.min_area_m2 {min_area}"))
            continue
        sel = _tin_triangles_inside(tin, p.kind, p.polygon, centroids)
        if len(sel) == 0:
            qa.append(c.QAEntry("pavement", eid, f"no TIN triangle labelled '{p.kind}' inside the polygon",
                                {"area_m2": float(p.polygon.area)}))
            continue
        tris = _ccw(vertices, triangles[sel])
        areas2 = np.abs((vertices[tris[:, 1], 0] - vertices[tris[:, 0], 0]) * (vertices[tris[:, 2], 1] - vertices[tris[:, 0], 1])
                        - (vertices[tris[:, 1], 1] - vertices[tris[:, 0], 1]) * (vertices[tris[:, 2], 0] - vertices[tris[:, 0], 0]))
        tris = tris[areas2 > 2e-12]
        if len(tris) == 0:
            qa.append(c.QAEntry("pavement", eid, "all TIN triangles degenerate"))
            continue
        verts, tri_idx = _compact_mesh(vertices, tris)
        tri_area = _tri_area(verts, tri_idx)
        ratio = tri_area / p.polygon.area if p.polygon.area > 0 else 0.0
        status = _STATUS_MAP.get(p.status, Status.UNSURE)
        notes: list[str] = []
        if abs(ratio - 1.0) > 0.02:
            status = Status.WARNING if status == Status.OK else status
            notes.append(f"TIN triangle area is {ratio * 100:.1f} % of the polygon area (TIN not built from the snapped polygons?)")
            qa.append(c.QAEntry("pavement", eid, notes[-1], {"area_ratio": float(ratio)}))
        origin = ParamOrigin.SNAPPED if key in topo.snapped_ids else ParamOrigin.MEASURED
        z = verts[:, 2]
        elements.append(PavementElement(
            id=eid, tag=f"{p.kind}-{p.pavement_id}",
            params=PavementParams(
                vertices=[tuple(map(float, v)) for v in verts],
                triangles=[tuple(int(i) for i in t) for t in tri_idx],
                kind=p.kind, thickness=thickness,
            ),
            param_origin={"vertices": origin, "triangles": origin, "kind": ParamOrigin.MEASURED, "thickness": ParamOrigin.ASSUMED},
            source_instance_ids=[str(p.pavement_id)],
            evidence={"polygon_area_m2": round(float(p.polygon.area), 4), "triangle_count": int(len(tri_idx)),
                      "z_range_m": round(float(z.max() - z.min()), 4), "area_ratio": round(float(ratio), 4),
                      "points": int(p.evidence.points)},
            status=status, notes=notes,
        ))
        elements[-1].metrics.point_count = int(p.evidence.points) if p.evidence.points else None
    return elements, qa


def category_counts(elements: Iterable) -> dict[str, int]:
    """Counts per pavement kind (keys ``road``, ``sidewalk``, ``path``) for ``qa_cad.count_consistency``."""
    counts = {k: 0 for k in c.PAVEMENT_KINDS}
    for e in elements:
        if isinstance(e, PavementElement):
            counts[e.params.kind] += 1
    return counts


# ---------------------------------------------------------------------------------- footprints / DXF


def element_footprint(element: PavementElement) -> Polygon | None:
    """Footprint (union of the top triangles) of a pavement element in local XY, or None."""
    v = np.asarray(element.params.vertices, dtype=np.float64)
    t = np.asarray(element.params.triangles, dtype=np.int64)
    if len(t) == 0:
        return None
    polys = shapely.polygons(v[t][:, :, :2])
    geom = shapely.union_all(polys)
    geom = make_valid(geom)
    parts = _explode(geom)
    return unary_union(parts) if parts else None


def element_footprints(elements: Iterable) -> dict[str, list[Polygon]]:
    """Footprint polygons per kind, e.g. as input for ``qa_cad.check_topology``."""
    out: dict[str, list[Polygon]] = {k: [] for k in c.PAVEMENT_KINDS}
    for e in elements:
        if isinstance(e, PavementElement):
            fp = element_footprint(e)
            if fp is not None:
                out[e.params.kind].extend(_explode(fp))
    return out


def _ring_with_z(ring_xy: np.ndarray, tree: cKDTree, vz: np.ndarray) -> np.ndarray:
    _, idx = tree.query(ring_xy)
    return np.column_stack([ring_xy, vz[idx]])


def _outline_entity(ring: np.ndarray, frame: c.LocalFrame, hatch: bool) -> dict:
    """Closed LWPOLYLINE at the median Z (Z range <= 0.3 m) or closed polyline3d with per-vertex Z."""
    abs_ring = frame.to_lv95(ring)
    zs = abs_ring[:, 2]
    if float(zs.max() - zs.min()) <= FLAT_Z_RANGE_M:
        return {"type": "lwpolyline", "points": abs_ring.tolist(), "closed": True,
                "elevation": float(np.median(zs)), "hatch": bool(hatch)}
    return {"type": "polyline3d", "points": abs_ring.tolist(), "closed": True}


def to_dxf_entities(elements: Iterable, frame: c.LocalFrame, *, config: dict | None = None) -> dict[str, list[dict]]:
    """DXF entity dicts per layer (S-PAVE / S-WALK / S-PATH), coordinates already in LV95.

    Outline = boundary of the element's top surface. Closed LWPOLYLINE at the median Z when the
    Z range is <= 0.3 m, else closed polyline3d with the TIN Z per vertex. Interior rings (holes)
    become additional closed polylines on the same layer. A HATCH is requested only with
    ``dxf.hatch``.
    """
    cfg = _pave_cfg(config)
    hatch = bool(cfg.get("dxf", {}).get("hatch", False))
    out: dict[str, list[dict]] = {}
    for e in elements:
        if not isinstance(e, PavementElement):
            continue
        fp = element_footprint(e)
        if fp is None:
            continue
        v = np.asarray(e.params.vertices, dtype=np.float64)
        tree = cKDTree(v[:, :2])
        layer = KIND_LAYERS[e.params.kind]
        for poly in _explode(fp):
            poly = poly.simplify(1e-4, preserve_topology=True)
            for k, r in enumerate([poly.exterior, *poly.interiors]):
                xy = np.asarray(r.coords, dtype=np.float64)[:-1]
                if len(xy) < 3:
                    continue
                out.setdefault(layer, []).append(_outline_entity(_ring_with_z(xy, tree, v[:, 2]), frame, hatch and k == 0))
    return out
