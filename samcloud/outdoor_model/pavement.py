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
    snapped_ids: set[tuple[str, int | str]] = field(default_factory=set)
    dropped: list[c.QAEntry] = field(default_factory=list)

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


def _clean_outline(geometry, outline: dict) -> tuple[list[Polygon], dict[str, int]]:
    """Clean one pavement or lawn geometry before it becomes a TIN constraint."""
    geom = make_valid(geometry)
    close = max(0.0, float(outline.get("close_m", 0.25)))
    opening = max(0.0, float(outline.get("open_m", 0.10)))
    if close:
        geom = geom.buffer(close).buffer(-close)
    if opening:
        geom = geom.buffer(-opening).buffer(opening)
    min_area = max(0.0, float(outline.get("min_area_m2", 2.0)))
    min_hole = max(0.0, float(outline.get("min_hole_m2", 1.0)))
    simplify = max(0.0, float(outline.get("simplify_m", 0.05)))
    dropped_parts = dropped_holes = 0
    cleaned: list[Polygon] = []
    for part in _explode(make_valid(geom)):
        if part.area < min_area:
            dropped_parts += 1
            continue
        holes = []
        for ring in part.interiors:
            hole = Polygon(ring)
            if hole.area < min_hole:
                dropped_holes += 1
            else:
                holes.append(ring.coords)
        polygon = Polygon(part.exterior.coords, holes)
        if simplify:
            polygon = polygon.simplify(simplify, preserve_topology=True)
        for clean_part in _explode(make_valid(polygon)):
            if clean_part.area >= min_area:
                cleaned.append(clean_part)
            else:
                dropped_parts += 1
    cleaned.sort(key=lambda p: (round(p.centroid.x, 9), round(p.centroid.y, 9), -p.area))
    return cleaned, {"dropped_parts": dropped_parts, "dropped_holes": dropped_holes}


def _part_identifier(identifier: int | str, number: int, count: int) -> int | str:
    """Preserve a source id for one part and give multipart sources stable suffixes."""
    return identifier if count == 1 else f"{identifier}_{number}"


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
    dropped: list[c.QAEntry] = []
    originals: list[c.PavementPolygon] = []
    for index, polygon in enumerate(polygons):
        ident = f"{getattr(polygon, 'kind', 'pavement')}-{getattr(polygon, 'pavement_id', index)}"
        geometry = getattr(polygon, "polygon", None)
        if geometry is None or geometry.is_empty:
            dropped.append(c.QAEntry("pavement", ident, "dropped empty pavement polygon"))
        elif getattr(polygon, "status", "ok") != "rejected":
            # The upstream classifier may already have split one semantic
            # pavement into a MultiPolygon.  Break it into simple polygons
            # before snap_to_breakline, which operates on polygon exteriors.
            for part in _explode(make_valid(geometry)):
                originals.append(c.PavementPolygon(
                    pavement_id=polygon.pavement_id, kind=polygon.kind, polygon=part,
                    evidence=polygon.evidence, status=polygon.status, reason=polygon.reason,
                ))
    lawns_in = [make_valid(p) for p in lawn_polygons if p is not None and not p.is_empty]
    lawns_in = [q for g in lawns_in for q in _explode(g)]
    axes: list[c.CurbAxis] = []
    for index, axis in enumerate(curbs):
        ident = f"curb-{getattr(axis, 'curb_id', index)}"
        points = getattr(axis, "points", None)
        valid = points is not None and np.asarray(points).ndim == 2 and len(points) >= 2
        if axis is None or not valid:
            dropped.append(c.QAEntry("curb", ident, "dropped empty curb axis"))
        elif getattr(axis, "status", "ok") != "rejected":
            axes.append(axis)

    snapped_items: list[c.PavementPolygon] = []
    if axes:
        for item in originals:
            try:
                snapped_items.extend(topology.snap_to_breakline([item], axes, tolerance=search))
            except AttributeError as exc:
                # A repaired input can still become a multipart geometry in
                # Shapely's segmentizer.  Preserve its simple valid pieces so
                # one malformed outline cannot block the complete TIN run.
                LOGGER.warning("could not snap pavement %s-%s: %s", item.kind, item.pavement_id, exc)
                snapped_items.extend(c.PavementPolygon(
                    pavement_id=item.pavement_id, kind=item.kind, polygon=part,
                    evidence=item.evidence, status=item.status, reason=item.reason,
                ) for part in _explode(make_valid(item.polygon)))
    else:
        snapped_items = list(originals)
    by_kind: dict[str, list[c.PavementPolygon]] = {k: [] for k in c.PAVEMENT_KINDS}
    for item in snapped_items:
        by_kind[item.kind].append(item)
    params = _surface_params(cfg)
    cleaned_groups: dict[str, list[Polygon]] = {}
    for kind in (*_PRIORITY_KINDS, "lawn"):
        source = lawns_in if kind == "lawn" else [p.polygon for p in by_kind[kind]]
        if not source:
            cleaned_groups[kind] = []
            continue
        parts, cleanup = _clean_outline(unary_union(source), params[kind])
        if cleanup["dropped_parts"] or cleanup["dropped_holes"]:
            dropped.append(c.QAEntry("lawn" if kind == "lawn" else "pavement", kind,
                                     "outline cleanup removed small geometry", cleanup))
        cleaned_groups[kind] = parts
    gap_tol = float(pav.get("gap_tolerance_m", max(search, 0.05)))
    groups = [cleaned_groups[k] for k in (*_PRIORITY_KINDS, "lawn")]
    resolved = topology.resolve_overlaps(groups, gap_tolerance=gap_tol)
    min_width = float(cfg.get("surfaces", {}).get("clean", {}).get("min_width_m", 0.0))
    slivers: dict[str, int] = {}
    if min_width > 0:
        resolved, slivers = topology.absorb_slivers(resolved, min_width, names=(*_PRIORITY_KINDS, "lawn"))
    # a common 1 um grid makes neighbouring edges bit-identical (no floating point gaps/overlaps)
    resolved = [[q for part in group for q in _explode(make_valid(shapely.set_precision(part, 1e-6)))] for group in resolved]
    # topology-preserving simplification of the shared linework (neighbours keep identical edges)
    tolerance = float(params["lawn"].get("simplify_m", 0.0))
    flat = [(g, q) for g, group in enumerate(resolved) for q in group]
    if tolerance > 0 and len(flat) > 1:
        try:
            simple = shapely.coverage_simplify([q for _g, q in flat], tolerance)
            resolved = [[] for _ in resolved]
            for (g, _q), new_geom in zip(flat, simple):
                resolved[g].extend(_explode(make_valid(new_geom)))
        except Exception as exc:  # noqa: BLE001 - invalid coverage: keep the resolved polygons
            LOGGER.warning("coverage simplification skipped: %s", exc)
    for number, kind in enumerate((*_PRIORITY_KINDS, "lawn")):
        small = float(params[kind].get("min_area_m2", 0.0))
        kept = [q for q in resolved[number] if q.area >= small]
        if len(kept) != len(resolved[number]):
            dropped.append(c.QAEntry("lawn" if kind == "lawn" else "pavement", kind, "parts below the minimum area fall back to terrain",
                                     {"dropped_parts": len(resolved[number]) - len(kept)}))
        resolved[number] = kept
    for kind, count in slivers.items():
        dropped.append(c.QAEntry("lawn" if kind == "lawn" else "pavement", kind, "narrow slivers merged into the neighbour or dropped",
                                 {"slivers": count}))

    out: list[c.PavementPolygon] = []
    snapped: set[tuple[str, int]] = set()
    orig_by_kind = {k: [p for p in originals if p.kind == k] for k in c.PAVEMENT_KINDS}
    # "snapped" means moved by curb snapping or overlap/gap resolution, not by outline cleaning
    raw_groups = [[p.polygon for p in by_kind[k]] for k in _PRIORITY_KINDS] + [lawns_in]
    raw_resolved = topology.resolve_overlaps(raw_groups, gap_tolerance=gap_tol)
    moved: set[tuple[str, int]] = set()
    for kind, parts in zip(_PRIORITY_KINDS, raw_resolved[:3]):
        pool = orig_by_kind[kind]
        for part in parts:
            src = max(pool, key=lambda p: p.polygon.intersection(part).area, default=None)
            if src is not None and part.symmetric_difference(src.polygon).area > 1e-6:
                moved.add((kind, src.pavement_id))
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
            if (kind, src.pavement_id) in moved:
                snapped.add((kind, src.pavement_id))
            out.append(c.PavementPolygon(
                pavement_id=src.pavement_id, kind=kind, polygon=part,
                evidence=src.evidence, status=src.status, reason=src.reason,
            ))
    grouped: dict[tuple[str, int | str], list[c.PavementPolygon]] = {}
    for item in out:
        grouped.setdefault((item.kind, item.pavement_id), []).append(item)
    cleaned_out: list[c.PavementPolygon] = []
    cleaned_snapped: set[tuple[str, int | str]] = set()
    for (kind, source_id), items in grouped.items():
        parts = _explode(unary_union([item.polygon for item in items]))
        parts.sort(key=lambda p: (round(p.centroid.x, 9), round(p.centroid.y, 9), -p.area))
        source = items[0]
        for number, part in enumerate(parts, start=1):
            part_id = _part_identifier(source_id, number, len(parts))
            cleaned_out.append(c.PavementPolygon(
                pavement_id=part_id, kind=kind, polygon=part, evidence=source.evidence,
                status=source.status, reason=source.reason,
            ))
            if (kind, source_id) in snapped:
                cleaned_snapped.add((kind, part_id))
    lawns: list[Polygon] = []
    for part in resolved[3]:
        lawns.extend(_explode(part))
    lawns.sort(key=lambda p: (round(p.centroid.x, 9), round(p.centroid.y, 9), -p.area))
    out = cleaned_out
    snapped = cleaned_snapped
    named = {k: [p.polygon for p in out if p.kind == k] for k in c.PAVEMENT_KINDS}
    named["lawn"] = lawns
    named = {k: v for k, v in named.items() if v}
    report = {f"{a}|{b}": r for (a, b), r in topology.pairwise_report(named, tolerance=0.05).items()}
    return PavementTopology(polygons=out, lawns=lawns, report=report, snapped_ids=snapped, dropped=dropped)


def _surface_params(cfg: dict) -> dict[str, dict]:
    """Cleaning parameters per class: road keeps ``pavement.outline``, the rest uses ``surfaces.clean``."""
    outline = dict(cfg.get("pavement", {}).get("outline", {}))
    surf = dict(cfg.get("surfaces", {}).get("clean", {}))
    simplify = float(surf.get("simplify_m", outline.get("simplify_m", 0.05)))
    if bool(cfg.get("edges", {}).get("enabled", False)):
        simplify = min(simplify, 0.02)  # the edge fit does the real simplification, keep arcs intact
    areas = surf.get("min_area_m2", {})
    params = {"road": {**outline, "min_area_m2": float(outline.get("min_area_m2", 2.0))}}
    for kind in ("sidewalk", "path", "lawn"):
        base = {**outline, **{k: v for k, v in surf.items() if k not in ("min_area_m2", "min_width_m")}}
        base["simplify_m"] = simplify
        base["min_area_m2"] = float(areas.get(kind, outline.get("min_area_m2", 2.0))) if isinstance(areas, dict) else float(areas)
        params[kind] = base
    return params


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


def _triangle_components(tris: np.ndarray) -> list[np.ndarray]:
    """Return face groups connected by full edges, never only by a vertex."""
    if len(tris) == 0:
        return []
    parent = np.arange(len(tris), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def join(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    owners: dict[tuple[int, int], int] = {}
    for face_index, (a, b, cc) in enumerate(tris):
        for edge in ((a, b), (b, cc), (cc, a)):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            if key in owners:
                join(face_index, owners[key])
            else:
                owners[key] = face_index
    groups: dict[int, list[int]] = {}
    for face_index in range(len(tris)):
        groups.setdefault(find(face_index), []).append(face_index)
    return [np.asarray(group, dtype=np.int64) for group in groups.values()]


def _split_pinch_vertices(verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give every fan of triangles that only meets another fan at one vertex its own copy of that vertex.

    A pavement outline that touches itself at a point (a hole touching the shell, a neck of zero width)
    would otherwise produce a pinched top surface whose boundary loops are ambiguous for the solid.
    """
    incident: dict[int, list[int]] = {}
    for k, tri in enumerate(tris):
        for x in tri:
            incident.setdefault(int(x), []).append(k)
    new_verts = [v for v in verts]
    out = tris.copy()
    for x, ks in incident.items():
        if len(ks) < 2:
            continue
        parent = {k: k for k in ks}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, k1 in enumerate(ks):
            for k2 in ks[i + 1:]:
                if len(set(tris[k1]) & set(tris[k2])) >= 2:
                    parent[find(k1)] = find(k2)
        groups: dict[int, list[int]] = {}
        for k in ks:
            groups.setdefault(find(k), []).append(k)
        for group in list(groups.values())[1:]:
            # the copy sits 0.2 mm inside its own fan so that the two vertices stay distinct when positions are merged
            others = np.unique(out[group].ravel())
            inward = verts[others[others != x], :2].mean(axis=0) - verts[x, :2]
            norm = float(np.hypot(*inward))
            moved = np.array(verts[x], dtype=float)
            if norm > 1e-12:
                moved[:2] += inward / norm * 2e-4
            new_verts.append(moved)
            for k in group:
                out[k][out[k] == x] = len(new_verts) - 1
    return np.asarray(new_verts), out


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
    topology_ready: bool = False,
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
        if p is not None and p.status == "rejected":
            qa.append(c.QAEntry("pavement", f"{p.kind}-{p.pavement_id}", f"rejected by detection: {p.reason}"))
    topo = (PavementTopology(polygons=list(polygons), lawns=list(lawn_polygons), report={}) if topology_ready
            else pavement_topology(polygons, lawn_polygons, curbs, config=cfg))
    qa.extend(topo.dropped)
    if topo.worst_cm2() >= limit_cm2:
        worst = max(topo.report.items(), key=lambda kv: kv[1]["overlap_cm2"] + kv[1]["gap_cm2"])
        qa.append(c.QAEntry("pavement_topology", worst[0], f"gap + overlap {topo.worst_cm2():.2f} cm2 >= {limit_cm2} cm2",
                            {"overlap_cm2": worst[1]["overlap_cm2"], "gap_cm2": worst[1]["gap_cm2"]}))

    vertices = np.asarray(tin.vertices, dtype=np.float64)
    triangles = np.asarray(tin.triangles, dtype=np.int64)
    centroids = vertices[triangles].mean(axis=1) if len(triangles) else np.zeros((0, 3))
    elements: list[PavementElement] = []
    for p in topo.polygons:
        key = (p.kind, p.pavement_id)
        eid = f"pavement-{p.kind}-{p.pavement_id}"
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
        components = _triangle_components(tris)
        for number, component in enumerate(components, start=1):
            verts, tri_idx = _split_pinch_vertices(*_compact_mesh(vertices, tris[component]))
            tri_area = _tri_area(verts, tri_idx)
            ratio = tri_area / p.polygon.area if p.polygon.area > 0 else 0.0
            status = _STATUS_MAP.get(p.status, Status.UNSURE)
            notes: list[str] = []
            if abs(ratio - 1.0) > 0.02:
                status = Status.WARNING if status == Status.OK else status
                notes.append(f"labelled TIN footprint area is {ratio * 100:.1f} % of the source polygon area")
            origin = ParamOrigin.SNAPPED if key in topo.snapped_ids else ParamOrigin.MEASURED
            z = verts[:, 2]
            part_id = eid if len(components) == 1 else f"{eid}_{number}"
            elements.append(PavementElement(
                id=part_id, tag=f"{p.kind}-{p.pavement_id}",
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
            elements[-1]._dxf_footprint = element_footprint(elements[-1])
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


def _outline_entity(ring: np.ndarray, frame: c.LocalFrame, hatch: bool, bulges: np.ndarray | None = None) -> dict:
    """Closed LWPOLYLINE at the median Z (Z range <= 0.3 m) or closed polyline3d with per-vertex Z.

    ``bulges`` (one per segment, DXF convention) turn the flat LWPOLYLINE into lines and arcs.
    """
    abs_ring = frame.to_lv95(ring)
    zs = abs_ring[:, 2]
    if float(zs.max() - zs.min()) <= FLAT_Z_RANGE_M:
        entity = {"type": "lwpolyline", "points": abs_ring.tolist(), "closed": True,
                  "elevation": float(np.median(zs)), "hatch": bool(hatch)}
        if bulges is not None:
            entity["bulges"] = [float(b) for b in bulges]
        return entity
    return {"type": "polyline3d", "points": abs_ring.tolist(), "closed": True}


def to_dxf_entities(elements: Iterable, frame: c.LocalFrame, *, config: dict | None = None, edge_fit=None) -> dict[str, list[dict]]:
    """DXF entity dicts per layer (S-PAVE / S-WALK / S-PATH), coordinates already in LV95.

    Outline = boundary of the element's top surface. Closed LWPOLYLINE at the median Z when the
    Z range is <= 0.3 m, else closed polyline3d with the TIN Z per vertex. Interior rings (holes)
    become additional closed polylines on the same layer. A HATCH is requested only with
    ``dxf.hatch``. With ``edge_fit`` (``samcloud.terrain.edges.EdgeFit``) flat outlines are written
    as lines and arcs (LWPOLYLINE bulges); non-flat ones stay densified (sagitta <= 0.01 m).
    """
    cfg = _pave_cfg(config)
    hatch = bool(cfg.get("dxf", {}).get("hatch", False))
    out: dict[str, list[dict]] = {}
    for e in elements:
        if not isinstance(e, PavementElement):
            continue
        fp = getattr(e, "_dxf_footprint", None) or element_footprint(e)
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
                dense = _ring_with_z(xy, tree, v[:, 2])
                bulge_line = edge_fit.bulge_ring(xy) if edge_fit is not None else None
                if bulge_line is not None and bulge_line.arcs and float(np.ptp(dense[:, 2])) <= FLAT_Z_RANGE_M:
                    anchors = _ring_with_z(bulge_line.pts[:-1], tree, v[:, 2])
                    out.setdefault(layer, []).append(_outline_entity(anchors, frame, hatch and k == 0, bulge_line.bulges))
                else:
                    out.setdefault(layer, []).append(_outline_entity(dense, frame, hatch and k == 0))
    return out


def curb_dxf_entities(curbs: Iterable, frame: c.LocalFrame, *, config: dict | None = None) -> dict[str, list[dict]]:
    """``S-CURB`` entities for fitted curbs (``samcloud.terrain.edges.FittedCurb``).

    Flat curbs (Z range <= ``edges.dxf_curb_flat_z_m``) become an elevated LWPOLYLINE with bulges for the
    arcs; others a 3D polyline with the arcs densified at sagitta <= ``edges.fit.sagitta_m``.
    """
    cfg = _pave_cfg(config)
    flat_limit = float(cfg.get("edges", {}).get("dxf_curb_flat_z_m", 0.05))
    out: list[dict] = []
    min_length = float(cfg.get("curb", {}).get("curb_min_length_m", cfg.get("curb", {}).get("min_length_m", 0.3)))
    for fc in curbs:
        if fc.status == "rejected" or float(np.hypot(*np.diff(fc.dense_xyz[:, :2], axis=0).T).sum()) < min_length:
            continue  # not emitted as a curb element either
        z = np.asarray(fc.z, dtype=np.float64)
        if len(fc.line.pts) >= 2 and float(np.ptp(fc.dense_xyz[:, 2])) <= flat_limit:
            pts = frame.to_lv95(np.column_stack([fc.line.pts, z])).tolist()
            out.append({"type": "lwpolyline", "points": pts, "closed": False, "bulges": [float(b) for b in fc.line.bulges],
                        "elevation": float(np.median(frame.to_lv95(np.column_stack([fc.line.pts, z]))[:, 2]))})
        else:
            from samcloud.outdoor_model.civil import thin_collinear_xy
            dense = thin_collinear_xy(fc.dense_xyz, float(cfg.get("curb", {}).get("dp_tolerance_m", 0.015)))
            out.append({"type": "polyline3d", "points": frame.to_lv95(dense).tolist(), "closed": False})
    return {"S-CURB": out} if out else {}
