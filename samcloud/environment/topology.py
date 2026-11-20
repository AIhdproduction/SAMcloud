"""Topological clean-up of neighbouring surfaces: the curb is the master breakline.

Functions accept shapely polygons or ``PavementPolygon`` objects (snapping keeps the type).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.validation import make_valid

from samcloud.drone import contracts as c

LOGGER = logging.getLogger(__name__)

#: Priority of surfaces when they overlap (highest first).
PRIORITY_ORDER = ("curb", "sidewalk", "road", "path", "lawn")
_MIN_AREA = 1e-9


def _explode(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_explode(g))
        return out
    return []


def _as_polygon(item) -> Polygon:
    return item.polygon if isinstance(item, c.PavementPolygon) else item


def _as_line(item) -> LineString:
    if isinstance(item, c.CurbAxis):
        return LineString(np.asarray(item.points)[:, :2])
    if isinstance(item, (LineString, MultiLineString)):
        return shapely.force_2d(item)
    return LineString(np.asarray(item)[:, :2])


def _clean(geom) -> list[Polygon]:
    return [p for p in _explode(make_valid(geom)) if p.area > _MIN_AREA]


def _snap_ring(coords: np.ndarray, target, tolerance: float) -> np.ndarray:
    pts = shapely.points(coords)
    dist = shapely.distance(pts, target)
    move = dist <= tolerance
    out = coords.copy()
    if move.any():
        near = shapely.shortest_line(pts[move], target)
        out[move] = shapely.get_coordinates(shapely.get_point(near, 1)).reshape(-1, 2)
    return out


def _snap_polygon(poly: Polygon, target, tolerance: float, max_segment: float) -> Polygon:
    dense = shapely.segmentize(poly, max_segment)
    shell = _snap_ring(np.asarray(dense.exterior.coords), target, tolerance)
    holes = [_snap_ring(np.asarray(r.coords), target, tolerance) for r in dense.interiors]
    return Polygon(shell, holes)


def snap_to_breakline(polygons: Iterable, breaklines: Iterable, *, tolerance: float = 0.01,
                      max_segment: float = 0.25) -> list:
    """Snap polygon boundaries onto the curb breaklines.

    Both are densified (``max_segment``); every polygon vertex closer than ``tolerance`` to a
    breakline (XY of curb axes, LineStrings or (M,>=2) arrays) is moved to its nearest point on
    that line, so neighbouring road/sidewalk polygons end up sharing the curb line. The result is
    repaired with make_valid. Same element type in and out; a polygon that splits yields several.
    """
    lines = [_as_line(b) for b in breaklines]
    items = list(polygons)
    if not lines:
        return items
    target = unary_union([shapely.segmentize(ln, max_segment) for ln in lines])
    out: list = []
    for item in items:
        poly = _as_polygon(item)
        snapped = _snap_polygon(poly, target, tolerance, max_segment)
        parts = _clean(snapped)
        if isinstance(item, c.PavementPolygon):
            for k, part in enumerate(parts):
                out.append(c.PavementPolygon(
                    pavement_id=item.pavement_id, kind=item.kind, polygon=part,
                    evidence=item.evidence, status=item.status,
                    reason=item.reason if k == 0 else (item.reason or "split by snapping"),
                ))
        else:
            out.extend(parts)
    return out


def _gap_pieces(union_geom, tolerance: float) -> list[Polygon]:
    """Pieces of ``closing(union, tolerance) - union``: gaps narrower than ``tolerance``."""
    r = tolerance / 2.0 + 1e-9
    closed = union_geom.buffer(r, join_style="mitre", mitre_limit=2.0).buffer(-r, join_style="mitre", mitre_limit=2.0)
    return [p for p in _explode(closed.difference(union_geom)) if p.area > _MIN_AREA]


def resolve_overlaps(
    polygons_by_priority: Sequence[Iterable],
    *,
    gap_tolerance: float = 0.05,
) -> list[list[Polygon]]:
    """Resolve overlaps and close narrow gaps between groups of polygons.

    ``polygons_by_priority`` is a sequence of polygon groups, highest priority first (see
    ``PRIORITY_ORDER``: curb buffer > sidewalk > road > path > lawn). Every group loses the area
    covered by higher-priority groups. Gaps narrower than ``gap_tolerance`` that touch at least
    two groups are given to the neighbour with the longest shared edge. Returns one polygon list
    per input group, in the same order.
    """
    materialised = [list(g) for g in polygons_by_priority]
    groups = [unary_union([_as_polygon(p) for p in g]) if g else Polygon() for g in materialised]
    resolved: list = []
    covered = Polygon()
    for geom in groups:
        geom = make_valid(geom)
        cut = geom.difference(covered) if not covered.is_empty else geom
        resolved.append(cut)
        covered = unary_union([covered, cut]) if not cut.is_empty else covered
    if gap_tolerance > 0 and not covered.is_empty:
        for piece in _gap_pieces(covered, gap_tolerance):
            ring = piece.buffer(1e-6)
            lengths = [
                0.0 if g.is_empty else float(ring.intersection(g.boundary).length) for g in resolved
            ]
            if sum(1 for ln in lengths if ln > 1e-4) < 2:
                continue
            best = int(np.argmax(lengths))
            resolved[best] = unary_union([resolved[best], piece])
    return [_clean(g) for g in resolved]


def _union(polys: Iterable) -> Polygon:
    items = [make_valid(_as_polygon(p)) for p in polys]
    return unary_union(items) if items else Polygon()


def gap_overlap_report(polys_a: Iterable, polys_b: Iterable, *, region=None, tolerance: float = 0.05,
                       max_cm2: float = 1.0) -> dict:
    """Overlap and gap between two polygon groups.

    ``overlap_m2`` = area of A intersect B. ``gap_m2`` = area of slivers narrower than
    ``tolerance`` that lie between A and B (closing of A+B minus A+B, touching both). ``region``
    optionally clips everything. ``ok`` is True when gap + overlap < ``max_cm2`` cm2 and all
    geometry is valid.
    """
    pa, pb = list(polys_a), list(polys_b)
    valid_a = all(_as_polygon(p).is_valid for p in pa)
    valid_b = all(_as_polygon(p).is_valid for p in pb)
    a, b = _union(pa), _union(pb)
    if region is not None:
        a, b = a.intersection(region), b.intersection(region)
    overlap = float(a.intersection(b).area)
    gap = 0.0
    if not a.is_empty and not b.is_empty:
        both = unary_union([a, b])
        for piece in _gap_pieces(both, tolerance):
            ring = piece.buffer(1e-6)
            if ring.intersects(a) and ring.intersects(b) and (
                ring.intersection(a.boundary).length > 1e-4 and ring.intersection(b.boundary).length > 1e-4
            ):
                gap += piece.area if region is None else piece.intersection(region).area
    return {
        "overlap_m2": overlap,
        "gap_m2": float(gap),
        "overlap_cm2": overlap * 1e4,
        "gap_cm2": float(gap) * 1e4,
        "is_valid_a": bool(valid_a),
        "is_valid_b": bool(valid_b),
        "is_valid": bool(valid_a and valid_b),
        "ok": bool(valid_a and valid_b and (overlap + gap) * 1e4 < max_cm2),
    }


def pairwise_report(named: dict[str, Iterable], *, tolerance: float = 0.05) -> dict[tuple[str, str], dict]:
    """:func:`gap_overlap_report` for every neighbouring pair of named polygon groups."""
    names = list(named)
    groups = {k: list(v) for k, v in named.items()}
    out: dict[tuple[str, str], dict] = {}
    for i, na in enumerate(names):
        for nb in names[i + 1 :]:
            ua, ub = _union(groups[na]), _union(groups[nb])
            if ua.is_empty or ub.is_empty or ua.distance(ub) > tolerance:
                continue
            out[(na, nb)] = gap_overlap_report(groups[na], groups[nb], tolerance=tolerance)
    return out
