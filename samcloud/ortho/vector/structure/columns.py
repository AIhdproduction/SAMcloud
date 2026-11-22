"""Free-standing columns: compact vertical structures with a constant cross-section from the floor to the ceiling.

Candidates are wall- / column-class points in the middle of the storey that lie in a room, clear of every wall body and
opening.  A candidate becomes a column only when

* its cross-section is compact (``min_section <= side <= max_section``, aspect <= ``max_aspect``),
* structural points cover at least ``min_height_share`` of the local clear height and reach the ceiling,
* the cross-section stays the same over the height (a lamp, a coat stand or a person does not),
* furniture-class points do not dominate it and it does not stand in a furniture footprint (unless the points are column class).

The profile is a rotated rectangle or a circle, whichever fits the points better (median distance to the outline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .common import BEAM, CEILING, COLUMN, FLOOR, FOREIGN_CODES, WALL, LevelPoints, StructureSettings, contains
from .levels import RoomLevel


_STRUCTURAL_LABELS = frozenset({"", "column", "pillar", "post", "support post", "structural column", "support pillar", "wall", "beam"})


@dataclass(slots=True)
class Column:
    shape: str  # rect | circle
    center: tuple[float, float]
    footprint: Polygon  # metric plan polygon
    width_m: float
    depth_m: float
    angle_rad: float  # direction of the longer side (rect)
    z_bottom_m: float
    z_top_m: float
    residual_m: float
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


def _voxel_unique(points: np.ndarray, size: float) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / size).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def _rectangle(xy: np.ndarray) -> tuple[Polygon, float, float, float]:
    """Minimum rotated rectangle of the points: (polygon, long side, short side, angle of the long side)."""
    hull = MultiPoint([tuple(p) for p in xy]).convex_hull
    if hull.geom_type != "Polygon":
        hull = hull.buffer(0.005)
    rectangle = hull.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        rectangle = rectangle.buffer(0.005).minimum_rotated_rectangle
    corners = np.asarray(rectangle.exterior.coords)[:4]
    first, second = corners[1] - corners[0], corners[2] - corners[1]
    a, b = float(np.hypot(*first)), float(np.hypot(*second))
    edge = first if a >= b else second
    angle = float(math.atan2(edge[1], edge[0]))
    # The hull rectangle is widened by the measurement noise of the faces: place every side at its face instead.
    u = np.asarray((math.cos(angle), math.sin(angle)))
    v = np.asarray((-u[1], u[0]))
    centre = corners.mean(axis=0)
    pu, pv = (xy - centre) @ u, (xy - centre) @ v
    (u0, u1), (v0, v1) = _face_extent(pu), _face_extent(pv)
    if u1 - u0 > 0.02 and v1 - v0 > 0.02:
        rectangle = Polygon([tuple(centre + u * s + v * t) for s, t in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])
        a, b = u1 - u0, v1 - v0
    return rectangle, max(a, b), min(a, b), angle if a >= b else angle + math.pi / 2.0


def _face_extent(values: np.ndarray, step: float = 0.005) -> tuple[float, float]:
    """Positions of the two outer faces along one rectangle axis.

    Points of a face perpendicular to the axis pile up at its position (their noise spreads along the axis); a hidden face
    leaves only the sharp ends of the two side faces.  Each end therefore takes the prominent peak of the outer quarter of the
    histogram, else the 0.5 % / 99.5 % percentile.
    """
    lo_p, hi_p = (float(v) for v in np.percentile(values, [0.5, 99.5]))
    if hi_p - lo_p < 4 * step:
        return lo_p, hi_p
    bins = int(math.ceil((hi_p - lo_p) / step)) + 1
    counts, edges = np.histogram(values, bins=bins, range=(lo_p, lo_p + bins * step))
    smooth = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    quarter = max(1, bins // 4)
    interior = np.median(smooth[quarter:bins - quarter]) if bins - 2 * quarter > 0 else np.median(smooth)
    centres = 0.5 * (edges[:-1] + edges[1:])
    result = []
    for segment, fallback in ((slice(0, quarter), lo_p), (slice(bins - quarter, bins), hi_p)):
        local = smooth[segment]
        peak = int(np.argmax(local))
        result.append(float(centres[segment][peak]) if local[peak] >= 2.5 * max(interior, 1.0) else fallback)
    return result[0], result[1]


def _circle_fit(xy: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Algebraic (Kasa) circle fit; works on arcs (only the visible side of a column)."""
    if len(xy) < 6:
        return None
    design = np.column_stack((xy[:, 0], xy[:, 1], np.ones(len(xy))))
    target = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    try:
        (d, e, f), *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    centre = np.asarray((-d / 2.0, -e / 2.0))
    squared = float(centre @ centre - f)
    if squared <= 0:
        return None
    return centre, math.sqrt(squared)


def _local_ceiling(point: Point, rooms: list[Polygon], levels: list[RoomLevel], default: float) -> float:
    for room, level in zip(rooms, levels, strict=True):
        if room.covers(point):
            return float(level.ceiling_at(np.asarray([[point.x, point.y]]), default)[0])
    return float(default)


def detect_columns(
    points: LevelPoints, rooms: list[Polygon], levels: list[RoomLevel], walls: BaseGeometry | None, openings: list[Polygon],
    furniture: list[Polygon], level_ceiling_m: float, settings: StructureSettings = StructureSettings(),
    instances: list[dict[str, Any]] | None = None,
) -> tuple[list[Column], list[dict[str, Any]]]:
    """Columns of one level (metric) and the rejected candidates with their reason.

    ``instances`` (optional): ``[{"hull": metric Polygon, "label": str}]`` of the scan instances; a candidate lying mostly
    inside an instance with a non-structural label (door, bed, shelf ...) is no column.
    """
    from sklearn.cluster import DBSCAN

    rejected: list[dict[str, Any]] = []
    if not rooms:
        return [], rejected
    storey = float(level_ceiling_m)
    structural = points.mask((WALL, COLUMN))
    band = structural & (points.xyz[:, 2] >= 0.20 * storey) & (points.xyz[:, 2] <= 0.90 * storey)
    candidates = points.xyz[band]
    room_union = unary_union(rooms).buffer(0.05)
    keep = contains(room_union, candidates[:, :2])
    if walls is not None and not walls.is_empty:
        keep &= ~contains(walls.buffer(settings.column_wall_clearance_m), candidates[:, :2])
    if openings:
        keep &= ~contains(unary_union(openings).buffer(settings.column_opening_clearance_m), candidates[:, :2])
    candidates = _voxel_unique(candidates[keep], 0.02)
    if len(candidates) < settings.column_min_points:
        return [], rejected
    clearance = settings.column_wall_clearance_m
    if walls is not None and not walls.is_empty:
        # Smeared wall faces (noisy / unfused clouds) leave wall points off the face: the fitted column must keep clear of
        # the walls by at least this measured spread (p90 of the off-face distance of wall points within 0.15 m of a face).
        wall_xyz = points.xyz[points.mask((WALL,)) & (points.xyz[:, 2] >= 0.30 * storey) & (points.xyz[:, 2] <= 0.90 * storey)]
        wall_xyz = wall_xyz[:: max(1, len(wall_xyz) // 100_000)]
        off_face = shapely.distance(walls, shapely.points(wall_xyz[:, :2])) if len(wall_xyz) else np.empty(0)
        off_face = off_face[(off_face > 1e-4) & (off_face < 0.15)]
        if len(off_face) >= 50:
            clearance = max(clearance, float(np.percentile(off_face, 90)))
    labels = DBSCAN(eps=max(3.0 * settings.column_cell_m, 0.10), min_samples=8).fit_predict(candidates[:, :2])
    furniture_union = unary_union([item for item in furniture if item.is_valid and not item.is_empty]) if furniture else None
    order = np.argsort(points.xyz[:, 0], kind="stable")
    sorted_xyz, sorted_x, sorted_codes = points.xyz[order], points.xyz[order, 0], points.codes[order]
    columns: list[Column] = []
    for label in sorted(set(labels.tolist()) - {-1}):
        cluster = candidates[labels == label]
        centre = cluster[:, :2].mean(axis=0)
        info: dict[str, Any] = {"center": [round(float(centre[0]), 3), round(float(centre[1]), 3)], "points": int(len(cluster))}
        if len(cluster) < settings.column_min_points:
            continue
        rectangle, long_side, short_side, angle = _rectangle(cluster[:, :2])
        centre = np.asarray(rectangle.centroid.coords[0])
        info.update(long_m=round(long_side, 3), short_m=round(short_side, 3))
        if long_side > settings.column_max_section_m or short_side < settings.column_min_section_m or long_side / max(short_side, 1e-6) > settings.column_max_aspect:
            rejected.append({**info, "reason": "cross_section"})
            continue
        if walls is not None and not walls.is_empty:
            gap = float(rectangle.distance(walls))
            info["wall_gap_m"] = round(gap, 3)
            if gap < clearance:
                rejected.append({**info, "reason": "touches_wall", "clearance_m": round(clearance, 3)})
                continue
            # attached to a wall (a smeared wall end, a stub): structural points fill the strip between the candidate and the
            # wall about as densely as the candidate's own faces.  A free-standing column has free space there.
            strip = rectangle.buffer(2.0 * clearance).intersection(walls.buffer(2.0 * clearance)).difference(rectangle.buffer(0.02)).difference(walls)
            if not strip.is_empty and strip.area > 1e-4:
                lo_x, hi_x = np.searchsorted(sorted_x, strip.bounds[0]), np.searchsorted(sorted_x, strip.bounds[2], "right")
                near_xyz, near_codes = sorted_xyz[lo_x:hi_x], sorted_codes[lo_x:hi_x]
                mid = np.isin(near_codes, (WALL, COLUMN)) & (near_xyz[:, 2] >= 0.2 * storey) & (near_xyz[:, 2] <= 0.9 * storey)
                in_strip = int(contains(strip, near_xyz[mid, :2]).sum()) / strip.area
                shell = rectangle.buffer(0.03).difference(rectangle.buffer(-0.03))
                lo_x, hi_x = np.searchsorted(sorted_x, shell.bounds[0]), np.searchsorted(sorted_x, shell.bounds[2], "right")
                own_xyz, own_codes = sorted_xyz[lo_x:hi_x], sorted_codes[lo_x:hi_x]
                own_mid = np.isin(own_codes, (WALL, COLUMN)) & (own_xyz[:, 2] >= 0.2 * storey) & (own_xyz[:, 2] <= 0.9 * storey)
                on_faces = int(contains(shell, own_xyz[own_mid, :2]).sum()) / max(shell.area, 1e-9)
                info["strip_density_ratio"] = round(in_strip / max(on_faces, 1e-9), 3)
                if in_strip >= settings.column_attached_density * on_faces:
                    rejected.append({**info, "reason": "attached_to_wall"})
                    continue
        point = Point(*centre)
        clear = _local_ceiling(point, rooms, levels, storey)
        lo, hi = np.searchsorted(sorted_x, rectangle.bounds[0] - 0.05), np.searchsorted(sorted_x, rectangle.bounds[2] + 0.05, "right")
        window, codes = sorted_xyz[lo:hi], sorted_codes[lo:hi]
        inside = contains(rectangle.buffer(0.03), window[:, :2]) & (window[:, 2] >= -0.05) & (window[:, 2] <= clear + 0.05)
        column_points, column_codes = window[inside], codes[inside]
        is_structural = np.isin(column_codes, (WALL, COLUMN))
        # class purity: every non-structural class (furniture, door / window frames, railings ...) counts against the column;
        # floor / ceiling / beam points at its ends do not
        own = ~np.isin(column_codes, (FLOOR, CEILING, BEAM))
        foreign = float(1.0 - is_structural[own].sum() / max(int(own.sum()), 1))
        info["foreign_share"] = round(foreign, 3)
        if foreign > settings.column_max_foreign_share:
            rejected.append({**info, "reason": "furniture_points"})
            continue
        z = column_points[is_structural, 2]
        top = float(np.percentile(z, 99)) if len(z) else 0.0
        reach = clear
        if top < clear - settings.column_top_gap_m and top > settings.column_min_height_share * clear:
            # A column may end under a soffit (beam / lowered ceiling): overhead points right above its top, around it.
            near = contains(rectangle.buffer(0.15), window[:, :2]) & np.isin(codes, (CEILING, BEAM))
            soffit = window[near & (window[:, 2] >= top - 0.05) & (window[:, 2] <= top + settings.column_top_gap_m), 2]
            if len(soffit) >= 10:
                reach = float(np.median(soffit))
                info["top"] = "soffit"
        bins = np.arange(0.05 * reach, 0.95 * reach + 1e-9, 0.10)
        counts, _ = np.histogram(z, bins=bins)
        coverage = float((counts >= 3).mean()) if len(counts) else 0.0
        info.update(coverage=round(coverage, 3), top_m=round(top, 3), clear_m=round(clear, 3))
        if coverage < settings.column_min_height_share or top < reach - settings.column_top_gap_m:
            rejected.append({**info, "reason": "height"})
            continue
        # constant cross-section over the height (four slabs of the middle 80 %)
        axis_u = np.asarray((math.cos(angle), math.sin(angle)))
        axis_v = np.asarray((-axis_u[1], axis_u[0]))
        structural_xyz = column_points[is_structural]
        extents = []
        for z0 in np.linspace(0.10 * reach, 0.70 * reach, 4):
            slab = structural_xyz[(structural_xyz[:, 2] >= z0) & (structural_xyz[:, 2] < z0 + 0.20 * reach)]
            if len(slab) < 10:
                continue
            rel = slab[:, :2] - centre
            extents.append(max(np.ptp(np.percentile(rel @ axis_u, [5, 95])), np.ptp(np.percentile(rel @ axis_v, [5, 95]))))
        ratio = float(max(extents) / max(np.median(extents), 1e-6)) if len(extents) >= 3 else float("inf")
        info["slab_ratio"] = round(ratio, 3) if math.isfinite(ratio) else None
        if ratio > settings.column_slab_extent_ratio:
            rejected.append({**info, "reason": "varying_cross_section"})
            continue
        share_108 = float((column_codes == COLUMN).sum() / max(int(is_structural.sum()), 1))
        info["column_class_share"] = round(share_108, 3)
        if furniture_union is not None and not furniture_union.is_empty:
            overlap = rectangle.intersection(furniture_union).area / max(rectangle.area, 1e-9)
            info["furniture_overlap"] = round(float(overlap), 3)
            if overlap > settings.column_max_furniture_overlap and share_108 < 0.5:
                rejected.append({**info, "reason": "in_furniture"})
                continue
        if instances:
            for instance in instances:
                label = str(instance.get("label") or "")
                if label in _STRUCTURAL_LABELS:
                    continue
                share = rectangle.intersection(instance["hull"]).area / max(rectangle.area, 1e-9)
                if share >= settings.column_max_instance_overlap and share_108 < 0.5:
                    info["instance"] = label
                    break
            if "instance" in info:
                rejected.append({**info, "reason": "in_instance"})
                continue
        # profile fit on the middle band
        mid = structural_xyz[(structural_xyz[:, 2] >= 0.2 * reach) & (structural_xyz[:, 2] <= 0.9 * reach), :2]
        mid = mid if len(mid) >= 10 else cluster[:, :2]
        rect_residual = float(np.median(shapely.distance(rectangle.exterior, shapely.points(mid))))
        shape, footprint, residual = "rect", rectangle, rect_residual
        width, depth = long_side, short_side
        fit = _circle_fit(mid)
        if fit is not None and long_side / max(short_side, 1e-6) <= 1.6:
            circle_centre, radius = fit
            circle_residual = float(np.median(np.abs(np.hypot(*(mid - circle_centre).T) - radius)))
            if (circle_residual < 0.7 * rect_residual and settings.column_min_section_m <= 2 * radius <= settings.column_max_section_m
                    and np.hypot(*(circle_centre - centre)) <= radius and 2 * radius >= 0.8 * long_side):
                shape, residual = "circle", circle_residual
                footprint = Point(*circle_centre).buffer(radius, quad_segs=8)
                centre = circle_centre
                width = depth = 2.0 * radius
        info.update(shape=shape, residual_m=round(residual, 4))
        # a column surface is a clean prism: the points lie on the fitted outline within the measurement noise
        limit = min(settings.column_max_residual_m, settings.column_max_relative_residual * min(width, depth))
        if residual > limit:
            rejected.append({**info, "reason": "poor_profile_fit", "residual_limit_m": round(limit, 4)})
            continue
        confidence = float(np.clip(coverage, 0.0, 1.0) * math.exp(-residual / 0.03) * (0.75 + 0.25 * share_108))
        columns.append(Column(
            shape=shape, center=(float(centre[0]), float(centre[1])), footprint=footprint, width_m=float(width), depth_m=float(depth),
            angle_rad=float(angle), z_bottom_m=0.0, z_top_m=float(reach), residual_m=residual, confidence=round(confidence, 3), evidence=info,
        ))
    return columns, rejected
