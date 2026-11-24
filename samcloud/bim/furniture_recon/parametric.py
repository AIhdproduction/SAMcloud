"""Parametric furniture models fitted to the cleaned item points (item frame, metres, z above the floor).

Scan points lie on surfaces: a horizontal slice through a cabinet holds only its outline, while a slice at a table top,
seat or mattress holds the whole area.  Horizontal surfaces are therefore found as slices with a large occupied area,
supports (legs, pedestal, trestle panels) as point columns below such a surface, backs / arms / headboards as raised
strips along the footprint edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.polygon import orient

from samcloud.bim.furniture_recon.geometry import Solid, box, cylinder, extrusion, robust_union
from samcloud.bim.furniture_recon.settings import ReconSettings


@dataclass(frozen=True, slots=True)
class Level:
    top: float  # z of the surface
    band_low: float  # lower end of the dense band below it (thickness evidence)
    area: float  # m2 occupied at the surface slice


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def surface_level(points: np.ndarray, low: float, high: float, *, cell: float = 0.03, slice_m: float = 0.02, min_area: float = 0.015, share: float = 0.5) -> Level | None:
    """Highest dense horizontal surface between ``low`` and ``high`` (slice area at least ``share`` of the densest slice)."""
    pts = points[(points[:, 2] >= low) & (points[:, 2] <= high)]
    if len(pts) < 10:
        return None
    edges = np.arange(low, high + slice_m, slice_m)
    if len(edges) < 2:
        return None
    index = np.clip(((pts[:, 2] - low) / slice_m).astype(int), 0, len(edges) - 2)
    cells = np.floor(pts[:, :2] / cell).astype(np.int64)
    keys = (cells[:, 0] * 1_000_003 + cells[:, 1]) * 4096 + index
    unique = np.unique(keys)
    per_slice = np.bincount((unique % 4096).astype(int), minlength=len(edges) - 1)[: len(edges) - 1]
    areas = per_slice * cell * cell
    best = float(areas.max())
    if best < min_area:
        return None
    candidates = np.flatnonzero(areas >= share * best)
    top_slice = int(candidates.max())
    bottom_slice = top_slice
    while bottom_slice - 1 >= 0 and areas[bottom_slice - 1] >= 0.35 * best:
        bottom_slice -= 1
    # the surface height is the densest 5 mm level around the top slice (the slice holds the surface and the faces below it)
    near = pts[(pts[:, 2] >= edges[top_slice] - slice_m) & (pts[:, 2] <= edges[top_slice + 1] + slice_m)]
    if len(near) >= 5:
        fine_edges = np.arange(edges[top_slice] - slice_m, edges[top_slice + 1] + slice_m + 0.005, 0.005)
        counts, fine_edges = np.histogram(near[:, 2], bins=fine_edges)
        smooth = np.convolve(counts, np.ones(3) / 3.0, mode="same")
        peak = int(np.argmax(smooth))
        top = float(fine_edges[peak + 1])
    else:
        top = float(edges[top_slice + 1])
    return Level(top=top, band_low=float(edges[bottom_slice]), area=float(areas[top_slice]))


# ----------------------------------------------------------------------------------------------------------------------
# outlines
# ----------------------------------------------------------------------------------------------------------------------

def occupancy_polygon(xy: np.ndarray, cell: float = 0.02) -> Polygon | None:
    if len(xy) < 3:
        return None
    unique = np.unique(np.round(xy / (cell / 2.0)).astype(np.int64), axis=0) * (cell / 2.0)
    shape = MultiPoint([tuple(p) for p in unique]).buffer(cell * 1.25, resolution=4).buffer(-cell * 0.75, resolution=4)
    if shape.is_empty:
        return None
    if shape.geom_type == "MultiPolygon":
        shape = max(shape.geoms, key=lambda item: item.area)
    return Polygon(shape.exterior) if isinstance(shape, Polygon) else None


def _iou(a: Polygon, b: Polygon) -> float:
    from shapely.errors import GEOSException

    try:
        union = a.union(b).area
        return float(a.intersection(b).area / union) if union > 0 else 0.0
    except GEOSException:
        a, b = a.buffer(0), b.buffer(0)
        union = robust_union([a, b]).area
        return float(a.buffer(1e-9).intersection(b.buffer(1e-9)).area / union) if union > 0 else 0.0


def top_outline(xy: np.ndarray, footprint: Polygon) -> tuple[str, Any]:
    """Best plan shape of a top surface: ("box", (x0, x1, y0, y1)) | ("cylinder", (cx, cy, r)) | ("extrusion", Polygon)."""
    x0, x1 = (float(v) for v in np.percentile(xy[:, 0], [1, 99]))
    y0, y1 = (float(v) for v in np.percentile(xy[:, 1], [1, 99]))
    # a top never reaches far beyond the plan footprint (points of neighbours, a chair back under the edge)
    fx0, fy0, fx1, fy1 = footprint.bounds
    x0, x1, y0, y1 = max(x0, fx0 - 0.05), min(x1, fx1 + 0.05), max(y0, fy0 - 0.05), min(y1, fy1 + 0.05)
    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
        return "box", (fx0, fx1, fy0, fy1)
    rect = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    occupied = occupancy_polygon(xy)
    if occupied is None or occupied.area < 0.3 * rect.area:
        return "box", (x0, x1, y0, y1)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    angles = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    ellipse = Polygon(np.column_stack((cx + hx * np.cos(angles), cy + hy * np.sin(angles))))
    scores = {"box": _iou(rect, occupied), "ellipse": _iou(ellipse, occupied)}
    hull = occupied.convex_hull
    scores["hull"] = _iou(hull, occupied) if isinstance(hull, Polygon) else 0.0
    if scores["ellipse"] > scores["box"] + 0.03:
        if max(hx, hy) / max(min(hx, hy), 1e-6) <= 1.12:
            return "cylinder", (cx, cy, (hx + hy) / 2.0)
        return "extrusion", ellipse
    if scores["hull"] > max(scores["box"], scores["ellipse"]) + 0.08 and isinstance(hull, Polygon):
        return "extrusion", hull.simplify(0.01)
    return "box", (x0, x1, y0, y1)


def _top_solid(kind: str, spec: Any, z0: float, z1: float, role: str) -> Solid:
    if kind == "cylinder":
        return cylinder(role, spec[0], spec[1], spec[2], z0, z1)
    if kind == "extrusion":
        return extrusion(role, spec, z0, z1)
    return box(role, spec[0], spec[1], spec[2], spec[3], z0, z1)


# ----------------------------------------------------------------------------------------------------------------------
# supports
# ----------------------------------------------------------------------------------------------------------------------

def find_supports(points: np.ndarray, z_low: float, z_high: float, top: Polygon, *, cell: float = 0.02, max_supports: int = 8) -> list[Solid]:
    """Point columns between ``z_low`` and ``z_high`` under ``top``: legs, panels (trestle / frame) and pedestals."""
    span = z_high - z_low
    if span < 0.08:
        return []
    pts = points[(points[:, 2] >= z_low) & (points[:, 2] <= z_high)]
    if len(pts) < 6:
        return []
    import shapely

    pts = pts[shapely.contains_xy(top.buffer(0.05), pts[:, 0], pts[:, 1])]
    if len(pts) < 6:
        return []
    origin = pts[:, :2].min(axis=0) - cell
    keys = np.floor((pts[:, :2] - origin) / cell).astype(int)
    shape = keys.max(axis=0) + 2
    grid = np.zeros(tuple(shape), dtype=np.int32)
    np.add.at(grid, (keys[:, 0], keys[:, 1]), 1)
    # only point columns that rise over a good part of the height carry the top: a stretcher or rail between two trestle
    # panels (one height band) must not join them into one block
    low = np.full(tuple(shape), np.inf)
    high = np.full(tuple(shape), -np.inf)
    np.minimum.at(low, (keys[:, 0], keys[:, 1]), pts[:, 2])
    np.maximum.at(high, (keys[:, 0], keys[:, 1]), pts[:, 2])
    columns = (grid > 0) & (high - low >= 0.25 * span)
    columns = ndimage.binary_dilation(columns, structure=np.ones((3, 3), bool), iterations=2) & (grid > 0)
    if not columns.any():
        columns = grid > 0
    occupied = ndimage.binary_dilation(columns, structure=np.ones((3, 3), bool))
    labels, count = ndimage.label(occupied, structure=np.ones((3, 3), bool))
    owner = labels[keys[:, 0], keys[:, 1]]
    owner[~columns[keys[:, 0], keys[:, 1]]] = 0
    candidates: list[tuple[float, Solid]] = []
    for label in range(1, count + 1):
        member = pts[owner == label]
        if len(member) < 6:
            continue
        z5, z95 = (float(v) for v in np.percentile(member[:, 2], [5, 95]))
        cover = (z95 - z5) / span
        # a support carries the top: it reaches up close to it (things lying on the floor under a table do not)
        if cover < 0.35 or z95 < z_low + 0.6 * span:
            continue
        u0, u1 = (float(v) for v in np.percentile(member[:, 0], [3, 97]))
        v0, v1 = (float(v) for v in np.percentile(member[:, 1], [3, 97]))
        eu, ev = u1 - u0 + cell * 0.5, v1 - v0 + cell * 0.5
        cu, cv = (u0 + u1) / 2.0, (v0 + v1) / 2.0
        major, minor = max(eu, ev), min(eu, ev)
        if major <= 0.12:
            size = _clamp(major, 0.025, 0.12)
            solid = box("leg", cu - size / 2, cu + size / 2, cv - size / 2, cv + size / 2, 0.0, z_high)
        elif minor <= 0.12:
            thickness = _clamp(minor, 0.02, 0.12)
            if eu >= ev:
                solid = box("panel", u0, u1, cv - thickness / 2, cv + thickness / 2, 0.0, z_high)
            else:
                solid = box("panel", cu - thickness / 2, cu + thickness / 2, v0, v1, 0.0, z_high)
        else:
            # a pedestal stands under the middle of the top; a big blob elsewhere is clutter (bag, chair seat, box)
            tx0, ty0, tx1, ty1 = top.bounds
            if abs(cu - (tx0 + tx1) / 2) > 0.35 * (tx1 - tx0) / 2 + 0.05 or abs(cv - (ty0 + ty1) / 2) > 0.35 * (ty1 - ty0) / 2 + 0.05:
                continue
            if cover < 0.6:  # a column carrying the top is seen over most of its height
                continue
            if major / max(minor, 1e-6) < 1.35:
                solid = cylinder("pedestal", cu, cv, (eu + ev) / 4.0, 0.0, z_high)
            else:
                solid = box("pedestal", u0, u1, v0, v1, 0.0, z_high)
        candidates.append((cover * len(member), solid))
    candidates.sort(key=lambda item: -item[0])
    return [solid for _, solid in candidates[:max_supports]]


def complete_legs(supports: list[Solid], top: tuple[float, float, float, float], z_high: float) -> list[Solid]:
    """Two or three corner legs of a rectangular top: the hidden ones mirrored across the top's axes (``symmetry``)."""
    # legs and short panels (a leg smeared by the scan) at the corners
    legs = [s for s in supports if s.role == "leg" or (s.role == "panel" and max(s.sx, s.sy) <= 0.25)]
    if not 2 <= len(legs) <= 3 or len(supports) != len(legs):
        return supports
    x0, x1, y0, y1 = top
    cu, cv, hu, hv = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    corner = [leg for leg in legs if abs(leg.cx - cu) > 0.5 * hu and abs(leg.cy - cv) > 0.5 * hv]
    if len(corner) < 2:
        return supports
    result = list(supports)
    for leg in corner:
        for su, sv in ((-1, 1), (1, -1), (-1, -1)):
            x, y = cu + su * (leg.cx - cu), cv + sv * (leg.cy - cv)
            if all(math.hypot(x - other.cx, y - other.cy) > 0.10 for other in result) and len(result) < 4:
                result.append(box("leg", x - leg.sx / 2, x + leg.sx / 2, y - leg.sy / 2, y + leg.sy / 2, 0.0, z_high, source="symmetry"))
    return result


def prior_legs(kind: str, spec: Any, z_high: float, settings: ReconSettings) -> list[Solid]:
    """Class prior supports for an elevated top without any support evidence (``prior``)."""
    if kind == "cylinder":
        return [cylinder("pedestal", spec[0], spec[1], max(0.04, 0.15 * spec[2]), 0.0, z_high, source="prior")]
    if kind == "extrusion":
        x0, y0, x1, y1 = spec.bounds
    else:
        x0, x1, y0, y1 = spec
    size, inset = settings.leg_prior_size_m, settings.leg_prior_inset_m
    legs = []
    for x in (x0 + inset, x1 - inset - size):
        for y in (y0 + inset, y1 - inset - size):
            legs.append(box("leg", x, x + size, y, y + size, 0.0, z_high, source="prior"))
    return legs


# ----------------------------------------------------------------------------------------------------------------------
# raised strips (backs, arms, headboards)
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Strip:
    edge: int
    length: float
    coverage: float
    thickness: float
    top: float
    solid_spec: tuple[float, float, float, float, float]  # cx, cy, along, thickness, yaw
    count: int


def raised_strips(points: np.ndarray, polygon: Polygon, depth_max: float, *, min_edge: float = 0.30, thickness_percentile: float = 90.0, min_thickness: float = 0.04) -> list[Strip]:
    """Per footprint edge: the points within ``depth_max`` inside it (a back, arm or headboard along that edge)."""
    if len(points) < 15:
        return []
    ring = np.asarray(orient(polygon, 1.0).exterior.coords, dtype=float)
    strips: list[Strip] = []
    for index in range(len(ring) - 1):
        a, b = ring[index], ring[index + 1]
        length = float(np.hypot(*(b - a)))
        if length < min_edge:
            continue
        d = (b - a) / length
        n = np.asarray((-d[1], d[0]))  # interior on the left of a counter-clockwise ring
        rel = points[:, :2] - a
        along = rel @ d
        inward = rel @ n
        inside = (along >= -0.02) & (along <= length + 0.02) & (inward >= -0.05) & (inward <= depth_max)
        if int(inside.sum()) < 15:
            continue
        a_in, w_in, z_in = along[inside], inward[inside], points[inside, 2]
        bins = np.clip((a_in / 0.10).astype(int), 0, max(int(math.ceil(length / 0.10)) - 1, 0))
        coverage = float(np.count_nonzero(np.bincount(bins, minlength=max(int(math.ceil(length / 0.10)), 1)) >= 2) / max(int(math.ceil(length / 0.10)), 1))
        thickness = _clamp(float(np.percentile(w_in, thickness_percentile)), min_thickness, depth_max)
        a0, a1 = (float(v) for v in np.percentile(a_in, [2, 98]))
        a0, a1 = max(a0, 0.0), min(a1, length)
        if a1 - a0 < 0.10:
            continue
        centre = a + d * (a0 + a1) / 2.0 + n * thickness / 2.0
        strips.append(Strip(index, length, coverage, thickness, float(np.percentile(z_in, 97)), (float(centre[0]), float(centre[1]), a1 - a0, thickness, math.atan2(d[1], d[0])), int(inside.sum())))
    return strips


def _strip_solid(strip: Strip, role: str, z0: float) -> Solid:
    cx, cy, along, thickness, yaw = strip.solid_spec
    return Solid("box", role, float(z0), float(strip.top), "measured", cx=cx, cy=cy, sx=along, sy=thickness, yaw=yaw)


# ----------------------------------------------------------------------------------------------------------------------
# families
# ----------------------------------------------------------------------------------------------------------------------

def _real_surface(points: np.ndarray, level: Level | None, footprint: Polygon, share: float = 0.3, band: float = 0.04, cell: float = 0.03) -> bool:
    """A seat / top / mattress surface covers a good part of the footprint seen from above (a back or armrest top does not).

    Soft surfaces (blanket, cushions) undulate, so the area is taken over a band of +-``band`` around the level."""
    if level is None:
        return False
    near = points[np.abs(points[:, 2] - level.top) <= band]
    if not len(near):
        return False
    area = len(np.unique(np.floor(near[:, :2] / cell).astype(np.int64), axis=0)) * cell * cell
    return area >= share * footprint.area


def fit_table(points: np.ndarray, footprint: Polygon, settings: ReconSettings, window: tuple[float, float]) -> list[Solid] | None:
    level = surface_level(points, max(0.25, window[0]), window[1])
    if level is None or level.top < 0.25 or not _real_surface(points, level, footprint):
        return None
    thickness = _clamp(level.top - level.band_low, 0.02, settings.table_top_max_thickness_m)
    top_points = points[(points[:, 2] >= level.top - max(thickness, 0.03) - 0.01) & (points[:, 2] <= level.top + 0.03)]
    if len(top_points) < 10:
        return None
    kind, spec = top_outline(top_points[:, :2], footprint)
    z_under = level.top - thickness
    top = _top_solid(kind, spec, z_under, level.top, "top")
    supports = find_supports(points, 0.03, z_under - 0.02, top.outline())
    for solid in supports:
        solid.z1 = z_under
    if kind == "box":
        supports = complete_legs(supports, spec, z_under)
    if not supports:
        supports = prior_legs(kind, spec, z_under, settings)
    return [top, *supports]


def fit_chair(points: np.ndarray, footprint: Polygon, settings: ReconSettings, window: tuple[float, float]) -> list[Solid] | None:
    level = surface_level(points, 0.28, min(0.70, window[1]))
    fx0, fy0, fx1, fy1 = footprint.bounds
    if _real_surface(points, level, footprint):
        seat_top = level.top
        thickness = _clamp(seat_top - level.band_low, 0.03, 0.10)
        seat_points = points[(points[:, 2] >= seat_top - thickness - 0.015) & (points[:, 2] <= seat_top + 0.02)]
        if len(seat_points) < 10:
            return None
        x0, x1 = (float(v) for v in np.percentile(seat_points[:, 0], [3, 97]))
        y0, y1 = (float(v) for v in np.percentile(seat_points[:, 1], [3, 97]))
        x0, x1, y0, y1 = max(x0, fx0 - 0.03), min(x1, fx1 + 0.03), max(y0, fy0 - 0.03), min(y1, fy1 + 0.03)
        seat_source = "measured"
    else:
        # seat hidden (chair pushed under a table) but the back is seen high above the floor: the seat is the class prior
        high_only = points[points[:, 2] >= settings.chair_seat_prior_m + 0.10]
        if len(high_only) < 30 or not (0.30 <= fx1 - fx0 <= 0.90 and 0.30 <= fy1 - fy0 <= 0.90):
            return None
        seat_top, thickness = settings.chair_seat_prior_m, 0.05
        x0, x1, y0, y1 = fx0 + 0.02, fx1 - 0.02, fy0 + 0.02, fy1 - 0.02
        seat_source = "prior"
    if x1 - x0 < 0.15 or y1 - y0 < 0.15:
        return None
    seat_low = seat_top - thickness
    solids = [box("seat", x0, x1, y0, y1, seat_low, seat_top, source=seat_source)]
    high = points[(points[:, 2] > seat_top + 0.08)]
    back_edge: tuple[int, int] | None = None
    if len(high) >= max(15, int(0.04 * len(points))):
        cu, cv, hu, hv = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        nu, nv = (float(np.median(high[:, 0])) - cu) / hu, (float(np.median(high[:, 1])) - cv) / hv
        axis = 0 if abs(nu) >= abs(nv) else 1
        sign = 1 if (nu if axis == 0 else nv) >= 0 else -1
        perpendicular = high[:, axis]
        along = high[:, 1 - axis]
        p5, p95 = (float(v) for v in np.percentile(perpendicular, [5, 95]))
        t = _clamp(p95 - p5, 0.02, 0.15)
        centre = float(np.median(perpendicular))
        a0, a1 = (float(v) for v in np.percentile(along, [3, 97]))
        limits = (y0, y1) if axis == 0 else (x0, x1)
        a0, a1 = max(a0, limits[0] - 0.03), min(a1, limits[1] + 0.03)
        top = float(np.percentile(high[:, 2], 98))
        # a back stands at the seat edge (not in the middle of the seat) and is clearly above it
        edge_distance = (hu if axis == 0 else hv) - abs(centre - (cu if axis == 0 else cv))
        if top - seat_top >= 0.12 and a1 - a0 >= 0.15 and edge_distance <= 0.12 + (0.10 if seat_source == "prior" else 0.0):
            if axis == 0:
                solids.append(box("back", centre - t / 2, centre + t / 2, a0, a1, seat_top, top))
            else:
                solids.append(box("back", a0, a1, centre - t / 2, centre + t / 2, seat_top, top))
            back_edge = (axis, sign)
    # armrests: raised side strips perpendicular to the back, below the back's top
    if back_edge is not None:
        seat_poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        arm_points = points[(points[:, 2] > seat_top + 0.12) & (points[:, 2] < seat_top + 0.40)]
        for strip in raised_strips(arm_points, seat_poly, 0.12, min_edge=0.15):
            yaw = strip.solid_spec[4]
            runs_along_u = abs(math.cos(yaw)) > 0.7
            # back thin along v (axis 1) runs along u: its arms run along v, and vice versa
            perpendicular_to_back = (not runs_along_u) if back_edge[0] == 1 else runs_along_u
            if perpendicular_to_back and strip.coverage >= 0.6 and strip.count >= 15 and strip.top - seat_top >= 0.15:
                solids.append(_strip_solid(strip, "arm", seat_top))
    if seat_source == "prior" and back_edge is None:
        return None  # neither seat nor back seen: nothing chair-like to model
    supports = find_supports(points, 0.03, seat_low - 0.02, solids[0].outline())
    for solid in supports:
        solid.z1 = seat_low
    supports = complete_legs(supports, (x0, x1, y0, y1), seat_low)
    if not supports and seat_low >= 0.25:
        supports = prior_legs("box", (x0, x1, y0, y1), seat_low, settings)
    return [*solids, *supports]


def fit_sofa(points: np.ndarray, footprint: Polygon, settings: ReconSettings, window: tuple[float, float]) -> list[Solid] | None:
    level = surface_level(points, 0.25, min(0.65, window[1]))
    if not _real_surface(points, level, footprint):
        return None
    seat_top = level.top
    base = extrusion("base", footprint, 0.0, seat_top)
    high = points[points[:, 2] > seat_top + 0.08]
    minx, miny, maxx, maxy = footprint.bounds
    depth_max = _clamp(0.45 * min(maxx - minx, maxy - miny), 0.10, 0.45)
    strips = [s for s in raised_strips(high, footprint, depth_max) if s.coverage >= 0.5 and s.top - seat_top >= 0.10 and s.count >= 30]
    if not strips:
        return [base]
    top_max = max(s.top for s in strips)
    long_edge = max(s.length for s in strips)
    backs = [s for s in strips if s.top >= top_max - 0.06 and s.length >= 0.6 * long_edge]
    solids = [base, *(_strip_solid(strip, "back", seat_top) for strip in backs)]
    # arms: the remaining edges, measured without the back's points (the back corner would lift them to its height)
    import shapely

    back_area = robust_union([solid.outline().buffer(0.02) for solid in solids[1:]])
    free = high[~shapely.contains_xy(back_area, high[:, 0], high[:, 1])]
    back_edges = {strip.edge for strip in backs}
    for strip in raised_strips(free, footprint, depth_max):
        if strip.edge in back_edges or strip.coverage < 0.5 or strip.top - seat_top < 0.10 or strip.count < 30:
            continue
        solids.append(_strip_solid(strip, "arm", seat_top))
    return solids


def _trim_to_sleeping_surface(points: np.ndarray, footprint: Polygon, top: float, settings: ReconSettings) -> Polygon:
    """The bed's base outline without a low neighbour the plan outline swallowed (a bench, draped blanket, bedside piece).

    The outline of a bed comes from the plan's height map, where a low object touching the bed joins its mask.  The sleeping
    surface itself is where points stand at mattress / bedding height (>= ``bed_surface_share`` of the surface level): a side of the outline
    that reaches more than ``bed_trim_gap_m`` beyond those points is cut back to them (+ the footprint margin).  Only the box of the
    outline is cut, and only when at least 60 % of its area remains; a bed measured over its whole outline keeps it.
    """
    raised = points[points[:, 2] >= settings.bed_surface_share * top]
    if len(raised) < max(50, int(0.3 * len(points))):
        return footprint
    margin = settings.footprint_margin_m
    bounds = list(footprint.bounds)  # minx, miny, maxx, maxy
    for axis in (0, 1):
        low, high = (float(v) for v in np.percentile(raised[:, axis], [0.5, 99.5]))
        if low - bounds[axis] > settings.bed_trim_gap_m:
            bounds[axis] = low - margin
        if bounds[axis + 2] - high > settings.bed_trim_gap_m:
            bounds[axis + 2] = high + margin
    if tuple(bounds) == tuple(footprint.bounds):
        return footprint
    from shapely.geometry import box as _box

    trimmed = footprint.intersection(_box(*bounds))
    if isinstance(trimmed, Polygon) and not trimmed.is_empty and trimmed.area >= 0.6 * footprint.area:
        return trimmed
    return footprint


def fit_bed(points: np.ndarray, footprint: Polygon, settings: ReconSettings, window: tuple[float, float]) -> list[Solid] | None:
    level = surface_level(points, 0.20, min(0.90, window[1]))
    if not _real_surface(points, level, footprint):
        return None
    top = level.top
    footprint = _trim_to_sleeping_surface(points, footprint, top, settings)
    solids = [extrusion("base_mattress", footprint, 0.0, top)]
    high = points[points[:, 2] > top + 0.12]
    minx, miny, maxx, maxy = footprint.bounds
    short = min(maxx - minx, maxy - miny)
    boards = [
        s for s in raised_strips(high, footprint, 0.20, thickness_percentile=75.0, min_thickness=0.03)
        if s.length <= 1.15 * short and s.coverage >= 0.5 and s.top >= top + 0.15 and s.count >= 30
    ]
    boards.sort(key=lambda s: -s.top)
    for index, strip in enumerate(boards[:2]):
        solids.append(_strip_solid(strip, "headboard" if index == 0 else "footboard", 0.0))
    return solids


def top_surface_below_objects(points: np.ndarray, footprint: Polygon, z_high: float, *, max_top: float = 1.30, cell: float = 0.03) -> float | None:
    """Height of a low piece's top when smaller things stand on it (TV on a TV stand, lamp on a nightstand): a dense
    surface covering at least half the footprint with less than 40 % of the footprint occupied above it; else None."""
    level = surface_level(points, 0.25, min(max_top, z_high))
    if level is None or level.top >= z_high - 0.05 or not _real_surface(points, level, footprint, share=0.5):
        return None
    above = points[points[:, 2] > level.top + 0.05]
    if not len(above):
        return level.top
    area = len(np.unique(np.floor(above[:, :2] / cell).astype(np.int64), axis=0)) * cell * cell
    return level.top if area < 0.4 * footprint.area else None


def fit_box(points: np.ndarray, footprint: Polygon, settings: ReconSettings, *, hanging: bool, low_piece: bool = False) -> list[Solid] | None:
    """Oriented box from robust percentiles; an axis the scan saw only partly keeps the footprint extent.

    ``low_piece`` (sideboard, TV stand, nightstand, chest of drawers): objects standing on its top are not part of it."""
    if len(points) < 10:
        return None
    fx0, fy0, fx1, fy1 = footprint.bounds
    margin = settings.footprint_margin_m
    extents = []
    for axis, (f0, f1) in enumerate(((fx0, fx1), (fy0, fy1))):
        m0, m1 = (float(v) for v in np.percentile(points[:, axis], [1, 99]))
        m0, m1 = max(m0, f0 - margin), min(m1, f1 + margin)
        extents.append((m0, m1) if m1 - m0 >= 0.6 * (f1 - f0) else (f0, f1))
    z_low, z_high = (float(v) for v in np.percentile(points[:, 2], [1, 99]))
    z0 = z_low if hanging and z_low > 0.15 else 0.0
    if low_piece:
        top = top_surface_below_objects(points, footprint, z_high)
        if top is not None and top > z0 + 0.15:
            z_high = top
    if z_high - z0 < 0.03:
        return None
    rectangle = footprint.minimum_rotated_rectangle
    if footprint.area < 0.92 * rectangle.area and len(footprint.exterior.coords) > 5:
        # an L / polygon outline keeps its shape; the scan gives the heights
        return [extrusion("body", footprint, z0, z_high)]
    (x0, x1), (y0, y1) = extents
    return [box("body", x0, x1, y0, y1, z0, z_high)]


def worktop_level(points: np.ndarray, low: float = 0.60, high: float = 1.20) -> float | None:
    """Measured worktop height: the dense horizontal surface of a kitchen run between ``low`` and ``high``."""
    level = surface_level(points, low, high, share=0.6)
    return None if level is None else level.top
