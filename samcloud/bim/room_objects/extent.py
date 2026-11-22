"""Vertical extent of plan furniture: what the scan shows hanging keeps its measured bottom edge.

Scans rarely see the lowest part of a standing object, so "no points near the floor" alone proves nothing: a wardrobe,
a sofa or a counter still reaches the floor.  An object floats only with positive evidence:

* **wall**: it stands out from a wall, no furniture point inside its outline reaches down to the floor, the space
  below it in front of the wall is empty, and the wall is seen directly below it.  A standing object would hide
  exactly that piece of wall, a wall shelf does not.  A fragment of a larger plan item never lifts the whole item.
* **resting**: its bottom edge lies on the top of another piece that carries most of its outline.

Furniture on thin legs or poles (tables, chairs, beds, sofas, lamps, plants, cat trees) shows the wall below it
although it stands, so only the second rule can lift it.  Everything else is decided by the points, not by its name.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Polygon, box

from samcloud.bim.furniture_recon.families import TALL_GAP_NAMES, STANDING_FAMILIES, family_of
from samcloud.bim.room_objects.evidence import (
    APPLIANCE, CEILING, CLUTTER, ELECTRICAL, FLOOR, FURNITURE, WALL, RoomEvidence,
)
from samcloud.bim.room_objects.geometry import hull_of, line_direction
from samcloud.bim.room_objects.settings import ExtentSettings

SHELF_LABELS = ("shelf", "bookshelf", "wall shelf")
_FIXED_KINDS = frozenset({
    "kitchen_base", "worktop", "kitchen_sink", "cooktop", "tall_unit", "oven_tower", "builtin_front", "upper_cabinet",
})


def _metric_polygon(points: list[list[float]], upm: float) -> Polygon | None:
    try:
        polygon = Polygon(np.asarray(points, dtype=float) / upm)
        polygon = polygon if polygon.is_valid else polygon.buffer(0)
    except (TypeError, ValueError):
        return None
    return polygon if isinstance(polygon, Polygon) and polygon.area > 0 else None


def _has_thin_supports(item: Any) -> bool:
    if family_of(item.label, item.semantic_label, item.catalogue_kind) in STANDING_FAMILIES:
        return True
    return any(str(text or "").strip().lower() in TALL_GAP_NAMES for text in (item.semantic_label, item.label))


def _wall_seen(
    evidence: RoomEvidence, member: np.ndarray, line: LineString, bottom: float, settings: ExtentSettings,
) -> tuple[float, LineString | None]:
    """(share of the band below the object in which the wall is seen, the measured wall face there).

    The plan wall only gives the direction: the face itself is measured from the wall points at the object, because a
    face drawn a few centimetres off would hide both the contact and the wall behind the object.
    """
    start = np.asarray(line.coords[0])
    direction = line_direction(line)
    normal = np.asarray((-direction[1], direction[0]))
    along = (member[:, :2] - start) @ direction
    u0, u1 = (float(value) for value in np.percentile(along, [1, 99]))
    if u1 - u0 < 0.15 or min(u1, line.length) - max(u0, 0.0) < 0.5 * (u1 - u0):
        return 0.0, None  # only touches the wall with its narrow side, or lies past the end of it
    inset = 0.10 * (u1 - u0)
    u0, u1 = u0 + inset, u1 - inset
    z0, z1 = max(0.05, bottom - settings.seen_band_m), bottom - 0.05
    if z1 - z0 < 0.10:
        return 0.0, None
    across = (member[:, :2] - start) @ normal
    sides = np.percentile(across, [1, 99])
    low, high = float(sides[0]) - settings.wall_gap_m, float(sides[1]) + settings.wall_gap_m
    corners = np.asarray([start + a * direction + b * normal for a in (u0, u1) for b in (low, high)])
    near = evidence.in_box(
        (*corners.min(axis=0), z0), (*corners.max(axis=0), float(member[:, 2].max()) + 0.30),
        classes=(WALL, ELECTRICAL),
    )
    if len(near) < 10:
        return 0.0, None
    relative = evidence.xyz[near, :2] - start
    u, offset = relative @ direction, relative @ normal
    inside = (u >= u0) & (u < u1) & (offset >= low) & (offset <= high)
    if inside.sum() < 10:
        return 0.0, None
    counts, edges = np.histogram(offset[inside], bins=max(int(math.ceil((high - low) / 0.01)), 1), range=(low, high))
    peak = int(np.argmax(np.convolve(counts, np.ones(3), mode="same")))
    face = 0.5 * float(edges[peak] + edges[peak + 1])
    if np.min(np.abs(sides - face)) > settings.wall_gap_m:
        return 0.0, None
    front = float(sides[1]) if abs(float(sides[1]) - face) >= abs(float(sides[0]) - face) else float(sides[0])
    if abs(front - face) < settings.min_protrusion_m:
        return 0.0, None  # in the plane of what is seen below it: a front that goes on downwards
    keep = inside & (np.abs(offset - face) <= settings.wall_band_m) & (evidence.xyz[near, 2] < z1)
    measured = LineString((start + face * normal, start + line.length * direction + face * normal))
    columns = max(1, int(math.ceil((u1 - u0) / 0.10)))
    rows = max(1, int(math.ceil((z1 - z0) / 0.10)))
    col = np.minimum(((u[keep] - u0) / (u1 - u0) * columns).astype(int), columns - 1)
    row = np.minimum(((evidence.xyz[near[keep], 2] - z0) / (z1 - z0) * rows).astype(int), rows - 1)
    seen = np.zeros((columns, rows), dtype=bool)
    seen[col, row] = True
    return float(seen.mean()), measured


def _support_points(
    evidence: RoomEvidence, outline: Polygon, own_id: int, others: list[tuple[Any, Polygon]], settings: ExtentSettings,
) -> int:
    """Furniture points at floor level inside the outline that do not belong to another piece."""
    grown = outline.buffer(0.02)
    x0, y0, x1, y1 = grown.bounds
    low = evidence.in_box(
        (x0, y0, 0.03), (x1, y1, settings.support_band_m), classes=(FURNITURE, APPLIANCE),
    )
    low = low[(evidence.instance[low] == 0) | (evidence.instance[low] == own_id)]
    if not len(low):
        return 0
    x, y = evidence.xyz[low, 0], evidence.xyz[low, 1]
    own = shapely.contains_xy(grown, x, y)
    for _item, polygon in others:
        if polygon.intersects(grown) and polygon.intersection(outline).area < 0.5 * outline.area:
            own &= ~shapely.contains_xy(polygon.buffer(0.03), x, y)
    return int(own.sum())


def _space_below_used(
    evidence: RoomEvidence, member: np.ndarray, line: LineString, bottom: float, own_id: int, others: list[tuple[Any, Polygon]],
    settings: ExtentSettings,
) -> bool:
    """Whether something fills the space under the object in front of the wall (front, sides or content of a piece)."""
    start = np.asarray(line.coords[0])
    direction = line_direction(line)
    normal = np.asarray((-direction[1], direction[0]))
    relative = member[:, :2] - start
    across = relative @ normal
    if float(np.median(across)) < 0:
        normal, across = -normal, -across
    u0, u1 = np.percentile(relative @ direction, [1, 99])
    depth = float(np.percentile(across, 99))
    z0, z1 = max(0.10, bottom - settings.seen_band_m), bottom - 0.08
    low = settings.wall_band_m + 0.01
    if depth <= low or z1 - z0 < 0.10:
        return False
    corners = np.asarray([start + a * direction + b * normal for a in (u0, u1) for b in (low, depth)])
    near = evidence.in_box((*corners.min(axis=0), z0), (*corners.max(axis=0), z1))
    near = near[~np.isin(evidence.cls[near], (FLOOR, CEILING))]
    if not len(near):
        return False
    relative = evidence.xyz[near, :2] - start
    u, v = relative @ direction, relative @ normal
    inside = (u >= u0) & (u <= u1) & (v >= low) & (v <= depth)
    for other, polygon in others:  # the bed under a shelf is no part of the shelf
        ids = evidence.instance[near]
        inside &= ~((ids != 0) & (ids != own_id) & (ids == (other.instance_id if other.instance_id is not None else -1)))
        top = math.inf if other.z_max is None else evidence.z_metres(other.z_max) + 0.25
        x0, y0, x1, y1 = polygon.bounds
        candidates = np.flatnonzero(inside)
        x, y, z = evidence.xyz[near[candidates]].T
        covered = (x >= x0 - 0.05) & (x <= x1 + 0.05) & (y >= y0 - 0.05) & (y <= y1 + 0.05) & (z <= top)
        if covered.any():
            covered[covered] = shapely.contains_xy(polygon.buffer(0.05), x[covered], y[covered])
            inside[candidates[covered]] = False
    if inside.sum() < settings.support_min_points:
        return False
    rows = max(1, int(math.ceil((z1 - z0) / 0.10)))
    row = np.minimum(((evidence.xyz[near[inside], 2] - z0) / (z1 - z0) * rows).astype(int), rows - 1)
    return int(np.count_nonzero(np.bincount(row, minlength=rows) >= 3)) >= settings.below_used_share * rows


def _wall_aligned_outline(member: np.ndarray, line: LineString) -> Polygon:
    """Rectangle of the object's own points along its wall, closed back to the wall face."""
    direction = line_direction(line)
    normal = np.asarray((-direction[1], direction[0]))
    start = np.asarray(line.coords[0])
    relative = member[:, :2] - start
    offset = relative @ normal
    if float(np.median(offset)) < 0:
        normal, offset = -normal, -offset
    along = relative @ direction
    u0, u1 = np.percentile(along, [0.5, 99.5])
    front = max(float(np.percentile(offset, 99)), 0.03)
    back = max(min(float(np.percentile(offset, 1)), 0.0), 0.0)  # the wall face itself, never inside the wall
    matrix = [direction[0], normal[0], direction[1], normal[1], start[0], start[1]]
    return affinity.affine_transform(box(u0, back, u1, front), matrix)


def apply_measured_extents(
    items: list[Any], evidence: RoomEvidence, wall_lines_m: list[LineString], settings: ExtentSettings,
) -> dict[str, Any]:
    """Give the items the scan shows hanging or resting their measured bottom (``items`` are modified in place)."""
    report: dict[str, Any] = {"wall": [], "resting": [], "standing": {}}
    if not settings.enabled:
        return report
    upm = evidence.units_per_metre
    outlines = [(item, polygon) for item in items if (polygon := _metric_polygon(item.footprint, upm)) is not None]
    for item, polygon in outlines:
        kind = str(item.catalogue_kind or "").lower()
        if item.instance_id is None or item.mounting or item.layer in ("A-OBEN", "P-SANI") or kind in _FIXED_KINDS:
            continue
        own_id = int(item.instance_id)
        member = evidence.xyz[(evidence.instance == own_id) & (evidence.cls != CLUTTER)]
        if len(member) < settings.min_points:
            continue
        bottom = float(np.percentile(member[:, 2], 0.5))
        top = float(np.percentile(member[:, 2], 99.5))
        if bottom < settings.min_bottom_m:
            continue  # reaches down to the floor
        outline = hull_of(member[:, :2])
        if outline is None:
            continue
        others = [(other, shape) for other, shape in outlines if other is not item]
        if polygon.area > settings.max_item_share * max(outline.area, settings.small_item_m2):
            report["standing"][item.id] = {"bottom_m": round(bottom, 3), "reason": "fragment of a larger item"}
            continue
        support = _support_points(evidence, outline, own_id, others, settings)
        if support >= settings.support_min_points:
            report["standing"][item.id] = {"bottom_m": round(bottom, 3), "support_points": support}
            continue
        carrier = None
        for other, shape in others:
            if other.z_max is None or shape.intersection(outline).area < 0.5 * outline.area:
                continue
            if abs(evidence.z_metres(other.z_max) - bottom) <= settings.carrier_gap_m:
                carrier = other
                break
        best_share, best_line = 0.0, None
        if carrier is None:
            if _has_thin_supports(item):
                continue
            for line in wall_lines_m:
                if line.length < 0.15 or line.distance(outline) > settings.wall_gap_m + settings.plan_face_error_m:
                    continue
                share, measured = _wall_seen(evidence, member, line, bottom, settings)
                if share > best_share:
                    best_share, best_line = share, measured
            if best_line is not None and best_share >= settings.seen_share:
                if _space_below_used(evidence, member, best_line, bottom, own_id, others, settings):
                    report["standing"][item.id] = {"bottom_m": round(bottom, 3), "reason": "space below is not empty"}
                    continue
            if best_line is None or best_share < settings.seen_share:
                report["standing"][item.id] = {"bottom_m": round(bottom, 3), "wall_seen": round(best_share, 3)}
                continue  # occluded: it still stands on the floor
        mounting = "resting" if carrier is not None else "wall"
        shape = outline.minimum_rotated_rectangle if best_line is None else _wall_aligned_outline(member, best_line)
        if top - bottom < 0.02:
            top = bottom + 0.02  # a board measured on one face only
        _apply(item, evidence, shape, bottom, top, mounting, carrier, best_share, len(member))
        report[mounting].append({"id": item.id, "bottom_m": round(bottom, 3), "wall_seen": round(best_share, 3)})
    return report


def discover_wall_shelves(
    items: list[Any], evidence: RoomEvidence, wall_lines_m: list[LineString], settings: ExtentSettings,
    item_factory: Callable[..., Any],
) -> list[dict[str, Any]]:
    """Shelf instances on a wall that no plan item stands for become items (appended to ``items``).

    The top view shows the sofa under a shelf, not the shelf.  Added only with the full evidence of a hanging piece:
    a shelf instance, no support to the floor, the wall seen directly below it, and no plan item at that height.
    """
    if not settings.enabled:
        return []
    upm = evidence.units_per_metre
    added: list[dict[str, Any]] = []
    represented = {int(item.instance_id) for item in items if item.instance_id is not None}
    outlines = [(item, polygon) for item in items if (polygon := _metric_polygon(item.footprint, upm)) is not None]
    number = max([int(item.id[1:]) for item in items if str(item.id)[1:].isdigit()] + [0]) + 1
    for identifier in evidence.instance_ids(SHELF_LABELS).tolist():
        if identifier in represented:
            continue
        member = evidence.xyz[(evidence.instance == identifier) & (evidence.cls == FURNITURE)]
        if len(member) < settings.discover_min_points:
            continue
        bottom, top = (float(value) for value in np.percentile(member[:, 2], [0.5, 99.5]))
        if bottom < settings.discover_min_bottom_m:
            continue
        outline = hull_of(member[:, :2])
        if outline is None or _support_points(evidence, outline, identifier, outlines, settings) >= settings.support_min_points:
            continue
        best_share, best_line = 0.0, None
        for line in wall_lines_m:
            if line.length < 0.15 or line.distance(outline) > settings.wall_gap_m + settings.plan_face_error_m:
                continue
            share, measured = _wall_seen(evidence, member, line, bottom, settings)
            if share > best_share:
                best_share, best_line = share, measured
        if best_line is None or best_share < settings.seen_share:
            continue
        if _space_below_used(evidence, member, best_line, bottom, identifier, outlines, settings):
            continue
        shape = _wall_aligned_outline(member, best_line)
        if any(
            polygon.intersection(shape).area >= 0.25 * min(polygon.area, shape.area)
            and _spans_height(other, evidence, bottom, top)
            for other, polygon in outlines
        ):
            continue  # that item stands for it
        top = max(top, bottom + 0.02)
        ring = evidence.xy_local(shape.exterior.coords[:-1])
        low, high = evidence.z_local(bottom), evidence.z_local(top)
        item = item_factory(
            id=f"F{number:03d}", instance_id=int(identifier), label="Regal", semantic_label="shelf", catalogue_kind=None,
            layer="A-FURN", footprint=ring, z_min=low, z_max=high, z_source="points", floor_contact=False, mounting="wall",
            shape_source="parametric",
            solids=[{"type": "extrusion", "role": "body", "part_source": "measured", "outline": ring, "z_min": low, "z_max": high}],
            fit={"added": "wall shelf", "model": "mounted", "points": int(len(member)), "wall_seen": round(best_share, 3)},
        )
        number += 1
        items.append(item)
        outlines.append((item, shape))
        added.append({"id": item.id, "instance_id": int(identifier), "bottom_m": round(bottom, 3)})
    return added


def _spans_height(item: Any, evidence: RoomEvidence, bottom: float, top: float) -> bool:
    if item.z_min is not None and item.z_max is not None:
        low, high = evidence.z_metres(item.z_min), evidence.z_metres(item.z_max)
    elif item.default_z_m:
        low, high = float(item.default_z_m[0]), float(item.default_z_m[1])
    else:
        low, high = 0.0, float(item.default_height_m or 0.8)
    return min(high, top) - max(low, bottom) >= -0.05


def _apply(
    item: Any, evidence: RoomEvidence, shape: Polygon, bottom: float, top: float, mounting: str, carrier: Any, seen: float,
    points: int,
) -> None:
    started_from = [[float(x), float(y)] for x, y in item.footprint]
    ring = evidence.xy_local(shape.exterior.coords[:-1])
    low, high = evidence.z_local(bottom), evidence.z_local(top)
    item.mounting = mounting
    item.footprint = ring
    item.z_min, item.z_max, item.z_source = low, high, "points"
    item.default_z_m = None
    item.parts = []
    item.floor_contact = False
    item.shape_source = "parametric"
    item.solids = [{
        "type": "extrusion", "role": "body", "part_source": "measured", "outline": ring, "z_min": low, "z_max": high,
    }]
    item.fit = {
        "model": "mounted", "points": int(points), "wall_seen": round(float(seen), 3),
        "carried_by": None if carrier is None else carrier.id,
        "plan_footprint": started_from,  # the 2D DXF redraws exactly this polyline
    }
