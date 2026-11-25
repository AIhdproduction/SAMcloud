"""Full-control edits of the indoor plan model (tool tier ``"full"``): new walls, room outlines, bulk operations.

These operations extend :data:`samcloud.assistant.edits.OPS` for the full-control tier only (the "elements" tier keeps
exactly the phase 1 tools).  They run through the same :class:`~samcloud.assistant.edits.Proposal` (all or nothing, the
base plan is never mutated), are stored in ``edits.json`` like every other op and replay after a pipeline recompute:
:func:`samcloud.assistant.edits.op_spec` finds them here.  Dependencies are computed deterministically, never by the AI:

* ``add_wall`` snaps its ends onto the centreline of a wall it ends on (a join), refuses to duplicate an existing wall
  or to block another wall's opening, hosts the unhosted openings inside its band, cuts its band out of the rooms it
  crosses (a room split in two gets a new room with area and label point) and joins the wall body rings;
* ``set_room_polygon`` / ``split_room`` / ``merge_rooms`` recompute area and label point; outlines are clipped to the wall
  bands, rooms must not overlap and merged rooms must touch (a wall between them has to be deleted first);
* ``move_opening_to_wall`` re-hosts an opening (centre on the new wall, its thickness and direction; it must fit inside
  the wall and must not overlap the other openings there);
* ceiling objects can be moved, deleted and set to a new height (footprint, mesh and ``z_min`` / ``z_max`` follow);
* ``apply_to_kind`` runs one element operation on every element of a kind that matches a filter (e.g. all interior walls
  to 0.12 m) as ONE op; ``rename_rooms_bulk`` renames several rooms at once.

Ids that an op refers to besides its ``target`` (the second room of a merge, the new host wall, explicit bulk ids) are
stored with their signature (``_refs``) and re-resolved by signature when the edit is replayed on a recomputed plan.
Coordinates are plan metres, heights metres above the floor.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split as split_geometry
from shapely.ops import unary_union

from samcloud.assistant.edits import (
    MAX_FURNITURE_MOVE_M,
    MAX_ROOM_NAME_LENGTH,
    MIN_WALL_LENGTH_M,
    OPS,
    THICKNESS_RANGE_M,
    Change,
    EditError,
    EditOp,
    OpSpec,
    _check_in_host,
    _check_params,
    _Context,
    _in_range,
    _number,
    _op_rename_room,
    _polygon_of,
    _require,
    _room_polygon,
    _schema,
    _update_bodies,
    _wall_axis,
    find_element,
    resolve_signature,
    signature_of,
    snapshot,
    wall_band,
)
from samcloud.bim.plan_model import BuildingPlan, PlanRoom, PlanWall

# ---------------------------------------------------------------------------------------------------------- limits
MAX_NEW_WALL_LENGTH_M = 60.0
WALL_KINDS = ("interior", "exterior", "partition")
WALL_HEIGHT_RANGE_M = (0.3, 10.0)
JOIN_SNAP_REACH_M = 0.15  # a new wall end this close to another wall's band is joined onto its centreline
JOIN_MIN_ANGLE_DEG = 30.0
WALL_OVERLAP_SHARE = 0.30  # a new wall overlapping an existing one by more than this share of its band is a duplicate
OPENING_HOST_REACH_M = 0.02
MIN_ROOM_AREA_M2 = 0.5
MAX_ROOM_AREA_M2 = 5000.0
MAX_OUTLINE_POINTS = 400
ROOM_OVERLAP_M2 = 0.05
ROOM_OVERLAP_SHARE = 0.02
MERGE_GAP_M = 0.30  # merging closes gaps up to twice this (a removed wall's band)
SPLIT_LINE_EXTEND_M = 0.30
OPENING_GAP_M = 0.01
CEILING_HEIGHT_RANGE_M = (0.3, 8.0)
BULK_MAX_ELEMENTS = 400
BULK_MAX_RENAMES = 200

# element operations apply_to_kind may run per kind (door / window / passage are openings)
BULK_OPS: dict[str, tuple[str, ...]] = {
    "wall": ("move_wall", "set_wall_thickness", "delete_wall"),
    "opening": ("set_opening", "delete_opening"),
    "furniture": ("move_furniture", "rotate_furniture", "delete_furniture", "set_furniture_class"),
    "ceiling_object": ("move_ceiling_object", "delete_ceiling_object", "set_ceiling_object_height"),
}
BULK_KINDS: dict[str, str] = {"wall": "wall", "opening": "opening", "door": "opening", "window": "opening",
                              "passage": "opening", "furniture": "furniture", "ceiling_object": "ceiling_object"}
BULK_CHAIN_OPS = ("move_wall", "set_wall_thickness")  # chain defaults to False in bulk (every match is changed itself)


# ----------------------------------------------------------------------------------------------------------- helpers

def _xy(value: Any, name: str) -> tuple[float, float]:
    """A plan point [x, y] in metres (finite numbers)."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise EditError(f"Parameter {name!r} must be [x, y] in plan metres.", key="assistant.edit.bad_params", params={"param": name})
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as error:
        raise EditError(f"Parameter {name!r} must be [x, y] in plan metres.", key="assistant.edit.bad_params",
                        params={"param": name}) from error
    if not (math.isfinite(x) and math.isfinite(y)):
        raise EditError(f"Parameter {name!r} must be finite.", key="assistant.edit.bad_params", params={"param": name})
    return x, y


def _next_id(existing: list[str], prefix: str) -> str:
    numbers = [int(item[len(prefix):]) for item in existing if item.startswith(prefix) and item[len(prefix):].isdigit()]
    return f"{prefix}{max(numbers or [0]) + 1:03d}"


def _polygons(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [part for part in getattr(geometry, "geoms", []) if part.geom_type == "Polygon" and not part.is_empty]


def _usable_parts(geometry: Any, upm: float) -> list[Polygon]:
    """Polygon parts of at least :data:`MIN_ROOM_AREA_M2`, largest first."""
    minimum = MIN_ROOM_AREA_M2 * upm * upm
    return sorted((part for part in _polygons(geometry) if part.area >= minimum), key=lambda part: -part.area)


def _set_room_geometry(plan: BuildingPlan, room: PlanRoom, polygon: Polygon) -> None:
    """Room outline, area and label point from ``polygon`` (local units); the label point stays when still inside."""
    upm = plan.units_per_metre or 1.0
    polygon = polygon.simplify(1e-6 * upm, preserve_topology=True) if polygon.is_valid else polygon
    room.polygon = {
        "exterior": [[float(x), float(y)] for x, y in list(polygon.exterior.coords)[:-1]],
        "holes": [[[float(x), float(y)] for x, y in list(ring.coords)[:-1]] for ring in polygon.interiors],
    }
    room.area_m2 = float(polygon.area) / (upm * upm)
    try:
        label = Point(float(room.label_point[0]), float(room.label_point[1]))
    except (TypeError, ValueError, IndexError):
        label = None
    if label is None or not polygon.contains(label):
        point = polygon.representative_point()
        room.label_point = [float(point.x), float(point.y)]


def _unique_name(plan: BuildingPlan, name: str) -> str:
    names = {room.name for room in plan.rooms}
    base = " ".join(str(name or "").split())[: MAX_ROOM_NAME_LENGTH - 4] or "Raum"
    index = 2
    candidate = f"{base} {index}"
    while candidate in names:
        index += 1
        candidate = f"{base} {index}"
    return candidate


def _room_name(value: Any) -> str | None:
    if value is None:
        return None
    name = " ".join(str(value).split())
    if not name or len(name) > MAX_ROOM_NAME_LENGTH:
        raise EditError(f"A room name needs 1-{MAX_ROOM_NAME_LENGTH} characters.", key="assistant.edit.invalid_value",
                        params={"name": "name", "value": len(name), "min": 1, "max": MAX_ROOM_NAME_LENGTH})
    return name


def _new_room(plan: BuildingPlan, template: PlanRoom, polygon: Polygon, name: str) -> PlanRoom:
    """A new room with the heights / ceiling of ``template`` on ``polygon``."""
    data = template.to_dict()
    data.update({"id": _next_id([room.id for room in plan.rooms], "R"), "name": name})
    room = PlanRoom.from_dict(copy.deepcopy(data))
    room.label_point = [float("nan"), float("nan")]
    _set_room_geometry(plan, room, polygon)
    return room


def _wall_union(plan: BuildingPlan, exclude: set[str] | None = None) -> Any:
    upm = plan.units_per_metre or 1.0
    bands = []
    for wall in plan.walls:
        if exclude and wall.id in exclude or len(wall.centreline) < 2:
            continue
        try:
            bands.append(wall_band(wall, upm))
        except EditError:
            continue
    return unary_union(bands) if bands else Polygon()


def _check_room_overlaps(plan: BuildingPlan, room_id: str, polygon: Polygon, ignore: set[str] | None = None) -> None:
    upm = plan.units_per_metre or 1.0
    for other in plan.rooms:
        if other.id == room_id or (ignore and other.id in ignore):
            continue
        other_polygon = _room_polygon(other)
        if other_polygon is None:
            continue
        overlap = polygon.intersection(other_polygon).area
        limit = max(ROOM_OVERLAP_M2 * upm * upm, ROOM_OVERLAP_SHARE * min(polygon.area, other_polygon.area))
        if overlap > limit:
            raise EditError(f"Room {room_id} would overlap room {other.id} by {overlap / (upm * upm):.2f} m2.",
                            key="assistant.edit.overlap", params={"id": room_id, "other": other.id})


def _ref(ctx: _Context, op: EditOp, element_id: Any, kind: str) -> tuple[str, Any]:
    """(current id, element) of an id an op refers to besides its target.

    The first application stores the element's signature in ``_refs``; a replay (stored edit on a recomputed plan)
    re-resolves the id by that signature first.
    """
    element_id = str(element_id or "").strip()
    if not element_id:
        raise EditError("An element id is missing.", key="assistant.edit.bad_params", params={"param": "id"})
    refs = op.params.get("_refs")
    stored = refs.get(element_id) if isinstance(refs, dict) else None
    current = element_id
    if isinstance(stored, dict):
        resolved = resolve_signature(ctx.plan, stored, preferred_id=element_id, strict=True)
        if resolved is None:
            raise EditError(f"Element {element_id} of this edit is no longer in the plan.", key="assistant.history.element_gone",
                            params={"id": element_id})
        current = resolved
    element = _require(ctx.plan, current, kind)
    signature = signature_of(ctx.plan, current)
    if signature is not None:
        ctx.resolved.setdefault("_refs", {})[current] = signature
    return current, element


# --------------------------------------------------------------------------------------------------------- add_wall

def _join_end(plan: BuildingPlan, end: np.ndarray, u: np.ndarray, reach: float) -> tuple[np.ndarray, str | None]:
    """``end`` moved along ``u`` onto the centreline of the wall it ends on (nearest, not parallel), else unchanged."""
    upm = plan.units_per_metre or 1.0
    sin_join = math.sin(math.radians(JOIN_MIN_ANGLE_DEG))
    best: tuple[float, np.ndarray, str] | None = None
    for wall in plan.walls:
        if len(wall.centreline) < 2:
            continue
        try:
            q0, _q1, d2, length = _wall_axis(wall)
        except EditError:
            continue
        cross = float(u[0] * d2[1] - u[1] * d2[0])
        if abs(cross) < sin_join:
            continue
        normal = np.asarray((-d2[1], d2[0]))
        half = float(wall.thickness_m) * upm / 2.0
        offset = float((end - q0) @ normal)
        along = float((end - q0) @ d2)
        if abs(offset) > half + reach or not (-half <= along <= length + half):
            continue
        moved = end - u * (offset / float(u @ normal))
        station = float((moved - q0) @ d2)
        if not (-half <= station <= length + half):
            continue
        if best is None or abs(offset) < best[0]:
            best = (abs(offset), moved, wall.id)
    if best is None:
        return end, None
    return best[1], best[2]


def _op_add_wall(ctx: _Context, op: EditOp) -> list[Change]:
    plan, upm = ctx.plan, ctx.upm
    start = np.asarray(_xy(op.params.get("start"), "start"), dtype=float) * upm
    end = np.asarray(_xy(op.params.get("end"), "end"), dtype=float) * upm
    thickness = float(_number(op.params, "thickness_m", required=True))
    _in_range("thickness", thickness, *THICKNESS_RANGE_M)
    wall_kind = str(op.params.get("wall_kind") or "interior").strip().lower()
    if wall_kind not in WALL_KINDS:
        raise EditError(f"Unknown wall kind {wall_kind!r} ({', '.join(WALL_KINDS)}).", key="assistant.edit.bad_params",
                        params={"param": "wall_kind"})
    height = _number(op.params, "height_m")
    if height is not None:
        _in_range("height", height, *WALL_HEIGHT_RANGE_M)
    length = float(np.linalg.norm(end - start))
    _in_range("length", length / upm, MIN_WALL_LENGTH_M, MAX_NEW_WALL_LENGTH_M)
    u = (end - start) / length
    joins: list[str] = []
    if op.params.get("join", True):
        reach = JOIN_SNAP_REACH_M * upm
        start, first = _join_end(plan, start, u, reach)
        end, second = _join_end(plan, end, u, reach)
        joins = [wall_id for wall_id in (first, second) if wall_id]
        length = float(np.linalg.norm(end - start))
        if length < MIN_WALL_LENGTH_M * upm or float((end - start) @ u) <= 0.0:
            raise EditError(f"The new wall would be shorter than {MIN_WALL_LENGTH_M} m.", key="assistant.edit.wall_too_short",
                            params={"id": "new"})
    wall_id = _next_id([wall.id for wall in plan.walls], "W")
    right = np.asarray((u[1], -u[0]))
    half = thickness * upm / 2.0
    line = np.asarray([start, end], dtype=float)
    with_faces = any(wall.face_a or wall.face_b for wall in plan.walls) or not plan.walls
    wall = PlanWall(
        id=wall_id, centreline=[[float(x), float(y)] for x, y in line], thickness_m=thickness, kind=wall_kind,
        evidence="assistant",
        face_a=[[float(x), float(y)] for x, y in line + right * half] if with_faces else None,
        face_b=[[float(x), float(y)] for x, y in line - right * half] if with_faces else None,
        body_id=None, height_m=None if height is None else float(height),
        height_source="storey" if height is None else "assistant",
    )
    band = wall_band(wall, upm)
    # no duplicate of an existing wall (the walls it joins only touch it at its ends)
    for other in plan.walls:
        if other.id in joins or len(other.centreline) < 2:
            continue
        try:
            other_band = wall_band(other, upm)
        except EditError:
            continue
        overlap = band.intersection(other_band).area
        if overlap > WALL_OVERLAP_SHARE * band.area:
            raise EditError(f"The new wall would lie on wall {other.id}.", key="assistant.edit.overlap",
                            params={"id": wall_id, "other": other.id})
    # it must not block an opening of another wall
    for opening in plan.openings:
        if not opening.host_wall_id:
            continue
        axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
        normal = np.asarray((-axis[1], axis[0]))
        centre = np.asarray(opening.center[:2], dtype=float)
        half_w = float(opening.width_m) * upm / 2.0
        half_t = max(float(opening.wall_thickness_m), 0.05) * upm / 2.0
        rectangle = Polygon([tuple(centre + axis * a + normal * b) for a, b in ((-half_w, -half_t), (half_w, -half_t),
                                                                                (half_w, half_t), (-half_w, half_t))])
        if band.intersection(rectangle).area > 1e-4 * upm * upm:
            raise EditError(f"The new wall would block opening {opening.id}.", key="assistant.edit.blocks_opening",
                            params={"id": wall_id, "opening": opening.id})
    plan.walls.append(wall)
    note = f"joined to {', '.join(joins)}" if joins else ""
    changes = [Change(wall.id, "wall", "added", None, snapshot(plan, "wall", wall), note)]
    if joins:
        ctx.notes.append(f"new wall {wall.id}: ends joined to {', '.join(joins)}")
    # unhosted openings inside the new band get it as host
    for opening in plan.openings:
        if opening.host_wall_id:
            continue
        if band.distance(Point(float(opening.center[0]), float(opening.center[1]))) > OPENING_HOST_REACH_M * upm:
            continue
        before = snapshot(plan, "opening", opening)
        opening.host_wall_id = wall.id
        opening.wall_thickness_m = thickness
        changes.append(Change(opening.id, "opening", "modified", before, snapshot(plan, "opening", opening),
                              f"hosted by new wall {wall.id}"))
    # rooms: the band is cut out; a room split in two gets a new room for each further part
    for room in list(plan.rooms):
        polygon = _room_polygon(room)
        if polygon is None or polygon.intersection(band).area <= 1e-6 * upm * upm:
            continue
        parts = _usable_parts(polygon.difference(band), upm)
        if not parts:
            raise EditError(f"The new wall would fill room {room.id}.", key="assistant.edit.geometry", params={"id": room.id})
        before = snapshot(plan, "room", room)
        label = Point(float(room.label_point[0]), float(room.label_point[1]))
        keep = next((part for part in parts if part.contains(label)), parts[0])
        rest = [part for part in parts if part is not keep]
        _set_room_geometry(plan, room, keep)
        changes.append(Change(room.id, "room", "modified", before, snapshot(plan, "room", room),
                              f"split by wall {wall.id}" if rest else f"edge follows wall {wall.id}"))
        for part in rest:
            new_room = _new_room(plan, room, part, _unique_name(plan, room.name))
            plan.rooms.append(new_room)
            changes.append(Change(new_room.id, "room", "added", None, snapshot(plan, "room", new_room), f"split from {room.id} by wall {wall.id}"))
            ctx.notes.append(f"room {room.id} split by wall {wall.id}: new room {new_room.id} ({new_room.name})")
    _update_bodies(ctx, Polygon(), band, wall.id)
    return changes


# ------------------------------------------------------------------------------------------------------------ rooms

def _op_set_room_polygon(ctx: _Context, op: EditOp) -> list[Change]:
    plan, upm = ctx.plan, ctx.upm
    room = _require(plan, op.target, "room")
    outline = op.params.get("outline")
    if not isinstance(outline, (list, tuple)) or not 3 <= len(outline) <= MAX_OUTLINE_POINTS:
        raise EditError(f"outline needs 3-{MAX_OUTLINE_POINTS} points [x, y] in plan metres.", key="assistant.edit.bad_params",
                        params={"param": "outline"})
    points = [_xy(point, "outline") for point in outline]
    polygon = _polygon_of([[x * upm, y * upm] for x, y in points])
    if polygon is None:
        raise EditError("The outline is not a usable polygon.", key="assistant.edit.geometry", params={"id": room.id})
    walls = _wall_union(plan)
    clipped = polygon.difference(walls) if not walls.is_empty else polygon
    parts = _usable_parts(clipped, upm)
    if not parts:
        raise EditError("The outline lies inside the walls or is too small.", key="assistant.edit.geometry", params={"id": room.id})
    result = parts[0]
    _in_range("area", result.area / (upm * upm), MIN_ROOM_AREA_M2, MAX_ROOM_AREA_M2)
    _check_room_overlaps(plan, room.id, result)
    if polygon.area - result.area > 0.01 * polygon.area:
        ctx.notes.append(f"room {room.id}: outline clipped to the wall faces ({(polygon.area - result.area) / (upm * upm):.2f} m2)")
    before = snapshot(plan, "room", room)
    _set_room_geometry(plan, room, result)
    return [Change(room.id, "room", "modified", before, snapshot(plan, "room", room), "new outline")]


def _op_split_room(ctx: _Context, op: EditOp) -> list[Change]:
    plan, upm = ctx.plan, ctx.upm
    room = _require(plan, op.target, "room")
    line = op.params.get("line")
    if not isinstance(line, (list, tuple)) or len(line) != 2:
        raise EditError("line needs two points [[x, y], [x, y]] in plan metres.", key="assistant.edit.bad_params", params={"param": "line"})
    a = np.asarray(_xy(line[0], "line"), dtype=float) * upm
    b = np.asarray(_xy(line[1], "line"), dtype=float) * upm
    length = float(np.linalg.norm(b - a))
    if length < MIN_WALL_LENGTH_M * upm:
        raise EditError("The split line is too short.", key="assistant.edit.bad_params", params={"param": "line"})
    direction = (b - a) / length
    extend = SPLIT_LINE_EXTEND_M * upm
    cutter = LineString([tuple(a - direction * extend), tuple(b + direction * extend)])
    polygon = _room_polygon(room)
    if polygon is None:
        raise EditError(f"Room {room.id} has no usable outline.", key="assistant.edit.geometry", params={"id": room.id})
    parts = _usable_parts(split_geometry(polygon, cutter), upm)
    if len(parts) < 2:
        raise EditError(f"The line does not split room {room.id} into two parts of at least {MIN_ROOM_AREA_M2} m2.",
                        key="assistant.edit.split_failed", params={"id": room.id})
    name = _room_name(op.params.get("name"))
    before = snapshot(plan, "room", room)
    label = Point(float(room.label_point[0]), float(room.label_point[1]))
    keep = next((part for part in parts if part.contains(label)), parts[0])
    _set_room_geometry(plan, room, keep)
    changes = [Change(room.id, "room", "modified", before, snapshot(plan, "room", room), "split")]
    for index, part in enumerate(part for part in parts if part is not keep):
        new_name = name if index == 0 and name else _unique_name(plan, room.name)
        new_room = _new_room(plan, room, part, new_name)
        plan.rooms.append(new_room)
        changes.append(Change(new_room.id, "room", "added", None, snapshot(plan, "room", new_room), f"split from {room.id}"))
    return changes


def _op_merge_rooms(ctx: _Context, op: EditOp) -> list[Change]:
    plan, upm = ctx.plan, ctx.upm
    room = _require(plan, op.target, "room")
    others_raw = op.params.get("rooms")
    if isinstance(others_raw, str):
        others_raw = [others_raw]
    if not isinstance(others_raw, (list, tuple)) or not others_raw:
        raise EditError("rooms needs the ids of the rooms merged into the target.", key="assistant.edit.bad_params", params={"param": "rooms"})
    others: list[PlanRoom] = []
    ids: list[str] = []
    for raw in others_raw:
        current, other = _ref(ctx, op, raw, "room")
        if current == room.id or current in ids:
            raise EditError(f"Room {current} is listed twice.", key="assistant.edit.bad_params", params={"param": "rooms"})
        ids.append(current)
        others.append(other)
    ctx.resolved["rooms"] = list(ids)
    polygons = [_room_polygon(item) for item in [room, *others]]
    if any(polygon is None for polygon in polygons):
        raise EditError("A room has no usable outline.", key="assistant.edit.geometry", params={"id": room.id})
    gap = MERGE_GAP_M * upm
    merged = unary_union([polygon.buffer(gap, join_style=2) for polygon in polygons]).buffer(-gap, join_style=2)
    merged = unary_union([merged, *polygons])
    walls = _wall_union(plan)
    if not walls.is_empty:
        merged = merged.difference(walls)
    parts = _usable_parts(merged, upm)
    if len(parts) != 1:
        raise EditError(f"Rooms {room.id} and {', '.join(ids)} do not touch (a wall between them must be deleted first).",
                        key="assistant.edit.not_adjacent", params={"id": room.id, "other": ", ".join(ids)})
    name = _room_name(op.params.get("name"))
    _check_room_overlaps(plan, room.id, parts[0], ignore=set(ids))
    before = snapshot(plan, "room", room)
    changes: list[Change] = []
    for other in others:
        changes.append(Change(other.id, "room", "removed", snapshot(plan, "room", other), None, f"merged into {room.id}"))
        plan.rooms.remove(other)
    _set_room_geometry(plan, room, parts[0])
    if name:
        room.name = name
    changes.insert(0, Change(room.id, "room", "modified", before, snapshot(plan, "room", room), f"merged with {', '.join(ids)}"))
    return changes


def _op_rename_rooms_bulk(ctx: _Context, op: EditOp) -> list[Change]:
    entries = op.params.get("names")
    if isinstance(entries, dict):
        entries = [{"id": key, "name": value} for key, value in entries.items()]
    if not isinstance(entries, (list, tuple)) or not entries or len(entries) > BULK_MAX_RENAMES:
        raise EditError(f"names needs 1-{BULK_MAX_RENAMES} entries {{id, name}}.", key="assistant.edit.bad_params", params={"param": "names"})
    changes: list[Change] = []
    resolved: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise EditError("Each names entry needs id and name.", key="assistant.edit.bad_params", params={"param": "names"})
        current, _room = _ref(ctx, op, entry.get("id"), "room")
        if current in seen:
            raise EditError(f"Room {current} is listed twice.", key="assistant.edit.bad_params", params={"param": "names"})
        seen.add(current)
        name = _room_name(entry.get("name"))
        changes.extend(_op_rename_room(ctx, EditOp("rename_room", current, {"name": name})))
        resolved.append({"id": current, "name": str(name)})
    ctx.resolved["names"] = resolved
    return changes


# --------------------------------------------------------------------------------------------------------- openings

def _op_move_opening_to_wall(ctx: _Context, op: EditOp) -> list[Change]:
    plan, upm = ctx.plan, ctx.upm
    opening = _require(plan, op.target, "opening")
    wall_id, wall = _ref(ctx, op, op.params.get("wall"), "wall")
    ctx.resolved["wall"] = wall_id
    p0, _p1, u, length = _wall_axis(wall)
    position = _number(op.params, "position_m")
    if position is not None and op.params.get("center") is not None:
        raise EditError("Give either position_m or center, not both.", key="assistant.edit.bad_params", params={"param": "position_m"})
    if position is not None:
        _in_range("position", position, 0.0, length / upm)
        station = position * upm
    else:
        point = np.asarray(_xy(op.params["center"], "center"), dtype=float) * upm if op.params.get("center") is not None \
            else np.asarray(opening.center[:2], dtype=float)
        station = float((point - p0) @ u)
    half = float(opening.width_m) * upm / 2.0
    if station - half < -1e-6 or station + half > length + 1e-6:
        station = min(max(station, half), length - half)
        if half * 2.0 > length + 1e-6:
            raise EditError(f"Opening {opening.id} ({opening.width_m:.2f} m) does not fit wall {wall.id}.",
                            key="assistant.edit.opening_outside_wall",
                            params={"id": opening.id, "wall": wall.id, "overhang_m": round((half * 2.0 - length) / upm, 2)})
        ctx.notes.append(f"opening {opening.id}: kept inside wall {wall.id}")
    for other in plan.openings:
        if other.id == opening.id or other.host_wall_id != wall.id:
            continue
        other_station = float((np.asarray(other.center[:2], dtype=float) - p0) @ u)
        other_half = float(other.width_m) * upm / 2.0
        if abs(other_station - station) < half + other_half + OPENING_GAP_M * upm - 1e-9:
            raise EditError(f"Opening {opening.id} would overlap opening {other.id} on wall {wall.id}.",
                            key="assistant.edit.overlap", params={"id": opening.id, "other": other.id})
    before = snapshot(plan, "opening", opening)
    old_host = opening.host_wall_id
    angle = math.atan2(float(u[1]), float(u[0]))
    if math.cos(angle - float(opening.axis_angle)) < 0.0:
        angle += math.pi
    centre = p0 + u * station
    opening.center = [float(centre[0]), float(centre[1])]
    opening.axis_angle = float(angle % (2.0 * math.pi))
    opening.wall_thickness_m = float(wall.thickness_m)
    opening.host_wall_id = wall.id
    _check_in_host(plan, opening, 0.0)
    return [Change(opening.id, "opening", "moved", before, snapshot(plan, "opening", opening),
                   f"moved from wall {old_host} to wall {wall.id}")]


# --------------------------------------------------------------------------------------------------- ceiling objects

def _ceiling_snapshot(plan: BuildingPlan, item: dict[str, Any]) -> dict[str, Any]:
    entry = snapshot(plan, "ceiling_object", item)
    bottom, top = _ceiling_z(item)
    upm = plan.units_per_metre or 1.0
    if bottom is not None and top is not None:
        entry["bottom_m"] = round((bottom - float(plan.floor_z)) / upm, 3)
        entry["top_m"] = round((top - float(plan.floor_z)) / upm, 3)
    return entry


def _ceiling_z(item: dict[str, Any]) -> tuple[float | None, float | None]:
    vertices = (item.get("mesh") or {}).get("vertices") if isinstance(item.get("mesh"), dict) else None
    if vertices:
        try:
            array = np.asarray(vertices, dtype=float)
            return float(array[:, 2].min()), float(array[:, 2].max())
        except (TypeError, ValueError, IndexError):
            pass
    try:
        return float(item["z_min"]), float(item["z_max"])
    except (KeyError, TypeError, ValueError):
        return None, None


def _op_move_ceiling_object(ctx: _Context, op: EditOp) -> list[Change]:
    plan, upm = ctx.plan, ctx.upm
    item = _require(plan, op.target, "ceiling_object")
    dx, dy = _number(op.params, "dx_m") or 0.0, _number(op.params, "dy_m") or 0.0
    _in_range("move", math.hypot(dx, dy), 0.001, MAX_FURNITURE_MOVE_M)
    before = _ceiling_snapshot(plan, item)
    offset = np.asarray((dx, dy), dtype=float) * upm
    if item.get("footprint"):
        item["footprint"] = [[float(p[0]) + offset[0], float(p[1]) + offset[1]] for p in item["footprint"]]
    mesh = item.get("mesh")
    if isinstance(mesh, dict) and mesh.get("vertices"):
        mesh["vertices"] = [[float(v[0]) + offset[0], float(v[1]) + offset[1], *[float(c) for c in v[2:]]] for v in mesh["vertices"]]
    if isinstance(item.get("center"), (list, tuple)) and len(item["center"]) >= 2:
        item["center"] = [float(item["center"][0]) + offset[0], float(item["center"][1]) + offset[1], *item["center"][2:]]
    return [Change(str(item.get("id")), "ceiling_object", "moved", before, _ceiling_snapshot(plan, item))]


def _op_delete_ceiling_object(ctx: _Context, op: EditOp) -> list[Change]:
    item = _require(ctx.plan, op.target, "ceiling_object")
    change = Change(str(item.get("id")), "ceiling_object", "removed", _ceiling_snapshot(ctx.plan, item), None)
    ctx.plan.ceiling_objects.remove(item)
    return [change]


def _op_set_ceiling_object_height(ctx: _Context, op: EditOp) -> list[Change]:
    plan, upm = ctx.plan, ctx.upm
    item = _require(plan, op.target, "ceiling_object")
    top_m, bottom_m = _number(op.params, "top_m"), _number(op.params, "bottom_m")
    if top_m is None and bottom_m is None:
        raise EditError("set_ceiling_object_height needs top_m or bottom_m.", key="assistant.edit.bad_params", params={"param": "top_m"})
    for name, value in (("top", top_m), ("bottom", bottom_m)):
        if value is not None:
            _in_range(name, value, *CEILING_HEIGHT_RANGE_M)
    old_bottom, old_top = _ceiling_z(item)
    if old_bottom is None or old_top is None:
        raise EditError(f"Ceiling object {item.get('id')} has no height.", key="assistant.edit.geometry", params={"id": item.get("id")})
    floor = float(plan.floor_z)
    new_top = floor + top_m * upm if top_m is not None else None
    new_bottom = floor + bottom_m * upm if bottom_m is not None else None
    if new_top is not None and new_bottom is not None:
        if new_top - new_bottom < 0.01 * upm:
            raise EditError("top_m must be above bottom_m.", key="assistant.edit.invalid_value",
                            params={"name": "top", "value": top_m, "min": bottom_m, "max": CEILING_HEIGHT_RANGE_M[1]})
        scale = (new_top - new_bottom) / max(old_top - old_bottom, 1e-9)

        def mapped(z: float) -> float:
            return new_bottom + (z - old_bottom) * scale
    else:
        shift = (new_top - old_top) if new_top is not None else (new_bottom - old_bottom)  # type: ignore[operator]

        def mapped(z: float) -> float:
            return z + shift
    before = _ceiling_snapshot(plan, item)
    mesh = item.get("mesh")
    if isinstance(mesh, dict) and mesh.get("vertices"):
        mesh["vertices"] = [[float(v[0]), float(v[1]), mapped(float(v[2])), *[float(c) for c in v[3:]]] for v in mesh["vertices"]]
    item["z_min"], item["z_max"] = mapped(old_bottom), mapped(old_top)
    item["z_source"] = "assistant"
    return [Change(str(item.get("id")), "ceiling_object", "modified", before, _ceiling_snapshot(plan, item), "height")]


# ------------------------------------------------------------------------------------------------------------ bulk

def _bulk_matches(ctx: _Context, op: EditOp, kind: str, spec: dict[str, Any]) -> list[str]:
    from samcloud.assistant.selection import element_list

    plan = ctx.plan
    ids_raw = spec.get("ids")
    if ids_raw is not None:
        if isinstance(ids_raw, str):
            ids_raw = [ids_raw]
        if not isinstance(ids_raw, (list, tuple)):
            raise EditError("filter.ids must be a list of ids.", key="assistant.edit.bad_params", params={"param": "filter"})
        wanted: list[str] = []
        for raw in ids_raw:
            current, _element = _ref(ctx, op, raw, BULK_KINDS[kind])
            if current not in wanted:
                wanted.append(current)
        ctx.resolved["filter"] = {**spec, "ids": list(wanted)}
    else:
        wanted = []
    query = {"kind": kind}
    if spec.get("room"):
        query["room"] = str(spec["room"])
    if spec.get("label"):
        query["label"] = str(spec["label"])
    rows = element_list(plan, query)
    wall_kinds = spec.get("wall_kind")
    if isinstance(wall_kinds, str):
        wall_kinds = [wall_kinds]
    result: list[str] = []
    for row in rows:
        element_id = str(row["id"])
        if ids_raw is not None and element_id not in wanted:
            continue
        if wall_kinds:
            found = find_element(plan, element_id)
            if found is None or found[0] != "wall" or str(found[1].kind) not in [str(k) for k in wall_kinds]:
                continue
        for field, low_key, high_key in (("thickness_m", "min_thickness_m", "max_thickness_m"),
                                         ("width_m", "min_width_m", "max_width_m")):
            value = row.get(field)
            low, high = spec.get(low_key), spec.get(high_key)
            if (low is not None or high is not None) and value is None:
                break
            if low is not None and float(value) < float(low) - 1e-9:
                break
            if high is not None and float(value) > float(high) + 1e-9:
                break
        else:
            result.append(element_id)
    return result


def _op_apply_to_kind(ctx: _Context, op: EditOp) -> list[Change]:
    kind = str(op.params.get("kind") or "").strip().lower()
    if kind not in BULK_KINDS:
        raise EditError(f"Unknown kind {kind!r} ({', '.join(BULK_KINDS)}).", key="assistant.edit.bad_params", params={"param": "kind"})
    name = str(op.params.get("op") or "").strip()
    allowed = BULK_OPS[BULK_KINDS[kind]]
    if name not in allowed:
        raise EditError(f"{name!r} cannot be applied to {kind} (allowed: {', '.join(allowed)}).", key="assistant.edit.bad_params",
                        params={"param": "op"})
    spec = OPS.get(name) or FULL_OPS.get(name)
    params = op.params.get("params") or {}
    filter_spec = op.params.get("filter") or {}
    if not isinstance(params, dict) or not isinstance(filter_spec, dict):
        raise EditError("params and filter must be objects.", key="assistant.edit.bad_params", params={"param": "params"})
    params = {key: copy.deepcopy(value) for key, value in params.items() if key != "target" and not str(key).startswith("_")}
    if name in BULK_CHAIN_OPS:
        params.setdefault("chain", False)
    targets = _bulk_matches(ctx, op, kind, filter_spec)
    if not targets:
        raise EditError(f"No {kind} matches the filter.", key="assistant.edit.no_match", params={"kind": kind})
    if len(targets) > BULK_MAX_ELEMENTS:
        raise EditError(f"{len(targets)} elements match; at most {BULK_MAX_ELEMENTS} at once.", key="assistant.edit.invalid_value",
                        params={"name": "count", "value": len(targets), "min": 1, "max": BULK_MAX_ELEMENTS})
    changes: list[Change] = []
    skipped: list[str] = []
    for element_id in targets:
        if find_element(ctx.plan, element_id) is None:
            skipped.append(element_id)  # removed by an earlier element of this op (e.g. an opening of a deleted wall)
            continue
        sub = EditOp(name, element_id, copy.deepcopy(params))
        _check_params(spec, sub)
        try:
            changes.extend(spec.handler(ctx, sub))
        except EditError as error:
            raise EditError(f"{element_id}: {error.message}", key=error.key, params={**error.params, "id": element_id}) from error
    if skipped:
        ctx.notes.append(f"{name}: {', '.join(skipped)} already removed by this edit")
    ctx.notes.append(f"{name} applied to {len(targets) - len(skipped)} {kind} element(s)")
    return changes


# -------------------------------------------------------------------------------------------------------------- OPS

_POINT = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "Plan metres [x, y]."}
_TARGET = {"type": "string", "description": "Plan id of the element (e.g. W003, O002, R001, C001)."}
_FILTER = {
    "type": "object",
    "description": "Which elements: all of the kind when empty. room (room id), label (text in label / id), wall_kind "
                   "(interior / exterior / partition, or a list), ids (explicit list), min_/max_thickness_m, min_/max_width_m.",
    "properties": {
        "room": {"type": "string"}, "label": {"type": "string"},
        "wall_kind": {"anyOf": [{"type": "string", "enum": list(WALL_KINDS)},
                                {"type": "array", "items": {"type": "string", "enum": list(WALL_KINDS)}}]},
        "ids": {"type": "array", "items": {"type": "string"}},
        "min_thickness_m": {"type": "number"}, "max_thickness_m": {"type": "number"},
        "min_width_m": {"type": "number"}, "max_width_m": {"type": "number"},
    },
    "additionalProperties": False,
}

FULL_OPS: dict[str, OpSpec] = {
    "add_wall": OpSpec(_op_add_wall, _schema({
        "start": _POINT, "end": _POINT,
        "thickness_m": {"type": "number", "minimum": THICKNESS_RANGE_M[0], "maximum": THICKNESS_RANGE_M[1]},
        "wall_kind": {"type": "string", "enum": list(WALL_KINDS), "description": "Default interior."},
        "height_m": {"type": "number", "description": "Only for a wall lower than the storey (metres)."},
        "join": {"type": "boolean", "description": "Snap the ends onto the walls they end on (default true)."},
    }, ["start", "end", "thickness_m"]), "Add a new straight wall (centreline from start to end).", None),
    "set_room_polygon": OpSpec(_op_set_room_polygon, _schema({
        "target": _TARGET,
        "outline": {"type": "array", "items": _POINT, "minItems": 3, "maxItems": MAX_OUTLINE_POINTS,
                    "description": "Closed outline in plan metres (do not repeat the first point)."},
    }, ["target", "outline"]), "Replace a room's outline.", "room"),
    "split_room": OpSpec(_op_split_room, _schema({
        "target": _TARGET,
        "line": {"type": "array", "items": _POINT, "minItems": 2, "maxItems": 2, "description": "Split line across the room."},
        "name": {"type": "string", "maxLength": MAX_ROOM_NAME_LENGTH, "description": "Name of the new room (optional)."},
    }, ["target", "line"]), "Split a room in two along a line.", "room"),
    "merge_rooms": OpSpec(_op_merge_rooms, _schema({
        "target": _TARGET,
        "rooms": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Rooms merged into the target."},
        "name": {"type": "string", "maxLength": MAX_ROOM_NAME_LENGTH, "description": "Name of the merged room (optional)."},
    }, ["target", "rooms"]), "Merge touching rooms into the target room.", "room"),
    "move_opening_to_wall": OpSpec(_op_move_opening_to_wall, _schema({
        "target": _TARGET,
        "wall": {"type": "string", "description": "Id of the new host wall."},
        "position_m": {"type": "number", "description": "Centre along the new wall, metres from its start (optional)."},
        "center": dict(_POINT, description="Point near the new centre in plan metres (optional)."),
    }, ["target", "wall"]), "Move a door / window / passage into another wall.", "opening"),
    "move_ceiling_object": OpSpec(_op_move_ceiling_object, _schema({
        "target": _TARGET, "dx_m": {"type": "number"}, "dy_m": {"type": "number"},
    }, ["target"]), "Move a ceiling object (lamp, detector, ...).", "ceiling_object"),
    "delete_ceiling_object": OpSpec(_op_delete_ceiling_object, _schema({"target": _TARGET}, ["target"]),
                                    "Delete a ceiling object.", "ceiling_object"),
    "set_ceiling_object_height": OpSpec(_op_set_ceiling_object_height, _schema({
        "target": _TARGET,
        "top_m": {"type": "number", "description": "Top above the floor (metres)."},
        "bottom_m": {"type": "number", "description": "Bottom above the floor (metres)."},
    }, ["target"]), "Set the height of a ceiling object (only top or bottom: shifted; both: stretched).", "ceiling_object"),
    "apply_to_kind": OpSpec(_op_apply_to_kind, _schema({
        "kind": {"type": "string", "enum": list(BULK_KINDS)},
        "op": {"type": "string", "enum": sorted({name for names in BULK_OPS.values() for name in names})},
        "params": {"type": "object", "description": "Parameters of op without target (e.g. {\"thickness_m\": 0.12})."},
        "filter": _FILTER,
    }, ["kind", "op"]), "Apply one element operation to every matching element of a kind at once.", None),
    "rename_rooms_bulk": OpSpec(_op_rename_rooms_bulk, _schema({
        "names": {"type": "array", "minItems": 1, "maxItems": BULK_MAX_RENAMES,
                  "items": {"type": "object", "properties": {"id": {"type": "string"},
                                                             "name": {"type": "string", "maxLength": MAX_ROOM_NAME_LENGTH}},
                            "required": ["id", "name"], "additionalProperties": False}},
    }, ["names"]), "Rename several rooms at once.", None),
}

FULL_OP_NAMES: tuple[str, ...] = tuple(FULL_OPS)

_FULL_NOTES: dict[str, str] = {
    "add_wall": "Plan metres. Ends near another wall are joined onto its centreline. Rooms crossed by the wall are cut "
                "(a room split in two gets a new room), unhosted openings in the band are hosted. Refused when it would lie "
                "on an existing wall or block an opening. Measure first (fit_surface kind wall) when the user did not give "
                "the position.",
    "set_room_polygon": "The outline is clipped to the wall faces; area and label point are recomputed; rooms must not overlap.",
    "split_room": "The new room keeps the heights of the old one; area and label points are recomputed.",
    "merge_rooms": "Rooms must touch; delete the wall between them first (delete_wall) in the same turn. The other rooms are removed.",
    "move_opening_to_wall": "The opening takes the new wall's thickness and direction and must fit without overlapping other openings.",
    "move_ceiling_object": "dx_m / dy_m in plan metres; footprint and 3D shape move together.",
    "set_ceiling_object_height": "Metres above the floor.",
    "apply_to_kind": "Use it for 'all ...' requests, e.g. kind wall, op set_wall_thickness, params {thickness_m: 0.12}, "
                     "filter {wall_kind: interior}. All matches change in one step (all or nothing); chain defaults to false.",
    "rename_rooms_bulk": "names: [{id, name}].",
}


def full_op_schemas() -> dict[str, dict[str, Any]]:
    """{op name: {"description", "parameters"}} of the full-control ops."""
    return {name: {"description": spec.description, "parameters": copy.deepcopy(spec.schema)} for name, spec in FULL_OPS.items()}


def full_edit_tools() -> list[Any]:
    """The full-control edit tools (:class:`~samcloud.assistant.domains.base.ToolDef`, handled like every edit tool)."""
    from samcloud.assistant.domains.base import ToolDef
    from samcloud.assistant.tools import _edit

    tools = []
    for name, entry in full_op_schemas().items():
        description = f"{entry['description']} {_FULL_NOTES.get(name, '')}".strip()
        tools.append(ToolDef(name, description, entry["parameters"], _edit, "edit"))
    return tools


__all__ = ["BULK_OPS", "FULL_OPS", "FULL_OP_NAMES", "full_edit_tools", "full_op_schemas"]
