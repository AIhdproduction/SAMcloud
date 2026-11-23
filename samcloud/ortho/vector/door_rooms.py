"""Rooms go from door to door: the interior floor is split only by walls and by door lines.

``rooms_from_wall_cells`` finds the rooms that the wall bodies enclose.  The wall bodies of a real flat are never
complete (a niche hides a wall, a window gap is wider than a crack), so large parts of the floor - a hall that
touches five doors, strips beside a cupboard, pockets behind furniture - belong to no cell.  This module adds the
missing floor and hands it out by one rule:

* a room boundary exists only along a wall body, a window span or a **door line** (the line through the wall centre,
  across the full door width: each side gets half of the door reveal);
* an opening without a door (a ``passage``) does not split; a corridor that is open to a room belongs to it;
* furniture never splits or carves a room (the floor outline closes over it and pockets are filled);
* every interior m2 belongs to exactly one room (a floor stretch that no room reaches without crossing a door
  line becomes a room of its own, e.g. a hall between doors).

Everything is metric and scaled by ``units_per_metre`` at use.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
import math
from typing import Any

import numpy as np
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from .raster_grid import RasterGrid, mask_to_polygons


@dataclass(frozen=True, slots=True)
class DoorRoomSettings:
    """Metric thresholds (config ``plan_polish:`` keys ``door_*``)."""

    enabled: bool = True
    floor_close_m: float = 0.50  # the free floor closes over furniture / gaps up to twice this
    floor_grow_m: float = 0.30  # the closed floor grows towards the wall faces (the free-floor mask keeps clear of walls)
    wall_reach_m: float = 0.60  # ... but only within this distance of a wall body
    strip_open_m: float = 0.20  # floor strips thinner than twice this (beyond a missing wall) are not interior
    merge_contact_m: float = 1.00  # rooms open to each other along this much boundary (wider than any door; no wall, no door line) are one room
    door_overshoot_m: float = 0.30  # a door line reaches this far beyond the door width, into the walls beside it
    line_buffer_m: float = 0.04  # half width of the cut a door line makes in the floor
    contact_m: float = 0.03  # two floor pieces touch when closer than this
    min_contact_m: float = 0.15  # open boundary needed to attach a piece to a room
    min_piece_m2: float = 0.02  # smaller leftover pieces are dropped
    min_room_m2: float = 0.50  # a floor piece closed in by doors becomes a room of its own from this area (the plan names it a hall from 1 m2)
    furniture_reach_m: float = 0.40  # a furniture blob this close to the free floor / a room is interior
    support_window_m: float = 0.50  # measured-support density window
    support_density: float = 0.50  # share of measured pixels that makes a window part of the measured ground
    support_step_m: float = 0.40  # steps of the measured-ground outline smaller than this are closed straight
    support_grow_m: float = 0.25  # the measured ground reaches this far beyond (into the wall thickness)
    gap_close_m: float = 0.30  # wall gaps narrower than twice this (any door is wider) are closed for room building
    gap_close_max_m2: float = 0.60  # ... when the closed piece is this small
    exterior_gap_max_m: float = 3.2  # a missing stretch of an exterior wall up to this long is bridged along its line
    pocket_close_m: float = 0.60  # pockets between a room and a wall / another room are closed up to twice this
    pocket_min_contact_m: float = 0.25
    wall_pocket_max_m2: float = 1.0  # a corner pocket bounded by walls and one room is closed up to this area (0 = off)
    wall_face_reach_m: float = 0.60  # a pocket between rooms reaches the wall faces within this distance even where the measured support ends (floor hidden behind furniture)
    pocket_open_share: float = 0.20  # a corner pocket outside every measured room is closed only when at most this share of its outline is open (neither wall nor room): a corner behind furniture is enclosed, a pocket open to unseen space is not
    passage_front_reach_m: float = 0.15  # a room is in front of a passage when it comes this close to the probes just beyond the passage's wall faces (face position noise of a few cm plus the floor_grow tolerance of the room outline)


def opening_slab(opening: Any) -> Polygon:
    """The wall stretch an opening cuts: ``width`` along its axis x ``wall_thickness`` across (plan units)."""
    width = float(opening.width)
    thickness = max(float(opening.wall_thickness), 1e-9)
    slab = box(-width / 2.0, -thickness / 2.0, width / 2.0, thickness / 2.0)
    slab = affinity.rotate(slab, float(opening.axis_angle), origin=(0.0, 0.0), use_radians=True)
    return affinity.translate(slab, float(opening.center[0]), float(opening.center[1]))


def door_line(opening: Any, overshoot: float = 0.0) -> LineString:
    """The door line: through the wall centre along the door axis, ``width`` long (plus ``overshoot`` at both ends)."""
    ux, uy = math.cos(float(opening.axis_angle)), math.sin(float(opening.axis_angle))
    half = float(opening.width) / 2.0 + overshoot
    cx, cy = float(opening.center[0]), float(opening.center[1])
    return LineString([(cx - ux * half, cy - uy * half), (cx + ux * half, cy + uy * half)])


def _half_slabs(opening: Any) -> list[Polygon]:
    """The two halves of the door slab on either side of the door line (each is half the door reveal)."""
    slab = opening_slab(opening)
    thickness = max(float(opening.wall_thickness), 1e-9)
    nx, ny = -math.sin(float(opening.axis_angle)), math.cos(float(opening.axis_angle))
    halves: list[Polygon] = []
    for sign in (1.0, -1.0):
        side = affinity.translate(
            affinity.rotate(box(-1e6, 0.0, 1e6, sign * thickness), float(opening.axis_angle), origin=(0.0, 0.0), use_radians=True),
            float(opening.center[0]), float(opening.center[1]),
        )
        halves.extend(_polygons(slab.intersection(side)))
    return halves


def _polygons(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        return [part for item in geometry.geoms for part in _polygons(item)]
    return []


def _fill_holes(geometry: Any) -> Any:
    parts = [Polygon(polygon.exterior) for polygon in _polygons(geometry)]
    return unary_union(parts) if parts else GeometryCollection()


def _support_limit(support: np.ndarray, grid: RasterGrid, settings: DoorRoomSettings, upm: float, theta: float = 0.0) -> Any:
    """Polygon of the measured ground: ``support`` pixels with at least ``support_density`` of a window of ``support_window_m`` measured."""
    import cv2

    window = max(3, int(round(settings.support_window_m * upm / grid.pixel_size)) | 1)
    density = cv2.blur(support.astype(np.float32), (window, window))
    dense = (density >= settings.support_density).astype(np.uint8)
    close = max(1, int(round(settings.floor_close_m * 0.4 * upm / grid.pixel_size)))
    dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1)))
    parts = [Polygon(polygon.exterior) for polygon in mask_to_polygons(dense.astype(bool), grid) if polygon.area >= 1.0 * upm * upm]
    if not parts:
        return GeometryCollection()
    from .plan_polish import rectilinear_outline

    # The contour of the measured ground is ragged: room edges that follow it (where no wall stands) are closed straight along the wall axes.
    straight = []
    for part in parts:
        simple = rectilinear_outline(part, theta, settings.support_step_m * upm)
        straight.append(unary_union([part, simple]) if simple is not None and not simple.is_empty and simple.area >= 0.9 * part.area else part)
    return unary_union(straight).buffer(settings.support_grow_m * upm, join_style=2)


def exterior_gap_bridges(pieces: list[dict[str, Any]], open_slabs: list[Polygon], upm: float, max_gap: float) -> list[Polygon]:
    """Wall rectangles that close the gaps between collinear exterior wall pieces (faces within 0.15 m, gap up to ``max_gap``).

    ``pieces`` are ``DoubleWallResult.pieces`` (``face`` / ``face_b`` end points, ``kind``); a gap that holds a door or passage
    span (``open_slabs``) is an opening and stays open.  The bridge continues the faces of the piece it starts at.
    """
    entries: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for piece in pieces:
        if piece.get("kind") != "exterior":
            continue
        a, b = np.asarray(piece["face"][0], dtype=float), np.asarray(piece["face"][1], dtype=float)
        a2, b2 = np.asarray(piece["face_b"][0], dtype=float), np.asarray(piece["face_b"][1], dtype=float)
        length = float(np.hypot(*(b - a)))
        if length < 1e-6:
            continue
        direction = (b - a) / length
        entries.append((a, b, a2, b2, direction))
        entries.append((b, a, b2, a2, -direction))  # both orientations
    line_tolerance = 0.15 * upm
    minimum = 0.05 * upm
    bridges: list[Polygon] = []
    seen: set[tuple[int, int]] = set()
    for first, (a, b, a2, b2, d) in enumerate(entries):
        across = np.array([-d[1], d[0]])
        for second, (c, e, c2, e2, d2) in enumerate(entries):
            if first == second or abs(float(d[0] * d2[1] - d[1] * d2[0])) > 0.03 or float(d @ d2) <= 0.0:
                continue
            if abs(float((c - b) @ across)) > line_tolerance or abs(float((c2 - b2) @ across)) > line_tolerance:
                continue
            gap = float((c - b) @ d)  # from the end of the first piece to the start of the second, along the wall line
            if not minimum < gap <= max_gap:
                continue
            rectangle = Polygon([tuple(b), tuple(b + d * gap), tuple(b2 + d * gap), tuple(b2)])
            if not rectangle.is_valid or rectangle.area <= 0.0:
                continue
            if any(rectangle.intersection(slab).area > 0.05 * min(rectangle.area, slab.area) for slab in open_slabs):
                continue
            key = (min(first, second), max(first, second))
            if key in seen:
                continue
            seen.add(key)
            bridges.append(rectangle)
    return bridges


def _joined(core: Polygon, pieces: list[Polygon], tolerance: float) -> Any:
    """``core`` with its attached ``pieces``; parts that stay apart only by a gap within ``tolerance`` are closed together."""
    joined = unary_union([core, *pieces])
    if len(_polygons(joined)) <= 1 or tolerance <= 0.0:
        return joined
    return joined.buffer(tolerance, join_style=2).buffer(-tolerance, join_style=2)


def _open_share(gap: Polygon, walls: Any, rooms: list[Any], tolerance: float, openings: Any = None) -> float:
    """Share of the outline of ``gap`` that runs along neither a wall body, a room nor a door / passage (``openings``: their slabs
    and door lines); the rest is its open side towards unseen space."""
    length = float(gap.boundary.length)
    if length <= 0.0:
        return 0.0
    closed = unary_union([shape for shape in (walls, openings, *rooms) if shape is not None and not shape.is_empty]).buffer(tolerance, join_style=2)
    return float(gap.boundary.difference(closed).length) / length


def _onto_wall_band(opening: Any, bodies: Any) -> Any:
    """``opening`` with centre and wall thickness of the wall band it stands in (probed across its centre and both ends).

    The band is the run of wall body along the opening normal that holds or touches the detected slab (within a quarter wall
    thickness), at most twice the
    opening's wall thickness wide (a thicker band is a crossing wall, not the host); without such a band the opening is unchanged.
    """
    if bodies is None or bodies.is_empty:
        return opening
    ux, uy = math.cos(float(opening.axis_angle)), math.sin(float(opening.axis_angle))
    normal = np.asarray((-uy, ux))
    centre = np.asarray((float(opening.center[0]), float(opening.center[1])))
    thickness = max(float(opening.wall_thickness), 1e-9)
    half = float(opening.width) / 2.0
    bands: list[tuple[float, float]] = []
    for along in (-0.9 * half, 0.0, 0.9 * half):
        point = centre + np.asarray((ux, uy)) * along
        line = LineString([tuple(point - normal * 2.0 * thickness), tuple(point + normal * 2.0 * thickness)])
        hit = line.intersection(bodies)
        for part in getattr(hit, "geoms", [hit]):
            if part.is_empty or part.geom_type != "LineString":
                continue
            values = sorted(float((np.asarray(xy) - point) @ normal) for xy in part.coords)
            low, high = values[0], values[-1]
            # the band holds or touches the detected slab (a quarter of a wall thickness of detection noise allowed)
            if high - low <= 2.0 * thickness and low <= 0.75 * thickness and high >= -0.75 * thickness:
                bands.append((low, high))
    if not bands:
        return opening
    low, high = float(np.median([band[0] for band in bands])), float(np.median([band[1] for band in bands]))
    if high - low <= 0.25 * thickness:
        return opening
    shifted = centre + normal * (low + high) / 2.0
    return SimpleNamespace(kind=opening.kind, center=(float(shifted[0]), float(shifted[1])), axis_angle=float(opening.axis_angle), width=float(opening.width), wall_thickness=high - low)


def _contact(piece: Polygon, room: Polygon, tolerance: float) -> float:
    """Length of the room outline that lies against ``piece`` (within ``tolerance``)."""
    shared = room.boundary.intersection(piece.buffer(tolerance, join_style=2))
    return float(shared.length)


def door_bounded_rooms(
    rooms: list[Polygon],
    wall_bodies: Iterable[Polygon],
    openings: Iterable[Any],
    floor: np.ndarray | None,
    grid: RasterGrid | None,
    units_per_metre: float,
    settings: DoorRoomSettings = DoorRoomSettings(),
    furniture: np.ndarray | None = None,
    support: np.ndarray | None = None,
    wall_pieces: Iterable[dict[str, Any]] = (),
    theta: float = 0.0,
    passage_reveals: Iterable[Polygon] = (),
) -> tuple[list[Polygon], dict[str, Any]]:
    """Complete ``rooms`` (wall cells / measured rooms) to the whole interior floor, split only by walls and door lines.

    ``openings`` are Opening-like objects (``kind`` DOOR / WINDOW / passage, ``center``, ``axis_angle``, ``width``,
    ``wall_thickness``).  Returns the rooms (the input rooms grown by what belongs to them, plus new rooms for floor
    that is reached only through doors) and statistics.  ``passage_reveals`` (optional) are the passages between both faces of
    their host wall: the detected slab lies on ONE face and need not reach the room behind the other one.
    """
    upm = max(float(units_per_metre), 1e-9)
    stats: dict[str, Any] = {"door_room_exterior_gaps": 0, "door_room_cores_cut": 0, "door_room_gaps_closed": 0, "door_room_pieces": 0, "door_room_new": 0, "door_room_pockets": 0, "door_room_halves": 0, "door_room_area_added_m2": 0.0}
    cores = [room for room in rooms if isinstance(room, Polygon) and not room.is_empty]
    evidenced = unary_union(cores) if cores else GeometryCollection()  # the measured rooms / wall cells as given (before any cut-back)
    if not settings.enabled or not cores:
        return list(rooms), stats
    bodies = [body for body in wall_bodies if isinstance(body, Polygon) and not body.is_empty]
    opening_list = list(openings)
    kind_of = lambda item: str(item.kind).upper()  # noqa: E731
    doors = [item for item in opening_list if kind_of(item) == "DOOR"]
    passages = [item for item in opening_list if kind_of(item) == "PASSAGE"]
    windows = [item for item in opening_list if kind_of(item) == "WINDOW"]
    # A window detected on one face of its wall stands in the wall's band: its span is moved onto the wall body it lies in, so it
    # never reaches into the room beside it (the room ends at the wall face, not at the detected face + half a wall).
    body_union = unary_union(bodies) if bodies else GeometryCollection()
    windows = [_onto_wall_band(item, body_union) for item in windows]
    walls = unary_union(bodies + [opening_slab(item) for item in windows]) if (bodies or windows) else GeometryCollection()
    # A gap in the wall line narrower than any door (a missing piece beside an opening, a short unmeasured stub) is no way through:
    # it is closed for room building, so rooms never flood through it.  Door, window and passage spans stay open / closed as they are.
    bridge_union: Any = GeometryCollection()
    all_slabs = [opening_slab(item) for item in opening_list if kind_of(item) in {"DOOR", "WINDOW", "PASSAGE"}]
    if settings.gap_close_m > 0.0 and (bodies or windows):
        shell = unary_union(bodies + all_slabs)
        radius = settings.gap_close_m * upm
        sealed = shell.buffer(radius, join_style=2).buffer(-radius, join_style=2)
        fill = sealed.difference(shell)
        bridges = [piece for piece in _polygons(fill) if piece.area <= settings.gap_close_max_m2 * upm * upm]
        if bridges:
            bridge_union = unary_union(bridges)
            walls = unary_union([walls, bridge_union])
            stats["door_room_gaps_closed"] = len(bridges)
    # A missing stretch of an exterior wall (not measured, not a door / passage span) is closed along the wall line between the
    # two collinear pieces, so the interior never floods out through it.
    if settings.exterior_gap_max_m > 0.0:
        window_pieces = []
        for item in windows:  # a window span is part of the exterior wall line: both orientations / face sides, so it chains with its neighbours
            ux, uy = math.cos(float(item.axis_angle)), math.sin(float(item.axis_angle))
            nx, ny = -uy, ux
            half, thick = float(item.width) / 2.0, float(item.wall_thickness)
            cx, cy = float(item.center[0]), float(item.center[1])
            for flip in (1.0, -1.0):
                for side in (1.0, -1.0):
                    base = (cx - flip * ux * half - side * nx * thick / 2.0, cy - flip * uy * half - side * ny * thick / 2.0)
                    end = (cx + flip * ux * half - side * nx * thick / 2.0, cy + flip * uy * half - side * ny * thick / 2.0)
                    # the other face of the SAME band (c -/+ t/2): the window's own wall stretch, never a band one thickness beside it
                    offset = (side * nx * thick, side * ny * thick)
                    window_pieces.append({"kind": "exterior", "face": [list(base), list(end)], "face_b": [[base[0] + offset[0], base[1] + offset[1]], [end[0] + offset[0], end[1] + offset[1]]]})
        bridges = exterior_gap_bridges(list(wall_pieces) + window_pieces, [opening_slab(item) for item in doors + passages], upm, settings.exterior_gap_max_m * upm)
        if bridges:
            walls = unary_union([walls, *bridges])
            bridge_union = unary_union([bridge_union, *bridges]) if not bridge_union.is_empty else unary_union(bridges)
            stats["door_room_exterior_gaps"] = len(bridges)
    lines = [door_line(item, settings.door_overshoot_m * upm) for item in doors]
    cut = unary_union(lines).buffer(settings.line_buffer_m * upm, cap_style=2) if lines else GeometryCollection()
    slabs = [opening_slab(item) for item in doors]  # door reveals are shared out last, half to each side
    slab_union = unary_union(slabs) if slabs else GeometryCollection()
    passage_slabs = [opening_slab(item) for item in passages]  # a passage is plain floor

    # --- a measured room that a door line runs through (its outline covers both sides of a door: the wall around the door was
    # not measured, a ceiling room reaches across it) is cut at the door line; the smaller parts are handled as floor pieces
    split_off: list[Polygon] = []
    blockers = unary_union([cut, bridge_union]) if not cut.is_empty else GeometryCollection()
    if not blockers.is_empty:
        kept_cores: list[Polygon] = []
        for core in cores:
            parts = sorted((part for part in _polygons(core.difference(blockers)) if part.area >= settings.min_piece_m2 * upm * upm), key=lambda item: -item.area)
            if len(parts) <= 1:
                kept_cores.append(core)
                continue
            kept_cores.append(parts[0])
            split_off.extend(parts[1:])
        cores = kept_cores
        stats["door_room_cores_cut"] = len(split_off)
    # --- measured support: the interior is where points were measured; unobserved ground (outside the wall line, a stairwell
    # behind a wall, the void where a wall is missing) never becomes room floor, whatever the wall bodies leave open
    limit: Any = None
    if support is not None and grid is not None and np.asarray(support).any():
        limit = _support_limit(np.asarray(support, dtype=bool), grid, settings, upm, theta)
    if limit is not None and not limit.is_empty:
        # a measured (ceiling) room reaching into unmeasured ground (past a missing wall, out of a window) is cut back to the measured ground
        clipped = []
        for core in cores:
            parts = sorted(_polygons(core.intersection(limit)), key=lambda item: -item.area)
            clipped.append(parts[0] if parts and parts[0].area >= 0.5 * core.area else core)
        cores = clipped
    # --- the interior floor: free floor + rooms, closed over furniture, grown to the wall faces
    base_parts: list[Any] = list(cores)
    floor_parts: list[Polygon] = []
    if floor is not None and grid is not None and np.asarray(floor).any():
        floor_parts = [Polygon(polygon.exterior) for polygon in mask_to_polygons(np.asarray(floor, dtype=bool), grid) if polygon.area >= 0.2 * upm * upm]
        base_parts.extend(floor_parts)
    if furniture is not None and grid is not None and np.asarray(furniture).any():
        # Furniture stands on the floor: a blob that lies against the free floor or a room is part of the interior; a
        # blob far from both (an object seen through a window) is not.
        anchor = unary_union([*floor_parts, *cores]).buffer(settings.furniture_reach_m * upm)
        for polygon in mask_to_polygons(np.asarray(furniture, dtype=bool), grid):
            if polygon.area >= 0.02 * upm * upm and polygon.intersects(anchor):
                base_parts.append(Polygon(polygon.exterior))
    close = settings.floor_close_m * upm
    closed = unary_union(base_parts).buffer(close, join_style=2).buffer(-close, join_style=2)
    closed = _fill_holes(closed)
    # The free-floor mask keeps clear of the walls: it grows to the wall faces, but only where a wall stands close (never
    # into the void where a wall is missing, never out of a window).
    grow = closed.buffer(settings.floor_grow_m * upm, join_style=2)
    grown = unary_union([closed, grow.intersection(walls.buffer(settings.wall_reach_m * upm, join_style=2))]) if not walls.is_empty else closed
    interior = grown.difference(walls) if not walls.is_empty else grown
    strip = settings.strip_open_m * upm
    interior = interior.buffer(-strip, join_style=2).buffer(strip, join_style=2)
    if limit is not None:
        interior = interior.intersection(limit)
    interior = unary_union([interior, *cores, *passage_slabs])
    if not walls.is_empty:
        interior = interior.difference(walls)
    reveal = unary_union([slab.difference(walls) for slab in slabs]) if slabs else GeometryCollection()

    # --- floor that no room covers, cut along the door lines, without the door slabs (they are shared out last)
    core_union = unary_union(cores)
    leftover = interior.difference(core_union)
    if not slab_union.is_empty:
        leftover = leftover.difference(slab_union)
    if not cut.is_empty:
        leftover = leftover.difference(cut)
    pieces = [piece for piece in _polygons(leftover) if piece.area >= settings.min_piece_m2 * upm * upm] + split_off
    members: list[list[Polygon]] = [[] for _ in cores]
    new_rooms: list[list[Polygon]] = []
    tolerance = settings.contact_m * upm
    unattached: list[Polygon] = []
    for piece in sorted(pieces, key=lambda item: -item.area):
        contacts = [_contact(piece, core, tolerance) for core in cores]
        best = max(range(len(cores)), key=lambda index: contacts[index])
        if contacts[best] >= settings.min_contact_m * upm:
            members[best].append(piece)
            stats["door_room_pieces"] += 1
        else:
            unattached.append(piece)
    # floor reached only through doors: touching pieces form one new room; tiny ones go to the neighbour with most contact
    groups: list[list[Polygon]] = []
    for piece in unattached:
        joined = [group for group in groups if any(piece.distance(other) <= tolerance for other in group)]
        merged = [piece]
        for group in joined:
            merged.extend(group)
            groups.remove(group)
        groups.append(merged)
    new_polygons: list[Polygon] = []
    for group in groups:
        merged = unary_union(group)
        if merged.area >= settings.min_room_m2 * upm * upm:
            new_polygons.extend(_polygons(merged))
            stats["door_room_new"] += 1

    # A piece joins its room because it lies within the contact tolerance of it: the hairline between them is floor too (else the
    # union stays two parts and the smaller one is dropped below).
    rooms_out: list[Any] = [_joined(core, members[index], tolerance) for index, core in enumerate(cores)]
    rooms_out.extend(new_polygons)
    # --- every door reveal: each half goes to the room it touches
    all_rooms = [_largest_polygon(item) for item in rooms_out]
    halves: list[list[Polygon]] = [[] for _ in all_rooms]
    for door in doors:
        for half in _half_slabs(door):
            part = half.difference(walls) if not walls.is_empty else half
            for piece in _polygons(part):
                if piece.area <= 0.0:
                    continue
                contacts = [_contact(piece, room, tolerance * 2.0) if room is not None else 0.0 for room in all_rooms]
                best = max(range(len(all_rooms)), key=lambda index: contacts[index])
                if contacts[best] > 0.0:
                    halves[best].append(piece)
                    stats["door_room_halves"] += 1
    rooms_out = [unary_union([room, *halves[index]]) if room is not None else None for index, room in enumerate(all_rooms)]
    # --- rooms that are open to each other along a boundary wider than any door (no wall, no door line) are one room
    door_cut = unary_union([door_line(item, 0.15 * upm) for item in doors]).buffer(0.10 * upm) if doors else GeometryCollection()
    blocked = unary_union([door_cut, walls.buffer(0.01 * upm)]) if not walls.is_empty else door_cut
    probes = [door_probes(item, tolerance) for item in doors]
    stats["door_room_open_merged"] = _merge_open_rooms(rooms_out, blocked, settings.merge_contact_m * upm, tolerance * 2.0, probes)
    # --- a passage is plain floor: the rooms on both sides of it are one room
    free = (lambda shape: shape.difference(walls)) if not walls.is_empty else (lambda shape: shape)
    reveal_list = [shape for shape in passage_reveals if isinstance(shape, Polygon) and not shape.is_empty]
    # Only the rooms in front of a passage (at its middle, on both sides of its wall) are joined by it: a detected span that
    # overshoots past a crossing wall touches the room behind that wall with its end, and that room is not reached through it.
    fronts = [passage_probes(item, [opening_slab(item), *(shape for shape in reveal_list if shape.intersects(opening_slab(item)))], tolerance) for item in passages]
    merged_count = _merge_through_passages(
        rooms_out, [free(slab) for slab in passage_slabs], tolerance * 2.0, [free(shape) for shape in reveal_list],
        fronts=fronts, front_reach=settings.passage_front_reach_m * upm, min_part=settings.min_room_m2 * upm * upm,
    )
    stats["door_room_merged"] = merged_count
    # --- pockets between a room and a wall / room outline (ragged around furniture) belong to the room they touch most
    live = [room for room in rooms_out if room is not None]
    if live:
        pocket_close = settings.pocket_close_m * upm
        union = unary_union(live)
        region = union.buffer(pocket_close, join_style=2).buffer(-pocket_close, join_style=2)
        gaps = region.difference(union)
        if limit is not None:
            # The measured ground limits the pockets against the void; floor a wall face hides from the camera (a strip between a room and
            # the wall beside it, furniture in front) stays: a room reaches the wall faces, whatever the measured support shows.
            hidden = gaps.intersection(walls.buffer(settings.wall_face_reach_m * upm, join_style=2)) if not walls.is_empty else GeometryCollection()
            gaps = unary_union([gaps.intersection(limit), hidden]) if not hidden.is_empty else gaps.intersection(limit)
        if not walls.is_empty:
            gaps = gaps.difference(walls)
        if not cut.is_empty:
            gaps = gaps.difference(cut)
        gaps = gaps.difference(reveal) if not reveal.is_empty else gaps
        for gap in _polygons(gaps):
            if gap.area < settings.min_piece_m2 * upm * upm:
                continue
            contacts = [_contact(gap, room, tolerance * 2.0) if room is not None else 0.0 for room in rooms_out]
            best = max(range(len(rooms_out)), key=lambda index: contacts[index])
            if contacts[best] >= settings.pocket_min_contact_m * upm:
                rooms_out[best] = unary_union([rooms_out[best], gap])
                stats["door_room_pockets"] += 1
    # --- corner pockets: a notch or strip bounded by walls and by ONE room (furniture / a tall unit hid the floor in the corner / along the wall) is closed over the
    # walls as well, so a room reaches the wall faces; a notch that touches two rooms, a door reveal or a door line is left alone
    stats["door_room_wall_pockets"] = 0
    live = [room for room in rooms_out if room is not None]
    if live and not walls.is_empty and settings.wall_pocket_max_m2 > 0.0:
        pocket_close = settings.pocket_close_m * upm
        union = unary_union(live)
        region = unary_union([union, walls]).buffer(pocket_close, join_style=2).buffer(-pocket_close, join_style=2)
        gaps = region.difference(union).difference(walls)
        gaps = gaps.difference(slab_union) if not slab_union.is_empty else gaps
        gaps = gaps.difference(cut) if not cut.is_empty else gaps
        gaps = gaps.difference(reveal) if not reveal.is_empty else gaps
        for gap in _polygons(gaps):
            thin = gap.length > 0.0 and 2.0 * gap.area / gap.length <= settings.wall_face_reach_m * upm  # a strip along a wall: mean width within the wall-face reach
            if gap.area < settings.min_piece_m2 * upm * upm or not (gap.area <= settings.wall_pocket_max_m2 * upm * upm or thin):
                continue
            doorways = unary_union([slab_union, cut, *passage_slabs])
            if gap.intersection(evidenced).area < 0.5 * gap.area and (doorways.is_empty or gap.distance(doorways) > tolerance * 2.0) and _open_share(gap, walls, rooms_out, tolerance * 2.0, doorways) > settings.pocket_open_share:
                continue  # a pocket hidden behind furniture is enclosed by walls and the room (a pocket at a doorway is its reveal); one that opens onto unseen space is no floor of the room
            contacts = [_contact(gap, room, tolerance * 2.0) if room is not None else 0.0 for room in rooms_out]
            touching = [index for index, value in enumerate(contacts) if value >= settings.pocket_min_contact_m * upm]
            if len(touching) != 1:
                continue
            rooms_out[touching[0]] = unary_union([rooms_out[touching[0]], gap])
            stats["door_room_wall_pockets"] += 1
    result: list[Polygon] = []
    final = [room for room in rooms_out if room is not None and not room.is_empty]
    for index, room in enumerate(final):
        cleaned = _fill_holes(room.buffer(0.01 * upm, join_style=2).buffer(-0.01 * upm, join_style=2))  # furniture leaves no hole; a pillar is cut out again
        if not walls.is_empty:
            cleaned = cleaned.difference(walls)
        # Every m2 belongs to one room: a room enclosed by this one, or a reveal half the smaller room took, is no floor of this one.
        for other_index, other in enumerate(final):
            if (other.area, other_index) < (room.area, index) and cleaned.intersection(other).area > 1e-6 * upm * upm:
                cleaned = cleaned.difference(other)
        polygon = _largest_polygon(cleaned)
        if polygon is not None and polygon.area > 0.0:
            result.append(polygon)
    stats["door_room_area_added_m2"] = float(sum(polygon.area for polygon in result) - sum(core.area for core in cores)) / (upm * upm)
    return result, stats


def door_probes(opening: Any, margin: float) -> list[tuple[Point, Point]]:
    """Points on the two sides of a door, just beyond its wall faces, at three places along the door width: ``(side +, side -)`` each."""
    ux, uy = math.cos(float(opening.axis_angle)), math.sin(float(opening.axis_angle))
    nx, ny = -uy, ux
    reach = max(float(opening.wall_thickness), 1e-9) / 2.0 + margin
    cx, cy = float(opening.center[0]), float(opening.center[1])
    probes: list[tuple[Point, Point]] = []
    for fraction in (-0.25, 0.0, 0.25):
        along = fraction * float(opening.width)
        px, py = cx + ux * along, cy + uy * along
        probes.append((Point(px + nx * reach, py + ny * reach), Point(px - nx * reach, py - ny * reach)))
    return probes


def _door_connects(a: Any, b: Any, probes: list[list[tuple[Point, Point]]], reach: float) -> bool:
    """True when a door stands between ``a`` and ``b``: one room lies at one side of the door, the other room at the other side.

    A door that only touches the shared boundary of the two rooms (it connects a third room to one of them, or both rooms touch its
    reveal sideways) does not separate them.
    """
    for door in probes:
        for plus, minus in door:
            if (a.distance(plus) <= reach and b.distance(minus) <= reach) or (a.distance(minus) <= reach and b.distance(plus) <= reach):
                return True
    return False


def _merge_open_rooms(rooms: list[Any], blocked: Any, minimum: float, tolerance: float, door_probes_: list[list[tuple[Point, Point]]]) -> int:
    """Merge (in place; a merged-away entry becomes None) rooms whose outlines touch along at least ``minimum`` outside ``blocked`` (door lines, walls).

    A shorter open stretch is a gap in the measured wall (a missing piece beside a door), not an opening.  A door that connects the
    two rooms (one at each side of it) always separates them; a door of a third room touching their shared boundary does not.
    """
    merged = 0
    changed = True
    while changed:
        changed = False
        for first in range(len(rooms)):
            for second in range(first + 1, len(rooms)):
                a, b = rooms[first], rooms[second]
                if a is None or b is None:
                    continue
                if _door_connects(a, b, door_probes_, tolerance):
                    continue  # a door stands between them: it always separates, whatever else is open
                shared = a.boundary.intersection(b.buffer(tolerance, join_style=2))
                if not blocked.is_empty:
                    shared = shared.difference(blocked)
                if shared.length >= minimum:
                    rooms[first], rooms[second] = unary_union([a, b]), None
                    merged += 1
                    changed = True
    return merged


def passage_probes(opening: Any, spans: Iterable[Polygon], margin: float) -> list[tuple[Point, Point]]:
    """Points just beyond both wall faces of a passage at the middle half of its width: ``(side +, side -)`` each.

    ``spans`` (the passage slab and its face-to-face reveal) give the extent across the wall; the probes lie ``margin`` beyond it.
    """
    ux, uy = math.cos(float(opening.axis_angle)), math.sin(float(opening.axis_angle))
    nx, ny = -uy, ux
    cx, cy = float(opening.center[0]), float(opening.center[1])
    thickness = max(float(opening.wall_thickness), 1e-9)
    span = unary_union([shape for shape in spans if shape is not None and not shape.is_empty])
    reach = 4.0 * thickness + float(opening.width)
    probes: list[tuple[Point, Point]] = []
    for fraction in (-0.25, 0.0, 0.25):
        px, py = cx + ux * fraction * float(opening.width), cy + uy * fraction * float(opening.width)
        low, high = -thickness / 2.0, thickness / 2.0
        if not span.is_empty:
            hit = LineString([(px - nx * reach, py - ny * reach), (px + nx * reach, py + ny * reach)]).intersection(span)
            values = [(x - px) * nx + (y - py) * ny for part in getattr(hit, "geoms", [hit]) if not part.is_empty and part.geom_type == "LineString" for x, y in part.coords]
            if values:
                low, high = min(values), max(values)
        probes.append((Point(px + nx * (high + margin), py + ny * (high + margin)), Point(px + nx * (low - margin), py + ny * (low - margin))))
    return probes


def _separate_parts(geometry: Any, minimum: float) -> int:
    return sum(part.area >= minimum for part in _polygons(geometry))


def _merge_through_passages(
    rooms: list[Any], passages: list[Any], reach: float, reveals: Iterable[Any] = (),
    fronts: list[list[tuple[Point, Point]]] | None = None, front_reach: float = 0.0, min_part: float = 0.0,
) -> int:
    """Merge (in place; a merged-away entry becomes None) all rooms that touch the same passage opening.

    ``reveals`` reach from face to face of the host wall: rooms that only meet across one are joined by its floor.  Where the
    passage itself already touches all of them nothing is added.  With ``fronts`` (``passage_probes`` per passage) only rooms
    within ``front_reach`` of a probe are joined.  With ``min_part`` two rooms are only joined when the joined room is one floor
    (no part of at least ``min_part`` would stand apart): the result keeps one polygon per room, a separate part would be lost.
    """
    reveal_list = [reveal for reveal in reveals if not reveal.is_empty]
    merged = 0
    for number, passage in enumerate(passages):
        if passage.is_empty:
            continue
        touching = [index for index, room in enumerate(rooms) if room is not None and room.distance(passage) <= reach]
        across = [reveal for reveal in reveal_list if reveal.intersects(passage)]
        if across:
            span = unary_union([passage, *across])
            meeting = [index for index, room in enumerate(rooms) if room is not None and room.distance(span) <= reach]
            if len(meeting) > max(len(touching), 1):
                touching = meeting
            else:
                across = []
        if fronts is not None and number < len(fronts):
            touching = [index for index in touching if any(rooms[index].distance(point) <= front_reach for pair in fronts[number] for point in pair)]
        for other in touching[1:]:
            joined = unary_union([rooms[touching[0]], rooms[other], *across])
            if min_part > 0.0:
                bridge = unary_union([passage, *across])
                parts = _separate_parts(unary_union([rooms[touching[0]], rooms[other], bridge]).buffer(reach, join_style=2), min_part)
                if parts >= _separate_parts(rooms[touching[0]], min_part) + _separate_parts(rooms[other], min_part):
                    continue  # the two rooms do not meet through this passage: joining them would drop one of them
            rooms[touching[0]] = joined
            rooms[other] = None
            merged += 1
    return merged


def _largest_polygon(geometry: Any) -> Polygon | None:
    parts = _polygons(geometry)
    return max(parts, key=lambda item: item.area) if parts else None
