"""Final polish of an indoor clean plan: rooms from wall cells, regular furniture, wall-hugging kitchen, clean labels.

All thresholds are metric and scaled by ``units_per_metre`` at use.  The functions work on the neutral
plan entities (``{"type": "polyline", "points": ...}`` dictionaries) after the double-line walls are built.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations
import math
import re
from typing import Any

import cv2
import numpy as np
import shapely
from shapely import affinity
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import polylabel, split, unary_union

from .furniture_shapes import FurnitureShapeSettings, fit_furniture_shape, is_sofa, sofa_details
from .raster_grid import RasterGrid, mask_to_polygons


@dataclass(frozen=True, slots=True)
class PlanPolishSettings:
    """Metric thresholds of the plan polish (config ``plan_polish:``)."""

    enabled: bool = True
    # Rooms: every wall-enclosed cell with floor is a room; a cell holding several measured ceiling rooms is
    # split only at a passage (straight or L-shaped cut between two wall corners) no wider than room_split_max_m.
    cell_rooms: bool = True
    room_min_area_m2: float = 1.0
    room_min_floor_share: float = 0.10
    room_split_max_m: float = 2.4
    room_split_min_area_m2: float = 2.0
    room_jog_m: float = 0.04
    wall_crack_cell_max_m2: float = 12.0  # a cell that exists only after the crack closing is accepted as a room up to this area (bath / WC, not a whole flat)
    wall_crack_m: float = 0.02  # hairline gaps (up to 2x this = 4 cm) beside a door/window span are closed before small cells are found
    barrier_bridge_m: float = 0.30  # two door/window spans closer than this are one opening with a wall stub between them (narrower than any passage)
    room_step_m: float = 0.30
    # Furniture: rectangle when it covers the footprint well, else a rectilinear outline along the wall axes.
    furniture: bool = True
    furniture_rect_iou: float = 0.75
    furniture_snap_deg: float = 15.0
    furniture_step_m: float = 0.15
    furniture_max_vertices: int = 8
    furniture_min_area_m2: float = 0.06
    furniture_sliver_m: float = 0.15
    furniture_nested_share: float = 0.80
    furniture_merge_gap_m: float = 0.10
    furniture_detail_tolerance_m: float = 0.03
    # Shape recognition per object (furniture_shapes.py): rectangle, L, U, round or rectilinear outline chosen by
    # (1 - IoU) + class-dependent complexity penalty against the traced footprint; sofas get backrest lines.
    furniture_shapes: bool = True
    # Object labels: inside the object when they fit, else beside it inside the room; tiny objects get none.
    label_min_object_area_m2: float = 0.15
    label_min_height_m: float = 0.06
    label_char_width: float = 0.62
    label_margin_m: float = 0.04  # clear gap between a label inside an object and its outline / detail lines
    # Kitchen run: legs hug the final inner wall faces, one consistent depth.
    kitchen: bool = True
    kitchen_wall_reach_m: float = 0.45
    kitchen_depth_m: float = 0.60
    kitchen_depth_tol_m: float = 0.15
    kitchen_end_reach_m: float = 0.30
    kitchen_segment_reach_m: float = 0.15
    kitchen_layout: bool = True
    # Sanitary blocks (WC, washbasin): back on the nearest final wall face, front into the room (sanitary_facing.py).
    sanitary_facing: bool = True
    sanitary_back_gap_m: float = 0.35
    sanitary_tie_m: float = 0.03
    sanitary_snap_deg: float = 12.0


# Labels whose object cannot be this large / tall; a bigger object keeps its outline but a generic name.
_PLAUSIBLE_SIZE: dict[str, tuple[float | None, float | None]] = {
    "Kissen": (0.8, 1.2), "Decke": (2.5, 1.2), "Kleidung": (1.2, None), "Pflanze": (1.2, None),
    "Stehlampe": (0.5, None), "Stuhl": (0.8, 1.3), "Bett": (None, 1.3), "Nachttisch": (0.6, 1.0),
}
_GENERIC_FURNITURE = "Möbel"
_HEIGHT_PATTERN = re.compile(r"\s+h=([0-9.]+)\s*$")


# --------------------------------------------------------------------------
# rooms
# --------------------------------------------------------------------------

def enclosed_cells(walls: Any, upm: float, minimum_m2: float = 0.5) -> list[Polygon]:
    """Free cells completely enclosed by wall bodies (the holes of the wall union, islands removed)."""
    cells: list[Polygon] = []
    if walls is None or walls.is_empty:
        return cells
    for part in _parts(walls):
        for ring in part.interiors:
            for piece in _parts(Polygon(ring).difference(walls)):
                if piece.area >= minimum_m2 * upm * upm:
                    cells.append(piece)
    return cells


def bridge_barriers(barriers: list[Polygon], reach: float, max_area: float) -> list[Polygon]:
    """Wall stubs between two close door / window spans (the wall bodies there were cut away, e.g. a 25 cm jamb between a door and a window).

    The stub is the part of the convex hull of the two spans that lies outside them and within ``reach`` of both; it is returned when the spans are at most ``reach`` apart
    and the stub is small (``max_area``), so two spans in different walls are not bridged across a room.
    """
    bridges: list[Polygon] = []
    for index, first in enumerate(barriers):
        for second in barriers[index + 1:]:
            if first.intersects(second) or first.distance(second) > reach:
                continue
            stub = unary_union([first, second]).convex_hull.difference(first).difference(second).intersection(first.buffer(reach, join_style=2)).intersection(second.buffer(reach, join_style=2))
            for piece in _parts(stub):
                if 0.0 < piece.area <= max_area:
                    bridges.append(piece)
    return bridges


def close_wall_cracks(walls: Any, tolerance: float, barriers: Any = None) -> Any:
    """Close narrow gaps (hairline cracks up to a short wall stretch) at door / window spans in the wall union.

    Adjacent door spans (or a span and the wall beside it) can leave a crack of a few mm, or a numerically
    zero-width seam, through the full wall thickness.  The cell behind it then leaks into the neighbouring room
    (a bath behind two touching door spans merged with the living room).  A morphological closing with
    ``tolerance`` fills gaps up to twice that width; only the filled pieces that touch a barrier (``barriers`` =
    the door / window span polygons) are kept, every other seam and every wider opening is left unchanged.
    Without ``barriers`` every crack is closed.
    """
    if walls is None or walls.is_empty or tolerance <= 0.0:
        return walls
    closed = walls.buffer(tolerance, join_style=2).buffer(-tolerance, join_style=2)
    if closed.is_empty:
        return walls
    filled = closed.difference(walls)
    if barriers is not None:
        if barriers.is_empty:
            return walls
        pieces = [piece for piece in _parts(filled) if isinstance(piece, Polygon) and piece.distance(barriers) <= tolerance]
        filled = unary_union(pieces) if pieces else GeometryCollection()
    return unary_union([walls, filled]) if not filled.is_empty else walls


def rooms_from_wall_cells(
    rooms: list[Polygon],
    wall_bodies: Iterable[Polygon],
    barriers: Iterable[Polygon],
    floor: np.ndarray | None,
    grid: RasterGrid | None,
    units_per_metre: float,
    theta: float,
    settings: PlanPolishSettings = PlanPolishSettings(),
) -> tuple[list[Polygon], dict[str, int]]:
    """Room outlines = wall cells (inner wall faces, closed by the door and window spans).

    ``rooms`` are the measured (ceiling) rooms; they only decide whether one cell holds several spaces.  A cell
    with two measured rooms is split at the shortest passage between two wall corners (straight or one L step
    along the wall axes, at most ``room_split_max_m``) that leaves each measured room on its own side; without
    such a passage the cell is one room (open plan).  Measured rooms outside every cell keep their outline,
    cut back from the walls and simplified to a rectilinear polygon.
    """
    upm = max(float(units_per_metre), 1e-9)
    stats = {"room_cells": 0, "room_cells_split": 0, "room_rooms_merged": 0, "rooms_outside_cells": 0}
    body_list = [item for item in wall_bodies if isinstance(item, Polygon) and not item.is_empty]
    barrier_list = [item for item in barriers if isinstance(item, Polygon) and not item.is_empty]
    if not body_list:
        return list(rooms), stats
    walls = unary_union(body_list + barrier_list)
    output: list[Polygon] = []
    used: set[int] = set()
    cell_candidates = enclosed_cells(walls, upm, settings.room_min_area_m2)
    if settings.wall_crack_m > 0.0 and barrier_list:
        # Door / window spans leave hairline cracks at their ends; a small space (bath, WC) that is closed only once those
        # cracks are filled is still a room.  Cells formed by the closing that are larger than ``wall_crack_cell_max_m2``
        # are not trusted (their leak may be a real passage): they stay with the measured ceiling rooms.
        bridges = bridge_barriers(barrier_list, settings.barrier_bridge_m * upm, 0.20 * upm * upm) if settings.barrier_bridge_m > 0.0 else []
        sealed = close_wall_cracks(unary_union([walls, *bridges]) if bridges else walls, settings.wall_crack_m * upm, unary_union(barrier_list))
        for extra in enclosed_cells(sealed, upm, settings.room_min_area_m2):
            if extra.area <= settings.wall_crack_cell_max_m2 * upm * upm and not any(extra.intersection(cell).area >= 0.5 * extra.area for cell in cell_candidates):
                cell_candidates.append(extra)
    for cell in cell_candidates:
        cell = _clean_outline(cell, settings.room_jog_m * upm) or cell
        if cell.area < settings.room_min_area_m2 * upm * upm:
            continue
        if floor is not None and grid is not None and _mask_share(cell, floor, grid) < settings.room_min_floor_share:
            continue
        members = [index for index, room in enumerate(rooms) if room.intersection(cell).area >= 0.5 * room.area]
        used.update(members)
        parts = _split_cell(cell, [rooms[index] for index in members], theta, upm, settings, stats)
        stats["room_rooms_merged"] += max(0, len([index for index in members if rooms[index].area >= settings.room_split_min_area_m2 * upm * upm]) - len(parts))
        output.extend(parts)
        stats["room_cells"] += 1
    taken = unary_union(output) if output else GeometryCollection()
    for index, room in enumerate(rooms):
        if index in used:
            continue
        rest = room.difference(walls)
        if not taken.is_empty:
            rest = rest.difference(taken)
        piece = _largest(rest)
        if piece is None or piece.area < max(settings.room_min_area_m2 * upm * upm, 0.5 * room.area):
            continue
        simple = rectilinear_outline(piece, theta, settings.room_step_m * upm)
        if simple is not None:
            simple = _largest(simple.difference(walls))
        if simple is None or simple.area < settings.room_min_area_m2 * upm * upm:
            simple = piece
        output.append(simple)
        stats["rooms_outside_cells"] += 1
    return output, stats


def _split_cell(cell: Polygon, members: list[Polygon], theta: float, upm: float, settings: PlanPolishSettings, stats: dict[str, int], depth: int = 0, shares: list[float] | None = None) -> list[Polygon]:
    """Split at the passage that best separates the measured rooms (misplaced room area + half the cut length)."""
    minimum = settings.room_split_min_area_m2 * upm * upm
    if shares is None:
        shares = [room.intersection(cell).area for room in members]
    # A measured room belongs to this piece when most of its share of the whole wall cell lies in it.
    significant = [k for k, room in enumerate(members) if shares[k] >= minimum and room.intersection(cell).area >= 0.5 * shares[k]]
    if len(significant) < 2 or depth > 6:
        return [cell]
    best: tuple[float, list[Polygon]] | None = None
    for path, bonus in _cut_candidates(cell, theta, settings.room_split_max_m * upm, upm):
        line = LineString(path)
        if line.length <= 1e-9 or not cell.buffer(1e-6 * upm).covers(line):
            continue
        if any(not cell.buffer(-1e-4 * upm).contains(Point((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)) for a, b in zip(path[:-1], path[1:])):
            continue  # the cut must cross the cell, not follow its boundary
        try:
            pieces = [piece for piece in _parts(split(cell, line)) if piece.area > 1e-9 * upm * upm]
        except (GEOSException, ValueError):
            continue
        if len(pieces) != 2 or min(piece.area for piece in pieces) < minimum:
            continue
        inside = [[members[k].intersection(piece).area for k in significant] for piece in pieces]
        owners = [{position for position, k in enumerate(significant) if inside[side][position] >= 0.5 * shares[k]} for side in (0, 1)]
        if not owners[0] or not owners[1]:
            continue
        misfit = sum(shares[k] - max(inside[0][position], inside[1][position]) for position, k in enumerate(significant))
        score = misfit / upm + 0.5 * line.length - (0.05 * upm if bonus else 0.0)
        if best is None or score < best[0]:
            best = (score, pieces)
    if best is None:
        return [cell]
    stats["room_cells_split"] += 1
    output: list[Polygon] = []
    for piece in best[1]:
        output.extend(_split_cell(piece, members, theta, upm, settings, stats, depth + 1, shares))
    return output


def _cut_candidates(cell: Polygon, theta: float, maximum: float, upm: float) -> list[tuple[list[tuple[float, float]], bool]]:
    """Straight or one-step (L) cuts along the wall axes between two reflex corners of the cell."""
    ring = list(orient(cell, 1.0).exterior.coords)[:-1]
    count = len(ring)
    reflex: list[int] = []
    for index in range(count):
        a, v, b = np.asarray(ring[index - 1]), np.asarray(ring[index]), np.asarray(ring[(index + 1) % count])
        first, second = v - a, b - v
        if float(first[0] * second[1] - first[1] * second[0]) < -1e-12 * upm * upm:
            reflex.append(index)
    axis_u = np.asarray((math.cos(theta), math.sin(theta)))
    axis_v = np.asarray((-math.sin(theta), math.cos(theta)))
    jog = 0.04 * upm
    output: list[tuple[list[tuple[float, float]], bool]] = []
    for i, j in combinations(reflex, 2):
        p, q = np.asarray(ring[i]), np.asarray(ring[j])
        distance = float(np.linalg.norm(q - p))
        if distance > maximum or distance < 0.30 * upm:
            continue
        du, dv = float((q - p) @ axis_u), float((q - p) @ axis_v)
        if abs(du) <= jog or abs(dv) <= jog:
            output.append(([tuple(p), tuple(q)], False))
            continue
        for corner in (p + du * axis_u, p + dv * axis_v):
            path = [tuple(p), tuple(corner), tuple(q)]
            long_start, long_end, vertex = (p, corner, i) if np.linalg.norm(corner - p) >= np.linalg.norm(q - corner) else (q, corner, j)
            output.append((path, _extends_edge(long_start, long_end, ring, vertex, upm)))
    return output


def _extends_edge(start: np.ndarray, end: np.ndarray, ring: list[tuple[float, float]], vertex: int, upm: float) -> bool:
    """True when the segment continues a boundary edge that meets the cell at ``vertex`` (a wall face line)."""
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return False
    direction /= length
    for neighbour in (ring[vertex - 1], ring[(vertex + 1) % len(ring)]):
        offset = np.asarray(neighbour) - start
        if abs(float(direction[0] * offset[1] - direction[1] * offset[0])) <= 0.02 * upm and float(offset @ direction) < 0.0:
            return True
    return False


def _clean_outline(polygon: Polygon, jog: float) -> Polygon | None:
    """Remove notches and spikes narrower than ``jog`` (mitred buffers keep the right angles)."""
    if jog <= 0.0:
        return polygon
    closed = polygon.buffer(jog / 2.0, join_style=2).buffer(-jog / 2.0, join_style=2)
    opened = closed.buffer(-jog / 2.0, join_style=2).buffer(jog / 2.0, join_style=2)
    result = _largest(opened)
    if result is None or result.area < 0.95 * polygon.area:
        return polygon
    return result.simplify(0.002 * jog, preserve_topology=True) if isinstance(result, Polygon) else polygon


def _mask_share(polygon: Polygon, mask: np.ndarray, grid: RasterGrid) -> float:
    inside = polygon_mask(polygon, grid)
    return float(np.mean(mask[inside])) if inside.any() else 0.0


def polygon_mask(polygon: Polygon, grid: RasterGrid) -> np.ndarray:
    """Raster mask of a polygon on ``grid``: exactly the pixels whose centre lies inside."""
    output = np.zeros((grid.height, grid.width), dtype=bool)
    if polygon.is_empty:
        return output
    min_x, min_y, max_x, max_y = polygon.bounds
    c0 = max(0, int(math.floor((min_x - grid.min_x) / grid.pixel_size)))
    c1 = min(grid.width, int(math.ceil((max_x - grid.min_x) / grid.pixel_size)) + 1)
    r0 = max(0, int(math.floor((grid.max_y - max_y) / grid.pixel_size)))
    r1 = min(grid.height, int(math.ceil((grid.max_y - min_y) / grid.pixel_size)) + 1)
    if c1 <= c0 or r1 <= r0:
        return output
    xs = grid.min_x + (np.arange(c0, c1) + 0.5) * grid.pixel_size
    ys = grid.max_y - (np.arange(r0, r1) + 0.5) * grid.pixel_size
    gx, gy = np.meshgrid(xs, ys)
    output[r0:r1, c0:c1] = shapely.contains_xy(polygon, gx, gy)
    return output


# --------------------------------------------------------------------------
# rectilinear shapes
# --------------------------------------------------------------------------

def rectilinear_outline(polygon: Polygon, theta: float, step: float, resolution: float | None = None) -> Polygon | None:
    """Rectilinear (wall-axis) approximation of ``polygon`` whose steps are at least ``step`` long."""
    if not isinstance(polygon, Polygon) or polygon.is_empty or step <= 0.0:
        return None
    res = resolution if resolution is not None else step / 4.0
    centre = polygon.centroid
    rotated = affinity.rotate(polygon, -theta, origin=(centre.x, centre.y), use_radians=True)
    min_x, min_y, max_x, max_y = rotated.bounds
    pad = 2.0 * step
    while ((max_x - min_x + 2 * pad) / res) * ((max_y - min_y + 2 * pad) / res) > 2_000_000:
        res *= 1.5
    width = int(math.ceil((max_x - min_x + 2 * pad) / res)) + 1
    height = int(math.ceil((max_y - min_y + 2 * pad) / res)) + 1
    grid = RasterGrid(min_x - pad, max_y + pad, res, width, height)
    mask = polygon_mask(rotated, grid).astype(np.uint8)
    if not mask.any():
        return None
    size = 2 * int(round(step / res / 2.0)) + 1  # odd: the morphology must not shift the outline
    kernel = np.ones((size, size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if not mask.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if count > 2:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == keep).astype(np.uint8)
    polygons = mask_to_polygons(mask.astype(bool), grid)
    shape = _largest(unary_union(polygons)) if polygons else None
    if shape is None:
        return None
    shape = collapse_short_edges(Polygon(shape.exterior).simplify(0.01 * res, preserve_topology=True), step)
    if shape is None:
        return None
    shape = affinity.rotate(shape, theta, origin=(centre.x, centre.y), use_radians=True)
    return shape if isinstance(shape, Polygon) and shape.is_valid and not shape.is_empty else None


def collapse_short_edges(polygon: Polygon, minimum: float, max_iterations: int = 500) -> Polygon | None:
    """Remove the edges shorter than ``minimum`` from an axis-parallel (rectilinear) polygon.

    The shortest edge is removed by moving the shorter of its two (parallel) neighbour edges onto the line of
    the longer one: a stair step becomes one straight edge, a thin spike or notch disappears.  Each move keeps
    the polygon rectilinear; a move that would make it invalid is skipped.
    """
    points = _ring_points(polygon)
    if points is None:
        return None
    blocked: set[tuple[float, float, float, float]] = set()
    for _ in range(max_iterations):
        count = len(points)
        if count <= 4:
            break
        lengths = [float(np.abs(points[(k + 1) % count] - points[k]).sum()) for k in range(count)]
        order = [k for k in np.argsort(lengths) if lengths[k] < minimum]
        moved = False
        for k in order:
            a, b, c, d = points[(k - 1) % count], points[k], points[(k + 1) % count], points[(k + 2) % count]
            key = (round(float(b[0]), 9), round(float(b[1]), 9), round(float(c[0]), 9), round(float(c[1]), 9))
            if key in blocked:
                continue
            shift = c - b
            candidate = [point.copy() for point in points]
            if lengths[(k - 1) % count] <= lengths[(k + 1) % count]:
                candidate[(k - 1) % count] = a + shift  # previous edge a-b moves onto the line of c-d
                candidate[k] = b + shift
            else:
                candidate[(k + 1) % count] = c - shift  # next edge c-d moves onto the line of a-b
                candidate[(k + 2) % count] = d - shift
            cleaned = _clean_ring(candidate)
            if cleaned is not None and len(cleaned) >= 4:
                trial = Polygon(cleaned)
                if trial.is_valid and trial.area > 0.25 * polygon.area:
                    points = cleaned
                    moved = True
                    break
            blocked.add(key)
        if not moved:
            break
    result = Polygon(points)
    return result if result.is_valid and not result.is_empty else polygon


def _ring_points(polygon: Polygon) -> list[np.ndarray] | None:
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return None
    return _clean_ring([np.asarray(point, dtype=float) for point in list(polygon.exterior.coords)[:-1]])


def _clean_ring(points: list[np.ndarray]) -> list[np.ndarray] | None:
    """Drop duplicate and collinear vertices of a closed ring (axis-parallel edges)."""
    ring = list(points)
    changed = True
    while changed and len(ring) >= 3:
        changed = False
        output: list[np.ndarray] = []
        for point in ring:
            if output and float(np.abs(point - output[-1]).sum()) <= 1e-9:
                changed = True
                continue
            output.append(point)
        if len(output) >= 2 and float(np.abs(output[0] - output[-1]).sum()) <= 1e-9:
            output.pop()
            changed = True
        ring = output
        count = len(ring)
        if count < 3:
            return None
        for k in range(count):
            a, b, c = ring[k - 1], ring[k], ring[(k + 1) % count]
            first, second = b - a, c - b
            if abs(float(first[0] * second[1] - first[1] * second[0])) <= 1e-12:
                del ring[k]
                changed = True
                break
    return ring if len(ring) >= 3 else None


def regular_furniture_shape(polygon: Polygon, theta: float, units_per_metre: float, settings: PlanPolishSettings = PlanPolishSettings()) -> tuple[Polygon, str] | None:
    """Rectangle along the wall axes, a truly rotated rectangle, or a simplified rectilinear outline."""
    upm = max(float(units_per_metre), 1e-9)
    shape = _largest(polygon.buffer(0))
    if shape is None:
        return None
    rectangle = shape.minimum_rotated_rectangle
    if isinstance(rectangle, Polygon) and rectangle.area > 0.0:
        corners = np.asarray(rectangle.exterior.coords[:4], dtype=float)
        edge = corners[1] - corners[0]
        deviation = abs(((math.degrees(math.atan2(edge[1], edge[0])) - math.degrees(theta) + 45.0) % 90.0) - 45.0)
        rect_iou = _iou(rectangle, shape)
        if deviation > settings.furniture_snap_deg and rect_iou >= settings.furniture_rect_iou:
            return rectangle, "rect_free"
        if deviation <= settings.furniture_snap_deg and rect_iou >= 0.90:
            snapped = affinity.rotate(rectangle, -_signed_axis_offset(edge, theta), origin=rectangle.centroid, use_radians=True)
            if _iou(snapped, shape) >= settings.furniture_rect_iou:
                return snapped, "rect"
    step = settings.furniture_step_m * upm
    core = rectilinear_outline(shape, theta, step)
    reference = core if core is not None and core.area >= 0.5 * shape.area else shape
    centre = reference.centroid
    local = affinity.rotate(reference, -theta, origin=(centre.x, centre.y), use_radians=True)
    axis_box = affinity.rotate(box(*local.bounds), theta, origin=(centre.x, centre.y), use_radians=True)
    # The rectangle is judged against the step-cleaned outline (a ragged trace understates any fit).
    if _iou(axis_box, reference) >= settings.furniture_rect_iou:
        return axis_box, "rect"
    candidates = [core] if core is not None else []
    for factor in (1.6, 2.56):
        if candidates and len(candidates[-1].exterior.coords) - 1 <= settings.furniture_max_vertices:
            break
        coarser = rectilinear_outline(shape, theta, step * factor)
        if coarser is not None and coarser.area >= 0.5 * shape.area:
            candidates.append(coarser)
    if candidates:
        simple = [item for item in candidates if len(item.exterior.coords) - 1 <= settings.furniture_max_vertices]
        return (simple[0] if simple else min(candidates, key=lambda item: len(item.exterior.coords))), "rectilinear"
    return axis_box, "rect"


def _signed_axis_offset(edge: np.ndarray, theta: float) -> float:
    angle = math.atan2(float(edge[1]), float(edge[0])) - theta
    return (angle + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0


def _iou(first: Polygon, second: Polygon) -> float:
    union = first.union(second).area
    return float(first.intersection(second).area / union) if union > 0.0 else 0.0


# --------------------------------------------------------------------------
# furniture
# --------------------------------------------------------------------------

def _closed_polyline(entity: dict[str, Any]) -> bool:
    return entity.get("type") == "polyline" and bool(entity.get("closed")) and len(entity.get("points", [])) >= 3


def _points(polygon: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in list(polygon.exterior.coords)[:-1]]


def _room_of(polygon: Polygon, rooms: list[Polygon]) -> Polygon | None:
    best = max(rooms, key=lambda room: room.intersection(polygon).area, default=None)
    if best is None or best.intersection(polygon).area <= 0.0:
        return None
    return best


def clip_to_interior(polygon: Polygon, rooms: list[Polygon], walls: Any, units_per_metre: float) -> Polygon | None:
    """Cut a furniture outline back from the wall bodies and to the room it mostly stands in."""
    clipped: Any = polygon
    if walls is not None and not walls.is_empty:
        clipped = clipped.difference(walls)
    piece = _largest(clipped)
    if piece is None:
        return None
    room = _room_of(piece, rooms)
    if room is not None and room.intersection(piece).area >= 0.5 * piece.area:
        piece = _largest(piece.intersection(room))
    if piece is None:
        return None
    if piece.area >= 0.995 * polygon.area:
        return polygon
    return piece.simplify(0.002 * units_per_metre, preserve_topology=True)


def _height_of(text: str) -> float | None:
    match = _HEIGHT_PATTERN.search(text)
    return float(match.group(1)) if match else None


def plausible_label(label: str, area_m2: float, height_m: float | None) -> str:
    """Generic name for an object far too large or tall for its semantic label (e.g. a 4 m² "Kissen")."""
    limits = _PLAUSIBLE_SIZE.get(label)
    if limits is None:
        return label
    max_area, max_height = limits
    if max_area is not None and area_m2 > max_area:
        return _GENERIC_FURNITURE
    if max_height is not None and height_m is not None and height_m > max_height:
        return _GENERIC_FURNITURE
    return label


def regularize_furniture(
    layers: dict[str, list[dict[str, Any]]],
    rooms: list[Polygon],
    walls: Any,
    theta: float,
    units_per_metre: float,
    settings: PlanPolishSettings = PlanPolishSettings(),
    footprint_of: Callable[[dict[str, Any]], Polygon | None] | None = None,
    evidence_of: Callable[[dict[str, Any]], Polygon | None] | None = None,
    height_of: Callable[[np.ndarray], np.ndarray] | None = None,
    shape_settings: FurnitureShapeSettings = FurnitureShapeSettings(),
) -> dict[str, Any]:
    """Regular A-FURN outlines: fragments of one instance merged, the simplest outline that explains the footprint
    (rectangle, L, U, round, rectilinear; see ``furniture_shapes``), nothing inside a wall, nested duplicates and
    clutter below ``furniture_min_area_m2`` dropped, ragged height-step details dropped, sofas with backrest /
    armrest lines from ``height_of``.  ``evidence_of`` returns the traced footprint of an outline (its instance
    mask); without it the outline itself is the evidence.  Texts linked by ``_furn_id`` follow (renamed when
    implausible, dropped with their object)."""
    upm = max(float(units_per_metre), 1e-9)
    stats: dict[str, Any] = {"furniture_merged": 0, "furniture_regularised": 0, "furniture_clipped": 0, "furniture_dropped_small": 0,
                             "furniture_dropped_nested": 0, "furniture_renamed": 0, "furniture_details_dropped": 0, "furniture_overlaps_cut": 0,
                             "furniture_shapes": [], "furniture_sofa_details": 0}
    furniture = layers.get("A-FURN", [])
    texts = {entity.get("_furn_id"): entity for entity in layers.get("A-ANNO", []) if entity.get("type") == "text" and entity.get("_furn_id") is not None}
    dropped_ids: set[Any] = set()
    polylines = [entity for entity in furniture if _closed_polyline(entity)]
    shapes: dict[int, Polygon] = {}
    evidences: dict[int, Polygon] = {}
    for entity in polylines:
        shape = _largest(Polygon(entity["points"]).buffer(0))
        if shape is not None:
            shapes[id(entity)] = shape
            measured = _largest(evidence_of(entity)) if evidence_of is not None else None
            # A traced footprint has no holes: a hole is an occlusion (a person on the sofa), not furniture.
            evidences[id(entity)] = Polygon(measured.exterior) if measured is not None else Polygon(shape.exterior)
    # 1. fragments of one object instance
    by_instance: dict[int, list[dict[str, Any]]] = {}
    for entity in polylines:
        if entity.get("_instance_id") and id(entity) in shapes:
            by_instance.setdefault(int(entity["_instance_id"]), []).append(entity)
    removed: set[int] = set()
    for members in by_instance.values():
        members.sort(key=lambda item: -shapes[id(item)].area)
        for position, keeper in enumerate(members):
            if id(keeper) in removed:
                continue
            for other in members[position + 1:]:
                if id(other) in removed:
                    continue
                if shapes[id(keeper)].distance(shapes[id(other)]) <= settings.furniture_merge_gap_m * upm:
                    joined = unary_union([shapes[id(keeper)], shapes[id(other)]]).buffer(settings.furniture_merge_gap_m * upm / 2.0, join_style=2).buffer(-settings.furniture_merge_gap_m * upm / 2.0, join_style=2)
                    merged = _largest(joined)
                    if merged is not None:
                        shapes[id(keeper)] = merged
                        joined_evidence = _largest(unary_union([evidences[id(keeper)], evidences[id(other)], merged]))
                        evidences[id(keeper)] = Polygon(joined_evidence.exterior) if joined_evidence is not None else merged
                        removed.add(id(other))
                        dropped_ids.add(other.get("_furn_id"))
                        stats["furniture_merged"] += 1
    # 2. regular shape, clipped to the room interior
    kept: list[dict[str, Any]] = []
    for entity in polylines:
        if id(entity) in removed or id(entity) not in shapes:
            if id(entity) not in removed:
                dropped_ids.add(entity.get("_furn_id"))
            continue
        shape = shapes[id(entity)]
        kind = str(entity.get("_footprint_kind", ""))
        if kind in {"circle", "ellipse"}:
            regular = shape
        else:
            fit = regular_furniture_shape(shape, theta, upm, settings)
            if fit is None:
                dropped_ids.add(entity.get("_furn_id"))
                continue
            regular, kind = fit
            if settings.furniture_shapes:
                label = entity.get("_semantic_label") or entity.get("_label")
                extra = [("rectilinear", regular, (len(regular.exterior.coords) - 5) / 2.0 + 0.5)] if kind == "rectilinear" else []
                choice = fit_furniture_shape(evidences[id(entity)], theta, upm, label, shape_settings, extra)
                # A notched or round outline replaces the rectangle when the evidence supports it; a rectilinear
                # trace that does not pay for its vertices becomes the best rectangle; a fitted rectangle stays.
                adopt = choice is not None and (choice.kind in {"L", "U", "circle", "ellipse"} or (kind == "rectilinear" and choice.kind == "rect" and choice.iou >= shape_settings.min_rect_iou))
                if choice is not None:
                    stats["furniture_shapes"].append({
                        "label": str(entity.get("_label")), "instance": entity.get("_instance_id"), "kind": choice.kind if adopt else kind,
                        "iou": round(choice.iou, 3), "boundary_m": round(choice.boundary_m, 3), "legacy": kind,
                        "scores": choice.scores,
                    })
                if adopt:
                    regular, kind = choice.polygon, choice.kind  # type: ignore[union-attr]
            stats["furniture_regularised"] += 1
        clipped = clip_to_interior(regular, rooms, walls, upm)
        if clipped is None:
            dropped_ids.add(entity.get("_furn_id"))
            stats["furniture_dropped_small"] += 1
            continue
        if clipped is not regular:
            stats["furniture_clipped"] += 1
            if kind in {"rect", "rectilinear", "L", "U"}:
                again = rectilinear_outline(clipped, theta, min(settings.furniture_step_m * upm, 0.05 * upm), resolution=0.01 * upm)
                clipped = _largest(again.intersection(clipped.buffer(0.005 * upm, join_style=2))) if again is not None else clipped
                clipped = clipped if clipped is not None else regular
        if clipped.area < settings.furniture_min_area_m2 * upm * upm:
            dropped_ids.add(entity.get("_furn_id"))
            stats["furniture_dropped_small"] += 1
            continue
        entity["points"] = _points(clipped)
        entity["holes"] = []
        entity["_footprint_kind"] = kind
        entity["_polished"] = True
        shapes[id(entity)] = clipped
        text = texts.get(entity.get("_furn_id")) if entity.get("_furn_id") is not None else None
        if text is not None:
            label = str(entity.get("_label") or str(text.get("text", "")).split(" h=")[0]).strip()
            height = _height_of(str(text.get("text", "")))
            new_label = plausible_label(label, clipped.area / (upm * upm), height)
            if new_label != label:
                entity["_label"] = new_label
                text["text"] = f"{new_label} h={height:.2f}" if height is not None else new_label
                stats["furniture_renamed"] += 1
        kept.append(entity)
    # 3. a smaller outline mostly inside a larger one (or inside a catalogue block) is a duplicate trace of the
    #    same object; a smaller outline that only reaches into a larger one is cut back.
    blocks = [shape for name in ("A-FURN", "P-SANI", "A-EINB") for entity in layers.get(name, []) if entity.get("type") == "insert" and footprint_of is not None and (shape := footprint_of(entity)) is not None]
    kept.sort(key=lambda item: -shapes[id(item)].area)
    final: list[dict[str, Any]] = []
    for entity in kept:
        shape = shapes[id(entity)]
        blockers = blocks + [shapes[id(other)] for other in final]
        covered = [shape.intersection(other).area for other in blockers if other.intersects(shape)]
        if any(area >= settings.furniture_nested_share * shape.area for area in covered) or sum(covered) >= 0.5 * shape.area:
            dropped_ids.add(entity.get("_furn_id"))
            stats["furniture_dropped_nested"] += 1
            continue
        if sum(covered) > 1e-9 * upm * upm:
            rest = _largest(shape.difference(unary_union([other for other in blockers if other.intersects(shape)])))
            if rest is None or rest.area < settings.furniture_min_area_m2 * upm * upm:
                dropped_ids.add(entity.get("_furn_id"))
                stats["furniture_dropped_nested"] += 1
                continue
            rest = rest.simplify(0.002 * upm, preserve_topology=True)
            shapes[id(entity)] = rest
            entity["points"] = _points(rest)
            stats["furniture_overlaps_cut"] += 1
        final.append(entity)
    # 3a. a thin sliver (narrower than furniture_sliver_m) glued to a much larger outline is a trace fragment of it
    if settings.furniture_shapes:
        for entity in list(final):
            shape = shapes[id(entity)]
            frame_rect = shape.minimum_rotated_rectangle
            corners = np.asarray(frame_rect.exterior.coords[:3], dtype=float) if isinstance(frame_rect, Polygon) else None
            if corners is None:
                continue
            narrow = min(float(np.hypot(*(corners[1] - corners[0]))), float(np.hypot(*(corners[2] - corners[1]))))
            if narrow >= settings.furniture_sliver_m * upm:
                continue
            if any(other is not entity and shapes[id(other)].area >= 4.0 * shape.area and shapes[id(other)].distance(shape) <= 0.03 * upm for other in final):
                final.remove(entity)
                dropped_ids.add(entity.get("_furn_id"))
                stats["furniture_dropped_nested"] += 1
    final_ids = {id(entity) for entity in final}
    layers["A-FURN"] = [entity for entity in furniture if not _closed_polyline(entity) or id(entity) in final_ids]
    # 3b. sofas: backrest / armrest bands from the height profile, corner modules of L / U sofas
    shape_details: list[dict[str, Any]] = []
    detailed: list[Polygon] = []
    if settings.furniture_shapes and height_of is not None:
        for entity in final:
            if not is_sofa(entity.get("_semantic_label") or entity.get("_label")) or entity.get("_footprint_kind") in {"circle", "ellipse", "rect_free"}:
                continue
            detail = sofa_details(shapes[id(entity)], theta, upm, height_of, shape_settings)
            if not detail.lines:
                continue
            stats["furniture_sofa_details"] += 1
            detailed.append(shapes[id(entity)])
            if detail.seat is not None:
                entity["_label_area"] = _points(detail.seat)
            shape_details.extend({"type": "polyline", "points": [[float(x), float(y)] for x, y in line.coords], "closed": False, "_shape_detail": True, "_furn_id": entity.get("_furn_id")} for line in detail.lines)
    dropped_ids.discard(None)
    if dropped_ids:
        layers["A-ANNO"] = [entity for entity in layers.get("A-ANNO", []) if entity.get("_furn_id") not in dropped_ids]
    # 4. height-step details: straight lines inside a kept outline only
    outlines = unary_union([shapes[id(entity)] for entity in final]) if final else GeometryCollection()
    details: list[dict[str, Any]] = []
    detailed_zone = unary_union(detailed).buffer(0.02 * upm) if detailed else None
    for entity in layers.get("A-FURN-DETL", []):
        if entity.get("type") != "polyline" or len(entity.get("points", [])) < 2:
            details.append(entity)
            continue
        if detailed_zone is not None and detailed_zone.intersects(LineString(entity["points"])):
            stats["furniture_details_dropped"] += 1  # the class-aware sofa detail replaces height-step traces
            continue
        line = LineString(entity["points"]).simplify(settings.furniture_detail_tolerance_m * upm, preserve_topology=False)
        (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
        off_axis = abs(((math.degrees(math.atan2(y1 - y0, x1 - x0) - math.degrees(theta)) + 45.0) % 90.0) - 45.0) > 10.0
        if len(line.coords) > 2 or off_axis or line.length < 0.10 * upm or outlines.is_empty or not outlines.buffer(0.02 * upm).covers(line):
            stats["furniture_details_dropped"] += 1
            continue
        entity["points"] = [[float(x), float(y)] for x, y in line.coords]
        details.append(entity)
    layers["A-FURN-DETL"] = details + shape_details
    return stats


# --------------------------------------------------------------------------
# kitchen
# --------------------------------------------------------------------------

def fit_kitchen_to_walls(
    layers: dict[str, list[dict[str, Any]]],
    rooms: list[Polygon],
    walls: Any,
    theta: float,
    units_per_metre: float,
    settings: PlanPolishSettings = PlanPolishSettings(),
) -> dict[str, Any]:
    """Move every kitchen-run leg onto the final inner wall face behind it and give all legs one depth.

    The run was built against the measured faces before the double-line walls (a cabinet front can stand
    there).  A leg with a wall within ``kitchen_wall_reach_m`` behind its back edge is moved flush onto that
    wall face, its ends extend to a wall within ``kitchen_end_reach_m``; all legs get the run depth (the
    standard ``kitchen_depth_m`` when the measured depths are within ``kitchen_depth_tol_m`` of it, else their
    median).  A peninsula leg (no wall behind it) keeps its position.  The run never overlaps a wall body and
    stays inside its room; loose built-in segments that only retrace the run are dropped.  Upper cabinets on
    A-OBEN move flush onto the same walls.
    """
    upm = max(float(units_per_metre), 1e-9)
    stats: dict[str, Any] = {"kitchen_legs_snapped": 0, "kitchen_depth_m": None, "kitchen_segments_dropped": 0, "kitchen_upper_snapped": 0}
    if walls is None or walls.is_empty:
        return stats
    entities = layers.get("A-EINB", [])
    runs = [entity for entity in entities if _closed_polyline(entity) and entity.get("_run_legs")]
    run_shapes: list[Polygon] = []
    for entity in runs:
        legs = [_LegBox.of(Polygon(points), theta) for points in entity["_run_legs"] if len(points) >= 3]
        legs = [leg for leg in legs if leg is not None]
        if not legs:
            continue
        local_walls = affinity.rotate(walls, -theta, origin=(0.0, 0.0), use_radians=True)
        originals = [(leg.u0, leg.u1, leg.v0, leg.v1, leg.horizontal) for leg in legs]
        chamfers = _chamfer_corners_of(Polygon(entity["points"]).buffer(0), theta, upm)
        measured: list[float] = []
        for leg in legs:
            if leg.snap_back(local_walls, settings.kitchen_wall_reach_m * upm):
                stats["kitchen_legs_snapped"] += 1
                measured.append(leg.depth)
        depth = settings.kitchen_depth_m * upm
        if measured:
            median = float(np.median(measured))
            if abs(median - depth) > settings.kitchen_depth_tol_m * upm:
                depth = median
        stats["kitchen_depth_m"] = round(depth / upm, 3)
        for leg in legs:
            if leg.back is not None:
                leg.set_depth(depth)
        for leg in legs:
            if leg.back is None:
                leg.set_depth(min(depth, leg.depth) if leg.depth > 0 else depth, centred=True)
        _join_legs(legs, originals, 0.05 * upm)
        for leg in legs:
            leg.snap_ends(local_walls, settings.kitchen_end_reach_m * upm)
        union = unary_union([leg.polygon(theta) for leg in legs])
        union = union.difference(walls)
        room = _room_of(union, rooms) if rooms else None
        if room is not None:
            union = union.intersection(room.buffer(1e-6 * upm, join_style=2))
        shape = _largest(union.buffer(0.002 * upm, join_style=2).buffer(-0.002 * upm, join_style=2))
        if shape is None:
            continue
        shape = shape.simplify(0.002 * upm, preserve_topology=True)
        # A diagonal inner-corner unit (chamfer) measured on the run is kept at the moved corner.
        kept_chamfers: list[Polygon] = []
        for corner, size in chamfers:
            triangle = _chamfer_at(shape, corner, size, upm)
            if triangle is None:
                continue
            triangle = triangle.difference(walls)
            if room is not None:
                triangle = triangle.intersection(room.buffer(1e-6 * upm, join_style=2))
            grown = _largest(unary_union([shape, triangle]).buffer(0.002 * upm, join_style=2).buffer(-0.002 * upm, join_style=2))
            if grown is not None and isinstance(triangle, Polygon) and triangle.area > 0.0:
                shape = grown.simplify(0.002 * upm, preserve_topology=True)
                kept_chamfers.append(triangle)
        entity["points"] = _points(shape)
        entity["_run_legs"] = [_points(leg.polygon(theta)) for leg in legs]
        entity["_chamfers"] = [_points(triangle) for triangle in kept_chamfers]
        entity["_kitchen_depth_m"] = depth / upm
        run_shapes.append(shape)
        for text in entities:
            if text.get("type") == "text" and str(text.get("text", "")).split(" h=")[0].strip() == "Küche" and shape.buffer(0.3 * upm).contains(Point(text["point"][:2])):
                text["point"] = [float(value) for value in shape.representative_point().coords[0]]
    if run_shapes:
        counter = unary_union(run_shapes)
        zone = counter.buffer(settings.kitchen_segment_reach_m * upm)
        wall_band = walls.buffer(0.15 * upm, join_style=2)
        kept: list[dict[str, Any]] = []
        for entity in layers.get("A-EINB", []):
            if entity.get("type") == "polyline" and not entity.get("closed") and len(entity.get("points", [])) >= 2 and not entity.get("_tall_unit_cross"):
                line = LineString(entity["points"])
                # a loose front line that retraces the run, or a line that only doubles a wall face
                if line.intersection(zone).length >= 0.5 * line.length or wall_band.covers(line):
                    stats["kitchen_segments_dropped"] += 1
                    continue
            kept.append(entity)
        layers["A-EINB"] = kept
        stats["kitchen_furniture_absorbed"] = _furniture_off_counter(layers, counter, theta, upm)
        local_walls = affinity.rotate(walls, -theta, origin=(0.0, 0.0), use_radians=True)
        for entity in layers.get("A-OBEN", []):
            if not _closed_polyline(entity) or not entity.get("_upper_cabinet"):
                continue
            leg = _LegBox.of(Polygon(entity["points"]), theta)
            if leg is None:
                continue
            thickness = leg.depth
            if leg.snap_back(local_walls, settings.kitchen_wall_reach_m * upm):
                leg.set_depth(thickness)
                shape = _largest(leg.polygon(theta).difference(walls))
                # Cutting the wall bodies out of a unit that was pushed into them can leave a sliver: keep the unit as detected then.
                if shape is not None and shape.area >= 0.5 * Polygon(entity["points"]).area:
                    entity["points"] = _points(shape)
                    stats["kitchen_upper_snapped"] += 1
    return stats


def _chamfer_corners_of(polygon: Polygon, theta: float, upm: float) -> list[tuple[tuple[float, float], float]]:
    """Inner corners (local point, leg length) of the diagonal chamfer edges of a run outline."""
    output: list[tuple[tuple[float, float], float]] = []
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return output
    cos, sin = math.cos(theta), math.sin(theta)
    coords = list(polygon.exterior.coords)
    grown = polygon.buffer(1e-3 * upm)
    for a, b in zip(coords[:-1], coords[1:]):
        vector = np.asarray(b) - np.asarray(a)
        length = float(np.hypot(*vector))
        deviation = abs(((math.degrees(math.atan2(vector[1], vector[0])) - math.degrees(theta) + 45.0) % 90.0) - 45.0)
        if length < 0.15 * upm or deviation < 20.0:
            continue
        ua, va = a[0] * cos + a[1] * sin, -a[0] * sin + a[1] * cos
        ub, vb = b[0] * cos + b[1] * sin, -b[0] * sin + b[1] * cos
        for cu, cv in ((ua, vb), (ub, va)):
            point = (cu * cos - cv * sin, cu * sin + cv * cos)
            if grown.covers(Point(point)):
                output.append((point, 0.5 * (abs(ub - ua) + abs(vb - va))))
                break
    return output


def _chamfer_at(shape: Polygon, corner: tuple[float, float], size: float, upm: float) -> Polygon | None:
    """Chamfer triangle of leg length ``size`` at the reflex vertex of ``shape`` nearest to ``corner``."""
    ring = list(orient(shape, 1.0).exterior.coords)[:-1]
    best: tuple[float, int] | None = None
    for index, vertex in enumerate(ring):
        a, v, b = np.asarray(ring[index - 1]), np.asarray(vertex), np.asarray(ring[(index + 1) % len(ring)])
        first, second = v - a, b - v
        if float(first[0] * second[1] - first[1] * second[0]) >= 0.0:
            continue  # convex
        distance = float(np.hypot(v[0] - corner[0], v[1] - corner[1]))
        if distance <= 0.6 * upm and (best is None or distance < best[0]):
            best = (distance, index)
    if best is None:
        return None
    index = best[1]
    v = np.asarray(ring[index])
    e1 = np.asarray(ring[index - 1]) - v
    e2 = np.asarray(ring[(index + 1) % len(ring)]) - v
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    if l1 <= 1e-9 or l2 <= 1e-9:
        return None
    c = min(size, 0.8 * l1, 0.8 * l2)
    if c < 0.10 * upm:
        return None
    return Polygon([tuple(v), tuple(v + c * e1 / l1), tuple(v + c * e2 / l2)])


def _furniture_off_counter(layers: dict[str, list[dict[str, Any]]], counter: Any, theta: float, upm: float) -> int:
    """Traced furniture standing mostly on the counter is part of it; furniture reaching into it is cut back."""
    removed_ids: set[Any] = set()
    output: list[dict[str, Any]] = []
    changed = 0
    for entity in layers.get("A-FURN", []):
        if not _closed_polyline(entity):
            output.append(entity)
            continue
        shape = _largest(Polygon(entity["points"]).buffer(0))
        overlap = shape.intersection(counter).area if shape is not None else 0.0
        if shape is None or overlap <= 1e-9 * upm * upm:
            output.append(entity)
            continue
        changed += 1
        rest = _largest(shape.difference(counter)) if overlap < 0.5 * shape.area else None
        if rest is None or rest.area < 0.10 * upm * upm:
            removed_ids.add(entity.get("_furn_id"))
            continue
        entity["points"] = _points(rest.simplify(0.002 * upm, preserve_topology=True))
        output.append(entity)
    layers["A-FURN"] = output
    removed_ids.discard(None)
    if removed_ids:
        layers["A-ANNO"] = [entity for entity in layers.get("A-ANNO", []) if entity.get("_furn_id") not in removed_ids]
    return changed


def _join_legs(legs: list["_LegBox"], originals: list[tuple[float, float, float, float, bool]], tolerance: float) -> None:
    """Legs that met before the depth change meet again: an end at a perpendicular leg runs through it."""
    for index, leg in enumerate(legs):
        u0, u1, v0, v1, horizontal = originals[index]
        for other_index, other in enumerate(legs):
            ou0, ou1, ov0, ov1, other_horizontal = originals[other_index]
            if other_index == index or other_horizontal == horizontal:
                continue
            if horizontal:
                if min(v1, ov1) - max(v0, ov0) < -tolerance:
                    continue
                if ou0 - tolerance <= u0 <= ou1 + tolerance:
                    leg.u0 = min(leg.u0, other.u0)
                if ou0 - tolerance <= u1 <= ou1 + tolerance:
                    leg.u1 = max(leg.u1, other.u1)
            else:
                if min(u1, ou1) - max(u0, ou0) < -tolerance:
                    continue
                if ov0 - tolerance <= v0 <= ov1 + tolerance:
                    leg.v0 = min(leg.v0, other.v0)
                if ov0 - tolerance <= v1 <= ov1 + tolerance:
                    leg.v1 = max(leg.v1, other.v1)


@dataclass(slots=True)
class _LegBox:
    """A kitchen leg as an axis-aligned box in the wall frame (rotated by -theta about the origin)."""

    u0: float
    u1: float
    v0: float
    v1: float
    back: str | None = None

    @classmethod
    def of(cls, polygon: Polygon, theta: float) -> "_LegBox | None":
        if polygon.is_empty or polygon.area <= 0.0:
            return None
        local = affinity.rotate(polygon, -theta, origin=(0.0, 0.0), use_radians=True)
        u0, v0, u1, v1 = local.bounds
        return cls(u0, u1, v0, v1)

    @property
    def horizontal(self) -> bool:
        return (self.u1 - self.u0) >= (self.v1 - self.v0)

    @property
    def depth(self) -> float:
        return (self.v1 - self.v0) if self.horizontal else (self.u1 - self.u0)

    def polygon(self, theta: float) -> Polygon:
        return affinity.rotate(box(self.u0, self.v0, self.u1, self.v1), theta, origin=(0.0, 0.0), use_radians=True)

    def _across(self) -> tuple[str, str]:
        return ("v0", "v1") if self.horizontal else ("u0", "u1")

    def snap_back(self, walls: Any, reach: float) -> bool:
        """Move the long edge nearest to a wall (within ``reach`` outside or inside the leg) onto the wall face."""
        lo_name, hi_name = self._across()
        along = (self.u0, self.u1) if self.horizontal else (self.v0, self.v1)
        margin = 0.15 * (along[1] - along[0])
        a0, a1 = along[0] + margin, along[1] - margin
        best: tuple[float, str, float] | None = None
        for name, outward in ((lo_name, -1.0), (hi_name, 1.0)):
            edge = getattr(self, name)
            far = edge + outward * reach
            near = edge - outward * min(reach, 0.5 * self.depth)
            low, high = min(far, near), max(far, near)
            strip = box(a0, low, a1, high) if self.horizontal else box(low, a0, high, a1)
            hit = walls.intersection(strip)
            if hit.is_empty or hit.area <= 1e-12:
                continue
            bounds = hit.bounds
            face = (bounds[3] if outward < 0 else bounds[1]) if self.horizontal else (bounds[2] if outward < 0 else bounds[0])
            # the wall must cover the strip across the leg (not a column touching one end)
            covered = bounds[2] - bounds[0] if self.horizontal else bounds[3] - bounds[1]
            if covered < 0.5 * (a1 - a0):
                continue
            gap = abs(face - edge)
            if best is None or gap < best[0]:
                best = (gap, name, face)
        if best is None:
            return False
        _, name, face = best
        setattr(self, name, face)
        self.back = name
        return True

    def set_depth(self, depth: float, *, centred: bool = False) -> None:
        lo_name, hi_name = self._across()
        low, high = getattr(self, lo_name), getattr(self, hi_name)
        if self.back == lo_name:
            setattr(self, hi_name, low + depth)
        elif self.back == hi_name:
            setattr(self, lo_name, high - depth)
        elif centred:
            middle = (low + high) / 2.0
            setattr(self, lo_name, middle - depth / 2.0)
            setattr(self, hi_name, middle + depth / 2.0)

    def snap_ends(self, walls: Any, reach: float) -> None:
        """Extend each end of the leg to a perpendicular wall face within ``reach``."""
        if self.horizontal:
            names, across = (("u0", -1.0), ("u1", 1.0)), (self.v0 + 0.1 * (self.v1 - self.v0), self.v1 - 0.1 * (self.v1 - self.v0))
        else:
            names, across = (("v0", -1.0), ("v1", 1.0)), (self.u0 + 0.1 * (self.u1 - self.u0), self.u1 - 0.1 * (self.u1 - self.u0))
        for name, outward in names:
            edge = getattr(self, name)
            low, high = (edge - reach, edge) if outward < 0 else (edge, edge + reach)
            strip = box(low, across[0], high, across[1]) if self.horizontal else box(across[0], low, across[1], high)
            hit = walls.intersection(strip)
            if hit.is_empty or hit.area <= 1e-12:
                continue
            bounds = hit.bounds
            span = (bounds[3] - bounds[1]) if self.horizontal else (bounds[2] - bounds[0])
            if span < 0.8 * (across[1] - across[0]):
                continue
            face = (bounds[2] if outward < 0 else bounds[0]) if self.horizontal else (bounds[3] if outward < 0 else bounds[1])
            setattr(self, name, face)


# --------------------------------------------------------------------------
# object labels
# --------------------------------------------------------------------------

def _text_box(text: str, height: float, point: tuple[float, float], rotation: float, char_width: float) -> Polygon:
    width = len(text) * char_width * height
    raw = box(point[0] - width / 2.0, point[1] - height / 2.0, point[0] + width / 2.0, point[1] + height / 2.0)
    return affinity.rotate(raw, rotation, origin=point) if abs(rotation) > 1e-12 else raw


def place_object_labels(
    layers: dict[str, list[dict[str, Any]]],
    rooms: list[Polygon],
    units_per_metre: float,
    theta: float,
    footprint_of: Callable[[dict[str, Any]], Polygon | None],
    settings: PlanPolishSettings = PlanPolishSettings(),
) -> dict[str, int]:
    """Put every furniture label (A-ANNO text linked by ``_furn_id``) inside its object, else beside it inside
    the room, never across a room outline (= under a wall).  Labels shrink down to ``label_min_height_m``;
    labels of objects below ``label_min_object_area_m2`` and labels without any free place are dropped."""
    upm = max(float(units_per_metre), 1e-9)
    stats = {"labels_inside_object": 0, "labels_beside_object": 0, "labels_dropped_small": 0, "labels_dropped_no_space": 0, "labels_orphaned": 0, "details_dropped_at_labels": 0}
    owners: dict[Any, Polygon] = {}
    for name in ("A-FURN", "P-SANI"):
        for entity in layers.get(name, []):
            if entity.get("_furn_id") is None:
                continue
            shape = _largest(Polygon(entity["points"]).buffer(0)) if _closed_polyline(entity) else footprint_of(entity) if entity.get("type") == "insert" else None
            if shape is not None:
                owners[entity["_furn_id"]] = shape
    obstacles = list(owners.values())
    builtins = [shape for entity in layers.get("A-EINB", []) if _closed_polyline(entity) and (shape := _largest(Polygon(entity["points"]).buffer(0))) is not None]
    placed: list[Polygon] = [
        _text_box(str(entity.get("text", "")), float(entity["height"]), (float(entity["point"][0]), float(entity["point"][1])), float(entity.get("rotation", 0.0)), settings.label_char_width)
        for name, entities in layers.items() for entity in entities
        if entity.get("type") == "text" and entity.get("_furn_id") is None and "point" in entity and float(entity.get("height", 0.0)) > 0.0 and name in {"A-ANNO", "A-DOOR", "A-EINB"}
    ]
    minimum_height = settings.label_min_height_m * upm
    axis_deg = math.degrees(theta)
    output: list[dict[str, Any]] = []
    labels = [entity for entity in layers.get("A-ANNO", []) if entity.get("type") == "text" and entity.get("_furn_id") is not None]
    labels.sort(key=lambda entity: -(owners[entity["_furn_id"]].area if entity["_furn_id"] in owners else 0.0))
    decided: dict[int, bool] = {}
    for entity in labels:
        shape = owners.get(entity["_furn_id"])
        if shape is None:
            stats["labels_orphaned"] += 1
            decided[id(entity)] = False
            continue
        if shape.area < settings.label_min_object_area_m2 * upm * upm:
            stats["labels_dropped_small"] += 1
            decided[id(entity)] = False
            continue
        text = str(entity.get("text", ""))
        start = float(entity.get("height", 0.08 * upm))
        room = _room_of(shape, rooms)
        owner = next((item for item in layers.get("A-FURN", []) if item.get("_furn_id") == entity["_furn_id"] and item.get("_label_area")), None)
        result = None
        if owner is not None:
            # a sofa label sits on the seat, clear of the backrest and corner-module lines
            result = _fit_inside(Polygon(owner["_label_area"]).buffer(-settings.label_margin_m * upm, join_style=2), text, start, minimum_height, axis_deg, placed, settings.label_char_width)
        if result is None:
            # inside the whole object the label also keeps clear of the object's own detail lines (sofa seat, steps)
            details = [LineString(item["points"]).buffer(settings.label_margin_m * upm, join_style=2) for item in layers.get("A-FURN-DETL", [])
                       if item.get("type") == "polyline" and len(item.get("points", [])) >= 2 and LineString(item["points"]).intersects(shape)]
            inner = shape.buffer(-settings.label_margin_m * upm, join_style=2)
            result = _fit_inside(inner.difference(unary_union(details)) if details else inner, text, start, minimum_height, axis_deg, placed, settings.label_char_width)
        if result is not None:
            stats["labels_inside_object"] += 1
        else:
            others = [item for item in obstacles if item is not shape]
            result = _fit_beside(shape, room, text, start, minimum_height, axis_deg, placed + others + builtins, settings.label_char_width, upm)
            if result is not None:
                stats["labels_beside_object"] += 1
        if result is None:
            stats["labels_dropped_no_space"] += 1
            decided[id(entity)] = False
            continue
        point, height, rotation = result
        entity["point"] = [float(point[0]), float(point[1])]
        entity["height"] = float(height)
        entity["rotation"] = float(rotation)
        entity["_label_placed"] = True
        placed.append(_text_box(text, height, point, rotation, settings.label_char_width))
        decided[id(entity)] = True
    for entity in layers.get("A-ANNO", []):
        if decided.get(id(entity), True):
            output.append(entity)
    layers["A-ANNO"] = output
    # A height-step detail line never runs through a label.
    label_boxes = [_text_box(str(entity.get("text", "")), float(entity["height"]), (float(entity["point"][0]), float(entity["point"][1])), float(entity.get("rotation", 0.0)), settings.label_char_width) for entity in output if entity.get("_label_placed")]
    if label_boxes:
        before = len(layers.get("A-FURN-DETL", []))
        layers["A-FURN-DETL"] = [
            entity for entity in layers.get("A-FURN-DETL", [])
            if entity.get("_shape_detail") or not (entity.get("type") == "polyline" and len(entity.get("points", [])) >= 2 and any(LineString(entity["points"]).intersects(item) for item in label_boxes))
        ]
        stats["details_dropped_at_labels"] = before - len(layers["A-FURN-DETL"])
    return stats


def _text_rotation(axis_deg: float) -> float:
    """Reading direction along the wall axis nearest to horizontal (0 for an axis within 0.5 degrees)."""
    base = ((axis_deg + 45.0) % 90.0) - 45.0
    return 0.0 if abs(base) < 0.5 else base


def _fit_inside(area: Any, text: str, start: float, minimum: float, axis_deg: float, placed: list[Polygon], char_width: float) -> tuple[tuple[float, float], float, float] | None:
    shape = _largest(area)
    if shape is None:
        return None
    try:
        pole = polylabel(shape, tolerance=max(1e-6, 0.002 * math.sqrt(shape.area)))
        anchors = [(float(pole.x), float(pole.y))]
    except GEOSException:
        anchors = []
    centre = shape.centroid
    anchors.append((float(centre.x), float(centre.y)))
    base = _text_rotation(axis_deg)
    rotations = [base, base + 90.0]
    height = start
    while height >= minimum - 1e-9:
        for rotation in rotations:
            for point in anchors:
                candidate = _text_box(text, height, point, rotation, char_width)
                if shape.covers(candidate) and not any(candidate.intersects(other) and candidate.intersection(other).area > 1e-9 for other in placed):
                    return point, height, rotation
        height *= 0.85
    return None


def _fit_beside(shape: Polygon, room: Polygon | None, text: str, start: float, minimum: float, axis_deg: float, blockers: list[Polygon], char_width: float, upm: float) -> tuple[tuple[float, float], float, float] | None:
    if room is None:
        return None
    rotation = _text_rotation(axis_deg)
    centre = shape.centroid
    height = start
    while height >= minimum - 1e-9:
        width = len(text) * char_width * height
        best: tuple[float, tuple[float, float]] | None = None
        for distance in (0.6 * height + 0.02 * upm, 1.2 * height + 0.02 * upm, 0.5 * width + 0.05 * upm, 2.5 * height + 0.05 * upm):
            ring = shape.buffer(distance, join_style=2).exterior
            count = max(16, int(ring.length / max(0.5 * height, 1e-9)))
            for k in range(count):
                point = ring.interpolate(k / count, normalized=True)
                anchor = (float(point.x), float(point.y))
                candidate = _text_box(text, height, anchor, rotation, char_width)
                if not room.covers(candidate) or candidate.intersection(shape).area > 1e-9:
                    continue
                if any(candidate.intersects(other) and candidate.intersection(other).area > 1e-9 for other in blockers):
                    continue
                key = math.hypot(anchor[0] - centre.x, anchor[1] - centre.y)
                if best is None or key < best[0]:
                    best = (key, anchor)
            if best is not None:
                return best[1], height, rotation
        height *= 0.85
    return None


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def _parts(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        output: list[Polygon] = []
        for member in geometry.geoms:
            output.extend(_parts(member))
        return output
    return []


def _largest(geometry: Any) -> Polygon | None:
    parts = [part for part in _parts(geometry) if part.area > 0.0]
    return max(parts, key=lambda part: part.area) if parts else None
