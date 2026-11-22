"""Exact 2D/3D ground-truth derivations of a :class:`SceneSpec` (scene frame, metres)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from .spec import FurnitureSpec, OpeningSpec, Part, SceneSpec, WallSpec

EPS = 1e-9


def wall_rectangle(wall: WallSpec, extend: float | None = None) -> Polygon:
    """Wall body as a rectangle; the ends are extended by half the thickness so corners close."""
    ext = wall.thickness / 2.0 if extend is None else extend
    d = wall.direction()
    n = wall.normal()
    p0 = np.asarray(wall.p0, dtype=float) - d * ext
    p1 = np.asarray(wall.p1, dtype=float) + d * ext
    h = wall.thickness / 2.0
    return Polygon([tuple(p0 + n * h), tuple(p1 + n * h), tuple(p1 - n * h), tuple(p0 - n * h)])


def opening_rectangle(spec: SceneSpec, opening: OpeningSpec, margin: float = 0.06) -> Polygon:
    """Footprint of the opening cut through its host wall (width along the wall, depth through it)."""
    wall = spec.wall(opening.wall_id)
    d, n = wall.direction(), wall.normal()
    c = np.asarray(wall.p0, dtype=float) + d * opening.offset
    hw, hd = opening.width / 2.0, wall.thickness / 2.0 + margin
    return Polygon([tuple(c + d * hw + n * hd), tuple(c - d * hw + n * hd), tuple(c - d * hw - n * hd), tuple(c + d * hw - n * hd)])


def opening_centre(spec: SceneSpec, opening: OpeningSpec) -> np.ndarray:
    wall = spec.wall(opening.wall_id)
    return np.asarray(wall.p0, dtype=float) + wall.direction() * opening.offset


def wall_solid(spec: SceneSpec) -> Polygon | MultiPolygon:
    """Full-height 2D wall solid: wall rectangles plus projections (cuts and openings are z-limited)."""
    parts = [wall_rectangle(w) for w in spec.walls] + [Polygon(p.polygon) for p in spec.projections]
    return unary_union(parts).buffer(0)


def wall_bands(spec: SceneSpec, height: float) -> list[tuple[float, float, Any]]:
    """Return [(z0, z1, polygon)] slices of the wall solid; openings and niches are removed in their z range."""
    solid = wall_solid(spec)
    cuts: list[tuple[Polygon, float, float]] = []
    for opening in spec.openings:
        z0 = -0.05 if opening.kind in ("door", "passage") else opening.sill
        cuts.append((opening_rectangle(spec, opening), z0, opening.head))
    for cut in spec.cuts:
        cuts.append((Polygon(cut.polygon), cut.z0, cut.z1))
    levels = {0.0, float(height)}
    for _poly, z0, z1 in cuts:
        for z in (z0, z1):
            if 0.0 < z < height:
                levels.add(round(float(z), 6))
    zs = sorted(levels)
    bands = []
    for za, zb in zip(zs[:-1], zs[1:]):
        active = [poly for poly, z0, z1 in cuts if z0 <= za + 1e-6 and z1 >= zb - 1e-6]
        shape = solid.difference(unary_union(active)) if active else solid
        shape = shape.buffer(0)
        if not shape.is_empty:
            bands.append((za, zb, shape))
    return bands


def room_polygons(spec: SceneSpec) -> list[dict[str, Any]]:
    """Closed free regions between the walls; names/ceiling heights come from the room seeds."""
    solid = wall_solid(spec)
    minx, miny, maxx, maxy = solid.bounds
    frame = box(minx - 1.0, miny - 1.0, maxx + 1.0, maxy + 1.0)
    free = frame.difference(solid)
    polys = list(free.geoms) if isinstance(free, MultiPolygon) else [free]
    outer = frame.exterior
    rooms = []
    for poly in polys:
        if poly.area < 0.5 or poly.exterior.distance(outer) < 1e-6 or poly.intersects(outer):
            continue
        rooms.append(poly)
    result = []
    used: set[int] = set()
    for index, poly in enumerate(sorted(rooms, key=lambda p: (round(p.centroid.x, 3), round(p.centroid.y, 3)))):
        name, ceiling = f"Room {index + 1}", spec.storey_height
        for seed_index, seed in enumerate(spec.rooms):
            if seed_index not in used and poly.contains(Point(seed.point)):
                name, ceiling = seed.name, seed.ceiling_h
                used.add(seed_index)
                break
        result.append({"name": name, "polygon": poly, "ceiling_h": float(ceiling), "area": float(poly.area)})
    return result


def footprint_union(spec: SceneSpec) -> Polygon | MultiPolygon:
    rooms = [r["polygon"] for r in room_polygons(spec)]
    return unary_union(rooms + [wall_solid(spec)]).buffer(0)


def room_at(rooms: list[dict[str, Any]], xy: tuple[float, float]) -> dict[str, Any] | None:
    point = Point(xy)
    for room in rooms:
        if room["polygon"].buffer(0.05).contains(point):
            return room
    return None


# ---------------------------------------------------------------------------------------------------------------
# furniture
# ---------------------------------------------------------------------------------------------------------------

def item_transform(item: FurnitureSpec) -> np.ndarray:
    a = math.radians(item.yaw_deg)
    t = np.eye(4)
    t[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    t[0, 3], t[1, 3] = item.center
    return t


def part_footprint(part: Part) -> Polygon:
    cx, cy, _cz = part.center
    if part.shape == "box":
        dx, dy, _dz = part.size
        poly = box(-dx / 2, -dy / 2, dx / 2, dy / 2)
        poly = rotate(poly, part.yaw_deg, origin=(0, 0))
        return translate(poly, cx, cy)
    radius = float(part.size[0])
    return Point(cx, cy).buffer(radius, quad_segs=12)


def part_z(part: Part) -> tuple[float, float]:
    cz = part.center[2]
    if part.shape == "box":
        h = part.size[2]
    elif part.shape == "cyl":
        h = part.size[2]
    else:
        h = 2 * part.size[0]
    return cz - h / 2.0, cz + h / 2.0


def item_footprint(item: FurnitureSpec) -> Polygon:
    local = unary_union([part_footprint(p) for p in item.parts]).buffer(0)
    placed = rotate(local, item.yaw_deg, origin=(0, 0))
    placed = translate(placed, *item.center)
    if isinstance(placed, MultiPolygon):  # legs-only parts: use the hull of the pieces
        placed = placed.convex_hull
    return placed


def item_z(item: FurnitureSpec) -> tuple[float, float]:
    zs = [part_z(p) for p in item.parts]
    return min(z[0] for z in zs), max(z[1] for z in zs)


def oriented_box(item: FurnitureSpec) -> dict[str, Any]:
    poly = item_footprint(item)
    rect = poly.minimum_rotated_rectangle
    coords = np.asarray(rect.exterior.coords)[:4]
    edges = [float(np.linalg.norm(coords[(i + 1) % 4] - coords[i])) for i in range(4)]
    z0, z1 = item_z(item)
    return {"corners": coords.tolist(), "length": max(edges[0], edges[1]), "depth": min(edges[0], edges[1]), "z0": z0, "z1": z1}


# ---------------------------------------------------------------------------------------------------------------
# validation helpers used by tests and the generator
# ---------------------------------------------------------------------------------------------------------------

def door_hinge_and_leaf(spec: SceneSpec, opening: OpeningSpec) -> tuple[np.ndarray, np.ndarray, float]:
    """Hinge point (on the wall face of the swing side), closed-leaf direction and leaf length of a door."""
    wall = spec.wall(opening.wall_id)
    d, n = wall.direction(), wall.normal()
    c = opening_centre(spec, opening)
    face = c + n * opening.swing * wall.thickness / 2.0
    sign = -1.0 if opening.hinge == "left" else 1.0
    hinge = face + d * sign * opening.width / 2.0
    closed = -d * sign
    return hinge, closed, opening.width


def door_sweep(spec: SceneSpec, opening: OpeningSpec) -> Polygon:
    """Area swept by the leaf between closed and its open angle (quarter disc on the swing side)."""
    wall = spec.wall(opening.wall_id)
    hinge, closed, length = door_hinge_and_leaf(spec, opening)
    n = wall.normal() * opening.swing
    angle = math.radians(max(opening.leaf_angle_deg, 10.0))
    pts = [tuple(hinge)]
    for k in range(13):
        a = angle * k / 12.0
        v = closed * math.cos(a) + n * math.sin(a)
        pts.append(tuple(hinge + v * (length + 0.03)))
    return Polygon(pts).buffer(0.03)


def blockers(spec: SceneSpec) -> list[tuple[str, Polygon, tuple[float, float]]]:
    """Footprints with z range of everything that occupies floor space (furniture, columns, stairs, door sweeps)."""
    result: list[tuple[str, Polygon, tuple[float, float]]] = []
    for item in spec.furniture:
        result.append((item.id, item_footprint(item), item_z(item)))
    for column in spec.columns:
        result.append((column.id, column_footprint(column), (0.0, spec.storey_height)))
    for stair in spec.stairs:
        result.append((stair.id, stair_footprint(stair), (0.0, spec.storey_height)))
    for opening in spec.openings:
        if opening.kind == "door":
            result.append((opening.id + "-sweep", door_sweep(spec, opening), (0.0, opening.head)))
    return result


def collides(candidate: Polygon, z: tuple[float, float], others: list[tuple[str, Polygon, tuple[float, float]]], clearance: float = 0.0) -> bool:
    shape = candidate.buffer(clearance) if clearance > 0 else candidate
    for _name, poly, oz in others:
        if z[1] <= oz[0] + 1e-6 or oz[1] <= z[0] + 1e-6:
            continue
        if shape.intersection(poly).area > 1e-4:
            return True
    return False


def validate_scene(spec: SceneSpec) -> list[str]:
    """Return a list of problems (empty when the scene is physically consistent)."""
    problems: list[str] = []
    rooms = room_polygons(spec)
    if not rooms:
        problems.append("no closed room")
    solid = wall_solid(spec)
    free = unary_union([r["polygon"] for r in rooms]) if rooms else Polygon()
    items = blockers(spec)
    for item in spec.furniture:
        fp = item_footprint(item)
        if not free.buffer(0.02).contains(fp):
            problems.append(f"{item.id} ({item.label}) leaves the rooms")
    for name, poly, _z in items:
        if name.startswith("S") and not free.buffer(0.02).contains(poly):
            problems.append(f"{name} leaves the rooms")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a[0].endswith("-sweep") and b[0].endswith("-sweep"):
                continue
            if a[2][1] <= b[2][0] + 1e-6 or b[2][1] <= a[2][0] + 1e-6:
                continue  # stacked (upper cabinets above worktop)
            inter = a[1].intersection(b[1]).area
            if inter > 1e-4:
                problems.append(f"{a[0]} overlaps {b[0]} ({inter:.4f} m2)")
    for opening in spec.openings:
        wall = spec.wall(opening.wall_id)
        if opening.offset - opening.width / 2 < 0.05 or opening.offset + opening.width / 2 > wall.length - 0.05:
            problems.append(f"{opening.id} leaves wall {wall.id}")
        if opening.head <= opening.sill:
            problems.append(f"{opening.id} head below sill")
    del solid
    return problems


def column_footprint(column: Any) -> Polygon:
    if column.shape == "cyl":
        return Point(column.center).buffer(column.size[0], quad_segs=12)
    dx, dy = column.size
    poly = rotate(box(-dx / 2, -dy / 2, dx / 2, dy / 2), column.yaw_deg, origin=(0, 0))
    return translate(poly, *column.center)


def beam_footprint(beam: Any) -> Polygon:
    return LineString([beam.p0, beam.p1]).buffer(beam.width / 2.0, cap_style="flat")


def stair_footprint(stair: Any) -> Polygon:
    a = math.radians(stair.yaw_deg)
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-d[1], d[0]])
    s = np.asarray(stair.start, dtype=float)
    length = stair.steps * stair.run
    hw = stair.width / 2
    return Polygon([tuple(s + n * hw), tuple(s + d * length + n * hw), tuple(s + d * length - n * hw), tuple(s - n * hw)])


def polygon_to_list(poly: Any) -> dict[str, Any]:
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)
    return {"exterior": [list(map(float, c)) for c in list(poly.exterior.coords)[:-1]],
            "holes": [[list(map(float, c)) for c in list(r.coords)[:-1]] for r in poly.interiors]}


def polygon_from_list(data: Any) -> Polygon:
    if isinstance(data, dict):
        return Polygon(data.get("exterior", []), data.get("holes", []) or None).buffer(0)
    return Polygon(data).buffer(0)


def triangulate(poly: Any) -> list[np.ndarray]:
    """Triangles (3x2 arrays) of a polygon with holes via shapely's constrained Delaunay triangulation."""
    polys = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
    tris: list[np.ndarray] = []
    for item in polys:
        if item.is_empty or item.area < 1e-9:
            continue
        result = shapely.constrained_delaunay_triangles(item)
        for tri in result.geoms:
            coords = np.asarray(tri.exterior.coords)[:3]
            if abs(Polygon(coords).area) > 1e-12:
                tris.append(coords)
    return tris
