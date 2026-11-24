"""Shapes of a reconstructed furniture item: vertical prisms (box / cylinder / extrusion) and closed triangle meshes.

Fitting works in a metric *item frame*: origin at the footprint centre, x along the item's long axis, z in metres above
the storey floor.  :class:`ItemFrame` converts to and from the plan-local frame of ``plan_model.json`` (local units,
absolute local z).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

CYLINDER_SEGMENTS = 24


def robust_union(geoms: Any) -> Any:
    """``unary_union`` that survives GEOS non-noded intersections of touching / collinear outlines."""
    import shapely
    from shapely.errors import GEOSException
    from shapely.validation import make_valid

    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return Polygon()
    try:
        return unary_union(geoms)
    except GEOSException:
        pass
    try:
        return unary_union([shapely.set_precision(g, 1e-6) for g in geoms])
    except GEOSException:
        pass
    try:
        return unary_union([make_valid(g) for g in geoms])
    except GEOSException:
        return unary_union([g.buffer(0) for g in geoms])


@dataclass(frozen=True, slots=True)
class ItemFrame:
    """Plan-local <-> item metric frame."""

    origin: tuple[float, float]  # local units
    angle: float  # radians: item x axis direction in the plan frame
    upm: float  # local units per metre
    floor_z: float  # local units

    def to_item(self, local_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(local_xyz, dtype=float)
        c, s = math.cos(self.angle), math.sin(self.angle)
        dx = (pts[:, 0] - self.origin[0]) / self.upm
        dy = (pts[:, 1] - self.origin[1]) / self.upm
        out = np.empty((len(pts), 3), dtype=float)
        out[:, 0] = c * dx + s * dy
        out[:, 1] = -s * dx + c * dy
        out[:, 2] = (pts[:, 2] - self.floor_z) / self.upm if pts.shape[1] > 2 else 0.0
        return out

    def xy_to_local(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=float).reshape(-1, 2)
        c, s = math.cos(self.angle), math.sin(self.angle)
        x = (c * pts[:, 0] - s * pts[:, 1]) * self.upm + self.origin[0]
        y = (s * pts[:, 0] + c * pts[:, 1]) * self.upm + self.origin[1]
        return np.column_stack((x, y))

    def z_to_local(self, z: float) -> float:
        return float(self.floor_z + float(z) * self.upm)

    def polygon_to_item(self, polygon: Polygon) -> Polygon:
        ring = np.asarray(polygon.exterior.coords, dtype=float)
        local = self.to_item(np.column_stack((ring, np.zeros(len(ring)))))
        return Polygon(local[:, :2])


def frame_for_footprint(footprint: Polygon, upm: float, floor_z: float) -> ItemFrame:
    """Item frame of a footprint: centre of its minimum rotated rectangle, x along the rectangle's long side."""
    rectangle = footprint.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        centre = footprint.centroid
        return ItemFrame((float(centre.x), float(centre.y)), 0.0, upm, floor_z)
    corners = np.asarray(rectangle.exterior.coords, dtype=float)[:4]
    first, second = corners[1] - corners[0], corners[2] - corners[1]
    edge = first if np.hypot(*first) >= np.hypot(*second) else second
    angle = math.atan2(edge[1], edge[0])
    # canonical: angle in (-pi/2, pi/2]
    if angle <= -math.pi / 2:
        angle += math.pi
    elif angle > math.pi / 2:
        angle -= math.pi
    centre = corners.mean(axis=0)
    return ItemFrame((float(centre[0]), float(centre[1])), float(angle), float(upm), float(floor_z))


@dataclass(slots=True)
class Solid:
    """One vertical prism in the item frame (metres). ``kind`` box | cylinder | extrusion."""

    kind: str
    role: str
    z0: float
    z1: float
    source: str = "measured"  # measured | symmetry | prior
    # box: centre, size, yaw (radians in the item frame); cylinder: centre + radius; extrusion: ring
    cx: float = 0.0
    cy: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    yaw: float = 0.0
    radius: float = 0.0
    ring: list[tuple[float, float]] = field(default_factory=list)

    def outline(self) -> Polygon:
        if self.kind == "cylinder":
            angles = np.linspace(0.0, 2.0 * math.pi, CYLINDER_SEGMENTS, endpoint=False)
            return Polygon(np.column_stack((self.cx + self.radius * np.cos(angles), self.cy + self.radius * np.sin(angles))))
        if self.kind == "box":
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            hx, hy = self.sx / 2.0, self.sy / 2.0
            corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
            return Polygon([(self.cx + c * x - s * y, self.cy + s * x + c * y) for x, y in corners])
        return Polygon(self.ring)

    @property
    def height(self) -> float:
        return float(self.z1 - self.z0)

    def volume(self) -> float:
        return float(self.outline().area * max(self.height, 0.0))


def box(role: str, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, source: str = "measured", yaw: float = 0.0) -> Solid:
    return Solid("box", role, float(z0), float(z1), source, cx=(x0 + x1) / 2.0, cy=(y0 + y1) / 2.0, sx=abs(x1 - x0), sy=abs(y1 - y0), yaw=yaw)


def cylinder(role: str, cx: float, cy: float, radius: float, z0: float, z1: float, source: str = "measured") -> Solid:
    return Solid("cylinder", role, float(z0), float(z1), source, cx=float(cx), cy=float(cy), radius=float(radius))


def extrusion(role: str, polygon: Polygon, z0: float, z1: float, source: str = "measured") -> Solid:
    ring = [(float(x), float(y)) for x, y in list(polygon.exterior.coords)[:-1]]
    return Solid("extrusion", role, float(z0), float(z1), source, ring=ring)


@dataclass(slots=True)
class Mesh:
    """Closed triangle mesh in the item frame (metres), outward normals."""

    vertices: np.ndarray  # (n, 3)
    faces: np.ndarray  # (m, 3) int

    def volume(self) -> float:
        v = self.vertices[self.faces]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)

    def outline(self) -> Polygon:
        return _mesh_outline(self)


def _mesh_outline(mesh: Mesh) -> Polygon:
    triangles = mesh.vertices[mesh.faces][:, :, :2]
    polygons = []
    for tri in triangles:
        polygon = Polygon(tri)
        if polygon.area > 1e-10:
            polygons.append(polygon)
    if not polygons:
        return MultiPoint([tuple(p) for p in mesh.vertices[:, :2]]).convex_hull
    union = robust_union(polygons).buffer(0.002).buffer(-0.002)
    if union.geom_type == "MultiPolygon":
        union = max(union.geoms, key=lambda item: item.area)
    return Polygon(union.exterior) if isinstance(union, Polygon) else MultiPoint([tuple(p) for p in mesh.vertices[:, :2]]).convex_hull


# ----------------------------------------------------------------------------------------------------------------------
# surface sampling (fit metrics)
# ----------------------------------------------------------------------------------------------------------------------

def _grid_inside(polygon: Polygon, spacing: float) -> np.ndarray:
    import shapely

    x0, y0, x1, y1 = polygon.bounds
    xs = np.arange(x0 + spacing / 2.0, x1, spacing)
    ys = np.arange(y0 + spacing / 2.0, y1, spacing)
    if not len(xs) or not len(ys):
        c = polygon.representative_point()
        return np.asarray([[c.x, c.y]])
    gx, gy = np.meshgrid(xs, ys)
    flat = np.column_stack((gx.ravel(), gy.ravel()))
    inside = shapely.contains_xy(polygon, flat[:, 0], flat[:, 1])
    pts = flat[inside]
    ring = np.asarray(polygon.exterior.coords, dtype=float)
    return np.vstack((pts, ring)) if len(pts) else ring


def _perimeter_points(polygon: Polygon, spacing: float) -> np.ndarray:
    ring = polygon.exterior
    count = max(int(math.ceil(ring.length / spacing)), 4)
    return np.asarray([ring.interpolate(i / count, normalized=True).coords[0] for i in range(count)], dtype=float)


def sample_solid(solid: Solid, spacing: float) -> np.ndarray:
    polygon = solid.outline()
    if polygon.is_empty or solid.height <= 0:
        return np.empty((0, 3))
    cap = _grid_inside(polygon, spacing)
    rim = _perimeter_points(polygon, spacing)
    levels = np.linspace(solid.z0, solid.z1, max(int(math.ceil(solid.height / spacing)) + 1, 2))
    side = np.column_stack((np.tile(rim, (len(levels), 1)), np.repeat(levels, len(rim))))
    return np.vstack((
        np.column_stack((cap, np.full(len(cap), solid.z0))),
        np.column_stack((cap, np.full(len(cap), solid.z1))),
        side,
    ))


def sample_mesh(mesh: Mesh, spacing: float) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    counts = np.maximum(np.ceil(areas / (spacing * spacing)).astype(int), 1)
    rng = np.random.default_rng(0)
    owner = np.repeat(np.arange(len(tri)), counts)
    r1, r2 = rng.random(len(owner)), rng.random(len(owner))
    flip = r1 + r2 > 1.0
    r1[flip], r2[flip] = 1.0 - r1[flip], 1.0 - r2[flip]
    t = tri[owner]
    pts = t[:, 0] + r1[:, None] * (t[:, 1] - t[:, 0]) + r2[:, None] * (t[:, 2] - t[:, 0])
    return np.vstack((pts, mesh.vertices))


def sample_shape(solids: list[Solid], mesh: Mesh | None, spacing: float = 0.01, budget: int = 250_000) -> np.ndarray:
    """Points on the surface of the shape (solids and/or mesh), spacing adapted to stay within ``budget``."""
    area = sum(2.0 * s.outline().area + s.outline().length * s.height for s in solids)
    if mesh is not None:
        tri = mesh.vertices[mesh.faces]
        area += float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    if area > 0 and area / (spacing * spacing) > budget:
        spacing = math.sqrt(area / budget)
    parts = [sample_solid(s, spacing) for s in solids]
    if mesh is not None:
        parts.append(sample_mesh(mesh, spacing))
    parts = [p for p in parts if len(p)]
    return np.vstack(parts) if parts else np.empty((0, 3))


def shape_volume(solids: list[Solid], mesh: Mesh | None, cell: float = 0.02) -> float:
    """Volume of the union of the parts (voxel estimate for overlapping solids, exact for one mesh)."""
    if mesh is not None and not solids:
        return abs(mesh.volume())
    if not solids:
        return 0.0
    if len(solids) == 1:
        return solids[0].volume()
    import shapely

    lows = [s.z0 for s in solids]
    highs = [s.z1 for s in solids]
    boxes = [s.outline().bounds for s in solids if not s.outline().is_empty]
    if not boxes:
        return float(sum(s.volume() for s in solids))
    bounds = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    xs = np.arange(bounds[0] + cell / 2, bounds[2], cell)
    ys = np.arange(bounds[1] + cell / 2, bounds[3], cell)
    zs = np.arange(min(lows) + cell / 2, max(highs), cell)
    if not len(xs) or not len(ys) or not len(zs):
        return float(sum(s.volume() for s in solids))
    gx, gy = np.meshgrid(xs, ys)
    flat = np.column_stack((gx.ravel(), gy.ravel()))
    occupied = np.zeros((len(flat), len(zs)), dtype=bool)
    for s in solids:
        inside = shapely.contains_xy(s.outline(), flat[:, 0], flat[:, 1])
        band = (zs >= s.z0) & (zs <= s.z1)
        occupied[np.ix_(inside, band)] = True
    return float(occupied.sum() * cell ** 3)


# ----------------------------------------------------------------------------------------------------------------------
# conversion to plan_model dictionaries (plan-local units)
# ----------------------------------------------------------------------------------------------------------------------

def _round(values: Any, digits: int = 4) -> Any:
    return [round(float(v), digits) for v in values]


def solid_to_plan(solid: Solid, frame: ItemFrame) -> dict[str, Any]:
    ring = frame.xy_to_local(np.asarray(solid.outline().exterior.coords, dtype=float)[:-1])
    out: dict[str, Any] = {
        "type": solid.kind, "role": solid.role, "part_source": solid.source,
        "z_min": round(frame.z_to_local(solid.z0), 4), "z_max": round(frame.z_to_local(solid.z1), 4),
        "outline": [_round(point) for point in ring],
    }
    centre = frame.xy_to_local(np.asarray([[solid.cx, solid.cy]]))[0]
    if solid.kind == "box":
        out["center"] = _round(centre)
        out["dims_m"] = _round((solid.sx, solid.sy, solid.height))
        out["rotation_deg"] = round(math.degrees(frame.angle + solid.yaw), 3)
    elif solid.kind == "cylinder":
        out["center"] = _round(centre)
        out["radius"] = round(solid.radius * frame.upm, 4)
        out["dims_m"] = _round((2.0 * solid.radius, 2.0 * solid.radius, solid.height))
    return out


def mesh_to_plan(mesh: Mesh, frame: ItemFrame) -> dict[str, Any]:
    xy = frame.xy_to_local(mesh.vertices[:, :2])
    z = frame.floor_z + mesh.vertices[:, 2] * frame.upm
    return {
        "vertices": [[round(float(x), 4), round(float(y), 4), round(float(h), 4)] for (x, y), h in zip(xy, z)],
        "faces": [[int(a), int(b), int(c)] for a, b, c in mesh.faces],
    }


def outline_to_plan(polygon: Polygon, frame: ItemFrame) -> list[list[float]]:
    ring = frame.xy_to_local(np.asarray(polygon.exterior.coords, dtype=float)[:-1])
    return [_round(point) for point in ring]


def projected_outline(solids: list[Solid], mesh: Mesh | None) -> Polygon | None:
    """Plan projection of the reconstructed shape (largest polygon, holes dropped, slightly simplified)."""
    pieces = [s.outline() for s in solids]
    if mesh is not None:
        pieces.append(mesh.outline())
    pieces = [p if p.is_valid else p.buffer(0) for p in pieces if not p.is_empty]
    if not pieces:
        return None
    union = robust_union(pieces).buffer(0.005).buffer(-0.005)
    if union.geom_type == "MultiPolygon":
        # disjoint legs under a top are inside the top's projection; separate pieces are joined by their hull
        union = union.convex_hull if union.convex_hull.area < 1.3 * sum(g.area for g in union.geoms) else max(union.geoms, key=lambda g: g.area)
    if not isinstance(union, Polygon) or union.is_empty:
        return None
    simple = Polygon(union.exterior).simplify(0.005, preserve_topology=True)
    return simple if isinstance(simple, Polygon) and simple.is_valid and not simple.is_empty else Polygon(union.exterior)
