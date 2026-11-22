"""Triangle meshes of a scene (scene frame) with per-face element index, class and instance id."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPolygon, Polygon

from . import geometry as geo
from .spec import (
    CLASS_BEAM, CLASS_CEILING, CLASS_COLUMN, CLASS_DOOR, CLASS_FLOOR, CLASS_STAIRS, CLASS_WALL, CLASS_WINDOW, SceneSpec,
)

PALETTE = {
    "wall": (205, 203, 196), "floor": (150, 112, 74), "ceiling": (238, 238, 236), "door": (236, 232, 220),
    "window_frame": (250, 250, 250), "glass": (150, 190, 222), "column": (182, 182, 178), "beam": (192, 190, 186),
    "stair": (122, 92, 64),
}
LABEL_COLOURS = {
    "sofa": (70, 90, 140), "armchair": (140, 60, 60), "bed": (210, 200, 180), "nightstand": (120, 80, 50),
    "table": (140, 95, 55), "dining table": (130, 85, 50), "coffee table": (100, 70, 40), "desk": (160, 120, 80),
    "chair": (60, 60, 60), "wardrobe": (225, 220, 210), "shelf": (180, 140, 90), "sideboard": (90, 60, 40),
    "tv stand": (40, 40, 40), "plant": (60, 120, 50), "toilet": (250, 250, 250), "washbasin": (245, 245, 245),
    "bathtub": (248, 248, 248), "shower tray": (240, 240, 240), "kitchen counter": (200, 200, 205),
    "kitchen sink": (170, 170, 175), "stove": (20, 20, 20), "refrigerator": (230, 230, 230), "kitchen cabinet": (210, 210, 200),
    "television": (30, 30, 34), "monitor": (35, 35, 40), "spotlight": (255, 255, 250), "ceiling lamp": (255, 255, 250),
    "smoke detector": (215, 215, 210), "ventilation grille": (150, 152, 150), "ceiling fixture": (120, 118, 115),
}


@dataclass
class Element:
    id: str
    kind: str  # wall | floor | ceiling | door | window | glass | furniture | column | beam | stair
    class_idx: int
    instance_id: int
    vertices: np.ndarray
    faces: np.ndarray
    colour: tuple[int, int, int]
    label: str = ""
    transparent: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneMesh:
    vertices: np.ndarray  # (n,3) float64
    faces: np.ndarray  # (m,3) int32
    face_element: np.ndarray  # (m,) int32 index into elements
    elements: list[Element]

    def face_class(self) -> np.ndarray:
        return np.array([e.class_idx for e in self.elements], dtype=np.uint8)[self.face_element]

    def face_instance(self) -> np.ndarray:
        return np.array([e.instance_id for e in self.elements], dtype=np.uint32)[self.face_element]

    def face_colour(self) -> np.ndarray:
        return np.array([e.colour for e in self.elements], dtype=np.uint8)[self.face_element]

    def face_transparent(self) -> np.ndarray:
        return np.array([e.transparent for e in self.elements], dtype=bool)[self.face_element]

    def face_normals(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------------------------------------------

def flat(poly: Any, z: float) -> tuple[np.ndarray, np.ndarray]:
    tris = geo.triangulate(poly)
    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    v = np.vstack([np.column_stack((t, np.full(3, z))) for t in tris])
    f = np.arange(len(v), dtype=np.int32).reshape(-1, 3)
    return v, f


def extrude(poly: Any, z0: float, z1: float, caps: tuple[bool, bool] = (True, True)) -> tuple[np.ndarray, np.ndarray]:
    """Prism of a (multi)polygon with holes between z0 and z1 (caps via constrained Delaunay)."""
    parts_v: list[np.ndarray] = []
    parts_f: list[np.ndarray] = []
    offset = 0

    def add(v: np.ndarray, f: np.ndarray) -> None:
        nonlocal offset
        parts_v.append(v)
        parts_f.append(f + offset)
        offset += len(v)

    polys = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
    for item in polys:
        if item.is_empty:
            continue
        if caps[0]:
            add(*flat(item, z0))
        if caps[1]:
            add(*flat(item, z1))
        for ring in [item.exterior, *item.interiors]:
            c = np.asarray(ring.coords)[:-1]
            if len(c) < 2:
                continue
            nxt = np.roll(c, -1, axis=0)
            k = len(c)
            v = np.vstack([np.column_stack((c, np.full(k, z0))), np.column_stack((nxt, np.full(k, z0))),
                           np.column_stack((nxt, np.full(k, z1))), np.column_stack((c, np.full(k, z1)))])
            i = np.arange(k)
            f = np.vstack([np.column_stack((i, i + k, i + 2 * k)), np.column_stack((i, i + 2 * k, i + 3 * k))]).astype(np.int32)
            add(v, f)
    if not parts_v:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    return np.vstack(parts_v), np.vstack(parts_f)


def _transform(yaw_deg: float, centre: tuple[float, float, float]) -> np.ndarray:
    a = math.radians(yaw_deg)
    t = np.eye(4)
    t[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    t[:3, 3] = centre
    return t


def box_mesh(centre, size, yaw_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    m = trimesh.creation.box(extents=np.asarray(size, dtype=float), transform=_transform(yaw_deg, tuple(centre)))
    return np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int32)


def cylinder_mesh(centre, radius: float, height: float, sections: int = 24) -> tuple[np.ndarray, np.ndarray]:
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections, transform=_transform(0.0, tuple(centre)))
    return np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int32)


def sphere_mesh(centre, radius: float) -> tuple[np.ndarray, np.ndarray]:
    m = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    return np.asarray(m.vertices) + np.asarray(centre, dtype=float), np.asarray(m.faces, dtype=np.int32)


def _merge(pieces: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vs, fs, off = [], [], 0
    for v, f in pieces:
        if len(f) == 0:
            continue
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    return np.vstack(vs), np.vstack(fs).astype(np.int32)


def part_mesh(part, item_tf: np.ndarray, item_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    centre_local = np.array([part.center[0], part.center[1], part.center[2], 1.0])
    centre = (item_tf @ centre_local)[:3]
    if part.shape == "box":
        return box_mesh(centre, part.size, item_yaw + part.yaw_deg)
    if part.shape == "cyl":
        return cylinder_mesh(centre, float(part.size[0]), float(part.size[2]))
    return sphere_mesh(centre, float(part.size[0]))


# ---------------------------------------------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------------------------------------------

def build_scene_mesh(spec: SceneSpec) -> SceneMesh:
    rooms = geo.room_polygons(spec)
    height = max([spec.storey_height] + [r["ceiling_h"] for r in rooms])
    elements: list[Element] = []
    rng = np.random.default_rng(spec.seed + 7)

    v, f = _merge([extrude(poly, z0, z1) for z0, z1, poly in geo.wall_bands(spec, height)])
    elements.append(Element("walls", "wall", CLASS_WALL, 0, v, f, PALETTE["wall"]))
    v, f = flat(geo.footprint_union(spec), 0.0)
    elements.append(Element("floor", "floor", CLASS_FLOOR, 0, v, f, PALETTE["floor"]))
    for room in rooms:
        v, f = flat(room["polygon"], room["ceiling_h"])
        elements.append(Element(f"ceiling:{room['name']}", "ceiling", CLASS_CEILING, 0, v, f, PALETTE["ceiling"], label=room["name"]))

    for opening in spec.openings:
        wall = spec.wall(opening.wall_id)
        d, n = wall.direction(), wall.normal()
        yaw = math.degrees(math.atan2(d[1], d[0]))
        c = geo.opening_centre(spec, opening)
        if opening.kind == "door":
            hinge, closed, length = geo.door_hinge_and_leaf(spec, opening)
            a = math.radians(opening.leaf_angle_deg)
            vdir = closed * math.cos(a) + n * opening.swing * math.sin(a)
            centre2 = hinge + vdir * (length - 0.02) / 2 + (n * opening.swing) * 0.025
            leaf_yaw = math.degrees(math.atan2(vdir[1], vdir[0]))
            leaf_h = opening.head - 0.015
            v, f = box_mesh((centre2[0], centre2[1], leaf_h / 2 + 0.005), (length - 0.02, 0.04, leaf_h), leaf_yaw)
            elements.append(Element(opening.id, "door", CLASS_DOOR, opening.instance_id, v, f, PALETTE["door"], label="door"))
        elif opening.kind == "window":
            hgt = opening.head - opening.sill
            zc = (opening.sill + opening.head) / 2
            fw = 0.06
            frame = [
                box_mesh((c[0], c[1], opening.sill + fw / 2), (opening.width, 0.08, fw), yaw),
                box_mesh((c[0], c[1], opening.head - fw / 2), (opening.width, 0.08, fw), yaw),
                box_mesh((*(c + d * (opening.width / 2 - fw / 2)), zc), (fw, 0.08, hgt), yaw),
                box_mesh((*(c - d * (opening.width / 2 - fw / 2)), zc), (fw, 0.08, hgt), yaw),
            ]
            if opening.width > 1.4:  # mullion
                frame.append(box_mesh((c[0], c[1], zc), (fw, 0.08, hgt), yaw))
            v, f = _merge(frame)
            elements.append(Element(opening.id, "window", CLASS_WINDOW, opening.instance_id, v, f, PALETTE["window_frame"], label="window"))
            v, f = box_mesh((c[0], c[1], zc), (opening.width - 2 * fw, 0.02, hgt - 2 * fw), yaw)
            elements.append(Element(opening.id + ":glass", "glass", CLASS_WINDOW, opening.instance_id, v, f, PALETTE["glass"], label="window", transparent=True))

    for item in spec.furniture:
        tf = geo.item_transform(item)
        v, f = _merge([part_mesh(p, tf, item.yaw_deg) for p in item.parts])
        base = np.asarray(LABEL_COLOURS.get(item.label, (150, 150, 150)), dtype=float)
        colour = tuple(int(x) for x in np.clip(base + rng.normal(0, 8, 3), 0, 255))
        if item.label in ("spotlight", "ceiling lamp"):
            colour = (255, 255, 255)  # a lamp that is on
        elements.append(Element(item.id, "furniture", item.class_idx, item.instance_id, v, f, colour, label=item.label, meta={"variant": item.variant}))

    for column in spec.columns:
        room = geo.room_at(rooms, column.center)
        top = room["ceiling_h"] if room else spec.storey_height
        if column.shape == "cyl":
            v, f = cylinder_mesh((column.center[0], column.center[1], top / 2), column.size[0], top, 32)
        else:
            v, f = box_mesh((column.center[0], column.center[1], top / 2), (column.size[0], column.size[1], top), column.yaw_deg)
        elements.append(Element(column.id, "column", CLASS_COLUMN, column.instance_id, v, f, PALETTE["column"], label="column"))

    for beam in spec.beams:
        mid = ((beam.p0[0] + beam.p1[0]) / 2, (beam.p0[1] + beam.p1[1]) / 2)
        room = geo.room_at(rooms, mid)
        top = room["ceiling_h"] if room else spec.storey_height
        length = math.dist(beam.p0, beam.p1)
        yaw = math.degrees(math.atan2(beam.p1[1] - beam.p0[1], beam.p1[0] - beam.p0[0]))
        v, f = box_mesh((mid[0], mid[1], top - beam.depth / 2), (length, beam.width, beam.depth), yaw)
        elements.append(Element(beam.id, "beam", CLASS_BEAM, 0, v, f, PALETTE["beam"], label="beam"))

    for stair in spec.stairs:
        a = math.radians(stair.yaw_deg)
        dvec = np.array([math.cos(a), math.sin(a)])
        steps = []
        for i in range(stair.steps):
            cxy = np.asarray(stair.start) + dvec * stair.run * (i + 0.5)
            ztop = stair.rise * (i + 1)
            steps.append(box_mesh((cxy[0], cxy[1], ztop / 2), (stair.run, stair.width, ztop), stair.yaw_deg))
        v, f = _merge(steps)
        elements.append(Element(stair.id, "stair", CLASS_STAIRS, 0, v, f, PALETTE["stair"], label="stairs"))

    vs, fs, fe, off = [], [], [], 0
    for index, element in enumerate(elements):
        if len(element.faces) == 0:
            continue
        vs.append(element.vertices)
        fs.append(element.faces + off)
        fe.append(np.full(len(element.faces), index, dtype=np.int32))
        off += len(element.vertices)
    return SceneMesh(np.vstack(vs), np.vstack(fs).astype(np.int32), np.concatenate(fe), elements)


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Area-weighted uniform samples on a triangle soup."""
    if len(faces) == 0 or count <= 0:
        return np.zeros((0, 3))
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return np.zeros((0, 3))
    pick = rng.choice(len(faces), size=count, p=area / area.sum())
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)
    a, b, c = tri[pick, 0], tri[pick, 1], tri[pick, 2]
    return a * (1 - s)[:, None] + b * (s * (1 - r2))[:, None] + c * (s * r2)[:, None]


def element_area(element: Element) -> float:
    if len(element.faces) == 0:
        return 0.0
    tri = element.vertices[element.faces]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def visible_surface_mask(points: np.ndarray, rooms: list[dict[str, Any]], height: float) -> np.ndarray:
    """True for samples that lie inside the building (excludes exterior wall faces and hidden caps)."""
    from shapely import contains_xy
    from shapely.ops import unary_union

    inside = unary_union([r["polygon"].buffer(0.02) for r in rooms])
    return contains_xy(inside, points[:, 0], points[:, 1]) & (points[:, 2] <= height + 1e-6)
