"""Ground truth of a scene in the scene frame (metres) as a JSON-friendly dict, plus surface samples per class."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import geometry as geo
from .furniture import label_group
from .mesh import SceneMesh, sample_surface, visible_surface_mask
from .spec import CLASS_CEILING_EQUIPMENT, CLASS_HVAC, SceneSpec


def ground_truth(spec: SceneSpec) -> dict[str, Any]:
    rooms = geo.room_polygons(spec)
    height = max([spec.storey_height] + [r["ceiling_h"] for r in rooms])
    walls = []
    for wall in spec.walls:
        walls.append({
            "id": wall.id, "p0": list(wall.p0), "p1": list(wall.p1), "thickness": wall.thickness, "exterior": wall.exterior,
            "length": wall.length,
        })
    junctions = _junctions(spec)
    openings = []
    for opening in spec.openings:
        wall = spec.wall(opening.wall_id)
        c = geo.opening_centre(spec, opening)
        openings.append({
            "id": opening.id, "kind": opening.kind, "host": opening.wall_id, "center": [float(c[0]), float(c[1])],
            "width": opening.width, "sill": 0.0 if opening.kind != "window" else opening.sill, "head": opening.head,
            "axis_angle": math.atan2(wall.direction()[1], wall.direction()[0]), "thickness": wall.thickness,
        })
    furniture = []
    ceiling_objects = []
    for item in spec.furniture:
        fp = geo.item_footprint(item)
        z0, z1 = geo.item_z(item)
        if item.class_idx in (CLASS_CEILING_EQUIPMENT, CLASS_HVAC):
            if not item.variant.startswith("neg_"):
                ceiling_objects.append({
                    "id": item.id, "label": item.label, "footprint": geo.polygon_to_list(fp), "z0": z0, "z1": z1,
                    "center": list(item.center), "variant": item.variant,
                })
            continue
        furniture.append({
            "id": item.id, "label": item.label, "group": label_group(item.label), "class_idx": item.class_idx,
            "variant": item.variant, "instance_id": item.instance_id, "footprint": geo.polygon_to_list(fp), "z0": z0, "z1": z1,
            "center": list(item.center), "yaw_deg": item.yaw_deg, "obb": geo.oriented_box(item),
        })
    room_list = []
    for room in rooms:
        room_list.append({"name": room["name"], "polygon": geo.polygon_to_list(room["polygon"]), "area": room["area"], "ceiling_h": room["ceiling_h"]})
    columns = [{"id": c.id, "footprint": geo.polygon_to_list(geo.column_footprint(c)), "center": list(c.center), "shape": c.shape} for c in spec.columns]
    beams = [{"id": b.id, "footprint": geo.polygon_to_list(geo.beam_footprint(b)), "p0": list(b.p0), "p1": list(b.p1), "width": b.width, "depth": b.depth} for b in spec.beams]
    stairs = [{"id": s.id, "footprint": geo.polygon_to_list(geo.stair_footprint(s)), "steps": s.steps, "rise": s.rise, "run": s.run, "width": s.width} for s in spec.stairs]
    return {
        "name": spec.name, "family": spec.family, "seed": spec.seed, "storey_height": spec.storey_height, "max_height": height,
        "world_yaw_deg": spec.world_yaw_deg, "world_offset": list(spec.world_offset), "walls": walls, "junctions": junctions,
        "rooms": room_list, "openings": openings, "furniture": furniture, "columns": columns, "beams": beams, "stairs": stairs,
        "ceiling_objects": ceiling_objects,
        "features": list(spec.features),
    }


def _junctions(spec: SceneSpec) -> list[dict[str, Any]]:
    """Points where a wall end meets another wall (corner, T); used for the gap metric."""
    result = []
    for wall in spec.walls:
        for end in (wall.p0, wall.p1):
            partners = []
            for other in spec.walls:
                if other.id == wall.id:
                    continue
                p, q = np.asarray(other.p0), np.asarray(other.p1)
                d = q - p
                t = float(np.clip(np.dot(np.asarray(end) - p, d) / max(np.dot(d, d), 1e-12), 0, 1))
                dist = float(np.linalg.norm(p + d * t - np.asarray(end)))
                if dist < other.thickness / 2 + 0.02:
                    partners.append(other.id)
            if partners:
                result.append({"point": [float(end[0]), float(end[1])], "walls": [wall.id] + partners})
    unique: list[dict[str, Any]] = []
    for item in result:
        if not any(math.dist(item["point"], u["point"]) < 0.05 for u in unique):
            unique.append(item)
    return unique


def class_samples(spec: SceneSpec, mesh: SceneMesh, density: float = 400.0, seed: int = 5) -> dict[str, np.ndarray]:
    """Surface samples (scene frame) per element kind, restricted to surfaces visible from inside the building."""
    rng = np.random.default_rng(seed)
    rooms = geo.room_polygons(spec)
    height = max([spec.storey_height] + [r["ceiling_h"] for r in rooms])
    groups: dict[str, list[np.ndarray]] = {}
    for element in mesh.elements:
        kind = element.kind
        if kind == "glass":
            continue
        tri = element.vertices[element.faces]
        area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
        count = int(min(200_000, max(50, area * density)))
        pts = sample_surface(element.vertices, element.faces, count, rng)
        if kind == "wall":
            pts = pts[visible_surface_mask(pts, rooms, height)]
            # drop the hidden horizontal caps of the wall bands (top / bottom of the solid outside openings)
            pts = pts[(pts[:, 2] > 0.01) & (pts[:, 2] < height - 0.01)]
        groups.setdefault(kind, []).append(pts)
    return {kind: np.vstack(parts) for kind, parts in groups.items()}


def element_samples(mesh: SceneMesh, element_id: str, count: int = 3000, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for element in mesh.elements:
        if element.id == element_id:
            return sample_surface(element.vertices, element.faces, count, rng)
    return np.zeros((0, 3))
