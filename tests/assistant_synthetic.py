"""Synthetic project for the AI assistant tests: a 2-room plan, a matching classified point cloud and a DXF result.

8 x 5 m flat (local plan metres, units_per_metre 1): exterior walls 0.30 m, a partition at x = 4 (0.20 m) with a door,
a window in the south wall, a room on each side and a table in the west room.  The point cloud (noise ~4 mm, points
never on grid lines) shows the walls, floor, ceiling, the table, and a chair in the east room that the plan does NOT
contain (for measured ``add_furniture``); the north-east corner of the east room is empty (no evidence).
Nothing here comes from a real project.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from samcloud.bim.plan_model import BuildingPlan, PlanFurniture, PlanOpening, PlanRoom, PlanWall, save_plan_model

FLOOR_Z = 0.20
CEILING_Z = 2.80
FRAME_ORIGIN = (120.0, 340.0, 5.0)
LEVEL_ID = "level_1"
PROJECT_NAME = "assistant_synthetic"
TABLE_BOX = (1.0, 2.2, 1.0, 1.8)  # x0, x1, y0, y1 of the plan table F001
TABLE_TOP_M = 0.75
CHAIR_BOX = (5.30, 5.75, 2.10, 2.60)  # chair in the point cloud only (not in the plan)
CHAIR_SEAT_M, CHAIR_BACK_M = 0.46, 0.92
EMPTY_BOX = {"x": [6.6, 7.4], "y": [3.6, 4.4]}  # no object points here
FINE_LABELS = ["table", "chair"]


def _wall(wall_id: str, a: tuple[float, float], b: tuple[float, float], thickness: float, kind: str) -> PlanWall:
    p0, p1 = np.asarray(a, float), np.asarray(b, float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    right = np.asarray((u[1], -u[0]))
    face_a = [list(p0 + right * thickness / 2.0), list(p1 + right * thickness / 2.0)]
    face_b = [list(p0 - right * thickness / 2.0), list(p1 - right * thickness / 2.0)]
    return PlanWall(id=wall_id, centreline=[list(map(float, a)), list(map(float, b))], thickness_m=thickness, kind=kind,
                    evidence="face", face_a=[[float(x), float(y)] for x, y in face_a], face_b=[[float(x), float(y)] for x, y in face_b],
                    body_id=0, height_m=None, height_source="storey")


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def make_plan() -> BuildingPlan:
    walls = [
        _wall("W001", (0.0, 0.0), (8.0, 0.0), 0.30, "exterior"),
        _wall("W002", (8.0, 0.0), (8.0, 5.0), 0.30, "exterior"),
        _wall("W003", (8.0, 5.0), (0.0, 5.0), 0.30, "exterior"),
        _wall("W004", (0.0, 5.0), (0.0, 0.0), 0.30, "exterior"),
        _wall("W005", (4.0, 0.0), (4.0, 5.0), 0.20, "interior"),
    ]
    rooms = [
        PlanRoom("R001", "Wohnen", {"exterior": _rect(0.15, 0.15, 3.9, 4.85), "holes": []}, 3.75 * 4.7, [2.0, 2.5]),
        PlanRoom("R002", "Schlafen", {"exterior": _rect(4.1, 0.15, 7.85, 4.85), "holes": []}, 3.75 * 4.7, [6.0, 2.5]),
    ]
    openings = [
        PlanOpening("O001", "door", [4.0, 2.5], 0.9, math.pi / 2.0, 0.20, host_wall_id="W005", hinge_side="left", swing_side=1,
                    leaf_detected=True, sill_m=0.0, head_m=2.05, evidence="synthetic"),
        PlanOpening("O002", "window", [2.0, 0.0], 1.2, 0.0, 0.30, host_wall_id="W001", sill_m=0.9, head_m=2.1, mullion_count=0,
                    evidence="synthetic"),
    ]
    x0, x1, y0, y1 = TABLE_BOX
    furniture = [
        PlanFurniture("F001", 5, "Tisch", "table", None, "A-FURN", _rect(x0, y0, x1, y1), z_min=FLOOR_Z, z_max=FLOOR_Z + TABLE_TOP_M,
                      z_source="instance_bbox", confidence=0.9, default_height_m=TABLE_TOP_M),
    ]
    body = unary_union([Polygon(_band(w)) for w in walls])
    return BuildingPlan(
        project_name=PROJECT_NAME, level_id=LEVEL_ID, level_name="EG", units_per_metre=1.0,
        frame={"origin": list(FRAME_ORIGIN), "scale_factor": 1.0, "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
        floor_z=FLOOR_Z, ceiling_z=CEILING_Z, walls=walls, rooms=rooms, openings=openings, furniture=furniture,
        wall_bodies=[{"exterior": [[float(x), float(y)] for x, y in list(body.exterior.coords)[:-1]],
                      "holes": [[[float(x), float(y)] for x, y in list(r.coords)[:-1]] for r in body.interiors]}],
        source={"result_fingerprint": "synthetic"},
    )


def _band(wall: PlanWall) -> list[tuple[float, float]]:
    p0, p1 = np.asarray(wall.centreline[0], float), np.asarray(wall.centreline[1], float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.asarray((-u[1], u[0])) * wall.thickness_m / 2.0
    return [tuple(p0 + n), tuple(p1 + n), tuple(p1 - n), tuple(p0 - n)]


# ---------------------------------------------------------------------------------------------------------- point cloud

def _jittered_grid(rng: np.random.Generator, x0: float, x1: float, y0: float, y1: float, step: float) -> np.ndarray:
    xs = np.arange(x0 + step * 0.37, x1, step)
    ys = np.arange(y0 + step * 0.41, y1, step)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return grid + rng.uniform(-0.3 * step, 0.3 * step, size=grid.shape)


def make_cloud_points(plan: BuildingPlan, seed: int = 7) -> dict[str, np.ndarray]:
    """Local plan coordinates (absolute local z) with class / instance / fine term per point."""
    rng = np.random.default_rng(seed)
    xyz, cls, inst, fine = [], [], [], []

    def add(points: np.ndarray, code: int, instance: int = 0, term: int = 0) -> None:
        points = np.asarray(points, float)
        points = points + rng.normal(0.0, 0.004, size=points.shape)
        xyz.append(points)
        cls.append(np.full(len(points), code, np.uint8))
        inst.append(np.full(len(points), instance, np.uint32))
        fine.append(np.full(len(points), term, np.uint8))

    height = CEILING_Z - FLOOR_Z
    for room in plan.rooms:
        x0, y0 = np.min(room.polygon["exterior"], axis=0)
        x1, y1 = np.max(room.polygon["exterior"], axis=0)
        floor = _jittered_grid(rng, x0, x1, y0, y1, 0.09)
        add(np.column_stack((floor, np.full(len(floor), FLOOR_Z))), 4)
        ceiling = _jittered_grid(rng, x0, x1, y0, y1, 0.12)
        add(np.column_stack((ceiling, np.full(len(ceiling), CEILING_Z))), 3)
    for wall in plan.walls:
        p0, p1 = np.asarray(wall.centreline[0], float), np.asarray(wall.centreline[1], float)
        length = float(np.linalg.norm(p1 - p0))
        u = (p1 - p0) / length
        n = np.asarray((-u[1], u[0]))
        grid = _jittered_grid(rng, 0.0, length, 0.0, height, 0.07)
        for side in (1.0, -1.0):
            base = p0 + n * side * wall.thickness_m / 2.0
            pts = base[None, :] + grid[:, :1] * u[None, :]
            keep = np.ones(len(grid), bool)
            for opening in plan.openings:
                if opening.host_wall_id != wall.id:
                    continue
                s = float((np.asarray(opening.center) - p0) @ u)
                inside = (np.abs(grid[:, 0] - s) < opening.width_m / 2.0) & (grid[:, 1] > opening.sill_m) & (grid[:, 1] < opening.head_m)
                keep &= ~inside
            add(np.column_stack((pts[keep], FLOOR_Z + grid[keep, 1])), 2)
    # table F001: top + 4 legs
    x0, x1, y0, y1 = TABLE_BOX
    top = _jittered_grid(rng, x0, x1, y0, y1, 0.025)
    add(np.column_stack((top, np.full(len(top), FLOOR_Z + TABLE_TOP_M))), 17, 5, 1)
    for lx, ly in ((x0 + 0.06, y0 + 0.06), (x1 - 0.06, y0 + 0.06), (x1 - 0.06, y1 - 0.06), (x0 + 0.06, y1 - 0.06)):
        heights = np.linspace(0.04, TABLE_TOP_M - 0.03, 40) + 0.0013
        for angle in np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False):
            add(np.column_stack((np.full(40, lx + 0.02 * math.cos(angle)), np.full(40, ly + 0.02 * math.sin(angle)), FLOOR_Z + heights)), 17, 5, 1)
    # chair (cloud only): seat, back, legs
    cx0, cx1, cy0, cy1 = CHAIR_BOX
    seat = _jittered_grid(rng, cx0, cx1, cy0, cy1, 0.02)
    add(np.column_stack((seat, np.full(len(seat), FLOOR_Z + CHAIR_SEAT_M))), 17, 9, 2)
    back = _jittered_grid(rng, cx0, cx1, CHAIR_SEAT_M, CHAIR_BACK_M, 0.02)
    add(np.column_stack((back[:, 0], np.full(len(back), cy1 - 0.02), FLOOR_Z + back[:, 1])), 17, 9, 2)
    for lx, ly in ((cx0 + 0.03, cy0 + 0.03), (cx1 - 0.03, cy0 + 0.03), (cx1 - 0.03, cy1 - 0.03), (cx0 + 0.03, cy1 - 0.03)):
        heights = np.linspace(0.03, CHAIR_SEAT_M - 0.02, 30) + 0.0017
        add(np.column_stack((np.full(30, lx), np.full(30, ly), FLOOR_Z + heights)), 17, 9, 2)
    return {"xyz": np.concatenate(xyz), "cls": np.concatenate(cls), "instance": np.concatenate(inst), "fine": np.concatenate(fine)}


def write_cloud(project: Path, plan: BuildingPlan, seed: int = 7) -> Path:
    from plyfile import PlyData, PlyElement

    points = make_cloud_points(plan, seed)
    world = points["xyz"] + np.asarray(FRAME_ORIGIN)  # identity rotation, scale 1
    directory = project / "products" / "classified"
    directory.mkdir(parents=True, exist_ok=True)
    data = np.empty(len(world), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                       ("scalar_class", "u1"), ("fine_term", "u1"), ("instance_id", "u4")])
    data["x"], data["y"], data["z"] = world[:, 0], world[:, 1], world[:, 2]
    data["red"] = data["green"] = data["blue"] = 128
    data["scalar_class"], data["fine_term"], data["instance_id"] = points["cls"], points["fine"], points["instance"]
    path = directory / "classified.ply"
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))
    (directory / "fine_labels.json").write_text(json.dumps({"labels": FINE_LABELS}), encoding="utf-8")
    instances = []
    for identifier, label in ((5, "table"), (9, "chair")):
        sel = points["instance"] == identifier
        pts = world[sel]
        instances.append({"id": identifier, "class_id": 17, "class_name": "furniture", "label": label, "point_count": int(sel.sum()),
                          "bbox_min": pts.min(axis=0).tolist(), "bbox_max": pts.max(axis=0).tolist(), "confidence": 0.9})
    (directory / "instances.json").write_text(json.dumps({"schema_version": 1, "instances": instances}), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------------------ DXF result

LAYER_COLORS = {"A-WALL": 7, "A-WALL-PATT": 253, "A-DOOR": 1, "A-WIND": 4, "A-FURN": 34, "P-SANI": 151, "A-ANNO": 7, "A-AREA": 140,
                "A-FURN-DETL": 8, "A-EINB": 30, "A-OBEN": 8}


def make_result(plan: BuildingPlan) -> Any:
    """A DxfPlanResult like the clean plan draws it (pipeline helpers for openings and room labels)."""
    from samcloud.dxf.las_source import Frame
    from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
    from samcloud.ortho.vector.indoor import Opening, opening_entities
    from samcloud.ortho.vector.room_names import room_label_text

    upm = plan.units_per_metre
    layers: dict[str, list[dict[str, Any]]] = {name: [] for name in LAYER_COLORS}
    voids = []
    for opening in plan.openings:
        axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
        normal = np.asarray((-axis[1], axis[0]))
        c = np.asarray(opening.center, float)
        hw, ht = opening.width_m / 2.0, opening.wall_thickness_m / 2.0 + 0.01
        voids.append(Polygon([tuple(c + axis * a + normal * b) for a, b in ((-hw, -ht), (hw, -ht), (hw, ht), (-hw, ht))]))
    mass = unary_union([Polygon(_band(w)) for w in plan.walls]).difference(unary_union(voids))
    for piece in sorted(getattr(mass, "geoms", [mass]), key=lambda p: -p.area):
        ring = [[float(x), float(y)] for x, y in list(piece.exterior.coords)[:-1]]
        holes = [[[float(x), float(y)] for x, y in list(r.coords)[:-1]] for r in piece.interiors]
        layers["A-WALL"].append({"type": "polyline", "points": ring, "closed": True, "holes": holes, "_double_wall": True})
        layers["A-WALL-PATT"].append({"type": "hatch", "points": ring, "holes": holes, "_double_wall": True})
    specs = []
    for opening in plan.openings:
        kind = {"door": "DOOR", "window": "WINDOW"}.get(opening.kind, "passage")
        specs.append(Opening(kind=kind, center=tuple(opening.center), width=opening.width_m * upm, axis_angle=opening.axis_angle,
                             wall_thickness=opening.wall_thickness_m * upm, evidence=opening.evidence, hinge_side=opening.hinge_side,
                             swing_side=opening.swing_side, leaf_detected=opening.leaf_detected, sill_m=opening.sill_m, head_m=opening.head_m))
    for entity in opening_entities(specs, None, width_scale=1.0 / upm, height_scale=1.0):
        entity["_on_face"] = True
        layers["A-WIND" if entity.get("_opening_kind") == "WINDOW" else "A-DOOR"].append(entity)
    for room in plan.rooms:
        polygon = Polygon(room.polygon["exterior"])
        layers["A-AREA"].append({"type": "polyline", "points": room.polygon["exterior"], "closed": True, "holes": []})
        point = polygon.representative_point()
        layers["A-ANNO"].append({"type": "text", "text": room_label_text(room.name, room.area_m2), "point": [point.x, point.y],
                                 "height": 0.12, "rotation": 0.0, "_room_name": room.name})
    for index, item in enumerate(plan.furniture, start=1):
        layers[item.layer].append({"type": "polyline", "points": item.footprint, "closed": True, "holes": [], "_furn_id": index,
                                   "_label": item.label, "_semantic_label": item.semantic_label})
        polygon = Polygon(item.footprint)
        point = polygon.representative_point()
        layers["A-ANNO"].append({"type": "text", "text": f"{item.label} h={TABLE_TOP_M:.2f}", "point": [point.x, point.y], "height": 0.08,
                                 "rotation": 0.0, "_furn_id": index, "_label_placed": True})
    frame = Frame(np.asarray(plan.frame["origin"], float), float(plan.frame["scale_factor"]), np.asarray(plan.frame["rotation"], float))
    statuses = {name: LayerStatus(name, "ok" if entities else "empty", None if entities else "no geometry survived filtering", len(entities))
                for name, entities in layers.items()}
    return DxfPlanResult(
        project_name=plan.project_name, level_name=plan.level_name, mode="indoor", profile_name=None, scale_factor=1.0, frame=frame,
        settings_fingerprint="synthetic", layers=layers, statuses=statuses, layer_colors=dict(LAYER_COLORS),
        background=BackgroundImage(None, None, (-1.0, -1.0, 9.0, 6.0)), layer_linetypes={name: "CONTINUOUS" for name in LAYER_COLORS},
        layer_hatches={name: False for name in LAYER_COLORS}, layer_styles={"A-WALL-PATT": {"hatch_pattern": "SOLID"}},
        blocks=sorted({str(e["name"]) for entities in layers.values() for e in entities if e.get("type") == "insert"}), variant="ortho",
    )


# --------------------------------------------------------------------------------------------------------------- project

def make_project(root: Path, *, with_cloud: bool = True) -> Path:
    """``<root>/project`` with plan_model.json + result.json in the ortho cache and (optionally) the classified cloud."""
    from samcloud.dxf.pipeline import save_cached_result

    project = Path(root) / "project"
    cache = project / "products" / "dxf" / LEVEL_ID / "ortho"
    plan = make_plan()
    save_plan_model(plan, cache)
    save_cached_result(make_result(plan), cache)
    if with_cloud:
        write_cloud(project, plan)
    return project


def renumbered_recompute(plan: BuildingPlan, shift: tuple[float, float] = (0.012, -0.008)) -> BuildingPlan:
    """A 'pipeline recompute' of ``plan``: reversed element order, new ids, geometry shifted by a few millimetres."""
    import copy

    new = copy.deepcopy(plan)
    dx, dy = shift

    def move(points: list[list[float]]) -> list[list[float]]:
        return [[float(x) + dx, float(y) + dy] for x, y in points]

    new.walls = list(reversed(new.walls))
    id_map: dict[str, str] = {}
    for index, wall in enumerate(new.walls, start=1):
        id_map[wall.id] = f"W{index:03d}"
        wall.id = id_map[wall.id]
        wall.centreline = move(wall.centreline)
        wall.face_a = move(wall.face_a) if wall.face_a else None
        wall.face_b = move(wall.face_b) if wall.face_b else None
    new.openings = list(reversed(new.openings))
    for index, opening in enumerate(new.openings, start=1):
        opening.id = f"O{index:03d}"
        opening.center = [opening.center[0] + dx, opening.center[1] + dy]
        opening.host_wall_id = id_map.get(opening.host_wall_id or "", opening.host_wall_id)
    new.rooms = list(reversed(new.rooms))
    for index, room in enumerate(new.rooms, start=1):
        room.id = f"R{index:03d}"
        room.polygon = {"exterior": move(room.polygon["exterior"]), "holes": [move(h) for h in room.polygon.get("holes", [])]}
        room.label_point = [room.label_point[0] + dx, room.label_point[1] + dy]
    for index, item in enumerate(new.furniture, start=1):
        item.id = f"F{index + 10:03d}"
        item.footprint = move(item.footprint)
    return new
