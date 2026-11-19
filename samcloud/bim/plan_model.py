"""Neutral structured building model persisted next to the DXF plan result.

The indoor clean plan (``samcloud.ortho.plan``) draws DXF entities and, since WP2,
also persists this ``BuildingPlan`` so DXF and IFC show the same walls, rooms,
openings and furniture.  Conventions:

* ``centreline``, ``center``, ``footprint``, ``polygon``, ``face_a`` / ``face_b`` and
  ``z_min`` / ``z_max`` are in *local plan units* (the DXF frame, ``units_per_metre`` units
  per metre).
* every ``*_m`` field (``thickness_m``, ``width_m``, ``sill_m`` ...) is in metres.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml
from shapely.geometry import LineString, MultiPoint, Point, Polygon

SCHEMA_VERSION = 1
PLAN_MODEL_FILENAME = "plan_model.json"
PASS1_PLAN_FILENAME = "plan_model_pass1.json"
FURNITURE_LAYERS = ("A-FURN", "A-EINB", "A-OBEN", "P-SANI", "A-FURN-DETL")

Point2 = list[float]


# --------------------------------------------------------------------------
# dataclasses
# --------------------------------------------------------------------------

@dataclass(slots=True)
class PlanWall:
    """One straight wall piece. ``thickness_m`` is in metres; geometry in local units."""

    id: str
    centreline: list[Point2]
    thickness_m: float
    kind: str = "interior"  # interior | exterior | partition
    evidence: str = ""
    face_a: list[Point2] | None = None
    face_b: list[Point2] | None = None
    body_id: int | None = None  # index into BuildingPlan.wall_bodies (closed wall body ring)
    height_m: float | None = None  # None -> storey height

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanWall":
        return _from_dict(cls, data)


@dataclass(slots=True)
class PlanRoom:
    id: str
    name: str
    polygon: dict[str, Any]  # {"exterior": [[x, y], ...], "holes": [[[x, y], ...], ...]}
    area_m2: float
    label_point: Point2
    floor_z: float | None = None  # None -> storey
    ceiling_z: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanRoom":
        return _from_dict(cls, data)


@dataclass(slots=True)
class PlanOpening:
    id: str
    kind: str  # door | window | passage
    center: Point2
    width_m: float
    axis_angle: float  # rad, plan frame
    wall_thickness_m: float
    host_wall_id: str | None = None
    hinge_side: str | None = None
    swing_side: int | None = None
    leaf_angle_deg: float | None = None
    leaf_detected: bool | None = None
    sill_m: float = 0.0
    head_m: float = 2.0
    height_source: str = "default"
    mullion_count: int = 0
    evidence: str = ""
    mullion_u: list[float] = field(default_factory=list)  # metres from the left edge along the axis (windows)
    glazed: bool = False  # glazed door / floor-to-ceiling glazing
    confidence: float | None = None  # 3D refinement confidence, None for 2D-only openings

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanOpening":
        return _from_dict(cls, data)


@dataclass(slots=True)
class PlanFurniture:
    id: str
    instance_id: int | None
    label: str
    semantic_label: str
    catalogue_kind: str | None  # catalogue kind (BED_DOUBLE ...), block name or built-in kind
    layer: str  # A-FURN | A-EINB | A-OBEN | P-SANI | A-FURN-DETL
    footprint: list[Point2]  # closed ring, local plan coordinates (first point not repeated)
    z_min: float | None = None  # local units, from instances.json bbox when instance_id is known
    z_max: float | None = None
    z_source: str = "default"  # instance_bbox | default
    confidence: float | None = None
    default_height_m: float | None = None  # metric default used when z_min/z_max are None
    default_z_m: list[float] | None = None  # metric [low, high] for hanging units (A-OBEN)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanFurniture":
        return _from_dict(cls, data)


@dataclass(slots=True)
class BuildingPlan:
    project_name: str
    level_id: str
    level_name: str
    units_per_metre: float
    frame: dict[str, Any]  # {"origin": [x,y,z], "scale_factor": s, "rotation": 3x3}
    floor_z: float  # local units
    ceiling_z: float | None  # local units
    walls: list[PlanWall] = field(default_factory=list)
    rooms: list[PlanRoom] = field(default_factory=list)
    openings: list[PlanOpening] = field(default_factory=list)
    furniture: list[PlanFurniture] = field(default_factory=list)
    wall_bodies: list[dict[str, Any]] = field(default_factory=list)  # closed wall body rings {"exterior", "holes"}
    source: dict[str, Any] = field(default_factory=dict)  # {"result_fingerprint": ...}
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "level_id": self.level_id,
            "level_name": self.level_name,
            "units_per_metre": self.units_per_metre,
            "frame": self.frame,
            "floor_z": self.floor_z,
            "ceiling_z": self.ceiling_z,
            "walls": [item.to_dict() for item in self.walls],
            "rooms": [item.to_dict() for item in self.rooms],
            "openings": [item.to_dict() for item in self.openings],
            "furniture": [item.to_dict() for item in self.furniture],
            "wall_bodies": self.wall_bodies,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildingPlan":
        version = int(data.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported plan_model schema_version {version}")
        return cls(
            project_name=str(data["project_name"]), level_id=str(data["level_id"]), level_name=str(data["level_name"]),
            units_per_metre=float(data["units_per_metre"]), frame=dict(data["frame"]), floor_z=float(data["floor_z"]),
            ceiling_z=None if data.get("ceiling_z") is None else float(data["ceiling_z"]),
            walls=[PlanWall.from_dict(item) for item in data.get("walls", [])],
            rooms=[PlanRoom.from_dict(item) for item in data.get("rooms", [])],
            openings=[PlanOpening.from_dict(item) for item in data.get("openings", [])],
            furniture=[PlanFurniture.from_dict(item) for item in data.get("furniture", [])],
            wall_bodies=list(data.get("wall_bodies", [])), source=dict(data.get("source", {})), schema_version=version,
        )


def _to_dict(item: Any) -> dict[str, Any]:
    return {f.name: getattr(item, f.name) for f in fields(item)}


def _from_dict(cls: Any, data: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in names})


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def plan_model_path(directory: Path | str) -> Path:
    return Path(directory) / PLAN_MODEL_FILENAME


def save_plan_model(plan: BuildingPlan, directory: Path | str) -> Path:
    """Write ``plan_model.json`` atomically into ``directory``."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    destination = plan_model_path(target)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(plan.to_dict(), separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def pass1_plan_path(directory: Path | str) -> Path:
    """Snapshot of the plan computed WITHOUT refined openings (walls + 2D-detected openings = refinement seeds)."""
    return Path(directory) / PASS1_PLAN_FILENAME


def save_pass1_snapshot(directory: Path | str) -> Path:
    """Copy the just-written ``plan_model.json`` to ``plan_model_pass1.json`` (atomic)."""
    target = Path(directory)
    destination = pass1_plan_path(target)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(plan_model_path(target).read_bytes())
    os.replace(temporary, destination)
    return destination


def load_refine_plan(directory: Path | str) -> BuildingPlan:
    """Plan the opening refinement works on: the pass-1 snapshot when it exists (never the refined plan), else plan_model.json."""
    snapshot = pass1_plan_path(directory)
    return load_plan_model(snapshot if snapshot.is_file() else directory)


def load_plan_model(directory_or_file: Path | str) -> BuildingPlan:
    path = Path(directory_or_file)
    if path.is_dir():
        path = plan_model_path(path)
    return BuildingPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

_FALLBACK_CONFIG: dict[str, Any] = {
    "defaults": {
        "door_head_m": 2.0, "window_sill_m": 0.9, "window_head_m": 2.1, "passage_head_m": 2.0,
        "upper_cabinet_z_m": [1.45, 2.15], "furniture_height_fallback_m": 0.8,
        "kitchen_base_height_m": 0.86, "kitchen_worktop_top_m": 0.90, "kitchen_sink_bottom_m": 0.80, "kitchen_hob_height_m": 0.03,
    },
    "furniture_height_by_kind_m": {}, "furniture_height_by_label_m": {}, "furniture_height_by_einb_kind_m": {},
}


def load_bim_export_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load ``config/bim_export.yaml`` (metric defaults); built-in fallbacks fill gaps."""
    if path is None:
        from samcloud.core.paths import CONFIG_DIRECTORY

        path = Path(CONFIG_DIRECTORY) / "bim_export.yaml"
    loaded: dict[str, Any] = {}
    if Path(path).is_file():
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config: dict[str, Any] = {key: dict(value) for key, value in _FALLBACK_CONFIG.items()}
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


# --------------------------------------------------------------------------
# building the plan from the clean-plan objects
# --------------------------------------------------------------------------

def load_instance_z_ranges(instances_path: Path | str | None, frame: Any) -> dict[int, tuple[float, float, float | None]]:
    """Return {instance_id: (z_min, z_max, confidence)} in local units from an instances.json."""
    if instances_path is None or not Path(instances_path).is_file():
        return {}
    payload = json.loads(Path(instances_path).read_text(encoding="utf-8"))
    ranges: dict[int, tuple[float, float, float | None]] = {}
    for item in payload.get("instances", []):
        low, high = item.get("bbox_min"), item.get("bbox_max")
        if low is None or high is None:
            continue
        corners = np.asarray([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])], dtype=float)
        z = frame.to_local(corners)[:, 2]
        confidence = item.get("confidence")
        ranges[int(item["id"])] = (float(z.min()), float(z.max()), None if confidence is None else float(confidence))
    return ranges


def load_instance_hulls(instances_path: Path | str | None, frame: Any) -> list[dict[str, Any]]:
    """Instances with their local plan hull: ``[{"id", "label", "hull", "z_min", "z_max", "confidence"}]`` (z in local units)."""
    if instances_path is None or not Path(instances_path).is_file():
        return []
    payload = json.loads(Path(instances_path).read_text(encoding="utf-8"))
    hulls: list[dict[str, Any]] = []
    for item in payload.get("instances", []):
        low, high = item.get("bbox_min"), item.get("bbox_max")
        if low is None or high is None:
            continue
        corners = np.asarray([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])], dtype=float)
        local = frame.to_local(corners)
        hull = MultiPoint([(float(x), float(y)) for x, y in local[:, :2]]).convex_hull
        if not isinstance(hull, Polygon) or hull.is_empty:
            continue
        confidence = item.get("confidence")
        hulls.append({
            "id": int(item["id"]), "label": str(item.get("label") or "").strip().lower(), "hull": hull,
            "z_min": float(local[:, 2].min()), "z_max": float(local[:, 2].max()), "confidence": None if confidence is None else float(confidence),
        })
    return hulls


_UPPER_LABELS = frozenset({"kitchen cabinet", "cabinet", "kitchen counter", "shelf", "bookcase"})
_TALL_LABELS = frozenset({"refrigerator", "fridge", "kitchen cabinet", "cabinet", "wardrobe", "oven", "microwave oven"})


def _measured_instance(
    footprint: Polygon, instances: list[dict[str, Any]], labels: frozenset[str], floor_z: float, upm: float, low_m: float, high_m: float, min_cover: float = 0.5,
) -> dict[str, Any] | None:
    """The instance (of one of ``labels``) whose plan hull covers most of ``footprint`` and whose lower edge lies at ``low_m`` .. ``high_m`` above the floor.

    Among several the smallest hull wins (a tight instance beats a big merged one).
    """
    best: tuple[float, dict[str, Any]] | None = None
    for item in instances:
        if item["label"] not in labels:
            continue
        bottom = (float(item["z_min"]) - floor_z) / upm
        if not low_m <= bottom <= high_m:
            continue
        cover = footprint.intersection(item["hull"]).area / max(footprint.area, 1e-12)
        if cover >= min_cover and (best is None or item["hull"].area < best[0]):
            best = (item["hull"].area, item)
    return None if best is None else best[1]


def _points(values: Iterable[Any]) -> list[Point2]:
    return [[float(point[0]), float(point[1])] for point in values]


def _polygon_dict(polygon: Polygon) -> dict[str, Any]:
    return {
        "exterior": _points(list(polygon.exterior.coords)[:-1]),
        "holes": [_points(list(ring.coords)[:-1]) for ring in polygon.interiors],
    }


def _walls_from_pieces(pieces: list[dict[str, Any]], bodies: list[Polygon]) -> list[PlanWall]:
    walls: list[PlanWall] = []
    for index, piece in enumerate(pieces, start=1):
        a = np.asarray(piece["face"], dtype=float)
        b = np.asarray(piece["face_b"], dtype=float)
        centre = (a + b) / 2.0
        midpoint = Point(*centre.mean(axis=0))
        body_id = next((i for i, body in enumerate(bodies) if body.buffer(1e-6).contains(midpoint)), None)
        if body_id is None and bodies:
            distances = [body.distance(midpoint) for body in bodies]
            nearest = int(np.argmin(distances))
            body_id = nearest if distances[nearest] <= 1e-3 + float(piece.get("thickness", 0.0)) else None
        walls.append(PlanWall(
            id=f"W{index:03d}", centreline=_points(centre), thickness_m=float(piece["thickness"]),
            kind=str(piece.get("kind") or "interior"), evidence=str(piece.get("evidence") or ""),
            face_a=_points(a), face_b=_points(b), body_id=body_id,
        ))
    return walls


def _walls_from_lines(entities: list[dict[str, Any]]) -> list[PlanWall]:
    walls: list[PlanWall] = []
    for entity in entities:
        if entity.get("type") == "line":
            points = [entity["start"], entity["end"]]
        elif entity.get("type") == "polyline" and not entity.get("closed") and len(entity.get("points", [])) >= 2:
            points = list(entity["points"])
        else:
            continue
        for start, end in zip(points[:-1], points[1:]):
            walls.append(PlanWall(
                id=f"W{len(walls) + 1:03d}", centreline=_points([start, end]), thickness_m=0.0, kind="interior",
                evidence="single_face", face_a=_points([start, end]),
            ))
    return walls


def _host_wall(center: np.ndarray, width: float, thickness: float, axis_angle: float, walls: list[PlanWall], upm: float) -> str | None:
    """Nearest wall piece whose centreline passes the opening centre and runs along its axis."""
    axis = np.asarray((math.cos(axis_angle), math.sin(axis_angle)))
    best: tuple[float, str] | None = None
    for wall in walls:
        line = LineString(wall.centreline)
        if line.length <= 1e-9:
            continue
        direction = np.asarray(wall.centreline[1]) - np.asarray(wall.centreline[0])
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        if abs(float(np.dot(direction, axis))) < 0.9:
            continue
        limit = width / 2.0 + max(wall.thickness_m * upm, thickness) / 2.0 + 0.05 * upm
        # Perpendicular offset must stay inside the wall body; along-axis the piece may end at the opening edge.
        distance = line.distance(Point(*center))
        offset = abs(float(np.dot(center - np.asarray(wall.centreline[0]), np.asarray((-direction[1], direction[0])))))
        if distance <= limit and offset <= max(wall.thickness_m * upm, thickness) / 2.0 + 0.05 * upm and (best is None or distance < best[0]):
            best = (distance, wall.id)
    return None if best is None else best[1]


def _openings(openings: list[Any], walls: list[PlanWall], upm: float, config: dict[str, Any]) -> list[PlanOpening]:
    defaults = config["defaults"]
    result: list[PlanOpening] = []
    for index, opening in enumerate(openings, start=1):
        kind = {"DOOR": "door", "WINDOW": "window", "PASSAGE": "passage"}.get(str(opening.kind).upper(), str(opening.kind).lower())
        if kind == "window":
            sill, head = float(defaults["window_sill_m"]), float(defaults["window_head_m"])
        elif kind == "passage":
            sill, head = 0.0, float(defaults["passage_head_m"])
        else:
            sill, head = 0.0, float(defaults["door_head_m"])
        centre = np.asarray(opening.center, dtype=float)[:2]
        measured = getattr(opening, "height_source", None) not in (None, "")
        mullion_u = [float(value) for value in getattr(opening, "mullion_u", ())]
        if measured:
            sill, head = float(opening.sill_m), float(opening.head_m)
        result.append(PlanOpening(
            id=f"O{index:03d}", kind=kind, center=[float(centre[0]), float(centre[1])], width_m=float(opening.width) / upm,
            axis_angle=float(opening.axis_angle), wall_thickness_m=float(opening.wall_thickness) / upm,
            host_wall_id=_host_wall(centre, float(opening.width), float(opening.wall_thickness), float(opening.axis_angle), walls, upm),
            hinge_side=opening.hinge_side, swing_side=None if opening.swing_side is None else int(opening.swing_side),
            leaf_angle_deg=None if opening.leaf_angle_deg is None else float(opening.leaf_angle_deg),
            leaf_detected=opening.leaf_detected, sill_m=sill, head_m=head, height_source=str(opening.height_source) if measured else "default",
            mullion_count=len(mullion_u), evidence=str(opening.evidence or ""), mullion_u=mullion_u, glazed=bool(getattr(opening, "glazed", False)),
            confidence=getattr(opening, "confidence", None),
        ))
    return result


def _insert_footprint(insert: dict[str, Any]) -> Polygon | None:
    from shapely import affinity
    from shapely.geometry import box

    from samcloud.ortho.vector.symbols import catalogue_entry_for_block, symbol_extents

    name = str(insert.get("name", ""))
    scale = float(insert.get("xscale", 1.0))
    entry = catalogue_entry_for_block(name)
    if entry is not None:
        y0, y1 = (0.0, entry.depth_m) if entry.wall_mounted else (-entry.depth_m / 2.0, entry.depth_m / 2.0)
        body = box(-entry.width_m / 2.0 * scale, y0 * scale, entry.width_m / 2.0 * scale, y1 * scale)
    else:
        try:
            x0, y0, x1, y1 = symbol_extents(name)
        except (KeyError, ValueError):
            return None
        sx, sy = scale, float(insert.get("yscale", scale))
        body = box(min(x0 * sx, x1 * sx), min(y0 * sy, y1 * sy), max(x0 * sx, x1 * sx), max(y0 * sy, y1 * sy))
        if body.area <= 0.0:
            return None
    body = affinity.rotate(body, float(insert.get("rotation", 0.0)), origin=(0.0, 0.0))
    return affinity.translate(body, float(insert["point"][0]), float(insert["point"][1]))


def _default_height(config: dict[str, Any], kind: str | None, label: str, einb_kind: str | None) -> float:
    if kind and kind in config["furniture_height_by_kind_m"]:
        return float(config["furniture_height_by_kind_m"][kind])
    if einb_kind and einb_kind in config["furniture_height_by_einb_kind_m"]:
        return float(config["furniture_height_by_einb_kind_m"][einb_kind])
    by_label = config["furniture_height_by_label_m"]
    for candidate in (label.strip().lower(), label.strip().lower().split(" h=")[0]):
        if candidate in by_label:
            return float(by_label[candidate])
    return float(config["defaults"]["furniture_height_fallback_m"])


def _furniture(
    layers: dict[str, list[dict[str, Any]]], z_ranges: dict[int, tuple[float, float, float | None]], config: dict[str, Any],
    instances: list[dict[str, Any]] | None = None, floor_z: float = 0.0, units_per_metre: float = 1.0,
) -> list[PlanFurniture]:
    result: list[PlanFurniture] = []
    instances = instances or []
    upm = max(float(units_per_metre), 1e-9)
    kitchen_cfg = config["defaults"]
    seen_runs: set[tuple[tuple[float, float], ...]] = set()
    for layer in FURNITURE_LAYERS:
        for entity in layers.get(layer, []):
            outline = entity.get("_counter_outline") or (entity.get("points") if entity.get("_run_legs") and entity.get("closed") else None)
            if outline and len(outline) >= 3:
                # A kitchen run (drawn as open front / end lines): base cabinets and worktop, once per run.
                key = tuple((round(float(x), 3), round(float(y), 3)) for x, y in outline)
                if key in seen_runs:
                    continue
                seen_runs.add(key)
                run = Polygon(outline)
                run = run if run.is_valid else run.buffer(0)
                if run.is_empty or run.geom_type != "Polygon":
                    continue
                ring = _points(list(run.exterior.coords)[:-1])
                base, top = float(kitchen_cfg["kitchen_base_height_m"]), float(kitchen_cfg["kitchen_worktop_top_m"])
                result.append(PlanFurniture(
                    id=f"F{len(result) + 1:03d}", instance_id=None, label="Unterschrank", semantic_label="kitchen cabinet", catalogue_kind="kitchen_base",
                    layer=layer, footprint=ring, default_height_m=base,
                ))
                result.append(PlanFurniture(
                    id=f"F{len(result) + 1:03d}", instance_id=None, label="Arbeitsplatte", semantic_label="worktop", catalogue_kind="worktop",
                    layer=layer, footprint=ring, default_z_m=[base, top],
                ))
                continue
            kind_name: str | None = None
            einb_kind: str | None = None
            instance_id: int | None = None
            if entity.get("type") == "insert":
                footprint = _insert_footprint(entity)
                attributes = entity.get("attributes") or {}
                label = str(attributes.get("LABEL") or entity.get("name") or "")
                semantic = label
                catalogue = entity.get("_catalogue") or {}
                kind_name = str(catalogue.get("kind") or entity.get("name") or "") or None
                if attributes.get("INSTANCE") not in (None, ""):
                    instance_id = int(attributes["INSTANCE"])
            elif entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points", [])) >= 3:
                footprint = Polygon(entity["points"], entity.get("holes") or None)
                if not footprint.is_valid:
                    footprint = footprint.buffer(0)
                    if footprint.is_empty or footprint.geom_type != "Polygon":
                        continue
                einb_kind = entity.get("_einb_kind") or ("upper_cabinet" if entity.get("_upper_cabinet") else None)
                label = str(entity.get("_label") or einb_kind or layer)
                semantic = str(entity.get("_semantic_label") or label)
                kind_name = einb_kind
                if entity.get("_instance_id") is not None:
                    instance_id = int(entity["_instance_id"])
            else:
                continue
            if footprint is None or footprint.is_empty or footprint.area <= 0.0:
                continue
            ring = _points(list(footprint.exterior.coords)[:-1])
            z_min = z_max = confidence = None
            z_source = "default"
            if instance_id is not None and instance_id in z_ranges:
                z_min, z_max, confidence = z_ranges[instance_id]
                z_source = "instance_bbox"
            default_z = [float(value) for value in config["defaults"]["upper_cabinet_z_m"]] if layer == "A-OBEN" else None
            catalogue_flags = entity.get("_catalogue") or {}
            measured_height = entity.get("_height_m")
            if measured_height is not None and default_z is None and kind_name in ("tall_unit",):
                default_height_override = float(measured_height)
            else:
                default_height_override = None
            if kind_name in ("KITCHEN_SINK", "COOKTOP") and catalogue_flags.get("on_kitchen_run"):
                # On the worktop: the sink basin sits flush in it, the hob lies on it.
                top = float(kitchen_cfg["kitchen_worktop_top_m"])
                default_z = [float(kitchen_cfg["kitchen_sink_bottom_m"]), top] if kind_name == "KITCHEN_SINK" else [top, top + float(kitchen_cfg["kitchen_hob_height_m"])]
            if z_source == "default" and instances and layer in ("A-OBEN", "A-EINB") and kind_name in ("upper_cabinet", "tall_unit", "builtin_front"):
                hanging = kind_name == "upper_cabinet"
                match = _measured_instance(
                    footprint, instances, _UPPER_LABELS if hanging else _TALL_LABELS, floor_z, upm,
                    low_m=0.8 if hanging else -0.2, high_m=1.8 if hanging else 0.8,
                )
                if match is not None:
                    # measured from the instance bounding box (an upper cabinet hangs at its measured height)
                    instance_id, z_min, z_max, confidence, z_source = match["id"], match["z_min"], match["z_max"], match["confidence"], "instance_bbox"
                    if not hanging:
                        z_min = min(z_min, floor_z)
                    default_z = None
            result.append(PlanFurniture(
                id=f"F{len(result) + 1:03d}", instance_id=instance_id, label=label, semantic_label=semantic,
                catalogue_kind=kind_name, layer=layer, footprint=ring, z_min=z_min, z_max=z_max, z_source=z_source,
                confidence=confidence, default_height_m=None if default_z else (default_height_override if default_height_override is not None else _default_height(config, kind_name, semantic, einb_kind)),
                default_z_m=default_z,
            ))
    return result


def build_plan_model(
    *,
    project_name: str,
    level: Any,
    frame: Any,
    units_per_metre: float,
    layers: dict[str, list[dict[str, Any]]],
    openings: list[Any],
    rooms: list[tuple[Polygon, str]],
    wall_pieces: list[dict[str, Any]] | None = None,
    wall_bodies: list[Polygon] | None = None,
    result_fingerprint: str = "",
    instances_path: Path | str | None = None,
    config: dict[str, Any] | None = None,
    source_extra: dict[str, Any] | None = None,
) -> BuildingPlan:
    """Assemble the BuildingPlan from the objects that also produce the DXF entities.

    ``layers`` are the FINAL DXF entity layers, ``openings`` the final Opening objects
    (after double-wall correction), ``rooms`` the kept ``(polygon, name)`` pairs that become
    A-AREA, ``wall_pieces`` ``DoubleWallResult.pieces`` (None -> single-line walls from A-WALL).
    """
    upm = max(float(units_per_metre), 1e-9)
    cfg = config or load_bim_export_config()
    bodies = list(wall_bodies or [])
    if wall_pieces is not None:
        walls = _walls_from_pieces(wall_pieces, bodies)
    else:
        walls = _walls_from_lines(layers.get("A-WALL", []))
        bodies = []
    plan_rooms = [
        PlanRoom(
            id=f"R{index:03d}", name=name, polygon=_polygon_dict(polygon), area_m2=float(polygon.area) / (upm * upm),
            label_point=[float(polygon.representative_point().x), float(polygon.representative_point().y)],
        )
        for index, (polygon, name) in enumerate(rooms, start=1)
    ]
    return BuildingPlan(
        project_name=project_name, level_id=str(level.id), level_name=str(level.name), units_per_metre=upm,
        frame={"origin": [float(v) for v in frame.origin], "scale_factor": float(frame.scale_factor), "rotation": np.asarray(frame.rotation, dtype=float).tolist()},
        floor_z=float(level.floor_z), ceiling_z=None if level.ceiling_z is None else float(level.ceiling_z),
        walls=walls, rooms=plan_rooms, openings=_openings(openings, walls, upm, cfg),
        furniture=_furniture(
            layers, load_instance_z_ranges(instances_path, frame), cfg, load_instance_hulls(instances_path, frame), float(level.floor_z), upm,
        ),
        wall_bodies=[_polygon_dict(body) for body in bodies], source={"result_fingerprint": result_fingerprint, **(source_extra or {})},
    )


def plan_model_is_current(cache_directory: Path | str) -> bool:
    """True when a readable plan_model.json of the current schema exists next to result.json."""
    path = plan_model_path(cache_directory)
    if not path.is_file():
        return False
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("schema_version", 0)) == SCHEMA_VERSION
    except (OSError, ValueError):
        return False
