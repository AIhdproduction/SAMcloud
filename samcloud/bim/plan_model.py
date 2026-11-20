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
from shapely.ops import unary_union

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
    # Optional measured sub-solids.  Each entry carries a local-plan footprint and
    # local z range; together they cover ``footprint`` without changing the DXF outline.
    parts: list[dict[str, Any]] = field(default_factory=list)
    # True when depth evidence found furniture-class voxels at floor level inside the footprint (stands on the floor); None = unknown / no depth
    floor_contact: bool | None = None
    # Sanitary block (WC, washbasin): direction the front looks (degrees counter-clockwise from +X of the local plan) and the wall
    # whose face carries its back.  Same values for the DXF block and the IFC property set; None = no facing rule applied.
    facing_deg: float | None = None
    back_wall_id: str | None = None

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


_TALL_COLUMN_LABELS = frozenset({"chest of drawers", "dresser", "cupboard"})
_UPPER_LABELS = frozenset({"kitchen cabinet", "cabinet", "kitchen counter", "shelf", "bookcase"})
_TALL_LABELS = frozenset({"refrigerator", "fridge", "kitchen cabinet", "cabinet", "wardrobe", "oven", "microwave oven"})
_WALL_MOUNTED_KINDS = frozenset({"upper_cabinet", "washbasin", "lavabo", "wall shelf", "hood", "wall tv", "mirror cabinet"})


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
    footprints: Any = None, depth: Any = None, depth_report: dict[str, Any] | None = None,
    row_points: np.ndarray | None = None, wall_union: Any = None, wall_lines: list[LineString] | None = None, row_settings: Any = None,
) -> list[PlanFurniture]:
    result: list[PlanFurniture] = []
    instances = instances or []
    upm = max(float(units_per_metre), 1e-9)
    kitchen_cfg = config["defaults"]
    seen_runs: set[tuple[tuple[float, float], ...]] = set()
    kitchen_legs: list[Polygon] = []
    for layer in FURNITURE_LAYERS:
        for entity in layers.get(layer, []):
            outline = entity.get("_counter_outline") or (entity.get("points") if entity.get("_run_legs") and entity.get("closed") else None)
            if outline and len(outline) >= 3:
                # A U/L run is represented by its measured rectangular legs, not its
                # enclosing outline.  The latter would incorrectly turn the free
                # centre of the kitchen into a solid slab.
                leg_outlines = entity.get("_run_legs") or [outline]
                emitted = False
                for leg_outline in leg_outlines:
                    if not leg_outline or len(leg_outline) < 3:
                        continue
                    key = tuple((round(float(x), 3), round(float(y), 3)) for x, y in leg_outline)
                    if key in seen_runs:
                        continue
                    seen_runs.add(key)
                    run = Polygon(leg_outline)
                    run = run if run.is_valid else run.buffer(0)
                    if run.is_empty or run.geom_type != "Polygon":
                        continue
                    emitted = True
                    kitchen_legs.append(run)
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
                if emitted:
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
            measured_band = entity.get("_z_range_m")
            if measured_band and layer == "A-OBEN" and z_source == "default":
                # An upper cabinet hangs at the robust (p5..p95) band of its own points, not at a loose instance bounding box.
                z_min, z_max = floor_z + float(measured_band[0]) * upm, floor_z + float(measured_band[1]) * upm
                z_source, default_z = "points", None
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
            # Bounding boxes often omit the lower, occluded portion of a
            # standing object.  Persist its physically plausible floor origin
            # in the plan model; only explicitly wall-mounted kinds may float.
            mounted = layer == "A-OBEN" or any(str(value or "").strip().lower() in _WALL_MOUNTED_KINDS for value in (kind_name, semantic, label))
            if z_source == "instance_bbox" and z_min is not None and not mounted:
                z_min = min(float(z_min), floor_z)
            result.append(PlanFurniture(
                id=f"F{len(result) + 1:03d}", instance_id=instance_id, label=label, semantic_label=semantic,
                catalogue_kind=kind_name, layer=layer, footprint=ring, z_min=z_min, z_max=z_max, z_source=z_source,
                confidence=confidence, default_height_m=None if default_z else (default_height_override if default_height_override is not None else _default_height(config, kind_name, semantic, einb_kind)),
                default_z_m=default_z,
                parts=_parts_matching_footprint(entity.get("_solid_parts"), footprint, units_per_metre=upm),
                facing_deg=None if entity.get("_facing_deg") is None else float(entity["_facing_deg"]),
            ))
    # A tall kitchen appliance may be visible only as an instance front, while
    # the 2D kitchen run has no matching catalogue block.  Add it only when
    # measured tall evidence is immediately next to a measured kitchen leg and
    # no regular furniture object already owns that instance.
    if kitchen_legs:
        kitchen_area = unary_union(kitchen_legs).buffer(0.35 * upm)
        represented = {item.instance_id for item in result if item.instance_id is not None}
        for item in sorted(instances, key=lambda value: int(value["id"])):
            # An instance bounding box is a loose world AABB (a diamond in the plan): with instance points the hull is the tight,
            # wall-aligned footprint.  A "chest of drawers" as high as a kitchen column next to a kitchen run is an oven tower.
            candidate_label = item["label"] in _TALL_LABELS or item["label"] in _TALL_COLUMN_LABELS
            tight = footprints.footprint(item["id"]) if footprints is not None and candidate_label else None
            if item["id"] in represented or not (item["label"] in _TALL_LABELS or (tight is not None and item["label"] in _TALL_COLUMN_LABELS)):
                continue
            bottom = (float(item["z_min"]) - floor_z) / upm
            top = (float(item["z_max"]) - floor_z) / upm
            hull = tight if tight is not None else item["hull"]
            if bottom > 0.80 or top < float(kitchen_cfg.get("kitchen_tall_min_top_m", 1.80)) or hull.distance(kitchen_area) > 1e-9:
                continue
            # A tall unit is one narrow cabinet column: a merged instance (hood + oven + counter) is wider than that.
            rectangle = hull.minimum_rotated_rectangle
            if rectangle.geom_type == "Polygon":
                corners = list(rectangle.exterior.coords)
                narrow = min(math.dist(corners[0], corners[1]), math.dist(corners[1], corners[2])) / upm
                if narrow > float(kitchen_cfg.get("kitchen_tall_max_width_m", 1.0)):
                    continue
            result.append(PlanFurniture(
                id=f"F{len(result) + 1:03d}", instance_id=int(item["id"]), label="Tall unit", semantic_label="kitchen cabinet",
                catalogue_kind="tall_unit", layer="A-EINB", footprint=_points(list(hull.exterior.coords)[:-1]),
                z_min=min(float(item["z_min"]), floor_z), z_max=float(item["z_max"]), z_source="instance_bbox", confidence=item["confidence"],
            ))
    if footprints is not None:
        _complete_thin_slabs(result, footprints, upm, floor_z, depth_report)
    if kitchen_legs:
        _square_kitchen_inserts(result, kitchen_legs, upm, depth_report)
        if row_points is not None and len(row_points):
            result = _apply_tall_rows(result, kitchen_legs, row_points, wall_union, wall_lines or [], footprints, upm, floor_z, _row_settings(config, row_settings), depth_report)
    if depth is not None:
        result = _apply_depth_rules(result, depth, upm, floor_z, depth_report)
    return result


def _assign_back_walls(items: list[PlanFurniture], walls: list[PlanWall], upm: float) -> None:
    """``back_wall_id`` of every item with a facing: the wall face nearest to the midpoint of the footprint side opposite its front."""
    faces = [(wall.id, LineString(np.asarray(face, dtype=float))) for wall in walls for face in (wall.face_a, wall.face_b) if face and len(face) >= 2]
    if not faces:
        return
    for item in items:
        if item.facing_deg is None:
            continue
        polygon = _footprint_polygon(item)
        if polygon is None:
            continue
        front = np.asarray((math.cos(math.radians(item.facing_deg)), math.sin(math.radians(item.facing_deg))))
        coords = np.asarray(polygon.exterior.coords[:-1], dtype=float)
        extent = coords @ front
        back = Point(*(coords[np.argsort(extent)[:2]].mean(axis=0)))
        item.back_wall_id = min(faces, key=lambda entry: entry[1].distance(back))[0]


def _row_settings(config: dict[str, Any], settings: Any) -> Any:
    """Tall-row settings: the explicit object, else ``kitchen_rows:`` of bim_export.yaml, else the defaults."""
    from samcloud.bim.kitchen_rows import KitchenRowSettings

    if settings is not None:
        return settings
    section = config.get("kitchen_rows") or {}
    return KitchenRowSettings(**{key: value for key, value in section.items() if key in KitchenRowSettings.__dataclass_fields__})


_FRIDGE_NAMES = frozenset({"refrigerator", "fridge", "k\u00fchlschrank", "kuehlschrank"})


def _is_fridge(item: PlanFurniture) -> bool:
    return any(str(value or "").strip().lower() in _FRIDGE_NAMES for value in (item.label, item.semantic_label, item.catalogue_kind))


def _leg_angle(leg: Polygon) -> float:
    rectangle = leg.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        return 0.0
    corners = list(rectangle.exterior.coords)
    first, second = np.asarray(corners[1]) - np.asarray(corners[0]), np.asarray(corners[2]) - np.asarray(corners[1])
    edge = first if np.hypot(*first) >= np.hypot(*second) else second
    return float(math.atan2(edge[1], edge[0]))


def _square_kitchen_inserts(items: list[PlanFurniture], legs: list[Polygon], upm: float, report: dict[str, Any] | None) -> None:
    """A sink / hob lies inside its kitchen run and is square to it: a diagonal or protruding insert is clipped to the leg it stands on."""
    from shapely import affinity
    from shapely.geometry import box as _box

    union = unary_union(legs).buffer(0.01 * upm)
    for item in items:
        if item.catalogue_kind not in ("KITCHEN_SINK", "COOKTOP") or item.layer == "A-OBEN":
            continue
        polygon = _footprint_polygon(item)
        if polygon is None:
            continue
        best = max(legs, key=lambda leg: polygon.intersection(leg).area)
        angle = _leg_angle(best)
        local = affinity.rotate(polygon, -angle, origin=(0.0, 0.0), use_radians=True)
        x0, y0, x1, y1 = local.bounds
        straight = abs(local.area - (x1 - x0) * (y1 - y0)) <= 0.01 * max(local.area, 1e-12)
        if straight and union.covers(polygon):
            continue
        clipped = polygon.intersection(best)
        if clipped.is_empty or clipped.area < 0.35 * polygon.area:
            if report is not None:
                report.setdefault("kitchen_inserts_unsquared", []).append({"id": item.id, "label": item.label})
            continue
        cx0, cy0, cx1, cy1 = affinity.rotate(clipped, -angle, origin=(0.0, 0.0), use_radians=True).bounds
        squared = affinity.rotate(_box(cx0, cy0, cx1, cy1), angle, origin=(0.0, 0.0), use_radians=True).intersection(best)
        if squared.is_empty or squared.geom_type != "Polygon":
            continue
        item.footprint = _points(list(squared.exterior.coords)[:-1])
        if report is not None:
            report.setdefault("kitchen_inserts_squared", []).append({"id": item.id, "label": item.label})


def _apply_tall_rows(
    items: list[PlanFurniture], legs: list[Polygon], points: np.ndarray, wall_union: Any, wall_lines: list[LineString], footprints: Any,
    upm: float, floor_z: float, settings: Any, report: dict[str, Any] | None,
) -> list[PlanFurniture]:
    """Tall kitchen rows measured from the points: fridge / oven tower / tall cabinet from the front face back to the wall.

    An existing fridge keeps its own item and height but takes the row's depth and its along-range; every other item that stands in
    the row (its footprint or its instance points mostly inside) is replaced by the row's units.
    """
    from shapely import contains_xy

    from samcloud.bim.kitchen_rows import detect_tall_rows, dominant_angle, split_row

    if not settings.enabled:
        return items
    area = unary_union(legs)
    angle = dominant_angle(wall_lines, (area.centroid.x, area.centroid.y)) if wall_lines else 0.0
    rows = detect_tall_rows(points, wall_union, area, upm, angle, settings, area)
    if not rows:
        return items
    hob_legs: list[Polygon] = []
    for item in items:
        polygon = _footprint_polygon(item) if item.catalogue_kind == "COOKTOP" else None
        if polygon is not None:
            hob_legs.extend(leg for leg in legs if leg.distance(polygon) <= 0.05 * upm)
    hob_area = unary_union(hob_legs) if hob_legs else None
    protected = {"kitchen_base", "worktop", "KITCHEN_SINK", "COOKTOP", "upper_cabinet"}
    next_number = max([int(item.id[1:]) for item in items if item.id[1:].isdigit()] or [0]) + 1
    oven_assigned = False
    produced: list[dict[str, Any]] = []
    for row in rows:
        grown = row.polygon.buffer(0.10 * upm)
        owned: list[PlanFurniture] = []
        fixed: list[PlanFurniture] = []
        for item in items:
            if item.layer == "A-OBEN" or item.catalogue_kind in protected:
                continue
            polygon = _footprint_polygon(item)
            inside = 0.0 if polygon is None else polygon.intersection(grown).area / max(polygon.area, 1e-12)
            if inside < 0.5 and footprints is not None and footprints.has(item.instance_id):
                member = footprints.points[int(item.instance_id)]
                inside = float(contains_xy(grown, member[:, 0], member[:, 1]).mean())
            if inside >= 0.5:
                (fixed if _is_fridge(item) else owned).append(item)
        for item in owned:
            items.remove(item)
        usable = [(item, _footprint_polygon(item)) for item in fixed]
        usable = [(item, polygon) for item, polygon in usable if polygon is not None]
        units = split_row(row.polygon, [polygon for _, polygon in usable], upm, settings.min_unit_m)
        top = floor_z + row.top_m * upm
        unit_ids: list[str] = []
        for polygon, index in units:
            ring = _points(list(polygon.exterior.coords)[:-1])
            if index is not None:
                usable[index][0].footprint = ring
                unit_ids.append(usable[index][0].id)
                continue
            oven = False
            if hob_area is not None and not oven_assigned and polygon.distance(hob_area) <= 0.15 * upm:
                oven = oven_assigned = True
            item = PlanFurniture(
                id=f"F{next_number:03d}", instance_id=None, label="Backofen" if oven else "Hochschrank", semantic_label="oven tower" if oven else "tall unit",
                catalogue_kind="oven_tower" if oven else "tall_unit", layer="A-EINB", footprint=ring, z_min=floor_z, z_max=top, z_source="points",
                floor_contact=True,
            )
            next_number += 1
            items.append(item)
            unit_ids.append(item.id)
        produced.append({
            "units": unit_ids, "bbox_m": [round(float(v) / upm, 3) for v in row.polygon.bounds], "top_m": round(row.top_m, 3), "front": row.front,
            "replaced": [{"id": item.id, "label": item.label, "instance_id": item.instance_id} for item in owned],
        })
    new_ids = {unit for entry in produced for unit in entry["units"]}
    row_union = unary_union([row.polygon for row in rows])
    trimmed: list[str] = []
    for item in list(items):
        if item.id in new_ids or item.layer == "A-OBEN" or item.catalogue_kind != "tall_unit":
            continue
        polygon = _footprint_polygon(item)
        if polygon is None or polygon.intersection(row_union).area <= 0.02 * polygon.area:
            continue
        rectangle: Polygon | None = polygon
        for row in sorted(rows, key=lambda candidate: -polygon.intersection(candidate.polygon).area):
            if rectangle is not None and rectangle.intersection(row.polygon).area > 0.02 * rectangle.area:
                rectangle = _largest_free_rectangle(rectangle, row.polygon, settings.min_unit_m * upm)
        if rectangle is None:
            items.remove(item)
        else:
            item.footprint = _points(list(rectangle.exterior.coords)[:-1])
        trimmed.append(item.id)
    if report is not None:
        report["kitchen_rows"] = produced
        if trimmed:
            report["kitchen_rows_trimmed_neighbours"] = trimmed
    return items


def _largest_free_rectangle(polygon: Polygon, blocked: Any, minimum: float) -> Polygon | None:
    """The biggest rectangle (in the frame of ``polygon``'s minimum rectangle) of ``polygon`` that keeps clear of ``blocked``; None when it is too small."""
    from shapely import affinity
    from shapely.geometry import box as _box

    angle = _leg_angle(polygon)
    local = affinity.rotate(polygon, -angle, origin=(0.0, 0.0), use_radians=True)
    x0, y0, x1, y1 = local.bounds
    cut = affinity.rotate(polygon.intersection(blocked), -angle, origin=(0.0, 0.0), use_radians=True)
    if cut.is_empty:
        return polygon
    cx0, cy0, cx1, cy1 = cut.bounds
    candidates = [(x0, y0, cx0, y1), (cx1, y0, x1, y1), (x0, y0, x1, cy0), (x0, cy1, x1, y1)]
    best: tuple[float, Polygon] | None = None
    for a, b, c, d in candidates:
        if c - a < minimum or d - b < minimum:
            continue
        rectangle = affinity.rotate(_box(a, b, c, d), angle, origin=(0.0, 0.0), use_radians=True).intersection(polygon)
        if isinstance(rectangle, Polygon) and not rectangle.is_empty and (best is None or rectangle.area > best[0]):
            best = (rectangle.area, rectangle)
    return None if best is None else best[1]


def _drop_unmounted_uppers(items: list[PlanFurniture], walls: list[PlanWall], upm: float, maximum_m: float, report: dict[str, Any] | None) -> list[PlanFurniture]:
    """Upper cabinets hang on a wall: drop one with no wall face / centreline within ``maximum_m`` of its footprint in the FINAL plan."""
    lines = [LineString(np.asarray(points, dtype=float)) for wall in walls for points in (wall.face_a, wall.face_b, wall.centreline) if points and len(points) >= 2]
    if not lines:
        return items
    kept: list[PlanFurniture] = []
    for item in items:
        polygon = _footprint_polygon(item) if item.layer == "A-OBEN" else None
        if polygon is not None and min(line.distance(polygon) for line in lines) > maximum_m * upm:
            if report is not None:
                report.setdefault("uppers_dropped_no_wall", []).append({"id": item.id, "label": item.label})
            continue
        kept.append(item)
    return kept


def _footprint_polygon(item: PlanFurniture) -> Polygon | None:
    try:
        polygon = Polygon(item.footprint)
        polygon = polygon if polygon.is_valid else polygon.buffer(0)
    except (ValueError, TypeError):
        return None
    return polygon if isinstance(polygon, Polygon) and not polygon.is_empty else None


def _complete_thin_slabs(items: list[PlanFurniture], footprints: Any, upm: float, floor_z: float, report: dict[str, Any] | None) -> None:
    """A tall standing slab (a fridge seen only from the front) takes the measured, wall-completed depth of its instance points."""
    from samcloud.bim.furniture_footprint import narrow_side_m

    settings = footprints.settings
    for item in items:
        if item.instance_id is None or item.layer == "A-OBEN" or item.parts or not footprints.has(item.instance_id):
            continue
        polygon = _footprint_polygon(item)
        if polygon is None or narrow_side_m(polygon, upm) >= settings.thin_depth_m:
            continue
        top = (float(item.z_max) - floor_z) / upm if item.z_max is not None else float(item.default_height_m or 0.0)
        if top < 1.0:
            continue
        tight = footprints.footprint(item.instance_id)
        if tight is None or narrow_side_m(tight, upm) < 0.40 or polygon.intersection(tight).area < 0.5 * polygon.area:
            continue
        item.footprint = _points(list(tight.exterior.coords)[:-1])
        if report is not None:
            report.setdefault("thin_slabs_completed", []).append({"id": item.id, "label": item.label, "depth_m": round(narrow_side_m(tight, upm), 3)})


def _apply_depth_rules(items: list[PlanFurniture], depth: Any, upm: float, floor_z: float, report: dict[str, Any] | None) -> list[PlanFurniture]:
    """Depth evidence on furniture: seen-empty space rejects boxes, floor-level voxels confirm floor contact."""
    contact_z = float(depth.settings["plan_floor_contact_z_m"]) * upm
    kept: list[PlanFurniture] = []
    furniture_points, _ = depth.furniture_points(colmap_only=False)
    low = furniture_points[furniture_points[:, 2] <= contact_z]
    for item in items:
        polygon = _footprint_polygon(item)
        if polygon is None:
            kept.append(item)
            continue
        z0 = (float(item.z_min) - floor_z) if item.z_min is not None else (float(item.default_z_m[0]) * upm if item.default_z_m else 0.0)
        z1 = (float(item.z_max) - floor_z) if item.z_max is not None else (float(item.default_z_m[1]) * upm if item.default_z_m else float(item.default_height_m or 0.0) * upm)
        if item.layer != "A-OBEN":
            z0 = max(z0, 0.05 * upm)
        share = depth.free_share(polygon, z0 + 0.03 * upm, z1 - 0.03 * upm) if z1 - z0 > 0.10 * upm else 0.0
        if report is not None:
            report.setdefault("free_share", {})[item.id] = round(share, 3)
        if share >= float(depth.settings["plan_empty_share"]):
            if report is not None:
                report.setdefault("rejected_in_free_space", []).append({"id": item.id, "label": item.label, "free_share": round(share, 3)})
            continue
        if len(low) and item.layer != "A-OBEN":
            item.floor_contact = bool(int(depth.points_inside(polygon.buffer(0.03 * upm), low).sum()) >= 8)
        kept.append(item)
    return kept


def _parts_matching_footprint(parts: Any, footprint: Polygon, minimum_iou: float = 0.70, units_per_metre: float = 1.0) -> list[dict[str, Any]]:
    """Keep measured solid parts only when together they cover (about) the exported footprint.

    A catalogue insert can be placed slightly differently from the fitted outline the parts were measured in;
    mismatching parts would then draw a different body than the DXF footprint, so one plain box is kept instead.
    """
    items = [dict(part) for part in (parts or []) if isinstance(part, dict) and len(part.get("footprint") or []) >= 3]
    if not items:
        return []
    try:
        union = unary_union([Polygon(part["footprint"]).buffer(0) for part in items]).buffer(0)
        overlap = union.intersection(footprint).area / max(union.union(footprint).area, 1e-12)
    except Exception:  # noqa: BLE001 - unusable measured geometry falls back to a single box
        return []
    if overlap < minimum_iou:
        return []
    # The parts become exact tiles of the exported footprint: unseen cells and cells under objects lying on the item are interpolated
    # from the nearest measured level, so the solids have no holes and no gaps.
    from samcloud.bim.furniture_parts import tile_parts_to_footprint

    return tile_parts_to_footprint(items, footprint, units_per_metre)


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
    instance_points: dict[int, np.ndarray] | None = None,
    tight_settings: Any = None,
    depth: Any = None,
    class_points: np.ndarray | None = None,
    row_settings: Any = None,
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
    footprints = None
    wall_lines = [LineString(np.asarray(face, dtype=float)) for wall in walls for face in (wall.face_a, wall.face_b) if face and len(face) >= 2]
    wall_union = unary_union([Polygon(body.exterior) for body in bodies]) if bodies else (unary_union(wall_lines).buffer(0.01 * upm) if wall_lines else None)
    if instance_points:
        from samcloud.bim.furniture_footprint import InstanceFootprints, TightFootprintSettings

        footprints = InstanceFootprints(instance_points, wall_lines, wall_union, upm, tight_settings or TightFootprintSettings())
    depth_report: dict[str, Any] = {}
    furniture = _furniture(
        layers, load_instance_z_ranges(instances_path, frame), cfg, load_instance_hulls(instances_path, frame), float(level.floor_z), upm,
        footprints=footprints, depth=depth, depth_report=depth_report,
        row_points=class_points, wall_union=wall_union, wall_lines=wall_lines, row_settings=row_settings,
    )
    _assign_back_walls(furniture, walls, upm)
    furniture = _drop_unmounted_uppers(furniture, walls, upm, float(cfg["defaults"].get("upper_cabinet_wall_max_m", 0.30)), depth_report)
    extra = {**(source_extra or {})}
    if depth_report:
        extra["depth_furniture"] = depth_report
    return BuildingPlan(
        project_name=project_name, level_id=str(level.id), level_name=str(level.name), units_per_metre=upm,
        frame={"origin": [float(v) for v in frame.origin], "scale_factor": float(frame.scale_factor), "rotation": np.asarray(frame.rotation, dtype=float).tolist()},
        floor_z=float(level.floor_z), ceiling_z=None if level.ceiling_z is None else float(level.ceiling_z),
        walls=walls, rooms=plan_rooms, openings=_openings(openings, walls, upm, cfg),
        furniture=furniture,
        wall_bodies=[_polygon_dict(body) for body in bodies], source={"result_fingerprint": result_fingerprint, **extra},
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
