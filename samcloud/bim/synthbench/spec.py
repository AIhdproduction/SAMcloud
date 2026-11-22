"""Scene specification of the synthetic indoor bench (ground truth, metres, scene frame).

A scene is described by explicit wall centre lines with thickness; rooms are the closed free regions between the
walls (derived exactly in :mod:`samcloud.bim.synthbench.geometry`).  Openings, niches and projections modify the wall
solid; furniture, columns, beams and stairs are independent elements.  Everything is in the *scene frame*; the scan
is delivered in a *world frame* ``world = R(world_yaw) @ scene + world_offset`` so the pipeline has to find the
alignment itself (as with real 360 scans).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

# Indoor class indices of config/classes.json ("las_codes" -> Cyclone codes 1, 180, 100, ...).
CLASS_UNKNOWN = 0
CLASS_CLUTTER = 1
CLASS_WALL = 2
CLASS_CEILING = 3
CLASS_FLOOR = 4
CLASS_DOOR = 5
CLASS_WINDOW = 6
CLASS_STAIRS = 7
CLASS_RAILING = 8
CLASS_COLUMN = 9
CLASS_BEAM = 10
CLASS_HVAC = 15
CLASS_CEILING_EQUIPMENT = 16
CLASS_FURNITURE = 17
CLASS_APPLIANCES = 18
CLASS_SANITARY = 19

CLASS_NAMES = {
    0: "unknown", 1: "loose items", 2: "wall", 3: "ceiling", 4: "floor", 5: "door", 6: "window", 7: "stairs",
    8: "railing", 9: "column", 10: "beam", 11: "steel structure", 12: "pipe", 13: "cable", 14: "electrical",
    15: "hvac", 16: "ceiling equipment", 17: "furniture", 18: "appliances", 19: "sanitary fixtures",
}


@dataclass(slots=True)
class WallSpec:
    id: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    thickness: float
    exterior: bool = False

    @property
    def length(self) -> float:
        return float(math.dist(self.p0, self.p1))

    def direction(self) -> np.ndarray:
        d = np.subtract(self.p1, self.p0).astype(float)
        return d / max(np.linalg.norm(d), 1e-12)

    def normal(self) -> np.ndarray:
        d = self.direction()
        return np.array([-d[1], d[0]])


@dataclass(slots=True)
class OpeningSpec:
    id: str
    kind: str  # door | window | passage
    wall_id: str
    offset: float  # centre distance from wall.p0 along the wall (m)
    width: float
    sill: float
    head: float
    swing: int = 1  # door leaf opens to +normal (1) or -normal (-1) of the host wall
    hinge: str = "left"  # left | right (seen along +direction)
    leaf_angle_deg: float = 85.0
    instance_id: int = 0


@dataclass(slots=True)
class CutSpec:
    """A 2D polygon removed from the wall solid between z0 and z1 (niche, recess)."""

    id: str
    polygon: list[tuple[float, float]]
    z0: float
    z1: float


@dataclass(slots=True)
class ProjectionSpec:
    """A 2D polygon added to the wall solid over the full storey (pilaster, wall projection)."""

    id: str
    polygon: list[tuple[float, float]]


@dataclass(slots=True)
class RoomSeed:
    name: str
    point: tuple[float, float]
    ceiling_h: float


@dataclass(slots=True)
class Part:
    """One primitive of a furniture item in the item frame (origin at the floor below the item centre)."""

    shape: str  # box | cyl | sphere
    center: tuple[float, float, float]
    size: tuple[float, float, float]  # box: dx, dy, dz ; cyl: radius, radius, height ; sphere: radius, radius, radius
    yaw_deg: float = 0.0  # extra rotation of the part about its own centre


@dataclass(slots=True)
class FurnitureSpec:
    id: str
    label: str  # SAM3 instance label vocabulary
    class_idx: int
    center: tuple[float, float]
    yaw_deg: float
    parts: list[Part]
    instance_id: int = 0
    variant: str = ""


@dataclass(slots=True)
class ColumnSpec:
    id: str
    center: tuple[float, float]
    shape: str  # box | cyl
    size: tuple[float, float]  # box: dx, dy ; cyl: radius, radius
    yaw_deg: float = 0.0
    instance_id: int = 0


@dataclass(slots=True)
class BeamSpec:
    id: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    width: float
    depth: float  # drop below the ceiling of the room it is in


@dataclass(slots=True)
class StairSpec:
    id: str
    start: tuple[float, float]  # centre of the first riser line
    yaw_deg: float  # walking direction
    width: float
    steps: int
    rise: float
    run: float


@dataclass(slots=True)
class ScanSpec:
    noise_m: float = 0.006
    voxel_m: float = 0.018
    angular_step_deg: float = 0.40
    station_spacing_m: float = 2.4
    max_range_m: float = 14.0
    dropout_patches: int = 0
    dropout_radius_m: tuple[float, float] = (0.2, 0.5)
    label_noise: float = 0.0
    window_return: float = 0.35  # share of rays that return a point on glass
    unvisited_rooms: list[str] = field(default_factory=list)
    unscanned: list[list[tuple[float, float]]] = field(default_factory=list)
    door_stations: bool = True
    photos_per_station: int = 4
    photo_size: tuple[int, int] = (480, 360)


@dataclass(slots=True)
class SceneSpec:
    name: str
    family: str
    seed: int
    storey_height: float
    walls: list[WallSpec]
    openings: list[OpeningSpec] = field(default_factory=list)
    rooms: list[RoomSeed] = field(default_factory=list)
    cuts: list[CutSpec] = field(default_factory=list)
    projections: list[ProjectionSpec] = field(default_factory=list)
    furniture: list[FurnitureSpec] = field(default_factory=list)
    columns: list[ColumnSpec] = field(default_factory=list)
    beams: list[BeamSpec] = field(default_factory=list)
    stairs: list[StairSpec] = field(default_factory=list)
    scan: ScanSpec = field(default_factory=ScanSpec)
    world_yaw_deg: float = 0.0
    world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    features: list[str] = field(default_factory=list)

    def wall(self, wall_id: str) -> WallSpec:
        for wall in self.walls:
            if wall.id == wall_id:
                return wall
        raise KeyError(wall_id)

    # -- frames ----------------------------------------------------------------------------------------------------
    def world_rotation(self) -> np.ndarray:
        a = math.radians(self.world_yaw_deg)
        return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])

    def to_world(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        return pts @ self.world_rotation().T + np.asarray(self.world_offset, dtype=np.float64)

    def to_scene(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        return (pts - np.asarray(self.world_offset, dtype=np.float64)) @ self.world_rotation()

    # -- serialisation ---------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneSpec":
        def build(kind: Any, items: list[dict[str, Any]]) -> list[Any]:
            return [_make(kind, item) for item in items]

        values = dict(data)
        values["walls"] = build(WallSpec, data.get("walls", []))
        values["openings"] = build(OpeningSpec, data.get("openings", []))
        values["rooms"] = build(RoomSeed, data.get("rooms", []))
        values["cuts"] = build(CutSpec, data.get("cuts", []))
        values["projections"] = build(ProjectionSpec, data.get("projections", []))
        values["furniture"] = [
            _make(FurnitureSpec, {**item, "parts": [_make(Part, part) for part in item.get("parts", [])]})
            for item in data.get("furniture", [])
        ]
        values["columns"] = build(ColumnSpec, data.get("columns", []))
        values["beams"] = build(BeamSpec, data.get("beams", []))
        values["stairs"] = build(StairSpec, data.get("stairs", []))
        values["scan"] = _make(ScanSpec, data.get("scan", {}))
        return _make(cls, values)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1, sort_keys=True)


def _make(kind: Any, data: dict[str, Any]) -> Any:
    names = {f.name for f in fields(kind)}
    values = {}
    for key, value in data.items():
        if key not in names:
            continue
        values[key] = _tuplify(value) if key in {"p0", "p1", "center", "size", "point", "start", "world_offset", "dropout_radius_m", "photo_size"} else value
        if key in {"polygon"}:
            values[key] = [tuple(v) for v in value]
        if key == "unscanned":
            values[key] = [[tuple(v) for v in poly] for poly in value]
    return kind(**values)


def _tuplify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        return round(value, 6)
    return value
