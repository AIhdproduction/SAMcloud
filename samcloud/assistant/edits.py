"""Deterministic edits of a project's ``plan_model`` for the AI assistant (no AI in here).

The assistant (package C) only calls the fixed operations of :data:`OPS`.  A :class:`Proposal` applies them on a deep
copy of the automatic plan; the base plan is never mutated.  Every distance of the API is in metres and converted with
``units_per_metre``; coordinates are plan metres (local plan coordinates / ``units_per_metre``).

Dependencies are computed here, deterministically, never by the AI:

* a wall that moves or changes thickness takes its hosted openings along (``host_wall_id``: centre moved with the wall,
  ``wall_thickness_m`` updated), walls that end on it are extended / trimmed so the joins stay closed, room polygon edges
  that lie on the old wall face (parallel, within tolerance) are offset by the same face displacement (``area_m2`` and
  ``label_point`` recomputed), ``face_a`` / ``face_b`` are rebuilt from centreline +- thickness / 2 and the wall body
  ring of ``wall_bodies`` is rebuilt;
* a deleted wall deletes its hosted openings (reported as extra changes with a note);
* an opening must stay inside its host wall; values are validated (thickness 0.03-1.0 m, width 0.3-6 m, sill 0-2.5 m,
  head above the sill and at most the storey height ...).

``move_wall(offset_m)`` shifts the wall perpendicular to its axis; **positive values move it towards its ``face_a``
side** (when the plan has no ``face_a``, the face_a side is the right-hand side of the centreline direction).
``detected_*`` fields keep the as-detected values and are never changed.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from samcloud.assistant.evidence import FURNITURE_KINDS, MANUAL_LAYER, FurnitureKind, check_evidence, resolve_kind
from samcloud.bim.plan_model import BuildingPlan, PlanFurniture, PlanOpening, PlanRoom, PlanWall

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------------------- limits
THICKNESS_RANGE_M = (0.03, 1.0)
OPENING_WIDTH_RANGE_M = (0.3, 6.0)
SILL_RANGE_M = (0.0, 2.5)
MIN_OPENING_HEIGHT_M = 0.10  # head must be at least this much above the sill
MAX_WALL_MOVE_M = 3.0
MAX_OPENING_SHIFT_M = 6.0
MAX_FURNITURE_MOVE_M = 30.0
MIN_WALL_LENGTH_M = 0.10
OPENING_OVERHANG_M = 0.02  # an opening may reach this far beyond its host wall's ends (or as far as it did before)
DEFAULT_STOREY_HEIGHT_M = 2.6
MAX_ROOM_NAME_LENGTH = 60
FURNITURE_SIZE_RANGE_M = (0.05, 6.0)
# dependency detection
FACE_MATCH_TOLERANCE_M = 0.06  # a room edge lies on a wall face within this distance
FACE_PARALLEL_DEG = 5.0
JOIN_MIN_ANGLE_DEG = 30.0  # a wall joining at a smaller angle is a continuation, not a join
JOIN_END_REACH_M = 0.15  # how far beyond the moved wall's ends a joining wall's end may lie
JOIN_BAND_SLACK_M = 0.05
BACK_WALL_REACH_M = 0.10
# collinear chain: pieces on the same line (within these limits) that touch or are bridged by a hosted opening move together
CHAIN_MAX_ANGLE_DEG = 3.0
CHAIN_OFFSET_M = 0.03
CHAIN_THICKNESS_M = 0.03
CHAIN_TOUCH_M = 0.05
CHAIN_GAP_SLACK_M = 0.10
CHAIN_CONFIRM_PIECES = 3  # a chain of this many pieces always needs an explicit confirmation
# a proposal needs explicit confirmation with any removal or more changed elements than this
CONFIRM_ELEMENT_COUNT = 5

ELEMENT_KINDS = ("wall", "opening", "furniture", "room", "ceiling_object")


class EditError(Exception):
    """A user-readable reason why an edit cannot be applied (English text + i18n ``key`` and ``params``)."""

    def __init__(self, message: str, *, key: str = "assistant.edit.invalid_value", params: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.key = key
        self.params = dict(params or {})

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "key": self.key, "params": dict(self.params)}


@dataclass
class EditOp:
    op: str
    target: str | None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "target": self.target, "params": copy.deepcopy(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditOp":
        return cls(str(data["op"]), None if data.get("target") is None else str(data["target"]), dict(data.get("params") or {}))

    @classmethod
    def from_tool_call(cls, name: str, arguments: dict[str, Any]) -> "EditOp":
        """An op from a tool call whose arguments carry ``target`` next to the parameters (see :func:`op_schemas`)."""
        arguments = dict(arguments or {})
        target = arguments.pop("target", None)
        return cls(str(name), None if target is None else str(target), arguments)


@dataclass
class Change:
    element_id: str
    kind: str  # wall | opening | furniture | room | ceiling_object
    change: str  # added | removed | modified | moved
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"element_id": self.element_id, "kind": self.kind, "change": self.change,
                "before": copy.deepcopy(self.before), "after": copy.deepcopy(self.after), "note": self.note}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Change":
        return cls(str(data["element_id"]), str(data["kind"]), str(data["change"]), data.get("before"), data.get("after"), str(data.get("note") or ""))


# ----------------------------------------------------------------------------------------------------- element access

def _r(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


def _collection(plan: BuildingPlan, kind: str) -> list[Any]:
    return {"wall": plan.walls, "opening": plan.openings, "furniture": plan.furniture, "room": plan.rooms,
            "ceiling_object": plan.ceiling_objects}[kind]


def _element_id(element: Any) -> str:
    return str(element["id"] if isinstance(element, dict) else element.id)


def find_element(plan: BuildingPlan, element_id: str) -> tuple[str, Any] | None:
    """(kind, element) of a plan id in any collection, else None."""
    for kind in ELEMENT_KINDS:
        for element in _collection(plan, kind):
            if _element_id(element) == str(element_id):
                return kind, element
    return None


def _require(plan: BuildingPlan, element_id: str | None, kind: str) -> Any:
    found = None if element_id is None else find_element(plan, element_id)
    if found is None or found[0] != kind:
        raise EditError(f"No {kind} with id {element_id!r} in the plan.", key="assistant.edit.not_found", params={"id": element_id, "kind": kind})
    return found[1]


def _polygon_of(ring: list[list[float]] | None) -> Polygon | None:
    if not ring or len(ring) < 3:
        return None
    try:
        polygon = Polygon([(float(p[0]), float(p[1])) for p in ring])
    except (TypeError, ValueError):
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda g: g.area)
    return polygon if isinstance(polygon, Polygon) and not polygon.is_empty else None


def _room_polygon(room: PlanRoom) -> Polygon | None:
    exterior = room.polygon.get("exterior") or []
    if len(exterior) < 3:
        return None
    polygon = Polygon(exterior, room.polygon.get("holes") or [])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda g: g.area)
    return polygon if isinstance(polygon, Polygon) and not polygon.is_empty else None


def _rect_size(polygon: Polygon) -> tuple[float, float, float]:
    """(long side, short side, angle deg of the long side) of the minimum rotated rectangle (local units)."""
    rectangle = polygon.minimum_rotated_rectangle
    corners = list(rectangle.exterior.coords) if rectangle.geom_type == "Polygon" else [(0, 0), (0, 0), (0, 0)]
    a = math.dist(corners[0], corners[1])
    b = math.dist(corners[1], corners[2])
    if a >= b:
        angle = math.degrees(math.atan2(corners[1][1] - corners[0][1], corners[1][0] - corners[0][0]))
        return a, b, angle % 180.0
    angle = math.degrees(math.atan2(corners[2][1] - corners[1][1], corners[2][0] - corners[1][0]))
    return b, a, angle % 180.0


def furniture_height_m(plan: BuildingPlan, item: PlanFurniture) -> float:
    upm = plan.units_per_metre or 1.0
    if item.z_min is not None and item.z_max is not None and item.z_source in ("instance_bbox", "points"):
        return max(0.0, (float(item.z_max) - float(plan.floor_z)) / upm)
    if item.default_z_m:
        return float(item.default_z_m[1])
    return float(item.default_height_m or 0.0)


def snapshot(plan: BuildingPlan, kind: str, element: Any) -> dict[str, Any]:
    """Compact JSON description of an element in metres (plan metres for positions) for the UI and the AI."""
    upm = plan.units_per_metre or 1.0
    if kind == "wall":
        p0, p1 = np.asarray(element.centreline[0], float) / upm, np.asarray(element.centreline[-1], float) / upm
        return {"id": element.id, "type": "wall", "wall_kind": element.kind,
                "centreline": [[_r(p0[0]), _r(p0[1])], [_r(p1[0]), _r(p1[1])]],
                "thickness_m": _r(element.thickness_m), "length_m": _r(float(np.linalg.norm(p1 - p0))),
                "height_m": None if element.height_m is None else _r(element.height_m)}
    if kind == "opening":
        return {"id": element.id, "type": element.kind, "center": [_r(element.center[0] / upm), _r(element.center[1] / upm)],
                "width_m": _r(element.width_m), "sill_m": _r(element.sill_m), "head_m": _r(element.head_m),
                "wall_thickness_m": _r(element.wall_thickness_m), "host_wall_id": element.host_wall_id,
                "axis_deg": _r(math.degrees(element.axis_angle) % 360.0, 2)}
    if kind == "furniture":
        polygon = _polygon_of(element.footprint)
        long_side, short_side, angle = _rect_size(polygon) if polygon is not None else (0.0, 0.0, 0.0)
        centre = polygon.centroid if polygon is not None else Point(0, 0)
        return {"id": element.id, "type": "furniture", "label": element.label, "semantic_label": element.semantic_label,
                "layer": element.layer, "center": [_r(centre.x / upm), _r(centre.y / upm)],
                "size_m": [_r(long_side / upm), _r(short_side / upm)], "height_m": _r(furniture_height_m(plan, element)),
                "angle_deg": _r(angle, 1), "evidence": (element.fit or {}).get("evidence", "scan")}
    if kind == "room":
        return {"id": element.id, "type": "room", "name": element.name, "area_m2": _r(element.area_m2, 2),
                "label_point": [_r(element.label_point[0] / upm), _r(element.label_point[1] / upm)]}
    polygon = _polygon_of(element.get("footprint"))
    centre = polygon.centroid if polygon is not None else Point(0, 0)
    return {"id": element.get("id"), "type": "ceiling_object", "label": element.get("label"),
            "semantic_label": element.get("semantic_label"), "center": [_r(centre.x / upm), _r(centre.y / upm)]}


# --------------------------------------------------------------------------------------------------------- signatures

SIGNATURE_POSITION_TOLERANCE_M = 0.30
SIGNATURE_SIZE_TOLERANCE = 0.25
SIGNATURE_ANGLE_TOLERANCE_DEG = 12.0
# strict resolution: two candidates whose scores (about metres) differ by less than this are ambiguous
SIGNATURE_AMBIGUITY_MARGIN_M = 0.05


def signature_of(plan: BuildingPlan, element_id: str) -> dict[str, Any] | None:
    """Kind + geometry of an element (plan metres), used to find it again after a pipeline recompute renumbered the ids."""
    found = find_element(plan, element_id)
    if found is None:
        return None
    kind, element = found
    upm = plan.units_per_metre or 1.0
    if kind == "wall":
        p0, p1 = np.asarray(element.centreline[0], float) / upm, np.asarray(element.centreline[-1], float) / upm
        return {"kind": kind, "p0": [_r(p0[0]), _r(p0[1])], "p1": [_r(p1[0]), _r(p1[1])], "thickness_m": _r(element.thickness_m),
                "length_m": _r(float(np.linalg.norm(p1 - p0)))}
    if kind == "opening":
        return {"kind": kind, "type": element.kind, "center": [_r(element.center[0] / upm), _r(element.center[1] / upm)],
                "width_m": _r(element.width_m), "axis_deg": _r(math.degrees(element.axis_angle) % 180.0, 2)}
    if kind == "furniture":
        polygon = _polygon_of(element.footprint)
        if polygon is None:
            return {"kind": kind, "label": element.label, "center": None, "size_m": [0.0, 0.0]}
        long_side, short_side, _angle = _rect_size(polygon)
        return {"kind": kind, "label": element.label, "semantic_label": element.semantic_label,
                "center": [_r(polygon.centroid.x / upm), _r(polygon.centroid.y / upm)], "size_m": [_r(long_side / upm), _r(short_side / upm)]}
    if kind == "room":
        polygon = _room_polygon(element)
        centre = polygon.centroid if polygon is not None else Point(element.label_point)
        return {"kind": kind, "name": element.name, "center": [_r(centre.x / upm), _r(centre.y / upm)], "area_m2": _r(element.area_m2, 2)}
    polygon = _polygon_of(element.get("footprint"))
    centre = polygon.centroid if polygon is not None else Point(0, 0)
    return {"kind": kind, "label": element.get("label"), "center": [_r(centre.x / upm), _r(centre.y / upm)]}


def _size_ok(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1e-6)


def _signature_score(plan: BuildingPlan, kind: str, element: Any, sig: dict[str, Any]) -> float | None:
    """Distance score (lower is better) when ``element`` matches ``sig`` within the tolerances, else None."""
    current = signature_of(plan, _element_id(element))
    if current is None or current.get("kind") != sig.get("kind"):
        return None
    pos = SIGNATURE_POSITION_TOLERANCE_M
    if kind == "wall":
        a0, a1 = np.asarray(sig["p0"]), np.asarray(sig["p1"])
        b0, b1 = np.asarray(current["p0"]), np.asarray(current["p1"])
        da, db = a1 - a0, b1 - b0
        angle = math.degrees(math.acos(min(1.0, abs(float(da @ db)) / max(np.linalg.norm(da) * np.linalg.norm(db), 1e-12))))
        if angle > SIGNATURE_ANGLE_TOLERANCE_DEG:
            return None
        mid_a, mid_b = (a0 + a1) / 2.0, (b0 + b1) / 2.0
        mid = float(np.linalg.norm(mid_a - mid_b))
        if mid > pos + 0.25 * max(float(sig["length_m"]), 0.0):
            return None
        direction = da / max(float(np.linalg.norm(da)), 1e-12)
        across = abs(float((mid_b - a0) @ np.asarray((-direction[1], direction[0]))))
        if across > pos:
            return None
        if not _size_ok(float(sig["length_m"]), float(current["length_m"]), SIGNATURE_SIZE_TOLERANCE):
            return None
        if abs(float(sig["thickness_m"]) - float(current["thickness_m"])) > max(0.05, SIGNATURE_SIZE_TOLERANCE * float(sig["thickness_m"])):
            return None
        return across + 0.5 * mid + 0.01 * angle
    if kind == "opening":
        if sig.get("type") != current.get("type"):
            return None
        distance = math.dist(sig["center"], current["center"])
        if distance > pos or not _size_ok(float(sig["width_m"]), float(current["width_m"]), SIGNATURE_SIZE_TOLERANCE):
            return None
        return distance
    if kind == "furniture":
        if sig.get("center") is None or current.get("center") is None:
            return None
        distance = math.dist(sig["center"], current["center"])
        if distance > pos:
            return None
        if not all(_size_ok(float(a), float(b), SIGNATURE_SIZE_TOLERANCE) for a, b in zip(sig["size_m"], current["size_m"])):
            return None
        same_label = str(sig.get("label") or "") == str(current.get("label") or "")
        return distance + (0.0 if same_label else 0.1)
    if kind == "room":
        distance = math.dist(sig["center"], current["center"])
        if distance > max(pos, 0.5 * math.sqrt(max(float(sig["area_m2"]), 0.0)) * 0.5):
            return None
        if not _size_ok(float(sig["area_m2"]), float(current["area_m2"]), SIGNATURE_SIZE_TOLERANCE):
            return None
        return distance
    distance = math.dist(sig["center"], current["center"])
    return distance if distance <= pos else None


def resolve_signature(plan: BuildingPlan, sig: dict[str, Any] | None, preferred_id: str | None = None, *, strict: bool = False) -> str | None:
    """The id of the element that matches ``sig`` best (``preferred_id`` first when it still matches), else None.

    ``strict`` (used when stored edits are re-applied): the best match must be clear.  When another candidate also lies
    within the tolerances and its score differs by less than :data:`SIGNATURE_AMBIGUITY_MARGIN_M`, an :class:`EditError`
    with key ``assistant.history.ambiguous`` is raised instead of silently picking one.  ``preferred_id`` only wins
    such a tie when its geometry is unchanged (score ~0: the same element, ids did not move).
    """
    if not sig or sig.get("kind") not in ELEMENT_KINDS:
        return None
    kind = str(sig["kind"])
    if not strict:
        if preferred_id is not None:
            found = find_element(plan, preferred_id)
            if found is not None and found[0] == kind and _signature_score(plan, kind, found[1], sig) is not None:
                return str(preferred_id)
        best: tuple[float, str] | None = None
        for element in _collection(plan, kind):
            score = _signature_score(plan, kind, element, sig)
            if score is not None and (best is None or score < best[0]):
                best = (score, _element_id(element))
        return None if best is None else best[1]
    scored = sorted(
        (score, _element_id(element)) for element in _collection(plan, kind)
        if (score := _signature_score(plan, kind, element, sig)) is not None
    )
    if not scored:
        return None
    by_id = {element_id: score for score, element_id in scored}
    if preferred_id is not None and by_id.get(str(preferred_id), math.inf) <= 1e-6:
        return str(preferred_id)
    chosen_score, chosen = scored[0]
    rivals = [element_id for score, element_id in scored[1:] if score - chosen_score < SIGNATURE_AMBIGUITY_MARGIN_M]
    if rivals:
        raise EditError(
            f"{kind} {preferred_id or ''} matches {chosen} and {rivals[0]} equally well; it cannot be found again safely.".replace("  ", " "),
            key="assistant.history.ambiguous", params={"id": preferred_id, "candidates": [chosen, *rivals]},
        )
    return chosen


# -------------------------------------------------------------------------------------------------------- wall geometry

def _wall_axis(wall: PlanWall) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    p0 = np.asarray(wall.centreline[0], dtype=float)
    p1 = np.asarray(wall.centreline[-1], dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    if length <= 1e-12:
        raise EditError(f"Wall {wall.id} has no length.", key="assistant.edit.geometry", params={"id": wall.id})
    return p0, p1, (p1 - p0) / length, length


def _face_a_side(wall: PlanWall) -> np.ndarray:
    """Unit vector from the centreline towards the face_a side (right-hand side when the plan has no face_a)."""
    p0, p1, u, _length = _wall_axis(wall)
    left = np.asarray((-u[1], u[0]))
    if wall.face_a:
        middle = np.mean(np.asarray(wall.face_a, dtype=float), axis=0) - (p0 + p1) / 2.0
        side = float(middle @ left)
        if abs(side) > 1e-9:
            return left if side > 0 else -left
    if wall.face_b:
        middle = np.mean(np.asarray(wall.face_b, dtype=float), axis=0) - (p0 + p1) / 2.0
        side = float(middle @ left)
        if abs(side) > 1e-9:
            return -left if side > 0 else left
    return -left


def _rebuild_faces(wall: PlanWall, upm: float, side: np.ndarray | None = None) -> None:
    if wall.face_a is None and wall.face_b is None:
        return
    side = _face_a_side(wall) if side is None else side
    half = float(wall.thickness_m) * upm / 2.0
    line = np.asarray(wall.centreline, dtype=float)
    wall.face_a = [[float(x), float(y)] for x, y in line + side * half]
    wall.face_b = [[float(x), float(y)] for x, y in line - side * half]


def wall_band(wall: PlanWall, upm: float) -> Polygon:
    """The wall's plan footprint: centreline +- thickness / 2 (local units)."""
    p0, p1, u, _length = _wall_axis(wall)
    normal = np.asarray((-u[1], u[0])) * float(wall.thickness_m) * upm / 2.0
    return Polygon([tuple(p0 + normal), tuple(p1 + normal), tuple(p1 - normal), tuple(p0 - normal)])


def opening_overhang_m(plan: BuildingPlan, opening: PlanOpening) -> float | None:
    """How far (metres) the opening reaches beyond the ends of its host wall (0 = inside); None without host."""
    host = None if opening.host_wall_id is None else find_element(plan, opening.host_wall_id)
    if host is None or host[0] != "wall":
        return None
    upm = plan.units_per_metre or 1.0
    p0, _p1, u, length = _wall_axis(host[1])
    s = float((np.asarray(opening.center, float) - p0) @ u)
    half = float(opening.width_m) * upm / 2.0
    return max(0.0, -(s - half), (s + half) - length) / upm


def _check_in_host(plan: BuildingPlan, opening: PlanOpening, allowed_m: float) -> None:
    overhang = opening_overhang_m(plan, opening)
    if overhang is not None and overhang > max(OPENING_OVERHANG_M, allowed_m) + 1e-6:
        raise EditError(
            f"Opening {opening.id} would reach {overhang:.2f} m beyond the ends of wall {opening.host_wall_id}.",
            key="assistant.edit.opening_outside_wall", params={"id": opening.id, "wall": opening.host_wall_id, "overhang_m": round(overhang, 2)},
        )


def _storey_height_m(plan: BuildingPlan) -> float:
    if plan.ceiling_z is not None and plan.ceiling_z > plan.floor_z:
        return (float(plan.ceiling_z) - float(plan.floor_z)) / (plan.units_per_metre or 1.0)
    return DEFAULT_STOREY_HEIGHT_M


# ------------------------------------------------------------------------------------------------------------ context

@dataclass
class _Context:
    plan: BuildingPlan
    cloud: Any
    notes: list[str] = field(default_factory=list)
    resolved: dict[str, Any] = field(default_factory=dict)  # parameters stored with the op for a deterministic re-apply
    confirm: list[str] = field(default_factory=list)  # reasons why the proposal needs an explicit confirmation

    @property
    def upm(self) -> float:
        return float(self.plan.units_per_metre or 1.0)


def _number(params: dict[str, Any], key: str, *, required: bool = False) -> float | None:
    value = params.get(key)
    if value is None:
        if required:
            raise EditError(f"Parameter {key!r} is missing.", key="assistant.edit.bad_params", params={"param": key})
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EditError(f"Parameter {key!r} must be a number.", key="assistant.edit.bad_params", params={"param": key}) from error
    if not math.isfinite(number):
        raise EditError(f"Parameter {key!r} must be finite.", key="assistant.edit.bad_params", params={"param": key})
    return number


def _in_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise EditError(f"{name} {value:.2f} m is outside {low:g}-{high:g} m.", key="assistant.edit.invalid_value",
                        params={"name": name, "value": round(value, 3), "min": low, "max": high})


# --------------------------------------------------------------------------------------------------- wall dependencies

def _shift_ring(ring: list[list[float]], p0: np.ndarray, u: np.ndarray, side: np.ndarray, length: float, half: float,
                deltas: tuple[float, float, float], tolerance: float, reach: float) -> tuple[list[list[float]], bool]:
    """Offset the ring edges lying on a wall face / the centreline by that line's displacement (along ``side``)."""
    points = [np.asarray(p, dtype=float) for p in ring]
    count = len(points)
    if count < 3:
        return ring, False
    sin_limit = math.sin(math.radians(FACE_PARALLEL_DEG))
    edge_shift: list[float | None] = []
    for index in range(count):
        a, b = points[index], points[(index + 1) % count]
        vector = b - a
        size = float(np.linalg.norm(vector))
        shift = None
        if size > 1e-9 and abs(float(u[0] * vector[1] - u[1] * vector[0])) / size <= sin_limit:
            oa, ob = float((a - p0) @ side), float((b - p0) @ side)
            sa, sb = float((a - p0) @ u), float((b - p0) @ u)
            overlaps = max(sa, sb) >= -reach and min(sa, sb) <= length + reach and max(sa, sb) - min(sa, sb) > 1e-9
            if overlaps:
                candidates = ((abs(oa - half) + abs(ob - half), deltas[0]), (abs(oa + half) + abs(ob + half), deltas[2]), (abs(oa) + abs(ob), deltas[1]))
                best = min(candidates, key=lambda item: item[0])
                if best[0] <= 2.0 * tolerance and max(abs(oa - ob), 0.0) <= tolerance and best[1] != 0.0:
                    shift = best[1]
        edge_shift.append(shift)
    if not any(shift is not None for shift in edge_shift):
        return ring, False

    def line_of(index: int, shift: float | None) -> tuple[np.ndarray, np.ndarray]:
        a, b = points[index], points[(index + 1) % count]
        offset = side * (shift or 0.0)
        return a + offset, b - a

    new_points: list[list[float]] = []
    for index in range(count):
        previous, following = (index - 1) % count, index
        sp, sf = edge_shift[previous], edge_shift[following]
        point = points[index]
        if sp is None and sf is None:
            new_points.append([float(point[0]), float(point[1])])
            continue
        if sp is not None and sf is not None and abs(sp - sf) <= 1e-12:
            moved = point + side * sp
            new_points.append([float(moved[0]), float(moved[1])])
            continue
        (a1, d1), (a2, d2) = line_of(previous, sp), line_of(following, sf)
        denominator = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(denominator) <= 1e-12 * max(float(np.linalg.norm(d1) * np.linalg.norm(d2)), 1e-12):
            moved = point + side * (sf if sf is not None else sp)
        else:
            t = float((a2 - a1)[0] * d2[1] - (a2 - a1)[1] * d2[0]) / denominator
            moved = a1 + d1 * t
        new_points.append([float(moved[0]), float(moved[1])])
    return new_points, True


def _update_rooms(ctx: _Context, wall: PlanWall, p0: np.ndarray, u: np.ndarray, side: np.ndarray, length: float, half: float,
                  deltas: tuple[float, float, float], changes: list[Change]) -> None:
    upm = ctx.upm
    tolerance = max(min(FACE_MATCH_TOLERANCE_M * upm, 0.45 * half * 2.0), 0.01 * upm)
    reach = JOIN_END_REACH_M * upm + half
    for room in ctx.plan.rooms:
        before = snapshot(ctx.plan, "room", room)
        exterior, moved = _shift_ring(room.polygon.get("exterior") or [], p0, u, side, length, half, deltas, tolerance, reach)
        holes = []
        for hole in room.polygon.get("holes") or []:
            new_hole, hole_moved = _shift_ring(hole, p0, u, side, length, half, deltas, tolerance, reach)
            holes.append(new_hole)
            moved = moved or hole_moved
        if not moved:
            continue
        polygon = Polygon(exterior, holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda g: g.area)
            ctx.notes.append(f"room {room.id}: polygon split by the edit, largest part kept")
        if not isinstance(polygon, Polygon) or polygon.is_empty or polygon.area <= 1e-9:
            raise EditError(f"The edit would collapse room {room.id}.", key="assistant.edit.geometry", params={"id": room.id})
        polygon = polygon.simplify(1e-6 * upm, preserve_topology=True) if polygon.is_valid else polygon
        room.polygon = {
            "exterior": [[float(x), float(y)] for x, y in list(polygon.exterior.coords)[:-1]],
            "holes": [[[float(x), float(y)] for x, y in list(ring.coords)[:-1]] for ring in polygon.interiors],
        }
        room.area_m2 = float(polygon.area) / (upm * upm)
        point = polygon.representative_point()
        room.label_point = [float(point.x), float(point.y)]
        changes.append(Change(room.id, "room", "modified", before, snapshot(ctx.plan, "room", room), f"edge follows wall {wall.id}"))


def _update_bodies(ctx: _Context, old_band: Any, new_band: Any | None, wall_ids: set[str] | str) -> None:
    """Rebuild the wall body rings: the old band (minus what other walls cover) out, the new band in."""
    if not ctx.plan.wall_bodies:
        return
    upm = ctx.upm
    try:
        excluded = {wall_ids} if isinstance(wall_ids, str) else set(wall_ids)
        others = unary_union([wall_band(w, upm) for w in ctx.plan.walls if w.id not in excluded])
        removed = old_band.difference(others.buffer(1e-6 * upm)) if not others.is_empty else old_band
        bodies: list[dict[str, Any]] = []
        added = False
        for body in ctx.plan.wall_bodies:
            polygon = Polygon(body.get("exterior") or [], body.get("holes") or [])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            hit = polygon.intersects(old_band) or (new_band is not None and polygon.intersects(new_band))
            if not hit or polygon.is_empty:
                bodies.append(body)
                continue
            result = polygon.difference(removed)
            if new_band is not None and not added:
                result = result.union(new_band)
                added = True
            result = result.buffer(1e-7 * upm).buffer(-1e-7 * upm)
            parts = [result] if result.geom_type == "Polygon" else sorted(getattr(result, "geoms", []), key=lambda g: -g.area)
            for part in parts:
                if isinstance(part, Polygon) and not part.is_empty and part.area > 1e-6 * upm * upm:
                    bodies.append({"exterior": [[float(x), float(y)] for x, y in list(part.exterior.coords)[:-1]],
                                   "holes": [[[float(x), float(y)] for x, y in list(r.coords)[:-1]] for r in part.interiors]})
        if new_band is not None and not added:
            for part in ([new_band] if new_band.geom_type == "Polygon" else list(getattr(new_band, "geoms", []))):
                bodies.append({"exterior": [[float(x), float(y)] for x, y in list(part.exterior.coords)[:-1]],
                               "holes": [[[float(x), float(y)] for x, y in list(r.coords)[:-1]] for r in part.interiors]})
        ctx.plan.wall_bodies = bodies
    except Exception as error:  # noqa: BLE001 - a body ring that shapely cannot rebuild stays as it was
        ctx.notes.append(f"wall body ring not rebuilt ({type(error).__name__})")


def collinear_chain(plan: BuildingPlan, wall: PlanWall) -> list[PlanWall]:
    """``wall`` and the pieces that continue it on the same line (parallel, same thickness, touching or separated only by an
    opening hosted on one of them): a straight wall the pipeline split at a door is moved as one."""
    upm = plan.units_per_metre or 1.0
    p0, _p1, u, _length = _wall_axis(wall)
    normal = np.asarray((-u[1], u[0]))
    sin_limit = math.sin(math.radians(CHAIN_MAX_ANGLE_DEG))

    def on_line(other: PlanWall) -> bool:
        try:
            q0, q1, d2, _l2 = _wall_axis(other)
        except EditError:
            return False
        return (abs(float(u[0] * d2[1] - u[1] * d2[0])) <= sin_limit
                and max(abs(float((q0 - p0) @ normal)), abs(float((q1 - p0) @ normal))) <= CHAIN_OFFSET_M * upm
                and abs(float(other.thickness_m) - float(wall.thickness_m)) <= CHAIN_THICKNESS_M)

    def interval(item: PlanWall) -> tuple[float, float]:
        a, b = (float((np.asarray(p, float) - p0) @ u) for p in (item.centreline[0], item.centreline[-1]))
        return min(a, b), max(a, b)

    def bridged(a: PlanWall, b: PlanWall) -> bool:
        (a0, a1), (b0, b1) = interval(a), interval(b)
        low, high = (a1, b0) if a1 <= b0 else (b1, a0) if b1 <= a0 else (0.0, 0.0)
        if high - low <= CHAIN_TOUCH_M * upm:
            return True
        for opening in plan.openings:
            if opening.host_wall_id not in (a.id, b.id):
                continue
            s = float((np.asarray(opening.center, float) - p0) @ u)
            half_width = float(opening.width_m) * upm / 2.0
            if s - half_width <= low + CHAIN_GAP_SLACK_M * upm and s + half_width >= high - CHAIN_GAP_SLACK_M * upm:
                return True
        return False

    candidates = [w for w in plan.walls if w.id != wall.id and len(w.centreline) >= 2 and on_line(w)]
    chain = [wall]
    grown = True
    while grown:
        grown = False
        for other in list(candidates):
            if any(bridged(member, other) for member in chain):
                chain.append(other)
                candidates.remove(other)
                grown = True
    return chain


def _shift_wall(ctx: _Context, wall: PlanWall, *, delta_a: float, delta_c: float, delta_b: float, thickness_m: float,
                parallel: np.ndarray | None, change: str, note: str = "", chain: list[PlanWall] | None = None) -> list[Change]:
    """Move / re-thicken ``wall`` (and its collinear ``chain`` pieces): deltas are perpendicular displacements (local units,
    + = towards face_a of ``wall``) of face_a, the centreline and face_b; ``parallel`` an extra shift along the axis.
    Applies all dependencies."""
    plan, upm = ctx.plan, ctx.upm
    chain = list(chain or [wall])
    chain_ids = {piece.id for piece in chain}
    changes: list[Change] = []
    p0, _p1, u, _length = _wall_axis(wall)
    side = _face_a_side(wall)
    half = float(wall.thickness_m) * upm / 2.0
    stations = [float((np.asarray(p, float) - p0) @ u) for piece in chain for p in (piece.centreline[0], piece.centreline[-1])]
    s_min, s_max = min(stations), max(stations)
    origin, length = p0 + u * s_min, s_max - s_min
    old_band = unary_union([wall_band(piece, upm) for piece in chain])
    shift = side * delta_c + (np.zeros(2) if parallel is None else np.asarray(parallel, dtype=float))
    overhang_before = {o.id: (opening_overhang_m(plan, o) or 0.0) for o in plan.openings}

    for piece in chain:
        before = snapshot(plan, "wall", piece)
        piece_side = _face_a_side(piece)
        piece.centreline = [[float(x), float(y)] for x, y in np.asarray(piece.centreline, dtype=float) + shift]
        piece.thickness_m = float(thickness_m)
        _rebuild_faces(piece, upm, piece_side)
        piece_note = note if piece is wall else ", ".join(text for text in (note, f"collinear with {wall.id}") if text)
        changes.append(Change(piece.id, "wall", change, before, snapshot(plan, "wall", piece), piece_note))

    for opening in plan.openings:
        if opening.host_wall_id not in chain_ids:
            continue
        opening_before = snapshot(plan, "opening", opening)
        opening.center = [float(opening.center[0] + shift[0]), float(opening.center[1] + shift[1])]
        if abs(float(opening.wall_thickness_m) - float(thickness_m)) > 1e-12:
            opening.wall_thickness_m = float(thickness_m)
        changes.append(Change(opening.id, "opening", "moved" if change == "moved" else "modified", opening_before,
                              snapshot(plan, "opening", opening), f"follows host wall {opening.host_wall_id}"))

    # walls that end on this wall follow the face (or centreline) they end at
    sin_join = math.sin(math.radians(JOIN_MIN_ANGLE_DEG))
    for other in plan.walls:
        if other.id in chain_ids or len(other.centreline) < 2:
            continue
        q0, q1, d2, _l2 = _wall_axis(other)
        if abs(float(d2 @ side)) < sin_join:
            continue
        half2 = float(other.thickness_m) * upm / 2.0
        band_limit = half + half2 + JOIN_BAND_SLACK_M * upm
        reach = half2 + JOIN_END_REACH_M * upm
        points = np.asarray(other.centreline, dtype=float)
        moved = False
        for index in (0, len(points) - 1):
            end = points[index]
            offset = float((end - origin) @ side)
            along = float((end - origin) @ u)
            if abs(offset) > band_limit or not (-reach <= along <= length + reach):
                continue
            delta = delta_a if offset > half / 2.0 else delta_b if offset < -half / 2.0 else delta_c
            if abs(delta) <= 1e-12:
                continue
            points[index] = end + d2 * (delta / float(d2 @ side))
            moved = True
        if not moved:
            continue
        if float(np.linalg.norm(points[-1] - points[0])) < MIN_WALL_LENGTH_M * upm or float((points[-1] - points[0]) @ d2) <= 0.0:
            raise EditError(f"The edit would shrink wall {other.id} below {MIN_WALL_LENGTH_M} m.", key="assistant.edit.wall_too_short",
                            params={"id": other.id})
        other_before = snapshot(plan, "wall", other)
        other_side = _face_a_side(other)
        other.centreline = [[float(x), float(y)] for x, y in points]
        _rebuild_faces(other, upm, other_side)
        changes.append(Change(other.id, "wall", "modified", other_before, snapshot(plan, "wall", other), f"end follows wall {wall.id}"))

    for opening in plan.openings:
        _check_in_host(plan, opening, overhang_before.get(opening.id, 0.0))

    if len(chain) >= CHAIN_CONFIRM_PIECES:
        text = f"wall {wall.id} is one straight wall of {len(chain)} pieces ({', '.join(piece.id for piece in chain)}): all of them changed"
        changes[0].note = ", ".join(item for item in (changes[0].note, f"with {len(chain) - 1} collinear pieces") if item)
        ctx.notes.append(text)
        ctx.confirm.append(text)
    _update_rooms(ctx, wall, origin, u, side, length, half, (delta_a, delta_c, delta_b), changes)
    _update_bodies(ctx, old_band, unary_union([wall_band(piece, upm) for piece in chain]), chain_ids)
    return changes


# --------------------------------------------------------------------------------------------------------- op handlers

def _op_move_wall(ctx: _Context, op: EditOp) -> list[Change]:
    wall = _require(ctx.plan, op.target, "wall")
    offset = _number(op.params, "offset_m")
    dx, dy = _number(op.params, "dx_m"), _number(op.params, "dy_m")
    upm = ctx.upm
    side = _face_a_side(wall)
    _p0, _p1, u, _length = _wall_axis(wall)
    if offset is not None and (dx is not None or dy is not None):
        raise EditError("Give either offset_m or dx_m/dy_m, not both.", key="assistant.edit.bad_params", params={"param": "offset_m"})
    if offset is not None:
        _in_range("offset", abs(offset), 0.001, MAX_WALL_MOVE_M)
        delta, parallel = offset * upm, None
    elif dx is not None or dy is not None:
        vector = np.asarray((dx or 0.0, dy or 0.0), dtype=float) * upm
        _in_range("move", float(np.linalg.norm(vector)) / upm, 0.001, MAX_WALL_MOVE_M)
        delta = float(vector @ side)
        parallel = u * float(vector @ u)
    else:
        raise EditError("move_wall needs offset_m or dx_m/dy_m.", key="assistant.edit.bad_params", params={"param": "offset_m"})
    chain = collinear_chain(ctx.plan, wall) if op.params.get("chain", True) else [wall]
    return _shift_wall(ctx, wall, delta_a=delta, delta_c=delta, delta_b=delta, thickness_m=float(wall.thickness_m),
                       parallel=parallel, change="moved", chain=chain)


def _op_set_wall_thickness(ctx: _Context, op: EditOp) -> list[Change]:
    wall = _require(ctx.plan, op.target, "wall")
    thickness = float(_number(op.params, "thickness_m", required=True))
    _in_range("thickness", thickness, *THICKNESS_RANGE_M)
    anchor = str(op.params.get("anchor") or "centre").strip().lower()
    if anchor == "center":
        anchor = "centre"
    if anchor not in ("centre", "face_a", "face_b"):
        raise EditError(f"Unknown anchor {anchor!r} (centre, face_a, face_b).", key="assistant.edit.bad_params", params={"param": "anchor"})
    grow = (thickness - float(wall.thickness_m)) * ctx.upm / 2.0  # each face moves outward by this when centred
    if anchor == "centre":
        delta_a, delta_c, delta_b = grow, 0.0, -grow
    elif anchor == "face_a":
        delta_a, delta_c, delta_b = 0.0, -grow, -2.0 * grow
    else:
        delta_a, delta_c, delta_b = 2.0 * grow, grow, 0.0
    chain = collinear_chain(ctx.plan, wall) if op.params.get("chain", True) else [wall]
    return _shift_wall(ctx, wall, delta_a=delta_a, delta_c=delta_c, delta_b=delta_b, thickness_m=thickness, parallel=None,
                       change="modified", note=f"anchor {anchor}", chain=chain)


def _op_delete_wall(ctx: _Context, op: EditOp) -> list[Change]:
    plan, upm = ctx.plan, ctx.upm
    wall = _require(plan, op.target, "wall")
    changes = [Change(wall.id, "wall", "removed", snapshot(plan, "wall", wall), None)]
    for opening in list(plan.openings):
        if opening.host_wall_id == wall.id:
            changes.append(Change(opening.id, "opening", "removed", snapshot(plan, "opening", opening), None, f"removed with host wall {wall.id}"))
            plan.openings.remove(opening)
    old_band = wall_band(wall, upm)
    plan.walls.remove(wall)
    _update_bodies(ctx, old_band, None, wall.id)
    ctx.notes.append(f"wall {wall.id} deleted: the rooms on both sides are not merged automatically")
    return changes


def _op_set_opening(ctx: _Context, op: EditOp) -> list[Change]:
    plan, upm = ctx.plan, ctx.upm
    opening = _require(plan, op.target, "opening")
    width, sill = _number(op.params, "width_m"), _number(op.params, "sill_m")
    head, shift = _number(op.params, "head_m"), _number(op.params, "shift_m")
    if width is None and sill is None and head is None and shift is None:
        raise EditError("set_opening needs width_m, sill_m, head_m or shift_m.", key="assistant.edit.bad_params", params={"param": "width_m"})
    before = snapshot(plan, "opening", opening)
    allowed = opening_overhang_m(plan, opening) or 0.0
    if width is not None:
        _in_range("width", width, *OPENING_WIDTH_RANGE_M)
        opening.width_m = float(width)
    if sill is not None:
        _in_range("sill", sill, *SILL_RANGE_M)
        if opening.kind != "window" and sill > 1e-9:
            raise EditError(f"A {opening.kind} has no sill.", key="assistant.edit.invalid_value", params={"name": "sill", "value": sill, "min": 0, "max": 0})
        opening.sill_m = float(sill)
    if head is not None:
        storey = _storey_height_m(plan)
        _in_range("head", head, MIN_OPENING_HEIGHT_M, storey + 1e-6)
        opening.head_m = float(head)
    if opening.head_m - opening.sill_m < MIN_OPENING_HEIGHT_M - 1e-9:
        raise EditError(f"The head ({opening.head_m:.2f} m) must be above the sill ({opening.sill_m:.2f} m).",
                        key="assistant.edit.invalid_value", params={"name": "head", "value": round(opening.head_m, 3), "min": round(opening.sill_m, 3), "max": round(_storey_height_m(plan), 3)})
    if shift is not None:
        _in_range("shift", abs(shift), 0.0, MAX_OPENING_SHIFT_M)
        axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
        if opening.host_wall_id is not None:
            host = find_element(plan, opening.host_wall_id)
            if host is not None and host[0] == "wall":
                axis_u = _wall_axis(host[1])[2]
                axis = axis_u if float(axis_u @ axis) >= 0 else -axis_u
        opening.center = [float(opening.center[0] + axis[0] * shift * upm), float(opening.center[1] + axis[1] * shift * upm)]
    _check_in_host(plan, opening, allowed)
    return [Change(opening.id, "opening", "moved" if shift is not None and width is None and sill is None and head is None else "modified",
                   before, snapshot(plan, "opening", opening))]


def _op_delete_opening(ctx: _Context, op: EditOp) -> list[Change]:
    opening = _require(ctx.plan, op.target, "opening")
    change = Change(opening.id, "opening", "removed", snapshot(ctx.plan, "opening", opening), None)
    ctx.plan.openings.remove(opening)
    return [change]


def _next_furniture_id(plan: BuildingPlan) -> str:
    numbers = [int(item.id[1:]) for item in plan.furniture if str(item.id)[:1] == "F" and str(item.id)[1:].isdigit()]
    return f"F{max(numbers or [0]) + 1:03d}"


def _kind(params: dict[str, Any]) -> FurnitureKind:
    spec = resolve_kind(params.get("kind"))
    if spec is None:
        raise EditError(f"Unknown furniture kind {params.get('kind')!r}.", key="assistant.edit.unknown_kind",
                        params={"kind": params.get("kind"), "known": ", ".join(sorted(FURNITURE_KINDS))})
    return spec


def _trimmed_rectangle(xy: np.ndarray, low: float = 1.0, high: float = 99.0) -> Polygon | None:
    """Oriented rectangle of the points without the outer 1 % along both principal axes."""
    if len(xy) < 3:
        return None
    centre = xy.mean(axis=0)
    centred = xy - centre
    _values, vectors = np.linalg.eigh(centred.T @ centred / len(xy))
    axis = vectors[:, -1]
    # snap the axis direction to 90 degree symmetry: the angle of the long axis in [0, 180)
    across = np.asarray((-axis[1], axis[0]))
    s, t = centred @ axis, centred @ across
    s0, s1 = np.percentile(s, [low, high])
    t0, t1 = np.percentile(t, [low, high])
    if s1 - s0 <= 1e-6 or t1 - t0 <= 1e-6:
        return None
    corners = [centre + axis * a + across * b for a, b in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))]
    polygon = Polygon([tuple(c) for c in corners])
    return polygon if polygon.is_valid and polygon.area > 0 else None


def _measured_item(ctx: _Context, box: dict[str, Any], spec: FurnitureKind, evidence: Any) -> PlanFurniture:
    from samcloud.assistant.evidence import EVIDENCE_FLOOR_CLEARANCE_M

    plan, cloud, upm = ctx.plan, ctx.cloud, ctx.upm
    selection = cloud.select(box, min_height_m=EVIDENCE_FLOOR_CLEARANCE_M, classes=spec.classes)
    xy = cloud.xy_m(selection)
    heights = cloud.height_m(selection)
    rectangle = _trimmed_rectangle(xy)
    if rectangle is None:
        raise EditError("The points in the box do not form an object outline.", key="assistant.edit.no_evidence",
                        params={"kind": spec.key, "points": int(len(selection)), "reason": "no outline"})
    top = float(np.percentile(heights, 98))
    hanging = spec.default_z_m is not None
    bottom = float(np.percentile(heights, 2)) if hanging else 0.0
    instances = cloud.instance[selection]
    instance_id = None
    if len(instances) and np.any(instances > 0):
        ids, counts = np.unique(instances[instances > 0], return_counts=True)
        if counts.max() >= 0.5 * len(instances):
            instance_id = int(ids[int(np.argmax(counts))])
    footprint = [[float(x) * upm, float(y) * upm] for x, y in list(rectangle.exterior.coords)[:-1]]
    item = PlanFurniture(
        id=_next_furniture_id(plan), instance_id=instance_id, label=spec.label, semantic_label=spec.semantic_label,
        catalogue_kind=spec.catalogue_kind, layer=spec.layer, footprint=footprint,
        z_min=float(plan.floor_z) + bottom * upm, z_max=float(plan.floor_z) + top * upm, z_source="points",
        default_height_m=round(top, 3), mounting="wall" if hanging else None,
    )
    try:
        from samcloud.bim.furniture_recon.pipeline import reconstruct_items

        reconstruct_items([item], cloud.scene_cloud(box, margin_m=0.05), units_per_metre=upm, floor_z=float(plan.floor_z),
                          ceiling_z=plan.ceiling_z)
    except Exception as error:  # noqa: BLE001 - the measured box stays when the shape reconstruction fails
        LOGGER.warning("assistant: furniture reconstruction failed (%s), measured box kept", type(error).__name__)
        ctx.notes.append(f"shape reconstruction failed ({type(error).__name__}): measured box kept")
    fit = dict(item.fit or {})
    fit.update({"added": "assistant", "evidence": "measured", "evidence_points": int(evidence.points)})
    item.fit = fit
    return item


def _manual_item(ctx: _Context, box: dict[str, Any], spec: FurnitureKind, size: list[float] | None) -> PlanFurniture:
    from samcloud.assistant.pointcloud import parse_box

    plan, upm = ctx.plan, ctx.upm
    x0, x1, y0, y1, _z0, _z1 = parse_box(box)
    width, depth, height = spec.size_m if size is None else (float(size[0]), float(size[1]), float(size[2]) if len(size) > 2 else spec.size_m[2])
    for value in (width, depth, height):
        _in_range("size", value, *FURNITURE_SIZE_RANGE_M)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    long_side, short_side = max(width, depth), min(width, depth)
    if (x1 - x0) >= (y1 - y0):
        half_x, half_y = long_side / 2.0, short_side / 2.0
    else:
        half_x, half_y = short_side / 2.0, long_side / 2.0
    footprint = [[(cx + dx) * upm, (cy + dy) * upm] for dx, dy in ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))]
    item = PlanFurniture(
        id=_next_furniture_id(plan), instance_id=None, label=spec.label, semantic_label=spec.semantic_label,
        catalogue_kind=spec.catalogue_kind, layer=MANUAL_LAYER, footprint=footprint, z_source="default",
        default_height_m=None if spec.default_z_m else float(height),
        default_z_m=None if spec.default_z_m is None else [float(spec.default_z_m[0]), float(spec.default_z_m[1])],
        fit={"added": "assistant", "evidence": "manual"},
    )
    ctx.notes.append(f"{spec.label} {item.id} added manually without point-cloud evidence (default dimensions, layer {MANUAL_LAYER})")
    return item


def _op_add_furniture(ctx: _Context, op: EditOp) -> list[Change]:
    from samcloud.assistant.pointcloud import parse_box

    plan = ctx.plan
    spec = _kind(op.params)
    box = op.params.get("box")
    try:
        parse_box(box)
    except ValueError as error:
        raise EditError(str(error), key="assistant.edit.bad_params", params={"param": "box"}) from error
    stored = op.params.get("_item")
    if isinstance(stored, dict):  # re-apply: the item as measured when the edit was accepted
        item = PlanFurniture.from_dict({**copy.deepcopy(stored), "id": _next_furniture_id(plan)})
    else:
        manual = bool(op.params.get("manual", False))
        size = op.params.get("size_m")
        if size is not None and (not isinstance(size, (list, tuple)) or len(size) < 2):
            raise EditError("size_m must be [width, depth] or [width, depth, height].", key="assistant.edit.bad_params", params={"param": "size_m"})
        evidence = check_evidence(ctx.cloud, box, spec.key) if ctx.cloud is not None else None
        if evidence is not None and evidence.ok:
            item = _measured_item(ctx, box, spec, evidence)
            if manual:
                ctx.notes.append(f"{spec.label}: the point cloud shows it, measured instead of manual")
        elif manual:
            item = _manual_item(ctx, box, spec, None if size is None else [float(v) for v in size])
        elif ctx.cloud is None:
            raise EditError("Adding an object needs the project point cloud as evidence.", key="assistant.edit.cloud_required", params={"kind": spec.key})
        else:
            raise EditError(f"No evidence for a {spec.label_en.lower()} in the point cloud: {evidence.reason}.", key="assistant.edit.no_evidence",
                            params={"kind": spec.key, "points": evidence.points, "reason": evidence.reason})
        ctx.resolved["_item"] = {key: value for key, value in item.to_dict().items() if key != "id"}
    plan.furniture.append(item)
    note = "manual, no point-cloud evidence" if (item.fit or {}).get("evidence") == "manual" else "measured in the point cloud"
    return [Change(item.id, "furniture", "added", None, snapshot(plan, "furniture", item), note)]


def _transform_furniture(item: PlanFurniture, transform: Callable[[np.ndarray], np.ndarray], angle_deg: float = 0.0) -> None:
    def ring(points: Any) -> list[list[float]]:
        array = np.asarray(points, dtype=float).reshape(-1, 2)
        return [[float(x), float(y)] for x, y in transform(array)]

    item.footprint = ring(item.footprint)
    for part in item.parts or []:
        if isinstance(part, dict) and part.get("footprint"):
            part["footprint"] = ring(part["footprint"])
    for solid in item.solids or []:
        if not isinstance(solid, dict):
            continue
        if solid.get("outline"):
            solid["outline"] = ring(solid["outline"])
        if solid.get("center") is not None and len(solid["center"]) >= 2:
            moved = transform(np.asarray(solid["center"][:2], dtype=float).reshape(1, 2))[0]
            solid["center"] = [float(moved[0]), float(moved[1]), *[float(v) for v in solid["center"][2:]]]
        if angle_deg and solid.get("rotation_deg") is not None:
            solid["rotation_deg"] = float(solid["rotation_deg"]) + angle_deg
    if isinstance(item.mesh, dict) and item.mesh.get("vertices"):
        vertices = np.asarray(item.mesh["vertices"], dtype=float)
        vertices[:, :2] = transform(vertices[:, :2])
        item.mesh["vertices"] = [[float(v) for v in row] for row in vertices]
    if item.fit and item.fit.get("plan_footprint"):
        item.fit["plan_footprint"] = ring(item.fit["plan_footprint"])
    if angle_deg and item.facing_deg is not None:
        item.facing_deg = (float(item.facing_deg) + angle_deg) % 360.0


def _check_back_wall(plan: BuildingPlan, item: PlanFurniture, notes: list[str]) -> None:
    if not item.back_wall_id:
        return
    found = find_element(plan, item.back_wall_id)
    polygon = _polygon_of(item.footprint)
    upm = plan.units_per_metre or 1.0
    if found is None or polygon is None or wall_band(found[1], upm).distance(polygon) > BACK_WALL_REACH_M * upm:
        notes.append(f"{item.id}: no longer against wall {item.back_wall_id}")
        item.back_wall_id = None


def _op_move_furniture(ctx: _Context, op: EditOp) -> list[Change]:
    item = _require(ctx.plan, op.target, "furniture")
    dx, dy = _number(op.params, "dx_m") or 0.0, _number(op.params, "dy_m") or 0.0
    _in_range("move", math.hypot(dx, dy), 0.001, MAX_FURNITURE_MOVE_M)
    before = snapshot(ctx.plan, "furniture", item)
    offset = np.asarray((dx, dy), dtype=float) * ctx.upm
    _transform_furniture(item, lambda points: points + offset)
    _check_back_wall(ctx.plan, item, ctx.notes)
    return [Change(item.id, "furniture", "moved", before, snapshot(ctx.plan, "furniture", item))]


def _op_rotate_furniture(ctx: _Context, op: EditOp) -> list[Change]:
    item = _require(ctx.plan, op.target, "furniture")
    angle = float(_number(op.params, "angle_deg", required=True))
    if abs(angle) < 1e-6 or abs(angle) > 360.0:
        raise EditError("angle_deg must be between -360 and 360 and not 0.", key="assistant.edit.invalid_value",
                        params={"name": "angle", "value": angle, "min": -360, "max": 360})
    polygon = _polygon_of(item.footprint)
    if polygon is None:
        raise EditError(f"Furniture {item.id} has no usable outline.", key="assistant.edit.geometry", params={"id": item.id})
    centre = np.asarray(polygon.centroid.coords[0], dtype=float)
    radians = math.radians(angle)
    rotation = np.asarray(((math.cos(radians), -math.sin(radians)), (math.sin(radians), math.cos(radians))))
    before = snapshot(ctx.plan, "furniture", item)
    _transform_furniture(item, lambda points: (points - centre) @ rotation.T + centre, angle)
    _check_back_wall(ctx.plan, item, ctx.notes)
    return [Change(item.id, "furniture", "moved", before, snapshot(ctx.plan, "furniture", item), f"rotated {angle:g} deg")]


def _op_delete_furniture(ctx: _Context, op: EditOp) -> list[Change]:
    item = _require(ctx.plan, op.target, "furniture")
    change = Change(item.id, "furniture", "removed", snapshot(ctx.plan, "furniture", item), None)
    ctx.plan.furniture.remove(item)
    return [change]


def _op_set_furniture_class(ctx: _Context, op: EditOp) -> list[Change]:
    item = _require(ctx.plan, op.target, "furniture")
    spec = _kind(op.params)
    before = snapshot(ctx.plan, "furniture", item)
    item.label, item.semantic_label, item.catalogue_kind = spec.label, spec.semantic_label, spec.catalogue_kind
    if item.layer != MANUAL_LAYER and (spec.layer == "P-SANI" or item.layer == "P-SANI"):
        item.layer = spec.layer
    return [Change(item.id, "furniture", "modified", before, snapshot(ctx.plan, "furniture", item), f"class {spec.key}")]


def _op_rename_room(ctx: _Context, op: EditOp) -> list[Change]:
    room = _require(ctx.plan, op.target, "room")
    name = " ".join(str(op.params.get("name") or "").split())
    if not name or len(name) > MAX_ROOM_NAME_LENGTH:
        raise EditError(f"A room name needs 1-{MAX_ROOM_NAME_LENGTH} characters.", key="assistant.edit.invalid_value",
                        params={"name": "name", "value": len(name), "min": 1, "max": MAX_ROOM_NAME_LENGTH})
    before = snapshot(ctx.plan, "room", room)
    room.name = name
    return [Change(room.id, "room", "modified", before, snapshot(ctx.plan, "room", room), "renamed")]


# ---------------------------------------------------------------------------------------------------------------- OPS

@dataclass(frozen=True)
class OpSpec:
    handler: Callable[[_Context, EditOp], list[Change]]
    schema: dict[str, Any]  # JSON schema of the tool arguments (``target`` + parameters)
    description: str
    target_kind: str | None


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_TARGET = {"type": "string", "description": "Plan id of the element (e.g. W003, O002, F007, R001)."}
_BOX = {
    "type": "object", "description": "Plan metres: {x: [x0, x1], y: [y0, y1], z: [z0, z1] floor-relative (optional)}.",
    "properties": {"x": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                   "y": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                   "z": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
    "required": ["x", "y"],
}
_KIND = {"type": "string", "enum": sorted(FURNITURE_KINDS)}

OPS: dict[str, OpSpec] = {
    "move_wall": OpSpec(_op_move_wall, _schema({
        "target": _TARGET,
        "offset_m": {"type": "number", "description": "Perpendicular shift in metres; positive = towards the wall's face_a side."},
        "dx_m": {"type": "number", "description": "Shift in plan x (metres); use with dy_m instead of offset_m."},
        "dy_m": {"type": "number", "description": "Shift in plan y (metres)."},
        "chain": {"type": "boolean", "description": "Also move the collinear pieces of the same straight wall (default true)."},
    }, ["target"]), "Move a wall; hosted openings, joining walls and room edges follow.", "wall"),
    "set_wall_thickness": OpSpec(_op_set_wall_thickness, _schema({
        "target": _TARGET, "thickness_m": {"type": "number", "minimum": THICKNESS_RANGE_M[0], "maximum": THICKNESS_RANGE_M[1]},
        "anchor": {"type": "string", "enum": ["centre", "face_a", "face_b"], "description": "What stays in place (default centre)."},
        "chain": {"type": "boolean", "description": "Also change the collinear pieces of the same straight wall (default true)."},
    }, ["target", "thickness_m"]), "Change a wall's thickness.", "wall"),
    "delete_wall": OpSpec(_op_delete_wall, _schema({"target": _TARGET}, ["target"]), "Delete a wall and its hosted openings.", "wall"),
    "set_opening": OpSpec(_op_set_opening, _schema({
        "target": _TARGET,
        "width_m": {"type": "number", "minimum": OPENING_WIDTH_RANGE_M[0], "maximum": OPENING_WIDTH_RANGE_M[1]},
        "sill_m": {"type": "number", "minimum": SILL_RANGE_M[0], "maximum": SILL_RANGE_M[1], "description": "Windows only."},
        "head_m": {"type": "number", "description": "Head height above the floor (metres), above the sill and at most the storey height."},
        "shift_m": {"type": "number", "description": "Shift along the host wall (metres)."},
    }, ["target"]), "Change a door / window / passage: width, sill, head or position along its wall.", "opening"),
    "delete_opening": OpSpec(_op_delete_opening, _schema({"target": _TARGET}, ["target"]), "Delete a door, window or passage.", "opening"),
    "add_furniture": OpSpec(_op_add_furniture, _schema({
        "box": _BOX, "kind": _KIND,
        "manual": {"type": "boolean", "description": "Only after the user explicitly confirmed: add without point-cloud evidence."},
        "size_m": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3, "description": "Manual items only: width, depth[, height]."},
    }, ["box", "kind"]), "Add a furniture item measured from the point cloud inside the box (needs evidence).", None),
    "move_furniture": OpSpec(_op_move_furniture, _schema({
        "target": _TARGET, "dx_m": {"type": "number"}, "dy_m": {"type": "number"},
    }, ["target"]), "Move a furniture item.", "furniture"),
    "rotate_furniture": OpSpec(_op_rotate_furniture, _schema({
        "target": _TARGET, "angle_deg": {"type": "number", "description": "Counter-clockwise about the footprint centroid."},
    }, ["target", "angle_deg"]), "Rotate a furniture item about its footprint centroid.", "furniture"),
    "delete_furniture": OpSpec(_op_delete_furniture, _schema({"target": _TARGET}, ["target"]), "Delete a furniture item.", "furniture"),
    "set_furniture_class": OpSpec(_op_set_furniture_class, _schema({"target": _TARGET, "kind": _KIND}, ["target", "kind"]),
                                  "Change what a furniture item is (label, IFC class and type follow).", "furniture"),
    "rename_room": OpSpec(_op_rename_room, _schema({"target": _TARGET, "name": {"type": "string", "maxLength": MAX_ROOM_NAME_LENGTH}},
                                                   ["target", "name"]), "Rename a room.", "room"),
}


def op_spec(name: str) -> OpSpec | None:
    """The :class:`OpSpec` of an op name: :data:`OPS` (the "elements" tier), else the full-control ops of
    :mod:`samcloud.assistant.edits_full` (looked up lazily so stored full-control edits replay everywhere)."""
    spec = OPS.get(str(name))
    if spec is None:
        from samcloud.assistant.edits_full import FULL_OPS

        spec = FULL_OPS.get(str(name))
    return spec


def op_schemas() -> dict[str, dict[str, Any]]:
    """{op name: {"description", "parameters" (JSON schema incl. target)}} for the tool definitions of package C."""
    return {name: {"description": spec.description, "parameters": copy.deepcopy(spec.schema)} for name, spec in OPS.items()}


def _check_params(spec: OpSpec, op: EditOp) -> None:
    allowed = set(spec.schema["properties"]) - {"target"}
    unknown = [key for key in op.params if key not in allowed and not key.startswith("_")]
    if unknown:
        raise EditError(f"Unknown parameter(s) {', '.join(sorted(unknown))} for {op.op}.", key="assistant.edit.bad_params", params={"param": unknown[0]})
    for key in spec.schema["required"]:
        if key == "target":
            if spec.target_kind is not None and not op.target:
                raise EditError(f"{op.op} needs a target.", key="assistant.edit.bad_params", params={"param": "target"})
        elif op.params.get(key) is None:
            raise EditError(f"Parameter {key!r} is missing.", key="assistant.edit.bad_params", params={"param": key})


# ---------------------------------------------------------------------------------------------------------- proposal

class Proposal:
    """Edits applied on a deep copy of ``base``; the base plan is never mutated."""

    def __init__(self, base: BuildingPlan) -> None:
        self.base: BuildingPlan = copy.deepcopy(base)
        self.plan: BuildingPlan = copy.deepcopy(base)
        self.ops: list[EditOp] = []
        self.changes: list[Change] = []
        self.notes: list[str] = []
        # signature of every op target as it was before this proposal touched it (re-resolved after a recompute)
        self.signatures: dict[str, dict[str, Any]] = {}
        self.created: list[str] = []  # ids this proposal added
        self.confirm_reasons: list[str] = []  # why needs_confirmation() is True beyond removals / count

    def apply(self, op: EditOp | dict[str, Any], *, cloud: Any | None = None, replay: bool = False) -> list[Change]:
        """Apply one op (all or nothing); raises :class:`EditError` and leaves the proposal unchanged on failure.

        Parameters starting with ``_`` are internal (the measured item of an accepted ``add_furniture``): they are dropped
        from every incoming op and only honoured with ``replay=True`` (re-applying a stored edit, ``history.reapply_edit``).
        """
        op = EditOp.from_dict(op) if isinstance(op, dict) else EditOp(op.op, op.target, copy.deepcopy(op.params))
        if not replay:
            op.params = {key: value for key, value in op.params.items() if not str(key).startswith("_")}
        spec = op_spec(op.op)
        if spec is None:
            raise EditError(f"Unknown edit operation {op.op!r}.", key="assistant.edit.unknown_op", params={"op": op.op})
        _check_params(spec, op)
        work = copy.deepcopy(self.plan)
        ctx = _Context(work, cloud)
        signature = None
        if op.target and op.target not in self.created:
            signature = signature_of(self.base, op.target) or signature_of(work, op.target)
        changes = spec.handler(ctx, op)
        created = [change.element_id for change in changes if change.change == "added"]
        if created:
            ctx.resolved["_created_ids"] = created
        op.params.update(ctx.resolved)
        if op.target and signature is not None and op.target not in self.signatures:
            self.signatures[op.target] = signature
        self.created.extend(created)
        self.plan = work
        self.ops.append(op)
        self.changes.extend(changes)
        self.notes.extend(ctx.notes)
        self.confirm_reasons.extend(ctx.confirm)
        return changes

    def summary(self) -> list[dict[str, Any]]:
        """Before -> after per changed element (first before, last after) for the UI."""
        order: list[str] = []
        merged: dict[str, dict[str, Any]] = {}
        for change in self.changes:
            entry = merged.get(change.element_id)
            if entry is None:
                order.append(change.element_id)
                merged[change.element_id] = {"element_id": change.element_id, "kind": change.kind, "change": change.change,
                                             "before": copy.deepcopy(change.before), "after": copy.deepcopy(change.after),
                                             "notes": [change.note] if change.note else []}
                continue
            entry["after"] = copy.deepcopy(change.after)
            if change.note and change.note not in entry["notes"]:
                entry["notes"].append(change.note)
            if entry["change"] == "added":
                entry["change"] = "added" if change.change != "removed" else "discarded"
            elif change.change == "removed":
                entry["change"] = "removed"
            elif entry["change"] != change.change:
                entry["change"] = "modified"
        return [merged[key] for key in order if merged[key]["change"] != "discarded"]

    def changed_ids(self) -> set[str]:
        return {entry["element_id"] for entry in self.summary()}

    def needs_confirmation(self) -> bool:
        """Any removal, more than :data:`CONFIRM_ELEMENT_COUNT` changed elements, or a wall edit that changed a chain of
        :data:`CHAIN_CONFIRM_PIECES` or more collinear pieces."""
        entries = self.summary()
        return (bool(self.confirm_reasons) or any(entry["change"] == "removed" for entry in entries)
                or len(entries) > CONFIRM_ELEMENT_COUNT)


def apply_ops(base: BuildingPlan, ops: list[EditOp | dict[str, Any]], *, cloud: Any | None = None) -> Proposal:
    """A proposal with ``ops`` applied in order (raises on the first failing op)."""
    proposal = Proposal(base)
    for op in ops:
        proposal.apply(op, cloud=cloud)
    return proposal


def diff_plans(base: BuildingPlan, edited: BuildingPlan) -> list[Change]:
    """Element changes between two plans that share ids (edits never renumber): added, removed, modified."""
    changes: list[Change] = []
    for kind in ELEMENT_KINDS:
        old = {_element_id(e): e for e in _collection(base, kind)}
        new = {_element_id(e): e for e in _collection(edited, kind)}
        for element_id, element in old.items():
            if element_id not in new:
                changes.append(Change(element_id, kind, "removed", snapshot(base, kind, element), None))
                continue
            a = element if isinstance(element, dict) else element.to_dict()
            other = new[element_id]
            b = other if isinstance(other, dict) else other.to_dict()
            if a != b:
                changes.append(Change(element_id, kind, "modified", snapshot(base, kind, element), snapshot(edited, kind, other)))
        for element_id, element in new.items():
            if element_id not in old:
                changes.append(Change(element_id, kind, "added", None, snapshot(edited, kind, element)))
    return changes


__all__ = [
    "CONFIRM_ELEMENT_COUNT", "Change", "EditError", "collinear_chain", "EditOp", "OPS", "OpSpec", "Proposal", "apply_ops", "diff_plans", "find_element",
    "op_schemas", "op_spec", "opening_overhang_m", "resolve_signature", "signature_of", "snapshot", "wall_band",
]
