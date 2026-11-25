"""Selections for the AI assistant: hit-testing in the plan, chip labels, IFC GlobalId mapping and the read tools.

Positions are plan metres (local plan coordinates / ``units_per_metre``), like everywhere in ``samcloud.assistant``.
GlobalIds are the deterministic ones of the IFC export: ``samcloud.bim.scene_builder._uid`` (uuid5 of
``"<project>/<level>/<kind>/<key>"``) compressed by ``samcloud.bim.core.ifc.ids.guid_from_uuid``.  Scene kinds used by
the export: ``wall`` (merged / split pieces keep or suffix the plan id, e.g. ``W004a``), ``opening`` (the void),
``door`` / ``window`` (the filling), ``space`` (``R001``, extra pieces ``R001.1``), ``furniture``, ``ceiling-object``.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon

from samcloud.assistant.edits import ELEMENT_KINDS, _polygon_of, _room_polygon, find_element, snapshot, wall_band
from samcloud.bim.plan_model import BuildingPlan

HIT_PRIORITY = ("opening", "furniture", "ceiling_object", "wall", "room")
_SCENE_KINDS = {"wall": ("wall",), "opening": ("door", "window", "opening"), "furniture": ("furniture",), "room": ("space",),
                "ceiling_object": ("ceiling-object",)}
_PLAN_KIND = {"wall": "wall", "opening": "opening", "door": "opening", "window": "opening", "passage": "opening",
              "opening_element": "opening", "room": "room", "space": "room", "furniture": "furniture",
              "ceiling_object": "ceiling_object", "ceiling-object": "ceiling_object"}
_SPLIT_SUFFIXES = ("a", "b", "aa", "ab", "ba", "bb")
_ROOM_PIECES = 10

_CHIP_FALLBACK = {
    "de": {"wall": "Wand {id}", "door": "Tür {id} ({width} m)", "window": "Fenster {id} ({width} m)", "passage": "Durchgang {id} ({width} m)",
           "furniture": "{label} {id}", "ceiling_object": "{label} {id}", "room": "Raum {label}", "point": "Punkt x {x} / y {y}"},
    "en": {"wall": "Wall {id}", "door": "Door {id} ({width} m)", "window": "Window {id} ({width} m)", "passage": "Passage {id} ({width} m)",
           "furniture": "{label} {id}", "ceiling_object": "{label} {id}", "room": "Room {label}", "point": "Point x {x} / y {y}"},
}


def _r(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


# ---------------------------------------------------------------------------------------------------------- hit test

def _label(kind: str, element: Any) -> str:
    if kind == "furniture":
        return str(element.label or element.id)
    if kind == "room":
        return str(element.name or element.id)
    if kind == "ceiling_object":
        return str(element.get("label") or element.get("id"))
    if kind == "opening":
        return f"{element.kind} {element.id}"
    return str(element.id)


def hit_test(plan: BuildingPlan, x: float, y: float, *, tolerance_m: float = 0.15) -> dict[str, Any]:
    """The element under a click at plan metres (x, y): priority opening > furniture > ceiling_object > wall > room.

    Returns ``{"type": "element", "kind", "id", "label"}`` (openings also ``subkind``) or ``{"type": "point", "x", "y"}``.
    """
    upm = plan.units_per_metre or 1.0
    point = Point(float(x) * upm, float(y) * upm)
    reach = float(tolerance_m) * upm
    candidates: dict[str, list[tuple[float, float, Any]]] = {kind: [] for kind in HIT_PRIORITY}
    for opening in plan.openings:
        axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
        normal = np.asarray((-axis[1], axis[0]))
        centre = np.asarray(opening.center[:2], dtype=float)
        half_w = float(opening.width_m) * upm / 2.0
        half_t = max(float(opening.wall_thickness_m), 0.10) * upm / 2.0
        rectangle = Polygon([tuple(centre + axis * a + normal * b) for a, b in ((-half_w, -half_t), (half_w, -half_t), (half_w, half_t), (-half_w, half_t))])
        distance = rectangle.distance(point)
        if distance <= reach:
            candidates["opening"].append((distance, rectangle.area, opening))
    for item in plan.furniture:
        polygon = _polygon_of(item.footprint)
        if polygon is not None and (distance := polygon.distance(point)) <= reach:
            candidates["furniture"].append((distance, polygon.area, item))
    for item in plan.ceiling_objects:
        polygon = _polygon_of(item.get("footprint"))
        if polygon is not None and (distance := polygon.distance(point)) <= reach:
            candidates["ceiling_object"].append((distance, polygon.area, item))
    for wall in plan.walls:
        if len(wall.centreline) < 2:
            continue
        try:
            band = wall_band(wall, upm)
        except Exception:  # noqa: BLE001 - a degenerate wall cannot be hit
            continue
        if (distance := band.distance(point)) <= reach:
            candidates["wall"].append((distance, band.area, wall))
    for room in plan.rooms:
        polygon = _room_polygon(room)
        if polygon is not None and polygon.covers(point):
            candidates["room"].append((0.0, polygon.area, room))
    for kind in HIT_PRIORITY:
        if candidates[kind]:
            _distance, _area, element = min(candidates[kind], key=lambda item: (item[0], item[1]))
            result = {"type": "element", "kind": kind, "id": str(element["id"] if isinstance(element, dict) else element.id),
                      "label": _label(kind, element)}
            if kind == "opening":
                result["subkind"] = element.kind
                result["width_m"] = _r(element.width_m, 2)
            return result
    return {"type": "point", "x": _r(x), "y": _r(y)}


def chip_label(sel: dict[str, Any], lang: str | None = None) -> str:
    """Short text for a selection chip (``assistant.chip.*`` translations when present, built-in de/en otherwise)."""
    language = (lang or "de").lower()[:2]
    if sel.get("type") == "point":
        key, values = "point", {"x": f"{float(sel.get('x', 0.0)):.2f}", "y": f"{float(sel.get('y', 0.0)):.2f}"}
    else:
        kind = str(sel.get("kind") or "")
        key = str(sel.get("subkind") or kind) if kind == "opening" else kind
        width = sel.get("width_m")
        values = {"id": sel.get("id", ""), "label": sel.get("label", ""), "width": "" if width is None else f"{float(width):.2f}"}
    template = None
    try:
        from samcloud.app.i18n import catalogue

        template = catalogue(language).get(f"assistant.chip.{key}")
    except Exception:  # noqa: BLE001 - the built-in texts are enough without the UI catalogues
        template = None
    if not template:
        template = _CHIP_FALLBACK.get(language, _CHIP_FALLBACK["en"]).get(key, "{id}")
    try:
        return str(template).format(**values)
    except (KeyError, IndexError, ValueError):
        return str(sel.get("label") or sel.get("id") or "")


# ---------------------------------------------------------------------------------------------------------- GlobalIds

def _guid(plan: BuildingPlan, scene_kind: str, key: str) -> str:
    from samcloud.bim.core.ifc.ids import guid_from_uuid
    from samcloud.bim.scene_builder import _uid

    return guid_from_uuid(uuid.UUID(_uid(plan, scene_kind, key)))


def global_id_for(plan: BuildingPlan, kind: str, plan_id: str) -> str | None:
    """IFC GlobalId of a plan element (None when the id is unknown).

    ``kind``: wall | opening | room | furniture | ceiling_object (the plan kinds) or a scene kind (door, window, space,
    opening_element ...).  For ``opening`` the GlobalId of the IfcDoor / IfcWindow is returned (the void for a passage).
    """
    plan_kind = _PLAN_KIND.get(str(kind))
    if plan_kind is None:
        raise ValueError(f"unknown element kind {kind!r}")
    found = find_element(plan, plan_id)
    if found is None or found[0] != plan_kind:
        return None
    element = found[1]
    if plan_kind == "opening":
        if kind == "opening_element" or element.kind not in ("door", "window"):
            return _guid(plan, "opening", element.id)
        return _guid(plan, element.kind, element.id)
    return _guid(plan, _SCENE_KINDS[plan_kind][0], str(plan_id))


def global_id_table(plan: BuildingPlan) -> dict[str, tuple[str, str]]:
    """{GlobalId: (plan kind, plan id)} of every GlobalId the export can give a plan element."""
    table: dict[str, tuple[str, str]] = {}
    for wall in plan.walls:
        for key in (wall.id, *[f"{wall.id}{suffix}" for suffix in _SPLIT_SUFFIXES]):
            table[_guid(plan, "wall", key)] = ("wall", wall.id)
    for opening in plan.openings:
        for scene_kind in ("door", "window", "opening"):
            table[_guid(plan, scene_kind, opening.id)] = ("opening", opening.id)
    for room in plan.rooms:
        for key in (room.id, *[f"{room.id}.{index}" for index in range(1, _ROOM_PIECES)]):
            table[_guid(plan, "space", key)] = ("room", room.id)
    for item in plan.furniture:
        table[_guid(plan, "furniture", item.id)] = ("furniture", item.id)
    for item in plan.ceiling_objects:
        table[_guid(plan, "ceiling-object", str(item.get("id")))] = ("ceiling_object", str(item.get("id")))
    return table


def plan_id_for_global_id(plan: BuildingPlan, global_id: str) -> tuple[str, str] | None:
    """(plan kind, plan id) of an IFC GlobalId of this plan's export, else None."""
    return global_id_table(plan).get(str(global_id))


# --------------------------------------------------------------------------------------------------------- read tools

def _position(entry: dict[str, Any]) -> list[float] | None:
    if entry.get("center") is not None:
        return entry["center"]
    if entry.get("label_point") is not None:
        return entry["label_point"]
    line = entry.get("centreline")
    if line:
        return [_r((line[0][0] + line[-1][0]) / 2.0), _r((line[0][1] + line[-1][1]) / 2.0)]
    return None


def element_list(plan: BuildingPlan, filter: Any = None) -> list[dict[str, Any]]:  # noqa: A002 - tool parameter name
    """Compact list (id, type, label, main dimensions in metres, position in plan metres).

    ``filter``: None, a kind (``wall`` | ``opening`` | ``door`` | ``window`` | ``passage`` | ``furniture`` | ``room`` |
    ``ceiling_object``), a list of kinds, or a dict ``{"kind": ..., "room": <room id>, "label": <text>}``.
    """
    spec = filter if isinstance(filter, dict) else {"kind": filter} if filter is not None else {}
    kinds_raw = spec.get("kind")
    kinds = None if kinds_raw is None else ([kinds_raw] if isinstance(kinds_raw, str) else list(kinds_raw))
    room_polygon = None
    if spec.get("room"):
        found = find_element(plan, str(spec["room"]))
        room_polygon = _room_polygon(found[1]) if found is not None and found[0] == "room" else None
        if room_polygon is None:
            return []
    text = str(spec.get("label") or "").strip().lower()
    upm = plan.units_per_metre or 1.0
    rows: list[dict[str, Any]] = []
    for kind in ELEMENT_KINDS:
        for element in {"wall": plan.walls, "opening": plan.openings, "furniture": plan.furniture, "room": plan.rooms,
                        "ceiling_object": plan.ceiling_objects}[kind]:
            entry = snapshot(plan, kind, element)
            subtype = entry.get("type")
            if kinds is not None and kind not in kinds and subtype not in kinds:
                continue
            position = _position(entry)
            if room_polygon is not None and kind != "room":
                if position is None or not room_polygon.buffer(0.3 * upm).covers(Point(position[0] * upm, position[1] * upm)):
                    continue
            label = entry.get("label") or entry.get("name") or entry["id"]
            if text and text not in str(label).lower() and text not in str(entry["id"]).lower():
                continue
            row: dict[str, Any] = {"id": entry["id"], "type": subtype, "label": label, "position": position}
            for key in ("thickness_m", "length_m", "width_m", "sill_m", "head_m", "size_m", "height_m", "area_m2", "host_wall_id", "evidence"):
                if entry.get(key) is not None:
                    row[key] = entry[key]
            rows.append(row)
    return rows


def element_details(plan: BuildingPlan, element_id: str) -> dict[str, Any]:
    """Everything the read tools tell about one element (metres), with its IFC GlobalId and neighbours."""
    found = find_element(plan, element_id)
    if found is None:
        return {"id": element_id, "found": False}
    kind, element = found
    upm = plan.units_per_metre or 1.0
    details = snapshot(plan, kind, element)
    details["found"] = True
    details["kind"] = kind
    try:
        details["global_id"] = global_id_for(plan, kind, element_id)
    except Exception:  # noqa: BLE001 - the mapping is informative only
        details["global_id"] = None
    if kind == "wall":
        band = wall_band(element, upm)
        details["openings"] = [o.id for o in plan.openings if o.host_wall_id == element.id]
        details["rooms"] = [r.id for r in plan.rooms if (p := _room_polygon(r)) is not None and p.distance(band) <= 0.05 * upm]
        details["joins"] = [w.id for w in plan.walls if w.id != element.id and len(w.centreline) >= 2 and wall_band(w, upm).distance(band) <= 1e-6 * upm]
        details["height_source"] = element.height_source
        details["face_a_side"] = "given" if element.face_a else "right of the centreline direction"
    elif kind == "opening":
        details["leaf_detected"] = element.leaf_detected
        details["hinge_side"] = element.hinge_side
        details["swing_side"] = element.swing_side
        details["glazed"] = bool(element.glazed)
        details["height_source"] = element.height_source
    elif kind == "furniture":
        polygon = _polygon_of(element.footprint)
        details["catalogue_kind"] = element.catalogue_kind
        details["instance_id"] = element.instance_id
        details["shape_source"] = element.shape_source
        details["mounting"] = element.mounting
        details["room"] = next((r.id for r in plan.rooms if polygon is not None and (p := _room_polygon(r)) is not None and p.covers(polygon.centroid)), None)
        details["footprint"] = [[_r(px / upm), _r(py / upm)] for px, py in element.footprint]
    elif kind == "room":
        polygon = _room_polygon(element)
        details["outline"] = [] if polygon is None else [[_r(px / upm), _r(py / upm)] for px, py in list(polygon.exterior.coords)[:-1]]
        details["furniture"] = [f.id for f in plan.furniture if polygon is not None and (q := _polygon_of(f.footprint)) is not None and polygon.covers(q.centroid)]
        details["openings"] = [o.id for o in plan.openings if polygon is not None and polygon.distance(Point(o.center[0], o.center[1])) <= 0.3 * upm]
        details["floor_source"] = element.floor_source
        details["ceiling_source"] = element.ceiling_source
    return details
