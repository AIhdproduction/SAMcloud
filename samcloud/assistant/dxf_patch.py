"""DXF patch layer: redraw only the plan elements the AI assistant edited.

The 2D DXF is not written from ``plan_model.json`` but from the cached clean-plan ``DxfPlanResult`` (neutral entity
dicts in the same local plan frame).  :func:`apply_dxf_patch` removes the old entities of every changed element (found
by geometry against the base element) and draws the new ones with the pipeline's own helpers, layers, blocks and text
formats:

* doors / windows / passages: entities whose ``_opening_center`` is the base opening centre are replaced by
  ``samcloud.ortho.vector.indoor.opening_entities`` of the edited opening (same block attributes and label formats);
* walls: the wall mass (``A-WALL`` closed polylines, ``A-WALL-PATT`` hatches) loses the old band and gains the new one
  (centreline +- thickness / 2, opening voids cut out); only the touched pieces are rewritten;
* rooms: the ``A-ANNO`` label with the base room's ``_room_name`` (``room_label_text`` format) and the ``A-AREA``
  outline follow; a room added by full control (split, new wall) gets a label and an outline like the other rooms;
  an opening moved into another wall redraws both host walls;
* furniture: outline polyline, catalogue inserts, label and detail lines of the base item move / rotate rigidly with
  it (or are removed / drawn new: ``Stuhl`` or ``Tisch h=0.75`` like the pipeline); manual items go on
  ``A-FURN-MANUAL`` (style from ``config/dxf_layers.yaml`` ``assistant_layers``).

Untouched entities stay the very same dicts in the same order; without changes the input object is returned.
Texts are produced exactly like the pipeline (German / neutral); export-time translation happens elsewhere.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from functools import lru_cache
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from samcloud.assistant.edits import Change, furniture_height_m, wall_band
from samcloud.assistant.evidence import MANUAL_LAYER
from samcloud.bim.plan_model import BuildingPlan, PlanFurniture, PlanOpening

LOGGER = logging.getLogger(__name__)

FURNITURE_LAYERS = ("A-FURN", "A-EINB", "A-OBEN", "P-SANI", "A-FURN-DETL")
OPENING_LAYERS = ("A-DOOR", "A-WIND")
WALL_LAYER, WALL_HATCH_LAYER, ANNO_LAYER, AREA_LAYER = "A-WALL", "A-WALL-PATT", "A-ANNO", "A-AREA"
OPENING_MATCH_M = 0.01  # an entity belongs to an opening when its _opening_center is this close to the opening centre
INSERT_OWNER_REACH_M = 0.15  # a catalogue insert point (block origin at a corner) may lie this far outside the outline
TEXT_OWNER_REACH_M = 0.02
DETAIL_OWNER_REACH_M = 0.05
OUTLINE_MATCH_IOU = 0.5
ROOM_OUTLINE_MATCH_IOU = 0.8
MIN_WALL_PIECE_M2 = 0.001
DEFAULT_FURNITURE_TEXT_HEIGHT_M = 0.08
_FALLBACK_MANUAL_STYLE = {"color": 6, "linetype": "DASHED", "lineweight": 18}


@lru_cache(maxsize=1)
def assistant_layer_styles() -> dict[str, dict[str, Any]]:
    """``assistant_layers`` of config/dxf_layers.yaml ({name: {color, linetype, lineweight}}), with a built-in fallback."""
    styles: dict[str, dict[str, Any]] = {MANUAL_LAYER: dict(_FALLBACK_MANUAL_STYLE)}
    try:
        import yaml

        from samcloud.core.paths import CONFIG_DIRECTORY

        data = yaml.safe_load((CONFIG_DIRECTORY / "dxf_layers.yaml").read_text(encoding="utf-8")) or {}
        for record in data.get("assistant_layers") or []:
            if isinstance(record, dict) and record.get("name"):
                styles[str(record["name"])] = {key: value for key, value in record.items() if key != "name"}
    except Exception as error:  # noqa: BLE001 - the fallback style keeps the patch working
        LOGGER.debug("assistant layer styles: fallback (%s)", error)
    return styles


def _key(points: Any) -> frozenset[tuple[float, float]]:
    return frozenset((round(float(p[0]), 3), round(float(p[1]), 3)) for p in points or [])


def _poly(points: Any, holes: Any = None) -> Polygon | None:
    try:
        polygon = Polygon([(float(p[0]), float(p[1])) for p in points], [[(float(p[0]), float(p[1])) for p in h] for h in (holes or [])])
    except (TypeError, ValueError, IndexError):
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda g: g.area)
    return polygon if isinstance(polygon, Polygon) and not polygon.is_empty else None


def _iou(a: Polygon, b: Polygon) -> float:
    try:
        union = a.union(b).area
        return float(a.intersection(b).area / union) if union > 0 else 0.0
    except Exception:  # noqa: BLE001 - degenerate shapes never match
        return 0.0


def _ring(polygon: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in list(polygon.exterior.coords)[:-1]]


def _holes(polygon: Polygon) -> list[list[list[float]]]:
    return [[[float(x), float(y)] for x, y in list(ring.coords)[:-1]] for ring in polygon.interiors]


def _polygons(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [g for g in getattr(geometry, "geoms", []) if g.geom_type == "Polygon" and not g.is_empty]


def _rigid(base: list[list[float]], new: list[list[float]], scale: float) -> tuple[float, np.ndarray, np.ndarray] | None:
    """(angle deg, rotation 2x2, translation) mapping ``base`` onto ``new`` vertex by vertex, None when not rigid."""
    if len(base) != len(new) or len(base) < 2:
        return None
    a, b = np.asarray(base, dtype=float), np.asarray(new, dtype=float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _s, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        return None
    translation = cb - rotation @ ca
    residual = float(np.max(np.linalg.norm(a @ rotation.T + translation - b, axis=1)))
    if residual > 1e-6 * max(scale, 1.0):
        return None
    return math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])), rotation, translation


class _Patch:
    def __init__(self, result: Any, base: BuildingPlan, edited: BuildingPlan) -> None:
        self.result = result
        self.base, self.edited = base, edited
        self.upm = float(edited.units_per_metre or base.units_per_metre or 1.0)
        self.layers: dict[str, list[dict[str, Any]]] = {name: list(entities) for name, entities in result.layers.items()}
        self.touched: set[str] = set()
        self.new_layers: set[str] = set()
        self.blocks: set[str] = set(result.blocks)

    # ------------------------------------------------------------------------------------------------------- helpers
    def _layer(self, name: str) -> list[dict[str, Any]]:
        if name not in self.layers:
            self.layers[name] = []
            self.new_layers.add(name)
        self.touched.add(name)
        return self.layers[name]

    def _remove(self, layer: str, indices: list[int]) -> int | None:
        """Remove entities by index; return the position the first one had (to insert replacements there)."""
        if not indices:
            return None
        drop = set(indices)
        self.layers[layer] = [entity for index, entity in enumerate(self.layers[layer]) if index not in drop]
        self.touched.add(layer)
        return min(indices)

    def _insert(self, layer: str, position: int | None, entities: list[dict[str, Any]]) -> None:
        target = self._layer(layer)
        if position is None or position > len(target):
            target.extend(entities)
        else:
            target[position:position] = entities

    # ------------------------------------------------------------------------------------------------------ openings
    def openings(self, ids: list[str]) -> None:
        base = {o.id: o for o in self.base.openings}
        edited = {o.id: o for o in self.edited.openings}
        reach = OPENING_MATCH_M * self.upm
        for opening_id in ids:
            old, new = base.get(opening_id), edited.get(opening_id)
            template: dict[str, Any] = {}
            positions: dict[str, int | None] = {}
            if old is not None:
                centre = np.asarray(old.center[:2], dtype=float)
                for layer in OPENING_LAYERS:
                    indices = [
                        index for index, entity in enumerate(self.layers.get(layer, []))
                        if entity.get("_opening_center") is not None
                        and float(np.linalg.norm(np.asarray(entity["_opening_center"][:2], dtype=float) - centre)) <= reach
                    ]
                    for index in indices:
                        entity = self.layers[layer][index]
                        if entity.get("type") == "insert":
                            template.update({key: entity.get(key) for key in ("_hinge_side", "_swing_side", "_leaf_mode") if entity.get(key) is not None})
                            template["insert"] = True
                            if entity.get("name") == "WINDOW":
                                template["window_sign"] = -1.0 if float(entity.get("yscale", 1.0)) < 0 else 1.0
                        if entity.get("_opening_label"):
                            template["label"] = True
                    positions[layer] = self._remove(layer, indices)
            if new is not None:
                entities = self._opening_entities(new, template)
                layer = "A-WIND" if new.kind == "window" else "A-DOOR"
                self._insert(layer, positions.get(layer), entities)
                self.blocks.update(str(e["name"]) for e in entities if e.get("type") == "insert")

    def _opening_entities(self, opening: PlanOpening, template: dict[str, Any]) -> list[dict[str, Any]]:
        from samcloud.ortho.vector.indoor import Opening, opening_entities

        kind = {"door": "DOOR", "window": "WINDOW"}.get(opening.kind, "passage")
        leaf_detected = opening.leaf_detected
        if kind == "DOOR" and template.get("label") and not template.get("insert"):
            leaf_detected = False  # the pipeline drew this door without a leaf: keep it that way
        swing = opening.swing_side if opening.swing_side in (1, -1) else template.get("_swing_side")
        spec = Opening(
            kind=kind, center=(float(opening.center[0]), float(opening.center[1])), width=float(opening.width_m) * self.upm,
            axis_angle=float(opening.axis_angle), wall_thickness=float(opening.wall_thickness_m) * self.upm,
            evidence=str(opening.evidence or "gap"), leaf_angle_deg=opening.leaf_angle_deg, leaf_mode=template.get("_leaf_mode"),
            hinge_side=opening.hinge_side or template.get("_hinge_side"), swing_side=None if swing is None else int(swing),
            leaf_detected=leaf_detected, sill_m=float(opening.sill_m), head_m=float(opening.head_m),
            mullion_u=tuple(float(v) for v in opening.mullion_u or ()), glazed=bool(opening.glazed), confidence=opening.confidence,
        )
        entities = opening_entities([spec], None, width_scale=1.0 / self.upm, height_scale=1.0)
        sign = float(template.get("window_sign", 1.0))
        for entity in entities:
            if sign < 0 and kind == "WINDOW":
                if entity.get("type") == "insert" and entity.get("name") == "WINDOW":
                    entity["yscale"] = -float(entity["yscale"])
                elif entity.get("_mullion"):
                    start = np.asarray(entity["start"], dtype=float)
                    entity["end"] = (start - (np.asarray(entity["end"], dtype=float) - start)).tolist()
            entity["_on_face"] = True
            entity["_assistant_edit"] = True
        return entities

    # --------------------------------------------------------------------------------------------------------- walls
    def walls(self, ids: list[str]) -> None:
        upm = self.upm
        base = {w.id: w for w in self.base.walls}
        edited = {w.id: w for w in self.edited.walls}
        changed = set(ids)
        old_bands = [wall_band(base[i], upm) for i in ids if i in base]
        new_bands = [wall_band(edited[i], upm) for i in ids if i in edited]
        if not old_bands and not new_bands:
            return
        others = unary_union([wall_band(w, upm) for w in self.edited.walls if w.id not in changed])
        old_union = unary_union(old_bands) if old_bands else Polygon()
        remove = old_union.difference(others.buffer(1e-6 * upm)) if not others.is_empty else old_union
        add = unary_union(new_bands) if new_bands else Polygon()
        voids = []
        for opening in self.edited.openings:
            axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
            normal = np.asarray((-axis[1], axis[0]))
            centre = np.asarray(opening.center[:2], dtype=float)
            half_w = float(opening.width_m) * upm / 2.0
            half_t = (max(float(opening.wall_thickness_m), 0.01) / 2.0 + 0.01) * upm
            rectangle = Polygon([tuple(centre + axis * a + normal * b) for a, b in ((-half_w, -half_t), (half_w, -half_t), (half_w, half_t), (-half_w, half_t))])
            if not add.is_empty and rectangle.intersects(add):
                voids.append(rectangle)
        if voids:
            add = add.difference(unary_union(voids))
        affected = unary_union([remove, add]).buffer(1e-6 * upm) if not (remove.is_empty and add.is_empty) else Polygon()
        if affected.is_empty:
            return
        minimum = MIN_WALL_PIECE_M2 * upm * upm
        for layer, kind in ((WALL_LAYER, "polyline"), (WALL_HATCH_LAYER, "hatch")):
            entities = self.layers.get(layer)
            if entities is None and layer == WALL_HATCH_LAYER:
                continue
            entities = entities or []
            touched: list[int] = []
            pieces: list[Polygon] = []
            for index, entity in enumerate(entities):
                if entity.get("type") != kind or (kind == "polyline" and not entity.get("closed")) or len(entity.get("points") or []) < 3:
                    continue
                polygon = _poly(entity["points"], entity.get("holes"))
                if polygon is not None and polygon.intersects(affected):
                    touched.append(index)
                    pieces.append(polygon)
            mass = unary_union(pieces) if pieces else Polygon()
            mass = mass.difference(remove).union(add) if not mass.is_empty else add
            mass = mass.buffer(1e-7 * upm).buffer(-1e-7 * upm)
            template = {k: v for k, v in entities[touched[0]].items() if k.startswith("_")} if touched else {}
            replacement = []
            for polygon in sorted(_polygons(mass), key=lambda p: (-p.area)):
                if polygon.area < minimum:
                    continue
                entity = {"type": kind, "points": _ring(polygon)}
                if kind == "polyline":
                    entity["closed"] = True
                entity["holes"] = _holes(polygon)
                entity.update(template)
                entity["_assistant_edit"] = True
                replacement.append(entity)
            if layer not in self.layers and not replacement:
                continue
            position = self._remove(layer, touched) if touched else None
            self._insert(layer, position, replacement)

    # --------------------------------------------------------------------------------------------------------- rooms
    def rooms(self, ids: list[str]) -> None:
        from samcloud.ortho.vector.room_names import room_label_text

        base = {r.id: r for r in self.base.rooms}
        edited = {r.id: r for r in self.edited.rooms}
        for room_id in ids:
            old, new = base.get(room_id), edited.get(room_id)
            if old is None:
                if new is not None:
                    self._new_room(new, room_label_text)
                continue
            old_polygon = _poly(old.polygon.get("exterior") or [], old.polygon.get("holes"))
            new_polygon = None if new is None else _poly(new.polygon.get("exterior") or [], new.polygon.get("holes"))
            # label
            texts = self.layers.get(ANNO_LAYER, [])
            candidates = [
                (index, entity) for index, entity in enumerate(texts)
                if entity.get("type") == "text" and entity.get("_room_name") == old.name
            ]
            if candidates:
                anchor = np.asarray(old.label_point[:2], dtype=float)

                def rank(item: tuple[int, dict[str, Any]]) -> tuple[int, float]:
                    point = np.asarray(item[1]["point"][:2], dtype=float)
                    inside = old_polygon is not None and old_polygon.buffer(1e-6).covers(Point(point))
                    return (0 if inside else 1, float(np.linalg.norm(point - anchor)))

                index, entity = min(candidates, key=rank)
                if new is None:
                    self._remove(ANNO_LAYER, [index])
                else:
                    point = entity["point"]
                    if new_polygon is not None and not new_polygon.covers(Point(float(point[0]), float(point[1]))):
                        point = [float(new.label_point[0]), float(new.label_point[1])]
                    texts[index] = {**entity, "text": room_label_text(new.name, float(new.area_m2)), "point": list(point),
                                    "_room_name": new.name, "_assistant_edit": True}
                    self.touched.add(ANNO_LAYER)
            # room outline
            if old_polygon is None:
                continue
            areas = self.layers.get(AREA_LAYER, [])
            best: tuple[float, int] | None = None
            for index, entity in enumerate(areas):
                if entity.get("type") != "polyline" or not entity.get("closed") or len(entity.get("points") or []) < 3:
                    continue
                polygon = _poly(entity["points"], entity.get("holes"))
                if polygon is None:
                    continue
                score = _iou(polygon, old_polygon)
                if score >= ROOM_OUTLINE_MATCH_IOU and (best is None or score > best[0]):
                    best = (score, index)
            if best is None:
                continue
            if new_polygon is None:
                self._remove(AREA_LAYER, [best[1]])
            elif _key(areas[best[1]]["points"]) != _key(_ring(new_polygon)) or old_polygon.symmetric_difference(new_polygon).area > 1e-9:
                areas[best[1]] = {**areas[best[1]], "points": _ring(new_polygon), "holes": _holes(new_polygon), "_assistant_edit": True}
                self.touched.add(AREA_LAYER)

    def _new_room(self, room: Any, label_text: Any) -> None:
        """A room the assistant added (split / new wall): label and outline drawn like the pipeline's other rooms
        (attributes copied from an existing room label / outline; nothing is drawn without such a template)."""
        polygon = _poly(room.polygon.get("exterior") or [], room.polygon.get("holes"))
        if polygon is None:
            return
        texts = self.layers.get(ANNO_LAYER, [])
        template = next((entity for entity in texts if entity.get("type") == "text" and entity.get("_room_name")), None)
        if template is not None:
            point = [float(room.label_point[0]), float(room.label_point[1])]
            self._insert(ANNO_LAYER, None, [{**template, "text": label_text(room.name, float(room.area_m2)), "point": point,
                                             "_room_name": room.name, "_assistant_edit": True}])
        areas = self.layers.get(AREA_LAYER, [])
        outline = next((entity for entity in areas if entity.get("type") == "polyline" and entity.get("closed")), None)
        if outline is not None:
            self._insert(AREA_LAYER, None, [{**outline, "points": _ring(polygon), "holes": _holes(polygon), "_assistant_edit": True}])

    # ----------------------------------------------------------------------------------------------------- furniture
    def furniture(self, ids: list[str]) -> None:
        upm = self.upm
        base = {f.id: f for f in self.base.furniture}
        edited = {f.id: f for f in self.edited.furniture}
        base_polygons = {fid: p for fid, item in base.items() if (p := _poly(item.footprint)) is not None}
        layers = [name for name in self.layers if name in FURNITURE_LAYERS or name == MANUAL_LAYER]

        def owner_by_point(point: Point, reach: float) -> str | None:
            hits = [(polygon.area, fid) for fid, polygon in base_polygons.items() if polygon.buffer(reach).covers(point)]
            return min(hits)[1] if hits else None

        def owner_by_outline(polygon: Polygon) -> str | None:
            scores = [(_iou(polygon, other), fid) for fid, other in base_polygons.items()]
            scores = [item for item in scores if item[0] >= OUTLINE_MATCH_IOU]
            return max(scores)[1] if scores else None

        text_height = self._furniture_text_height()
        for furniture_id in ids:
            old, new = base.get(furniture_id), edited.get(furniture_id)
            matched: dict[str, list[int]] = {}
            outline_hits: set[tuple[str, int]] = set()
            if old is not None and furniture_id in base_polygons:
                polygon = base_polygons[furniture_id]
                keys = {_key(old.footprint), _key((old.fit or {}).get("plan_footprint"))}
                if new is not None:
                    keys |= {_key(new.footprint), _key((new.fit or {}).get("plan_footprint"))}
                keys.discard(frozenset())
                furn_ids: set[Any] = set()
                for layer in layers:
                    for index, entity in enumerate(self.layers[layer]):
                        kind = entity.get("type")
                        if kind == "polyline" and entity.get("closed") and len(entity.get("points") or []) >= 3:
                            shape = _poly(entity["points"])
                            if _key(entity["points"]) in keys or (shape is not None and owner_by_outline(shape) == furniture_id):
                                matched.setdefault(layer, []).append(index)
                                outline_hits.add((layer, index))
                                if entity.get("_furn_id") is not None:
                                    furn_ids.add(entity["_furn_id"])
                        elif kind == "insert" and entity.get("point") is not None:
                            point = Point(float(entity["point"][0]), float(entity["point"][1]))
                            if owner_by_point(point, INSERT_OWNER_REACH_M * upm) == furniture_id:
                                matched.setdefault(layer, []).append(index)
                for layer in layers:
                    for index, entity in enumerate(self.layers[layer]):
                        if entity.get("type") == "polyline" and not entity.get("closed") and layer == "A-FURN-DETL":
                            points = entity.get("points") or []
                            own = entity.get("_furn_id") is not None and entity.get("_furn_id") in furn_ids
                            if not own and points:
                                inside = all(polygon.buffer(DETAIL_OWNER_REACH_M * upm).covers(Point(float(p[0]), float(p[1]))) for p in points)
                                own = inside and owner_by_point(Point(float(points[0][0]), float(points[0][1])), DETAIL_OWNER_REACH_M * upm) == furniture_id
                            if own:
                                matched.setdefault(layer, []).append(index)
                for index, entity in enumerate(self.layers.get(ANNO_LAYER, [])):
                    if entity.get("type") != "text" or "_room_name" in entity or entity.get("_opening_label") or entity.get("point") is None:
                        continue
                    own = entity.get("_furn_id") is not None and entity.get("_furn_id") in furn_ids
                    if not own and entity.get("_furn_id") is None:
                        point = Point(float(entity["point"][0]), float(entity["point"][1]))
                        own = owner_by_point(point, TEXT_OWNER_REACH_M * upm) == furniture_id
                    if own:
                        matched.setdefault(ANNO_LAYER, []).append(index)
            if new is None:
                for layer, indices in matched.items():
                    self._remove(layer, indices)
                continue
            transform = None
            if old is not None and matched:
                transform = _rigid(old.footprint, new.footprint, upm)
            if transform is not None and old is not None and old.label == new.label and new.layer == old.layer:
                self._transform(matched, outline_hits, transform, new)
                continue
            if transform is not None and old is not None and new.layer == old.layer:
                self._transform(matched, outline_hits, transform, new)
                self._relabel(matched, old, new)
                continue
            # not rigid (or a new / re-layered item): redraw outline + label, keep nothing of the old drawing
            positions = {layer: self._remove(layer, indices) for layer, indices in matched.items()}
            layer = self._furniture_layer(new)
            self._dedupe_outline(new.footprint)
            outline = {"type": "polyline", "points": [[float(x), float(y)] for x, y in new.footprint], "closed": True, "holes": [],
                       "_label": new.label, "_semantic_label": new.semantic_label, "_assistant_edit": True}
            if (new.fit or {}).get("evidence") == "manual":
                outline["_evidence"] = "manual"
            self._insert(layer, positions.get(layer), [outline])
            polygon = _poly(new.footprint)
            if polygon is not None:
                point = polygon.representative_point()
                text = {"type": "text", "text": self._label_text(new), "point": [float(point.x), float(point.y)],
                        "height": text_height, "rotation": 0.0, "_assistant_edit": True}
                self._insert(ANNO_LAYER, positions.get(ANNO_LAYER), [text])

    def _furniture_text_height(self) -> float:
        heights = [float(e["height"]) for e in self.layers.get(ANNO_LAYER, []) if e.get("type") == "text" and e.get("_furn_id") is not None and e.get("height")]
        return float(np.median(heights)) if heights else DEFAULT_FURNITURE_TEXT_HEIGHT_M * self.upm

    def _label_text(self, item: PlanFurniture) -> str:
        """Pipeline format (``samcloud.ortho.plan._furniture_label_text``): "Stuhl" for a chair, else "Name h=0.76"."""
        if str(item.semantic_label or "").strip().lower() == "chair":
            return item.label
        return f"{item.label} h={furniture_height_m(self.edited, item):.2f}"

    def _furniture_layer(self, item: PlanFurniture) -> str:
        if item.layer in self.layers:
            return item.layer
        if item.layer == MANUAL_LAYER or item.layer in assistant_layer_styles():
            return item.layer
        return "A-FURN"

    def _dedupe_outline(self, footprint: list[list[float]]) -> None:
        """Drop outlines already drawn for this footprint (``apply_reconstructed_outlines`` draws added items too)."""
        key = _key(footprint)
        for layer in [name for name in self.layers if name in FURNITURE_LAYERS or name == MANUAL_LAYER]:
            indices = [i for i, e in enumerate(self.layers[layer]) if e.get("type") == "polyline" and e.get("closed") and _key(e.get("points")) == key]
            self._remove(layer, indices)

    def _transform(self, matched: dict[str, list[int]], outline_hits: set[tuple[str, int]], transform: tuple[float, np.ndarray, np.ndarray],
                   new: PlanFurniture) -> None:
        angle, rotation, translation = transform

        def move(point: Any) -> list[float]:
            moved = rotation @ np.asarray(point[:2], dtype=float) + translation
            return [float(moved[0]), float(moved[1]), *[float(v) for v in point[2:]]]

        identity = abs(angle) < 1e-9 and float(np.linalg.norm(translation)) < 1e-12
        for layer, indices in matched.items():
            entities = self.layers[layer]
            for index in indices:
                entity = dict(entities[index])
                if not identity:
                    if entity.get("points") is not None:
                        if (layer, index) in outline_hits and _key(entity["points"]) in (_key((new.fit or {}).get("plan_footprint")), _key(new.footprint)):
                            pass  # already drawn at the new place (dxf_outline redrew it from the edited plan)
                        else:
                            entity["points"] = [move(p) for p in entity["points"]]
                            if entity.get("holes"):
                                entity["holes"] = [[move(p) for p in hole] for hole in entity["holes"]]
                    for key in ("point", "start", "end"):
                        if entity.get(key) is not None:
                            entity[key] = move(entity[key])
                    if angle and entity.get("rotation") is not None:
                        entity["rotation"] = float(entity["rotation"]) + angle
                    if entity.get("_solid_parts"):
                        entity["_solid_parts"] = [
                            {**part, "footprint": [move(p) for p in part.get("footprint") or []]} if isinstance(part, dict) else part
                            for part in entity["_solid_parts"]
                        ]
                entity["_assistant_edit"] = True
                entities[index] = entity
            self.touched.add(layer)

    def _relabel(self, matched: dict[str, list[int]], old: PlanFurniture, new: PlanFurniture) -> None:
        texts = self.layers.get(ANNO_LAYER, [])
        found = False
        for index in matched.get(ANNO_LAYER, []):
            entity = dict(texts[index])
            entity["text"] = self._label_text(new)
            entity["_assistant_edit"] = True
            texts[index] = entity
            found = True
        for layer, indices in matched.items():
            for index in indices:
                entity = self.layers[layer][index]
                if entity.get("type") == "polyline" and "_label" in entity:
                    self.layers[layer][index] = {**entity, "_label": new.label, "_semantic_label": new.semantic_label, "_assistant_edit": True}
                elif entity.get("type") == "insert" and isinstance(entity.get("attributes"), dict) and "LABEL" in entity["attributes"]:
                    self.layers[layer][index] = {**entity, "attributes": {**entity["attributes"], "LABEL": new.label}, "_assistant_edit": True}
        if not found:
            polygon = _poly(new.footprint)
            if polygon is not None:
                point = polygon.representative_point()
                self._insert(ANNO_LAYER, None, [{"type": "text", "text": self._label_text(new), "point": [float(point.x), float(point.y)],
                                                 "height": self._furniture_text_height(), "rotation": 0.0, "_assistant_edit": True}])
        self.touched.add(ANNO_LAYER)

    # -------------------------------------------------------------------------------------------------------- result
    def finish(self) -> Any:
        from samcloud.dxf.result import LayerStatus

        statuses = dict(self.result.statuses)
        for name in self.touched:
            count = len(self.layers.get(name, []))
            old = statuses.get(name)
            statuses[name] = LayerStatus(name, "ok" if count else "empty", (old.reason if old is not None and count else None) if count else "no geometry survived filtering", count)
        colors, linetypes = dict(self.result.layer_colors), dict(self.result.layer_linetypes)
        hatches, styles = dict(self.result.layer_hatches), {k: dict(v) for k, v in self.result.layer_styles.items()}
        for name in self.new_layers:
            style = assistant_layer_styles().get(name)
            if style is None:
                continue
            colors.setdefault(name, int(style.get("color", 7)))
            linetypes.setdefault(name, str(style.get("linetype", "CONTINUOUS")))
            hatches.setdefault(name, False)
            if style.get("lineweight") is not None:
                styles.setdefault(name, {})["lineweight"] = int(style["lineweight"])
        layers = {name: entities for name, entities in self.layers.items() if entities or name in self.result.layers}
        return replace(self.result, layers=layers, statuses=statuses, layer_colors=colors, layer_linetypes=linetypes,
                       layer_hatches=hatches, layer_styles=styles, blocks=sorted(self.blocks))


def apply_dxf_patch(result: Any, base: BuildingPlan, edited: BuildingPlan, changes: list[Change]) -> Any:
    """``result`` (DxfPlanResult) with only the changed elements redrawn; the very same object without changes."""
    if not changes:
        return result
    by_kind: dict[str, list[str]] = {}
    for change in changes:
        ids = by_kind.setdefault(change.kind, [])
        if change.element_id not in ids:
            ids.append(change.element_id)
    patch = _Patch(result, base, edited)
    # an opening moved into another wall: both host walls are redrawn (the void moves with the opening)
    base_hosts = {o.id: o.host_wall_id for o in base.openings}
    for opening in edited.openings:
        if opening.id in by_kind.get("opening", []) and opening.id in base_hosts and base_hosts[opening.id] != opening.host_wall_id:
            walls = by_kind.setdefault("wall", [])
            for wall_id in (base_hosts[opening.id], opening.host_wall_id):
                if wall_id and wall_id not in walls and any(w.id == wall_id for w in edited.walls):
                    walls.append(wall_id)
    if by_kind.get("opening"):
        patch.openings(by_kind["opening"])
    if by_kind.get("wall"):
        patch.walls(by_kind["wall"])
    if by_kind.get("room"):
        patch.rooms(by_kind["room"])
    if by_kind.get("furniture"):
        patch.furniture(by_kind["furniture"])
    if not patch.touched:
        return result
    return patch.finish()
