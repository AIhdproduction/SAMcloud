"""2D DXF: furniture outlines drawn as the plan projection of the 3D reconstruction.

The clean-plan ``result.json`` keeps the fitted furniture outlines; ``plan_model.json`` beside it knows, per reconstructed
item, the outline it started from (``fit["plan_footprint"]``) and its true projected outline (``footprint``).  At DXF
export a closed furniture polyline whose points equal such a start outline is redrawn with the projected outline.
Catalogue block inserts (WC, washbasin, bed blocks ...) and kitchen runs are never touched.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from samcloud.bim.furniture_recon.families import family_of

CEILING_LAYER = "A-CLNG-OBJ"
CEILING_LAYER_COLOUR = 2
CEILING_LINETYPE = "DASHED"
FURNITURE_LAYERS = ("A-FURN", "A-EINB", "A-OBEN", "P-SANI", "A-FURN-DETL")
_KITCHEN = frozenset({"kitchen_base", "worktop", "kitchen_insert"})


def _key(points: list[Any]) -> frozenset[tuple[float, float]]:
    return frozenset((round(float(p[0]), 3), round(float(p[1]), 3)) for p in points)


def reconstructed_outlines(plan_model_file: Path | str) -> dict[frozenset[tuple[float, float]], list[list[float]]]:
    """{start outline key: projected outline} of the reconstructed (parametric / mesh) items of a plan_model.json."""
    path = Path(plan_model_file)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    mapping: dict[frozenset[tuple[float, float]], list[list[float]]] = {}
    for item in payload.get("furniture", []):
        fit = item.get("fit") or {}
        start = fit.get("plan_footprint")
        if item.get("shape_source") not in ("parametric", "mesh") or not start or len(item.get("footprint") or []) < 3:
            continue
        if family_of(item.get("label"), item.get("semantic_label"), item.get("catalogue_kind")) in _KITCHEN:
            continue
        mapping[_key(start)] = [[float(x), float(y)] for x, y in item["footprint"]]
    return mapping


def added_items(plan_model_file: Path | str) -> list[tuple[str, list[list[float]]]]:
    """(layer, outline) of the items the reconstruction added from scan instances (``fit["added"]``)."""
    path = Path(plan_model_file)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [
        (str(item.get("layer") or "A-FURN"), [[float(x), float(y)] for x, y in item["footprint"]])
        for item in payload.get("furniture", [])
        if (item.get("fit") or {}).get("added") and len(item.get("footprint") or []) >= 3
    ]


# A tall row unit is already drawn where a closed built-in outline of the plan covers this much of it (a tall unit of the wall stage).
_ROW_UNIT_DRAWN_SHARE = 0.50


def row_units(plan_model_file: Path | str) -> list[tuple[list[list[float]], str, float]]:
    """(outline, "HS" / "KS", units per metre) of the tall kitchen row units measured for the 3D model (``kitchen_rows`` of a
    plan_model.json); "KS" = a fridge that keeps its own item in the row.
    """
    path = Path(plan_model_file)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = ((payload.get("source") or {}).get("depth_furniture") or {}).get("kitchen_rows") or []
    ids = {str(unit) for row in rows for unit in row.get("units") or []}
    fridges = {str(item.get("id")) for row in rows for item in row.get("kept") or []}
    upm = float(payload.get("units_per_metre") or 1.0)
    return [
        ([[float(x), float(y)] for x, y in item["footprint"]], "KS" if str(item.get("id")) in fridges else "HS", upm)
        for item in payload.get("furniture", [])
        if str(item.get("id")) in ids and len(item.get("footprint") or []) >= 3
        and (item.get("catalogue_kind") in ("tall_unit", "oven_tower") or str(item.get("id")) in fridges)
    ]


def row_replaced_outlines(plan_model_file: Path | str) -> tuple[list[list[list[float]]], set[int]]:
    """(plan outlines, scan instance ids) a tall kitchen row took the place of in the 3D model: the furniture items it replaced
    (they stand inside the row) and what a fridge was before it became a unit of the row.  The instance id finds the plan outline
    of an item whose footprint was completed before the row took it."""
    path = Path(plan_model_file)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = ((payload.get("source") or {}).get("depth_furniture") or {}).get("kitchen_rows") or []
    items = [item for row in rows for item in [*(row.get("replaced") or []), *(row.get("kept") or [])]]
    outlines = [[[float(x), float(y)] for x, y in item["footprint"]] for item in items if len(item.get("footprint") or []) >= 3]
    return outlines, {int(item["instance_id"]) for item in items if item.get("instance_id") is not None}


# A loose built-in line is the front of a drawn row unit when this share of it lies within _ROW_FRONT_REACH_M of the unit's outline
# (0.15 m: the detectors place the same face up to 13 cm apart, measured on the real scans).
_ROW_FRONT_SHARE = 0.50
_ROW_FRONT_REACH_M = 0.15


def _without_row_duplicates(
    layers: dict[str, list[dict[str, Any]]], unit_entities: list[dict[str, Any]], replaced: list[list[list[float]]], upm: float,
    instances: set[int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """The plan without what a drawn tall row makes double: the outlines (and labels) of the items it replaced, and loose
    built-in lines that only retrace its front."""
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union

    gone = {_key(outline) for outline in replaced}
    outlines = [Polygon(entity["points"]).exterior for entity in unit_entities if entity.get("_einb_kind") == "tall_unit"]
    near = unary_union(outlines).buffer(_ROW_FRONT_REACH_M * upm) if outlines else None

    def replaced_outline(name: str, entity: dict[str, Any]) -> bool:
        # (a tall unit the wall stage drew stays: it is the drawing of the row unit that took its place)
        points = entity.get("points") or []
        return (name in FURNITURE_LAYERS and entity.get("type") == "polyline" and bool(entity.get("closed")) and len(points) >= 3
                and entity.get("_einb_kind") != "tall_unit"
                and (_key(points) in gone or (entity.get("_instance_id") is not None and int(entity["_instance_id"]) in (instances or ()))))

    gone_shapes = [Polygon(entity["points"]).buffer(0) for name, entities in layers.items() for entity in entities if replaced_outline(name, entity)]
    removed = 0
    cleaned: dict[str, list[dict[str, Any]]] = {}
    for name, entities in layers.items():
        if name not in FURNITURE_LAYERS and name != "A-ANNO":
            cleaned[name] = entities
            continue
        kept = []
        for entity in entities:
            kind, points = entity.get("type"), entity.get("points") or []
            if replaced_outline(name, entity):
                removed += 1
                continue
            if kind == "text" and entity.get("point") and not entity.get("_tall_unit_label") and any(shape.covers(Point(entity["point"][:2])) for shape in gone_shapes):
                removed += 1
                continue
            if (near is not None and name == "A-EINB" and kind == "polyline" and not entity.get("closed") and len(points) >= 2
                    and entity.get("_einb_kind") == "segment"):
                line = LineString(points)
                if line.length > 0.0 and line.intersection(near).length >= _ROW_FRONT_SHARE * line.length:
                    removed += 1
                    continue
            kept.append(entity)
        cleaned[name] = kept
    return cleaned, removed


def _row_unit_entities(layers: dict[str, list[dict[str, Any]]], units: list[tuple[list[list[float]], str, float]]) -> list[dict[str, Any]]:
    """Plan entities (outline, cross, label) of the row units that no built-in outline of the plan shows yet."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from samcloud.ortho.vector.kitchen_layout import tall_unit_entities

    drawn = []
    for entity in layers.get("A-EINB", []):
        if entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points") or []) >= 3 and not entity.get("_run_legs"):
            shape = Polygon(entity["points"]).buffer(0)
            if not shape.is_empty:
                drawn.append(shape)
    covered = unary_union(drawn) if drawn else None
    entities: list[dict[str, Any]] = []
    for outline, label, upm in units:
        shape = Polygon(outline).buffer(0)
        if shape.is_empty or shape.geom_type != "Polygon" or shape.area <= 0.0:
            continue
        if covered is not None and shape.intersection(covered).area >= _ROW_UNIT_DRAWN_SHARE * shape.area:
            continue
        rectangle = shape.minimum_rotated_rectangle
        entities.extend(tall_unit_entities(rectangle if rectangle.geom_type == "Polygon" else shape, label, upm, _kitchen_row=True))
    return entities


def overhead_outlines(plan_model_file: Path | str) -> list[list[list[float]]]:
    """Plan outlines of the objects on the ceiling (``ceiling_objects`` of a plan_model.json)."""
    path = Path(plan_model_file)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [
        [[float(x), float(y)] for x, y in item["footprint"]]
        for item in payload.get("ceiling_objects") or [] if len(item.get("footprint") or []) >= 3
    ]


def apply_overhead_outlines(result: Any, cache_dir: Path | str) -> Any:
    """``result`` with the objects on the ceiling on their own dashed overhead layer (they are above the plan cut)."""
    outlines = overhead_outlines(Path(cache_dir) / "plan_model.json")
    if not outlines:
        return result
    entities = [{"type": "polyline", "points": outline, "closed": True, "holes": []} for outline in outlines]
    return replace(
        result,
        layers={**result.layers, CEILING_LAYER: entities},
        layer_colors={**result.layer_colors, CEILING_LAYER: CEILING_LAYER_COLOUR},
        layer_linetypes={**result.layer_linetypes, CEILING_LAYER: CEILING_LINETYPE},
        layer_styles={**result.layer_styles, CEILING_LAYER: {**result.layer_styles.get(CEILING_LAYER, {}), "lineweight": 18}},
    )


def apply_reconstructed_outlines(result: Any, cache_dir: Path | str) -> Any:
    """``result`` (DxfPlanResult) with its matching furniture polylines redrawn as reconstructed outlines and the items
    added from scan instances drawn as closed polylines."""
    result = apply_overhead_outlines(result, cache_dir)
    mapping = reconstructed_outlines(Path(cache_dir) / "plan_model.json")
    extra = added_items(Path(cache_dir) / "plan_model.json")
    rows = row_units(Path(cache_dir) / "plan_model.json") if "A-EINB" in result.layers else []
    if not mapping and not extra and not rows:
        return result
    layers: dict[str, list[dict[str, Any]]] = {}
    changed = 0
    for name, entities in result.layers.items():
        if name not in FURNITURE_LAYERS:
            layers[name] = entities
            continue
        redrawn = []
        for entity in entities:
            if entity.get("type") == "polyline" and entity.get("closed") and not entity.get("holes") and len(entity.get("points") or []) >= 3:
                outline = mapping.get(_key(entity["points"]))
                if outline is not None:
                    entity = {**entity, "points": outline}
                    changed += 1
            redrawn.append(entity)
        layers[name] = redrawn
    for layer, outline in extra:
        target = layer if layer in layers else ("A-FURN" if "A-FURN" in layers else None)
        if target is None:
            continue  # never invent a layer the plan result does not style
        layers[target] = [*layers[target], {"type": "polyline", "points": outline, "closed": True, "holes": []}]
        changed += 1
    # tall kitchen rows exist only in the 3D model: the 2D plan shows them like the tall units of the wall stage
    units = _row_unit_entities(layers, rows) if rows else []
    if rows:
        outlines, instances = row_replaced_outlines(Path(cache_dir) / "plan_model.json")
        layers, removed = _without_row_duplicates(layers, units, outlines, rows[0][2], instances)
        changed += removed
    if units:
        layers["A-EINB"] = [*layers["A-EINB"], *units]
        changed += len(units)
    return replace(result, layers=layers) if changed else result
