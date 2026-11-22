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
    if not mapping and not extra:
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
    return replace(result, layers=layers) if changed else result
