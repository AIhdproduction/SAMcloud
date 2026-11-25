"""Outdoor domain: the environment-chain Scene of a drone project, edited with the operations of ``outdoor_edits``.

Registered on import (:func:`register`) for drone / environment-chain projects (the registry's
``uses_environment_chain``); it replaces the ``outdoor_pending`` placeholder.  Tiers: ``"elements"`` (trees, manholes,
fences, site walls, curbs) and ``"full"`` (additionally pavements, buildings, terrain and ``apply_to_category``).

The model sees LV95 (``e`` / ``n``) and LN02; selections, hit tests and outlines use the LOCAL metres of the Scene
(``view_frame()["origin_offset"]`` converts).  After an accept / undo the page runs the fast environment job
(:meth:`OutdoorDomain.export_actions`: steps ``scene,export,qa`` with ``cad_only``).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from samcloud.assistant.domains.base import TIER_ELEMENTS, TIER_FULL, AssistantDomain, HistoryBackend, ToolDef, normalize_tier

KIND = "outdoor"
REASON_NO_SCENE = "assistant.agent.outdoor_no_scene"
EXPORT_ACTION = "env:scene,export,qa:cad_only"
MAX_LIST_ROWS = 60
PROFILE_MAX_STATIONS = 60

_VOCABULARY: dict[str, dict[str, Any]] = {
    "de": {
        "project_type": "Aussenprojekt (Drohne)",
        "model_phrase": "das Umgebungsmodell (DXF/IFC: Bäume, Schächte, Zäune, Mauern, Randsteine, Beläge, Gebäude, Gelände)",
        "element_types": {"tree": "Baum", "manhole": "Schacht", "fence": "Zaun", "site_wall": "Gartenmauer", "curb": "Randstein",
                          "pavement": "Belag", "building": "Gebäude", "terrain": "Gelände"},
        "rules": [
            "Keine Evidenz = nichts. Bevor du einen Baum oder Schacht hinzufügst, schau mit points_in_box oder measure_tree in der "
            "Punktwolke nach.\n   Findest du dort keine passenden Punkte, sag es und frag nach. Setze manual=true nur, wenn der "
            "Benutzer das ausdrücklich bestätigt hat.",
            "Gemessen, nicht geschätzt. Masse kommen aus den Mess-Werkzeugen (measure_tree, points_in_box, terrain_height, profile) "
            "oder aus dem Satz des Benutzers.",
            "Ändere nur die Elemente, die der Benutzer meint (Auswahl im Kontext, sonst nachfragen). Nur die Bearbeitungs-Werkzeuge "
            "ändern etwas;\n   es gibt keinen Code, keine Shell und keinen Dateizugriff.",
            "Koordinaten sind LV95 (e = Ost, n = Nord, Meter), Höhen LN02 in Metern. Höhen von Bäumen, Zäunen und Mauern sind "
            "Meter über dem Gelände.",
            "Abhängigkeiten rechnet SAMcloud: Baumfüsse, Zaun- und Mauerunterkanten liegen bündig auf dem Gelände, Randsteine "
            "werden auf das Gelände\n   gelegt und die Rinne neu berechnet. Du musst das nicht nachführen.",
            "{tier}",
            "Schlägt ein Werkzeug fehl, lies den Fehler und korrigiere, oder erkläre kurz, warum es nicht geht.",
        ],
        "closing": {
            "elements": "Am Ende beschreibst du den Vorschlag in ein bis drei Sätzen (was vorher, was nachher) oder stellst deine "
                        "Rückfrage.\nFormuliere immer als Vorschlag, nie als erledigt: z. B. \"Vorschlag: Baum B1 von 8.2 m auf "
                        "10.2 m erhöhen. Nach\nÜbernehmen ist er 10.2 m hoch.\" Schreibe nie \"wurde geändert\", \"ist jetzt\" "
                        "oder \"habe geändert\".",
        },
    },
    "en": {
        "project_type": "outdoor project (drone)",
        "model_phrase": "the site model (DXF/IFC: trees, manholes, fences, walls, curbs, pavements, buildings, terrain)",
        "element_types": {"tree": "tree", "manhole": "manhole", "fence": "fence", "site_wall": "garden wall", "curb": "curb",
                          "pavement": "pavement", "building": "building", "terrain": "terrain"},
        "rules": [
            "No evidence = nothing. Before you add a tree or a manhole, look in the point cloud with points_in_box or measure_tree.\n"
            "   If there are no matching points, say so and ask. Use manual=true only after the user explicitly confirmed it.",
            "Measured, not estimated. Dimensions come from the measuring tools (measure_tree, points_in_box, terrain_height, "
            "profile) or from the user's sentence.",
            "Only change the elements the user means (the selection in the context, otherwise ask). Only the edit tools change "
            "anything;\n   there is no code, no shell and no file access.",
            "Coordinates are LV95 (e = east, n = north, metres), heights LN02 in metres. Heights of trees, fences and walls are "
            "metres above the terrain.",
            "Dependencies are computed by SAMcloud: tree bases, fence and wall bottoms sit flush on the terrain, curbs are placed "
            "on the terrain\n   and their gutter is recomputed. You do not need to update them.",
            "{tier}",
            "When a tool fails, read the error and correct the call, or briefly explain why it is not possible.",
        ],
        "closing": {
            "elements": "At the end describe the proposal in one to three sentences (before and after) or ask your question.\n"
                        "Always phrase it as a proposal, never as done, e.g. \"Proposal: raise tree B1 from 8.2 m to 10.2 m. "
                        "After you accept it,\nit will be 10.2 m high.\" Never write \"was changed\", \"is now\" or \"I changed\".",
        },
    },
}


def _r(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


# ------------------------------------------------------------------------------------------------------------ tools

_POINT_PROPS = {"e": {"type": "number", "description": "LV95 east in metres."}, "n": {"type": "number", "description": "LV95 north in metres."}}
_BOX = {"type": "object", "description": "LV95 box {e: [e0, e1], n: [n0, n1]} with optional z: [z0, z1] (LN02); at most 100 m.",
        "properties": {"e": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                       "n": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                       "z": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}, "required": ["e", "n"]}

_READ_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "list_elements": (
        "List the site elements compactly: id, type, label, main dimensions in metres, LV95 position. Optional: category "
        "(tree, manhole, fence, site_wall, curb, pavement, building, terrain, wall, shrub, marking, vegetation_area), label "
        "(text in id or name).",
        {"type": "object", "properties": {"category": {"type": "string"}, "label": {"type": "string"}}, "additionalProperties": False}),
    "element_details": (
        "Everything about one element: LV95 axis / position, LN02 heights, dimensions, origin of each value, evidence.",
        {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False}),
    "points_in_box": (
        "Measure the classified point cloud in an LV95 box: point count, classes (tree canopy, tree trunk, manhole cover, "
        "fence, garden wall, curb, road, ...) with count, height range and bounding box, a height histogram (LN02, 0.5 m "
        "bins) and the terrain height. Use it before adding anything: no points = no evidence.",
        {"type": "object", "properties": {"box": _BOX}, "required": ["box"], "additionalProperties": False}),
    "terrain_height": (
        "Terrain height (LN02) at an LV95 point (the site's terrain surface).",
        {"type": "object", "properties": dict(_POINT_PROPS), "required": ["e", "n"], "additionalProperties": False}),
    "profile": (
        "Height profile along an LV95 polyline: per station (every step_m, default 1 m, at most 60 stations) the terrain "
        "height and the highest / lowest cloud point within half_width_m (default 0.3 m) with its class.",
        {"type": "object", "properties": {
            "polyline": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, "minItems": 2},
            "step_m": {"type": "number"}, "half_width_m": {"type": "number"}}, "required": ["polyline"], "additionalProperties": False}),
    "measure_tree": (
        "Measure a tree in the point cloud around an LV95 point (radius_m, default 3 m): canopy / trunk points, trunk "
        "position, height above the terrain, crown diameter and shape, trunk diameter (only when the trunk fit succeeds).",
        {"type": "object", "properties": {**_POINT_PROPS, "radius_m": {"type": "number"}}, "required": ["e", "n"],
         "additionalProperties": False}),
}


def _to_json(value: Any) -> Any:
    from samcloud.assistant.tools import ToolResult, to_json

    return ToolResult(to_json(value))


def _scene(ctx: Any) -> Any:
    return ctx.proposal.plan


def _support(ctx: Any) -> Any:
    return ctx.proposal.support


def _local(scene: Any, e: float, n: float) -> tuple[float, float]:
    from samcloud.assistant.domains.outdoor_scene import to_local_xy

    return to_local_xy(scene, e, n)


def _num(arguments: dict[str, Any], key: str, default: float | None = None) -> float:
    value = arguments.get(key, default)
    if value is None or isinstance(value, bool):
        raise ValueError(f"{key} is missing")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _region_of_box(scene: Any, box: Any) -> Any:
    from samcloud.assistant.domains.outdoor_cloud import Region
    from samcloud.assistant.domains.outdoor_scene import to_local_z

    if not isinstance(box, dict):
        raise ValueError("box must be an object {e, n}")
    e0, e1 = sorted(float(v) for v in box["e"])
    n0, n1 = sorted(float(v) for v in box["n"])
    if not all(math.isfinite(v) for v in (e0, e1, n0, n1)) or e1 - e0 <= 0 or n1 - n0 <= 0:
        raise ValueError("box must have a positive finite size")
    if e1 - e0 > 100 or n1 - n0 > 100:
        raise ValueError("box must be at most 100 m wide")
    x0, y0 = _local(scene, e0, n0)
    x1, y1 = _local(scene, e1, n1)
    z = box.get("z")
    z_range = None
    if z is not None:
        z0, z1 = sorted(float(v) for v in z)
        z_range = (to_local_z(scene, z0), to_local_z(scene, z1))
    return Region(box=(x0, x1, y0, y1), z=z_range)


def _surface(ctx: Any) -> Callable[[np.ndarray], np.ndarray]:
    from samcloud.assistant.domains.outdoor_edits import surface_heights

    scene, support = _scene(ctx), _support(ctx)
    return lambda xy: surface_heights(scene, support, xy)


def _read_list(ctx: Any, arguments: dict[str, Any]) -> Any:
    from samcloud.assistant.domains.outdoor_selection import element_list

    spec = {key: arguments[key] for key in ("category", "label") if arguments.get(key) not in (None, "")}
    rows = element_list(_scene(ctx), spec or None)
    payload: dict[str, Any] = {"total": len(rows), "elements": rows[:MAX_LIST_ROWS]}
    if len(rows) > MAX_LIST_ROWS:
        payload["truncated"] = f"{len(rows) - MAX_LIST_ROWS} more; use the category / label filters"
    from samcloud.assistant.tools import ToolResult, compact

    return ToolResult(json.dumps(compact(payload, max_list=MAX_LIST_ROWS + 1), ensure_ascii=False, separators=(",", ":")))


def _read_details(ctx: Any, arguments: dict[str, Any]) -> Any:
    from samcloud.assistant.domains.outdoor_selection import element_details
    from samcloud.assistant.tools import _error

    element_id = str(arguments.get("id") or "").strip()
    if not element_id:
        raise ValueError("id is missing")
    details = element_details(_scene(ctx), element_id)
    if not details.get("found"):
        return _error(f"No element {element_id!r} in the site model.", "assistant.edit.not_found", {"id": element_id},
                      "Use list_elements to see the ids.")
    return _to_json(details)


def _lv95_rows(scene: Any, result: dict[str, Any]) -> dict[str, Any]:
    from samcloud.assistant.domains.outdoor_scene import origin_offset

    offset = origin_offset(scene)
    for row in result.get("classes", []):
        row["z_min"], row["z_max"] = _r(row["z_min"] + offset[2]), _r(row["z_max"] + offset[2])
        row["bbox"] = [[_r(x + offset[0]), _r(y + offset[1])] for x, y in row["bbox"]]
    histogram = result.get("z_histogram")
    if histogram:
        histogram["z0"] = _r(histogram["z0"] + offset[2])
    if result.get("terrain_z") is not None:
        result["terrain_ln02"] = _r(result.pop("terrain_z") + offset[2])
    return result


def _read_points(ctx: Any, arguments: dict[str, Any]) -> Any:
    scene = _scene(ctx)
    region = _region_of_box(scene, arguments.get("box"))
    return _to_json(_lv95_rows(scene, ctx.get_cloud().points_in_box(region, _surface(ctx))))


def _read_terrain(ctx: Any, arguments: dict[str, Any]) -> Any:
    scene = _scene(ctx)
    e, n = _num(arguments, "e"), _num(arguments, "n")
    x, y = _local(scene, e, n)
    z = float(_surface(ctx)(np.asarray([[x, y]]))[0])
    from samcloud.assistant.domains.outdoor_scene import origin_offset

    if not math.isfinite(z):
        return _to_json({"e": _r(e), "n": _r(n), "terrain_ln02": None, "note": "no terrain at this position"})
    return _to_json({"e": _r(e), "n": _r(n), "terrain_ln02": _r(z + origin_offset(scene)[2])})


def _read_profile(ctx: Any, arguments: dict[str, Any]) -> Any:
    from samcloud.assistant.domains.outdoor_scene import origin_offset

    scene = _scene(ctx)
    raw = arguments.get("polyline")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("polyline needs at least two [e, n] points")
    xy = np.asarray([_local(scene, float(p[0]), float(p[1])) for p in raw], dtype=np.float64)
    length = float(np.hypot(*np.diff(xy, axis=0).T).sum())
    if length <= 0 or length > 500:
        raise ValueError("polyline length must be 0-500 m")
    step = max(_num(arguments, "step_m", 1.0), length / PROFILE_MAX_STATIONS, 0.1)
    half = min(max(_num(arguments, "half_width_m", 0.3), 0.05), 2.0)
    offset = origin_offset(scene)
    try:
        cloud = ctx.get_cloud()
    except Exception:  # noqa: BLE001 - the terrain profile works without the cloud
        cloud = None
    if cloud is not None:
        rows = cloud.profile(xy, step, half, _surface(ctx))
    else:
        from samcloud.assistant.domains.outdoor_cloud import OutdoorCloud

        rows = OutdoorCloud(np.zeros((0, 3)), np.zeros(0)).profile(xy, step, half, _surface(ctx))
    for row in rows:
        row["e"], row["n"] = _r(row.pop("x") + offset[0]), _r(row.pop("y") + offset[1])
        for key in ("terrain_z", "top_z", "low_z"):
            if row.get(key) is not None:
                row[key.replace("_z", "_ln02")] = _r(row.pop(key) + offset[2])
            else:
                row.pop(key, None)
    return _to_json({"length_m": _r(length, 2), "step_m": _r(step, 2), "stations": rows})


def _read_measure_tree(ctx: Any, arguments: dict[str, Any]) -> Any:
    from samcloud.assistant.domains.outdoor_cloud import Region, measure_tree
    from samcloud.assistant.domains.outdoor_scene import origin_offset

    scene = _scene(ctx)
    x, y = _local(scene, _num(arguments, "e"), _num(arguments, "n"))
    radius = min(max(_num(arguments, "radius_m", 3.0), 0.2), 15.0)
    result = measure_tree(ctx.get_cloud(), Region(circle=(x, y, radius)), _surface(ctx), _support(ctx).config)
    offset = origin_offset(scene)
    if "trunk_xy" in result:
        tx, ty = result.pop("trunk_xy")
        result["trunk"] = [_r(tx + offset[0]), _r(ty + offset[1])]
    for key in ("base_z", "top_z", "crown_base_z"):
        if result.get(key) is not None:
            result[key.replace("_z", "_ln02")] = _r(result.pop(key) + offset[2])
    from samcloud.assistant.domains.outdoor_cloud import MIN_TREE_POINTS

    result["evidence"] = "enough" if result["points"] >= MIN_TREE_POINTS else "none"
    return _to_json(result)


_READ_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    "list_elements": _read_list, "element_details": _read_details, "points_in_box": _read_points,
    "terrain_height": _read_terrain, "profile": _read_profile, "measure_tree": _read_measure_tree,
}


def _read_handler(name: str) -> Callable[..., Any]:
    def handler(ctx: Any, call: Any, arguments: dict[str, Any]) -> Any:
        return _READ_HANDLERS[name](ctx, arguments)

    handler.__name__ = f"outdoor_read_{name}"
    return handler


def outdoor_read_tools() -> list[ToolDef]:
    return [ToolDef(name, description, json.loads(json.dumps(schema)), _read_handler(name), "read")
            for name, (description, schema) in _READ_SPECS.items()]


_ADD_OPS = ("add_tree", "add_manhole")


def _edit(ctx: Any, call: Any, arguments: dict[str, Any]) -> Any:
    from samcloud.assistant.edits import EditOp
    from samcloud.assistant.tools import MANUAL_NOT_ALLOWED_KEY, _error, _summary_entry, to_json, ToolResult

    internal = [key for key in arguments if str(key).startswith("_")]
    if internal:
        return _error(f"Parameter {internal[0]!r} is not allowed.", "assistant.edit.bad_params", {"param": internal[0]})
    cloud = None
    if call.name in _ADD_OPS:
        if bool(arguments.get("manual")) and not ctx.allow_manual:
            return _error("Adding an element without point-cloud evidence (manual=true) needs the user's explicit consent.",
                          MANUAL_NOT_ALLOWED_KEY, {"kind": call.name.removeprefix("add_")},
                          "Do not retry. Ask the user whether to place it anyway without evidence.")
        if not bool(arguments.get("manual")):
            cloud = ctx.cloud_or_none()  # None -> the op fails with cloud_required
    op = EditOp.from_tool_call(call.name, arguments)
    changes = ctx.proposal.apply(op, cloud=cloud)
    applied = {change.element_id for change in changes}
    entries = [_summary_entry(entry) for entry in ctx.proposal.summary() if entry["element_id"] in applied]
    payload: dict[str, Any] = {"ok": True, "op": call.name, "changes": entries[:40], "changed_elements": len(entries),
                               "proposal_elements": len(ctx.proposal.summary()), "needs_confirmation": ctx.proposal.needs_confirmation()}
    notes = ctx.proposal.notes[-5:]
    if notes:
        payload["notes"] = notes
    return ToolResult(to_json(payload))


def outdoor_edit_tools(tier: str = TIER_ELEMENTS) -> list[ToolDef]:
    from samcloud.assistant.domains.outdoor_edits import ops_for_tier

    tools = []
    for spec in ops_for_tier(normalize_tier(tier)):
        description = spec.description + (" (Full control.)" if spec.tier == TIER_FULL else "")
        tools.append(ToolDef(spec.name, description, spec.schema, _edit, "edit"))
    return tools


# ------------------------------------------------------------------------------------------------------------ domain

class OutdoorDomain(AssistantDomain):
    """The environment-chain Scene of a drone project (level ``site``)."""

    kind = KIND
    supported_tiers: tuple[str, ...] = (TIER_ELEMENTS, TIER_FULL)

    def __init__(self, project: Path | str, level_id: str, *, mode: str | None = None, scene: Any = None) -> None:
        super().__init__(project, level_id, mode=mode)
        self._scene = scene  # optional fixed base Scene (tests)
        self._support_obj: Any = None
        self._backend: HistoryBackend | None = None

    # ---------------------------------------------------------------------------------------------------- status
    def availability(self) -> tuple[bool, str | None]:
        from samcloud.assistant.domains.outdoor_scene import raw_model_path, scene_pickle_path

        if self._scene is not None or scene_pickle_path(self.project).is_file() or raw_model_path(self.project, self.level_id).is_file():
            return True, None
        return False, REASON_NO_SCENE

    # ----------------------------------------------------------------------------------------------------- model
    def load_base(self) -> Any:
        """The current outdoor model.  Read-only use writes nothing: without ``edits.json`` the pipeline Scene of
        ``scene.pkl`` is used from memory; ``products/assistant/site/scene.json`` is only kept (and refreshed) once the
        assistant has edits (the history writes it on the first accept)."""
        if self._scene is not None:
            return self._scene
        from samcloud.assistant.domains.outdoor_scene import (
            edits_file, load_model_cached, load_pipeline_scene, model_path, raw_model_path, scene_pickle_path,
        )

        if edits_file(self.project, self.level_id).is_file():
            path = model_path(self.project, self.level_id)
            if path.is_file():
                return load_model_cached(path)
        if scene_pickle_path(self.project).is_file():
            return load_pipeline_scene(self.project)
        raw = raw_model_path(self.project, self.level_id)
        if raw.is_file():
            return load_model_cached(raw)
        return load_pipeline_scene(self.project)

    def _model(self, model: Any) -> Any:
        return self.current_model() if model is None else model

    def support(self) -> Any:
        if self._support_obj is None:
            from samcloud.assistant.domains.outdoor_scene import EnvSupport

            self._support_obj = EnvSupport.for_project(self.project)
        return self._support_obj

    def new_proposal(self, base: Any) -> Any:
        from samcloud.assistant.domains.outdoor_edits import OutdoorProposal

        return OutdoorProposal(base, support=self.support(), project=self.project)

    def view_frame(self) -> dict[str, Any]:
        scene = self.current_model()
        crs = scene.settings.crs
        return {"units_per_metre": 1.0, "floor_z": None, "origin_offset": [float(v) for v in scene.settings.origin_offset],
                "crs": {"epsg": None if crs is None else crs.epsg, "vertical_datum": None if crs is None else crs.vertical_datum}}

    # ------------------------------------------------------------------------------------------------- elements
    def element_list(self, filter: Any = None, model: Any = None) -> list[dict[str, Any]]:  # noqa: A002
        from samcloud.assistant.domains.outdoor_selection import element_list

        return element_list(self._model(model), filter)

    def element_details(self, element_id: str, model: Any = None) -> dict[str, Any]:
        from samcloud.assistant.domains.outdoor_selection import element_details

        return element_details(self._model(model), element_id)

    # ------------------------------------------------------------------------------------------------ selection
    def hit_test(self, x: float, y: float, model: Any = None) -> dict[str, Any]:
        from samcloud.assistant.domains.outdoor_selection import hit_test

        return hit_test(self._model(model), float(x), float(y))

    def selection_from_global_id(self, global_id: str, point: list[float] | None = None, model: Any = None) -> dict[str, Any]:
        from samcloud.assistant.domains.outdoor_selection import global_id_map, selection_for

        scene = self._model(model)
        element_id = global_id_map(scene).get(str(global_id or "")) if global_id else None
        if element_id is not None:
            return selection_for(scene, element_id)
        if point is not None and len(point) >= 2:
            return {"type": "point", "x": _r(point[0]), "y": _r(point[1])}
        return {"type": "unknown", "global_id": str(global_id or "")}

    def chip_label(self, sel: dict[str, Any], lang: str | None = None) -> str:
        from samcloud.assistant.domains.outdoor_selection import chip_label

        scene = None
        if isinstance(sel, dict) and sel.get("type") == "point":
            try:
                scene = self.current_model()
            except Exception:  # noqa: BLE001 - a label without LV95 is still a label
                scene = None
        return chip_label(sel or {}, lang, scene)

    def outline(self, sel: dict[str, Any], model: Any = None) -> list[list[list[float]]]:
        from samcloud.assistant.domains.outdoor_selection import outline

        if not isinstance(sel, dict) or sel.get("type") != "element":
            return []
        return outline(self._model(model), sel)

    def z_range(self, sel: dict[str, Any], model: Any = None) -> tuple[float, float] | None:
        from samcloud.assistant.domains.outdoor_selection import z_range

        if not isinstance(sel, dict) or sel.get("type") != "element":
            return None
        return z_range(self._model(model), sel)

    def global_ids_for(self, selections: list[dict[str, Any]], model: Any = None) -> list[str]:
        from samcloud.assistant.domains.outdoor_selection import global_id_of

        scene = None
        ids: list[str] = []
        for sel in selections or []:
            if not isinstance(sel, dict) or sel.get("type") != "element" or not sel.get("id"):
                continue
            scene = scene if scene is not None else self._model(model)
            if str(sel["id"]) not in scene.elements:
                continue
            gid = global_id_of(str(sel["id"]))
            if gid not in ids:
                ids.append(gid)
        return ids

    def refresh_selection(self, sel: dict[str, Any], model: Any = None) -> dict[str, Any] | None:
        from samcloud.assistant.domains.outdoor_selection import selection_for

        if not isinstance(sel, dict):
            return None
        if sel.get("type") == "point":
            return dict(sel)
        if sel.get("type") == "element" and sel.get("id"):
            updated = selection_for(self._model(model), str(sel["id"]))
            return updated if updated.get("type") == "element" else None
        return None

    # ----------------------------------------------------------------------------------------------------- tools
    def read_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        return outdoor_read_tools()

    def edit_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        return outdoor_edit_tools(tier)

    def cloud_loader(self, base: Any) -> Callable[[], Any] | None:
        project = self.project

        def loader() -> Any:
            from samcloud.assistant.domains.outdoor_cloud import OutdoorCloud

            return OutdoorCloud.load(project)

        return loader

    # -------------------------------------------------------------------------------------------------------- QA
    def run_qa(self, proposal: Any, baseline: Any, *, work_dir: Path) -> Any:
        from samcloud.assistant.domains.outdoor_qa import run_outdoor_qa

        return run_outdoor_qa(self.project, proposal.plan, baseline if baseline is not None else proposal.base, work_dir=work_dir)

    # --------------------------------------------------------------------------------------------------- prompts
    @staticmethod
    def vocabulary_for(lang: str) -> dict[str, Any]:
        """The outdoor prompt vocabulary (element names from ``assistant.outdoor.element.*`` when translated)."""
        code = str(lang or "").lower()[:2]
        code = code if code in _VOCABULARY else "en"
        vocabulary = copy.deepcopy(_VOCABULARY[code])
        try:
            from samcloud.app.i18n import catalogue

            texts = catalogue(code)
            for kind in list(vocabulary["element_types"]):
                name = texts.get(f"assistant.outdoor.element.{kind}")
                if name:
                    vocabulary["element_types"][kind] = name
            project_type = texts.get("assistant.outdoor.project_type")
            if project_type:
                vocabulary["project_type"] = project_type
        except Exception:  # noqa: BLE001 - the built-in names are enough
            pass
        return vocabulary

    def vocabulary(self, lang: str) -> dict[str, Any]:
        return self.vocabulary_for(lang)

    def context_message(self, model: Any, selection: list[dict[str, Any]] | None, lang: str | None = None, *,
                        max_elements: int = 80) -> str:
        from samcloud.assistant.domains.outdoor_scene import origin_offset

        scene = self._model(model)
        rows = self.element_list(model=scene)
        cap = max(0, int(max_elements))
        offset = origin_offset(scene)
        details = []
        for sel in list(selection or []):
            if not isinstance(sel, dict):
                continue
            item: dict[str, Any] = {"label": self.chip_label(sel, lang)}
            if sel.get("type") == "element" and sel.get("id"):
                item["id"] = sel["id"]
                if sel.get("ref"):
                    item["ref"] = sel["ref"]
            elif sel.get("type") == "point":
                item["e"], item["n"] = _r(float(sel.get("x", 0.0)) + offset[0]), _r(float(sel.get("y", 0.0)) + offset[1])
            details.append(item)
        payload: dict[str, Any] = {"project_type": self.vocabulary(str(lang or "en")).get("project_type"),
                                   "coordinates": "LV95 e/n (m), heights LN02 (m)",
                                   "refs": "ref = short element label (e.g. B9 tree, S3 manhole, Z2 fence, M1 garden wall, R4 curb); "
                                           "use it when talking to the user, tools accept ref or id as target",
                                   "selection": details, "elements": rows[:cap]}
        if len(rows) > cap:
            payload["more"] = len(rows) - cap
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)

    # ------------------------------------------------------------------------------------------- history / export
    def history_backend(self) -> HistoryBackend:
        if self._backend is None:
            from samcloud.assistant.domains.outdoor_pipeline import outdoor_backend

            self._backend = outdoor_backend(self.project)
        return self._backend

    def export_actions(self, after: str = "accept") -> list[str]:
        return [EXPORT_ACTION]

    def preview(self, proposal: Any) -> dict[str, list[dict[str, Any]]]:
        from samcloud.assistant.domains.outdoor_selection import element_rings, element_z_range, global_id_of

        preview: dict[str, list[dict[str, Any]]] = {"added": [], "removed": [], "moved": []}
        if proposal is None:
            return preview
        for entry in proposal.summary():
            element_id, kind, change = entry["element_id"], entry["kind"], entry["change"]
            before = proposal.base.elements.get(element_id)
            after = proposal.plan.elements.get(element_id)
            item: dict[str, Any] = {"id": element_id, "kind": kind, "change": change, "global_id": global_id_of(element_id),
                                    "before": element_rings(before) if before is not None else None,
                                    "after": element_rings(after) if after is not None else None}
            source = before if change == "removed" else after
            heights = None
            if source is not None:
                try:
                    heights = element_z_range(proposal.base if change == "removed" else proposal.plan, source)
                except Exception:  # noqa: BLE001 - informative only
                    heights = None
            item["z_min"], item["z_max"] = heights if heights is not None else (None, None)
            bucket = "added" if change == "added" else "removed" if change == "removed" else "moved"
            preview[bucket].append(item)
        return preview


# ---------------------------------------------------------------------------------------------------- registration

def matches(project: Path, level_id: str, mode: str | None) -> bool:
    """Registry matcher: a drone / environment-chain project (outdoor level)."""
    from samcloud.assistant.domains.registry import uses_environment_chain

    try:
        return uses_environment_chain(project, level_id, mode)
    except Exception:  # noqa: BLE001 - an unreadable project is no drone project
        return False


def factory(project: Path, level_id: str, mode: str | None) -> OutdoorDomain:
    return OutdoorDomain(project, level_id, mode=mode)


def register() -> None:
    """Replace the ``outdoor_pending`` placeholder of the registry (called by the registry on import)."""
    from samcloud.assistant.domains.registry import PRIORITY_OUTDOOR, register_domain

    register_domain(KIND, factory, matches, priority=PRIORITY_OUTDOOR)


__all__ = ["EXPORT_ACTION", "OutdoorDomain", "REASON_NO_SCENE", "factory", "matches", "outdoor_edit_tools", "outdoor_read_tools",
           "register"]
