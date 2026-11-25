"""The fixed tools of the AI assistant: read / measure tools and one edit tool per edit operation.

The tools come from the project's domain (:mod:`samcloud.assistant.domains`) and its tool tier: :func:`tool_specs` lists
them for the model, :func:`dispatch` runs one call with the domain of the :class:`ToolContext` (indoor when it has none).
This module also holds the indoor tools (:func:`indoor_read_tools`, :func:`indoor_edit_tools`).

The model never gets code, shell or file access: it can only call the tools of :func:`tool_specs`.  :func:`dispatch`
runs one tool call against a :class:`ToolContext` and always returns a :class:`ToolResult` (compact JSON in metres),
never an exception: edit errors, a missing point cloud and bad arguments become error results the model can react to.

SAMcloud rules enforced here (not only in the prompt):

* edits only go through :class:`samcloud.assistant.edits.Proposal` (all or nothing, base plan never mutated);
* ``add_furniture`` with ``manual=true`` is refused (``assistant.edit.manual_not_allowed``) unless the session was
  started with ``allow_manual=True`` (the user explicitly answered "set it anyway");
* internal parameters (names starting with ``_``) are never accepted from the model;
* section images are only produced when images are allowed (``text_only`` off and an image-capable model).
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from samcloud.assistant.domains.base import TIER_ELEMENTS, ToolDef, normalize_tier
from samcloud.assistant.edits import EditError, EditOp, Proposal, op_schemas
from samcloud.assistant.pointcloud import CloudUnavailable
from samcloud.assistant.providers.base import ToolCall, ToolSpec
from samcloud.bim.plan_model import BuildingPlan

LOGGER = logging.getLogger(__name__)

MAX_LIST_ROWS = 60  # list_elements: rows returned before truncation
MAX_JSON_LIST = 40  # any other list in a tool result
MAX_RESULT_CHARS = 12000  # hard cap of one tool result (characters)
FLOAT_DIGITS = 3

READ_TOOL_NAMES: tuple[str, ...] = ("list_elements", "element_details", "points_in_box", "section", "fit_surface")
NO_EVIDENCE_KEY = "assistant.edit.no_evidence"
MANUAL_NOT_ALLOWED_KEY = "assistant.edit.manual_not_allowed"

_BOX_SCHEMA = {
    "type": "object",
    "description": "Box in plan metres: {x: [x0, x1], y: [y0, y1], z: [z0, z1]}; z is the height above the floor (optional).",
    "properties": {"x": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                   "y": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                   "z": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
    "required": ["x", "y"],
}

_READ_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "list_elements": (
        "List the plan elements compactly: id, type, label, main dimensions in metres, position in plan metres. "
        "Optional filters: kind (wall, opening, door, window, passage, furniture, room, ceiling_object), room (room id: only "
        "elements in that room), label (text contained in the label or id).",
        {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["wall", "opening", "door", "window", "passage", "furniture", "room", "ceiling_object"]},
            "room": {"type": "string", "description": "Room id, e.g. R001."},
            "label": {"type": "string", "description": "Text contained in the label or id."},
        }, "additionalProperties": False},
    ),
    "element_details": (
        "Everything about one element (metres, plan metres): geometry, host wall, neighbours, rooms, sources.",
        {"type": "object", "properties": {"id": {"type": "string", "description": "Plan id, e.g. W003, O002, F007, R001."}},
         "required": ["id"], "additionalProperties": False},
    ),
    "points_in_box": (
        "Measure the classified point cloud inside a box: point count, height histogram (0.1 m bins, floor-relative), "
        "SAM3 classes and instances with their bounding boxes. Use it before adding anything: no points = no evidence.",
        {"type": "object", "properties": {"box": _BOX_SCHEMA}, "required": ["box"], "additionalProperties": False},
    ),
    "section": (
        "Horizontal floor-plan slice of the point cloud at height z_m above the floor (+-0.05 m): occupied outlines in plan "
        "metres, point and class counts. Optional box limits the area.",
        {"type": "object", "properties": {
            "z_m": {"type": "number", "description": "Height above the floor in metres (e.g. 1.0 for walls, 0.75 for tables)."},
            "box": _BOX_SCHEMA,
            "cell_m": {"type": "number", "minimum": 0.02, "maximum": 0.5, "description": "Grid cell size in metres (default 0.05)."},
        }, "required": ["z_m"], "additionalProperties": False},
    ),
    "fit_surface": (
        "Fit a surface in the point cloud inside a box: kind wall (face offset, normal; thickness and centreline when both "
        "faces are visible), table (top height and size), floor or ceiling (height). Measured values, metres.",
        {"type": "object", "properties": {
            "box": _BOX_SCHEMA,
            "kind": {"type": "string", "enum": ["wall", "table", "floor", "ceiling"]},
        }, "required": ["box", "kind"], "additionalProperties": False},
    ),
}

# extra usage notes appended to the edit tool descriptions of package B (units, sign conventions, defaults)
_EDIT_NOTES: dict[str, str] = {
    "move_wall": "Units: metres. offset_m > 0 moves the wall towards its face_a side (element_details tells which side "
                 "that is), offset_m < 0 towards face_b; or give dx_m / dy_m in plan axes instead. chain (default true) also "
                 "moves the collinear pieces of the same straight wall. Hosted openings, joining walls and room edges are "
                 "recomputed by SAMcloud.",
    "set_wall_thickness": "Units: metres (0.03-1.0). anchor: centre (default), face_a or face_b stays in place. chain "
                          "(default true) changes the collinear pieces too. Rooms and joins follow automatically.",
    "delete_wall": "Deletes the wall and the openings it hosts. Needs the user's confirmation before it is accepted.",
    "set_opening": "Units: metres. width_m 0.3-6.0; sill_m windows only; head_m above the floor; shift_m moves the opening "
                   "along its host wall (sign = wall direction). The opening must stay inside its wall.",
    "delete_opening": "Deletes a door, window or passage. Needs the user's confirmation before it is accepted.",
    "add_furniture": "box in plan metres around the object. SAMcloud measures the object from the point cloud (size, "
                     "height, IFC class, DXF layer); without evidence the call fails with no_evidence: then ask the user. "
                     "manual=true is only allowed after the user explicitly said to place it anyway.",
    "move_furniture": "dx_m / dy_m in plan metres.",
    "rotate_furniture": "angle_deg counter-clockwise (positive) about the footprint centroid.",
    "delete_furniture": "Needs the user's confirmation before it is accepted.",
    "set_furniture_class": "kind from the enum; label, IFC class and layer follow automatically.",
    "rename_room": "New room name (1-60 characters).",
}


@dataclass
class ToolResult:
    """Outcome of one tool call: JSON text for the model, optional PNG images, error flag and i18n key."""

    content: str
    images: list[bytes] = field(default_factory=list)
    error: bool = False
    error_key: str | None = None
    error_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    """What the tools of one turn act on.

    ``plan`` is the base plan of the turn; the read tools look at ``proposal.plan`` (the base with the edits of this turn).
    The point cloud is loaded lazily on first use by ``cloud_loader`` (or taken from ``cloud``); when it is unavailable
    the tools that need it return an error result.  ``domain`` (None = indoor) provides the tools, ``tier`` selects them.
    """

    plan: BuildingPlan
    proposal: Proposal
    settings: Any = None
    allow_manual: bool = False
    images_allowed: bool = False
    cloud: Any | None = None
    cloud_loader: Callable[[], Any] | None = None
    project: Path | None = None
    level_id: str | None = None
    domain: Any | None = None  # AssistantDomain of the turn (None = the indoor plan tools)
    tier: str = TIER_ELEMENTS  # tool tier of the turn ("elements" / "full")
    last_error_key: str | None = None  # key of the last failing tool call
    last_edit_key: str | None = None  # key of the last failing edit call (None after a successful edit)
    _cloud_error: str | None = None

    @classmethod
    def for_project(cls, project: Path | str, level_id: str, plan: BuildingPlan, *, settings: Any = None,
                    allow_manual: bool = False, images_allowed: bool = False, domain: Any = None,
                    tier: str = TIER_ELEMENTS) -> "ToolContext":
        """Context with a lazy :class:`~samcloud.assistant.pointcloud.ProjectCloud` loader for one project level."""
        project_path = Path(project)

        def loader() -> Any:
            from samcloud.assistant.pointcloud import ProjectCloud

            return ProjectCloud.load(project_path, str(level_id), plan=plan)

        return cls(plan=plan, proposal=Proposal(plan), settings=settings, allow_manual=allow_manual,
                   images_allowed=images_allowed, cloud_loader=loader, project=project_path, level_id=str(level_id),
                   domain=domain, tier=normalize_tier(tier))

    def get_cloud(self) -> Any:
        """The point cloud (loaded once); raises :class:`CloudUnavailable` when there is none."""
        if self.cloud is not None:
            return self.cloud
        if self._cloud_error is not None:
            raise CloudUnavailable(self._cloud_error)
        if self.cloud_loader is None:
            self._cloud_error = "no project point cloud"
            raise CloudUnavailable(self._cloud_error)
        try:
            self.cloud = self.cloud_loader()
        except CloudUnavailable as error:
            self._cloud_error = str(error) or "no project point cloud"
            raise
        except Exception as error:  # noqa: BLE001 - an unreadable cloud is "unavailable" for the tools
            LOGGER.warning("assistant: point cloud could not be loaded (%s)", type(error).__name__)
            self._cloud_error = f"point cloud could not be read ({type(error).__name__})"
            raise CloudUnavailable(self._cloud_error) from None
        return self.cloud

    def cloud_or_none(self) -> Any | None:
        try:
            return self.get_cloud()
        except CloudUnavailable:
            return None


# ------------------------------------------------------------------------------------------------------------- specs

def tool_specs(domain: Any = None, tier: str = TIER_ELEMENTS, images_allowed: bool = False) -> list[ToolSpec]:
    """All tools the model may call for ``domain`` (None = indoor) and ``tier``: read / measure tools first, then the
    edit tools (indoor: one per entry of ``edits.OPS``)."""
    specs: list[ToolSpec] = []
    for tool in tool_defs(domain, tier):
        description = tool.description
        if images_allowed and tool.image_note:
            description += tool.image_note
        specs.append(ToolSpec(tool.name, description, copy.deepcopy(tool.parameters)))
    return specs


def tool_defs(domain: Any = None, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
    """The :class:`~samcloud.assistant.domains.base.ToolDef` list of a domain and tier (None = the indoor tools)."""
    if domain is None:
        return [*indoor_read_tools(), *indoor_edit_tools()]
    return list(domain.tools(normalize_tier(tier)))


def tool_names(domain: Any = None, tier: str = TIER_ELEMENTS) -> list[str]:
    return [tool.name for tool in tool_defs(domain, tier)]


# ------------------------------------------------------------------------------------------------------------ helpers

def compact(value: Any, *, digits: int = FLOAT_DIGITS, max_list: int = MAX_JSON_LIST) -> Any:
    """JSON-safe copy with floats rounded and long lists truncated (a marker string tells how many were left out)."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    if isinstance(value, dict):
        return {str(k): compact(v, digits=digits, max_list=max_list) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [compact(v, digits=digits, max_list=max_list) for v in list(value)[:max_list]]
        if len(value) > max_list:
            items.append(f"... {len(value) - max_list} more")
        return items
    try:  # numpy scalars
        import numpy as np

        if isinstance(value, np.generic):
            return compact(value.item(), digits=digits, max_list=max_list)
        if isinstance(value, np.ndarray):
            return compact(value.tolist(), digits=digits, max_list=max_list)
    except Exception:  # noqa: BLE001 - numpy missing or exotic type
        pass
    if isinstance(value, Path):
        return value.name
    return str(value)


def to_json(value: Any) -> str:
    text = json.dumps(compact(value), ensure_ascii=False, separators=(",", ":"))
    if len(text) > MAX_RESULT_CHARS:
        text = json.dumps({"truncated": True, "preview": text[:MAX_RESULT_CHARS - 200]}, ensure_ascii=False)
    return text


def _error(message: str, key: str, params: dict[str, Any] | None = None, hint: str | None = None) -> ToolResult:
    payload: dict[str, Any] = {"error": message, "key": key}
    if params:
        payload["params"] = params
    if hint:
        payload["hint"] = hint
    return ToolResult(to_json(payload), [], True, key, dict(params or {}))


def short_args(arguments: Any, limit: int = 160) -> str:
    """Short text of tool arguments for logs and the chat history (never more than ``limit`` characters)."""
    try:
        text = json.dumps(compact(arguments, max_list=6), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(arguments)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _box_arg(arguments: dict[str, Any], name: str = "box", *, required: bool = True) -> dict[str, Any] | None:
    from samcloud.assistant.pointcloud import parse_box

    box = arguments.get(name)
    if box is None and not required:
        return None
    parse_box(box)  # raises ValueError with a readable reason
    return box


# --------------------------------------------------------------------------------------------------------- read tools

def _list_elements(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    from samcloud.assistant.selection import element_list

    spec = {key: arguments[key] for key in ("kind", "room", "label") if arguments.get(key) not in (None, "")}
    rows = element_list(ctx.proposal.plan, spec or None)
    payload: dict[str, Any] = {"total": len(rows), "elements": rows[:MAX_LIST_ROWS]}
    if len(rows) > MAX_LIST_ROWS:
        payload["truncated"] = f"{len(rows) - MAX_LIST_ROWS} more; use the kind / room / label filters"
    return ToolResult(json.dumps(compact(payload, max_list=MAX_LIST_ROWS + 1), ensure_ascii=False, separators=(",", ":")))


def _element_details(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    from samcloud.assistant.selection import element_details

    element_id = str(arguments.get("id") or "").strip()
    if not element_id:
        raise ValueError("id is missing")
    details = element_details(ctx.proposal.plan, element_id)
    if not details.get("found"):
        return _error(f"No element {element_id!r} in the plan.", "assistant.edit.not_found", {"id": element_id},
                      "Use list_elements to see the ids.")
    return ToolResult(to_json(details))


def _points_in_box(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    box = _box_arg(arguments)
    return ToolResult(to_json(ctx.get_cloud().points_in_box(box)))


def _section(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    if arguments.get("z_m") is None:
        raise ValueError("z_m is missing")
    z_m = float(arguments["z_m"])
    if not math.isfinite(z_m):
        raise ValueError("z_m must be finite")
    box = _box_arg(arguments, required=False)
    cell = float(arguments.get("cell_m") or 0.05)
    cell = min(max(cell, 0.02), 0.5)
    cloud = ctx.get_cloud()
    result = cloud.section(z_m, box, cell)
    images: list[bytes] = []
    if ctx.images_allowed:
        try:
            images.append(cloud.section_png(z_m, box, cell))
            result["image"] = "attached (dark = occupied, north up)"
        except Exception as error:  # noqa: BLE001 - the numbers are enough without the picture
            LOGGER.info("assistant: section image failed (%s)", type(error).__name__)
    return ToolResult(to_json(result), images)


def _fit_surface(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    box = _box_arg(arguments)
    kind = str(arguments.get("kind") or "").strip().lower()
    return ToolResult(to_json(ctx.get_cloud().fit_surface(box, kind)))


_READ_HANDLERS: dict[str, Callable[[ToolContext, dict[str, Any]], ToolResult]] = {
    "list_elements": _list_elements,
    "element_details": _element_details,
    "points_in_box": _points_in_box,
    "section": _section,
    "fit_surface": _fit_surface,
}
_IMAGE_NOTES = {"section": " The slice is also attached as an image."}


def _read_handler(name: str) -> Callable[[ToolContext, ToolCall, dict[str, Any]], ToolResult]:
    def handler(ctx: ToolContext, call: ToolCall, arguments: dict[str, Any]) -> ToolResult:
        return _READ_HANDLERS[name](ctx, arguments)

    handler.__name__ = f"read_{name}"
    return handler


def indoor_read_tools() -> list[ToolDef]:
    """The indoor read / measure tools (plan elements and the classified point cloud)."""
    tools = []
    for name in READ_TOOL_NAMES:
        description, schema = _READ_SPECS[name]
        tools.append(ToolDef(name, description, json.loads(json.dumps(schema)), _read_handler(name), "read",
                             _IMAGE_NOTES.get(name, "")))
    return tools


def indoor_edit_tools() -> list[ToolDef]:
    """One indoor edit tool per entry of ``edits.OPS`` (description with the usage notes of :data:`_EDIT_NOTES`)."""
    tools = []
    for name, entry in op_schemas().items():
        note = _EDIT_NOTES.get(name, "")
        description = f"{entry['description']} {note}".strip()
        tools.append(ToolDef(name, description, copy.deepcopy(entry["parameters"]), _edit, "edit"))
    return tools


# --------------------------------------------------------------------------------------------------------- edit tools

def _summary_entry(entry: dict[str, Any]) -> dict[str, Any]:
    result = {"element_id": entry["element_id"], "kind": entry["kind"], "change": entry["change"],
              "before": entry.get("before"), "after": entry.get("after")}
    if entry.get("notes"):
        result["notes"] = entry["notes"]
    return result


def _edit(ctx: ToolContext, call: ToolCall, arguments: dict[str, Any]) -> ToolResult:
    internal = [key for key in arguments if str(key).startswith("_")]
    if internal:
        return _error(f"Parameter {internal[0]!r} is not allowed.", "assistant.edit.bad_params", {"param": internal[0]})
    cloud = None
    if call.name == "add_furniture":
        if bool(arguments.get("manual")) and not ctx.allow_manual:
            return _error("Adding an object without point-cloud evidence (manual=true) needs the user's explicit consent.",
                          MANUAL_NOT_ALLOWED_KEY, {"kind": arguments.get("kind")},
                          "Do not retry. Ask the user whether to place it anyway without evidence.")
        cloud = ctx.cloud_or_none()
    op = EditOp.from_tool_call(call.name, arguments)
    changes = ctx.proposal.apply(op, cloud=cloud)
    applied = {change.element_id for change in changes}
    entries = [_summary_entry(entry) for entry in ctx.proposal.summary() if entry["element_id"] in applied]
    payload: dict[str, Any] = {"ok": True, "op": call.name, "changes": entries,
                               "proposal_elements": len(ctx.proposal.summary()),
                               "needs_confirmation": ctx.proposal.needs_confirmation()}
    notes = ctx.proposal.notes[-5:]
    if notes:
        payload["notes"] = notes
    return ToolResult(to_json(payload))


# ----------------------------------------------------------------------------------------------------------- dispatch

def dispatch(ctx: ToolContext, call: ToolCall) -> ToolResult:
    """Run one tool call; every failure becomes an error result (``error=True``), never an exception."""
    name = str(call.name or "")
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    try:
        tools = {tool.name: tool for tool in tool_defs(ctx.domain, ctx.tier)}
    except Exception as error:  # noqa: BLE001 - a broken domain must not crash the turn
        LOGGER.warning("assistant: tools of the domain unavailable (%s)", type(error).__name__)
        tools = {}
    tool = tools.get(name)
    is_edit = tool is not None and tool.is_edit
    try:
        if "_raw" in arguments or "_value" in arguments:
            result = _error("The tool arguments are not a JSON object.", "assistant.agent.bad_arguments", {"tool": name})
        elif tool is not None:
            result = tool.handler(ctx, call, dict(arguments))
        else:
            result = _error(f"Unknown tool {name!r}.", "assistant.agent.unknown_tool", {"tool": name},
                            "Only the listed tools exist.")
    except EditError as error:
        hint = None
        if error.key == NO_EVIDENCE_KEY:
            hint = ("There is no evidence in the point cloud. Tell the user and ask whether to place it anyway "
                    "(manual); do not invent it.")
        result = _error(error.message, error.key, dict(error.params), hint)
    except CloudUnavailable as error:
        result = _error(f"The project point cloud is not available ({error}).", "assistant.agent.cloud_unavailable", {},
                        "Work with the plan data only and say that nothing could be measured.")
    except (ValueError, TypeError, KeyError) as error:
        result = _error(f"Bad arguments for {name}: {str(error)[:200]}", "assistant.agent.bad_arguments", {"tool": name})
    except Exception as error:  # noqa: BLE001 - a tool must never crash the turn
        LOGGER.warning("assistant: tool %s failed (%s)", name, type(error).__name__)
        result = _error(f"The tool {name} failed ({type(error).__name__}).", "assistant.agent.tool_failed", {"tool": name})
    if result.error:
        ctx.last_error_key = result.error_key
        if is_edit:
            ctx.last_edit_key = result.error_key
    elif is_edit:
        ctx.last_edit_key = None
    if result.images and not ctx.images_allowed:
        result.images = []
    return result


__all__ = ["MANUAL_NOT_ALLOWED_KEY", "NO_EVIDENCE_KEY", "READ_TOOL_NAMES", "ToolContext", "ToolResult", "compact", "dispatch",
           "indoor_edit_tools", "indoor_read_tools", "short_args", "to_json", "tool_defs", "tool_names", "tool_specs"]
