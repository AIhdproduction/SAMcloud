"""AI assistant panel of the /dxf page: shared selection, chat, coloured proposal preview, accept / discard, change list.

Only built by the page while :meth:`samcloud.assistant.service.AssistantService.is_available` is true; without it the
page and both viewers stay exactly as before (no panel, no extra props, no handlers).

One :class:`AssistantPanel` per page and level holds the state (selection, draft, last answer); it is rendered once,
as the one chat card between the 2D result plan and the IFC viewer (:meth:`AssistantPanel.render`).  Both viewers
show the same selection and preview.  After accept / undo the page updates the DXF and the IFC by itself; the card
shows that progress, the result and a retry button when an update failed (:meth:`AssistantPanel._update_box`).
With full control (tool tier ``"full"``) the service applies a change right after its passed QA: the card shows one
"Applied -- Undo" line instead of the accept buttons and starts the same automatic update as after an accept.

A question runs as an app-level task (:class:`_Turn`, one per project level), not bound to the card or the browser
connection that asked it: when the page or the panel is rebuilt while the model works (page refresh, reconnect, a second
tab), the finished turn is handed to every live panel of that level (:func:`_deliver_turn`), and a panel built during a
running turn shows the spinner and picks the result up itself (timer).  Everything shown comes from the service state
(``pending()``, ``chat()``), so a rebuilt panel always shows the pending proposal and the chat.

Coordinates: the 2D canvas works in plan-local units (``units_per_metre`` per metre), the service in plan metres and
the GLB of the IFC viewer in its model frame (IFC metres, Z up -> glTF Y up: ``(x, y, z) -> (x, z, -y)``, minus
``origin_offset_yup`` from ``props.json``; see ``samcloud.bim.viewer_export``).

Outdoor (drone) projects: the domain works in the LOCAL metres of its Scene and says where that frame sits with
``view_frame()["origin_offset"]`` (LV95 E / N and LN02 height of the local origin).  The 2D view is the SVG preview of
the published DXF (:meth:`AssistantPanel.attach_preview`): its picks arrive as LV95 E / N and its overlays are sent as
LV95 rings (local + offset).  The outdoor IFC is written in the local frame, so the IFC pick point maps to local metres
exactly like the indoor plan metres; only the absolute heights of the domain (LN02) lose the origin height for the
preview boxes.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import re
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nicegui import background_tasks, run, ui

from samcloud.app.i18n import t, translate_or

LOGGER = logging.getLogger(__name__)

CHANGE_COLORS = {"added": "#4bc692", "removed": "#ef6b73", "moved": "#ffb74d", "modified": "#ffb74d"}
SELECT_COLOR = "#26b7ef"  # selection outline in the DXF preview (the accent colour)
MAX_SELECTION = 20
UNDO_CONFIRM_COUNT = 5  # undo of an edit that touched more elements (or deleted some) asks first
CHAT_LIMIT = 40
_UPDATE_KEYS = {"export": "assistant.panel.update_dxf", "export_ifc": "assistant.panel.update_ifc",
                "run_all": "assistant.panel.update_all"}
"""Job action of the automatic update after accept / undo -> its progress text in the chat card."""
ENV_ACTION_PREFIX = "env:"  # outdoor: the fast environment run (``env:<steps>[:cad_only]``) updates DXF and IFC together
DETAIL_LIMIT = 4
_EMPTY_PREVIEW: dict[str, list[dict[str, Any]]] = {"added": [], "removed": [], "moved": []}
_FIELD_ORDER = ("width_m", "sill_m", "head_m", "thickness_m", "wall_thickness_m", "length_m", "height_m", "size_m", "center",
                "centreline", "angle_deg", "axis_deg", "name", "label", "semantic_label", "area_m2", "wall_kind", "host_wall_id",
                "layer", "label_point")
_ADDED_FIELDS = ("width_m", "size_m", "length_m", "height_m", "center")
_OUTDOOR_KINDS = frozenset({"tree", "manhole", "fence", "site_wall", "curb", "pavement", "building", "terrain"})
_OUTDOOR_FIELDS = ("height", "crown_diameter", "trunk_diameter", "crown_shape", "diameter", "cover_ln02", "thickness", "width",
                   "gutter_width", "length", "base_ln02", "top_ln02", "type", "pavement_kind", "trunk", "center", "start", "end",
                   "area_m2")
"""Snapshot fields of outdoor elements (``outdoor_edits.snapshot``) shown as 'Height 5.80 → 7.80 m'."""
_OUTDOOR_METRES = frozenset({"height", "crown_diameter", "trunk_diameter", "diameter", "cover_ln02", "thickness", "width",
                             "gutter_width", "length", "base_ln02", "top_ln02"})
_OUTDOOR_ADDED = ("height", "crown_diameter", "diameter", "length", "thickness", "trunk", "center")
_CLAIMED_CHANGE = re.compile(
    r"\b(geändert|umbenannt|gelöscht|entfernt|verschoben|hinzugefügt|gesetzt|angepasst|verbreitert|verschmälert|"
    r"changed|renamed|deleted|removed|moved|added|updated|widened|narrowed)\b", re.IGNORECASE)

_STYLE = """
<style>
.sam-ai-card { border-left:3px solid var(--sam-accent)!important; }
.sam-ai-section { font-size:11px; text-transform:uppercase; letter-spacing:.7px; color:var(--sam-muted); font-weight:600; }
.sam-ai-chat { max-height:300px; overflow-y:auto; display:flex; flex-direction:column; gap:6px; padding:2px 4px 2px 0; }
.sam-ai-msg { padding:6px 10px; border-radius:7px; font-size:13px; line-height:1.45; white-space:pre-wrap; overflow-wrap:anywhere; max-width:92%; }
.sam-ai-msg.is-user { align-self:flex-end; background:rgba(38,183,239,.13); border:1px solid #24719b; }
.sam-ai-msg.is-assistant { align-self:flex-start; background:#0b1722; border:1px solid var(--sam-line); }
.sam-ai-who { display:block; font-size:10px; letter-spacing:.5px; text-transform:uppercase; color:var(--sam-muted); margin-bottom:2px; }
.sam-ai-box { border:1px solid var(--sam-line); border-radius:8px; padding:10px 12px; background:#0b1722; }
.sam-ai-change { display:flex; gap:8px; align-items:baseline; font-size:13px; padding:4px 0; border-bottom:1px solid var(--sam-line-soft); min-width:0; }
.sam-ai-change:last-child { border-bottom:0; }
.sam-ai-badge { font-size:11px; padding:1px 8px; border-radius:10px; border:1px solid; white-space:nowrap; flex:none; }
.sam-ai-badge.is-added, .sam-ai-badge.is-active { color:#4bc692; border-color:#2f7d5c; background:rgba(75,198,146,.10); }
.sam-ai-badge.is-removed { color:#ef6b73; border-color:#9b3c43; background:rgba(239,107,115,.10); }
.sam-ai-badge.is-moved, .sam-ai-badge.is-modified, .sam-ai-badge.is-stale { color:#ffb74d; border-color:#a8752b; background:rgba(255,183,77,.10); }
.sam-ai-badge.is-undone { color:var(--sam-muted); border-color:#456176; }
.sam-ai-dot { width:10px; height:10px; border-radius:2px; display:inline-block; flex:none; }
.sam-ai-detail { color:var(--sam-muted); font-size:12px; overflow-wrap:anywhere; min-width:0; }
.sam-ai-history { max-height:420px; overflow-y:auto; padding-right:4px; }
.sam-ai-edit { border-bottom:1px solid var(--sam-line-soft); padding:6px 0; }
.sam-ai-edit:last-child { border-bottom:0; }
.sam-ai-chips .q-chip { margin:0; }
</style>
"""


# ------------------------------------------------------------------------------------------------- coordinate helpers

def plan_to_model(x: float, y: float, z: float, offset: list[float]) -> list[float]:
    """Plan metres (x, y) + absolute height z -> GLB model frame (Y up, origin offset removed)."""
    return [float(x) - offset[0], float(z) - offset[1], -float(y) - offset[2]]


def model_to_plan(point: list[float], offset: list[float]) -> list[float]:
    """GLB model frame point -> plan metres [x, y] (inverse of :func:`plan_to_model` without the height)."""
    return [round(float(point[0]) + offset[0], 4), round(-(float(point[2]) + offset[2]), 4)]


def rings_of(value: Any) -> list[list[list[float]]]:
    """One ring (``[[x, y], ...]``, indoor) or several rings (``[[[x, y], ...], ...]``, outdoor elements) -> a list of
    rings; empty for None / empty values."""
    if not value:
        return []
    try:
        first = value[0]
        nested = bool(first) and isinstance(first[0], (list, tuple))
    except (TypeError, IndexError, KeyError):
        return []
    return [ring for ring in value if ring] if nested else [value]


def shift_ring(ring: list[list[float]], offset: list[float]) -> list[list[float]]:
    """Ring in local metres -> map coordinates (``+ offset``; outdoor: LV95 E / N)."""
    return [[round(float(point[0]) + float(offset[0]), 4), round(float(point[1]) + float(offset[1]), 4)] for point in ring]


def model_box(ring: list[list[float]], z_min: float, z_max: float, offset: list[float], color: str) -> dict[str, Any]:
    """Axis-aligned box of a plan ring (metres; or several rings, see :func:`rings_of`) between two heights, in the GLB
    model frame."""
    points = [point for item in rings_of(ring) for point in item]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    low = plan_to_model(min(xs), max(ys), float(z_min), offset)
    high = plan_to_model(max(xs), min(ys), float(z_max), offset)
    return {"min": [min(a, b) for a, b in zip(low, high)], "max": [max(a, b) for a, b in zip(low, high)], "color": color}


def _scale_ring(ring: list[list[float]] | None, factor: float) -> list[list[float]] | None:
    if not ring:
        return None
    return [[round(float(x) * factor, 4), round(float(y) * factor, 4)] for x, y, *_ in ring]


def _origin_offset(props_data: dict[str, Any] | None) -> list[float]:
    raw = (props_data or {}).get("origin_offset_yup")
    try:
        values = [float(value) for value in raw]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0]
    return values if len(values) == 3 else [0.0, 0.0, 0.0]


# ------------------------------------------------------------------------------------------------- text helpers

def _group(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def _money(cost: float) -> str:
    if cost >= 1.0:
        return f"{cost:.2f}"
    if cost >= 0.001 or cost == 0.0:
        return f"{cost:.3f}"
    return f"{cost:.4f}"


def usage_text(result: Any) -> str | None:
    """'1 234 in / 210 out · 16 414 / 32 768 tokens context · ≈ $0.004' (Ollama: 'local' instead of a price); None
    without usage.  The context part (largest request of the turn / context window) only when both are known."""
    usage = getattr(result, "usage", None)
    tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
    cost = getattr(result, "cost_usd", None)
    try:
        prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
        window = int(getattr(result, "context_window", 0) or 0)
    except (TypeError, ValueError):
        prompt_tokens = window = 0
    has_context = prompt_tokens > 0 and window > 0
    if not tokens_in and not tokens_out and cost is None and not has_context:
        return None
    parts = [t("assistant.panel.usage", input=_group(tokens_in), output=_group(tokens_out))]
    if has_context:
        parts.append(t("assistant.panel.context_use", used=_group(prompt_tokens), window=_group(window)))
    if str(getattr(result, "provider", "")) == "ollama":
        parts.append(t("assistant.panel.local"))
    elif cost is not None:
        parts.append(t("assistant.panel.cost", cost=_money(float(cost))))
    else:
        parts.append(t("assistant.panel.cost_unknown"))
    return " · ".join(parts)


def error_title(kind: str | None) -> str:
    """Short translated title of an ``error_kind`` (generic title for unknown kinds)."""
    return translate_or(f"assistant.panel.error_kind.{kind}", t("assistant.panel.error_title")) if kind else t("assistant.panel.error_title")


def _format_value(name: str, value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        unit = " m²" if name.endswith("_m2") else " m" if name.endswith("_m") else "°" if name.endswith("_deg") else ""
        return f"{float(value):.2f}{unit}"
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (list, tuple)) for item in value):
            return " – ".join(_format_value("", item) for item in value)
        if all(isinstance(item, (int, float)) for item in value):
            text = " / ".join(f"{float(item):.2f}" for item in value)
            return f"({text})" if name in ("", "center", "label_point") else f"{text} m" if name.endswith("_m") else text
    return str(value)


def element_ref(entry: dict[str, Any]) -> str:
    """The name of a changed element in the UI: its short display label (outdoor ``ref``, e.g. 'B9') or its id."""
    for source in (entry, entry.get("after") or {}, entry.get("before") or {}):
        ref = source.get("ref") if isinstance(source, dict) else None
        if ref:
            return str(ref)
    return str(entry.get("element_id", ""))


def change_title(entry: dict[str, Any]) -> str:
    """'Door O001' / 'Furniture F001 (Tisch)' / 'Tree B9' (outdoor: the short label) for a summary entry."""
    kind = str(entry.get("kind") or "")
    snapshot = entry.get("after") or entry.get("before") or {}
    sub = str(snapshot.get("type") or kind) if kind == "opening" else kind
    label = translate_or(f"assistant.panel.kind.{sub}", sub)
    name = snapshot.get("label") or snapshot.get("name") if kind in ("furniture", "room", "ceiling_object") else None
    text = f"{label} {element_ref(entry)}".strip()
    return f"{text} ({name})" if name else text


def _outdoor_value(name: str, value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _format_value("", value) if isinstance(value, (list, tuple)) else str(value)
    return f"{float(value):.2f}"


def _outdoor_details(before: dict[str, Any], after: dict[str, Any], change: str) -> list[str]:
    """'Height 5.80 → 7.80 m' lines of an outdoor element (unit once, at the end)."""
    lines = []
    names = _OUTDOOR_ADDED if change == "added" else _OUTDOOR_FIELDS
    for name in names:
        label = translate_or(f"assistant.panel.field.{name}", name)
        unit = " m" if name in _OUTDOOR_METRES else (" m²" if name.endswith("_m2") else "")
        if change == "added":
            if after.get(name) is not None:
                lines.append(f"{label} {_outdoor_value(name, after[name])}{unit}")
        elif before.get(name) != after.get(name) and (name in before or name in after):
            lines.append(f"{label} {_outdoor_value(name, before.get(name))} → {_outdoor_value(name, after.get(name))}{unit}")
    return lines


def change_details(entry: dict[str, Any]) -> list[str]:
    """'Width 0.80 m → 0.90 m' lines of a summary entry (at most DETAIL_LIMIT plus a '+n more' line)."""
    before = entry.get("before") or {}
    after = entry.get("after") or {}
    change = str(entry.get("change") or "")
    lines: list[str] = []
    if change == "removed":
        return lines
    if str(entry.get("kind") or "") in _OUTDOOR_KINDS:
        lines = _outdoor_details(before, after, change)
    elif change == "added":
        for name in _ADDED_FIELDS:
            if after.get(name) is not None:
                lines.append(f"{translate_or(f'assistant.panel.field.{name}', name)} {_format_value(name, after[name])}")
    else:
        for name in _FIELD_ORDER:
            if before.get(name) != after.get(name) and (name in before or name in after):
                label = translate_or(f"assistant.panel.field.{name}", name)
                lines.append(f"{label} {_format_value(name, before.get(name))} → {_format_value(name, after.get(name))}")
    if len(lines) > DETAIL_LIMIT:
        lines = lines[:DETAIL_LIMIT] + [t("assistant.panel.summary_more", count=len(lines) - DETAIL_LIMIT)]
    return lines


def claims_change(reply: str | None) -> bool:
    """True when a reply reads as if the plan had been changed (de / en past participles such as 'umbenannt',
    'renamed'); used only to warn when the turn produced no proposal.  The model's text itself is never altered."""
    return bool(_CLAIMED_CHANGE.search(str(reply or "")))


def _history_message(error: Any) -> str:
    key = str(getattr(error, "key", "") or "")
    params = dict(getattr(error, "params", None) or {})
    params.pop("key", None)
    if key:
        try:
            text = translate_or(key, "", **params)
        except (KeyError, IndexError, ValueError):
            text = ""
        if text:
            return text
    return str(error)


def _history_item(change: dict[str, Any]) -> str:
    """'Door O001 (changed)' for one change of an edit in the change list; outdoor elements with their short label and the
    changed values ('Tree B9 (changed): Height 5.80 → 7.80 m')."""
    kind = str(change.get("kind") or "")
    state = str(change.get("change") or "")
    text = (f"{translate_or(f'assistant.panel.kind.{kind}', kind)} {element_ref(change)} "
            f"({translate_or(f'assistant.panel.change.{state}', state)})")
    if kind in _OUTDOOR_KINDS and (change.get("before") or change.get("after")):
        details = [line for line in change_details(change) if line]
        if details:
            text += ": " + " · ".join(details)
    return text


def _selection_key(selection: dict[str, Any]) -> tuple[str, str] | None:
    if selection.get("type") == "element" and selection.get("id"):
        return str(selection.get("kind") or ""), str(selection["id"])
    return None


# ------------------------------------------------------------------------------------------------------ turns

@dataclass
class _Turn:
    """One question of a project level, running independently of the panel that asked it."""

    seq: int
    done: bool = False
    result: Any = None
    exports_started: bool = False  # the automatic update after a directly applied change was requested once
    task: Any = field(default=None, repr=False)


_TURNS: dict[tuple[str, str], _Turn] = {}
_PANELS: dict[tuple[str, str], "weakref.WeakSet[AssistantPanel]"] = {}
_TURN_SEQ = itertools.count(1)
TURN_POLL_S = 0.5


def _level_key(service: Any) -> tuple[str, str]:
    try:
        project = str(Path(service.project).resolve())
    except (OSError, TypeError, ValueError):
        project = str(getattr(service, "project", ""))
    return project, str(getattr(service, "level_id", ""))


def running_turn(service: Any) -> _Turn | None:
    """The unfinished question of the service's project level, if any."""
    turn = _TURNS.get(_level_key(service))
    return turn if turn is not None and not turn.done else None


def live_panels(key: tuple[str, str]) -> list["AssistantPanel"]:
    """Panels of a project level whose card is shown in a connected page."""
    return [panel for panel in list(_PANELS.get(key, ())) if panel.is_live()]


async def _run_turn(service: Any, turn: _Turn, prompt: str, selection: list[dict[str, Any]], allow_manual: bool) -> None:
    result = None
    try:
        result = await run.io_bound(service.ask, prompt, selection, allow_manual=allow_manual)
    except Exception as error:  # noqa: BLE001 - never a stack trace in the UI
        LOGGER.warning("assistant panel: request failed (%s)", type(error).__name__)
        from samcloud.assistant.agent import TurnResult

        result = TurnResult(error=t("assistant.panel.unexpected", detail=type(error).__name__), error_kind="unexpected")
    finally:  # also when the app shuts down mid-turn: waiting panels stop their spinner
        turn.result = result
        turn.done = True


def _deliver_turn(key: tuple[str, str], turn: _Turn, origin: "AssistantPanel | None" = None,
                  clear_draft: str | None = None) -> None:
    """Show a finished turn in every live panel waiting for it; the first one starts the automatic update of a directly
    applied change.  Without any live panel the asking panel still starts it (the change is applied already)."""
    targets = [panel for panel in live_panels(key) if panel.awaiting == turn.seq]
    if not targets and origin is not None and origin.awaiting == turn.seq:
        targets = [origin]
    for panel in targets:
        primary = not turn.exports_started
        if primary:
            turn.exports_started = True
        panel.show_turn(turn.result, primary=primary, clear_draft=clear_draft if panel is origin else None)


# ----------------------------------------------------------------------------------------------------------- panel

class AssistantPanel:
    """State and UI of the assistant for one project level on the /dxf page."""

    def __init__(
        self,
        service: Any,
        *,
        request_exports: Callable[[], bool],
        jobs_running: Callable[[], bool],
        export_state: Callable[[], tuple[str | None, list[str]]] | None = None,
        export_outcome: Callable[[], dict[str, Any]] | None = None,
        ifc_export: bool = True,
    ) -> None:
        self.service = service
        self.level_id = str(service.level_id)
        self.request_exports = request_exports
        self.jobs_running = jobs_running
        self.export_state = export_state or (lambda: (None, []))
        # {"state": None | "done" | "failed", "error": str | None, "plan_stale": bool} of the last automatic update
        self.export_outcome = export_outcome or (lambda: {"state": None, "error": None, "plan_stale": False})
        self.ifc_export = ifc_export
        self.selection: list[dict[str, Any]] = []
        self.draft = ""
        self.busy = False
        self.result: Any = None  # TurnResult of the last question of this panel
        self.manual_dismissed = False
        self.last_prompt = ""
        self.last_selection: list[dict[str, Any]] = []
        self.accept_notes: list[str] = []
        self.canvas: Any = None
        self.units_per_metre = 1.0
        self.viewer: Any = None
        self.offset = [0.0, 0.0, 0.0]
        self.preview_view: Any = None  # outdoor: the SVG preview of the published DXF (LV95 picks and overlays)
        self.map_offset: list[float] | None = None  # view_frame()["origin_offset"] (local -> LV95 / LN02), read once
        self._lock_buttons: list[Any] = []
        self._lock_hints: list[Any] = []
        self._export_labels: list[Any] = []
        self._locked: bool | None = None
        self._card: Any = None
        self.awaiting: int | None = None  # seq of the running turn this panel shows the spinner for
        pending = service.pending()
        if pending is not None:  # a proposal from an earlier visit of the page stays visible
            self.result = pending
        self.key = _level_key(service)
        _PANELS.setdefault(self.key, weakref.WeakSet()).add(self)
        turn = running_turn(service)
        if turn is not None:  # built while a question runs (page rebuilt / reconnected): wait for it, then show it
            self.busy = True
            self.awaiting = turn.seq
        ui.add_head_html(_STYLE)

    def is_live(self) -> bool:
        """True while the card of this panel is shown in a connected page."""
        card = self._card
        if card is None or card.is_deleted:
            return False
        client = card._client() if hasattr(card, "_client") else None
        return client is not None and not client.is_deleted

    def _context(self) -> Any:
        """UI context for notifications after an await (the card; nothing when it is gone)."""
        return self._card if self.is_live() else nullcontext()

    # ------------------------------------------------------------------------------------------ viewer wiring
    @property
    def outdoor(self) -> bool:
        """True for the outdoor (drone) domain: LV95 preview, outdoor wording."""
        try:
            return str(getattr(self.service, "domain_kind", "") or "") == "outdoor"
        except Exception:  # noqa: BLE001 - an unreadable domain counts as indoor wording
            return False

    def frame_offset(self) -> list[float]:
        """``[E, N, H]`` of the domain's local origin (``view_frame()["origin_offset"]``); zeros indoors / unreadable."""
        if self.map_offset is None:
            try:
                raw = self.service.view_frame().get("origin_offset")
                values = [float(value) for value in raw] if raw is not None else []
            except Exception as error:  # noqa: BLE001 - an unreadable model only disables the conversion
                LOGGER.warning("assistant panel: view frame unreadable (%s)", type(error).__name__)
                values = []
            self.map_offset = (values + [0.0, 0.0, 0.0])[:3] if values else [0.0, 0.0, 0.0]
        return self.map_offset

    def attach_preview(self, view: Any) -> None:
        """Outdoor: picking and overlays in the SVG preview of the published DXF.

        ``view`` (built by the page) has ``set_overlay(selection=..., preview=...)`` with LV95 items (``{"ring": [[e, n],
        ...] | "point": [e, n], "color", "dashed", "fill", "width"}``) and ``is_deleted``; the page forwards its clicks to
        :meth:`on_preview_pick`."""
        self.preview_view = view
        self.frame_offset()
        self.sync_views()

    def attach_canvas(self, canvas: Any) -> None:
        """Enable click picking on the 2D result canvas and show the current selection / preview there."""
        self.canvas = canvas
        try:
            self.units_per_metre = float(self.service.view_frame().get("units_per_metre") or 1.0)
        except Exception as error:  # noqa: BLE001 - an unreadable plan only disables the overlays
            LOGGER.warning("assistant panel: plan unreadable (%s)", type(error).__name__)
            self.units_per_metre = 1.0
        canvas.set_pick_enabled(True)
        canvas.on("pick", self.on_canvas_pick)
        self.sync_views()

    def attach_viewer(self, viewer: Any, props_data: dict[str, Any] | None) -> None:
        """Enable assistant picking in the IFC viewer (with ctrl and hit point) and show selection / preview there."""
        self.viewer = viewer
        self.offset = _origin_offset(props_data)
        floor = None
        try:
            floor_z = self.service.view_frame().get("floor_z")
            floor = None if floor_z is None else float(floor_z) - self.offset[1]
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("assistant panel: plan unreadable (%s)", type(error).__name__)
        viewer.set_pick_enabled(True, floor)
        viewer.on("pick", self.on_ifc_pick)
        self.sync_views()

    # ------------------------------------------------------------------------------------------ selection
    def on_canvas_pick(self, event: Any) -> None:
        args = event.args or {}
        try:
            x = float(args["x"]) / self.units_per_metre
            y = float(args["y"]) / self.units_per_metre
            selection = self.service.hit_test(x, y)
        except Exception as error:  # noqa: BLE001 - a broken plan or a bad event never breaks the page
            LOGGER.warning("assistant panel: 2D pick failed (%s)", type(error).__name__)
            ui.notify(t("assistant.panel.pick_failed"), type="warning")
            return
        self.apply_pick(selection, bool(args.get("ctrl")))

    def on_preview_pick(self, e: float, n: float, ctrl: bool = False) -> None:
        """A click in the outdoor DXF preview at LV95 (e, n): the element there, else a point (chip in LV95)."""
        try:
            offset = self.frame_offset()
            selection = self.service.hit_test(float(e) - offset[0], float(n) - offset[1])
        except Exception as error:  # noqa: BLE001 - a broken model or a bad event never breaks the page
            LOGGER.warning("assistant panel: preview pick failed (%s)", type(error).__name__)
            ui.notify(t("assistant.panel.pick_failed"), type="warning")
            return
        self.apply_pick(selection, bool(ctrl))

    def on_ifc_pick(self, event: Any) -> None:
        args = event.args or {}
        if "ctrl" not in args and "point" not in args:  # an old client without assistant picking
            return
        point = args.get("point")
        plan_point = model_to_plan(point, self.offset) if isinstance(point, (list, tuple)) and len(point) >= 3 else None
        try:
            selection = self.service.selection_from_global_id(str(args.get("global_id") or ""), plan_point)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("assistant panel: 3D pick failed (%s)", type(error).__name__)
            ui.notify(t("assistant.panel.pick_failed"), type="warning")
            return
        if selection.get("type") == "unknown":
            # neither a plan element nor a point: a plain click clears the selection and its highlight (like a click
            # into empty space; the page's property panel shows the clicked element or "nothing"); Ctrl+click keeps it
            ui.notify(t("assistant.panel.unknown_selection_outdoor" if self.outdoor else "assistant.panel.unknown_selection"),
                      type="warning")
            if not args.get("ctrl"):
                self.clear_selection()
            return
        self.apply_pick(selection, bool(args.get("ctrl")))

    def apply_pick(self, selection: dict[str, Any], add: bool) -> None:
        """Plain click replaces the selection; Ctrl+click toggles an element / adds a point."""
        if selection.get("type") not in ("element", "point"):
            return
        key = _selection_key(selection)
        if not add:
            self.selection = [selection]
        elif key is not None and any(_selection_key(item) == key for item in self.selection):
            self.selection = [item for item in self.selection if _selection_key(item) != key]
        elif len(self.selection) < MAX_SELECTION:
            self.selection = [*self.selection, selection]
        self.selection_changed()

    def remove_selection(self, index: int) -> None:
        if 0 <= index < len(self.selection):
            self.selection = [item for position, item in enumerate(self.selection) if position != index]
            self.selection_changed()

    def clear_selection(self) -> None:
        self.selection = []
        self.selection_changed()

    def selection_changed(self) -> None:
        self.sync_selection()
        self.chips.refresh()

    # ------------------------------------------------------------------------------------------ viewer sync
    def sync_views(self) -> None:
        self.sync_selection()
        self.sync_preview()

    def sync_selection(self) -> None:
        try:
            outlines = self.service.outlines_for(self.selection)
            global_ids = self.service.global_ids_for(self.selection)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("assistant panel: selection outlines failed (%s)", type(error).__name__)
            outlines, global_ids = [], []
        if self.canvas is not None and not self.canvas.is_deleted:
            factor = self.units_per_metre
            items = []
            for item in outlines:
                if item.get("ring"):
                    items.append({"ring": _scale_ring(item["ring"], factor)})
                elif item.get("point"):
                    items.append({"point": [round(float(item["point"][0]) * factor, 4), round(float(item["point"][1]) * factor, 4)]})
            self.canvas.set_selection_overlay(items)
        if self.viewer is not None and not self.viewer.is_deleted:
            self.viewer.set_selected_many(global_ids)
        if self.preview_view is not None and not self.preview_view.is_deleted:
            offset = self.frame_offset()
            marks: list[dict[str, Any]] = []
            for item in outlines:
                if item.get("ring"):
                    marks.append({"ring": shift_ring(item["ring"], offset), "color": SELECT_COLOR, "width": 3.0})
                elif item.get("point"):
                    point = item["point"]
                    marks.append({"point": [round(float(point[0]) + offset[0], 4), round(float(point[1]) + offset[1], 4)],
                                  "color": SELECT_COLOR, "width": 2.5})
            self.preview_view.set_overlay(selection=marks)

    def preview(self) -> dict[str, list[dict[str, Any]]]:
        if self.service.pending() is None:
            return {key: [] for key in _EMPTY_PREVIEW}
        try:
            return self.service.preview()
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("assistant panel: preview failed (%s)", type(error).__name__)
            return {key: [] for key in _EMPTY_PREVIEW}

    def sync_preview(self) -> None:
        """Colour the pending proposal in both viewers (added green, removed red, moved / modified amber)."""
        preview = self.preview()
        overlay: list[dict[str, Any]] = []
        highlight: dict[str, str] = {}
        boxes: list[dict[str, Any]] = []
        marks: list[dict[str, Any]] = []  # outdoor DXF preview (LV95)
        factor = self.units_per_metre
        map_offset = self.frame_offset() if self.preview_view is not None or self.outdoor else [0.0, 0.0, 0.0]
        z_datum = map_offset[2]  # outdoor: the domain's heights are LN02, the IFC model is in the local frame
        for bucket in ("added", "removed", "moved"):
            for item in preview.get(bucket, []):
                change = bucket if bucket != "moved" else ("moved" if item.get("change") == "moved" else "modified")
                color = CHANGE_COLORS[change]
                before_rings, after_rings = rings_of(item.get("before")), rings_of(item.get("after"))
                overlay.append({"change": change, "id": item.get("id"),
                                "before": _scale_ring(before_rings[0] if before_rings else None, factor),
                                "after": _scale_ring(after_rings[0] if after_rings else None, factor)})
                if bucket == "removed":
                    marks.extend({"ring": shift_ring(ring, map_offset), "color": color, "fill": 0.25, "dashed": True}
                                 for ring in before_rings)
                elif bucket == "added":
                    marks.extend({"ring": shift_ring(ring, map_offset), "color": color, "fill": 0.25} for ring in after_rings)
                else:
                    marks.extend({"ring": shift_ring(ring, map_offset), "color": color, "dashed": True, "width": 1.5}
                                 for ring in before_rings)
                    marks.extend({"ring": shift_ring(ring, map_offset), "color": color, "fill": 0.2} for ring in after_rings)
                global_id = item.get("global_id")
                if bucket != "added" and global_id:
                    highlight[str(global_id)] = color
                ring = item.get("before") if bucket == "removed" else item.get("after")
                z_min, z_max = item.get("z_min"), item.get("z_max")
                wants_box = bucket != "removed" or not global_id  # removed elements in the model are tinted instead
                if wants_box and rings_of(ring) and z_min is not None and z_max is not None:
                    boxes.append(model_box(ring, float(z_min) - z_datum, float(z_max) - z_datum, self.offset, color))
        if self.canvas is not None and not self.canvas.is_deleted:
            self.canvas.set_preview_overlay(overlay)
        if self.preview_view is not None and not self.preview_view.is_deleted:
            self.preview_view.set_overlay(preview=marks)
        if self.viewer is not None and not self.viewer.is_deleted:
            self.viewer.set_highlight(highlight)
            self.viewer.set_boxes(boxes)

    # ------------------------------------------------------------------------------------------ jobs / locks
    def is_locked(self) -> bool:
        """True while a DXF/IFC job runs or another writer holds the edit history lock (accept / undo disabled)."""
        try:
            if self.jobs_running():
                return True
        except Exception:  # noqa: BLE001
            return True
        try:
            from samcloud.assistant.history import history_locked

            return bool(history_locked(self.service.project, self.level_id))
        except Exception:  # noqa: BLE001
            return False

    def export_text(self, running: bool | None = None, status: dict[str, Any] | None = None) -> str:
        """Progress line of the automatic update after accept / undo ("Updating the DXF ... 40 %", "Waiting: ...") or,
        once everything finished, the done line; '' while nothing runs, after a failure or before the first update."""
        job, queue = self.export_state()
        parts = []
        is_running = running if running is not None else self._jobs_running_safe()
        if job is not None and is_running:
            fraction = float((status or {}).get("fraction", 0.0) or 0.0)
            key = _UPDATE_KEYS.get(job) or ("assistant.panel.update_env" if str(job).startswith(ENV_ACTION_PREFIX) else None)
            percent = f"{fraction * 100:.0f}"
            parts.append(t(key, percent=percent) if key else t("assistant.panel.export_running",
                                                                 action=translate_or(f"dxf.action.{job}", job), percent=percent))
        if queue:
            parts.append(t("assistant.panel.export_queued", actions=", ".join(translate_or(f"dxf.action.{a}", a) for a in queue)))
        if not parts and job is None and self._outcome().get("state") == "done":
            parts.append(t("assistant.panel.update_done" if self.ifc_export else "assistant.panel.update_done_dxf"))
        return " · ".join(parts)

    def _outcome(self) -> dict[str, Any]:
        try:
            outcome = self.export_outcome()
        except Exception:  # noqa: BLE001
            return {}
        return outcome if isinstance(outcome, dict) else {}

    def update_jobs(self, running: bool, status: dict[str, Any] | None = None) -> None:
        """Called by the page poll: enable / disable accept and undo, update the export line (in place, no re-render)."""
        self._lock_buttons = [button for button in self._lock_buttons if not button.is_deleted]
        self._lock_hints = [label for label in self._lock_hints if not label.is_deleted]
        self._export_labels = [label for label in self._export_labels if not label.is_deleted]
        locked = running or self.is_locked()
        if locked != self._locked:
            self._locked = locked
            for button in self._lock_buttons:
                button.set_enabled(not locked)
            for label in self._lock_hints:
                label.set_visibility(locked)
        text = self.export_text(running, status)
        for label in self._export_labels:
            label.set_text(text)
            label.set_visibility(bool(text))

    # ------------------------------------------------------------------------------------------ rendering
    def refresh_all(self) -> None:
        self.render.refresh()

    @ui.refreshable_method
    def chips(self) -> None:
        """Selection chips (the same selection is highlighted in both viewers)."""
        self._chip_row("assistant-chip-")

    def _chip_row(self, prefix: str) -> None:
        with ui.row().classes("items-center gap-2 w-full sam-ai-chips").mark(f"{prefix}row"):
            if not self.selection:
                key = "assistant.panel.selection_empty_outdoor" if self.outdoor else "assistant.panel.selection_empty"
                ui.label(t(key)).classes("sam-muted text-sm")
                return
            for index, selection in enumerate(self.selection):
                label = self.service.chip_label(selection)
                icon = "place" if selection.get("type") == "point" else "crop_square"
                ui.chip(label, icon=icon, removable=True, color="blue-grey-9", text_color="white",
                        on_value_change=lambda event, position=index: self.remove_selection(position) if not event.value else None,
                        ).props("dense square").mark(f"{prefix}{index}")
            ui.button(t("assistant.panel.clear_selection"), icon="backspace", on_click=self.clear_selection).props(
                "flat dense no-caps size=sm").mark(f"{prefix}clear")

    @ui.refreshable_method
    def render(self) -> None:
        """The one chat card of the page (between the 2D result plan and the IFC viewer)."""
        self._lock_buttons = []
        self._lock_hints = []
        self._export_labels = []
        self._locked = None
        with ui.card().classes("w-full gap-3 sam-ai-card").mark("assistant-panel") as card:
            self._card = card
            if self.busy and self.awaiting is not None:
                ui.timer(TURN_POLL_S, self.poll_turn)  # picks up a turn this panel did not start (rebuilt meanwhile)
            self._header()
            with ui.row().classes("w-full gap-5 no-wrap items-start"):
                with ui.column().classes("gap-3").style("flex:1 1 0;min-width:0"):
                    self._chat_box()
                    self._answer_box()
                    self._update_box()
                    ui.label(t("assistant.panel.selection")).classes("sam-ai-section")
                    self.chips()
                    self._input_row()
                with ui.column().classes("gap-2").style("flex:0 0 360px;min-width:0"):
                    self._history_box()
        self.update_jobs(self._jobs_running_safe())

    def _update_box(self) -> None:
        """State of the automatic DXF / IFC update after accept / undo: progress (updated in place by the page poll),
        the done line, a note when the exported plan itself is stale, and the error with a retry button."""
        export = ui.label(self.export_text()).classes("text-sm").style("color:var(--sam-accent)").mark("assistant-export-status")
        export.set_visibility(bool(export.text))
        self._export_labels.append(export)
        outcome = self._outcome()
        job, queue = self.export_state()
        if job is not None or queue:
            return
        if outcome.get("state") == "done" and outcome.get("plan_stale"):
            ui.label(t("assistant.panel.update_plan_stale")).classes("sam-muted text-xs").mark("assistant-export-plan-stale")
        if outcome.get("state") == "failed":
            with ui.column().classes("gap-1 w-full sam-dxf-banner is-error").mark("assistant-export-error"):
                ui.label(str(outcome.get("error") or t("assistant.panel.update_failed_generic"))).classes("text-sm").style(
                    "white-space:pre-wrap")
                ui.label(t("assistant.panel.update_kept")).classes("text-xs")
                retry = ui.button(t("assistant.panel.update_retry"), icon="refresh", on_click=self.retry_exports).props(
                    "unelevated dense no-caps color=primary").classes("self-start").mark("assistant-export-retry")
                self._lock_buttons.append(retry)

    def _jobs_running_safe(self) -> bool:
        try:
            return bool(self.jobs_running())
        except Exception:  # noqa: BLE001
            return False

    def _header(self) -> None:
        with ui.row().classes("items-center gap-3 w-full no-wrap"):
            ui.icon("smart_toy").classes("text-2xl").style("color:var(--sam-accent)")
            ui.label(t("assistant.panel.title")).classes("sam-section-title")
            model_line = self._model_line()
            if model_line:
                ui.label(model_line).classes("sam-muted text-sm").mark("assistant-model")
            ui.element("div").style("flex:1")
            ui.button(t("assistant.panel.clear_chat"), icon="delete_sweep", on_click=self.clear_chat).props(
                "flat dense no-caps").mark("assistant-clear-chat")
        if self.outdoor:
            ui.label(t("assistant.panel.outdoor_scope")).classes("sam-muted text-xs").mark("assistant-outdoor-scope")
        if self.full_control():
            ui.label(t("assistant.panel.full_mode")).classes("sam-muted text-xs").mark("assistant-full-mode")

    def full_control(self) -> bool:
        """True when the next question runs with full control (changes applied directly after a passed QA)."""
        try:
            return str(getattr(self.service, "tool_tier", "elements")) == "full"
        except Exception:  # noqa: BLE001 - unreadable settings = proposal mode
            return False

    @staticmethod
    def _model_line() -> str:
        try:
            from samcloud.assistant.settings import load_settings

            settings = load_settings()
        except Exception:  # noqa: BLE001
            return ""
        provider = translate_or(f"assistant.settings.provider.{settings.provider}", settings.provider)
        return t("assistant.panel.model_line", provider=provider, model=settings.model) if settings.model else provider

    def _chat_box(self) -> None:
        try:
            messages = self.service.chat()[-CHAT_LIMIT:]
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("assistant panel: chat unreadable (%s)", type(error).__name__)
            messages = []
        pending = self.service.pending() is not None
        with ui.element("div").classes("sam-ai-chat w-full").mark("assistant-chat"):
            if not messages:
                ui.label(t("assistant.panel.chat_empty")).classes("sam-muted text-sm")
            for position, message in enumerate(messages):
                user = message.get("role") == "user"
                with ui.element("div").classes(f"sam-ai-msg {'is-user' if user else 'is-assistant'}"):
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.label(t("assistant.panel.you") if user else t("assistant.panel.assistant")).classes("sam-ai-who")
                        if pending and not user and position == len(messages) - 1:  # the reply that made the proposal
                            ui.label(t("assistant.panel.pending_badge")).classes("sam-ai-badge is-modified").mark("assistant-pending-chat")
                    ui.label(str(message.get("text") or ""))
                    meta = message.get("meta") or {}
                    if user and meta.get("selection"):
                        ui.label(", ".join(str(item) for item in meta["selection"])).classes("sam-ai-detail")

    def _last_chat_text(self) -> str | None:
        """Text of the last chat message (the reply is not repeated in the answer box when the chat shows it)."""
        try:
            messages = self.service.chat()
        except Exception:  # noqa: BLE001
            return None
        return str(messages[-1].get("text") or "") if messages else None

    def _legend(self) -> None:
        with ui.row().classes("items-center gap-3"):
            for key, color in (("legend_added", CHANGE_COLORS["added"]), ("legend_removed", CHANGE_COLORS["removed"]),
                               ("legend_moved", CHANGE_COLORS["moved"])):
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.element("span").classes("sam-ai-dot").style(f"background:{color}")
                    ui.label(t(f"assistant.panel.{key}")).classes("sam-muted text-xs")

    def _answer_box(self) -> None:
        result = self.result
        pending = self.service.pending()
        if self.busy:
            with ui.row().classes("items-center gap-2").mark("assistant-busy"):
                ui.spinner(size="md")
                ui.label(t("assistant.panel.thinking")).classes("sam-muted")
            return
        if self.accept_notes:
            with ui.column().classes("gap-1 w-full").mark("assistant-accept-notes"):
                for note in self.accept_notes:
                    ui.label(note).classes("sam-dxf-banner is-info w-full text-sm")
        if result is None and pending is None:
            return
        shown = result if result is not None else pending
        with ui.column().classes("sam-ai-box w-full gap-2").mark("assistant-answer"):
            if pending is not None and shown is pending:
                ui.label(t("assistant.panel.pending_badge")).classes("sam-ai-badge is-modified self-start").mark("assistant-pending")
            if shown.reply and not shown.error and shown.reply != self._last_chat_text():
                ui.label(str(shown.reply)).classes("text-sm").style("white-space:pre-wrap").mark("assistant-reply")
            notes = [str(note) for note in (getattr(shown, "notes", None) or [])]
            if getattr(shown, "retried_fresh", False):
                # the question was asked again in a fresh chat context (the agent puts that note first)
                fresh = notes.pop(0) if notes else t("assistant.agent.retried_fresh")
                ui.label(fresh).classes("text-xs sam-dxf-banner is-info").mark("assistant-retried-fresh")
            if shown.error:
                with ui.column().classes("gap-0 w-full sam-dxf-banner is-error").mark("assistant-error"):
                    ui.label(error_title(shown.error_kind)).classes("text-sm").style("font-weight:600")
                    ui.label(str(shown.error)).classes("text-sm").style("white-space:pre-wrap")
            if shown.summary and pending is not None and shown is pending:
                with ui.row().classes("items-center gap-3 w-full"):
                    ui.label(t("assistant.panel.proposal")).classes("sam-ai-section")
                    ui.label(t("assistant.panel.changes_count", count=len(shown.summary))).classes("text-sm")
                    ui.element("div").style("flex:1")
                    self._legend()
                with ui.column().classes("gap-0 w-full").mark("assistant-summary"):
                    for entry in shown.summary:
                        change = str(entry.get("change") or "modified")
                        with ui.element("div").classes("sam-ai-change"):
                            ui.label(translate_or(f"assistant.panel.change.{change}", change)).classes(f"sam-ai-badge is-{change}")
                            ui.label(change_title(entry)).style("flex:none")
                            details = change_details(entry)
                            if details:
                                ui.label(" · ".join(details)).classes("sam-ai-detail")
            if getattr(shown, "applied", False) and shown.summary and pending is None:
                self._applied_box(shown)
            elif pending is None and getattr(shown, "proposal", None) is None and not shown.error:
                # a finished turn without a proposal: say plainly that nothing changed (even if the reply claims it)
                with ui.column().classes("gap-0 w-full sam-dxf-banner is-info").mark("assistant-no-proposal"):
                    ui.label(t("assistant.panel.no_proposal")).classes("text-sm")
                    if claims_change(shown.reply):
                        ui.label(t("assistant.panel.no_proposal_claim")).classes("text-sm").style("font-weight:600").mark(
                            "assistant-no-proposal-claim")
            for note in notes:
                ui.label(note).classes("sam-muted text-xs")
            usage = usage_text(shown)
            if usage:
                ui.label(usage).classes("sam-muted text-xs").mark("assistant-usage")
            if shown.asks_manual and not self.manual_dismissed and not shown.error:
                with ui.row().classes("items-center gap-2 w-full").mark("assistant-manual"):
                    ui.label(t("assistant.panel.manual_question")).classes("text-sm")
                    ui.button(t("assistant.panel.manual_yes"), icon="add_location_alt", on_click=self.manual_yes).props(
                        "unelevated dense no-caps color=primary").mark("assistant-manual-yes")
                    ui.button(t("assistant.panel.manual_no"), on_click=self.manual_no).props("flat dense no-caps").mark("assistant-manual-no")
            if pending is not None:
                with ui.row().classes("items-center gap-2 w-full"):
                    accept = ui.button(t("assistant.panel.accept"), icon="check", on_click=self.accept_clicked).props(
                        "unelevated no-caps color=positive").mark("assistant-accept")
                    ui.button(t("assistant.panel.discard"), icon="close", on_click=self.discard_clicked).props(
                        "flat no-caps color=negative").mark("assistant-discard")
                    self._lock_buttons.append(accept)
                    hint = ui.label(t("assistant.panel.locked_hint")).classes("sam-muted text-xs").mark("assistant-locked")
                    hint.set_visibility(False)
                    self._lock_hints.append(hint)

    def _applied_box(self, shown: Any) -> None:
        """Full control: the change was applied right away -- one "Applied -- Undo" line with the changed elements."""
        edit_id = str(getattr(shown, "edit_id", "") or "")
        with ui.column().classes("gap-1 w-full sam-dxf-banner is-info").mark("assistant-applied"):
            with ui.row().classes("items-center gap-2 w-full no-wrap"):
                ui.label(t("assistant.panel.applied_title")).classes("sam-ai-badge is-active")
                ui.label(t("assistant.panel.applied_line", count=len(shown.summary), id=edit_id)).classes("text-sm")
                ui.element("div").style("flex:1")
                if edit_id and self._edit_active(edit_id):
                    button = ui.button(t("assistant.panel.applied_undo"), icon="undo",
                                       on_click=lambda _event=None: self.undo_applied(edit_id)).props(
                        "flat dense no-caps color=warning").mark("assistant-applied-undo")
                    self._lock_buttons.append(button)
            with ui.column().classes("gap-0 w-full").mark("assistant-applied-summary"):
                for entry in shown.summary[:12]:
                    change = str(entry.get("change") or "modified")
                    with ui.element("div").classes("sam-ai-change"):
                        ui.label(translate_or(f"assistant.panel.change.{change}", change)).classes(f"sam-ai-badge is-{change}")
                        ui.label(change_title(entry)).style("flex:none")
                if len(shown.summary) > 12:
                    ui.label(t("assistant.panel.summary_more", count=len(shown.summary) - 12)).classes("sam-muted text-xs")

    def _edit_active(self, edit_id: str) -> bool:
        try:
            return any(str(edit.get("id")) == edit_id and edit.get("status") == "active" for edit in self.service.edits())
        except Exception:  # noqa: BLE001
            return False

    def undo_applied(self, edit_id: str) -> None:
        """Undo of a directly applied change (same confirmation rules as the change list)."""
        record = None
        try:
            record = next((edit for edit in self.service.edits() if str(edit.get("id")) == edit_id), None)
        except Exception:  # noqa: BLE001
            record = None
        self.undo_clicked(record or {"id": edit_id, "summary": []})

    def _input_row(self) -> None:
        with ui.row().classes("w-full items-end gap-2 no-wrap"):
            area = ui.textarea(
                t("assistant.panel.input_label"), placeholder=t("assistant.panel.input_placeholder"),
                value=self.draft, on_change=lambda event: self._set_draft(event.value),
            ).props("outlined dense autogrow rows=2").classes("flex-1").mark("assistant-input")
            area.set_enabled(not self.busy)
            area.on("keydown.enter.exact.prevent", lambda _event: self.send_clicked())
            button = ui.button(t("assistant.panel.send"), icon="send", on_click=lambda _event=None: self.send_clicked()).props(
                "unelevated no-caps color=primary").mark("assistant-send")
            button.set_enabled(not self.busy)
        ui.label(t("assistant.panel.input_hint")).classes("sam-muted text-xs")

    def _set_draft(self, value: Any) -> None:
        self.draft = str(value or "")

    def _history_box(self) -> None:
        ui.label(t("assistant.panel.history_title")).classes("sam-ai-section")
        for message in self._pipeline_errors():
            ui.label(message).classes("sam-dxf-banner is-warning w-full text-xs").mark("assistant-pipeline-error")
        try:
            edits = list(reversed(self.service.edits()))
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("assistant panel: edits unreadable (%s)", type(error).__name__)
            edits = []
        with ui.column().classes("gap-0 w-full sam-ai-history").mark("assistant-history"):
            if not edits:
                ui.label(t("assistant.panel.history_empty")).classes("sam-muted text-sm")
            for edit in edits:
                self._edit_row(edit)

    def _edit_row(self, edit: dict[str, Any]) -> None:
        edit_id = str(edit.get("id") or "")
        status = str(edit.get("status") or "active")
        summary = list(edit.get("summary") or [])
        with ui.column().classes("gap-1 w-full sam-ai-edit").mark(f"assistant-edit-{edit_id}"):
            with ui.row().classes("items-center gap-2 w-full no-wrap"):
                ui.label(translate_or(f"assistant.panel.status.{status}", status)).classes(f"sam-ai-badge is-{status}")
                ui.label(edit_id).classes("text-sm").style("font-weight:600")
                ui.label(t("assistant.panel.changes_count", count=len(summary))).classes("sam-muted text-xs")
                ui.element("div").style("flex:1")
                if status in ("active", "stale"):
                    label = t("assistant.panel.undo") if status == "active" else t("assistant.panel.remove_stale")
                    button = ui.button(label, icon="undo" if status == "active" else "delete_outline",
                                       on_click=lambda _event=None, record=edit: self.undo_clicked(record)).props(
                        "flat dense no-caps size=sm").mark(f"assistant-undo-{edit_id}")
                    self._lock_buttons.append(button)
            # the stored changes carry the before / after snapshots (labels, values); the summary only ids and kinds
            changes = [change for change in edit.get("changes") or [] if isinstance(change, dict)] or summary
            items = ", ".join(_history_item(change) for change in changes[:4])
            if len(changes) > 4:
                items += " " + t("assistant.panel.summary_more", count=len(changes) - 4)
            if items:
                ui.label(items).classes("sam-ai-detail")
            if edit.get("prompt"):
                prompt = str(edit["prompt"])
                ui.label(f"„{prompt[:140]}{'…' if len(prompt) > 140 else ''}“").classes("sam-muted text-xs").style("font-style:italic")
            if status == "stale":
                reason = str(edit.get("stale_reason") or "")
                ui.label(t("assistant.panel.stale_reason", reason=reason)).classes("text-xs").style("color:#ffb74d").mark(f"assistant-stale-{edit_id}")

    def _pipeline_errors(self) -> list[str]:
        """Steps of the last pipeline / export runs whose edits could not be re-applied (pipeline_report.json)."""
        try:
            from samcloud.assistant.paths import assistant_dir

            path = assistant_dir(self.service.project, self.level_id) / "pipeline_report.json"
            if not path.is_file():
                return []
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        messages = []
        for step, info in (report.items() if isinstance(report, dict) else []):
            if isinstance(info, dict) and info.get("error"):
                messages.append(t("assistant.panel.pipeline_error", step=str(step), error=str(info["error"])[:200]))
        return messages

    # ------------------------------------------------------------------------------------------ actions
    async def send_clicked(self) -> None:
        await self.send(self.draft.strip(), list(self.selection), clear_draft="main")

    async def send(self, prompt: str, selection: list[dict[str, Any]], *, allow_manual: bool = False,
                   clear_draft: str | None = None) -> None:
        """Ask the assistant (blocking call in a worker thread); shows the answer and the preview afterwards."""
        if self.busy:
            return
        if not prompt.strip():
            ui.notify(t("assistant.agent.empty_prompt"), type="warning")
            return
        if running_turn(self.service) is not None:
            ui.notify(t("assistant.agent.busy"), type="warning")
            return
        turn = _Turn(seq=next(_TURN_SEQ))
        _TURNS[self.key] = turn
        self.busy = True
        self.awaiting = turn.seq
        self.last_prompt = prompt
        self.last_selection = [dict(item) for item in selection]
        self.manual_dismissed = False
        self.accept_notes = []
        for panel in live_panels(self.key):  # other tabs of the same level show the spinner too
            if panel is not self and not panel.busy:
                panel.busy, panel.awaiting = True, turn.seq
                panel.refresh_all()
        self.refresh_all()
        # an app-level task: it finishes (and its result is delivered) even when this card or its connection is gone
        turn.task = background_tasks.create(_run_turn(self.service, turn, prompt, self.last_selection, allow_manual),
                                            name="assistant-turn")
        try:
            await asyncio.shield(turn.task)
        except asyncio.CancelledError:
            return  # the page went away; a rebuilt panel picks the result up (poll_turn)
        _deliver_turn(self.key, turn, origin=self, clear_draft=clear_draft)

    def poll_turn(self) -> None:
        """Timer of a panel waiting for a turn: show it once it is done (or stop waiting when it vanished)."""
        if not self.busy or self.awaiting is None:
            return
        turn = _TURNS.get(self.key)
        if turn is None or turn.seq != self.awaiting:
            self.busy, self.awaiting = False, None
            self.refresh_all()
            return
        if turn.done:
            _deliver_turn(self.key, turn)

    def show_turn(self, result: Any, *, primary: bool = True, clear_draft: str | None = None) -> None:
        """Show a finished turn (answer, proposal preview); a directly applied change refreshes the viewers and, on the
        ``primary`` panel, starts the automatic DXF / IFC update."""
        self.busy = False
        self.awaiting = None
        self.result = result
        if result is not None and result.error_kind != "busy" and clear_draft == "main":
            self.draft = ""
        with self._context():
            if result is not None and getattr(result, "applied", False):
                # full control: applied after the passed QA -- refresh like after an accept and start the automatic exports
                self._after_plan_change()
                key = "assistant.panel.applied_notify" if self.ifc_export else "assistant.panel.applied_notify_dxf"
                try:
                    ui.notify(t(key, id=str(getattr(result, "edit_id", "") or "")), type="positive")
                except RuntimeError:  # no page left to notify
                    pass
                if primary:
                    try:
                        self.request_exports()
                    except Exception as error:  # noqa: BLE001 - the change is applied; "Export again" stays possible
                        LOGGER.warning("assistant panel: automatic update not started (%s)", type(error).__name__)
            self.sync_preview()
            self.refresh_all()

    async def manual_yes(self) -> None:
        await self.send(self.last_prompt, self.last_selection, allow_manual=True)

    def manual_no(self) -> None:
        self.manual_dismissed = True
        self.render.refresh()

    def clear_chat(self) -> None:
        try:
            self.service.clear_chat()
        except OSError as error:
            ui.notify(str(error), type="negative")
            return
        ui.notify(t("assistant.panel.chat_cleared"))
        self.render.refresh()

    def discard_clicked(self) -> None:
        self.service.discard()
        self.result = None
        self.sync_preview()
        self.refresh_all()
        ui.notify(t("assistant.panel.discarded"))

    def accept_clicked(self) -> None:
        pending = self.service.pending()
        if pending is None:
            ui.notify(t("assistant.agent.no_pending"), type="warning")
            self.refresh_all()
            return
        if self.is_locked():
            ui.notify(t("assistant.panel.locked_hint"), type="warning")
            return
        if pending.needs_confirmation:
            self.confirm_dialog(pending)
            return
        self.accept(confirmed=False)

    def confirm_dialog(self, pending: Any) -> None:
        removed = [change_title(entry) for entry in pending.summary if entry.get("change") == "removed"]
        with ui.dialog() as dialog, ui.card().classes("gap-3").style("min-width:380px").mark("assistant-confirm-dialog"):
            ui.label(t("assistant.panel.confirm_title")).classes("sam-section-title")
            ui.label(t("assistant.panel.confirm_body", count=len(pending.summary))).classes("text-sm")
            if removed:
                ui.label(t("assistant.panel.confirm_deletions", items=", ".join(removed))).classes("text-sm").style("color:#ef6b73")

            def confirm() -> None:
                dialog.close()
                self.accept(confirmed=True)

            with ui.row().classes("justify-end w-full gap-2"):
                ui.button(t("assistant.panel.confirm_cancel"), on_click=dialog.close).props("flat no-caps").mark("assistant-confirm-cancel")
                ui.button(t("assistant.panel.confirm_ok"), icon="check", on_click=confirm).props(
                    "unelevated no-caps color=positive").mark("assistant-confirm-ok")
        dialog.open()

    def accept(self, *, confirmed: bool) -> None:
        from samcloud.assistant.history import HistoryError

        try:
            edit_id = self.service.accept(confirmed=confirmed)
        except HistoryError as error:
            message = _history_message(error)
            if error.key in ("assistant.history.outdated", "assistant.history.ambiguous"):
                self.service.discard()
                self.result = None
                message = f"{message} {t('assistant.panel.ask_again')}"
                self.sync_preview()
            self.accept_notes = [message]
            ui.notify(message, type="warning")
            self.refresh_all()
            return
        report = getattr(self.service, "last_accept_report", None)
        notes = [str(note) for note in (getattr(report, "notes", None) or [])]
        rebased = dict(getattr(report, "rebased", None) or {})
        if rebased:
            notes.insert(0, t("assistant.panel.rebased", mapping=", ".join(f"{old} → {new}" for old, new in rebased.items())))
        self.accept_notes = notes
        self.result = None
        self._after_plan_change()
        ui.notify(t("assistant.panel.accepted" if self.ifc_export else "assistant.panel.accepted_dxf", id=edit_id), type="positive")
        self.request_exports()
        self.refresh_all()

    def undo_clicked(self, edit: dict[str, Any]) -> None:
        if self.is_locked():
            ui.notify(t("assistant.panel.locked_hint"), type="warning")
            return
        summary = list(edit.get("summary") or [])
        if len(summary) > UNDO_CONFIRM_COUNT or any(entry.get("change") == "removed" for entry in summary):
            edit_id = str(edit.get("id") or "")
            with ui.dialog() as dialog, ui.card().classes("gap-3").style("min-width:360px").mark("assistant-undo-dialog"):
                ui.label(t("assistant.panel.undo_confirm_title", id=edit_id)).classes("sam-section-title")
                ui.label(t("assistant.panel.undo_confirm_body", count=len(summary))).classes("text-sm")

                def confirm() -> None:
                    dialog.close()
                    self.undo(edit_id)

                with ui.row().classes("justify-end w-full gap-2"):
                    ui.button(t("assistant.panel.confirm_cancel"), on_click=dialog.close).props("flat no-caps").mark("assistant-undo-cancel")
                    ui.button(t("assistant.panel.undo"), icon="undo", on_click=confirm).props(
                        "unelevated no-caps color=warning").mark("assistant-undo-ok")
            dialog.open()
            return
        self.undo(str(edit.get("id") or ""))

    def undo(self, edit_id: str) -> None:
        from samcloud.assistant.history import HistoryError

        try:
            self.service.undo(edit_id)
        except HistoryError as error:
            ui.notify(_history_message(error), type="warning")
            self.refresh_all()
            return
        self.result = None
        self.accept_notes = []
        self._after_plan_change()
        ui.notify(t("assistant.panel.undone", id=edit_id), type="positive")
        self.request_exports()
        self.refresh_all()

    def retry_exports(self) -> None:
        """'Export again' after a failed automatic update: run the DXF / IFC update queue once more (the edit is kept)."""
        if self.is_locked():
            ui.notify(t("assistant.panel.locked_hint"), type="warning")
            return
        self.request_exports()
        self.refresh_all()

    def _after_plan_change(self) -> None:
        """The plan on disk changed: refresh the selection labels (drop vanished elements) and both viewers."""
        try:
            self.selection = self.service.refresh_selections(self.selection)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("assistant panel: selection refresh failed (%s)", type(error).__name__)
            self.selection = []
        self.sync_views()
        self.chips.refresh()


def assistant_available(project: Any, level_id: str, mode: str | None = None) -> bool:
    """True when the assistant is enabled, configured and the level's project type can be edited (indoor: its ortho
    plan_model); reads only the assistant settings while it is switched off."""
    try:
        from samcloud.assistant.settings import load_settings

        if not load_settings().enabled:
            return False
        from samcloud.assistant.service import AssistantService

        return bool(AssistantService(project, level_id, mode=mode).is_available()[0])
    except Exception as error:  # noqa: BLE001 - a broken assistant never breaks the page
        LOGGER.warning("assistant panel: not available (%s)", type(error).__name__)
        return False


def assistant_unavailable_reason(project: Any, level_id: str, mode: str | None = None) -> str | None:
    """i18n key of the one hint line the page shows instead of the panel: the assistant is enabled and configured, but
    the project type / level cannot be edited (no plan yet, outdoor without environment chain, outdoor pending).  None
    while the assistant is off, not configured (the settings page tells that) or available.  Reads only the assistant
    settings while it is switched off."""
    try:
        from samcloud.assistant.settings import load_settings

        if not load_settings().enabled:
            return None
        from samcloud.assistant.domains.registry import DOMAIN_REASONS
        from samcloud.assistant.service import AssistantService

        available, reason = AssistantService(project, level_id, mode=mode).is_available()
    except Exception as error:  # noqa: BLE001 - a broken assistant never breaks the page
        LOGGER.warning("assistant panel: availability unknown (%s)", type(error).__name__)
        return None
    if available or reason not in DOMAIN_REASONS:
        return None
    return reason


def create_panel(project: Any, level_id: str, *, mode: str | None = None, **callbacks: Any) -> AssistantPanel | None:
    """The panel for a project level, or None when the assistant is off, not configured or the level cannot be edited.

    ``mode`` is the project mode of the page (picks the project's assistant domain).  Reads only the assistant settings
    when the assistant is switched off (no credentials, no plan)."""
    try:
        from samcloud.assistant.settings import load_settings

        if not load_settings().enabled:
            return None
        from samcloud.assistant.service import AssistantService

        service = AssistantService(project, level_id, mode=mode)
        available, _reason = service.is_available()
    except Exception as error:  # noqa: BLE001 - a broken assistant never breaks the page
        LOGGER.warning("assistant panel: not available (%s)", type(error).__name__)
        return None
    if not available:
        return None
    return AssistantPanel(service, **callbacks)


__all__ = ["AssistantPanel", "live_panels", "running_turn", "assistant_available", "assistant_unavailable_reason", "claims_change", "create_panel", "change_details", "change_title", "model_box", "model_to_plan", "plan_to_model",
           "element_ref", "rings_of", "shift_ring", "usage_text"]
