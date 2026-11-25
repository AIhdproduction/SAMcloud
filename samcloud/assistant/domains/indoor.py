"""Indoor domain: the plan model (``plan_model.json``) of one level, edited with the operations of ``edits.OPS``.

A thin wrapper around the existing modules (``edits``, ``selection``, ``qa``, ``pointcloud``, ``history``, ``prompts``):
nothing here changes their behaviour.  The tier ``"full"`` (full control) adds the project-wide operations of
:mod:`samcloud.assistant.edits_full` (new walls, room outlines, ceiling objects, bulk operations); the service applies a
full-tier proposal directly after a passed QA.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from samcloud.assistant.domains.base import TIER_ELEMENTS, TIER_FULL, AssistantDomain, HistoryBackend, ToolDef, normalize_tier

KIND = "indoor"
OUTDOOR_SITE_LEVEL = "site"

_VOCABULARY: dict[str, dict[str, Any]] = {
    "de": {
        "project_type": "Innenprojekt",
        "model_phrase": "das Planmodell (DXF/IFC)",
        "element_types": {"wall": "Wand", "opening": "Öffnung", "door": "Tür", "window": "Fenster", "passage": "Durchgang",
                          "furniture": "Möbel", "room": "Raum", "ceiling_object": "Deckenobjekt"},
        "rules": [
            "Keine Evidenz = nichts. Bevor du etwas hinzufügst, schau mit den Mess-Werkzeugen (points_in_box, section, "
            "fit_surface) in der\n   Punktwolke nach. Findest du dort keine Punkte, sag es und frag nach, z. B.: \"Dort ist in "
            "der Punktwolke kein Stuhl. Soll ich\n   trotzdem einen setzen?\" Setze manual=true nur, wenn der Benutzer das "
            "ausdrücklich bestätigt hat.",
            "Gemessen, nicht geschätzt. Masse kommen aus den Mess-Werkzeugen oder aus dem Satz des Benutzers, nie aus "
            "Vermutungen.",
            "Ändere nur die Elemente, die der Benutzer meint (Auswahl im Kontext, sonst nachfragen). Nur die "
            "Bearbeitungs-Werkzeuge ändern\n   etwas; es gibt keinen Code, keine Shell und keinen Dateizugriff.",
            "Abhängigkeiten (Räume, Öffnungen, Wandanschlüsse) rechnet SAMcloud selbst. Du musst sie nicht nachführen.",
            "{tier}",
            "Einheiten: Meter, Planmeter für Positionen (x/y), Höhen über dem Boden. Grad gegen den Uhrzeigersinn.",
            "Schlägt ein Werkzeug fehl, lies den Fehler und korrigiere, oder erkläre kurz, warum es nicht geht.",
        ],
        "closing": {
            "elements": "Am Ende beschreibst du den Vorschlag in ein bis drei Sätzen (was vorher, was nachher) oder stellst "
                        "deine Rückfrage.\nFormuliere immer als Vorschlag, nie als erledigt: z. B. \"Vorschlag: Tür O008 "
                        "von 0.90 m auf 0.80 m verschmälern. Nach\nÜbernehmen wird sie 0.80 m breit.\" Schreibe nie "
                        "\"wurde geändert\", \"ist jetzt\" oder \"habe geändert\".",
            "full": "Am Ende beschreibst du in ein bis drei Sätzen, was nach bestandener QA direkt angewendet wird (was vorher, "
                    "was nachher), oder\nstellst deine Rückfrage. Bei Sammeländerungen nennst du die Anzahl der Elemente. "
                    "Behaupte nie, etwas sei angewendet,\nwenn kein Bearbeitungs-Werkzeug erfolgreich war.",
        },
    },
    "en": {
        "project_type": "indoor project",
        "model_phrase": "the plan model (DXF/IFC)",
        "element_types": {"wall": "wall", "opening": "opening", "door": "door", "window": "window", "passage": "passage",
                          "furniture": "furniture", "room": "room", "ceiling_object": "ceiling object"},
        "rules": [
            "No evidence = nothing. Before you add anything, look in the point cloud with the measuring tools "
            "(points_in_box, section,\n   fit_surface). If there are no points, say so and ask, e.g.: \"There is no chair in "
            "the point cloud there. Shall I place one\n   anyway?\" Use manual=true only after the user explicitly confirmed it.",
            "Measured, not estimated. Dimensions come from the measuring tools or from the user's sentence, never from "
            "guesses.",
            "Only change the elements the user means (the selection in the context, otherwise ask). Only the edit tools "
            "change anything;\n   there is no code, no shell and no file access.",
            "Dependencies (rooms, openings, wall joins) are computed by SAMcloud. You do not need to update them.",
            "{tier}",
            "Units: metres, plan metres for positions (x/y), heights above the floor. Degrees counter-clockwise.",
            "When a tool fails, read the error and correct the call, or briefly explain why it is not possible.",
        ],
        "closing": {
            "elements": "At the end describe the proposal in one to three sentences (before and after) or ask your "
                        "question.\nAlways phrase it as a proposal, never as done, e.g. \"Proposal: narrow door O008 from "
                        "0.90 m to 0.80 m. After you accept it,\nit will be 0.80 m wide.\" Never write \"was changed\", "
                        "\"is now\" or \"I changed\".",
            "full": "At the end describe in one to three sentences what will be applied directly after a passed QA (before "
                    "and after), or ask\nyour question. For bulk changes name the number of elements. Never claim that "
                    "anything is applied when no edit tool\nsucceeded.",
        },
    },
}


def _r(value: float) -> float:
    return float(round(float(value), 3))


# ----------------------------------------------------------------------------------------------- plan geometry helpers

def outline_ring(plan: Any, kind: str, element_id: str) -> list[list[float]] | None:
    """2D outline (closed ring, plan metres) of an element for the DXF overlay; None when unknown."""
    from samcloud.assistant.edits import _polygon_of, _room_polygon, find_element, wall_band

    found = find_element(plan, element_id)
    if found is None:
        return None
    kind, element = found
    upm = plan.units_per_metre or 1.0
    polygon = None
    try:
        if kind == "wall":
            polygon = wall_band(element, upm)
        elif kind == "opening":
            axis = (math.cos(element.axis_angle), math.sin(element.axis_angle))
            normal = (-axis[1], axis[0])
            cx, cy = float(element.center[0]), float(element.center[1])
            hw = float(element.width_m) * upm / 2.0
            ht = max(float(element.wall_thickness_m), 0.05) * upm / 2.0
            return [[_r((cx + axis[0] * a + normal[0] * b) / upm), _r((cy + axis[1] * a + normal[1] * b) / upm)]
                    for a, b in ((-hw, -ht), (hw, -ht), (hw, ht), (-hw, ht))]
        elif kind == "furniture":
            polygon = _polygon_of(element.footprint)
        elif kind == "room":
            polygon = _room_polygon(element)
        else:
            polygon = _polygon_of(element.get("footprint"))
    except Exception:  # noqa: BLE001 - a degenerate element has no overlay
        return None
    if polygon is None or polygon.is_empty:
        return None
    return [[_r(x / upm), _r(y / upm)] for x, y in list(polygon.exterior.coords)[:-1]]


def height_range(plan: Any, kind: str, element_id: str) -> tuple[float, float] | None:
    """Absolute height range (metres, the IFC frame) of a wall / opening / furniture / ceiling object for the 3D
    preview boxes; None for rooms and unknown elements."""
    from samcloud.assistant.edits import find_element, furniture_height_m

    found = find_element(plan, element_id)
    if found is None:
        return None
    kind, element = found
    upm = plan.units_per_metre or 1.0
    floor = float(plan.floor_z) / upm
    ceiling = float(plan.ceiling_z) / upm if plan.ceiling_z is not None and plan.ceiling_z > plan.floor_z else floor + 2.6
    try:
        if kind == "wall":
            top = floor + float(element.height_m) if element.height_m else ceiling
            return _r(floor), _r(top)
        if kind == "opening":
            return _r(floor + float(element.sill_m)), _r(floor + max(float(element.head_m), float(element.sill_m) + 0.05))
        if kind == "furniture":
            if element.z_min is not None and element.z_max is not None and float(element.z_max) > float(element.z_min):
                return _r(float(element.z_min) / upm), _r(float(element.z_max) / upm)
            return _r(floor), _r(floor + max(furniture_height_m(plan, element), 0.05))
        if kind == "ceiling_object":
            return _r(ceiling - 0.10), _r(ceiling)
    except (TypeError, ValueError):
        return None
    return None


def selection_for(plan: Any, kind: str, plan_id: str) -> dict[str, Any]:
    """Selection dict of a plan element (``{"type": "unknown", "id"}`` when it does not exist)."""
    from samcloud.assistant.edits import find_element

    found = find_element(plan, plan_id)
    if found is None:
        return {"type": "unknown", "id": plan_id}
    kind, element = found
    if kind == "furniture":
        label = str(element.label or element.id)
    elif kind == "room":
        label = str(element.name or element.id)
    elif kind == "ceiling_object":
        label = str(element.get("label") or element.get("id"))
    elif kind == "opening":
        label = f"{element.kind} {element.id}"
    else:
        label = str(plan_id)
    selection = {"type": "element", "kind": kind, "id": str(plan_id), "label": label}
    if kind == "opening":
        selection["subkind"] = element.kind
        selection["width_m"] = float(round(float(element.width_m), 2))
    return selection


# ------------------------------------------------------------------------------------------------------------- domain

class IndoorDomain(AssistantDomain):
    """The plan model of one indoor level (also an outdoor ortho site that has a ``plan_model.json``)."""

    kind = KIND
    supported_tiers: tuple[str, ...] = (TIER_ELEMENTS, TIER_FULL)

    def __init__(self, project: Path | str, level_id: str, *, mode: str | None = None, plan: Any = None) -> None:
        super().__init__(project, level_id, mode=mode)
        self._plan = plan  # optional fixed base plan (tests, a turn with an explicit plan)

    # ---------------------------------------------------------------------------------------------------- status
    @property
    def plan_path(self) -> Path:
        from samcloud.assistant.paths import plan_model_path

        return plan_model_path(self.project, self.level_id)

    def availability(self) -> tuple[bool, str | None]:
        if self._plan is not None or self.plan_path.is_file():
            return True, None
        return False, "assistant.agent.no_plan"

    # ----------------------------------------------------------------------------------------------------- model
    def load_base(self) -> Any:
        if self._plan is not None:
            return self._plan
        from samcloud.bim.plan_model import load_plan_model

        return load_plan_model(self.plan_path)

    def current_model(self) -> Any:
        from samcloud.bim.plan_model import load_plan_model

        return load_plan_model(self.plan_path)

    def _model(self, model: Any) -> Any:
        return self.current_model() if model is None else model

    def new_proposal(self, base: Any) -> Any:
        from samcloud.assistant.edits import Proposal

        return Proposal(base)

    def view_frame(self) -> dict[str, Any]:
        plan = self.current_model()
        upm = float(plan.units_per_metre or 1.0)
        return {"units_per_metre": upm, "floor_z": float(plan.floor_z) / upm}

    # ------------------------------------------------------------------------------------------------- elements
    def element_list(self, filter: Any = None, model: Any = None) -> list[dict[str, Any]]:  # noqa: A002
        from samcloud.assistant.selection import element_list

        return element_list(self._model(model), filter)

    def element_details(self, element_id: str, model: Any = None) -> dict[str, Any]:
        from samcloud.assistant.selection import element_details

        return element_details(self._model(model), element_id)

    # ------------------------------------------------------------------------------------------------ selection
    def hit_test(self, x: float, y: float, model: Any = None) -> dict[str, Any]:
        from samcloud.assistant.selection import hit_test

        return hit_test(self._model(model), float(x), float(y))

    def selection_from_global_id(self, global_id: str, point: list[float] | None = None,
                                 model: Any = None) -> dict[str, Any]:
        from samcloud.assistant.selection import plan_id_for_global_id

        plan = self._model(model)
        found = plan_id_for_global_id(plan, str(global_id)) if global_id else None
        if found is not None:
            return selection_for(plan, found[0], found[1])
        if point is not None and len(point) >= 2:
            return {"type": "point", "x": _r(point[0]), "y": _r(point[1])}
        return {"type": "unknown", "global_id": str(global_id or "")}

    def chip_label(self, sel: dict[str, Any], lang: str | None = None) -> str:
        from samcloud.assistant.selection import chip_label

        return chip_label(sel, lang)

    def outline(self, sel: dict[str, Any], model: Any = None) -> list[list[list[float]]]:
        if not isinstance(sel, dict) or not sel.get("id"):
            return []
        ring = outline_ring(self._model(model), str(sel.get("kind") or ""), str(sel["id"]))
        return [ring] if ring else []

    def z_range(self, sel: dict[str, Any], model: Any = None) -> tuple[float, float] | None:
        if not isinstance(sel, dict) or not sel.get("id"):
            return None
        return height_range(self._model(model), str(sel.get("kind") or ""), str(sel["id"]))

    def global_ids_for(self, selections: list[dict[str, Any]], model: Any = None) -> list[str]:
        from samcloud.assistant.selection import global_id_for

        plan = None
        ids: list[str] = []
        for sel in selections or []:
            if not isinstance(sel, dict) or sel.get("type") != "element" or not sel.get("id"):
                continue
            plan = plan if plan is not None else self._model(model)
            try:
                gid = global_id_for(plan, str(sel.get("kind") or ""), str(sel["id"]))
            except ValueError:
                gid = None
            if gid and gid not in ids:
                ids.append(gid)
        return ids

    def refresh_selection(self, sel: dict[str, Any], model: Any = None) -> dict[str, Any] | None:
        if not isinstance(sel, dict):
            return None
        if sel.get("type") == "point":
            return dict(sel)
        if sel.get("type") == "element" and sel.get("id"):
            updated = selection_for(self._model(model), str(sel.get("kind") or ""), str(sel["id"]))
            return updated if updated.get("type") == "element" else None
        return None

    # ----------------------------------------------------------------------------------------------------- tools
    def read_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        from samcloud.assistant.tools import indoor_read_tools

        return indoor_read_tools()

    def edit_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        from samcloud.assistant.tools import indoor_edit_tools

        tools = indoor_edit_tools()
        if normalize_tier(tier) == TIER_FULL:
            from samcloud.assistant.edits_full import full_edit_tools

            tools.extend(full_edit_tools())
        return tools

    def cloud_loader(self, base: Any) -> Callable[[], Any] | None:
        project, level_id = self.project, self.level_id

        def loader() -> Any:
            from samcloud.assistant.pointcloud import ProjectCloud

            return ProjectCloud.load(project, level_id, plan=base)

        return loader

    # -------------------------------------------------------------------------------------------------------- QA
    def run_qa(self, proposal: Any, baseline: Any, *, work_dir: Path) -> Any:
        from samcloud.assistant.qa import run_qa

        return run_qa(proposal.plan, baseline=baseline, work_dir=work_dir)

    # --------------------------------------------------------------------------------------------------- prompts
    @staticmethod
    def vocabulary_for(lang: str) -> dict[str, Any]:
        """The indoor prompt vocabulary of a language (German / English, else English)."""
        import copy

        code = str(lang or "").lower()[:2]
        return copy.deepcopy(_VOCABULARY[code if code in _VOCABULARY else "en"])

    def vocabulary(self, lang: str) -> dict[str, Any]:
        return self.vocabulary_for(lang)

    def context_message(self, model: Any, selection: list[dict[str, Any]] | None, lang: str | None = None, *,
                        max_elements: int = 80) -> str:
        from samcloud.assistant.prompts import context_message

        return context_message(self._model(model), selection, lang, max_elements=max_elements)

    # ------------------------------------------------------------------------------------------- history / export
    def history_backend(self) -> HistoryBackend:
        from samcloud.assistant.history import indoor_backend

        return indoor_backend()

    @property
    def has_ifc_export(self) -> bool:
        """An indoor level exports DXF and IFC; an outdoor ortho site only its DXF."""
        if self.mode is not None:
            return self.mode == "indoor"
        return self.level_id != OUTDOOR_SITE_LEVEL

    def export_actions(self, after: str = "accept") -> list[str]:
        return ["export", "export_ifc"] if self.has_ifc_export else ["export"]

    def preview(self, proposal: Any) -> dict[str, list[dict[str, Any]]]:
        from samcloud.assistant.selection import global_id_for

        preview: dict[str, list[dict[str, Any]]] = {"added": [], "removed": [], "moved": []}
        if proposal is None:
            return preview
        for entry in proposal.summary():
            element_id, kind, change = entry["element_id"], entry["kind"], entry["change"]
            source = proposal.base if change == "removed" else proposal.plan
            try:
                gid = global_id_for(source, kind, element_id)
            except Exception:  # noqa: BLE001 - informative only
                gid = None
            item = {"id": element_id, "kind": kind, "change": change, "global_id": gid,
                    "before": None if change == "added" else outline_ring(proposal.base, kind, element_id),
                    "after": None if change == "removed" else outline_ring(proposal.plan, kind, element_id)}
            try:
                heights = height_range(source, kind, element_id)
            except Exception:  # noqa: BLE001 - informative only
                heights = None
            item["z_min"], item["z_max"] = heights if heights is not None else (None, None)
            bucket = "added" if change == "added" else "removed" if change == "removed" else "moved"
            preview[bucket].append(item)
        return preview


def matches(project: Path, level_id: str, mode: str | None) -> bool:
    """Registry matcher: the level has a ``plan_model.json``."""
    from samcloud.assistant.paths import plan_model_path

    try:
        return plan_model_path(project, level_id).is_file()
    except Exception:  # noqa: BLE001 - an unusable project is not an indoor project
        return False


def factory(project: Path, level_id: str, mode: str | None) -> IndoorDomain:
    return IndoorDomain(project, level_id, mode=mode)


__all__ = ["IndoorDomain", "factory", "height_range", "matches", "outline_ring", "selection_for"]
