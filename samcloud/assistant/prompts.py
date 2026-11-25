"""Prompts of the AI assistant (German / English): the fixed SAMcloud rules and the per-turn context message.

All prompt texts live in this module only.  The rules are also enforced in code (``samcloud.assistant.tools`` and
``samcloud.assistant.edits``); the prompt only tells the model how to behave so it does not fight them.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from samcloud.bim.plan_model import BuildingPlan

LOGGER = logging.getLogger(__name__)

MAX_CONTEXT_ELEMENTS = 80  # elements listed in the context message (the model can call list_elements for more)
MAX_SELECTION = 20

_SYSTEM = {
    "de": """Du bist der KI-Assistent von SAMcloud. Du verbesserst ausschliesslich das Planmodell (DXF/IFC) DIESES Projekts.
Du antwortest auf Deutsch, kurz und sachlich.

Regeln (SAMcloud prüft sie zusätzlich im Code):
1. Keine Evidenz = nichts. Bevor du etwas hinzufügst, schau mit den Mess-Werkzeugen (points_in_box, section, fit_surface) in der
   Punktwolke nach. Findest du dort keine Punkte, sag es und frag nach, z. B.: "Dort ist in der Punktwolke kein Stuhl. Soll ich
   trotzdem einen setzen?" Setze manual=true nur, wenn der Benutzer das ausdrücklich bestätigt hat.
2. Gemessen, nicht geschätzt. Masse kommen aus den Mess-Werkzeugen oder aus dem Satz des Benutzers, nie aus Vermutungen.
3. Ändere nur die Elemente, die der Benutzer meint (Auswahl im Kontext, sonst nachfragen). Nur die Bearbeitungs-Werkzeuge ändern
   etwas; es gibt keinen Code, keine Shell und keinen Dateizugriff.
4. Abhängigkeiten (Räume, Öffnungen, Wandanschlüsse) rechnet SAMcloud selbst. Du musst sie nicht nachführen.
5. Deine Änderungen sind nur ein Vorschlag: Der Benutzer übernimmt oder verwirft ihn. Behaupte nie, Dateien geändert zu haben.
6. Einheiten: Meter, Planmeter für Positionen (x/y), Höhen über dem Boden. Grad gegen den Uhrzeigersinn.
7. Schlägt ein Werkzeug fehl, lies den Fehler und korrigiere, oder erkläre kurz, warum es nicht geht.

Am Ende beschreibst du den Vorschlag in ein bis drei Sätzen (was vorher, was nachher) oder stellst deine Rückfrage.
Formuliere immer als Vorschlag, nie als erledigt: z. B. "Vorschlag: Tür O008 von 0.90 m auf 0.80 m verschmälern. Nach
Übernehmen wird sie 0.80 m breit." Schreibe nie "wurde geändert", "ist jetzt" oder "habe geändert".""",
    "en": """You are the AI assistant of SAMcloud. You only improve the plan model (DXF/IFC) of THIS project.
Answer in English, short and factual.

Rules (SAMcloud also enforces them in code):
1. No evidence = nothing. Before you add anything, look in the point cloud with the measuring tools (points_in_box, section,
   fit_surface). If there are no points, say so and ask, e.g.: "There is no chair in the point cloud there. Shall I place one
   anyway?" Use manual=true only after the user explicitly confirmed it.
2. Measured, not estimated. Dimensions come from the measuring tools or from the user's sentence, never from guesses.
3. Only change the elements the user means (the selection in the context, otherwise ask). Only the edit tools change anything;
   there is no code, no shell and no file access.
4. Dependencies (rooms, openings, wall joins) are computed by SAMcloud. You do not need to update them.
5. Your changes are only a proposal: the user accepts or discards it. Never claim to have changed files.
6. Units: metres, plan metres for positions (x/y), heights above the floor. Degrees counter-clockwise.
7. When a tool fails, read the error and correct the call, or briefly explain why it is not possible.

At the end describe the proposal in one to three sentences (before and after) or ask your question.
Always phrase it as a proposal, never as done, e.g. "Proposal: narrow door O008 from 0.90 m to 0.80 m. After you accept it,
it will be 0.80 m wide." Never write "was changed", "is now" or "I changed".""",
}

# Generic frame of the system prompt; the domain vocabulary fills the project specifics and the tool tier picks the
# proposal or full-control wording.  For the indoor domain on the "elements" tier the result equals _SYSTEM exactly.
_FRAME = {
    "de": "Du bist der KI-Assistent von SAMcloud. Du verbesserst ausschliesslich {model_phrase} DIESES Projekts.\n"
          "Du antwortest auf Deutsch, kurz und sachlich.\n\nRegeln (SAMcloud prüft sie zusätzlich im Code):\n{rules}\n\n{closing}",
    "en": "You are the AI assistant of SAMcloud. You only improve {model_phrase} of THIS project.\n"
          "Answer in English, short and factual.\n\nRules (SAMcloud also enforces them in code):\n{rules}\n\n{closing}",
}
_TIER_RULE = {
    "elements": {
        "de": "Deine Änderungen sind nur ein Vorschlag: Der Benutzer übernimmt oder verwirft ihn. Behaupte nie, Dateien "
              "geändert zu haben.",
        "en": "Your changes are only a proposal: the user accepts or discards it. Never claim to have changed files.",
    },
    "full": {
        "de": "Vollkontrolle aktiv: Deine Änderung wird nach bestandener SAMcloud-QA direkt angewendet (ohne Übernehmen-Klick)\n"
              "   und landet im Verlauf (Rückgängig möglich). Du darfst alle Ergebnisse dieses Projekts ändern, auch viele\n"
              "   Elemente auf einmal (Sammel-Werkzeuge), nie Code, Einstellungen oder andere Projekte. Bei QA-Befunden wird nichts\n"
              "   angewendet: korrigiere es oder melde es. Behaupte nie, etwas sei angewendet, was nicht angewendet wurde.\n"
              "   Ändere nur, was der Benutzer verlangt; ohne Evidenz nur nach ausdrücklicher Bestätigung im Chat.",
        "en": "Full control active: your change is applied directly after a passed SAMcloud QA (no accept click) and is listed\n"
              "   in the history (undo possible). You may change all results of this project, also many elements at once\n"
              "   (bulk tools), never code, settings or other projects. With QA findings nothing is applied: correct it or\n"
              "   report it. Never claim that something was applied when it was not. Only change what the user asks for;\n"
              "   without evidence only after an explicit confirmation in the chat.",
    },
}
_CLOSING = {
    "elements": {
        "de": "Am Ende beschreibst du den Vorschlag in ein bis drei Sätzen (was vorher, was nachher) oder stellst deine "
              "Rückfrage.\nFormuliere immer als Vorschlag, nie als erledigt. Schreibe nie \"wurde geändert\", \"ist jetzt\" "
              "oder \"habe geändert\".",
        "en": "At the end describe the proposal in one to three sentences (before and after) or ask your question.\n"
              "Always phrase it as a proposal, never as done. Never write \"was changed\", \"is now\" or \"I changed\".",
    },
    "full": {
        "de": "Am Ende beschreibst du in ein bis drei Sätzen, was nach bestandener QA direkt angewendet wird (was vorher, was "
              "nachher), oder stellst deine Rückfrage. Behaupte nie, etwas sei angewendet, wenn kein Bearbeitungs-Werkzeug "
              "erfolgreich war.",
        "en": "At the end describe in one to three sentences what will be applied directly after a passed QA (before and after) "
              "or ask your question. Never claim that anything is applied when no edit tool succeeded.",
    },
}
_PROJECT_TYPE = {"de": "Projektart: {project_type}. Elementtypen: {element_types}.",
                 "en": "Project type: {project_type}. Element types: {element_types}."}

_QA_FEEDBACK = {
    "de": "Die SAMcloud-QA hat im Vorschlag neue Befunde gefunden:\n{findings}\nKorrigiere das mit den Werkzeugen oder erkläre, "
          "warum es nicht geht.",
    "en": "The SAMcloud QA found new problems in the proposal:\n{findings}\nFix it with the tools or explain why it is not possible.",
}

_IMAGES_NOTE = {
    "de": "Schnittbild(er) zu den Werkzeug-Ergebnissen oben (dunkel = belegt, Norden oben).",
    "en": "Section image(s) for the tool results above (dark = occupied, north up).",
}

_LABELS = {
    "de": {"intro": "Kontext (SAMcloud, Planmeter):", "selection": "Auswahl des Benutzers", "none": "keine",
           "elements": "Elemente", "more": "weitere: list_elements verwenden", "units": "Einheiten",
           "units_text": "Meter; x/y = Planmeter; Höhen über Boden", "storey": "Geschosshöhe (Boden bis Decke)",
           "level": "Geschoss"},
    "en": {"intro": "Context (SAMcloud, plan metres):", "selection": "User selection", "none": "none",
           "elements": "Elements", "more": "more: use list_elements", "units": "Units",
           "units_text": "metres; x/y = plan metres; heights above the floor", "storey": "Storey height (floor to ceiling)",
           "level": "Level"},
}


def _lang(lang: str | None) -> str:
    code = str(lang or "").lower()[:2]
    if code in _SYSTEM:
        return code
    try:
        from samcloud.app.i18n import current_language

        code = current_language()
    except Exception:  # noqa: BLE001 - prompts work without the UI
        code = "en"
    return code if code in _SYSTEM else "en"


def _tier(tier: Any) -> str:
    return "full" if str(tier or "").strip().lower() == "full" else "elements"


def tier_rule(tier: str | None, lang: str | None = None) -> str:
    """The rule that tells the model whether its changes are a proposal (``"elements"``) or applied (``"full"``)."""
    return _TIER_RULE[_tier(tier)][_lang(lang)]


def system_prompt(lang: str | None = None, *, vocabulary: dict[str, Any] | None = None, tier: str | None = None) -> str:
    """The SAMcloud rules in German or English (reply language = UI language when ``lang`` is None).

    Without ``vocabulary`` (and on the ``"elements"`` tier) this is the indoor phase 1 prompt.  ``vocabulary`` is the
    domain's :meth:`~samcloud.assistant.domains.base.AssistantDomain.vocabulary`: ``model_phrase``, ``rules`` (the entry
    ``"{tier}"`` marks the tier rule; without it the tier rule goes last), optional ``closing`` {tier: text},
    ``project_type`` and ``element_types`` (named at the end on the ``"full"`` tier, or always when the vocabulary sets
    ``announce_project``).  ``tier`` ``"full"`` uses the full-control wording.
    """
    language = _lang(lang)
    level = _tier(tier)
    if vocabulary is None:
        if level == "elements":
            return _SYSTEM[language]
        from samcloud.assistant.domains.indoor import IndoorDomain

        vocabulary = IndoorDomain.vocabulary_for(language)
    rules = [str(rule) for rule in vocabulary.get("rules") or []]
    if "{tier}" not in rules:
        rules.append("{tier}")
    rule_text = "\n".join(f"{index}. {tier_rule(level, language) if rule == '{tier}' else rule}"
                           for index, rule in enumerate(rules, start=1))
    closing = dict(vocabulary.get("closing") or {}).get(level) or _CLOSING[level][language]
    text = _FRAME[language].format(model_phrase=vocabulary.get("model_phrase") or "", rules=rule_text, closing=closing)
    if vocabulary.get("project_type") and bool(vocabulary.get("announce_project", level == "full")):
        names = vocabulary.get("element_types") or {}
        listed = ", ".join(str(name) for name in (names.values() if isinstance(names, dict) else names))
        text = f"{text}\n\n" + _PROJECT_TYPE[language].format(project_type=vocabulary["project_type"], element_types=listed or "-")
    return text


def qa_feedback(findings: list[dict[str, Any]], lang: str | None = None) -> str:
    """The one QA correction message sent back to the model."""
    return _QA_FEEDBACK[_lang(lang)].format(findings=format_findings(findings))


def images_note(lang: str | None = None) -> str:
    return _IMAGES_NOTE[_lang(lang)]


def format_findings(findings: list[dict[str, Any]], limit: int = 8) -> str:
    """Plain lines ``- check subject: detail`` of QA findings (bounded)."""
    lines = []
    for item in list(findings)[:limit]:
        detail = item.get("detail")
        if isinstance(detail, (dict, list)):
            detail = json.dumps(detail, ensure_ascii=False, separators=(",", ":"))
        text = f"- {item.get('check')} {item.get('subject')}"
        if detail not in (None, ""):
            text += f": {str(detail)[:160]}"
        lines.append(text)
    if len(findings) > limit:
        lines.append(f"- ... +{len(findings) - limit}")
    return "\n".join(lines)


def _selection_rows(plan: BuildingPlan | None, selection: list[dict[str, Any]], lang: str) -> list[dict[str, Any]]:
    from samcloud.assistant.selection import chip_label

    rows = []
    for sel in list(selection or [])[:MAX_SELECTION]:
        if not isinstance(sel, dict):
            continue
        if sel.get("type") == "point":
            try:
                rows.append({"point": [round(float(sel.get("x")), 3), round(float(sel.get("y")), 3)], "chip": chip_label(sel, lang)})
            except (TypeError, ValueError):
                continue
            continue
        row: dict[str, Any] = {"id": sel.get("id"), "kind": sel.get("kind"), "chip": chip_label(sel, lang)}
        if sel.get("subkind"):
            row["type"] = sel.get("subkind")
        if plan is not None and sel.get("id"):
            from samcloud.assistant.edits import find_element, snapshot

            found = find_element(plan, str(sel["id"]))
            if found is None:
                row["missing"] = True
            else:
                row["now"] = snapshot(plan, found[0], found[1])
        rows.append(row)
    return rows


def _anchors(rows: list[dict[str, Any]], selection: list[dict[str, Any]]) -> tuple[set[str], list[tuple[float, float]]]:
    """Ids and plan positions of the user's selection (selected elements and clicked points)."""
    ids = {str(sel.get("id")) for sel in selection if isinstance(sel, dict) and sel.get("id")}
    points: list[tuple[float, float]] = []
    for sel in selection:
        if isinstance(sel, dict) and sel.get("type") == "point":
            try:
                points.append((float(sel.get("x")), float(sel.get("y"))))
            except (TypeError, ValueError):
                continue
    for row in rows:
        position = row.get("position")
        if str(row.get("id")) in ids and isinstance(position, (list, tuple)) and len(position) >= 2:
            try:
                points.append((float(position[0]), float(position[1])))
            except (TypeError, ValueError):
                continue
    return ids, points


def ordered_elements(rows: list[dict[str, Any]], selection: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """``rows`` with the selected elements first, then the elements nearest to the selection (stable otherwise)."""
    ids, points = _anchors(rows, list(selection or []))
    if not ids and not points:
        return list(rows)

    def key(item: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
        index, row = item
        if str(row.get("id")) in ids:
            return (0, 0.0, index)
        position = row.get("position")
        if not points or not isinstance(position, (list, tuple)) or len(position) < 2:
            return (2, 0.0, index)
        try:
            x, y = float(position[0]), float(position[1])
        except (TypeError, ValueError):
            return (2, 0.0, index)
        return (1, min((x - px) ** 2 + (y - py) ** 2 for px, py in points), index)

    return [row for _, row in sorted(enumerate(rows), key=key)]


def context_message(plan: BuildingPlan, selection: list[dict[str, Any]] | None, lang: str | None = None, *,
                    max_elements: int = MAX_CONTEXT_ELEMENTS) -> str:
    """Selection chips, a bounded compact element list, units and floor / ceiling heights of the level.

    The element list holds at most ``max_elements`` rows: the selected elements first, then those nearest to the selection
    (the agent lowers ``max_elements`` to keep the prompt inside the model's context window).
    """
    from samcloud.assistant.selection import element_list
    from samcloud.assistant.tools import compact

    language = _lang(lang)
    labels = _LABELS[language]
    cap = max(0, int(max_elements))
    upm = plan.units_per_metre or 1.0
    storey = None
    if plan.ceiling_z is not None and plan.ceiling_z > plan.floor_z:
        storey = round((float(plan.ceiling_z) - float(plan.floor_z)) / upm, 3)
    rows = ordered_elements(element_list(plan), selection)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("type"))] = counts.get(str(row.get("type")), 0) + 1
    selected = _selection_rows(plan, list(selection or []), language)
    payload: dict[str, Any] = {
        labels["level"]: plan.level_name or plan.level_id,
        labels["units"]: labels["units_text"],
        labels["storey"]: storey,
        labels["selection"]: selected or labels["none"],
        "counts": counts,
        labels["elements"]: rows[:cap],
    }
    if len(rows) > cap:
        payload[labels["more"]] = len(rows) - cap
    text = json.dumps(compact(payload, max_list=max(cap, MAX_SELECTION) + 1), ensure_ascii=False, separators=(",", ":"))
    return f"{labels['intro']}\n{text}"


__all__ = ["context_message", "format_findings", "images_note", "ordered_elements", "qa_feedback", "system_prompt", "tier_rule"]
