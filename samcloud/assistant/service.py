"""Service API of the AI assistant for the UI (package D only calls this module).

One :class:`AssistantService` per (project, level).  The pending proposal is kept per (project, level) at module
level, so a new service instance (e.g. another page or a reload) sees the same proposal.  ``ask`` may run while a
DXF/IFC job runs; ``accept`` / ``undo`` raise :class:`~samcloud.assistant.history.HistoryError` with the key
``assistant.history.locked`` right away when the history lock is held (e.g. by the job runner re-applying edits).
Re-exporting DXF/IFC after ``accept`` / ``undo`` is done by the UI through the existing job runner.

The project type is a domain (:func:`samcloud.assistant.domains.registry.domain_for`): availability, selection, outlines,
preview, tools, QA and the export jobs come from it, so indoor and outdoor levels plug in their own model.  Without a
matching domain the indoor plan helpers are used (phase 1 behaviour) and :attr:`AssistantService.unsupported_reason` says
why the level cannot be edited.

Writes: ``ask`` appends to ``chat.json`` and, when ``edits.json`` exists, may run
:func:`~samcloud.assistant.history.ensure_applied` (the pipeline re-apply hook, which can rewrite ``plan_model.json``
after a recompute) before the turn.  :func:`~samcloud.assistant.agent.run_turn` itself never writes the plan;
only ``accept`` and ``undo`` change ``plan_model.json`` / ``edits.json`` -- and, with full control (resolved tool tier
``"full"`` of any domain), ``ask`` itself: a proposal that passed the domain's QA is stored in the history right away
(``TurnResult.applied`` / ``edit_id``; undo as usual); the UI then starts the same exports as after an accept.
"""

from __future__ import annotations

import copy
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from samcloud.app.i18n import current_language, t, translate_or
from samcloud.assistant.agent import TurnResult, run_turn
from samcloud.assistant.chat_store import ChatStore
from samcloud.assistant.domains.base import TIER_ELEMENTS, TIER_FULL, AssistantDomain, UnavailableDomain
from samcloud.assistant.history import History, HistoryError, ensure_applied, history_locked

LOGGER = logging.getLogger(__name__)

HISTORY_FOR_MODEL = 6  # previous chat messages sent with a new question


@dataclass
class _Pending:
    result: TurnResult
    prompt: str
    provider: str
    model: str


_PENDING: dict[tuple[str, str], _Pending] = {}
_ASK_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_STATE_LOCK = threading.Lock()


def _key(project: Path, level_id: str) -> tuple[str, str]:
    try:
        resolved = str(project.resolve())
    except OSError:
        resolved = str(project)
    return resolved, str(level_id)


def _r(value: float) -> float:
    return float(round(float(value), 3))


# indoor plan helpers (kept under their phase 1 names; the implementation lives in the indoor domain)
def _outline(plan: Any, kind: str, element_id: str) -> Any:
    from samcloud.assistant.domains.indoor import outline_ring

    return outline_ring(plan, kind, element_id)


def _z_range(plan: Any, kind: str, element_id: str) -> Any:
    from samcloud.assistant.domains.indoor import height_range

    return height_range(plan, kind, element_id)


def _selection_for(plan: Any, kind: str, plan_id: str) -> dict[str, Any]:
    from samcloud.assistant.domains.indoor import selection_for

    return selection_for(plan, kind, plan_id)


def _current_model(domain: AssistantDomain) -> Any:
    """The domain's model on disk; None for a domain that has none (an unavailable project type)."""
    if isinstance(domain, UnavailableDomain):
        return None
    return domain.current_model()


class _NoDomain(UnavailableDomain):
    """Stand-in for a level no domain can edit and that is no indoor plan (e.g. outdoor without environment chain):
    every query is empty, nothing falls back to the indoor plan."""

    def __init__(self, project: Path | str, level_id: str, *, mode: str | None = None, reason: str | None = None) -> None:
        super().__init__(project, level_id, mode=mode)
        if reason:
            self.reason_key = reason


class AssistantService:
    """Everything the UI needs from the assistant for one project level."""

    def __init__(self, project: Path | str, level_id: str, *, mode: str | None = None) -> None:
        self.project = Path(project)
        self.level_id = str(level_id)
        self.mode = mode  # project mode of the page ("indoor" / "outdoor"), None = derived from the level
        self._key = _key(self.project, self.level_id)
        self.last_accept: Any = None  # history.AcceptReport of the last accept of this instance
        self._domain: AssistantDomain | None = None
        self._domain_resolved = False

    # ---------------------------------------------------------------------------------------------------- domain
    def _resolve_domain(self) -> AssistantDomain | None:
        from samcloud.assistant.domains.registry import domain_for

        try:
            self._domain = domain_for(self.project, self.level_id, mode=self.mode)
        except Exception as error:  # noqa: BLE001 - a broken registry means "no domain"
            LOGGER.warning("assistant: project type not recognised (%s)", type(error).__name__)
            self._domain = None
        self._domain_resolved = True
        return self._domain

    @property
    def domain(self) -> AssistantDomain | None:
        """The domain of this level (looked up again by :meth:`is_available`), None when the project is not supported."""
        if not self._domain_resolved:
            self._resolve_domain()
        return self._domain

    def _work_domain(self) -> AssistantDomain:
        """The domain for selection / preview / history: the level's domain when it is available; otherwise the indoor
        plan (phase 1 behaviour) only when the registry says indoor / no plan -- an outdoor level (environment chain, or
        an outdoor project without it) never falls back to the indoor plan model."""
        from samcloud.assistant.domains.registry import REASON_NO_PLAN, reason_without_domain

        domain = self.domain
        if domain is not None and domain.availability()[0]:
            return domain
        if domain is not None and str(domain.kind) != "indoor":
            return domain  # e.g. the outdoor domain while it cannot edit this level: empty answers, no indoor fallback
        if domain is None and reason_without_domain(self.project, self.level_id, mode=self.mode) != REASON_NO_PLAN:
            return _NoDomain(self.project, self.level_id, mode=self.mode,
                             reason=reason_without_domain(self.project, self.level_id, mode=self.mode))
        from samcloud.assistant.domains.indoor import IndoorDomain

        return IndoorDomain(self.project, self.level_id, mode=self.mode)

    @property
    def domain_kind(self) -> str | None:
        """``"indoor"`` / ``"outdoor"`` / ... of the level's domain, None when no domain matches."""
        domain = self.domain
        return None if domain is None else str(domain.kind)

    @property
    def unsupported_reason(self) -> str | None:
        """i18n key why the project level cannot be edited (no domain or domain unavailable), else None."""
        from samcloud.assistant.domains.registry import reason_without_domain

        domain = self._resolve_domain()
        if domain is None:
            return reason_without_domain(self.project, self.level_id, mode=self.mode)
        available, reason = domain.availability()
        return None if available else (reason or "assistant.agent.no_plan")

    @property
    def tool_tier(self) -> str:
        """The tool tier the next turn uses (``"elements"`` / ``"full"``): full control from the settings, if the domain
        supports it."""
        from samcloud.assistant.domains.registry import resolve_tier
        from samcloud.assistant.settings import load_settings

        try:
            settings = load_settings()
        except Exception:  # noqa: BLE001 - unreadable settings = default tier
            return TIER_ELEMENTS
        return resolve_tier(self.domain, settings)

    def export_actions(self, after: str = "accept") -> list[str]:
        """Page job actions that update DXF / IFC after ``"accept"`` / ``"undo"`` (started with ``assistant=True``)."""
        return list(self._work_domain().export_actions(after))

    def view_frame(self) -> dict[str, Any]:
        """``{"units_per_metre", "floor_z", ...}`` of the level's model for the 2D canvas and the IFC viewer; extra keys
        of the domain (e.g. ``origin_offset`` / ``crs`` outdoors) are passed through unchanged."""
        frame = dict(self._work_domain().view_frame() or {})
        frame.setdefault("units_per_metre", 1.0)
        frame.setdefault("floor_z", None)
        return frame

    @property
    def last_accept_report(self) -> Any:
        """``AcceptReport(edit_id, rebased {old id: new id}, notes)`` of the last ``accept`` of this instance (None before);
        a non-empty ``rebased`` means the proposal was re-applied on a recomputed plan with new ids (e.g. O002 -> O001)."""
        return self.last_accept

    # ---------------------------------------------------------------------------------------------------- status
    def is_available(self) -> tuple[bool, str | None]:
        """(True, None) when enabled, configured, a model is chosen and the level's domain can edit it (indoor: the
        level has a plan_model); else the reason key (settings reasons first, then the domain's)."""
        from samcloud.assistant.credentials import load_ai_credentials
        from samcloud.assistant.settings import is_configured, load_settings

        try:
            settings = load_settings()
            creds = load_ai_credentials()
        except Exception as error:  # noqa: BLE001 - broken settings = not available
            LOGGER.info("assistant: settings unreadable (%s)", type(error).__name__)
            return False, "assistant.agent.not_configured"
        if not settings.enabled:
            return False, "assistant.agent.disabled"
        if not is_configured(settings, creds):
            return False, "assistant.agent.not_configured"
        if not settings.model.strip():
            return False, "assistant.agent.no_model"
        reason = self.unsupported_reason
        if reason is not None:
            return False, reason
        return True, None

    def model_path(self) -> Path | None:
        """The level's model file (indoor ``plan_model.json``, outdoor the domain's model), None when the level has no
        editable domain."""
        try:
            backend = self._work_domain().history_backend()
        except NotImplementedError:
            return None
        return Path(backend.model_path(self.project, self.level_id))

    def model(self) -> Any:
        """The level's model on disk (with the accepted edits), through its domain."""
        return self._work_domain().current_model()

    @property
    def plan_path(self) -> Path:
        """Indoor-compatible alias: ``plan_model.json`` of the level (see :meth:`model_path` for any domain)."""
        from samcloud.assistant.paths import plan_model_path

        return plan_model_path(self.project, self.level_id)

    def plan(self) -> Any:
        """Indoor-compatible alias: the plan on disk (with the accepted edits); see :meth:`model` for any domain."""
        from samcloud.bim.plan_model import load_plan_model

        return load_plan_model(self.plan_path)

    # -------------------------------------------------------------------------------------------------- selection
    def hit_test(self, x: float, y: float) -> dict[str, Any]:
        """Element under a click at plan metres (x, y), else ``{"type": "point", "x", "y"}``."""
        return self._work_domain().hit_test(float(x), float(y))

    def selection_from_global_id(self, global_id: str, point: list[float] | None = None) -> dict[str, Any]:
        """Selection of an IFC pick: the plan element of ``global_id``, else a point (plan metres) when given, else
        ``{"type": "unknown", "global_id": ...}``."""
        return self._work_domain().selection_from_global_id(str(global_id or ""), point)

    def chip_label(self, sel: dict[str, Any]) -> str:
        return self._work_domain().chip_label(sel, current_language())

    def global_ids_for(self, selections: list[dict[str, Any]]) -> list[str]:
        """IFC GlobalIds of the selected elements (points and unknown ids are skipped), for the IFC highlight."""
        return self._work_domain().global_ids_for(list(selections or []))

    def outlines_for(self, selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """2D highlight shapes of selections in plan metres for the DXF overlay: ``{"id", "ring"}`` for elements with an
        outline (elements that no longer exist are skipped) and ``{"point": [x, y]}`` for point selections."""
        items: list[dict[str, Any]] = []
        domain = None
        model = None
        for sel in selections or []:
            if not isinstance(sel, dict):
                continue
            if sel.get("type") == "point":
                try:
                    items.append({"point": [_r(sel["x"]), _r(sel["y"])]})
                except (KeyError, TypeError, ValueError):
                    continue
            elif sel.get("type") == "element" and sel.get("id"):
                if domain is None:
                    domain = self._work_domain()
                    model = _current_model(domain)
                for ring in domain.outline(sel, model):
                    if ring:
                        items.append({"id": str(sel["id"]), "ring": ring})
        return items

    def refresh_selections(self, selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Selections re-read from the current plan (new labels / widths after an accept or undo); elements that no
        longer exist are dropped, point selections are kept."""
        domain = None
        model = None
        fresh: list[dict[str, Any]] = []
        for sel in selections or []:
            if not isinstance(sel, dict):
                continue
            if sel.get("type") == "element" and sel.get("id"):
                if domain is None:
                    domain = self._work_domain()
                    model = _current_model(domain)
                updated = domain.refresh_selection(sel, model)
                if updated is not None and updated.get("type") == "element":
                    fresh.append(updated)
            elif sel.get("type") == "point":
                fresh.append(dict(sel))
        return fresh

    # -------------------------------------------------------------------------------------------------------- ask
    def _ask_lock(self) -> threading.Lock:
        with _STATE_LOCK:
            return _ASK_LOCKS.setdefault(self._key, threading.Lock())

    def ask(self, prompt: str, selection: list[dict[str, Any]], *, allow_manual: bool = False) -> TurnResult:
        """Run one assistant turn; the question and the answer are appended to chat.json.  A new proposal replaces the
        pending one; a turn without proposal keeps nothing pending.

        Before the turn, when ``edits.json`` exists, the accepted edits are re-applied if the pipeline recomputed the plan
        (``ensure_applied``, may write ``plan_model.json``); the turn itself (``run_turn``) never writes the plan.
        Returns ``error_kind="busy"`` (pending unchanged) while another question of this level is running."""
        from samcloud.assistant.credentials import load_ai_credentials, redact
        from samcloud.assistant.settings import load_settings

        prompt = redact(str(prompt or ""))
        available, reason = self.is_available()
        if not available:
            result = TurnResult(error=t(reason or "assistant.agent.not_configured"), error_kind="unavailable")
            result.reply = result.error or ""
            return result
        lock = self._ask_lock()
        if not lock.acquire(blocking=False):
            result = TurnResult(error=t("assistant.agent.busy"), error_kind="busy")
            result.reply = result.error or ""
            return result
        try:
            settings = load_settings()
            creds = load_ai_credentials()
            store = ChatStore(self.project, self.level_id)
            previous = store.load()[-HISTORY_FOR_MODEL:]
            domain = self._work_domain()
            try:  # the pipeline may have recomputed the plan: re-apply the accepted edits first (pipeline hook)
                if not history_locked(self.project, self.level_id):  # a running job re-applies them itself
                    ensure_applied(self.project, self.level_id, backend=domain.history_backend())
            except HistoryError as error:
                LOGGER.info("assistant: edits not re-applied before the question (%s)", error.key)
            selection = [dict(sel) for sel in (selection or []) if isinstance(sel, dict)]
            store.append("user", prompt, {"selection": [self.chip_label(sel) for sel in selection],
                                          "provider": settings.provider, "model": settings.model})
            from samcloud.assistant.domains.registry import resolve_tier

            tier = resolve_tier(domain, settings)
            result = run_turn(project=self.project, level_id=self.level_id, prompt=prompt, selection=selection,
                              settings=settings, creds=creds, allow_manual=allow_manual, history_messages=previous,
                              domain=domain, tier=tier)
            result.tier = tier
            if tier == TIER_FULL:
                self._apply_directly(result, prompt, domain)
            meta = {
                "tools": result.tool_log, "cost_usd": result.cost_usd, "provider": result.provider, "model": result.model,
                "input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens,
                "error_kind": result.error_kind, "changes": len(result.summary),
                "qa_ok": None if result.qa is None else result.qa.ok, "needs_confirmation": result.needs_confirmation,
                "asks_manual": result.asks_manual,
            }
            if tier == TIER_FULL:
                meta.update({"tier": tier, "applied": result.applied, "edit_id": result.edit_id})
            store.append("assistant", result.reply or result.error or "", meta)
            with _STATE_LOCK:
                if result.proposal is not None and not result.applied:
                    _PENDING[self._key] = _Pending(result, prompt, result.provider, result.model)
                else:
                    _PENDING.pop(self._key, None)
            return result
        finally:
            lock.release()

    def _apply_directly(self, result: TurnResult, prompt: str, domain: AssistantDomain) -> None:
        """Full control: a proposal that passed the domain's QA is stored in the history right away (exactly like an
        accept, without the confirmation for deletions / many elements).  A failing QA never reaches this point (the
        turn discards the proposal); when the history refuses (locked, outdated ...) the proposal stays pending and a
        note says why, so nothing is claimed as applied that was not."""
        if result.error is not None or result.proposal is None or not result.summary:
            return
        if result.qa is not None and not result.qa.ok:
            return
        backend = domain.history_backend()
        try:
            unchanged = backend.semantic_hash(result.proposal.plan) == backend.semantic_hash(result.proposal.base)
        except Exception:  # noqa: BLE001 - a model that cannot be hashed is treated as changed
            unchanged = False
        # every element reads the same before and after (e.g. a wall set to the thickness it already has; derived data
        # such as body rings may be rebuilt with float noise)
        unchanged = unchanged or all(entry.get("change") in ("modified", "moved") and entry.get("before") == entry.get("after")
                                     for entry in result.summary)
        if unchanged:  # e.g. the same change asked twice: nothing to apply, no history entry
            result.proposal = None
            result.summary = []
            result.needs_confirmation = False
            result.notes = [*result.notes, t("assistant.agent.no_change")]
            return
        try:
            report = History.load(self.project, self.level_id, backend).accept_with_report(
                result.proposal, prompt=prompt, provider=result.provider, model=result.model)
        except HistoryError as error:
            LOGGER.info("assistant: full-control change not applied (%s)", error.key)
            result.notes = [*result.notes, t("assistant.agent.apply_failed", reason=translate_or(error.key, error.message, **dict(error.params or {})))]
            return
        self.last_accept = report
        result.applied = True
        result.edit_id = report.edit_id
        result.needs_confirmation = False
        rebased = dict(report.rebased or {})
        if rebased:
            result.notes = [*result.notes, *[str(note) for note in report.notes]]

    # ---------------------------------------------------------------------------------------------------- pending
    def pending(self) -> TurnResult | None:
        """The last proposal awaiting accept / discard."""
        with _STATE_LOCK:
            entry = _PENDING.get(self._key)
        return None if entry is None else entry.result

    def preview(self) -> dict[str, Any]:
        """Viewer colours of the pending proposal: ``added`` (green), ``removed`` (red), ``moved`` (yellow: moved and
        modified elements).  Each entry: ``id``, ``kind``, ``change``, ``global_id``, ``before`` / ``after`` (closed 2D
        rings in plan metres, None when the element does not exist on that side) and ``z_min`` / ``z_max`` (absolute
        height range in metres of the element after the change, before it for removals; None for rooms)."""
        result = self.pending()
        if result is None or result.proposal is None:
            return {"added": [], "removed": [], "moved": []}
        return self._work_domain().preview(result.proposal)

    # ------------------------------------------------------------------------------------------------ accept / undo
    def _check_lock(self) -> None:
        """Raise ``assistant.history.locked`` at once when another writer holds the history lock (fast pre-check; the
        ``assistant.history.locked`` error that ``History.accept`` / ``undo`` raise themselves stays authoritative)."""
        if history_locked(self.project, self.level_id):
            raise HistoryError("Another assistant edit is being saved; try again.", key="assistant.history.locked")

    def accept(self, *, confirmed: bool = False) -> str:
        """Store the pending proposal in edits.json and write the edited plan_model.json; returns the edit id.

        A proposal made on an older plan (another accept, an undo or a pipeline recompute since) is re-applied by
        signature onto the current plan by ``History.accept``; the ids that changed on the way are in
        :attr:`last_accept` (``AcceptReport.rebased`` / ``notes``).  When that is not possible ``assistant.history.outdated``
        (or ``ambiguous``) is raised and the proposal stays pending (the UI should discard it and ask again).

        Raises :class:`HistoryError` (``assistant.agent.no_pending``, ``assistant.agent.confirm_required`` when the
        proposal needs a confirmation and ``confirmed`` is False, ``assistant.history.locked`` / ``outdated`` ...)."""
        with _STATE_LOCK:
            entry = _PENDING.get(self._key)
        if entry is None or entry.result.proposal is None:
            raise HistoryError("There is no proposal to accept.", key="assistant.agent.no_pending")
        proposal = entry.result.proposal
        if proposal.needs_confirmation() and not confirmed:
            raise HistoryError("This proposal deletes elements or changes more than 5; confirm it first.",
                               key="assistant.agent.confirm_required", params={"count": len(proposal.summary())})
        self._check_lock()
        # a HistoryError from History.accept (locked, outdated, ...) propagates and the proposal stays pending
        report = History.load(self.project, self.level_id, self._backend()).accept_with_report(
            proposal, prompt=entry.prompt, provider=entry.provider, model=entry.model)
        self.last_accept = report
        with _STATE_LOCK:
            if _PENDING.get(self._key) is entry:
                _PENDING.pop(self._key, None)
        return report.edit_id

    def _backend(self) -> Any:
        return self._work_domain().history_backend()

    def discard(self) -> None:
        """Forget the pending proposal (nothing on disk changes)."""
        with _STATE_LOCK:
            _PENDING.pop(self._key, None)

    def edits(self) -> list[dict[str, Any]]:
        """All edits (active, undone, stale) for the change list, oldest first; each with a short ``summary``."""
        try:
            history = History.load(self.project, self.level_id, self._backend())
        except (HistoryError, OSError, ValueError, NotImplementedError) as error:
            LOGGER.info("assistant: edits.json unreadable (%s)", type(error).__name__)
            return []
        rows = []
        for record in history.edits():
            row = {key: copy.deepcopy(value) for key, value in record.items() if key != "signatures"}
            row["summary"] = [{"element_id": c.get("element_id"), "kind": c.get("kind"), "change": c.get("change")}
                              for c in record.get("changes") or []]
            row["tools"] = [op.get("op") for op in record.get("ops") or []]
            rows.append(row)
        return rows

    def undo(self, edit_id: str) -> dict[str, Any]:
        """Undo one edit (the remaining active edits are re-applied); returns the UndoReport as a dict.  A pending
        proposal is discarded because its base plan changed."""
        self._check_lock()
        report = History.load(self.project, self.level_id, self._backend()).undo(str(edit_id))
        self.discard()
        return asdict(report)

    def stale(self) -> list[dict[str, Any]]:
        """Edits that no longer apply after a recompute (with ``stale_reason``)."""
        try:
            return [copy.deepcopy(e) for e in History.load(self.project, self.level_id, self._backend()).stale_edits()]
        except (HistoryError, OSError, ValueError, NotImplementedError):
            return []

    # ------------------------------------------------------------------------------------------------------- chat
    def chat(self) -> list[dict[str, Any]]:
        return ChatStore(self.project, self.level_id).load()

    def clear_chat(self) -> None:
        ChatStore(self.project, self.level_id).clear()


__all__ = ["AssistantService"]
