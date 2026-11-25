"""Edit history of the AI assistant: accept, undo, and re-apply after a pipeline recompute.

Files in ``<project>/products/assistant/<level_id>/`` (:func:`samcloud.assistant.paths.assistant_dir`):

* ``edits.json`` (version 1)::

      {"version": 1, "edits": [{"id", "timestamp", "prompt", "provider", "model", "ops": [...], "changes": [...],
                                "signatures": {element_id: signature}, "status": "active" | "undone" | "stale",
                                "stale_reason"}]}

  It never holds a key or credential field (they are scrubbed recursively before writing).
* ``plan_model.base.json``: the automatic plan the edits are applied on (taken on the first accept and again whenever
  the pipeline rewrote ``plan_model.json``).
* ``state.json``: SHA-256 of the ``plan_model.json`` this module wrote last (plus ``pending_sha256`` while a commit is
  in progress).  A different hash means the pipeline recomputed the plan, unless the plan still equals base + the
  active edits: :func:`ensure_applied` then takes it as the new base and re-applies the active and stale edits.
  Commits write state (pending) -> edits.json -> plan_model.json -> state, so a crash never loses or doubles an edit.

Because a recompute renumbers the ids, every edit's targets are re-resolved by **signature** (kind + geometry with
tolerances), never by id; an edit that no longer applies is marked ``stale`` with a reason, never silently dropped.
Without ``edits.json`` nothing is read or written.  Writes are atomic; a lock file keeps concurrent writers apart.

The model access is pluggable (:class:`~samcloud.assistant.domains.base.HistoryBackend`, from the domain's
``history_backend()``); the default :func:`indoor_backend` is the indoor ``plan_model.json`` described above (the model
file, its base snapshot ``plan_model.base.json`` and the hashes).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from samcloud.assistant.domains.base import HistoryBackend
from samcloud.assistant.edits import Change, EditError, EditOp, Proposal, diff_plans, resolve_signature
from samcloud.assistant.paths import assistant_dir, plan_model_path
from samcloud.bim.plan_model import BuildingPlan, load_plan_model, save_plan_model

LOGGER = logging.getLogger(__name__)

EDITS_FILENAME = "edits.json"
BASE_FILENAME = "plan_model.base.json"
STATE_FILENAME = "state.json"
LOCK_FILENAME = ".lock"
EDITS_VERSION = 1
LOCK_TIMEOUT_S = 30.0
MAX_PROMPT_LENGTH = 4000
# a dict key containing one of these (case-insensitive) is never written to edits.json
_SECRET_KEY_PARTS = ("key", "token", "secret", "password", "passwd", "authorization", "credential", "bearer")


class HistoryError(EditError):
    """History file problem (outdated proposal, unknown edit, lock timeout)."""


@dataclass
class UndoReport:
    edit_id: str
    applied: list[str] = field(default_factory=list)
    stale: list[dict[str, Any]] = field(default_factory=list)
    changed: bool = False


@dataclass
class AcceptReport:
    edit_id: str
    rebased: dict[str, str] = field(default_factory=dict)  # {id in the proposal: id in the current plan}, only changed ids
    notes: list[str] = field(default_factory=list)


@dataclass
class ReapplyReport:
    applied: list[str] = field(default_factory=list)
    stale: list[dict[str, Any]] = field(default_factory=list)
    changed: bool = False


# ----------------------------------------------------------------------------------------------------------- helpers

def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _plan_hash(plan: BuildingPlan) -> str:
    return hashlib.sha256(json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _secret_key(name: Any) -> bool:
    lowered = str(name).strip().lower()
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _redact(text: str) -> str:
    """The prompt with stored keys and ``sk-...`` / ``Bearer ...`` tokens replaced (``samcloud.assistant.credentials``)."""
    try:
        from samcloud.assistant.credentials import redact
    except Exception:  # noqa: BLE001 - without the credentials module no key is stored; tokens are still masked
        import re

        return re.sub(r"(?i)bearer\s+\S+", "Bearer ***", re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "***", text))
    return redact(text)


def _scrub(value: Any) -> Any:
    """Drop every key / credential field (recursively, matched by name part) before anything is written to edits.json."""
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if not _secret_key(k)}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_THREAD_LOCK = threading.RLock()


def _try_lock(handle: Any) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(handle: Any) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """Exclusive writer lock (OS file lock on ``.lock``: released by the OS when a writer crashes)."""
    directory.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK:
        with open(directory / LOCK_FILENAME, "a+b") as handle:
            started = time.monotonic()
            while not _try_lock(handle):
                if time.monotonic() - started > LOCK_TIMEOUT_S:
                    raise HistoryError("Another assistant edit is being saved; try again.", key="assistant.history.locked")
                time.sleep(0.05)
            try:
                yield
            finally:
                _unlock(handle)


# ----------------------------------------------------------------------------------------------------------- History

class History:
    """The edit history of one project level (``backend``: the domain's model access, default indoor plan model)."""

    def __init__(self, project: Path | str, level_id: str, backend: HistoryBackend | None = None) -> None:
        self.project = Path(project)
        self.level_id = str(level_id)
        self.backend = backend if backend is not None else indoor_backend()
        self.directory = assistant_dir(self.project, self.level_id)
        self.data: dict[str, Any] = {"version": EDITS_VERSION, "edits": []}
        self.last_rebase: dict[str, str] = {}  # {old id: new id} of the last accept that was re-applied on a newer plan
        self.last_notes: list[str] = []

    # -------------------------------------------------------------------------------------------------------- files
    @property
    def edits_path(self) -> Path:
        return self.directory / EDITS_FILENAME

    @property
    def base_path(self) -> Path:
        return self.directory / self.backend.base_filename

    @property
    def state_path(self) -> Path:
        return self.directory / STATE_FILENAME

    @property
    def plan_path(self) -> Path:
        return self.backend.model_path(self.project, self.level_id)

    @classmethod
    def load(cls, project: Path | str, level_id: str, backend: HistoryBackend | None = None) -> "History":
        history = cls(project, level_id, backend)
        history._read()
        return history

    def exists(self) -> bool:
        return self.edits_path.is_file()

    def _read(self) -> None:
        if not self.edits_path.is_file():
            self.data = {"version": EDITS_VERSION, "edits": []}
            return
        data = json.loads(self.edits_path.read_text(encoding="utf-8"))
        if int(data.get("version", 0)) != EDITS_VERSION:
            raise HistoryError(f"Unsupported edits.json version {data.get('version')!r}.", key="assistant.history.version")
        self.data = {"version": EDITS_VERSION, "edits": list(data.get("edits") or [])}

    # -------------------------------------------------------------------------------------------------------- writing
    def _write(self) -> None:
        _atomic_write(self.edits_path, json.dumps(_scrub(self.data), indent=1, ensure_ascii=False) + "\n")

    def _save_plan(self, plan: BuildingPlan) -> None:
        self.backend.save(plan, self.plan_path)

    def _write_state(self, plan_hash: str | None, pending: str | None = None) -> None:
        state: dict[str, Any] = {"plan_model_sha256": plan_hash, "updated": _now()}
        if pending is not None:
            state["pending_sha256"] = pending
        _atomic_write(self.state_path, json.dumps(state, indent=1) + "\n")

    def _commit(self, plan: BuildingPlan) -> None:
        """Crash-safe write order: state (pending hash) -> edits.json -> plan_model.json -> state (final hash).

        A crash in any window leaves a state that :meth:`_sync` resolves: before edits.json the plan still equals base + the
        stored edits; before plan_model.json it is rebuilt from base + edits; before the final state the pending hash
        is recognised.  An edit is therefore never lost and never applied twice.
        """
        self._write_state(_sha256(self.plan_path), self.backend.file_hash(plan))
        self._write()
        self._save_plan(plan)
        self._write_state(_sha256(self.plan_path))

    def _state(self) -> dict[str, Any] | None:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return state if isinstance(state, dict) and state.get("plan_model_sha256") else None

    def _state_hash(self) -> str | None:
        state = self._state()
        return None if state is None else state.get("plan_model_sha256")

    def base_plan(self) -> BuildingPlan | None:
        return self.backend.load(self.base_path) if self.base_path.is_file() else None

    def current_plan(self) -> BuildingPlan:
        return self.backend.load(self.plan_path)

    # ---------------------------------------------------------------------------------------------------------- query
    def edits(self) -> list[dict[str, Any]]:
        return list(self.data["edits"])

    def active_edits(self) -> list[dict[str, Any]]:
        return [edit for edit in self.data["edits"] if edit.get("status") == "active"]

    def stale_edits(self) -> list[dict[str, Any]]:
        return [edit for edit in self.data["edits"] if edit.get("status") == "stale"]

    def edit(self, edit_id: str) -> dict[str, Any]:
        for edit in self.data["edits"]:
            if edit.get("id") == edit_id:
                return edit
        raise HistoryError(f"No assistant edit {edit_id!r}.", key="assistant.history.unknown_edit", params={"id": edit_id})

    # ----------------------------------------------------------------------------------------------------------- sync
    def _expected(self, base: BuildingPlan) -> BuildingPlan | None:
        """base + the active edits (statuses untouched); None when one of them does not apply."""
        plan = copy.deepcopy(base)
        try:
            for record in self.active_edits():
                plan = reapply_edit(plan, record, backend=self.backend)
        except EditError:
            return None
        return plan

    def _sync(self, *, cloud: Any | None = None) -> ReapplyReport:
        """Bring plan_model.json, state.json and the base snapshot in line (call under the lock, after ``_read``).

        * hashes agree: nothing to do;
        * the pending hash of an interrupted commit matches: the plan was written, only the state is finalised;
        * otherwise (state missing / corrupt / pending, or the pipeline rewrote the plan): when the current plan equals
          base + the active edits, only state.json is rewritten; after a commit interrupted before plan_model.json it is
          rebuilt from base + edits; anything else is a recompute: the current plan becomes the new base and every active
          or stale edit is re-applied by signature.
        """
        if not self.edits_path.is_file() or not self.plan_path.is_file():
            return ReapplyReport()
        current_hash = _sha256(self.plan_path)
        state = self._state()
        active = [str(e["id"]) for e in self.active_edits()]
        if state is not None and not state.get("pending_sha256") and current_hash == state.get("plan_model_sha256"):
            return ReapplyReport(active, [], False)
        if state is not None and state.get("pending_sha256") and current_hash == state["pending_sha256"]:
            self._write_state(current_hash)
            return ReapplyReport(active, [], False)
        base = self.base_plan()
        current = self.current_plan()
        if base is not None:
            expected = self._expected(base)
            if expected is not None and self.backend.semantic_hash(expected) == self.backend.semantic_hash(current):
                self._write_state(current_hash)
                return ReapplyReport(active, [], False)
            if state is not None and state.get("pending_sha256") and current_hash == state.get("plan_model_sha256"):
                # a commit stopped before plan_model.json was written: rebuild from base + the stored edits
                plan, applied, stale = self._rebuild(base, cloud=cloud)
                self._commit(plan)
                return ReapplyReport(applied, stale, True)
        # the pipeline rewrote plan_model.json: it is the new automatic base
        _atomic_write(self.base_path, self.plan_path.read_text(encoding="utf-8"))
        plan, applied, stale = self._rebuild(self.backend.load(self.base_path), cloud=cloud)
        self._commit(plan)
        LOGGER.info("assistant: re-applied %d edit(s), %d stale, level %s", len(applied), len(stale), self.level_id)
        return ReapplyReport(applied, stale, True)

    def _rebuild(self, base: BuildingPlan, *, cloud: Any | None = None) -> tuple[BuildingPlan, list[str], list[dict[str, Any]]]:
        """base + every active or stale edit in order; stale edits that apply again turn active, failing ones stale."""
        plan = copy.deepcopy(base)
        applied: list[str] = []
        stale: list[dict[str, Any]] = []
        for record in self.data["edits"]:
            if record.get("status") not in ("active", "stale"):
                continue
            try:
                plan = reapply_edit(plan, record, cloud=cloud, backend=self.backend)
            except EditError as error:
                record["status"] = "stale"
                record["stale_reason"] = error.message
                record["stale_code"] = error.key
                stale.append({"id": record["id"], "reason": error.message, "key": error.key, "params": dict(error.params)})
                continue
            record["status"] = "active"
            record["stale_reason"] = None
            record.pop("stale_code", None)
            applied.append(str(record["id"]))
        return plan, applied, stale

    # --------------------------------------------------------------------------------------------------------- accept
    def accept(self, proposal: Proposal, *, prompt: str, provider: str, model: str) -> str:
        """Store the proposal as a new active edit and write the edited ``plan_model.json``; returns the edit id.

        A proposal made on an older plan (another accept, an undo or a recompute since) is re-applied by signature onto
        the current plan; when that fails, :class:`HistoryError` ``assistant.history.outdated`` (or ``ambiguous``) is
        raised.  Ids that changed on the way are in :attr:`last_rebase` / :attr:`last_notes` (see :meth:`accept_with_report`).
        """
        return self.accept_with_report(proposal, prompt=prompt, provider=provider, model=model).edit_id

    def accept_with_report(self, proposal: Proposal, *, prompt: str, provider: str, model: str) -> AcceptReport:
        """Like :meth:`accept`, returning ``AcceptReport(edit_id, rebased {old id: new id}, notes)`` for the UI."""
        self.last_rebase = {}
        self.last_notes = []
        if not proposal.ops:
            raise HistoryError("The proposal holds no edit.", key="assistant.history.empty")
        with _locked(self.directory):
            self._read()
            if not self.plan_path.is_file():
                raise HistoryError(f"{self.plan_path.name} of this level is missing.", key="assistant.history.no_plan")
            if self.edits_path.is_file():
                self._sync()
            if not self.base_path.is_file() or not self.active_edits() and not self.stale_edits():
                # no edit is applied: the current plan is the automatic base
                _atomic_write(self.base_path, self.plan_path.read_text(encoding="utf-8"))
            current = self.current_plan()
            ops = [op.to_dict() for op in proposal.ops]
            plan, changes, signatures, notes = proposal.plan, proposal.changes, proposal.signatures, proposal.notes
            rebase: dict[str, str] = {}
            if self.backend.semantic_hash(current) != self.backend.semantic_hash(proposal.base):
                try:
                    rebased, mapping = _replay_with_mapping(current, {"ops": ops, "signatures": proposal.signatures},
                                                            backend=self.backend)
                except HistoryError:
                    raise
                except EditError as error:
                    raise HistoryError(f"The plan changed since the proposal was made ({error.message}); ask again.",
                                       key="assistant.history.outdated", params=dict(error.params)) from error
                ops = [op.to_dict() for op in rebased.ops]
                plan, changes, signatures, notes = rebased.plan, rebased.changes, rebased.signatures, list(rebased.notes)
                rebase = {old: new for old, new in mapping.items() if old != new}
                if rebase:
                    notes.append("re-applied on the current plan, ids changed: " + ", ".join(f"{old} -> {new}" for old, new in rebase.items()))
            edit_id = f"E{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
            record = {
                "id": edit_id, "timestamp": _now(), "prompt": _redact(str(prompt or ""))[:MAX_PROMPT_LENGTH],
                "provider": str(provider or ""), "model": str(model or ""), "ops": ops,
                "changes": [change.to_dict() for change in changes], "signatures": copy.deepcopy(signatures),
                "notes": list(notes), "status": "active", "stale_reason": None,
            }
            if rebase:
                record["rebased"] = dict(rebase)
            self.data["edits"].append(record)
            self._commit(plan)
            self.last_rebase = dict(rebase)
            self.last_notes = list(notes)
            return AcceptReport(edit_id, dict(rebase), list(notes))

    # ----------------------------------------------------------------------------------------------------------- undo
    def undo(self, edit_id: str, *, cloud: Any | None = None) -> UndoReport:
        """Undo one edit: rebuild = base + the remaining edits in order (after a recompute the new plan is the base)."""
        with _locked(self.directory):
            self._read()
            record = self.edit(edit_id)
            if record.get("status") not in ("active", "stale"):
                raise HistoryError(f"Edit {edit_id} is not active.", key="assistant.history.not_active", params={"id": edit_id})
            self._sync(cloud=cloud)
            record = self.edit(edit_id)
            base = self.base_plan()
            if base is None:
                raise HistoryError("The base plan snapshot is missing.", key="assistant.history.no_base")
            record["status"] = "undone"
            record["undone_at"] = _now()
            plan, applied, stale = self._rebuild(base, cloud=cloud)
            self._commit(plan)
            return UndoReport(edit_id, applied, stale, True)

    # ----------------------------------------------------------------------------------------------------------- lock
    def is_locked(self) -> bool:
        """Whether an assistant writer holds this level's lock right now (thread lock or OS file lock); never waits."""
        return history_locked(self.project, self.level_id)

    # -------------------------------------------------------------------------------------------------------- reapply
    def ensure_applied(self, *, cloud: Any | None = None) -> ReapplyReport:
        if not self.edits_path.is_file():
            return ReapplyReport()
        with _locked(self.directory):
            self._read()
            return self._sync(cloud=cloud)


def _file_hash_of(plan: BuildingPlan) -> str:
    """SHA-256 of the bytes ``save_plan_model`` writes for ``plan`` (compact JSON, ASCII, no newline)."""
    return hashlib.sha256(json.dumps(plan.to_dict(), separators=(",", ":")).encode("utf-8")).hexdigest()


def _replay(plan: BuildingPlan, record: dict[str, Any], *, cloud: Any | None = None,
            backend: HistoryBackend | None = None) -> Proposal:
    """A proposal with one stored edit replayed on ``plan`` (targets re-resolved by signature)."""
    return _replay_with_mapping(plan, record, cloud=cloud, backend=backend)[0]


def _replay_with_mapping(plan: BuildingPlan, record: dict[str, Any], *, cloud: Any | None = None,
                         backend: HistoryBackend | None = None) -> tuple[Proposal, dict[str, str]]:
    """(proposal, {old id: new id}) of one stored edit replayed on ``plan``; targets re-resolved strictly by signature.

    Raises :class:`HistoryError` ``assistant.history.ambiguous`` when a target matches two elements about equally well,
    ``EditError`` ``assistant.history.element_gone`` when it matches none.
    """
    backend = backend if backend is not None else indoor_backend()
    signatures = record.get("signatures") or {}
    mapping: dict[str, str] = {}
    for old_id, signature in signatures.items():
        try:
            new_id = backend.resolve_signature(plan, signature, preferred_id=old_id, strict=True)
        except EditError as error:
            raise HistoryError(error.message, key="assistant.history.ambiguous", params=dict(error.params)) from error
        if new_id is None:
            raise EditError(f"Element {old_id} of this edit is no longer in the plan.", key="assistant.history.element_gone", params={"id": old_id})
        mapping[old_id] = new_id
    if len(set(mapping.values())) != len(mapping):
        raise HistoryError("Two targets of this edit now match the same element.", key="assistant.history.ambiguous")
    proposal = backend.new_proposal(plan)
    for data in record.get("ops") or []:
        op = backend.op_from_dict(copy.deepcopy(data))
        if op.target is not None:
            op.target = mapping.get(op.target, op.target)
        created = list(op.params.pop("_created_ids", []) or [])
        changes = proposal.apply(op, cloud=cloud, replay=True)
        for old_id, change in zip(created, [c for c in changes if c.change == "added"]):
            mapping[old_id] = change.element_id
    return proposal, mapping


def reapply_edit(plan: BuildingPlan, record: dict[str, Any], *, cloud: Any | None = None,
                 backend: HistoryBackend | None = None) -> BuildingPlan:
    """``plan`` with one stored edit applied; targets re-resolved by signature (raises EditError when it no longer applies)."""
    return _replay(plan, record, cloud=cloud, backend=backend).plan


# ------------------------------------------------------------------------------------------------------ indoor backend

def _save_indoor_plan(plan: BuildingPlan, path: Path) -> None:
    save_plan_model(plan, Path(path).parent)


_INDOOR_BACKEND: HistoryBackend | None = None


def indoor_backend() -> HistoryBackend:
    """Model access of the indoor plan model: ``plan_model.json`` of the level, base snapshot ``plan_model.base.json``."""
    global _INDOOR_BACKEND
    if _INDOOR_BACKEND is None:
        _INDOOR_BACKEND = HistoryBackend(
            kind="indoor", base_filename=BASE_FILENAME, model_path=plan_model_path, load=load_plan_model,
            save=_save_indoor_plan, semantic_hash=_plan_hash, file_hash=_file_hash_of, new_proposal=Proposal,
            op_from_dict=EditOp.from_dict, resolve_signature=resolve_signature, diff=diff_plans)
    return _INDOOR_BACKEND


# ------------------------------------------------------------------------------------------------- module functions

def history_locked(project: Path | str, level_id: str) -> bool:
    """Non-blocking: True while a writer holds the history lock of a level (in this process or another one)."""
    if not _THREAD_LOCK.acquire(blocking=False):
        return True
    try:
        lock = assistant_dir(project, level_id) / LOCK_FILENAME
        if not lock.is_file():
            return False
        try:
            handle = open(lock, "a+b")
        except OSError:
            return False
        with handle:
            if not _try_lock(handle):
                return True
            _unlock(handle)
            return False
    finally:
        _THREAD_LOCK.release()


def ensure_applied(project: Path | str, level_id: str, *, cloud: Any | None = None,
                   backend: HistoryBackend | None = None) -> ReapplyReport:
    """Re-apply the active edits when the pipeline recomputed ``plan_model.json``; no ``edits.json`` = touch nothing.

    Pipeline hook: call after the clean plan wrote ``plan_model.json`` (and before the IFC export reads it).
    ``backend`` is the domain's model access (default: the indoor plan model).
    """
    history = History(project, level_id, backend)
    if not history.edits_path.is_file():
        return ReapplyReport()
    return history.ensure_applied(cloud=cloud)


def edit_changes(project: Path | str, level_id: str,
                 backend: HistoryBackend | None = None) -> tuple[BuildingPlan | None, BuildingPlan | None, list[Change]]:
    """(base plan, edited plan, element changes between them) of a level with active edits, else (None, None, [])."""
    history = History.load(project, level_id, backend)
    if not history.exists() or not history.active_edits():
        return None, None, []
    base = history.base_plan()
    if base is None or not history.plan_path.is_file():
        return None, None, []
    edited = history.current_plan()
    return base, edited, history.backend.diff(base, edited)


def apply_to_result(result: Any, project: Path | str, level_id: str, language: str | None = None) -> Any:
    """DXF hook: ``result`` with the assistant's edits patched in (the same object when there is nothing to patch).

    Call in ``samcloud.dxf.jobs.run_export`` after ``apply_reconstructed_outlines``.  ``language`` is accepted for the
    hook signature; texts stay pipeline-identical (export-time translation is done elsewhere).
    """
    del language
    if not History(project, level_id).edits_path.is_file():
        return result
    ensure_applied(project, level_id)
    base, edited, changes = edit_changes(project, level_id)
    if base is None or edited is None or not changes:
        return result
    from samcloud.assistant.dxf_patch import apply_dxf_patch

    return apply_dxf_patch(result, base, edited, changes)


__all__ = ["AcceptReport", "History", "HistoryError", "ReapplyReport", "UndoReport", "apply_to_result", "edit_changes",
           "ensure_applied", "history_locked", "indoor_backend", "reapply_edit"]
