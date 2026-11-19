"""Patch log with exact undo and redo (MASTERPROMPT section 5; docs/DATA_MODEL.md, Patches).

Every change to a scene is a :class:`PatchRecord`: RFC 6902 operations over the scene's JSON
document (``Scene.model_dump(mode="json")``) plus the inverse operations, computed from the
document before the change. :class:`PatchEngine` validates a patch, applies it all-or-nothing,
recomputes the derived values and appends the record to ``scene/patches.jsonl``.

Log format: one JSON object per line (UTF-8, LF), append-only. A record has ``action``
``apply`` (a new patch), ``undo`` or ``redo``. Undo and redo records carry ``ref``, the id of
the ``apply`` record they act on, and the operations that were applied then (``forward`` is the
original inverse for an undo, the original forward for a redo). Replaying the log from the
base scene therefore only applies the ``forward`` operations in file order.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from samcloud.bim.core.model.enums import ElementKind
from samcloud.bim.core.model.graph import RecomputeResult, recompute
from samcloud.bim.core.model.json_patch import (
    JsonPatchError,
    Operation,
    apply_patch_with_inverse,
    format_pointer,
    parse_pointer,
    validate_operation,
)
from samcloud.bim.core.model.params import PARAM_SCHEMAS, check_param_ranges
from samcloud.bim.core.model.scene import Scene

LAYER_TOLERANCE = 1e-9


class AuthorKind(StrEnum):
    """Who makes a change."""

    USER = "user"
    AI = "ai"
    SCRIPT = "script"


class PatchError(Exception):
    """A patch, undo or redo cannot be performed (the scene is unchanged)."""


class PatchRejected(PatchError):
    """A patch violates a rule; ``messages`` lists every reason."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = messages
        super().__init__("patch rejected: " + "; ".join(messages))


def _utc_now() -> datetime:
    return datetime.now(UTC)


class PatchRecord(BaseModel):
    """One entry of ``patches.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    action: Literal["apply", "undo", "redo"] = "apply"
    ref: str | None = None
    """For ``undo`` and ``redo``: id of the ``apply`` record this record acts on."""
    group_id: str | None = None
    author: str
    author_kind: AuthorKind
    timestamp: str
    reason: str = ""
    forward: list[dict[str, Any]]
    inverse: list[dict[str, Any]]
    recompute: Any = Field(default=None, exclude=True, repr=False)
    """Not logged: the :class:`RecomputeResult` of the step (``changed`` = derived ids)."""

    @property
    def changed_derived(self) -> tuple[str, ...]:
        """Ids whose derived values changed in this step."""
        result: RecomputeResult | None = self.recompute
        return () if result is None else result.changed


def _format_time(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _error_messages(error: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()]


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _changed_ids(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    old: dict[str, Any] = before.get("elements", {})
    new: dict[str, Any] = after.get("elements", {})
    return sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))


def _is_unlock_only(operations: list[Operation], author_kind: AuthorKind) -> bool:
    if author_kind != AuthorKind.USER:
        return False
    effective = [op for op in operations if op["op"] != "test"]
    if not effective:
        return False
    for op in effective:
        if op["op"] not in ("add", "replace") or op.get("value") is not False:
            return False
        tokens = parse_pointer(op["path"])
        if len(tokens) != 3 or tokens[0] != "elements" or tokens[2] != "locked":
            return False
    return True


class PatchEngine:
    """Apply patches to a scene with validation, logging, undo and redo."""

    def __init__(
        self,
        scene: Scene,
        log_path: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.scene = scene
        self.log_path = None if log_path is None else Path(log_path)
        self._clock = clock or _utc_now
        self._history: list[PatchRecord] = []
        self._undo_stack: list[PatchRecord] = []
        self._redo_stack: list[PatchRecord] = []
        self.last_recompute: RecomputeResult | None = None

    # ------------------------------------------------------------------ public state

    @property
    def history(self) -> list[PatchRecord]:
        """All records in log order (apply, undo and redo)."""
        return list(self._history)

    @property
    def applied(self) -> list[PatchRecord]:
        """The ``apply`` records currently in effect, oldest first (the undo stack)."""
        return list(self._undo_stack)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    # ------------------------------------------------------------------------ apply

    def apply(
        self,
        ops: list[dict[str, Any]],
        author: str,
        author_kind: AuthorKind | str,
        reason: str = "",
        group_id: str | None = None,
    ) -> PatchRecord:
        """Validate and apply a patch. Raises :class:`PatchRejected`; nothing changes then."""
        kind = AuthorKind(author_kind)
        try:
            operations = [validate_operation(op) for op in ops]
        except JsonPatchError as error:
            raise PatchRejected([str(error)]) from error
        if not operations:
            raise PatchRejected(["patch has no operations"])
        before = self.scene.model_dump(mode="json")
        self._check_paths(before, operations, kind)
        try:
            work, inverse = apply_patch_with_inverse(before, operations)
        except JsonPatchError as error:
            raise PatchRejected([str(error)]) from error
        forward = list(operations)
        for extra in self._derive_ops(before, work, kind):
            try:
                work, extra_inverse = apply_patch_with_inverse(work, [extra])
            except JsonPatchError as error:  # pragma: no cover - extra ops are well-formed
                raise PatchRejected([str(error)]) from error
            forward.append(extra)
            inverse = extra_inverse + inverse
        changed = _changed_ids(before, work)
        candidate, result = self._validate(before, work, changed)
        record = PatchRecord(
            group_id=group_id,
            author=author,
            author_kind=kind,
            timestamp=_format_time(self._clock()),
            reason=reason,
            forward=forward,
            inverse=inverse,
            recompute=result,
        )
        self._append(record)
        self._adopt(candidate, result)
        self._history.append(record)
        self._undo_stack.append(record)
        self._redo_stack.clear()
        return record

    # ------------------------------------------------------------------ undo / redo

    def undo(
        self, author: str = "user", author_kind: AuthorKind | str = AuthorKind.USER
    ) -> PatchRecord | None:
        """Undo the latest applied patch; ``None`` when there is nothing to undo."""
        if not self._undo_stack:
            return None
        original = self._undo_stack[-1]
        record = self._step(
            "undo", original, original.inverse, original.forward, author, author_kind
        )
        self._undo_stack.pop()
        self._redo_stack.append(original)
        return record

    def redo(
        self, author: str = "user", author_kind: AuthorKind | str = AuthorKind.USER
    ) -> PatchRecord | None:
        """Reapply the most recently undone patch; ``None`` when there is nothing to redo."""
        if not self._redo_stack:
            return None
        original = self._redo_stack[-1]
        record = self._step(
            "redo", original, original.forward, original.inverse, author, author_kind
        )
        self._redo_stack.pop()
        self._undo_stack.append(original)
        return record

    def undo_group(
        self,
        group_id: str,
        author: str = "user",
        author_kind: AuthorKind | str = AuthorKind.USER,
    ) -> list[PatchRecord]:
        """Undo the newest patches that share ``group_id`` (they must be on top of the stack).

        Returns the undo records, newest patch first. Raises :class:`PatchError` when the
        newest applied patch does not belong to the group.
        """
        if not self._undo_stack or self._undo_stack[-1].group_id != group_id:
            raise PatchError(f"group {group_id!r} is not at the top of the undo stack")
        records: list[PatchRecord] = []
        while self._undo_stack and self._undo_stack[-1].group_id == group_id:
            undone = self.undo(author, author_kind)
            assert undone is not None
            records.append(undone)
        return records

    def _step(
        self,
        action: Literal["undo", "redo"],
        original: PatchRecord,
        forward: list[dict[str, Any]],
        inverse: list[dict[str, Any]],
        author: str,
        author_kind: AuthorKind | str,
    ) -> PatchRecord:
        before = self.scene.model_dump(mode="json")
        try:
            work, _ = apply_patch_with_inverse(before, forward)
        except JsonPatchError as error:
            raise PatchError(f"cannot {action} patch {original.id}: {error}") from error
        candidate, result = self._validate(before, work, _changed_ids(before, work), rules=False)
        record = PatchRecord(
            action=action,
            ref=original.id,
            group_id=original.group_id,
            author=author,
            author_kind=AuthorKind(author_kind),
            timestamp=_format_time(self._clock()),
            reason=f"{action} of {original.id}",
            forward=copy.deepcopy(forward),
            inverse=copy.deepcopy(inverse),
            recompute=result,
        )
        self._append(record)
        self._adopt(candidate, result)
        self._history.append(record)
        return record

    # ----------------------------------------------------------------------- replay

    @classmethod
    def load(
        cls,
        scene: Scene,
        log_path: str | Path,
        clock: Callable[[], datetime] | None = None,
    ) -> PatchEngine:
        """Replay ``log_path`` onto ``scene`` (the base scene) and return an engine bound to it.

        The scene is modified in place; derived values are recomputed once at the end. The
        engine keeps appending to the same log. A missing log file gives an unchanged scene.
        """
        engine = cls(scene, log_path, clock)
        path = Path(log_path)
        if not path.exists():
            return engine
        document = scene.model_dump(mode="json")
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            if not line.strip():
                continue
            try:
                record = PatchRecord.model_validate_json(line)
                document, _ = apply_patch_with_inverse(document, record.forward)
            except (ValidationError, JsonPatchError) as error:
                raise PatchError(f"{path}:{number}: cannot replay record: {error}") from error
            engine._replay_stacks(record, number)
            engine._history.append(record)
        try:
            candidate = Scene.model_validate(document)
            result = recompute(candidate)
        except (ValidationError, ValueError) as error:
            raise PatchError(f"{path}: replayed scene is invalid: {error}") from error
        engine._adopt(candidate, result)
        return engine

    def _replay_stacks(self, record: PatchRecord, number: int) -> None:
        if record.action == "apply":
            self._undo_stack.append(record)
            self._redo_stack.clear()
            return
        source = self._undo_stack if record.action == "undo" else self._redo_stack
        target = self._redo_stack if record.action == "undo" else self._undo_stack
        if not source or source[-1].id != record.ref:
            raise PatchError(f"log line {number}: {record.action} of {record.ref} is out of order")
        target.append(source.pop())

    # --------------------------------------------------------------------- internals

    def _check_paths(
        self, before: dict[str, Any], operations: list[Operation], author_kind: AuthorKind
    ) -> None:
        """Reject /derived edits and edits of locked elements."""
        problems: list[str] = []
        elements: dict[str, Any] = before.get("elements", {})
        unlock = _is_unlock_only(operations, author_kind)
        for op in operations:
            if op["op"] == "test":
                continue
            paths = [op["path"]]
            if op["op"] == "move":
                paths.append(op["from"])
            for path in paths:
                tokens = parse_pointer(path)
                if not tokens or tokens[0] == "derived":
                    problems.append(
                        f"{path or '<root>'}: derived values are never edited directly "
                        "(they are recomputed)"
                    )
                    continue
                if tokens[0] != "elements" or unlock:
                    continue
                if len(tokens) == 1:
                    touched = [i for i, e in elements.items() if e.get("locked")]
                else:
                    touched = [tokens[1]] if elements.get(tokens[1], {}).get("locked") else []
                for element_id in touched:
                    problems.append(
                        f"{path}: element {element_id} is locked "
                        "(only the user can unlock it, with a patch that sets locked to false)"
                    )
        if problems:
            raise PatchRejected(problems)

    def _derive_ops(
        self, before: dict[str, Any], work: dict[str, Any], author_kind: AuthorKind
    ) -> list[Operation]:
        """Extra operations that belong to the patch: layer sync, origins, provenance, lock."""
        extra: list[Operation] = []
        old_elements: dict[str, Any] = before.get("elements", {})
        new_elements: dict[str, Any] = work.get("elements", {})
        auto_lock = bool(work.get("settings", {}).get("auto_lock_on_manual_edit"))
        for element_id in _changed_ids(before, work):
            new = new_elements.get(element_id)
            if new is None:
                continue
            old = old_elements.get(element_id)
            base = ["elements", element_id]
            new_params: dict[str, Any] = new.get("params", {})
            old_params: dict[str, Any] = {} if old is None else old.get("params", {})
            changed_params = sorted(
                key
                for key in new_params.keys() | old_params.keys()
                if new_params.get(key) != old_params.get(key)
            )
            if old is not None:
                extra.extend(self._layer_sync(element_id, old, new))
            if not changed_params and old is not None:
                continue
            if author_kind == AuthorKind.USER:
                origins: dict[str, Any] = new.get("param_origin", {})
                if old is not None:
                    for key in changed_params:
                        if key in new_params and origins.get(key) != "manual":
                            path = format_pointer([*base, "param_origin", key])
                            extra.append({"op": "add", "path": path, "value": "manual"})
                if new.get("provenance") != "manual":
                    path = format_pointer([*base, "provenance"])
                    extra.append({"op": "replace", "path": path, "value": "manual"})
                if auto_lock and old is not None and not new.get("locked"):
                    extra.append(
                        {"op": "replace", "path": format_pointer([*base, "locked"]), "value": True}
                    )
            elif author_kind == AuthorKind.AI and new.get("provenance") != "ai":
                extra.append(
                    {"op": "replace", "path": format_pointer([*base, "provenance"]), "value": "ai"}
                )
        return extra

    def _layer_sync(
        self, element_id: str, old: dict[str, Any], new: dict[str, Any]
    ) -> list[Operation]:
        if new.get("kind") not in ("wall", "slab"):
            return []
        old_thickness = old.get("params", {}).get("thickness")
        new_thickness = new.get("params", {}).get("thickness")
        if not _is_number(old_thickness) or not _is_number(new_thickness):
            return []
        if old_thickness == new_thickness:
            return []
        old_layers = old.get("params", {}).get("layers", [])
        new_layers = new.get("params", {}).get("layers", [])
        if old_layers != new_layers:
            return []
        if len(old_layers) == 1 and math.isclose(
            old_layers[0]["thickness"], old_thickness, abs_tol=LAYER_TOLERANCE
        ):
            path = format_pointer(["elements", element_id, "params", "layers", "0", "thickness"])
            return [{"op": "replace", "path": path, "value": new_thickness}]
        raise PatchRejected(
            [
                f"element {element_id}: thickness changed but it has {len(old_layers)} layers "
                "that were not changed; the patch must also change the layers"
            ]
        )

    def _validate(
        self,
        before: dict[str, Any],
        work: dict[str, Any],
        changed: list[str],
        *,
        rules: bool = True,
    ) -> tuple[Scene, RecomputeResult]:
        """Build the candidate scene, check the rules and recompute; nothing is committed."""
        try:
            candidate = Scene.model_validate(work)
        except ValidationError as error:
            raise PatchRejected(_error_messages(error)) from error
        if rules:
            messages: list[str] = []
            for element_id in changed:
                element = candidate.elements.get(element_id)
                if element is None or element.kind not in PARAM_SCHEMAS:
                    continue
                params = element.params.model_dump(mode="json")
                messages.extend(
                    f"{element.kind.value} {element_id}: {m}"
                    for m in check_param_ranges(ElementKind(element.kind), params)
                )
            old_problems = set(Scene.model_validate(before).check_references())
            messages.extend(p for p in candidate.check_references() if p not in old_problems)
            if messages:
                raise PatchRejected(messages)
        try:
            result = recompute(candidate, [i for i in changed if i in candidate.elements])
        except (ValueError, KeyError, ArithmeticError) as error:
            raise PatchRejected([f"recompute failed: {error}"]) from error
        return candidate, result

    def _adopt(self, candidate: Scene, result: RecomputeResult) -> None:
        for name in Scene.model_fields:
            setattr(self.scene, name, getattr(candidate, name))
        self.last_recompute = result

    def _append(self, record: PatchRecord) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()


__all__ = [
    "AuthorKind",
    "PatchEngine",
    "PatchError",
    "PatchRecord",
    "PatchRejected",
]
