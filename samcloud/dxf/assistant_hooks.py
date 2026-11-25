"""Pipeline hooks of the AI assistant that can never break the pipeline.

Every hook is a no-op (no file read beyond one ``is_file`` check, nothing from ``samcloud.assistant`` imported) when
``<project>/products/assistant/<level_key>/edits.json`` does not exist. With edits:

* :func:`ensure_applied_safely` re-applies the accepted edits after the pipeline rewrote ``plan_model.json`` (clean plan,
  opening refinement, before the IFC export reads the plan),
* :func:`apply_to_result_safely` patches the edits into the DXF result of the export.

An exception inside the assistant only logs a warning. Each hook returns an info dict (``level``, ``step``,
``applied`` edit ids, ``stale`` edits with id and reason, ``changed``, ``error``) that the callers store as
``assistant`` in ``last_export.json`` / ``export.json``; the latest info of each step is also kept in
``products/assistant/<level_key>/pipeline_report.json`` so the UI can show problems of steps without a manifest.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

EDITS_FILENAME = "edits.json"
STATE_FILENAME = "state.json"
PIPELINE_REPORT_FILENAME = "pipeline_report.json"
# Fields of an edit that a share package may contain (never the user's prompt sentence or the chat).
SHARED_EDIT_FIELDS = ("id", "timestamp", "ops", "changes", "provider", "model")


def level_key_of(level: Any) -> str:
    """Assistant level key of a ``Level`` (or None = the outdoor site)."""
    return "site" if level is None else str(getattr(level, "id", level))


def assistant_level_dir(project_dir: Path | str, level_key: str) -> Path:
    return Path(project_dir) / "products" / "assistant" / str(level_key)


def edits_path(project_dir: Path | str, level_key: str) -> Path:
    return assistant_level_dir(project_dir, level_key) / EDITS_FILENAME


def has_edits(project_dir: Path | str, level_key: str) -> bool:
    """True when the level has an assistant edit history (``edits.json``)."""
    return edits_path(project_dir, level_key).is_file()


def levels_with_edits(project_dir: Path | str) -> list[str]:
    """Level keys that have an assistant ``edits.json`` (directory listing only)."""
    root = Path(project_dir) / "products" / "assistant"
    if not root.is_dir():
        return []
    return sorted(path.parent.name for path in root.glob(f"*/{EDITS_FILENAME}") if path.is_file())


def edits_newer_than(project_dir: Path | str, level_key: str, path: Path | str) -> bool:
    """True when the level has ``edits.json`` and it (or the assistant's ``state.json``) is newer than ``path``.

    An export older than an accepted / undone / re-applied edit misses it: callers treat it as stale.
    """
    edits = edits_path(project_dir, level_key)
    if not edits.is_file():
        return False
    try:
        exported = Path(path).stat().st_mtime_ns
    except OSError:
        return True
    for candidate in (edits, edits.with_name(STATE_FILENAME)):
        try:
            if candidate.stat().st_mtime_ns > exported:
                return True
        except OSError:
            continue
    return False


def read_edits(project_dir: Path | str, level_key: str) -> list[dict[str, Any]]:
    """The edit records of ``edits.json`` (plain JSON read, no assistant import); [] without or with a broken file."""
    try:
        data = json.loads(edits_path(project_dir, level_key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    edits = data.get("edits") if isinstance(data, dict) else None
    return [edit for edit in edits if isinstance(edit, dict)] if isinstance(edits, list) else []


def stale_edits(project_dir: Path | str, level_key: str) -> list[dict[str, Any]]:
    """``[{"id", "reason"}]`` of the level's stale edits."""
    return [{"id": str(edit.get("id")), "reason": str(edit.get("stale_reason") or "")}
            for edit in read_edits(project_dir, level_key) if edit.get("status") == "stale"]


def shared_active_edits(project_dir: Path | str, level_key: str) -> list[dict[str, Any]] | None:
    """The ACTIVE edits reduced to :data:`SHARED_EDIT_FIELDS` (no prompt); None when the level has no ``edits.json``."""
    if not has_edits(project_dir, level_key):
        return None
    return [{field: edit.get(field) for field in SHARED_EDIT_FIELDS} for edit in read_edits(project_dir, level_key)
            if edit.get("status") == "active"]


def _record(project_dir: Path | str, level_key: str, info: dict[str, Any]) -> None:
    """Keep the latest info of each step in ``pipeline_report.json`` (best effort)."""
    target = assistant_level_dir(project_dir, level_key) / PIPELINE_REPORT_FILENAME
    try:
        try:
            report = json.loads(target.read_text(encoding="utf-8"))
            report = report if isinstance(report, dict) else {}
        except (OSError, ValueError):
            report = {}
        report[str(info.get("step"))] = info
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)
    except (OSError, TypeError, ValueError) as error:  # unwritable folder or a value json cannot serialise
        LOGGER.warning("assistant: cannot write %s: %s", target, error)


def _info(level_key: str, step: str) -> dict[str, Any]:
    return {"level": str(level_key), "step": step, "time": datetime.now(timezone.utc).isoformat(), "applied": [], "stale": [],
            "changed": False, "error": None}


def _active_ids(project_dir: Path | str, level_key: str) -> list[str]:
    return [str(edit.get("id")) for edit in read_edits(project_dir, level_key) if edit.get("status") == "active"]


def ensure_applied_safely(project_dir: Path | str, level_key: str, step: str, logger: logging.Logger | None = None) -> dict[str, Any] | None:
    """``samcloud.assistant.history.ensure_applied`` that never raises; None without ``edits.json``."""
    if not has_edits(project_dir, level_key):
        return None
    log = logger or LOGGER
    info = _info(level_key, step)
    try:
        from samcloud.assistant.history import ensure_applied

        report = ensure_applied(project_dir, level_key)
        info.update(applied=[str(item) for item in report.applied], changed=bool(report.changed))
    except Exception as error:  # noqa: BLE001 - the assistant never breaks the pipeline
        info["error"] = f"{type(error).__name__}: {error}"
        log.warning(f"Assistant edits of level '{level_key}' could not be re-applied ({step}): {info['error']}")
    info["stale"] = stale_edits(project_dir, level_key)
    if info["stale"]:
        log.warning(f"Assistant edits of level '{level_key}' no longer apply: {[item['id'] for item in info['stale']]}")
    _record(project_dir, level_key, info)
    return info


def apply_to_result_safely(result: Any, project_dir: Path | str, level_key: str,
                           logger: logging.Logger | None = None) -> tuple[Any, dict[str, Any] | None]:
    """``samcloud.assistant.history.apply_to_result`` that never raises: (result, info); unchanged result + None without edits."""
    if not has_edits(project_dir, level_key):
        return result, None
    log = logger or LOGGER
    info = _info(level_key, "dxf_export")
    patched = result
    try:
        from samcloud.assistant.history import apply_to_result

        patched = apply_to_result(result, project_dir, level_key)
        info["changed"] = patched is not result
    except Exception as error:  # noqa: BLE001 - the export goes on with the unpatched plan
        patched = result
        info["error"] = f"{type(error).__name__}: {error}"
        log.warning(f"Assistant edits of level '{level_key}' could not be drawn into the DXF: {info['error']}")
    info["applied"] = [] if info["error"] else _active_ids(project_dir, level_key)
    info["stale"] = stale_edits(project_dir, level_key)
    if info["stale"]:
        log.warning(f"Assistant edits of level '{level_key}' no longer apply: {[item['id'] for item in info['stale']]}")
    _record(project_dir, level_key, info)
    return patched, info
