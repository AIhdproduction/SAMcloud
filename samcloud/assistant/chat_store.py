"""Chat history of the AI assistant for one project level: ``<project>/products/assistant/<level_id>/chat.json``.

Only the conversation text and small metadata are stored (tool names with short arguments, cost, tokens, provider /
model, status): never keys, never raw tool payloads or point data.  Everything written goes through
:func:`samcloud.assistant.credentials.redact` and secret-looking fields are dropped.  The file lives outside
``products/dxf`` and ``products/bim`` and is therefore never part of a project share.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from samcloud.assistant.credentials import redact
from samcloud.assistant.paths import assistant_dir

LOGGER = logging.getLogger(__name__)

CHAT_FILENAME = "chat.json"
CHAT_VERSION = 1
MAX_MESSAGES = 200
MAX_TEXT_CHARS = 8000
MAX_TOOLS = 30
ROLES = ("user", "assistant", "system")
# metadata the store keeps; anything else is dropped
META_FIELDS = frozenset({"tools", "cost_usd", "provider", "model", "input_tokens", "output_tokens", "error_kind", "changes",
                         "qa_ok", "status", "selection", "needs_confirmation", "asks_manual", "edit_id", "tier", "applied"})
_SECRET_FIELDS = frozenset({"api_key", "apikey", "key", "token", "access_token", "secret", "password", "authorization",
                            "credential", "credentials", "bearer"})
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(value: Any, depth: int = 0) -> Any:
    """Redacted, bounded, JSON-safe copy without secret-looking fields."""
    if depth > 4:
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v, depth + 1) for k, v in list(value.items())[:40] if str(k).strip().lower() not in _SECRET_FIELDS}
    if isinstance(value, (list, tuple)):
        return [_clean(v, depth + 1) for v in list(value)[:MAX_TOOLS]]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6)
    text = redact(str(value))
    return text if len(text) <= 300 else text[:297] + "..."


def clean_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Only the allowed metadata fields; tool entries reduced to ``{"tool", "args"}`` (short)."""
    result: dict[str, Any] = {}
    for name, value in (meta or {}).items():
        if name not in META_FIELDS:
            continue
        if name == "tools":
            tools = []
            for entry in list(value or [])[:MAX_TOOLS]:
                if isinstance(entry, dict):
                    item = {"tool": str(entry.get("tool") or entry.get("name") or "")[:60],
                            "args": str(entry.get("args") or "")[:160]}
                    if "ok" in entry:
                        item["ok"] = bool(entry["ok"])
                    tools.append(item)
                else:
                    tools.append({"tool": str(entry)[:60]})
            result[name] = _clean(tools)
        else:
            result[name] = _clean(value)
    return result


class ChatStore:
    """The chat of one project level (list of ``{"id", "role", "text", "timestamp", "meta"}``)."""

    def __init__(self, project: Path | str, level_id: str) -> None:
        self.project = Path(project)
        self.level_id = str(level_id)
        self.path = assistant_dir(self.project, self.level_id) / CHAT_FILENAME

    def load(self) -> list[dict[str, Any]]:
        """All messages (oldest first); a missing or broken file yields an empty list."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        messages = data.get("messages") if isinstance(data, dict) else None
        return [m for m in messages if isinstance(m, dict)] if isinstance(messages, list) else []

    def _write(self, messages: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"version": CHAT_VERSION, "messages": messages[-MAX_MESSAGES:]}, ensure_ascii=False, indent=1)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex[:8]}.tmp")
        temporary.write_text(redact(text) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def append(self, role: str, text: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Add one message (redacted; metadata filtered) and return it."""
        role = role if role in ROLES else "system"
        body = redact(str(text or ""))
        if len(body) > MAX_TEXT_CHARS:
            body = body[: MAX_TEXT_CHARS - 3] + "..."
        entry = {"id": uuid.uuid4().hex[:12], "role": role, "text": body, "timestamp": _now(), "meta": clean_meta(meta)}
        with _LOCK:
            messages = self.load()
            messages.append(entry)
            self._write(messages)
        return entry

    def clear(self) -> None:
        """Empty the chat (the file stays, with no messages); nothing is created when there is no chat yet."""
        with _LOCK:
            if self.path.is_file():
                self._write([])


__all__ = ["CHAT_FILENAME", "ChatStore", "clean_meta"]
