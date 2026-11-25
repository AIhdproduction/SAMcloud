"""API keys and the Ollama server address of the AI assistant, stored outside the repository.

File: ``<user config dir>/ai_credentials.yaml`` (see :func:`samcloud.core.secretfile.config_dir`), owner-only.
Keys are never logged; use :func:`masked_ai_credentials` for display and :func:`redact` for every text that could
contain request data (error messages, log lines).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable

from samcloud.core import secretfile
from samcloud.core.secretfile import mask

LOGGER = logging.getLogger(__name__)

FILE_NAME = "ai_credentials.yaml"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
KEY_FIELDS: dict[str, str] = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "openrouter": "openrouter_api_key",
}
FIELDS: tuple[str, ...] = ("ollama_url", *KEY_FIELDS.values())

_TOKEN_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[^\s\"',;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"(?i)(x-api-key[\"']?\s*[:=]\s*[\"']?)[^\s\"',;]+"),
)
_MIN_SECRET_LENGTH = 6


def credentials_path() -> Path:
    """Location of the AI credentials file (not created)."""
    return secretfile.config_dir() / FILE_NAME


def _normalize_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/") or DEFAULT_OLLAMA_URL


def load_ai_credentials() -> dict[str, str]:
    """All fields of the credentials file; a missing or broken file yields the defaults (no keys, local Ollama)."""
    data = secretfile.read_yaml(credentials_path()) or {}
    result = {name: str(data.get(name) or "").strip() for name in FIELDS}
    result["ollama_url"] = _normalize_url(result["ollama_url"])
    return result


def save_ai_credentials(values: dict[str, Any]) -> Path:
    """Merge ``values`` into the stored credentials and write the file owner-only; returns its path.

    Missing fields, ``None`` and empty key strings keep the stored value (use :func:`delete_ai_key` to remove a key).
    An empty ``ollama_url`` resets the address to ``http://localhost:11434``.
    """
    data = load_ai_credentials()
    for name in FIELDS:
        if name not in values or values[name] is None:
            continue
        text = str(values[name]).strip()
        if name == "ollama_url":
            data[name] = _normalize_url(text)
        elif text:
            data[name] = text
    path = secretfile.write_private_yaml(credentials_path(), data)
    LOGGER.info("AI assistant credentials saved (%s)", ", ".join(
        f"{name}={'set' if data[name] else 'empty'}" for name in KEY_FIELDS.values()))
    return path


def delete_ai_key(provider: str) -> bool:
    """Remove the stored key of ``provider`` (for ``"ollama"``: reset the address); False when nothing was stored."""
    path = credentials_path()
    if not path.is_file():
        return False
    data = load_ai_credentials()
    if provider == "ollama":
        changed = data["ollama_url"] != DEFAULT_OLLAMA_URL
        data["ollama_url"] = DEFAULT_OLLAMA_URL
    else:
        name = KEY_FIELDS.get(provider)
        if name is None:
            raise ValueError(f"unknown AI provider: {provider}")
        changed = bool(data[name])
        data[name] = ""
    if changed:
        secretfile.write_private_yaml(path, data)
    return changed


def masked_ai_credentials() -> dict[str, str]:
    """Copy safe for display and logs: keys masked (empty when not set), the Ollama address shown."""
    data = load_ai_credentials()
    for name in KEY_FIELDS.values():
        data[name] = mask(data[name]) if data[name] else ""
    return data


def key_for(provider: str, creds: dict[str, Any] | None = None) -> str:
    """The stored API key of ``provider`` ("" for Ollama or when none is stored)."""
    name = KEY_FIELDS.get(provider)
    if name is None:
        return ""
    source = creds if creds is not None else load_ai_credentials()
    return str(source.get(name) or "").strip()


def redact(text: Any, extra: Iterable[str] = ()) -> str:
    """``text`` with every stored key, the ``extra`` secrets and any ``sk-...``/``Bearer ...`` token replaced by ``***``."""
    result = str(text)
    secrets: set[str] = {str(item) for item in extra if item}
    try:
        stored = load_ai_credentials()
        secrets.update(stored[name] for name in KEY_FIELDS.values() if stored[name])
    except Exception:  # redaction must never fail
        pass
    for secret in sorted(secrets, key=len, reverse=True):
        if len(secret) >= _MIN_SECRET_LENGTH:
            result = result.replace(secret, "***")
    result = _TOKEN_PATTERNS[0].sub("Bearer ***", result)
    result = _TOKEN_PATTERNS[1].sub("***", result)
    result = _TOKEN_PATTERNS[2].sub(r"\1***", result)
    return result
