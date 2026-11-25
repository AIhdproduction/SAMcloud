"""Cloudflare R2 credentials of the SAMcloud user, stored outside the repository.

File: ``%APPDATA%\\SAMcloud\\share_credentials.yaml`` (Windows) or ``~/.config/samcloud/share_credentials.yaml``.
The secret is never logged; use ``mask`` for any display. File helpers live in :mod:`samcloud.core.secretfile`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from samcloud.core import secretfile
from samcloud.core.secretfile import mask

LOGGER = logging.getLogger(__name__)
FIELDS = ("account_id", "access_key_id", "secret_access_key", "bucket", "public_base_url")
FILE_NAME = "share_credentials.yaml"

__all__ = ["FIELDS", "FILE_NAME", "credentials_path", "normalize", "is_complete", "load_credentials",
           "save_credentials", "delete_credentials", "mask", "masked"]


def credentials_path() -> Path:
    """Location of the credentials file (not created)."""
    return secretfile.config_dir() / FILE_NAME


def normalize(values: dict[str, Any]) -> dict[str, str]:
    """Strip whitespace; ``public_base_url`` loses a trailing slash."""
    data = {field: str(values.get(field) or "").strip() for field in FIELDS}
    data["public_base_url"] = data["public_base_url"].rstrip("/")
    return data


def is_complete(values: dict[str, Any] | None) -> bool:
    """True when all five fields are set and the public URL is http(s)."""
    if not values:
        return False
    data = normalize(values)
    return all(data.values()) and data["public_base_url"].lower().startswith(("http://", "https://"))


def load_credentials() -> dict[str, str] | None:
    """The stored credentials, or ``None`` when missing, unreadable or incomplete."""
    data = secretfile.read_yaml(credentials_path())
    if data is None or not is_complete(data):
        return None
    return normalize(data)


def _restrict(path: Path) -> None:
    """Owner-only access (alias of :func:`samcloud.core.secretfile.restrict`, kept patchable)."""
    secretfile.restrict(path)


def _regrant(path: Path) -> None:
    """Restore full owner rights (alias of :func:`samcloud.core.secretfile.regrant`, kept patchable)."""
    secretfile.regrant(path)


def save_credentials(values: dict[str, Any]) -> Path:
    """Validate and write the credentials file with restrictive permissions; returns its path."""
    data = normalize(values)
    if not is_complete(data):
        raise ValueError("incomplete credentials")
    # The aliases are looked up at call time so tests may monkeypatch them.
    return secretfile.write_private_yaml(credentials_path(), data,
                                         restrict_fn=lambda path: _restrict(path),
                                         regrant_fn=lambda path: _regrant(path))


def delete_credentials() -> bool:
    """Remove the credentials file; False when none existed."""
    return secretfile.delete_private(credentials_path(), regrant_fn=lambda path: _regrant(path))


def masked(values: dict[str, Any]) -> dict[str, str]:
    """Copy of the credentials safe for logging (secret and key id masked)."""
    data = normalize(values)
    data["secret_access_key"] = mask(data["secret_access_key"], 0)
    data["access_key_id"] = mask(data["access_key_id"])
    return data
