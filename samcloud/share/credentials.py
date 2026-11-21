"""Cloudflare R2 credentials of the SAMcloud user, stored outside the repository.

File: ``%APPDATA%\\SAMcloud\\share_credentials.yaml`` (Windows) or ``~/.config/samcloud/share_credentials.yaml``.
The secret is never logged; use ``mask`` for any display.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)
FIELDS = ("account_id", "access_key_id", "secret_access_key", "bucket", "public_base_url")
FILE_NAME = "share_credentials.yaml"


def credentials_path() -> Path:
    """Location of the credentials file (not created)."""
    if sys.platform.startswith("win") or os.environ.get("APPDATA"):
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "SAMcloud" / FILE_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "samcloud" / FILE_NAME


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
    path = credentials_path()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict) or not is_complete(data):
        return None
    return normalize(data)


def _restrict(path: Path) -> None:
    """Owner-only access: chmod 600 (POSIX) or an ACL without inherited rights (Windows)."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    if os.name == "nt":
        import subprocess

        user = os.environ.get("USERNAME")
        if user:
            try:
                subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(F)"],
                               capture_output=True, check=False, timeout=15, creationflags=0x08000000)
            except (OSError, subprocess.SubprocessError):
                LOGGER.info("could not restrict permissions of the credentials file")


def _regrant(path: Path) -> None:
    """Restore full owner rights on a file written with an older, too narrow ACL."""
    if os.name == "nt":
        import subprocess

        user = os.environ.get("USERNAME")
        if user:
            try:
                subprocess.run(["icacls", str(path), "/grant:r", f"{user}:(F)"],
                               capture_output=True, check=False, timeout=15, creationflags=0x08000000)
            except (OSError, subprocess.SubprocessError):
                LOGGER.info("could not regrant permissions of the credentials file")
    else:
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass


def save_credentials(values: dict[str, Any]) -> Path:
    """Validate and write the credentials file with restrictive permissions; returns its path."""
    data = normalize(values)
    if not is_complete(data):
        raise ValueError("incomplete credentials")
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False)
    try:
        path.write_text(text, encoding="utf-8")
    except PermissionError:
        _regrant(path)
        path.write_text(text, encoding="utf-8")
    _restrict(path)
    return path


def delete_credentials() -> bool:
    """Remove the credentials file; False when none existed."""
    path = credentials_path()
    if path.is_file():
        try:
            path.unlink()
        except PermissionError:
            _regrant(path)
            path.unlink()
        return True
    return False


def mask(secret: str, keep: int = 4) -> str:
    """``****abcd`` form of a secret for display and logs."""
    secret = str(secret or "")
    if keep <= 0 or len(secret) <= keep:
        return "*" * 8
    return "*" * 8 + secret[-keep:]


def masked(values: dict[str, Any]) -> dict[str, str]:
    """Copy of the credentials safe for logging (secret and key id masked)."""
    data = normalize(values)
    data["secret_access_key"] = mask(data["secret_access_key"], 0)
    data["access_key_id"] = mask(data["access_key_id"])
    return data
