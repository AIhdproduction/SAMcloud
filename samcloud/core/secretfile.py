"""Private per-user files outside the repository (credentials, user settings).

All files live in the SAMcloud user config directory: ``%APPDATA%\\SAMcloud`` on Windows (or whenever ``APPDATA`` is set),
else ``$XDG_CONFIG_HOME/samcloud`` or ``~/.config/samcloud``. Files written with :func:`write_private_yaml` are restricted to
the owner (chmod 600 / an ``icacls`` ACL without inherited rights). Secrets are never logged; use :func:`mask` for display.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

LOGGER = logging.getLogger(__name__)

PathHook = Callable[[Path], None]


def config_dir() -> Path:
    """The SAMcloud user config directory (not created)."""
    if sys.platform.startswith("win") or os.environ.get("APPDATA"):
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "SAMcloud"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "samcloud"


def restrict(path: Path) -> None:
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


def regrant(path: Path) -> None:
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


def read_yaml(path: Path) -> dict[str, Any] | None:
    """The mapping stored in ``path``, or ``None`` when missing, unreadable or not a mapping."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def write_private_yaml(path: Path, data: dict[str, Any], *, restrict_fn: PathHook | None = None,
                       regrant_fn: PathHook | None = None) -> Path:
    """Write ``data`` as UTF-8 YAML (key order kept) and restrict the file to the owner; returns ``path``.

    A file left with a too narrow ACL by an older version is regranted and written again.
    ``restrict_fn``/``regrant_fn`` replace :func:`restrict`/:func:`regrant` (used by modules that keep patchable aliases).
    """
    restrict_fn = restrict_fn or restrict
    regrant_fn = regrant_fn or regrant
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False)
    try:
        path.write_text(text, encoding="utf-8")
    except PermissionError:
        regrant_fn(path)
        path.write_text(text, encoding="utf-8")
    restrict_fn(path)
    return path


def delete_private(path: Path, *, regrant_fn: PathHook | None = None) -> bool:
    """Remove a private file; ``False`` when none existed."""
    regrant_fn = regrant_fn or regrant
    path = Path(path)
    if path.is_file():
        try:
            path.unlink()
        except PermissionError:
            regrant_fn(path)
            path.unlink()
        return True
    return False


def mask(secret: str, keep: int = 4) -> str:
    """``****abcd`` form of a secret for display and logs."""
    secret = str(secret or "")
    if keep <= 0 or len(secret) <= keep:
        return "*" * 8
    return "*" * 8 + secret[-keep:]
