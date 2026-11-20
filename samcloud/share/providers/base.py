"""Interface of the upload providers that publish a static share package."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[int, int], None]
_SHARE_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class ProviderError(RuntimeError):
    """An upload or delete operation failed; the message is shown to the user."""


def validate_share_id(share_id: str) -> str:
    """Return ``share_id`` when it is a safe token (no path separators), else raise ``ProviderError``."""
    if not isinstance(share_id, str) or not _SHARE_ID.fullmatch(share_id):
        raise ProviderError("invalid share id")
    return share_id


def directory_size(local_dir: Path) -> int:
    """Total size in bytes of all files below ``local_dir``."""
    return sum(path.stat().st_size for path in Path(local_dir).rglob("*") if path.is_file())


class ShareProvider(ABC):
    """Publishes a package directory under a secret ``share_id`` and can remove it again."""

    name = "base"
    expiry_days: int | None = None  # lifetime of a published package; ``None`` = unlimited

    @abstractmethod
    def upload(self, local_dir: Path, share_id: str, progress_cb: ProgressCallback | None = None) -> str:
        """Publish ``local_dir`` (contains index.html) and return the public URL of the viewer.

        ``progress_cb(bytes_done, bytes_total)`` is called while data is transferred.
        """

    @abstractmethod
    def delete(self, share_id: str) -> None:
        """Remove the published package; unknown ids are ignored."""

    def extend(self, share_id: str) -> "datetime":
        """Restart the lifetime of a published package without re-uploading; returns the new expiry."""
        raise ProviderError("extending is not supported by this provider")

    def ensure_lifecycle(self, days: int | None = None) -> str:
        """Make sure the storage deletes packages after ``days``; providers without a lifecycle do nothing."""
        return "present"
