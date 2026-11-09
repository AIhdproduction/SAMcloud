"""Cancellation and progress primitives shared by DXF processing operations."""

from __future__ import annotations

from threading import Event
from typing import Protocol


class Cancelled(RuntimeError):
    """Raised when a DXF processing operation is cancelled."""


class ProgressCallback(Protocol):
    """Receives a normalized progress fraction and a human-readable stage."""

    def __call__(self, fraction: float, stage: str) -> None:
        """Report progress for an explicit processing operation."""


class CancelToken:
    """Thread-safe cancellation token for explicit DXF processing operations."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`Cancelled` if cancellation was requested."""
        if self.is_cancelled():
            raise Cancelled("DXF processing was cancelled")


def check_cancel(cancel: CancelToken | None) -> None:
    """Check an optional cancellation token."""
    if cancel is not None:
        cancel.raise_if_cancelled()


def report_progress(progress: ProgressCallback | None, fraction: float, stage: str) -> None:
    """Report bounded progress when a callback was supplied."""
    if progress is not None:
        progress(min(1.0, max(0.0, fraction)), stage)
