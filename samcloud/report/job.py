"""Background job for the processing protocol; progress in ``reports/protocol_progress.json``.

Modelled on ``samcloud.share.job``: a daemon thread keeps the NiceGUI loop free and the UI polls the file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROGRESS_RELATIVE = Path("reports") / "protocol_progress.json"


class ProtocolCancelled(Exception):
    """Raised by ``ProtocolJob.check`` after a cancel request."""


def read_progress(project_dir: Path | str) -> dict[str, Any]:
    try:
        data = json.loads((Path(project_dir) / PROGRESS_RELATIVE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"state": "idle", "fraction": 0.0}
    return data if isinstance(data, dict) else {"state": "idle", "fraction": 0.0}


class _ProgressWriter:
    """Atomic, thread-safe progress document."""

    def __init__(self, project_dir: Path) -> None:
        self.path = project_dir / PROGRESS_RELATIVE
        self.lock = threading.Lock()
        self.doc: dict[str, Any] = {"state": "running", "fraction": 0.0, "phase": "", "message": "", "result_path": None,
                                    "error": None, "started_at": time.time(), "finished_at": None}

    def update(self, **changes: Any) -> None:
        with self.lock:
            self.doc.update(changes)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.doc, ensure_ascii=False), encoding="utf-8")
            for attempt in range(40):  # Windows refuses the replace while a poller has the file open
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == 39:
                        raise
                    time.sleep(0.025)


class ProtocolJob:
    def __init__(self, project_dir: Path | str, *, language: str | None = None, options: dict[str, Any] | None = None) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.language = language
        self.options = options
        self.cancel_event = threading.Event()
        self.progress = _ProgressWriter(self.project_dir)
        self.thread: threading.Thread | None = None

    def cancel(self) -> None:
        self.cancel_event.set()

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise ProtocolCancelled()

    def start(self) -> "ProtocolJob":
        self.progress.update(state="running")
        self.thread = threading.Thread(target=self.run, name="protocol-job", daemon=True)
        self.thread.start()
        return self

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _report(self, fraction: float, message: str = "") -> None:
        """Progress callback of the generator; ``message`` is the phase key (collect, charts, pdf) or free text."""
        self.check()
        fraction = min(max(float(fraction), 0.0), 1.0)
        changes: dict[str, Any] = {"fraction": fraction, "message": message}
        key = message.split(":", 1)[0].strip().lower() if message else ""
        if key in ("collect", "charts", "pdf"):
            changes["phase"] = key
        self.progress.update(**changes)

    def run(self) -> Path | None:
        """Never raises; always ends in done, error or cancelled."""
        try:
            self.progress.update(state="running", started_at=time.time(), error=None, result_path=None)
            from samcloud.report.generate import generate_protocol  # lazy: heavy imports (charts, pdf)

            path = generate_protocol(self.project_dir, language=self.language, options=self.options,
                                     progress=self._report, cancel_check=self.check)
            self.progress.update(state="done", fraction=1.0, message="done", result_path=str(path), finished_at=time.time())
            return Path(path)
        except ProtocolCancelled:
            self.progress.update(state="cancelled", message="cancelled", finished_at=time.time())
        except Exception as error:
            if self.cancel_event.is_set():
                self.progress.update(state="cancelled", message="cancelled", finished_at=time.time())
            else:
                LOGGER.exception("protocol job failed")
                self.progress.update(state="error", error=str(error) or error.__class__.__name__,
                                     message="error", finished_at=time.time())
        return None


class ProtocolManager:
    """Keeps the running job per project."""

    def __init__(self) -> None:
        self._jobs: dict[Path, ProtocolJob] = {}
        self._lock = threading.Lock()

    def is_running(self, project_dir: Path | str) -> bool:
        with self._lock:
            job = self._jobs.get(Path(project_dir).resolve())
        return job is not None and job.is_alive()

    def start(self, project_dir: Path | str, **options: Any) -> ProtocolJob:
        directory = Path(project_dir).resolve()
        with self._lock:
            current = self._jobs.get(directory)
            if current is not None and current.is_alive():
                return current
            job = ProtocolJob(directory, **options)
            self._jobs[directory] = job
        return job.start()

    def cancel(self, project_dir: Path | str) -> bool:
        with self._lock:
            job = self._jobs.get(Path(project_dir).resolve())
        if job is None or not job.is_alive():
            return False
        job.cancel()
        return True

    def status(self, project_dir: Path | str) -> dict[str, Any]:
        progress = read_progress(project_dir)
        if progress.get("state") == "running" and not self.is_running(project_dir):
            progress = {**progress, "state": "error", "error": "Der Vorgang wurde unterbrochen."}
        return progress


protocol_manager = ProtocolManager()
