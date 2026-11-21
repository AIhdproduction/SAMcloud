"""Background job for the outdoor environment chain, controlled purely through files.

The UI never talks to the worker process directly: ``start_environment_job`` launches
``python -m samcloud.drone.run_environment`` and records the run in
``products/environment/job_status.json``; ``environment_job_status`` merges that file with the
chain's own ``state.json`` (and the QA report) so a page only has to poll files.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from samcloud.core.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Display order of the chain (mirrors ``run_environment.STEPS``; kept local so the UI need not import the heavy chain).
ENVIRONMENT_STEPS: tuple[str, ...] = (
    "load", "semantics", "terrain", "vegetation", "pavement", "tin",
    "buildings", "site", "scene", "export", "qa", "report",
)
_TERMINAL = {"done", "failed", "skipped"}
_WINDOWS_NO_CONSOLE = 0x08000000
_ERROR_MAX_CHARS = 140

_handles: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


def environment_dir(project_dir: Path | str) -> Path:
    """Folder holding the chain's state, QA report and the job status."""
    return Path(project_dir) / "products" / "environment"


def job_status_path(project_dir: Path | str) -> Path:
    """Location of ``job_status.json``."""
    return environment_dir(project_dir) / "job_status.json"


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pid_alive(pid: int) -> bool:
    """True while a process with ``pid`` exists (no psutil needed)."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
            return str(pid) in result.stdout
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _short(text: Any) -> str:
    one_line = " ".join(str(text or "").split())
    return one_line if len(one_line) <= _ERROR_MAX_CHARS else one_line[: _ERROR_MAX_CHARS - 1] + "…"


def build_command(project_dir: Path | str, rerun_all: bool) -> list[str]:
    """Command line of the chain subprocess; ``--no-resume`` only for a full recompute."""
    command = [sys.executable, "-m", "samcloud.drone.run_environment", "--project", str(project_dir)]
    if rerun_all:
        command.append("--no-resume")
    return command


def _is_running(record: dict[str, Any]) -> bool:
    return record.get("state") == "running" and _pid_alive(int(record.get("pid") or 0))


def _watch(project_dir: Path, process: subprocess.Popen, started_at: str) -> None:
    """Record the exit of ``process`` in job_status.json (daemon thread)."""
    code = process.wait()
    with _lock:
        _handles.pop(str(project_dir), None)
        record = _read_json(job_status_path(project_dir), {}) or {}
        if record.get("started_at") != started_at:
            return
        record.update(state="finished" if code == 0 else "failed", returncode=code, finished_at=_now(),
                      error=None if code == 0 else f"exit code {code}")
        _write_json(job_status_path(project_dir), record)


def start_environment_job(project_dir: Path | str, rerun_all: bool = False) -> dict[str, Any]:
    """Start the environment chain in a background subprocess and return the new job record.

    Only one job per project may run: raises ``RuntimeError`` while ``job_status.json`` says running and its pid is alive.
    """
    project = Path(project_dir)
    with _lock:
        current = _read_json(job_status_path(project), {}) or {}
        if _is_running(current):
            raise RuntimeError("environment job already running")
        out = environment_dir(project)
        out.mkdir(parents=True, exist_ok=True)
        command = build_command(project, rerun_all)
        started_at = _now()
        with (out / "job.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
                creationflags=_WINDOWS_NO_CONSOLE if sys.platform == "win32" else 0,
            )
        record = {"state": "running", "pid": process.pid, "started_at": started_at, "finished_at": None,
                  "rerun_all": bool(rerun_all), "returncode": None, "error": None, "command": command}
        _write_json(job_status_path(project), record)
        _handles[str(project)] = process
    threading.Thread(target=_watch, args=(project, process, started_at), daemon=True).start()
    return record


def is_environment_job_running(project_dir: Path | str) -> bool:
    """True while this project's environment job is running."""
    return _is_running(_read_json(job_status_path(project_dir), {}) or {})


def qa_summary(project_dir: Path | str) -> dict[str, Any] | None:
    """``{passed, total, failed}`` from products/environment/qa_cad_report.json, or None when absent."""
    results = _read_json(environment_dir(project_dir) / "qa_cad_report.json")
    if not isinstance(results, list):
        return None
    rows = [item for item in results if isinstance(item, dict)]
    failed = [str(item.get("rule", "?")) for item in rows if not item.get("ok")]
    return {"passed": len(rows) - len(failed), "total": len(rows), "failed": failed}


def environment_job_status(project_dir: Path | str) -> dict[str, Any]:
    """Merged job + chain status.

    Keys: ``state`` (idle / running / finished / failed), ``running``, ``pid``, ``started_at``, ``finished_at``,
    ``rerun_all``, ``error``, ``steps`` (one ``{name, status, seconds, error}`` per chain step; status is done / failed /
    skipped / running / pending), ``progress`` (0..1) and ``qa``.
    """
    project = Path(project_dir)
    record = _read_json(job_status_path(project), {}) or {}
    state_path = environment_dir(project) / "state.json"
    raw = _read_json(state_path, {}) or {}
    raw_steps = raw.get("steps", {}) if isinstance(raw, dict) else {}
    state = str(record.get("state") or "idle")
    running = _is_running(record)
    if state == "running" and not running:
        # The worker vanished without the watcher recording its exit (e.g. the app restarted).
        finished = raw_steps.get("report", {}).get("status") in ("done", "skipped")
        state = "finished" if finished else "failed"
        record = {**record, "error": None if finished else "interrupted"}
    if running and record.get("rerun_all"):
        # A full recompute starts from an empty state; ignore the previous run's leftovers until it rewrites the file.
        try:
            stale = state_path.stat().st_mtime < datetime.fromisoformat(str(record["started_at"])).timestamp()
        except (OSError, KeyError, ValueError):
            stale = False
        if stale:
            raw_steps = {}
    steps: list[dict[str, Any]] = []
    active_marked = False
    for name in ENVIRONMENT_STEPS:
        item = raw_steps.get(name) if isinstance(raw_steps.get(name), dict) else {}
        status = str(item.get("status") or "pending")
        if status not in _TERMINAL:
            status = "pending"
            if running and not active_marked:
                status, active_marked = "running", True
        error = _short(item["error"]) if status == "failed" and item.get("error") else (
            _short(item["reason"]) if status == "skipped" and item.get("reason") and "dependency" in str(item["reason"]) else "")
        seconds = item.get("seconds")
        steps.append({"name": name, "status": status, "seconds": float(seconds) if isinstance(seconds, (int, float)) else None,
                      "error": error})
    finished_count = sum(1 for item in steps if item["status"] in _TERMINAL)
    return {
        "state": state, "running": running, "pid": record.get("pid"), "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"), "rerun_all": bool(record.get("rerun_all")),
        "error": record.get("error"), "steps": steps, "progress": finished_count / len(ENVIRONMENT_STEPS),
        "qa": qa_summary(project),
    }
