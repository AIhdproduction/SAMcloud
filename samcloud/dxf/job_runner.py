"""Subprocess runner for DXF jobs with cooperative cancellation and process termination."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from samcloud.core.paths import PROJECT_ROOT
from samcloud.dxf.jobs import replace_with_retry

_WINDOWS_NO_CONSOLE = 0x08000000


def is_process_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still active without psutil."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


class DxfJobRunner:
    """Manages execution and lifecycle of DXF subprocess jobs for a single project."""

    def __init__(self, project_dir: Path | str) -> None:
        self.project_dir = Path(project_dir).resolve()
        self._process: subprocess.Popen[str] | None = None

    @property
    def dxf_directory(self) -> Path:
        return self.project_dir / "products" / "dxf"

    @property
    def status_file(self) -> Path:
        return self.dxf_directory / "job_status.json"

    @property
    def cancel_flag(self) -> Path:
        return self.dxf_directory / "cancel.flag"

    @property
    def log_file(self) -> Path:
        return self.dxf_directory / "job.log"

    def is_running(self) -> bool:
        """Return True if a DXF job subprocess is actively executing."""
        if self._process is not None:
            if self._process.poll() is None:
                return True
            self._process = None
        current_status = self.status()
        if current_status.get("state") == "running":
            pid = current_status.get("pid")
            if isinstance(pid, int) and is_process_alive(pid):
                return True
        return False

    def status(self) -> dict[str, Any]:
        """Return the current parsed job_status.json payload or idle defaults."""
        if not self.status_file.is_file():
            return {
                "action": "",
                "state": "idle",
                "fraction": 0.0,
                "stage": "",
                "message": "",
                "started": "",
                "finished": None,
                "pid": None,
            }
        try:
            data = json.loads(self.status_file.read_text(encoding="utf-8"))
            if data.get("state") == "running" and self._process is not None:
                code = self._process.poll()
                if code is not None:
                    data["state"] = "cancelled" if code == 2 else ("done" if code == 0 else "failed")
                    data["finished"] = datetime.now(timezone.utc).isoformat()
            return data
        except Exception:
            return {
                "action": "",
                "state": "unknown",
                "fraction": 0.0,
                "stage": "Reading status failed",
                "message": "",
                "started": "",
                "finished": None,
                "pid": None,
            }

    def start(
        self,
        action: str,
        las: Path | str,
        level: str | None = None,
        mode: str | None = None,
    ) -> subprocess.Popen[str]:
        """Launch a background DXF job subprocess with sys.executable (only one at a time)."""
        if self.is_running():
            raise RuntimeError(f"A DXF job is already running for project '{self.project_dir.name}'")

        if self.cancel_flag.exists():
            try:
                self.cancel_flag.unlink()
            except OSError:
                pass

        command = [
            sys.executable,
            "-m",
            "samcloud.dxf.jobs",
            str(action),
            "--project",
            str(self.project_dir),
            "--las",
            str(Path(las).resolve()),
        ]
        if level is not None:
            command.extend(["--level", str(level)])
        if mode is not None:
            command.extend(["--mode", str(mode)])
        launched = time.time()
        command.extend(["--launched", repr(launched)])

        # Replace the previous job's final status before launching so pollers never mistake the
        # last run's "done" for the new job; the subprocess overwrites this as soon as it starts.
        self._write_status(
            {
                "action": str(action),
                "state": "running",
                "fraction": 0.0,
                "stage": "Starting",
                "message": "",
                "started": datetime.now(timezone.utc).isoformat(),
                "finished": None,
                "pid": None,
            }
        )

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self.log_file.open("a", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=_WINDOWS_NO_CONSOLE if sys.platform == "win32" else 0,
            )
        finally:
            log_handle.close()

        return self._process

    def cancel(self) -> None:
        """Request cooperative cancellation, then terminate after 5 s and kill after another 5 s."""
        self.cancel_flag.parent.mkdir(parents=True, exist_ok=True)
        self.cancel_flag.write_text("cancel", encoding="utf-8")

        proc = self._process
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            self._process = None
        else:
            status = self.status()
            if status.get("state") == "running":
                pid = status.get("pid")
                if isinstance(pid, int) and is_process_alive(pid):
                    start_time = time.monotonic()
                    while time.monotonic() - start_time < 5.0:
                        if not is_process_alive(pid):
                            break
                        time.sleep(0.2)
                    if is_process_alive(pid):
                        if sys.platform == "win32":
                            subprocess.run(["taskkill", "/PID", str(pid), "/T"], capture_output=True)
                            wait_start = time.monotonic()
                            while time.monotonic() - wait_start < 5.0:
                                if not is_process_alive(pid):
                                    break
                                time.sleep(0.2)
                            if is_process_alive(pid):
                                subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], capture_output=True)
                        else:
                            try:
                                os.kill(pid, signal.SIGTERM)
                            except OSError:
                                pass
                            wait_start = time.monotonic()
                            while time.monotonic() - wait_start < 5.0:
                                if not is_process_alive(pid):
                                    break
                                time.sleep(0.2)
                            if is_process_alive(pid):
                                try:
                                    os.kill(pid, signal.SIGKILL)
                                except OSError:
                                    pass

        # Ensure status file reflects cancellation if not already updated
        current = self.status()
        if current.get("state") == "running":
            current["state"] = "cancelled"
            current["stage"] = "Cancelled"
            current["message"] = "Job was cancelled."
            current["finished"] = datetime.now(timezone.utc).isoformat()
            try:
                self._write_status(current)
            except Exception:
                pass

    def _write_status(self, payload: dict[str, Any]) -> None:
        """Atomically replace job_status.json."""
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="job_status_", suffix=".tmp", dir=self.status_file.parent)
        try:
            with open(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            replace_with_retry(tmp_path, self.status_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def log_tail(self, n: int = 50) -> list[str]:
        """Return the last n lines of products/dxf/job.log."""
        if not self.log_file.is_file():
            return []
        try:
            lines = self.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            return lines[-max(1, int(n)) :]
        except Exception:
            return []
