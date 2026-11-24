"""Protocol background job: states, single running job, restart handling, cancel."""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from samcloud.report import job as job_module
from samcloud.report.job import ProtocolManager


def _fake(monkeypatch: pytest.MonkeyPatch, func: Any) -> None:
    module = types.ModuleType("samcloud.report.generate")
    module.generate_protocol = func
    monkeypatch.setitem(sys.modules, "samcloud.report.generate", module)


def _wait(manager: ProtocolManager, project: Path, timeout: float = 5.0) -> dict[str, Any]:
    end = time.time() + timeout
    while time.time() < end:
        status = manager.status(project)
        if status["state"] != "running":
            return status
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def generate(project_dir, *, language=None, options=None, progress=None, cancel_check=None, now=None):
        seen.update(language=language, options=options)
        progress(0.5, "charts")
        out = project_dir / "reports" / "protocol_x.pdf"
        out.write_bytes(b"%PDF")
        return out

    _fake(monkeypatch, generate)
    (tmp_path / "reports").mkdir()
    manager = ProtocolManager()
    manager.start(tmp_path, language="de", options={"camera_list": False})
    status = _wait(manager, tmp_path)
    assert status["state"] == "done" and status["result_path"].endswith("protocol_x.pdf")
    assert seen == {"language": "de", "options": {"camera_list": False}}
    assert (tmp_path / job_module.PROGRESS_RELATIVE).is_file()


def test_exception_sets_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def generate(project_dir, **kwargs):
        raise RuntimeError("boom")

    _fake(monkeypatch, generate)
    manager = ProtocolManager()
    manager.start(tmp_path)
    status = _wait(manager, tmp_path)
    assert status["state"] == "error" and "boom" in status["error"]


def test_second_start_returns_same_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def generate(project_dir, **kwargs):
        release.wait(5)
        return project_dir / "x.pdf"

    _fake(monkeypatch, generate)
    manager = ProtocolManager()
    first = manager.start(tmp_path)
    assert manager.start(tmp_path) is first and manager.is_running(tmp_path)
    release.set()
    assert _wait(manager, tmp_path)["state"] == "done"


def test_stale_running_file_becomes_error(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / job_module.PROGRESS_RELATIVE).write_text('{"state": "running", "fraction": 0.3}', encoding="utf-8")
    status = ProtocolManager().status(tmp_path)
    assert status["state"] == "error" and status["error"]


def test_cancel_between_phases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reached = threading.Event()
    proceed = threading.Event()

    def generate(project_dir, *, progress=None, cancel_check=None, **kwargs):
        progress(0.1, "collect")
        reached.set()
        proceed.wait(5)
        cancel_check()
        raise AssertionError("not cancelled")

    _fake(monkeypatch, generate)
    manager = ProtocolManager()
    manager.start(tmp_path)
    assert reached.wait(5)
    assert manager.cancel(tmp_path)
    proceed.set()
    assert _wait(manager, tmp_path)["state"] == "cancelled"
