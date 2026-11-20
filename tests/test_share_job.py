"""Share job: phases in order, cancel, errors, stored result, delete and re-create."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from samcloud.core.project_manager import ProjectManager
from samcloud.share import export, job as share_job
from samcloud.share.job import ShareJob, ShareManager, delete_share, get_share, read_progress, share_available
from samcloud.share.providers import LocalProvider
from test_share_export import make_project, needs_converter

PHASE_KEYS = ["prepare", "octree", "assemble", "upload", "qr"]


def _job(tmp_path: Path, project: Path, **kwargs: Any) -> ShareJob:
    provider = kwargs.pop("provider", None) or LocalProvider(tmp_path / "site")
    config = {"provider": "local", "max_points": 8000, "octree_encoding": "BROTLI", "providers": {}}
    return ShareJob(project, config=config, provider=provider, project_manager=ProjectManager(tmp_path / "projects"),
                    **kwargs)


def _project_yaml(project: Path) -> dict:
    return yaml.safe_load((project / "project.samcloud.yml").read_text(encoding="utf-8"))


def _wait(predicate: Any, seconds: float = 20.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@needs_converter
def test_job_runs_all_phases_in_order_and_stores_the_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = make_project(tmp_path, points=20_000)
    snapshots: list[dict] = []
    original = share_job._ProgressWriter.update

    def recording(self: Any, **changes: Any) -> None:
        original(self, **changes)
        snapshots.append(dict(self.doc))

    monkeypatch.setattr(share_job._ProgressWriter, "update", recording)
    job = _job(tmp_path, project)
    block = job.run()
    assert block is not None
    progress = read_progress(project)
    assert progress["state"] == "done" and progress["percent"] == 100.0
    assert progress["phases_done"] == PHASE_KEYS
    percents = [s["percent"] for s in snapshots]
    assert percents == sorted(percents), "progress never goes backwards"
    assert [s["phase"] for s in snapshots if s["phase"]][0] == "Vorbereiten"
    order = [s["phase_index"] for s in snapshots]
    assert order == sorted(order)
    stored = _project_yaml(project)["share"]
    assert stored == {k: block[k] for k in stored} and set(stored) == {"url", "share_id", "created_at", "provider",
                                                                       "points", "size_mb"}
    assert stored["provider"] == "local" and 0 < stored["points"] <= 8000 and stored["size_mb"] > 1
    assert len(stored["share_id"]) >= 20
    assert (tmp_path / "site" / stored["share_id"] / "index.html").is_file()
    assert (project / "products" / "share" / "qr.png").is_file() and (project / "products" / "share" / "qr.svg").is_file()
    assert not (project / "products" / "share" / "package").exists(), "staging is removed"
    assert get_share(project)["url"] == stored["url"]


def test_job_ends_in_error_with_message_when_a_phase_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = make_project(tmp_path, points=2000)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Konverter explodiert")

    monkeypatch.setattr(export, "build_octree", boom)
    job = _job(tmp_path, project)
    assert job.run() is None
    progress = read_progress(project)
    assert progress["state"] == "error" and "Konverter explodiert" in progress["error"]
    assert progress["phases_done"] == ["prepare", "octree"]
    assert "share" not in _project_yaml(project)


def test_job_thread_error_state_is_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = make_project(tmp_path, points=2000)
    monkeypatch.setattr(export, "prepare_points", lambda *a, **k: (_ for _ in ()).throw(ValueError("kaputt")))
    job = _job(tmp_path, project).start()
    assert _wait(lambda: read_progress(project)["state"] == "error")
    job.thread.join(5)
    assert not job.is_alive() and "kaputt" in read_progress(project)["error"]


def test_job_refuses_before_step_6(tmp_path: Path) -> None:
    project = make_project(tmp_path, refinement="not_started", with_cloud=False)
    assert not share_available(project)
    job = _job(tmp_path, project)
    assert job.run() is None
    progress = read_progress(project)
    assert progress["state"] == "error" and "Schritt 6" in progress["error"]


def test_share_available_by_workflow_or_classified_file(tmp_path: Path) -> None:
    assert share_available(make_project(tmp_path, "a", refinement="completed", with_cloud=False))
    assert share_available(make_project(tmp_path, "b", points=100, refinement="running"))
    assert not share_available(make_project(tmp_path, "c", refinement="running", with_cloud=False))


def test_cancel_stops_the_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = make_project(tmp_path, points=2000)
    entered: list[bool] = []

    def slow_prepare(project_dir: Path, max_points: int, progress: Any, check: Any) -> None:
        entered.append(True)
        for _ in range(2000):
            check()
            time.sleep(0.01)
        raise AssertionError("cancel did not stop the phase")

    monkeypatch.setattr(export, "prepare_points", slow_prepare)
    job = _job(tmp_path, project).start()
    assert _wait(lambda: bool(entered))
    job.cancel()
    assert _wait(lambda: read_progress(project)["state"] == "cancelled", 5)
    job.thread.join(5)
    assert not job.is_alive()
    assert "share" not in _project_yaml(project)
    assert not (project / "products" / "share" / "qr.png").exists()


@needs_converter
def test_cancel_during_upload_rolls_back_the_published_package(tmp_path: Path) -> None:
    project = make_project(tmp_path, points=5000)
    holder: dict[str, ShareJob] = {}

    class CancellingProvider(LocalProvider):
        def upload(self, local_dir: Path, share_id: str, progress_cb: Any = None) -> str:
            holder["job"].cancel()
            return super().upload(local_dir, share_id, progress_cb)

    job = holder["job"] = _job(tmp_path, project, provider=CancellingProvider(tmp_path / "site"))
    assert job.run() is None
    assert read_progress(project)["state"] == "cancelled"
    assert not list((tmp_path / "site").glob("*")), "partial upload removed"
    assert "share" not in _project_yaml(project)


def test_privacy_scan_failure_blocks_the_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = make_project(tmp_path, points=3000)
    monkeypatch.setattr(export, "scan_package", lambda *a, **k: ["image file: x.jpg"])
    uploads: list[str] = []

    class Spy(LocalProvider):
        def upload(self, *args: Any, **kwargs: Any) -> str:
            uploads.append("called")
            return super().upload(*args, **kwargs)

    job = _job(tmp_path, project, provider=Spy(tmp_path / "site"))
    assert job.run() is None
    progress = read_progress(project)
    assert progress["state"] == "error" and "Datenschutz" in progress["error"] and not uploads


@needs_converter
def test_delete_share_removes_package_qr_and_block_and_recreate_replaces_old(tmp_path: Path) -> None:
    project = make_project(tmp_path, points=5000)
    provider = LocalProvider(tmp_path / "site")
    first = _job(tmp_path, project, provider=provider).run()
    assert first is not None
    config = {"provider": "local", "providers": {"local": {"directory": str(tmp_path / "site")}}}
    second = ShareJob(project, config={**config, "max_points": 8000}, provider=provider,
                      project_manager=ProjectManager(tmp_path / "projects")).run()
    assert second is not None and second["share_id"] != first["share_id"]
    assert not (tmp_path / "site" / first["share_id"]).exists(), "re-creating removes the old link"
    assert (tmp_path / "site" / second["share_id"]).is_dir()
    assert delete_share(project, config=config, provider=provider, project_manager=ProjectManager(tmp_path / "projects"))
    assert not (tmp_path / "site" / second["share_id"]).exists()
    assert get_share(project) is None and "share" not in _project_yaml(project)
    assert not (project / "products" / "share" / "qr.png").exists()
    assert not delete_share(project, config=config, provider=provider)


def test_manager_runs_one_job_per_project_and_flags_stale_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = make_project(tmp_path, points=2000)
    release: list[bool] = []

    def waiting(project_dir: Path, max_points: int, progress: Any, check: Any) -> None:
        while not release:
            check()
            time.sleep(0.01)
        raise RuntimeError("stop")

    monkeypatch.setattr(export, "prepare_points", waiting)
    manager = ShareManager()
    options = {"config": {"provider": "local", "max_points": 8000, "providers": {}},
               "provider": LocalProvider(tmp_path / "site"), "project_manager": ProjectManager(tmp_path / "projects")}
    first = manager.start(project, **options)
    assert manager.start(project, **options) is first
    assert manager.is_running(project) and manager.status(project)["state"] == "running"
    release.append(True)
    first.thread.join(5)
    assert manager.status(project)["state"] == "error" and not manager.is_running(project)
    assert not manager.cancel(project)
    # a "running" file without a live job (application restart) is reported as an error, not as running
    share_job._ProgressWriter(project).update(state="running")
    assert ShareManager().status(project)["state"] == "error"


@needs_converter
def test_expiring_provider_sets_expires_at_lifecycle_and_extend_restarts_it(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    class Expiring(LocalProvider):
        expiry_days = 15

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.lifecycle: list[int] = []
            self.extended: list[str] = []

        def ensure_lifecycle(self, days: int | None = None) -> None:
            self.lifecycle.append(int(days or 0))

        def extend(self, share_id: str) -> datetime:
            self.extended.append(share_id)
            return datetime.now(timezone.utc) + timedelta(days=15)

    project = make_project(tmp_path, points=4000)
    provider = Expiring(tmp_path / "site")
    block = _job(tmp_path, project, provider=provider).run()
    assert block is not None and provider.lifecycle == [15]
    created = datetime.fromisoformat(block["created_at"])
    assert datetime.fromisoformat(block["expires_at"]) - created == timedelta(days=15)
    assert share_job.share_state(block) == "active"
    assert share_job.share_state({**block, "expires_at": "2000-01-01T00:00:00+00:00"}) == "expired"
    assert share_job.share_state(None) == "none" and share_job.share_state({"url": "x"}) == "active"
    updated = share_job.extend_share(project, provider=provider, project_manager=ProjectManager(tmp_path / "projects"))
    assert provider.extended == [block["share_id"]] and updated["url"] == block["url"]
    assert _project_yaml(project)["share"]["expires_at"] == updated["expires_at"]
