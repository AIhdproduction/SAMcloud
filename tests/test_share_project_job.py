"""Share-job kind separation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import ezdxf
import pytest
import yaml

from samcloud.core.project_manager import ProjectManager

from samcloud.share import job
from samcloud.share.providers import LocalProvider


def test_project_kind_has_independent_storage_contract() -> None:
    assert job.PROJECT.key == "project_share"
    assert job.PROJECT.output_relative == Path("products") / "share" / "project"
    assert [key for key, _label, _weight in job.PROJECT.phases] == ["prepare", "octree", "dxf", "ifc", "assemble", "upload", "qr"]


def test_progress_and_result_accessors_are_kind_specific(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    job._ProgressWriter(tmp_path, job.PROJECT).update(state="done", percent=100)
    assert job.read_progress(tmp_path, kind=job.PROJECT)["state"] == "done"
    assert job.read_progress(tmp_path)["state"] == "idle"


def test_manager_keys_jobs_by_kind(tmp_path: Path, monkeypatch) -> None:
    manager = job.ShareManager()

    class Dummy:
        def __init__(self, directory, **options): self.directory, self.kind = directory, options.get("kind", job.POINTCLOUD)
        def start(self): return self
        def is_alive(self): return True
        def cancel(self): return None

    monkeypatch.setattr(job, "ShareJob", Dummy)
    pointcloud = manager.start(tmp_path)
    project = manager.start(tmp_path, kind=job.PROJECT)
    assert pointcloud is not project and manager.is_running(tmp_path) and manager.is_running(tmp_path, kind=job.PROJECT)


def test_project_job_stores_its_own_block_and_keeps_pointcloud_share(tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "demo"; (directory / "products" / "classified").mkdir(parents=True)
    (directory / "products" / "classified" / "classified.ply").write_bytes(b"ply")
    export_dir = directory / "products" / "dxf" / "export"; export_dir.mkdir(parents=True)
    drawing = ezdxf.new(); drawing.modelspace().add_line((0, 0), (1, 1)); drawing.saveas(export_dir / "demo_Site_clean.dxf")
    bim = directory / "products" / "bim" / "site"; bim.mkdir(parents=True); (bim / "model.glb").write_bytes(b"glTF")
    document = {"share": {"url": "point", "share_id": "old", "provider": "local"}}
    (directory / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")

    def build(_directory: Path, package: Path, *, progress_for_phase: Any, **_options: Any) -> dict[str, Any]:
        for index in range(5):  # the build phases; upload and qr are driven by the job
            progress_for_phase(index)(1.0, "x")
        package.mkdir(parents=True); (package / "index.html").write_text("ok", encoding="utf-8")
        return {"points": 1, "views": ["dxf", "pc", "ifc"], "levels": ["site"]}

    kind = replace(job.PROJECT, build=build)
    manager_data: dict[str, Any] = {}

    class Manager:
        def load(self, _directory: Path) -> dict[str, Any]: return dict(document)
        def save(self, _directory: Path, value: dict[str, Any]) -> None: manager_data.update(value)

    provider = LocalProvider(tmp_path / "site")
    result = job.ShareJob(directory, kind=kind, provider=provider, project_manager=Manager(), config={"max_points": 10}).run()
    assert result and manager_data["share"]["url"] == "point" and manager_data["project_share"]["views"] == ["dxf", "pc", "ifc"]
    assert (directory / "products" / "share" / "project" / "qr.png").is_file()
    stored = manager_data["project_share"]
    assert stored["url"] == result["url"] and stored["share_id"] == result["share_id"] and stored["levels"] == ["site"]
    assert (tmp_path / "site" / stored["share_id"] / "index.html").is_file()
    progress = job.read_progress(directory, kind=job.PROJECT)
    assert progress["state"] == "done" and progress["result"]["share_id"] == stored["share_id"]
    assert (directory / "reports" / "project_share_progress.json").is_file()
    assert progress["phases_done"] == [key for key, _label, _weight in job.PROJECT.phases]


def _project(root: Path, *, outputs: bool = True, extra: dict[str, Any] | None = None) -> Path:
    directory = root / "demo"; (directory / "products" / "classified").mkdir(parents=True)
    (directory / "products" / "classified" / "classified.ply").write_bytes(b"ply")
    if outputs:
        export_dir = directory / "products" / "dxf" / "export"; export_dir.mkdir(parents=True)
        drawing = ezdxf.new(); drawing.modelspace().add_line((0, 0), (1, 1)); drawing.saveas(export_dir / "demo_Site_clean.dxf")
        bim = directory / "products" / "bim" / "site"; bim.mkdir(parents=True); (bim / "model.glb").write_bytes(b"glTF")
    document: dict[str, Any] = {"project": {"name": "demo"}, **(extra or {})}
    (directory / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return directory


def _fake_build(_directory: Path, package: Path, *, progress_for_phase: Any, check: Any, **_options: Any) -> dict[str, Any]:
    for index in range(5):
        progress_for_phase(index)(1.0, "x")
        check()
    package.mkdir(parents=True); (package / "index.html").write_text("ok", encoding="utf-8")
    return {"points": 1, "views": ["dxf", "pc", "ifc"], "levels": ["site"]}


def _yaml(directory: Path) -> dict[str, Any]:
    return yaml.safe_load((directory / "project.samcloud.yml").read_text(encoding="utf-8"))


def test_pointcloud_and_project_shares_are_independent_on_delete(tmp_path: Path) -> None:
    share_id = "abcdefghijklmnopqrstuvwx"
    directory = _project(tmp_path, extra={"share": {"url": "point", "share_id": share_id, "provider": "local"}})
    (directory / "products" / "share").mkdir(parents=True); (directory / "products" / "share" / "qr.png").write_bytes(b"png")
    manager = ProjectManager(tmp_path / "projects"); provider = LocalProvider(tmp_path / "site")
    kind = replace(job.PROJECT, build=_fake_build)
    assert job.ShareJob(directory, kind=kind, provider=provider, project_manager=manager, config={"max_points": 10}).run()
    project_qr = directory / "products" / "share" / "project" / "qr.png"
    assert project_qr.is_file() and "project_share" in _yaml(directory)

    assert job.delete_share(directory, kind=job.PROJECT, provider=provider, project_manager=manager, config={})
    document = _yaml(directory)
    assert "project_share" not in document and document["share"]["share_id"] == share_id
    assert not project_qr.exists() and (directory / "products" / "share" / "qr.png").is_file()

    # reverse: a project share survives the deletion of the point-cloud share
    assert job.ShareJob(directory, kind=kind, provider=provider, project_manager=manager, config={"max_points": 10}).run()
    assert job.delete_share(directory, provider=provider, project_manager=manager, config={})
    document = _yaml(directory)
    assert "share" not in document and document["project_share"]["url"]
    assert project_qr.is_file() and not (directory / "products" / "share" / "qr.png").exists()


def test_project_job_cancelled_during_octree_publishes_nothing(tmp_path: Path) -> None:
    directory = _project(tmp_path)
    holder: list[job.ShareJob] = []

    def build(_directory: Path, package: Path, *, progress_for_phase: Any, check: Any, **_options: Any) -> dict[str, Any]:
        progress_for_phase(0)(1.0, "x")
        progress_for_phase(1)(0.5, "octree")
        holder[0].cancel()
        check()
        raise AssertionError("cancel did not stop the build")

    provider = LocalProvider(tmp_path / "site")
    manager = ProjectManager(tmp_path / "projects")
    running = job.ShareJob(directory, kind=replace(job.PROJECT, build=build), provider=provider, project_manager=manager, config={"max_points": 10})
    holder.append(running)
    assert running.run() is None
    assert job.read_progress(directory, kind=job.PROJECT)["state"] == "cancelled"
    assert not (tmp_path / "site").exists() or not any((tmp_path / "site").iterdir())
    assert "project_share" not in _yaml(directory)


def test_project_job_refuses_without_dxf_and_glb(tmp_path: Path) -> None:
    directory = _project(tmp_path, outputs=False)
    provider = LocalProvider(tmp_path / "site")
    result = job.ShareJob(directory, kind=job.PROJECT, provider=provider, project_manager=ProjectManager(tmp_path / "projects"), config={"max_points": 10}).run()
    progress = job.read_progress(directory, kind=job.PROJECT)
    assert result is None and progress["state"] == "error" and "DXF" in progress["error"]
    assert "project_share" not in _yaml(directory)


def test_manager_keeps_pointcloud_and_project_jobs_side_by_side(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = job.ShareManager()

    class Alive:
        def __init__(self, directory, **options): self.kind = options.get("kind", job.POINTCLOUD)
        def start(self): return self
        def is_alive(self): return True
        def cancel(self): self.cancelled = True

    monkeypatch.setattr(job, "ShareJob", Alive)
    first = manager.start(tmp_path); second = manager.start(tmp_path, kind=job.PROJECT)
    assert first is not second and first.kind is job.POINTCLOUD and second.kind is job.PROJECT
    assert manager.start(tmp_path) is first and manager.start(tmp_path, kind=job.PROJECT) is second
    assert manager.cancel(tmp_path, kind=job.PROJECT) and getattr(second, "cancelled", False) and not hasattr(first, "cancelled")
