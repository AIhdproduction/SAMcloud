"""Project-share controls on the dashboard and in the shared dialog."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from samcloud.app import i18n
from samcloud.app.pages import home
from samcloud.share import ui as share_ui
from samcloud.share.job import POINTCLOUD, PROJECT
from test_apriltag_gui import _element, _page_user
from test_share_export import make_project


CREDS = {
    "account_id": "acc123",
    "access_key_id": "AKIAEXAMPLEKEY1234",
    "secret_access_key": "s3cr3t",
    "bucket": "my-bucket",
    "public_base_url": "https://pub-abc.r2.dev",
}


def _home_route() -> None:
    home.render()


_home_route.__module__ = "tests.share_project_ui_route"


def _markers(user: Any, marker: str) -> int:
    try:
        return len(user.find(marker=marker).elements)
    except AssertionError:
        return 0


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.delenv("SAMCLOUD_SHARE_PROVIDER", raising=False)


def _use_english(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(i18n, "_read_preferences", lambda: {})
    monkeypatch.setattr(i18n, "_read_config_language", lambda: "en")
    i18n.reload_translations()


def _complete_project(tmp_path: Path, name: str = "complete") -> Path:
    project = make_project(tmp_path, name, with_cloud=False)
    (project / "products" / "classified" / "classified.ply").write_bytes(b"ply")
    dxf = project / "products" / "dxf" / "export"
    dxf.mkdir(parents=True)
    (dxf / "complete_Site_clean.dxf").write_bytes(b"dxf")
    bim = project / "products" / "bim" / "site"
    bim.mkdir(parents=True)
    (bim / "x_site.ifc").write_bytes(b"ifc")
    (bim / "model.glb").write_bytes(b"glTF")
    path = project / "project.samcloud.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document.setdefault("workflow", {}).update({"classification": "completed", "refinement": "completed"})
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return project


def _document(project: Path) -> dict[str, Any]:
    return yaml.safe_load((project / "project.samcloud.yml").read_text(encoding="utf-8"))


@pytest.mark.anyio
async def test_home_hides_project_share_without_dxf_and_ifc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    with monkeypatch.context() as language:
        _use_english(language)
        project = make_project(tmp_path, "incomplete", with_cloud=False)
        async with _page_user("/", _home_route, project, monkeypatch) as user:
            await asyncio.sleep(0.3)
            assert _markers(user, "project-share-button") == 0
    i18n.reload_translations()


@pytest.mark.anyio
async def test_home_shows_project_share_for_complete_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    with monkeypatch.context() as language:
        _use_english(language)
        project = _complete_project(tmp_path)
        async with _page_user("/", _home_route, project, monkeypatch) as user:
            await asyncio.sleep(0.3)
            assert _element(user, "project-share-button").text == "Share project"
    i18n.reload_translations()


@pytest.mark.anyio
async def test_home_shows_project_share_without_storage_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    with monkeypatch.context() as language:
        _use_english(language)
        project = _complete_project(tmp_path)
        assert not (project / "reports" / "storage_summary.json").exists()
        async with _page_user("/", _home_route, project, monkeypatch) as user:
            await asyncio.sleep(0.3)
            assert _markers(user, "project-share-button") == 1
    i18n.reload_translations()


@pytest.mark.anyio
async def test_project_share_opens_setup_then_project_dialog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(share_ui, "needs_credentials", lambda: True)
    with monkeypatch.context() as language:
        _use_english(language)
        project = _complete_project(tmp_path)
        async with _page_user("/", _home_route, project, monkeypatch) as user:
            await asyncio.sleep(0.3)
            user.find(marker="project-share-button").click()
            await user.should_see(marker="share-setup-dialog")
            for key, value in CREDS.items():
                with user:
                    _element(user, f"share-field-{key}").value = value
            user.find(marker="share-save").click()
            await user.should_see(marker="share-dialog")
            await user.should_see("Share project via QR code")
    i18n.reload_translations()


@pytest.mark.anyio
async def test_project_share_uses_own_privacy_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(share_ui, "needs_credentials", lambda: False)
    started: list[Any] = []
    monkeypatch.setattr(share_ui.share_manager, "start", lambda directory, **options: started.append((directory, options)))
    with monkeypatch.context() as language:
        _use_english(language)
        project = _complete_project(tmp_path)
        async with _page_user("/", _home_route, project, monkeypatch) as user:
            await asyncio.sleep(0.3)
            user.find(marker="project-share-button").click()
            await user.should_see(marker="share-dialog")
            user.find(marker="share-start").click()
            await user.should_see(marker="share-confirm")
            await user.should_see("point cloud, plan (DXF) and IFC model")
            user.find(marker="share-confirm-yes").click()
            await asyncio.sleep(0.3)
    document = _document(project)
    assert document["settings"]["project_share_confirmed"] is True
    assert "share_confirmed" not in document["settings"]
    assert started and started[0][1]["kind"] is PROJECT
    i18n.reload_translations()


@pytest.mark.anyio
async def test_project_dialog_reads_only_project_share_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(share_ui, "needs_credentials", lambda: False)
    with monkeypatch.context() as language:
        _use_english(language)
        project = _complete_project(tmp_path)
        document = _document(project)
        document["share"] = {"url": "https://example.test/point", "share_id": "point", "provider": "local"}
        document["project_share"] = {
            "url": "https://example.test/project",
            "share_id": "project",
            "provider": "local",
            "points": 42,
            "size_mb": 1.5,
            "views": ["dxf", "pc", "ifc"],
            "levels": ["site", "upper"],
        }
        (project / "products" / "share" / "project").mkdir(parents=True)
        (project / "products" / "share" / "project" / "qr.png").write_bytes(b"png")
        (project / "project.samcloud.yml").write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
        async with _page_user("/", _home_route, project, monkeypatch) as user:
            await asyncio.sleep(0.3)
            user.find(marker="project-share-button").click()
            await user.should_see(marker="share-link")
            assert _element(user, "share-link").text == "https://example.test/project"
            assert "Plan (DXF) · Point cloud · IFC model · 2 levels" in _element(user, "share-project-views").text
            assert _markers(user, "share-qr") == 1
        def point_share_route() -> None:
            share_ui.ShareDialog(project, kind=POINTCLOUD).open()

        point_share_route.__module__ = "tests.share_project_ui_route"
        async with _page_user("/point-share", point_share_route, project, monkeypatch) as user:
            await user.should_see(marker="share-link")
            assert _element(user, "share-link").text == "https://example.test/point"
    i18n.reload_translations()


@pytest.mark.anyio
async def test_project_share_button_appears_live_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    with monkeypatch.context() as language:
        _use_english(language)
        project = make_project(tmp_path, "live", with_cloud=False)
        job_file = project / "products" / "dxf" / "job_status.json"
        job_file.parent.mkdir(parents=True, exist_ok=True)
        job_file.write_text(json.dumps({"action": "run_all", "state": "running", "fraction": 0.4}), encoding="utf-8")
        assert home._dxf_ifc_step(project)["status"] == "running"
        async with _page_user("/", _home_route, project, monkeypatch) as user:
            await asyncio.sleep(0.3)
            assert _markers(user, "project-share-button") == 0
            (project / "products" / "classified").mkdir(parents=True, exist_ok=True)
            (project / "products" / "classified" / "classified.ply").write_bytes(b"ply")
            export = project / "products" / "dxf" / "export"
            export.mkdir(parents=True, exist_ok=True)
            (export / "live_Site_clean.dxf").write_bytes(b"dxf")
            site = project / "products" / "bim" / "site"
            site.mkdir(parents=True, exist_ok=True)
            (site / "x_site.ifc").write_bytes(b"ifc")
            (site / "model.glb").write_bytes(b"glTF")
            job_file.write_text(json.dumps({"action": "run_all", "state": "completed", "fraction": 1.0}), encoding="utf-8")
            assert home._dxf_ifc_step(project)["status"] == "completed"
            await asyncio.sleep(3.0)
            assert _markers(user, "project-share-button") == 1
            await user.should_see("Share project")
            await asyncio.sleep(4.5)
            assert _markers(user, "project-share-button") == 1
    i18n.reload_translations()
