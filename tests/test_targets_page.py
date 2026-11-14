"""Targets page: printable AprilTag and cross target PDFs, with and without an open project."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import yaml
from nicegui import ui
from nicegui.testing.user import User
from nicegui.testing.user_download import UserDownload
from nicegui.testing.user_simulation import user_simulation

from samcloud.app import layout
from samcloud.app.pages import targets as targets_page
from samcloud.app.state import state as app_state
from samcloud.core.project_manager import ProjectManager

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _targets_route() -> None:
    targets_page.render()


# NiceGUI's test reset drops sys.modules entries of page modules outside "tests."; keep samcloud loaded.
_targets_route.__module__ = "tests.targets_page_route"


@asynccontextmanager
async def _targets_user(project: Path | None, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[User]:
    monkeypatch.setattr(app_state, "active_project_directory", project)
    async with user_simulation(root=lambda: None) as user:
        ui.page("/targets")(_targets_route)
        await user.open("/targets")
        yield user


def _element(user: User, marker: str) -> Any:
    elements = user.find(marker=marker).elements
    assert len(elements) == 1, f"expected one element marked {marker!r}, found {len(elements)}"
    return next(iter(elements))


def _set(user: User, marker: str, value: Any) -> None:
    with user:
        _element(user, marker).value = value


def _record_downloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    downloads: list[dict[str, Any]] = []

    def fake_call(self: Any, src: Any, filename: str | None = None, media_type: str = "") -> None:
        downloads.append({"data": src, "filename": filename, "media_type": media_type})

    monkeypatch.setattr(UserDownload, "__call__", fake_call)
    return downloads


@pytest.mark.anyio
async def test_apriltag_download_without_project(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _targets_user(None, monkeypatch) as user:
        downloads = _record_downloads(monkeypatch)
        await user.should_see(marker="targets-apriltag-pdf")
        await user.should_see(marker="targets-cross-pdf")
        assert _element(user, "targets-apriltag-start").value == 0
        user.find(marker="targets-apriltag-pdf").click()

        assert len(downloads) == 1
        assert downloads[0]["filename"] == "apriltags_0-9.pdf"
        assert downloads[0]["media_type"] == "application/pdf"
        assert downloads[0]["data"][:4] == b"%PDF"


@pytest.mark.anyio
async def test_cross_download_uses_start_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _targets_user(None, monkeypatch) as user:
        downloads = _record_downloads(monkeypatch)
        _set(user, "targets-cross-count", 2)
        _set(user, "targets-cross-start", 7)
        user.find(marker="targets-cross-pdf").click()

        assert len(downloads) == 1
        assert downloads[0]["filename"] == "kreuz_targets_7-8.pdf"
        assert downloads[0]["media_type"] == "application/pdf"
        assert downloads[0]["data"][:4] == b"%PDF"


@pytest.mark.anyio
async def test_invalid_cross_range_shows_no_download(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _targets_user(None, monkeypatch) as user:
        downloads = _record_downloads(monkeypatch)
        _set(user, "targets-cross-count", 5)
        _set(user, "targets-cross-start", 9998)
        user.find(marker="targets-cross-pdf").click()
        assert downloads == []


def test_targets_navigation_visible_without_project() -> None:
    visible = layout._visible_navigation(False)
    assert ("Targets", "/targets", "qr_code_2") in visible
    assert "/targets" not in layout._PROJECT_PAGES


@pytest.mark.anyio
async def test_apriltag_start_follows_open_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ProjectManager(tmp_path)
    project = manager.create_project("TagsProject", str(tmp_path), class_set="indoor", apriltags_enabled=True)
    monkeypatch.setattr(app_state, "projects", manager)
    async with _targets_user(project, monkeypatch) as user:
        downloads = _record_downloads(monkeypatch)
        _set(user, "targets-apriltag-count", 3)
        _set(user, "targets-apriltag-start", 4)
        user.find(marker="targets-apriltag-pdf").click()
        assert downloads[0]["filename"] == "apriltags_4-6.pdf"

    saved = yaml.safe_load((project / "project.samcloud.yml").read_text(encoding="utf-8"))
    assert saved["settings"]["apriltags"]["last_printed_id"] == 6

    # Reopening the page proposes the id after the last printed one.
    async with _targets_user(project, monkeypatch) as user:
        assert _element(user, "targets-apriltag-start").value == 7
