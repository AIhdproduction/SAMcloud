"""Protocol card on the control-points page."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from samcloud.app.pages import control_points as control_points_page
from test_apriltag_gui import _element, _page_user
from test_share_export import make_project

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _project(root: Path, name: str, **kwargs: Any) -> Path:
    """Share-test project; its minimal AprilTag report lacks the fields the page table needs, so drop it."""
    project = make_project(root, name, **kwargs)
    (project / "reports" / "apriltag_report.json").unlink()
    return project


def _route() -> None:
    control_points_page.render()


_route.__module__ = "tests.protocol_panel_route"


def _markers(user: Any, marker: str) -> int:
    try:
        return len(user.find(marker=marker).elements)
    except AssertionError:
        return 0


def _disabled(user: Any) -> bool:
    return bool(_element(user, "protocol-button").props.get("disable"))


@pytest.mark.anyio
async def test_disabled_when_step6_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, "early", refinement="running", with_cloud=False)
    async with _page_user("/control-points", _route, project, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert _disabled(user)


@pytest.mark.anyio
async def test_enabled_when_done_and_hint_without_control_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, "done")
    async with _page_user("/control-points", _route, project, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert not _disabled(user)
        assert _markers(user, "protocol-hint") == 1
        assert _markers(user, "protocol-list") == 1


@pytest.mark.anyio
async def test_disabled_when_points_without_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, "pts")
    (project / "control_points" / "control_points.csv").write_text("name,x,y,z\nP1,1,2,3\n", encoding="utf-8")
    async with _page_user("/control-points", _route, project, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert _disabled(user)
        assert _markers(user, "protocol-hint") == 0


@pytest.mark.anyio
async def test_existing_protocols_are_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, "listed")
    for name in ("protocol_20260101_1000.pdf", "protocol_20260102_1000.pdf"):
        (project / "reports" / name).write_bytes(b"%PDF-1.4")
    async with _page_user("/control-points", _route, project, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert _markers(user, "protocol-download") == 2
        await user.should_see("protocol_20260102_1000.pdf")
