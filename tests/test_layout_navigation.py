"""Tests for conditional navigation visibility in the layout sidebar."""

from samcloud.app import layout


def test_navigation_hides_project_pages_without_project() -> None:
    visible_paths = [path for _, path, _ in layout._visible_navigation(False)]
    assert visible_paths == ["/", "/projects", "/settings"]


def test_navigation_shows_all_pages_with_project() -> None:
    visible_paths = [path for _, path, _ in layout._visible_navigation(True)]
    assert visible_paths == [
        "/",
        "/projects",
        "/sparse",
        "/control-points",
        "/dense",
        "/dxf",
        "/settings",
    ]


# --------------------------------------------------------------------------- DXF readiness gating

from pathlib import Path
from typing import Any

import pytest
import yaml
from nicegui import ui
from nicegui.testing.user import User
from nicegui.testing.user_simulation import user_simulation

from samcloud.app.dxf_controller import DXF_NO_LAS_HINT, DXF_NO_PROJECT_HINT, dxf_unavailable_reason
from samcloud.app.pages import home as home_page
from samcloud.app.state import state as app_state
from samcloud.dxf.state import DxfProjectState, save_state


def _make_project(root: Path) -> Path:
    """Create a minimal project folder without any products."""
    project = (root / "Gating").resolve()
    project.mkdir(parents=True)
    document = {"project": {"name": "Gating"}, "workflow": {}}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return project


def _write_classified_las(project: Path) -> Path:
    # Readiness is an existence check only; the file content is never read.
    las = project / "products" / "classified" / "classified.las"
    las.parent.mkdir(parents=True, exist_ok=True)
    las.write_bytes(b"")
    return las


def _shell_route() -> None:
    layout.render_page("Gating", lambda: None, current_path="/")


def _workflow_route() -> None:
    home_page._render_workflow(app_state.active_project())


# NiceGUI's test reset drops sys.modules entries of page modules outside "tests."; keep samcloud loaded.
_shell_route.__module__ = "tests.layout_gating_route"
_workflow_route.__module__ = "tests.layout_gating_route"


def _single(user: User, marker: str) -> Any:
    elements = user.find(marker=marker).elements
    assert len(elements) == 1, f"expected one element marked {marker!r}, found {len(elements)}"
    return next(iter(elements))


def _tooltip_text(element: Any) -> str | None:
    """Return the text of the tooltip attached to the element's wrapping slot (disabled buttons get no hover)."""
    parent = element.parent_slot.parent if element.parent_slot is not None else None
    if parent is None:
        return None
    target = f"#{parent.html_id}"
    texts = [tip.text for tip in element.client.elements.values() if isinstance(tip, ui.tooltip) and tip.props.get("target") == target]
    assert len(texts) <= 1
    return texts[0] if texts else None


def test_dxf_readiness_requires_project_and_las(tmp_path: Path) -> None:
    assert dxf_unavailable_reason(None) == DXF_NO_PROJECT_HINT
    project = _make_project(tmp_path)
    assert dxf_unavailable_reason(project) == DXF_NO_LAS_HINT
    assert layout._navigation_disabled_reason("/dxf", project) == DXF_NO_LAS_HINT
    assert layout._navigation_disabled_reason("/dense", project) is None
    _write_classified_las(project)
    assert dxf_unavailable_reason(project) is None
    assert layout._navigation_disabled_reason("/dxf", project) is None


def test_dxf_readiness_accepts_existing_chosen_las(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    external = tmp_path / "external" / "scan.las"
    dxf_state = DxfProjectState()
    dxf_state.view["las_path"] = str(external)
    save_state(project, dxf_state)
    assert dxf_unavailable_reason(project) == DXF_NO_LAS_HINT, "a chosen LAS that does not exist must not count"
    external.parent.mkdir(parents=True)
    external.write_bytes(b"")
    assert dxf_unavailable_reason(project) is None


@pytest.mark.anyio
async def test_dxf_nav_hidden_without_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_state, "active_project_directory", None)
    async with user_simulation(root=lambda: None) as user:
        ui.page("/shell")(_shell_route)
        await user.open("/shell")
        await user.should_see(marker="nav-overview")
        await user.should_not_see(marker="nav-dxf")


@pytest.mark.anyio
async def test_dxf_nav_disabled_without_classified_las(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path)
    monkeypatch.setattr(app_state, "active_project_directory", project)
    async with user_simulation(root=lambda: None) as user:
        ui.page("/shell")(_shell_route)
        await user.open("/shell")
        button = _single(user, "nav-dxf")
        assert button.props.get("disable") is True
        assert "sam-nav-disabled" in button.classes
        assert _tooltip_text(button) == DXF_NO_LAS_HINT
        dense = _single(user, "nav-dense")
        assert not dense.props.get("disable")
        user.find(marker="nav-dxf").click()
        assert "/dxf" not in user.back_history


@pytest.mark.anyio
async def test_dxf_nav_enabled_with_classified_las(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path)
    _write_classified_las(project)
    monkeypatch.setattr(app_state, "active_project_directory", project)
    async with user_simulation(root=lambda: None) as user:
        ui.page("/shell")(_shell_route)
        await user.open("/shell")
        button = _single(user, "nav-dxf")
        assert not button.props.get("disable")
        assert "sam-nav-disabled" not in button.classes
        assert _tooltip_text(button) is None


@pytest.mark.anyio
async def test_home_dxf_button_disabled_without_classified_las(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path)
    monkeypatch.setattr(app_state, "active_project_directory", project)
    async with user_simulation(root=lambda: None) as user:
        ui.page("/workflow")(_workflow_route)
        await user.open("/workflow")
        button = _single(user, "home-open-dxf")
        assert button.props.get("disable") is True
        assert _tooltip_text(button) == DXF_NO_LAS_HINT
        await user.should_see(DXF_NO_LAS_HINT, marker="home-dxf-status")


@pytest.mark.anyio
async def test_home_dxf_button_enabled_with_classified_las(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path)
    _write_classified_las(project)
    monkeypatch.setattr(app_state, "active_project_directory", project)
    async with user_simulation(root=lambda: None) as user:
        ui.page("/workflow")(_workflow_route)
        await user.open("/workflow")
        button = _single(user, "home-open-dxf")
        assert not button.props.get("disable")
        await user.should_not_see(marker="home-dxf-status")
        await user.should_see("optional")


@pytest.mark.anyio
async def test_home_dxf_button_absent_without_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_state, "active_project_directory", None)
    async with user_simulation(root=lambda: None) as user:
        ui.page("/workflow")(_workflow_route)
        await user.open("/workflow")
        await user.should_see("DXF export (optional)")
        await user.should_not_see(marker="home-open-dxf")
