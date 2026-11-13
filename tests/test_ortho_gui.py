"""Behavioural tests for orthophoto controls, jobs, cached routes, and review-canvas integration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator
import uuid

import numpy as np
import pytest
from nicegui import run as nicegui_run, ui
from nicegui.testing.user import User
from nicegui.testing.user_simulation import user_simulation

from ortho_synthetic import make_indoor_project, make_outdoor_project
from samcloud.app import application
from samcloud.app.dxf_controller import DxfPageController
from samcloud.app.pages import dxf as dxf_page
from samcloud.app.state import state as app_state
from samcloud.dxf import jobs
from samcloud.dxf.cancel import CancelToken
from samcloud.dxf.job_runner import DxfJobRunner
from samcloud.dxf.las_source import Frame
from samcloud.dxf.pipeline import save_cached_result
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.ortho import build as ortho_build


_LOGGER = logging.getLogger("samcloud.tests.ortho_gui")


@pytest.fixture(scope="module")
def work_dir(request: pytest.FixtureRequest) -> Path:
    """Return an isolated directory compatible with the sandbox's disabled tmp_path plugin."""
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"ortho_gui_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("ortho_gui")


def _no_progress(_fraction: float, _stage: str) -> None:
    return None


def _configure_fast_ortho(project: Path) -> DxfPageController:
    """Persist small-raster settings so every synthetic orthophoto remains quick."""
    controller = DxfPageController(project)
    controller.set_ortho_option("pixel_size_outdoor", 0.1)
    controller.set_ortho_option("pixel_size_indoor", 0.1)
    controller.set_ortho_option("max_side", 1000)
    return controller


def _build_ortho(project: Path, level_id: str | None = None) -> DxfPageController:
    """Build one small orthophoto in-process for cache-only UI tests."""
    controller = _configure_fast_ortho(project)
    assert controller.las_path is not None
    jobs.run_ortho(project, controller.las_path, level_id, _no_progress, CancelToken(), _LOGGER, mode=controller.mode)
    controller.refresh_state()
    return controller


def _wait_for_job(runner: DxfJobRunner, timeout: float = 180.0) -> dict[str, Any]:
    """Wait for one subprocess job without relying on pytest's tmp_path machinery."""
    deadline = time.monotonic() + timeout
    while runner.is_running():
        assert time.monotonic() < deadline, "Orthophoto job did not finish in time"
        time.sleep(0.05)
    return runner.status()


def _dxf_route() -> None:
    dxf_page.render()


_dxf_route.__module__ = "tests.ortho_gui_route"


@asynccontextmanager
async def _dxf_user(project: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[User]:
    monkeypatch.setattr(app_state, "active_project_directory", project)
    monkeypatch.setattr(dxf_page, "_RUNNERS", {})
    monkeypatch.setattr(nicegui_run, "setup", lambda: None)
    async with user_simulation(root=lambda: None) as user:
        ui.page("/dxf")(_dxf_route)
        await user.open("/dxf")
        yield user


def _element(user: User, marker: str) -> Any:
    elements = user.find(marker=marker).elements
    assert len(elements) == 1, f"expected one element marked {marker!r}, found {len(elements)}"
    return next(iter(elements))


def _set(user: User, marker: str, value: Any) -> None:
    with user:
        _element(user, marker).value = value


def _cache_minimal_dxf_result(project: Path, variant: str = "pointcloud") -> None:
    """Persist just enough DXF cache data to render the review canvas without running DXF generation."""
    result = DxfPlanResult(
        project.name,
        "site",
        "outdoor",
        "sam3_outdoor",
        1.0,
        Frame(np.zeros(3)),
        "test-cache",
        {},
        {},
        {},
        BackgroundImage(None, None, (0.0, 0.0, 8.0, 7.0)),
    )
    cache_dir = project / "products" / "dxf" / "site"
    if variant == "ortho":
        result = replace(result, settings_fingerprint="clean-test", variant="ortho")
    save_cached_result(result, cache_dir if variant == "pointcloud" else cache_dir / "ortho")


@pytest.mark.anyio
async def test_page_open_and_ortho_settings_start_nothing(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Page opening, orthophoto settings, source toggling, and level switching are all cache-only operations."""
    synthetic = make_indoor_project(work_dir / "nothing")
    controller = _configure_fast_ortho(synthetic.project_dir)
    controller.set_profile("sam3_indoor")
    second = controller.add_level(3.0, 5.6)
    calls: list[str] = []

    def record(name: str):
        def inner(*_args: Any, **_kwargs: Any) -> None:
            calls.append(name)
            raise AssertionError(f"{name} must not run automatically")

        return inner

    monkeypatch.setattr(jobs, "build_orthophoto", record("jobs.build_orthophoto"))
    monkeypatch.setattr(ortho_build, "build_orthophoto", record("ortho.build_orthophoto"))
    monkeypatch.setattr(DxfJobRunner, "start", record("DxfJobRunner.start"))
    fake_result = SimpleNamespace(grid=SimpleNamespace(bounds=(0.0, 0.0, 8.0, 7.0)))
    monkeypatch.setattr(DxfPageController, "ortho_result", lambda _self, _level: fake_result)
    monkeypatch.setattr(DxfPageController, "ortho_status", lambda _self, _level: "ok")

    async with _dxf_user(synthetic.project_dir, monkeypatch) as user:
        await user.should_see(marker="ortho-build")
        _set(user, "ortho-pixel-outdoor", 0.12)
        _set(user, "ortho-pixel-indoor", 0.08)
        _set(user, "ortho-max-side", 1200)
        _set(user, "ortho-cut", 0.2)
        _set(user, "segment-source", "pointcloud")
        _set(user, "ortho-background", "ortho")
        _set(user, "review-level", second["id"])

    assert calls == []
    stored = DxfPageController(synthetic.project_dir).state
    assert stored.ortho == {
        "pixel_size_outdoor": pytest.approx(0.12),
        "pixel_size_indoor": pytest.approx(0.08),
        "max_side": 1200,
        "cut_below_ceiling": pytest.approx(0.2),
    }
    assert stored.view["background_source"] == "ortho"
    assert stored.segment_source == "pointcloud"


@pytest.mark.anyio
async def test_build_orthophoto_button_starts_ortho_job_with_level(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The explicit build button starts the ortho action for the currently selected indoor level."""
    synthetic = make_indoor_project(work_dir / "button")
    controller = _configure_fast_ortho(synthetic.project_dir)
    controller.set_profile("sam3_indoor")
    started: list[tuple[str, str | None]] = []

    def start(_self: DxfJobRunner, action: str, las: Path | str, level: str | None = None, mode: str | None = None,
              force: bool = False, **_options: Any) -> None:
        assert Path(las).is_file() and mode == "indoor"
        assert force is False
        started.append((action, level))

    monkeypatch.setattr(DxfJobRunner, "start", start)
    async with _dxf_user(synthetic.project_dir, monkeypatch) as user:
        user.find(marker="ortho-build").click()
    assert started == [("ortho", synthetic.level.id)]


def test_ortho_job_subprocess_builds_outputs_and_cancel_keeps_cache(work_dir: Path) -> None:
    """The real subprocess writes cache artifacts, and a cancelled replacement preserves the previous PNG."""
    synthetic = make_outdoor_project(work_dir / "subprocess")
    controller = _configure_fast_ortho(synthetic.project_dir)
    assert controller.las_path is not None
    runner = DxfJobRunner(synthetic.project_dir)
    runner.start("ortho", controller.las_path, mode="outdoor")
    assert _wait_for_job(runner)["state"] == "done"
    output = controller.ortho_dir(None)
    previous = (output / "ortho.png").read_bytes()
    assert (output / "ortho.json").is_file()

    controller.set_ortho_option("pixel_size_outdoor", 0.08)
    runner.start("ortho", controller.las_path, mode="outdoor")
    runner.cancel()
    assert _wait_for_job(runner)["state"] == "cancelled"
    assert (output / "ortho.png").read_bytes() == previous


def test_ortho_status_missing_ok_stale(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Freshness checks compare fingerprints only and never invoke the orthophoto builder."""
    synthetic = make_outdoor_project(work_dir / "status")
    controller = _configure_fast_ortho(synthetic.project_dir)
    assert controller.ortho_status(None) == "missing"
    _build_ortho(synthetic.project_dir)
    controller = DxfPageController(synthetic.project_dir)
    assert controller.ortho_status(None) == "ok"

    def fail_build(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("freshness inspection must not build an orthophoto")

    monkeypatch.setattr(jobs, "build_orthophoto", fail_build)
    monkeypatch.setattr(ortho_build, "build_orthophoto", fail_build)
    controller.set_ortho_option("pixel_size_outdoor", 0.08)
    assert controller.ortho_status(None) == "stale"


def test_ortho_routes_serve_cache_only(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The HTTP endpoint is 404 before a build and serves only existing artifacts afterwards."""
    synthetic = make_outdoor_project(work_dir / "routes")
    _configure_fast_ortho(synthetic.project_dir)
    monkeypatch.setattr(app_state, "active_project_directory", synthetic.project_dir)
    assert application.dxf_ortho_file("site", "preview").status_code == 404

    _build_ortho(synthetic.project_dir)
    monkeypatch.setattr(jobs, "build_orthophoto", lambda *_args, **_kwargs: pytest.fail("route must not build"))
    preview = application.dxf_ortho_file("site", "preview")
    png = application.dxf_ortho_file("site", "png")
    assert preview.status_code == 200 and preview.media_type == "image/jpeg"
    assert png.status_code == 200 and png.media_type == "image/png"
    assert Path(preview.path).name == "preview.jpg" and Path(png.path).name == "ortho.png"


@pytest.mark.anyio
async def test_background_toggle_uses_ortho_preview_and_bounds(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting Orthophoto switches the review image to preview.jpg and uses the grid's outer edge bounds."""
    synthetic = make_outdoor_project(work_dir / "canvas")
    controller = _build_ortho(synthetic.project_dir)
    controller.set_profile("sam3_outdoor")
    _cache_minimal_dxf_result(synthetic.project_dir)
    expected_bounds = list(controller.ortho_result(None).grid.bounds)

    async with _dxf_user(synthetic.project_dir, monkeypatch) as user:
        await user.should_see(marker="review-canvas")
        _set(user, "ortho-background", "ortho")
        canvas = _element(user, "review-canvas")
        assert canvas.props["image-url"].startswith("/api/dxf/ortho/site/preview?v=")
        assert canvas.props["image-bounds"] == pytest.approx(expected_bounds)


@pytest.mark.anyio
@pytest.mark.parametrize("action, marker", [
    ("ortho", "ortho-build"),
    ("segment", "segment-build"),
    ("clean_plan", "clean-plan-build"),
    ("export", "export"),
])
async def test_wp4_buttons_start_the_matching_explicit_job(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch, action: str, marker: str,
) -> None:
    """Every WP4 action is started only by its matching button and carries the selected state."""
    synthetic = make_outdoor_project(work_dir / f"button_{action}")
    controller = _configure_fast_ortho(synthetic.project_dir)
    controller.set_profile("sam3_outdoor")
    _cache_minimal_dxf_result(synthetic.project_dir)
    started: list[tuple[str, dict[str, Any]]] = []

    def start(_self: DxfJobRunner, name: str, las: Path | str, **kwargs: Any) -> None:
        assert Path(las).is_file()
        started.append((name, kwargs))

    monkeypatch.setattr(DxfJobRunner, "start", start)
    monkeypatch.setattr(DxfPageController, "ortho_status", lambda _self, _level: "ok")
    monkeypatch.setattr(DxfPageController, "segment_status", lambda _self, _level: "ok")
    monkeypatch.setattr(DxfPageController, "plan_status", lambda _self, _level, _variant="pointcloud": "ok")
    async with _dxf_user(synthetic.project_dir, monkeypatch) as user:
        user.find(marker=marker).click()
    assert started and started[0][0] == action
    if action in {"segment", "clean_plan"}:
        assert started[0][1]["source"] == "combined"
    if action == "export":
        assert started[0][1]["variant"] == "pointcloud"


@pytest.mark.anyio
async def test_wp4_status_chips_reflect_cached_step_freshness(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page shows independently computed missing, fresh, and stale states for its workflow steps."""
    synthetic = make_outdoor_project(work_dir / "statuses")
    controller = _configure_fast_ortho(synthetic.project_dir)
    controller.set_profile("sam3_outdoor")
    _cache_minimal_dxf_result(synthetic.project_dir)
    monkeypatch.setattr(DxfPageController, "ortho_status", lambda _self, _level: "ok")
    monkeypatch.setattr(DxfPageController, "segment_status", lambda _self, _level: "stale")
    monkeypatch.setattr(DxfPageController, "plan_status", lambda _self, _level, variant="pointcloud": "missing" if variant == "ortho" else "ok")
    async with _dxf_user(synthetic.project_dir, monkeypatch) as user:
        assert _element(user, "ortho-status").text == "Orthophoto: up to date"
        assert _element(user, "segment-status").text == "Segment: stale"
        assert _element(user, "clean-plan-status").text == "Clean plan: missing"


@pytest.mark.anyio
async def test_wp4_variant_switch_changes_review_payload(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching between fast and clean plans changes the review canvas payload URL without a job."""
    synthetic = make_outdoor_project(work_dir / "variant")
    controller = _configure_fast_ortho(synthetic.project_dir)
    controller.set_profile("sam3_outdoor")
    _cache_minimal_dxf_result(synthetic.project_dir)
    _cache_minimal_dxf_result(synthetic.project_dir, "ortho")
    async with _dxf_user(synthetic.project_dir, monkeypatch) as user:
        assert "variant=ortho" not in _element(user, "review-canvas").props["payload-url"]
        _set(user, "plan-variant", "ortho")
        assert "variant=ortho" in _element(user, "review-canvas").props["payload-url"]
    assert DxfPageController(synthetic.project_dir).state.plan_variant == "ortho"


@pytest.mark.anyio
async def test_wp4_compare_mode_supplies_both_vector_overlays(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare mode keeps the orthophoto background and provides both cached vector payloads to the canvas."""
    synthetic = make_outdoor_project(work_dir / "compare")
    controller = _build_ortho(synthetic.project_dir)
    controller.set_profile("sam3_outdoor")
    _cache_minimal_dxf_result(synthetic.project_dir)
    _cache_minimal_dxf_result(synthetic.project_dir, "ortho")
    async with _dxf_user(synthetic.project_dir, monkeypatch) as user:
        _set(user, "compare-plans", True)
        canvas = _element(user, "review-canvas")
        assert canvas.props["image-url"].startswith("/api/dxf/ortho/site/preview?v=")
        assert "variant=ortho" not in canvas.props["payload-url"]
        assert "variant=ortho" in canvas.props["compare-payload-url"]
        await user.should_see(marker="compare-pointcloud-opacity")
        await user.should_see(marker="compare-clean-opacity")
