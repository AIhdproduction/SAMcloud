"""DXF export GUI tests: no automatic runs, staleness, compositing, jobs, culling, profiles, page smoke, end to end."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import time
from typing import Any, AsyncIterator

import ezdxf
import laspy
import numpy as np
import pytest
import yaml
from nicegui import ui
from nicegui.testing.user import User
from nicegui.testing.user_simulation import user_simulation

from dxf_synthetic import write_indoor_building, write_outdoor_scene
from samcloud.app import dxf_controller as controller_module
from samcloud.app.dxf_controller import DxfPageController
from samcloud.app.pages import dxf as dxf_page
from samcloud.app.state import state as app_state
from samcloud.dxf import jobs, pipeline, preview, writer
from samcloud.dxf.cancel import CancelToken
from samcloud.dxf.job_runner import DxfJobRunner
from samcloud.dxf.las_source import Frame
from samcloud.dxf.overlay import composite
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus, vector_payload, visible_cells, visible_entity_indices

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_LOGGER = logging.getLogger("samcloud.tests.dxf_gui")


def _no_progress(_fraction: float, _stage: str) -> None:
    return None


def _make_project(root: Path, name: str, kind: str) -> tuple[Path, Path]:
    """Create a minimal SAMcloud project folder with a classified LAS at the default location."""
    project = (root / name).resolve()
    las = project / "products" / "classified" / "classified.las"
    if kind == "indoor":
        write_indoor_building(las)
    elif kind == "outdoor":
        write_outdoor_scene(las)
    elif kind == "flurseg":
        write_outdoor_scene(las, profile="flurseg")
    elif kind == "no_profile":
        _write_las_without_provenance(las)
    else:
        raise ValueError(kind)
    class_set = "indoor" if kind == "indoor" else "outdoor"
    document = {"project": {"name": name}, "settings": {"classification": {"class_set": class_set}}, "workflow": {}}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return project, las


def _write_las_without_provenance(path: Path) -> None:
    rng = np.random.default_rng(7)
    xyz = np.column_stack((rng.uniform(0.0, 20.0, 3000), rng.uniform(0.0, 12.0, 3000), rng.normal(0.0, 0.01, 3000)))
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(np.min(xyz, axis=0))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cloud.classification = np.full(len(xyz), 2, dtype=np.uint8)
    cloud.write(path)


def _prepare_indoor(project: Path, las: Path) -> list[str]:
    """Run preview, alignment and level detection in-process; return level ids."""
    token = CancelToken()
    jobs.run_preview(project, las, _no_progress, token, _LOGGER)
    jobs.run_align(project, las, _no_progress, token, _LOGGER)
    jobs.run_levels(project, las, _no_progress, token, _LOGGER)
    return [str(level["id"]) for level in DxfPageController(project).available_levels()]


def _generate(project: Path, las: Path, level_ids: list[str | None], mode: str) -> None:
    for level_id in level_ids:
        jobs.run_generate(project, las, level_id, _no_progress, CancelToken(), _LOGGER, mode=mode)


@pytest.fixture
def heavy_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every heavy entry point by a recorder; any call is a failure of 'nothing runs automatically'."""
    calls: list[str] = []

    def recorder(name: str) -> Any:
        def record(*_args: Any, **_kwargs: Any) -> None:
            calls.append(name)
            raise AssertionError(f"{name} must not run automatically")

        return record

    for module, name in (
        (jobs, "create_preview"), (jobs, "estimate_alignment"), (jobs, "detect_levels"), (jobs, "generate_plan"),
        (jobs, "export_plan"), (preview, "create_preview"), (pipeline, "generate_plan"), (writer, "export_plan"),
    ):
        monkeypatch.setattr(module, name, recorder(f"{module.__name__}.{name}"))
    monkeypatch.setattr(DxfJobRunner, "start", recorder("DxfJobRunner.start"))
    return calls


def _dxf_route() -> None:
    dxf_page.render()


# NiceGUI's test reset drops sys.modules entries of page modules outside "tests."; keep samcloud loaded.
_dxf_route.__module__ = "tests.dxf_gui_route"


@asynccontextmanager
async def _dxf_user(project: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[User]:
    monkeypatch.setattr(app_state, "active_project_directory", project)
    monkeypatch.setattr(dxf_page, "_RUNNERS", {})
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


def _no_error_logs(caplog: pytest.LogCaptureFixture) -> None:
    errors = [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
    assert not errors, errors


# --------------------------------------------------------------------------- 1. nothing runs automatically


def test_controller_settings_never_start_heavy_work(tmp_path: Path, heavy_calls: list[str]) -> None:
    project, las = _make_project(tmp_path, "Calls", "indoor")
    controller = DxfPageController(project)
    controller.refresh_state()
    controller.set_mode_override("indoor")
    controller.set_mode_override(None)
    controller.set_profile("sam3_indoor")
    controller.set_scale_factor(1.25)
    controller.measure_scale([0.0, 0.0], [2.0, 0.0], 2.0)
    level = controller.add_level_at(0.0)
    controller.add_level(3.0, 5.6)
    controller.rename_level(level["id"], "Ground")
    controller.move_level(level["id"], floor_z=0.1, ceiling_z=2.7)
    controller.set_level_checked(level["id"], False)
    controller.set_cell_size(0.03)
    controller.set_layer_mode("A-WALL", "footprint")
    controller.set_layer_z_band("A-COLS", 0.5, 1.5)
    controller.reset_layer_override("A-COLS")
    for key, value in (("image_opacity", 0.3), ("dxf_opacity", 0.6), ("image_source", "classes"), ("coordinates", "original"),
                       ("wall_centerlines", False), ("door_swing_arc", True), ("side_axis", "y"), ("slab_center", 2.0), ("slab_thickness", 1.0)):
        controller.set_view_option(key, value)
    controller.delete_level(level["id"])
    controller.set_las_path(las)
    for level_id in controller.review_targets():
        controller.is_stale(level_id)
        controller.export_enabled(level_id)
        controller.cached_result(level_id)
    controller.stale_levels()
    assert heavy_calls == []
    assert not controller.is_running()


@pytest.mark.anyio
@pytest.mark.parametrize("with_cache", [True, False])
async def test_page_open_and_every_setting_start_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, with_cache: bool,
) -> None:
    project, las = _make_project(tmp_path, "Haus", "indoor")
    level_ids = _prepare_indoor(project, las)
    if with_cache:
        _generate(project, las, level_ids, "indoor")
    calls: list[str] = []
    for module, name in ((jobs, "create_preview"), (jobs, "estimate_alignment"), (jobs, "detect_levels"), (jobs, "generate_plan"),
                         (jobs, "export_plan"), (pipeline, "generate_plan"), (writer, "export_plan")):
        monkeypatch.setattr(module, name, lambda *_a, _n=f"{module.__name__}.{name}", **_k: calls.append(_n))
    monkeypatch.setattr(DxfJobRunner, "start", lambda *_a, **_k: calls.append("DxfJobRunner.start"))
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see("Side view and levels")
        await user.should_see("DXF review")
        if with_cache:
            await user.should_see(marker="review-canvas")
        else:
            await user.should_see(marker="no-cache")
        first = level_ids[0]
        _set(user, "cell-size", 0.04)
        _set(user, f"layer-mode-A-WALL", "footprint")
        _set(user, f"layer-zmin-A-COLS", 0.4)
        _set(user, "wall-output", "polygons")
        _set(user, "door-swing", True)
        _set(user, "coordinates", "original")
        _set(user, "slab-center", 3.0)
        _set(user, "slab-thickness", 2.0)
        _set(user, "side-axis", "y")
        _set(user, f"level-name-{first}", "EG")
        user.find(marker=f"level-name-{first}").trigger("blur")
        _set(user, f"level-floor-{first}", 0.05)
        user.find(marker="side-canvas").trigger("level_move", {"id": first, "edge": "ceiling", "z": 2.55})
        user.find(marker="side-canvas").trigger("add_level", {"z": 7.0})
        _set(user, "scale-factor", 1.0)
        user.find(marker="scale-apply").click()
        await user.should_see("DXF review")
        if with_cache:
            _set(user, "image-opacity", 40)
            _set(user, "dxf-opacity", 70)
            _set(user, "image-source", "classes")
        _set(user, "mode-select", "indoor")
        await user.should_see("DXF review")
    assert calls == []
    _no_error_logs(caplog)
    stored = json.loads((project / "products" / "dxf" / "state.json").read_text(encoding="utf-8"))
    assert stored["overrides"]["cell_size"] == pytest.approx(0.04)
    assert stored["overrides"]["layers"]["A-WALL"]["mode"] == "footprint"
    assert stored["view"]["wall_centerlines"] is False and stored["view"]["door_swing_arc"] is True
    assert stored["view"]["side_axis"] == "y" and stored["view"]["coordinates"] == "original"
    assert any(level["name"] == "EG" and level["ceiling_z"] == pytest.approx(2.55) for level in stored["levels"])
    assert len(stored["levels"]) == 3
    if with_cache:
        assert stored["view"]["image_opacity"] == pytest.approx(0.4)
        assert stored["view"]["dxf_opacity"] == pytest.approx(0.7)
        assert stored["view"]["image_source"] == "classes"


# --------------------------------------------------------------------------- 2. stale after settings change


@pytest.mark.anyio
async def test_setting_change_marks_cache_stale_and_disables_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    project, las = _make_project(tmp_path, "Stale", "indoor")
    level_ids = _prepare_indoor(project, las)
    _generate(project, las, level_ids, "indoor")
    controller = DxfPageController(project)
    assert all(not controller.is_stale(level) and controller.export_enabled(level) for level in level_ids)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="review-canvas")
        await user.should_not_see(marker="review-stale")
        assert _element(user, "export").enabled
        _set(user, "cell-size", 0.05)
        await user.should_see(marker="review-stale")
        await user.should_see(marker="stale-banner")
        assert not _element(user, "export").enabled
        assert not _element(user, "export-all").enabled
    _no_error_logs(caplog)
    controller = DxfPageController(project)
    assert controller.is_stale(level_ids[0]) and not controller.export_enabled(level_ids[0])
    controller.set_cell_size(None)
    assert not controller.is_stale(level_ids[0])
    controller.rename_level(level_ids[1], "Attic")
    assert controller.is_stale(level_ids[1]) and not controller.export_enabled(level_ids[1])
    controller.set_scale_factor(1.1)
    assert controller.is_stale(level_ids[0]) and not controller.all_exports_enabled()


# --------------------------------------------------------------------------- 3. compositing


@pytest.mark.parametrize("image_opacity", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("dxf_opacity", [0.0, 0.5, 1.0])
def test_composite_matches_reference_values(image_opacity: float, dxf_opacity: float) -> None:
    image = np.array([[[200, 100, 50], [200, 100, 50]]], dtype=np.uint8)
    overlay = np.array([[[0, 0, 255, 255], [0, 0, 255, 0]]], dtype=np.uint8)
    expected_base = {0.0: (255.0, 255.0, 255.0), 0.5: (227.5, 177.5, 152.5), 1.0: (200.0, 100.0, 50.0)}[image_opacity]
    covered = tuple(0.0 * dxf_opacity + value * (1.0 - dxf_opacity) for value in expected_base)
    covered = (covered[0], covered[1], 255.0 * dxf_opacity + expected_base[2] * (1.0 - dxf_opacity))
    reference = {
        (0.0, 0.0): (255, 255, 255), (0.0, 0.5): (128, 128, 255), (0.0, 1.0): (0, 0, 255),
        (0.5, 0.0): (228, 178, 153), (0.5, 0.5): (114, 89, 204), (0.5, 1.0): (0, 0, 255),
        (1.0, 0.0): (200, 100, 50), (1.0, 0.5): (100, 50, 153), (1.0, 1.0): (0, 0, 255),
    }[(image_opacity, dxf_opacity)]
    result = composite(image, overlay, image_opacity, dxf_opacity)
    assert np.allclose(result[0, 0], reference, atol=1)
    assert np.allclose(result[0, 0], covered, atol=1)
    assert np.allclose(result[0, 1], expected_base, atol=1)
    dark = composite(image, overlay, image_opacity, dxf_opacity, background=controller_module.REVIEW_BACKGROUND)
    dark_base = np.array([200, 100, 50]) * image_opacity + np.array(controller_module.REVIEW_BACKGROUND) * (1 - image_opacity)
    assert np.allclose(dark[0, 1], dark_base, atol=1)


def test_snapshot_uses_compositing(tmp_path: Path) -> None:
    project, las = _make_project(tmp_path, "Snap", "outdoor")
    _generate(project, las, [None], "outdoor")
    controller = DxfPageController(project)
    path = controller.save_snapshot(None, "rgb", image_opacity=0.0, dxf_opacity=0.0)
    assert path.parent == project / "products" / "dxf" / "snapshots" and path.suffix == ".png"
    import cv2

    pixels = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    assert np.all(np.abs(pixels.astype(int) - np.array(controller_module.REVIEW_BACKGROUND)) <= 1)
    full = cv2.imread(str(controller.save_snapshot(None, "classes", image_opacity=1.0, dxf_opacity=1.0)))
    assert full.shape == pixels.shape and not np.array_equal(cv2.cvtColor(full, cv2.COLOR_BGR2RGB), pixels)


# --------------------------------------------------------------------------- 4. job runner (real subprocess)


def _wait_for_job(runner: DxfJobRunner, timeout: float = 180.0) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while runner.is_running():
        seen.append(runner.status())
        assert time.monotonic() < deadline, "DXF job did not finish in time"
        time.sleep(0.05)
    seen.append(runner.status())
    return seen


def test_job_runner_progress_completion_and_cancel_keeps_cache(tmp_path: Path) -> None:
    project, las = _make_project(tmp_path, "Jobs", "outdoor")
    runner = DxfJobRunner(project)
    controller = DxfPageController(project, runner=runner)
    controller.start_job("preview")
    with pytest.raises(RuntimeError):
        runner.start("preview", las)
    seen = _wait_for_job(runner)
    assert seen[0]["state"] == "running" and seen[0]["action"] == "preview"
    assert seen[-1]["state"] == "done" and seen[-1]["fraction"] == pytest.approx(1.0)
    assert (project / "products" / "dxf" / "preview.npz").is_file()
    assert any("%]" in line for line in runner.log_tail(200)), "progress callback lines missing from job.log"

    controller.start_job("generate")
    assert _wait_for_job(runner)[-1]["state"] == "done"
    cache = project / "products" / "dxf" / "site"
    before = {path.name: path.read_bytes() for path in cache.iterdir() if path.is_file()}
    assert "result.json" in before

    controller.start_job("generate")
    runner.cancel()
    final = _wait_for_job(runner)[-1]
    assert final["state"] == "cancelled"
    after = {path.name: path.read_bytes() for path in cache.iterdir() if path.is_file()}
    assert after == before
    assert DxfPageController(project).export_enabled(None)


# --------------------------------------------------------------------------- 5. culling


def test_vector_payload_culling_returns_only_intersecting_cells() -> None:
    entities = [{"type": "line", "start": [float(x), 1.0], "end": [float(x) + 0.5, 1.0]} for x in range(10)]
    entities.append({"type": "line", "start": [9.6, 10.0], "end": [10.0, 10.0]})
    result = DxfPlanResult(
        "Cull", "site", "outdoor", "sam3_outdoor", 1.0, Frame(np.zeros(3)), "fp", {"S-CURB": entities},
        {"S-CURB": LayerStatus("S-CURB", "ok", None, len(entities))}, {"S-CURB": 140},
        BackgroundImage(None, None, (0.0, 0.0, 10.0, 10.0)),
    )
    payload = vector_payload(result, grid_size=10)
    layer = payload["layers"]["S-CURB"]
    left, bottom = layer["bounds"][0], layer["bounds"][1]
    assert layer["cell_size"] == pytest.approx(1.0) and (left, bottom) == pytest.approx((0.0, 1.0))
    viewport = (2.1, 1.1, 4.9, 1.9)
    cells = visible_cells(layer, viewport)
    assert cells == ["2:0", "3:0", "4:0"]
    for key in cells:
        x, y = (int(part) for part in key.split(":"))
        cell_left, cell_bottom = left + x * 1.0, bottom + y * 1.0
        assert cell_left <= viewport[2] and cell_left + 1.0 >= viewport[0]
        assert cell_bottom <= viewport[3] and cell_bottom + 1.0 >= viewport[1]
    assert visible_entity_indices(layer, viewport) == [2, 3, 4]
    assert visible_cells(layer, (9.0, 10.0, 9.8, 10.5)) == ["9:9"]
    assert visible_entity_indices(layer, (9.0, 10.0, 9.8, 10.5)) == [10]
    assert visible_cells(layer, (20.0, 20.0, 30.0, 30.0)) == []
    assert set(visible_cells(layer, (-1.0, -1.0, 11.0, 11.0))) == set(layer["grid"])


# --------------------------------------------------------------------------- 6. profile required / blocked


@pytest.mark.anyio
async def test_missing_profile_requires_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    project, _las = _make_project(tmp_path, "Scanner", "no_profile")
    controller = DxfPageController(project)
    assert controller.profile_required
    with pytest.raises(ValueError, match="profile"):
        controller.start_job("preview")
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="profile-required")
        await user.should_not_see("Preparation")
        _set(user, "profile-required", "sam3_outdoor")
        await user.should_see("Preparation")
    _no_error_logs(caplog)
    assert not DxfPageController(project).profile_required
    assert DxfPageController(project).profile == "sam3_outdoor"


@pytest.mark.anyio
async def test_flurseg_indoor_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    project, _las = _make_project(tmp_path, "Strasse", "flurseg")
    controller = DxfPageController(project)
    assert controller.profile == "flurseg_v1" and controller.blocked_message is None
    controller.set_mode_override("indoor")
    assert controller.blocked_message is not None and "outdoor" in controller.blocked_message
    assert not controller.export_enabled(None)
    with pytest.raises(ValueError, match="FlurSeg"):
        controller.start_job("preview")
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see("The FlurSeg v1 profile supports outdoor projects only", marker="blocked")
        await user.should_not_see("Preparation")
        _set(user, "mode-select", "outdoor")
        await user.should_see("Preparation")
    _no_error_logs(caplog)


@pytest.mark.anyio
async def test_dxf_page_without_classified_las_explains_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, heavy_calls: list[str]
) -> None:
    project = (tmp_path / "NoLas").resolve()
    project.mkdir()
    document = {"project": {"name": "NoLas"}, "workflow": {}}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    assert controller_module.dxf_unavailable_reason(project) == controller_module.DXF_NO_LAS_HINT
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see("Export a classified LAS first", marker="dxf-not-ready")
        await user.should_not_see("Preparation")
        await user.should_not_see(marker="las-path")
    assert heavy_calls == []
    assert not (project / "products" / "dxf").exists(), "the not-ready page must not create DXF state or caches"
    _no_error_logs(caplog)


# --------------------------------------------------------------------------- 7. GUI smoke


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["indoor", "outdoor"])
async def test_dxf_page_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, kind: str) -> None:
    project, las = _make_project(tmp_path, f"Smoke_{kind}", kind)
    if kind == "indoor":
        _generate(project, las, _prepare_indoor(project, las)[:1], "indoor")
    else:
        _generate(project, las, [None], "outdoor")
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see("DXF export")
        await user.should_see("Preparation")
        await user.should_see(marker="review-canvas")
        if kind == "indoor":
            await user.should_see("Side view and levels")
            await user.should_see("Align to gravity")
        else:
            await user.should_not_see("Side view and levels")
            await user.should_see("Site plan")
        canvas = _element(user, "review-canvas")
        assert canvas.props["payload-url"].startswith("/api/dxf/payload/")
        assert canvas.props["image-opacity"] == pytest.approx(1.0) and canvas.props["dxf-opacity"] == pytest.approx(1.0)
    _no_error_logs(caplog)


def test_review_http_routes_serve_cache_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app import application

    project, las = _make_project(tmp_path, "Routes", "outdoor")
    monkeypatch.setattr(app_state, "active_project_directory", project)
    assert application.dxf_review_payload("site").status_code == 404
    assert application.dxf_review_payload("..").status_code == 400
    assert application.dxf_review_background("site", "rgb").status_code == 404
    _generate(project, las, [None], "outdoor")
    payload = json.loads(application.dxf_review_payload("site").body)
    assert "S-GRND" in payload["layers"] and payload["layers"]["S-GRND"]["cell_size"] > 0
    image = application.dxf_review_background("site", "rgb")
    assert Path(image.path) == project / "products" / "dxf" / "site" / "background_rgb.png"
    assert application.dxf_review_background("site", "depth").status_code == 400


# --------------------------------------------------------------------------- 8. end to end without GUI


def test_end_to_end_indoor_and_outdoor_exports(tmp_path: Path) -> None:
    project, las = _make_project(tmp_path, "Musterhaus", "indoor")
    level_ids = _prepare_indoor(project, las)
    controller = DxfPageController(project)
    assert controller.mode == "indoor" and controller.alignment is not None
    assert [level["name"] for level in controller.available_levels()] == ["1", "2"]
    _generate(project, las, level_ids, "indoor")
    controller = DxfPageController(project)
    assert controller.all_exports_enabled()
    jobs.run_export(project, las, None, _no_progress, CancelToken(), _LOGGER, mode="indoor")
    export = project / "products" / "dxf" / "export"
    for name in ("Musterhaus_1.dxf", "Musterhaus_2.dxf"):
        document = ezdxf.readfile(export / name)
        assert not document.audit().has_errors
        assert len(document.modelspace()) > 1
    manifest = controller.last_export()
    assert manifest is not None and manifest["audit"] == "passed" and len(manifest["files"]) == 2

    site, site_las = _make_project(tmp_path, "Platz", "outdoor")
    _generate(site, site_las, [None], "outdoor")
    jobs.run_export(site, site_las, None, _no_progress, CancelToken(), _LOGGER, mode="outdoor")
    document = ezdxf.readfile(site / "products" / "dxf" / "export" / "Platz_site.dxf")
    assert not document.audit().has_errors


def test_two_point_scale_uses_unscaled_distance_and_rescales_levels(tmp_path: Path) -> None:
    project, _las = _make_project(tmp_path, "Scale", "indoor")
    controller = DxfPageController(project)
    controller.add_level(2.0, 4.0)
    controller.set_scale_factor(2.0)
    assert controller.available_levels()[0]["floor_z"] == pytest.approx(4.0)
    factor = controller.measure_scale([0.0, 0.0], [4.0, 0.0], 3.0)
    assert factor == pytest.approx(1.5)
    assert controller.available_levels()[0]["floor_z"] == pytest.approx(3.0)
    assert controller.available_levels()[0]["ceiling_z"] == pytest.approx(6.0)
