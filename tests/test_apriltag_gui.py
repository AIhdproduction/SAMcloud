"""AprilTag GUI: print dialog on the control-points page, scale report table, project-creation switch."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import yaml
from nicegui import ui
from nicegui.elements.upload_files import SmallFileUpload
from nicegui.testing.user import User
from nicegui.testing.user_download import UserDownload
from nicegui.testing.user_interaction import UserInteraction
from nicegui.testing.user_simulation import user_simulation

from samcloud.app.pages import control_points as control_points_page
from samcloud.app.pages import projects as projects_page
from samcloud.app.state import state as app_state

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _make_project(root: Path, name: str, *, last_printed_id: int = -1) -> Path:
    project = (root / name).resolve()
    (project / "control_points").mkdir(parents=True)
    (project / "reports").mkdir()
    document = {
        "project": {"name": name, "updated_at": "2026-01-01T00:00:00+00:00"},
        "settings": {
            "classification": {"class_set": "indoor"},
            "apriltags": {"enabled": True, "outlier_pct": 3.0, "last_printed_id": last_printed_id},
        },
        "workflow": {},
    }
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return project


def _control_points_route() -> None:
    control_points_page.render()


def _projects_route() -> None:
    projects_page.render()


# NiceGUI's test reset drops sys.modules entries of page modules outside "tests."; keep samcloud loaded.
_control_points_route.__module__ = "tests.apriltag_gui_route"
_projects_route.__module__ = "tests.apriltag_gui_route"


@asynccontextmanager
async def _page_user(path: str, route: Any, project: Path | None, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[User]:
    monkeypatch.setattr(app_state, "active_project_directory", project)
    async with user_simulation(root=lambda: None) as user:
        ui.page(path)(route)
        await user.open(path)
        yield user


def _element(user: User, marker: str) -> Any:
    elements = user.find(marker=marker).elements
    assert len(elements) == 1, f"expected one element marked {marker!r}, found {len(elements)}"
    return next(iter(elements))


def _set(user: User, marker: str, value: Any) -> None:
    with user:
        _element(user, marker).value = value


def _record_downloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture ui.download calls. NiceGUI's simulated user re-installs its own download object on every
    attribute access, so the recorder is patched on the class and keeps filename and media type."""
    downloads: list[dict[str, Any]] = []

    def fake_call(self: Any, src: Any, filename: str | None = None, media_type: str = "") -> None:
        downloads.append({"data": src, "filename": filename, "media_type": media_type})

    monkeypatch.setattr(UserDownload, "__call__", fake_call)
    return downloads


def _saved_last_printed_id(project: Path) -> int:
    return yaml.safe_load((project / "project.samcloud.yml").read_text(encoding="utf-8"))["settings"]["apriltags"]["last_printed_id"]


@pytest.mark.anyio
async def test_print_dialog_downloads_pdf_and_remembers_last_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path, "tags")
    async with _page_user("/control-points", _control_points_route, project, monkeypatch) as user:
        downloads = _record_downloads(monkeypatch)
        await user.should_see(marker="apriltag-print")
        user.find(marker="apriltag-print").click()
        assert _element(user, "apriltag-start").value == 0
        _set(user, "apriltag-count", 2)
        _set(user, "apriltag-start", 5)
        user.find(marker="apriltag-pdf").click()

        assert len(downloads) == 1
        assert downloads[0]["media_type"] == "application/pdf"
        assert downloads[0]["filename"] == "apriltags_5-6.pdf"
        assert downloads[0]["data"][:4] == b"%PDF"
        assert _saved_last_printed_id(project) == 6

        # The next dialog proposes the id after the last printed one.
        user.find(marker="apriltag-print").click()
        assert _element(user, "apriltag-start").value == 7


@pytest.mark.anyio
async def test_print_dialog_rejects_ids_beyond_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path, "toomany", last_printed_id=3)
    async with _page_user("/control-points", _control_points_route, project, monkeypatch) as user:
        downloads = _record_downloads(monkeypatch)
        user.find(marker="apriltag-print").click()
        assert _element(user, "apriltag-start").value == 4
        _set(user, "apriltag-count", 5)
        _set(user, "apriltag-start", 584)
        user.find(marker="apriltag-pdf").click()

        assert downloads == []
        assert _saved_last_printed_id(project) == 3


@pytest.mark.anyio
async def test_report_table_shows_scale_and_tags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path, "report")
    report = {
        "mode": "scale", "scale_factor": 2.3712, "scale_applied": 2.3712,
        "warnings": ["Nur 2 AprilTag(s) gefunden – keine Kontrolle möglich"],
        "tags": [
            {"id": 11, "images": 4, "size_mm_before": 63.3, "size_mm_after": 150.1, "deviation_pct": 0.07, "status": "ok"},
            {"id": 12, "images": 3, "size_mm_before": 62.9, "size_mm_after": 149.2, "deviation_pct": -0.53, "status": "ok"},
        ],
    }
    (project / "reports" / "apriltag_report.json").write_text(json.dumps(report), encoding="utf-8")
    async with _page_user("/control-points", _control_points_route, project, monkeypatch) as user:
        table = _element(user, "apriltag-report")
        assert [row["id"] for row in table.rows] == [11, 12]
        assert table.rows[1]["after"] == "149.2"
        assert table.rows[1]["deviation"] == "-0.53"
        assert [column["label"] for column in table.columns] == ["ID", "Bilder", "Größe vorher mm", "Größe nachher mm", "Abweichung %", "Status"]
        await user.should_see("Faktor 2.3712")
        await user.should_see("keine Kontrolle möglich")


@pytest.mark.anyio
async def test_no_report_table_without_report_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path, "noreport")
    async with _page_user("/control-points", _control_points_route, project, monkeypatch) as user:
        await user.should_see(marker="apriltag-print")
        await user.should_not_see(marker="apriltag-report")


def _select_by_label(user: User, label: str) -> Any:
    with user:
        selects = [e for e in user.find(kind=ui.select).elements if e.props.get("label") == label]
    assert len(selects) == 1
    return selects[0]


@pytest.mark.anyio
async def test_projects_page_shows_apriltag_switch_for_both_class_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        class_set = _select_by_label(user, "Semantic class set")
        await asyncio.sleep(0.3)
        assert class_set.value == "outdoor"
        await user.should_see(marker="apriltag-switch")
        assert _element(user, "apriltag-switch").value is False

        with user:
            class_set.value = "indoor"
        await asyncio.sleep(0.3)
        await user.should_see(marker="apriltag-switch")
        assert _element(user, "apriltag-switch").value is True

        with user:
            class_set.value = "outdoor"
        await asyncio.sleep(0.3)
        await user.should_see(marker="apriltag-switch")
        assert _element(user, "apriltag-switch").value is False


@pytest.mark.anyio
async def test_projects_page_panorama_selects_indoor_and_stays_manual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        class_set = _select_by_label(user, "Semantic class set")
        capture_type = _select_by_label(user, "Capture type")
        await asyncio.sleep(0.3)
        assert class_set.value == "outdoor"

        with user:
            capture_type.value = "panorama_360"
        await asyncio.sleep(0.3)
        assert class_set.value == "indoor"
        assert _element(user, "apriltag-switch").value is True

        with user:
            class_set.value = "outdoor"
        await asyncio.sleep(0.3)
        assert class_set.value == "outdoor"
        assert capture_type.value == "panorama_360"
        assert _element(user, "apriltag-switch").value is False

        with user:
            _element(user, "apriltag-switch").value = True
        await asyncio.sleep(0.3)
        assert class_set.value == "outdoor"
        assert _element(user, "apriltag-switch").value is True

        with user:
            capture_type.value = "camera"
        await asyncio.sleep(0.3)
        assert class_set.value == "outdoor"


@pytest.mark.anyio
async def test_projects_page_outdoor_create_passes_apriltags_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create_project(self: Any, name: str, image_directory: str, **kwargs: Any) -> Path:
        calls.append({"name": name, **kwargs})
        raise ValueError("stop after capture")

    monkeypatch.setattr(type(app_state.projects), "create_project", fake_create_project)
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        class_set = _select_by_label(user, "Semantic class set")
        await asyncio.sleep(0.3)
        assert class_set.value == "outdoor"
        with user:
            _element(user, "apriltag-switch").value = True
        await asyncio.sleep(0.3)
        with user:
            # The header also has a "Create project" navigation button; the form button has the highest id.
            form_button = max(user.find("Create project", kind=ui.button).elements, key=lambda e: e.id)
        UserInteraction(user, {form_button}, "Create project").click()
        await asyncio.sleep(0.3)

    assert len(calls) == 1
    assert calls[0]["class_set"] == "outdoor"
    assert calls[0]["apriltags_enabled"] is True
    assert calls[0]["apriltag_coordinates_csv"] is None


async def _import_tags(user: User, text: str) -> None:
    upload = _element(user, "apriltag-coords-import")
    file = SmallFileUpload(name="tags.csv", content_type="text/csv", _data=text.encode("utf-8"))
    with user:
        await upload.handle_uploads([file])
    await asyncio.sleep(0.3)


@pytest.mark.anyio
async def test_coords_import_and_save_writes_csv_and_crs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path, "coords")
    async with _page_user("/control-points", _control_points_route, project, monkeypatch) as user:
        await _import_tags(user, "id,x,y,z,role\n1,2600000,1200000,400,pass\n2,2600010,1200000,400,pass\nTAG7,2600010,1200010,400,control\n")
        table = _element(user, "apriltag-coords-table")
        assert [row["id"] for row in table.rows] == [1, 2, 7]
        with user:
            crs = [e for e in user.find(kind=ui.select).elements if e.props.get("label") == "Control-point CRS"][0]
            crs.value = "EPSG:2056"
        user.find(marker="apriltag-coords-save").click()
        await asyncio.sleep(0.3)

    written = project / "control_points" / "apriltags.csv"
    assert written.is_file()
    lines = written.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id,x,y,z,role" and len(lines) == 4
    assert lines[3].startswith("7,") and lines[3].endswith("control")
    document = yaml.safe_load((project / "project.samcloud.yml").read_text(encoding="utf-8"))
    assert document["settings"]["control_points"]["crs"] == "EPSG:2056"


@pytest.mark.anyio
async def test_coords_warning_shown_with_two_pass_tags_hidden_with_three(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path, "warn")
    async with _page_user("/control-points", _control_points_route, project, monkeypatch) as user:
        await _import_tags(user, "id,x,y,z,role\n1,0,0,0,pass\n2,1,0,0,pass\n3,0,1,0,control\n")
        await user.should_see(marker="apriltag-coords-warning")
        await _import_tags(user, "id,x,y,z,role\n1,0,0,0,pass\n2,1,0,0,pass\n3,0,1,0,pass\n")
        await user.should_not_see(marker="apriltag-coords-warning")


@pytest.mark.anyio
async def test_projects_page_coords_upload_and_crs_follow_switch_for_both_class_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        class_set = _select_by_label(user, "Semantic class set")
        await asyncio.sleep(0.3)
        for chosen in ("outdoor", "indoor"):
            with user:
                class_set.value = chosen
            await asyncio.sleep(0.3)
            with user:
                _element(user, "apriltag-switch").value = False
            await asyncio.sleep(0.3)
            await user.should_not_see(marker="apriltag-coords-upload")
            await user.should_not_see(marker="apriltag-crs")

            with user:
                _element(user, "apriltag-switch").value = True
            await asyncio.sleep(0.3)
            await user.should_see(marker="apriltag-coords-upload")
            await user.should_see(marker="apriltag-crs")
            await user.should_see("Koordinaten der Tag-Mitte je ID")


@pytest.mark.anyio
async def test_projects_page_tag_crs_select_offers_local_and_shows_hint_only_for_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        class_set = _select_by_label(user, "Semantic class set")
        await asyncio.sleep(0.3)
        with user:
            class_set.value = "indoor"
        await asyncio.sleep(0.3)
        crs = _element(user, "apriltag-crs")
        assert "LOCAL" in crs.options
        await user.should_not_see(marker="apriltag-crs-local-hint")
        with user:
            crs.value = "LOCAL"
        await asyncio.sleep(0.3)
        await user.should_see(marker="apriltag-crs-local-hint")
        await user.should_see("Kein Export in LV95/WGS84 möglich")
        with user:
            crs.value = "EPSG:2056"
        await asyncio.sleep(0.3)
        await user.should_not_see(marker="apriltag-crs-local-hint")


@pytest.mark.anyio
async def test_control_points_page_crs_select_offers_local_and_hint_follows_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_project(tmp_path, "localcrs")
    async with _page_user("/control-points", _control_points_route, project, monkeypatch) as user:
        crs = _element(user, "control-points-crs")
        assert "LOCAL" in crs.options
        await user.should_not_see(marker="control-points-crs-local-hint")
        with user:
            crs.value = "LOCAL"
        await asyncio.sleep(0.3)
        await user.should_see(marker="control-points-crs-local-hint")
        await user.should_see("Kein Export in LV95/WGS84 möglich")
        with user:
            crs.value = "EPSG:4979"
        await asyncio.sleep(0.3)
        await user.should_not_see(marker="control-points-crs-local-hint")
    document = yaml.safe_load((project / "project.samcloud.yml").read_text(encoding="utf-8"))
    assert document["settings"]["control_points"]["crs"] == "EPSG:4979"
