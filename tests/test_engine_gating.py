"""Engine gating: DXF/IFC exist only for SAM3 projects; FlurSeg projects deliver point clouds only."""

from __future__ import annotations

import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import use_language
from samcloud.app import layout
from samcloud.app.dxf_controller import dxf_unavailable_reason
from samcloud.app.pages import home
from samcloud.app.pages.projects import creation_engine_and_auto_dxf
from samcloud.app.processing import ProcessingManager, auto_dxf_requested, project_uses_flurseg
from samcloud.core.project_manager import PROJECT_FILENAME, ProjectManager
from samcloud.engine import auto_dxf, pipeline


def _write_project(directory: Path, engine: str, class_set: str = "outdoor", auto_dxf_flag: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    project = {"settings": {"classification": {"engine": engine, "class_set": class_set}, "workflow": {"auto_dxf": auto_dxf_flag}}}
    (directory / PROJECT_FILENAME).write_text(yaml.safe_dump(project), encoding="utf-8")
    return directory


def _classified(directory: Path) -> None:
    target = directory / "products" / "classified"
    target.mkdir(parents=True, exist_ok=True)
    (target / "classified.ply").write_bytes(b"ply\n")


def _command(engine: str, auto_dxf_flag: bool, class_set: str = "outdoor") -> list[str]:
    project = {
        "settings": {
            "workflow": {"auto_dxf": auto_dxf_flag},
            "classification": {"class_set": class_set, "engine": engine},
            "coordinate_systems": {},
        },
        "acquisition": {"type": "drone_gps", "sources": {"image_directory": "D:/images"}, "georeferencing": {}},
    }
    return ProcessingManager._pipeline_command(object(), Path("D:/demo"), project, "none")


# 1 ---------------------------------------------------------------- dxf_unavailable_reason
def test_dxf_unavailable_for_flurseg_available_for_sam3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_language(monkeypatch, "de")
    flur = _write_project(tmp_path / "flur", "flurseg")
    sam = _write_project(tmp_path / "sam", "sam3")
    _classified(flur)
    _classified(sam)
    assert project_uses_flurseg(flur) is True
    assert project_uses_flurseg(sam) is False
    assert dxf_unavailable_reason(flur) == "DXF/IFC nur mit SAM3"
    assert dxf_unavailable_reason(sam) is None


def test_flurseg_on_indoor_class_set_is_still_sam3(tmp_path: Path) -> None:
    project = _write_project(tmp_path / "indoor", "flurseg", class_set="indoor")
    _classified(project)
    assert project_uses_flurseg(project) is False
    assert dxf_unavailable_reason(project) is None


# 2 ---------------------------------------------------------------- navigation
def test_navigation_hides_dxf_for_flurseg(tmp_path: Path) -> None:
    flur = _write_project(tmp_path / "flur", "flurseg")
    sam = _write_project(tmp_path / "sam", "sam3")
    flur_paths = [path for _, path, _ in layout._visible_navigation(True, flur)]
    sam_paths = [path for _, path, _ in layout._visible_navigation(True, sam)]
    assert "/dxf" not in flur_paths
    assert "/dense" in flur_paths and "/settings" in flur_paths
    assert "/dxf" in sam_paths
    assert "/dxf" not in [path for _, path, _ in layout._visible_navigation(False, None)]


def test_home_workflow_has_no_dxf_step_for_flurseg() -> None:
    flur = {"settings": {"classification": {"engine": "flurseg", "class_set": "outdoor"}}}
    sam = {"settings": {"classification": {"engine": "sam3", "class_set": "outdoor"}}}
    assert "dxf" not in [key for _, key in home._workflow_steps(flur)]
    assert "dxf" in [key for _, key in home._workflow_steps(sam)]
    assert "dxf" in [key for _, key in home._workflow_steps(None)]


# 3 ---------------------------------------------------------------- command line
def test_command_line_auto_dxf_zero_for_flurseg_even_if_enabled() -> None:
    command = _command("flurseg", True)
    assert command[command.index("--auto-dxf") + 1] == "0"
    assert command[command.index("--classification-engine") + 1] == "flurseg"


def test_command_line_auto_dxf_one_for_sam3() -> None:
    command = _command("sam3", True)
    assert command[command.index("--auto-dxf") + 1] == "1"
    off = _command("sam3", False)
    assert off[off.index("--auto-dxf") + 1] == "0"
    assert auto_dxf_requested({"classification": {"engine": "sam3"}, "workflow": {}}) is True


# 4 ---------------------------------------------------------------- pipeline guard
def _args(tmp_path: Path, engine: str, class_set: str = "outdoor") -> Namespace:
    return Namespace(
        auto_dxf="1", classification_engine=engine, outdir=str(tmp_path / "proj" / "products"),
        progress_file=None, capture_type="drone_gps", class_set=class_set,
    )


@pytest.mark.parametrize("class_set", ["outdoor", "indoor"])
def test_stage_skips_flurseg_without_calling_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], class_set: str,
) -> None:
    calls: list[str] = []
    fake = types.ModuleType("samcloud.drone.run_environment")
    fake.run_environment = lambda *a, **k: calls.append("run_environment")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "samcloud.drone.run_environment", fake)
    monkeypatch.setattr(auto_dxf, "run_after_pipeline", lambda *a, **k: calls.append("run_after_pipeline"))
    monkeypatch.setattr(auto_dxf, "pipeline_run_completed", lambda *a, **k: True)
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", None)
    pipeline._request_auto_dxf(_args(tmp_path, "flurseg", class_set))
    pipeline.run_auto_dxf_stage()
    assert calls == []
    assert capsys.readouterr().out.count("FlurSeg") == 1


def test_stage_still_runs_environment_for_sam3_outdoor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake = types.ModuleType("samcloud.drone.run_environment")
    fake.run_environment = lambda *a, **k: calls.append("run_environment") or {"steps": {}}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "samcloud.drone.run_environment", fake)
    monkeypatch.setattr(auto_dxf, "pipeline_run_completed", lambda *a, **k: True)
    monkeypatch.setattr(pipeline, "_record_environment_stage", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_AUTO_DXF_REQUEST", None)
    pipeline._request_auto_dxf(_args(tmp_path, "sam3"))
    pipeline.run_auto_dxf_stage()
    assert calls == ["run_environment"]


# 5 ---------------------------------------------------------------- project creation
def test_creation_helper_flurseg_forces_auto_dxf_off() -> None:
    assert creation_engine_and_auto_dxf("outdoor", "flurseg", True) == ("flurseg", False)
    assert creation_engine_and_auto_dxf("outdoor", "sam3") == ("sam3", True)
    assert creation_engine_and_auto_dxf("outdoor", None) == ("sam3", True)
    assert creation_engine_and_auto_dxf("indoor", "flurseg", True) == ("sam3", True)
    assert creation_engine_and_auto_dxf("outdoor", "sam3", False) == ("sam3", False)


def test_project_manager_stores_auto_dxf_by_engine(tmp_path: Path) -> None:
    def data(engine: str, **extra: Any) -> dict:
        return ProjectManager._new_project_data(
            "demo", str(tmp_path), capture_type="drone_gps", class_set="outdoor", panorama_image_directory="",
            classification_engine=engine, panorama_video_path="", frame_interval_seconds=None,
            rtk_enabled=True, control_points_mode="auto", **extra,
        )

    assert data("sam3")["settings"]["workflow"]["auto_dxf"] is True
    assert data("flurseg", auto_dxf=True)["settings"]["workflow"]["auto_dxf"] is False
    assert data("sam3", auto_dxf=False)["settings"]["workflow"]["auto_dxf"] is False


# ---------------------------------------------------------------- dxf page text and settings rule
def test_dxf_page_text_shows_engine_reason_for_flurseg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app.pages.dxf import page_blocked_text

    use_language(monkeypatch, "de")
    flur = _write_project(tmp_path / "flur", "flurseg")
    _classified(flur)
    sam_open = _write_project(tmp_path / "sam_ready", "sam3")
    _classified(sam_open)
    sam_new = _write_project(tmp_path / "sam_new", "sam3")
    assert page_blocked_text(flur) == "DXF/IFC nur mit SAM3"
    assert page_blocked_text(sam_open) is None
    blocked = page_blocked_text(sam_new)
    assert blocked is not None and blocked != "DXF/IFC nur mit SAM3"


def test_settings_page_uses_engine_rule_for_auto_dxf() -> None:
    # Save rule used by settings.save_settings: FlurSeg (outdoor) stores False, everything else keeps the switch.
    assert creation_engine_and_auto_dxf("outdoor", "flurseg", True)[1] is False
    assert creation_engine_and_auto_dxf("indoor", "flurseg", True)[1] is True
    assert creation_engine_and_auto_dxf("outdoor", "sam3", True)[1] is True
    source = (Path(__file__).resolve().parents[1] / "samcloud" / "app" / "pages" / "settings.py").read_text(encoding="utf-8")
    assert 'workflow["auto_dxf"] = creation_engine_and_auto_dxf(' in source
    assert "auto_dxf.set_visibility(" in source
