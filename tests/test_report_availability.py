"""Availability rules of the processing protocol."""

from __future__ import annotations

from pathlib import Path

from samcloud.report.availability import protocol_availability
from test_share_export import make_project

CSV = "name,x,y,z\nP1,1,2,3\n"


def _csv(project: Path, text: str) -> None:
    (project / "control_points" / "control_points.csv").write_text(text, encoding="utf-8")


def _report(project: Path) -> None:
    (project / "reports" / "control_points_report.json").write_text("{}", encoding="utf-8")


def test_step6_missing_disables(tmp_path: Path) -> None:
    project = make_project(tmp_path, refinement="running", with_cloud=False)
    result = protocol_availability(project)
    assert not result.enabled and "step6" in result.missing


def test_points_without_report_missing_control_points(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    _csv(project, CSV)
    result = protocol_availability(project)
    assert not result.enabled and result.missing == ["control_points"] and not result.without_control_points


def test_report_and_step6_enabled(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    _csv(project, CSV)
    _report(project)
    result = protocol_availability(project)
    assert result.enabled and result.missing == [] and not result.without_control_points


def test_no_csv_is_without_control_points(tmp_path: Path) -> None:
    result = protocol_availability(make_project(tmp_path))
    assert result.enabled and result.without_control_points


def test_header_only_csv_is_without_control_points(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    _csv(project, "name,x,y,z\n")
    result = protocol_availability(project)
    assert result.enabled and result.without_control_points
