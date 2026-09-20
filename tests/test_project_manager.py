"""Tests for persistent project creation and reopening."""

from pathlib import Path

from samcloud.core.project_manager import PROJECT_FILENAME, ProjectManager


def test_create_project_writes_metadata(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")

    project_directory = manager.create_project("Test site", "D:/images")

    assert (project_directory / PROJECT_FILENAME).exists()
    assert manager.load(project_directory)["paths"]["image_directory"] == "D:/images"
