"""Tests for persistent project creation and reopening."""

from pathlib import Path

from samcloud.core.project_manager import PROJECT_FILENAME, ProjectManager


def test_create_project_writes_metadata(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")

    project_directory = manager.create_project("Test site", "D:/images")

    assert (project_directory / PROJECT_FILENAME).exists()
    assert manager.load(project_directory)["paths"]["image_directory"] == "D:/images"


def test_create_project_stores_camera_capture_configuration(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")

    project_directory = manager.create_project(
        "Indoor scan",
        "D:/images",
        capture_type="camera",
        class_set="indoor",
    )

    project = manager.load(project_directory)
    assert project["acquisition"]["type"] == "camera"
    assert project["acquisition"]["georeferencing"]["mode"] == "local"
    assert project["settings"]["classification"]["class_set"] == "indoor"
    assert project["settings"]["coordinate_systems"]["working_crs"] == "LOCAL"


def test_create_project_stores_panorama_sources(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")

    project_directory = manager.create_project(
        "360 tour",
        capture_type="panorama_360",
        class_set="indoor",
        panorama_image_directory="D:/panoramas",
        panorama_video_path="D:/tour.mp4",
        frame_interval_seconds=0.5,
    )

    project = manager.load(project_directory)
    assert project["acquisition"]["sources"]["panorama_image_directory"] == "D:/panoramas"
    assert project["acquisition"]["sources"]["panorama_video_path"] == "D:/tour.mp4"
    assert project["acquisition"]["panorama"]["frame_interval_seconds"] == 0.5
