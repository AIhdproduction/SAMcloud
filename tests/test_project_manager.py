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


def test_create_project_defaults_to_automatic_processing_without_control_points(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")

    project = manager.load(manager.create_project("Automatic run", "D:/images"))

    assert project["workflow"]["control_points_mode"] == "none"
    assert project["workflow"]["control_points"] == "skipped"
    assert project["settings"]["workflow"]["control_points_enabled"] is False


def test_create_project_can_pause_after_alignment_for_control_points(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")

    project = manager.load(
        manager.create_project(
            "Survey with points",
            "D:/images",
            control_points_mode="after_alignment",
        )
    )

    assert project["workflow"]["control_points_mode"] == "after_alignment"
    assert project["workflow"]["control_points"] == "not_started"
    assert project["settings"]["workflow"]["control_points_enabled"] is True


def test_create_project_accepts_automatic_control_point_mode(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path)
    project_directory = ProjectManager(tmp_path).create_project(
        "auto_test", str(tmp_path), control_points_mode="automatic"
    )
    project = manager.load(project_directory)

    assert project["workflow"]["control_points_mode"] == "automatic"
    assert project["workflow"]["control_points"] == "not_started"


def test_indoor_project_enables_apriltags_by_default(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    directory = manager.create_project("indoor site", "D:/images", capture_type="camera", class_set="indoor")

    assert manager.load(directory)["settings"]["apriltags"]["enabled"] is True


def test_outdoor_project_disables_apriltags_by_default(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    directory = manager.create_project("outdoor site", "D:/images", capture_type="camera", class_set="outdoor")

    assert manager.load(directory)["settings"]["apriltags"]["enabled"] is False


def test_indoor_project_can_opt_out_of_apriltags(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    directory = manager.create_project(
        "indoor off", "D:/images", capture_type="camera", class_set="indoor", apriltags_enabled=False,
    )

    project = manager.load(directory)
    assert project["settings"]["apriltags"]["enabled"] is False
    assert project["settings"]["apriltags"]["last_printed_id"] == -1


def _write_tag_csv(path: Path, rows: str) -> Path:
    path.write_text("id,x,y,z,role\n" + rows, encoding="utf-8")
    return path


def test_indoor_project_with_apriltag_coordinates_sets_crs_and_export(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    csv_path = _write_tag_csv(tmp_path / "tags.csv", "1,2600000.5,1200000.5,400.1,pass\n2,2600005,1200000,400.2,control\n")

    directory = manager.create_project(
        "tag site", "D:/images", capture_type="camera", class_set="indoor",
        apriltag_coordinates_csv=csv_path, apriltag_crs="EPSG:2056",
    )

    written = directory / "control_points" / "apriltags.csv"
    assert written.is_file()
    lines = written.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id,x,y,z,role" and len(lines) == 3
    project = manager.load(directory)
    assert project["settings"]["control_points"]["crs"] == "EPSG:2056"
    assert project["settings"]["coordinate_systems"]["export_crs"] == "EPSG:2056"
    assert project["settings"]["coordinate_systems"]["working_crs"] == "LOCAL"
    assert project["settings"]["coordinate_systems"]["input_crs"] == "LOCAL"


def test_apriltag_coordinates_with_duplicate_id_are_rejected(tmp_path: Path) -> None:
    import pytest

    manager = ProjectManager(tmp_path / "projects")
    csv_path = _write_tag_csv(tmp_path / "dup.csv", "3,1,2,3,pass\n3,4,5,6,pass\n")

    with pytest.raises(ValueError):
        manager.create_project(
            "dup site", "D:/images", capture_type="camera", class_set="indoor", apriltag_coordinates_csv=csv_path,
        )
    assert not (tmp_path / "projects" / "dup-site").exists()


def test_indoor_project_with_local_apriltag_coordinates_stays_local(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    csv_path = _write_tag_csv(tmp_path / "local_tags.csv", "1,10.5,20.5,1.1,pass\n2,15,20,1.2,pass\n3,10,26,1.0,control\n")

    directory = manager.create_project(
        "local site", "D:/images", capture_type="camera", class_set="indoor",
        apriltag_coordinates_csv=csv_path, apriltag_crs="LOCAL",
    )

    settings = manager.load(directory)["settings"]
    assert settings["control_points"]["crs"] == "LOCAL"
    assert settings["coordinate_systems"]["export_crs"] == "LOCAL"
    assert settings["coordinate_systems"]["working_crs"] == "LOCAL"
    assert settings["coordinate_systems"]["input_crs"] == "LOCAL"


def test_drone_gps_project_rejects_local_apriltag_crs(tmp_path: Path) -> None:
    import pandas as pd
    import pytest

    from samcloud.core.project_manager import set_apriltag_coordinates

    manager = ProjectManager(tmp_path / "projects")
    csv_path = _write_tag_csv(tmp_path / "local_tags.csv", "1,10.5,20.5,1.1,pass\n")

    with pytest.raises(ValueError, match="ohne GPS"):
        manager.create_project(
            "drone local", "D:/images", capture_type="drone_gps", class_set="outdoor",
            apriltag_coordinates_csv=csv_path, apriltag_crs="LOCAL",
        )
    assert not (tmp_path / "projects" / "drone-local").exists()

    directory = manager.create_project("drone plain", "D:/images", capture_type="drone_gps", class_set="outdoor")
    with pytest.raises(ValueError, match="ohne GPS"):
        set_apriltag_coordinates(
            directory, manager.load(directory), pd.DataFrame([{"id": 1, "x": 1.0, "y": 2.0, "z": 3.0, "role": "pass"}]), "LOCAL",
        )
