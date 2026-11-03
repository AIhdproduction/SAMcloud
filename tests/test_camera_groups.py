"""Tests for DJI-style flight grouping and COLMAP database camera updates."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from samcloud.colmap.camera_groups import assign_camera_groups, rewrite_camera_groups


def test_sequence_number_resets_create_three_groups() -> None:
    names = [
        "DJI_20250101123000_0001_D.JPG",
        "DJI_20250101123000_0002_D.JPG",
        "DJI_20250101124000_0001_D.JPG",
        "DJI_20250101124000_0002_D.JPG",
        "DJI_20250101125000_0001_D.JPG",
    ]
    groups = assign_camera_groups(names)
    assert [groups[name] for name in names] == [0, 0, 1, 1, 2]


def test_sequence_input_is_sorted_before_grouping() -> None:
    names = ["DJI_20250101123000_0002_D.JPG", "DJI_20250101123000_0001_D.JPG"]
    assert set(assign_camera_groups(names).values()) == {0}


def test_names_without_sequence_pattern_use_one_group() -> None:
    names = ["photo-b.jpg", "photo-a.jpg", "photo-c.jpg"]
    assert set(assign_camera_groups(names).values()) == {0}


def test_folder_mode_assigns_one_group_per_subfolder() -> None:
    names = ["flight_b/image.jpg", "flight_a/image.jpg", "flight_b/other.jpg"]
    groups = assign_camera_groups(names, "folder")
    assert groups["flight_a/image.jpg"] != groups["flight_b/image.jpg"]
    assert groups["flight_b/image.jpg"] == groups["flight_b/other.jpg"]


def test_database_rewrite_shares_one_camera_per_group(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE cameras ("
            "camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "
            "model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, "
            "params BLOB, prior_focal_length INTEGER NOT NULL);"
            "CREATE TABLE images ("
            "image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, name TEXT NOT NULL UNIQUE, "
            "camera_id INTEGER NOT NULL);"
        )
        for camera_id in range(1, 5):
            connection.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) "
                "VALUES (?, 4, 4000, 3000, ?, 1)",
                (camera_id, b"params"),
            )
        for image_id, name in enumerate(("a_0001_x.jpg", "a_0002_x.jpg", "b_0001_x.jpg", "b_0002_x.jpg"), 1):
            connection.execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                (image_id, name, image_id),
            )
    groups = assign_camera_groups(["a_0001_x.jpg", "a_0002_x.jpg", "b_0001_x.jpg", "b_0002_x.jpg"])
    counts = rewrite_camera_groups(str(database), groups)
    with sqlite3.connect(database) as connection:
        camera_ids = dict(connection.execute("SELECT name, camera_id FROM images"))
    assert counts == {0: 2, 1: 2}
    assert camera_ids["a_0001_x.jpg"] == camera_ids["a_0002_x.jpg"]
    assert camera_ids["b_0001_x.jpg"] == camera_ids["b_0002_x.jpg"]
    assert camera_ids["a_0001_x.jpg"] != camera_ids["b_0001_x.jpg"]
