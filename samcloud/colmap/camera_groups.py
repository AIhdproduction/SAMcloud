"""Camera grouping helpers for multi-flight drone image sets."""

from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
import re
import sqlite3


_SEQUENCE_NUMBER = re.compile(r"_(\d{4})_")


def assign_camera_groups(
    image_names: list[str], grouping: str = "sequence_number"
) -> dict[str, int]:
    """Assign deterministic camera groups from image names and relative folders."""
    if grouping not in {"sequence_number", "folder", "single", "per_image"}:
        raise ValueError(f"Unsupported camera grouping: {grouping}")
    names = sorted(image_names, key=lambda name: name.replace("\\", "/").casefold())
    if grouping == "single":
        return {name: 0 for name in names}
    if grouping == "per_image":
        return {name: index for index, name in enumerate(names)}

    folders: dict[str, list[str]] = {}
    for name in names:
        normalized = name.replace("\\", "/")
        folder = str(PurePosixPath(normalized).parent)
        folders.setdefault(folder, []).append(name)
    assigned: dict[str, int] = {}
    next_group = 0
    for folder in sorted(folders, key=str.casefold):
        if grouping == "folder":
            for name in folders[folder]:
                assigned[name] = next_group
            next_group += 1
            continue
        previous: int | None = None
        current_group = next_group
        for name in folders[folder]:
            match = _SEQUENCE_NUMBER.search(PurePosixPath(name.replace("\\", "/")).name)
            number = int(match.group(1)) if match else None
            if number is not None and previous is not None and number < previous:
                current_group += 1
            assigned[name] = current_group
            if number is not None:
                previous = number
        next_group = current_group + 1
    return assigned


def rewrite_camera_groups(database: str, groups: dict[str, int]) -> dict[int, int]:
    """Make every logical group share one compatible COLMAP camera row."""
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        camera_columns = [row[1] for row in connection.execute("PRAGMA table_info(cameras)")]
        required = {"camera_id", "model", "width", "height"}
        if not required.issubset(camera_columns):
            raise ValueError("The database does not contain a compatible COLMAP cameras table")
        selected_columns = ", ".join(camera_columns)
        cameras = {
            row[0]: dict(zip(camera_columns, row))
            for row in connection.execute(f"SELECT {selected_columns} FROM cameras")
        }
        images = connection.execute("SELECT image_id, name, camera_id FROM images ORDER BY name").fetchall()
        allocated: dict[tuple[int, int, int, int], int] = {}
        group_counts: Counter[int] = Counter()
        for image_id, name, source_camera_id in images:
            group = groups.get(name, 0)
            source = cameras[source_camera_id]
            key = (group, source["model"], source["width"], source["height"])
            target_camera_id = allocated.get(key)
            if target_camera_id is None:
                if source_camera_id not in allocated.values():
                    target_camera_id = source_camera_id
                else:
                    values = [source[column] for column in camera_columns if column != "camera_id"]
                    columns = [column for column in camera_columns if column != "camera_id"]
                    placeholders = ", ".join("?" for _ in values)
                    cursor = connection.execute(
                        f"INSERT INTO cameras ({', '.join(columns)}) VALUES ({placeholders})", values
                    )
                    target_camera_id = int(cursor.lastrowid)
                    cameras[target_camera_id] = dict(zip(columns, values)) | {"camera_id": target_camera_id}
                allocated[key] = target_camera_id
            connection.execute("UPDATE images SET camera_id = ? WHERE image_id = ?", (target_camera_id, image_id))
            group_counts[group] += 1
        has_pose_priors = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pose_priors'"
        ).fetchone()
        if has_pose_priors:
            connection.execute(
                "UPDATE pose_priors SET corr_sensor_id = "
                "(SELECT camera_id FROM images WHERE image_id = pose_priors.corr_data_id) "
                "WHERE corr_sensor_type = 0"
            )
        return dict(group_counts)
