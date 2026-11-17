"""AprilTag report frame transform and indoor tag photo payload."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from samcloud.app.gcp_review import apriltag_marker_payload, has_apriltag_review_data, marker_payload
from samcloud.control_points.apriltags import TagObservation, save_tag_detections, transform_tag_report

ROTATION = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]  # +90 degrees about x
TRANSLATION = [1.0, 2.0, 3.0]
CORNERS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]


def _report() -> dict:
    return {"mode": "scale", "tags": [
        {"id": 5, "status": "ok", "images": 2, "center_xyz": [1.0, 2.0, 3.0], "corners_xyz": [list(c) for c in CORNERS]},
        {"id": 6, "status": "ok", "images": 2, "center_xyz": None, "corners_xyz": None},
    ]}


def _project(tmp_path: Path, report: dict | None = None) -> Path:
    (tmp_path / "reports").mkdir(parents=True)
    (tmp_path / "control_points").mkdir()
    (tmp_path / "reports" / "apriltag_report.json").write_text(json.dumps(report or _tag_report()), encoding="utf-8")
    return tmp_path


def _tag_report() -> dict:
    return {"mode": "scale", "tags": [
        {"id": 5, "status": "ok", "images": 2, "center_xyz": [1.0, 2.0, 3.0], "corners_xyz": CORNERS},
        {"id": 9, "status": "ok", "images": 1, "center_xyz": [4.0, 5.0, 6.0], "corners_xyz": CORNERS},
    ]}


def _obs(image: str, tag_id: int, u: float, v: float) -> TagObservation:
    corners = ((u - 5, v - 5), (u + 5, v - 5), (u + 5, v + 5), (u - 5, v + 5))
    return TagObservation(image=image, tag_id=tag_id, corners=corners, center=(u, v))


# 1
def test_transform_tag_report_moves_center_and_corners_and_is_idempotent() -> None:
    report = transform_tag_report(_report(), ROTATION, TRANSLATION)

    assert report["frame"] == "gravity_aligned"
    assert report["gravity_transform"] == {"rotation": ROTATION, "translation": TRANSLATION}
    first = report["tags"][0]
    assert first["center_xyz"] == pytest.approx([2.0, -1.0, 5.0])
    expected = (np.asarray(CORNERS) @ np.asarray(ROTATION).T + np.asarray(TRANSLATION)).tolist()
    assert np.allclose(first["corners_xyz"], expected)
    assert np.allclose(first["corners_xyz"], [[1, 2, 3], [2, 2, 3], [2, 2, 4], [1, 2, 4]])
    assert report["tags"][1]["center_xyz"] is None and report["tags"][1]["corners_xyz"] is None
    snapshot = json.dumps(report, sort_keys=True)

    again = transform_tag_report(report, ROTATION, [9.0, 9.0, 9.0])

    assert json.dumps(again, sort_keys=True) == snapshot


# 4
def test_apriltag_marker_payload_lists_detection_photos_sorted(tmp_path: Path) -> None:
    project = _project(tmp_path)
    save_tag_detections(project / "control_points" / "apriltag_detections.json", {
        "front/b.jpg": [_obs("front/b.jpg", 5, 30.0, 40.0)],
        "front/a.jpg": [_obs("front/a.jpg", 5, 10.0, 20.0), _obs("front/a.jpg", 77, 1.0, 1.0)],
    })

    markers = {marker["tag_id"]: marker for marker in apriltag_marker_payload(project)}

    photos = markers[5]["photos"]
    assert [photo["image"] for photo in photos] == ["front/a.jpg", "front/b.jpg"]
    assert (photos[0]["u"], photos[0]["v"]) == (10.0, 20.0)
    assert all(photo["kind"] == "auto" and photo["source"] == "apriltag" and photo["enabled"] is True for photo in photos)
    assert photos[0]["corners"] == [[5.0, 15.0], [15.0, 15.0], [15.0, 25.0], [5.0, 25.0]]
    assert markers[5]["observations"] == photos
    assert markers[9]["photos"] == [] and markers[9]["observations"] == []


# 5
@pytest.mark.parametrize("content", [None, "{not json", json.dumps({"schema_version": 99, "images": {}}),
                                     json.dumps({"schema_version": 1, "images": {"a.jpg": [{"tag_id": 5}]}})])
def test_missing_or_corrupt_detections_file_gives_empty_photos(tmp_path: Path, content: str | None) -> None:
    project = _project(tmp_path)
    if content is not None:
        (project / "control_points" / "apriltag_detections.json").write_text(content, encoding="utf-8")

    markers = apriltag_marker_payload(project)

    assert [marker["tag_id"] for marker in markers] == [5, 9]
    assert all(marker["photos"] == [] and marker["observations"] == [] for marker in markers)


# 6
def test_tag_project_without_detections_json_yields_review_markers_with_photos(tmp_path: Path) -> None:
    project = _project(tmp_path)
    save_tag_detections(project / "control_points" / "apriltag_detections.json", {"front/a.jpg": [_obs("front/a.jpg", 9, 7.0, 8.0)]})
    assert not (project / "control_points" / "detections.json").exists()

    payload = marker_payload(project)

    assert has_apriltag_review_data(project) is True
    by_name = {marker["name"]: marker for marker in payload["markers"]}
    assert set(by_name) == {"TAG5", "TAG9"}
    assert by_name["TAG9"]["photos"][0]["image"] == "front/a.jpg"
    assert by_name["TAG9"]["photos"][0]["source"] == "apriltag"
    assert by_name["TAG5"]["photos"] == []


def test_has_apriltag_review_data_false_for_empty_project(tmp_path: Path) -> None:
    assert has_apriltag_review_data(tmp_path) is False
