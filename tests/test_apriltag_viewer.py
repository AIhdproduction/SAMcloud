"""AprilTag markers in the sparse/dense point-cloud viewers (Paket D)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import laspy
import numpy as np
import pytest

from samcloud.app import application
from samcloud.app.gcp_review import apriltag_marker_payload, marker_payload
from samcloud.app.state import state
from samcloud.control_points.repository import APRILTAG_CSV_NAME, save_detections
from samcloud.engine import pipeline
from samcloud.viewer.potree import write_viewer_las


def _corners(centre, half=0.08):
    x, y, z = centre
    return [[x, y - half, z - half], [x, y + half, z - half], [x, y + half, z + half], [x, y - half, z + half]]


def _write_report(project: Path, tags: list[dict] | None = None, warnings: list[str] | None = None) -> None:
    if tags is None:
        tags = [
            {"id": 5, "images": 6, "size_mm_before": 158.0, "size_mm_after": 160.2, "deviation_pct": 1.4, "status": "ok",
             "center_xyz": [0.0, 1.0, 1.5], "corners_xyz": _corners([0.0, 1.0, 1.5]), "reprojection_rms_px": 0.4, "error_3d_m": 0.012},
            {"id": 6, "images": 4, "size_mm_before": 150.0, "size_mm_after": 165.0, "deviation_pct": 9.7, "status": "outlier",
             "center_xyz": [0.0, 2.0, 1.5], "corners_xyz": _corners([0.0, 2.0, 1.5]), "reprojection_rms_px": 1.2},
            {"id": 7, "images": 1, "status": "insufficient"},
        ]
    (project / "reports").mkdir(parents=True, exist_ok=True)
    (project / "reports" / "apriltag_report.json").write_text(
        json.dumps({"mode": "scale", "scale_factor": 1.01, "warnings": warnings or [], "tags": tags}), encoding="utf-8")


def _observation(source: str) -> dict:
    return {"image": "a.jpg", "u": 1.0, "v": 2.0, "score": 1.0, "source": source, "enabled": True}


def test_apriltag_marker_payload_filters_insufficient_and_flags_outlier(tmp_path: Path) -> None:
    assert apriltag_marker_payload(tmp_path) == []
    _write_report(tmp_path)
    markers = {item["name"]: item for item in apriltag_marker_payload(tmp_path)}
    assert set(markers) == {"TAG5", "TAG6"}
    assert markers["TAG5"]["kind"] == "apriltag" and markers["TAG5"]["tag_id"] == 5
    assert markers["TAG5"]["state"] == markers["TAG5"]["state_color"] == "apriltag"
    assert markers["TAG6"]["state"] == markers["TAG6"]["state_color"] == "needs_review"
    assert len(markers["TAG5"]["corners"]) == 4 and len(markers["TAG6"]["corners"]) == 4
    assert markers["TAG5"]["errors"] == {"total_m": 0.012} and "errors" not in markers["TAG6"]
    assert markers["TAG5"]["tag_size_mm"] == 160.2 and markers["TAG5"]["tag_deviation_pct"] == 1.4
    assert markers["TAG5"]["photos"] == [] and markers["TAG5"]["observations"] == []


def test_marker_payload_merges_tags_without_duplicates(tmp_path: Path) -> None:
    (tmp_path / "control_points").mkdir()
    save_detections(tmp_path / "control_points" / "detections.json", {
        "TAG5": {"role": "pass", "enabled": True, "status": "accepted", "triangulated_model_xyz": [0.0, 1.0, 1.5],
                 "observations": [_observation("apriltag")]},
        "PP1": {"role": "pass", "enabled": True, "status": "accepted", "triangulated_model_xyz": [4.0, 5.0, 6.0],
                "observations": [_observation("auto")]},
    })
    _write_report(tmp_path)
    markers = marker_payload(tmp_path)["markers"]
    names = [item["name"] for item in markers]
    assert sorted(names) == ["PP1", "TAG5", "TAG6"]
    by_name = {item["name"]: item for item in markers}
    assert by_name["TAG5"]["state_color"] == "pass"  # detections.json entry wins, not replaced by the report one
    assert by_name["TAG5"]["observations"] and by_name["TAG5"]["kind"] == "apriltag"
    assert by_name["TAG6"]["state_color"] == "needs_review" and by_name["TAG6"]["corners"]
    assert "kind" not in by_name["PP1"] and by_name["PP1"]["coordinates"] == [4.0, 5.0, 6.0]


@pytest.fixture
def active_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = tmp_path / "tag-project"
    project.mkdir()
    monkeypatch.setattr(state, "active_project_directory", project)
    return project


def test_apriltag_markers_endpoint_returns_markers_and_warnings(active_project: Path, monkeypatch) -> None:
    empty = json.loads(application.apriltag_markers().body)
    assert empty == {"markers": [], "warnings": []}
    _write_report(active_project, warnings=["Nur 2 AprilTag(s) gefunden – keine Kontrolle möglich"])
    response = application.apriltag_markers()
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert sorted(item["name"] for item in payload["markers"]) == ["TAG5", "TAG6"]
    assert payload["warnings"] == ["Nur 2 AprilTag(s) gefunden – keine Kontrolle möglich"]
    monkeypatch.setattr(state, "active_project_directory", None)
    assert json.loads(application.apriltag_markers().body) == {"markers": [], "warnings": []}


def test_dense_page_passes_apriltag_marker_endpoint(tmp_path: Path, monkeypatch) -> None:
    from samcloud.app.pages import dense_cloud

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(dense_cloud, "render_pointcloud_viewer", lambda kind, **kwargs: calls.append((kind, kwargs)))
    monkeypatch.setattr(dense_cloud, "ui", MagicMock())
    monkeypatch.setattr(dense_cloud, "render_page", lambda title, content, **kwargs: content())
    monkeypatch.setattr(state, "active_project_directory", tmp_path)
    monkeypatch.setattr(state, "active_project", lambda: {"settings": {"classification": {"class_set": "outdoor"}}})
    dense_cloud.render()
    assert len(calls) == 1
    kind, kwargs = calls[0]
    assert kind == "dense" and kwargs["marker_endpoint"] == "/api/apriltags/markers"


def test_potree_dense_viewer_has_tag_colour_click_priority_and_label(german_ui: None) -> None:
    from samcloud.app import potree_viewer

    config = potree_viewer._viewer_config("dense", "Dense", [[1, 2, 3]], marker_endpoint="/api/apriltags/markers")
    assert config["show_markers"] is True and config["enable_point_info"] is True
    script = potree_viewer._script(config)
    assert 'apriltag: "#b06cff"' in script
    assert "/control-points?tag=${encodeURIComponent(marker.tag_id)}" in script
    assert "/control-points?marker=${encodeURIComponent(marker.name)}" in script  # non-tag behaviour unchanged
    assert "THREE.LineLoop" in script and "SphereGeometry(radius" in script and 'marker.kind === "apriltag"' in script
    assert "`AprilTag ${marker.tag_id}`" in script and "Gesehen in" in script and "Ausreisser" in script
    click = script[script.index('area.addEventListener("click"'):]
    assert click.index("tagAt(event)") < click.index("pickCamera(event, true)") < click.index("handlePointClick(event)")
    clear = script[script.index("const clearPointInfo"):script.index("const readPickedValue")]
    assert "highlightedTag" in clear and "pointHelper" in clear
    assert 'event.key === "Escape"' in script
    dense_html = potree_viewer._html("sam-potree-dense", show_markers=True, markers_label="AprilTags")
    assert ">AprilTags</button>" in dense_html
    assert ">Passpunkte</button>" in potree_viewer._html("sam-potree-sparse", show_markers=True)


def test_render_potree_dense_toolbar_label_is_apriltags(monkeypatch, german_ui: None) -> None:
    from samcloud.app import potree_viewer

    html_calls: list[str] = []

    class _Element:
        def classes(self, *_args, **_kwargs):
            return self

    fake = MagicMock()
    fake.html.side_effect = lambda content, sanitize=False: (html_calls.append(content), _Element())[1]
    monkeypatch.setattr(potree_viewer, "ui", fake)
    potree_viewer.render_potree_viewer("dense", title="t", class_colors=[[1, 2, 3]], marker_endpoint="/api/apriltags/markers")
    potree_viewer.render_potree_viewer("sparse", title="t", class_colors=[[1, 2, 3]], marker_endpoint="/api/control-points/markers")
    assert ">AprilTags</button>" in html_calls[0] and "Passpunkte" not in html_calls[0]
    assert ">Passpunkte</button>" in html_calls[1]


def test_canvas_viewer_knows_apriltag_colour_and_tag_navigation() -> None:
    from samcloud.app import pointcloud_viewer

    script = pointcloud_viewer._script("sam-cloud-dense", "/api/pointcloud/dense", marker_endpoint="/api/apriltags/markers")
    assert "apriltag: '#b06cff'" in script
    assert "/control-points?tag=" in script and "tag_size_mm" in script


def test_tag_markers_land_in_the_frame_of_the_dense_viewer_points(tmp_path: Path, active_project: Path) -> None:
    """The dense Potree LAS only quantises with a header offset (no shift): endpoint coordinates match its points."""
    rng = np.random.default_rng(31)
    grid = np.stack(np.meshgrid(np.arange(0.0, 3.0, 0.02), np.arange(0.0, 2.5, 0.02), indexing="ij"), axis=-1).reshape(-1, 2)
    # Wall at x = 1234.5 in a large model frame; the tag centre lies on that wall.
    wall = np.column_stack((1234.5 + rng.normal(0, 0.003, len(grid)), 50.0 + grid[:, 0], 7.0 + grid[:, 1]))
    centre = [1234.5, 51.0, 8.5]
    _write_report(active_project, tags=[
        {"id": 1, "images": 5, "size_mm_after": 160.0, "deviation_pct": 0.5, "status": "ok", "center_xyz": centre,
         "corners_xyz": _corners(centre)},
    ])
    las_path = tmp_path / "dense.las"
    write_viewer_las(las_path, wall, np.full((len(wall), 3), 128, dtype=np.uint8))
    viewer_points = np.column_stack((laspy.read(las_path).x, laspy.read(las_path).y, laspy.read(las_path).z))
    marker = json.loads(application.apriltag_markers().body)["markers"][0]
    coordinate = np.asarray(marker["coordinates"])
    assert np.min(np.linalg.norm(viewer_points - coordinate, axis=1)) <= 0.02
    for corner in marker["corners"]:
        assert abs(corner[0] - np.median(viewer_points[:, 0])) <= 0.02


def _tag_project(root: Path, source: str, names=("TAG1", "TAG2", "TAG3"), gps: bool = True) -> tuple[Path, Path]:
    control = root / "control_points"
    control.mkdir(parents=True)
    (control / APRILTAG_CSV_NAME).write_text("id,x,y,z,role\n1,0,0,0,pass\n", encoding="utf-8")
    save_detections(control / "detections.json", {
        name: {"role": "pass", "enabled": True, "status": "accepted", "triangulated_model_xyz": [0.0, 0.0, float(index)],
               "observations": [_observation(source)]}
        for index, name in enumerate(names)
    })
    gps_ref = root / "gps_ref.txt"
    if gps:
        gps_ref.write_text("a.jpg 47.0 8.0 400.0\n", encoding="utf-8")
    return control, gps_ref


def test_is_tag_project_requires_apriltag_observations(tmp_path: Path) -> None:
    control, gps_ref = _tag_project(tmp_path / "auto", "auto")
    assert pipeline._is_tag_project(control, gps_ref) is False
    control, gps_ref = _tag_project(tmp_path / "tag", "apriltag")
    assert pipeline._is_tag_project(control, gps_ref) is True
    mixed_control, mixed_ref = _tag_project(tmp_path / "mixed", "apriltag")
    detections = json.loads((mixed_control / "detections.json").read_text(encoding="utf-8"))
    first = next(iter(detections.get("detections", detections)))
    target = detections.get("detections", detections)
    target[first]["observations"].append(_observation("manual"))
    (mixed_control / "detections.json").write_text(json.dumps(detections), encoding="utf-8")
    assert pipeline._is_tag_project(mixed_control, mixed_ref) is False
    # Unchanged: surveyed tags without any GPS reference stay a tag project.
    control, no_gps = _tag_project(tmp_path / "nogps", "auto", gps=False)
    assert pipeline._is_tag_project(control, no_gps) is True
