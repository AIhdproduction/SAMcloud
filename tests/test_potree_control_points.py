"""Control point markers (AprilTags + surveyed Passpunkte) of the dense Potree viewer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from pyproj import Transformer

from samcloud.app import potree_viewer
from samcloud.app.state import state

LAT, LON, H0 = 47.134, 8.755, 450.0
_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
_ECEF_TO_LV95 = Transformer.from_crs("EPSG:4978", "EPSG:2056+5728", always_xy=True)
ENU = np.array([[12.3, -45.6, 3.2], [-80.7, 60.1, -1.4], [101.9, 77.7, 5.5]])
NAMES = ("PP1", "PP2", "CP3")


def _georef() -> dict:
    lat, lon = np.radians(LAT), np.radians(LON)
    east = [-np.sin(lon), np.cos(lon), 0.0]
    north = [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    up = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    rotation = np.array([east, north, up]).T
    origin = np.array(_TO_ECEF.transform(LON, LAT, H0))
    return {"scale": 1.0003, "rotation": rotation.tolist(), "translation": origin.tolist(),
            "pairs": 5, "residual_m": {"mean": 0.0, "p95": 0.0, "max": 0.0}}


def _enu_to_lv95(xyz: np.ndarray) -> np.ndarray:
    g = _georef()
    ecef = (np.asarray(xyz) @ np.asarray(g["rotation"]).T) * g["scale"] + np.asarray(g["translation"])
    return np.column_stack(_ECEF_TO_LV95.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2]))


def _drone_project(root: Path) -> Path:
    project = root / "drone"
    (project / "products").mkdir(parents=True)
    (project / "products" / "georeference.json").write_text(json.dumps(_georef()), encoding="utf-8")
    lv95 = _enu_to_lv95(ENU)
    roles = ("pass", "pass", "control")
    rows = ["name,x,y,z,role"] + [f"{n},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{r}" for n, p, r in zip(NAMES, lv95, roles)]
    target = project / "inputs" / "survey" / "points.csv"
    target.parent.mkdir(parents=True)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return project


def test_survey_points_map_back_to_enu_within_1cm(tmp_path: Path) -> None:
    markers = {m["name"]: m for m in potree_viewer.control_point_markers(_drone_project(tmp_path))}
    assert set(markers) == set(NAMES)
    assert [markers[n]["kind"] for n in NAMES] == ["passpunkt", "passpunkt", "kontrollpunkt"]
    for name, expected in zip(NAMES, ENU):
        assert np.abs(np.asarray(markers[name]["coordinates"]) - expected).max() < 0.01
    assert markers["CP3"]["state_color"] == "kontrollpunkt"


def test_control_points_csv_has_priority_over_inputs_survey(tmp_path: Path) -> None:
    project = _drone_project(tmp_path)
    (project / "control_points").mkdir()
    x, y, z = _enu_to_lv95(ENU[:1])[0]
    (project / "control_points" / "control_points.csv").write_text(
        f"name,x,y,z,role\nONLY,{x:.4f},{y:.4f},{z:.4f},pass\n", encoding="utf-8")
    assert [m["name"] for m in potree_viewer.control_point_markers(project)] == ["ONLY"]


def test_apriltag_project_still_yields_tag_markers(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    center = [0.0, 1.0, 1.5]
    corners = [[0.0, 0.9, 1.4], [0.0, 1.1, 1.4], [0.0, 1.1, 1.6], [0.0, 0.9, 1.6]]
    tag = {"id": 5, "images": 6, "status": "ok", "center_xyz": center, "corners_xyz": corners, "size_mm_after": 160.0}
    (tmp_path / "reports" / "apriltag_report.json").write_text(json.dumps({"tags": [tag]}), encoding="utf-8")
    markers = potree_viewer.control_point_markers(tmp_path)
    assert [m["name"] for m in markers] == ["TAG5"]
    assert markers[0]["kind"] == "apriltag" and markers[0]["coordinates"] == center and len(markers[0]["corners"]) == 4


def test_no_control_points_gives_empty_list_and_disabled_button(tmp_path: Path, german_ui: None) -> None:
    assert potree_viewer.control_point_markers(tmp_path) == []
    assert potree_viewer.control_point_markers(None) == []
    html = potree_viewer._html("x", show_markers=True, markers_label="Passpunkte", markers_disabled=True)
    assert "disabled" in html and "Keine Passpunkte im Projekt" in html
    enabled = potree_viewer._html("x", show_markers=True, markers_label="Passpunkte")
    assert 'data-action="markers" class="is-selected"' in enabled and "Keine Passpunkte" not in enabled


def test_survey_without_georeference_uses_model_coordinates_or_nothing(tmp_path: Path) -> None:
    project = tmp_path / "local"
    (project / "control_points").mkdir(parents=True)
    (project / "control_points" / "control_points.csv").write_text("name,x,y,z,role\nPP1,1,2,3,pass\n", encoding="utf-8")
    assert potree_viewer.control_point_markers(project) == []
    detections = {"PP1": {"role": "pass", "enabled": True, "status": "accepted",
                          "triangulated_model_xyz": [0.5, 0.6, 0.7], "observations": []}}
    (project / "control_points" / "detections.json").write_text(json.dumps(detections), encoding="utf-8")
    markers = potree_viewer.control_point_markers(project)
    assert [m["name"] for m in markers] == ["PP1"] and markers[0]["coordinates"] == [0.5, 0.6, 0.7]


def test_checkpoint_report_adds_tooltips(tmp_path: Path) -> None:
    project = _drone_project(tmp_path)
    (project / "reports").mkdir()
    points = [{"name": "PP1", "status": "ok", "dH": 0.2124, "dE": 0.01, "dN": 0.0, "d3D": 0.213},
              {"name": "PP2", "status": "not_found", "reason": "Target in keinem Bild erkannt"}]
    (project / "reports" / "checkpoints_baseline.json").write_text(json.dumps({"points": points}), encoding="utf-8")
    markers = {m["name"]: m for m in potree_viewer.control_point_markers(project)}
    assert markers["PP1"]["tooltip"] == "PP1 · dH +0.212 m" and markers["PP1"]["state_color"] == "passpunkt"
    assert markers["PP2"]["tooltip"] == "PP2 · nicht gefunden" and markers["PP2"]["state_color"] == "disabled"
    assert markers["CP3"]["tooltip"] == "CP3"


def test_dense_html_uses_passpunkte_label_and_disables_without_points(monkeypatch, german_ui: None, tmp_path: Path) -> None:
    html_calls: list[str] = []

    class _Element:
        def classes(self, *_args, **_kwargs):
            return self

    fake = MagicMock()
    fake.html.side_effect = lambda content, sanitize=False: (html_calls.append(content), _Element())[1]
    monkeypatch.setattr(potree_viewer, "ui", fake)
    monkeypatch.setattr(potree_viewer, "register_control_point_routes", lambda: None)
    monkeypatch.setattr(state, "active_project_directory", _drone_project(tmp_path))
    potree_viewer.render_potree_viewer("dense", title="t", class_colors=[[1, 2, 3]], marker_endpoint="/api/apriltags/markers")
    assert ">Passpunkte</button>" in html_calls[0] and "AprilTags</button>" not in html_calls[0]
    assert "disabled" not in html_calls[0].split('data-action="markers"')[1].split(">")[0]
    config = potree_viewer._viewer_config("dense", "t", [[1, 2, 3]], marker_endpoint=potree_viewer.CONTROL_POINT_MARKERS_URL)
    script = potree_viewer._script(config)
    assert 'passpunkt: "#ff9f1a"' in script and 'kontrollpunkt: "#18d6e6"' in script
    monkeypatch.setattr(state, "active_project_directory", tmp_path / "empty")
    potree_viewer.render_potree_viewer("dense", title="t", class_colors=[[1, 2, 3]], marker_endpoint="/api/apriltags/markers")
    assert "Keine Passpunkte im Projekt" in html_calls[1] and "disabled" in html_calls[1]


def test_fit_action_includes_visible_markers() -> None:
    script = potree_viewer._script(potree_viewer._viewer_config("dense", "t", [[1, 2, 3]]))
    fit = script[script.index('action === "fit"'):][:420]
    assert "getBoundingBox" in fit and "expandByPoint(item.sphere.position)" in fit and "markersVisible" in fit
