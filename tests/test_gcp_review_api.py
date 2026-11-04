import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from samcloud.app import application, processing
from samcloud.app.gcp_review import _image_roots, displayed_pixel_to_original, marker_payload
from samcloud.app.state import state
from samcloud.control_points.adjustment import add_pass_points_to_reconstruction
from samcloud.control_points.repository import load_detections, save_detections
from samcloud.viewer.preview import ensure_preview


def _response_json(response):
    return json.loads(response.body)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "review-project"
    (project / "control_points").mkdir(parents=True)
    (project / "reports").mkdir()
    images = project / "inputs" / "images"
    images.mkdir(parents=True)
    pixels = np.zeros((40, 40, 3), dtype=np.uint8)
    pixels[12, 10] = (0, 0, 255)
    pixels[15, 20] = (0, 255, 0)
    assert cv2.imwrite(str(images / "camera.jpg"), pixels)
    assert cv2.imwrite(str(images / "other.jpg"), pixels)
    detections = {
        "P1": {
            "role": "pass", "enabled": True, "status": "accepted", "triangulated_model_xyz": [1.0, 2.0, 3.0],
            "observations": [
                {"image": "camera.jpg", "u": 10.5, "v": 12.5, "u_undist": 10.5, "v_undist": 12.5, "score": 0.91, "source": "auto", "enabled": True, "reprojection_px": 0.22},
                {"image": "other.jpg", "u": 20.5, "v": 15.5, "u_undist": 20.5, "v_undist": 15.5, "score": 0.88, "source": "auto", "enabled": True, "reprojection_px": 0.41},
            ],
        },
        "CP1": {
            "role": "control", "enabled": True, "status": "needs_review", "triangulated_model_xyz": [4.0, 5.0, 6.0],
            "observations": [{"image": "camera.jpg", "u": 10.5, "v": 12.5, "source": "auto", "enabled": True, "reprojection_px": 1.5}],
        },
    }
    save_detections(project / "control_points" / "detections.json", detections)
    (project / "control_points" / "control_points.csv").write_text("name,x,y,z,role\nP1,1,2,3,pass\nCP1,4,5,6,control\n", encoding="utf-8")
    report = {
        "gate": {"passed": False, "reasons": ["P1 needs review"]},
        "points": [
            {"name": "P1", "role": "pass", "enabled": True, "projections": 2, "error_x_m": 0.01, "error_y_m": -0.02, "error_z_m": 0.03, "error_3d_m": 0.04, "reprojection_rms_px": 0.31},
            {"name": "CP1", "role": "control", "enabled": True, "projections": 1, "error_x_m": 0.3, "error_y_m": 0.2, "error_z_m": 0.1, "error_3d_m": 0.37, "reprojection_rms_px": 1.5},
        ],
    }
    (project / "reports" / "control_points_report.json").write_text(json.dumps(report), encoding="utf-8")
    return project


def test_image_roots_falls_back_when_undistorted_images_are_missing(tmp_path: Path) -> None:
    project = _project(tmp_path)
    roots = _image_roots(project, "undistorted")
    assert roots == [(project / "inputs" / "images").resolve(), (project / "inputs").resolve()]


@pytest.fixture
def active_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = _project(tmp_path)
    monkeypatch.setattr(state, "active_project_directory", project)
    return project


def test_markers_endpoint_returns_review_fields_in_final_preview_frame(active_project: Path) -> None:
    response = application.control_point_markers()

    assert response.status_code == 200
    payload = _response_json(response)
    markers = {item["name"]: item for item in payload["markers"]}
    assert set(markers) == {"P1", "CP1"}
    assert markers["P1"]["role"] == "pass"
    assert markers["P1"]["enabled"] is True
    assert markers["P1"]["coordinates"] == [1.0, 2.0, 3.0]
    assert markers["P1"]["state_color"] == "pass"
    assert markers["CP1"]["state_color"] == "needs_review"
    assert markers["CP1"]["errors"]["total_m"] == pytest.approx(0.37)


def test_crop_endpoint_centres_subpixel_crop_clamps_and_rejects_unsafe_paths(active_project: Path) -> None:
    response = application.control_point_crop("camera.jpg", "10.5", "12.5", "1", "original")

    assert response.status_code == 200
    assert response.headers["X-SAMcloud-Crop-Size"] == "16"
    crop = cv2.imdecode(np.frombuffer(response.body, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[:2] == (32, 32)
    assert int(crop[16, 16, 2]) == int(crop[:, :, 2].max()) and int(crop[16, 16, 2]) > 0
    clamped = application.control_point_crop("camera.jpg", "10.5", "12.5", "9999", "original")
    assert clamped.headers["X-SAMcloud-Crop-Size"] == "512"
    assert cv2.imdecode(np.frombuffer(clamped.body, dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2] == (1024, 1024)
    assert application.control_point_crop("../camera.jpg", "1", "1", "96", "original").status_code == 400
    assert application.control_point_crop("missing.jpg", "1", "1", "96", "original").status_code == 404


def test_manual_move_remove_toggle_and_invalid_input_preserve_other_observations(active_project: Path) -> None:
    moved = application.move_control_point_observation({"name": "P1", "image": "camera.jpg", "u": 11.25, "v": 13.75})

    assert moved.status_code == 200
    after_move = load_detections(active_project / "control_points" / "detections.json")["P1"]["observations"]
    camera = next(item for item in after_move if item["image"] == "camera.jpg")
    assert camera["source"] == "manual" and camera["u"] == pytest.approx(11.25) and camera["v"] == pytest.approx(13.75)
    assert camera["u_undist"] == pytest.approx(11.25) and camera["v_undist"] == pytest.approx(13.75)
    assert next(item for item in after_move if item["image"] == "other.jpg")["source"] == "auto"
    toggled = application.toggle_control_point_observation({"name": "P1", "image": "other.jpg", "enabled": False})
    assert _response_json(toggled)["result"] is False
    removed = application.remove_control_point_observation({"name": "P1", "image": "camera.jpg"})
    assert removed.status_code == 200
    remaining = load_detections(active_project / "control_points" / "detections.json")["P1"]["observations"]
    assert len(remaining) == 1 and remaining[0]["image"] == "other.jpg" and remaining[0]["enabled"] is False
    assert application.move_control_point_observation({"name": "P1", "image": "other.jpg", "u": "nan", "v": 4}).status_code == 400


def test_role_and_enabled_toggles_persist_and_controls_are_excluded_from_adjustment(active_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert application.change_control_point_role({"name": "P1", "role": "control"}).status_code == 200
    assert application.change_control_point_role({"name": "CP1", "role": "pass"}).status_code == 200
    assert application.toggle_control_point_marker({"name": "P1", "enabled": False}).status_code == 200
    detections = load_detections(active_project / "control_points" / "detections.json")
    assert detections["P1"]["role"] == "control" and detections["P1"]["enabled"] is False
    assert "P1,1,2,3,control" in (active_project / "control_points" / "control_points.csv").read_text(encoding="utf-8")
    detections["CP1"]["observations"].append({"image": "other.jpg", "u": 20.5, "v": 15.5, "source": "auto", "enabled": True, "reprojection_px": 0.2})

    captured = []
    class Track:
        def __init__(self): self.elements = []
        def add_element(self, image_id, point2d_idx):
            captured.append((image_id, point2d_idx))
            self.elements.append((image_id, point2d_idx))
        def length(self): return len(self.elements)
    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace(Track=Track, Point2D=lambda value: value))
    class Image:
        def __init__(self, image_id, name): self.image_id, self.name, self.points2D = image_id, name, []
        def num_points2D(self): return len(self.points2D)
    class Reconstruction:
        def __init__(self): self.images = {1: Image(1, "camera.jpg"), 2: Image(2, "other.jpg")}
        def reg_image_ids(self): return [1, 2]
        def image(self, image_id): return self.images[image_id]
        def add_point3D(self, _xyz, track): return 99
    added = add_pass_points_to_reconstruction(Reconstruction(), detections, {"P1": np.zeros(3), "CP1": np.ones(3)})
    assert added == [99]
    assert captured == [(1, 0), (2, 0)]


def test_optimize_reuses_pristine_model_updates_report_and_refuses_parallel_pipeline(active_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sparse = active_project / "products" / "sparse"
    (sparse / "pre_gcp").mkdir(parents=True)
    (sparse / "pre_gcp_txt").mkdir()
    calls = []
    def fake_adjust(model, model_txt, image_dir, control_dir, crs, gps_ref, settings, report_path, *, rtk):
        calls.append((model, model_txt, control_dir, settings["crs"]))
        report_path.write_text(json.dumps({"gate": {"passed": True, "reasons": []}, "points": []}), encoding="utf-8")
        return model
    import samcloud.engine.pipeline as pipeline
    monkeypatch.setattr(pipeline, "_automatic_control_points", fake_adjust)
    project_data = {"paths": {"image_directory": ""}, "acquisition": {"sources": {"image_directory": ""}, "georeferencing": {"rtk_enabled": False}}, "settings": {"control_points": {"crs": "LOCAL"}}}
    assert processing.optimize_control_points(active_project, project_data) is True
    assert calls == [(sparse / "pre_gcp", sparse / "pre_gcp_txt", active_project / "control_points", "LOCAL")]
    assert json.loads((active_project / "reports" / "control_points_report.json").read_text(encoding="utf-8"))["gate"]["passed"] is True

    manager = object.__new__(processing.ProcessingManager)
    manager._jobs = {active_project.resolve(): SimpleNamespace(poll=lambda: None)}
    manager._review_jobs = set()
    manager._lock = processing.Lock()
    manager._state = SimpleNamespace(projects=SimpleNamespace(load=lambda _path: project_data, save=lambda *_args: None))
    with pytest.raises(ValueError, match="already running"):
        manager.optimize_control_points(active_project)


def test_click_to_pixel_mapping_preserves_pixel_centres_at_all_zoom_levels() -> None:
    for zoom in (1, 4, 16):
        display_x, display_y = 37.0 + (12.5 * zoom), 19.0 + (8.5 * zoom)
        u, v = displayed_pixel_to_original(display_x, display_y, origin_x=37.0, origin_y=19.0, zoom=zoom)
        assert (u, v) == pytest.approx((12.5, 8.5))


def test_sparse_preview_prefers_final_adjusted_model_and_invalidates_changed_source(tmp_path: Path) -> None:
    project = tmp_path / "preview-project"
    original = project / "products" / "sparse" / "0" / "points3D.bin"
    adjusted = project / "products" / "sparse" / "gcp_adjusted" / "points3D.bin"
    _write_sparse_points(original, [1.0])
    _write_sparse_points(adjusted, [9.0])

    preview, count = ensure_preview(project, "sparse")
    assert count == 1
    assert struct.unpack("<f", preview.read_bytes()[16:20])[0] == pytest.approx(9.0)
    _write_sparse_points(adjusted, [11.0, 12.0])
    refreshed, refreshed_count = ensure_preview(project, "sparse")
    assert refreshed == preview and refreshed_count == 2
    assert struct.unpack("<f", refreshed.read_bytes()[16:20])[0] == pytest.approx(11.0)


def _write_sparse_points(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [struct.pack("<QdddBBBdQ", index, value, 0.0, 0.0, 100, 110, 120, 0.1, 0) for index, value in enumerate(values)]
    path.write_bytes(struct.pack("<Q", len(records)) + b"".join(records))


def test_payload_merges_observations_and_predictions(active_project: Path) -> None:
    detections_path = active_project / "control_points" / "detections.json"
    detections = load_detections(detections_path)
    detections["P1"]["observations"][0]["source"] = "manual"
    detections["P1"]["observations"][0]["reprojected_u"] = 10.45
    detections["P1"]["observations"][0]["reprojected_v"] = 12.55
    detections["P1"]["predictions"] = [
        {"image": "camera.jpg", "u": 10.45, "v": 12.55},
        {"image": "third.jpg", "u": 412.37, "v": 189.64},
    ]
    save_detections(detections_path, detections)

    response = application.control_point_markers()
    assert response.status_code == 200
    payload = _response_json(response)
    p1 = next(item for item in payload["markers"] if item["name"] == "P1")
    photos = p1["photos"]

    assert len(photos) == 3
    assert [p["image"] for p in photos] == ["camera.jpg", "other.jpg", "third.jpg"]
    assert [p["kind"] for p in photos] == ["manual", "auto", "predicted"]

    camera_photo = photos[0]
    assert camera_photo["enabled"] is True
    assert camera_photo["u"] == pytest.approx(10.5)
    assert camera_photo["v"] == pytest.approx(12.5)
    assert camera_photo["reprojected_u"] == pytest.approx(10.45)
    assert camera_photo["reprojected_v"] == pytest.approx(12.55)
    assert camera_photo["reprojection_px"] == pytest.approx(0.22)
    assert camera_photo["score"] == pytest.approx(0.91)

    predicted_photo = photos[2]
    assert predicted_photo["enabled"] is False
    assert predicted_photo["u"] == pytest.approx(412.37)
    assert predicted_photo["v"] == pytest.approx(189.64)
    assert predicted_photo["reprojected_u"] == pytest.approx(412.37)
    assert predicted_photo["reprojected_v"] == pytest.approx(189.64)
    assert predicted_photo["reprojection_px"] is None
    assert predicted_photo["score"] is None


def test_crop_headers_report_exact_unclamped_origin(active_project: Path) -> None:
    source_image = cv2.imread(str(active_project / "inputs" / "images" / "camera.jpg"))
    assert source_image is not None

    s = 16
    border_response = application.control_point_crop("camera.jpg", "3.2", "2.7", str(s), "original")
    assert border_response.status_code == 200
    expected_border_origin = (math.floor(3.2 - s / 2.0), math.floor(2.7 - s / 2.0))
    border_origin_u = int(border_response.headers["X-Crop-Origin-U"])
    border_origin_v = int(border_response.headers["X-Crop-Origin-V"])
    assert (border_origin_u, border_origin_v) == expected_border_origin
    assert border_origin_u < 0 and border_origin_v < 0
    assert border_response.headers["X-Crop-Size"] == str(s)
    assert border_response.headers["X-SAMcloud-Crop-Size"] == str(s)

    centre_u, centre_v = 20.37, 19.63
    centre_response = application.control_point_crop("camera.jpg", str(centre_u), str(centre_v), str(s), "original")
    assert centre_response.status_code == 200
    expected_centre_origin = (math.floor(centre_u - s / 2.0), math.floor(centre_v - s / 2.0))
    centre_origin_u = int(centre_response.headers["X-Crop-Origin-U"])
    centre_origin_v = int(centre_response.headers["X-Crop-Origin-V"])
    assert (centre_origin_u, centre_origin_v) == expected_centre_origin
    assert centre_response.headers["X-Crop-Size"] == str(s)

    border_crop = cv2.imdecode(np.frombuffer(border_response.body, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert border_crop is not None
    u_px_out, v_px_out = -1, -2
    row_out = int(v_px_out) - border_origin_v
    col_out = int(u_px_out) - border_origin_u
    pixel_outside = border_crop[row_out * 2, col_out * 2]
    assert np.array_equal(pixel_outside, np.zeros(3, dtype=np.uint8))

    centre_crop = cv2.imdecode(np.frombuffer(centre_response.body, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert centre_crop is not None
    u_px_in, v_px_in = 20, 15
    row_in = int(v_px_in) - centre_origin_v
    col_in = int(u_px_in) - centre_origin_u
    pixel_inside = centre_crop[row_in * 2, col_in * 2]
    assert np.array_equal(pixel_inside, source_image[int(v_px_in), int(u_px_in)])


def test_manual_observation_on_predicted_photo_becomes_manual(active_project: Path) -> None:
    detections_path = active_project / "control_points" / "detections.json"
    detections = load_detections(detections_path)
    detections["P1"]["predictions"] = [
        {"image": "camera.jpg", "u": 10.5, "v": 12.5},
        {"image": "predicted_pic.jpg", "u": 412.37, "v": 285.91},
    ]
    save_detections(detections_path, detections)

    initial_response = application.control_point_markers()
    assert initial_response.status_code == 200
    initial_p1 = next(item for item in _response_json(initial_response)["markers"] if item["name"] == "P1")
    initial_predicted = next(p for p in initial_p1["photos"] if p["image"] == "predicted_pic.jpg")
    assert initial_predicted["kind"] == "predicted"

    post_response = application.move_control_point_observation({
        "name": "P1",
        "image": "predicted_pic.jpg",
        "u": 412.37,
        "v": 285.91,
    })
    assert post_response.status_code == 200

    updated_response = application.control_point_markers()
    assert updated_response.status_code == 200
    updated_p1 = next(item for item in _response_json(updated_response)["markers"] if item["name"] == "P1")
    matching_photos = [p for p in updated_p1["photos"] if p["image"] == "predicted_pic.jpg"]
    assert len(matching_photos) == 1

    manual_photo = matching_photos[0]
    assert manual_photo["kind"] == "manual"
    assert manual_photo["enabled"] is True
    assert manual_photo["u"] == pytest.approx(412.37)
    assert manual_photo["v"] == pytest.approx(285.91)
    assert not any(p["image"] == "predicted_pic.jpg" and p["kind"] == "predicted" for p in updated_p1["photos"])


def test_old_detections_without_predictions_still_render(active_project: Path) -> None:
    detections_path = active_project / "control_points" / "detections.json"
    detections = load_detections(detections_path)
    for entry in detections.values():
        entry.pop("predictions", None)
    save_detections(detections_path, detections)

    response = application.control_point_markers()
    assert response.status_code == 200
    payload = _response_json(response)
    markers = {item["name"]: item for item in payload["markers"]}

    assert set(markers.keys()) == {"P1", "CP1"}
    assert markers["P1"]["coordinates"] == [1.0, 2.0, 3.0]
    assert markers["P1"]["state"] == "accepted"
    assert markers["P1"]["errors"]["total_m"] == pytest.approx(0.04)
    assert len(markers["P1"]["photos"]) == 2
    assert [p["kind"] for p in markers["P1"]["photos"]] == ["auto", "auto"]
    assert len(markers["CP1"]["photos"]) == 1
    assert markers["CP1"]["photos"][0]["kind"] == "auto"
    assert markers["CP1"]["photos"][0]["reprojection_px"] == pytest.approx(1.5)


def test_markers_payload_is_strict_json(active_project: Path) -> None:
    report_path = active_project / "reports" / "control_points_report.json"
    report_data = {
        "gate": {"passed": True, "reasons": []},
        "points": [],
        "totals": {
            "pass": {
                "rms_x_m": 0.01,
                "rms_y_m": 0.0,
                "rms_z_m": 0.0,
                "rms_xy_m": 0.01,
                "rms_3d_m": 0.01,
                "mean_reprojection_px": 0.2,
                "count": 1,
            },
            "control": {
                "rms_x_m": float("nan"),
                "rms_y_m": float("nan"),
                "rms_z_m": float("nan"),
                "rms_xy_m": float("nan"),
                "rms_3d_m": float("nan"),
                "mean_reprojection_px": float("nan"),
                "count": 0,
            },
        },
    }
    report_path.write_text(json.dumps(report_data, allow_nan=True), encoding="utf-8")

    direct_payload = marker_payload(active_project)
    assert json.dumps(direct_payload, allow_nan=False) is not None
    assert direct_payload["totals"]["control"]["rms_x_m"] is None
    assert direct_payload["totals"]["control"]["rms_y_m"] is None
    assert direct_payload["totals"]["control"]["rms_z_m"] is None
    assert direct_payload["totals"]["control"]["rms_xy_m"] is None
    assert direct_payload["totals"]["control"]["rms_3d_m"] is None
    assert direct_payload["totals"]["control"]["mean_reprojection_px"] is None

    response = application.control_point_markers()
    assert response.status_code == 200
    payload = _response_json(response)
    assert json.dumps(payload, allow_nan=False) is not None
    assert payload["totals"]["control"]["rms_x_m"] is None
    assert payload["totals"]["control"]["rms_y_m"] is None
    assert payload["totals"]["control"]["rms_z_m"] is None
    assert payload["totals"]["control"]["rms_xy_m"] is None
    assert payload["totals"]["control"]["rms_3d_m"] is None
    assert payload["totals"]["control"]["mean_reprojection_px"] is None


