"""Unit coverage for conservative automatic control-point helpers."""

from pathlib import Path

import numpy as np
import pytest

from samcloud.control_points import auto_detect
from samcloud.control_points.auto_detect import assign_detection, robust_prediction_correction, search_radius_px
from samcloud.control_points.repository import load_detections, save_detections, validate_detections
from samcloud.control_points.targets import TargetDetection


def test_undistorted_coordinates_round_trip(tmp_path: Path) -> None:
    import cv2
    import pycolmap

    distorted_dir, undistorted_dir = tmp_path / "distorted", tmp_path / "undistorted"
    distorted_dir.mkdir(); undistorted_dir.mkdir()
    distorted = pycolmap.Reconstruction(); undistorted = pycolmap.Reconstruction()
    source_camera = pycolmap.Camera(camera_id=1, model="OPENCV", width=1600, height=1200, params=np.array([1000., 980., 800., 600., -0.15, 0.05, 0., 0.]))
    target_camera = pycolmap.Camera(camera_id=1, model="PINHOLE", width=1600, height=1200, params=np.array([1000., 980., 800., 600.]))
    for reconstruction, camera in ((distorted, source_camera), (undistorted, target_camera)):
        reconstruction.add_camera_with_trivial_rig(camera)
        reconstruction.add_image_with_trivial_frame(pycolmap.Image(name="image.jpg", camera_id=1, image_id=1), pycolmap.Rigid3d())
    distorted.write(distorted_dir); undistorted.write(undistorted_dir)
    samples = np.array([[800., 600.], [30., 40.], [1570., 1160.]], dtype=np.float32)
    detections = {"P1": {"observations": [{"image": "image.jpg", "u": float(u), "v": float(v)} for u, v in samples]}}
    auto_detect.populate_undistorted_coordinates(pycolmap.Reconstruction(distorted_dir), pycolmap.Reconstruction(undistorted_dir), detections)
    expected = cv2.undistortPoints(samples.reshape(-1, 1, 2), np.array([[1000., 0., 800.], [0., 980., 600.], [0., 0., 1.]]), np.array([-0.15, 0.05, 0., 0.]), P=np.array([[1000., 0., 800.], [0., 980., 600.], [0., 0., 1.]]))[:, 0, :]
    actual = np.array([[item["u_undist"], item["v_undist"]] for item in detections["P1"]["observations"]])
    assert np.allclose(actual, expected, atol=0.01)


def test_stage_one_correction_recovers_translation() -> None:
    source = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 1.]])
    scale, rotation, translation = robust_prediction_correction(source, source + [2.5, -1.0, 1.0])
    assert np.allclose(scale * (rotation @ source.T).T + translation, source + [2.5, -1.0, 1.0], atol=1e-8)


def test_overlapping_windows_drop_ambiguous_detection() -> None:
    detection = TargetDetection(101., 100., "cross", 0.9, 10., {})
    assert not assign_detection(detection, {"first": (100., 100., 25.), "second": (102., 100., 25.)}, "first")


class _Pose:
    def __mul__(self, value): return np.asarray(value, dtype=float)


class _Camera:
    width, height = 800, 600
    def img_from_cam(self, point):
        point = np.asarray(point, dtype=float)
        return np.array([400 + 100 * point[0] / point[2], 300 + 100 * point[1] / point[2]]) if point[2] > 0 else None


class _Image:
    def cam_from_world(self): return _Pose()


def test_prediction_matches_manual_distorted_projection() -> None:
    projected = auto_detect.project_point(_Image(), _Camera(), np.array([1., -0.5, 10.]))
    assert projected is not None and np.allclose(projected, (410., 295.), atol=.01)


def test_search_radius_grows_and_is_clamped() -> None:
    image, camera, point = _Image(), _Camera(), np.array([0., 0., 10.])
    assert search_radius_px(image, camera, point, .01, .01, 40, 800) == 40
    assert search_radius_px(image, camera, point, 100., 100., 40, 800) == 800


def test_manual_observations_are_never_overwritten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = TargetDetection(20., 20., "cross", .9, 8., {})
    monkeypatch.setattr(auto_detect, "detect_target", lambda *_args, **_kwargs: target)
    image = np.zeros((50, 50), dtype=np.uint8); path = tmp_path / "one.png"
    import cv2; cv2.imwrite(str(path), image)
    prediction = {"P1": [auto_detect.PredictedObservation("one.png", 20., 20., 10.)]}
    existing = {"P1": {"observations": [{"image": "one.png", "u": 1., "v": 2., "source": "manual"}]}}
    result, _ = auto_detect.detect_stage({"one.png": path}, prediction, radius_override=None, target_type="cross", settings={}, existing=existing)
    assert result["P1"] == []


def test_detections_json_round_trip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "detections.json"
    value = {"P1": {"role": "pass", "status": "accepted", "triangulated_model_xyz": [1, 2, 3], "observations": [{"image": "one.jpg", "u": 1, "v": 2}]}}
    save_detections(path, value)
    assert load_detections(path)["P1"]["observations"][0]["u"] == 1.
    with pytest.raises(ValueError):
        validate_detections({"P1": {"role": "invalid", "observations": []}})


def test_automatic_detection_finds_rendered_cross_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exercise the two-stage plumbing with deterministic sub-pixel detections."""
    class Reconstruction:
        def reg_image_ids(self): return [1, 2]
        def image(self, image_id): return type("I", (), {"name": f"{image_id}.png", "camera_id": 1, "cam_from_world": lambda self: _Pose()})()
        def camera(self, _): return _Camera()
        def find_image_with_name(self, name): return self.image(int(name.split(".")[0]))
    monkeypatch.setattr(auto_detect, "triangulate_observations", lambda *_args, **_kwargs: (np.array([0., 0., 10.]), [0.1, 0.1]))
    monkeypatch.setattr(auto_detect, "detect_target", lambda _img, uv, *_args, **_kwargs: TargetDetection(uv[0] + .1, uv[1] - .1, "cross", .9, 8., {}))
    paths = {}
    import cv2
    for name in ("1.png", "2.png"):
        path = tmp_path / name; cv2.imwrite(str(path), np.zeros((600, 800), np.uint8)); paths[name] = path
    detections, _ = auto_detect.automatic_detection(Reconstruction(), {"P1": np.array([0., 0., 10.])}, {"P1": "pass"}, paths, {"detection": {"target_type": "cross"}}, .01, .01)
    assert detections["P1"]["status"] == "accepted"


def test_triangulation_least_squares_beats_best_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pycolmap
    from synthetic_scene import make_scene

    scene = make_scene(tmp_path)
    reconstruction = pycolmap.Reconstruction(scene.model_dir)
    gcp = scene.references["P1"]

    clean_observations = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        camera = reconstruction.camera(image.camera_id)
        uv = auto_detect.project_point(image, camera, gcp, margin_px=0.0)
        if uv is not None:
            clean_observations.append({
                "image": image.name,
                "u": float(uv[0]),
                "v": float(uv[1]),
                "enabled": True,
            })
    best, _ = auto_detect.triangulate_observations(reconstruction, clean_observations)
    assert np.linalg.norm(best - gcp) < 1e-6

    rng = np.random.default_rng(11)
    observations = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        camera = reconstruction.camera(image.camera_id)
        uv = auto_detect.project_point(image, camera, gcp, margin_px=0.0)
        if uv is not None:
            noisy = np.asarray(uv) + rng.normal(0.0, 0.8, 2)
            observations.append({
                "image": image.name,
                "u": float(noisy[0]),
                "v": float(noisy[1]),
                "enabled": True,
            })
    refined, errors = auto_detect.triangulate_observations(reconstruction, observations)
    monkeypatch.setattr(auto_detect, "_refine_least_squares", lambda xyz, views, iterations=10: xyz)
    pair, _ = auto_detect.triangulate_observations(reconstruction, observations)
    rms_error = float(np.sqrt(np.mean(np.square(errors))))
    assert np.linalg.norm(refined - gcp) < np.linalg.norm(pair - gcp)
    assert np.linalg.norm(refined - gcp) < 0.1
    assert rms_error < 1.0

