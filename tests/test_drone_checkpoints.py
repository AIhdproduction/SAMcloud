"""Checkpoint evaluation on a synthetic scene: 4 cameras over a plane with rendered bow-tie targets."""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
import pycolmap
import pytest

from samcloud.drone import checkpoints as cp

OFFSET = np.array([2_699_900.0, 1_221_200.0, 900.0])  # fake "LV95" = model + offset
TARGETS = {"A": (0.0, 0.0, 0.0), "B": (3.1, 1.7, 0.0), "C": (-2.9, 1.3, 0.0)}


@dataclass
class _Image:
    name: str
    camera_id: int
    _pose: object

    def cam_from_world(self):
        return self._pose


class _Reconstruction:
    def __init__(self, images, cameras):
        self._images, self._cameras = dict(enumerate(images, 1)), cameras

    def reg_image_ids(self):
        return list(self._images)

    def image(self, image_id):
        return self._images[image_id]

    def camera(self, camera_id):
        return self._cameras[camera_id]

    def find_image_with_name(self, name):
        return next((i for i in self._images.values() if i.name == name), None)


def _pose(centre, look_at):
    forward = np.asarray(look_at, float) - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])  # world -> camera
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -rotation @ np.asarray(centre, float))


def _render(camera, pose, targets, seed):
    rng = np.random.default_rng(seed)
    image = np.full((camera.height, camera.width), 120.0)
    image += rng.normal(0, 3, image.shape)

    def project(points):
        out = []
        for p in points:
            uv = camera.img_from_cam(np.asarray(pose * np.array(p, float)))
            out.append([(uv[0] - 0.5) * 16, (uv[1] - 0.5) * 16])
        return np.round(np.array(out)).astype(np.int32)

    def poly(points, value):
        cv2.fillPoly(image, [project(points)], value, lineType=cv2.LINE_AA, shift=4)

    for x, y, z in targets:
        half = 0.32
        poly([(x - half, y - half, z), (x + half, y - half, z), (x + half, y + half, z), (x - half, y + half, z)], 245)
        poly([(x - 0.25, y - 0.25, z), (x - 0.25, y + 0.25, z), (x, y, z)], 10)
        poly([(x + 0.25, y - 0.25, z), (x + 0.25, y + 0.25, z), (x, y, z)], 10)
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.fixture
def scene(tmp_path):
    camera = pycolmap.Camera.create_from_model_name(1, "OPENCV", 1500, 1600, 1200)
    camera.params = [1500.0, 1500.0, 800.0, 600.0, -0.05, 0.0, 0.0, 0.0]
    cameras = {1: camera}
    images = []
    paths = {}
    centres = [(-3.0, -2.5, 11.0), (3.5, -3.0, 10.5), (3.0, 3.5, 11.5), (-3.5, 3.0, 10.0)]
    for index, centre in enumerate(centres):
        pose = _pose(np.array(centre), [0.3, 0.2, 0.0])
        name = f"cam{index}.png"
        cv2.imwrite(str(tmp_path / name), _render(camera, pose, list(TARGETS.values()), index))
        images.append(_Image(name, 1, pose))
        paths[name] = tmp_path / name
    return _Reconstruction(images, cameras), paths


def _survey(extra=None):
    survey = {name: np.array(xyz) + OFFSET for name, xyz in TARGETS.items()}
    survey.update(extra or {})
    return survey


def _run(scene, survey, out_dir=None):
    reconstruction, paths = scene
    return cp.evaluate_checkpoints_model(
        reconstruction, survey, paths,
        to_model=lambda xyz: xyz - OFFSET, from_model=lambda xyz: xyz + OFFSET, out_dir=out_dir,
    )


def test_triangulation_within_one_cm(scene):
    result = _run(scene, _survey())
    assert [p["status"] for p in result["points"]] == ["ok"] * 3
    for point in result["points"]:
        assert point["d3D"] < 0.01
        assert point["images_detected"] >= 3
        assert point["reproj_rms_px"] < 1.5


def test_target_outside_all_images_is_not_found(scene):
    far = np.array([60.0, 60.0, 0.0]) + OFFSET
    result = _run(scene, _survey({"FAR": far}))
    by_name = {p["name"]: p for p in result["points"]}
    assert by_name["FAR"]["status"] == "not_found"
    assert by_name["FAR"]["reason"]
    assert result["totals"]["n_ok"] == 3 and result["totals"]["n_points"] == 4


def test_totals_and_bias(scene):
    shifts = {"A": (0.03, 0.0, 0.0), "B": (0.0, 0.0, 0.0), "C": (-0.03, 0.0, 0.0)}
    survey = {n: np.array(TARGETS[n]) + OFFSET + np.array(shifts[n]) for n in TARGETS}
    result = _run(scene, survey)
    points = result["points"]
    totals = result["totals"]
    assert totals["n_ok"] == 3
    assert totals["rms_E"] == pytest.approx(np.sqrt(np.mean([p["dE"] ** 2 for p in points])))
    assert totals["bias_E"] == pytest.approx(np.mean([p["dE"] for p in points]))
    assert totals["bias_E"] == pytest.approx(0.0, abs=0.01)
    assert totals["rms_E"] == pytest.approx(0.03 * np.sqrt(2 / 3), abs=0.01)
    assert totals["rms_3D"] >= totals["rms_2D"] >= 0


def test_shifted_survey_height_sign_is_measured_minus_surveyed(scene):
    survey = _survey()
    survey["B"] = survey["B"] + np.array([0.0, 0.0, 0.05])  # survey 5 cm higher than reality
    point = {p["name"]: p for p in _run(scene, survey)["points"]}["B"]
    assert point["dH"] == pytest.approx(-0.05, abs=0.01)
    assert abs(point["dE"]) < 0.01 and abs(point["dN"]) < 0.01


def test_reports_and_crops_written(scene, tmp_path):
    out = tmp_path / "reports"
    result = _run(scene, _survey({"FAR": np.array([60.0, 60.0, 0.0]) + OFFSET}), out_dir=out)
    crops = list((out / "checkpoint_crops").glob("*.png"))
    accepted = [c for c in crops if not c.name.startswith("pred_")]
    assert accepted and all(cv2.imread(str(c)).shape[:2] == (cp.CROP_PX, cp.CROP_PX) for c in accepted)
    for name in TARGETS:
        assert len([c for c in accepted if c.name.startswith(f"{name}_")]) <= cp.MAX_CROPS_PER_POINT
    text = cp.render_markdown(result, "synthetic")
    assert "| A | ok |" in text and "FAR" in text and "Bias" in text
    (out / "checkpoints_t.json").write_text(json.dumps(result), encoding="utf-8")
    assert json.loads((out / "checkpoints_t.json").read_text())["totals"]["n_ok"] == 3


def test_implausible_deviation_excluded_from_totals(scene):
    reconstruction, paths = scene
    survey = _survey()
    survey["B"] = survey["B"] + np.array([0.0, 0.0, 0.05])
    result = cp.evaluate_checkpoints_model(
        reconstruction, survey, paths, to_model=lambda xyz: xyz - OFFSET, from_model=lambda xyz: xyz + OFFSET,
        max_plausible_m=0.02,
    )
    by_name = {p["name"]: p for p in result["points"]}
    assert by_name["B"]["status"] == "implausible" and by_name["B"]["reason"]
    assert result["totals"]["n_ok"] == 2
