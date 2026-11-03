"""Deterministic pycolmap scene used by control-point integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap


@dataclass
class SyntheticScene:
    model_dir: Path
    references: dict[str, np.ndarray]
    detections: dict[str, dict[str, object]]
    camera_centres: dict[str, np.ndarray]


def make_scene(root: Path, *, perturbed: bool = False, seed: int = 20261103) -> SyntheticScene:
    """Write a 14-image drone scene with 300 ties and six surveyed targets."""
    rng = np.random.default_rng(seed)
    model_dir = root / ("perturbed_model" if perturbed else "pristine_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction = pycolmap.Reconstruction()
    cameras = {
        1: _camera(1, -0.055 if not perturbed else -0.005),
        2: _camera(2, -0.080 if not perturbed else -0.010),
    }
    for camera in cameras.values():
        reconstruction.add_camera_with_trivial_rig(camera)

    centres = _camera_centres()
    camera_centres: dict[str, np.ndarray] = {}
    for image_id, centre in enumerate(centres, start=1):
        name = f"image_{image_id:02d}.jpg"
        initial_centre = _perturb_position(centre) if perturbed else centre
        image = pycolmap.Image(name=name, camera_id=1 if image_id % 2 else 2, image_id=image_id)
        reconstruction.add_image_with_trivial_frame(image, _pose(initial_centre, image_id > 12))
        camera_centres[name] = initial_centre

    tie_points = np.column_stack((
        rng.uniform(-58.0, 58.0, 300),
        rng.uniform(-38.0, 38.0, 300),
        rng.uniform(-3.0, 3.0, 300),
    ))
    for point in tie_points:
        track = pycolmap.Track()
        initial_point = _perturb_position(point) if perturbed else point
        for image_id in reconstruction.reg_image_ids():
            name = reconstruction.image(image_id).name
            true_pose = _pose(centres[image_id - 1], image_id > 12)
            camera = _camera(int(reconstruction.image(image_id).camera_id), -0.055 if image_id % 2 else -0.080)
            uv = camera.img_from_cam(true_pose * point)
            if uv is None or not _inside(uv):
                continue
            noisy = np.asarray(uv) + rng.normal(0.0, 0.3, 2)
            image = reconstruction.image(image_id)
            image.points2D.append(pycolmap.Point2D(noisy))
            track.add_element(image_id, image.num_points2D() - 1)
        if track.length() >= 3:
            reconstruction.add_point3D(initial_point, track)

    references = {
        "P1": np.array([-47.3, -27.7, -0.8]),
        "P2": np.array([43.6, -24.2, 1.4]),
        "P3": np.array([-35.1, 25.8, -1.7]),
        "P4": np.array([38.4, 29.3, 2.2]),
        "CP1": np.array([-4.7, -1.9, 0.4]),
        "CP2": np.array([17.2, 11.6, -2.1]),
    }
    detections: dict[str, dict[str, object]] = {}
    for name, xyz in references.items():
        observations: list[dict[str, object]] = []
        for image_id in reconstruction.reg_image_ids():
            camera = _camera(int(reconstruction.image(image_id).camera_id), -0.055 if image_id % 2 else -0.080)
            uv = camera.img_from_cam(_pose(centres[image_id - 1], image_id > 12) * xyz)
            if uv is not None and _inside(uv):
                observed = np.asarray(uv) + rng.normal(0.0, 0.2, 2)
                observations.append({"image": reconstruction.image(image_id).name, "u": float(observed[0]), "v": float(observed[1]), "enabled": True, "source": "auto"})
        detections[name] = {
            "role": "pass" if name.startswith("P") else "control",
            "status": "accepted",
            "enabled": True,
            "triangulated_model_xyz": None,
            "observations": observations,
        }
    reconstruction.write(model_dir)
    return SyntheticScene(model_dir, references, detections, camera_centres)


def clone_detections(detections: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Return an independent detection graph without relying on disk serialization."""
    import copy
    return copy.deepcopy(detections)


def _camera(camera_id: int, k1: float) -> pycolmap.Camera:
    return pycolmap.Camera(
        camera_id=camera_id,
        model="OPENCV",
        width=2000,
        height=1500,
        params=np.array([1450.0, 1435.0, 1000.0, 750.0, k1, 0.012, 0.0, 0.0]),
    )


def _camera_centres() -> list[np.ndarray]:
    return [np.array([x, y, 60.0]) for y in (-30.0, 0.0, 30.0) for x in (-45.0, -15.0, 15.0, 45.0)] + [np.array([-55.0, 0.0, 64.0]), np.array([55.0, 0.0, 64.0])]


def _pose(centre: np.ndarray, oblique: bool) -> pycolmap.Rigid3d:
    if not oblique:
        rotation = np.diag([1.0, -1.0, -1.0])
    else:
        target = np.array([0.0, 0.0, 0.0])
        forward = (target - centre) / np.linalg.norm(target - centre)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0])); right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        rotation = np.vstack((right, up, forward))
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -rotation @ centre)


def _perturb_position(xyz: np.ndarray) -> np.ndarray:
    value = np.asarray(xyz, dtype=float)
    dome = 0.15 * ((value[0] / 60.0) ** 2 + (value[1] / 40.0) ** 2)
    return np.array([1.01 * value[0], 1.01 * value[1], 1.01 * value[2] + dome])


def _inside(uv: np.ndarray) -> bool:
    return 10.0 < float(uv[0]) < 1990.0 and 10.0 < float(uv[1]) < 1490.0
