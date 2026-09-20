"""
Minimal reader for COLMAP's TEXT model output (cameras.txt, images.txt).
No dependency on pycolmap. Only reads what the classification step needs:
per-image pose (world -> camera) and per-camera intrinsics (PINHOLE model,
which is what image_undistorter produces).
"""

import numpy as np
from pathlib import Path


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


class Camera:
    __slots__ = ("id", "model", "width", "height", "params")

    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

    def intrinsics(self):
        # Undistorted models from colmap image_undistorter are PINHOLE: fx, fy, cx, cy
        if self.model != "PINHOLE":
            raise ValueError(
                f"Camera {self.id} has model {self.model}, expected PINHOLE. "
                "Make sure image_undistorter ran before this step."
            )
        fx, fy, cx, cy = self.params
        return fx, fy, cx, cy


class Image:
    __slots__ = ("id", "qvec", "tvec", "camera_id", "name")

    def __init__(self, id, qvec, tvec, camera_id, name):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name

    def world_to_cam(self, points_world):
        # points_world: (N, 3)
        R = qvec2rotmat(self.qvec)
        return (R @ points_world.T).T + self.tvec


def read_cameras_txt(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = Camera(cam_id, model, width, height, params)
    return cameras


def read_images_txt(path):
    images = {}
    with open(path, "r") as f:
        lines = [l for l in f if not l.startswith("#")]
    # Each image takes two lines: pose line + points2D line
    for i in range(0, len(lines), 2):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        img_id = int(parts[0])
        qvec = np.array([float(p) for p in parts[1:5]])
        tvec = np.array([float(p) for p in parts[5:8]])
        cam_id = int(parts[8])
        name = parts[9]
        images[img_id] = Image(img_id, qvec, tvec, cam_id, name)
    return images


def load_model(sparse_txt_dir):
    sparse_txt_dir = Path(sparse_txt_dir)
    cameras = read_cameras_txt(sparse_txt_dir / "cameras.txt")
    images = read_images_txt(sparse_txt_dir / "images.txt")
    return cameras, images
