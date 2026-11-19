"""Choose, per wall face, the photographs that look at it (greedy coverage with angle/distance weights)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from samcloud.dxf.cancel import Cancelled
from samcloud.bim.plan_model import BuildingPlan
from samcloud.dxf.las_source import Frame
from samcloud.ortho.cameras import OrthoCamera, project_points

from .config import Settings
from .walls import WallFace, face_grid_points

ProgressFn = Callable[[float, str], None]


def camera_key(camera: OrthoCamera) -> str:
    """File-system safe identifier of a camera/image."""
    return camera.name.replace("/", "__").replace(".jpg", "")


def usable_cameras(cameras: list[OrthoCamera], config: Settings) -> list[OrthoCamera]:
    """Cube-map side faces whose image file exists."""
    prefixes = tuple(config.selection.camera_prefixes)
    return [camera for camera in cameras if camera.name.split("/")[0] in prefixes and camera.image_path is not None and camera.image_path.is_file()]


def select_images(
    faces: list[WallFace],
    cameras: list[OrthoCamera],
    plan: BuildingPlan,
    frame: Frame,
    config: Settings,
    *,
    device: str = "cpu",
    cancel: Callable[[], bool] | None = None,
) -> dict[tuple[str, int], list[tuple[str, float]]]:
    """Greedy image selection per face: ``{(line_id, side): [(camera_key, gain), ...]}``."""
    cfg = config.selection
    upm = plan.units_per_metre
    by_key = {camera_key(camera): camera for camera in cameras}
    keys = sorted(by_key)  # fixed order: ties between equal gains must not depend on the camera file order
    centres_local = frame.to_local(np.asarray([by_key[key].centre for key in keys], float)) if keys else np.zeros((0, 3))
    max_cos = math.cos(math.radians(cfg.max_angle_deg))
    selection: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for face in faces:
        if cancel is not None and cancel():
            raise Cancelled("Opening refinement cancelled")
        line = face.line
        _, _, points = face_grid_points(face, plan, cfg.grid_step_m, row_step=2)
        flat = points.reshape(-1, 3)
        if len(flat) == 0:
            selection[face.key] = []
            continue
        world = torch.from_numpy(frame.to_original(flat)).double().to(device)
        centre_xy = line.a + line.d * line.length / 2.0 + face.n * line.thickness / 2.0
        normal3 = np.array([face.n[0], face.n[1], 0.0])
        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        for index, key in enumerate(keys):
            local = centres_local[index]
            if float((local[:2] - centre_xy) @ face.n) < cfg.min_front_offset_m * upm:
                continue
            _, _, _, valid_t = project_points(by_key[key], world)
            valid = valid_t.cpu().numpy()
            if valid.sum() < cfg.min_valid_points:
                continue
            ray = flat[valid] - local
            distance = np.linalg.norm(ray, axis=1)
            cosine = -(ray @ normal3) / distance
            ok = (cosine > max_cos) & (distance > cfg.min_distance_m * upm) & (distance < cfg.max_distance_m * upm)
            picked = np.flatnonzero(valid)[ok]
            if len(picked) < cfg.min_valid_points:
                continue
            candidates.append((key, picked, cosine[ok] / (distance[ok] / upm)))
        cover = np.zeros(len(flat))
        chosen: list[tuple[str, float]] = []
        used: set[str] = set()
        budget = int(min(cfg.max_images_per_face, max(cfg.min_images_per_face, cfg.images_per_metre * line.length_m)))
        for _ in range(budget):
            best, best_gain = None, 0.0
            for candidate_index, (key, picked, weight) in enumerate(candidates):
                if key in used:
                    continue
                gain = float((weight * np.maximum(cfg.coverage_k - cover[picked], 0) / cfg.coverage_k).sum())
                if gain > best_gain:
                    best, best_gain = candidate_index, gain
            if best is None or best_gain < cfg.min_gain:
                break
            key, picked, _ = candidates[best]
            cover[picked] += 1
            used.add(key)
            chosen.append((key, best_gain))
        selection[face.key] = chosen
    return selection
