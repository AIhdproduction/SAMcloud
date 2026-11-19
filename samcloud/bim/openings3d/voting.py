"""Vote SAM3 window/door/wall/furniture scores of the photographs onto the wall planes.

Every cell of a wall-face grid is projected into each selected photograph.  A sample counts only when the
cell is visible (z-buffer of the classified cloud, so furniture and walls in front occlude it), weighted by
viewing angle and distance and down-weighted where SAM sees furniture.  Besides the class votes the pass
accumulates the observed colour, the occluded/see-through/ray statistics used by the detection rules.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as functional

from samcloud.dxf.cancel import Cancelled
from samcloud.bim.plan_model import BuildingPlan
from samcloud.dxf.las_source import Frame
from samcloud.ortho.cameras import OrthoCamera, project_points
from samcloud.ortho.config import OrthoConfig
from samcloud.ortho.zbuffer import build_zbuffer

from .config import Settings
from .scoring import load_score_map
from .walls import WallFace, face_grid_points

_LOGGER = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]

# accumulator channels
SW, WIN, DOOR, WALL, NIMG, OCC, FUR, BEHIND, RAYS, VIS = range(10)
CHANNELS = ("sw", "win", "door", "wall", "n", "occ", "fur", "behind", "rays", "vis")
_RGB = 10  # three colour channels follow


@dataclass
class LineVotes:
    """Votes of both faces of a wall line, all (nv, nu); ``rgb`` is (3, nv, nu) in [0, 1] (RGB order)."""

    sw: np.ndarray
    win: np.ndarray
    door: np.ndarray
    wall: np.ndarray
    n: np.ndarray
    occ: np.ndarray
    fur: np.ndarray
    behind: np.ndarray
    rays: np.ndarray
    vis: np.ndarray
    rgb: np.ndarray


def _empty_like(nv: int, nu: int) -> LineVotes:
    zeros = lambda: np.zeros((nv, nu))  # noqa: E731
    return LineVotes(zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), np.zeros((3, nv, nu)))


def merge_faces(parts: list[LineVotes]) -> LineVotes:
    """Sum the votes of the two faces of a line (colours are weight-averaged)."""
    merged = _empty_like(*parts[0].sw.shape)
    rgb_sum = np.zeros_like(merged.rgb)
    for part in parts:
        for name in CHANNELS:
            setattr(merged, name, getattr(merged, name) + getattr(part, name))
        rgb_sum += part.rgb * part.sw[None]
    merged.rgb = rgb_sum / np.maximum(merged.sw, 1e-9)[None]
    return merged


def vote_faces(
    selection: dict[tuple[str, int], list[tuple[str, float]]],
    faces: list[WallFace],
    cameras: dict[str, OrthoCamera],
    score_paths: dict[str, Path],
    cloud: np.ndarray,
    plan: BuildingPlan,
    frame: Frame,
    config: Settings,
    *,
    device: str = "cpu",
    progress: ProgressFn | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[tuple[str, int], LineVotes]:
    """Accumulate the photograph votes per face (``{(line_id, side): LineVotes}``)."""
    import cv2

    cfg = config.vote
    cell = float(config.grid.cell_m)
    upm = plan.units_per_metre
    scale = float(frame.scale_factor)
    unit = upm / scale  # world units per metre
    ortho_config = OrthoConfig()
    face_by_key = {face.key: face for face in faces}
    geometry: dict[tuple[str, int], tuple[torch.Tensor, tuple[int, int], torch.Tensor]] = {}
    accumulators: dict[tuple[str, int], torch.Tensor] = {}
    by_image: dict[str, list[tuple[str, int]]] = {}
    for face_key, chosen in selection.items():
        if not chosen:
            continue
        face = face_by_key[face_key]
        _, _, points = face_grid_points(face, plan, cell)
        shape = points.shape[:2]
        world = torch.from_numpy(frame.to_original(points.reshape(-1, 3))).double().to(device)
        normal_world = frame.to_original(np.array([face.n[0], face.n[1], 0.0])) - frame.to_original(np.zeros(3))
        normal_world = torch.from_numpy(normal_world / max(float(np.linalg.norm(normal_world)), 1e-12)).double().to(device)
        geometry[face_key] = (world, shape, normal_world)
        accumulators[face_key] = torch.zeros((_RGB + 3,) + shape, dtype=torch.float64, device=device)
        for key, _ in chosen:
            by_image.setdefault(key, []).append(face_key)
    cloud_t = torch.from_numpy(np.asarray(cloud, dtype=np.float64)).to(device)
    max_cos = math.cos(math.radians(cfg.max_angle_deg))
    started = time.time()
    for index, (key, face_keys) in enumerate(sorted(by_image.items())):
        if cancel is not None and cancel():
            raise Cancelled("Opening refinement cancelled")
        camera = cameras[key]
        scores = torch.from_numpy(load_score_map(score_paths[key]).astype(np.float32) / 255.0).to(device)
        image = cv2.cvtColor(cv2.imread(str(camera.image_path)), cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        depth_buffer, factor = build_zbuffer(camera, cloud_t, ortho_config, indoor=True)
        finite = torch.where(torch.isfinite(depth_buffer), depth_buffer, torch.full_like(depth_buffer, 1e9))
        kernel = int(cfg.zbuffer_min_filter_px)
        depth_buffer = -functional.max_pool2d(-finite[None, None].float(), kernel, 1, kernel // 2)[0, 0]
        zh, zw = depth_buffer.shape
        centre = torch.from_numpy(camera.centre).double().to(device)
        for face_key in face_keys:
            world, shape, normal_world = geometry[face_key]
            px, py, depth, valid = project_points(camera, world)
            ray = world - centre
            distance = ray.norm(dim=1)
            cosine = -(ray @ normal_world) / distance
            distance_local = distance * scale
            ok = valid & (cosine > max_cos) & (distance_local > cfg.min_distance_m * upm) & (distance_local < cfg.max_distance_m * upm)
            gx = (px / camera.width * 2 - 1).float()
            gy = (py / camera.height * 2 - 1).float()
            grid = torch.stack([gx, gy], 1)[None, :, None, :]
            sampled = functional.grid_sample(scores[None], grid, mode="bilinear", align_corners=False)[0, :, :, 0]
            zx = torch.clamp((px * factor).long(), 0, zw - 1)
            zy = torch.clamp((py * factor).long(), 0, zh - 1)
            z_depth = depth_buffer[zy, zx].double()
            visible = ok & (z_depth >= depth - cfg.zbuffer_tolerance_m * unit)
            weight = (cosine.clamp(min=0) / (distance_local / upm)) * visible.double() * (1 - cfg.furniture_downweight * sampled[3].double())
            acc = accumulators[face_key]
            reshape = lambda values: values.reshape(shape)  # noqa: E731
            acc[SW] += reshape(weight)
            acc[WIN] += reshape(weight * sampled[0].double())
            acc[DOOR] += reshape(weight * sampled[1].double())
            acc[WALL] += reshape(weight * sampled[2].double())
            acc[NIMG] += reshape(visible.double())
            acc[OCC] += reshape((ok & ~visible).double())
            acc[FUR] += reshape(visible.double() * sampled[3].double())
            behind = ok & (z_depth > depth + cfg.behind_margin_m * unit) & torch.isfinite(z_depth) & (z_depth < 1e8)
            acc[BEHIND] += reshape(behind.double())
            acc[RAYS] += reshape(ok.double())
            acc[VIS] += reshape(visible.double())
            colour = functional.grid_sample(rgb_t[None], grid, mode="bilinear", align_corners=False)[0, :, :, 0].double()
            acc[_RGB:_RGB + 3] += (weight[None] * colour).reshape((3,) + shape)
        if progress is not None and ((index + 1) % 10 == 0 or index + 1 == len(by_image)):
            progress((index + 1) / len(by_image), f"Voting photographs {index + 1}/{len(by_image)}")
    _LOGGER.info("openings3d: voted %d photographs in %.0f s", len(by_image), time.time() - started)
    result: dict[tuple[str, int], LineVotes] = {}
    for face_key, acc in accumulators.items():
        values = acc.cpu().numpy()
        weight = np.maximum(values[SW], 1e-9)[None]
        result[face_key] = LineVotes(*(values[i] for i in range(_RGB)), rgb=values[_RGB:_RGB + 3] / weight)
    return result
