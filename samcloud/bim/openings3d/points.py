"""Point-cloud evidence on the wall lines (one streaming pass over the classified cloud).

Per wall line five count grids (wall / window / door / other / any point inside the wall band) are filled at
the cell size of the detection grid; the same pass collects a voxel-thinned copy of the cloud for the
photograph occlusion buffers and the voxel-thinned mid-height door-class points around the wall lines
(door leaves, :mod:`.leaves`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from samcloud.dxf.cancel import Cancelled
from samcloud.bim.plan_model import BuildingPlan
from samcloud.dxf.las_source import Frame, iter_source_chunks

from .config import Settings
from .walls import WallLine, grid_shape

ProgressFn = Callable[[float, str], None]
CHUNK_SIZE = 1_000_000
_KEY_OFFSET = 1 << 20


@dataclass
class PointEvidence:
    """``grids[line_id]`` = {wall, win, door, other, any} int32 (nv, nu) plus the thinned world cloud.

    ``any`` counts every point of every class in the wall band (ceiling and floor included); ``door_points`` are
    the voxel-thinned door-class points between ``points.leaf_z_m`` near a wall line, (N, 3) = plan x, plan y in
    metres (plan units / units_per_metre) and height above the floor in metres.
    """

    grids: dict[str, dict[str, np.ndarray]]
    cloud: np.ndarray | None = None  # (N, 3) float64, source frame, voxel-thinned
    door_points: np.ndarray | None = None
    point_count: int = 0
    classes_present: dict[str, int] = field(default_factory=dict)


def _voxel_keys(local: np.ndarray, voxel_units: float) -> np.ndarray:
    cells = np.floor(local / voxel_units).astype(np.int64) + _KEY_OFFSET
    return (cells[:, 0] << 42) + (cells[:, 1] << 21) + cells[:, 2]


def collect_point_evidence(
    source: Path,
    lines: list[WallLine],
    plan: BuildingPlan,
    frame: Frame,
    config: Settings,
    *,
    want_cloud: bool,
    progress: ProgressFn | None = None,
    cancel: Callable[[], bool] | None = None,
) -> PointEvidence:
    """Stream ``source`` (classified PLY/LAS) once and fill the per-line grids (and the thinned cloud)."""
    upm = plan.units_per_metre
    cell = float(config.grid.cell_m)
    cfg = config.points
    codes = {"wall": cfg.wall_code, "win": cfg.window_code, "door": cfg.door_code}
    structural = np.asarray([cfg.wall_code, cfg.ceiling_code, cfg.floor_code, cfg.door_code, cfg.window_code], dtype=np.uint8)
    shapes = {line.id: grid_shape(line, plan, cell) for line in lines}
    grids = {line.id: {name: np.zeros(shapes[line.id], dtype=np.int32) for name in ("wall", "win", "door", "other", "any")} for line in lines}
    half_band = {line.id: line.thickness / upm / 2.0 + cfg.band_margin_m for line in lines}
    thinned_xyz: list[np.ndarray] = []
    thinned_keys: list[np.ndarray] = []
    leaf_xyz: list[np.ndarray] = []
    leaf_keys: list[np.ndarray] = []
    leaf_voxel = float(cfg.leaf_voxel_m)
    leaf_z0, leaf_z1 = (float(value) for value in cfg.leaf_z_m)
    leaf_reach = float(cfg.leaf_reach_m) * upm
    total = 0
    present: dict[str, int] = {}
    voxel_units = float(config.vote.cloud_voxel_m) * upm
    vertex_total = max(1, _vertex_hint(source))
    for chunk in iter_source_chunks(source, CHUNK_SIZE, cancel=None, stage="Streaming points for openings"):
        if cancel is not None and cancel():
            raise Cancelled("Opening refinement cancelled")
        xyz = np.column_stack((chunk.x, chunk.y, chunk.z))
        classes = np.asarray(chunk.classification, dtype=np.uint8)
        local = frame.to_local(xyz)
        total += len(xyz)
        for code in np.unique(classes):
            present[str(int(code))] = present.get(str(int(code)), 0) + int((classes == code).sum())
        if want_cloud:
            keys = _voxel_keys(local, voxel_units)
            _, first = np.unique(keys, return_index=True)
            thinned_xyz.append(xyz[first])
            thinned_keys.append(keys[first])
        height_m = (local[:, 2] - plan.floor_z) / upm
        near_door = (classes == cfg.door_code) & (height_m > leaf_z0) & (height_m < leaf_z1)
        if near_door.any():
            door_xy = local[near_door, :2]
            reach = np.zeros(len(door_xy), dtype=bool)
            for line in lines:
                relative = door_xy - line.a
                reach |= (np.abs(relative @ line.n) < leaf_reach) & ((relative @ line.d) > -leaf_reach) & ((relative @ line.d) < line.length + leaf_reach)
            if reach.any():
                chosen = np.column_stack((door_xy[reach] / upm, height_m[near_door][reach]))
                keys = _voxel_keys(chosen, leaf_voxel)
                _, first = np.unique(keys, return_index=True)
                leaf_xyz.append(chosen[first])
                leaf_keys.append(keys[first])
        for line in lines:
            nv, nu = shapes[line.id]
            relative = local[:, :2] - line.a
            along = (relative @ line.d) / upm
            across = np.abs(relative @ line.n) / upm
            mask = (across < half_band[line.id]) & (along > 0) & (along < nu * cell)
            if not mask.any():
                continue
            iu = (along[mask] / cell).astype(np.int64)
            iv = (height_m[mask] / cell).astype(np.int64)
            inside = (iv >= 0) & (iv < nv) & (iu >= 0) & (iu < nu)
            flat = (iv * nu + iu)[inside]
            cls = classes[mask][inside]
            targets = {name: cls == code for name, code in codes.items()}
            targets["other"] = ~np.isin(cls, structural)
            targets["any"] = np.ones(len(cls), dtype=bool)
            for name, selected in targets.items():
                if selected.any():
                    grids[line.id][name] += np.bincount(flat[selected], minlength=nv * nu).reshape(nv, nu).astype(np.int32)
        if progress is not None:
            progress(min(1.0, total / vertex_total), f"Reading points {total / 1e6:.1f} M")
    cloud = None
    if want_cloud and thinned_xyz:
        merged = np.concatenate(thinned_xyz)
        keys = np.concatenate(thinned_keys)
        _, first = np.unique(keys, return_index=True)
        cloud = merged[np.sort(first)]
    door_points = np.zeros((0, 3))
    if leaf_xyz:
        merged = np.concatenate(leaf_xyz)
        _, first = np.unique(np.concatenate(leaf_keys), return_index=True)
        door_points = merged[np.sort(first)]
    return PointEvidence(grids=grids, cloud=cloud, door_points=door_points, point_count=total, classes_present=present)


def _vertex_hint(source: Path) -> int:
    """Vertex count for the progress fraction (PLY header); 0 when unknown."""
    try:
        from samcloud.engine.export_las import read_ply_header

        return int(read_ply_header(source)[1])
    except Exception:  # noqa: BLE001 - progress hint only
        return 0
