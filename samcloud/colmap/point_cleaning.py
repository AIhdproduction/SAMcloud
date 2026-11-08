"""Remove floating dense points that the depth maps prove to be in free space.

A real surface point blocks the view: a camera looking at it cannot measure
anything behind it. A stray point floating in a room fails that test - many
cameras look straight through it and reliably measure the wall behind.

For every fused point and every dense view the filtered (geometric) depth map
is sampled at the point's pixel:

* ``free``    the view measured a surface clearly behind the point
              (depth > point depth * (1 + free_space_margin));
* ``support`` the view measured the point itself (relative error <= support_tolerance).

A point is removed when ``free >= min_free_views`` and ``free >= free_ratio *
support``. Thin structures that the depth maps miss keep their many supporting
views and stay. The pass is purely geometric and identical for drone, camera
and 360 captures.

Indoors an aligned Depth-Anything-V2 prior (see ``depth_prior``) fills the
pixels COLMAP left empty (white walls): there ``free`` uses the looser
``prior_free_margin`` and ``support`` the ``prior_support_tolerance``.

``clean_classified_cloud`` runs in workflow step 6 on the classified cloud
(rows of classified PLY and SAM score archive are removed together);
``clean_dense_workspace`` cleans ``fused.ply``/``fused.ply.vis`` directly.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Callable

import numpy as np

from samcloud.engine.ply_io import read_vertex_array
from samcloud.colmap.plane_fill import View, load_views, read_colmap_array

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "free_space_margin": 0.05,
    "support_tolerance": 0.01,
    "min_free_views": 3,
    "free_ratio": 1.0,
    "min_depth": 0.05,
    "prior_free_margin": 0.10,
    "prior_support_tolerance": 0.03,
}


def free_space_counts(
    points: np.ndarray, views: list[View], stereo_dir: Path, settings: dict[str, Any],
    *, prior_reader: Callable[[str], np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per point (views seeing through it, views measuring it)."""
    points = np.asarray(points, dtype=np.float64)
    free = np.zeros(len(points), dtype=np.int32)
    support = np.zeros(len(points), dtype=np.int32)
    settings = {**DEFAULTS, **settings}
    margin = float(settings["free_space_margin"])
    tolerance = float(settings["support_tolerance"])
    prior_margin = float(settings["prior_free_margin"])
    prior_tolerance = float(settings["prior_support_tolerance"])
    for view in views:
        path = stereo_dir / "depth_maps" / f"{view.name}.geometric.bin"
        if not path.is_file():
            continue
        depth = read_colmap_array(path)
        height, width = depth.shape[:2]
        cam = points @ view.rotation.T + view.translation
        index = np.flatnonzero(cam[:, 2] > float(settings["min_depth"]))
        z = cam[index, 2]
        sx, sy = width / view.width, height / view.height
        px = np.floor((view.fx * cam[index, 0] / z + view.cx) * sx).astype(np.int64)
        py = np.floor((view.fy * cam[index, 1] / z + view.cy) * sy).astype(np.int64)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        index, z, px, py = index[inside], z[inside], px[inside], py[inside]
        measured = depth[py, px]
        valid = measured > 0
        free[index[valid & (measured > z * (1.0 + margin))]] += 1
        support[index[valid & (np.abs(measured - z) <= tolerance * z)]] += 1
        prior = prior_reader(view.name) if prior_reader is not None else None
        if prior is not None and prior.shape[:2] == (height, width):
            predicted = prior[py, px]
            gap = ~valid & (predicted > 0)
            free[index[gap & (predicted > z * (1.0 + prior_margin))]] += 1
            support[index[gap & (np.abs(predicted - z) <= prior_tolerance * z)]] += 1
    return free, support


def floating_mask(free: np.ndarray, support: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    """True for points the depth maps prove to float in free space."""
    return (free >= int(settings["min_free_views"])) & (free >= float(settings["free_ratio"]) * support)


def _rewrite_vis(vis_path: Path, keep: np.ndarray) -> None:
    data = vis_path.read_bytes()
    count = struct.unpack_from("<Q", data, 0)[0]
    if count != len(keep):
        raise ValueError(f"fused.ply.vis has {count} entries but fused.ply has {len(keep)} points")
    offset = 8
    chunks = [struct.pack("<Q", int(np.count_nonzero(keep)))]
    for index in range(count):
        size = struct.unpack_from("<I", data, offset)[0]
        end = offset + 4 + 4 * size
        if keep[index]:
            chunks.append(data[offset:end])
        offset = end
    vis_path.write_bytes(b"".join(chunks))


def clean_dense_workspace(
    dense_dir: Path, settings: dict[str, Any] | None = None, *, views: list[View] | None = None,
) -> dict[str, Any]:
    """Remove depth-proven floating points from fused.ply/fused.ply.vis in place."""
    from plyfile import PlyData, PlyElement

    dense_dir = Path(dense_dir)
    settings = {**DEFAULTS, **(settings or {})}
    if not bool(settings.get("enabled", True)):
        return {"enabled": False, "removed_points": 0}
    ply_path, vis_path = dense_dir / "fused.ply", dense_dir / "fused.ply.vis"
    stereo_dir = dense_dir / "stereo"
    if not ply_path.is_file() or not vis_path.is_file():
        return {"enabled": True, "removed_points": 0, "reason": "fused.ply or fused.ply.vis missing"}
    if not (stereo_dir / "depth_maps").is_dir():
        return {"enabled": True, "removed_points": 0, "reason": "depth maps missing"}
    # No memory-mapping: the same file is rewritten below (Windows Errno 22).
    vertex = read_vertex_array(ply_path)
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    if views is None:
        views = load_views(dense_dir / "sparse")
    free, support = free_space_counts(points, views, stereo_dir, settings)
    remove = floating_mask(free, support, settings)
    report = {
        "enabled": True,
        "original_points": int(len(points)),
        "removed_points": int(np.count_nonzero(remove)),
        "removed_percent": round(100.0 * float(np.mean(remove)), 3) if len(points) else 0.0,
        "settings": {key: settings[key] for key in DEFAULTS if key != "enabled"},
    }
    if np.any(remove):
        keep = ~remove
        kept = vertex[keep]
        del vertex
        _rewrite_vis(vis_path, keep)
        PlyData([PlyElement.describe(kept, "vertex")], text=False, byte_order="<").write(str(ply_path))
    (dense_dir / "point_cleaning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def clean_classified_cloud(
    dense_dir: Path,
    classified_ply: Path,
    settings: dict[str, Any] | None = None,
    *,
    scores_path: Path | None = None,
    views: list[View] | None = None,
    prior_reader: Callable[[str], np.ndarray | None] | None = None,
) -> dict[str, Any]:
    """Remove depth-proven floating points from a classified cloud (and its scores) in place."""
    from plyfile import PlyData, PlyElement

    dense_dir, classified_ply = Path(dense_dir), Path(classified_ply)
    settings = {**DEFAULTS, **(settings or {})}
    if not bool(settings.get("enabled", True)):
        return {"enabled": False, "removed_points": 0}
    stereo_dir = dense_dir / "stereo"
    if not (stereo_dir / "depth_maps").is_dir():
        return {"enabled": True, "removed_points": 0, "reason": "depth maps missing"}
    vertex = read_vertex_array(classified_ply)
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    if views is None:
        views = load_views(dense_dir / "sparse")
    free, support = free_space_counts(points, views, stereo_dir, settings, prior_reader=prior_reader)
    remove = floating_mask(free, support, settings)
    report = {
        "enabled": True,
        "depth_prior": prior_reader is not None,
        "original_points": int(len(points)),
        "removed_points": int(np.count_nonzero(remove)),
        "removed_percent": round(100.0 * float(np.mean(remove)), 3) if len(points) else 0.0,
    }
    if np.any(remove):
        keep = ~remove
        kept = vertex[keep]
        del vertex
        PlyData([PlyElement.describe(kept, "vertex")], text=False).write(str(classified_ply))
        if scores_path is not None and Path(scores_path).is_file():
            with np.load(scores_path) as archive:
                arrays = {name: archive[name] for name in archive.files}
            if all(len(values) == len(keep) for values in arrays.values()):
                np.savez(scores_path, **{name: values[keep] for name, values in arrays.items()})
                report["scores_filtered"] = True
            else:
                report["scores_filtered"] = False
    (dense_dir / "point_cleaning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
