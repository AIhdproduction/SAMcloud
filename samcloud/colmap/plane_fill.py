"""Depth-verified hole filling on large planes of a COLMAP dense cloud.

White walls and other weakly textured surfaces often stay almost empty after
COLMAP fusion: the filtered (geometric) depth maps drop them although the raw
photometric PatchMatch estimates frequently do hit the surface. This module
only adds a point when the depth maps confirm it:

1. Large planes are detected in ``fused.ply`` (RANSAC on a voxel subsample).
2. Enclosed holes in each plane's occupancy grid become candidate points.
3. Each candidate is projected into every dense view. It is accepted only if
   * at least ``min_support_views`` raw photometric depth maps measure the
     plane depth there (relative error <= ``max_depth_error``) with a matching
     normal, and
   * no filtered geometric depth map reliably sees a surface *behind* it
     (free-space check, e.g. doors and windows stay open).

Accepted points are appended to ``fused.ply`` and ``fused.ply.vis``. The
visibility list holds the supporting views as COLMAP image indices (rank of
the image id in the dense model), exactly like COLMAP's own fusion output, so
classification treats them like every other dense point.

The procedure works on the undistorted dense workspace and is therefore the
same for drone, camera and 360 captures (cubemap faces are pinhole views).
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from samcloud.engine.ply_io import read_vertex_array

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "max_planes": 24,
    "min_plane_share": 0.01,
    "min_support_views": 3,
    "max_depth_error": 0.02,
    "free_space_margin": 0.05,
    "max_free_space_violations": 0,
    "max_view_angle_deg": 75.0,
    "max_normal_error_deg": 25.0,
    "max_candidates": 4_000_000,
    "ransac_points": 300_000,
    "ransac_iterations": 400,
    "max_vis_views": 16,
    "fill_spacing_m": None,
    "classes": ["floor", "ceiling", "wall"],
    # Rule A ("sam_plane_free_space"): no depth hit required, see verify_by_class.
    "rule": "depth",
    "min_class_ratio": 0.6,
    "verify_cell_m": 0.03,
    "region_cell_m": 0.05,
    "max_gap_m": 1.5,
    "prior_free_margin": 0.10,
    "min_prior_violations": 2,
    "seed": 0,
}


@dataclass
class Plane:
    normal: np.ndarray
    offset: float
    inliers: int


@dataclass
class View:
    """One undistorted pinhole view of the dense workspace."""

    index: int
    name: str
    rotation: np.ndarray
    translation: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.translation


@dataclass
class FillResult:
    xyz: np.ndarray
    rgb: np.ndarray
    normals: np.ndarray
    visibility: list[np.ndarray]
    report: dict[str, Any] = field(default_factory=dict)
    classes: np.ndarray | None = None


def read_colmap_array(path: Path) -> np.ndarray:
    """Read a COLMAP depth/normal map (``width&height&channels&`` + float32)."""
    with open(path, "rb") as handle:
        header = b""
        separators = 0
        while separators < 3:
            char = handle.read(1)
            if not char:
                raise ValueError(f"Truncated COLMAP array header: {path}")
            header += char
            if char == b"&":
                separators += 1
        width, height, channels = (int(value) for value in header.decode("ascii").strip("&").split("&"))
        data = np.fromfile(handle, dtype=np.float32)
    if data.size != width * height * channels:
        raise ValueError(f"COLMAP array size mismatch in {path}")
    array = data.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze(axis=2) if channels == 1 else np.transpose(array, (1, 0, 2))


def write_colmap_array(path: Path, array: np.ndarray) -> None:
    """Write a COLMAP depth/normal map; used by tests and tooling."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    height, width, channels = array.shape
    with open(path, "wb") as handle:
        handle.write(f"{width}&{height}&{channels}&".encode("ascii"))
        np.transpose(array, (1, 0, 2)).reshape(-1, order="F").tofile(handle)


def _pose_matrices(image) -> tuple[np.ndarray, np.ndarray]:
    pose = image.cam_from_world() if callable(getattr(image, "cam_from_world", None)) else image.cam_from_world
    return np.asarray(pose.rotation.matrix(), dtype=float), np.asarray(pose.translation, dtype=float)


def load_views(sparse_model: Path) -> list[View]:
    """Load dense-workspace views in COLMAP's MVS order (the indices used in fused.ply.vis).

    That order is ``reg_image_ids()`` (frame by frame), which differs from
    ascending image ids for rigs such as 360 cubemaps.
    """
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(sparse_model))
    views = []
    for index, image_id in enumerate(reconstruction.reg_image_ids()):
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        params = [float(value) for value in camera.params]
        if len(params) == 3:  # SIMPLE_PINHOLE
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:  # PINHOLE (undistorted workspaces)
            fx, fy, cx, cy = params[:4]
        rotation, translation = _pose_matrices(image)
        views.append(View(index, image.name, rotation, translation, fx, fy, cx, cy, int(camera.width), int(camera.height)))
    return views


def _voxel_keys(points: np.ndarray, size: float) -> np.ndarray:
    cells = np.floor((points - points.min(axis=0)) / size).astype(np.int64)
    span = cells.max(axis=0) + 1
    return (cells[:, 0] * span[1] + cells[:, 1]) * span[2] + cells[:, 2]


def voxel_size_for(points: np.ndarray, target: int) -> float:
    """Return a voxel size whose occupied-voxel count is close to ``target``."""
    extent = float(np.max(np.ptp(points, axis=0))) or 1.0
    low, high = extent / 1e5, extent
    for _ in range(24):
        middle = math.sqrt(low * high)
        if np.unique(_voxel_keys(points, middle)).size > target:
            low = middle
        else:
            high = middle
    return high


def _downsample(points: np.ndarray, size: float) -> np.ndarray:
    _, first = np.unique(_voxel_keys(points, size), return_index=True)
    return points[np.sort(first)]


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    return normal, -float(normal @ centroid)


def detect_planes(points: np.ndarray, threshold: float, settings: dict[str, Any]) -> list[Plane]:
    """Sequential RANSAC for the largest planes of a (subsampled) cloud."""
    rng = np.random.default_rng(int(settings["seed"]))
    remaining = points
    minimum = max(50, int(len(points) * float(settings["min_plane_share"])))
    planes: list[Plane] = []
    for _ in range(int(settings["max_planes"])):
        if len(remaining) < minimum:
            break
        scoring = remaining if len(remaining) <= 20_000 else remaining[rng.choice(len(remaining), 20_000, replace=False)]
        best_normal, best_offset, best_score = None, 0.0, 0
        for _ in range(int(settings["ransac_iterations"])):
            sample = remaining[rng.choice(len(remaining), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            length = np.linalg.norm(normal)
            if length < 1e-12:
                continue
            normal /= length
            offset = -float(normal @ sample[0])
            score = int(np.count_nonzero(np.abs(scoring @ normal + offset) < threshold))
            if score > best_score:
                best_normal, best_offset, best_score = normal, offset, score
        if best_normal is None:
            break
        inliers = np.abs(remaining @ best_normal + best_offset) < threshold
        if np.count_nonzero(inliers) < minimum:
            break
        normal, offset = _fit_plane(remaining[inliers])
        inliers = np.abs(remaining @ normal + offset) < threshold
        planes.append(Plane(normal, offset, int(np.count_nonzero(inliers))))
        remaining = remaining[~inliers]
    return planes


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def _fill_enclosed(occupied: np.ndarray) -> np.ndarray:
    """Return cells that are fully enclosed by occupied cells (holes)."""
    import cv2

    closed = cv2.morphologyEx(occupied.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    padded = np.pad(closed, 1, constant_values=0)
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    outside = padded.copy()
    cv2.floodFill(outside, mask, (0, 0), 2)
    holes = (outside == 0)[1:-1, 1:-1]
    return holes


def plane_candidates(
    points: np.ndarray, plane: Plane, threshold: float, cell: float, max_cells: int = 16_000_000,
    *, spacing: float | None = None, max_points: int | None = None,
) -> np.ndarray:
    """Candidate 3D points in enclosed empty cells of one plane.

    Holes are found on an occupancy grid of ``cell`` (about the dense point
    spacing). Without ``spacing`` each hole cell yields its centre; with
    ``spacing`` (metres, e.g. 0.01) each hole cell is filled with a regular
    sub-grid of that spacing, coarsened only if ``max_points`` would be exceeded.
    """
    inliers = points[np.abs(points @ plane.normal + plane.offset) < threshold]
    if len(inliers) < 3:
        return np.empty((0, 3))
    u, v = _plane_basis(plane.normal)
    coords = np.column_stack((inliers @ u, inliers @ v))
    origin = coords.min(axis=0)
    shape = np.floor((coords.max(axis=0) - origin) / cell).astype(int) + 1
    if int(shape[0]) * int(shape[1]) > max_cells:
        scale = math.sqrt(shape[0] * shape[1] / max_cells)
        cell *= scale
        shape = np.floor((coords.max(axis=0) - origin) / cell).astype(int) + 1
    index = np.floor((coords - origin) / cell).astype(int)
    occupied = np.zeros((int(shape[0]), int(shape[1])), dtype=bool)
    occupied[index[:, 0], index[:, 1]] = True
    holes = np.argwhere(_fill_enclosed(occupied) & ~occupied)
    if not len(holes):
        return np.empty((0, 3))
    plane_uv = origin + (holes + 0.5) * cell
    if spacing and spacing < cell:
        per_axis = int(math.ceil(cell / float(spacing)))
        if max_points:
            per_axis = min(per_axis, max(1, int(math.sqrt(max_points / len(holes)))))
        if per_axis > 1:
            offsets = ((np.arange(per_axis) + 0.5) / per_axis - 0.5) * cell
            grid = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)
            plane_uv = (plane_uv[:, None, :] + grid[None, :, :]).reshape(-1, 2)
    anchor = -plane.offset * plane.normal
    return anchor + plane_uv[:, :1] * u + plane_uv[:, 1:] * v


def _depth_paths(stereo_dir: Path, name: str) -> tuple[Path, Path, Path]:
    base = stereo_dir / "depth_maps" / name
    return (
        base.with_name(base.name + ".photometric.bin"),
        base.with_name(base.name + ".geometric.bin"),
        stereo_dir / "normal_maps" / f"{name}.photometric.bin",
    )


def verify_candidates(
    candidates: np.ndarray,
    candidate_normals: np.ndarray,
    views: Iterable[View],
    stereo_dir: Path,
    images_dir: Path,
    settings: dict[str, Any],
    *,
    image_reader: Callable[[Path], np.ndarray | None] | None = None,
    candidate_classes: np.ndarray | None = None,
    label_reader: Callable[[str], np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], dict[str, int]]:
    """Check candidates against all depth maps; return accepted mask, colours and views.

    With ``candidate_classes`` a view only supports a candidate if, besides the
    depth and normal agreement, its SAM label raster (``label_reader(view.name)``)
    shows exactly the candidate's class at that pixel.
    """
    count = len(candidates)
    support = np.zeros(count, dtype=np.int32)
    depth_support = np.zeros(count, dtype=np.int32)
    limit = int(settings["max_vis_views"])
    vis_views = np.zeros((count, limit), dtype=np.uint32)
    vis_count = np.zeros(count, dtype=np.int32)
    violations = np.zeros(count, dtype=np.int32)
    colour_sum = np.zeros((count, 3), dtype=np.float64)
    colour_count = np.zeros(count, dtype=np.int32)
    tolerance = float(settings["max_depth_error"])
    margin = float(settings["free_space_margin"])
    cos_view = math.cos(math.radians(float(settings["max_view_angle_deg"])))
    cos_normal = math.cos(math.radians(float(settings["max_normal_error_deg"])))
    stats = {"views": 0, "views_with_depth": 0}
    reader = image_reader or _read_image
    for view in views:
        stats["views"] += 1
        photometric_path, geometric_path, normal_path = _depth_paths(stereo_dir, view.name)
        if not photometric_path.is_file() and not geometric_path.is_file():
            continue
        cam = candidates @ view.rotation.T + view.translation
        z = cam[:, 2]
        front = z > 1e-9
        if not np.any(front):
            continue
        rays = candidates - view.center
        distance = np.linalg.norm(rays, axis=1)
        facing = np.abs(np.einsum("ij,ij->i", rays, candidate_normals)) / np.maximum(distance, 1e-12)
        selected = np.flatnonzero(front & (facing >= cos_view))
        if not len(selected):
            continue
        photometric = read_colmap_array(photometric_path) if photometric_path.is_file() else None
        geometric = read_colmap_array(geometric_path) if geometric_path.is_file() else None
        reference = photometric if photometric is not None else geometric
        stats["views_with_depth"] += 1
        height, width = reference.shape[:2]
        sx, sy = width / view.width, height / view.height
        zs = z[selected]
        px = np.floor((view.fx * cam[selected, 0] / zs + view.cx) * sx).astype(np.int64)
        py = np.floor((view.fy * cam[selected, 1] / zs + view.cy) * sy).astype(np.int64)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        selected, zs, px, py = selected[inside], zs[inside], px[inside], py[inside]
        if not len(selected):
            continue
        if geometric is not None:
            reliable = geometric[py, px]
            behind = (reliable > 0) & (reliable > zs * (1.0 + margin))
            violations[selected[behind]] += 1
        if photometric is None:
            measured = geometric[py, px]
        else:
            measured = photometric[py, px]
        agrees = (measured > 0) & (np.abs(measured - zs) <= tolerance * zs)
        if normal_path.is_file() and np.any(agrees):
            normal_map = read_colmap_array(normal_path)
            if normal_map.shape[:2] == (height, width):
                world_normals = normal_map[py, px] @ view.rotation
                agreement = np.abs(np.einsum("ij,ij->i", world_normals, candidate_normals[selected]))
                agrees &= agreement >= cos_normal
        depth_support[selected[agrees]] += 1
        if candidate_classes is not None and np.any(agrees):
            labels = label_reader(view.name) if label_reader is not None else None
            if labels is None:
                agrees[:] = False
            else:
                lh, lw = labels.shape[:2]
                lx = np.clip((px / sx * lw / view.width).astype(np.int64), 0, lw - 1)
                ly = np.clip((py / sy * lh / view.height).astype(np.int64), 0, lh - 1)
                agrees &= labels[ly, lx] == candidate_classes[selected]
        hit = selected[agrees]
        if not len(hit):
            continue
        support[hit] += 1
        slot = vis_count[hit]
        room = slot < limit
        vis_views[hit[room], slot[room]] = view.index
        vis_count[hit[room]] += 1
        image = reader(images_dir / view.name)
        if image is not None:
            ih, iw = image.shape[:2]
            ix = np.clip((px[agrees] / sx * iw / view.width).astype(np.int64), 0, iw - 1)
            iy = np.clip((py[agrees] / sy * ih / view.height).astype(np.int64), 0, ih - 1)
            colour_sum[hit] += image[iy, ix, :3]
            colour_count[hit] += 1
    accepted = (support >= int(settings["min_support_views"])) & (violations <= int(settings["max_free_space_violations"]))
    colours = np.where(
        colour_count[:, None] > 0, colour_sum / np.maximum(colour_count, 1)[:, None], 200.0,
    ).round().clip(0, 255).astype(np.uint8)
    empty_views = np.empty(0, dtype=np.uint32)
    visibility: list[np.ndarray] = [empty_views] * count
    for index in np.flatnonzero(accepted):
        visibility[index] = vis_views[index, : vis_count[index]].copy()
    if candidate_classes is not None:
        minimum = int(settings["min_support_views"])
        stats["rejected_class"] = int(np.count_nonzero((depth_support >= minimum) & (support < minimum)))
    stats["rejected_free_space"] = int(np.count_nonzero(violations > int(settings["max_free_space_violations"])))
    stats["rejected_support"] = int(np.count_nonzero(support < int(settings["min_support_views"])))
    return accepted, colours, visibility, stats


def _read_image(path: Path) -> np.ndarray | None:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if image is None else image[:, :, ::-1]


def fill_planes(
    points: np.ndarray,
    views: list[View],
    stereo_dir: Path,
    images_dir: Path,
    settings: dict[str, Any] | None = None,
    *,
    image_reader: Callable[[Path], np.ndarray | None] | None = None,
) -> FillResult:
    """Detect planes, build hole candidates and keep only depth-verified points."""
    settings = {**DEFAULTS, **(settings or {})}
    points = np.asarray(points, dtype=np.float64)
    empty = FillResult(np.empty((0, 3)), np.empty((0, 3), np.uint8), np.empty((0, 3)), [], {})
    if len(points) < 100:
        empty.report = {"added_points": 0, "reason": "too few dense points"}
        return empty
    coarse = voxel_size_for(points, int(settings["ransac_points"]))
    fine = voxel_size_for(points, max(1000, len(points) // 2))
    threshold = 1.5 * coarse
    sample = _downsample(points, coarse)
    planes = detect_planes(sample, threshold, settings)
    candidate_blocks, normal_blocks = [], []
    budget = int(settings["max_candidates"])
    for plane in planes:
        block = plane_candidates(points, plane, threshold, fine)
        if not len(block):
            continue
        block = block[: max(0, budget)]
        budget -= len(block)
        candidate_blocks.append(block)
        normal_blocks.append(np.repeat(plane.normal[None, :], len(block), axis=0))
        if budget <= 0:
            break
    report: dict[str, Any] = {
        "planes": len(planes),
        "voxel_size": coarse,
        "fill_spacing": fine,
        "candidates": int(sum(len(block) for block in candidate_blocks)),
    }
    if not candidate_blocks:
        empty.report = {**report, "added_points": 0}
        return empty
    candidates = np.concatenate(candidate_blocks)
    normals = np.concatenate(normal_blocks)
    accepted, colours, visibility, stats = verify_candidates(
        candidates, normals, views, stereo_dir, images_dir, settings, image_reader=image_reader,
    )
    report.update(stats)
    report["added_points"] = int(np.count_nonzero(accepted))
    return FillResult(
        candidates[accepted], colours[accepted], normals[accepted],
        [visibility[index] for index in np.flatnonzero(accepted)], report,
    )


def append_to_fused(ply_path: Path, vis_path: Path, result: FillResult) -> None:
    """Append verified points to COLMAP's fused.ply and fused.ply.vis in place."""
    from plyfile import PlyData, PlyElement

    if not len(result.xyz):
        return
    ply = PlyData.read(str(ply_path), mmap=False)
    vertex = ply["vertex"].data
    extra = np.zeros(len(result.xyz), dtype=vertex.dtype)
    names = vertex.dtype.names or ()
    for axis, name in enumerate(("x", "y", "z")):
        extra[name] = result.xyz[:, axis]
    for axis, name in enumerate(("nx", "ny", "nz")):
        if name in names:
            extra[name] = result.normals[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        if name in names:
            extra[name] = result.rgb[:, axis]
    combined = np.concatenate([vertex, extra])
    PlyData([PlyElement.describe(combined, "vertex")], text=False, byte_order="<").write(str(ply_path))
    with open(vis_path, "r+b") as handle:
        existing = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(0)
        handle.write(struct.pack("<Q", existing + len(result.visibility)))
        handle.seek(0, 2)
        for views in result.visibility:
            handle.write(struct.pack("<I", len(views)))
            handle.write(np.asarray(views, dtype="<u4").tobytes())


def fill_dense_workspace(dense_dir: Path, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run depth-verified plane filling on a COLMAP dense workspace."""
    from plyfile import PlyData

    dense_dir = Path(dense_dir)
    settings = {**DEFAULTS, **(settings or {})}
    if not bool(settings.get("enabled", True)):
        return {"enabled": False, "added_points": 0}
    ply_path, vis_path = dense_dir / "fused.ply", dense_dir / "fused.ply.vis"
    stereo_dir = dense_dir / "stereo"
    if not ply_path.is_file() or not vis_path.is_file():
        return {"enabled": True, "added_points": 0, "reason": "fused.ply or fused.ply.vis missing"}
    if not (stereo_dir / "depth_maps").is_dir():
        return {"enabled": True, "added_points": 0, "reason": "depth maps missing"}
    # Read without memory-mapping: append_to_fused rewrites this very file, and
    # Windows refuses to write a file that is still mapped (Errno 22).
    vertex = read_vertex_array(ply_path)
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    del vertex
    views = load_views(dense_dir / "sparse")
    result = fill_planes(points, views, stereo_dir, dense_dir / "images", settings)
    append_to_fused(ply_path, vis_path, result)
    report = {"enabled": True, "original_points": int(len(points)), **result.report}
    (dense_dir / "plane_fill.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# --- Class-confirmed filling (indoor) -------------------------------------
#
# Runs after SAM classification. Planes are searched separately inside the
# points SAM labelled floor, ceiling or wall; holes in those planes are filled
# on a regular grid (``fill_spacing_m``, e.g. 1 cm). A candidate is kept only
# if at least ``min_support_views`` cameras confirm it twice: their depth map
# measures the plane there AND their SAM label raster shows the same class at
# that pixel. No camera may see a surface behind it (free-space check).


def fill_classified_planes(
    points: np.ndarray,
    classes: np.ndarray,
    fill_classes: dict[int, str],
    views: list[View],
    stereo_dir: Path,
    images_dir: Path,
    settings: dict[str, Any] | None = None,
    *,
    label_reader: Callable[[str], np.ndarray | None],
    image_reader: Callable[[Path], np.ndarray | None] | None = None,
) -> FillResult:
    """Fill holes in floor/ceiling/wall planes, confirmed by depth maps and SAM labels."""
    settings = {**DEFAULTS, **(settings or {})}
    points = np.asarray(points, dtype=np.float64)
    classes = np.asarray(classes)
    empty = FillResult(np.empty((0, 3)), np.empty((0, 3), np.uint8), np.empty((0, 3)), [], {},
                       np.empty(0, np.uint8))
    if len(points) < 100 or not fill_classes:
        empty.report = {"added_points": 0, "reason": "too few points or no fill classes"}
        return empty
    coarse = voxel_size_for(points, int(settings["ransac_points"]))
    fine = voxel_size_for(points, max(1000, len(points) // 2))
    threshold = 1.5 * coarse
    spacing = settings.get("fill_spacing_m")
    spacing = float(spacing) if spacing else None
    budget = int(settings["max_candidates"])
    blocks, normal_blocks, class_blocks = [], [], []
    planes_per_class: dict[str, int] = {}
    for class_id, name in fill_classes.items():
        own = points[classes == class_id]
        if len(own) < 100:
            planes_per_class[name] = 0
            continue
        planes = detect_planes(_downsample(own, coarse), threshold, settings)
        planes_per_class[name] = len(planes)
        for plane in planes:
            if budget <= 0:
                break
            block = plane_candidates(points, plane, threshold, fine, spacing=spacing, max_points=budget)[:budget]
            if not len(block):
                continue
            budget -= len(block)
            blocks.append(block)
            normal_blocks.append(np.repeat(plane.normal[None, :], len(block), axis=0))
            class_blocks.append(np.full(len(block), class_id, dtype=np.uint8))
    report: dict[str, Any] = {
        "planes": planes_per_class,
        "voxel_size": coarse,
        "hole_cell": fine,
        "fill_spacing_m": spacing if spacing and spacing < fine else fine,
        "candidates": int(sum(len(block) for block in blocks)),
        "candidate_limit_reached": budget <= 0,
    }
    if not blocks:
        empty.report = {**report, "added_points": 0}
        return empty
    candidates = np.concatenate(blocks)
    normals = np.concatenate(normal_blocks)
    candidate_classes = np.concatenate(class_blocks)
    accepted, colours, visibility, stats = verify_candidates(
        candidates, normals, views, stereo_dir, images_dir, settings,
        image_reader=image_reader, candidate_classes=candidate_classes, label_reader=label_reader,
    )
    report.update(stats)
    report["added_points"] = int(np.count_nonzero(accepted))
    report["added_per_class"] = {
        name: int(np.count_nonzero(accepted & (candidate_classes == class_id)))
        for class_id, name in fill_classes.items()
    }
    return FillResult(
        candidates[accepted], colours[accepted], normals[accepted],
        [visibility[index] for index in np.flatnonzero(accepted)], report, candidate_classes[accepted],
    )


def fill_class_ids(class_names: list[str], wanted: Iterable[str]) -> dict[int, str]:
    """Map wanted class names (first term, case-insensitive) to 1-based class ids."""
    short = [name.split(",", 1)[0].strip().lower() for name in class_names]
    return {short.index(name.lower()) + 1: name.lower() for name in wanted if name.lower() in short}


def _read_label_raster(labels_dir: Path) -> Callable[[str], np.ndarray | None]:
    import cv2

    def reader(name: str) -> np.ndarray | None:
        path = Path(labels_dir) / f"{name}.png"
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None

    return reader


def _extend_scores(scores_path: Path, original: int, result: FillResult) -> bool:
    """Append confidence rows for filled points so refine.py still sees one row per point."""
    with np.load(scores_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(len(values) != original for values in arrays.values()):
        return False
    added = len(result.xyz)
    extra = {
        "top1_class": result.classes.astype(np.uint8),
        "top1_conf": np.ones(added, dtype=np.float16),
        "top2_class": np.zeros(added, dtype=np.uint8),
        "top2_conf": np.zeros(added, dtype=np.float16),
        "view_count": np.array([len(views) for views in result.visibility], dtype=np.uint16),
    }
    merged = {
        name: np.concatenate([values, extra.get(name, np.zeros(added, dtype=values.dtype)).astype(values.dtype)])
        for name, values in arrays.items()
    }
    np.savez(scores_path, **merged)
    return True


def fill_classified_cloud(
    dense_dir: Path,
    classified_ply: Path,
    labels_dir: Path,
    class_names: list[str],
    settings: dict[str, Any] | None = None,
    *,
    scores_path: Path | None = None,
    views: list[View] | None = None,
    image_reader: Callable[[Path], np.ndarray | None] | None = None,
    prior_reader: Callable[[str], np.ndarray | None] | None = None,
) -> dict[str, Any]:
    """Fill floor/ceiling/wall holes of a classified cloud in place (indoor only)."""
    from plyfile import PlyData, PlyElement

    dense_dir, classified_ply = Path(dense_dir), Path(classified_ply)
    settings = {**DEFAULTS, **(settings or {})}
    report_path = dense_dir / "plane_fill.json"

    def finish(report: dict[str, Any]) -> dict[str, Any]:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    if not bool(settings.get("enabled", True)):
        return {"enabled": False, "added_points": 0}
    fill_classes = fill_class_ids(class_names, settings.get("classes") or [])
    stereo_dir = dense_dir / "stereo"
    if not fill_classes:
        return finish({"enabled": True, "added_points": 0, "reason": "class set has no floor/ceiling/wall classes"})
    if not (stereo_dir / "depth_maps").is_dir():
        return finish({"enabled": True, "added_points": 0, "reason": "depth maps missing"})
    if not Path(labels_dir).is_dir():
        return finish({"enabled": True, "added_points": 0, "reason": "SAM label rasters missing"})
    # No memory-mapping: the same file is rewritten below (Windows Errno 22).
    vertex = read_vertex_array(classified_ply)
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    classes = np.asarray(vertex["scalar_class"], dtype=np.uint8)
    if views is None:
        views = load_views(dense_dir / "sparse")
    if settings.get("rule") == "sam_plane_free_space":
        result = fill_by_class_rule(
            points, classes, fill_classes, views, stereo_dir, dense_dir / "images", settings,
            label_reader=_read_label_raster(Path(labels_dir)), image_reader=image_reader, prior_reader=prior_reader,
        )
    else:
        result = fill_classified_planes(
            points, classes, fill_classes, views, stereo_dir, dense_dir / "images", settings,
            label_reader=_read_label_raster(Path(labels_dir)), image_reader=image_reader,
        )
    report = {"enabled": True, "original_points": int(len(points)), **result.report}
    if len(result.xyz):
        extra = np.zeros(len(result.xyz), dtype=vertex.dtype)
        for axis, name in enumerate(("x", "y", "z")):
            extra[name] = result.xyz[:, axis]
        for axis, name in enumerate(("red", "green", "blue")):
            extra[name] = result.rgb[:, axis]
        extra["scalar_class"] = result.classes
        if all(name in vertex.dtype.names for name in ("nx", "ny", "nz")):
            for axis, name in enumerate(("nx", "ny", "nz")):
                extra[name] = result.normals[:, axis]
        combined = np.concatenate([vertex, extra])
        del vertex
        PlyData([PlyElement.describe(combined, "vertex")], text=False).write(str(classified_ply))
        if scores_path is not None and Path(scores_path).is_file():
            report["scores_extended"] = _extend_scores(Path(scores_path), len(points), result)
    return finish(report)


# --- Rule A: SAM class + measured plane + free space (no depth hit required) -
#
# White walls have no depth measurements at all, so requiring depth hits
# leaves them empty. Under rule A a point on a plane fitted to measured
# floor/ceiling/wall points is added when
#   * at least ``min_support_views`` cameras see it unoccluded and their SAM
#     label at that pixel is the plane's class,
#   * those cameras are at least ``min_class_ratio`` of all cameras that see
#     the spot unoccluded (a doorway seen from several sides fails this), and
#   * no camera reliably measures a surface behind it (free space).
# A camera whose depth map measures something clearly in front of the point
# does not see the point (occluded) and neither supports nor counts.


def verify_by_class(
    candidates: np.ndarray,
    candidate_normals: np.ndarray,
    candidate_classes: np.ndarray,
    views: Iterable[View],
    stereo_dir: Path,
    images_dir: Path,
    settings: dict[str, Any],
    *,
    label_reader: Callable[[str], np.ndarray | None],
    image_reader: Callable[[Path], np.ndarray | None] | None = None,
    prior_reader: Callable[[str], np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], dict[str, int]]:
    """Accept candidates by SAM agreement, visibility ratio and free space.

    With ``prior_reader`` (aligned Depth-Anything-V2 depth) the pixels without
    COLMAP depth are judged too: a prior surface clearly in front occludes the
    candidate, one clearly behind counts as a prior free-space violation.
    """
    count = len(candidates)
    support = np.zeros(count, dtype=np.int32)
    visible = np.zeros(count, dtype=np.int32)
    violations = np.zeros(count, dtype=np.int32)
    prior_violations = np.zeros(count, dtype=np.int32)
    colour_sum = np.zeros((count, 3), dtype=np.float64)
    colour_count = np.zeros(count, dtype=np.int32)
    limit = int(settings["max_vis_views"])
    vis_views = np.zeros((count, limit), dtype=np.uint32)
    vis_count = np.zeros(count, dtype=np.int32)
    margin = float(settings["free_space_margin"])
    prior_margin = float(settings["prior_free_margin"])
    cos_view = math.cos(math.radians(float(settings["max_view_angle_deg"])))
    reader = image_reader or _read_image
    stats = {"views": 0, "views_with_labels": 0}
    for view in views:
        stats["views"] += 1
        labels = label_reader(view.name)
        if labels is None:
            continue
        stats["views_with_labels"] += 1
        cam = candidates @ view.rotation.T + view.translation
        z = cam[:, 2]
        rays = candidates - view.center
        distance = np.linalg.norm(rays, axis=1)
        facing = np.abs(np.einsum("ij,ij->i", rays, candidate_normals)) / np.maximum(distance, 1e-12)
        selected = np.flatnonzero((z > 1e-6) & (facing >= cos_view))
        if not len(selected):
            continue
        zs = z[selected]
        u = view.fx * cam[selected, 0] / zs + view.cx
        v = view.fy * cam[selected, 1] / zs + view.cy
        inside = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)
        selected, zs, u, v = selected[inside], zs[inside], u[inside], v[inside]
        if not len(selected):
            continue
        seen = np.ones(len(selected), dtype=bool)
        measured_any = np.zeros(len(selected), dtype=bool)
        geometric_path = stereo_dir / "depth_maps" / f"{view.name}.geometric.bin"
        if geometric_path.is_file():
            depth = read_colmap_array(geometric_path)
            dh, dw = depth.shape[:2]
            dx = np.clip((u * dw / view.width).astype(np.int64), 0, dw - 1)
            dy = np.clip((v * dh / view.height).astype(np.int64), 0, dh - 1)
            measured = depth[dy, dx]
            measured_any = measured > 0
            behind = measured_any & (measured > zs * (1.0 + margin))
            occluded = measured_any & (measured < zs * (1.0 - margin))
            violations[selected[behind]] += 1
            seen = ~behind & ~occluded
        prior = prior_reader(view.name) if prior_reader is not None else None
        if prior is not None:
            ph, pw = prior.shape[:2]
            qx = np.clip((u * pw / view.width).astype(np.int64), 0, pw - 1)
            qy = np.clip((v * ph / view.height).astype(np.int64), 0, ph - 1)
            predicted = prior[qy, qx]
            gap = ~measured_any & (predicted > 0)
            prior_behind = gap & (predicted > zs * (1.0 + prior_margin))
            prior_front = gap & (predicted < zs * (1.0 - prior_margin))
            prior_violations[selected[prior_behind]] += 1
            seen &= ~prior_behind & ~prior_front
        lh, lw = labels.shape[:2]
        lx = np.clip((u * lw / view.width).astype(np.int64), 0, lw - 1)
        ly = np.clip((v * lh / view.height).astype(np.int64), 0, lh - 1)
        agrees = seen & (labels[ly, lx] == candidate_classes[selected])
        visible[selected[seen]] += 1
        hit = selected[agrees]
        if not len(hit):
            continue
        support[hit] += 1
        slot = vis_count[hit]
        room = slot < limit
        vis_views[hit[room], slot[room]] = view.index
        vis_count[hit[room]] += 1
        image = reader(images_dir / view.name)
        if image is not None:
            ih, iw = image.shape[:2]
            ix = np.clip((u[agrees] * iw / view.width).astype(np.int64), 0, iw - 1)
            iy = np.clip((v[agrees] * ih / view.height).astype(np.int64), 0, ih - 1)
            colour_sum[hit] += image[iy, ix, :3]
            colour_count[hit] += 1
    minimum = int(settings["min_support_views"])
    ratio_ok = support >= float(settings["min_class_ratio"]) * np.maximum(visible, 1)
    prior_ok = ~(
        (prior_violations >= int(settings["min_prior_violations"])) & (prior_violations > 0.5 * support)
    )
    free_ok = (violations <= int(settings["max_free_space_violations"])) & prior_ok
    accepted = (support >= minimum) & ratio_ok & free_ok
    colours = np.where(
        colour_count[:, None] > 0, colour_sum / np.maximum(colour_count, 1)[:, None], 200.0,
    ).round().clip(0, 255).astype(np.uint8)
    empty_views = np.empty(0, dtype=np.uint32)
    visibility: list[np.ndarray] = [empty_views] * count
    for index in np.flatnonzero(accepted):
        visibility[index] = vis_views[index, : vis_count[index]].copy()
    stats["rejected_free_space"] = int(np.count_nonzero(~free_ok))
    stats["rejected_by_depth_prior"] = int(np.count_nonzero(~prior_ok))
    stats["rejected_support"] = int(np.count_nonzero(free_ok & (support < minimum)))
    stats["rejected_class_ratio"] = int(np.count_nonzero(free_ok & (support >= minimum) & ~ratio_ok))
    return accepted, colours, visibility, stats


def plane_region_candidates(
    points: np.ndarray, plane: Plane, threshold: float, settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check cells of a plane region that are too empty, plus the plane's 2D frame.

    The region is the plane's occupied area with gaps up to ``max_gap_m``
    closed and enclosed holes filled (``region_cell`` grid). Inside it, every
    ``verify_cell`` square with fewer measured points than a ``fill_spacing_m``
    grid would hold becomes a candidate (its centre is verified later).
    Returns (centres 3D, centres uv, uv origin, basis rows u/v).
    """
    import cv2

    nothing = (np.empty((0, 3)), np.empty((0, 2)), np.zeros(2), np.zeros((2, 3)))
    inliers = points[np.abs(points @ plane.normal + plane.offset) < threshold]
    if len(inliers) < 3:
        return nothing
    u_axis, v_axis = _plane_basis(plane.normal)
    coords = np.column_stack((inliers @ u_axis, inliers @ v_axis))
    origin = coords.min(axis=0)
    region_cell = float(settings["region_cell_m"])
    verify_cell = float(settings["verify_cell_m"])
    shape = np.floor((coords.max(axis=0) - origin) / region_cell).astype(int) + 1
    if int(shape[0]) * int(shape[1]) > 25_000_000:
        return nothing
    occupied = np.zeros((int(shape[0]), int(shape[1])), dtype=np.uint8)
    cells = np.floor((coords - origin) / region_cell).astype(int)
    occupied[cells[:, 0], cells[:, 1]] = 1
    radius = max(1, int(math.ceil(float(settings["max_gap_m"]) / 2.0 / region_cell)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    padded = np.pad(occupied, radius, constant_values=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)[radius:-radius, radius:-radius]
    region = (closed > 0) | _fill_enclosed(closed > 0)
    vshape = np.floor((coords.max(axis=0) - origin) / verify_cell).astype(int) + 1
    counts = np.zeros((int(vshape[0]), int(vshape[1])), dtype=np.int32)
    vcells = np.floor((coords - origin) / verify_cell).astype(int)
    np.add.at(counts, (vcells[:, 0], vcells[:, 1]), 1)
    spacing = float(settings.get("fill_spacing_m") or verify_cell)
    expected = (verify_cell / spacing) ** 2
    sparse_cells = np.argwhere(counts < 0.5 * expected)
    centres_uv = origin + (sparse_cells + 0.5) * verify_cell
    in_region = np.floor((centres_uv - origin) / region_cell).astype(int)
    in_region = np.clip(in_region, 0, np.array(region.shape) - 1)
    keep = region[in_region[:, 0], in_region[:, 1]]
    centres_uv = centres_uv[keep]
    anchor = -plane.offset * plane.normal
    centres = anchor + centres_uv[:, :1] * u_axis + centres_uv[:, 1:] * v_axis
    return centres, centres_uv, origin, np.vstack((u_axis, v_axis))


def expand_accepted_cells(
    centres_uv: np.ndarray, plane: Plane, basis: np.ndarray, existing_uv: np.ndarray, settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Turn accepted verify cells into a ``fill_spacing_m`` point grid.

    Grid points whose spacing cell already holds a measured point are skipped.
    Returns (points 3D, index of the source cell for every point).
    """
    verify_cell = float(settings["verify_cell_m"])
    spacing = float(settings.get("fill_spacing_m") or verify_cell)
    per_axis = max(1, int(round(verify_cell / spacing)))
    offsets = ((np.arange(per_axis) + 0.5) / per_axis - 0.5) * verify_cell
    grid = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)
    uv = (centres_uv[:, None, :] + grid[None, :, :]).reshape(-1, 2)
    source = np.repeat(np.arange(len(centres_uv)), len(grid))
    if len(existing_uv) and len(uv):
        base = np.minimum(uv.min(axis=0), existing_uv.min(axis=0))
        occupied_keys = np.unique(_grid_keys(existing_uv, base, spacing))
        free = ~np.isin(_grid_keys(uv, base, spacing), occupied_keys)
        uv, source = uv[free], source[free]
    anchor = -plane.offset * plane.normal
    return anchor + uv[:, :1] * basis[0] + uv[:, 1:] * basis[1], source


def _grid_keys(uv: np.ndarray, base: np.ndarray, spacing: float) -> np.ndarray:
    cells = np.floor((uv - base) / spacing).astype(np.int64)
    return cells[:, 0] * 10_000_000 + cells[:, 1]


def fill_by_class_rule(
    points: np.ndarray,
    classes: np.ndarray,
    fill_classes: dict[int, str],
    views: list[View],
    stereo_dir: Path,
    images_dir: Path,
    settings: dict[str, Any],
    *,
    label_reader: Callable[[str], np.ndarray | None],
    image_reader: Callable[[Path], np.ndarray | None] | None = None,
    prior_reader: Callable[[str], np.ndarray | None] | None = None,
) -> FillResult:
    """Rule A fill: verify ``verify_cell`` squares, then emit a ``fill_spacing_m`` grid."""
    settings = {**DEFAULTS, **settings}
    coarse = voxel_size_for(points, int(settings["ransac_points"]))
    threshold = 1.5 * coarse
    budget = int(settings["max_candidates"])
    plane_jobs = []
    planes_per_class: dict[str, int] = {}
    for class_id, name in fill_classes.items():
        own = points[classes == class_id]
        if len(own) < 100:
            planes_per_class[name] = 0
            continue
        planes = detect_planes(_downsample(own, coarse), threshold, settings)
        planes_per_class[name] = len(planes)
        for plane in planes:
            centres, centres_uv, _origin, basis = plane_region_candidates(points, plane, threshold, settings)
            if not len(centres) or budget <= 0:
                continue
            centres, centres_uv = centres[:budget], centres_uv[:budget]
            budget -= len(centres)
            plane_jobs.append((class_id, plane, centres, centres_uv, basis))
    report: dict[str, Any] = {
        "rule": "sam_plane_free_space",
        "planes": planes_per_class,
        "plane_threshold": threshold,
        "verify_cell_m": float(settings["verify_cell_m"]),
        "fill_spacing_m": float(settings.get("fill_spacing_m") or settings["verify_cell_m"]),
        "candidate_cells": int(sum(len(job[2]) for job in plane_jobs)),
        "candidate_limit_reached": budget <= 0,
    }
    empty = FillResult(np.empty((0, 3)), np.empty((0, 3), np.uint8), np.empty((0, 3)), [], {},
                       np.empty(0, np.uint8))
    if not plane_jobs:
        empty.report = {**report, "added_points": 0}
        return empty
    centres = np.concatenate([job[2] for job in plane_jobs])
    normals = np.concatenate([np.repeat(job[1].normal[None, :], len(job[2]), axis=0) for job in plane_jobs])
    cell_classes = np.concatenate([np.full(len(job[2]), job[0], dtype=np.uint8) for job in plane_jobs])
    accepted, colours, visibility, stats = verify_by_class(
        centres, normals, cell_classes, views, stereo_dir, images_dir, settings,
        label_reader=label_reader, image_reader=image_reader, prior_reader=prior_reader,
    )
    report.update(stats)
    report["depth_prior"] = prior_reader is not None
    report["accepted_cells"] = int(np.count_nonzero(accepted))
    xyz_blocks, rgb_blocks, normal_blocks, class_blocks, vis_out = [], [], [], [], []
    start = 0
    for class_id, plane, job_centres, job_uv, basis in plane_jobs:
        stop = start + len(job_centres)
        ok = np.flatnonzero(accepted[start:stop])
        if len(ok):
            inliers = points[np.abs(points @ plane.normal + plane.offset) < threshold]
            existing_uv = np.column_stack((inliers @ basis[0], inliers @ basis[1]))
            xyz, source = expand_accepted_cells(job_uv[ok], plane, basis, existing_uv, settings)
            cell = start + ok[source]
            xyz_blocks.append(xyz)
            rgb_blocks.append(colours[cell])
            normal_blocks.append(np.repeat(plane.normal[None, :], len(xyz), axis=0))
            class_blocks.append(np.full(len(xyz), class_id, dtype=np.uint8))
            vis_out.extend(visibility[index] for index in cell)
        start = stop
    if not xyz_blocks:
        empty.report = {**report, "added_points": 0}
        return empty
    result_classes = np.concatenate(class_blocks)
    report["added_points"] = int(len(result_classes))
    report["added_per_class"] = {
        name: int(np.count_nonzero(result_classes == class_id)) for class_id, name in fill_classes.items()
    }
    return FillResult(
        np.concatenate(xyz_blocks), np.concatenate(rgb_blocks), np.concatenate(normal_blocks),
        vis_out, report, result_classes,
    )
