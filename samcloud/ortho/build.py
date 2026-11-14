"""Public true-orthophoto orchestration and cache inspection API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import cv2
import numpy as np
import pycolmap
from scipy.spatial import cKDTree
import torch

from samcloud.core.sparse_model import resolve_sparse_model
from samcloud.core.configuration import load_yaml
from samcloud.core.paths import CONFIG_DIRECTORY
from samcloud.dxf.cancel import CancelToken, ProgressCallback, check_cancel, report_progress
from samcloud.dxf.las_source import Frame, read_las_info
from samcloud.dxf.levels import Level
from samcloud.dxf.state import load_state, plan_settings_for
from samcloud.ortho.cameras import load_cameras
from samcloud.ortho.colorize import colorize
from samcloud.ortho.config import OrthoConfig, load_ortho_config
from samcloud.ortho.dsm import build_dsm, read_cloud
from samcloud.ortho.grid import OrthoGrid, grid_for_bounds
from samcloud.ortho.outputs import write_outputs


class OrthoError(RuntimeError):
    """A recoverable orthophoto input or consistency failure."""


@dataclass(frozen=True, slots=True)
class OrthoResult:
    output_dir: Path
    ortho_path: Path
    dsm_path: Path
    dtm_path: Path
    classes_path: Path
    validity_path: Path
    view_count_path: Path
    grid: OrthoGrid
    stats: dict[str, Any]
    fingerprint: str


def ortho_output_dir(project_dir: Path, level: Level | None) -> Path:
    return Path(project_dir) / "products" / "ortho" / ("site" if level is None else level.id)


def _progress_wrapper(progress: ProgressCallback | None, start: float, end: float) -> Callable[[float, str], None] | None:
    if progress is None:
        return None

    def forward(fraction: float, stage: str) -> None:
        clamped = max(0.0, min(float(fraction), 1.0))
        report_progress(progress, start + (end - start) * clamped, stage)

    return forward


def build_orthophoto(project_dir: Path, *, level: Level | None, crop: tuple[float, float, float, float] | None = None, config: OrthoConfig | None = None, progress: ProgressCallback | None = None, cancel: CancelToken | None = None, device: str | None = None) -> OrthoResult:
    """Build an outdoor-site or single-level true orthophoto from original images and sparse cameras."""
    started = time.perf_counter()
    project = Path(project_dir)
    settings = config or load_ortho_config()
    point_source = _point_source(project)
    if point_source is None or not point_source.is_file():
        raise OrthoError(f"Classified point cloud was not found under {project / 'products' / 'classified'}")
    info = read_las_info(point_source)
    state = load_state(project)
    frame = plan_settings_for(state, level, base_frame=info.frame, project_directory=project).frame
    voxel_size = settings.zbuffer.voxel_indoor if level is not None else settings.zbuffer.voxel_outdoor
    check_cancel(cancel)
    read_progress = _progress_wrapper(progress, 0.0, 0.15)
    # With return_instance_ids=True the result has 7 items (instance ids after fine terms);
    # readers that ignore the keyword return the legacy 6 / 5 item formats.
    cloud = read_cloud(project, point_source, frame, voxel_size=voxel_size, progress=read_progress, return_instance_ids=True)
    instance_ids = None
    if len(cloud) == 7:
        raw_xyz, local_xyz, classes, rgb, fine_terms, instance_ids, voxel_xyz = cloud
    elif len(cloud) == 6:
        raw_xyz, local_xyz, classes, rgb, fine_terms, voxel_xyz = cloud
    else:  # Compatibility with callers that provide the pre-fine-term reader result.
        raw_xyz, local_xyz, classes, rgb, voxel_xyz = cloud
        fine_terms = np.zeros(len(raw_xyz), dtype=np.uint8)
    check_cancel(cancel)
    model_dir, model_distance = select_sparse_model(project, raw_xyz, settings)
    check_cancel(cancel)
    cameras = load_cameras(model_dir, project)
    check_cancel(cancel)
    missing_count = sum(camera.image_path is None for camera in cameras)
    if not cameras:
        raise OrthoError(f"No registered cameras were found in sparse model: {model_dir}")
    if missing_count > len(cameras) / 2:
        raise OrthoError(f"More than half of original photos are missing ({missing_count}/{len(cameras)})")
    mask = np.ones(len(local_xyz), dtype=bool)
    if level is not None:
        cut = level.ceiling_z - settings.indoor.cut_below_ceiling if level.ceiling_z is not None else level.floor_z + settings.indoor.cut_fallback_above_floor
        mask &= local_xyz[:, 2] >= level.floor_z - settings.indoor.floor_margin
        mask &= local_xyz[:, 2] <= cut
        mask &= ~np.isin(classes, settings.indoor.ceiling_codes)
    if not np.any(mask):
        raise OrthoError("No classified points remain within the requested orthophoto extent")
    bounds = (float(local_xyz[mask, 0].min()), float(local_xyz[mask, 1].min()), float(local_xyz[mask, 0].max()), float(local_xyz[mask, 1].max()))
    if crop is not None:
        bounds = _validated_crop(crop)
    grid = grid_for_bounds(bounds, settings.pixel_size_indoor if level is not None else settings.pixel_size_outdoor, settings.max_side)
    output_dir = _crop_output_dir(project, crop) if crop is not None else ortho_output_dir(project, level)
    work_dir = output_dir / "work"
    check_cancel(cancel)
    dsm_progress = _progress_wrapper(progress, 0.15, 0.25)
    dsm = build_dsm(raw_xyz, local_xyz, classes, grid, level, settings, work_dir, rgb=rgb, fine_terms=fine_terms, instance_ids=instance_ids, voxel_colmap_xyz=voxel_xyz, progress=dsm_progress)
    if level is not None and settings.dsm.surface_fill_indoor:
        dsm.full_colmap_xyz = _append_surface_fill_zbuffer_points(dsm.full_colmap_xyz, dsm.colmap_xyz, dsm.validity, settings.zbuffer.voxel_indoor)
    check_cancel(cancel)
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    report_progress(progress, 0.25, "Preparing orthophoto colour reconstruction")
    local_centres = {camera.name: frame.to_local(np.asarray(camera.centre, dtype=np.float64)[None, :])[0] for camera in cameras}
    colours = colorize(grid, dsm, cameras, settings, indoor=level is not None, device=chosen_device, camera_local_centres=local_centres, progress=progress, cancel=cancel)
    check_cancel(cancel)
    fine_labels_source = point_source.with_name("fine_labels.json")
    fingerprint = ortho_fingerprint(project, level=level, crop=crop, config=settings, frame=frame, point_source=point_source, sparse_model=model_dir)
    tilt = max(abs(float(frame.rotation[2, 0])), abs(float(frame.rotation[2, 1])))
    original_world = grid.original_world_file(frame) if tilt < 1e-6 else None
    metadata: dict[str, Any] = {
        "bounds": list(grid.bounds), "pixel_size": grid.pixel_size, "width": grid.width, "height": grid.height,
        "crop": None if crop is None else list(crop),
        "frame": {"origin": frame.origin.tolist(), "scale": frame.scale_factor, "rotation": frame.rotation.tolist()}, "mode": "indoor" if level is not None else "outdoor", "level": None if level is None else asdict(level),
        "source_paths": {"point_source": str(point_source), "fine_labels": str(fine_labels_source) if fine_labels_source.is_file() else None, "sparse_model": str(model_dir)}, "source_mtimes": _source_mtimes(point_source, model_dir), "sparse_model_used": str(model_dir), "model_cloud_distance": model_distance,
        "camera_counts": {"total": len(cameras), "available": len(cameras) - len(colours.missing_photos)}, "missing_photos": colours.missing_photos,
        "affine": {"method": dsm.xyz_method, "residual": dsm.affine_residual}, "settings_fingerprint": fingerprint, "timings": {"seconds": time.perf_counter() - started},
        "world_file_notes": "Written in original CRS with rotated affine terms." if original_world is not None else "Skipped because the DXF frame has non-z tilt.",
    }
    paths = write_outputs(output_dir, rgba=colours.rgba, dsm=dsm.dsm, dtm=dsm.dtm, classes=dsm.classes, validity=dsm.validity, view_count=colours.view_count, metadata=metadata, fingerprint=fingerprint, local_world_file=grid.local_world_file(), original_world_file=original_world, jpeg_sides=settings.jpeg_long_sides, jpeg_quality=settings.jpeg_quality, fine_pc=dsm.fine_pc, fine_pc_confidence=dsm.fine_pc_confidence, instance_pc=dsm.instance_pc, fine_labels_source=fine_labels_source)
    colour_source_path = output_dir / "colour_source.png"
    if not cv2.imwrite(str(colour_source_path), colours.colour_source, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Could not write colour source map: {colour_source_path}")
    footprint_path = output_dir / "footprint.png"
    if not cv2.imwrite(str(footprint_path), dsm.footprint.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Could not write footprint mask: {footprint_path}")
    return OrthoResult(output_dir, paths["ortho"], paths["dsm"], paths["dtm"], paths["classes"], paths["valid"], paths["view_count"], grid, metadata, fingerprint)


def load_ortho_result(directory: Path) -> OrthoResult | None:
    """Load an existing result only; this function deliberately never computes or creates anything."""
    root = Path(directory)
    metadata_path, fingerprint_path = root / "ortho.json", root / "fingerprint.txt"
    required = [metadata_path, fingerprint_path, root / "ortho.png", root / "dsm.npy", root / "dtm.npy", root / "classes_pc.npy", root / "valid.png", root / "view_count.png"]
    if not all(path.is_file() for path in required):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        grid = OrthoGrid(float(metadata["bounds"][0]), float(metadata["bounds"][1]), float(metadata["bounds"][2]), float(metadata["bounds"][3]), float(metadata["pixel_size"]), int(metadata["width"]), int(metadata["height"]))
        return OrthoResult(root, root / "ortho.png", root / "dsm.npy", root / "dtm.npy", root / "classes_pc.npy", root / "valid.png", root / "view_count.png", grid, metadata, fingerprint_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError, KeyError, TypeError):
        return None


def ortho_fingerprint(project_dir: Path, *, level: Level | None, crop: tuple[float, float, float, float] | None = None, config: OrthoConfig | None = None, frame: Frame | None = None, point_source: Path | None = None, sparse_model: Path | None = None) -> str:
    """Fingerprint every input that can affect cached orthophoto pixels."""
    project = Path(project_dir)
    source = point_source or _point_source(project)
    model = sparse_model or resolve_sparse_model(project / "products" / "sparse")
    if frame is None and source is not None:
        info = read_las_info(source)
        frame = plan_settings_for(load_state(project), level, base_frame=info.frame, project_directory=project).frame
    payload = {"point_source": _stamp(source), "fine_terms": _fine_term_stamp(source), "sparse_model": _tree_stamps(model), "frame": None if frame is None else {"origin": frame.origin.tolist(), "scale": frame.scale_factor, "rotation": frame.rotation.tolist()}, "level": None if level is None else asdict(level), "crop": None if crop is None else list(_validated_crop(crop)), "config": asdict(config or load_ortho_config())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def select_sparse_model(project_dir: Path, cloud_sample: np.ndarray, config: OrthoConfig | None = None) -> tuple[Path, float]:
    """Select the sparse reconstruction geometrically consistent with COLMAP-cloud coordinates."""
    settings = config or load_ortho_config()
    root = Path(project_dir) / "products" / "sparse"
    preferred = resolve_sparse_model(root)
    candidates = [candidate for candidate in [preferred, *sorted((path for path in root.iterdir() if path.is_dir()), key=lambda value: value.name)] if candidate is not None]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    sample = np.asarray(cloud_sample, dtype=np.float64)
    rng = np.random.default_rng(0)
    if len(sample) > 3_000_000:
        sample = sample[rng.choice(len(sample), size=3_000_000, replace=False)]
    cloud_tree = cKDTree(sample)
    distances: dict[str, float] = {}
    for candidate in unique:
        reconstruction = pycolmap.Reconstruction(candidate)
        points = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=np.float64)
        if len(points) == 0:
            continue
        if len(points) > 20_000:
            points = points[rng.choice(len(points), size=20_000, replace=False)]
        distance = float(np.median(cloud_tree.query(points, k=1)[0]))
        distances[str(candidate)] = distance
    if not distances:
        raise OrthoError(f"No sparse model with 3D points was found in {root}")
    chosen_path, chosen_distance = min(distances.items(), key=lambda item: item[1])
    scale = max(np.ptp(sample, axis=0).max(), 1.0)
    limit = max(settings.max_model_cloud_distance, settings.max_model_cloud_distance * scale * 1e-4)
    if chosen_distance > limit:
        detail = ", ".join(f"{path}: {distance:.5g}" for path, distance in distances.items())
        raise OrthoError(f"No sparse model matches classified cloud coordinates (median distances: {detail})")
    return Path(chosen_path), chosen_distance


def _point_source(project: Path) -> Path | None:
    classified = project / "products" / "classified"
    las = classified / "classified.las"
    ply = classified / "classified.ply"
    if las.is_file() and ply.is_file():
        return las if las.stat().st_mtime_ns >= ply.stat().st_mtime_ns else ply
    return las if las.is_file() else (ply if ply.is_file() else None)


def _stamp(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return {"path": os.path.normcase(str(Path(path).resolve())), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def _fine_term_stamp(point_source: Path | None) -> dict[str, Any] | None:
    """Hash the per-point fine property and its vocabulary for ortho cache invalidation."""
    if point_source is None or not point_source.is_file():
        return None
    source = Path(point_source)
    digest = hashlib.sha256()
    labels_path = source.with_name("fine_labels.json")
    if labels_path.is_file():
        digest.update(labels_path.read_bytes())
    else:
        digest.update(b"no-fine-labels")
    count = 0
    present = False
    instance_digest = hashlib.sha256()
    instance_count = 0
    instance_present = False
    if source.suffix.lower() == ".ply":
        from samcloud.engine.export_las import _ply_chunks, read_ply_header

        offset, vertex_count = read_ply_header(source)
        with source.open("rb") as handle:
            for points in _ply_chunks(handle, offset, vertex_count, 500_000):
                names = points.dtype.names or ()
                if "fine_term" in names:
                    values = np.ascontiguousarray(points["fine_term"], dtype=np.uint8)
                    digest.update(values.tobytes())
                    count += len(values)
                    present = True
                if "instance_id" in names:
                    values = np.ascontiguousarray(points["instance_id"], dtype=np.uint32)
                    instance_digest.update(values.tobytes())
                    instance_count += len(values)
                    instance_present = True
    else:
        import laspy

        with laspy.open(source) as reader:
            dimensions = {str(name).lower() for name in reader.header.point_format.dimension_names}
            if "fine_term" in dimensions:
                for points in reader.chunk_iterator(500_000):
                    values = np.ascontiguousarray(points.fine_term, dtype=np.uint8)
                    digest.update(values.tobytes())
                    count += len(values)
                present = True
            if "instance_id" in dimensions:
                for points in reader.chunk_iterator(500_000):
                    values = np.ascontiguousarray(points.instance_id, dtype=np.uint32)
                    instance_digest.update(values.tobytes())
                    instance_count += len(values)
                instance_present = True
    return {"present": present, "points": count, "sha256": digest.hexdigest(), "labels": _stamp(labels_path), "instance_id": {"present": instance_present, "points": instance_count, "sha256": instance_digest.hexdigest()}}


def _tree_stamps(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_dir():
        return []
    return [_stamp(item) for item in sorted(path.rglob("*")) if item.is_file()]


def _source_mtimes(point_source: Path, model: Path) -> dict[str, Any]:
    return {"point_source": _stamp(point_source), "sparse_model": _tree_stamps(model)}


def _validated_crop(crop: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Validate local crop bounds supplied by interactive quality iteration."""
    if len(crop) != 4:
        raise ValueError("Crop must contain min_x, min_y, max_x and max_y")
    min_x, min_y, max_x, max_y = (float(value) for value in crop)
    if not np.isfinite((min_x, min_y, max_x, max_y)).all() or min_x >= max_x or min_y >= max_y:
        raise ValueError("Crop bounds must be finite and ordered")
    return min_x, min_y, max_x, max_y


def _crop_output_dir(project_dir: Path, crop: tuple[float, float, float, float] | None) -> Path:
    """Return a deterministic, isolated output directory for a requested local crop."""
    if crop is None:
        raise ValueError("Crop output directory requires crop bounds")
    encoded = json.dumps(_validated_crop(crop), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return Path(project_dir) / "products" / "ortho" / f"_crop_{hashlib.sha256(encoded).hexdigest()[:12]}"


def ortho_quality_metrics(output_dir: Path) -> dict[str, float]:
    """Measure uncoloured DSM cells, green road leakage and view coverage for an ortho output."""
    root = Path(output_dir)
    image = cv2.imread(str(root / "ortho.png"), cv2.IMREAD_UNCHANGED)
    views = cv2.imread(str(root / "view_count.png"), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4 or views is None:
        raise OrthoError(f"Incomplete orthophoto output: {root}")
    rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    dsm = np.load(root / "dsm.npy", allow_pickle=False)
    classes = np.load(root / "classes_pc.npy", allow_pickle=False)
    validity = cv2.imread(str(root / "valid.png"), cv2.IMREAD_UNCHANGED)
    footprint = cv2.imread(str(root / "footprint.png"), cv2.IMREAD_UNCHANGED)
    if dsm.shape != rgba.shape[:2] or classes.shape != dsm.shape or views.shape != dsm.shape or validity is None or validity.shape != dsm.shape:
        raise OrthoError(f"Orthophoto output dimensions do not agree: {root}")
    valid = np.isfinite(dsm)
    roads = valid & np.isin(classes, _road_codes())
    green = (rgba[..., 1].astype(np.int16) > rgba[..., 0].astype(np.int16) + 25) & (rgba[..., 1].astype(np.int16) > rgba[..., 2].astype(np.int16) + 25)
    valid_count = int(np.count_nonzero(valid))
    road_count = int(np.count_nonzero(roads))
    surface_fill = validity == 3
    footprint_mask = footprint > 0 if footprint is not None and footprint.shape == dsm.shape else np.zeros(dsm.shape, dtype=bool)
    footprint_count = int(np.count_nonzero((validity > 0) | footprint_mask))
    return {
        "alpha_holes": float(np.count_nonzero(valid & (rgba[..., 3] == 0)) / valid_count) if valid_count else 0.0,
        "road_green_fraction": float(np.count_nonzero(green & roads) / road_count) if road_count else 0.0,
        "mean_views": float(views[valid].mean()) if valid_count else 0.0,
        "single_view_fraction": float(np.count_nonzero((views == 1) & valid) / valid_count) if valid_count else 0.0,
        "surface_fill_fraction": float(np.mean(surface_fill)),
        "surface_fill_coloured": float(np.count_nonzero(surface_fill & (rgba[..., 3] == 255)) / np.count_nonzero(surface_fill)) if np.any(surface_fill) else 0.0,
        "alpha_holes_in_footprint": float(np.count_nonzero(((validity > 0) | footprint_mask) & (rgba[..., 3] == 0)) / footprint_count) if footprint_count else 0.0,
    }


def _append_surface_fill_zbuffer_points(existing: np.ndarray, colmap_xyz: np.ndarray, validity: np.ndarray, voxel: float) -> np.ndarray:
    """Add one finite validity-3 surface point per z-buffer voxel."""
    filled = np.asarray(colmap_xyz)[np.asarray(validity) == 3]
    filled = filled[np.isfinite(filled).all(axis=1)]
    if len(filled) == 0:
        return existing
    keys = np.floor(filled / voxel).astype(np.int64)
    _unique, first = np.unique(keys, axis=0, return_index=True)
    return np.vstack((existing, filled[first]))


def _road_codes() -> tuple[int, ...]:
    """Read road and sidewalk classes from the maintained DXF layer configuration."""
    layers = load_yaml(CONFIG_DIRECTORY / "dxf_layers.yaml")
    codes: set[int] = set()
    for layer in layers.get("layers", []):
        if layer.get("mode") == "outdoor" and layer.get("name") in {"S-PAVE", "S-WALK"}:
            codes.update(int(value) for value in layer.get("codes", []))
    return tuple(sorted(codes))
