"""Pure measurement functions for the processing protocol.

Used both by the pipeline at run time (``run_metrics_writer``) and by the collector for old runs
(``collect``). Every function is read-only, never raises for missing optional input where a ``None`` /
empty result is meaningful, and keeps source units in the key suffix (``_m``, ``_px``, ``_s``, ``_pct``).
"""

from __future__ import annotations

import ctypes
import gzip
import json
import logging
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

LOGGER = logging.getLogger(__name__)

SAMPLE_SEED = 20240611
MAX_OBSERVATIONS_PER_IMAGE = 150

_PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "int64": "i8", "uint64": "u8",
}


# ------------------------------------------------------------------------------------ alignment


def _load_reconstruction(model: Any):
    """``model`` may already be a ``pycolmap.Reconstruction``; otherwise it is a model directory."""
    import pycolmap

    if hasattr(model, "num_reg_images"):
        return model
    return pycolmap.Reconstruction(str(model))


def _camera_block(camera: Any) -> dict[str, Any]:
    names = [name.strip() for name in str(camera.params_info).split(",") if name.strip()]
    values = [float(value) for value in camera.params]
    params = {name: values[index] for index, name in enumerate(names) if index < len(values)}
    if len(names) < len(values):
        for index in range(len(names), len(values)):
            params[f"p{index}"] = values[index]
    if "f" in params:
        params["focal_px"] = params["f"]
    elif "fx" in params and "fy" in params:
        params["focal_px"] = float((params["fx"] + params["fy"]) / 2.0)
    return {
        "camera_id": int(camera.camera_id), "model": str(camera.model_name),
        "width": int(camera.width), "height": int(camera.height), "params": params,
    }


def _image_reprojection(image: Any, reconstruction: Any) -> tuple[float | None, int]:
    """RMS pixel error of the observations of ``image`` and the number of observed 3D points.

    The error is the true per-observation reprojection error (``project_point`` of the triangulated point
    against the measured 2D position). At most ``MAX_OBSERVATIONS_PER_IMAGE`` evenly spaced observations are
    projected, which keeps huge models fast; the observation count is exact.
    """
    observed: list[tuple[Any, int]] = []
    for point in image.points2D:
        if point.has_point3D():
            observed.append((point.xy, int(point.point3D_id)))
    count = len(observed)
    if count == 0:
        return None, 0
    step = max(1, count // MAX_OBSERVATIONS_PER_IMAGE)
    squares: list[float] = []
    for xy, point_id in observed[::step]:
        if not reconstruction.exists_point3D(point_id):
            continue
        projected = image.project_point(reconstruction.points3D[point_id].xyz)
        if projected is None:
            continue
        delta = np.asarray(projected, dtype=float) - np.asarray(xy, dtype=float)
        squares.append(float(delta @ delta))
    if not squares:
        return None, count
    return float(np.sqrt(np.mean(squares))), count


def alignment_metrics(model_dir: Any, input_image_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Registration quality of a COLMAP model.

    Keys: ``registered_images``, ``total_images`` (None without ``input_image_names``),
    ``unregistered_images`` (names; [] when the input list is unknown), ``submodels`` (number of sibling
    numbered models ``0``, ``1`` ... next to a numbered ``model_dir``, else None), ``tie_points``,
    ``mean_track_length``, ``mean_reprojection_error_px`` (pycolmap's mean point error), ``cameras`` and
    ``images`` (name, camera_id, center_xyz, reprojection_rms_px, observations).
    """
    reconstruction = _load_reconstruction(model_dir)
    registered = [reconstruction.images[image_id] for image_id in sorted(reconstruction.reg_image_ids())]
    names = {image.name for image in registered}
    total = None
    unregistered: list[str] = []
    if input_image_names is not None:
        wanted = [str(name) for name in input_image_names]
        total = len(wanted)
        known = {Path(name).as_posix() for name in names} | {Path(name).name for name in names}
        unregistered = [name for name in wanted if Path(name).as_posix() not in known and Path(name).name not in known]
    images = []
    for image in registered:
        error, observations = _image_reprojection(image, reconstruction)
        images.append({
            "name": image.name, "camera_id": int(image.camera_id),
            "center_xyz": [float(v) for v in image.projection_center()],
            "reprojection_rms_px": error, "observations": observations,
        })
    points = int(reconstruction.num_points3D())
    return {
        "registered_images": len(registered), "total_images": total, "unregistered_images": unregistered,
        "submodels": _count_submodels(model_dir),
        "tie_points": points,
        "mean_track_length": float(reconstruction.compute_mean_track_length()) if points else None,
        "mean_reprojection_error_px": float(reconstruction.compute_mean_reprojection_error()) if points else None,
        "cameras": [_camera_block(camera) for camera in reconstruction.cameras.values()],
        "images": images,
    }


def _count_submodels(model_dir: Any) -> int | None:
    if hasattr(model_dir, "num_reg_images"):
        return None
    path = Path(model_dir)
    if not re.fullmatch(r"\d+", path.name) or not path.parent.is_dir():
        return None
    try:
        return sum(1 for item in path.parent.iterdir()
                   if item.is_dir() and re.fullmatch(r"\d+", item.name) and (item / "points3D.bin").is_file())
    except OSError:
        return None


def overlap_grid(model: Any, max_cells: int = 120, max_points: int = 300000) -> dict[str, Any]:
    """Top-view image overlap: per grid cell the number of DISTINCT images that see a 3D point of the cell.

    The grid covers the 0.5-99.5 percentile box of the X/Y coordinates of the model frame (outliers do not
    stretch it); ``counts[row][col]`` has row 0 at the minimum Y. More than ``max_points`` points are
    sampled with a fixed seed.
    """
    reconstruction = _load_reconstruction(model)
    ids = np.asarray(list(reconstruction.points3D.keys()), dtype=np.int64)
    empty = {"cell_size": None, "origin_xy": None, "shape": [0, 0], "counts": [], "max": 0, "mean_over_occupied": None}
    if ids.size == 0:
        return empty
    if ids.size > max_points:
        ids = np.sort(np.random.default_rng(SAMPLE_SEED).choice(ids, size=max_points, replace=False))
    xy = np.empty((ids.size, 2))
    owners: list[int] = []
    owner_point: list[int] = []
    for row, point_id in enumerate(ids):
        point = reconstruction.points3D[int(point_id)]
        xy[row] = point.xyz[:2]
        for element in point.track.elements:
            owners.append(int(element.image_id))
            owner_point.append(row)
    low = np.percentile(xy, 0.5, axis=0)
    high = np.percentile(xy, 99.5, axis=0)
    extent = np.maximum(high - low, 1e-9)
    max_cells = max(1, int(max_cells))
    cell = float(max(extent) / max_cells)
    shape = (int(np.floor(extent[1] / cell)) + 1, int(np.floor(extent[0] / cell)) + 1)
    col = np.floor((xy[:, 0] - low[0]) / cell).astype(np.int64)
    row_index = np.floor((xy[:, 1] - low[1]) / cell).astype(np.int64)
    inside = (col >= 0) & (col < shape[1]) & (row_index >= 0) & (row_index < shape[0])
    point_cell = np.where(inside, row_index * shape[1] + col, -1)
    pair_cell = point_cell[np.asarray(owner_point, dtype=np.int64)]
    pair_image = np.asarray(owners, dtype=np.int64)
    keep = pair_cell >= 0
    counts = np.zeros(shape[0] * shape[1], dtype=np.int64)
    if keep.any():
        unique = np.unique(np.stack([pair_cell[keep], pair_image[keep]], axis=1), axis=0)
        cells, per_cell = np.unique(unique[:, 0], return_counts=True)
        counts[cells] = per_cell
    grid = counts.reshape(shape)
    occupied = grid[grid > 0]
    return {
        "cell_size": cell, "origin_xy": [float(low[0]), float(low[1])], "shape": [int(shape[0]), int(shape[1])],
        "counts": grid.tolist(), "max": int(grid.max()) if grid.size else 0,
        "mean_over_occupied": float(occupied.mean()) if occupied.size else None,
        "sampled_points": int(ids.size),
    }


# --------------------------------------------------------------------------------- classification


def _load_scores(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file() and path.with_name(path.name + ".gz").is_file():
        import io

        with gzip.open(path.with_name(path.name + ".gz"), "rb") as handle:
            buffer = io.BytesIO(handle.read())
        with np.load(buffer) as data:
            return {key: data[key] for key in ("top1_class", "top1_conf") if key in data.files}
    with np.load(path) as data:
        return {key: data[key] for key in ("top1_class", "top1_conf") if key in data.files}


def confidence_histogram(scores_npz_path: Any, bins: int = 20, class_names: Any = None,
                         placeholder_values: Any = ()) -> dict[str, Any]:
    """Histogram of the best-class confidence over [0, 1] plus per-class p10/p50/p90.

    Points with confidence 0 and class 0 (never written / unknown) are counted in ``unknown_points`` and kept
    out of the histogram. ``class_names`` maps class id to name (dict or list where index+1 is the id).
    A ``.npz.gz`` next to a missing ``.npz`` is read as well.

    Classified points without their own SAM score are kept out of the histogram and the percentiles and counted
    in ``without_own_score``: points with a class but confidence exactly 0 (never scored) and points whose
    confidence equals one of ``placeholder_values`` (depth fill writes 0.5, plane fill 1.0 for appended points).
    """
    scores = _load_scores(Path(scores_npz_path))
    if "top1_conf" not in scores:
        raise ValueError("scores file has no top1_conf")
    conf = scores["top1_conf"].astype(np.float32)
    classes = scores["top1_class"].astype(np.int64) if "top1_class" in scores else np.ones(conf.size, dtype=np.int64)
    classified = classes > 0
    placeholder = np.zeros(conf.size, dtype=bool)
    for value in placeholder_values or ():
        placeholder |= conf == np.float16(value).astype(np.float32)
    own = classified & (conf > 0.0) & ~placeholder
    without_own = int((classified & ~own).sum())
    values = np.clip(conf[own], 0.0, 1.0)
    counts, edges = np.histogram(values, bins=int(bins), range=(0.0, 1.0))
    if isinstance(class_names, Mapping):
        names = {int(key): str(value) for key, value in class_names.items()}
    elif class_names:
        names = {index + 1: str(value) for index, value in enumerate(class_names)}
    else:
        names = {}
    per_class: dict[str, dict[str, Any]] = {}
    for class_id in np.unique(classes[own]):
        sel = np.clip(conf[own & (classes == class_id)], 0.0, 1.0)
        p10, p50, p90 = (float(v) for v in np.percentile(sel, [10, 50, 90]))
        per_class[names.get(int(class_id), str(int(class_id)))] = {
            "class_id": int(class_id), "points": int(sel.size), "p10": p10, "p50": p50, "p90": p90,
        }
    return {
        "bins": int(bins), "edges": [float(v) for v in edges], "counts": [int(v) for v in counts],
        "points": int(values.size), "unknown_points": int((~classified).sum()),
        "without_own_score": without_own, "placeholder_values": [float(v) for v in (placeholder_values or ())],
        "mean_conf": float(values.mean()) if values.size else None, "per_class": per_class,
    }


# ------------------------------------------------------------------------------------ dense cloud


def ply_vertex_count(path: Path) -> int | None:
    """Vertex count from the PLY header only (plain or ``.gz``)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as handle:
            for _ in range(200):
                line = handle.readline().decode("ascii", errors="replace").strip()
                if line.startswith("element vertex "):
                    return int(line.rsplit(" ", 1)[1])
                if line == "end_header":
                    return None
    except (OSError, ValueError, EOFError):
        return None
    return None


def _ply_xy_chunks(path: Path, chunk: int = 4_000_000):
    """Yield (N, 2) float64 arrays of x/y of a binary PLY via memory map, without loading other columns."""
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path.name}")
        fmt, count, props, in_vertex = None, 0, [], False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header without end_header")
            words = line.decode("ascii", errors="replace").split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "end_header":
                offset = handle.tell()
                break
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                in_vertex = words[1] == "vertex"
                if in_vertex:
                    count = int(words[2])
                elif count:
                    in_vertex = False
            elif words[0] == "property" and in_vertex:
                if words[1] == "list" or words[1] not in _PLY_DTYPES:
                    raise ValueError("unsupported PLY property")
                props.append((words[2], _PLY_DTYPES[words[1]]))
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError("only binary PLY is supported")
    order = "<" if fmt == "binary_little_endian" else ">"
    dtype = np.dtype([(name, order + kind) for name, kind in props])
    data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    for start in range(0, count, chunk):
        part = data[start:start + chunk]
        yield np.column_stack([np.asarray(part["x"], dtype=np.float64), np.asarray(part["y"], dtype=np.float64)])


def dense_density(source: Any, metric: bool, cell_m: float = 0.25) -> dict[str, Any]:
    """Point count, occupied top-view area and density of a cloud.

    ``source`` is an (N, 2+) array or a PLY path. Occupied area = occupied grid cells x ``cell_m``^2 over all
    points (chunked, nothing is sampled so the area is exact). Without a metric scale the density is None
    with ``reason`` "no metric scale".
    """
    occupied_keys: np.ndarray | None = None
    total = 0
    cell = float(cell_m)
    if isinstance(source, (str, Path)):
        chunks = _ply_xy_chunks(Path(source))
    else:
        array = np.asarray(source, dtype=np.float64)
        chunks = iter([array[:, :2]] if array.size else [])
    origin: np.ndarray | None = None
    for part in chunks:
        part = part[np.isfinite(part).all(axis=1)]
        if part.size == 0:
            continue
        if origin is None:
            origin = part.min(axis=0)
        total += int(part.shape[0])
        idx = np.floor((part - origin) / cell).astype(np.int64)
        key = (idx[:, 0] + (1 << 30)) * (1 << 31) + (idx[:, 1] + (1 << 30))
        unique = np.unique(key)
        occupied_keys = unique if occupied_keys is None else np.union1d(occupied_keys, unique)
    cells = int(occupied_keys.size) if occupied_keys is not None else 0
    area = cells * cell * cell
    result: dict[str, Any] = {"points": total, "cell_m": cell, "occupied_cells": cells, "area_m2": area if metric else None}
    if not metric:
        result.update(density_pts_m2=None, reason="no metric scale")
    elif area > 0:
        result.update(density_pts_m2=total / area, reason=None)
    else:
        result.update(density_pts_m2=None, reason="no points")
    return result


# ------------------------------------------------------------------------------ refinement table


def _num(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _sum_counts(counts: Any) -> int | None:
    if not isinstance(counts, Mapping) or not counts:
        return None
    try:
        return int(sum(int(v) for v in counts.values()))
    except (TypeError, ValueError):
        return None


def refinement_stages(run_metrics: Mapping[str, Any], warnings: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Unified step-6 table in pipeline order.

    Each row: ``stage, before, after, removed, added, skipped, reason, seconds``. Values that the run did not
    record stay None (never guessed). A stage is ``skipped`` when its settings block is absent, disabled, or
    carries an ``error`` / ``skipped`` entry (reason filled), or when a run-summary warning (``warnings``)
    says the step was skipped.
    """
    settings = run_metrics.get("settings") if isinstance(run_metrics.get("settings"), Mapping) else {}
    seconds = run_metrics.get("stage_seconds") if isinstance(run_metrics.get("stage_seconds"), Mapping) else {}
    rows: list[dict[str, Any]] = []

    def block(key: str) -> Mapping[str, Any] | None:
        value = settings.get(key)
        return value if isinstance(value, Mapping) else None

    def row(stage: str, section: Mapping[str, Any] | None, stage_key: str | None, *, before=None, after=None,
            removed=None, added=None, absent_reason: str = "not run") -> dict[str, Any]:
        entry = {"stage": stage, "before": _num(before), "after": _num(after), "removed": _num(removed),
                 "added": _num(added), "skipped": False, "reason": None,
                 "seconds": _num(seconds.get(stage_key)) if stage_key else None}
        if section is None:
            entry.update(skipped=True, reason=absent_reason)
        elif section.get("error"):
            entry.update(skipped=True, reason=str(section["error"]))
        elif section.get("enabled") is False:
            entry.update(skipped=True, reason=str(section.get("skipped") or "disabled"))
        elif section.get("skipped"):
            entry.update(skipped=True, reason=str(section["skipped"]))
        return entry

    prior = block("depth_prior")
    rows.append(row("depth_prior", prior, "Depth prior", added=None))
    fusion = block("surface_fusion")
    mirrors = fusion.get("mirrors") if fusion else None
    mirror_removed = _num(fusion.get("mirror_points_removed")) if fusion else None
    mirror = row("mirror_detection", fusion if fusion is not None else None, "Mirror detection",
                 removed=mirror_removed, absent_reason="surface fusion not run")
    mirror["skipped"] = fusion is None or not isinstance(mirrors, list)
    mirror["reason"] = None if not mirror["skipped"] else (mirror["reason"] or "no mirror report")
    if isinstance(mirrors, list):
        mirror["detected"] = len(mirrors)
    rows.append(mirror)
    fill = block("depth_fill")
    fill_before = fill.get("original_points") if fill else None
    fill_added = fill.get("added_points") if fill else None
    if fill_added is None and fill:
        fill_added = _sum_counts(fill.get("added_per_class_id"))
    rows.append(row("depth_fill", fill, "Depth fill", before=fill_before,
                    after=(fill_before + fill_added) if _num(fill_before) is not None and _num(fill_added) is not None else None,
                    added=fill_added))
    cleaning = block("point_cleaning")
    cl_before = cleaning.get("original_points") if cleaning else None
    cl_removed = cleaning.get("removed_points") if cleaning else None
    rows.append(row("point_cleaning", cleaning, "Point cleaning", before=cl_before,
                    after=(cl_before - cl_removed) if _num(cl_before) is not None and _num(cl_removed) is not None else None,
                    removed=cl_removed))
    rows.append(row("surface_fusion", fusion, "Surface fusion",
                    before=fusion.get("original_points") if fusion else None,
                    after=fusion.get("final_points") if fusion else None,
                    removed=(fusion["original_points"] - fusion["final_points"]) if fusion
                    and _num(fusion.get("original_points")) is not None and _num(fusion.get("final_points")) is not None
                    else None))
    sub_keys = ("mirror_points_removed", "removed_visibility", "removed_outside_levels",
                "removed_visibility_after_planes", "removed_hidden", "removed_unconfirmed", "raymarch_removed",
                "behind_wall_removed", "mirror_image_removed", "removed_fragments", "removed_sparse")
    if fusion and not fusion.get("error") and fusion.get("enabled") is not False:
        for key in sub_keys:
            value = _num(fusion.get(key))
            if value is not None:
                rows.append({"stage": f"surface_fusion.{key}", "before": None, "after": None, "removed": value,
                             "added": None, "skipped": False, "reason": None, "seconds": None})
        unseen = fusion.get("removed_unseen_walls")
        unseen_total = _sum_counts(unseen) if isinstance(unseen, Mapping) else _num(unseen)
        if unseen_total is not None:
            rows.append({"stage": "surface_fusion.removed_unseen_walls", "before": None, "after": None,
                         "removed": unseen_total, "added": None, "skipped": False, "reason": None, "seconds": None})
    plane = block("plane_fill")
    rows.append(row("plane_fill", plane, "Plane filling", added=plane.get("added_points") if plane else None))
    vote = block("superpoint_vote")
    changed = vote.get("changed_points") if vote else None
    changed_count = changed.get("count") if isinstance(changed, Mapping) else None
    entry = row("superpoint_vote", vote, "Superpoint voting",
                before=vote.get("point_count") if vote else None, after=vote.get("point_count") if vote else None)
    entry["changed"] = _num(changed_count)
    rows.append(entry)
    evidence = block("depth_evidence")
    rows.append(row("depth_evidence", evidence, "Depth evidence", added=evidence.get("points") if evidence else None))
    marker = {"depth_fill": "Depth-Anything fill was skipped", "point_cleaning": "Point cleaning was skipped",
              "surface_fusion": "Surface fusion was skipped", "plane_fill": "Plane filling was skipped",
              "superpoint_vote": "Superpoint voting was skipped"}
    for line in warnings or ():
        for item in rows:
            text = marker.get(item["stage"])
            if text and text in str(line) and not item["skipped"]:
                item.update(skipped=True, reason=str(line).split(":", 2)[-1].strip() or "skipped (warning)")
    return rows


# ------------------------------------------------------------------------------------ environment


def software_info(repo_root: Path | None = None) -> dict[str, Any]:
    from samcloud import __version__

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    short = git("rev-parse", "--short", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no") if short else None
    return {"version": __version__, "git_hash": short or None,
            "git_dirty": None if status is None else bool(status)}


def _ram_gb() -> float | None:
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("pt", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.total / 1024 ** 3, 1)
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 ** 2, 1)
    except Exception:  # noqa: BLE001 - hardware info is optional
        return None
    return None


def hardware_info() -> dict[str, Any]:
    gpu_name = vram = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name, vram = props.name, round(props.total_memory / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001 - torch is optional here
        pass
    return {"gpu_name": gpu_name, "vram_gb": vram, "ram_gb": _ram_gb(),
            "cpu": platform.processor() or None, "cpu_count": os.cpu_count(),
            "os": f"{platform.system()} {platform.release()}".strip()}


def log_warnings(log_path: Any, limit: int = 200) -> list[str]:
    """Lines of a log that contain "WARNING" (deduplicated, order kept, at most ``limit``)."""
    seen: set[str] = set()
    found: list[str] = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if "WARNING" in line and line not in seen:
                    seen.add(line)
                    found.append(line)
                    if len(found) >= limit:
                        break
    except OSError:
        return []
    return found


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
