"""Build and query the compact depth-evidence product.

COLMAP geometric depths are preferred.  Locally aligned Depth-Anything priors
only supply pixels where geometric depth is absent.  All views are processed in
bounded streaming batches: every batch is reduced to per-voxel statistics (integer
voxel keys, vectorised numpy) and merged into one running state, so raw depth
samples are never kept and memory stays bounded.  The final voxel vote keeps the
product useful after the original dense cache has been released.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from samcloud.bim.plan_model import load_bim_export_config
from samcloud.colmap.plane_fill import load_views, read_colmap_array
from samcloud.engine.ply_io import read_vertex_array

SOURCE_COLMAP = np.uint8(1)
SOURCE_DEPTH_ANYTHING = np.uint8(2)

LABEL_BINS = 20  # SAM/scalar classes are 0..19 (class 0 = unlabelled is dropped); higher ids are ignored
VOTE_UNITS = 20  # votes are stored as uint16 in units of 1/20 confidence (memory: 40 B per voxel instead of 96 B)
_KEY_BITS = 21
_KEY_OFFSET = 1 << (_KEY_BITS - 1)

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "voxel_m": 0.015,
    "free_space_voxel_m": 0.05,
    "stride": 2,
    # 0 deliberately means every usable view.  A cap is only appropriate for
    # an explicitly requested preview, never for the retained evidence.
    "max_views": 0,
    "batch_views": 128,
    "free_ray_stride": 24,
    "max_geometric_depth_m": 10.0,
    "max_prior_depth_m": 6.0,
    "min_prior_views": 2,
    "min_label_views": 2,
    "min_free_through": 2,
    "depth_edge_rel": 0.04,
    "gap_radius_m": 0.03,
    "instance_radius_m": 0.08,
    "alignment_max_residual_m": 0.08,
    "alignment_sample": 400000,
    "workers": 8,
}


def load_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return depth-evidence settings merged with ``bim_export.yaml``."""
    configured = load_bim_export_config().get("depth_evidence", {})
    return {**DEFAULTS, **configured, **(settings or {})}


def _product_dir(project_dir: Path) -> Path:
    return Path(project_dir) / "products" / "depth"


def _dense_dir(project_dir: Path) -> Path | None:
    project_dir = Path(project_dir)
    candidates = (project_dir / "processing_cache" / "dense", project_dir / "products" / "dense")
    return next((path for path in candidates if (path / "stereo" / "depth_maps").is_dir()), None)


def _depth_path(dense_dir: Path, name: str, suffix: str) -> Path:
    return dense_dir / "stereo" / "depth_maps" / f"{name}.{suffix}.bin"


def _label_path(dense_dir: Path, name: str) -> Path | None:
    candidate = dense_dir / "sam_labels" / f"{name}.png"
    return candidate if candidate.is_file() else None


def _read_png(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.shape != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(image, dtype=np.uint8)


def _read_labels(dense_dir: Path, name: str, shape: tuple[int, int]) -> np.ndarray:
    image = _read_png(_label_path(dense_dir, name), shape)
    return np.zeros(shape, np.uint8) if image is None else image


def _read_mirror(dense_dir: Path, name: str, shape: tuple[int, int]) -> np.ndarray:
    """True where the view's SAM mirror raster flags a reflecting surface."""
    image = _read_png(dense_dir / "sam_mirror" / f"{name}.png", shape)
    return np.zeros(shape, bool) if image is None else image > 0


def _world_points(view, rows: np.ndarray, cols: np.ndarray, depth: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    z = depth[rows, cols].astype(np.float64)
    x = ((cols.astype(float) + 0.5) * view.width / width - view.cx) * z / view.fx
    y = ((rows.astype(float) + 0.5) * view.height / height - view.cy) * z / view.fy
    local = np.column_stack((x, y, z)) - np.asarray(view.translation)
    rotation = np.asarray(view.rotation, np.float64)
    # Explicit row combination instead of ``@``: BLAS matmul inside the worker threads deadlocked
    # intermittently (observed twice on v12, all threads idle in this call).
    return local[:, 0:1] * rotation[0] + local[:, 1:2] * rotation[1] + local[:, 2:3] * rotation[2]


def _voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    """Pack signed integer voxel coordinates into sortable int64 keys."""
    cells = np.floor(np.asarray(points, np.float64) / voxel).astype(np.int64) + _KEY_OFFSET
    cells = np.clip(cells, 0, (1 << _KEY_BITS) - 1)
    return (cells[:, 0] << (2 * _KEY_BITS)) | (cells[:, 1] << _KEY_BITS) | cells[:, 2]


def _decode_keys(keys: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_voxel_keys`; returns signed integer cells (int32)."""
    mask = (1 << _KEY_BITS) - 1
    cells = np.column_stack((keys >> (2 * _KEY_BITS), (keys >> _KEY_BITS) & mask, keys & mask)).astype(np.int64)
    return (cells - _KEY_OFFSET).astype(np.int32)


def _group_starts(sorted_keys: np.ndarray) -> np.ndarray:
    if not len(sorted_keys):
        return np.empty(0, np.int64)
    return np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]


# ---------------------------------------------------------------------------
# Point statistics (voxel keys, sums, votes) - streaming and mergeable
# ---------------------------------------------------------------------------

def _empty_state() -> dict[str, np.ndarray]:
    return {"keys": np.empty(0, np.int64), "sum": np.empty((0, 3), np.float64), "samples": np.empty(0, np.uint32),
            "views": np.empty(0, np.uint32), "source": np.empty(0, np.uint8),
            "votes": np.empty((0, LABEL_BINS), np.uint16)}


def _or_by_group(target: np.ndarray, inverse: np.ndarray, values: np.ndarray) -> None:
    """Bitwise-OR ``values`` into ``target[inverse]`` without the slow ``ufunc.at`` loop (bit flags < 16)."""
    for bit in (1, 2, 4, 8):
        flagged = (values & bit) > 0
        if flagged.any():
            target[np.unique(inverse[flagged])] |= bit


def _aggregate_batch(
    xyz: np.ndarray, labels: np.ndarray, confidence: np.ndarray, sources: np.ndarray, views: np.ndarray, voxel: float,
) -> dict[str, np.ndarray]:
    """Reduce one bounded batch of samples into per-voxel statistics (vectorised)."""
    if not len(xyz):
        return _empty_state()
    keys = _voxel_keys(xyz, voxel)
    unique, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1).astype(np.int64)
    count = len(unique)
    sums = np.column_stack([np.bincount(inverse, weights=xyz[:, axis], minlength=count) for axis in range(3)])
    samples = np.bincount(inverse, minlength=count).astype(np.uint32)
    source = np.zeros(count, np.uint8)
    _or_by_group(source, inverse, np.asarray(sources, np.uint8))
    # distinct (voxel, view) pairs: a view id never appears in two batches.
    local_views = np.unique(views, return_inverse=True)[1].reshape(-1).astype(np.int64)
    width = int(local_views.max()) + 1
    pair_voxel = np.unique(inverse * width + local_views) // width
    n_views = np.bincount(pair_voxel, minlength=count).astype(np.uint32)
    votes = np.bincount(inverse * LABEL_BINS + labels.astype(np.int64), weights=np.rint(confidence * VOTE_UNITS),
                        minlength=count * LABEL_BINS).reshape(count, LABEL_BINS)
    votes = np.minimum(votes, 65535).astype(np.uint16)
    return {"keys": unique, "sum": sums, "samples": samples, "views": n_views, "source": source, "votes": votes}


def _merge_states(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Merge two per-voxel states built from disjoint view sets (view counts add).

    The inputs are consumed (their arrays are released field by field) to keep the peak memory low.
    """
    if not len(a["keys"]):
        return b
    if not len(b["keys"]):
        return a
    keys = np.concatenate((a.pop("keys"), b.pop("keys")))
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    starts = _group_starts(keys)
    out: dict[str, np.ndarray] = {"keys": keys[starts]}
    del keys
    for name, dtype in (("sum", np.float64), ("samples", np.uint32), ("views", np.uint32), ("votes", np.uint16)):
        merged = np.concatenate((a.pop(name), b.pop(name)))[order]
        reduced = np.add.reduceat(merged, starts, axis=0, dtype=np.float64 if dtype == np.float64 else np.uint32)
        del merged
        out[name] = np.minimum(reduced, 65535).astype(dtype) if dtype == np.uint16 else reduced.astype(dtype, copy=False)
    out["source"] = np.bitwise_or.reduceat(np.concatenate((a.pop("source"), b.pop("source")))[order], starts)
    return out


def _finalise_state(state: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Voxel mean position, winning label, confidence, view count, source flags."""
    samples = np.maximum(state["samples"], 1).astype(np.float64)
    xyz = state["sum"] / samples[:, None]
    labels = np.argmax(state["votes"], axis=1).astype(np.uint8)
    # Confidence = share of the sample confidence mass that backs the winning label
    # (source confidence 0.9 COLMAP / 0.45 Depth-Anything times label agreement).
    confidence = (state["votes"].max(axis=1) / (VOTE_UNITS * samples)).astype(np.float32)
    return xyz, labels, confidence, np.minimum(state["views"], 65535).astype(np.uint16), state["source"]


# ---------------------------------------------------------------------------
# Free space (vectorised ray marching)
# ---------------------------------------------------------------------------

def _free_space_records(
    camera: np.ndarray, points: np.ndarray, voxel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """March rays camera -> point in ``voxel`` steps; return (pass keys, counts, end keys, counts).

    The ray stops one voxel before the surface so an observed surface is never
    counted as seen-empty.  The pass count is a ray-sample count (a diagonal ray
    can hit a cell twice), i.e. an approximate number of rays.
    """
    camera = np.asarray(camera, np.float64)
    points = np.asarray(points, np.float64)
    empty = np.empty(0, np.int64)
    if not len(points):
        return empty, empty, empty, empty
    delta = points - camera
    distance = np.linalg.norm(delta, axis=1)
    steps = np.maximum(np.floor((distance - 1.5 * voxel) / voxel), 0).astype(np.int64)
    end_keys, ends = np.unique(_voxel_keys(points, voxel), return_counts=True)
    total = int(steps.sum())
    if total == 0:
        return empty, empty, end_keys, ends
    ray = np.repeat(np.arange(len(points)), steps)
    first = np.cumsum(steps) - steps
    step_index = np.arange(total) - np.repeat(first, steps) + 1
    direction = delta / np.maximum(distance, 1e-9)[:, None]
    positions = camera + direction[ray] * (step_index[:, None] * voxel)
    keys, through = np.unique(_voxel_keys(positions, voxel), return_counts=True)
    return keys, through, end_keys, ends


def _combine_ray_batches(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold per-view (pass keys, pass counts, end keys, end counts) into sorted (keys, through, ending)."""
    empty = np.empty(0, np.int64)
    if not parts:
        return empty, empty, empty
    pass_keys = np.concatenate([part[0] for part in parts])
    pass_counts = np.concatenate([part[1] for part in parts]).astype(np.int64)
    end_keys = np.concatenate([part[2] for part in parts])
    end_counts = np.concatenate([part[3] for part in parts]).astype(np.int64)
    keys = np.concatenate((pass_keys, end_keys))
    if not len(keys):
        return empty, empty, empty
    through = np.concatenate((pass_counts, np.zeros(len(end_keys), np.int64)))
    ending = np.concatenate((np.zeros(len(pass_keys), np.int64), end_counts))
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    starts = _group_starts(keys)
    return keys[starts], np.add.reduceat(through[order], starts), np.add.reduceat(ending[order], starts)


def _merge_counts(
    a: tuple[np.ndarray, np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge two sorted (keys, through, ending) triples."""
    if not len(a[0]):
        return b
    if not len(b[0]):
        return a
    keys = np.concatenate((a[0], b[0]))
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    starts = _group_starts(keys)
    through = np.add.reduceat(np.concatenate((a[1], b[1]))[order], starts)
    ending = np.add.reduceat(np.concatenate((a[2], b[2]))[order], starts)
    return keys[starts], through, ending


# ---------------------------------------------------------------------------
# Per-view sampling
# ---------------------------------------------------------------------------

def _depth_edges(depth: np.ndarray, rel: float) -> np.ndarray:
    """True where depth jumps by more than ``rel`` (relative) inside a 5x5 window (edges, grazing surfaces)."""
    import cv2

    valid = depth > 0
    kernel = np.ones((5, 5), np.uint8)
    high = cv2.dilate(np.where(valid, depth, 0).astype(np.float32), kernel)
    low = cv2.erode(np.where(valid, depth, np.float32(1e6)).astype(np.float32), kernel)
    return valid & ((high - low) > rel * np.maximum(depth, 1e-3))


def _sample_view(dense: Path, view, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Back-project one view; returns samples for the point product and rays for free space."""
    geometric_path = _depth_path(dense, view.name, "geometric")
    prior_path = dense / "depth_prior" / f"{view.name}.npy"
    geometric = read_colmap_array(geometric_path) if geometric_path.is_file() else None
    prior = np.load(prior_path).astype(np.float32) if prior_path.is_file() else None
    if geometric is None and prior is None:
        return None
    shape = geometric.shape if geometric is not None else prior.shape
    if prior is not None and prior.shape != shape:
        import cv2

        prior = cv2.resize(prior, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    depth = np.zeros(shape, np.float32)
    source = np.zeros(shape, np.uint8)
    if geometric is not None:
        valid = (geometric > 0) & (geometric <= float(cfg["max_geometric_depth_m"]))
        depth[valid], source[valid] = geometric[valid], SOURCE_COLMAP
    agreement: np.ndarray | None = None
    if prior is not None:
        if geometric is not None:
            both = (geometric > 0) & (prior > 0)
            if both.any():  # independent of the cloud: aligned prior vs COLMAP at the same pixel
                agreement = np.abs(prior[both][::97] - geometric[both][::97])
        valid = (depth <= 0) & (prior > 0) & (prior <= float(cfg["max_prior_depth_m"]))
        depth[valid], source[valid] = prior[valid], SOURCE_DEPTH_ANYTHING
    depth[_read_mirror(dense, view.name, shape)] = 0.0
    depth[_depth_edges(depth, float(cfg["depth_edge_rel"]))] = 0.0
    labels_full = _read_labels(dense, view.name, shape)
    stride = max(1, int(cfg["stride"]))
    rows, cols = np.mgrid[0:shape[0]:stride, 0:shape[1]:stride]
    rows, cols = rows.ravel(), cols.ravel()
    label = labels_full[rows, cols]
    keep = (depth[rows, cols] > 0) & (label > 0) & (label < LABEL_BINS)
    rows, cols = rows[keep], cols[keep]
    points = _world_points(view, rows, cols, depth).astype(np.float32) if len(rows) else np.empty((0, 3), np.float32)
    selected_source = source[rows, cols]
    # free-space rays use every valid pixel (also unlabelled ones) on a coarse grid
    fstride = max(1, int(cfg["free_ray_stride"]))
    frows, fcols = np.mgrid[0:shape[0]:fstride, 0:shape[1]:fstride]
    frows, fcols = frows.ravel(), fcols.ravel()
    fkeep = depth[frows, fcols] > 0
    frows, fcols = frows[fkeep], fcols[fkeep]
    ray_points = _world_points(view, frows, fcols, depth) if len(frows) else np.empty((0, 3))
    return {
        "xyz": points, "labels": labels_full[rows, cols], "source": selected_source,
        "confidence": np.where(selected_source == SOURCE_COLMAP, 0.9, 0.45).astype(np.float32),
        "view": np.full(len(points), view.index, np.int64), "camera": np.asarray(view.center, np.float64),
        "ray_points": ray_points, "agreement": agreement,
    }


# ---------------------------------------------------------------------------
# Comparison with the classified cloud
# ---------------------------------------------------------------------------

def _load_cloud(classified: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if classified is None or not Path(classified).is_file():
        return None
    vertex = read_vertex_array(classified)
    if "scalar_class" not in vertex.dtype.names:
        return None
    cloud = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    return cloud, np.asarray(vertex["scalar_class"], dtype=np.uint8)


def _alignment(
    xyz: np.ndarray, labels: np.ndarray, cloud: tuple[np.ndarray, np.ndarray] | None, maximum: float,
    sample: int = 400000, sources: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Estimate (and apply) the median same-class offset to the classified cloud.

    Returns (xyz, offset norm, detail).  ``detail`` holds the median nearest-cloud
    distance per class and per source.  It is a nearest-neighbour statistic and,
    because the cloud was fused from the same depth maps, an optimistic one.
    """
    if cloud is None or not len(xyz):
        return xyz, float("nan"), {}
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return xyz, float("nan"), {}
    cloud_xyz, cloud_labels = cloud
    rng = np.random.default_rng(0)
    take = np.arange(len(xyz)) if len(xyz) <= sample else np.sort(rng.choice(len(xyz), sample, replace=False))
    residuals: list[np.ndarray] = []
    per_class: dict[str, float] = {}
    per_source: dict[str, list[np.ndarray]] = {}
    for label in np.unique(labels[take]):
        target = cloud_xyz[cloud_labels == label]
        picked = take[labels[take] == label]
        if len(target) < 8 or not len(picked):
            continue
        distances, nearest = cKDTree(target).query(xyz[picked], k=1, workers=-1)
        close = distances <= maximum
        if close.any():
            residuals.append((target[nearest] - xyz[picked])[close])
            per_class[str(int(label))] = float(np.median(distances[close]))
        if sources is not None:
            for flag in np.unique(sources[picked]):
                per_source.setdefault(str(int(flag)), []).append(distances[sources[picked] == flag])
    detail: dict[str, Any] = {
        "median_nn_distance_per_class_m": per_class,
        "median_nn_distance_per_source_m": {key: float(np.median(np.concatenate(value))) for key, value in per_source.items()},
    }
    if not residuals:
        return xyz, float("nan"), detail
    offset = np.median(np.concatenate(residuals), axis=0)
    norm = float(np.linalg.norm(offset))
    detail["median_offset_xyz_m"] = [float(value) for value in offset]
    if norm > maximum:
        return xyz, norm, detail
    return xyz + offset, norm, detail


def _gap_fill_mask(xyz: np.ndarray, cloud: tuple[np.ndarray, np.ndarray] | None, radius: float) -> np.ndarray:
    """True where no classified-cloud point lies within ``radius`` (the product adds information)."""
    if cloud is None or not len(xyz):
        return np.ones(len(xyz), bool)
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(cloud[0]).query(xyz, k=1, distance_upper_bound=radius, workers=-1)
    return ~np.isfinite(distances)


def _count_per_class(labels: np.ndarray, mask: np.ndarray | None = None) -> dict[str, int]:
    selected = labels if mask is None else labels[mask]
    return {str(int(key)): int(np.count_nonzero(selected == key)) for key in np.unique(selected)}


# ---------------------------------------------------------------------------
# Product builder
# ---------------------------------------------------------------------------

def build_depth_product(
    project_dir: Path | str, *, dense_dir: Path | str | None = None, classified_path: Path | str | None = None,
    settings: dict[str, Any] | None = None, progress=None,
) -> dict[str, Any]:
    """Build ``products/depth`` from the current dense workspace or its cache.

    It deliberately returns a report instead of raising for ordinary missing
    inputs so a pipeline run remains successful when depth is unavailable.
    """
    project_dir = Path(project_dir)
    cfg = load_settings(settings)
    started = time.perf_counter()
    dense = Path(dense_dir) if dense_dir is not None else _dense_dir(project_dir)
    if not bool(cfg["enabled"]) or dense is None:
        return {"enabled": bool(cfg["enabled"]), "written": False, "reason": "depth maps missing"}
    if not (dense / "sparse").is_dir():
        return {"enabled": True, "written": False, "reason": "sparse cameras missing"}
    views = load_views(dense / "sparse")
    # Sparse registration can include images excluded from PatchMatch.  Use only
    # views that retained a depth source.
    views = [view for view in views if _depth_path(dense, view.name, "geometric").is_file()
             or (dense / "depth_prior" / f"{view.name}.npy").is_file()]
    max_views = int(cfg["max_views"])
    if max_views > 0 and len(views) > max_views:
        indices = np.linspace(0, len(views) - 1, max_views, dtype=int)
        views = [views[index] for index in indices]
    voxel, free_voxel = float(cfg["voxel_m"]), float(cfg["free_space_voxel_m"])
    batch_size = max(1, int(cfg["batch_views"]))
    state = _empty_state()
    free_state = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
    views_used, samples_total = 0, 0
    agreement: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=max(1, int(cfg["workers"]))) as pool:
        for start in range(0, len(views), batch_size):
            batch = views[start:start + batch_size]
            results = [item for item in pool.map(lambda view: _sample_view(dense, view, cfg), batch) if item is not None]
            if results:
                views_used += len(results)
                xyz = np.concatenate([item["xyz"] for item in results])
                samples_total += len(xyz)
                state = _merge_states(state, _aggregate_batch(
                    xyz, np.concatenate([item["labels"] for item in results]),
                    np.concatenate([item["confidence"] for item in results]),
                    np.concatenate([item["source"] for item in results]),
                    np.concatenate([item["view"] for item in results]), voxel,
                ))
                records = [_free_space_records(item["camera"], item["ray_points"], free_voxel)
                           for item in results if len(item["ray_points"])]
                free_state = _merge_counts(free_state, _combine_ray_batches(records))
                agreement.extend(item["agreement"] for item in results if item["agreement"] is not None)
            if progress is not None:
                done = min(start + len(batch), len(views))
                progress(done / len(views), f"Depth evidence {done}/{len(views)}")
    if not len(state["keys"]):
        return {"enabled": True, "written": False, "reason": "no readable depth samples"}
    xyz, labels, confidence, n_views, sources = _finalise_state(state)
    del state
    # A prior-only surface is retained only after independent views agree;
    # geometric COLMAP samples remain valid on their own.
    keep = (sources != SOURCE_DEPTH_ANYTHING) | (n_views >= int(cfg["min_prior_views"]))
    xyz, labels, confidence, n_views, sources = (value[keep] for value in (xyz, labels, confidence, n_views, sources))
    if not len(xyz):
        return {"enabled": True, "written": False, "reason": "no voxel passed the view-agreement rule"}
    classified = Path(classified_path) if classified_path is not None else project_dir / "products" / "classified" / "classified.ply"
    cloud = _load_cloud(classified)
    xyz, residual, alignment_detail = _alignment(
        xyz, labels, cloud, float(cfg["alignment_max_residual_m"]), int(cfg["alignment_sample"]), sources)
    xyz = xyz.astype(np.float32)
    gap = _gap_fill_mask(xyz, cloud, float(cfg["gap_radius_m"]))
    free_keys, through, ending = free_state
    product = _product_dir(project_dir)
    product.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(product / "depth_points.npz", xyz=xyz, class_id=labels, confidence=confidence,
                        n_views=n_views, source=sources, gap_fill=gap)
    np.savez_compressed(product / "free_space.npz", origin=np.zeros(3, np.float32), voxel_m=np.float32(free_voxel),
                        indices=_decode_keys(free_keys), through=np.minimum(through, 65535).astype(np.uint16),
                        ending=np.minimum(ending, 65535).astype(np.uint16))
    point_file = product / "depth_points.npz"
    agree = np.concatenate(agreement) if agreement else np.empty(0)
    multi = n_views >= int(cfg["min_label_views"])
    report = {
        "enabled": True, "written": True, "views_used": views_used, "views_total": len(views), "points": int(len(xyz)),
        "samples": int(samples_total), "free_space_voxels": int(len(free_keys)),
        "counts_per_class": _count_per_class(labels),
        "counts_per_source": {"colmap": int(np.count_nonzero(sources == SOURCE_COLMAP)),
                              "depth_anything": int(np.count_nonzero(sources == SOURCE_DEPTH_ANYTHING)),
                              "both": int(np.count_nonzero(sources == (SOURCE_COLMAP | SOURCE_DEPTH_ANYTHING)))},
        "points_multi_view": int(np.count_nonzero(multi)),
        "gap_fill": {"radius_m": float(cfg["gap_radius_m"]), "points": int(np.count_nonzero(gap)),
                     "points_multi_view": int(np.count_nonzero(gap & multi)),
                     "per_class": _count_per_class(labels, gap),
                     "per_class_multi_view": _count_per_class(labels, gap & multi)} if cloud is not None else None,
        "alignment_residual_m": None if not np.isfinite(residual) else residual,
        "alignment": alignment_detail,
        "depth_anything_vs_colmap_abs_depth_diff_m": (
            {"median": float(np.median(agree)), "p90": float(np.percentile(agree, 90)), "pixels": int(len(agree))}
            if len(agree) else None),
        "settings": cfg,
        "size_bytes": int(point_file.stat().st_size + (product / "free_space.npz").stat().st_size),
        "runtime_s": round(time.perf_counter() - started, 3),
    }
    (product / "depth_evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


class DepthEvidence:
    """Fast region queries over ``products/depth`` in either cloud or plan frame."""

    def __init__(self, product: Path, frame=None) -> None:
        with np.load(product / "depth_points.npz") as archive:
            self.xyz = archive["xyz"].astype(np.float32)
            self.class_id = archive["class_id"].astype(np.uint8)
            self.confidence = archive["confidence"].astype(np.float32)
            self.n_views = archive["n_views"].astype(np.uint16)
            self.source = archive["source"].astype(np.uint8)
            # True where the classified cloud had no point within gap_radius_m at build time
            self.gap_fill = archive["gap_fill"].astype(bool) if "gap_fill" in archive.files else np.zeros(len(self.xyz), bool)
        with np.load(product / "free_space.npz") as archive:
            self.free_origin = archive["origin"].astype(np.float32)
            self.free_voxel = float(archive["voxel_m"])
            self.free_indices = archive["indices"].astype(np.int32)
            self.free_through = archive["through"].astype(np.uint16)
            self.free_ending = archive["ending"].astype(np.uint16)
        self.frame = frame
        self._plan_cache: np.ndarray | None = None

    @classmethod
    def open(cls, project_dir: Path | str, plan_frame=None) -> "DepthEvidence | None":
        product = _product_dir(Path(project_dir))
        if (product / "depth_points.npz").is_file() and (product / "free_space.npz").is_file():
            return cls(product, plan_frame)
        # Existing projects may still have the raw cache but not the compact
        # product.  Materialise the same product once rather than exposing a
        # second, subtly different query implementation.
        if _dense_dir(Path(project_dir)) is not None:
            report = build_depth_product(project_dir)
            if report.get("written"):
                return cls(product, plan_frame)
        return None

    def _to_plan(self, points: np.ndarray) -> np.ndarray:
        return points if self.frame is None else self.frame.to_local(points)

    @staticmethod
    def _inside(points: np.ndarray, polygon: Iterable[Iterable[float]]) -> np.ndarray:
        ring = np.asarray(list(polygon), dtype=float)
        x, y = points[:, 0], points[:, 1]
        inside = np.zeros(len(points), dtype=bool)
        previous = len(ring) - 1
        for current in range(len(ring)):
            xi, yi, xj, yj = ring[current, 0], ring[current, 1], ring[previous, 0], ring[previous, 1]
            inside ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
            previous = current
        return inside

    def plan_points(self) -> np.ndarray:
        """All product points in the plan frame (computed once, then cached)."""
        if self._plan_cache is None:
            self._plan_cache = np.asarray(self._to_plan(self.xyz), dtype=np.float32)
        return self._plan_cache

    def select(self, classes=None, *, gap_only: bool = False, min_views: int = 1, min_confidence: float = 0.0,
               colmap_only: bool = False, max_views=None) -> np.ndarray:
        """Boolean mask over all product voxels (no region): class / gap-fill / support / source filters.

        ``colmap_only`` keeps voxels with the COLMAP source bit (``source & 1``): only those carry trustworthy
        edge, face and plane positions; Depth-Anything-only voxels are occupancy evidence at best.
        """
        mask = np.ones(len(self.xyz), bool)
        if classes is not None:
            mask &= np.isin(self.class_id, list(classes))
        if max_views is not None:
            mask &= self.n_views <= int(max_views)
        if gap_only:
            mask &= self.gap_fill
        if min_views > 1:
            mask &= self.n_views >= int(min_views)
        if min_confidence > 0.0:
            mask &= self.confidence >= float(min_confidence)
        if colmap_only:
            mask &= (self.source & SOURCE_COLMAP) > 0
        return mask

    def points(self, region_xy_polygon, z_range, classes=None, max_views=None, *, gap_only: bool = False,
               min_views: int = 1, min_confidence: float = 0.0, colmap_only: bool = False, with_source: bool = False):
        """Product points inside a plan region: (xyz_plan, class_id, confidence, n_views).

        ``gap_only`` keeps only voxels the classified cloud did not cover;
        ``min_views`` / ``min_confidence`` drop weakly supported voxels; ``colmap_only`` keeps COLMAP-flagged voxels;
        ``with_source`` appends the per-voxel ``source`` bit flags (1 = COLMAP, 2 = Depth-Anything) as a fifth item.
        """
        points = self.plan_points()
        ring = np.asarray(list(region_xy_polygon), float)
        box = (points[:, 0] >= ring[:, 0].min()) & (points[:, 0] <= ring[:, 0].max()) \
            & (points[:, 1] >= ring[:, 1].min()) & (points[:, 1] <= ring[:, 1].max()) \
            & (points[:, 2] >= float(z_range[0])) & (points[:, 2] <= float(z_range[1]))
        index = np.flatnonzero(box)
        index = index[self._inside(points[index, :2], ring)]
        mask = np.zeros(len(points), bool)
        mask[index] = True
        mask &= self.select(classes, gap_only=gap_only, min_views=min_views, min_confidence=min_confidence,
                            colmap_only=colmap_only, max_views=max_views)
        result = (points[mask], self.class_id[mask], self.confidence[mask], self.n_views[mask])
        return (*result, self.source[mask]) if with_source else result

    def free_space(self, region_xy_polygon, z_range, min_through: int = 2):
        cloud = self.free_origin + (self.free_indices.astype(np.float32) + 0.5) * self.free_voxel
        points = self._to_plan(cloud)
        mask = self._inside(points[:, :2], region_xy_polygon)
        mask &= (points[:, 2] >= float(z_range[0])) & (points[:, 2] <= float(z_range[1]))
        mask &= (self.free_through >= int(min_through)) & (self.free_through > self.free_ending)
        return points[mask], self.free_through[mask], self.free_ending[mask]

    def cache_key(self, region_xy_polygon, z_range, settings: dict[str, Any] | None = None) -> str:
        payload = json.dumps({"region": list(region_xy_polygon), "z": list(z_range), "settings": settings or {}}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
