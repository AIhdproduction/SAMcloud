"""Geometry and colour-aware semantic voting inside point-cloud superpoints."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from samcloud.engine.ply_io import read_classified_ply, write_classified_ply


DEFAULTS = {
    "enabled": True,
    "normal_k": 16,
    "graph_k": 10,
    "max_edge_m": 0.06,
    "color_weight": 0.5,
    "k_thresh": 0.05,
    "min_segment": 30,
    "min_vote_points": 30,
    "majority_share": 0.6,
    "protect_min_points": 200,
    "relabel_classes": ["unknown", "construction material", "wall", "ceiling", "floor", "door", "window", "furniture"],
    "chunk_size": 100000,
}
_SUPPRESSED_CLASS = 256


class UnionFind:
    """Union-find for the Felzenszwalb criterion, using compact NumPy arrays."""

    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)
        self.internal = np.zeros(count, dtype=np.float32)

    def find(self, item: int) -> int:
        parent = self.parent
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            next_item = int(parent[item])
            parent[item] = root
            item = next_item
        return root

    def join(self, left: int, right: int, edge_weight: float) -> int:
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        self.internal[left] = max(self.internal[left], self.internal[right], edge_weight)
        return left

    def roots(self) -> np.ndarray:
        """Fully compress parents without a Python loop over points."""
        parent = self.parent
        while True:
            compressed = parent[parent]
            if np.array_equal(compressed, parent):
                return parent.copy()
            parent[:] = compressed


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _seconds_since(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def estimate_normals(
    xyz: np.ndarray, normal_k: int, chunk_size: int, progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, cKDTree]:
    """Estimate unoriented PCA normals in vectorised query chunks."""
    point_count = len(xyz)
    if point_count < 4:
        raise ValueError("At least four points are required to estimate normals.")
    effective_k = min(normal_k, point_count - 1)
    if effective_k < 3:
        raise ValueError("At least three neighbours are required to estimate normals.")
    tree = cKDTree(xyz)
    normals = np.empty((point_count, 3), dtype=np.float64)
    chunk_count = (point_count + chunk_size - 1) // chunk_size
    for chunk_index, start in enumerate(range(0, point_count, chunk_size), start=1):
        stop = min(start + chunk_size, point_count)
        _, neighbour_ids = tree.query(xyz[start:stop], k=effective_k + 1, workers=-1)
        neighbours = xyz[neighbour_ids[:, 1:]]
        centred = neighbours - neighbours.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centred, centred, optimize=True) / effective_k
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, :, 0]
        normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-12)[:, None]
        normals[start:stop] = normal
        _report(progress, f"Normals: chunk {chunk_index}/{chunk_count}")
    return normals, tree


def build_graph(
    xyz: np.ndarray, rgb: np.ndarray, normals: np.ndarray, tree: cKDTree, graph_k: int,
    max_edge_m: float, color_weight: float, chunk_size: int, progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build unique undirected weighted kNN edges, retaining only short edges."""
    point_count = len(xyz)
    effective_k = min(graph_k, point_count - 1)
    if effective_k < 1:
        raise ValueError("At least two points are required to build a graph.")
    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    chunk_count = (point_count + chunk_size - 1) // chunk_size
    colour_scale = 255.0 * np.sqrt(3.0)
    for chunk_index, start in enumerate(range(0, point_count, chunk_size), start=1):
        stop = min(start + chunk_size, point_count)
        distances, neighbours = tree.query(xyz[start:stop], k=effective_k + 1, workers=-1)
        source = np.arange(start, stop, dtype=np.int32)[:, None]
        target = neighbours[:, 1:].astype(np.int32, copy=False)
        source_full = np.broadcast_to(source, target.shape)
        keep = (source_full < target) & (distances[:, 1:] <= max_edge_m)
        if np.any(keep):
            left = source_full[keep].astype(np.int32, copy=True)
            right = target[keep].astype(np.int32, copy=True)
            normal_dot = np.abs(np.einsum("ij,ij->i", normals[left], normals[right], optimize=True))
            normal_term = 1.0 - np.clip(normal_dot, 0.0, 1.0)
            colour_term = np.linalg.norm(rgb[left].astype(np.float64) - rgb[right].astype(np.float64), axis=1) / colour_scale
            left_chunks.append(left)
            right_chunks.append(right)
            weight_chunks.append((normal_term + color_weight * colour_term).astype(np.float32))
        _report(progress, f"Graph: chunk {chunk_index}/{chunk_count}")
    if not left_chunks:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    return np.concatenate(left_chunks), np.concatenate(right_chunks), np.concatenate(weight_chunks)


def segment_graph(
    point_count: int, left: np.ndarray, right: np.ndarray, weights: np.ndarray, k_thresh: float,
    min_segment: int, progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Felzenszwalb segmentation followed by its standard small-component pass."""
    _report(progress, f"Segmentation: sorting {len(weights):,} graph edges")
    order = np.argsort(weights, kind="mergesort")
    sorted_left, sorted_right, sorted_weights = left[order], right[order], weights[order]
    uf = UnionFind(point_count)
    _report(progress, "Segmentation: merging by adaptive threshold")
    for source, target, weight in zip(sorted_left, sorted_right, sorted_weights):
        root_source, root_target = uf.find(int(source)), uf.find(int(target))
        if root_source != root_target and (
            weight <= uf.internal[root_source] + k_thresh / uf.size[root_source]
            and weight <= uf.internal[root_target] + k_thresh / uf.size[root_target]
        ):
            uf.join(root_source, root_target, float(weight))
    _report(progress, "Segmentation: merging small components")
    for source, target, weight in zip(sorted_left, sorted_right, sorted_weights):
        root_source, root_target = uf.find(int(source)), uf.find(int(target))
        if root_source != root_target and (uf.size[root_source] < min_segment or uf.size[root_target] < min_segment):
            uf.join(root_source, root_target, float(weight))
    _, segment_ids = np.unique(uf.roots(), return_inverse=True)
    segment_ids = segment_ids.astype(np.int32, copy=False)
    return segment_ids, np.bincount(segment_ids)


def resolve_relabel_class_ids(class_names: list[str], names: list[str] | tuple[str, ...]) -> set[int]:
    """Resolve configured class names to the stable one-based ids of this run."""
    ids = {"unknown": 0}
    ids.update({entry.split(",", 1)[0].strip().lower(): index for index, entry in enumerate(class_names, start=1)})
    return {ids[name.strip().lower()] for name in names if name.strip().lower() in ids}


def vote_classes(
    class_ids: np.ndarray, segment_ids: np.ndarray, segment_sizes: np.ndarray, min_vote_points: int,
    majority_share: float, relabel_classes: set[int], protect_min_points: int,
    suppressed: np.ndarray | None = None, protected_clutter: np.ndarray | None = None,
) -> np.ndarray:
    """Apply constrained voting while preserving fine classes and intentional Unknown points."""
    class_ids = np.asarray(class_ids, dtype=np.uint8)
    suppressed = np.zeros(len(class_ids), dtype=bool) if suppressed is None else np.asarray(suppressed, dtype=bool)
    if len(suppressed) != len(class_ids):
        raise ValueError("suppressed must have one value per point")
    protected_clutter = np.zeros(len(class_ids), dtype=bool) if protected_clutter is None else np.asarray(protected_clutter, dtype=bool)
    if len(protected_clutter) != len(class_ids):
        raise ValueError("protected_clutter must have one value per point")
    segment_count = len(segment_sizes)
    result = class_ids.copy()
    result[suppressed] = 0
    voting_ids = class_ids.astype(np.uint16)
    voting_ids[suppressed] = _SUPPRESSED_CLASS
    valid_label = (voting_ids != 0) & (voting_ids != 255)
    labelled_counts = np.bincount(segment_ids[valid_label], minlength=segment_count)
    majority_counts = np.zeros(segment_count, dtype=np.int32)
    majority_class = np.zeros(segment_count, dtype=np.uint16)
    own_count = np.zeros(len(class_ids), dtype=np.int32)
    for class_id in np.unique(voting_ids[valid_label]):
        mask = voting_ids == class_id
        counts = np.bincount(segment_ids[mask], minlength=segment_count)
        better = counts > majority_counts
        majority_counts[better] = counts[better]
        majority_class[better] = class_id
        if class_id != _SUPPRESSED_CLASS:
            own_count[mask] = counts[segment_ids[mask]]
    share = np.divide(majority_counts, labelled_counts, out=np.zeros(segment_count, dtype=np.float64), where=labelled_counts > 0)
    eligible_segment = (segment_sizes >= min_vote_points) & (share >= majority_share)
    segment_majority = majority_class[segment_ids]
    can_relabel = np.isin(class_ids, np.fromiter(sorted(relabel_classes), dtype=np.uint8)) & ~suppressed & ~protected_clutter
    protected = (class_ids != 0) & (class_ids != 255) & (own_count >= protect_min_points)
    change = eligible_segment[segment_ids] & can_relabel & (class_ids != segment_majority) & ~protected
    result[change] = np.where(segment_majority[change] == _SUPPRESSED_CLASS, 0, segment_majority[change]).astype(np.uint8)
    result[class_ids == 255] = 255
    result[suppressed] = 0
    result[protected_clutter] = class_ids[protected_clutter]
    return result


def _class_count_map(class_ids: np.ndarray, class_names: list[str]) -> dict[str, int]:
    ids = [0, *range(1, len(class_names) + 1), 255]
    return {str(class_id): int(np.count_nonzero(class_ids == class_id)) for class_id in ids}


def _change_matrix(before: np.ndarray, after: np.ndarray) -> list[dict[str, int]]:
    changed = before != after
    if not np.any(changed):
        return []
    pairs = before[changed].astype(np.int32) * 256 + after[changed].astype(np.int32)
    values, counts = np.unique(pairs, return_counts=True)
    rows = [{"from": int(value // 256), "to": int(value % 256), "count": int(count)} for value, count in zip(values, counts)]
    return sorted(rows, key=lambda row: (-row["count"], row["from"], row["to"]))


def vote_classified_cloud(
    classified_ply: Path | str, settings: dict[str, Any] | None, class_names: list[str], *,
    scores_path: Path | str | None = None, report_dir: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Vote classes in-place, retaining point order, coordinates and RGB values."""
    settings = {**DEFAULTS, **(settings or {})}
    classified_ply = Path(classified_ply)
    if not bool(settings["enabled"]):
        return {"enabled": False}
    started = time.perf_counter()
    read_started = time.perf_counter()
    xyz, rgb, original_classes = read_classified_ply(classified_ply)
    timings: dict[str, float] = {"read": _seconds_since(read_started)}
    if len(xyz) < 4:
        raise ValueError("At least four points are required for superpoint voting.")
    suppressed = np.zeros(len(xyz), dtype=bool)
    protected_clutter = np.zeros(len(xyz), dtype=bool)
    scores_file = Path(scores_path) if scores_path is not None else None
    score_arrays: dict[str, np.ndarray] | None = None
    if scores_file is not None and scores_file.is_file():
        with np.load(scores_file) as archive:
            score_arrays = {name: archive[name] for name in archive.files}
        if not all(len(values) == len(xyz) for values in score_arrays.values()):
            score_arrays = None
        elif "suppressed" in score_arrays:
            suppressed = np.asarray(score_arrays["suppressed"], dtype=bool)
        if score_arrays is not None and "protected_clutter" in score_arrays:
            protected_clutter = np.asarray(score_arrays["protected_clutter"], dtype=bool)
    geometry = np.asarray(xyz, dtype=np.float64)
    normal_started = time.perf_counter()
    _report(progress, f"Normals: estimating PCA normals for {len(geometry):,} points")
    normals, tree = estimate_normals(geometry, int(settings["normal_k"]), int(settings["chunk_size"]), progress)
    timings["normals"] = _seconds_since(normal_started)
    graph_started = time.perf_counter()
    _report(progress, "Graph: building unique kNN edges")
    left, right, weights = build_graph(
        geometry, rgb, normals, tree, int(settings["graph_k"]), float(settings["max_edge_m"]),
        float(settings["color_weight"]), int(settings["chunk_size"]), progress,
    )
    timings["graph"] = _seconds_since(graph_started)
    _report(progress, f"Graph: retained {len(weights):,} edges")
    segment_started = time.perf_counter()
    segment_ids, segment_sizes = segment_graph(
        len(geometry), left, right, weights, float(settings["k_thresh"]), int(settings["min_segment"]), progress,
    )
    timings["segmentation"] = _seconds_since(segment_started)
    _report(progress, f"Segmentation: produced {len(segment_sizes):,} superpoints")
    voting_started = time.perf_counter()
    relabel_names = list(settings.get("relabel_classes") or [])
    relabel_classes = resolve_relabel_class_ids(class_names, relabel_names)
    new_classes = vote_classes(
        original_classes, segment_ids, segment_sizes, int(settings["min_vote_points"]),
        float(settings["majority_share"]), relabel_classes, int(settings["protect_min_points"]), suppressed,
        protected_clutter,
    )
    timings["voting"] = _seconds_since(voting_started)
    changed = original_classes != new_classes
    changed_count = int(changed.sum())
    _report(progress, f"Voting: changed {changed_count:,} points ({changed_count / len(new_classes) * 100:.3f}%)")
    write_started = time.perf_counter()
    handle, temporary_name = tempfile.mkstemp(prefix=f".{classified_ply.stem}.superpoint_", suffix=classified_ply.suffix, dir=classified_ply.parent)
    os.close(handle)
    temporary_path = Path(temporary_name)
    write_classified_ply(temporary_path, xyz, rgb, new_classes)
    os.replace(temporary_path, classified_ply)
    timings["ply_outputs"] = _seconds_since(write_started)
    scores_updated = False
    if score_arrays is not None and "top1_class" in score_arrays:
        score_arrays["top1_class"] = np.asarray(score_arrays["top1_class"], dtype=np.uint8).copy()
        score_arrays["top1_class"][changed] = new_classes[changed]
        np.savez(scores_file, **score_arrays)
        scores_updated = True
    percentiles = np.percentile(segment_sizes, [5, 50, 95]).tolist()
    parameters = {
        "normal_k": int(settings["normal_k"]), "graph_k": int(settings["graph_k"]),
        "max_edge_m": float(settings["max_edge_m"]), "color_weight": float(settings["color_weight"]),
        "k_thresh": float(settings["k_thresh"]), "min_segment": int(settings["min_segment"]),
        "min_vote_points": int(settings["min_vote_points"]), "majority_share": float(settings["majority_share"]),
        "relabel_classes": sorted(relabel_classes), "protect_min_points": int(settings["protect_min_points"]),
        "chunk_size": int(settings["chunk_size"]),
    }
    stats: dict[str, Any] = {
        "input_path": str(classified_ply), "output_path": str(classified_ply),
        "report_dir": str(report_dir) if report_dir is not None else None, "parameters": parameters,
        "point_count": int(len(xyz)), "number_of_superpoints": int(len(segment_sizes)),
        "segment_size_percentiles": {"p5": float(percentiles[0]), "p50": float(percentiles[1]), "p95": float(percentiles[2]), "max": int(segment_sizes.max())},
        "changed_points": {"count": changed_count, "percent": changed_count / len(new_classes) * 100.0},
        "per_class_counts_before": _class_count_map(original_classes, class_names),
        "per_class_counts_after": _class_count_map(new_classes, class_names),
        "change_matrix": _change_matrix(original_classes, new_classes), "runtimes_seconds": timings,
        "suppressed_points": int(suppressed.sum()), "protected_clutter_points": int(protected_clutter.sum()),
        "scores_updated": scores_updated,
    }
    timings["total"] = _seconds_since(started)
    if report_dir is not None:
        destination = Path(report_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "superpoint_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
