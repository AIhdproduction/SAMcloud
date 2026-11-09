"""Refine georeferenced SAM classifications using local geometric evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

if __package__ in {None, ""}:  # Supports ``python samcloud/.../refine.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from samcloud.engine.geometry.features import compute_voxel_features
from samcloud.engine.geometry.neighbors import build_voxel_neighbour_index, voxel_neighbour_histogram
from samcloud.engine.geometry.rules import passes_for_class
from samcloud.engine.geometry.rules.building import flat_roof_passes
from samcloud.engine.geometry.support import build_local_support, height_above_support
from samcloud.engine.geometry.surface import apply_surface_rule, neighbourhood_majority
from samcloud.engine.geometry.terrain import build_dtm, height_above_ground, populated_terrain_cells
from samcloud.engine.ply_io import read_classified_ply, write_classified_ply
from samcloud.classification.profiles import NEVER_CLASSIFIED_ID, flurseg_profile


def class_name_mapping(classes_json: Path, class_set: str, engine: str = "sam3") -> tuple[dict[int, str], dict[str, int]]:
    """Map configured class IDs by their stable first-token names."""
    data = json.loads(classes_json.read_text(encoding="utf-8"))
    if class_set not in data:
        raise ValueError(f"Class set not found in {classes_json}: {class_set}")
    if engine != "flurseg":
        names = {index: str(label).split(",", 1)[0].strip() for index, label in enumerate(data[class_set], start=1)}
        return names, {name: index for index, name in names.items()}
    aliases = load_flurseg_aliases()
    entries = flurseg_profile("outdoor").entries
    names = {entry.model_id + 1: aliases.get(entry.model_name, entry.model_name) or entry.model_name for entry in entries}
    name_to_id: dict[str, int] = {}
    for class_id, name in names.items():
        name_to_id.setdefault(name, class_id)
    return names, name_to_id


def load_flurseg_aliases() -> dict[str, str | None]:
    """Return FlurSeg-to-SAM names used solely for geometry-rule lookup."""
    path = Path(__file__).resolve().parents[3] / "config" / "flurseg.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(name): target for name, target in data.get("class_mapping", {}).items()}


def load_config(path: Path) -> dict:
    """Load the refinement configuration as a mapping."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Geometry configuration must contain a mapping: {path}")
    return data


def load_scores(path: Path, point_count: int) -> dict[str, np.ndarray]:
    """Load and validate the compact SegEarth-OV3 confidence arrays."""
    required = ("top1_class", "top1_conf", "top2_class", "top2_conf", "view_count")
    with np.load(path) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"Score file is missing arrays: {', '.join(missing)}")
        scores = {name: archive[name] for name in required}
    mismatched = [name for name, values in scores.items() if len(values) != point_count]
    if mismatched:
        raise ValueError(
            f"Score file point count does not match input PLY ({point_count}): {', '.join(mismatched)}"
        )
    return scores


def confidence_percentiles(
    class_ids: np.ndarray, confidence: np.ndarray, id_to_name: dict[int, str]
) -> dict[str, dict[str, float]]:
    """Summarise SAM top-one confidence by original assigned class."""
    result = {}
    for class_id, name in id_to_name.items():
        values = confidence[(class_ids == class_id) & np.isfinite(confidence)]
        if len(values):
            p10, p50, p90 = np.percentile(values, [10, 50, 90])
            result[name] = {"p10": float(p10), "p50": float(p50), "p90": float(p90)}
    return result


def _subset(features: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[indices] for name, values in features.items()}


def _curb_adjacency_for_indices(
    class_ids: np.ndarray,
    passed: np.ndarray,
    voxel_coords: np.ndarray | None,
    name_to_id: dict[str, int] | None,
    rules_config: dict,
    query_indices: np.ndarray,
    context: dict | None = None,
) -> np.ndarray:
    """Count valid road/sidewalk points only around requested curb candidates."""
    result = np.zeros(len(query_indices), dtype=np.int32)
    if voxel_coords is None or not name_to_id:
        return result
    curb_config = rules_config.get("curb", {})
    adjacent_names = curb_config.get("adjacent_classes", ["road", "sidewalk"])
    adjacent_ids = [name_to_id[name] for name in adjacent_names if name in name_to_id]
    if not adjacent_ids:
        return result
    class_count = max(name_to_id.values(), default=0)
    radius = int(curb_config.get("adjacent_voxel_radius", 2))
    adjacent_passed = passed & np.isin(class_ids, adjacent_ids)
    voxel_index = None if context is None else context.get("voxel_index")
    if voxel_index is None:
        voxel_index = build_voxel_neighbour_index(
            voxel_coords, class_ids, adjacent_passed, class_count
        )
        if context is not None:
            context["voxel_index"] = voxel_index
    histogram = voxel_neighbour_histogram(
        voxel_coords, class_ids, adjacent_passed, query_indices, class_count, radius, voxel_index
    )
    return histogram[:, adjacent_ids].sum(axis=1)


def _passes_curb_candidates(
    features: dict[str, np.ndarray],
    candidate_indices: np.ndarray,
    class_ids: np.ndarray,
    passed_rules: np.ndarray,
    voxel_coords: np.ndarray | None,
    name_to_id: dict[str, int] | None,
    rules_config: dict,
    context: dict | None = None,
) -> np.ndarray:
    """Evaluate curb geometry after lazily obtaining only its local adjacency."""
    curb_features = dict(features)
    curb_features["adjacent_road_sidewalk"] = _curb_adjacency_for_indices(
        class_ids, passed_rules, voxel_coords, name_to_id, rules_config, candidate_indices, context
    )
    return passes_for_class("curb", curb_features, rules_config)


def evaluate_rules(
    class_ids: np.ndarray,
    features: dict[str, np.ndarray],
    id_to_name: dict[int, str],
    rules_config: dict,
    voxel_coords: np.ndarray | None = None,
    name_to_id: dict[str, int] | None = None,
    protected: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate each current class rule, treating unruled/unknown classes as valid."""
    passed = np.ones(len(class_ids), dtype=bool)
    for class_id, class_name in id_to_name.items():
        if class_name == "curb":
            continue
        indices = np.flatnonzero(class_ids == class_id)
        if len(indices):
            passed[indices] = passes_for_class(class_name, _subset(features, indices), rules_config)
    if protected is not None:
        passed |= protected
    curb_id = (name_to_id or {}).get("curb")
    if curb_id is not None:
        indices = np.flatnonzero(class_ids == curb_id)
        if len(indices):
            passed[indices] = _passes_curb_candidates(
                _subset(features, indices), indices, class_ids, passed, voxel_coords,
                name_to_id, rules_config,
            )
    return passed


def apply_flat_roof_rule(
    class_ids: np.ndarray,
    features: dict[str, np.ndarray],
    name_to_id: dict[str, int],
    config: dict,
) -> tuple[np.ndarray, int]:
    """Convert elevated, planar source labels to building before neighbour control."""
    roof_config = config.get("flat_roof", {})
    building_id = name_to_id.get("building")
    source_ids = [name_to_id[name] for name in roof_config.get("source_classes", []) if name in name_to_id]
    if building_id is None or not source_ids:
        return class_ids.copy(), 0
    convert = np.isin(class_ids, source_ids) & flat_roof_passes(features, roof_config)
    result = class_ids.copy()
    result[convert] = building_id
    return result, int(convert.sum())


def _candidate_passes(
    candidate_ids: np.ndarray,
    features: dict[str, np.ndarray],
    id_to_name: dict[int, str],
    rules_config: dict,
    candidate_indices: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    passed_rules: np.ndarray | None = None,
    voxel_coords: np.ndarray | None = None,
    name_to_id: dict[str, int] | None = None,
    curb_context: dict | None = None,
) -> np.ndarray:
    """Evaluate each candidate class, calculating curb adjacency only if needed."""
    passed = np.ones(len(candidate_ids), dtype=bool)
    for class_id in np.unique(candidate_ids):
        if class_id == 0:
            continue
        indices = np.flatnonzero(candidate_ids == class_id)
        name = id_to_name.get(int(class_id))
        if name is not None:
            local_features = _subset(features, indices)
            if (
                name == "curb" and candidate_indices is not None and class_ids is not None
                and passed_rules is not None
            ):
                passed[indices] = _passes_curb_candidates(
                    local_features, candidate_indices[indices], class_ids, passed_rules,
                    voxel_coords, name_to_id, rules_config, curb_context,
                )
            else:
                passed[indices] = passes_for_class(name, local_features, rules_config)
    return passed


def neighbour_reassign(
    class_ids: np.ndarray,
    failed: np.ndarray,
    passed_rules: np.ndarray,
    features: dict[str, np.ndarray],
    voxel_coords: np.ndarray,
    id_to_name: dict[int, str],
    rules_config: dict,
    neighbour_config: dict,
    second_classes: np.ndarray | None = None,
    second_confidences: np.ndarray | None = None,
    confidence_config: dict | None = None,
    batch_size: int = 250_000,
    return_details: bool = False,
) -> tuple[np.ndarray, Counter, int] | tuple[np.ndarray, Counter, int, Counter]:
    """Reassign invalid points via SAM second choice, neighbours, then unknown."""
    result = class_ids.copy()
    targets: Counter = Counter()
    unknown = 0
    resolution = Counter()
    query_all = np.flatnonzero(failed)
    class_count = max(id_to_name, default=0)
    candidate_name_to_id = {name: class_id for class_id, name in id_to_name.items()}
    min_share = float(neighbour_config.get("min_share", 0.0))
    radius = int(neighbour_config.get("voxel_radius", 1))
    curb_context: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"voxel_index": None}
    neighbour_index = build_voxel_neighbour_index(
        voxel_coords, class_ids, passed_rules, class_count
    )
    for start in range(0, len(query_all), batch_size):
        query = query_all[start:start + batch_size]
        assigned = np.zeros(len(query), dtype=bool)
        local_features = _subset(features, query)
        if second_classes is not None and second_confidences is not None:
            second = second_classes[query]
            threshold = float((confidence_config or {}).get("second_best_min", np.inf))
            eligible = (second > 0) & (second <= class_count) & (second_confidences[query] >= threshold)
            if eligible.any():
                positions = np.flatnonzero(eligible)
                candidate_ok = _candidate_passes(
                    second[eligible], _subset(local_features, positions), id_to_name, rules_config,
                    query[positions], class_ids, passed_rules, voxel_coords, candidate_name_to_id, curb_context,
                )
                accepted = positions[candidate_ok]
                if len(accepted):
                    result[query[accepted]] = second[accepted]
                    assigned[accepted] = True
                    resolution["second_best"] += int(len(accepted))
        remaining = query[~assigned]
        if not len(remaining):
            labels = result[query]
            labels = labels[labels > 0]
            for class_id, count in zip(*np.unique(labels, return_counts=True)):
                targets[int(class_id)] += int(count)
            continue
        histogram = voxel_neighbour_histogram(
            voxel_coords, class_ids, passed_rules, remaining, class_count, radius, neighbour_index
        )
        histogram[np.arange(len(remaining)), class_ids[remaining]] = 0
        totals = histogram.sum(axis=1)
        order = np.argsort(-histogram, axis=1, kind="stable")
        remaining_assigned = np.zeros(len(remaining), dtype=bool)
        remaining_features = _subset(features, remaining)
        for rank in range(class_count + 1):
            candidates = order[:, rank]
            support = histogram[np.arange(len(remaining)), candidates]
            eligible = (~remaining_assigned & (candidates > 0) & (support > 0)
                        & (support >= totals * min_share))
            if not eligible.any():
                continue
            eligible_indices = np.flatnonzero(eligible)
            candidate_ok = _candidate_passes(
                candidates[eligible], _subset(remaining_features, eligible_indices), id_to_name,
                rules_config, remaining[eligible_indices], class_ids, passed_rules, voxel_coords,
                candidate_name_to_id, curb_context,
            )
            accepted = eligible_indices[candidate_ok]
            if len(accepted):
                result[remaining[accepted]] = candidates[accepted]
                remaining_assigned[accepted] = True
        resolution["neighbours"] += int(remaining_assigned.sum())
        unresolved = ~remaining_assigned
        result[remaining[unresolved]] = 0
        labels = result[query]
        labels = labels[labels > 0]
        for class_id, count in zip(*np.unique(labels, return_counts=True)):
            targets[int(class_id)] += int(count)
        unknown += int(unresolved.sum())
    resolution["unknown"] = unknown
    if return_details:
        return result, targets, unknown, resolution
    return result, targets, unknown


def refine_classification(
    xyz: np.ndarray,
    class_ids: np.ndarray,
    config: dict,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    scores: dict[str, np.ndarray] | None = None,
    rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Run the complete geometry refinement on in-memory points and labels."""
    unseen = class_ids == NEVER_CLASSIFIED_ID
    class_ids = class_ids.copy()
    class_ids[unseen] = 0
    started = time.perf_counter()
    terrain_config = config["terrain"]
    ground_ids = [name_to_id[name] for name in terrain_config.get("ground_classes", []) if name in name_to_id]
    ground_mask = np.isin(class_ids, ground_ids)
    point_count = len(class_ids)
    confidence = np.full(point_count, np.nan)
    second_class = np.full(point_count, np.nan)
    second_conf = np.full(point_count, np.nan)
    if scores is not None:
        confidence = scores["top1_conf"].astype(np.float64, copy=False)
        second_class = scores["top2_class"].astype(np.float64, copy=False)
        second_conf = scores["top2_conf"].astype(np.float64, copy=False)
    confidence_config = config.get("confidence", {})
    dtm_mask = ground_mask.copy()
    dtm_confidence_fallback = False
    if scores is not None:
        dtm_mask &= (~np.isfinite(confidence) | (confidence >= float(confidence_config.get("dtm_min_confidence", 0.0))))
        enough_cells = populated_terrain_cells(
            xyz, dtm_mask, float(terrain_config["cell_size"]), int(terrain_config["min_points"])
        )
        if enough_cells < 3:
            dtm_mask = ground_mask
            dtm_confidence_fallback = True
    dtm, origin, cell_size = build_dtm(
        xyz, dtm_mask, float(terrain_config["cell_size"]), float(terrain_config["percentile"]),
        int(terrain_config["min_points"]), float(terrain_config["opening_window_m"]),
        method=str(terrain_config.get("method", "opening")), pmf=terrain_config.get("pmf"),
    )
    feature_values, voxel_coords, _ = compute_voxel_features(
        xyz, float(config["features"]["voxel_size"]), int(config["features"]["min_points"]),
    )
    features = {
        "hag": height_above_ground(xyz, dtm, origin, cell_size),
        "confidence": confidence,
        "second_class": second_class,
        "second_conf": second_conf,
        **feature_values,
    }
    surface_input, surface_stats = apply_surface_rule(class_ids, xyz, rgb, features["hag"], name_to_id, config.get("surface", {}))
    refined, flat_roof_count = apply_flat_roof_rule(surface_input, features, name_to_id, config)
    support_config = config.get("support", {})
    horizontal = np.isfinite(features["normal_z"]) & (features["normal_z"] >= 0.9)
    support_ids = [name_to_id[name] for name in terrain_config.get("ground_classes", []) if name in name_to_id]
    if "building" in name_to_id:
        support_ids.append(name_to_id["building"])
    support_mask = horizontal & np.isin(refined, support_ids)
    try:
        support, support_origin, support_cell_size = build_local_support(
            xyz, support_mask, float(support_config.get("cell_size", 1.0)),
            float(support_config.get("percentile", 5)), int(support_config.get("min_points", 3)),
            float(support_config.get("window_m", 3)),
        )
        features["hls"] = height_above_support(xyz, support, support_origin, support_cell_size)
    except ValueError:
        features["hls"] = np.full(point_count, np.nan)
    failures: Counter = Counter()
    reassigned: Counter = Counter()
    resolution: Counter = Counter()
    unknown = 0
    iterations = max(1, int(config.get("neighbors", {}).get("iterations", 1)))
    for _ in range(iterations):
        protect_above = float(confidence_config.get("protect_above", np.inf))
        protected = np.isfinite(confidence) & (confidence >= protect_above)
        alias_ids = [class_id for class_id, name in id_to_name.items() if list(id_to_name.values()).count(name) > 1]
        if alias_ids:
            protected |= np.isin(refined, alias_ids)
        passed = evaluate_rules(
            refined, features, id_to_name, config.get("rules", {}), voxel_coords, name_to_id, protected
        )
        failed = (refined > 0) & ~passed
        failures.update(id_to_name.get(int(class_id), str(class_id)) for class_id in refined[failed])
        if not failed.any():
            break
        refined, targets, unresolved, resolved_by = neighbour_reassign(
            refined, failed, passed, features, voxel_coords, id_to_name, config.get("rules", {}), config.get("neighbors", {}),
            second_class.astype(np.uint8, copy=False) if scores is not None else None,
            second_conf,
            confidence_config,
            return_details=True,
        )
        for class_id, count in targets.items():
            reassigned[id_to_name.get(class_id, str(class_id))] += count
        unknown += unresolved
        resolution.update(resolved_by)
    smoothing = config.get("smoothing", {})
    smoothed_count = 0
    if smoothing.get("enabled", True):
        protected_ids = tuple({*([name_to_id[name] for name in smoothing.get("protected_classes", []) if name in name_to_id]), *alias_ids})
        before_smoothing = refined
        refined = neighbourhood_majority(
            xyz, refined, max(int(refined.max(initial=0)), max(id_to_name, default=0)) + 1,
            voxel=float(smoothing.get("voxel_size", 0.3)), iterations=int(smoothing.get("iterations", 2)),
            own_max_share=float(smoothing.get("own_max_share", 0.25)),
            majority_min_share=float(smoothing.get("majority_min_share", 0.6)), protected=protected_ids,
        )
        smoothed_count = int(np.count_nonzero(before_smoothing != refined))
    changed = np.flatnonzero(class_ids != refined)
    final_transitions = Counter(zip(map(int, class_ids[changed]), map(int, refined[changed])))
    refined[unseen] = NEVER_CLASSIFIED_ID
    return refined, {
        "flat_roof_conversions": flat_roof_count,
        "failures_per_class": dict(failures),
        "reassigned_per_target_class": dict(reassigned),
        "set_to_unknown": unknown,
        "resolved_by_second_best": int(resolution["second_best"]),
        "resolved_by_neighbours": int(resolution["neighbours"]),
        "resolved_as_unknown": int(resolution["unknown"]),
        "dtm_confidence_filter_used": scores is not None,
        "dtm_confidence_filter_fallback": dtm_confidence_fallback,
        "unknown_before": int((class_ids == 0).sum()),
        "unknown_after": int((refined == 0).sum()),
        "top_transitions": [
            {"from": id_to_name.get(source, "unknown"), "to": id_to_name.get(target, "unknown"), "count": count}
            for (source, target), count in final_transitions.most_common(12)
        ],
        "surface_rule": surface_stats,
        "smoothed_points": smoothed_count,
        "dtm_method": str(terrain_config.get("method", "opening")),
        "runtime_seconds": time.perf_counter() - started,
        "terrain_cells": [int(dtm.shape[1]), int(dtm.shape[0])],
    }


def _write_progress(path: Path | None, detail: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "percent": 99.0,
        "stage": "Geometric refinement",
        "detail": detail,
        "photo_processed": 1,
        "photo_total": 1,
        "photo_percent": 100.0,
    }), encoding="utf-8")


def _print_summary(summary: dict) -> None:
    print("\nGeometric refinement")
    print("-" * 40)
    print(f"Flat-roof conversions: {summary['flat_roof_conversions']:,}")
    print(f"Unknown points: {summary['unknown_before']:,} -> {summary['unknown_after']:,}")
    print(f"Set to unknown: {summary['set_to_unknown']:,}")
    print(f"Resolved by SAM second-best: {summary['resolved_by_second_best']:,}")
    print(f"Resolved by neighbours: {summary['resolved_by_neighbours']:,}")
    print(f"Runtime: {summary['runtime_seconds']:.1f} s")
    for transition in summary["top_transitions"][:6]:
        print(f"{transition['from']:>24s} -> {transition['to']:<24s} {transition['count']:>10,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine a georeferenced classified point cloud using geometric rules")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--classes-json", required=True, type=Path)
    parser.add_argument("--class-set", required=True)
    parser.add_argument("--engine", choices=("sam3", "flurseg"), default="sam3")
    parser.add_argument("--config", type=Path, default=Path("config/geometry.yml"))
    parser.add_argument("--summary-file", required=True, type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--scores", type=Path,
                        help="optional SegEarth-OV3 confidence NPZ written by classify.py")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.class_set not in config.get("enabled_class_sets", []):
        print(f"Geometric refinement skipped: class set '{args.class_set}' is not enabled.")
        summary = {"skipped": f"class set '{args.class_set}' is not enabled", "effective_config": config}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.input, args.output)
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return
    _write_progress(args.progress_file, "Reading classified point cloud")
    xyz, rgb, class_ids = read_classified_ply(args.input)
    id_to_name, name_to_id = class_name_mapping(args.classes_json, args.class_set, args.engine)
    scores = load_scores(args.scores, len(xyz)) if args.scores else None
    _write_progress(args.progress_file, "Estimating terrain and local geometry")
    refined, summary = refine_classification(xyz, class_ids, config, id_to_name, name_to_id, scores, rgb=rgb)
    summary["confidence_percentiles_per_class"] = (
        confidence_percentiles(class_ids, scores["top1_conf"], id_to_name) if scores is not None else {}
    )
    summary["effective_config"] = config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_classified_ply(args.output, xyz, rgb, refined)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_progress(args.progress_file, "Completed")
    _print_summary(summary)


if __name__ == "__main__":
    main()
