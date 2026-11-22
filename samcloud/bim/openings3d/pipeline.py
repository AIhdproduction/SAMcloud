"""Orchestration of the 3D opening refinement for one level, with freshness and safe fallbacks.

``refine_level_openings`` loads the level's ``plan_model.json`` (first plan pass), measures every window and
door on the wall planes (photograph votes + point evidence) and writes ``openings_3d.json`` next to it.

* Inputs: the refinement works on the *pass-1 snapshot* ``plan_model_pass1.json`` (walls and 2D-detected openings of
  the plan pass computed without refined openings), never on a plan that already contains refined openings, so
  running it again from a refined plan gives the same result (idempotent).  Without a snapshot (legacy projects)
  ``plan_model.json`` is used.
* Freshness: the refinement fingerprint contains the wall set (never the openings), the threshold hash,
  the sparse model / cloud stamps and whether the photograph path is available.  An unchanged
  fingerprint skips the work, so ``plan -> refine -> plan`` cannot loop.
* No CUDA or no usable photographs: points-only mode (observed hole rectangles only).
* Any other exception: warning in the log, no ``openings_3d.json`` is written; the pipeline continues
  with the 2D openings.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from samcloud.bim.invariants import refinement_violations
from samcloud.bim.plan_model import BuildingPlan, load_refine_plan
from samcloud.dxf.cancel import Cancelled

from .config import Settings, config_hash, load_openings_config
from .classify import classify_openings
from .detect import Detection, detect_openings
from .evidence import LineEvidence, build_line_evidence
from .images import camera_key, select_images, usable_cameras
from .model import OpeningsResult, load_openings_3d, openings_3d_path, openings_3d_sha256, save_openings_3d
from .points import collect_point_evidence
from .pointsonly import detect_point_openings, points_only_detection
from .scoring import Scorer, Sam3Scorer, score_images, scores_directory
from .voting import LineVotes, merge_faces, vote_faces
from .walls import WallLine, plan_frame, wall_faces, wall_lines, walls_fingerprint, walls_match

_LOGGER = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class RefineOutcome:
    status: str  # refined | fresh | failed
    mode: str = ""
    openings: int = 0
    changed: bool = False  # openings_3d.json content differs from before the call
    seconds: float = 0.0
    message: str = ""


def level_cache_dir(project_root: Path | str, level_key: str) -> Path:
    """Folder of the level's clean-plan cache (result.json, plan_model.json, openings_3d.json)."""
    return Path(project_root) / "products" / "dxf" / level_key / "ortho"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def _stamp(path: Path | None) -> list[Any] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
        return [path.name, int(stat.st_size), int(stat.st_mtime_ns)]
    except OSError:
        return None


def resolve_sparse_model(project_root: Path, level_key: str) -> Path | None:
    """Sparse model the level's orthophoto used (same cameras as the plan), else ``products/dense/sparse``."""
    ortho_json = project_root / "products" / "ortho" / level_key / "ortho.json"
    try:
        metadata = json.loads(ortho_json.read_text(encoding="utf-8"))
        candidate = Path(str(metadata.get("sparse_model_used") or (metadata.get("source_paths") or {}).get("sparse_model") or ""))
        if candidate.is_dir():
            return candidate
    except (OSError, ValueError):
        pass
    fallback = project_root / "products" / "dense" / "sparse"
    return fallback if fallback.is_dir() else None


def resolve_point_source(project_root: Path, level_key: str, preferred: Path | None = None) -> Path | None:
    if preferred is not None and Path(preferred).is_file():
        return Path(preferred)
    ortho_json = project_root / "products" / "ortho" / level_key / "ortho.json"
    try:
        candidate = Path(str((json.loads(ortho_json.read_text(encoding="utf-8")).get("source_paths") or {}).get("point_source") or ""))
        if candidate.is_file():
            return candidate
    except (OSError, ValueError):
        pass
    for name in ("classified.las", "classified.ply"):
        candidate = project_root / "products" / "classified" / name
        if candidate.is_file():
            return candidate
    return None


def photographs_available(project_root: Path) -> bool:
    """True when the cube-map side images exist (cheap check used for freshness)."""
    cubemap = project_root / "inputs" / "cubemap"
    return cubemap.is_dir() and any(cubemap.glob("*/*.jpg"))


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def current_inputs(plan: BuildingPlan, lines: list[WallLine], project_root: Path, level_key: str, config: Settings, point_source: Path | None) -> dict[str, Any]:
    """Fingerprint of every input that can change the refinement. Walls only, never the openings."""
    sparse = resolve_sparse_model(project_root, level_key)
    return {
        "walls": walls_fingerprint(lines),
        "config_hash": config_hash(config),
        "sparse_model": _stamp(sparse / "images.bin") if sparse is not None else None,
        "point_source": _stamp(point_source),
        "photo_path": bool(cuda_available() and photographs_available(project_root) and sparse is not None),
    }


def inputs_match(stored: dict[str, Any], current: dict[str, Any], tolerance_units: float) -> bool:
    if not stored:
        return False
    for key in ("config_hash", "sparse_model", "point_source", "photo_path"):
        if stored.get(key) != current.get(key):
            return False
    return walls_match(list(stored.get("walls", [])), list(current["walls"]), tolerance_units)


def refine_status(project_root: Path | str, level_key: str, config: Settings | None = None, point_source: Path | None = None) -> str:
    """'missing' without openings_3d.json, 'fresh' when its fingerprint matches the current plan walls, else 'stale'."""
    root = Path(project_root)
    directory = level_cache_dir(root, level_key)
    if not openings_3d_path(directory).is_file():
        return "missing"
    try:
        settings = config or load_openings_config()
        stored = load_openings_3d(directory)
        plan = load_refine_plan(directory)
        lines = wall_lines(plan, settings)
        source = resolve_point_source(root, level_key, point_source)
        current = current_inputs(plan, lines, root, level_key, settings, source)
        return "fresh" if inputs_match(stored.inputs, current, settings.freshness.wall_tolerance_m * plan.units_per_metre) else "stale"
    except Exception:  # noqa: BLE001 - unreadable file counts as stale
        return "stale"


# --------------------------------------------------------------------------
# computation
# --------------------------------------------------------------------------

def _seeds(plan: BuildingPlan) -> list[Any]:
    return [opening for opening in plan.openings if opening.kind in ("door", "passage")]


def _unconfirmed(plan: BuildingPlan, evidence: dict[str, LineEvidence], detection: Detection, config: Settings) -> list[dict[str, Any]]:
    """Plan windows and point-cloud windows that no photograph confirmed (unknown, not emitted)."""
    upm = plan.units_per_metre
    limit = config.dedupe.unconfirmed_distance_m
    windows = [r for r in detection.openings if r.kind == "window"]
    result: list[dict[str, Any]] = []

    def covered(centre: list[float]) -> bool:
        return any(np.hypot(r.center[0] - centre[0], r.center[1] - centre[1]) / upm < limit for r in windows)

    for opening in plan.openings:
        if opening.kind == "window" and not covered(list(opening.center)):
            result.append({"source": "plan", "plan_id": opening.id, "center": [round(float(opening.center[0]), 2), round(float(opening.center[1]), 2)], "width_m": opening.width_m, "note": "plan window with no observed window evidence: dropped (unknown)"})
    for ev in evidence.values():
        for item in detect_point_openings(ev, config):
            if item["kind"] != "window":
                continue
            line = ev.line
            centre = line.a + line.d * ((item["u0"] + item["u1"]) / 2) * upm
            if not covered([float(centre[0]), float(centre[1])]):
                result.append({
                    "source": "points", "wall": line.id, "center": [round(float(centre[0]), 3), round(float(centre[1]), 3)], "width_m": round(item["u1"] - item["u0"], 2),
                    "sill_m": round(item["v0"], 2), "head_m": round(item["v1"], 2), "note": "point-hole window without observed photograph vote: unknown, not emitted",
                })
    return result


def _classify_doors(detection: Detection, lines: list[WallLine], evidence_points: Any, plan: BuildingPlan, config: Settings) -> dict[str, Any]:
    """Classify door / passage / none, assign open leaves, scan for entrances (in place on ``detection``)."""
    kept, dropped, summary = classify_openings(detection.openings, lines, evidence_points, plan, config, detection.duplicates)
    detection.openings = kept
    violations = refinement_violations(kept)
    summary["invariant_violations"] = violations
    for item in violations:
        _LOGGER.warning("openings3d: invariant violated (%s) by %s: %s", item["rule"], item["opening"], item["reason"])
    for item in dropped:
        detection.rejected.append({"what": "door candidate", "center": item["center"], "line": item["line"], "reason": f"classified as no opening: {item['reason']}", "features": item["features"]})
    detection.closed = [{key: item[key] for key in ("center", "width_m", "axis_angle", "wall_thickness_m", "host_wall_id", "line", "u", "reason")} for item in dropped]
    summary["dropped_candidates"] = dropped
    return summary


def _compute_full(
    root: Path, level_key: str, plan: BuildingPlan, lines: list[WallLine], config: Settings, point_source: Path, scorer: Scorer | None, device: str,
    progress: ProgressFn, cancel: Callable[[], bool] | None,
) -> tuple[Detection, dict[str, Any], list[dict[str, Any]]] | None:
    """Photograph + point detection; None when no usable photographs exist (caller falls back to points)."""
    from samcloud.ortho.cameras import load_cameras

    sparse = resolve_sparse_model(root, level_key)
    if sparse is None:
        return None
    cameras = usable_cameras(load_cameras(sparse, root), config)
    if not cameras:
        return None
    frame = plan_frame(plan)
    faces = wall_faces(lines)
    started = time.time()
    progress(0.0, "Selecting photographs")
    selection = select_images(faces, cameras, plan, frame, config, device=device, cancel=cancel)
    keys = sorted({key for chosen in selection.values() for key, _ in chosen})
    if not keys:
        return None
    by_key = {camera_key(camera): camera for camera in cameras}
    t_select = time.time() - started
    directory = scores_directory(root, config)
    paths = score_images(keys, by_key, directory, config, scorer or Sam3Scorer(config), progress=lambda f, s: progress(0.08 + 0.37 * f, s), cancel=cancel)
    t_score = time.time() - started - t_select
    evidence_points = collect_point_evidence(point_source, lines, plan, frame, config, want_cloud=True, progress=lambda f, s: progress(0.45 + 0.2 * f, s), cancel=cancel)
    if evidence_points.cloud is None or not len(evidence_points.cloud):
        return None
    votes = vote_faces(selection, faces, by_key, paths, evidence_points.cloud, plan, frame, config, device=device, progress=lambda f, s: progress(0.65 + 0.25 * f, s), cancel=cancel)
    progress(0.9, "Detecting windows and doors")
    by_line: dict[str, list[LineVotes]] = {}
    for (line_id, _), face_votes in votes.items():
        by_line.setdefault(line_id, []).append(face_votes)
    line_by_id = {line.id: line for line in lines}
    evidence = {line_id: build_line_evidence(line_by_id[line_id], evidence_points.grids[line_id], merge_faces(parts), config) for line_id, parts in by_line.items()}
    detection = detect_openings(evidence, _seeds(plan), plan, config)
    unconfirmed = _unconfirmed(plan, evidence, detection, config)
    door_summary = _classify_doors(detection, lines, evidence_points, plan, config)
    # Only deterministic numbers go into openings_3d.json (its hash tells whether the openings changed); timings are logged.
    stats = {"images": len(keys), "faces_voted": len(votes), "points": evidence_points.point_count, "doors": door_summary}
    _LOGGER.info("openings3d: selection %.1f s, SAM scoring %.1f s, total %.1f s", t_select, t_score, time.time() - started)
    return detection, stats, unconfirmed


def _compute_points(root: Path, plan: BuildingPlan, lines: list[WallLine], config: Settings, point_source: Path, progress: ProgressFn, cancel: Callable[[], bool] | None) -> tuple[Detection, dict[str, Any], list[dict[str, Any]]]:
    started = time.time()
    frame = plan_frame(plan)
    evidence_points = collect_point_evidence(point_source, lines, plan, frame, config, want_cloud=False, progress=lambda f, s: progress(0.8 * f, s), cancel=cancel)
    progress(0.85, "Detecting openings from points")
    evidence = {line.id: build_line_evidence(line, evidence_points.grids[line.id], None, config) for line in lines}
    detection = points_only_detection(evidence, _seeds(plan), plan, config)
    door_summary = _classify_doors(detection, lines, evidence_points, plan, config)
    _LOGGER.info("openings3d: points-only detection took %.1f s", time.time() - started)
    return detection, {"points": evidence_points.point_count, "doors": door_summary}, []


def refine_level_openings(
    project_root: Path | str,
    level_key: str,
    *,
    point_source: Path | None = None,
    config: Settings | None = None,
    scorer: Scorer | None = None,
    device: str | None = None,
    progress: ProgressFn | None = None,
    cancel: Callable[[], bool] | None = None,
    force: bool = False,
) -> RefineOutcome:
    """Refine the level's openings and write ``openings_3d.json``. Never raises except for cancellation."""
    root = Path(project_root)
    report: ProgressFn = progress or (lambda _f, _s: None)
    started = time.time()
    directory = level_cache_dir(root, level_key)
    try:
        settings = config or load_openings_config()
        plan = load_refine_plan(directory)
        lines = wall_lines(plan, settings)
        source = resolve_point_source(root, level_key, point_source)
        if source is None:
            raise RuntimeError("no classified point cloud found")
        inputs = current_inputs(plan, lines, root, level_key, settings, source)
        before = openings_3d_sha256(directory)
        if not force and before is not None:
            try:
                stored = load_openings_3d(directory)
                if inputs_match(stored.inputs, inputs, settings.freshness.wall_tolerance_m * plan.units_per_metre):
                    report(1.0, "Refined openings are up to date")
                    return RefineOutcome("fresh", stored.mode, len(stored.openings), False, time.time() - started)
            except Exception:  # noqa: BLE001 - recompute
                pass
        compute_device = device or ("cuda" if cuda_available() else "")
        computed = None
        mode = "points"
        if compute_device and inputs["photo_path"]:
            computed = _compute_full(root, level_key, plan, lines, settings, source, scorer, compute_device, report, cancel)
            mode = "full"
        if computed is None:
            if mode == "full":
                _LOGGER.warning("openings3d: no usable photographs; falling back to points-only openings")
            else:
                _LOGGER.info("openings3d: no CUDA device or no photographs; points-only openings")
            mode = "points"
            computed = _compute_points(root, plan, lines, settings, source, report, cancel)
        detection, stats, unconfirmed = computed
        result = OpeningsResult(
            openings=detection.openings, curtain_regions=detection.curtain_regions, unconfirmed=unconfirmed, rejected=detection.rejected, closed=detection.closed, inputs=inputs, mode=mode, stats=stats,
        )
        save_openings_3d(result, directory)
        after = openings_3d_sha256(directory)
        report(1.0, f"Refined {len(result.openings)} openings ({mode})")
        return RefineOutcome("refined", mode, len(result.openings), before != after, time.time() - started)
    except Cancelled:
        raise
    except Exception as error:  # noqa: BLE001 - the pipeline continues with the 2D openings
        _LOGGER.warning("openings3d: refinement failed, keeping the 2D openings: %s", error, exc_info=True)
        return RefineOutcome("failed", "", 0, False, time.time() - started, message=str(error))


def refined_openings_applied(project_root: Path | str, level_key: str) -> tuple[bool, str]:
    """(applied, reason): whether the level's plan_model.json contains the refined openings of its openings_3d.json.

    True without an openings_3d.json (nothing to apply).  The reason names why a refinement was not applied (stale pass-1
    snapshot, different wall set, unreadable plan).
    """
    directory = level_cache_dir(project_root, level_key)
    if openings_3d_sha256(directory) is None:
        return True, "no refined openings"
    try:
        source = json.loads((directory / "plan_model.json").read_text(encoding="utf-8")).get("source") or {}
    except (OSError, ValueError) as error:
        return False, f"plan_model.json unreadable: {error}"
    info = source.get("openings_3d") or {}
    if info.get("applied"):
        return True, "applied"
    return False, str(info.get("reason") or ("openings_3d.json belongs to a different wall set" if info.get("sha256") else "no refinement info in the plan"))


def plan_has_current_openings(project_root: Path | str, level_key: str) -> bool:
    """True when the level's plan_model.json was generated with the current openings_3d.json (or none exists).

    The clean plan records the sha256 of the openings file it used (``source.openings_3d.sha256``), also when the
    file was rejected because of a different wall set, so a rejected file does not trigger endless re-planning.
    """
    directory = level_cache_dir(project_root, level_key)
    current = openings_3d_sha256(directory)
    if current is None:
        return True
    try:
        source = json.loads((directory / "plan_model.json").read_text(encoding="utf-8")).get("source") or {}
    except (OSError, ValueError):
        return False
    return (source.get("openings_3d") or {}).get("sha256") == current
