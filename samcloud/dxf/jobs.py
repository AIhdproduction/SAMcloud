"""Subprocess CLI jobs for DXF preview, alignment, level detection, generation, and export."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from samcloud.dxf.alignment import estimate_alignment
from samcloud.dxf.cancel import CancelToken, Cancelled, ProgressCallback
from samcloud.dxf.las_source import Frame, plausibility_warnings, read_las_info
from samcloud.dxf.levels import Level, detect_levels
from samcloud.dxf.pipeline import cache_directory, generate_plan, is_stale, load_cached_result, save_cached_result
from samcloud.dxf.preview import create_preview
from samcloud.dxf.state import load_state, plan_settings_for, save_state
from samcloud.dxf.writer import export_plan


class JobCancelToken(CancelToken):
    """CancelToken that cooperatively checks a cancel.flag file on disk."""

    def __init__(self, flag_path: Path) -> None:
        super().__init__()
        self._flag_path = flag_path

    def is_cancelled(self) -> bool:
        if super().is_cancelled():
            return True
        if self._flag_path.is_file():
            self.cancel()
            return True
        return False


def setup_job_logging(log_path: Path) -> logging.Logger:
    """Configure logging to products/dxf/job.log and console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("samcloud.dxf.jobs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def replace_with_retry(source: str | Path, target: Path, attempts: int = 50, delay: float = 0.02) -> None:
    """os.replace that tolerates a reader briefly holding the target open (Windows sharing violation)."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def write_job_status(
    status_file: Path,
    action: str,
    state: str,
    fraction: float,
    stage: str,
    message: str = "",
    started: str = "",
    finished: str | None = None,
    pid: int | None = None,
) -> None:
    """Write job status JSON atomically."""
    status_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "action": action,
        "state": state,
        "fraction": max(0.0, min(1.0, float(fraction))),
        "stage": stage,
        "message": message,
        "started": started,
        "finished": finished,
        "pid": pid if pid is not None else os.getpid(),
    }
    payload = json.dumps(data, indent=2).encode("utf-8")
    descriptor, temp_path = tempfile.mkstemp(prefix=".status.", suffix=".tmp", dir=status_file.parent)
    try:
        with open(descriptor, "wb") as file:
            file.write(payload)
        replace_with_retry(temp_path, status_file)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


class _ProgressTracker:
    """Tracks the last reported progress fraction for cancellation recording."""

    def __init__(self) -> None:
        self.fraction: float = 0.0


def make_progress_callback(
    status_file: Path,
    cancel_flag: Path,
    action: str,
    started_iso: str,
    logger: logging.Logger,
    tracker: _ProgressTracker | None = None,
) -> ProgressCallback:
    """Build a ProgressCallback that polls cancel.flag and writes job_status.json."""
    def callback(fraction: float, stage: str) -> None:
        if tracker is not None:
            tracker.fraction = float(fraction)
        if cancel_flag.is_file():
            raise Cancelled(f"Job '{action}' cancelled via cancel.flag")
        logger.info(f"[{int(fraction * 100):3d}%] {stage}")
        write_job_status(
            status_file,
            action=action,
            state="running",
            fraction=fraction,
            stage=stage,
            started=started_iso,
            pid=os.getpid(),
        )
    return callback


def run_preview(
    project_dir: Path,
    las_path: Path,
    progress: ProgressCallback,
    cancel: CancelToken,
    logger: logging.Logger,
) -> None:
    """Generate voxel preview and LAS metadata with plausibility warnings."""
    dxf_dir = project_dir / "products" / "dxf"
    dxf_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(project_dir)

    logger.info("Reading LAS info...")
    info = read_las_info(las_path, progress=progress, cancel=cancel)

    factor = float(state.scale.get("factor", 1.0))
    # Preview stores points in UNROTATED local frame (origin + scale only, rotation = identity)
    frame = Frame(origin=info.frame.origin, scale_factor=factor)

    logger.info("Creating voxel preview...")
    preview = create_preview(las_path, frame, info=info, progress=progress, cancel=cancel)
    warnings = plausibility_warnings(info)

    # Save preview.npz atomically including used frame (origin, scale_factor)
    temp_npz = dxf_dir / ".preview.tmp.npz"
    arrays: dict[str, Any] = {
        "xyz": preview.xyz,
        "cls": preview.cls,
        "voxel_size": np.array(preview.voxel_size),
        "origin": np.asarray(frame.origin, dtype=np.float64),
        "scale_factor": np.float64(factor),
    }
    if preview.rgb is not None:
        arrays["rgb"] = preview.rgb
    np.savez_compressed(temp_npz, **arrays)
    os.replace(temp_npz, dxf_dir / "preview.npz")

    # Save las_info.json atomically
    las_info_dict = {
        "las_path": str(Path(las_path).resolve()),
        "point_count": info.point_count,
        "bounds_min": info.bounds_min.tolist(),
        "bounds_max": info.bounds_max.tolist(),
        "has_rgb": info.has_rgb,
        "class_histogram": {int(k): int(v) for k, v in info.class_histogram.items()},
        "profile": info.profile.profile_name if info.profile else None,
        "mode": info.mode,
        "warnings": warnings,
    }
    temp_json = dxf_dir / ".las_info.tmp.json"
    temp_json.write_text(json.dumps(las_info_dict, indent=2), encoding="utf-8")
    os.replace(temp_json, dxf_dir / "las_info.json")

    logger.info("Preview and LAS info written successfully.")


def _load_points_for_analysis(
    project_dir: Path,
    las_path: Path,
    progress: ProgressCallback,
    cancel: CancelToken,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Load unrotated XYZ and classification points from preview.npz or generate preview."""
    state = load_state(project_dir)
    current_scale = float(state.scale.get("factor", 1.0))
    dxf_dir = project_dir / "products" / "dxf"
    preview_file = dxf_dir / "preview.npz"
    if preview_file.is_file():
        logger.info("Loading points from preview.npz...")
        data = np.load(preview_file)
        xyz = np.array(data["xyz"])
        cls = np.array(data["cls"])
        if "scale_factor" in data:
            stored_scale = float(data["scale_factor"])
            if not np.isclose(stored_scale, current_scale, atol=1e-6):
                if stored_scale > 0:
                    logger.info(f"Rescaling preview points from factor {stored_scale} to {current_scale}")
                    xyz = xyz * (current_scale / stored_scale)
                else:
                    raise RuntimeError("Preview is stale, run Create preview again")
        else:
            if not np.isclose(current_scale, 1.0, atol=1e-6):
                raise RuntimeError("Preview is stale, run Create preview again")
        return xyz, cls

    logger.info("preview.npz not found; reading LAS directly...")
    info = read_las_info(las_path, cancel=cancel)
    frame = Frame(origin=info.frame.origin, scale_factor=current_scale)
    preview = create_preview(las_path, frame, info=info, progress=progress, cancel=cancel)
    return preview.xyz, preview.cls


def run_align(
    project_dir: Path,
    las_path: Path,
    progress: ProgressCallback,
    cancel: CancelToken,
    logger: logging.Logger,
) -> None:
    """Estimate gravity alignment on unrotated points and persist to project state."""
    xyz, cls = _load_points_for_analysis(project_dir, las_path, progress, cancel, logger)
    logger.info("Estimating gravity alignment...")
    alignment_result = estimate_alignment(xyz, cls, progress=progress, cancel=cancel)

    state = load_state(project_dir)
    state.alignment = alignment_result.to_dict()
    save_state(project_dir, state)
    logger.info(f"Alignment complete: tilt={alignment_result.tilt_degrees:.2f} deg, yaw={alignment_result.yaw_degrees:.2f} deg.")


def run_levels(
    project_dir: Path,
    las_path: Path,
    progress: ProgressCallback,
    cancel: CancelToken,
    logger: logging.Logger,
) -> None:
    """Detect storeys and persist to project state."""
    xyz, cls = _load_points_for_analysis(project_dir, las_path, progress, cancel, logger)
    state = load_state(project_dir)

    if state.alignment and "rotation" in state.alignment:
        rot = np.asarray(state.alignment["rotation"], dtype=np.float64)
        xyz = xyz @ rot.T

    logger.info("Detecting levels...")
    detected = detect_levels(xyz, cls, progress=progress, cancel=cancel)
    levels_list = [
        {
            "id": lvl.id,
            "name": lvl.name,
            "floor_z": float(lvl.floor_z),
            "ceiling_z": float(lvl.ceiling_z) if lvl.ceiling_z is not None else None,
            "checked": True,
        }
        for lvl in detected
    ]
    state.levels = levels_list
    save_state(project_dir, state)
    logger.info(f"Detected {len(levels_list)} levels.")


def _level_from_dict(item: dict[str, Any]) -> Level:
    """Rebuild a Level dataclass from a state dict."""
    return Level(
        id=item["id"],
        name=item["name"],
        floor_z=float(item["floor_z"]),
        ceiling_z=float(item["ceiling_z"]) if item.get("ceiling_z") is not None else None,
        point_count=0,
        confidence=1.0,
        low_confidence=False,
    )


def run_generate(
    project_dir: Path,
    las_path: Path,
    level_id: str | None,
    progress: ProgressCallback,
    cancel: CancelToken,
    logger: logging.Logger,
    mode: str | None = None,
) -> None:
    """Generate DXF vector plan for level or outdoor site and cache result."""
    state = load_state(project_dir)
    info = read_las_info(las_path, cancel=cancel)
    mode = _effective_mode(mode, state.mode_override, info.mode)

    level_obj: Level | None = None
    if mode == "indoor":
        if not level_id:
            raise ValueError("Indoor plan generation requires an explicit level ID (--level <id>)")
        matched = next((lvl for lvl in state.levels if lvl["id"] == level_id), None)
        if matched is None:
            raise ValueError(f"Level '{level_id}' not found in state levels")
        level_obj = _level_from_dict(matched)

    settings = plan_settings_for(state, level_obj, base_frame=info.frame, project_directory=project_dir)
    target_cache_dir = cache_directory(las_path, level_obj, settings)

    logger.info(f"Generating plan: mode={mode}, level={level_obj.name if level_obj else 'site'}...")
    result = generate_plan(las_path, mode, level_obj, settings=settings, progress=progress, cancel=cancel)

    save_cached_result(result, target_cache_dir)
    logger.info(f"Cached result saved to {target_cache_dir}")


def run_export(
    project_dir: Path,
    las_path: Path,
    level_id: str | None,
    progress: ProgressCallback,
    cancel: CancelToken,
    logger: logging.Logger,
    mode: str | None = None,
) -> None:
    """Export DXF files from cached results into products/dxf/export/."""
    state = load_state(project_dir)
    info = read_las_info(las_path, cancel=cancel)
    mode = _effective_mode(mode, state.mode_override, info.mode)
    dxf_dir = project_dir / "products" / "dxf"
    export_dir = dxf_dir / "export"
    coordinates = state.view.get("coordinates", "local")

    levels_to_export: list[Level | None] = []
    if mode == "indoor":
        if level_id:
            matched = next((lvl for lvl in state.levels if lvl["id"] == level_id), None)
            if matched is None:
                raise ValueError(f"Level '{level_id}' not found in state levels")
            levels_to_export.append(_level_from_dict(matched))
        else:
            checked = [lvl for lvl in state.levels if lvl.get("checked", True)]
            if not checked:
                checked = state.levels
            for item in checked:
                levels_to_export.append(_level_from_dict(item))
            if not levels_to_export:
                raise RuntimeError("No indoor levels defined in state to export")
    else:
        levels_to_export.append(None)

    # First verify every checked level has a cache and is not stale; fail before writing anything
    for lvl in levels_to_export:
        if cancel.is_cancelled():
            raise Cancelled("Job 'export' cancelled")
        settings = plan_settings_for(state, lvl, base_frame=info.frame, project_directory=project_dir)
        cache_dir = cache_directory(las_path, lvl, settings)
        level_label = lvl.name if lvl else "site"
        result_path = cache_dir / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"No cached result found for level '{level_label}' in {cache_dir}. Run generate first.")
        cached_result = load_cached_result(cache_dir)
        # Generation records the LAS provenance profile, so compare against that (profile_name=None);
        # a profile chosen manually for a LAS without provenance must not make every cache stale.
        if is_stale(cached_result, settings, level=lvl, profile_name=None):
            raise RuntimeError(f"Cached result for level '{level_label}' in {cache_dir} is stale. Run generate first.")

    # Export into a temp dir and move DXF files into products/dxf/export/ only when all succeeded
    with tempfile.TemporaryDirectory(prefix=".export.", dir=dxf_dir) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        exported_files: list[Path] = []
        total = len(levels_to_export)

        for idx, lvl in enumerate(levels_to_export):
            # Check cancel before each level
            if cancel.is_cancelled():
                raise Cancelled("Job 'export' cancelled")

            settings = plan_settings_for(state, lvl, base_frame=info.frame, project_directory=project_dir)
            cache_dir = cache_directory(las_path, lvl, settings)
            logger.info(f"Loading cached result from {cache_dir}...")
            cached_result = replace(load_cached_result(cache_dir), project_name=project_dir.name)
            dest = export_plan(cached_result, temp_dir, coordinates=coordinates)
            exported_files.append(dest)
            logger.info(f"Rendered temporary DXF to {dest}")
            progress((idx + 0.9) / total, f"Rendered {dest.name}")

        if cancel.is_cancelled():
            raise Cancelled("Job 'export' cancelled")

        export_dir.mkdir(parents=True, exist_ok=True)
        final_files: list[Path] = []
        for temp_file in exported_files:
            target = export_dir / temp_file.name
            os.replace(temp_file, target)
            final_files.append(target)
            logger.info(f"Moved exported DXF to {target}")
        write_export_manifest(export_dir, final_files, coordinates)

        progress(1.0, f"Exported {len(exported_files)} DXF file(s)")


def _effective_mode(requested: str | None, override: str | None, detected: str) -> str:
    """Pick the project mode: explicit CLI mode, then the stored override, then LAS detection."""
    for candidate in (requested, override):
        if candidate in ("indoor", "outdoor"):
            return str(candidate)
    return detected


def export_manifest_path(project_dir: Path) -> Path:
    """Location of the summary written by the last successful export job."""
    return Path(project_dir) / "products" / "dxf" / "export" / "last_export.json"


def write_export_manifest(export_dir: Path, files: list[Path], coordinates: str) -> Path:
    """Record exported files and their audit result; export_plan already re-read and audited each file."""
    manifest = {
        "finished": datetime.now(timezone.utc).isoformat(),
        "coordinates": coordinates,
        "files": [str(path.resolve()) for path in files],
        "audit": "passed",
        "audit_detail": "Every file was re-read with ezdxf and audited without errors before it was moved into place.",
    }
    target = export_dir / "last_export.json"
    temporary = export_dir / ".last_export.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return target


def main() -> None:
    """Entry point for CLI module."""
    parser = argparse.ArgumentParser(description="SAMcloud DXF background jobs CLI")
    parser.add_argument("action", choices=["preview", "align", "levels", "generate", "export"], help="Job action to perform")
    parser.add_argument("--project", required=True, type=Path, help="Path to project directory")
    parser.add_argument("--las", required=True, type=Path, help="Path to LAS file")
    parser.add_argument("--level", required=False, default=None, type=str, help="Optional level ID")
    parser.add_argument("--mode", required=False, default=None, choices=["indoor", "outdoor"], help="Project mode chosen in the GUI")
    parser.add_argument("--launched", required=False, default=None, type=float, help="Epoch seconds when the runner launched this job")

    args = parser.parse_args()
    project_dir = Path(args.project).resolve()
    las_path = Path(args.las).resolve()
    action = args.action
    level_id = args.level
    mode = args.mode

    dxf_dir = project_dir / "products" / "dxf"
    dxf_dir.mkdir(parents=True, exist_ok=True)
    status_file = dxf_dir / "job_status.json"
    cancel_flag = dxf_dir / "cancel.flag"
    log_file = dxf_dir / "job.log"

    logger = setup_job_logging(log_file)
    started_iso = datetime.now(timezone.utc).isoformat()
    # A cancel.flag written after the runner launched this process is a real request (for example a
    # cancel during interpreter start-up); only flags older than the launch are stale leftovers.
    job_start_ts = float(args.launched) if args.launched is not None else time.time()
    logger.info(f"Starting job '{action}' for project '{project_dir.name}', LAS='{las_path.name}', level='{level_id}'")

    write_job_status(status_file, action, "running", 0.0, "Starting", started=started_iso, pid=os.getpid())

    # Remove pre-existing cancel.flag if mtime is older than job start
    if cancel_flag.is_file():
        try:
            if cancel_flag.stat().st_mtime <= job_start_ts:
                cancel_flag.unlink(missing_ok=True)
                logger.info("Removed stale cancel.flag from an earlier run.")
        except OSError:
            pass

    tracker = _ProgressTracker()
    token = JobCancelToken(cancel_flag)
    progress = make_progress_callback(status_file, cancel_flag, action, started_iso, logger, tracker=tracker)

    try:
        if action == "preview":
            run_preview(project_dir, las_path, progress, token, logger)
        elif action == "align":
            run_align(project_dir, las_path, progress, token, logger)
        elif action == "levels":
            run_levels(project_dir, las_path, progress, token, logger)
        elif action == "generate":
            run_generate(project_dir, las_path, level_id, progress, token, logger, mode=mode)
        elif action == "export":
            run_export(project_dir, las_path, level_id, progress, token, logger, mode=mode)

        finished_iso = datetime.now(timezone.utc).isoformat()
        write_job_status(status_file, action, "done", 1.0, "Complete", started=started_iso, finished=finished_iso, pid=os.getpid())
        logger.info(f"Job '{action}' finished successfully.")
    except Cancelled as exc:
        finished_iso = datetime.now(timezone.utc).isoformat()
        logger.warning(f"Job '{action}' was cancelled: {exc}")
        write_job_status(status_file, action, "cancelled", tracker.fraction, "Cancelled", message=str(exc), started=started_iso, finished=finished_iso, pid=os.getpid())
        sys.exit(2)
    except Exception as exc:
        finished_iso = datetime.now(timezone.utc).isoformat()
        logger.exception(f"Job '{action}' failed: {exc}")
        write_job_status(status_file, action, "failed", 0.0, "Failed", message=str(exc), started=started_iso, finished=finished_iso, pid=os.getpid())
        sys.exit(1)


if __name__ == "__main__":
    main()
