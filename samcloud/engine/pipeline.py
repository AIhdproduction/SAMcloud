import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from samcloud.acquisition import CaptureValidationError, prepare_panorama_inputs
from samcloud.colmap.alignment import (
    adaptive_dense_size,
    extract_features_by_group,
    feature_extractor_command,
    filter_pass_is_safe,
    mapper_command,
    matching_command,
    parse_model_analyzer_stats,
    pose_prior_covariance_count,
    pose_prior_count,
    resolve_camera_model,
    select_matching_strategy,
    stereo_fusion_command,
    tiepoint_pass_commands,
    write_pose_priors_from_gps_ref,
)
from samcloud.colmap.camera_groups import assign_camera_groups
from samcloud.core.configuration import load_global_config
from samcloud.core.project_manager import ProjectManager
from samcloud.core.storage import compact_project, ensure_available, ensure_text_model, release_after_fusion, remove_generated, storage_settings
from samcloud.engine.run_summary import RunSummary
from samcloud.engine.extract_gps import is_dji_rtk_project
from samcloud.control_points.georeference import coordinates_to_ecef, enu_rotation_at_ecef, georeference_from_control_points, save_control_point_report
from samcloud.control_points.model_transform import transform_text_model
from samcloud.control_points.repository import load_detections, load_observations, read_control_points, save_detections, save_observations
from samcloud.control_points.auto_detect import automatic_detection, populate_undistorted_coordinates
from samcloud.control_points.adjustment import build_control_point_report, camera_intrinsics, leave_one_out_errors, populate_review_projections, pose_priors_in_model_frame, quality_gate, run_gcp_bundle_adjustment
from samcloud.pointcloud.georeference import fit_enu_to_ecef, fit_similarity, format_residuals, save_georeference
from samcloud.engine.eta import ProgressEstimator, RateStore, stage_units


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProgressTracker:
    """Persist pipeline progress and a stage-based ETA for the local dashboard."""

    def __init__(self, path: Path, estimator: "ProgressEstimator | None" = None) -> None:
        self.path = path
        self.started_at = time.time()
        self.estimator = estimator
        self._last_write = 0.0
        self._stage = "Preparing images"
        self._detail = ""
        self._percent = 0.0

    def update(self, percent: float, stage: str, detail: str = "") -> None:
        self._percent, self._stage, self._detail = float(percent), stage, detail
        self.write(force=True)

    def begin(self, key: str, units: float | None = None) -> None:
        if self.estimator is not None:
            self.estimator.start_stage(key, units)
            self.write(force=True)

    def finish(self) -> None:
        if self.estimator is not None:
            self.estimator.finish_stage()
            self.estimator.rates.save()
            self.write(force=True)

    def feed_line(self, line: str) -> None:
        if self.estimator is not None and self.estimator.parse_line(line):
            self.write()

    def write(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < 2.0:
            return
        self._last_write = now
        payload = {
            "percent": round(max(0.0, min(100.0, self._percent)), 1),
            "stage": self._stage,
            "detail": self._detail,
            "started_at": self.started_at,
            "updated_at": now,
        }
        if self.estimator is not None:
            snapshot = self.estimator.snapshot()
            payload.update(snapshot)
            if self._percent >= 100.0:
                payload["percent"] = 100.0
                payload["eta_seconds"] = 0.0
            if snapshot.get("stage_key") is None:
                payload["stage"] = self._stage
            current = next((item for item in snapshot["stages"] if item["status"] == "running"), None)
            if current is not None and current["key"] == "classification":
                stage_obj = self.estimator.current
                total = int(round(stage_obj.units)) if stage_obj is not None else 0
                payload["photo_total"] = total
                payload["photo_processed"] = int(round(total * current["percent"] / 100.0))
                payload["photo_percent"] = current["percent"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")


_ACTIVE_PROGRESS: ProgressTracker | None = None


def run(
    cmd, *, summary: RunSummary | None = None, stage: str | None = None,
    allow_failure: bool = False, **kwargs,
):
    print("\n>>>", " ".join(str(c) for c in cmd))
    started_at = time.perf_counter()
    if _ACTIVE_PROGRESS is not None and not ({"capture_output", "stdout", "stderr"} & set(kwargs)):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1, **kwargs)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            _ACTIVE_PROGRESS.feed_line(line)
        sys.stdout.flush()
        result = subprocess.CompletedProcess(cmd, process.wait())
    else:
        result = subprocess.run(cmd, **kwargs)
    if summary and stage:
        summary.record_stage(stage, time.perf_counter() - started_at)
    if result.returncode != 0 and not allow_failure:
        sys.exit(f"Command failed with exit code {result.returncode}: {' '.join(str(c) for c in cmd)}")
    return result


def _build_estimator(
    args, num_images: int, colmap_config: dict, *, resume: bool, dense_max_image_size: int | None = None,
) -> ProgressEstimator:
    dense = colmap_config.get("dense_reconstruction", {})
    camera = colmap_config.get("camera", {})
    alignment = colmap_config.get("alignment", {})
    tiepoints = colmap_config.get("tiepoint_filter", {})
    configured_size = dense.get("max_image_size", camera.get("max_image_size", 3200))
    size = dense_max_image_size if dense_max_image_size is not None else (
        int(camera.get("max_image_size", 3200)) if configured_size == "auto" else int(configured_size)
    )
    iterations = int(dense.get("num_iterations", 3))
    geometric = bool(dense.get("use_geometric_consistency", True))
    strategy = str(alignment.get("matching_strategy", "auto"))
    matching = "exhaustive" if strategy == "exhaustive" or (strategy == "auto" and num_images <= 300) else "spatial"
    automatic = args.control_points_mode == "automatic"
    plan: list[tuple[str, float]] = []
    if not resume:
        plan += [("feature_extraction", num_images),
                 ("matching", stage_units("matching", images=num_images, matching=matching,
                                          spatial_neighbors=int(alignment.get("spatial_max_num_neighbors", 50)))),
                 ("mapping", num_images)]
        if tiepoints.get("enabled", True):
            plan.append(("tiepoint_filtering", num_images))
    if automatic:
        plan.append(("control_points", num_images))
    plan.append(("undistortion", num_images))
    if not args.stop_after_alignment:
        plan += [("dense_stereo", stage_units("dense_stereo", images=num_images, max_image_size=size,
                                              num_iterations=iterations, geometric_consistency=geometric)),
                 ("dense_fusion", num_images), ("classification", num_images)]
        if args.geometry_refinement == "1":
            plan.append(("geometry_refinement", num_images))
    return ProgressEstimator(plan, RateStore(PROJECT_ROOT / "projects" / ".samcloud_stage_rates.json"))


def _dense_size_from_inputs(dense_config: dict, outdir: Path, images_dir: Path) -> tuple[int, str | None]:
    """Resolve numeric or automatic dense resolution from the first project image."""
    configured = dense_config.get("max_image_size", 3200)
    if configured != "auto":
        return int(configured), None
    image_list = outdir / "image_list.txt"
    names = [line.strip() for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError("Cannot choose automatic dense resolution without images")
    image_path = images_dir / names[0]
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except (ImportError, OSError, ValueError):
        try:
            import cv2

            image = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_2)
            if image is None:
                raise ValueError
            height, width = image.shape[:2]
        except (ImportError, ValueError):
            maximum = int(dense_config.get("auto", {}).get("maximum", 3200))
            return maximum, f"Dense size {maximum}px because the first image header could not be read."
    options = dense_config.get("auto", {})
    size, reason = adaptive_dense_size(
        max(width, height), len(names), shutil.disk_usage(outdir).free,
        ratio=float(options.get("ratio", 0.55)),
        minimum=int(options.get("minimum", 2000)),
        maximum=int(options.get("maximum", 3200)),
    )
    return size, reason


def _compact_completed_project(outdir: Path, summary: RunSummary) -> None:
    """Compact only valid project products after successful completion."""
    project_dir = outdir.parent
    if not (project_dir / "project.samcloud.yml").is_file():
        summary.settings["storage"] = {"skipped": "project settings are unavailable"}
        return
    try:
        project = ProjectManager().load(project_dir)
        settings = storage_settings(project.get("settings", {}))
        summary.settings["storage"] = compact_project(outdir, settings)
    except (OSError, ValueError) as error:
        print(f"WARNING: Storage compaction was skipped: {error}")
        summary.settings["storage"] = {"skipped": str(error)}


def _ensure_if_compressed(path: Path) -> None:
    """Restore an optional re-run input only when its compressed copy exists."""
    if path.exists() or path.with_name(path.name + ".gz").exists():
        ensure_available(path)


def _release_fusion_cache(dense_dir: Path, summary: RunSummary) -> None:
    products = dense_dir.parent
    freed = release_after_fusion(dense_dir) if (products.parent / "project.samcloud.yml").is_file() else 0
    if freed:
        print(f"Released {freed / 1024 ** 3:.2f} GB of dense stereo cache.")
    summary.settings["storage_freed_after_fusion_gb"] = round(
        float(summary.settings.get("storage_freed_after_fusion_gb", 0)) + freed / 1024 ** 3, 3,
    )


def _release_sparse_intermediates(sparse_dir: Path, model0: Path) -> None:
    """Discard superseded sparse filtering models once their successor is selected."""
    products = sparse_dir.parent
    if not (products.parent / "project.samcloud.yml").is_file():
        return
    for path in sparse_dir.iterdir() if sparse_dir.is_dir() else ():
        is_intermediate = (
            path.name in {"0", "0_geo"}
            or "_filter_" in path.name
            or "_ba_" in path.name
        )
        if path.is_dir() and is_intermediate and path != model0 and path.name not in {"pre_gcp", "gcp_adjusted"}:
            remove_generated(path, products)


def _write_final_model(sparse_dir: Path, model0: Path) -> None:
    """Record the sparse model that later processing continues to use."""
    (sparse_dir / "final_model.txt").write_text(model0.name + "\n", encoding="utf-8")


def find_colmap_exe(colmap_bin):
    if colmap_bin:
        return colmap_bin
    local = PROJECT_ROOT / "colmap" / "COLMAP.bat"
    if local.exists():
        return str(local)
    found = shutil.which("colmap")
    if found:
        return found
    sys.exit(
        "colmap.exe not found. Either add it to PATH, put a portable COLMAP "
        "release in .\\colmap\\, or pass --colmap-bin explicitly."
    )


def write_image_list(images_dir, out_path):
    """Write image names relative to images_dir for COLMAP's recursive input mode."""
    extensions = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )
    if not image_paths:
        sys.exit(f"No supported images found in {images_dir}")
    out_path.write_text(
        "\n".join(p.relative_to(images_dir).as_posix() for p in image_paths) + "\n",
        encoding="utf-8",
    )
    return len(image_paths)


def _model_point_count(model_path: Path) -> int | None:
    """Read a COLMAP binary point count without requiring COLMAP bindings."""
    points_path = model_path / "points3D.bin"
    try:
        with points_path.open("rb") as file:
            raw = file.read(8)
        return int.from_bytes(raw, "little") if len(raw) == 8 else None
    except OSError:
        return None


def _analyze_model(colmap: str, model_path: Path, summary: RunSummary) -> dict[str, float]:
    """Collect model_analyzer output, tolerating format changes across COLMAP builds."""
    result = run(
        [colmap, "model_analyzer", "--path", str(model_path)],
        summary=summary,
        stage="Tie point analysis",
        capture_output=True,
        text=True,
        allow_failure=True,
    )
    output = getattr(result, "stdout", "") or ""
    error_output = getattr(result, "stderr", "") or ""
    if output:
        print(output.strip())
    stats = parse_model_analyzer_stats(output + "\n" + error_output)
    if "points" not in stats:
        point_count = _model_point_count(model_path)
        if point_count is not None:
            stats["points"] = float(point_count)
    return stats


def _run_tiepoint_filtering(
    colmap: str,
    model_path: Path,
    settings: dict,
    summary: RunSummary,
    *,
    use_pose_priors: bool = False,
    database_path: Path | None = None,
    prior_std_xy_m: float = 3.0,
    prior_std_z_m: float = 5.0,
) -> tuple[Path, list[dict[str, object]]]:
    """Run guarded filtering and bundle adjustment passes in separate model folders."""
    if not settings.get("enabled", True):
        return model_path, []
    passes = settings.get("passes", [])
    if not isinstance(passes, list):
        raise ValueError("tiepoint_filter.passes must be a list")
    current_model = model_path
    statistics: list[dict[str, object]] = []
    for index, pass_settings in enumerate(passes, start=1):
        if not isinstance(pass_settings, dict):
            raise ValueError("Every tiepoint_filter pass must be a mapping")
        before = _analyze_model(colmap, current_model, summary)
        filtered_model = current_model.parent / f"{current_model.name}_filter_{index}"
        adjusted_model = current_model.parent / f"{current_model.name}_ba_{index}"
        filtering, adjustment = tiepoint_pass_commands(
            colmap, current_model, filtered_model, adjusted_model, pass_settings
        )
        filtered_model.mkdir(parents=True, exist_ok=True)
        run(filtering, summary=summary, stage=f"Tie point filtering pass {index}")
        after = _analyze_model(colmap, filtered_model, summary)
        before_points = int(before["points"]) if "points" in before else None
        after_points = int(after["points"]) if "points" in after else None
        entry: dict[str, object] = {
            "pass": index,
            "thresholds": dict(pass_settings),
            "before": before,
            "after_filter": after,
        }
        if not filter_pass_is_safe(before_points, after_points):
            entry["skipped"] = "removed more than 60 percent of points or left fewer than 1000"
            print(f"WARNING: Tie point filtering pass {index} was skipped because it is too destructive.")
            statistics.append(entry)
            continue
        adjusted_model.mkdir(parents=True, exist_ok=True)
        if use_pose_priors:
            if database_path is None:
                raise ValueError("A COLMAP database is required for pose-prior bundle adjustment")
            started_at = time.perf_counter()
            _run_pose_prior_bundle_adjustment(
                filtered_model, adjusted_model, database_path, prior_std_xy_m, prior_std_z_m,
            )
            summary.record_stage(f"Bundle adjustment pass {index}", time.perf_counter() - started_at)
        else:
            run(adjustment, summary=summary, stage=f"Bundle adjustment pass {index}")
        current_model = adjusted_model
        entry["skipped"] = False
        statistics.append(entry)
    return current_model, statistics


def _run_pose_prior_bundle_adjustment(
    input_model: Path,
    output_model: Path,
    database_path: Path,
    prior_std_xy_m: float,
    prior_std_z_m: float,
) -> None:
    """Refine a filtered GPS model without discarding its pose priors."""
    import pycolmap

    reconstruction = pycolmap.Reconstruction(input_model)
    with pycolmap.Database.open(database_path) as database:
        priors = database.read_all_pose_priors()
    if not priors:
        raise ValueError("Pose-prior bundle adjustment requires pose priors in the COLMAP database")
    priors = pose_priors_in_model_frame(reconstruction, priors, fallback_std_xy_m=float(prior_std_xy_m), fallback_std_z_m=float(prior_std_z_m))
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        config.add_image(image_id)
    options = pycolmap.BundleAdjustmentOptions(
        refine_focal_length=True,
        refine_principal_point=True,
        refine_extra_params=True,
    )
    options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    options.ceres.loss_function_scale = 1.0
    prior_options = pycolmap.PosePriorBundleAdjustmentOptions(
        prior_position_fallback_stddev=float(max(prior_std_xy_m, prior_std_z_m)),
    )
    prior_options.ceres.prior_position_loss_function_type = pycolmap.LossFunctionType.CAUCHY
    prior_options.ceres.prior_position_loss_scale = float(max(prior_std_xy_m, prior_std_z_m))
    adjuster = pycolmap.create_pose_prior_bundle_adjuster(
        options, prior_options, config, priors, reconstruction,
    )
    adjuster.solve()
    output_model.mkdir(parents=True, exist_ok=True)
    reconstruction.write(output_model)


def resolve_input_images(args, outdir: Path, parser: argparse.ArgumentParser) -> tuple[Path, bool]:
    """Resolve normal images or prepare virtual perspective images for 360 media."""
    if args.capture_type != "panorama_360":
        if not args.images:
            parser.error("--images is required for drone_gps and camera capture types")
        images_dir = Path(args.images).resolve()
        if not images_dir.is_dir():
            parser.error(f"Image directory was not found: {images_dir}")
        return images_dir, False

    try:
        prepared = prepare_panorama_inputs(
            args.panorama_images or "",
            args.panorama_video or "",
            Path(args.panorama_workdir).resolve() if args.panorama_workdir else outdir / "inputs",
            args.frame_interval,
            args.cubemap_face_size,
        )
    except CaptureValidationError as error:
        parser.error(str(error))
    print(
        "Prepared 360 degree inputs: "
        f"{prepared.panorama_count} panoramas, {prepared.video_frame_count} video frames, "
        f"{prepared.cubemap_image_count} virtual camera images."
    )
    print(f"Panorama source manifest: {prepared.manifest_path}")
    return prepared.cubemap_directory, True


def _resume_automatic_control_points(
    args, outdir: Path, report_dir: Path, summary: RunSummary, progress: ProgressTracker | None,
    colmap_config: dict, dense_max_image_size: int, dense_num_iterations: int,
    dense_gpu_index: int, classified_output: Path, control_points_dir: Path,
) -> None:
    """Continue reviewed automatic GCP projects exactly like a passed automatic run."""
    if not args.images:
        raise ValueError("The original image directory (--images) is required to continue automatic control points")
    images_dir = Path(args.images).resolve()
    image_list_path = outdir / "image_list.txt"
    if progress and progress.estimator is None:
        num_images = len([l for l in image_list_path.read_text(encoding="utf-8").splitlines() if l.strip()]) if image_list_path.is_file() else 0
        progress.estimator = _build_estimator(
            args, num_images, colmap_config, resume=True, dense_max_image_size=dense_max_image_size,
        )
        global _ACTIVE_PROGRESS
        _ACTIVE_PROGRESS = progress
    sparse_dir = outdir / "sparse"
    dense_dir = outdir / "dense"
    dense_config = colmap_config.get("dense_reconstruction", {})
    colmap = find_colmap_exe(args.colmap_bin)
    if progress:
        progress.begin("control_points")
        progress.update(57, "Applying control points")
    adjusted_model = _automatic_control_points(
        sparse_dir / "pre_gcp", sparse_dir / "pre_gcp_txt", images_dir, control_points_dir, args.control_points_crs,
        outdir / "gps_ref.txt", dict(load_global_config("defaults.yml").get("control_points", {})),
        report_dir / "control_points_report.json", rtk=bool(args.rtk),
    )
    if adjusted_model is None:
        if progress:
            progress.finish()
        sys.exit("Control-point quality gate did not pass. Review the control points and optimise again.")
    dense_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.begin("undistortion")
    run([colmap, "image_undistorter",
         "--image_path", str(images_dir),
         "--image_list_path", str(image_list_path),
         "--input_path", str(adjusted_model),
         "--output_path", str(dense_dir),
         "--output_type", "COLMAP",
         "--max_image_size", str(dense_max_image_size)], summary=summary, stage="Preparing dense reconstruction")
    _prepare_detections_for_resume(
        control_points_dir / "detections.json", control_points_dir / "observations.json", adjusted_model, dense_dir / "sparse",
    )
    _clear_stereo_cache(dense_dir)
    if progress:
        progress.begin("dense_stereo")
        progress.update(62, "Building dense cloud")
    run([colmap, "patch_match_stereo", "--workspace_path", str(dense_dir),
         "--PatchMatchStereo.geom_consistency", "1" if dense_config.get("use_geometric_consistency", True) else "0",
         "--PatchMatchStereo.gpu_index", str(dense_gpu_index),
         "--PatchMatchStereo.num_iterations", str(dense_num_iterations),
         "--PatchMatchStereo.max_image_size", str(dense_max_image_size)], summary=summary, stage="Dense stereo")
    if progress:
        progress.begin("dense_fusion")
    run(stereo_fusion_command(colmap, dense_dir, dense_dir / "fused.ply", dense_config.get("fusion", {})),
        summary=summary, stage="Dense fusion")
    _release_fusion_cache(dense_dir, summary)
    sparse_txt = dense_dir / "sparse_txt"
    sparse_txt.mkdir(parents=True, exist_ok=True)
    run([colmap, "model_converter", "--input_path", str(dense_dir / "sparse"), "--output_path", str(sparse_txt),
         "--output_type", "TXT"], summary=summary, stage="Preparing classification")
    _classify_and_finish(args, outdir, dense_dir, sparse_txt, report_dir, summary, progress, classified_output, True)


def _resume_after_control_points(
    args, outdir: Path, report_dir: Path, summary: RunSummary, progress: ProgressTracker | None,
    colmap_config: dict, dense_max_image_size: int, dense_num_iterations: int,
    dense_gpu_index: int, classified_output: Path,
) -> None:
    """Run Phase B without repeating feature extraction, mapping, or undistortion."""
    if progress and progress.estimator is None:
        image_list_path = outdir / "image_list.txt"
        num_images = len([l for l in image_list_path.read_text(encoding="utf-8").splitlines() if l.strip()]) if image_list_path.is_file() else 0
        progress.estimator = _build_estimator(
            args, num_images, colmap_config, resume=True, dense_max_image_size=dense_max_image_size,
        )
        global _ACTIVE_PROGRESS
        _ACTIVE_PROGRESS = progress
    dense_dir = outdir / "dense"
    fusion_config = colmap_config.get("dense_reconstruction", {}).get("fusion", {})
    dense_sparse = dense_dir / "sparse"
    sparse_txt = dense_dir / "sparse_txt"
    images_dir = dense_dir / "images"
    control_points_dir = Path(args.control_points_dir).resolve() if args.control_points_dir else outdir.parent / "control_points"
    if (control_points_dir / "detections.json").is_file() and (outdir / "sparse" / "pre_gcp").is_dir():
        _resume_automatic_control_points(
            args, outdir, report_dir, summary, progress, colmap_config, dense_max_image_size,
            dense_num_iterations, dense_gpu_index, classified_output, control_points_dir,
        )
        return
    missing = [str(path) for path in (dense_sparse, sparse_txt, images_dir) if not path.exists()]
    if missing:
        raise ValueError(
            "Cannot resume after control points; Phase A output is missing: " + ", ".join(missing)
        )
    pristine_sparse = dense_dir / "sparse_pre_gcp"
    pristine_sparse_txt = dense_dir / "sparse_txt_pre_gcp"
    _create_pristine_gcp_backups(dense_sparse, sparse_txt, pristine_sparse, pristine_sparse_txt)
    settings = dict(load_global_config("defaults.yml").get("control_points", {}))
    gps_residuals = None
    gps_georeference = outdir / "georeference.json"
    if gps_georeference.exists():
        try:
            gps_residuals = json.loads(gps_georeference.read_text(encoding="utf-8")).get("residual_m")
        except (OSError, json.JSONDecodeError):
            pass
    georeference, report, transform = georeference_from_control_points(
        pristine_sparse_txt, control_points_dir / "control_points.csv", control_points_dir / "observations.json",
        args.control_points_crs, settings, gps_residuals=gps_residuals,
    )
    save_georeference(outdir / "georeference.json", georeference)
    save_control_point_report(report_dir / "control_points_report.json", report)
    transformed_txt = dense_dir / "sparse_txt_gcp"
    transform_text_model(pristine_sparse_txt, transformed_txt, *transform)
    transformed_sparse = dense_dir / "sparse_gcp"
    transformed_sparse.mkdir(parents=True, exist_ok=True)
    colmap = find_colmap_exe(args.colmap_bin)
    run([colmap, "model_converter", "--input_path", str(transformed_txt), "--output_path", str(transformed_sparse), "--output_type", "BIN"], summary=summary, stage="Applying control points")
    for path in transformed_sparse.iterdir():
        if path.is_file():
            shutil.copy2(path, dense_sparse / path.name)
    shutil.rmtree(transformed_sparse)
    shutil.rmtree(transformed_txt)
    _clear_stereo_cache(dense_dir)
    if progress:
        progress.begin("dense_stereo")
        progress.update(62, "Building dense cloud")
    run([colmap, "patch_match_stereo", "--workspace_path", str(dense_dir),
         "--PatchMatchStereo.geom_consistency", "1" if colmap_config.get("dense_reconstruction", {}).get("use_geometric_consistency", True) else "0", "--PatchMatchStereo.gpu_index", str(dense_gpu_index),
         "--PatchMatchStereo.num_iterations", str(dense_num_iterations),
         "--PatchMatchStereo.max_image_size", str(dense_max_image_size)], summary=summary, stage="Dense stereo")
    dense_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.begin("dense_fusion")
    run(stereo_fusion_command(colmap, dense_dir, dense_dir / "fused.ply", fusion_config), summary=summary, stage="Dense fusion")
    _release_fusion_cache(dense_dir, summary)
    shutil.rmtree(sparse_txt)
    sparse_txt.mkdir(parents=True, exist_ok=True)
    run([colmap, "model_converter", "--input_path", str(dense_sparse), "--output_path", str(sparse_txt), "--output_type", "TXT"], summary=summary, stage="Preparing classification")
    _classify_and_finish(args, outdir, dense_dir, sparse_txt, report_dir, summary, progress, classified_output, True)


def _prepare_detections_for_resume(
    detections_path: Path, observations_path: Path, distorted_model: Path | None = None, undistorted_model: Path | None = None,
) -> None:
    """Expose reviewed detections to the legacy dense-coordinate consumer.

    The original-image detections remain authoritative. This compatibility
    projection lets existing control-point editing projects continue while the
    GCP BA reads the same reviewed entries on the next automatic run.
    """
    detections = load_detections(detections_path)
    if distorted_model is not None and undistorted_model is not None and distorted_model.is_dir() and undistorted_model.is_dir():
        import pycolmap
        populate_undistorted_coordinates(
            pycolmap.Reconstruction(distorted_model), pycolmap.Reconstruction(undistorted_model), detections,
        )
        save_detections(detections_path, detections)
    observations = load_observations(observations_path)
    detected = {
        name: [
            {"image": str(item["image"]), "u": float(item["u_undist"]), "v": float(item["v_undist"])}
            for item in entry.get("observations", []) if item.get("enabled", True)
        ]
        for name, entry in detections.items()
    }
    for name, items in detected.items():
        observations.setdefault(name, items)
    if detected:
        save_observations(observations_path, observations)


def _create_pristine_gcp_backups(
    sparse: Path, sparse_txt: Path, pristine_sparse: Path, pristine_sparse_txt: Path
) -> None:
    """Capture Phase-A model inputs once so revised clicks never double-transform them."""
    if not pristine_sparse.exists():
        shutil.copytree(sparse, pristine_sparse)
    if not pristine_sparse_txt.exists():
        shutil.copytree(sparse_txt, pristine_sparse_txt)


def _image_path_map(images_dir: Path) -> dict[str, Path]:
    """Map COLMAP image names (relative POSIX paths, incl. sub-folders) to files; unique bare names as fallback."""
    files = [path for path in images_dir.rglob("*") if path.is_file()]
    mapping = {path.relative_to(images_dir).as_posix(): path for path in files}
    counts: dict[str, int] = {}
    for path in files:
        counts[path.name] = counts.get(path.name, 0) + 1
    for path in files:
        if counts[path.name] == 1:
            mapping.setdefault(path.name, path)
    return mapping


def _automatic_control_points(
    model_path: Path, model_txt: Path, images_dir: Path, control_points_dir: Path, control_points_crs: str,
    gps_ref_path: Path, settings: dict, report_path: Path, *, rtk: bool,
) -> Path | None:
    """Detect, adjust and transform GPS projects into the GCP-local ENU frame."""
    if not gps_ref_path.is_file():
        report = {"gate": {"passed": False, "reasons": ["manual seeds needed"]}, "points": []}
        save_control_point_report(report_path, report)
        return None
    import pycolmap

    ensure_text_model(model_path, model_txt)

    points = read_control_points(control_points_dir / "control_points.csv")
    georef = fit_enu_to_ecef(model_txt, gps_ref_path)
    scale = float(georef["scale"])
    rotation = np.asarray(georef["rotation"], dtype=float)
    translation = np.asarray(georef["translation"], dtype=float)
    ecef = coordinates_to_ecef(points[["x", "y", "z"]].to_numpy(dtype=float), control_points_crs)
    references = {
        str(row.name): rotation.T @ ((ecef[index] - translation) / scale)
        for index, (_, row) in enumerate(points.set_index("name").iterrows())
    }
    roles = dict(zip(points["name"].astype(str), points["role"].astype(str), strict=True))
    paths = _image_path_map(images_dir)
    detections, detection_statistics = automatic_detection(
        pycolmap.Reconstruction(model_path), references, roles, paths, settings,
        0.03 if rtk else 3.0, 0.05 if rtk else 5.0,
        existing=load_detections(control_points_dir / "detections.json"),
    )
    before = camera_intrinsics(pycolmap.Reconstruction(model_path))
    pose_priors = None
    database_path = model_path.parents[1] / "database.db"
    try:
        ensure_available(database_path)
    except FileNotFoundError:
        pass
    if database_path.is_file():
        with pycolmap.Database.open(database_path) as database:
            pose_priors = database.read_all_pose_priors()
    raw_adjusted = model_path.parent / "gcp_adjusted_raw"
    adjustment: dict[str, object] = {}
    gate: tuple[bool, list[str], dict[str, dict[str, float]]] | None = None
    for iteration in range(int(settings.get("max_gate_iterations", 3))):
        iteration_model = raw_adjusted if iteration == 0 else model_path.parent / f"gcp_adjusted_raw_{iteration + 1}"
        adjustment = run_gcp_bundle_adjustment(
            model_path, iteration_model, detections, references, pose_priors=pose_priors,
            prior_stddev=0.05 if rtk else 5.0,
        )
        adjusted = pycolmap.Reconstruction(iteration_model)
        gate = quality_gate(adjusted, detections, references, settings)
        if gate[0] or iteration + 1 >= int(settings.get("max_gate_iterations", 3)):
            raw_adjusted = iteration_model
            break
    assert gate is not None
    adjusted = pycolmap.Reconstruction(raw_adjusted)
    populate_review_projections(adjusted, detections, references)
    save_detections(control_points_dir / "detections.json", detections)
    if not gate[0]:
        report = build_control_point_report(detections, gate, before, adjustment["intrinsics"])
        report["detection_statistics"] = detection_statistics
        save_control_point_report(report_path, report)
        return None
    centroid = ecef.mean(axis=0)
    enu_rotation = enu_rotation_at_ecef(centroid)
    target_enu = (enu_rotation @ (ecef - centroid).T).T
    enabled_pass = [str(name) for name in points.loc[points["role"] == "pass", "name"] if detections.get(str(name), {}).get("enabled")]
    source = np.asarray([references[name] for name in enabled_pass], dtype=float)
    destination = np.asarray([target_enu[points.index[points["name"] == name][0]] for name in enabled_pass], dtype=float)
    local_scale, local_rotation, local_translation, residuals = fit_similarity(source, destination)
    if float(residuals["max"]) >= 0.001:
        raise ValueError(f"GCP-local frame change residual is {residuals['max']:.6f} m, expected below 0.001 m")
    print(f"GCP-local frame change residual: {residuals['max']:.6f} m")
    adjusted.transform(pycolmap.Sim3d(np.column_stack((local_scale * local_rotation, local_translation))))
    local_model = model_path.parent / "gcp_adjusted"
    local_model.mkdir(parents=True, exist_ok=True)
    adjusted.write(local_model)
    _write_final_model(model_path.parent, local_model)
    for name, entry in detections.items():
        if entry["triangulated_model_xyz"] is not None:
            entry["triangulated_model_xyz"] = (local_scale * local_rotation @ np.asarray(entry["triangulated_model_xyz"], dtype=float) + local_translation).tolist()
    local_references = {name: local_scale * local_rotation @ reference + local_translation for name, reference in references.items()}
    local_recon = pycolmap.Reconstruction(local_model)
    final_gate = quality_gate(local_recon, detections, local_references, settings)
    populate_review_projections(local_recon, detections, local_references)
    leave_one_out = leave_one_out_errors(
        model_path, model_path.parent / "gcp_leave_one_out", detections, references,
        pose_priors=pose_priors, prior_stddev=0.05 if rtk else 5.0,
    )
    leave_one_out = {name: None if error is None else float(local_scale) * error for name, error in leave_one_out.items()}
    report = build_control_point_report(detections, final_gate, before, adjustment["intrinsics"], leave_one_out=leave_one_out)
    report["detection_statistics"] = detection_statistics
    print("Marker | Error [px] | Projections | Status")
    for point in report["points"]:
        error = point["reprojection_rms_px"]
        error_text = "n/a" if error is None or not np.isfinite(error) else f"{error:.3f}"
        print(f"{point['name']} | {error_text} | {point['projections']} | {point['status']}")
    save_detections(control_points_dir / "detections.json", detections)
    report["residual_m"] = residuals
    save_control_point_report(report_path, report)
    if not final_gate[0]:
        return None
    save_georeference(model_path.parents[1] / "georeference.json", {
        "source_frame": "gcp_local_enu", "method": "gcp_bundle_adjustment", "scale": 1.0,
        "rotation": enu_rotation.T.tolist(), "translation": centroid.tolist(), "pairs": len(points),
        "residual_m": residuals, "control_points": {"crs": control_points_crs, "origin_ecef": centroid.tolist()},
    })
    products = model_path.parents[1]
    if (products.parent / "project.samcloud.yml").is_file():
        for path in model_path.parent.glob("gcp_adjusted_raw*"):
            if path != local_model:
                remove_generated(path, products)
    return local_model


def _clear_stereo_cache(dense_dir: Path) -> None:
    """Discard depths/normals because COLMAP may otherwise reuse old geometry.

    COLMAP does not create sub-folders for images stored in sub-directories, so the
    empty cache folders are recreated with the same structure as ``dense/images``.
    """
    stereo = dense_dir / "stereo"
    images = dense_dir / "images"
    for name in ("depth_maps", "normal_maps"):
        cache = stereo / name
        if cache.exists():
            shutil.rmtree(cache)
        cache.mkdir(parents=True, exist_ok=True)
        if images.is_dir():
            for folder in images.rglob("*"):
                if folder.is_dir():
                    (cache / folder.relative_to(images)).mkdir(parents=True, exist_ok=True)


def _classify_and_finish(
    args, outdir: Path, dense_dir: Path, dense_sparse_txt: Path, report_dir: Path,
    summary: RunSummary, progress: ProgressTracker | None, classified_output: Path, georeferenced: bool,
) -> None:
    """Run the unchanged classification, refinement, and optional export tail."""
    classify_script = Path(__file__).resolve().parent / "classify.py"
    geometry_enabled = args.geometry_refinement == "1"
    classified_raw = classified_output.with_name("classified_raw.ply") if geometry_enabled else classified_output
    classified_scores = classified_output.parent / "classified_scores.npz"
    classified_output.parent.mkdir(parents=True, exist_ok=True)
    _ensure_if_compressed(dense_dir / "fused.ply")
    _ensure_if_compressed(dense_dir / "fused.ply.vis")
    if progress:
        progress.begin("classification")
    classify_command = [sys.executable, str(classify_script), "--dense-dir", str(dense_dir),
        "--images-dir", str(dense_dir / "images"), "--model-dir", str(dense_sparse_txt),
        "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"), "--class-set", args.class_set,
        "--min-votes", str(args.min_votes), "--tile-size", str(args.sam_tile_size),
        "--tile-min-overlap", str(args.sam_tile_min_overlap), "--mode", args.classification_mode,
        "--background-threshold", str(args.background_threshold), "--query-threshold", str(args.query_threshold),
        "--summary-file", str(report_dir / "classification_summary.json"), "--out", str(classified_raw)]
    if args.threshold_sweep:
        classify_command.extend(["--threshold-sweep", args.threshold_sweep])
    if geometry_enabled and args.classification_mode == "segearth_ov3":
        classify_command.extend(["--scores-out", str(classified_scores)])
    if args.progress_file:
        classify_command.extend(["--progress-file", str(Path(args.progress_file).with_name("classification_progress.json"))])
    run(classify_command, summary=summary, stage="SAM3 classification")
    if geometry_enabled and georeferenced:
        refine_script = Path(__file__).resolve().parent / "geometry" / "refine.py"
        _ensure_if_compressed(classified_raw)
        if args.classification_mode == "segearth_ov3":
            _ensure_if_compressed(classified_scores)
        if progress:
            progress.begin("geometry_refinement")
        run([sys.executable, str(refine_script), "--input", str(classified_raw), "--output", str(classified_output),
            "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"), "--class-set", args.class_set,
            "--config", str(PROJECT_ROOT / "config" / "geometry.yml"), "--summary-file", str(report_dir / "geometry_summary.json"),
            *(["--scores", str(classified_scores)] if args.classification_mode == "segearth_ov3" else [])], summary=summary, stage="Geometric refinement")
    elif geometry_enabled:
        shutil.copy2(classified_raw, classified_output)
    if args.skip_export:
        if progress:
            progress.finish()
            progress.update(100, "Complete")
        _compact_completed_project(outdir, summary)
        summary.write("Completed")
        return
    run([sys.executable, str(Path(__file__).resolve().parent / "export_las.py"), "--input", str(classified_output),
         "--output", str(outdir / "classified.las"), "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
         "--class-set", args.class_set, "--source-crs", args.source_crs, "--target-crs", args.export_crs,
         "--georeference", str(outdir / "georeference.json")], summary=summary, stage="LAS export")
    if progress:
        progress.finish()
    _compact_completed_project(outdir, summary)
    summary.write("Completed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-type", choices=("drone_gps", "camera", "panorama_360"), default="drone_gps")
    ap.add_argument("--images", help="folder with normal drone or camera photos")
    ap.add_argument("--panorama-images", help="folder with exported 2:1 JPG panoramas")
    ap.add_argument("--panorama-video", help="exported 2:1 MP4 panorama video")
    ap.add_argument("--panorama-workdir", help="project-owned directory for frames, cubemaps, and manifest")
    ap.add_argument("--frame-interval", type=float, default=1.0,
                    help="fixed 360 degree video frame interval in seconds")
    ap.add_argument("--cubemap-face-size", type=int, default=1024,
                    help="side length in pixels for generated cubemap faces")
    ap.add_argument("--outdir", required=True, help="project/output folder")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="minimum agreeing images per point; default 1 keeps single-view classifications")
    ap.add_argument("--sam-tile-size", type=int, default=1008,
                    help="SAM3 input tile size in pixels")
    ap.add_argument("--sam-tile-min-overlap", type=float, default=0.2,
                    help="SAM3 tile overlap fraction")
    ap.add_argument("--classification-mode", choices=("legacy", "segearth_ov3"), default="segearth_ov3",
                    help="SAM3 classification path (default: segearth_ov3; legacy remains available for comparison)")
    ap.add_argument("--background-threshold", type=float, default=0.4,
                    help="SegEarth-OV3 background threshold")
    ap.add_argument("--query-threshold", type=float, default=0.4,
                    help="SegEarth-OV3 decoder query threshold")
    ap.add_argument("--threshold-sweep", default="",
                    help="optional comma-separated SegEarth-OV3 background thresholds")
    ap.add_argument("--class-set", choices=("outdoor", "indoor"), default="outdoor",
                    help="semantic class set used for SAM3 classification")
    ap.add_argument("--geometry-refinement", choices=("0", "1"), default="1",
                    help="1 = apply post-classification height and flatness checks to georeferenced outdoor clouds (default on; skipped for non-georeferenced projects)")
    ap.add_argument("--colmap-bin", default=None)
    ap.add_argument("--gpu", default="1", help="1 = use GPU for colmap steps, 0 = CPU only")
    ap.add_argument("--rtk", action="store_true",
                     help="photos were flown with RTK, GPS EXIF is cm-accurate. "
                          "Without this flag, standard consumer GPS accuracy (~meters) is assumed.")
    ap.add_argument("--gps-max-error", type=float, default=None,
                    help="maximum GPS alignment error in meters. Defaults to 0.1 for --rtk "
                         "and 3.0 for standard GPS.")
    ap.add_argument("--source-crs", default="LOCAL",
                    help="CRS of the reconstructed point cloud before LAS export")
    ap.add_argument("--export-crs", default="LOCAL",
                    help="target CRS of the exported LAS file")
    ap.add_argument(
        "--control-points-mode",
        choices=("none", "after_alignment", "automatic"),
        default="none",
        help="continue without control points, stop for manual points, or detect targets automatically",
    )
    ap.add_argument(
        "--stop-after-alignment",
        action="store_true",
        help="finish after sparse alignment so control points can be added",
    )
    ap.add_argument(
        "--skip-export",
        action="store_true",
        help="leave classified.ply ready for the manual LAS export step",
    )
    ap.add_argument(
        "--classified-output",
        default=None,
        help="path for the classified PLY (defaults to <outdir>/classified.ply)",
    )
    ap.add_argument(
        "--resume-after-control-points",
        action="store_true",
        help="reuse the undistorted dense workspace and continue from marked control points",
    )
    ap.add_argument("--control-points-dir", default=None, help="project control_points directory")
    ap.add_argument("--control-points-crs", default="EPSG:2056+5728")
    ap.add_argument(
        "--progress-file",
        default=None,
        help="path for the dashboard progress JSON file",
    )
    args = ap.parse_args()

    if args.control_points_mode == "after_alignment":
        args.stop_after_alignment = True
    if args.resume_after_control_points and args.stop_after_alignment:
        ap.error("--resume-after-control-points cannot be combined with --stop-after-alignment")

    progress = ProgressTracker(Path(args.progress_file)) if args.progress_file else None
    if progress:
        progress.update(0, "Preparing images")

    if args.gps_max_error is not None and args.gps_max_error <= 0:
        ap.error("--gps-max-error must be greater than zero")
    if args.frame_interval <= 0:
        ap.error("--frame-interval must be greater than zero")
    if args.cubemap_face_size <= 0:
        ap.error("--cubemap-face-size must be greater than zero")
    if args.sam_tile_size not in {768, 1008, 1280}:
        ap.error("--sam-tile-size must be 768, 1008, or 1280")
    if not 0.0 <= args.sam_tile_min_overlap <= 0.8:
        ap.error("--sam-tile-min-overlap must be between 0.00 and 0.80")
    if not 0.0 <= args.background_threshold <= 1.0:
        ap.error("--background-threshold must be between 0 and 1")
    if not 0.0 <= args.query_threshold <= 1.0:
        ap.error("--query-threshold must be between 0 and 1")
    try:
        threshold_sweep = [float(item.strip()) for item in args.threshold_sweep.split(",") if item.strip()]
    except ValueError:
        ap.error("--threshold-sweep must be comma-separated floats")
    if any(not 0.0 <= threshold <= 1.0 for threshold in threshold_sweep):
        ap.error("--threshold-sweep values must be between 0 and 1")
    gps_alignment_max_error = args.gps_max_error if args.gps_max_error is not None else (0.1 if args.rtk else 3.0)
    colmap_config = load_global_config("colmap.yml")
    camera_config = colmap_config.get("camera", {})
    features_config = colmap_config.get("features", {})
    alignment_config = colmap_config.get("alignment", {})
    tiepoint_config = colmap_config.get("tiepoint_filter", {})
    georeferencing_config = colmap_config.get("georeferencing", {})
    dense_config = colmap_config.get("dense_reconstruction", {})
    max_image_size = int(camera_config.get("max_image_size", 4000))
    configured_dense_size = dense_config.get("max_image_size", max_image_size)
    dense_max_image_size = max_image_size if configured_dense_size == "auto" else int(configured_dense_size)
    dense_num_iterations = int(dense_config.get("num_iterations", 3))
    dense_gpu_index = int(dense_config.get("gpu_index", 0))

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    control_points_dir = Path(args.control_points_dir).resolve() if args.control_points_dir else outdir.parent / "control_points"
    if args.control_points_mode == "automatic" and not (control_points_dir / "control_points.csv").is_file():
        print("Automatic control points skipped: control_points/control_points.csv is missing.")
        args.control_points_mode = "none"
    classified_output = (
        Path(args.classified_output).resolve()
        if args.classified_output
        else outdir / "classified.ply"
    )
    report_dir = outdir.parent / "reports"
    summary = RunSummary(outdir, {
        "image_count": None,
        "capture_type": args.capture_type,
        "gps_mode": "RTK" if args.rtk else "Standard GPS" if args.capture_type == "drone_gps" else "Not used",
        "gps_max_error_m": gps_alignment_max_error if args.capture_type == "drone_gps" else "Not used",
        "gpu": args.gpu == "1",
        "dense_max_image_size": dense_max_image_size,
        "dense_num_iterations": dense_num_iterations,
        "alignment_quality": {
            "camera_grouping": camera_config.get("grouping", "sequence_number"),
            "camera_model": None,
            "matching_strategy": None,
            "mapper": None,
            "tiepoint_filter": {"enabled": bool(tiepoint_config.get("enabled", True)), "passes": []},
        },
        "class_set": args.class_set,
        "min_votes": args.min_votes,
        "sam_tile_size": args.sam_tile_size,
        "sam_tile_min_overlap": args.sam_tile_min_overlap,
        "classification_mode": args.classification_mode,
        "background_threshold": args.background_threshold,
        "query_threshold": args.query_threshold,
        "geometry_refinement": args.geometry_refinement == "1",
        "source_crs": args.source_crs,
        "export_crs": args.export_crs,
        "classified_output": str(classified_output),
        "classification_summary": str(report_dir / "classification_summary.json"),
    })
    if args.resume_after_control_points:
        resume_images = Path(args.images).resolve() if args.images else outdir / "dense" / "images"
        dense_max_image_size, dense_reason = _dense_size_from_inputs(dense_config, outdir, resume_images)
        if dense_reason:
            print(dense_reason)
        summary.settings["dense_max_image_size"] = dense_max_image_size
        summary.settings["dense_size_reason"] = dense_reason
        _resume_after_control_points(
            args, outdir, report_dir, summary, progress, colmap_config, dense_max_image_size,
            dense_num_iterations, dense_gpu_index, classified_output,
        )
        return
    images_dir, is_panorama = resolve_input_images(args, outdir, ap)

    db_path = outdir / "database.db"
    sparse_dir = outdir / "sparse"
    dense_dir = outdir / "dense"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)
    image_list_path = outdir / "image_list.txt"
    num_images = write_image_list(images_dir, image_list_path)
    dense_max_image_size, dense_reason = _dense_size_from_inputs(dense_config, outdir, images_dir)
    if dense_reason:
        print(dense_reason)
    summary.settings["dense_max_image_size"] = dense_max_image_size
    summary.settings["dense_size_reason"] = dense_reason
    summary.settings["image_count"] = num_images
    print(f"Using {num_images} supported images from {images_dir}")
    if progress:
        progress.estimator = _build_estimator(
            args, num_images, colmap_config, resume=False, dense_max_image_size=dense_max_image_size,
        )
        global _ACTIVE_PROGRESS
        _ACTIVE_PROGRESS = progress
    image_paths = [images_dir / name for name in image_list_path.read_text(encoding="utf-8").splitlines()]
    auto_rtk = args.capture_type == "drone_gps" and is_dji_rtk_project(image_paths)
    effective_rtk = args.rtk or auto_rtk
    if auto_rtk and not args.rtk:
        print("Detected DJI RTK fixed metadata in at least 90 percent of images; using RTK settings.")
    if args.gps_max_error is None:
        gps_alignment_max_error = 0.1 if effective_rtk else 3.0
    summary.settings["gps_mode"] = "RTK" if effective_rtk else "Standard GPS" if args.capture_type == "drone_gps" else "Not used"
    summary.settings["gps_max_error_m"] = gps_alignment_max_error if args.capture_type == "drone_gps" else "Not used"
    summary.settings["auto_rtk"] = auto_rtk
    if progress:
        progress.update(5, "Preparing images", f"Found {num_images} supported images")

    colmap = find_colmap_exe(args.colmap_bin)
    gps_ref_path = outdir / "gps_ref.txt"
    if args.capture_type == "drone_gps":
        gps_script = Path(__file__).resolve().parent / "extract_gps.py"
        run([sys.executable, str(gps_script),
             "--images-dir", str(images_dir),
             "--out", str(gps_ref_path),
             "--require-all"], summary=summary, stage="Preparing GPS references")

    # 1) Feature extraction
    if progress:
        progress.begin("feature_extraction")
        progress.update(5, "Extracting features")
    camera_model = resolve_camera_model(camera_config.get("camera_model"), is_panorama)
    summary.settings["alignment_quality"]["camera_model"] = camera_model
    if is_panorama:
        feature_command = feature_extractor_command(
            colmap, db_path, images_dir, image_list_path,
            gpu=args.gpu == "1",
            camera_model=camera_model,
            max_image_size=max_image_size,
            max_num_features=int(features_config.get("max_num_features", 16384)),
            estimate_affine_shape=bool(features_config.get("estimate_affine_shape", False)),
            domain_size_pooling=bool(features_config.get("domain_size_pooling", True)),
            single_camera=True,
        )
        feature_result = run(
            feature_command, summary=summary, stage="Feature extraction",
            allow_failure=bool(features_config.get("domain_size_pooling", True)),
        )
        if getattr(feature_result, "returncode", 0) and bool(features_config.get("domain_size_pooling", True)):
            print("WARNING: COLMAP rejected domain-size pooling; retrying feature extraction without it.")
            feature_command[feature_command.index("--SiftExtraction.domain_size_pooling") + 1] = "0"
            run(feature_command, summary=summary, stage="Feature extraction")
    else:
        grouping = str(camera_config.get("grouping", "sequence_number"))
        names = [line.strip() for line in image_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        groups = assign_camera_groups(names, grouping)
        group_counts = extract_features_by_group(
            lambda command: run(command, summary=summary, stage="Feature extraction"),
            colmap,
            db_path,
            images_dir,
            names,
            groups,
            outdir,
            gpu=args.gpu == "1",
            camera_model=camera_model,
            max_image_size=max_image_size,
            max_num_features=int(features_config.get("max_num_features", 16384)),
            estimate_affine_shape=bool(features_config.get("estimate_affine_shape", False)),
            domain_size_pooling=bool(features_config.get("domain_size_pooling", True)),
        )
        for group, count in sorted(group_counts.items()):
            print(f"Camera group {group + 1:02d}: {count} images")
        summary.settings["alignment_quality"]["camera_groups"] = group_counts

    has_pose_priors = False
    if args.capture_type == "drone_gps":
        prior_count = write_pose_priors_from_gps_ref(
            db_path, gps_ref_path, images_dir,
            standard_gps_std_xy_m=float(georeferencing_config.get("standard_gps_prior_std_xy_m", 3.0)),
            standard_gps_std_z_m=float(georeferencing_config.get("standard_gps_prior_std_z_m", 5.0)),
        )
        print(f"Wrote {prior_count} GPS pose priors to the COLMAP database.")
        prior_count = pose_prior_count(db_path)
        has_pose_priors = prior_count == num_images
        if not has_pose_priors:
            print("WARNING: GPS pose priors are incomplete; using the incremental mapper.")
    if progress:
        progress.update(18, "Extracting features")

    # 2) Video panoramas use sequential matching. A photo directory additionally
    # uses exhaustive matching so it can connect to the video frames as well.
    if progress:
        progress.begin("matching")
    if is_panorama and args.panorama_video:
        if progress:
            progress.update(18, "Matching images")
        run([colmap, "sequential_matcher",
             "--database_path", str(db_path),
             "--FeatureMatching.use_gpu", args.gpu,
             "--SequentialMatching.overlap", "10"], summary=summary, stage="Image matching")
        if progress:
            progress.update(25, "Matching images")
    if not is_panorama or args.panorama_images:
        if progress:
            progress.update(18 if not is_panorama else 25, "Matching images")
        matching_strategy = select_matching_strategy(
            str(alignment_config.get("matching_strategy", "auto")), num_images, has_pose_priors,
        ) if args.capture_type == "drone_gps" and not is_panorama else "exhaustive"
        summary.settings["alignment_quality"]["matching_strategy"] = matching_strategy
        run(matching_command(
            colmap, db_path, gpu=args.gpu == "1", strategy=matching_strategy,
            max_num_neighbors=int(alignment_config.get("spatial_max_num_neighbors", 50)),
            max_distance=float(alignment_config.get("spatial_max_distance_m", 200)),
        ), summary=summary, stage="Image matching")
        if progress:
            progress.update(32, "Matching images")

    # 3) Sparse reconstruction (SfM)
    if progress:
        progress.begin("mapping")
        progress.update(32, "Aligning cameras")
    use_pose_priors = (
        args.capture_type == "drone_gps"
        and str(alignment_config.get("mapper", "pose_prior")) == "pose_prior"
        and has_pose_priors
    )
    if args.capture_type == "drone_gps" and not use_pose_priors:
        print("GPS pose-prior mapper is unavailable; falling back to the incremental mapper.")
    summary.settings["alignment_quality"]["mapper"] = "pose_prior" if use_pose_priors else "incremental"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    run(mapper_command(
        colmap, db_path, images_dir, sparse_dir,
        use_pose_priors=use_pose_priors,
        rtk=effective_rtk,
        georeferencing={
            "rtk_prior_std_xy_m": float(georeferencing_config.get("rtk_prior_std_xy_m", 0.03)),
            "rtk_prior_std_z_m": float(georeferencing_config.get("rtk_prior_std_z_m", 0.05)),
            "standard_gps_prior_std_xy_m": float(georeferencing_config.get("standard_gps_prior_std_xy_m", 3.0)),
            "standard_gps_prior_std_z_m": float(georeferencing_config.get("standard_gps_prior_std_z_m", 5.0)),
        },
        per_image_covariances=pose_prior_covariance_count(db_path) == num_images,
    ), summary=summary, stage="Sparse reconstruction")
    if progress:
        progress.update(52, "Aligning cameras")

    model0 = sparse_dir / "0"
    if not model0.exists():
        sys.exit("colmap mapper did not produce a reconstruction (sparse/0 missing). Check image overlap/quality.")

    # Keep GPS priors active during every filtering-pass adjustment. Otherwise
    # ordinary BA can reintroduce the drift that the pose-prior mapper removed.
    if use_pose_priors:
        if progress:
            progress.begin("tiepoint_filtering")
        model0, tiepoint_statistics = _run_tiepoint_filtering(
            colmap, model0, tiepoint_config, summary,
            use_pose_priors=True,
            database_path=db_path,
            prior_std_xy_m=float(georeferencing_config.get(
                "rtk_prior_std_xy_m" if effective_rtk else "standard_gps_prior_std_xy_m", 0.03 if effective_rtk else 3.0,
            )),
            prior_std_z_m=float(georeferencing_config.get(
                "rtk_prior_std_z_m" if effective_rtk else "standard_gps_prior_std_z_m", 0.05 if effective_rtk else 5.0,
            )),
        )
        summary.settings["alignment_quality"]["tiepoint_filter"]["passes"].extend(tiepoint_statistics)

    # 3b) GPS alignment only applies to drone images. Other capture types retain
    # a local frame until control points or another georeferencing step is used.
    georeferenced = False
    if args.capture_type == "drone_gps":
        if progress:
            progress.update(55, "Applying GPS alignment")

        print(
            "Using GPS alignment maximum error: "
            f"{gps_alignment_max_error:.3f} m "
            f"({'RTK' if effective_rtk else 'standard GPS'})"
        )

        model0_geo = sparse_dir / "0_geo"
        model0_geo.mkdir(parents=True, exist_ok=True)
        print("COLMAP model_aligner will log the ENU similarity transform it applies below.")
        run([colmap, "model_aligner",
             "--input_path", str(model0),
             "--output_path", str(model0_geo),
             "--ref_images_path", str(gps_ref_path),
             "--ref_is_gps", "1",
             "--alignment_type", "enu",
             "--alignment_max_error", str(gps_alignment_max_error)], summary=summary, stage="GPS alignment")
        if progress:
            progress.update(57, "Applying GPS alignment")

        if not model0_geo.exists() or not any(model0_geo.iterdir()):
            sys.exit(
                "model_aligner produced no output. Check the gps_ref.txt and the "
                "colmap model_aligner flags for your installed COLMAP version."
            )
        model0 = model0_geo
        georeferenced = True
    else:
        print("GPS alignment skipped. The reconstruction remains in LOCAL coordinates.")

    if not use_pose_priors:
        if progress:
            progress.begin("tiepoint_filtering")
        model0, tiepoint_statistics = _run_tiepoint_filtering(
            colmap, model0, tiepoint_config, summary
        )
        summary.settings["alignment_quality"]["tiepoint_filter"]["passes"].extend(tiepoint_statistics)

    # Preserve the distorted sparse model before target observations or GCP BA.
    # Existing snapshots are deliberately retained so a review re-run starts
    # from exactly the same Phase-A reconstruction.
    if args.control_points_mode == "automatic":
        pristine_distorted = sparse_dir / "pre_gcp"
        if not pristine_distorted.exists():
            shutil.copytree(model0, pristine_distorted)
        pristine_text = sparse_dir / "pre_gcp_txt"
        pristine_text.mkdir(parents=True, exist_ok=True)
        run([colmap, "model_converter", "--input_path", str(model0), "--output_path", str(pristine_text), "--output_type", "TXT"], summary=summary, stage="Preparing automatic control points")
        if progress:
            progress.begin("control_points")
        automatic_model = _automatic_control_points(
            pristine_distorted, pristine_text, images_dir, control_points_dir, args.control_points_crs,
            gps_ref_path, dict(load_global_config("defaults.yml").get("control_points", {})),
            report_dir / "control_points_report.json", rtk=effective_rtk,
        )
        if automatic_model is None:
            args.stop_after_alignment = True
            print("Automatic control-point quality gate did not pass. Waiting for control-point review.")
        else:
            model0 = automatic_model

    _release_sparse_intermediates(sparse_dir, model0)
    _write_final_model(sparse_dir, model0)

    # 4) Undistort for dense stereo, this also writes a PINHOLE sparse model
    #    we can read as plain text for the camera poses.
    if progress:
        progress.begin("undistortion")
        progress.update(57, "Preparing dense reconstruction")
    dense_dir.mkdir(parents=True, exist_ok=True)
    run([colmap, "image_undistorter",
         "--image_path", str(images_dir),
         "--image_list_path", str(image_list_path),
         "--input_path", str(model0),
         "--output_path", str(dense_dir),
         "--output_type", "COLMAP",
         "--max_image_size", str(dense_max_image_size)], summary=summary, stage="Preparing dense reconstruction")
    if args.control_points_mode == "automatic":
        detections_path = control_points_dir / "detections.json"
        if detections_path.exists():
            _prepare_detections_for_resume(detections_path, control_points_dir / "observations.json", model0, dense_dir / "sparse")
    if progress:
        progress.update(62, "Preparing dense reconstruction")

    # Phase A makes the actual marker workspace available: undistorted images
    # and a PINHOLE TEXT model. It intentionally stops before dense stereo.
    if args.stop_after_alignment:
        dense_sparse_txt = dense_dir / "sparse_txt"
        dense_sparse_txt.mkdir(parents=True, exist_ok=True)
        run([colmap, "model_converter",
             "--input_path", str(dense_dir / "sparse"),
             "--output_path", str(dense_sparse_txt),
             "--output_type", "TXT"], summary=summary, stage="Preparing control points")
        if georeferenced:
            try:
                georeference = fit_enu_to_ecef(dense_sparse_txt, gps_ref_path)
                save_georeference(outdir / "georeference.json", georeference)
            except Exception as error:
                print(f"WARNING: Could not create GPS georeference: {error}")
        if progress:
            progress.finish()
            progress.update(100, "Waiting for control points")
        print("Undistortion finished. Waiting for control points before dense reconstruction.")
        summary.write("Waiting for control points")
        return

    # 5) Dense stereo
    if progress:
        progress.begin("dense_stereo")
        progress.update(62, "Building dense cloud")
    run([colmap, "patch_match_stereo",
         "--workspace_path", str(dense_dir),
         "--PatchMatchStereo.geom_consistency", "1" if dense_config.get("use_geometric_consistency", True) else "0",
         "--PatchMatchStereo.gpu_index", str(dense_gpu_index),
         "--PatchMatchStereo.num_iterations", str(dense_num_iterations),
         "--PatchMatchStereo.max_image_size", str(dense_max_image_size)], summary=summary, stage="Dense stereo")
    if progress:
        progress.update(78, "Building dense cloud")

    # 6) Fusion -> fused.ply + fused.ply.vis
    if progress:
        progress.begin("dense_fusion")
        progress.update(78, "Fusing dense cloud")
    dense_dir.mkdir(parents=True, exist_ok=True)
    run(stereo_fusion_command(
        colmap, dense_dir, dense_dir / "fused.ply", dense_config.get("fusion", {})
    ), summary=summary, stage="Dense fusion")
    _release_fusion_cache(dense_dir, summary)
    if progress:
        progress.update(84, "Fusing dense cloud")

    # 7) Export the dense/sparse model as TEXT so classify.py can read poses
    #    without any COLMAP python bindings.
    dense_sparse_txt = dense_dir / "sparse_txt"
    dense_sparse_txt.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.update(84, "Preparing classification")
    run([colmap, "model_converter",
         "--input_path", str(dense_dir / "sparse"),
         "--output_path", str(dense_sparse_txt),
         "--output_type", "TXT"], summary=summary, stage="Preparing classification")
    georeference_path = outdir / "georeference.json"
    if georeferenced:
        started_at = time.perf_counter()
        try:
            georeference = fit_enu_to_ecef(dense_sparse_txt, gps_ref_path)
            save_georeference(georeference_path, georeference)
        except Exception as error:
            print(f"WARNING: Could not create GPS georeference: {error}")
        else:
            summary.record_stage("Georeference", time.perf_counter() - started_at)
            summary.settings["georeference_residual_m"] = georeference["residual_m"]
            print(f"Georeference residuals: {format_residuals(georeference)}")
    if progress:
        progress.update(87, "Preparing classification")

    # 8) SAM3 classification + multi-view voting
    classify_script = Path(__file__).resolve().parent / "classify.py"
    geometry_enabled = args.geometry_refinement == "1"
    classified_raw = classified_output.with_name("classified_raw.ply") if geometry_enabled else classified_output
    classified_scores = classified_output.parent / "classified_scores.npz"
    classified_output.parent.mkdir(parents=True, exist_ok=True)
    _ensure_if_compressed(dense_dir / "fused.ply")
    _ensure_if_compressed(dense_dir / "fused.ply.vis")
    if progress:
        progress.begin("classification")
        progress.update(87, "Classifying point cloud")
    classify_command = [sys.executable, str(classify_script),
         "--dense-dir", str(dense_dir),
         "--images-dir", str(dense_dir / "images"),
         "--model-dir", str(dense_sparse_txt),
         "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
         "--class-set", args.class_set,
         "--min-votes", str(args.min_votes),
         "--tile-size", str(args.sam_tile_size),
         "--tile-min-overlap", str(args.sam_tile_min_overlap),
         "--mode", args.classification_mode,
         "--background-threshold", str(args.background_threshold),
         "--query-threshold", str(args.query_threshold),
         "--summary-file", str(report_dir / "classification_summary.json"),
         "--out", str(classified_raw)]
    if args.threshold_sweep:
        classify_command.extend(["--threshold-sweep", args.threshold_sweep])
    if geometry_enabled and args.classification_mode == "segearth_ov3":
        classify_command.extend(["--scores-out", str(classified_scores)])
    if args.progress_file:
        classify_command.extend(["--progress-file", str(Path(args.progress_file).with_name("classification_progress.json"))])
    run(classify_command, summary=summary, stage="SAM3 classification")
    if progress:
        progress.update(98, "Classifying point cloud")

    if geometry_enabled:
        geometry_summary = report_dir / "geometry_summary.json"
        if georeferenced:
            refine_script = Path(__file__).resolve().parent / "geometry" / "refine.py"
            _ensure_if_compressed(classified_raw)
            if args.classification_mode == "segearth_ov3":
                _ensure_if_compressed(classified_scores)
            if progress:
                progress.begin("geometry_refinement")
                progress.update(98, "Geometric refinement")
            run([
                sys.executable, str(refine_script),
                "--input", str(classified_raw),
                "--output", str(classified_output),
                "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
                "--class-set", args.class_set,
                "--config", str(PROJECT_ROOT / "config" / "geometry.yml"),
                "--summary-file", str(geometry_summary),
                *(["--scores", str(classified_scores)]
                  if args.classification_mode == "segearth_ov3" else []),
                *(["--progress-file", str(Path(args.progress_file).with_name("geometry_progress.json"))] if args.progress_file else []),
            ], summary=summary, stage="Geometric refinement")
        else:
            print("Geometric refinement skipped: reconstruction is not georeferenced.")
            shutil.copy2(classified_raw, classified_output)
            geometry_summary.parent.mkdir(parents=True, exist_ok=True)
            geometry_summary.write_text(json.dumps({"skipped": "not georeferenced"}, indent=2), encoding="utf-8")

    if args.skip_export:
        if progress:
            progress.finish()
            progress.update(100, "Complete")
        _compact_completed_project(outdir, summary)
        summary.write("Completed")
        print(f"Finished. Classified point cloud ready for manual export: {classified_output}")
        return

    # 9) Export a standards-compliant LAS file with RGB and class IDs.
    las_script = Path(__file__).resolve().parent / "export_las.py"
    if progress:
        progress.update(98, "Exporting LAS")
    run([sys.executable, str(las_script),
         "--input", str(classified_output),
         "--output", str(outdir / "classified.las"),
         "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
         "--class-set", args.class_set,
         "--source-crs", args.source_crs,
         "--target-crs", args.export_crs,
         *(["--georeference", str(georeference_path)] if georeference_path.is_file() else [])],
        summary=summary, stage="LAS export")
    if progress:
        progress.finish()
        progress.update(100, "Complete")

    _compact_completed_project(outdir, summary)
    summary.write("Completed")
    print(f"\nFinished. Classified point cloud: {classified_output}")
    print(f"LAS with class IDs: {outdir / 'classified.las'}")


if __name__ == "__main__":
    main()
