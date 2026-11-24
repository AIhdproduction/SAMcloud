import argparse
import copy
import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import NamedTuple

import numpy as np

from samcloud.acquisition import CaptureValidationError, prepare_panorama_inputs
from samcloud.colmap.alignment import (
    adaptive_dense_size,
    extract_features_by_group,
    feature_extractor_command,
    filter_pass_is_safe,
    mapper_command,
    matching_command,
    panorama_matching_command,
    parse_model_analyzer_stats,
    pose_prior_covariance_count,
    pose_prior_count,
    resolve_camera_model,
    select_panorama_matching_strategy,
    select_matching_strategy,
    dense_profile,
    patch_match_arguments,
    stereo_fusion_command,
    tiepoint_pass_commands,
    write_panorama_rig_config,
    write_pose_priors_from_gps_ref,
)
from samcloud.colmap.camera_groups import assign_camera_groups
from samcloud.colmap.gravity import align_camera_model, align_sparse_model
from samcloud.colmap.plane_fill import fill_classified_cloud
from samcloud.core.configuration import load_global_config
from samcloud.core.project_manager import ProjectManager
from samcloud.core.storage import compact_project, ensure_available, ensure_text_model, release_after_fusion, remove_generated, storage_settings
from samcloud.engine.run_summary import RunSummary
from samcloud.engine.fine_terms import LABELS_NAME, SOURCE_NAME, attach_fine_terms, snapshot_fine_terms
from samcloud.engine.instances import SOURCE_NAME as INSTANCES_SOURCE_NAME, attach_instances, snapshot_instances
from samcloud.engine.extract_gps import is_dji_rtk_project
from samcloud.control_points.georeference import coordinates_to_ecef, enu_rotation_at_ecef, gcp_georeference_from_ecef, georeference_from_control_points, save_control_point_report
from samcloud.control_points.model_transform import transform_text_model
from samcloud.control_points.repository import APRILTAG_CSV_NAME, load_detections, load_observations, read_apriltag_coordinates, read_control_points, save_detections, save_observations
from samcloud.control_points.apriltags import (
    MIN_GEOREF_TAGS, WARNING_GEOREF_UNAVAILABLE, TagGeorefUnavailable, apply_scale, build_tag_report, detect_apriltags,
    load_tag_detections, save_tag_detections, save_tag_report, tag_detection_entries, tag_edge_lengths, tag_scale,
    tags_are_non_collinear, transform_tag_report, triangulate_tags,
)
from samcloud.control_points.auto_detect import _similarity, automatic_detection, populate_undistorted_coordinates, triangulate_observations
from samcloud.control_points.adjustment import build_control_point_report, camera_intrinsics, leave_one_out_errors, populate_review_projections, pose_priors_in_model_frame, quality_gate, run_gcp_bundle_adjustment
from samcloud.pointcloud.georeference import (
    fit_enu_to_ecef, fit_similarity, format_residuals, gcp_georeference_for_cloud_frame,
    georeference_is_control_point_based, load_georeference, save_georeference,
)
from samcloud.engine.eta import ProgressEstimator, RateStore, stage_units
from samcloud.core.project_manager import LOCAL_CRS_GPS_MESSAGE


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

    def report_prepare(self, done: int, total: int, detail: str = "") -> None:
        """Progress callback of the image preparation (frame extraction and cubemap faces)."""
        if self.estimator is None:
            return
        self.estimator.update(done, total)
        self._detail = detail
        self.write(force=total > 0 and done >= total)

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
    prepare_units: float | None = None,
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
    capture_type = getattr(args, "capture_type", "drone_gps")
    faces = 1
    if capture_type == "panorama_360":
        from samcloud.colmap.alignment import PANORAMA_FACES_PER_FRAME, PANORAMA_NEIGHBORING_FRAMES, select_panorama_matching_strategy

        faces = PANORAMA_FACES_PER_FRAME
        matching = select_panorama_matching_strategy(max(num_images, 1))
        overlap = PANORAMA_FACES_PER_FRAME * PANORAMA_NEIGHBORING_FRAMES
    else:
        matching = "exhaustive" if strategy == "exhaustive" or (strategy == "auto" and num_images <= 300) else "spatial"
        overlap = 60
    automatic = args.control_points_mode == "automatic"
    plan: list[tuple[str, float]] = []
    if not resume:
        if prepare_units is not None:
            plan.append(("prepare_images", prepare_units))
        plan += [("feature_extraction", num_images),
                 ("matching", stage_units("matching", images=num_images, matching=matching,
                                          spatial_neighbors=int(alignment.get("spatial_max_num_neighbors", 50)),
                                          faces_per_frame=faces, sequential_overlap=overlap)),
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
        if _depth_prior_settings(args) is not None:
            plan.append(("depth_prior", num_images))
        if _depth_fill_settings(args) is not None:
            plan.append(("depth_fill", num_images))
        if _cleaning_settings(args) is not None:
            plan.append(("point_cleaning", num_images))
        if _surface_fusion_settings(args) is not None:
            plan.append(("surface_fusion", num_images))
        if _plane_fill_settings(args) is not None:
            plan.append(("plane_fill", num_images))
        if _superpoint_vote_settings(args) is not None:
            plan.append(("superpoint_vote", num_images))
        if args.geometry_refinement == "1":
            plan.append(("geometry_refinement", num_images))
    return ProgressEstimator(
        plan, RateStore(PROJECT_ROOT / "projects" / ".samcloud_stage_rates.json", profile=capture_type),
    )


def _plan_geometry_refinement(progress: "ProgressTracker | None", georeferenced: bool) -> None:
    """Drop geometric refinement from the ETA as soon as it is known to be skipped (not georeferenced).

    Only the estimate changes; whether refinement runs is still decided at the end of step 6.
    """
    if georeferenced or progress is None or progress.estimator is None:
        return
    progress.estimator.skip_stage("geometry_refinement")
    progress.write(force=True)


def _may_be_georeferenced(outdir: Path) -> bool:
    """Georeference state a resume run starts from (GPS reference to fit, or an AprilTag survey frame)."""
    return (outdir / "gps_ref.txt").is_file() or _local_tag_georeferenced(outdir)


def _panorama_source_width(args) -> int | None:
    """Width of the first equirectangular source (photo header or video stream), None when unknown."""
    from samcloud.acquisition.panorama import find_panorama_images

    if getattr(args, "panorama_images", None):
        try:
            photos = find_panorama_images(Path(args.panorama_images))
            if photos:
                from PIL import Image

                with Image.open(photos[0]) as image:
                    return int(image.size[0])
        except Exception:  # noqa: BLE001 - only an estimate
            pass
    if getattr(args, "panorama_video", None):
        try:
            import cv2

            from samcloud.acquisition.video import PANORAMA_VIDEO_EXTENSIONS, find_videos

            first_video = next(iter(find_videos(args.panorama_video, PANORAMA_VIDEO_EXTENSIONS)), None)
            if first_video is None and not Path(args.panorama_video).is_dir():
                first_video = Path(args.panorama_video)
            width = 0
            if first_video is not None:
                capture = cv2.VideoCapture(str(first_video))
                try:
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if capture.isOpened() else 0
                finally:
                    capture.release()
            if width > 0:
                return width
        except Exception:  # noqa: BLE001 - only an estimate
            pass
    return None


def _expected_input_long_side(args) -> int | None:
    """Long side of the images COLMAP will see, estimated before the inputs are prepared."""
    if getattr(args, "capture_type", None) == "panorama_360":
        from samcloud.acquisition.panorama import native_face_size

        requested = int(getattr(args, "cubemap_face_size", 0) or 0)
        width = _panorama_source_width(args)
        if width:
            native = native_face_size(width)
            return native if requested <= 0 else min(requested, native)
        return requested if requested > 0 else None
    if getattr(args, "video", None):
        # the frames folder is empty before extraction: the video header tells the frame size
        try:
            import cv2

            from samcloud.acquisition.video import CAMERA_VIDEO_EXTENSIONS, find_videos

            first_video = next(iter(find_videos(args.video, CAMERA_VIDEO_EXTENSIONS)), None)
            if first_video is None:
                return None
            capture = cv2.VideoCapture(str(first_video))
            try:
                if not capture.isOpened():
                    return None
                longest = int(max(capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            finally:
                capture.release()
            return longest if longest > 0 else None
        except Exception:  # noqa: BLE001 - only an estimate
            return None
    if not getattr(args, "images", None):
        return None
    try:
        extensions = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
        directory = Path(args.images)
        first = next((path for path in sorted(directory.iterdir())
                      if path.is_file() and path.suffix.lower() in extensions), None)
        if first is None:
            first = next((path for path in sorted(directory.rglob("*"))
                          if path.is_file() and path.suffix.lower() in extensions), None)
        if first is None:
            return None
        from PIL import Image

        with Image.open(first) as image:
            return int(max(image.size))
    except Exception:  # noqa: BLE001 - only an estimate
        return None


def _planned_dense_size(
    dense_config: dict, fallback: int, long_side: int | None, image_count: int, outdir: Path,
    capture_type: str | None = None,
) -> int:
    """Dense max image size for the initial ETA plan, on the same basis as ``_dense_size_from_inputs``.

    The real size is resolved from the prepared images; before that, ``auto`` uses the expected input
    long side (cubemap face size for 360 inputs) so dense stereo is not planned at full camera resolution.
    """
    from samcloud.colmap.alignment import dense_size_settings

    resolved = dense_size_settings({**dense_config, "max_image_size": dense_config.get("max_image_size", fallback)}, capture_type)
    configured = resolved["max_image_size"]
    if configured != "auto":
        return int(configured)
    options = resolved["auto"]
    maximum = int(options.get("maximum", 3200))
    if not long_side:
        return min(int(fallback), maximum)
    try:
        free_bytes = shutil.disk_usage(outdir).free
    except OSError:
        free_bytes = 1 << 62
    size, _reason = adaptive_dense_size(
        int(long_side), max(int(image_count), 1), free_bytes,
        ratio=float(options.get("ratio", 0.55)),
        minimum=int(options.get("minimum", 2000)),
        maximum=maximum,
    )
    return size


def _initial_estimator(args, colmap_config: dict, outdir: Path, fallback_dense_size: int) -> ProgressEstimator:
    """Estimator of a fresh run, built before the input images are prepared (estimated counts and sizes)."""
    prepare_units, estimated_images = _estimate_prepare_work(args)
    dense_size = _planned_dense_size(
        colmap_config.get("dense_reconstruction", {}), fallback_dense_size,
        _expected_input_long_side(args), estimated_images, outdir, getattr(args, "capture_type", None),
    )
    return _build_estimator(
        args, estimated_images, colmap_config, resume=False,
        dense_max_image_size=dense_size, prepare_units=prepare_units,
    )


def _dense_size_from_inputs(
    dense_config: dict, outdir: Path, images_dir: Path, capture_type: str | None = None,
) -> tuple[int, str | None]:
    """Resolve numeric or automatic dense resolution from the first project image.

    ``capture_type`` applies an optional ``capture_profiles.<type>.dense_size`` override.
    """
    from samcloud.colmap.alignment import dense_size_settings

    resolved = dense_size_settings(dense_config, capture_type)
    configured = resolved["max_image_size"]
    dense_config = {**dense_config, "max_image_size": configured, "auto": resolved["auto"]}
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


LOCAL_TAGS_FRAME = "local_tags"


def _is_local_crs(crs: str | None) -> bool:
    """Local (non-geodetic) survey system: coordinates are metres in the target frame itself."""
    return str(crs or "").strip().upper() == "LOCAL"


def _local_tag_georeferenced(outdir: Path) -> bool:
    """True when the passed AprilTag step placed the model into local survey coordinates (no georeference.json)."""
    try:
        report = json.loads((outdir.parent / "reports" / "control_points_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(report, dict) and report.get("source_frame") == LOCAL_TAGS_FRAME


def _is_georeferenced(outdir: Path) -> bool:
    return (outdir / "georeference.json").is_file() or _local_tag_georeferenced(outdir)


def _write_camera_centre_georeference(
    model_dir: Path,
    gps_ref_path: Path,
    georeference_path: Path,
    *,
    automatic_model_used: bool = False,
) -> dict | None:
    """Write a GPS fit, or express a retained GCP solution in the cloud frame."""
    if georeference_is_control_point_based(georeference_path):
        gcp_model = georeference_path.parent / "sparse" / "gcp_adjusted"
        if gcp_model.is_dir():
            georeference = gcp_georeference_for_cloud_frame(
                load_georeference(georeference_path), model_dir, gcp_model,
            )
            save_georeference(georeference_path, georeference)
            return georeference
        print("keeping control-point georeference")
        return None
    if automatic_model_used:
        print("keeping control-point georeference")
        return None
    georeference = fit_enu_to_ecef(model_dir, gps_ref_path, _configured_height_reference())
    save_georeference(georeference_path, georeference)
    return georeference


class TagStepResult(NamedTuple):
    """Outcome of the AprilTag step: model to continue with, report, and georeference state."""

    model: Path
    report: dict
    georeferenced: bool = False
    gate_failed: bool = False


def _apriltag_step(
    model0: Path, sparse_dir: Path, images_dir: Path, control_points_dir: Path, report_dir: Path,
    *, outlier_pct: float, crs: str = "EPSG:2056+5728", settings: dict | None = None,
) -> TagStepResult:
    """Georeference from >=3 surveyed AprilTags, otherwise scale the sparse model from tag sizes."""
    import pycolmap

    reconstruction = pycolmap.Reconstruction(model0)
    image_map = _image_path_map(images_dir)
    paths = {
        image.name: image_map[image.name]
        for image in reconstruction.images.values() if image.name in image_map
    }
    detections_path = control_points_dir / "apriltag_detections.json"
    detections = None
    if detections_path.is_file():
        try:
            cached = load_tag_detections(detections_path)
            if set(paths) <= set(cached):
                detections = cached
        except (ValueError, KeyError, OSError):
            detections = None
    if detections is None:
        detections = detect_apriltags(paths)
        save_tag_detections(detections_path, detections)
    geometries = triangulate_tags(reconstruction, detections)
    result = tag_scale(geometries, outlier_pct=outlier_pct)
    report_path = report_dir / "apriltag_report.json"
    gate_failed = False
    if (control_points_dir / APRILTAG_CSV_NAME).is_file():
        if settings is None:
            settings = dict(load_global_config("defaults.yml").get("control_points", {}))
        pristine = sparse_dir / "pre_gcp"
        if not pristine.exists():
            shutil.copytree(model0, pristine)
        try:
            adjusted = _apriltag_control_points(
                pristine, sparse_dir / "pre_gcp_txt", images_dir, control_points_dir, crs, settings,
                report_dir / "control_points_report.json", geometries, detections,
            )
        except TagGeorefUnavailable as error:
            print(f"AprilTags: georeferencing unavailable ({error}); scale only.")
            result = dataclasses.replace(result, warnings=[*result.warnings, WARNING_GEOREF_UNAVAILABLE])
        else:
            if adjusted is not None:
                report = _georef_tag_report(adjusted, result, geometries, detections, report_dir / "control_points_report.json")
                save_tag_report(report_path, report)
                print(f"AprilTags: mode=georef, {len(report['tags'])} tag(s)")
                return TagStepResult(adjusted, report, True, False)
            gate_failed = True
            result = dataclasses.replace(
                result, warnings=[*result.warnings, "AprilTag-Georeferenzierung: Qualitätsprüfung nicht bestanden – Review nötig"],
            )
    if gate_failed or result.scale_factor is None:
        report = build_tag_report(result, geometries, scale_applied=None, mode="none")
        new_model = model0
    else:
        new_model = apply_scale(model0, sparse_dir / "0_tagscale", result.scale_factor)
        report = build_tag_report(result, geometries, scale_applied=result.scale_factor, mode="scale")
    save_tag_report(report_path, report)
    factor = "none" if result.scale_factor is None else f"{result.scale_factor:.4f}"
    print(f"AprilTags: mode={report['mode']}, scale factor {factor}, {len(geometries)} tag(s)"
          + (f", warnings: {'; '.join(result.warnings)}" if result.warnings else ""))
    return TagStepResult(new_model, report, False, gate_failed)


def _georef_tag_report(
    final_model: Path, result, geometries: dict, detections: dict, control_report_path: Path,
) -> dict:
    """Tag report in the final (GCP-local ENU) frame: re-triangulated centres/corners plus 3D errors."""
    import pycolmap

    final_geometries = triangulate_tags(pycolmap.Reconstruction(final_model), detections)
    report = build_tag_report(result, final_geometries, scale_applied=1.0, mode="georef")
    errors: dict[str, float | None] = {}
    try:
        control_report = json.loads(control_report_path.read_text(encoding="utf-8"))
        errors = {str(point["name"]): point.get("error_3d_m") for point in control_report.get("points", [])}
    except (OSError, ValueError):
        pass
    factors = []
    for tag in report["tags"]:
        tag_id = int(tag["id"])
        final_edges = tag_edge_lengths(final_geometries[tag_id].corners_xyz)
        tag["size_mm_after"] = float(final_edges[:4].mean() * 1000.0)
        if tag_id in geometries:
            original_edges = tag_edge_lengths(geometries[tag_id].corners_xyz)
            factors.append(float(np.median(final_edges / original_edges)))
            tag["size_mm_before"] = float(original_edges[:4].mean() * 1000.0)
        error = errors.get(f"TAG{tag_id}")
        tag["error_3d_m"] = None if error is None or not math.isfinite(float(error)) else float(error)
    report["scale_applied"] = float(np.median(factors)) if factors else None
    return report


class _TagSeed(NamedTuple):
    detections: dict
    references: dict
    ecef_by_name: dict
    pass_names: list


def _apriltag_control_points(
    pristine_distorted: Path, pristine_text: Path, images_dir: Path, control_points_dir: Path, crs: str,
    settings: dict, report_path: Path, tag_geometries: dict | None = None, tag_detections: dict | None = None,
) -> Path | None:
    """GCP adjustment from surveyed AprilTag centres; raises TagGeorefUnavailable when <3 usable pass tags.

    With tag_geometries the detections are built from the fresh tag triangulation; without them the
    reviewed control_points/detections.json is re-triangulated on the pristine model (resume/optimise).
    """
    import pycolmap

    csv_path = control_points_dir / APRILTAG_CSV_NAME
    if not csv_path.is_file():
        raise TagGeorefUnavailable("apriltags.csv is missing")
    table = read_apriltag_coordinates(csv_path)
    try:
        existing = load_detections(control_points_dir / "detections.json")
    except (OSError, ValueError):
        existing = {}
    if tag_geometries is not None:
        detections = tag_detection_entries(table, tag_geometries, tag_detections or {})
        for name, entry in detections.items():
            if existing.get(name, {}).get("disabled_by_reviewer"):
                entry["enabled"], entry["status"], entry["disabled_by_reviewer"] = False, "disabled", True
    else:
        ensure_text_model(pristine_distorted, pristine_text)
        reconstruction = pycolmap.Reconstruction(pristine_distorted)
        detections = {}
        for row in table.itertuples(index=False):
            name = f"TAG{int(row.id)}"
            if name not in existing:
                continue
            entry = copy.deepcopy(existing[name])
            entry["role"] = str(row.role)
            active = [item for item in entry.get("observations", []) if item.get("enabled", True)]
            if len(active) >= 2:
                try:
                    entry["triangulated_model_xyz"] = [float(v) for v in triangulate_observations(reconstruction, active)[0]]
                except ValueError:
                    pass
            detections[name] = entry
    seed = _seed_tag_references(table, detections, crs)
    ensure_text_model(pristine_distorted, pristine_text)
    return _finish_control_point_adjustment(
        pristine_distorted, control_points_dir, crs, settings, report_path, seed.detections,
        {"apriltags": len(seed.detections)}, seed.references, seed.ecef_by_name, seed.pass_names,
        rtk=False, use_pose_priors=False, local_frame=_is_local_crs(crs),
    )


def _seed_tag_references(table, detections: dict, crs: str) -> _TagSeed:
    """Model-frame reference positions for every tag, from a similarity fit of the pass tags to local ENU."""
    rows = {f"TAG{int(row.id)}": row for row in table.itertuples(index=False)}
    names = [name for name in detections if name in rows]
    usable = [
        name for name in names
        if detections[name].get("role") == "pass" and not detections[name].get("disabled_by_reviewer")
        and detections[name].get("triangulated_model_xyz") is not None
    ]
    if len(usable) < MIN_GEOREF_TAGS:
        raise TagGeorefUnavailable(f"{len(usable)} surveyed pass tag(s)")
    surveyed = np.asarray([[rows[n].x, rows[n].y, rows[n].z] for n in names], dtype=float)
    if _is_local_crs(crs):
        # Local survey system (metres, z up): the coordinates are already the target frame.
        ecef_by_name = dict(zip(names, surveyed, strict=True))
        enu = dict(ecef_by_name)
    else:
        ecef = coordinates_to_ecef(surveyed, crs)
        ecef_by_name = dict(zip(names, ecef, strict=True))
        centroid = np.mean([ecef_by_name[n] for n in usable], axis=0)
        rotation_enu = enu_rotation_at_ecef(centroid)
        enu = {n: rotation_enu @ (ecef_by_name[n] - centroid) for n in names}
    if not tags_are_non_collinear(np.asarray([enu[n] for n in usable])):
        raise TagGeorefUnavailable("surveyed pass tags are collinear")
    scale, rotation, translation = _similarity(
        np.asarray([detections[n]["triangulated_model_xyz"] for n in usable], dtype=float),
        np.asarray([enu[n] for n in usable], dtype=float),
    )
    references = {n: rotation.T @ (enu[n] - translation) / scale for n in names}
    return _TagSeed(
        {n: detections[n] for n in names}, references, ecef_by_name,
        [n for n in names if detections[n].get("role") == "pass"],
    )


def _rerun_control_points(
    model_path: Path, model_txt: Path, images_dir: Path, control_points_dir: Path, control_points_crs: str,
    gps_ref_path: Path, settings: dict, report_path: Path, *, rtk: bool,
) -> Path | None:
    """Re-run the reviewed adjustment: AprilTag seeding for tag projects, GPS seeding otherwise."""
    if _is_tag_project(control_points_dir, gps_ref_path):
        try:
            return _apriltag_control_points(
                model_path, model_txt, images_dir, control_points_dir, control_points_crs, settings, report_path,
            )
        except TagGeorefUnavailable as error:
            save_control_point_report(report_path, {
                "gate": {"passed": False, "reasons": [f"AprilTag georeferencing unavailable: {error}"]}, "points": [],
            })
            return None
    return _automatic_control_points(
        model_path, model_txt, images_dir, control_points_dir, control_points_crs, gps_ref_path, settings,
        report_path, rtk=rtk,
    )


def _is_tag_project(control_points_dir: Path, gps_ref_path: Path) -> bool:
    """Surveyed-AprilTag project: apriltags.csv present and no GPS reference (or detections are all TAG<id>)."""
    if not (control_points_dir / APRILTAG_CSV_NAME).is_file():
        return False
    if not gps_ref_path.is_file():
        return True
    try:
        names = list(load_detections(control_points_dir / "detections.json"))
    except (OSError, ValueError):
        return False
    if not (names and all(re.fullmatch(r"TAG\d+", name) for name in names)):
        return False
    try:
        detections = load_detections(control_points_dir / "detections.json")
    except (OSError, ValueError):
        return False
    for entry in detections.values():
        observations = [obs for obs in entry.get("observations", []) if isinstance(obs, dict)]
        if observations and not all(obs.get("source") == "apriltag" for obs in observations):
            return False
    return True


def _align_panorama_gravity(sparse_dir: Path, model0: Path, report_dir: Path) -> tuple[Path, dict[str, object]]:
    """Write the gravity-aligned panorama model selected for dense reconstruction."""
    aligned_model = sparse_dir / "0_gravity"
    report = align_sparse_model(model0, aligned_model)
    report_path = report_dir / "gravity_alignment.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plane = report.get("largest_horizontal_plane")
    if isinstance(plane, dict) and plane.get("warning"):
        print(
            "WARNING: Sparse horizontal plane differs from gravity-aligned Z by "
            f"{float(plane['angle_to_aligned_z_deg']):.2f} degrees."
        )
    print(
        "Panorama gravity alignment: "
        f"{report['frame_count']} rig frames, "
        f"up-axis spread {float(report['up_spread_deg_rms']):.3f} degrees."
    )
    return aligned_model, report


def _align_camera_gravity(sparse_dir: Path, model0: Path, report_dir: Path) -> tuple[Path, dict[str, object]]:
    """Level an indoor camera model (floor horizontal); same outputs as the panorama gravity step."""
    aligned_model = sparse_dir / "0_gravity"
    report = align_camera_model(model0, aligned_model)
    report_path = report_dir / "gravity_alignment.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for warning in report.get("warnings") or []:
        print(f"WARNING: {warning}")
    print(
        f"Camera gravity alignment ({report['method']}): {report['image_count']} images, "
        f"tilt corrected {float(report['tilt_corrected_deg']):.2f} degrees."
    )
    return aligned_model, report


def _transform_tag_report_to_gravity(report_path: Path, gravity_report: dict[str, object]) -> None:
    """Bring a scale-only AprilTag report into the gravity-aligned frame of the final model."""
    report_path = Path(report_path)
    rotation = gravity_report.get("rotation") if isinstance(gravity_report, dict) else None
    translation = gravity_report.get("translation") if isinstance(gravity_report, dict) else None
    if not report_path.is_file() or rotation is None or translation is None:
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("mode") == "georef":
            return
        save_tag_report(report_path, transform_tag_report(report, rotation, translation))
    except Exception as error:
        print(f"WARNING: AprilTag report could not be moved to the gravity frame: {error}")


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


def write_image_list(images_dir, out_path, *, include_roots: tuple[str, ...] | None = None):
    """Write image names relative to images_dir for COLMAP's recursive input mode."""
    extensions = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )
    if include_roots is not None:
        allowed = set(include_roots)
        image_paths = [
            path for path in image_paths
            if path.relative_to(images_dir).parts[0] in allowed
        ]
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


def _model_registered_image_count(model_path: Path) -> int | None:
    """Read the registered-image count from a COLMAP binary model."""
    images_path = model_path / "images.bin"
    try:
        with images_path.open("rb") as file:
            raw = file.read(8)
        return int.from_bytes(raw, "little") if len(raw) == 8 else None
    except OSError:
        return None


def _promote_largest_sparse_model(sparse_dir: Path) -> dict[str, object]:
    """Make the sparse model with the most registered images "sparse/0".

    COLMAP's mapper numbers models in creation order, not by size, so a small
    side model can be "0". The largest model (ties: more points) is swapped
    into "0"; the old "0" takes the largest model's former number.
    """
    models: list[dict[str, object]] = []
    if sparse_dir.is_dir():
        for child in sparse_dir.iterdir():
            if not (child.is_dir() and child.name.isdigit()):
                continue
            registered = _model_registered_image_count(child)
            if registered is None:
                continue
            models.append({
                "index_original": int(child.name),
                "registered_images": registered,
                "points": _model_point_count(child) or 0,
            })
    models.sort(key=lambda m: int(m["index_original"]))
    info: dict[str, object] = {"models": models, "promoted_from": None}
    if len(models) < 2:
        return info
    best = max(models, key=lambda m: (int(m["registered_images"]), int(m["points"]), -int(m["index_original"])))
    counts = ", ".join(str(m["registered_images"]) for m in models)
    best_index = int(best["index_original"])
    if best_index == 0:
        print(f"Sparse models: {len(models)} (registered {counts}); sparse/0 is already the largest.")
        return info
    print(
        f"Sparse models: {len(models)} (registered {counts}); "
        f"using the largest ({best['registered_images']} images) as sparse/0."
    )
    first = sparse_dir / "0"
    other = sparse_dir / str(best_index)
    temp = sparse_dir / f"_swap_{uuid.uuid4().hex[:8]}"
    first.rename(temp)
    other.rename(first)
    temp.rename(other)
    info["promoted_from"] = best_index
    return info


def _require_usable_sparse_model(model_path: Path, input_image_count: int) -> None:
    """Reject a syntactically valid but unusable sparse reconstruction."""
    registered_images = _model_registered_image_count(model_path)
    points = _model_point_count(model_path)
    minimum_registered_images = max(1, math.ceil(input_image_count * 0.30))
    if registered_images is None or points is None:
        sys.exit(
            "COLMAP produced an unreadable sparse reconstruction; expected images.bin and points3D.bin."
        )
    if registered_images < minimum_registered_images or points < 1000:
        sys.exit(
            "COLMAP sparse reconstruction is too small to continue: "
            f"registered {registered_images}/{input_image_count} images and {points} points "
            f"(requires at least {minimum_registered_images} registered images and 1000 points)."
        )


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


def _estimate_prepare_work(args) -> tuple[int, int]:
    """Expected (prepare-stage units, COLMAP image count) before the images are prepared."""
    sources = 0
    if args.capture_type == "panorama_360":
        from samcloud.acquisition.panorama import expected_video_frame_count, find_panorama_images

        if args.panorama_images:
            try:
                sources += len(find_panorama_images(Path(args.panorama_images)))
            except (OSError, ValueError, TypeError):
                pass
        if args.panorama_video:
            try:
                if Path(args.panorama_video).is_dir():
                    from samcloud.acquisition.video import PANORAMA_VIDEO_EXTENSIONS, find_videos

                    for video in find_videos(args.panorama_video, PANORAMA_VIDEO_EXTENSIONS):
                        sources += expected_video_frame_count(video, args.frame_interval) or 0
                else:
                    sources += expected_video_frame_count(Path(args.panorama_video), args.frame_interval) or 0
            except (OSError, ValueError, TypeError):
                pass
        sources = max(sources, 1)
        return sources, 6 * sources
    if getattr(args, "video", None):
        try:
            from samcloud.acquisition.video import CAMERA_VIDEO_EXTENSIONS, expected_frame_count, find_videos

            sources = expected_frame_count(find_videos(args.video, CAMERA_VIDEO_EXTENSIONS), args.frame_interval) or 0
        except (OSError, ValueError, TypeError):
            sources = 0
        sources = max(sources, 1)
        return sources, sources
    try:
        if args.images:
            extensions = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
            sources = sum(1 for p in Path(args.images).rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    except (OSError, ValueError, TypeError):
        pass
    sources = max(sources, 1)
    return sources, sources


def resolve_input_images(
    args, outdir: Path, parser: argparse.ArgumentParser, progress_callback=None,
) -> tuple[Path, bool, int | None]:
    """Resolve normal images or prepare virtual perspective images for 360 media."""
    if args.capture_type != "panorama_360":
        if not args.images:
            parser.error("--images is required for drone_gps and camera capture types")
        images_dir = Path(args.images).resolve()
        if getattr(args, "video", None):
            from samcloud.acquisition.video import prepare_camera_video_frames

            try:
                prepared_video = prepare_camera_video_frames(
                    args.video, images_dir, args.frame_interval, progress_callback,
                )
            except CaptureValidationError as error:
                parser.error(str(error))
            print(
                f"Prepared camera video frames: {prepared_video.frame_count} frames "
                f"from {prepared_video.video_count} video(s) in {prepared_video.frames_directory}"
            )
            return prepared_video.frames_directory, False, None
        if not images_dir.is_dir():
            parser.error(f"Image directory was not found: {images_dir}")
        return images_dir, False, None

    try:
        prepared = prepare_panorama_inputs(
            args.panorama_images or "",
            args.panorama_video or "",
            Path(args.panorama_workdir).resolve() if args.panorama_workdir else outdir / "inputs",
            args.frame_interval,
            args.cubemap_face_size,
            progress_callback=progress_callback,
        )
    except CaptureValidationError as error:
        parser.error(str(error))
    print(
        "Prepared 360 degree inputs: "
        f"{prepared.panorama_count} panoramas, {prepared.video_frame_count} video frames, "
        f"{prepared.cubemap_image_count} virtual camera images."
    )
    print(f"Panorama source manifest: {prepared.manifest_path}")
    return prepared.cubemap_directory, True, prepared.face_size


def _configured_height_reference() -> str:
    """Describe the EXIF altitude datum configured in colmap.yml (georeferencing.input_height)."""
    from samcloud.engine.extract_gps import input_height_description

    mode = load_global_config("colmap.yml").get("georeferencing", {}).get("input_height", "ellipsoidal")
    return input_height_description(str(mode))


def _resume_automatic_control_points(
    args, outdir: Path, report_dir: Path, summary: RunSummary, progress: ProgressTracker | None,
    colmap_config: dict, dense_max_image_size: int, dense_num_iterations: int,
    dense_gpu_index: int, classified_output: Path, control_points_dir: Path,
) -> None:
    """Continue reviewed automatic GCP projects exactly like a passed automatic run."""
    cubemap_dir = outdir.parent / "inputs" / "cubemap"
    if getattr(args, "capture_type", None) == "panorama_360" and cubemap_dir.is_dir():
        images_dir = cubemap_dir.resolve()
    elif args.images:
        images_dir = Path(args.images).resolve()
    else:
        raise ValueError("The original image directory (--images) is required to continue automatic control points")
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
    adjusted_model = _rerun_control_points(
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
    _run_dense_stereo(
        colmap, dense_dir, dense_config, dense_gpu_index, dense_num_iterations,
        dense_max_image_size, summary, capture_type=getattr(args, "capture_type", None),
    )
    if progress:
        progress.begin("dense_fusion")
    _fuse_dense(colmap, dense_dir, dense_config, getattr(args, "capture_type", None), summary,
        keep_depth_maps=_needs_depth_maps(args))
    sparse_txt = dense_dir / "sparse_txt"
    sparse_txt.mkdir(parents=True, exist_ok=True)
    run([colmap, "model_converter", "--input_path", str(dense_dir / "sparse"), "--output_path", str(sparse_txt),
         "--output_type", "TXT"], summary=summary, stage="Preparing classification")
    _classify_and_finish(args, outdir, dense_dir, sparse_txt, report_dir, summary, progress, classified_output, True)


def _resume_dense_stereo(
    args, outdir: Path, report_dir: Path, summary: RunSummary, progress: ProgressTracker | None,
    colmap_config: dict, dense_max_image_size: int, dense_num_iterations: int,
    dense_gpu_index: int, classified_output: Path,
) -> None:
    """Continue an existing undistorted workspace without redoing sparse reconstruction."""
    dense_dir = outdir / "dense"
    required = (
        (dense_dir / "images", (dense_dir / "images").is_dir()),
        (dense_dir / "sparse", (dense_dir / "sparse").is_dir()),
        (dense_dir / "stereo" / "patch-match.cfg", (dense_dir / "stereo" / "patch-match.cfg").is_file()),
        (dense_dir / "stereo" / "fusion.cfg", (dense_dir / "stereo" / "fusion.cfg").is_file()),
    )
    missing = [str(path) for path, valid in required if not valid]
    if missing:
        raise ValueError("Cannot resume dense stereo; workspace files are missing: " + ", ".join(missing))
    if progress and progress.estimator is None:
        image_list_path = outdir / "image_list.txt"
        num_images = len([line for line in image_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]) if image_list_path.is_file() else 0
        progress.estimator = _build_estimator(
            args, num_images, colmap_config, resume=True, dense_max_image_size=dense_max_image_size,
        )
        global _ACTIVE_PROGRESS
        _ACTIVE_PROGRESS = progress
    _plan_geometry_refinement(progress, _may_be_georeferenced(outdir))

    dense_config = colmap_config.get("dense_reconstruction", {})
    colmap = find_colmap_exe(args.colmap_bin)
    if progress:
        progress.begin("dense_stereo")
        progress.update(62, "Building dense cloud")
    summary.settings["dense_stereo_resumed"] = True
    summary.settings["dense_stereo_excluded_images"] = _run_dense_stereo(
        colmap, dense_dir, dense_config, dense_gpu_index, dense_num_iterations,
        dense_max_image_size, summary, capture_type=getattr(args, "capture_type", None),
    )
    if progress:
        progress.begin("dense_fusion")
        progress.update(78, "Fusing dense cloud")
    _fuse_dense(colmap, dense_dir, dense_config, getattr(args, "capture_type", None), summary,
        keep_depth_maps=_needs_depth_maps(args))

    sparse_txt = dense_dir / "sparse_txt"
    sparse_txt.mkdir(parents=True, exist_ok=True)
    run([colmap, "model_converter", "--input_path", str(dense_dir / "sparse"), "--output_path", str(sparse_txt),
         "--output_type", "TXT"], summary=summary, stage="Preparing classification")
    georeferenced = (outdir / "gps_ref.txt").is_file()
    if georeferenced:
        try:
            georeference = _write_camera_centre_georeference(
                sparse_txt, outdir / "gps_ref.txt", outdir / "georeference.json",
            )
            if georeference is not None:
                summary.settings["georeference_residual_m"] = georeference["residual_m"]
        except Exception as error:
            print(f"WARNING: Could not create GPS georeference: {error}")
            georeferenced = False
    elif _local_tag_georeferenced(outdir):
        georeferenced = True
    _classify_and_finish(
        args, outdir, dense_dir, sparse_txt, report_dir, summary, progress, classified_output, georeferenced,
    )


def _resume_classification(
    args, outdir: Path, report_dir: Path, summary: RunSummary, progress: ProgressTracker | None,
    colmap_config: dict, dense_max_image_size: int, dense_num_iterations: int,
    dense_gpu_index: int, classified_output: Path,
) -> None:
    """Continue directly at classification using an existing fused dense workspace."""
    dense_dir = outdir / "dense"
    sparse_txt = dense_dir / "sparse_txt"
    required = (
        (dense_dir / "images", (dense_dir / "images").is_dir()),
        (dense_dir / "sparse", (dense_dir / "sparse").is_dir()),
        (dense_dir / "fused.ply", (dense_dir / "fused.ply").is_file()),
        (dense_dir / "fused.ply.vis", (dense_dir / "fused.ply.vis").is_file()),
        (sparse_txt, sparse_txt.is_dir()),
    )
    missing = [str(path) for path, valid in required if not valid]
    if missing:
        raise ValueError("Cannot resume classification; workspace files are missing: " + ", ".join(missing))
    if progress and progress.estimator is None:
        image_list_path = outdir / "image_list.txt"
        num_images = len([line for line in image_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]) if image_list_path.is_file() else 0
        progress.estimator = _build_estimator(
            args, num_images, colmap_config, resume=True, dense_max_image_size=dense_max_image_size,
        )
        global _ACTIVE_PROGRESS
        _ACTIVE_PROGRESS = progress
    _plan_geometry_refinement(progress, _may_be_georeferenced(outdir))
    summary.settings["classification_resumed"] = True
    georeferenced = (outdir / "gps_ref.txt").is_file()
    if georeferenced:
        try:
            georeference = _write_camera_centre_georeference(
                sparse_txt, outdir / "gps_ref.txt", outdir / "georeference.json",
            )
            if georeference is not None:
                summary.settings["georeference_residual_m"] = georeference["residual_m"]
        except Exception as error:
            print(f"WARNING: Could not create GPS georeference: {error}")
            georeferenced = False
    elif _local_tag_georeferenced(outdir):
        georeferenced = True
    _classify_and_finish(
        args, outdir, dense_dir, sparse_txt, report_dir, summary, progress, classified_output, georeferenced,
    )


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
    if _is_local_crs(args.control_points_crs):
        raise ValueError(LOCAL_CRS_GPS_MESSAGE)
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
    _run_dense_stereo(
        colmap, dense_dir, colmap_config.get("dense_reconstruction", {}), dense_gpu_index,
        dense_num_iterations, dense_max_image_size, summary, capture_type=getattr(args, "capture_type", None),
    )
    dense_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.begin("dense_fusion")
    _fuse_dense(colmap, dense_dir, colmap_config.get("dense_reconstruction", {}), getattr(args, "capture_type", None), summary,
        keep_depth_maps=_needs_depth_maps(args))
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


def _resolve_control_points_mode(mode: str, control_points_dir: Path) -> str:
    """Resolve the coordinate-aware default without changing explicit user choices."""
    if mode != "auto":
        return mode
    csv_path = control_points_dir / "control_points.csv"
    if not csv_path.is_file():
        return "none"
    try:
        points = read_control_points(csv_path)
    except (OSError, ValueError) as error:
        print(f"WARNING: Automatic control points skipped: cannot read {csv_path.name}: {error}")
        return "none"
    coordinates = points[["x", "y", "z"]].to_numpy(dtype=float)
    if len(points) >= 3 and np.isfinite(coordinates).all():
        return "automatic"
    print("Automatic control points skipped: at least three points with finite coordinates are required.")
    return "none"


def _evaluate_pre_gcp_checkpoints(
    model_path: Path, images_dir: Path, points, control_points_crs: str, georef: dict,
    settings: dict, report_dir: Path,
) -> dict | None:
    """Write the unadjusted checkpoint baseline; evaluation errors never stop processing."""
    try:
        import pycolmap
        from pyproj import Transformer

        from samcloud.drone.checkpoints import evaluate_checkpoints_model, render_markdown

        scale = float(georef["scale"])
        rotation = np.asarray(georef["rotation"], dtype=float)
        translation = np.asarray(georef["translation"], dtype=float)
        survey = {
            str(row.name): np.asarray([row.x, row.y, row.z], dtype=float)
            for row in points.itertuples(index=False)
        }
        to_survey = Transformer.from_crs("EPSG:4978", control_points_crs, always_xy=True)

        def to_model(coordinates: np.ndarray) -> np.ndarray:
            ecef = coordinates_to_ecef(np.asarray(coordinates, dtype=float), control_points_crs)
            return (rotation.T @ ((ecef - translation) / scale).T).T

        def from_model(coordinates: np.ndarray) -> np.ndarray:
            ecef = (scale * (rotation @ np.asarray(coordinates, dtype=float).T)).T + translation
            x, y, z = to_survey.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
            return np.column_stack((x, y, z))

        result = evaluate_checkpoints_model(
            pycolmap.Reconstruction(model_path), survey, _image_path_map(images_dir),
            to_model=to_model, from_model=from_model, settings=settings, out_dir=report_dir,
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "checkpoints_pre_gcp.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        (report_dir / "checkpoints_pre_gcp.md").write_text(
            render_markdown(result, "pre_gcp"), encoding="utf-8",
        )
        return result
    except Exception as error:  # checkpoint QA is informative and must not block GCP adjustment
        print(f"WARNING: Pre-GCP checkpoint evaluation failed: {error}")
        return None


def _quick_height_correction(checkpoint_report: Path) -> dict | None:
    """Estimate a safe vertical correction from the pre-GCP checkpoint report."""
    try:
        from samcloud.drone.height_correction import estimate_height_shift

        estimate = estimate_height_shift(
            checkpoint_report, exclude=("not_found", "implausible", "few_views"),
            min_points=3, max_spread_m=0.05,
        )
    except Exception as error:  # the optional correction must never stop the pipeline
        print(f"WARNING: Quick height correction was skipped: {error}")
        return None
    if estimate is None:
        return None
    if dataclasses.is_dataclass(estimate):
        result = dataclasses.asdict(estimate)
    elif isinstance(estimate, dict):
        result = dict(estimate)
    else:
        result = {"shift_m": float(estimate)}
    for key in ("height_offset_m", "shift_m", "shift", "height_shift_m"):
        if key in result and np.isfinite(float(result[key])):
            result["height_offset_m"] = float(result[key])
            return result
    print("WARNING: Quick height correction was skipped: no finite height shift was returned.")
    return None


def _apply_quick_height_correction(project_dir: Path, correction: dict | None) -> bool:
    """Apply the checkpoint-derived height shift only after a failed GCP quality gate."""
    if not correction or "height_offset_m" not in correction:
        return False
    try:
        from samcloud.drone.height_correction import apply_height_shift

        apply_height_shift(project_dir, correction)
        return True
    except Exception as error:  # a failed fallback must leave the pipeline reviewable
        print(f"WARNING: Quick height correction could not be applied: {error}")
        return False


def _automatic_control_points(
    model_path: Path, model_txt: Path, images_dir: Path, control_points_dir: Path, control_points_crs: str,
    gps_ref_path: Path, settings: dict, report_path: Path, *, rtk: bool, summary: RunSummary | None = None,
) -> Path | None:
    """Detect, adjust and transform GPS projects into the GCP-local ENU frame."""
    if not gps_ref_path.is_file():
        report = {"gate": {"passed": False, "reasons": ["manual seeds needed"]}, "points": []}
        save_control_point_report(report_path, report)
        return None
    import pycolmap

    ensure_text_model(model_path, model_txt)

    points = read_control_points(control_points_dir / "control_points.csv")
    georef = fit_enu_to_ecef(model_txt, gps_ref_path, _configured_height_reference())
    scale = float(georef["scale"])
    rotation = np.asarray(georef["rotation"], dtype=float)
    translation = np.asarray(georef["translation"], dtype=float)
    try:
        ecef = coordinates_to_ecef(points[["x", "y", "z"]].to_numpy(dtype=float), control_points_crs)
    except Exception:
        if not _is_local_crs(control_points_crs):
            raise
        print(f"WARNING: {LOCAL_CRS_GPS_MESSAGE}.")
        save_control_point_report(report_path, {"gate": {"passed": False, "reasons": [LOCAL_CRS_GPS_MESSAGE]}, "points": []})
        return None
    references = {
        str(row.name): rotation.T @ ((ecef[index] - translation) / scale)
        for index, (_, row) in enumerate(points.set_index("name").iterrows())
    }
    roles = dict(zip(points["name"].astype(str), points["role"].astype(str), strict=True))
    checkpoint_result = _evaluate_pre_gcp_checkpoints(
        model_path, images_dir, points, control_points_crs, georef, settings, report_path.parent,
    )
    quick_correction = _quick_height_correction(report_path.parent / "checkpoints_pre_gcp.json") if checkpoint_result else None
    if summary is not None and quick_correction is not None:
        summary.settings["quick_correction"] = quick_correction
    paths = _image_path_map(images_dir)
    detections, detection_statistics = automatic_detection(
        pycolmap.Reconstruction(model_path), references, roles, paths, settings,
        0.03 if rtk else 3.0, 0.05 if rtk else 5.0,
        existing=load_detections(control_points_dir / "detections.json"),
    )
    return _finish_control_point_adjustment(
        model_path, control_points_dir, control_points_crs, settings, report_path,
        detections, detection_statistics, references,
        {str(name): ecef[index] for index, name in enumerate(points["name"].astype(str))},
        [str(name) for name in points.loc[points["role"] == "pass", "name"]],
        rtk=rtk, use_pose_priors=True, quick_correction=quick_correction, summary=summary,
    )


def _finish_control_point_adjustment(
    model_path: Path, control_points_dir: Path, control_points_crs: str, settings: dict, report_path: Path,
    detections: dict[str, dict], detection_statistics: dict, references: dict[str, np.ndarray],
    ecef_by_name: dict[str, np.ndarray], pass_names: list[str], *, rtk: bool, use_pose_priors: bool,
    local_frame: bool = False, quick_correction: dict | None = None, summary: RunSummary | None = None,
) -> Path | None:
    """Shared GCP adjustment: BA loop, gate, GCP-local ENU transform, leave-one-out, reports, georeference."""
    import pycolmap

    before = camera_intrinsics(pycolmap.Reconstruction(model_path))
    pose_priors = None
    database_path = model_path.parents[1] / "database.db"
    if use_pose_priors:
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
        if quick_correction is not None:
            quick_correction = {**quick_correction, "applied": _apply_quick_height_correction(model_path.parents[2], quick_correction)}
            report["quick_correction"] = quick_correction
            if summary is not None:
                summary.settings["quick_correction"] = quick_correction
        save_control_point_report(report_path, report)
        return None
    enabled_pass = [str(name) for name in pass_names if detections.get(str(name), {}).get("enabled")]
    names = enabled_pass
    if local_frame:
        # Local survey coordinates are the target frame: no ECEF/ENU conversion and no centroid shift.
        target_enu = {name: np.asarray(ecef_by_name[name], dtype=float) for name in names}
    else:
        ecef = np.asarray([ecef_by_name[name] for name in names], dtype=float)
        georeference = gcp_georeference_from_ecef(
            ecef, control_points_crs, {"mean": 0.0, "p95": 0.0, "max": 0.0},
        )
        centroid = np.asarray(georeference["translation"], dtype=float)
        enu_rotation = np.asarray(georeference["rotation"], dtype=float).T
        target_enu = dict(zip(names, (enu_rotation @ (ecef - centroid).T).T, strict=True))
    source = np.asarray([references[name] for name in enabled_pass], dtype=float)
    destination = np.asarray([target_enu[name] for name in enabled_pass], dtype=float)
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
    if quick_correction is not None:
        report["quick_correction"] = {**quick_correction, "applied": False}
        if summary is not None:
            summary.settings["quick_correction"] = report["quick_correction"]
    if local_frame and final_gate[0]:
        report["crs"] = "LOCAL"
        report["source_frame"] = report["method"] = LOCAL_TAGS_FRAME
    save_control_point_report(report_path, report)
    if not final_gate[0]:
        if quick_correction is not None:
            quick_correction = {**quick_correction, "applied": _apply_quick_height_correction(model_path.parents[2], quick_correction)}
            report["quick_correction"] = quick_correction
            if summary is not None:
                summary.settings["quick_correction"] = quick_correction
            save_control_point_report(report_path, report)
        return None
    if local_frame:
        # The model already is in the target frame; LOCAL -> LOCAL export needs no georeference.json.
        products = model_path.parents[1]
        if (products.parent / "project.samcloud.yml").is_file():
            for path in model_path.parent.glob("gcp_adjusted_raw*"):
                if path != local_model:
                    remove_generated(path, products)
        return local_model
    save_georeference(
        model_path.parents[1] / "georeference.json",
        gcp_georeference_from_ecef(ecef, control_points_crs, residuals),
    )
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


def _stereo_workspace_image_data(sparse_model: Path) -> tuple[set[str], set[str], dict[str, list[str]]]:
    """Return sparse image names, usable names, and ranked usable MVS source images."""
    import pycolmap

    reconstruction = pycolmap.Reconstruction(sparse_model)
    names = {image.image_id: image.name for image in reconstruction.images.values()}
    rotations = {
        image.image_id: np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float32)
        for image in reconstruction.images.values()
    }
    translations = {
        image.image_id: np.asarray(image.cam_from_world().translation, dtype=np.float32)
        for image in reconstruction.images.values()
    }
    usable_image_ids: set[int] = set()
    for point in reconstruction.points3D.values():
        xyz = np.asarray(point.xyz, dtype=np.float32)
        for element in point.track.elements:
            image_id = element.image_id
            depth = np.dot(rotations[image_id][2], xyz) + translations[image_id][2]
            if np.isfinite(depth) and depth > 0:
                usable_image_ids.add(image_id)

    source_scores: dict[int, dict[int, int]] = {image_id: {} for image_id in usable_image_ids}
    for point in reconstruction.points3D.values():
        xyz = np.asarray(point.xyz, dtype=np.float32)
        visible_ids = [
            element.image_id for element in point.track.elements
            if element.image_id in usable_image_ids
            and np.isfinite(np.dot(rotations[element.image_id][2], xyz) + translations[element.image_id][2])
            and np.dot(rotations[element.image_id][2], xyz) + translations[element.image_id][2] > 0
        ]
        for reference_id in visible_ids:
            scores = source_scores[reference_id]
            for source_id in visible_ids:
                if source_id != reference_id:
                    scores[source_id] = scores.get(source_id, 0) + 1

    ranked_sources = {
        names[reference_id]: [names[source_id] for source_id, _ in sorted(
            scores.items(), key=lambda item: (-item[1], names[item[0]]),
        )]
        for reference_id, scores in source_scores.items()
    }
    return set(names.values()), {names[image_id] for image_id in usable_image_ids}, ranked_sources


def _stereo_images_with_depth_ranges(sparse_model: Path) -> set[str]:
    """Return images for which COLMAP's MVS model can derive a positive depth range."""
    _, usable_names, _ = _stereo_workspace_image_data(sparse_model)
    return usable_names


def _parse_stereo_sources(sources: str) -> tuple[str, int | list[str] | None]:
    """Parse one COLMAP PatchMatch source-image line."""
    values = [value.strip() for value in sources.split(",") if value.strip()]
    if not values:
        return "explicit", []
    if values[0] == "__auto__":
        if len(values) != 2:
            raise ValueError(f"Invalid COLMAP automatic source selector: {sources}")
        try:
            source_count = int(values[1])
        except ValueError as exc:
            raise ValueError(f"Invalid COLMAP automatic source selector: {sources}") from exc
        if source_count < 0:
            raise ValueError(f"Invalid COLMAP automatic source selector: {sources}")
        return "auto", source_count
    if values == ["__all__"]:
        return "all", None
    return "explicit", values


def _filtered_stereo_sources(
    sources: str, reference: str, retained_names: set[str], ranked_sources: dict[str, list[str]],
    convert_automatic: bool,
) -> str | None:
    """Remove unavailable sources and expand automatic selectors when required."""
    mode, value = _parse_stereo_sources(sources)
    if mode == "auto" and not convert_automatic:
        return sources.strip()
    if mode == "auto":
        selected = [
            name for name in ranked_sources.get(reference, []) if name in retained_names and name != reference
        ][:int(value)]
    elif mode == "all":
        selected = sorted(retained_names - {reference})
    else:
        selected = []
        for name in value or []:
            if name in retained_names and name != reference and name not in selected:
                selected.append(name)
    return ", ".join(selected) if selected else None


def _filter_fusion_config(
    fusion_config: Path, retained_names: set[str], ranked_sources: dict[str, list[str]],
    convert_automatic: bool,
) -> None:
    """Filter a standard fusion image list or an optional reference/source configuration."""
    lines = [line.strip() for line in fusion_config.read_text(encoding="utf-8").splitlines() if line.strip()]
    has_source_selectors = any(
        "," in line or line == "__all__" or line.startswith("__auto__") for line in lines[1::2]
    )
    if has_source_selectors and len(lines) % 2 == 0:
        entries = list(zip(lines[::2], lines[1::2]))
        retained = []
        for reference, sources in entries:
            if reference not in retained_names:
                continue
            filtered_sources = _filtered_stereo_sources(
                sources, reference, retained_names, ranked_sources, convert_automatic,
            )
            if filtered_sources is not None:
                retained.append((reference, filtered_sources))
        fusion_config.write_text(
            "".join(f"{reference}\n{sources}\n" for reference, sources in retained), encoding="utf-8",
        )
        return
    fusion_config.write_text(
        "".join(f"{name}\n" for name in lines if name in retained_names), encoding="utf-8",
    )


def _filter_stereo_workspace(dense_dir: Path) -> list[str]:
    """Remove unusable stereo references and sources from COLMAP configurations."""
    sparse_model = dense_dir / "sparse"
    patch_config = dense_dir / "stereo" / "patch-match.cfg"
    fusion_config = dense_dir / "stereo" / "fusion.cfg"
    missing = [
        path for path, valid in (
            (sparse_model, sparse_model.is_dir()),
            (patch_config, patch_config.is_file()),
            (fusion_config, fusion_config.is_file()),
        ) if not valid
    ]
    if missing:
        print(
            "WARNING: Dense stereo workspace filtering was skipped because files are missing: "
            + ", ".join(map(str, missing))
        )
        return []

    patch_lines = patch_config.read_text(encoding="utf-8").splitlines()
    nonempty_lines = [line.strip() for line in patch_lines if line.strip() and not line.lstrip().startswith("#")]
    if len(nonempty_lines) % 2:
        raise ValueError(f"Invalid COLMAP patch-match configuration: {patch_config}")
    entries = list(zip(nonempty_lines[::2], nonempty_lines[1::2]))
    sparse_names, usable_names, ranked_sources = _stereo_workspace_image_data(sparse_model)
    excluded = sorted(sparse_names - usable_names)
    retained = [(name, sources) for name, sources in entries if name in usable_names]
    if not retained:
        raise ValueError("Cannot run dense stereo because no image has a usable sparse depth range")

    retained_names = {name for name, _ in retained}
    convert_automatic = bool(excluded)
    dropped_references: list[str] = []
    while True:
        filtered_entries = []
        newly_dropped = []
        for name, sources in retained:
            if name not in retained_names:
                continue
            filtered_sources = _filtered_stereo_sources(
                sources, name, retained_names, ranked_sources, convert_automatic,
            )
            if filtered_sources is None:
                newly_dropped.append(name)
                continue
            filtered_entries.append((name, filtered_sources))
        if not newly_dropped:
            break
        dropped_references.extend(newly_dropped)
        retained_names.difference_update(newly_dropped)
    if not filtered_entries:
        raise ValueError("Cannot run dense stereo because no image has usable source images")

    patch_config.write_text(
        "".join(f"{name}\n{sources}\n" for name, sources in filtered_entries), encoding="utf-8",
    )
    retained_names = {name for name, _ in filtered_entries}
    _filter_fusion_config(
        fusion_config, retained_names, ranked_sources, convert_automatic,
    )
    if excluded:
        print(
            "Dense stereo excluded images without a usable sparse depth range: "
            + ", ".join(excluded)
        )
    if dropped_references:
        print(
            "Dense stereo excluded references without usable source images: "
            + ", ".join(dropped_references)
        )
    return excluded


def _run_dense_stereo(
    colmap: str, dense_dir: Path, dense_config: dict, dense_gpu_index: int,
    dense_num_iterations: int, dense_max_image_size: int, summary: RunSummary,
    *, capture_type: str | None = None,
) -> list[str]:
    """Run COLMAP PatchMatch after excluding reference images with no depth range."""
    excluded = _filter_stereo_workspace(dense_dir)
    patch_match = dense_profile(dense_config, capture_type)["patch_match"]
    summary.settings["dense_patch_match"] = dict(patch_match)
    run([colmap, "patch_match_stereo", "--workspace_path", str(dense_dir),
         "--PatchMatchStereo.geom_consistency", "1" if dense_config.get("use_geometric_consistency", True) else "0",
         "--PatchMatchStereo.gpu_index", str(dense_gpu_index),
         "--PatchMatchStereo.num_iterations", str(dense_num_iterations),
         "--PatchMatchStereo.max_image_size", str(dense_max_image_size),
         *patch_match_arguments(patch_match)], summary=summary, stage="Dense stereo")
    return excluded


def _plane_fill_settings(args) -> dict | None:
    """Settings of the class-confirmed plane fill, or None when it does not run.

    The fill needs SAM's floor/ceiling/wall classes and therefore only runs for
    the indoor class set; outdoor and drone projects are never filled.
    """
    if getattr(args, "class_set", None) != "indoor":
        return None
    dense_config = load_global_config("colmap.yml").get("dense_reconstruction", {})
    settings = dense_profile(dense_config, getattr(args, "capture_type", None))["plane_fill"]
    return settings if bool(settings.get("enabled", False)) else None


def _dense_settings(args) -> dict:
    """Dense settings merged with the profile of this capture type."""
    dense_config = load_global_config("colmap.yml").get("dense_reconstruction", {})
    return dense_profile(dense_config, getattr(args, "capture_type", None))


def _cleaning_settings(args) -> dict | None:
    """Free-space point cleaning of workflow step 6 (all capture types).

    Not needed when surface fusion rebuilds the cloud from the fused field:
    every camera's free space is already carved out there.
    """
    if _fusion_rebuilds_cloud(args):
        return None
    fusion = _surface_fusion_settings(args)
    if fusion is not None and bool(fusion.get("visibility_check", True)):
        return None  # the fusion's GPU visibility check does the same test in seconds
    settings = _dense_settings(args)["point_cleaning"]
    return settings if bool(settings.get("enabled", False)) else None


def _depth_prior_settings(args) -> dict | None:
    """Depth-Anything-V2 prior for step 6: indoor class set only, model chosen per project."""
    if getattr(args, "class_set", None) != "indoor":
        return None
    if str(getattr(args, "depth_fill", "1")) != "1":  # Depth-Anything switched off in the project settings
        return None
    model = str(getattr(args, "depth_model", "large") or "off").lower()
    if model == "off":
        return None
    from samcloud.colmap.depth_prior import DEFAULTS as PRIOR_DEFAULTS

    return {**PRIOR_DEFAULTS, "model": model}


def _depth_fill_settings(args) -> dict | None:
    """Depth-Anything point fill of step 6: indoor only, with a depth model.

    Replaced by surface fusion in ``extract`` mode, which fills from the same
    priors but only where they agree with COLMAP.
    """
    if _depth_prior_settings(args) is None or _fusion_rebuilds_cloud(args):
        return None
    settings = _dense_settings(args)["depth_fill"]
    return settings if bool(settings.get("enabled", False)) else None


def _surface_fusion_settings(args) -> dict | None:
    """TSDF surface fusion of step 6 (indoor): one surface layer from all depth maps.

    Works with COLMAP depth alone; Depth-Anything (when on) adds depth only
    where it agrees with COLMAP. Switchable per project.
    """
    if getattr(args, "class_set", None) != "indoor" or str(getattr(args, "surface_fusion", "1")) != "1":
        return None
    settings = _dense_settings(args)["surface_fusion"]
    return settings if bool(settings.get("enabled", False)) else None


def _superpoint_vote_settings(args) -> dict | None:
    """Final geometry-aware semantic vote, enabled only for indoor SAM3 clouds."""
    if getattr(args, "class_set", None) != "indoor":
        return None
    settings = _dense_settings(args)["superpoint_vote"]
    return settings if bool(settings.get("enabled", False)) else None


def _fusion_rebuilds_cloud(args) -> bool:
    settings = _surface_fusion_settings(args)
    return settings is not None and str(settings.get("mode", "extract")) == "extract"


def _needs_depth_maps(args) -> bool:
    """Keep COLMAP's depth maps after fusion when step 6 still reads them."""
    return (
        _cleaning_settings(args) is not None
        or _plane_fill_settings(args) is not None
        or _depth_fill_settings(args) is not None
        or _surface_fusion_settings(args) is not None
    )


def _fuse_dense(
    colmap: str, dense_dir: Path, dense_config: dict, capture_type: str | None, summary: RunSummary,
    *, keep_depth_maps: bool = False,
) -> None:
    """Fuse depth maps, then release the stereo cache unless step 6 still needs it."""
    profile = dense_profile(dense_config, capture_type)
    run(stereo_fusion_command(colmap, dense_dir, dense_dir / "fused.ply", profile["fusion"]),
        summary=summary, stage="Dense fusion")
    try:  # remembered for reports/processing_metrics.json: fused.ply is compressed away before the run ends
        from samcloud.report.metrics import ply_vertex_count

        fused_points = ply_vertex_count(dense_dir / "fused.ply")
        if fused_points:
            summary.dense_points_fused = fused_points
    except Exception as error:  # noqa: BLE001 - optional protocol metric
        print(f"WARNING: dense point count not recorded: {error}", flush=True)
    if not keep_depth_maps:
        _release_fusion_cache(dense_dir, summary)


def _labels_arguments(args, dense_dir: Path) -> list[str]:
    """Ask classify.py to keep per-view SAM label rasters when a fill step will use them."""
    needed = _plane_fill_settings(args) is not None or _depth_fill_settings(args) is not None
    return ["--labels-dir", str(dense_dir / "sam_labels")] if needed else []


def _estimator_progress(progress: "ProgressTracker | None"):
    if progress is None or progress.estimator is None:
        return None

    def report(done: int, total: int) -> None:
        progress.estimator.update(done, total)
        progress.write()

    return report


def _resolve_units_per_metre(args, dense_dir: Path, classified_raw: Path, fill_settings: dict | None,
                             fusion: dict | None, summary: RunSummary):
    """Set ``units_per_metre`` of depth fill and surface fusion (``auto`` = from the storey height).

    Unscaled indoor projects (no georeference) take floor/ceiling heights against a standard storey;
    georeferenced projects keep 1.0. A number in the config overrides.
    """
    wanted = [item for item in (fill_settings, fusion) if item is not None]
    if not wanted:
        return fill_settings, fusion
    setting = next((item.get("units_per_metre", "auto") for item in wanted), "auto")
    if isinstance(setting, str) and setting.lower() == "auto":
        outdir = dense_dir.parent
        info: dict = {"units_per_metre": 1.0, "source": "georeferenced"}
        metric = None
        if not (_is_georeferenced(outdir) or (outdir / "gps_ref.txt").is_file()):
            # Same decision as the indoor plan (samcloud.core.metric_scale): an AprilTag-scaled or explicitly metric
            # cloud keeps 1.0 instead of the storey-height approximation.
            from samcloud.core.metric_scale import metric_scale_decision

            metric = metric_scale_decision(outdir.parent)
            if metric.known_scale:
                unit = 1.0 if metric.units_per_metre is None else float(metric.units_per_metre)
                info = {"units_per_metre": unit, "source": metric.source}
        if not (_is_georeferenced(outdir) or (outdir / "gps_ref.txt").is_file() or (metric is not None and metric.known_scale)):
            try:
                from samcloud.colmap.surface_fusion import derive_units_per_metre
                from samcloud.engine.ply_io import read_vertex_array

                names = json.loads(Path(args.classes_json).read_text(encoding="utf-8"))[args.class_set]
                short = [name.split(",", 1)[0].strip().lower() for name in names]
                ids = {name: short.index(name) + 1 for name in ("floor", "ceiling") if name in short}
                vertex = read_vertex_array(classified_raw)
                info = derive_units_per_metre(np.asarray(vertex["z"], dtype=np.float64), np.asarray(vertex["scalar_class"]),
                                              ids, float(wanted[0].get("nominal_storey_height_m", 2.5)))
            except Exception as error:  # noqa: BLE001 - metric units stay valid
                info = {"units_per_metre": 1.0, "source": "error", "error": str(error)}
        print(f"Project scale: {info['units_per_metre']} units per metre ({info.get('source')}).")
    else:
        info = {"units_per_metre": float(setting), "source": "config"}
    summary.settings["units_per_metre"] = info
    unit = float(info["units_per_metre"])
    fill_settings = None if fill_settings is None else {**fill_settings, "units_per_metre": unit}
    fusion = None if fusion is None else {**fusion, "units_per_metre": unit}
    return fill_settings, fusion


def _refine_after_classification(
    args, dense_dir: Path, classified_raw: Path, scores_path: Path, summary: RunSummary,
    progress: "ProgressTracker | None" = None, *, report_dir: Path | None = None,
    keep_caches: bool = False, reuse_priors: bool = False,
) -> None:
    """Workflow step 6 "Verbessern & filtern" (before the geometric refinement).

    1. indoor: Depth-Anything-V2 depth priors aligned to COLMAP,
    2. indoor: fill measurement gaps with prior depth confirmed by several views,
    3. all capture types: remove points that any camera proves to float in free
       space (after the fill, so it also checks every filled point against all
       cameras, not only the fill's neighbouring views),
    4. optional plane grid fill (off by default).
    Afterwards the depth caches, SAM label rasters and priors are released.
    """
    from samcloud.colmap.depth_prior import availability, compute_depth_priors, prior_reader
    from samcloud.colmap.plane_fill import load_views
    from samcloud.colmap.point_cleaning import clean_classified_cloud

    if progress:
        progress.update(93, "Improving and filtering")
    try:  # step 6 keeps the score file row-aligned; the fine terms wait in a snapshot (see fine_terms.py)
        snapshot_fine_terms(classified_raw, scores_path)
    except Exception as error:  # noqa: BLE001 - the classification itself stays valid without fine terms
        print(f"WARNING: Fine terms could not be set aside for step 6: {error}")
    try:
        snapshot_instances(classified_raw, classified_raw.with_name(INSTANCES_SOURCE_NAME))
    except Exception as error:  # noqa: BLE001 - classifications remain usable without instances
        print(f"WARNING: Instances could not be set aside for step 6: {error}")
    scores = scores_path if scores_path.is_file() else None
    has_depth = (dense_dir / "stereo" / "depth_maps").is_dir()
    views = None
    if has_depth and (dense_dir / "sparse").is_dir():
        try:
            views = load_views(dense_dir / "sparse")
        except Exception as error:  # noqa: BLE001
            print(f"WARNING: Dense views could not be loaded for step 6: {error}")
    priors = None
    prior_settings = _depth_prior_settings(args)
    prior_dir = dense_dir / "depth_prior"
    if prior_settings is not None and views is not None and reuse_priors and prior_dir.is_dir() and any(prior_dir.rglob("*.npy")):
        print("Reusing the cached Depth-Anything-V2 priors.")
        summary.settings["depth_prior"] = {"reused": True}
        priors = prior_reader(dense_dir)
    elif prior_settings is not None and views is not None:
        reason = availability(prior_settings["model"])
        if reason:
            print(f"WARNING: Depth-Anything-V2 skipped: {reason}")
            summary.settings["depth_prior"] = {"skipped": reason}
        else:
            if progress:
                progress.begin("depth_prior", len(views))
            started_at = time.perf_counter()
            try:
                report = compute_depth_priors(dense_dir, views, prior_settings, progress=_estimator_progress(progress))
            except Exception as error:  # noqa: BLE001 - the run continues without priors
                print(f"WARNING: Depth-Anything-V2 failed: {error}")
                summary.settings["depth_prior"] = {"error": str(error)}
            else:
                summary.record_stage("Depth prior", time.perf_counter() - started_at)
                summary.settings["depth_prior"] = report
                print(f"Depth-Anything-V2 ({report['model']}) aligned {report['aligned_views']} + "
                      f"{report['global_fit_views']} views to COLMAP.")
                priors = prior_reader(dense_dir)
                if not keep_caches and str(getattr(args, "keep_processing_cache", "0")) == "1":
                    from samcloud.engine.processing_cache import save_depth_priors

                    save_depth_priors(dense_dir.parent.parent, dense_dir)
    labels_dir = dense_dir / "sam_labels"
    fill_settings = _depth_fill_settings(args)
    fusion = _surface_fusion_settings(args)
    fill_settings, fusion = _resolve_units_per_metre(args, dense_dir, classified_raw, fill_settings, fusion, summary)
    mirrors = []
    if views is not None and (fill_settings is not None or fusion is not None):
        from samcloud.colmap.mirror_filter import detect_mirrors
        from samcloud.colmap.surface_fusion import DEFAULTS as FUSION_DEFAULTS

        active_fusion = fusion or FUSION_DEFAULTS
        mirror_settings = {**active_fusion.get("mirror_filter", {}),
                           "depth_edge_filter": active_fusion.get("depth_edge_filter", {}),
                           "units_per_metre": active_fusion.get("units_per_metre", 1.0)}
        if bool(mirror_settings.get("enabled", True)):
            started_at = time.perf_counter()
            try:
                mirrors = detect_mirrors(views, dense_dir / "stereo", dense_dir / "sam_mirror", mirror_settings, labels_dir)
            except Exception as error:  # noqa: BLE001 - depth fill remains available without masks
                print(f"WARNING: Mirror detection was skipped: {error}")
                mirrors = []
            else:
                summary.record_stage("Mirror detection", time.perf_counter() - started_at)
        if fill_settings is not None:
            fill_settings = {**fill_settings, "depth_edge_filter": active_fusion.get("depth_edge_filter", {})}
    if fill_settings is not None and views is not None and priors is not None:
        from samcloud.colmap.depth_fill import fill_classified_from_prior

        if progress:
            progress.begin("depth_fill", len(views))
        started_at = time.perf_counter()
        try:
            class_names = json.loads(Path(args.classes_json).read_text(encoding="utf-8"))[args.class_set]
            report = fill_classified_from_prior(
                dense_dir, classified_raw, views, priors, fill_settings, scores_path=scores,
                labels_dir=labels_dir, progress=_estimator_progress(progress), class_names=class_names, mirrors=mirrors,
            )
        except Exception as error:  # noqa: BLE001 - the unfilled classification stays valid
            print(f"WARNING: Depth-Anything fill was skipped: {error}")
            report = {"enabled": True, "added_points": 0, "error": str(error)}
        else:
            summary.record_stage("Depth fill", time.perf_counter() - started_at)
            print(f"Depth-Anything fill added {report.get('added_points', 0):,} points confirmed by several views.")
        summary.settings["depth_fill"] = report
    cleaning = _cleaning_settings(args)
    if cleaning is not None and views is not None:
        if progress:
            progress.begin("point_cleaning")
        started_at = time.perf_counter()
        try:
            report = clean_classified_cloud(dense_dir, classified_raw, cleaning, scores_path=scores,
                                            views=views, prior_reader=priors)
        except Exception as error:  # noqa: BLE001 - the uncleaned cloud stays valid
            print(f"WARNING: Point cleaning was skipped: {error}")
            report = {"enabled": True, "removed_points": 0, "error": str(error)}
        else:
            summary.record_stage("Point cleaning", time.perf_counter() - started_at)
            print(f"Point cleaning removed {report.get('removed_points', 0):,} floating points.")
        summary.settings["point_cleaning"] = report
    if fusion is not None and views is not None:
        from samcloud.colmap.surface_fusion import consolidate_classified_cloud

        if progress:
            progress.begin("surface_fusion", len(views))
        started_at = time.perf_counter()
        try:
            class_names = json.loads(Path(args.classes_json).read_text(encoding="utf-8"))[args.class_set]
            report = consolidate_classified_cloud(
                dense_dir, classified_raw, views, priors, fusion, scores_path=scores,
                labels_dir=labels_dir, class_names=class_names, progress=_estimator_progress(progress), mirrors=mirrors,
            )
        except Exception as error:  # noqa: BLE001 - the unfused cloud stays valid
            print(f"WARNING: Surface fusion was skipped: {error}")
            report = {"enabled": True, "error": str(error)}
        else:
            summary.record_stage("Surface fusion", time.perf_counter() - started_at)
            print(f"Surface fusion moved {report.get('snapped', 0):,} points onto the fused surface, "
                  f"{report.get('final_points', 0):,} points remain.")
        summary.settings["surface_fusion"] = report
    settings = _plane_fill_settings(args)
    if settings is not None and views is not None:
        if progress:
            progress.begin("plane_fill")
        started_at = time.perf_counter()
        try:
            class_names = json.loads(Path(args.classes_json).read_text(encoding="utf-8"))[args.class_set]
            report = fill_classified_cloud(
                dense_dir, classified_raw, labels_dir, class_names, settings,
                scores_path=scores, views=views, prior_reader=priors,
            )
        except Exception as error:  # noqa: BLE001 - the unfilled classification stays valid
            print(f"WARNING: Plane filling was skipped: {error}")
            report = {"enabled": True, "added_points": 0, "error": str(error)}
        else:
            summary.record_stage("Plane filling", time.perf_counter() - started_at)
            print(f"Plane filling added {report.get('added_points', 0):,} points (SAM + plane + free space).")
        summary.settings["plane_fill"] = report
    superpoint = _superpoint_vote_settings(args)
    if superpoint is not None:
        from samcloud.colmap.superpoint_vote import vote_classified_cloud

        if progress:
            progress.begin("superpoint_vote")
        started_at = time.perf_counter()
        try:
            class_names = json.loads(Path(args.classes_json).read_text(encoding="utf-8"))[args.class_set]
            report = vote_classified_cloud(
                classified_raw, superpoint, class_names, scores_path=scores,
                report_dir=report_dir or summary.report_dir, progress=print,
            )
        except Exception as error:  # noqa: BLE001 - the pre-vote cloud remains valid
            print(f"WARNING: Superpoint voting was skipped: {error}")
            report = {"enabled": True, "error": str(error)}
        else:
            summary.record_stage("Superpoint voting", time.perf_counter() - started_at)
            print(f"Superpoint voting changed {report.get('changed_points', {}).get('count', 0):,} points.")
        summary.settings["superpoint_vote"] = report
    # The depth maps and SAM rasters are released immediately below.  Preserve
    # their fused BIM evidence first, independently of keep_processing_cache.
    if getattr(args, "class_set", None) == "indoor":
        started_at = time.perf_counter()
        try:
            from samcloud.bim.depth_evidence import build_depth_product

            if progress:
                progress.update(96, "Writing compact depth evidence")
            report = build_depth_product(dense_dir.parent.parent, dense_dir=dense_dir, classified_path=classified_raw)
        except Exception as error:  # noqa: BLE001 - evidence must never fail the processing run
            print(f"WARNING: Compact depth evidence was skipped: {error}")
            report = {"enabled": True, "written": False, "error": str(error)}
        else:
            summary.record_stage("Depth evidence", time.perf_counter() - started_at)
            if report.get("written"):
                print(f"Compact depth evidence wrote {report.get('points', 0):,} points from {report.get('views_used', 0)} views.")
        summary.settings["depth_evidence"] = report
    if keep_caches:  # running on the processing cache itself: never delete it
        return
    if (dense_dir.parent.parent / "project.samcloud.yml").is_file():
        for generated in (labels_dir, dense_dir / "depth_prior"):
            if generated.exists():
                remove_generated(generated, dense_dir.parent)
    _release_fusion_cache(dense_dir, summary)


def _attach_fine_terms(args, classified_scores: Path, classified_output: Path, summary: RunSummary) -> None:
    """Write the per-point fine SAM term into the product PLY (property ``fine_term``) and fine_labels.json."""
    source = Path(classified_scores).with_name(SOURCE_NAME)
    if not source.is_file() or not Path(classified_output).is_file():
        return
    try:
        report = attach_fine_terms(source, classified_output, Path(args.classes_json), args.class_set)
    except Exception as error:  # noqa: BLE001 - the classified cloud stays valid without fine terms
        print(f"WARNING: Fine terms were not written: {error}")
        summary.settings["fine_terms"] = {"enabled": True, "error": str(error)}
        return
    summary.settings["fine_terms"] = {key: value for key, value in report.items() if key != "labels"}
    print(f"Fine terms: {report['labelled_points']:,} of {report['points']:,} points carry a fine term "
          f"({len(report['labels'])} labels, {LABELS_NAME} written next to the PLY).")


def _instance_params(args) -> dict:
    return {
        "min_points": int(args.instance_min_points),
        "attach_radius_m": float(args.instance_attach_radius_m),
    }


def _instances_active(args) -> bool:
    """Instance IDs are produced only when enabled and not switched off."""
    return getattr(args, "instances_enabled", "0") == "1" and not bool(getattr(args, "no_instances", False))


def _discard_stale_instances(classified_raw: Path) -> None:
    """A fresh classification invalidates any instance snapshot left by an earlier run."""
    stale = Path(classified_raw).with_name(INSTANCES_SOURCE_NAME)
    try:
        stale.unlink(missing_ok=True)
    except OSError as error:
        print(f"WARNING: Stale instance snapshot could not be removed: {error}")


def _snapshot_instances_safely(args, classified_raw: Path) -> None:
    """Set the instance IDs aside for step 6; a failure must never abort the classification."""
    if not _instances_active(args):
        return
    try:
        snapshot_instances(classified_raw, Path(classified_raw).with_name(INSTANCES_SOURCE_NAME))
    except Exception as error:  # noqa: BLE001 - classifications remain usable without instances
        print(f"WARNING: Instances could not be set aside for step 6: {error}")


def _attach_instances(args, classified_scores: Path, classified_output: Path, summary: RunSummary) -> None:
    """Restore the per-point IDs after step 6 has produced its final point set."""
    if not _instances_active(args):
        return
    source = Path(classified_scores).with_name(INSTANCES_SOURCE_NAME)
    if not source.is_file() or not Path(classified_output).is_file():
        return
    try:
        report = attach_instances(source, classified_output, _instance_params(args))
    except Exception as error:  # noqa: BLE001 - product PLY remains valid without an instance field
        print(f"WARNING: Instances were not written: {error}")
        summary.settings["instances"] = {"enabled": True, "error": str(error)}
        return
    summary.settings["instances"] = {key: value for key, value in report.items() if key != "instances"}
    print(f"Instances: {report.get('count', 0):,} objects, {report.get('points', 0):,} points carry an instance ID.")


def _cache_after_sam(args, dense_dir: Path, classified_raw: Path, classified_scores: Path) -> None:
    """Keep the SAM result and dense workspace when the project wants a processing cache."""
    if str(getattr(args, "keep_processing_cache", "0")) != "1":
        return
    from samcloud.engine.processing_cache import save_after_sam

    try:
        manifest = save_after_sam(
            dense_dir.parent.parent, dense_dir, classified_raw, classified_scores,
            class_set=args.class_set, capture_type=getattr(args, "capture_type", None),
        )
    except OSError as error:
        print(f"WARNING: Processing cache was not saved: {error}")
    else:
        print(f"Processing cache saved for step 6 re-runs: {manifest['files']}")


def _refine_from_cache(
    args, outdir: Path, report_dir: Path, summary: RunSummary, progress: "ProgressTracker | None",
    classified_output: Path,
) -> None:
    """Re-run workflow step 6 from the processing cache (no COLMAP, no SAM)."""
    from samcloud.engine.processing_cache import (
        INSTANCES_SOURCE, SAM_CLOUD, SAM_SCORES, cache_directory, load_manifest, stale_cache_reason,
    )

    project_directory = outdir.parent
    manifest = load_manifest(project_directory)
    if manifest is None:
        sys.exit("No usable processing cache: enable 'Schritt 6 schnell neu rechnen' and process the project once.")
    if manifest.get("class_set") != args.class_set:
        sys.exit(f"The processing cache was made with class set {manifest.get('class_set')}, not {args.class_set}.")
    stale = stale_cache_reason(project_directory)
    if stale:
        sys.exit(stale)
    cache = cache_directory(project_directory)
    dense_dir = cache / "dense"
    geometry_enabled = args.geometry_refinement == "1"
    classified_raw = classified_output.with_name("classified_raw.ply") if geometry_enabled else classified_output
    classified_scores = classified_output.parent / "classified_scores.npz"
    classified_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache / SAM_CLOUD, classified_raw)
    if (cache / SAM_SCORES).is_file():
        shutil.copy2(cache / SAM_SCORES, classified_scores)
    elif classified_scores.exists():
        classified_scores.unlink()
    if (cache / INSTANCES_SOURCE).is_file():
        shutil.copy2(cache / INSTANCES_SOURCE, classified_scores.with_name(INSTANCES_SOURCE))
    views = int(manifest.get("files", {}).get("images", 0)) or 1
    if progress:
        plan = []
        if _depth_prior_settings(args) is not None and not manifest.get("depth_prior"):
            plan.append(("depth_prior", views))
        if _depth_fill_settings(args) is not None:
            plan.append(("depth_fill", views))
        if _cleaning_settings(args) is not None:
            plan.append(("point_cleaning", views))
        if _surface_fusion_settings(args) is not None:
            plan.append(("surface_fusion", views))
        if _plane_fill_settings(args) is not None:
            plan.append(("plane_fill", views))
        if _superpoint_vote_settings(args) is not None:
            plan.append(("superpoint_vote", views))
        if geometry_enabled:
            plan.append(("geometry_refinement", views))
        progress.estimator = ProgressEstimator(
            plan, RateStore(PROJECT_ROOT / "projects" / ".samcloud_stage_rates.json",
                            profile=getattr(args, "capture_type", None)),
        )
        global _ACTIVE_PROGRESS
        _ACTIVE_PROGRESS = progress
    _plan_geometry_refinement(progress, _is_georeferenced(outdir))
    summary.settings["refined_from_cache"] = manifest.get("created_at")
    print(f"Re-running step 6 from the processing cache of {manifest.get('created_at')}.")
    _refine_after_classification(args, dense_dir, classified_raw, classified_scores, summary, progress,
                                 report_dir=report_dir, keep_caches=True, reuse_priors=True)
    georeferenced = _is_georeferenced(outdir)
    _finish_classification(args, outdir, report_dir, summary, progress, classified_raw, classified_scores,
                           classified_output, georeferenced)


def _classification_engine(args) -> str:
    """2D model for classify.py: FlurSeg only for the outdoor class set, SAM3 otherwise."""
    engine = getattr(args, "classification_engine", "sam3") or "sam3"
    return engine if engine == "flurseg" and args.class_set == "outdoor" else "sam3"


def _writes_scores(args) -> bool:
    """Whether classify.py writes weighted-vote confidence scores (SegEarth-OV3 voting or FlurSeg)."""
    return args.classification_mode == "segearth_ov3" or _classification_engine(args) == "flurseg"


def _instance_arguments(args) -> list[str]:
    """Forward the instance configuration to the standalone classifier CLI."""
    values = [
        "--instances-enabled", "0" if args.no_instances else "1" if args.instances_enabled == "1" else "0",
        "--instance-query-threshold", str(args.instance_query_threshold),
        "--instance-mask-threshold", str(args.instance_mask_threshold),
        "--instance-min-mask-pixels", str(args.instance_min_mask_pixels),
        "--instance-merge-overlap", str(args.instance_merge_overlap),
        "--instance-min-shared-points", str(args.instance_min_shared_points),
        "--instance-min-views", str(args.instance_min_views),
        "--instance-min-points", str(args.instance_min_points),
        "--instance-split-radius-m", str(args.instance_split_radius_m),
        "--instance-debug-ply", "1" if args.instance_debug_ply == "1" else "0",
    ]
    return [*values, *( ["--no-instances"] if args.no_instances else [])]


RECLASSIFY_RESERVE_BYTES = 5 * 1024 ** 3  # working space for SAM rasters and depth priors next to copied inputs


def _reclassify_project(
    args, outdir: Path, report_dir: Path, summary: RunSummary, progress: "ProgressTracker | None",
    classified_output: Path,
) -> None:
    """Re-run only classification and step 6 on the project's finished dense cloud.

    Uses the kept classification inputs (fused.ply, fused.ply.vis, dense/sparse;
    zstd-compressed after compaction). When the undistorted images were released,
    they are recreated from the input photos with COLMAP's image_undistorter at
    the resolution of the dense cameras, so every engine sees exactly the same
    views and points. Depth maps are not recomputed: the per-view depth check and
    the indoor depth steps are skipped when they are gone.
    """
    import pycolmap

    from samcloud.core.sparse_model import resolve_sparse_model

    dense_dir = outdir / "dense"
    cache_dense = outdir.parent / "processing_cache" / "dense"
    staged = not (dense_dir / "fused.ply").exists() and (cache_dense / "fused.ply").is_file()
    if staged:
        dense_dir = _stage_reclassify_cache(cache_dense)
        print(f"Reclassifying from the processing cache; staging folder: {dense_dir.parent}")
    for name in ("fused.ply", "fused.ply.vis"):
        _ensure_if_compressed(dense_dir / name)
    missing = [str(path) for path in (dense_dir / "fused.ply", dense_dir / "fused.ply.vis", dense_dir / "sparse")
               if not path.exists()]
    if missing:
        raise ValueError("Cannot reclassify; the dense cloud was not kept (storage option "
                         "keep_classification_inputs): " + ", ".join(missing))
    if progress:
        progress.update(87, "Preparing images for reclassification")
    images_dir = dense_dir / "images"
    if not images_dir.is_dir() or not any(images_dir.iterdir()):
        source_images = Path(args.images).resolve() if args.images else (
            Path(args.panorama_workdir).resolve() / "cubemap" if args.panorama_workdir else None)
        if source_images is None or not source_images.is_dir():
            raise ValueError(f"Cannot reclassify; the input image folder is missing: {source_images}")
        input_model = resolve_sparse_model(outdir / "sparse")
        if input_model is None:
            raise ValueError(f"Cannot reclassify; no sparse model found in {outdir / 'sparse'}")
        dense_cameras = pycolmap.Reconstruction(str(dense_dir / "sparse")).cameras
        max_size = max(max(int(camera.width), int(camera.height)) for camera in dense_cameras.values())
        workspace = dense_dir / "_reclassify_undistort"
        if workspace.exists():
            remove_generated(workspace, outdir)
        command = [find_colmap_exe(args.colmap_bin), "image_undistorter", "--image_path", str(source_images),
                   "--input_path", str(input_model), "--output_path", str(workspace), "--output_type", "COLMAP",
                   "--max_image_size", str(max_size)]
        if (outdir / "image_list.txt").is_file():
            command.extend(["--image_list_path", str(outdir / "image_list.txt")])
        run(command, summary=summary, stage="Preparing images for reclassification")
        new_cameras = pycolmap.Reconstruction(str(workspace / "sparse")).cameras
        for camera_id, camera in dense_cameras.items():
            other = new_cameras[camera_id] if camera_id in new_cameras else None
            if other is None or (other.width, other.height) != (camera.width, camera.height) or not np.allclose(
                    np.asarray(other.params), np.asarray(camera.params), rtol=1e-4, atol=1e-3):
                raise ValueError(f"Recreated undistorted camera {camera_id} does not match the dense cloud camera; "
                                 "the photos or the sparse model changed since the dense cloud was built")
        if images_dir.exists():
            remove_generated(images_dir, outdir)
        shutil.move(str(workspace / "images"), str(images_dir))
        remove_generated(workspace, outdir)
    sparse_txt = ensure_text_model(dense_dir / "sparse", dense_dir / "sparse_txt")
    georeferenced = _is_georeferenced(outdir)
    if not georeferenced and (outdir / "gps_ref.txt").is_file():
        try:
            _write_camera_centre_georeference(sparse_txt, outdir / "gps_ref.txt", outdir / "georeference.json")
            georeferenced = True
        except Exception as error:  # noqa: BLE001
            print(f"WARNING: Could not create GPS georeference: {error}")
    summary.settings["reclassified"] = True
    _classify_and_finish(
        args, outdir, dense_dir, sparse_txt, report_dir, summary, progress, classified_output, georeferenced,
        cache_after_sam=False, keep_caches=staged,
    )


def _stage_reclassify_cache(cache_dense: Path) -> Path:
    """Create a disposable read-only-input staging workspace from a processing cache.

    The workspace is a sibling of ``processing_cache`` in the project. It is thus on the volume of the cache, where
    hard links work, and step 6 finds the project (metric scale, depth product) two levels above it exactly as when it
    runs on the cache. The cached files are only linked as inputs. Files step 6 may produce are deliberately omitted,
    so SAM and refinement never modify the cache copy. When a link is impossible the file is copied, but only after a
    check that all copies and the step 6 output fit.
    """
    cache_dense = Path(cache_dense)
    staging_base = cache_dense.parent.parent
    # ``tempfile.mkdtemp`` can create an ACL-isolated directory under the
    # Windows service account. Make the unique directory directly instead.
    staging_root = staging_base / f"reclassify_staging_{uuid.uuid4().hex}"
    staging_root.mkdir()
    staging = staging_root / "dense"
    mutable_trees = {"sam_labels", "sam_mirror", "depth_prior"}
    # Step 6 writes reports beside the read-only dense inputs. Do not stage
    # report files as hard links: replacing or truncating one must never alter
    # the cache inode.
    mutable_files = {"surface_fusion.json", "depth_prior.json", "depth_fill.json", "point_cleaning.json", "plane_fill.json"}
    unlinked: list[tuple[Path, Path]] = []
    for source in cache_dense.rglob("*"):
        relative = source.relative_to(cache_dense)
        if relative.parts and relative.parts[0] in mutable_trees:
            continue
        if source.is_file() and source.name not in mutable_files and source.suffix.lower() != ".json":
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                unlinked.append((source, target))
    if unlinked:
        needed = sum(source.stat().st_size for source, _target in unlinked) + RECLASSIFY_RESERVE_BYTES
        free = shutil.disk_usage(staging_base).free
        if needed > free:
            raise ValueError(
                f"Cannot reclassify; {len(unlinked)} cached files cannot be hard-linked into {staging_root} and "
                f"copying them needs {needed / 1024 ** 3:.1f} GB (including working space), but only "
                f"{free / 1024 ** 3:.1f} GB are free. Free some space on that drive or keep the processing cache "
                "on the drive of the project."
            )
        for source, target in unlinked:
            shutil.copy2(source, target)
    return staging


def _classify_and_finish(
    args, outdir: Path, dense_dir: Path, dense_sparse_txt: Path, report_dir: Path,
    summary: RunSummary, progress: ProgressTracker | None, classified_output: Path, georeferenced: bool,
    *, cache_after_sam: bool = True, keep_caches: bool = False,
) -> None:
    """Run the unchanged classification, refinement, and optional export tail."""
    _plan_geometry_refinement(progress, georeferenced)
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
        "--classes-json", str(args.classes_json), "--class-set", args.class_set,
        "--min-votes", str(args.min_votes), "--tile-size", str(args.sam_tile_size),
        "--tile-min-overlap", str(args.sam_tile_min_overlap), "--mode", args.classification_mode,
        "--engine", _classification_engine(args),
        "--background-threshold", str(args.background_threshold), "--query-threshold", str(args.query_threshold),
        "--summary-file", str(report_dir / "classification_summary.json"), "--out", str(classified_raw),
        *_labels_arguments(args, dense_dir), *_instance_arguments(args)]
    if args.threshold_sweep:
        classify_command.extend(["--threshold-sweep", args.threshold_sweep])
    if _writes_scores(args):
        classify_command.extend(["--scores-out", str(classified_scores)])
    if args.progress_file:
        classify_command.extend(["--progress-file", str(Path(args.progress_file).with_name("classification_progress.json"))])
    _discard_stale_instances(classified_raw)
    run(classify_command, summary=summary, stage="SAM3 classification")
    _snapshot_instances_safely(args, classified_raw)
    if cache_after_sam:
        _cache_after_sam(args, dense_dir, classified_raw, classified_scores)
    else:  # a reclassification: the cached SAM result is the old one from here on
        from samcloud.engine.processing_cache import mark_reclassified

        if mark_reclassified(outdir.parent):
            print("The processing cache still holds the previous classes; step 6 cannot be recomputed from it any more.")
    _refine_after_classification(args, dense_dir, classified_raw, classified_scores, summary, progress,
                                 report_dir=report_dir, keep_caches=keep_caches)
    _finish_classification(args, outdir, report_dir, summary, progress, classified_raw, classified_scores,
                           classified_output, georeferenced)


def _finish_classification(
    args, outdir: Path, report_dir: Path, summary: RunSummary, progress: "ProgressTracker | None",
    classified_raw: Path, classified_scores: Path, classified_output: Path, georeferenced: bool,
) -> None:
    """Geometric refinement (end of step 6), optional LAS export, compaction and summary."""
    geometry_enabled = args.geometry_refinement == "1"
    if geometry_enabled and georeferenced:
        refine_script = Path(__file__).resolve().parent / "geometry" / "refine.py"
        _ensure_if_compressed(classified_raw)
        with_scores = _writes_scores(args) and (
            classified_scores.is_file() or Path(f"{classified_scores}.zst").is_file())
        if with_scores:
            _ensure_if_compressed(classified_scores)
        if progress:
            progress.begin("geometry_refinement")
        run([sys.executable, str(refine_script), "--input", str(classified_raw), "--output", str(classified_output),
            "--classes-json", str(args.classes_json), "--class-set", args.class_set, "--engine", _classification_engine(args),
            "--config", str(PROJECT_ROOT / "config" / "geometry.yml"), "--summary-file", str(report_dir / "geometry_summary.json"),
            *(["--scores", str(classified_scores)] if with_scores else [])], summary=summary, stage="Geometric refinement")
    elif geometry_enabled:
        if progress and progress.estimator is not None:
            progress.estimator.skip_stage("geometry_refinement")
        shutil.copy2(classified_raw, classified_output)
    _attach_fine_terms(args, classified_scores, classified_output, summary)
    _attach_instances(args, classified_scores, classified_output, summary)
    if args.skip_export:
        if progress:
            progress.finish()
            progress.update(100, "Complete")
        _compact_completed_project(outdir, summary)
        summary.write("Completed")
        return
    run([sys.executable, str(Path(__file__).resolve().parent / "export_las.py"), "--input", str(classified_output),
         "--output", str(outdir / "classified.las"), "--classes-json", str(args.classes_json),
         "--class-set", args.class_set, "--engine", _classification_engine(args), "--source-crs", args.source_crs, "--target-crs", args.export_crs,
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
    ap.add_argument("--video", help="camera video file or folder of videos; frames are extracted into --images")
    ap.add_argument("--panorama-video", help="2:1 MP4 panorama video or folder of such videos")
    ap.add_argument("--panorama-workdir", help="project-owned directory for frames, cubemaps, and manifest")
    ap.add_argument("--frame-interval", type=float, default=1.0,
                    help="fixed 360 degree video frame interval in seconds")
    ap.add_argument("--cubemap-face-size", type=int, default=0,
                    help="side length in pixels for generated cubemap faces; 0 = native (panorama width / 4)")
    ap.add_argument("--outdir", required=True, help="project/output folder")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="minimum agreeing images per point; default 1 keeps single-view classifications")
    ap.add_argument("--sam-tile-size", type=int, default=1008,
                    help="SAM3 input tile size in pixels")
    ap.add_argument("--sam-tile-min-overlap", type=float, default=0.2,
                    help="SAM3 tile overlap fraction")
    ap.add_argument("--classification-mode", choices=("legacy", "segearth_ov3"), default="segearth_ov3",
                    help="SAM3 classification path (default: segearth_ov3; legacy remains available for comparison)")
    ap.add_argument("--classification-engine", choices=("sam3", "flurseg"), default="sam3",
                    help="2D model for the outdoor class set: sam3 (default) or flurseg (own EoMT model); "
                         "indoor always uses sam3")
    ap.add_argument("--reclassify", action="store_true",
                    help="only re-run classification and step 6 on the project's finished dense cloud")
    ap.add_argument("--background-threshold", type=float, default=0.4,
                    help="SegEarth-OV3 background threshold")
    ap.add_argument("--query-threshold", type=float, default=0.4,
                    help="SegEarth-OV3 decoder query threshold")
    ap.add_argument("--instances-enabled", choices=("0", "1"), default="0")
    ap.add_argument("--no-instances", action="store_true", help="disable SAM object instance IDs")
    ap.add_argument("--instance-query-threshold", type=float, default=0.5)
    ap.add_argument("--instance-mask-threshold", type=float, default=0.5)
    ap.add_argument("--instance-min-mask-pixels", type=int, default=12)
    ap.add_argument("--instance-merge-overlap", type=float, default=0.5)
    ap.add_argument("--instance-min-shared-points", type=int, default=10)
    ap.add_argument("--instance-min-views", type=int, default=2)
    ap.add_argument("--instance-min-points", type=int, default=100)
    ap.add_argument("--instance-split-radius-m", type=float, default=0.05)
    ap.add_argument("--instance-attach-radius-m", type=float, default=0.05)
    ap.add_argument("--instance-debug-ply", choices=("0", "1"), default="0")
    ap.add_argument("--threshold-sweep", default="",
                    help="optional comma-separated SegEarth-OV3 background thresholds")
    ap.add_argument("--classes-json", default=str(PROJECT_ROOT / "config" / "classes.json"),
                    help="class configuration (a project's own config/classes.json when its classes are customised)")
    ap.add_argument("--class-set", choices=("outdoor", "indoor"), default="outdoor",
                    help="semantic class set used for SAM3 classification")
    ap.add_argument("--keep-processing-cache", choices=("0", "1"), default="0",
                    help="keep SAM result and dense workspace so step 6 can be re-run quickly")
    ap.add_argument("--refine-from-cache", action="store_true",
                    help="only re-run step 6 (improve & filter) from the project's processing cache")
    ap.add_argument("--depth-fill", choices=("0", "1"), default="1",
                    help="indoor step 6: 1 = use Depth-Anything-V2 to fill textureless gaps (only where it agrees with COLMAP)")
    ap.add_argument("--surface-fusion", choices=("0", "1"), default="1",
                    help="indoor step 6: 1 = merge all depth maps into one surface layer (TSDF)")
    ap.add_argument("--depth-model", choices=("large", "small", "base", "off"), default="large",
                    help="Depth-Anything-V2 model for the indoor step 6 (default large; off disables it)")
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
        choices=("none", "after_alignment", "automatic", "auto"),
        default="auto",
        help="use surveyed points automatically when available, or select an explicit control-point workflow",
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
    ap.add_argument(
        "--resume-dense-stereo",
        action="store_true",
        help="reuse the undistorted dense workspace and continue from dense stereo",
    )
    ap.add_argument(
        "--resume-classification",
        action="store_true",
        help="reuse the fused dense workspace and continue directly at SAM classification",
    )
    ap.add_argument("--control-points-dir", default=None, help="project control_points directory")
    ap.add_argument("--control-points-crs", default="EPSG:2056+5728")
    ap.add_argument("--apriltags", action="store_true", help="search AprilTags and scale the sparse model from their size")
    ap.add_argument(
        "--progress-file",
        default=None,
        help="path for the dashboard progress JSON file",
    )
    ap.add_argument(
        "--auto-dxf", choices=("0", "1"), default="0",
        help="after a successful run, create the DXF/IFC export as a separate final stage",
    )
    args = ap.parse_args()
    if args.video and args.capture_type != "camera":
        ap.error("--video is only supported with --capture-type camera")
    _request_auto_dxf(args)

    if args.control_points_mode == "after_alignment":
        args.stop_after_alignment = True
    if args.resume_after_control_points and args.stop_after_alignment:
        ap.error("--resume-after-control-points cannot be combined with --stop-after-alignment")
    if args.resume_dense_stereo and args.stop_after_alignment:
        ap.error("--resume-dense-stereo cannot be combined with --stop-after-alignment")
    if args.resume_dense_stereo and args.resume_after_control_points:
        ap.error("--resume-dense-stereo cannot be combined with --resume-after-control-points")
    if args.resume_classification and args.stop_after_alignment:
        ap.error("--resume-classification cannot be combined with --stop-after-alignment")
    if args.resume_classification and args.resume_after_control_points:
        ap.error("--resume-classification cannot be combined with --resume-after-control-points")
    if args.resume_classification and args.resume_dense_stereo:
        ap.error("--resume-classification cannot be combined with --resume-dense-stereo")
    if args.classification_engine == "flurseg" and args.class_set != "outdoor":
        print("FlurSeg is outdoor only; the indoor class set is classified with SAM3.")

    progress = ProgressTracker(Path(args.progress_file)) if args.progress_file else None
    if progress:
        progress.update(0, "Preparing images")

    if args.gps_max_error is not None and args.gps_max_error <= 0:
        ap.error("--gps-max-error must be greater than zero")
    if args.frame_interval <= 0:
        ap.error("--frame-interval must be greater than zero")
    if args.cubemap_face_size < 0:
        ap.error("--cubemap-face-size must be 0 (native) or greater")
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
    args.control_points_mode = _resolve_control_points_mode(args.control_points_mode, control_points_dir)
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
        "classification_engine": _classification_engine(args),
        "background_threshold": args.background_threshold,
        "query_threshold": args.query_threshold,
        "instances_enabled": args.instances_enabled == "1" and not args.no_instances,
        "geometry_refinement": args.geometry_refinement == "1",
        "source_crs": args.source_crs,
        "export_crs": args.export_crs,
        "classified_output": str(classified_output),
        "classification_summary": str(report_dir / "classification_summary.json"),
    })
    if args.refine_from_cache:
        _refine_from_cache(args, outdir, report_dir, summary, progress, classified_output)
        return
    if args.reclassify:
        _reclassify_project(args, outdir, report_dir, summary, progress, classified_output)
        return
    if args.resume_after_control_points:
        resume_images = Path(args.images).resolve() if args.images else outdir / "dense" / "images"
        dense_max_image_size, dense_reason = _dense_size_from_inputs(dense_config, outdir, resume_images, getattr(args, "capture_type", None))
        if dense_reason:
            print(dense_reason)
        summary.settings["dense_max_image_size"] = dense_max_image_size
        summary.settings["dense_size_reason"] = dense_reason
        _resume_after_control_points(
            args, outdir, report_dir, summary, progress, colmap_config, dense_max_image_size,
            dense_num_iterations, dense_gpu_index, classified_output,
        )
        return
    if args.resume_classification:
        dense_max_image_size, dense_reason = _dense_size_from_inputs(dense_config, outdir, outdir / "dense" / "images", getattr(args, "capture_type", None))
        if dense_reason:
            print(dense_reason)
        summary.settings["dense_max_image_size"] = dense_max_image_size
        summary.settings["dense_size_reason"] = dense_reason
        _resume_classification(
            args, outdir, report_dir, summary, progress, colmap_config, dense_max_image_size,
            dense_num_iterations, dense_gpu_index, classified_output,
        )
        return
    if args.resume_dense_stereo:
        dense_max_image_size, dense_reason = _dense_size_from_inputs(dense_config, outdir, outdir / "dense" / "images", getattr(args, "capture_type", None))
        if dense_reason:
            print(dense_reason)
        summary.settings["dense_max_image_size"] = dense_max_image_size
        summary.settings["dense_size_reason"] = dense_reason
        _resume_dense_stereo(
            args, outdir, report_dir, summary, progress, colmap_config, dense_max_image_size,
            dense_num_iterations, dense_gpu_index, classified_output,
        )
        return
    if progress:
        progress.estimator = _initial_estimator(args, colmap_config, outdir, dense_max_image_size)
        progress.begin("prepare_images")
    prepare_started_at = time.perf_counter()
    images_dir, is_panorama, cubemap_face_size = resolve_input_images(
        args, outdir, ap, progress.report_prepare if progress else None,
    )
    summary.record_stage("Preparing images", time.perf_counter() - prepare_started_at)

    db_path = outdir / "database.db"
    sparse_dir = outdir / "sparse"
    dense_dir = outdir / "dense"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)
    image_list_path = outdir / "image_list.txt"
    panorama_face_roots = ("front", "right", "back", "left", "up", "down") if is_panorama else None
    num_images = write_image_list(images_dir, image_list_path, include_roots=panorama_face_roots)
    dense_max_image_size, dense_reason = _dense_size_from_inputs(dense_config, outdir, images_dir, getattr(args, "capture_type", None))
    if dense_reason:
        print(dense_reason)
    summary.settings["dense_max_image_size"] = dense_max_image_size
    summary.settings["dense_size_reason"] = dense_reason
    summary.settings["image_count"] = num_images
    print(f"Using {num_images} supported images from {images_dir}")
    if progress:
        if progress.estimator is not None and progress.estimator.current is not None:
            # Learn the preparation rate from the real source count, then plan the rest with real image counts.
            progress.estimator.set_units("prepare_images", num_images / 6 if is_panorama else num_images)
            progress.finish()
            progress.estimator.replan(_build_estimator(
                args, num_images, colmap_config, resume=False, dense_max_image_size=dense_max_image_size,
            ).plan)
        else:
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
             "--input-height", str(georeferencing_config.get("input_height", "ellipsoidal")),
             "--require-all"], summary=summary, stage="Preparing GPS references")

    # 1) Feature extraction
    if progress:
        progress.begin("feature_extraction")
        progress.update(5, "Extracting features")
    camera_model = resolve_camera_model(camera_config.get("camera_model"), is_panorama)
    summary.settings["alignment_quality"]["camera_model"] = camera_model
    if is_panorama:
        if cubemap_face_size is None:
            raise RuntimeError("Panorama cubemap face size is missing")
        cubemap_focal_length = cubemap_face_size / 2
        feature_command = feature_extractor_command(
            colmap, db_path, images_dir, image_list_path,
            gpu=args.gpu == "1",
            camera_model=camera_model,
            max_image_size=max_image_size,
            max_num_features=int(features_config.get("max_num_features", 16384)),
            estimate_affine_shape=bool(features_config.get("estimate_affine_shape", False)),
            domain_size_pooling=bool(features_config.get("domain_size_pooling", True)),
            capture_type=getattr(args, "capture_type", None),
            panorama_peak_threshold=float(features_config.get("panorama_peak_threshold", 0.003)),
            panorama_edge_threshold=int(features_config.get("panorama_edge_threshold", 20)),
            single_camera=False,
            single_camera_per_folder=True,
            camera_params=(
                f"{cubemap_focal_length:g},{cubemap_focal_length:g},"
                f"{cubemap_focal_length:g},{cubemap_focal_length:g}"
            ),
        )
        feature_result = run(
            feature_command, summary=summary, stage="Feature extraction",
            allow_failure=bool(features_config.get("domain_size_pooling", True)),
        )
        if getattr(feature_result, "returncode", 0) and bool(features_config.get("domain_size_pooling", True)):
            print("WARNING: COLMAP rejected domain-size pooling; retrying feature extraction without it.")
            feature_command[feature_command.index("--SiftExtraction.domain_size_pooling") + 1] = "0"
            run(feature_command, summary=summary, stage="Feature extraction")
        rig_config_path = write_panorama_rig_config(outdir / "rig_config.json", cubemap_face_size)
        run(
            [colmap, "rig_configurator", "--database_path", str(db_path),
             "--rig_config_path", str(rig_config_path)],
            summary=summary, stage="Configuring panorama rig",
        )
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

    # 2) Cubemap faces need matching across multiple neighbouring panoramas.
    if progress:
        progress.begin("matching")
    if is_panorama:
        if progress:
            progress.update(18, "Matching images")
        panorama_strategy = select_panorama_matching_strategy(num_images)
        summary.settings["alignment_quality"]["matching_strategy"] = f"panorama_{panorama_strategy}"
        run(
            panorama_matching_command(colmap, db_path, gpu=args.gpu == "1", image_count=num_images),
            summary=summary, stage="Image matching",
        )
        if progress:
            progress.update(32, "Matching images")
    else:
        if progress:
            progress.update(18, "Matching images")
        matching_strategy = select_matching_strategy(
            str(alignment_config.get("matching_strategy", "auto")), num_images, has_pose_priors,
        ) if args.capture_type == "drone_gps" else "exhaustive"
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
        fixed_intrinsics=is_panorama,
        fixed_rig_sensors=is_panorama,
        init_min_num_inliers=30 if is_panorama else None,
    ), summary=summary, stage="Sparse reconstruction")
    if progress:
        progress.update(52, "Aligning cameras")

    sparse_models_info = _promote_largest_sparse_model(sparse_dir)
    summary.settings["sparse_models"] = sparse_models_info["models"]
    model0 = sparse_dir / "0"
    if not model0.exists():
        sys.exit("colmap mapper did not produce a reconstruction (sparse/0 missing). Check image overlap/quality.")
    _require_usable_sparse_model(model0, num_images)

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
    automatic_model_used = False
    if args.control_points_mode == "automatic":
        pristine_distorted = sparse_dir / "pre_gcp"
        if not pristine_distorted.exists():
            shutil.copytree(model0, pristine_distorted)
        pristine_text = sparse_dir / "pre_gcp_txt"
        pristine_text.mkdir(parents=True, exist_ok=True)
        run([colmap, "model_converter", "--input_path", str(model0), "--output_path", str(pristine_text), "--output_type", "TXT"], summary=summary, stage="Preparing automatic control points")
        if progress:
            progress.begin("control_points")
        if _is_tag_project(control_points_dir, gps_ref_path):
            automatic_model = _rerun_control_points(
                pristine_distorted, pristine_text, images_dir, control_points_dir, args.control_points_crs,
                gps_ref_path, dict(load_global_config("defaults.yml").get("control_points", {})),
                report_dir / "control_points_report.json", rtk=effective_rtk,
            )
        else:
            automatic_model = _automatic_control_points(
                pristine_distorted, pristine_text, images_dir, control_points_dir, args.control_points_crs,
                gps_ref_path, dict(load_global_config("defaults.yml").get("control_points", {})),
                report_dir / "control_points_report.json", rtk=effective_rtk, summary=summary,
            )
        if automatic_model is None:
            args.stop_after_alignment = True
            print("Automatic control-point quality gate did not pass. Waiting for control-point review.")
        else:
            model0 = automatic_model
            automatic_model_used = True

    tag_georeferenced = False
    if args.apriltags and not georeferenced and not automatic_model_used and args.control_points_mode != "automatic":
        try:
            if progress:
                progress.update(56, "Searching AprilTags")
            tag_result = _apriltag_step(
                model0, sparse_dir, images_dir, control_points_dir, report_dir,
                outlier_pct=float(load_global_config("defaults.yml").get("apriltags", {}).get("outlier_pct", 3.0)),
                crs=args.control_points_crs,
            )
            model0, tag_report = tag_result.model, tag_result.report
            summary.settings["apriltags"] = tag_report
            if tag_result.georeferenced:
                georeferenced = tag_georeferenced = True
            elif tag_result.gate_failed:
                args.stop_after_alignment = True
                print("AprilTag control-point quality gate did not pass. Waiting for control-point review.")
        except Exception as error:
            print(f"WARNING: AprilTag scaling failed, model stays unscaled: {error}")
    # The georeference state is final here; a skipped geometric refinement leaves the ETA now.
    _plan_geometry_refinement(progress, georeferenced)

    _release_sparse_intermediates(sparse_dir, model0)
    if args.capture_type == "panorama_360" and bool(alignment_config.get("gravity_align_panorama", True)) and not tag_georeferenced:
        model0, gravity_report = _align_panorama_gravity(sparse_dir, model0, report_dir)
        summary.settings["gravity_alignment"] = gravity_report
        _transform_tag_report_to_gravity(report_dir / "apriltag_report.json", gravity_report)
    elif (
        args.capture_type == "camera" and getattr(args, "class_set", None) == "indoor"
        and bool(alignment_config.get("gravity_align_camera_indoor", True))
        and not tag_georeferenced and not georeferenced and not automatic_model_used
    ):
        try:
            model0, gravity_report = _align_camera_gravity(sparse_dir, model0, report_dir)
            summary.settings["gravity_alignment"] = gravity_report
            _transform_tag_report_to_gravity(report_dir / "apriltag_report.json", gravity_report)
        except Exception as error:  # noqa: BLE001 - an unlevelled model is still usable
            print(f"WARNING: Camera gravity alignment failed, the model keeps its orientation: {error}")
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
        if georeferenced and not tag_georeferenced:
            try:
                _write_camera_centre_georeference(
                    dense_sparse_txt, gps_ref_path, outdir / "georeference.json",
                    automatic_model_used=automatic_model_used,
                )
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
    excluded_stereo_images = _run_dense_stereo(
        colmap, dense_dir, dense_config, dense_gpu_index, dense_num_iterations,
        dense_max_image_size, summary, capture_type=getattr(args, "capture_type", None),
    )
    summary.settings["dense_stereo_excluded_images"] = excluded_stereo_images
    if progress:
        progress.update(78, "Building dense cloud")

    # 6) Fusion -> fused.ply + fused.ply.vis
    if progress:
        progress.begin("dense_fusion")
        progress.update(78, "Fusing dense cloud")
    dense_dir.mkdir(parents=True, exist_ok=True)
    _fuse_dense(colmap, dense_dir, dense_config, getattr(args, "capture_type", None), summary,
        keep_depth_maps=_needs_depth_maps(args))
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
    if georeferenced and not tag_georeferenced:
        started_at = time.perf_counter()
        try:
            georeference = _write_camera_centre_georeference(
                dense_sparse_txt, gps_ref_path, georeference_path,
                automatic_model_used=automatic_model_used,
            )
        except Exception as error:
            print(f"WARNING: Could not create GPS georeference: {error}")
        else:
            if georeference is not None:
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
         "--classes-json", str(args.classes_json),
         "--class-set", args.class_set,
         "--min-votes", str(args.min_votes),
         "--tile-size", str(args.sam_tile_size),
         "--tile-min-overlap", str(args.sam_tile_min_overlap),
         "--mode", args.classification_mode,
         "--engine", _classification_engine(args),
         "--background-threshold", str(args.background_threshold),
         "--query-threshold", str(args.query_threshold),
         "--summary-file", str(report_dir / "classification_summary.json"),
         "--out", str(classified_raw),
         *_labels_arguments(args, dense_dir), *_instance_arguments(args)]
    if args.threshold_sweep:
        classify_command.extend(["--threshold-sweep", args.threshold_sweep])
    if _writes_scores(args):
        classify_command.extend(["--scores-out", str(classified_scores)])
    if args.progress_file:
        classify_command.extend(["--progress-file", str(Path(args.progress_file).with_name("classification_progress.json"))])
    _discard_stale_instances(classified_raw)
    run(classify_command, summary=summary, stage="SAM3 classification")
    _snapshot_instances_safely(args, classified_raw)
    _cache_after_sam(args, dense_dir, classified_raw, classified_scores)
    _refine_after_classification(args, dense_dir, classified_raw, classified_scores, summary, progress,
                                 report_dir=report_dir)
    if progress:
        progress.update(97, "Improving and filtering")

    if geometry_enabled:
        geometry_summary = report_dir / "geometry_summary.json"
        if georeferenced:
            refine_script = Path(__file__).resolve().parent / "geometry" / "refine.py"
            _ensure_if_compressed(classified_raw)
            if _writes_scores(args):
                _ensure_if_compressed(classified_scores)
            if progress:
                progress.begin("geometry_refinement")
                progress.update(98, "Geometric refinement")
            run([
                sys.executable, str(refine_script),
                "--input", str(classified_raw),
                "--output", str(classified_output),
                "--classes-json", str(args.classes_json),
                "--class-set", args.class_set,
                "--engine", _classification_engine(args),
                "--config", str(PROJECT_ROOT / "config" / "geometry.yml"),
                "--summary-file", str(geometry_summary),
                *(["--scores", str(classified_scores)]
                  if _writes_scores(args) else []),
                *(["--progress-file", str(Path(args.progress_file).with_name("geometry_progress.json"))] if args.progress_file else []),
            ], summary=summary, stage="Geometric refinement")
        else:
            print("Geometric refinement skipped: reconstruction is not georeferenced.")
            if progress and progress.estimator is not None:
                progress.estimator.skip_stage("geometry_refinement")
            shutil.copy2(classified_raw, classified_output)
            geometry_summary.parent.mkdir(parents=True, exist_ok=True)
            geometry_summary.write_text(json.dumps({"skipped": "not georeferenced"}, indent=2), encoding="utf-8")

    _attach_fine_terms(args, classified_scores, classified_output, summary)
    _attach_instances(args, classified_scores, classified_output, summary)
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
         "--classes-json", str(args.classes_json),
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


_AUTO_DXF_REQUEST: dict | None = None


def _request_auto_dxf(args) -> None:
    """Remember whether the final export stage was requested (``--auto-dxf 1``) for this process."""
    global _AUTO_DXF_REQUEST
    requested = getattr(args, "auto_dxf", "0") == "1"
    flurseg = (getattr(args, "classification_engine", "sam3") or "sam3") == "flurseg"
    _AUTO_DXF_REQUEST = {
        "enabled": requested and not flurseg,
        "flurseg": requested and flurseg,
        "project_dir": Path(args.outdir).resolve().parent,
        "progress_file": args.progress_file,
        "started": time.time(),
        "capture_type": getattr(args, "capture_type", ""),
        "class_set": getattr(args, "class_set", ""),
        "rerun_environment": bool(getattr(args, "refine_from_cache", False) or getattr(args, "reclassify", False)),
    }


def _record_environment_stage(project_dir: Path, progress_file, result: dict | None, elapsed: float, error: Exception | None = None) -> None:
    """Add the final outdoor stage to reports without changing the completed pipeline run."""
    reports = project_dir / "reports"
    result = result if isinstance(result, dict) else {}
    steps = result.get("steps") if isinstance(result.get("steps"), dict) else {}
    statuses = {
        str(name): str(value.get("status", "unknown")) if isinstance(value, dict) else "unknown"
        for name, value in steps.items()
    }
    failures = [
        f"{name}: {value.get('error') or value.get('reason') or 'failed'}"
        for name, value in steps.items()
        if isinstance(value, dict) and value.get("status") == "failed"
    ]
    if error is not None:
        failures.append(str(error))
    status = "failed" if failures else "completed"
    report_path = Path(result.get("out_dir") or project_dir / "products" / "environment") / "environment_report.md"
    stage = {
        "status": status,
        "total_seconds": round(elapsed, 3),
        "steps": statuses,
        "report": str(report_path),
    }

    try:
        metrics_path = reports / "run_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_id = metrics.get("run_id") if isinstance(metrics, dict) else None
        metric_paths = [metrics_path]
        if run_id:
            metric_paths.append(reports / f"run_metrics_{run_id}.json")
        for path in metric_paths:
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            data.setdefault("stage_seconds", {})["Umgebungsmodell"] = round(elapsed, 3)
            data.setdefault("settings", {})["environment_model"] = stage
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError, TypeError):
        pass

    lines = ["", "Umgebungsmodell", "-" * 16, f"Status: {status}", f"Bericht: {report_path}"]
    lines.extend(f"{name}: {value}" for name, value in statuses.items())
    if failures:
        lines.append(f"WARNING [Umgebungsmodell] {'; '.join(failures)}")
    try:
        metrics = json.loads((reports / "run_metrics.json").read_text(encoding="utf-8"))
        run_id = metrics.get("run_id") if isinstance(metrics, dict) else None
        summary_paths = [reports / "run_summary.txt"]
        if run_id:
            summary_paths.append(reports / f"run_summary_{run_id}.txt")
        for path in summary_paths:
            if path.is_file():
                path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError):
        pass

    if progress_file:
        progress_path = Path(progress_file)
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
            if not isinstance(payload, dict):
                payload = {}
            payload.update({
                "percent": 100.0, "stage": "Umgebungsmodell", "detail": status,
                "stage_key": None, "updated_at": time.time(), "environment_model": stage,
            })
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(payload), encoding="utf-8")
        except (OSError, ValueError, TypeError):
            pass


def run_auto_dxf_stage() -> None:
    """Final stage after ``main()``: DXF/IFC or outdoor environment export, when requested and completed.

    run_metrics*.json and run_summary.txt are final at this point, so the pipeline's total time never
    contains the DXF/IFC time (the home page adds dxf_run_metrics.json separately). Never raises."""
    request = _AUTO_DXF_REQUEST
    if request and request.get("flurseg"):
        print("DXF/IFC export skipped: FlurSeg projects deliver point clouds only (DXF/IFC need SAM3).", flush=True)
        return
    if not request or not request["enabled"]:
        return
    if request.get("capture_type") == "drone_gps" or request.get("class_set") == "outdoor":
        started = time.perf_counter()
        result = None
        error = None
        try:
            from samcloud.engine.auto_dxf import pipeline_run_completed
            if not pipeline_run_completed(request["project_dir"], request["started"]):
                return
            from samcloud.drone.run_environment import run_environment
            if request.get("rerun_environment"):
                result = run_environment(request["project_dir"], steps=None, resume=False)
            else:
                result = run_environment(request["project_dir"], resume=True)
        except Exception as caught:  # noqa: BLE001 - the classified cloud remains a valid successful result
            error = caught
            print(f"WARNING: Umgebungsmodell konnte nicht erstellt werden: {caught}", flush=True)
        finally:
            if result is not None or error is not None:
                _record_environment_stage(request["project_dir"], request["progress_file"], result, time.perf_counter() - started, error)
                report_path = request["project_dir"] / "products" / "environment" / "environment_report.md"
                print(f"Umgebungsmodell-Bericht: {report_path}", flush=True)
        return
    try:
        from samcloud.engine.auto_dxf import run_after_pipeline
        run_after_pipeline(request["project_dir"], progress_file=request["progress_file"], not_before=request["started"])
    except Exception as error:
        print(f"DXF/IFC export (automatic) could not run: {error}", flush=True)


def run_cli() -> None:
    main()
    run_auto_dxf_stage()


if __name__ == "__main__":
    run_cli()
