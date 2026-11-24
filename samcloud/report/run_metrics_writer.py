"""Run-time writer of ``reports/processing_metrics.json`` (the data source of the processing protocol).

Called at the end of a pipeline run (``RunSummary.write``) and wrapped in try/except there: a failing metric only
adds a ``WARNING:`` line to the file's ``warnings`` list, never breaks or noticeably slows the run (big point
sets are chunked or sampled). Also holds the helpers that read project-level facts, shared with ``collect``.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from samcloud.report import metrics

LOGGER = logging.getLogger(__name__)
REPORT_VERSION = 1
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ------------------------------------------------------------------------------- shared helpers


def clean_json(value: Any) -> Any:
    """NaN/inf become None so the written JSON is valid for every reader."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def read_json_file(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_project_config(project_dir: Path) -> dict[str, Any]:
    """``project.samcloud.yml`` as a mapping ({} when missing or unreadable)."""
    try:
        import yaml

        data = yaml.safe_load((Path(project_dir) / "project.samcloud.yml").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - optional input
        return {}


def input_image_names(project_dir: Path) -> list[str] | None:
    """Names of the images given to the pipeline (``products/image_list.txt``), None when unknown."""
    path = Path(project_dir) / "products" / "image_list.txt"
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return None
    return lines or None


def final_sparse_model(project_dir: Path) -> Path | None:
    """The final COLMAP model of a project (regular products/sparse, else cached dense/sparse variants)."""
    from samcloud.core.sparse_model import resolve_sparse_model

    project_dir = Path(project_dir)
    for relative in ("products/sparse", "products/dense/sparse", "processing_cache/dense/sparse"):
        folder = project_dir / relative
        if folder.is_dir():
            model = resolve_sparse_model(folder)
            if model is not None:
                return model
    return None


def _image_directory(project_dir: Path, config: dict[str, Any]) -> Path | None:
    acquisition = config.get("acquisition") if isinstance(config.get("acquisition"), dict) else {}
    sources = acquisition.get("sources") if isinstance(acquisition.get("sources"), dict) else {}
    candidates = [sources.get("image_directory"), (config.get("paths") or {}).get("image_directory"),
                  sources.get("panorama_image_directory")]
    for value in candidates:
        if value and Path(str(value)).is_dir():
            return Path(str(value))
    return None


def camera_exif(image_dir: Path | None, names: list[str] | None) -> dict[str, Any] | None:
    """make / model / focal length / sensor info from the EXIF of the first input image (None when unavailable)."""
    if image_dir is None:
        return None
    try:
        import exifread

        first = None
        for name in (names or [])[:5]:
            if (image_dir / name).is_file():
                first = image_dir / name
                break
        if first is None:
            first = next((p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg"}), None)
        if first is None:
            return None
        with first.open("rb") as handle:
            tags = exifread.process_file(handle, details=False)

        def text(key: str) -> str | None:
            return str(tags[key]).strip() if key in tags else None

        def ratio(key: str) -> float | None:
            if key not in tags:
                return None
            value = tags[key].values[0]
            return float(value.num) / float(value.den) if getattr(value, "den", 1) else None

        block = {"make": text("Image Make"), "model": text("Image Model"), "focal_mm": ratio("EXIF FocalLength"),
                 "focal_35mm": text("EXIF FocalLengthIn35mmFilm"),
                 "sensor": {"pixel_width": text("EXIF ExifImageWidth"), "pixel_height": text("EXIF ExifImageLength")}}
        return block if any(block[k] for k in ("make", "model", "focal_mm")) else None
    except Exception:  # noqa: BLE001 - EXIF is optional
        return None


def import_block(project_dir: Path, run_metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Facts about the image input: source type, image count, frame interval, discarded frames, camera, GPS."""
    project_dir = Path(project_dir)
    config = read_project_config(project_dir)
    acquisition = config.get("acquisition") if isinstance(config.get("acquisition"), dict) else {}
    sources = acquisition.get("sources") if isinstance(acquisition.get("sources"), dict) else {}
    settings = (run_metrics or {}).get("settings") if isinstance((run_metrics or {}).get("settings"), dict) else {}
    capture = settings.get("capture_type") or acquisition.get("type")
    if capture == "panorama_360":
        source_type = "panorama"
    elif capture == "camera" and sources.get("video_path") or (project_dir / "inputs" / "video_frames_manifest.json").is_file():
        source_type = "video"
    else:
        source_type = "photos"
    names = input_image_names(project_dir)
    interval = None
    for key in ("video", "panorama"):
        block = acquisition.get(key)
        if isinstance(block, dict) and block.get("frame_interval_seconds") and (
                (key == "video" and source_type == "video") or (key == "panorama" and source_type == "panorama")):
            interval = block["frame_interval_seconds"]
    discarded: list[Any] = []
    manifest = project_dir / "inputs" / "video_frames_manifest.json"
    if manifest.is_file():
        try:
            data = read_json_file(manifest)
            discarded = list(data.get("discarded") or [])
            interval = data.get("frame_interval_seconds", interval)
        except Exception:  # noqa: BLE001 - optional input
            pass
    image_count = settings.get("image_count") or (run_metrics or {}).get("images") or (len(names) if names else None)
    return {
        "source_type": source_type, "image_count": image_count, "frame_interval": interval,
        "discarded_frames": discarded,
        "camera": camera_exif(_image_directory(project_dir, config), names) if source_type == "photos" else None,
        "gps": {"mode": settings.get("gps_mode"), "rtk": settings.get("auto_rtk") or None,
                "max_error_m": settings.get("gps_max_error_m")},
    }


def is_metric(project_dir: Path) -> bool:
    try:
        from samcloud.core.metric_scale import metric_scale_decision

        decision = metric_scale_decision(project_dir)
        return bool(getattr(decision, "metric", False) or getattr(decision, "known_scale", False))
    except Exception:  # noqa: BLE001
        return False


def classification_class_names(class_set: str | None) -> list[str]:
    """Short class names of a class set; class id = index + 1."""
    from samcloud.core.paths import CONFIG_DIRECTORY

    try:
        data = read_json_file(CONFIG_DIRECTORY / "classes.json")
        entries = data.get(class_set or "") or []
        return [str(entry).split(",", 1)[0].strip() for entry in entries]
    except Exception:  # noqa: BLE001
        return []


def scores_path(project_dir: Path) -> Path | None:
    base = Path(project_dir) / "products" / "classified" / "classified_scores.npz"
    for candidate in (base, base.with_name(base.name + ".gz")):
        if candidate.is_file():
            return base
    return None


def placeholder_confidences(project_dir: Path) -> tuple[float, ...]:
    """Confidence values that fill steps wrote for appended points (no own SAM score): depth fill 0.5, plane fill 1.0."""
    values: list[float] = []
    for name, value in (("depth_fill.json", 0.5), ("plane_fill.json", 1.0)):
        try:
            report = read_json_file(Path(project_dir) / "products" / "dense" / name)
        except (OSError, ValueError):
            continue
        if isinstance(report, dict) and any(isinstance(v, (int, float)) and v > 0 and "added" in k
                                            for k, v in report.items()):
            values.append(value)
    return tuple(values)


def cloud_for_density(project_dir: Path, run_metrics: dict[str, Any] | None = None) -> Path | None:
    candidates = [Path(project_dir) / "products" / "dense" / "fused.ply"]
    settings = (run_metrics or {}).get("settings")
    if isinstance(settings, dict) and settings.get("classified_output"):
        candidates.append(Path(str(settings["classified_output"])))
    candidates.append(Path(project_dir) / "products" / "classified" / "classified.ply")
    return next((p for p in candidates if p.is_file()), None)


# --------------------------------------------------------------------------------------- writer


def build_processing_metrics(project_dir: Path, run_id: str, run_metrics: dict[str, Any] | None = None,
                             dense_points_fused: int | None = None) -> dict[str, Any]:
    """All protocol metrics of one run; every metric is independent and fails into ``warnings``."""
    project_dir = Path(project_dir)
    warnings: list[str] = []
    settings = (run_metrics or {}).get("settings") if isinstance((run_metrics or {}).get("settings"), dict) else {}

    def guarded(name: str, function: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            return function()
        except Exception as error:  # noqa: BLE001 - a metric must never break a run
            warnings.append(f"WARNING: metric '{name}' failed: {error}")
            LOGGER.warning("processing metric %s failed: %s", name, error)
            return None
        finally:
            LOGGER.debug("metric %s took %.2fs", name, time.perf_counter() - started)

    model = final_sparse_model(project_dir)
    names = input_image_names(project_dir)
    alignment = guarded("alignment", lambda: metrics.alignment_metrics(model, names)) if model else None
    if model is None:
        warnings.append("WARNING: no sparse model found, alignment metrics not recorded")
    from samcloud.report.config import load_protocol_config

    config = load_protocol_config()
    overlap = guarded("overlap", lambda: metrics.overlap_grid(
        model, max_cells=int(config["overlap"]["max_cells"]))) if model else None

    def dense() -> dict[str, Any]:
        points = dense_points_fused
        source = "fused.ply"
        if points is None:
            fused = metrics.ply_vertex_count(project_dir / "products" / "dense" / "fused.ply")
            fusion = settings.get("surface_fusion") if isinstance(settings.get("surface_fusion"), dict) else {}
            points = fused if fused is not None else (fusion.get("original_points") or None)
            if fused is None and points is not None:
                source = "surface_fusion.original_points"
        block: dict[str, Any] = {"points": points, "points_source": source if points is not None else None}
        cloud = cloud_for_density(project_dir, run_metrics)
        if cloud is not None:
            density = metrics.dense_density(cloud, is_metric(project_dir), float(config["density"]["cell_m"]))
            block.update(area_m2=density["area_m2"], density_reason=density["reason"], density_cloud=cloud.name)
            block["density_pts_m2"] = (points / density["area_m2"]) if (
                points and density["area_m2"]) else density["density_pts_m2"]
        return block

    dense_block = guarded("dense", dense) or {"points": dense_points_fused}
    path = scores_path(project_dir)
    class_set = settings.get("class_set")
    classification = {"mode": settings.get("classification_mode"), "engine": settings.get("classification_engine"),
                      "confidence_histogram": None}
    if path is not None:
        classification["confidence_histogram"] = guarded(
            "confidence_histogram",
            lambda: metrics.confidence_histogram(path, class_names=classification_class_names(class_set),
                                                 placeholder_values=placeholder_confidences(project_dir)))
    stages = guarded("refinement_stages", lambda: metrics.refinement_stages(
        run_metrics or {}, metrics.log_warnings(project_dir / "reports" / "run_summary.txt")))
    warnings.extend(metrics.log_warnings(project_dir / "reports" / "pipeline.log"))
    return {
        "report_version": REPORT_VERSION, "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software": guarded("software", metrics.software_info),
        "hardware": guarded("hardware", metrics.hardware_info),
        "import": guarded("import", lambda: import_block(project_dir, run_metrics)),
        "alignment": alignment, "overlap": overlap, "dense": dense_block,
        "refinement_stages": stages, "classification": classification,
        "warnings": list(dict.fromkeys(warnings)),
    }


def write_processing_metrics(project_dir: Path, run_id: str, run_metrics: dict[str, Any] | None = None,
                             dense_points_fused: int | None = None) -> Path:
    """Write ``reports/processing_metrics.json`` and, once, ``processing_metrics_<run_id>.json``."""
    project_dir = Path(project_dir)
    reports = project_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    data = build_processing_metrics(project_dir, run_id, run_metrics, dense_points_fused)
    text = json.dumps(clean_json(data), indent=2, ensure_ascii=False, default=str)
    target = reports / "processing_metrics.json"
    target.write_text(text, encoding="utf-8")
    per_run = reports / f"processing_metrics_{run_id}.json"
    if not per_run.exists():
        per_run.write_text(text, encoding="utf-8")
    return target


def archive_pipeline_log(project_dir: Path, not_before: float | None = None) -> Path | None:
    """Copy ``reports/pipeline.log`` to ``pipeline_<run_id>.log`` after the pipeline process has exited.

    The run id comes from ``reports/run_metrics.json``. Nothing is copied when that file is older than
    ``not_before`` (a failed run that never wrote metrics would otherwise archive its log under the previous
    run) and an existing copy is never overwritten. Never raises.
    """
    try:
        reports = Path(project_dir) / "reports"
        log, metrics_file = reports / "pipeline.log", reports / "run_metrics.json"
        if not log.is_file() or not metrics_file.is_file():
            return None
        if not_before is not None and metrics_file.stat().st_mtime < not_before:
            return None
        run_id = read_json_file(metrics_file).get("run_id")
        if not run_id:
            return None
        target = reports / f"pipeline_{run_id}.log"
        if target.exists():
            return target
        shutil.copy2(log, target)
        return target
    except Exception as error:  # noqa: BLE001 - archiving is best effort
        LOGGER.warning("pipeline log archive failed: %s", error)
        return None
