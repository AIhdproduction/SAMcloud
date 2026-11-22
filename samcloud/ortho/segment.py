"""Explicit semantic segmentation of cached orthophotos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Literal

import cv2
import numpy as np

from samcloud.classification.custom import class_name, load_project_classes
from samcloud.classification.profiles import ModelProfile, profile_for_engine
from samcloud.core.configuration import load_yaml
from samcloud.core.paths import CONFIG_DIRECTORY
from samcloud.dxf.cancel import CancelToken, ProgressCallback, check_cancel, report_progress
from samcloud.dxf.levels import Level
from samcloud.ortho.build import OrthoError, load_ortho_result, ortho_output_dir

# The fine list up to classes revision 1. Outdoor never reads fine prompts, so its fingerprint keeps this list.
_FINE_PROMPTS_V1 = (
    "toilet", "washbasin", "bathtub", "shower tray", "kitchen sink", "stove", "bed", "sofa",
    "armchair", "table", "chair", "wardrobe", "shelf", "desk", "refrigerator", "washing machine",
    "radiator", "cushion", "pillow", "blanket", "clothes", "nightstand", "coffee table",
    "sideboard", "tv stand", "kitchen counter", "kitchen cabinet", "cat tree", "plant",
    "floor lamp", "television", "mirror",
)
# Screens only: the indoor top view is cut below the ceiling, so ceiling fixtures are never visible in it.
_SCREEN_PROMPTS = ("flat screen", "computer monitor", "laptop")

SegmentSource = Literal["pointcloud", "sam3", "flurseg", "combined"]
ImageModel = Literal["sam3", "flurseg"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SegmentConfig:
    """Validated semantic-segmentation settings for a cached orthophoto."""

    chunk_px: int = 4096
    chunk_overlap_px: int = 256
    min_image_confidence: float = 0.5
    pointcloud_confidence: float = 0.6
    high_threshold: float = 0.5
    low_threshold: float = 0.2
    outdoor_high_codes: tuple[int, ...] = (5, 6, 72, 84, 91)
    outdoor_low_codes: tuple[int, ...] = (2, 3, 9, 11, 71, 73, 82)
    indoor_high_codes: tuple[int, ...] = (100, 108, 117)
    indoor_low_codes: tuple[int, ...] = (102,)
    fine_prompts_indoor: tuple[str, ...] = (*_FINE_PROMPTS_V1, *_SCREEN_PROMPTS)
    fine_allowed_codes_indoor: tuple[int, ...] = (114, 117, 120, 121, 180)

    def fine_allowed_codes(self, mode: Literal["indoor", "outdoor"] = "indoor") -> tuple[int, ...]:
        """Return allowed Cyclone codes for fine labels."""
        if mode == "indoor":
            return self.fine_allowed_codes_indoor
        return ()

    def code_groups(self, mode: Literal["indoor", "outdoor"]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return high and low Cyclone-code groups for a project mode."""
        if mode == "indoor":
            return self.indoor_high_codes, self.indoor_low_codes
        return self.outdoor_high_codes, self.outdoor_low_codes


@dataclass(frozen=True, slots=True)
class SegmentResult:
    """Paths and metadata of one cached segmentation, without any compute side effect."""

    output_dir: Path
    source: SegmentSource
    classes_path: Path
    confidence_path: Path
    metadata_path: Path
    preview_path: Path
    fingerprint: str
    ortho_fingerprint: str
    is_stale: bool
    fine_classes_path: Path | None = None
    fine_confidence_path: Path | None = None
    fine_labels_path: Path | None = None
    metadata: dict[str, Any] | None = None


def load_segment_config(path: Path | None = None) -> SegmentConfig:
    """Load and validate segmentation configuration without constructing a model."""
    data = load_yaml(path or CONFIG_DIRECTORY / "ortho_segment.yaml")
    groups = dict(data.get("code_groups", {}) or {})
    outdoor = dict(groups.get("outdoor", {}) or {})
    indoor = dict(groups.get("indoor", {}) or {})
    fine = dict(data.get("fine_prompts", {}) or {})
    fine_allowed = dict(data.get("fine_allowed_codes", {}) or {})
    result = SegmentConfig(
        chunk_px=int(data.get("chunk_px", 4096)),
        chunk_overlap_px=int(data.get("chunk_overlap_px", 256)),
        min_image_confidence=float(data.get("min_image_confidence", 0.5)),
        pointcloud_confidence=float(data.get("pointcloud_confidence", 0.6)),
        high_threshold=float(data.get("high_threshold", 0.5)),
        low_threshold=float(data.get("low_threshold", 0.2)),
        outdoor_high_codes=_codes(outdoor.get("high_codes", (5, 6, 72, 84, 91))),
        outdoor_low_codes=_codes(outdoor.get("low_codes", (2, 3, 9, 11, 71, 73, 82))),
        indoor_high_codes=_codes(indoor.get("high_codes", (100, 108, 117))),
        indoor_low_codes=_codes(indoor.get("low_codes", (102,))),
        fine_prompts_indoor=tuple(str(value) for value in fine.get("indoor", SegmentConfig().fine_prompts_indoor)),
        fine_allowed_codes_indoor=_codes(fine_allowed.get("indoor", (114, 117, 120, 121, 180))),
    )
    if result.chunk_px < 1 or not 0 <= result.chunk_overlap_px < result.chunk_px:
        raise ValueError("chunk_overlap_px must be non-negative and smaller than chunk_px")
    if not 0.0 <= result.low_threshold <= result.high_threshold:
        raise ValueError("low_threshold and high_threshold must satisfy 0 <= low <= high")
    if any(not 0.0 <= value <= 1.0 for value in (result.min_image_confidence, result.pointcloud_confidence)):
        raise ValueError("confidence thresholds must be in [0, 1]")
    if not result.fine_prompts_indoor or len(set(result.fine_prompts_indoor)) != len(result.fine_prompts_indoor):
        raise ValueError("fine_prompts.indoor must be a non-empty unique list")
    if not result.fine_allowed_codes_indoor or len(set(result.fine_allowed_codes_indoor)) != len(result.fine_allowed_codes_indoor):
        raise ValueError("fine_allowed_codes.indoor must be a non-empty unique list of Cyclone codes")
    return result


def segment_ortho(
    project_dir: Path,
    *,
    level: Level | None,
    source: SegmentSource = "combined",
    image_model: ImageModel = "sam3",
    config: SegmentConfig | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
    device: str | None = None,
    labeler_factory: Callable[..., Any] | None = None,
) -> SegmentResult:
    """Segment an existing orthophoto and atomically cache its Cyclone-code rasters."""
    if source not in ("pointcloud", "sam3", "flurseg", "combined"):
        raise ValueError(f"Unsupported segmentation source: {source}")
    if image_model not in ("sam3", "flurseg"):
        raise ValueError(f"Unsupported image model: {image_model}")
    project = Path(project_dir)
    output_dir = ortho_output_dir(project, level)
    ortho = load_ortho_result(output_dir)
    if ortho is None:
        raise OrthoError("Build orthophoto first")
    settings = config or load_segment_config()
    mode = _project_mode(project, level)
    flurseg_runs = source == "flurseg" or (source == "combined" and image_model == "flurseg")
    if flurseg_runs and mode == "indoor":
        raise ValueError("FlurSeg segmentation is only available for outdoor orthophotos")
    active_image_model = image_model if source == "combined" else source
    if active_image_model not in ("sam3", "flurseg"):
        active_image_model = "sam3" if mode == "indoor" else image_model
    started = time.perf_counter()
    rgba = cv2.imread(str(ortho.ortho_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise OrthoError(f"Could not read RGBA orthophoto: {ortho.ortho_path}")
    rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
    point_classes = np.load(ortho.classes_path, allow_pickle=False).astype(np.uint8, copy=False)
    dsm = np.load(ortho.dsm_path, allow_pickle=False).astype(np.float32, copy=False)
    dtm = np.load(ortho.dtm_path, allow_pickle=False).astype(np.float32, copy=False)
    _validate_rasters(rgba, point_classes, dsm, dtm)
    valid = rgba[..., 3] > 0
    classes_config, class_names, prompts, profile = _model_context(project, mode, active_image_model)
    model_codes = _model_codes(profile, len(class_names))
    image_labels = np.zeros_like(point_classes)
    image_confidence = np.zeros(point_classes.shape, dtype=np.float16)
    uses_image = source != "pointcloud"
    if uses_image:
        report_progress(progress, 0.01, "Preparing orthophoto semantic segmentation")
        labeler = _create_labeler(labeler_factory, active_image_model, class_names, prompts, device)
        labels, image_confidence = _segment_image_chunks(
            rgba, labeler, settings, progress=progress, cancel=cancel, stage="Segmenting orthophoto",
        )
        image_labels = _map_labels_to_codes(labels, model_codes)
    check_cancel(cancel)
    if source == "pointcloud":
        classes = point_classes.copy()
        confidence = np.where(classes > 0, 1.0, 0.0).astype(np.float16)
    elif source in ("sam3", "flurseg"):
        classes = image_labels
        confidence = image_confidence
    else:
        classes, confidence = _combine(point_classes, image_labels, image_confidence, dsm, dtm, valid, settings, mode)
    classes[~valid] = 0
    confidence[~valid] = 0.0
    fine_paths: tuple[Path | None, Path | None, Path | None] = (None, None, None)
    fine_counts: dict[str, int] = {}
    point_fine_available = mode == "indoor" and has_point_fine_labels(output_dir)
    if point_fine_available and source in ("sam3", "combined"):
        LOGGER.info("fine labels from point cloud, ortho fine SAM skipped")
        report_progress(progress, 0.92, "Fine labels from point cloud, ortho fine SAM skipped")
    elif mode == "indoor" and source in ("sam3", "combined"):
        report_progress(progress, 0.92, "Segmenting indoor fine labels")
        fine_labels, fine_confidence = _fine_labels(rgba, classes, settings, labeler_factory, device, progress, cancel)
        fine_classes_path = _write_npy(output_dir / "fine_class.npy", fine_labels)
        fine_confidence_path = _write_npy(output_dir / "fine_conf.npy", fine_confidence)
        fine_labels_path = _write_json(output_dir / "fine_labels.json", list(settings.fine_prompts_indoor))
        fine_paths = (fine_classes_path, fine_confidence_path, fine_labels_path)
        fine_counts = {name: int(np.count_nonzero(fine_labels == index)) for index, name in enumerate(settings.fine_prompts_indoor, start=1)}
    fingerprint = segment_fingerprint(project, level=level, source=source, image_model=image_model, config=settings)
    counts = {str(code): int(count) for code, count in enumerate(np.bincount(classes.ravel(), minlength=256)) if count}
    metadata = {
        "source": source,
        "image_model": image_model,
        "mode": mode,
        "class_list": class_names,
        "prompts": prompts,
        "model_profile": profile.name,
        "code_groups": {"high_codes": list(settings.code_groups(mode)[0]), "low_codes": list(settings.code_groups(mode)[1])},
        "config": asdict(settings),
        "timings": {"total_seconds": time.perf_counter() - started},
        "class_pixel_counts": counts,
        "fine_pixel_counts": fine_counts,
        "fine_label_source": "pointcloud" if point_fine_available else ("ortho_sam" if fine_counts else "none"),
        "fingerprint": fingerprint,
        "ortho_fingerprint": ortho.fingerprint,
    }
    classes_path = _write_npy(output_dir / f"class_{source}.npy", classes)
    confidence_path = _write_npy(output_dir / f"conf_{source}.npy", confidence)
    metadata_path = _write_json(output_dir / f"segment_{source}.json", metadata)
    preview_path = _write_preview(output_dir / f"class_{source}.png", classes, classes_config, mode)
    report_progress(progress, 1.0, "Orthophoto segmentation complete")
    return SegmentResult(output_dir, source, classes_path, confidence_path, metadata_path, preview_path, fingerprint, ortho.fingerprint, False, *fine_paths, metadata)


def has_point_fine_labels(output_dir: Path) -> bool:
    """Return whether the ortho carries point-cloud fine labels (fine_pc.npy with a non-zero pixel)."""
    path = Path(output_dir) / "fine_pc.npy"
    labels = Path(output_dir) / "fine_pc_labels.json"
    if not path.is_file() or not labels.is_file():
        return False
    try:
        return bool(np.any(np.load(path, allow_pickle=False)))
    except (OSError, ValueError):
        return False


def load_segment_result(project_dir: Path, level: Level | None, source: SegmentSource) -> SegmentResult | None:
    """Load a cached segmentation only; this function never creates models or computes data."""
    output_dir = ortho_output_dir(Path(project_dir), level)
    classes_path = output_dir / f"class_{source}.npy"
    confidence_path = output_dir / f"conf_{source}.npy"
    metadata_path = output_dir / f"segment_{source}.json"
    preview_path = output_dir / f"class_{source}.png"
    if not all(path.is_file() for path in (classes_path, confidence_path, metadata_path, preview_path)):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        current = (output_dir / "fingerprint.txt").read_text(encoding="utf-8").strip()
        source_value = metadata["source"]
        if source_value != source:
            return None
        fine_classes = output_dir / "fine_class.npy"
        fine_confidence = output_dir / "fine_conf.npy"
        fine_labels = output_dir / "fine_labels.json"
        has_fine = all(path.is_file() for path in (fine_classes, fine_confidence, fine_labels))
        return SegmentResult(
            output_dir, source, classes_path, confidence_path, metadata_path, preview_path,
            str(metadata["fingerprint"]), str(metadata["ortho_fingerprint"]), str(metadata["ortho_fingerprint"]) != current,
            fine_classes if has_fine else None, fine_confidence if has_fine else None, fine_labels if has_fine else None, metadata,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def segment_fingerprint(
    project_dir: Path,
    *,
    level: Level | None,
    source: SegmentSource,
    image_model: ImageModel = "sam3",
    config: SegmentConfig | None = None,
) -> str:
    """Fingerprint cached-ortho identity, selected model/profile/classes/prompts and settings."""
    project = Path(project_dir)
    output_dir = ortho_output_dir(project, level)
    ortho_fingerprint_path = output_dir / "fingerprint.txt"
    if not ortho_fingerprint_path.is_file():
        raise OrthoError("Build orthophoto first")
    mode = _project_mode(project, level)
    active = image_model if source == "combined" else source
    if active not in ("sam3", "flurseg"):
        active = "sam3" if mode == "indoor" else image_model
    _classes_config, class_names, prompts, profile = _model_context(project, mode, active)
    payload = {
        "ortho_fingerprint": ortho_fingerprint_path.read_text(encoding="utf-8").strip(),
        "source": source,
        "image_model": image_model,
        "model_profile": profile.name,
        "class_list": class_names,
        "prompts": prompts,
        "config": _fingerprint_settings(config or load_segment_config(), mode),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _fingerprint_settings(config: SegmentConfig, mode: str) -> dict[str, Any]:
    """Settings that enter the fingerprint; the indoor-only fine prompts cannot invalidate an outdoor result."""
    values = asdict(config)
    if mode == "outdoor":
        values["fine_prompts_indoor"] = _FINE_PROMPTS_V1
    return values


def _codes(values: Any) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if any(value < 0 or value > 255 for value in result):
        raise ValueError("Cyclone codes must be in [0, 255]")
    return result


def _project_mode(project: Path, level: Level | None) -> Literal["indoor", "outdoor"]:
    project_file = project / "project.samcloud.yml"
    if project_file.is_file():
        settings = dict(load_yaml(project_file).get("settings", {}) or {})
        class_set = str(dict(settings.get("classification", {}) or {}).get("class_set", ""))
        if class_set == "indoor":
            return "indoor"
        if class_set == "outdoor":
            return "outdoor"
    return "indoor" if level is not None else "outdoor"


def _model_context(project: Path, mode: Literal["indoor", "outdoor"], engine: str) -> tuple[dict[str, Any], list[str], list[Any] | None, ModelProfile]:
    classes_config = load_project_classes(project)
    profile = profile_for_engine(engine, mode, classes_config)
    if engine == "flurseg":
        class_names = [entry.model_name for entry in profile.entries]
        return classes_config, class_names, None, profile
    class_names = list(classes_config[mode])
    prompts = classes_config.get("prompts", {}).get(mode) or None
    if prompts is not None and len(prompts) != len(class_names):
        raise ValueError(f"classes.json: prompts.{mode} must list one prompt per class")
    return classes_config, class_names, prompts, profile


def _model_codes(profile: ModelProfile, class_count: int) -> np.ndarray:
    exported = profile.codes_for_scalar_classes()
    if len(exported) != class_count + 1:
        raise ValueError(f"Model profile '{profile.name}' does not match its class list")
    codes = np.zeros(class_count + 1, dtype=np.uint8)
    codes[1:] = exported[1:]
    return codes


def _create_labeler(factory: Callable[..., Any] | None, engine: ImageModel, class_names: list[str], prompts: list[Any] | None, device: str | None) -> Any:
    if factory is not None:
        return factory(engine=engine, class_names=class_names, prompts=prompts, device=device)
    if engine == "sam3":
        from samcloud.engine.sam_labeler import SemanticLabeler

        return SemanticLabeler(class_names, prompts=prompts)
    from samcloud.engine.flurseg_labeler import FlurSegLabeler

    return FlurSegLabeler(class_names, device=device)


def _segment_image_chunks(rgba: np.ndarray, labeler: Any, settings: SegmentConfig, *, progress: ProgressCallback | None, cancel: CancelToken | None, stage: str) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgba.shape[:2]
    labels = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.float32)
    priority = np.zeros((height, width), dtype=np.float32)
    chunks = list(_chunk_boxes(width, height, settings.chunk_px, settings.chunk_overlap_px))
    for index, (x0, y0, x1, y1) in enumerate(chunks, start=1):
        check_cancel(cancel)
        chunk = rgba[y0:y1, x0:x1]
        if np.any(chunk[..., 3]):
            chunk_labels, chunk_confidence = labeler.label_image_scores(chunk[..., :3])
            chunk_labels = np.asarray(chunk_labels, dtype=np.uint8)
            chunk_confidence = np.asarray(chunk_confidence, dtype=np.float32)
            if chunk_labels.shape != chunk.shape[:2] or chunk_confidence.shape != chunk.shape[:2]:
                raise ValueError("label_image_scores must return rasters matching the input chunk")
            feather = _feather(y0, y1, x0, x1, height, width, settings.chunk_overlap_px)
            candidate = chunk_confidence * feather
            old_priority = priority[y0:y1, x0:x1]
            old_confidence = confidence[y0:y1, x0:x1]
            better = candidate > old_priority
            region = labels[y0:y1, x0:x1]
            region[better] = chunk_labels[better]
            old_priority[better] = candidate[better]
            old_confidence[better] = chunk_confidence[better]
            labels[y0:y1, x0:x1] = region
            priority[y0:y1, x0:x1] = old_priority
            confidence[y0:y1, x0:x1] = old_confidence
        report_progress(progress, index / len(chunks) * 0.9, stage)
    return labels, confidence.astype(np.float16)


def _chunk_boxes(width: int, height: int, chunk_px: int, overlap: int) -> list[tuple[int, int, int, int]]:
    stride = chunk_px - overlap
    starts_x = _starts(width, chunk_px, stride)
    starts_y = _starts(height, chunk_px, stride)
    return [(x0, y0, min(width, x0 + chunk_px), min(height, y0 + chunk_px)) for y0 in starts_y for x0 in starts_x]


def _starts(length: int, chunk_px: int, stride: int) -> list[int]:
    if length <= chunk_px:
        return [0]
    values = list(range(0, length - chunk_px + 1, stride))
    final = length - chunk_px
    if values[-1] != final:
        values.append(final)
    return values


def _feather(y0: int, y1: int, x0: int, x1: int, height: int, width: int, overlap: int) -> np.ndarray:
    weights_y = _axis_feather(y0, y1, height, overlap)
    weights_x = _axis_feather(x0, x1, width, overlap)
    return np.outer(weights_y, weights_x).astype(np.float32)


def _axis_feather(start: int, end: int, length: int, overlap: int) -> np.ndarray:
    weights = np.ones(end - start, dtype=np.float32)
    if overlap <= 0:
        return weights
    ramp = np.linspace(1.0 / (overlap + 1), 1.0, overlap, dtype=np.float32)
    if start > 0:
        weights[:min(overlap, len(weights))] = ramp[:min(overlap, len(weights))]
    if end < length:
        weights[-min(overlap, len(weights)):] = ramp[:min(overlap, len(weights))][::-1]
    return weights


def _map_labels_to_codes(labels: np.ndarray, model_codes: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.uint8)
    if values.max(initial=0) >= len(model_codes):
        raise ValueError("label_image_scores returned a class ID outside the configured model class list")
    return model_codes[values]


def _combine(point_classes: np.ndarray, image_classes: np.ndarray, image_confidence: np.ndarray, dsm: np.ndarray, dtm: np.ndarray, valid: np.ndarray, settings: SegmentConfig, mode: Literal["indoor", "outdoor"]) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(point_classes, dtype=np.uint8)
    image = np.asarray(image_classes, dtype=np.uint8)
    image_conf = np.asarray(image_confidence, dtype=np.float32)
    point_conf = np.where(point > 0, settings.pointcloud_confidence, 0.0).astype(np.float32)
    output = point.copy()
    confidence = point_conf.copy()
    image_only = (point == 0) & (image > 0)
    output[image_only] = image[image_only]
    confidence[image_only] = image_conf[image_only]
    disagree = (point > 0) & (image > 0) & (point != image)
    eligible = disagree & (image_conf >= settings.min_image_confidence)
    height = dsm - dtm
    high_codes, low_codes = settings.code_groups(mode)
    for group, chooser in ((high_codes, height >= settings.high_threshold), (low_codes, height <= settings.low_threshold)):
        point_group = np.isin(point, group)
        image_group = np.isin(image, group)
        prefer_image = eligible & chooser & image_group & ~point_group
        prefer_point = eligible & chooser & point_group & ~image_group
        higher_image = eligible & chooser & ((image_group == point_group) | ~(prefer_image | prefer_point)) & (image_conf > point_conf)
        select = prefer_image | higher_image
        output[select] = image[select]
        confidence[select] = image_conf[select]
    middle = eligible & (height > settings.low_threshold) & (height < settings.high_threshold)
    select_middle = middle & (image_conf > point_conf)
    output[select_middle] = image[select_middle]
    confidence[select_middle] = image_conf[select_middle]
    output[~valid] = 0
    confidence[~valid] = 0.0
    return output, confidence.astype(np.float16)


def _fine_labels(rgba: np.ndarray, classes: np.ndarray, settings: SegmentConfig, labeler_factory: Callable[..., Any] | None, device: str | None, progress: ProgressCallback | None, cancel: CancelToken | None) -> tuple[np.ndarray, np.ndarray]:
    labeler = _create_labeler(labeler_factory, "sam3", list(settings.fine_prompts_indoor), list(settings.fine_prompts_indoor), device)
    labels, confidence = _segment_image_chunks(rgba, labeler, settings, progress=progress, cancel=cancel, stage="Segmenting indoor fine labels")
    allowed = np.isin(classes, settings.fine_allowed_codes_indoor)
    labels[~allowed] = 0
    confidence[~allowed] = 0.0
    return labels, confidence


def _validate_rasters(rgba: np.ndarray, point_classes: np.ndarray, dsm: np.ndarray, dtm: np.ndarray) -> None:
    shape = rgba.shape[:2]
    if point_classes.shape != shape or dsm.shape != shape or dtm.shape != shape:
        raise OrthoError("Orthophoto and DSM rasters must have the same shape")


def _write_npy(path: Path, values: np.ndarray) -> Path:
    return _atomic_write(path, lambda temporary: _save_npy(temporary, values))


def _save_npy(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)


def _write_json(path: Path, values: Any) -> Path:
    return _atomic_write(path, lambda temporary: temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"))


def _write_preview(path: Path, classes: np.ndarray, classes_config: dict[str, Any], mode: Literal["indoor", "outdoor"]) -> Path:
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:] = (96, 96, 96)
    palette[0] = (0, 0, 0)
    codes = classes_config.get("las_codes", {}).get(mode, [])
    colours = classes_config.get("colors", {}).get(mode, [])
    for code, colour in zip(codes, colours):
        if isinstance(colour, str) and len(colour) == 7 and colour.startswith("#"):
            palette[int(code)] = tuple(int(colour[offset:offset + 2], 16) for offset in (1, 3, 5))
    rgb = palette[np.asarray(classes, dtype=np.uint8)]
    return _atomic_write(path, lambda temporary: _write_png(temporary, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Could not write image: {path}")


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=f".tmp{path.suffix}", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    writer(temporary)
    os.replace(temporary, path)
    return path
