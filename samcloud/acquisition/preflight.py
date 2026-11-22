"""Image count and storage estimate shown while a project is created.

Pure functions without UI. Probing (reading a video header, walking a folder) is separated from the
estimate so that a change of interval or storage switches only recomputes numbers. Nothing here ever
raises for bad user input: unknown, missing or unreadable sources become ``unknown`` results with a
reason key, never a made-up number. Heavy modules (cv2, PIL, the COLMAP helpers) are imported lazily.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

GIB = 1024 ** 3

# --- Storage model, measured on a real run of this application (one 360 and one drone project) --------------
# Dense stereo dominates the peak: photometric + geometric maps = depth (4 B) + normal (12 B) per map and
# pixel, twice (photometric and geometric) = 32 B per dense pixel and image.
STEREO_BYTES_PER_PIXEL = 32.0
# 360: only part of the cubemap faces get maps (1454 of 2136 measured); photos: every image gets maps.
PANORAMA_FACE_MAP_FRACTION = 0.68
PHOTO_MAP_FRACTION = 1.0
# Everything except dense stereo (frames, cubemap, undistorted images, depth prior, sparse text, products),
# calibrated on 356 panoramas / 2136 faces at 1232 px (total 81.1e9 B, stereo 70.6e9 B) ...
REST_BYTES_PER_FACE_PIXEL = (81.1e9 - 70.6e9) / (2136 * 1232 ** 2)
# ... and on 291 drone photos at a dense long side of 2900 px, 4:3 (total 75.7e9 B, stereo 56.7e9 B).
# Calibrated against the model's own stereo term so that the total reproduces the measurement.
REST_BYTES_PER_PHOTO_PIXEL = (75.7e9 - STEREO_BYTES_PER_PIXEL * 291 * 2900 * 2175) / (291 * 2900 * 2175)
# Finished project with the processing cache off (360: products 2.87 + kept inputs 1.27 GiB for 2136 faces;
# drone: products 9.89 GiB for 291 photos, inputs stay external).
AFTER_PRODUCTS_BYTES_PER_FACE_PIXEL = 2.87 * GIB / (2136 * 1232 ** 2)
AFTER_INPUTS_BYTES_PER_FACE_PIXEL = 1.27 * GIB / (2136 * 1232 ** 2)
AFTER_BYTES_PER_PHOTO_PIXEL = 9.89 * GIB / (291 * 2900 * 2175)

# Status thresholds: ok = free >= upper bound + 10 %, tight = free >= typical, otherwise short.
UPPER_BOUND_MARGIN = 0.10
INTERVAL_ADVICE_STEP = 0.5
INTERVAL_ADVICE_LIMIT = 600.0

VIDEO_EXTENSIONS = (".mp4",)
PANORAMA_PHOTO_EXTENSIONS = (".jpg", ".jpeg")
# The extensions the pipeline really uses for drone/camera folders (``write_image_list``).
PHOTO_FOLDER_EXTENSIONS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
# Image-like files nobody will process; only counted so that the user notices them.
IGNORED_IMAGE_EXTENSIONS = (
    ".png", ".tif", ".tiff", ".heic", ".heif", ".dng", ".raw", ".cr2", ".cr3", ".nef", ".arw", ".bmp", ".webp",
)
STORAGE_INPUT_KEYS = ("keep_control_point_inputs", "keep_classification_inputs", "keep_refinement_inputs")
PANORAMA_FACES = 6


@dataclass(frozen=True)
class VideoInfo:
    path: str
    status: str  # ok | not_found | unreadable
    is_mp4: bool = True
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    size_bytes: int = 0

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def is_2to1(self) -> bool:
        return self.height > 0 and abs(self.width / self.height - 2.0) <= 0.03


@dataclass(frozen=True)
class FolderInfo:
    path: str
    status: str  # ok | not_found | unreadable | no_images
    count: int = 0
    size_bytes: int = 0
    ignored: int = 0
    first_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class SourceProbe:
    """What was found on disk; independent of interval and storage switches."""

    capture_type: str
    video: VideoInfo | None = None
    folder: FolderInfo | None = None  # panorama photos (360) or image folder (drone/camera)


@dataclass(frozen=True)
class Note:
    key: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreflightResult:
    state: str  # empty | unknown | ready
    notes: tuple[Note, ...] = ()
    capture_type: str = ""
    panoramas: int = 0
    faces: int = 0
    images: int = 0
    video_panoramas: int = 0
    photo_count: int = 0
    interval: float = 0.0
    video: VideoInfo | None = None
    folder: FolderInfo | None = None
    source_bytes: int = 0
    dense_size: int = 0
    peak_typical: int = 0
    peak_upper: int = 0
    after: int = 0
    free: int | None = None
    free_drive: str = ""
    status: str = "unknown"  # ok | tight | short | unknown
    advice_interval: float | None = None


# --------------------------------------------------------------------------------------------- probing


def kept_frame_count(frame_count: int, fps: float, interval: float) -> int:
    """Frames kept at ``interval``; the same rule as ``expected_video_frame_count`` (parity is tested)."""
    if frame_count <= 0 or fps <= 0 or interval <= 0:
        return 0
    interval_frames = max(1, round(fps * interval))
    return int((int(frame_count) - 1) // interval_frames) + 1


def read_video_info(path: str) -> VideoInfo:
    """Read the header of a video; never raises."""
    try:
        file = Path(path)
        if not file.is_file():
            return VideoInfo(path, "not_found")
        size = file.stat().st_size
        is_mp4 = file.suffix.lower() in VIDEO_EXTENSIONS
        import cv2

        capture = cv2.VideoCapture(str(file))
        try:
            if not capture.isOpened():
                return VideoInfo(path, "unreadable", is_mp4, size_bytes=size)
            width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if not all(math.isfinite(v) for v in (width, height, fps, frames)) or min(width, height, fps, frames) <= 0:
            return VideoInfo(path, "unreadable", is_mp4, size_bytes=size)
        return VideoInfo(path, "ok", is_mp4, int(width), int(height), fps, int(frames), size)
    except Exception:  # noqa: BLE001 - this is a pre-check, never an error source
        return VideoInfo(path, "unreadable")


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.size[0]), int(image.size[1])
    except Exception:  # noqa: BLE001
        return None


def scan_folder(path: str, extensions: tuple[str, ...], *, count_ignored: bool = True) -> FolderInfo:
    """Count the images the pipeline would use (recursive, case-insensitive); never raises."""
    try:
        root = Path(path)
        if not root.is_dir():
            return FolderInfo(path, "not_found")
        used = {e.lower() for e in extensions}
        ignored_extensions = {e for e in IGNORED_IMAGE_EXTENSIONS if e not in used}
        count = size = ignored = 0
        first: Path | None = None
        first_name: tuple[str, ...] | None = None
        for current, _dirs, files in os.walk(root):
            for name in files:
                suffix = os.path.splitext(name)[1].lower()
                if suffix in used:
                    full = os.path.join(current, name)
                    try:
                        size += os.path.getsize(full)
                    except OSError:
                        continue
                    count += 1
                    key = Path(full).relative_to(root).parts
                    if first_name is None or key < first_name:  # the pipeline reads the first of the sorted list
                        first_name, first = key, Path(full)
                elif count_ignored and suffix in ignored_extensions:
                    ignored += 1
        if count == 0:
            return FolderInfo(path, "no_images", ignored=ignored)
        return FolderInfo(path, "ok", count, size, ignored, _image_size(first) if first else None)
    except Exception:  # noqa: BLE001
        return FolderInfo(path, "unreadable")


def probe_sources(
    capture_type: str, image_directory: str = "", panorama_video_path: str = "", panorama_image_directory: str = "",
) -> SourceProbe:
    """Look at the sources the form currently names (blocking disk access: call from a worker thread)."""
    if capture_type == "panorama_360":
        video = read_video_info(panorama_video_path.strip()) if (panorama_video_path or "").strip() else None
        folder = (
            scan_folder(panorama_image_directory.strip(), PANORAMA_PHOTO_EXTENSIONS)
            if (panorama_image_directory or "").strip() else None
        )
        return SourceProbe(capture_type, video, folder)
    folder = scan_folder(image_directory.strip(), PHOTO_FOLDER_EXTENSIONS) if (image_directory or "").strip() else None
    return SourceProbe(capture_type, None, folder)


def free_space(projects_directory: Path | str) -> tuple[int | None, str]:
    """Free bytes and drive label of the drive the project will be created on."""
    try:
        path = Path(projects_directory).absolute()
        existing = path
        while not existing.exists() and existing.parent != existing:
            existing = existing.parent
        return shutil.disk_usage(existing).free, (path.drive or str(existing.anchor or ""))
    except Exception:  # noqa: BLE001
        return None, ""


# ---------------------------------------------------------------------------------------------- model


def _dense_size(capture_type: str, long_side: int, images: int, free_bytes: int | None, dense_config: Mapping) -> int:
    """Dense long side exactly as the pipeline resolves it (incl. its free-space rule)."""
    from samcloud.colmap.alignment import adaptive_dense_size, dense_size_settings

    try:
        resolved = dense_size_settings(dict(dense_config), capture_type)
    except ValueError:
        resolved = {"max_image_size": "auto", "auto": {}}
    configured = resolved["max_image_size"]
    if configured != "auto":
        return max(1, min(int(configured), long_side))
    options = resolved["auto"]
    size, _reason = adaptive_dense_size(
        int(long_side), max(int(images), 1), free_bytes if free_bytes is not None else 1 << 62,
        ratio=float(options.get("ratio", 0.55)),
        minimum=int(options.get("minimum", 2000)),
        maximum=int(options.get("maximum", 3200)),
    )
    return size


def _estimate_bytes(
    *, is_360: bool, images: int, dense_size: int, aspect: float, storage: Mapping[str, bool],
) -> tuple[int, int, int]:
    """(typical peak, upper-bound peak, after completion) in bytes."""
    pixels = dense_size * dense_size * aspect  # aspect = short / long side
    stereo_upper = images * pixels * STEREO_BYTES_PER_PIXEL
    fraction = PANORAMA_FACE_MAP_FRACTION if is_360 else PHOTO_MAP_FRACTION
    rest = images * pixels * (REST_BYTES_PER_FACE_PIXEL if is_360 else REST_BYTES_PER_PHOTO_PIXEL)
    typical = stereo_upper * fraction + rest
    upper = stereo_upper + rest
    if storage.get("keep_processing_cache"):
        after = typical
    elif is_360:
        kept = sum(1 for key in STORAGE_INPUT_KEYS if storage.get(key, True)) / len(STORAGE_INPUT_KEYS)
        after = images * pixels * (AFTER_PRODUCTS_BYTES_PER_FACE_PIXEL + AFTER_INPUTS_BYTES_PER_FACE_PIXEL * kept)
    else:
        after = images * pixels * AFTER_BYTES_PER_PHOTO_PIXEL
    return int(typical), int(upper), int(min(after, typical))


def storage_status(free: int | None, typical: int, upper: int) -> str:
    if free is None:
        return "unknown"
    if free >= upper * (1 + UPPER_BOUND_MARGIN):
        return "ok"
    if free >= typical:
        return "tight"
    return "short"


def _face_size(width: int, requested: int) -> int:
    from samcloud.acquisition.panorama import native_face_size

    native = native_face_size(width)
    return native if requested <= 0 else min(requested, native)


def estimate(
    probe: SourceProbe,
    *,
    interval: float,
    storage: Mapping[str, bool],
    free_bytes: int | None,
    free_drive: str = "",
    dense_config: Mapping | None = None,
    requested_face_size: int = 1232,
    _search_advice: bool = True,
) -> PreflightResult:
    """Pure estimate for the probed sources; ``unknown`` results carry a reason, never a number."""
    dense_config = dense_config if dense_config is not None else {}
    capture_type = probe.capture_type
    notes: list[Note] = []
    is_360 = capture_type == "panorama_360"
    video, folder = probe.video, probe.folder

    if video is None and folder is None:
        return PreflightResult("empty", (), capture_type)

    video_panoramas = 0
    if video is not None:
        if video.status == "not_found":
            notes.append(Note("video_not_found"))
        elif video.status != "ok":
            notes.append(Note("video_unreadable"))
        else:
            if not video.is_mp4:
                notes.append(Note("video_not_mp4"))
            if not video.is_2to1:
                notes.append(Note("video_not_2to1", {"width": video.width, "height": video.height}))
            video_panoramas = kept_frame_count(video.frame_count, video.fps, interval) if interval > 0 else 0
            if interval <= 0:
                notes.append(Note("interval_invalid"))
    photos = 0
    if folder is not None:
        if folder.status == "not_found":
            notes.append(Note("folder_not_found"))
        elif folder.status == "unreadable":
            notes.append(Note("folder_unreadable"))
        elif folder.status == "no_images":
            notes.append(Note("folder_no_images", {"ignored": folder.ignored}))
        else:
            photos = folder.count
            if is_360 and folder.first_size and abs(folder.first_size[0] / max(folder.first_size[1], 1) - 2.0) > 0.03:
                notes.append(Note("photos_not_2to1", {"width": folder.first_size[0], "height": folder.first_size[1]}))
            if folder.ignored:
                notes.append(Note("ignored_images", {"count": folder.ignored}))

    sources = video_panoramas + photos
    if sources <= 0:
        return PreflightResult("unknown", tuple(notes), capture_type, video=video, folder=folder, interval=interval)

    # Long side and aspect of the images COLMAP will see.
    if is_360:
        width = 0
        if folder is not None and folder.status == "ok" and folder.first_size:
            width = folder.first_size[0]
        elif video is not None and video.status == "ok":
            width = video.width
        if width <= 0:
            return PreflightResult("unknown", (*notes, Note("size_unknown")), capture_type, video=video, folder=folder)
        long_side, aspect = _face_size(width, int(requested_face_size or 0)), 1.0
        images = sources * PANORAMA_FACES
    else:
        if not folder or not folder.first_size:
            return PreflightResult("unknown", (*notes, Note("size_unknown")), capture_type, video=video, folder=folder)
        w, h = folder.first_size
        long_side, aspect = max(w, h), min(w, h) / max(w, h)
        images = photos
    dense = _dense_size(capture_type, long_side, images, free_bytes, dense_config)
    typical, upper, after = _estimate_bytes(
        is_360=is_360, images=images, dense_size=dense, aspect=aspect, storage=storage,
    )
    status = storage_status(free_bytes, typical, upper)
    source_bytes = (folder.size_bytes if folder and folder.status == "ok" else 0) + (
        video.size_bytes if video is not None and video.status == "ok" else 0
    )
    result = PreflightResult(
        "ready", tuple(notes), capture_type,
        panoramas=sources if is_360 else 0, faces=images if is_360 else 0, images=images,
        video_panoramas=video_panoramas, photo_count=photos, interval=interval, video=video, folder=folder,
        source_bytes=source_bytes, dense_size=dense, peak_typical=typical, peak_upper=upper, after=after,
        free=free_bytes, free_drive=free_drive, status=status,
    )
    if _search_advice and status == "short" and video is not None and video.status == "ok":
        return replace(result, advice_interval=_advice_interval(
            probe, interval=interval, storage=storage, free_bytes=free_bytes, free_drive=free_drive,
            dense_config=dense_config, requested_face_size=requested_face_size,
        ))
    return result


def _advice_interval(probe: SourceProbe, *, interval: float, storage: Mapping[str, bool], free_bytes: int | None,
                     free_drive: str, dense_config: Mapping, requested_face_size: int) -> float | None:
    """Smallest interval in 0.5 s steps above the current one whose typical peak fits the free space."""
    candidate = (math.floor(max(interval, 0) / INTERVAL_ADVICE_STEP + 1e-9) + 1) * INTERVAL_ADVICE_STEP
    while candidate <= INTERVAL_ADVICE_LIMIT:
        result = estimate(
            probe, interval=candidate, storage=storage, free_bytes=free_bytes, free_drive=free_drive,
            dense_config=dense_config, requested_face_size=requested_face_size, _search_advice=False,
        )
        if result.state == "ready" and result.status in ("ok", "tight"):
            return candidate
        if result.state != "ready" or result.video_panoramas == 0:
            return None
        candidate += INTERVAL_ADVICE_STEP
    return None


# ----------------------------------------------------------------------------------------- formatting


def format_count(value: int, language: str = "de") -> str:
    """1'000 for German (Swiss) and 1,000 for English."""
    text = f"{int(value):,}"
    return text.replace(",", "'" if language == "de" else ",")


def format_gb(value: float, language: str = "de") -> str:
    """GiB labelled "GB" like Windows Explorer: whole numbers from 10 GB, one decimal below."""
    gib = float(value) / GIB
    if gib >= 10:
        return f"{format_count(round(gib), language)} GB"
    return f"{gib:.1f} GB"


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_interval(value: float) -> str:
    return f"{value:g}"


def load_dense_config() -> dict:
    """``dense_reconstruction`` section of the central COLMAP configuration (empty when unreadable)."""
    try:
        from samcloud.core.configuration import load_global_config

        return dict(load_global_config("colmap.yml").get("dense_reconstruction") or {})
    except Exception:  # noqa: BLE001
        return {}
