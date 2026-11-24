"""Video sources: discovery, frame extraction and camera video preparation.

One generic extraction routine serves two consumers: 360 degree videos (first frame of each interval
window, 2:1 check) and plain camera videos (sharpest frame of each interval window). Frames of several
videos are numbered continuously as ``frame_000000.jpg``, ``frame_000001.jpg`` ...
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2

CAMERA_VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
PANORAMA_VIDEO_EXTENSIONS = (".mp4",)
VIDEO_FRAMES_RELATIVE_DIRECTORY = "inputs/video_frames"
VIDEO_FRAMES_MANIFEST_NAME = "video_frames_manifest.json"
SHARPNESS_LONG_SIDE = 640
_STALE_FRAME_PATTERN = re.compile(r"^frame_\d{6}\.jpg$")


class CaptureValidationError(ValueError):
    """Raised when a capture source cannot be processed safely."""


ProgressCallback = Callable[[int, int, str], None]


def notify_progress(callback: ProgressCallback | None, done: int, total: int, detail: str) -> None:
    """Call a progress callback; a failing callback must never break the preparation."""
    if callback is None:
        return
    try:
        callback(int(done), int(total), detail)
    except Exception:
        pass


@dataclass(frozen=True)
class ExtractedFrame:
    """One frame written to disk and where it came from."""

    path: Path
    source: Path
    time_seconds: float


@dataclass(frozen=True)
class PreparedVideoFrames:
    """Result of extracting the frames of a camera video source."""

    frames_directory: Path
    frame_count: int
    video_count: int
    manifest_path: Path


# ------------------------------------------------------------------------------------------ discovery


def find_videos(source: str | Path, extensions: Iterable[str]) -> list[Path]:
    """Videos of ``source``: the file itself, or every matching file below a folder (sorted)."""
    allowed = {extension.lower() for extension in extensions}
    try:
        path = Path(source)
        if path.is_file():
            return [path] if path.suffix.lower() in allowed else []
        if path.is_dir():
            found = [
                item for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in allowed
            ]
            return sorted(found, key=lambda item: item.relative_to(path).as_posix().lower())
    except (OSError, ValueError):
        pass
    return []


def _video_header(video: Path) -> tuple[float, float] | None:
    """(frame count, frames per second) of a video header, None when unusable."""
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            return None
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frames_per_second = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    try:
        frame_count = float(frame_count)
        frames_per_second = float(frames_per_second)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(frame_count) or not math.isfinite(frames_per_second):
        return None
    if frame_count <= 0 or frames_per_second <= 0:
        return None
    return frame_count, frames_per_second


def window_frame_count(frame_count: float, frames_per_second: float, interval_seconds: float) -> int:
    """Number of interval windows (= kept frames) of one video."""
    interval_frames = max(1, round(frames_per_second * interval_seconds))
    return int((int(frame_count) - 1) // interval_frames) + 1


def expected_frame_count(videos: Iterable[Path], interval_seconds: float) -> int | None:
    """Frames ``extract_frames`` keeps over all videos, None when no video reports a frame count."""
    total = 0
    known = False
    for video in videos:
        header = _video_header(Path(video))
        if header is None:
            continue
        known = True
        total += window_frame_count(header[0], header[1], interval_seconds)
    return total if known else None


# ---------------------------------------------------------------------------------------- extraction

# Upper bound of the "discarded" entries kept in the manifest (long videos would otherwise bloat it).
MAX_DISCARDED_IN_MANIFEST = 20000


def frame_sharpness(frame: Any) -> float:
    """Variance of the Laplacian of the grayscale frame, downscaled to a long side of at most 640 px."""
    gray = frame if getattr(frame, "ndim", 3) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest > SHARPNESS_LONG_SIDE:
        scale = SHARPNESS_LONG_SIDE / longest
        gray = cv2.resize(
            gray, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
        )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    videos: Iterable[Path],
    output_directory: Path,
    interval_seconds: float,
    *,
    pick: str = "first",
    frame_check: Callable[[Any], None] | None = None,
    progress_callback: ProgressCallback | None = None,
    label: str = "Video",
    discarded: list[dict[str, Any]] | None = None,
) -> list[ExtractedFrame]:
    """Write one frame per interval window of every video, numbered continuously.

    ``pick="first"`` keeps the first frame of each window, ``pick="sharpest"`` the sharpest one (ties:
    earliest). ``frame_check(frame)`` may raise ``CaptureValidationError`` for an unsuitable frame.
    ``progress_callback(kept, expected, detail)`` counts over all videos; ``expected`` is 0 when no video
    reports its frame count. Callback errors are ignored. With ``pick="sharpest"`` and a ``discarded`` list,
    every considered frame that lost against a sharper one of its window is appended as
    ``{kept_frame, source_path, time_seconds, sharpness, kept_sharpness}``.
    """
    if pick not in ("first", "sharpest"):
        raise ValueError("pick must be 'first' or 'sharpest'")
    video_list = [Path(video) for video in videos]
    if not video_list:
        raise CaptureValidationError(f"No {label.lower()} was found")
    if interval_seconds <= 0:
        raise CaptureValidationError("Video frame interval must be greater than zero")
    output_directory.mkdir(parents=True, exist_ok=True)
    expected_frames = 0
    if progress_callback is not None:
        expected_frames = expected_frame_count(video_list, interval_seconds) or 0

    extracted: list[ExtractedFrame] = []

    def write(frame: Any, source: Path, frame_index: int, fps: float) -> None:
        if frame_check is not None:
            frame_check(frame)
        frame_path = output_directory / f"frame_{len(extracted):06d}.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            raise CaptureValidationError(f"Video frame could not be written: {frame_path.name}")
        extracted.append(ExtractedFrame(frame_path, source, frame_index / fps))
        if expected_frames:
            shown = min(len(extracted), expected_frames)
            notify_progress(progress_callback, shown, expected_frames, f"Frames extrahieren {shown}/{expected_frames}")
        else:
            notify_progress(progress_callback, len(extracted), 0, f"Frames extrahieren {len(extracted)}")

    for video in video_list:
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise CaptureValidationError(f"{label} cannot be opened: {video.name}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise CaptureValidationError(f"{label} does not provide a valid frame rate: {video.name}")
            window = max(1, round(fps * interval_seconds))
            frame_index = 0
            best: Any = None
            best_score = -1.0
            best_index = 0
            window_scores: list[tuple[int, float]] = []

            def note_discarded() -> None:
                if discarded is None:
                    return
                kept_name = f"frame_{len(extracted):06d}.jpg"
                for other_index, other_score in window_scores:
                    if other_index != best_index:
                        discarded.append({
                            "kept_frame": kept_name, "source_path": str(video),
                            "time_seconds": round(other_index / fps, 6), "sharpness": round(other_score, 3),
                            "kept_sharpness": round(best_score, 3),
                        })
                window_scores.clear()

            while True:
                success, frame = capture.read()
                if not success:
                    break
                if pick == "first":
                    if frame_index % window == 0:
                        write(frame, video, frame_index, fps)
                else:
                    if frame_index % window == 0 and best is not None:
                        note_discarded()
                        write(best, video, best_index, fps)
                        best = None
                    score = frame_sharpness(frame)
                    window_scores.append((frame_index, score))
                    if best is None or score > best_score:
                        best, best_score, best_index = frame.copy(), score, frame_index
                frame_index += 1
            if best is not None:
                note_discarded()
                write(best, video, best_index, fps)
        finally:
            capture.release()
    if not extracted:
        raise CaptureValidationError(f"No frames could be extracted from the {label.lower()}")
    return extracted


# ------------------------------------------------------------------------------- camera preparation


def prepare_camera_video_frames(
    video_source: str,
    frames_directory: Path,
    interval_seconds: float,
    progress_callback: ProgressCallback | None = None,
) -> PreparedVideoFrames:
    """Extract the sharpest frame per interval of a camera video (file or folder) into ``frames_directory``.

    Stale ``frame_NNNNNN.jpg`` files directly in the folder are removed first, because COLMAP reads the
    whole folder and a re-run with another interval must not leave old frames behind. A manifest
    ``video_frames_manifest.json`` is written next to the folder.
    """
    if interval_seconds <= 0:
        raise CaptureValidationError("Video frame interval must be greater than zero")
    videos = find_videos(video_source, CAMERA_VIDEO_EXTENSIONS) if video_source else []
    if not videos:
        raise CaptureValidationError(
            "No video was found: select a video file (" + ", ".join(CAMERA_VIDEO_EXTENSIONS)
            + ") or a folder that contains videos"
        )
    frames_directory = Path(frames_directory)
    frames_directory.mkdir(parents=True, exist_ok=True)
    for stale in frames_directory.iterdir():
        if stale.is_file() and _STALE_FRAME_PATTERN.match(stale.name):
            stale.unlink()
    discarded: list[dict[str, Any]] = []
    frames = extract_frames(
        videos, frames_directory, interval_seconds, pick="sharpest", progress_callback=progress_callback,
        discarded=discarded,
    )
    manifest_path = frames_directory.parent / VIDEO_FRAMES_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "frame_interval_seconds": interval_seconds,
                "pick": "sharpest",
                "frames": [
                    {
                        "frame": frame.path.name,
                        "source_path": str(frame.source),
                        "time_seconds": round(frame.time_seconds, 6),
                    }
                    for frame in frames
                ],
                "discarded": discarded[:MAX_DISCARDED_IN_MANIFEST],
                "discarded_count": len(discarded),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PreparedVideoFrames(frames_directory, len(frames), len(videos), manifest_path)


# --------------------------------------------------------------------------------- project helpers


def effective_image_directory(project_directory: Path, project: Mapping[str, Any]) -> Path | None:
    """Folder COLMAP reads photos from: the extracted video frames of a camera video project, else the configured folder."""
    acquisition = project.get("acquisition") or {}
    sources = acquisition.get("sources") or {}
    capture_type = acquisition.get("capture_type") or acquisition.get("type")
    if capture_type == "camera" and sources.get("video_path"):
        return Path(project_directory) / VIDEO_FRAMES_RELATIVE_DIRECTORY
    configured = sources.get("image_directory") or (project.get("paths") or {}).get("image_directory")
    return Path(configured) if configured else None
