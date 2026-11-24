"""Prepare equirectangular 360 degree media for the COLMAP workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from samcloud.acquisition.video import (
    CAMERA_VIDEO_EXTENSIONS,
    PANORAMA_VIDEO_EXTENSIONS,
    CaptureValidationError,
    ProgressCallback,
    expected_frame_count,
    extract_frames,
    find_videos,
    notify_progress as _notify,
)

CAPTURE_TYPES = ("drone_gps", "camera", "panorama_360")
PANORAMA_IMAGE_EXTENSIONS = (".jpg", ".jpeg")
CUBEMAP_FACE_NAMES = ("front", "right", "back", "left", "up", "down")
CUBEMAP_LAYOUT_VERSION = 2


def expected_video_frame_count(video_path: Path, interval_seconds: float) -> int | None:
    """Number of frames ``extract_video_frames`` will keep, or None when the video does not tell."""
    return expected_frame_count([Path(video_path)], interval_seconds)


@dataclass(frozen=True)
class PreparedPanoramaInputs:
    """Project-owned paths and source counts created for a 360 degree job."""

    cubemap_directory: Path
    manifest_path: Path
    panorama_count: int
    video_frame_count: int
    cubemap_image_count: int
    face_size: int


def validate_capture_configuration(
    capture_type: str,
    image_directory: str = "",
    panorama_image_directory: str = "",
    panorama_video_path: str = "",
    frame_interval_seconds: float = 1.0,
    *,
    check_paths: bool = False,
    video_path: str = "",
) -> None:
    """Validate the selected acquisition mode without changing project data.

    ``video_path`` is the camera video source (one video file or a folder of videos); a camera project
    uses either an image directory or a video, never both.
    """
    if capture_type not in CAPTURE_TYPES:
        raise CaptureValidationError("Select a supported capture type")
    if frame_interval_seconds <= 0:
        raise CaptureValidationError("Video frame interval must be greater than zero")

    if capture_type == "panorama_360":
        if not panorama_image_directory and not panorama_video_path:
            raise CaptureValidationError(
                "Select a 360 degree video, a folder of 360 degree videos, a panorama image directory, or a combination"
            )
        if panorama_video_path:
            video_location = Path(panorama_video_path)
            is_directory = video_location.is_dir()
            if not is_directory and video_location.suffix and (
                video_location.suffix.lower() not in PANORAMA_VIDEO_EXTENSIONS
            ):
                raise CaptureValidationError("360 degree video must be an exported MP4 file")
            if check_paths:
                if is_directory:
                    if not find_videos(video_location, PANORAMA_VIDEO_EXTENSIONS):
                        raise CaptureValidationError("Selected 360 degree video folder contains no MP4 videos")
                elif not video_location.is_file():
                    raise CaptureValidationError("Selected 360 degree video file was not found")
        if check_paths and panorama_image_directory and not Path(panorama_image_directory).is_dir():
            raise CaptureValidationError("Selected panorama image directory was not found")
        return

    if video_path and capture_type != "camera":
        raise CaptureValidationError("A video source is only supported for the standard camera capture type")
    if capture_type == "camera" and image_directory and video_path:
        raise CaptureValidationError("Select either an image directory or a video, not both")
    if not image_directory and not video_path:
        raise CaptureValidationError(
            "Select an image directory before creating the project"
            if capture_type != "camera"
            else "Select an image directory or a video before creating the project"
        )
    if video_path:
        video_location = Path(video_path)
        if video_location.suffix and not video_location.is_dir() and (
            video_location.suffix.lower() not in CAMERA_VIDEO_EXTENSIONS
        ):
            raise CaptureValidationError(
                "Video must be one of: " + ", ".join(CAMERA_VIDEO_EXTENSIONS)
            )
        if check_paths and not find_videos(video_location, CAMERA_VIDEO_EXTENSIONS):
            raise CaptureValidationError("Selected video was not found or the folder contains no videos")
        return
    if check_paths and not Path(image_directory).is_dir():
        raise CaptureValidationError("Selected image directory was not found")


def find_panorama_images(directory: Path) -> list[Path]:
    """Return supported panorama photos in a stable order."""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in PANORAMA_IMAGE_EXTENSIONS
    )


def validate_panorama_images(image_paths: Iterable[Path]) -> list[Path]:
    """Ensure that every selected panorama is a readable 2:1 image."""
    valid_paths: list[Path] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise CaptureValidationError(f"Panorama image cannot be read: {image_path.name}")
        height, width = image.shape[:2]
        if height == 0 or abs((width / height) - 2.0) > 0.03:
            raise CaptureValidationError(
                f"Panorama image must use a 2:1 equirectangular format: {image_path.name}"
            )
        valid_paths.append(image_path)
    if not valid_paths:
        raise CaptureValidationError("No JPG panorama images were found in the selected directory")
    return valid_paths


def _require_2to1(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    if height == 0 or abs((width / height) - 2.0) > 0.03:
        raise CaptureValidationError("360 degree video frames must use a 2:1 equirectangular format")


def extract_video_frames(
    video_path: Path, output_directory: Path, interval_seconds: float,
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    """Extract the first equirectangular frame of each requested fixed time interval.

    ``progress_callback(kept, expected, detail)`` reports every kept frame;
    ``expected`` is 0 when the video does not report its frame count.
    """
    frames = extract_frames(
        [Path(video_path)], output_directory, interval_seconds,
        pick="first", frame_check=_require_2to1, progress_callback=progress_callback,
        label="360 degree video",
    )
    return [frame.path for frame in frames]


def cubemap_face_directions(
    horizontal: np.ndarray, vertical: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return unnormalised equirectangular directions for cubemap pixels."""
    return {
        "front": (horizontal, -vertical, np.ones_like(horizontal)),
        "right": (np.ones_like(horizontal), -vertical, -horizontal),
        "back": (-horizontal, -vertical, -np.ones_like(horizontal)),
        "left": (-np.ones_like(horizontal), -vertical, horizontal),
        "up": (horizontal, np.ones_like(horizontal), vertical),
        "down": (horizontal, -np.ones_like(horizontal), -vertical),
    }


def equirectangular_to_cubemap(image: np.ndarray, face_size: int) -> dict[str, np.ndarray]:
    """Render six 90 degree perspective views from a 2:1 panorama."""
    if face_size <= 0:
        raise ValueError("Cubemap face size must be greater than zero")
    height, width = image.shape[:2]
    if height == 0 or abs((width / height) - 2.0) > 0.03:
        raise CaptureValidationError("Cubemap input must be a readable 2:1 equirectangular image")

    coordinates = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    directions = cubemap_face_directions(horizontal, vertical)
    faces: dict[str, np.ndarray] = {}
    for face_name, (x_axis, y_axis, z_axis) in directions.items():
        norm = np.sqrt(x_axis**2 + y_axis**2 + z_axis**2)
        x_axis, y_axis, z_axis = x_axis / norm, y_axis / norm, z_axis / norm
        longitude = np.arctan2(x_axis, z_axis)
        latitude = np.arcsin(y_axis)
        map_x = ((longitude / (2 * np.pi)) + 0.5) * (width - 1)
        map_y = (0.5 - (latitude / np.pi)) * (height - 1)
        faces[face_name] = cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
    return faces


def native_face_size(panorama_width: int) -> int:
    """Cubemap face side matching the equirectangular resolution (width / 4, multiple of 8)."""
    return max(8, int(round(panorama_width / 4 / 8)) * 8)


def prepare_panorama_inputs(
    panorama_image_directory: str,
    panorama_video_path: str,
    project_inputs_directory: Path,
    frame_interval_seconds: float = 1.0,
    face_size: int = 0,
    progress_callback: ProgressCallback | None = None,
) -> PreparedPanoramaInputs:
    """Create video frames, cubemap images, and an origin manifest for COLMAP.

    ``face_size`` 0 means native: a quarter of the first panorama's width
    (5760 px -> 1440 px), the resolution a 90 degree face really contains.
    Larger values are capped at native. The default (1232 px) sits between
    1024 and native: about 25 % fewer pixels than native for dense matching
    and SAM, with nearly the same detail.

    ``progress_callback(done, total, detail)`` reports one combined stage:
    ``total`` is twice the number of panorama sources (frame extraction plus
    cubemap faces; photos count as already extracted). ``total`` is 0 while a
    video does not report its frame count. Callback errors are ignored.
    """
    validate_capture_configuration(
        "panorama_360",
        panorama_image_directory=panorama_image_directory,
        panorama_video_path=panorama_video_path,
        frame_interval_seconds=frame_interval_seconds,
        check_paths=True,
    )
    project_inputs_directory.mkdir(parents=True, exist_ok=True)
    frame_directory = project_inputs_directory / "panorama_frames"
    cubemap_directory = project_inputs_directory / "cubemap"
    manifest_path = project_inputs_directory / "panorama_manifest.json"

    image_paths: list[Path] = []
    source_kind: dict[Path, str] = {}
    if panorama_image_directory:
        photo_paths = validate_panorama_images(find_panorama_images(Path(panorama_image_directory)))
        image_paths.extend(photo_paths)
        source_kind.update({path: "photo" for path in photo_paths})
    photo_count = len(image_paths)
    extracted_frames: list[Path] = []
    if panorama_video_path:
        video_folder = Path(panorama_video_path).is_dir()
        video_progress: ProgressCallback | None = None
        if progress_callback is not None:
            def video_progress(kept: int, expected: int, detail: str) -> None:
                total = 2 * (photo_count + expected) if expected > 0 else 0
                _notify(progress_callback, photo_count + kept, total, detail)

        if video_folder:
            extracted_frames = [
                frame.path for frame in extract_frames(
                    find_videos(panorama_video_path, PANORAMA_VIDEO_EXTENSIONS), frame_directory,
                    frame_interval_seconds, pick="first", frame_check=_require_2to1,
                    progress_callback=video_progress, label="360 degree video",
                )
            ]
        elif video_progress is None:
            extracted_frames = extract_video_frames(
                Path(panorama_video_path), frame_directory, frame_interval_seconds
            )
        else:
            extracted_frames = extract_video_frames(
                Path(panorama_video_path), frame_directory, frame_interval_seconds, video_progress
            )
        image_paths.extend(extracted_frames)
        source_kind.update({path: "video_frame" for path in extracted_frames})

    cubemap_directory.mkdir(parents=True, exist_ok=True)
    # COLMAP groups matching image suffixes after an image prefix into one
    # frame. The per-face folders provide those prefixes without touching a
    # legacy flat cubemap that may still be present in an existing project.
    for face_name in CUBEMAP_FACE_NAMES:
        (cubemap_directory / face_name).mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    source_total = len(image_paths)
    for source_index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise CaptureValidationError(f"Panorama image cannot be read: {image_path.name}")
        if source_index == 0:
            # 0 = native; larger requests are capped at native (upscaling adds no detail)
            native = native_face_size(image.shape[1])
            face_size = native if face_size <= 0 else min(face_size, native)
        source_id = f"panorama_{source_index:06d}"
        face_records = []
        for face_name, face_image in equirectangular_to_cubemap(image, face_size).items():
            face_path = cubemap_directory / face_name / f"{source_id}.jpg"
            if not cv2.imwrite(str(face_path), face_image):
                raise CaptureValidationError(f"Cubemap face could not be written: {face_path.name}")
            face_records.append({
                "name": face_name,
                "path": face_path.relative_to(cubemap_directory).as_posix(),
            })
        manifest_entries.append(
            {
                "source_id": source_id,
                "source_path": str(image_path),
                "source_kind": source_kind[image_path],
                "faces": face_records,
            }
        )
        _notify(progress_callback, source_total + source_index + 1, 2 * source_total,
                f"Würfelflächen {source_index + 1}/{source_total}")
    manifest_path.write_text(
        json.dumps(
            {
                "cubemap_layout_version": CUBEMAP_LAYOUT_VERSION,
                "cubemap_layout": "face_subdirectories",
                "frame_interval_seconds": frame_interval_seconds,
                "face_size": face_size,
                "panoramas": manifest_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PreparedPanoramaInputs(
        cubemap_directory=cubemap_directory,
        manifest_path=manifest_path,
        panorama_count=len(image_paths),
        video_frame_count=len(extracted_frames),
        cubemap_image_count=len(manifest_entries) * len(CUBEMAP_FACE_NAMES),
        face_size=face_size,
    )
