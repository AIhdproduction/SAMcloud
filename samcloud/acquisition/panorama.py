"""Prepare equirectangular 360 degree media for the COLMAP workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


CAPTURE_TYPES = ("drone_gps", "camera", "panorama_360")
PANORAMA_IMAGE_EXTENSIONS = (".jpg", ".jpeg")
PANORAMA_VIDEO_EXTENSIONS = (".mp4",)
CUBEMAP_FACE_NAMES = ("front", "right", "back", "left", "up", "down")


class CaptureValidationError(ValueError):
    """Raised when a capture source cannot be processed safely."""


@dataclass(frozen=True)
class PreparedPanoramaInputs:
    """Project-owned paths and source counts created for a 360 degree job."""

    cubemap_directory: Path
    manifest_path: Path
    panorama_count: int
    video_frame_count: int
    cubemap_image_count: int


def validate_capture_configuration(
    capture_type: str,
    image_directory: str = "",
    panorama_image_directory: str = "",
    panorama_video_path: str = "",
    frame_interval_seconds: float = 1.0,
    *,
    check_paths: bool = False,
) -> None:
    """Validate the selected acquisition mode without changing project data."""
    if capture_type not in CAPTURE_TYPES:
        raise CaptureValidationError("Select a supported capture type")
    if frame_interval_seconds <= 0:
        raise CaptureValidationError("Video frame interval must be greater than zero")

    if capture_type == "panorama_360":
        if not panorama_image_directory and not panorama_video_path:
            raise CaptureValidationError(
                "Select a 360 degree video, a panorama image directory, or both"
            )
        if panorama_video_path and Path(panorama_video_path).suffix.lower() not in PANORAMA_VIDEO_EXTENSIONS:
            raise CaptureValidationError("360 degree video must be an exported MP4 file")
        if check_paths:
            if panorama_video_path and not Path(panorama_video_path).is_file():
                raise CaptureValidationError("Selected 360 degree video file was not found")
            if panorama_image_directory and not Path(panorama_image_directory).is_dir():
                raise CaptureValidationError("Selected panorama image directory was not found")
        return

    if not image_directory:
        raise CaptureValidationError("Select an image directory before creating the project")
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


def extract_video_frames(video_path: Path, output_directory: Path, interval_seconds: float) -> list[Path]:
    """Extract one equirectangular frame at each requested fixed time interval."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise CaptureValidationError("360 degree video cannot be opened")
    frames_per_second = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(frames_per_second) or frames_per_second <= 0:
        capture.release()
        raise CaptureValidationError("360 degree video does not provide a valid frame rate")

    output_directory.mkdir(parents=True, exist_ok=True)
    next_frame_index = 0
    interval_frames = max(1, round(frames_per_second * interval_seconds))
    frame_index = 0
    extracted: list[Path] = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_index >= next_frame_index:
            height, width = frame.shape[:2]
            if height == 0 or abs((width / height) - 2.0) > 0.03:
                capture.release()
                raise CaptureValidationError("360 degree video frames must use a 2:1 equirectangular format")
            frame_path = output_directory / f"frame_{len(extracted):06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                capture.release()
                raise CaptureValidationError(f"Video frame could not be written: {frame_path.name}")
            extracted.append(frame_path)
            next_frame_index += interval_frames
        frame_index += 1
    capture.release()
    if not extracted:
        raise CaptureValidationError("No frames could be extracted from the 360 degree video")
    return extracted


def equirectangular_to_cubemap(image: np.ndarray, face_size: int) -> dict[str, np.ndarray]:
    """Render six 90 degree perspective views from a 2:1 panorama."""
    if face_size <= 0:
        raise ValueError("Cubemap face size must be greater than zero")
    height, width = image.shape[:2]
    if height == 0 or abs((width / height) - 2.0) > 0.03:
        raise CaptureValidationError("Cubemap input must be a readable 2:1 equirectangular image")

    coordinates = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    directions = {
        "front": (horizontal, -vertical, np.ones_like(horizontal)),
        "right": (np.ones_like(horizontal), -vertical, -horizontal),
        "back": (-horizontal, -vertical, -np.ones_like(horizontal)),
        "left": (-np.ones_like(horizontal), -vertical, horizontal),
        "up": (horizontal, np.ones_like(horizontal), vertical),
        "down": (horizontal, -np.ones_like(horizontal), -vertical),
    }
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


def prepare_panorama_inputs(
    panorama_image_directory: str,
    panorama_video_path: str,
    project_inputs_directory: Path,
    frame_interval_seconds: float = 1.0,
    face_size: int = 1024,
) -> PreparedPanoramaInputs:
    """Create video frames, cubemap images, and an origin manifest for COLMAP."""
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
    extracted_frames: list[Path] = []
    if panorama_video_path:
        extracted_frames = extract_video_frames(
            Path(panorama_video_path), frame_directory, frame_interval_seconds
        )
        image_paths.extend(extracted_frames)
        source_kind.update({path: "video_frame" for path in extracted_frames})

    cubemap_directory.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    for source_index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise CaptureValidationError(f"Panorama image cannot be read: {image_path.name}")
        source_id = f"panorama_{source_index:06d}"
        face_records = []
        for face_name, face_image in equirectangular_to_cubemap(image, face_size).items():
            face_path = cubemap_directory / f"{source_id}_{face_name}.jpg"
            if not cv2.imwrite(str(face_path), face_image):
                raise CaptureValidationError(f"Cubemap face could not be written: {face_path.name}")
            face_records.append({"name": face_name, "path": str(face_path)})
        manifest_entries.append(
            {
                "source_id": source_id,
                "source_path": str(image_path),
                "source_kind": source_kind[image_path],
                "faces": face_records,
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
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
    )
