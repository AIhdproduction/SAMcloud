from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from samcloud.acquisition import panorama
from samcloud.acquisition.panorama import (
    CaptureValidationError,
    extract_video_frames,
    prepare_panorama_inputs,
    validate_capture_configuration,
    validate_panorama_images,
)


def _write_panorama(path: Path, width: int = 80, height: int = 40) -> None:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_panorama_requires_at_least_one_source() -> None:
    with pytest.raises(CaptureValidationError, match="Select a 360 degree video"):
        validate_capture_configuration("panorama_360")


def test_panorama_rejects_raw_insta360_video() -> None:
    with pytest.raises(CaptureValidationError, match="MP4"):
        validate_capture_configuration("panorama_360", panorama_video_path="tour.insv")


def test_panorama_rejects_non_equirectangular_images(tmp_path: Path) -> None:
    image_path = tmp_path / "invalid.jpg"
    _write_panorama(image_path, width=60, height=40)

    with pytest.raises(CaptureValidationError, match="2:1"):
        validate_panorama_images([image_path])


def test_prepare_panorama_photos_creates_cubemap_and_manifest(tmp_path: Path) -> None:
    photo_directory = tmp_path / "photos"
    photo_directory.mkdir()
    _write_panorama(photo_directory / "pano.jpg")

    prepared = prepare_panorama_inputs(
        str(photo_directory), "", tmp_path / "project-inputs", face_size=16
    )

    assert prepared.panorama_count == 1
    assert prepared.video_frame_count == 0
    assert prepared.cubemap_image_count == 6
    assert prepared.manifest_path.exists()
    assert [
        path.relative_to(prepared.cubemap_directory).as_posix()
        for path in sorted(prepared.cubemap_directory.rglob("*.jpg"))
    ] == [
        "back/panorama_000000.jpg",
        "down/panorama_000000.jpg",
        "front/panorama_000000.jpg",
        "left/panorama_000000.jpg",
        "right/panorama_000000.jpg",
        "up/panorama_000000.jpg",
    ]
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["cubemap_layout_version"] == 2
    assert manifest["panoramas"][0]["faces"][0]["path"] == "front/panorama_000000.jpg"


def test_prepare_panorama_combines_photo_and_video_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    photo_directory = tmp_path / "photos"
    photo_directory.mkdir()
    _write_panorama(photo_directory / "photo.jpg")
    video_path = tmp_path / "tour.mp4"
    video_path.write_bytes(b"placeholder")

    def fake_extract(_: Path, output_directory: Path, __: float) -> list[Path]:
        output_directory.mkdir(parents=True, exist_ok=True)
        frame_path = output_directory / "frame_000000.jpg"
        _write_panorama(frame_path)
        return [frame_path]

    monkeypatch.setattr(panorama, "extract_video_frames", fake_extract)
    prepared = prepare_panorama_inputs(
        str(photo_directory), str(video_path), tmp_path / "project-inputs", face_size=16
    )

    assert prepared.panorama_count == 2
    assert prepared.video_frame_count == 1
    assert prepared.cubemap_image_count == 12


def test_extract_video_frames_uses_fixed_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, _: int) -> float:
            return 2.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index == 5:
                return False, None
            self.index += 1
            return True, np.full((40, 80, 3), self.index, dtype=np.uint8)

        def release(self) -> None:
            return None

    monkeypatch.setattr(panorama.cv2, "VideoCapture", lambda _: FakeCapture())
    frames = extract_video_frames(tmp_path / "tour.mp4", tmp_path / "frames", 1.0)

    assert [frame.name for frame in frames] == [
        "frame_000000.jpg",
        "frame_000001.jpg",
        "frame_000002.jpg",
    ]


def test_native_face_size_is_a_quarter_of_the_panorama_width(tmp_path: Path) -> None:
    from samcloud.acquisition.panorama import native_face_size

    assert native_face_size(5760) == 1440
    assert native_face_size(3840) == 960
    photo_directory = tmp_path / "photos"
    photo_directory.mkdir()
    _write_panorama(photo_directory / "pano.jpg")

    prepared = prepare_panorama_inputs(str(photo_directory), "", tmp_path / "project-inputs", face_size=0)

    width = cv2.imread(str(photo_directory / "pano.jpg")).shape[1]
    assert prepared.face_size == native_face_size(width)
    face = cv2.imread(str(prepared.cubemap_directory / "front" / "panorama_000000.jpg"))
    assert face.shape[:2] == (prepared.face_size, prepared.face_size)


def test_requested_face_size_is_capped_at_native(tmp_path: Path) -> None:
    import cv2
    from samcloud.acquisition.panorama import native_face_size, prepare_panorama_inputs

    photo_directory = tmp_path / "panoramas"
    photo_directory.mkdir()
    image = np.random.default_rng(1).integers(0, 255, (64, 128, 3), dtype=np.uint8)
    cv2.imwrite(str(photo_directory / "pano.jpg"), image)
    prepared = prepare_panorama_inputs(str(photo_directory), "", tmp_path / "inputs", face_size=1232)
    assert prepared.face_size == native_face_size(128) == 32
    prepared = prepare_panorama_inputs(str(photo_directory), "", tmp_path / "inputs2", face_size=24)
    assert prepared.face_size == 24
