"""Tests for AprilTag detection on warped, noisy synthetic images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from samcloud.control_points.apriltags import (
    detect_apriltags,
    detect_apriltags_in_image,
    load_tag_detections,
    save_tag_detections,
)

MODULE = 50
SIZE = 8 * MODULE
MARGIN = MODULE


def _marker_with_margin(tag_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, SIZE, borderBits=1)
    return np.pad(marker, MARGIN, constant_values=255)


def _warped_scene(tag_id: int = 17, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return the noisy image and the true corners in COLMAP pixel convention."""
    source = _marker_with_margin(tag_id)
    # Pixel centres sit at integer coordinates for OpenCV warping.
    src = np.float32([[MARGIN - 0.5, MARGIN - 0.5], [MARGIN + SIZE - 0.5, MARGIN - 0.5],
                      [MARGIN + SIZE - 0.5, MARGIN + SIZE - 0.5], [MARGIN - 0.5, MARGIN + SIZE - 0.5]])
    half = 130.0
    base = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    angle = np.deg2rad(23.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    perspective = np.array([[-14.3, 9.7], [11.1, -6.2], [20.7, 12.9], [-8.3, -17.4]])
    dst = (base @ rotation.T + np.array([430.37, 350.81]) + perspective).astype(np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(source, homography, (900, 700), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=120)
    rng = np.random.default_rng(seed)
    noisy = np.clip(warped.astype(float) + rng.normal(0.0, 3.0, warped.shape), 0, 255).astype(np.uint8)
    return noisy, dst.astype(float) + 0.5


def _intersection(corners: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = corners
    d1, d2 = p2 - p0, p3 - p1
    t = np.linalg.solve(np.array([d1, -d2]).T, p1 - p0)[0]
    return p0 + t * d1


def test_detects_id_and_corners_within_subpixel_tolerance() -> None:
    image, truth = _warped_scene()

    found = detect_apriltags_in_image(image, "warp.jpg")

    assert len(found) == 1
    observation = found[0]
    assert observation.tag_id == 17
    assert observation.image == "warp.jpg"
    detected = np.asarray(observation.corners)
    assert np.abs(detected - truth).max() < 0.3


def test_center_is_diagonal_intersection_not_corner_mean() -> None:
    image, truth = _warped_scene()
    observation = detect_apriltags_in_image(image, "warp.jpg")[0]
    corners = np.asarray(observation.corners)

    assert np.allclose(observation.center, _intersection(corners), atol=1e-9)
    assert np.linalg.norm(np.asarray(observation.center) - _intersection(truth)) < 0.3
    assert np.linalg.norm(corners.mean(axis=0) - _intersection(corners)) > 0.5


def test_color_images_are_converted_to_grayscale() -> None:
    image, _ = _warped_scene()
    colour = cv2.merge([image, image, image])

    found = detect_apriltags_in_image(colour, "c.jpg")

    assert [o.tag_id for o in found] == [17]


def test_image_without_tag_returns_empty_list() -> None:
    rng = np.random.default_rng(3)
    image = np.clip(rng.normal(128.0, 20.0, (480, 640)), 0, 255).astype(np.uint8)

    assert detect_apriltags_in_image(image, "empty.jpg") == []


def test_detection_json_round_trip(tmp_path: Path) -> None:
    image, _ = _warped_scene()
    detections = {"a.jpg": detect_apriltags_in_image(image, "a.jpg"), "b.jpg": []}
    path = tmp_path / "sub" / "apriltag_detections.json"

    save_tag_detections(path, detections)
    loaded = load_tag_detections(path)

    assert loaded == detections


def test_detect_apriltags_reads_files_and_skips_unreadable(tmp_path: Path) -> None:
    image, _ = _warped_scene(tag_id=99)
    good = tmp_path / "good.png"
    cv2.imwrite(str(good), image)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")

    result = detect_apriltags({"good.png": good, "broken.png": broken, "missing.png": tmp_path / "missing.png"}, workers=2)

    assert set(result) == {"good.png"}
    assert [o.tag_id for o in result["good.png"]] == [99]
    assert result["good.png"][0].image == "good.png"
