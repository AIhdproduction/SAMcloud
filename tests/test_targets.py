"""Synthetic accuracy tests for non-coded ground-control target detection."""

import cv2
import numpy as np
import pytest

from samcloud.control_points.targets import DetectionSettings, detect_target


SCALE = 8
SHAPE = (640, 800)


def _finish(canvas: np.ndarray, seed: int = 1) -> np.ndarray:
    image = cv2.resize(canvas, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_AREA)
    image = cv2.GaussianBlur(image, (0, 0), 0.6)
    noise = np.random.default_rng(seed).normal(0, 3, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _cross(center: tuple[float, float], angle: float = 0, contrast: int = 220, perspective: bool = False) -> np.ndarray:
    canvas = np.full((SHAPE[0] * SCALE, SHAPE[1] * SCALE), 240, np.uint8)
    cx, cy = np.array(center) * SCALE
    radius = 34 * SCALE
    points = []
    for sign in (-1, 1):
        triangle = np.array([[-radius, sign * radius], [radius, sign * radius], [0, 0]], np.float32)
        rotation = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
        triangle = triangle @ rotation.T + (cx, cy)
        points.append(triangle)
    for triangle in points:
        cv2.fillConvexPoly(canvas, np.round(triangle).astype(np.int32), 240 - contrast)
    if perspective:
        source = np.float32([[cx - 80 * SCALE, cy - 80 * SCALE], [cx + 80 * SCALE, cy - 80 * SCALE], [cx + 80 * SCALE, cy + 80 * SCALE], [cx - 80 * SCALE, cy + 80 * SCALE]])
        target = source + np.float32([[-20 * SCALE, -14 * SCALE], [22 * SCALE, 10 * SCALE], [12 * SCALE, 18 * SCALE], [-16 * SCALE, -10 * SCALE]])
        canvas = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(source, target), (canvas.shape[1], canvas.shape[0]), borderValue=240)
    return _finish(canvas, int(angle * 10) + int(center[0]))


def _circle(center: tuple[float, float], dark: bool = True, tilted: bool = False) -> np.ndarray:
    canvas = np.full((SHAPE[0] * SCALE, SHAPE[1] * SCALE), 235 if dark else 20, np.uint8)
    colour = 20 if dark else 235
    axes = (35 * SCALE, (22 if tilted else 35) * SCALE)
    cv2.ellipse(canvas, tuple(np.round(np.array(center) * SCALE).astype(int)), axes, 29 if tilted else 0, 0, 360, colour, -1, lineType=cv2.LINE_AA)
    return _finish(canvas, int(center[1]))


def _error(detection, center: tuple[float, float]) -> float:
    """Distance to the rendered truth: canvas index c*SCALE downsampled by INTER_AREA lies at c + 0.5/SCALE."""
    assert detection is not None
    offset = 0.5 / SCALE
    return float(np.hypot(detection.u - (center[0] + offset), detection.v - (center[1] + offset)))


def _warped_center(center: tuple[float, float]) -> tuple[float, float]:
    cx, cy = np.array(center) * SCALE
    source = np.float32([[cx - 80 * SCALE, cy - 80 * SCALE], [cx + 80 * SCALE, cy - 80 * SCALE], [cx + 80 * SCALE, cy + 80 * SCALE], [cx - 80 * SCALE, cy + 80 * SCALE]])
    target = source + np.float32([[-20 * SCALE, -14 * SCALE], [22 * SCALE, 10 * SCALE], [12 * SCALE, 18 * SCALE], [-16 * SCALE, -10 * SCALE]])
    return tuple(cv2.perspectiveTransform(np.float32([[[cx, cy]]]), cv2.getPerspectiveTransform(source, target))[0, 0] / SCALE)


@pytest.mark.parametrize("angle", [0, 23, 45, 71])
def test_cross_subpixel_accuracy_at_multiple_rotations(angle: float) -> None:
    center = (412.37, 288.81)
    assert _error(detect_target(_cross(center, angle), (center[0] + 0.5, center[1] + 0.5), 110, "cross"), center) < 0.15


def test_cross_under_perspective_warp() -> None:
    center = (412.37, 288.81)
    warped = _warped_center(center)
    assert _error(detect_target(_cross(center, 31, perspective=True), (warped[0] + 0.5, warped[1] + 0.5), 120, "cross"), warped) < 0.3


@pytest.mark.parametrize("dark,tilted,limit", [(True, False, 0.2), (False, False, 0.2), (True, True, 0.3)])
def test_circle_subpixel_accuracy(dark: bool, tilted: bool, limit: float) -> None:
    center = (412.37, 288.81)
    assert _error(detect_target(_circle(center, dark, tilted), (center[0] + 0.5, center[1] + 0.5), 110, "circle"), center) < limit


def test_texture_and_ambiguous_crosses_are_rejected() -> None:
    texture = np.random.default_rng(4).normal(128, 18, SHAPE).clip(0, 255).astype(np.uint8)
    assert detect_target(texture, (400.5, 320.5), 180, "cross") is None
    image = np.full(SHAPE, 240, np.uint8)
    for center in ((330.37, 300.81), (470.37, 300.81)):
        target = _cross(center)
        mask = target < 128
        image[mask] = target[mask]
    assert detect_target(image, (400.5, 300.5), 130, "cross") is None


def test_stronger_cross_wins_and_input_variants_work() -> None:
    center = (412.37, 288.81)
    image = _cross(center)
    weak = _cross((510.37, 320.81), contrast=55)
    image[weak < 180] = weak[weak < 180]
    detection = detect_target(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), (450.5, 300.5), 110, "cross")
    assert _error(detection, center) < 0.15
    sixteen_bit = image.astype(np.uint16) * 257
    assert _error(detect_target(sixteen_bit, (center[0] - 70 + 0.5, center[1] + 0.5), 90, "cross"), center) < 0.15


def test_pixel_convention_and_large_prediction_offset() -> None:
    center = (60.0, 60.0)
    assert _error(detect_target(_cross(center), (60.5, 60.5), 30, "cross"), center) < 0.1
    center = (412.37, 288.81)
    assert _error(detect_target(_cross(center), (center[0] - 72 + 0.5, center[1] + 0.5), 90, "cross"), center) < 0.15


def _bowtie_bgr(center, angle, foreground, background, checker=False, seed=7):
    canvas = np.full((SHAPE[0] * SCALE, SHAPE[1] * SCALE, 3), background, np.uint8)
    cx, cy = np.array(center) * SCALE
    if checker:
        disc = np.zeros(canvas.shape[:2], np.uint8)
        cv2.circle(disc, (int(cx), int(cy)), 70 * SCALE, 255, -1)
        yy, xx = np.indices(canvas.shape[:2])
        pattern = (((xx // (4 * SCALE)) + (yy // (4 * SCALE))) % 2 == 0)
        canvas[(disc > 0) & pattern] = (150, 150, 150)
        canvas[(disc > 0) & ~pattern] = (190, 190, 190)
    length, half_width = 30 * SCALE, 16.5 * SCALE
    rotation = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
    for sign in (-1, 1):
        triangle = np.array([[0, 0], [sign * length, -half_width], [sign * length, half_width]], np.float32) @ rotation.T + (cx, cy)
        cv2.fillConvexPoly(canvas, np.round(triangle).astype(np.int32), foreground)
    small = cv2.resize(canvas, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 0.8)
    noise = np.random.default_rng(seed).normal(0, 4, small.shape)
    return np.clip(small.astype(np.float32) + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("angle", [0, 23, 45, 71])
def test_bowtie_white_on_grey_asphalt_rotations(angle: float) -> None:
    center = (412.37, 288.81)
    image = _bowtie_bgr(center, angle, (240, 240, 240), (130, 130, 130))
    settings = DetectionSettings(polarity="bright", min_contrast=50, min_score=0.72)
    assert _error(detect_target(image, (center[0] + 9.3, center[1] - 7.1), 40, "cross", settings=settings), center) < 0.3


def test_bowtie_on_manhole_like_checker_background() -> None:
    center = (412.37, 288.81)
    image = _bowtie_bgr(center, 31, (245, 245, 245), (110, 110, 110), checker=True)
    settings = DetectionSettings(polarity="bright", min_contrast=50, min_score=0.72)
    # textured surroundings fall back to the triangle-apex midpoint (typically 0.3-0.7 px)
    assert _error(detect_target(image, (center[0] + 0.5, center[1] + 0.5), 40, "cross", settings=settings), center) < 0.8


def test_bright_polarity_ignores_dark_bowtie() -> None:
    center = (412.37, 288.81)
    image = _bowtie_bgr(center, 17, (30, 30, 30), (220, 220, 220))
    assert detect_target(image, (center[0] + 0.5, center[1] + 0.5), 40, "cross", settings=DetectionSettings(polarity="bright")) is None
    assert detect_target(image, (center[0] + 0.5, center[1] + 0.5), 40, "cross", settings=DetectionSettings(polarity="dark")) is not None


def test_saturated_bright_blob_rejected() -> None:
    center = (412.37, 288.81)
    settings = DetectionSettings(polarity="bright", min_contrast=50, max_saturation=70)
    green = _bowtie_bgr(center, 12, (0, 230, 0), (60, 60, 60))
    white = _bowtie_bgr(center, 12, (235, 235, 235), (60, 60, 60))
    assert detect_target(green, (center[0] + 0.5, center[1] + 0.5), 40, "cross", settings=settings) is None
    assert _error(detect_target(white, (center[0] + 0.5, center[1] + 0.5), 40, "cross", settings=settings), center) < 0.3
