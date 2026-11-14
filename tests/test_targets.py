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


# --- dark ground: painted white crosses must stay detectable, black-on-white sheets must be found, clutter rejected ---

def _asphalt(shape: tuple[int, int], level: int = 60, seed: int = 3) -> np.ndarray:
    """Grey ground with low-frequency mottling (+-10) and pixel noise, as float canvas."""
    rng = np.random.default_rng(seed)
    mottling = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), 6.0)
    mottling *= 10.0 / max(float(mottling.std()) * 1.5, 1e-6)
    return level + mottling


def _bowtie_points(center: tuple[float, float], length: float, half_width: float, angle: float) -> list[np.ndarray]:
    rotation = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
    cx, cy = np.array(center) * SCALE
    return [
        np.round(np.array([[0, 0], [s * length, -half_width], [s * length, half_width]], np.float32) * SCALE @ rotation.T + (cx, cy)).astype(np.int32)
        for s in (-1, 1)
    ]


def _dark_ground_scene(painted=(), sheets=(), level: int = 60, seed: int = 3, noise: float = 4.0) -> np.ndarray:
    """painted: (center, length, angle) white bow-ties; sheets: (center, side) white squares with black bow-tie."""
    canvas = np.full((SHAPE[0] * SCALE, SHAPE[1] * SCALE), 0, np.float32)
    ground = cv2.resize(_asphalt((SHAPE[0], SHAPE[1]), level, seed), (SHAPE[1] * SCALE, SHAPE[0] * SCALE), interpolation=cv2.INTER_LINEAR)
    canvas[:] = ground
    for center, length, angle in painted:
        for triangle in _bowtie_points(center, length, 0.55 * length, angle):
            cv2.fillConvexPoly(canvas, triangle, 235)
    for center, side in sheets:
        half = int(round((side / 2 + 1.2) * SCALE))  # thin white paper rim around the triangles (cut sheet)
        cx, cy = int(round(center[0] * SCALE)), int(round(center[1] * SCALE))
        cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), 245, -1)
        for triangle in _bowtie_points(center, side / 2, side / 2, 0):
            cv2.fillConvexPoly(canvas, triangle, 12)
    small = cv2.resize(canvas, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 0.7)
    noisy = small + np.random.default_rng(seed + 100).normal(0, noise, small.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("length", [20, 30, 45])
@pytest.mark.parametrize("angle", [0, 31])
def test_painted_white_cross_on_textured_asphalt_regression(length: float, angle: float) -> None:
    """Painted white crosses on mottled asphalt (grey 60) keep sub-pixel accuracy with polarity auto."""
    center = (412.37, 288.81)
    image = _dark_ground_scene(painted=[(center, length, angle)])
    found = detect_target(image, (center[0] + 6.1, center[1] - 4.3), 60, "cross", settings=DetectionSettings())
    assert _error(found, center) < 1.0


def test_textured_asphalt_and_plain_white_sheet_are_not_targets() -> None:
    """No target: bare ground (asphalt, concrete) and a white A4-like sheet without triangles are rejected."""
    center = (412.37, 288.81)
    for level in (60, 120):
        bare = _dark_ground_scene(level=level, seed=level)
        assert detect_target(bare, (center[0], center[1]), 80, "cross", settings=DetectionSettings()) is None
        sheet = bare.copy()
        sheet[int(center[1]) - 32 : int(center[1]) + 33, int(center[0]) - 46 : int(center[0]) + 47] = 235
        sheet = cv2.GaussianBlur(sheet, (0, 0), 0.7)
        assert detect_target(sheet, (center[0], center[1]), 80, "cross", settings=DetectionSettings()) is None


def test_black_sheet_and_white_painted_cross_in_one_window_are_both_candidates() -> None:
    """A printed black bow-tie sheet and a painted white cross side by side are both reported."""
    from samcloud.control_points.targets import detect_targets_in_window

    sheet_center, cross_center = (330.4, 290.2), (470.6, 288.8)
    image = _dark_ground_scene(painted=[(cross_center, 30, 12)], sheets=[(sheet_center, 56)])
    found = detect_targets_in_window(image, (400.0, 289.5), 110, "cross", settings=DetectionSettings(), max_results=5)
    for expected in (sheet_center, cross_center):
        assert any(np.hypot(item.u - (expected[0] + 0.5), item.v - (expected[1] + 0.5)) < 2.0 for item in found), (expected, [(i.u, i.v, i.score) for i in found])
