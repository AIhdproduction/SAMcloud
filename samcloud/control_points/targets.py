"""Sub-pixel detection of non-coded ground-control targets.

All public coordinates use COLMAP's image convention: the centre of the
top-left pixel is ``(0.5, 0.5)``.  OpenCV internally uses pixel-array indices,
so conversions happen only at this module's boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionSettings:
    min_contrast: float = 40.0
    min_score: float = 0.5
    ambiguity_ratio: float = 0.85
    min_size_px: float = 6.0
    max_size_px: float = 400.0
    subpixel_window: int = 5
    polarity: str = "auto"          # "bright" | "dark" | "auto" (both)
    max_saturation: float = 70.0    # HSV saturation limit for bright (white) targets when colour is available


@dataclass(frozen=True)
class TargetDetection:
    u: float
    v: float
    target_type: str
    score: float
    size_px: float
    details: dict[str, Any]


def load_settings(config: dict) -> DetectionSettings:
    """Read detection options from a full config or its detection subsection."""
    section = config.get("control_points", config)
    section = section.get("detection", section)
    fields = DetectionSettings.__dataclass_fields__
    return DetectionSettings(**{name: section[name] for name in fields if name in section})


def detect_target(
    image: np.ndarray,
    predicted_uv: tuple[float, float],
    search_radius_px: float,
    target_type: str,
    *,
    expected_size_px: float | None = None,
    settings: DetectionSettings | None = None,
) -> TargetDetection | None:
    """Find the single unambiguous best target around a predicted COLMAP point."""
    candidates = _detect_window(
        image, predicted_uv, search_radius_px, target_type, max_results=20, settings=settings,
        expected_size_px=expected_size_px,
    )
    if not candidates:
        return None
    best = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        active = settings or DetectionSettings()
        distance = float(np.hypot(best.u - second.u, best.v - second.v))
        if second.score >= best.score * active.ambiguity_ratio and distance > 0.5 * max(best.size_px, second.size_px):
            return None
    return best


def detect_targets_in_window(
    image: np.ndarray,
    center_uv: tuple[float, float],
    radius_px: float,
    target_type: str,
    *,
    max_results: int = 5,
    settings: DetectionSettings | None = None,
) -> list[TargetDetection]:
    """Return scored target candidates in a cropped search window, best first."""
    return _detect_window(image, center_uv, radius_px, target_type, max_results=max_results, settings=settings)


def _detect_window(
    image: np.ndarray,
    center_uv: tuple[float, float],
    radius_px: float,
    target_type: str,
    *,
    max_results: int,
    settings: DetectionSettings | None,
    expected_size_px: float | None = None,
) -> list[TargetDetection]:
    """Shared cropped candidate search, with optional single-target size prior."""
    active = settings or DetectionSettings()
    if target_type not in {"cross", "circle"}:
        raise ValueError("target_type must be 'cross' or 'circle'")
    gray = _gray8(image)
    cx, cy = center_uv[0] - 0.5, center_uv[1] - 0.5
    pad = int(np.ceil(radius_px))
    x0, y0 = max(0, int(np.floor(cx)) - pad), max(0, int(np.floor(cy)) - pad)
    x1, y1 = min(gray.shape[1], int(np.floor(cx)) + pad + 1), min(gray.shape[0], int(np.floor(cy)) + pad + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return []
    crop = gray[y0:y1, x0:x1]
    if target_type == "cross":
        margin = int(min(90.0, active.max_size_px / 2.0))
        bx0, by0 = max(0, int(np.floor(cx)) - pad - margin), max(0, int(np.floor(cy)) - pad - margin)
        bx1 = min(gray.shape[1], int(np.floor(cx)) + pad + margin + 1)
        by1 = min(gray.shape[0], int(np.floor(cy)) + pad + margin + 1)
        saturation = None
        if image.ndim == 3 and image.dtype == np.uint8:
            saturation = cv2.cvtColor(image[by0:by1, bx0:bx1], cv2.COLOR_BGR2HSV)[:, :, 1]
        detections = [
            item for item in _detect_bowties(gray[by0:by1, bx0:bx1], bx0, by0, active, saturation)
            if np.hypot(item.u - center_uv[0], item.v - center_uv[1]) <= radius_px
        ]
    else:
        detections = _detect_circles(crop, x0, y0, active, expected_size_px)
    detections = [item for item in detections if item.score >= active.min_score]
    detections.sort(key=lambda item: item.score, reverse=True)
    return _deduplicate(detections)[:max_results]


def refine_subpixel(
    image: np.ndarray, uv: tuple[float, float], target_type: str, settings: DetectionSettings | None = None
) -> tuple[float, float]:
    """Refine an approximate COLMAP coordinate and return a COLMAP coordinate."""
    active = settings or DetectionSettings()
    gray = _gray8(image)
    x, y = uv[0] - 0.5, uv[1] - 0.5
    if target_type == "circle":
        found = detect_target(gray, uv, max(8.0, active.max_size_px / 2), "circle", settings=active)
        return (found.u, found.v) if found else uv
    point = np.array([[[x, y]]], dtype=np.float32)
    try:
        cv2.cornerSubPix(gray, point, (active.subpixel_window, active.subpixel_window), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 1e-4))
    except cv2.error:
        return uv
    return float(point[0, 0, 0] + 0.5), float(point[0, 0, 1] + 0.5)


def _gray8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2 or image.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("image must be a grayscale or BGR uint8/uint16 array")
    if image.dtype == np.uint16:
        return cv2.convertScaleAbs(image, alpha=255.0 / 65535.0)
    return image


def _bowtie_from_contour(contour: np.ndarray, shape: tuple[int, int]) -> dict[str, Any] | None:
    """Recognise two triangles touching at their tips as one blob with a narrow waist."""
    area = cv2.contourArea(contour)
    hull_index = cv2.convexHull(contour, returnPoints=False)
    if hull_index is None or len(hull_index) < 4:
        return None
    try:
        defects = cv2.convexityDefects(contour, hull_index)
    except cv2.error:
        return None
    if defects is None or len(defects) < 2:
        return None
    scale = float(np.sqrt(area))
    deepest = sorted(defects[:, 0, :], key=lambda row: -row[3])[:2]
    depths = [row[3] / 256.0 for row in deepest]
    if min(depths) < 0.18 * scale:
        return None
    first = contour[deepest[0][2], 0].astype(float)
    second = contour[deepest[1][2], 0].astype(float)
    waist = float(np.linalg.norm(first - second))
    if waist > 0.45 * scale:
        return None
    junction = (first + second) / 2.0
    mask = np.zeros(shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(float)
    normal = np.array([-(second - first)[1], (second - first)[0]])
    normal /= np.linalg.norm(normal) + 1e-9
    side = (points - junction) @ normal
    lobe_areas = ((side > 0).sum(), (side < 0).sum())
    if min(lobe_areas) == 0:
        return None
    ratio = min(lobe_areas) / max(lobe_areas)
    if ratio < 0.5:
        return None
    triangle_fits = []
    apexes: list[np.ndarray] = []
    for sign in (1, -1):
        lobe = points[sign * side > 0].astype(np.float32).reshape(-1, 1, 2)
        if len(lobe) < 6:
            return None
        _, triangle = cv2.minEnclosingTriangle(lobe)
        triangle_area = abs(cv2.contourArea(triangle.reshape(3, 2)))
        triangle_fits.append(len(lobe) / triangle_area if triangle_area > 0 else 0.0)
        vertices = triangle.reshape(3, 2).astype(float)
        apexes.append(vertices[np.argmin(np.linalg.norm(vertices - junction, axis=1))])
    if min(triangle_fits) < 0.6:
        return None
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return {
        "junction": junction, "apex_junction": (apexes[0] + apexes[1]) / 2.0, "scale": scale, "ratio": ratio,
        "waist": waist, "depth": min(depths) / scale, "mask": mask, "fill": area / hull_area if hull_area else 0.0,
        "triangle_fit": min(1.0, min(triangle_fits)),
    }


def _detect_bowties(
    gray: np.ndarray, x0: int, y0: int, settings: DetectionSettings, saturation: np.ndarray | None = None,
) -> list[TargetDetection]:
    """Detect bow-tie crosses (two triangles touching at the tips) with a multi-threshold blob search."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    polarities = ("bright", "dark") if settings.polarity == "auto" else (settings.polarity,)
    found: list[TargetDetection] = []
    for polarity in polarities:
        image = blur if polarity == "bright" else 255 - blur
        low, high = float(np.percentile(image, 50)), float(np.percentile(image, 99.7))
        for threshold in np.linspace(high - 4, low, 12):
            mask = (image > threshold).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < max(40.0, settings.min_size_px ** 2) or area > settings.max_size_px ** 2:
                    continue
                bowtie = _bowtie_from_contour(contour, gray.shape)
                if bowtie is None:
                    continue
                inner = bowtie["mask"]
                ring = cv2.dilate(inner, np.ones((9, 9), np.uint8)) & ~cv2.dilate(inner, np.ones((3, 3), np.uint8))
                if not ring.any():
                    continue
                contrast = abs(float(image[inner > 0].mean()) - float(image[ring > 0].mean()))
                if contrast < settings.min_contrast:
                    continue
                if saturation is not None and polarity == "bright" and float(saturation[inner > 0].mean()) > settings.max_saturation:
                    continue
                window = int(np.clip(round(0.10 * bowtie["scale"]), 3, 9))
                point = np.array([[bowtie["apex_junction"]]], dtype=np.float32)
                try:
                    cv2.cornerSubPix(gray, point, (window, window), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 1e-4))
                except cv2.error:
                    continue
                refined = point[0, 0].astype(float)
                # cornerSubPix is very precise on clean backgrounds but can be pulled away by textured surroundings
                # (e.g. checkered manhole covers); then keep the midpoint of the two triangle apexes.
                if np.linalg.norm(refined - bowtie["apex_junction"]) > 0.8:
                    refined = bowtie["apex_junction"]
                score = float(np.mean([
                    bowtie["ratio"], min(1.0, contrast / 80.0), min(1.0, bowtie["depth"] / 0.35), bowtie["triangle_fit"],
                    max(0.0, 1.0 - bowtie["waist"] / (0.45 * bowtie["scale"])), bowtie["fill"],
                ]))
                found.append(TargetDetection(
                    u=float(refined[0] + x0 + 0.5), v=float(refined[1] + y0 + 0.5), target_type="cross", score=score,
                    size_px=2.0 * bowtie["scale"],
                    details={"polarity": polarity, "contrast": contrast, "threshold": float(threshold),
                             "lobe_ratio": bowtie["ratio"], "triangle_fit": bowtie["triangle_fit"]},
                ))
    found.sort(key=lambda item: item.score, reverse=True)
    merged: list[TargetDetection] = []
    for item in found:
        if all(np.hypot(item.u - other.u, item.v - other.v) > 0.25 * other.size_px for other in merged):
            merged.append(item)
    return merged


def _detect_crosses(gray: np.ndarray, x0: int, y0: int, settings: DetectionSettings, expected_size: float | None) -> list[TargetDetection]:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    ixx = cv2.Sobel(blur, cv2.CV_32F, 2, 0, ksize=3)
    iyy = cv2.Sobel(blur, cv2.CV_32F, 0, 2, ksize=3)
    ixy = cv2.Sobel(blur, cv2.CV_32F, 1, 1, ksize=3)
    saddle = -(ixx * iyy - ixy * ixy)
    local = saddle == cv2.dilate(saddle, np.ones((5, 5), np.uint8))
    threshold = max(float(np.percentile(saddle, 99.2)), 80.0)
    ys, xs = np.where(local & (saddle >= threshold))
    order = np.argsort(saddle[ys, xs])[::-1][:600]
    radii = [max(settings.min_size_px / 2, 3.0), 4.0, 6.0, 8.0, 11.0, 15.0, 20.0, 28.0, 38.0, 55.0, 75.0, 100.0]
    if expected_size:
        radii = [expected_size * value for value in (0.23, 0.30, 0.38, 0.48, 0.60)]
    radii = [radius for radius in radii if settings.min_size_px / 2 <= radius <= settings.max_size_px / 2]
    result: list[TargetDetection] = []
    for index in order:
        x, y = float(xs[index]), float(ys[index])
        best: tuple[float, float, float, float] | None = None
        for radius in radii:
            if x - radius - 2 < 0 or y - radius - 2 < 0 or x + radius + 2 >= gray.shape[1] or y + radius + 2 >= gray.shape[0]:
                continue
            score, contrast, regularity, symmetry = _cross_pattern_score(gray, x, y, radius, settings.min_contrast)
            if best is None or score > best[0]:
                best = (score, radius, contrast, regularity * symmetry)
        if best is None or best[0] < settings.min_score * 0.75:
            continue
        px, py = _subpixel_corner(gray, x, y, settings.subpixel_window)
        # Do not let a noisy corner optimisation jump from the saddle to an edge.
        if np.hypot(px - x, py - y) > 2.5:
            px, py = x, y
        score, radius, contrast, regularity = best
        # The high-resolution anti-aliased render and OpenCV contour/corner
        # coordinates describe pixel corners here; convert to COLMAP centres.
        result.append(TargetDetection(px + x0 + 1.0, py + y0 + 1.0, "cross", float(min(1.0, score)), float(radius * 2), {"contrast": float(contrast), "sector_regularity": float(regularity)}))
    return result


def _cross_pattern_score(gray: np.ndarray, x: float, y: float, radius: float, min_contrast: float) -> tuple[float, float, float, float]:
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    values = _sample_bilinear(gray, x + radius * np.cos(angles), y + radius * np.sin(angles))
    basis = np.column_stack((np.ones_like(angles), np.cos(2 * angles), np.sin(2 * angles)))
    fitted = basis @ np.linalg.lstsq(basis, values, rcond=None)[0]
    amplitude = float(np.hypot(np.linalg.lstsq(basis, values, rcond=None)[0][1], np.linalg.lstsq(basis, values, rcond=None)[0][2]))
    residual = float(np.std(values - fitted))
    contrast = 2.0 * amplitude
    regularity = max(0.0, 1.0 - residual / max(amplitude, 1.0))
    opposite = float(np.mean(np.abs(values - np.roll(values, 16))))
    symmetry = max(0.0, 1.0 - opposite / max(contrast, 1.0))
    contrast_score = min(1.0, contrast / max(min_contrast, 1.0))
    return contrast_score * regularity * symmetry, contrast, regularity, symmetry


def _subpixel_corner(gray: np.ndarray, x: float, y: float, window: int) -> tuple[float, float]:
    point = np.array([[[x, y]]], dtype=np.float32)
    try:
        cv2.cornerSubPix(gray, point, (window, window), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 1e-5))
        return float(point[0, 0, 0]), float(point[0, 0, 1])
    except cv2.error:
        return x, y


def _detect_circles(gray: np.ndarray, x0: int, y0: int, settings: DetectionSettings, expected_size: float | None) -> list[TargetDetection]:
    blur = cv2.GaussianBlur(gray, (0, 0), 0.8)
    result: list[TargetDetection] = []
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, binary = cv2.threshold(blur, 0, 255, mode | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 20:
                continue
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major, minor), angle = ellipse
            size = max(major, minor)
            axis_ratio = max(major, minor) / max(min(major, minor), 1e-6)
            if not settings.min_size_px <= size <= settings.max_size_px or axis_ratio > 3.0:
                continue
            area = cv2.contourArea(contour)
            ellipse_area = np.pi * major * minor / 4.0
            coverage = min(area, ellipse_area) / max(area, ellipse_area, 1.0)
            if coverage < 0.65:
                continue
            mask = np.zeros(gray.shape, np.uint8)
            cv2.ellipse(mask, ellipse, 255, -1)
            ring = cv2.dilate(mask, np.ones((5, 5), np.uint8)) & ~mask
            inside = gray[mask > 0]
            outside = gray[ring > 0]
            if not len(inside) or not len(outside):
                continue
            contrast = abs(float(np.mean(inside)) - float(np.mean(outside)))
            uniformity = max(0.0, 1.0 - float(np.std(inside)) / max(contrast, 1.0))
            size_score = 1.0 if expected_size is None else max(0.0, 1.0 - abs(size - expected_size) / max(expected_size, 1.0))
            score = min(1.0, contrast / max(settings.min_contrast, 1.0)) * coverage * uniformity * (0.7 + 0.3 * size_score)
            result.append(TargetDetection(x + x0 + 0.5, y + y0 + 0.5, "circle", float(score), float(size), {"contrast": contrast, "ellipse_axes": (float(major), float(minor)), "axis_ratio": float(axis_ratio), "angle_deg": float(angle)}))
    return result


def _sample_bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(image.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1)


def _deduplicate(items: list[TargetDetection]) -> list[TargetDetection]:
    kept: list[TargetDetection] = []
    for item in items:
        if all(np.hypot(item.u - other.u, item.v - other.v) > max(item.size_px, other.size_px) * 0.5 for other in kept):
            kept.append(item)
    return kept
