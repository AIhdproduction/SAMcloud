"""Tests for printable numbered cross targets."""

from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
import pytest
import yaml

from samcloud.control_points.cross_target_print import (
    build_cross_target_pdf,
    cross_target_center_px,
    cross_target_side_px,
    render_cross_target_page,
)
from samcloud.control_points.targets import detect_target, detect_targets_in_window, load_settings


def test_build_cross_target_pdf() -> None:
    """PDF generation returns valid PDF with exact page count."""
    pdf_bytes = build_cross_target_pdf(3, 2)
    assert pdf_bytes.startswith(b"%PDF")
    page_matches = re.findall(rb"/Type\s*/Page(?!s)", pdf_bytes)
    assert len(page_matches) == 2


def test_cross_target_geometry() -> None:
    """Black bow-tie triangles span 180 mm; left/right of the centre black, above/below white."""
    img = render_cross_target_page(5)
    arr = np.array(img)
    cx_colmap, cy_colmap = cross_target_center_px()
    cx, cy = cx_colmap - 0.5, cy_colmap - 0.5
    mm = 300.0 / 25.4

    # Horizontal extent of the dark triangles across the centre row (limited to the square area).
    row_y = int(round(cy))
    near = np.abs(np.arange(arr.shape[1]) - cx) <= 95.0 * mm
    dark_cols = np.where((arr[row_y, :] < 50) & near)[0]
    assert len(dark_cols) > 0
    assert abs((dark_cols.max() - dark_cols.min() + 1) / mm - 180.0) <= 0.5

    # Vertical extent of the dark triangles at their outer edges (x = 1 mm inside left edge).
    col_x = int(round(cx - 89.0 * mm))
    near_y = np.abs(np.arange(arr.shape[0]) - cy) <= 95.0 * mm
    dark_rows = np.where((arr[:, col_x] < 50) & near_y)[0]
    assert len(dark_rows) > 0
    assert abs((dark_rows.max() - dark_rows.min() + 1) / mm - 178.0) <= 2.5  # 2 mm chord loss at 1 mm from the edge

    px_20mm = int(round(20.0 * mm))
    col = int(round(cx))
    # 20 mm left and right of the centre: black
    assert arr[row_y, int(round(cx - px_20mm))] < 50
    assert arr[row_y, int(round(cx + px_20mm))] < 50
    # 20 mm above and below the centre: white
    assert arr[int(round(cy - px_20mm)), col] > 200
    assert arr[int(round(cy + px_20mm)), col] > 200


def _defaults_settings():
    defaults_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yml"
    return load_settings(yaml.safe_load(defaults_path.read_text(encoding="utf-8")))


def _ground_with_target(page: np.ndarray, ground: int, mode: str) -> np.ndarray:
    side = cross_target_side_px()
    cx, cy = cross_target_center_px()
    left = (page.shape[1] - side) // 2
    top = int(round(cy - 0.5)) - side // 2
    canvas = np.full_like(page, ground)
    if mode == "uncut":
        # whole A4 sheet (white paper) lying on the ground
        return page.copy()
    # cut at the crop marks: 2 mm of white paper remains around the square
    margin = int(round(2.0 * 300.0 / 25.4))
    y0, x0 = top - margin, left - margin
    canvas[y0 : top + side + 1 + margin, x0 : left + side + 1 + margin] = page[
        y0 : top + side + 1 + margin, x0 : left + side + 1 + margin
    ]
    return canvas


# Black printed triangles must be found on dark ground (asphalt 60, concrete 120) as well as on light ground,
# both for the whole sheet and for the square cut at the crop marks (thin white rim).
@pytest.mark.parametrize("mode", ["cut", "uncut"])
@pytest.mark.parametrize("ground", [60, 120, 180, 200])
@pytest.mark.parametrize("target_px", [60, 104, 150])
def test_cross_target_detectability(target_px: int, ground: int, mode: str) -> None:
    """Downscaled, noisy black bow-tie on dark and light ground is found by the real detector within 2.0 px."""
    page = np.array(render_cross_target_page(1))
    cx, cy = cross_target_center_px()
    side = cross_target_side_px()
    scene = _ground_with_target(page, ground, mode)

    scale = target_px / side
    scaled = cv2.resize(scene, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pad = int(target_px * 1.5)
    canvas = np.full((scaled.shape[0] + 2 * pad, scaled.shape[1] + 2 * pad), ground, np.uint8)
    canvas[pad : pad + scaled.shape[0], pad : pad + scaled.shape[1]] = scaled
    rng = np.random.default_rng(42)
    noisy = np.clip(canvas.astype(np.float32) + rng.normal(0, 3.0, canvas.shape), 0, 255).astype(np.uint8)

    true_u = (cx - 0.5) * scale + 0.5 + pad
    true_v = (cy - 0.5) * scale + 0.5 + pad
    candidates = detect_targets_in_window(
        noisy,
        center_uv=(true_u + 3.3, true_v - 2.7),
        radius_px=target_px * 0.5,
        target_type="cross",
        settings=_defaults_settings(),
    )
    assert len(candidates) >= 1
    best = candidates[0]
    assert np.hypot(best.u - true_u, best.v - true_v) < 2.0

    # the ambiguity-checked single-target API must return it as well (not None)
    if target_px == 104:
        single = detect_target(
            noisy, (true_u + 3.3, true_v - 2.7), target_px * 0.5, "cross", settings=_defaults_settings(),
        )
        assert single is not None
        assert np.hypot(single.u - true_u, single.v - true_v) < 2.0


def test_cross_target_validation() -> None:
    """Invalid count or start_number values raise ValueError."""
    with pytest.raises(ValueError):
        build_cross_target_pdf(1, 0)
    with pytest.raises(ValueError):
        build_cross_target_pdf(1, 101)
    with pytest.raises(ValueError):
        build_cross_target_pdf(0, 10)
    with pytest.raises(ValueError):
        build_cross_target_pdf(9999, 2)
