"""Tests for the printable AprilTag PDF sheets."""

from __future__ import annotations

import re

import cv2
import numpy as np
import pytest

from samcloud.control_points.apriltag_print import (
    PRINT_DPI,
    build_apriltag_pdf,
    render_apriltag_page,
    tag_box,
)
from samcloud.control_points.apriltags import detect_apriltags_in_image


def _dark_extent_mm(page: np.ndarray) -> tuple[float, float]:
    left, top, right, bottom = tag_box()
    pad = 6
    window = page[top - pad:bottom + pad, left - pad:right + pad]
    cols = np.where((window < 128).any(axis=0))[0]
    rows = np.where((window < 128).any(axis=1))[0]
    mm_per_px = 25.4 / PRINT_DPI
    return (cols[-1] - cols[0] + 1) * mm_per_px, (rows[-1] - rows[0] + 1) * mm_per_px


def test_pdf_has_one_page_per_tag() -> None:
    pdf = build_apriltag_pdf(10, 3)

    assert pdf.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page\b(?!s)", pdf)) == 3


def test_rendered_pages_decode_to_their_ids() -> None:
    for tag_id in (0, 7, 311, 586):
        page = np.asarray(render_apriltag_page(tag_id))
        small = cv2.resize(page, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        found = detect_apriltags_in_image(small, "page")
        assert [observation.tag_id for observation in found] == [tag_id]


def test_black_square_is_150_mm_and_page_is_a4() -> None:
    image = render_apriltag_page(42)
    width_mm, height_mm = _dark_extent_mm(np.asarray(image))

    assert width_mm == pytest.approx(150.0, abs=0.1)
    assert height_mm == pytest.approx(150.0, abs=0.1)
    assert image.size[0] / PRINT_DPI * 25.4 == pytest.approx(210.0, abs=0.5)
    assert image.size[1] / PRINT_DPI * 25.4 == pytest.approx(297.0, abs=0.5)


def test_pdf_media_box_is_a4_in_points() -> None:
    pdf = build_apriltag_pdf(3, 1)
    match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)

    assert match is not None
    assert float(match.group(1)) == pytest.approx(595.0, abs=1.5)
    assert float(match.group(2)) == pytest.approx(842.0, abs=1.5)


@pytest.mark.parametrize(
    ("start_id", "count"),
    [(0, 0), (0, 101), (-1, 5), (580, 8), (587, 1)],
)
def test_invalid_ranges_raise(start_id: int, count: int) -> None:
    with pytest.raises(ValueError):
        build_apriltag_pdf(start_id, count)


def test_boundary_range_is_accepted() -> None:
    assert build_apriltag_pdf(586, 1).startswith(b"%PDF")
