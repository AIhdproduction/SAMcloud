"""Snapshot raster text: Unicode glyphs, rotation and MIDDLE_CENTER anchoring."""

from __future__ import annotations

import numpy as np

from samcloud.dxf.las_source import Frame
from samcloud.dxf.overlay import _unicode_font_path, render_dxf_rgba
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus

SIZE = 400
BOUNDS = (0.0, 0.0, 4.0, 4.0)


def _render(text: str, *, point: tuple[float, float] = (2.0, 2.0), height: float = 0.3, **extra: object) -> np.ndarray:
    entity = {"type": "text", "text": text, "point": list(point), "height": height, "rotation": 0.0, **extra}
    result = DxfPlanResult(
        "overlay", "site", "indoor", None, 1.0, Frame(np.zeros(3), 1.0, np.eye(3)), "overlay-text",
        {"A-ANNO": [entity]}, {"A-ANNO": LayerStatus("A-ANNO", "ok", None, 1)}, {"A-ANNO": 7},
        BackgroundImage(None, None, BOUNDS),
    )
    return render_dxf_rgba(result, bounds=BOUNDS, size=(SIZE, SIZE))


def _ink(rendered: np.ndarray) -> np.ndarray:
    return np.argwhere(rendered[..., 3] > 40)


def test_unicode_font_is_available() -> None:
    assert _unicode_font_path() is not None


def test_umlaut_dots_rise_above_x_height() -> None:
    plain = _ink(_render("u"))
    umlaut = _ink(_render("\u00fc"))
    assert len(plain) > 0 and len(umlaut) > 0
    # same baseline: the diaeresis dots sit above the top of the plain "u"
    assert int(umlaut[:, 0].min()) <= int(plain[:, 0].min()) - 3
    assert int(umlaut[:, 0].max()) == int(plain[:, 0].max())


def test_superscript_two_renders_ink() -> None:
    plain = _render("m")
    squared = _render("m\u00b2")
    assert not np.array_equal(plain, squared)
    plain_ink, squared_ink = _ink(plain), _ink(squared)
    assert len(squared_ink) > len(plain_ink)
    # the superscript adds ink to the right of the "m" and reaches higher than its x-height
    assert int(squared_ink[:, 1].max()) > int(plain_ink[:, 1].max()) + 3
    assert int(squared_ink[:, 0].min()) < int(plain_ink[:, 0].min())


def test_middle_center_text_is_centred_on_anchor() -> None:
    scale = (SIZE - 1) / 4.0
    anchor_u, anchor_v = round(1.5 * scale), round((4.0 - 2.5) * scale)
    ink = _ink(_render("Gaste-WC 1.5 m", point=(1.5, 2.5), align="MIDDLE_CENTER"))
    text_width = int(ink[:, 1].max() - ink[:, 1].min() + 1)
    centroid_v, centroid_u = ink.mean(axis=0)
    assert abs(centroid_u - anchor_u) <= 0.15 * text_width
    assert abs(centroid_v - anchor_v) <= 0.15 * text_width
    # default alignment equals the explicit MIDDLE_CENTER
    assert np.array_equal(_render("Tuer", point=(1.5, 2.5)), _render("Tuer", point=(1.5, 2.5), align="MIDDLE_CENTER"))


def test_left_alignment_puts_text_to_the_right_of_anchor() -> None:
    scale = (SIZE - 1) / 4.0
    anchor_u = round(1.0 * scale)
    ink = _ink(_render("Regal", point=(1.0, 2.0), align="LEFT"))
    assert int(ink[:, 1].min()) >= anchor_u - 3


def test_rotated_text_is_taller_than_wide() -> None:
    flat = _ink(_render("T\u00fcr 0.89 m", rotation=0.0))
    upright = _ink(_render("T\u00fcr 0.89 m", rotation=90.0))
    flat_w, flat_h = flat[:, 1].max() - flat[:, 1].min() + 1, flat[:, 0].max() - flat[:, 0].min() + 1
    up_w, up_h = upright[:, 1].max() - upright[:, 1].min() + 1, upright[:, 0].max() - upright[:, 0].min() + 1
    assert flat_w > flat_h
    assert up_h > up_w
    assert abs(up_h - flat_w) <= 3 and abs(up_w - flat_h) <= 3


def test_text_ink_uses_layer_colour() -> None:
    rendered = _render("H")
    solid = rendered[rendered[..., 3] == 255]
    assert len(solid) > 0
    assert (solid[:, :3] == np.array([255, 255, 255])).all()  # ACI 7 is white
