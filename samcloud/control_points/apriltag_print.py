"""Printable A4 AprilTag sheets (tag36h11, black square exactly 150 mm)."""

from __future__ import annotations

import io

import cv2
from PIL import Image, ImageDraw, ImageFont

from samcloud.control_points.apriltags import APRILTAG_FAMILY, APRILTAG_ID_MAX, APRILTAG_SIZE_M

TAG_MM = APRILTAG_SIZE_M * 1000.0
MODULE_PX = 222
TAG_PX = 8 * MODULE_PX
PRINT_DPI = TAG_PX / (TAG_MM / 25.4)
PX_PER_MM = PRINT_DPI / 25.4
PAGE_W, PAGE_H = round(210 * PX_PER_MM), round(297 * PX_PER_MM)
QUIET = MODULE_PX
CUT_W = TAG_PX + 2 * QUIET
STRIP = round(14 * PX_PER_MM)
CUT_H = CUT_W + STRIP
X0 = (PAGE_W - CUT_W) // 2
Y0 = round(18 * PX_PER_MM)

_FONT = "C:/Windows/Fonts/arial.ttf"
_FONT_B = "C:/Windows/Fonts/arialbd.ttf"


def _mm(value: float) -> int:
    return round(value * PX_PER_MM)


def _font(path: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


def tag_box() -> tuple[int, int, int, int]:
    """Pixel box (left, top, right, bottom exclusive) of the black square on a page."""
    left, top = X0 + QUIET, Y0 + QUIET
    return left, top, left + TAG_PX, top + TAG_PX


def _dashed_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], dash: int, gap: int, width: int, fill: int) -> None:
    x0, y0, x1, y1 = box
    for a, b, fixed, horizontal in ((x0, x1, y0, True), (x0, x1, y1, True), (y0, y1, x0, False), (y0, y1, x1, False)):
        p = a
        while p < b:
            q = min(p + dash, b)
            draw.line([(p, fixed), (q, fixed)] if horizontal else [(fixed, p), (fixed, q)], fill=fill, width=width)
            p = q + gap


def render_apriltag_page(tag_id: int) -> Image.Image:
    if not 0 <= int(tag_id) <= APRILTAG_ID_MAX:
        raise ValueError(f"AprilTag id must be between 0 and {APRILTAG_ID_MAX}")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), TAG_PX, borderBits=1)
    tx, ty = X0 + QUIET, Y0 + QUIET
    img.paste(Image.fromarray(marker), (tx, ty))
    d = ImageDraw.Draw(img)
    _dashed_rect(d, (X0, Y0, X0 + CUT_W, Y0 + CUT_H), _mm(4), _mm(2.5), 3, 150)

    tick, tw = QUIET // 3, 4
    cx, cy = tx + TAG_PX // 2, ty + TAG_PX // 2
    d.line([(cx, Y0), (cx, Y0 + tick)], fill=0, width=tw)
    d.line([(cx, ty + TAG_PX + QUIET - tick), (cx, ty + TAG_PX + QUIET)], fill=0, width=tw)
    d.line([(X0, cy), (X0 + tick, cy)], fill=0, width=tw)
    d.line([(X0 + CUT_W - tick, cy), (X0 + CUT_W, cy)], fill=0, width=tw)

    f_label = _font(_FONT_B, _mm(6))
    label = f"ID {tag_id}  ·  {APRILTAG_FAMILY}  ·  150 mm"
    lw = d.textlength(label, font=f_label)
    d.text((X0 + (CUT_W - lw) / 2, Y0 + CUT_W + (STRIP - _mm(6)) / 2 - _mm(0.8)), label, font=f_label, fill=0)

    f_small, f_note = _font(_FONT, _mm(3.6)), _font(_FONT_B, _mm(4.2))
    by = Y0 + CUT_H + _mm(16)
    bx = (PAGE_W - _mm(100)) // 2
    d.rectangle([bx, by, bx + _mm(100), by + _mm(3)], fill=0)
    for c in range(11):
        x = bx + _mm(10 * c)
        d.line([(x, by - _mm(3 if c % 5 == 0 else 1.8)), (x, by)], fill=0, width=3)
        if c % 5 == 0:
            t = f"{c * 10}"
            d.text((x - d.textlength(t, font=f_small) / 2, by - _mm(8)), t, font=f_small, fill=0)
    t = "Kontrollstrecke 100 mm – nach dem Druck nachmessen"
    d.text(((PAGE_W - d.textlength(t, font=f_small)) / 2, by + _mm(5)), t, font=f_small, fill=0)
    notes = [
        "Drucken mit 100 % / „Tatsächliche Größe“ – NICHT „An Seite anpassen“.",
        "Schwarzes Quadrat muss 150,0 mm messen. Entlang der grauen Linie ausschneiden.",
        "Flach auf die Wand kleben (nicht gewellt). Koordinate = Tag-Mitte (Schnittpunkt der Randmarken).",
    ]
    y = by + _mm(14)
    for index, note in enumerate(notes):
        font = f_note if index == 0 else f_small
        d.text(((PAGE_W - d.textlength(note, font=font)) / 2, y), note, font=font, fill=0)
        y += _mm(7)
    footer = f"SAMcloud AprilTag  ·  ID {tag_id}"
    d.text(((PAGE_W - d.textlength(footer, font=f_small)) / 2, PAGE_H - _mm(12)), footer, font=f_small, fill=110)
    return img


def build_apriltag_pdf(start_id: int, count: int) -> bytes:
    if not 1 <= count <= 100:
        raise ValueError("count must be between 1 and 100")
    if start_id < 0 or start_id + count - 1 > APRILTAG_ID_MAX:
        raise ValueError(f"AprilTag ids must stay within 0..{APRILTAG_ID_MAX}")
    pages = [render_apriltag_page(i) for i in range(start_id, start_id + count)]
    buffer = io.BytesIO()
    pages[0].save(buffer, "PDF", resolution=PRINT_DPI, save_all=True, append_images=pages[1:])
    return buffer.getvalue()
