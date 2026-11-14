"""Printable A4 numbered cross target sheets (two black bow-tie triangles on white paper in a 180 mm square area)."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

CROSS_TARGET_MM = 180.0
PRINT_DPI = 300.0
CROSS_MAX_COUNT = 100
CROSS_MAX_NUMBER = 9999

PX_PER_MM = PRINT_DPI / 25.4
PAGE_W, PAGE_H = round(210.0 * PX_PER_MM), round(297.0 * PX_PER_MM)
CROSS_TARGET_PX = round(CROSS_TARGET_MM * PX_PER_MM)
TOP_MARGIN_MM = 20.0
TOP_MARGIN_PX = round(TOP_MARGIN_MM * PX_PER_MM)

_FONT = "C:/Windows/Fonts/arial.ttf"
_FONT_B = "C:/Windows/Fonts/arialbd.ttf"


def _mm(value: float) -> int:
    return round(value * PX_PER_MM)


def _font(path: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


def cross_target_side_px() -> int:
    """Side length of the 180 mm target square (triangle extents) in page pixels at 300 dpi."""
    return CROSS_TARGET_PX


def cross_target_center_px() -> tuple[float, float]:
    """COLMAP coordinates (+0.5 convention) of the square centre on the page."""
    cx = (PAGE_W - CROSS_TARGET_PX) // 2 + CROSS_TARGET_PX / 2.0
    cy = TOP_MARGIN_PX + CROSS_TARGET_PX / 2.0
    return (cx + 0.5, cy + 0.5)


def render_cross_target_page(number: int) -> Image.Image:
    """Render a single A4 printable page with a numbered cross target."""
    num = int(number)
    if not (1 <= num <= CROSS_MAX_NUMBER):
        raise ValueError(f"Target number must be between 1 and {CROSS_MAX_NUMBER}")

    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    draw = ImageDraw.Draw(img)

    square_left = (PAGE_W - CROSS_TARGET_PX) // 2
    square_top = TOP_MARGIN_PX
    square_right = square_left + CROSS_TARGET_PX
    square_bottom = square_top + CROSS_TARGET_PX

    cx = square_left + CROSS_TARGET_PX / 2.0
    cy = square_top + CROSS_TARGET_PX / 2.0

    # Two black triangles on white paper (bow-tie): tips meet exactly at the square centre.
    # Their outer vertical edges lie on the left/right edge of the 180 mm square area.
    draw.polygon([(cx, cy), (square_left, square_top), (square_left, square_bottom)], fill=0)
    draw.polygon([(cx, cy), (square_right, square_top), (square_right, square_bottom)], fill=0)

    # Corner crop marks 2 mm outside the square: the square may be cut out here (a 2 mm white paper
    # border remains, harmless for the dark triangles) or the whole sheet may be laid out.
    strip = _mm(36.0)
    cut_bottom = square_bottom + strip
    gap, length = _mm(2.0), _mm(8.0)
    for px, py, sx, sy in (
        (square_left, square_top, -1, -1),
        (square_right, square_top, 1, -1),
        (square_left, square_bottom, -1, 1),
        (square_right, square_bottom, 1, 1),
    ):
        draw.line([(px + sx * gap, py + sy * gap), (px + sx * (gap + length), py + sy * gap)], fill=0, width=3)
        draw.line([(px + sx * gap, py + sy * gap), (px + sx * gap, py + sy * (gap + length))], fill=0, width=3)

    # Number label inside cut area below the square
    f_num = _font(_FONT_B, _mm(14.0))
    f_sub = _font(_FONT, _mm(4.2))
    f_small, f_note = _font(_FONT, _mm(3.6)), _font(_FONT_B, _mm(4.2))

    num_text = f"Nr. {num}"
    num_w = draw.textlength(num_text, font=f_num)
    draw.text(((PAGE_W - num_w) / 2.0, square_bottom + _mm(6.0)), num_text, font=f_num, fill=0)

    sub_text = "Kreuz-Target · 180 mm · schwarze Dreiecke auf weiss · optional an den Marken ausschneiden"
    sub_w = draw.textlength(sub_text, font=f_sub)
    draw.text(((PAGE_W - sub_w) / 2.0, square_bottom + _mm(24.0)), sub_text, font=f_sub, fill=0)

    # 100 mm check bar outside cut area
    by = cut_bottom + _mm(10.0)
    bx = (PAGE_W - _mm(100.0)) // 2
    draw.rectangle([bx, by, bx + _mm(100.0), by + _mm(3.0)], fill=0)
    for c in range(11):
        x = bx + _mm(10.0 * c)
        draw.line([(x, by - _mm(3.0 if c % 5 == 0 else 1.8)), (x, by)], fill=0, width=3)
        if c % 5 == 0:
            t = f"{c * 10}"
            draw.text((x - draw.textlength(t, font=f_small) / 2.0, by - _mm(8.0)), t, font=f_small, fill=0)
    bar_label = "Kontrollstrecke 100 mm – nach dem Druck nachmessen"
    draw.text(((PAGE_W - draw.textlength(bar_label, font=f_small)) / 2.0, by + _mm(5.0)), bar_label, font=f_small, fill=0)

    # German hints outside cut area
    notes = [
        "Mit 100 % / „Tatsächliche Größe“ drucken.",
        "Blatt oder Quadrat (an den Marken) flach auf den Untergrund legen. Mitte einmessen.",
        "Funktioniert auf hellem und dunklem Untergrund.",
    ]
    y = by + _mm(14.0)
    for index, note in enumerate(notes):
        font = f_note if index == 0 else f_small
        draw.text(((PAGE_W - draw.textlength(note, font=font)) / 2.0, y), note, font=font, fill=0)
        y += _mm(7.0)

    footer = f"SAMcloud Kreuz-Target  ·  Nr. {num}"
    draw.text(((PAGE_W - draw.textlength(footer, font=f_small)) / 2.0, PAGE_H - _mm(12.0)), footer, font=f_small, fill=110)

    return img


def build_cross_target_pdf(start_number: int, count: int) -> bytes:
    """Build a multi-page PDF with numbered cross targets."""
    if not (1 <= count <= CROSS_MAX_COUNT):
        raise ValueError(f"count must be between 1 and {CROSS_MAX_COUNT}")
    if start_number < 1:
        raise ValueError("start_number must be at least 1")
    if start_number + count - 1 > CROSS_MAX_NUMBER:
        raise ValueError(f"target numbers must stay within 1..{CROSS_MAX_NUMBER}")

    pages = [render_cross_target_page(i) for i in range(start_number, start_number + count)]
    buffer = io.BytesIO()
    pages[0].save(buffer, "PDF", resolution=PRINT_DPI, save_all=True, append_images=pages[1:])
    return buffer.getvalue()
