"""Alpha compositing of background images and DXF vector overlays, plus snapshot rasterization."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from samcloud.dxf.result import DxfPlanResult, aci_to_rgb, expand_insert

# DejaVu Sans cap height as a fraction of the em size (cap height 1493 / 2048 units).
_DEJAVU_CAP_RATIO = 1493.0 / 2048.0


@lru_cache(maxsize=1)
def _unicode_font_path() -> str | None:
    """Path of the bundled DejaVu Sans TTF (Unicode), or None if Pillow or the font is unavailable."""
    try:
        import matplotlib
        from PIL import ImageFont  # noqa: F401
    except Exception:
        return None
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return str(path) if path.is_file() else None


@lru_cache(maxsize=128)
def _unicode_font(size_tenths: int):
    from PIL import ImageFont

    path = _unicode_font_path()
    if path is None:
        return None
    try:
        return ImageFont.truetype(path, size_tenths / 10.0)
    except Exception:
        return None


def _text_offsets(align: str) -> tuple[str, float]:
    """Horizontal Pillow anchor letter and vertical shift (in cap heights, +down) for a DXF alignment name."""
    name = align.upper()
    if name in {"MIDDLE_CENTER", "MIDDLE"}:
        return "m", 0.5
    if name == "MIDDLE_LEFT":
        return "l", 0.5
    if name == "MIDDLE_RIGHT":
        return "r", 0.5
    if name == "TOP_CENTER":
        return "m", 1.0
    if name == "TOP_LEFT":
        return "l", 1.0
    if name == "TOP_RIGHT":
        return "r", 1.0
    if name in {"CENTER", "BOTTOM_CENTER"}:
        return "m", 0.0
    if name in {"RIGHT", "BOTTOM_RIGHT"}:
        return "r", 0.0
    return "l", 0.0


def _draw_unicode_text(
    canvas: np.ndarray, text: str, anchor: tuple[int, int], cap_px: float, rotation: float, align: str,
    color_bgra: tuple[int, int, int, int],
) -> bool:
    """Blend Unicode text (Pillow, DejaVu Sans) into a BGRA canvas; returns False if it cannot be drawn this way."""
    if not text:
        return True
    font = _unicode_font(max(10, int(round(cap_px / _DEJAVU_CAP_RATIO * 10.0))))
    if font is None:
        return False
    from PIL import Image, ImageDraw

    horizontal, vertical_caps = _text_offsets(align)
    width = font.getlength(text)
    extent = int(np.ceil(width + 2.0 * cap_px / _DEJAVU_CAP_RATIO)) + 8
    side = 2 * extent + 1
    centre = side // 2
    mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(mask).text((centre, centre + vertical_caps * cap_px), text, fill=255, font=font, anchor=horizontal + "s")
    if abs(rotation % 360.0) > 1e-9:
        mask = mask.rotate(rotation, resample=Image.Resampling.BICUBIC, center=(centre, centre))
    coverage = np.asarray(mask, dtype=np.float32) / 255.0
    rows, cols = np.nonzero(coverage > 0.0)
    if rows.size == 0:
        return True
    top, bottom, left, right = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    coverage = coverage[top:bottom, left:right]
    y0, x0 = anchor[1] + int(top) - centre, anchor[0] + int(left) - centre
    cy0, cx0 = max(0, y0), max(0, x0)
    cy1, cx1 = min(canvas.shape[0], y0 + coverage.shape[0]), min(canvas.shape[1], x0 + coverage.shape[1])
    if cy1 <= cy0 or cx1 <= cx0:
        return True
    alpha = coverage[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, np.newaxis]
    region = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    blended = region * (1.0 - alpha) + np.asarray(color_bgra, dtype=np.float32) * alpha
    canvas[cy0:cy1, cx0:cx1] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    return True


def composite(
    image_rgb: np.ndarray,
    dxf_rgba: np.ndarray,
    image_opacity: float,
    dxf_opacity: float,
    background: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Composite an RGB background and RGBA DXF overlay over a neutral background.

    image_opacity and dxf_opacity range from 0.0 to 1.0.
    Returns uint8 (H, W, 3) RGB array.
    """
    img_op = float(np.clip(image_opacity, 0.0, 1.0))
    dxf_op = float(np.clip(dxf_opacity, 0.0, 1.0))

    img = np.asarray(image_rgb, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64).reshape((1, 1, 3))

    # 1. Image over background with image_opacity
    base = img * img_op + bg * (1.0 - img_op)

    # 2. DXF over base with dxf_alpha * dxf_opacity
    dxf_rgb = np.asarray(dxf_rgba[..., :3], dtype=np.float64)
    dxf_a = (np.asarray(dxf_rgba[..., 3], dtype=np.float64) / 255.0) * dxf_op
    dxf_a = dxf_a[..., np.newaxis]

    result = dxf_rgb * dxf_a + base * (1.0 - dxf_a)
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def render_dxf_rgba(
    result: DxfPlanResult,
    bounds: tuple[float, float, float, float] | None = None,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Rasterize vector plan entities into an RGBA image using layer colors."""
    chosen_bounds = bounds if bounds is not None else result.background.bounds
    min_x, min_y, max_x, max_y = chosen_bounds

    if size is not None:
        width, height = size
    else:
        png_data = result.background.rgb_png or result.background.class_png
        if png_data is not None:
            decoded = cv2.imdecode(np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if decoded is not None and decoded.ndim >= 2:
                height, width = decoded.shape[:2]
            else:
                width = 1024
                span_x = max(1e-3, max_x - min_x)
                span_y = max(1e-3, max_y - min_y)
                height = max(1, int(round(width * span_y / span_x)))
        else:
            width = 1024
            span_x = max(1e-3, max_x - min_x)
            span_y = max(1e-3, max_y - min_y)
            height = max(1, int(round(width * span_y / span_x)))

    width = max(1, int(width))
    height = max(1, int(height))

    # Canvas in BGRA order for OpenCV drawing
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    scale_x = (width - 1) / span_x
    scale_y = (height - 1) / span_y

    def to_u(x_val: float) -> int:
        return max(0, min(width - 1, int(round((x_val - min_x) * scale_x))))

    def to_v(y_val: float) -> int:
        return max(0, min(height - 1, int(round((max_y - y_val) * scale_y))))

    for layer_name, entities in result.layers.items():
        if not entities:
            continue
        aci = result.layer_colors.get(layer_name, 7)
        r, g, b = aci_to_rgb(aci)
        color_bgra = (int(b), int(g), int(r), 255)

        drawable = [part for entity in entities for part in (expand_insert(entity) if entity.get("type") == "insert" else (entity,))]
        for entity in drawable:
            etype = entity.get("type")
            if etype == "line":
                start = entity["start"]
                end = entity["end"]
                p1 = (to_u(float(start[0])), to_v(float(start[1])))
                p2 = (to_u(float(end[0])), to_v(float(end[1])))
                cv2.line(canvas, p1, p2, color_bgra, thickness=1, lineType=cv2.LINE_AA)
            elif etype == "polyline":
                pts = np.array(
                    [[to_u(float(p[0])), to_v(float(p[1]))] for p in entity.get("points", [])],
                    dtype=np.int32,
                )
                if len(pts) >= 2:
                    closed = bool(entity.get("closed", False))
                    cv2.polylines(canvas, [pts], isClosed=closed, color=color_bgra, thickness=1, lineType=cv2.LINE_AA)
                for hole in entity.get("holes", []):
                    hole_pts = np.array(
                        [[to_u(float(p[0])), to_v(float(p[1]))] for p in hole],
                        dtype=np.int32,
                    )
                    if len(hole_pts) >= 2:
                        cv2.polylines(canvas, [hole_pts], isClosed=True, color=color_bgra, thickness=1, lineType=cv2.LINE_AA)
            elif etype == "circle":
                center = entity["center"]
                cx, cy = to_u(float(center[0])), to_v(float(center[1]))
                radius = float(entity.get("radius", 0.5))
                r_px = max(1, int(round(radius * (scale_x + scale_y) / 2.0)))
                cv2.circle(canvas, (cx, cy), r_px, color_bgra, thickness=1, lineType=cv2.LINE_AA)
            elif etype == "point":
                point = entity["point"]
                px, py = to_u(float(point[0])), to_v(float(point[1]))
                cv2.circle(canvas, (px, py), 2, color_bgra, thickness=-1)
            elif etype == "arc":
                center = entity["center"]
                cx, cy = to_u(float(center[0])), to_v(float(center[1]))
                radius = float(entity.get("radius", 0.5))
                r_px = max(1, int(round(radius * (scale_x + scale_y) / 2.0)))
                start_deg = float(entity.get("start_angle", 0.0))
                end_deg = float(entity.get("end_angle", 360.0))
                cv2.ellipse(canvas, (cx, cy), (r_px, r_px), 0, -end_deg, -start_deg, color_bgra, thickness=1, lineType=cv2.LINE_AA)
            elif etype == "text":
                point = entity.get("point", (0.0, 0.0))
                text = str(entity.get("text", ""))
                height_m = max(0.01, float(entity.get("height", 0.12)))
                anchor = (to_u(float(point[0])), to_v(float(point[1])))
                cap_px = height_m * float(np.sqrt(scale_x * scale_y))
                if _draw_unicode_text(
                    canvas, text, anchor, cap_px, float(entity.get("rotation", 0.0)),
                    str(entity.get("align", "MIDDLE_CENTER")), color_bgra,
                ):
                    continue
                # Fallback without Pillow/DejaVu: Hershey Simplex (ASCII only, no rotation), ~21 px cap height at fontScale=1.
                font_scale = max(0.01, cap_px / 21.0)
                cv2.putText(canvas, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgra, 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
