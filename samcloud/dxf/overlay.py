"""Alpha compositing of background images and DXF vector overlays, plus snapshot rasterization."""

from __future__ import annotations

import cv2
import numpy as np

from samcloud.dxf.result import DxfPlanResult, aci_to_rgb


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

        for entity in entities:
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
            elif etype == "insert":
                point = entity["point"]
                px, py = to_u(float(point[0])), to_v(float(point[1]))
                cv2.circle(canvas, (px, py), 3, color_bgra, thickness=-1)

    return cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
