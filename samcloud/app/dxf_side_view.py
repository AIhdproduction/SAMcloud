"""Server-side orthographic side-view PNG rendering for indoor point clouds."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _build_color_table() -> np.ndarray:
    """Precompute RGB palette indexed by classification code (0-255)."""
    table = np.zeros((256, 3), dtype=np.uint8)
    table[:] = (180, 180, 180)

    # Standard / ASPRS codes
    table[0] = (140, 140, 140)   # Unknown / noise
    table[1] = (180, 180, 180)   # Unclassified
    table[2] = (166, 124, 82)    # Ground
    table[3] = (126, 200, 80)    # Low vegetation
    table[4] = (80, 160, 60)     # Medium vegetation
    table[5] = (34, 139, 34)     # Trees / high vegetation
    table[6] = (217, 83, 79)     # Buildings
    table[7] = (130, 130, 130)   # Low point / noise
    table[9] = (30, 144, 255)    # Water
    table[11] = (77, 77, 77)     # Road
    table[14] = (0, 206, 209)    # Wire

    # Cyclone / indoor codes
    table[100] = (245, 245, 245) # Wall (bright white/silver)
    table[101] = (130, 150, 175) # Ceiling (slate blue)
    table[102] = (180, 130, 85)  # Floor (warm ochre/wood)
    table[103] = (255, 140, 0)   # Door (orange)
    table[104] = (38, 183, 239)  # Window (SAMcloud cyan)
    table[105] = (255, 215, 0)   # Stairs (gold)
    table[107] = (210, 105, 210) # Railing (magenta)
    table[108] = (220, 220, 220) # Column
    table[109] = (170, 170, 195) # Beam
    table[110] = (135, 135, 145) # Steel
    for code in (111, 112, 113, 114): # MEP
        table[code] = (235, 155, 55)
    for code in (117, 118, 119, 120): # Furniture
        table[code] = (155, 120, 225) # Purple
    table[121] = (75, 198, 146)  # Sanitary (mint green)
    table[180] = (210, 110, 40)  # Construction material
    return table


COLOR_TABLE = _build_color_table()


def _empty_png() -> bytes:
    """Create a minimal 16x16 dark placeholder PNG."""
    arr = np.full((16, 16, 3), (9, 19, 29), dtype=np.uint8)
    success, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if success:
        return buf.tobytes()
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, format="PNG")
    return bio.getvalue()


def encode_png_data_url(png_bytes: bytes) -> str:
    """Encode PNG bytes as a browser-ready base64 data URL."""
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_side_view(
    preview_npz_path: Path | str,
    axis: str = "x",
    slab_center: float | None = None,
    slab_thickness: float | None = None,
    alignment: Any = None,
    scale: Any = None,
) -> tuple[bytes, tuple[float, float, float, float]]:
    """Render an orthographic per-class density raster PNG and compute world bounds.

    Args:
        preview_npz_path: Path to products/dxf/preview.npz.
        axis: View axis 'x' or 'y'.
        slab_center: Center of the depth section slab along the view axis, or None.
        slab_thickness: Thickness of the depth section slab, or None.
        alignment: Alignment dict, AlignmentResult, or rotation matrix, or None.
        scale: Scale factor number, scale state dict, or None.

    Returns:
        A tuple of (png_bytes, (horizontal_min, horizontal_max, z_min, z_max)).
    """
    path = Path(preview_npz_path)
    if not path.is_file():
        return _empty_png(), (0.0, 1.0, 0.0, 1.0)

    try:
        with np.load(path) as data:
            if "xyz" not in data:
                return _empty_png(), (0.0, 1.0, 0.0, 1.0)
            xyz = np.asarray(data["xyz"], dtype=np.float64)
            cls = np.asarray(data["cls"], dtype=np.uint8) if "cls" in data else np.zeros(len(xyz), dtype=np.uint8)
            stored_scale = float(data["scale_factor"]) if "scale_factor" in data else 1.0
    except Exception:
        return _empty_png(), (0.0, 1.0, 0.0, 1.0)

    if len(xyz) == 0:
        return _empty_png(), (0.0, 1.0, 0.0, 1.0)

    # 1. Apply scale adjustment
    scale_factor = 1.0
    if isinstance(scale, dict):
        scale_factor = float(scale.get("factor", 1.0))
    elif scale is not None:
        try:
            scale_factor = float(scale)
        except (ValueError, TypeError):
            scale_factor = 1.0

    if scale_factor > 0 and stored_scale > 0 and not np.isclose(scale_factor, stored_scale, atol=1e-6):
        xyz = xyz * (scale_factor / stored_scale)
    elif scale_factor > 0 and not np.isclose(scale_factor, 1.0, atol=1e-6):
        xyz = xyz * scale_factor

    # 2. Apply alignment rotation if present
    rot: np.ndarray | None = None
    if isinstance(alignment, dict) and "rotation" in alignment:
        rot = np.asarray(alignment["rotation"], dtype=np.float64)
    elif hasattr(alignment, "rotation"):
        rot = np.asarray(alignment.rotation, dtype=np.float64)
    elif isinstance(alignment, np.ndarray) and alignment.shape == (3, 3):
        rot = np.asarray(alignment, dtype=np.float64)

    if rot is not None and rot.shape == (3, 3):
        xyz = xyz @ rot.T

    # 3. Project according to view axis
    axis_clean = str(axis).lower().strip()
    if axis_clean == "x":
        depth = xyz[:, 0]
        h = xyz[:, 1]
        v = xyz[:, 2]
    else:
        depth = xyz[:, 1]
        h = xyz[:, 0]
        v = xyz[:, 2]

    # 4. Optional depth slab filtering
    if slab_center is not None and slab_thickness is not None:
        try:
            center_val = float(slab_center)
            thick_val = float(slab_thickness)
            if thick_val > 0:
                half = thick_val / 2.0
                mask = (depth >= center_val - half) & (depth <= center_val + half)
                if np.any(mask):
                    h = h[mask]
                    v = v[mask]
                    cls = cls[mask]
                else:
                    h = np.empty(0, dtype=np.float64)
                    v = np.empty(0, dtype=np.float64)
                    cls = np.empty(0, dtype=np.uint8)
        except (ValueError, TypeError):
            pass

    if len(h) == 0:
        return _empty_png(), (0.0, 1.0, 0.0, 1.0)

    h_min = float(np.min(h))
    h_max = float(np.max(h))
    z_min = float(np.min(v))
    z_max = float(np.max(v))

    if h_max - h_min < 1e-3:
        h_max = h_min + 1.0
    if z_max - z_min < 1e-3:
        z_max = z_min + 1.0

    world_bounds = (h_min, h_max, z_min, z_max)

    # 5. Determine image dimensions (maximum 1600 px wide, proportional)
    delta_h = h_max - h_min
    delta_z = z_max - z_min
    aspect = delta_h / delta_z

    if aspect >= 1.0:
        width = 1600
        height = max(50, min(1600, int(round(1600 / aspect))))
    else:
        height = 1600
        width = max(50, min(1600, int(round(1600 * aspect))))

    col = np.clip(np.floor((h - h_min) / delta_h * (width - 1)).astype(np.int32), 0, width - 1)
    row = np.clip(np.floor((z_max - v) / delta_z * (height - 1)).astype(np.int32), 0, height - 1)

    # 6. Count point density and store class
    bg_color = np.array([9, 19, 29], dtype=np.float32)
    count_grid = np.zeros((height, width), dtype=np.int32)
    np.add.at(count_grid, (row, col), 1)

    class_grid = np.zeros((height, width), dtype=np.uint8)
    class_grid[row, col] = cls

    occupied = count_grid > 0
    result_rgb = np.full((height, width, 3), bg_color, dtype=np.float32)

    if np.any(occupied):
        p95 = float(np.percentile(count_grid[occupied], 95))
        norm_density = np.log1p(count_grid[occupied].astype(np.float32)) / np.log1p(max(4.0, p95))
        alpha = np.clip(0.42 + 0.58 * norm_density, 0.42, 1.0)[:, None]

        point_colors = COLOR_TABLE[class_grid[occupied]].astype(np.float32)
        blended = bg_color * (1.0 - alpha) + point_colors * alpha
        result_rgb[occupied] = blended

    img_uint8 = np.clip(np.round(result_rgb), 0, 255).astype(np.uint8)

    success, buf = cv2.imencode(".png", cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    if success:
        return buf.tobytes(), world_bounds

    bio = io.BytesIO()
    Image.fromarray(img_uint8).save(bio, format="PNG")
    return bio.getvalue(), world_bounds
