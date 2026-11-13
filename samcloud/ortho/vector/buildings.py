"""Roof ridge / eave heights of regularised building footprints from the DSM."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon

from .raster_grid import RasterGrid


@dataclass(frozen=True, slots=True)
class RoofSettings:
    """Roof height sampling (metres, square metres)."""

    ridge_percentile: float = 98.0
    eave_ring_inner_m: float = 0.3
    eave_ring_outer_m: float = 1.2
    flat_roof_difference_m: float = 0.5
    text_height_m: float = 0.8
    min_area_m2: float = 20.0


def roof_heights(polygon: Polygon, dsm: np.ndarray, grid: RasterGrid, settings: RoofSettings = RoofSettings()) -> tuple[float, float] | None:
    """(ridge, eave) DSM heights of a building: the ridge is a high DSM percentile inside the
    footprint, the eave the median DSM in a thin ring just inside the outline (the roof edge).
    Heights are returned exactly as in the DSM; None when the footprint has no DSM data."""
    heights = np.asarray(dsm, dtype=np.float32)
    if heights.shape != (grid.height, grid.width):
        raise ValueError("dsm shape must match RasterGrid")
    window = _window(polygon, grid)
    if window is None:
        return None
    local_grid, (row_slice, col_slice) = window
    local_heights = heights[row_slice, col_slice]
    inside = _raster(polygon, local_grid)
    values = local_heights[inside & np.isfinite(local_heights)]
    if len(values) < 4:
        return None
    ridge = float(np.percentile(values, settings.ridge_percentile))
    inner_polygon = polygon.buffer(-settings.eave_ring_inner_m)
    outer_polygon = polygon.buffer(-settings.eave_ring_outer_m)
    inner = _raster(inner_polygon, local_grid) if not inner_polygon.is_empty else np.zeros_like(inside)
    outer = _raster(outer_polygon, local_grid) if not outer_polygon.is_empty else np.zeros_like(inside)
    ring = inner & ~outer
    ring_values = local_heights[ring & np.isfinite(local_heights)]
    eave = float(np.median(ring_values)) if len(ring_values) >= 4 else float(np.percentile(values, 10.0))
    return ridge, min(eave, ridge)


def roof_height_entities(polygon: Polygon, ridge_abs: float, eave_abs: float, settings: RoofSettings = RoofSettings()) -> list[dict[str, object]]:
    """Roof text for a building: ``First +x.xx  Traufe +y.yy`` (absolute heights), or ``Dach +x.xx`` for flat roofs."""
    if ridge_abs - eave_abs < settings.flat_roof_difference_m:
        text = f"Dach {ridge_abs:+.2f}"
    else:
        text = f"First {ridge_abs:+.2f}  Traufe {eave_abs:+.2f}"
    anchor = polygon.representative_point()
    return [{"type": "text", "text": text, "point": [float(anchor.x), float(anchor.y) - 1.6 * settings.text_height_m], "height": settings.text_height_m, "rotation": 0.0, "layer_role": "roof_heights"}]


def _window(polygon: Polygon, grid: RasterGrid) -> tuple[RasterGrid, tuple[slice, slice]] | None:
    """The raster window around a polygon (with a 2 px margin) and its own grid."""
    if polygon.is_empty:
        return None
    min_x, min_y, max_x, max_y = polygon.bounds
    row0f, col0f = grid.local_to_pixel(min_x, max_y)
    row1f, col1f = grid.local_to_pixel(max_x, min_y)
    row0, col0 = max(0, int(np.floor(row0f)) - 2), max(0, int(np.floor(col0f)) - 2)
    row1, col1 = min(grid.height, int(np.ceil(row1f)) + 3), min(grid.width, int(np.ceil(col1f)) + 3)
    if row1 <= row0 or col1 <= col0:
        return None
    local = RasterGrid(grid.min_x + col0 * grid.pixel_size, grid.max_y - row0 * grid.pixel_size, grid.pixel_size, col1 - col0, row1 - row0)
    return local, (slice(row0, row1), slice(col0, col1))


def _raster(polygon: Polygon, grid: RasterGrid) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    if polygon.is_empty:
        return mask.astype(bool)
    geometries = [polygon] if isinstance(polygon, Polygon) else list(getattr(polygon, "geoms", []))
    for geometry in geometries:
        exterior = np.asarray([[round(grid.local_to_pixel(x, y)[1]), round(grid.local_to_pixel(x, y)[0])] for x, y in geometry.exterior.coords], dtype=np.int32)
        cv2.fillPoly(mask, [exterior], 1)
        for hole in geometry.interiors:
            cv2.fillPoly(mask, [np.asarray([[round(grid.local_to_pixel(x, y)[1]), round(grid.local_to_pixel(x, y)[0])] for x, y in hole.coords], dtype=np.int32)], 0)
    return mask.astype(bool)
