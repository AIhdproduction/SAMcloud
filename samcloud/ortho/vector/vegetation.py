"""Shrubs and hedges as smooth closed outlines with a "Gebüsch h=… m" label."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from skimage.measure import find_contours

from .raster_grid import RasterGrid


@dataclass(frozen=True, slots=True)
class HedgeSettings:
    """Shrub / hedge detection and outline smoothing (metres)."""

    min_height_m: float = 0.5
    max_height_m: float = 2.0
    min_area_m2: float = 2.0
    close_radius_m: float = 0.3
    smooth_sigma_m: float = 0.25
    simplify_m: float = 0.04
    chaikin_iterations: int = 1


def hedge_entities(mask: np.ndarray, chm: np.ndarray, grid: RasterGrid, settings: HedgeSettings = HedgeSettings()) -> list[dict[str, object]]:
    """Closed, smooth outline polylines plus a ``Gebüsch h=… m`` label per shrub cluster.

    ``mask`` marks candidate vegetation pixels; ``chm`` is the canopy height model.  The
    outline comes from the sub-pixel contour of a Gaussian-smoothed mask, so it carries
    no raster staircase.  Height is the 90th percentile of the CHM inside the cluster.
    """
    binary = np.asarray(mask, dtype=bool)
    heights = np.asarray(chm, dtype=np.float32)
    if binary.shape != (grid.height, grid.width) or heights.shape != binary.shape:
        raise ValueError("mask and chm must match RasterGrid")
    candidate = binary & np.isfinite(heights) & (heights >= settings.min_height_m) & (heights < settings.max_height_m)
    if not candidate.any():
        return []
    radius = max(1, int(round(settings.close_radius_m / grid.pixel_size)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    closed = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    closed = ndimage.binary_fill_holes(closed)
    labels, count = ndimage.label(closed)
    entities: list[dict[str, object]] = []
    pad = max(3, int(round(3.0 * settings.smooth_sigma_m / grid.pixel_size)) + 2)
    for identifier, component_slice in enumerate(ndimage.find_objects(labels), start=1):
        if component_slice is None:
            continue
        region = labels[component_slice] == identifier
        area = float(region.sum()) * grid.pixel_size**2
        if area < settings.min_area_m2:
            continue
        padded = np.pad(region.astype(np.float32), pad)
        smoothed = ndimage.gaussian_filter(padded, sigma=max(0.5, settings.smooth_sigma_m / grid.pixel_size))
        contours = find_contours(smoothed, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        row0, col0 = component_slice[0].start - pad, component_slice[1].start - pad
        points = [grid.pixel_to_local(float(r) + row0, float(c) + col0) for r, c in contour]
        polygon = Polygon(points).buffer(0)
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            continue
        polygon = polygon.simplify(settings.simplify_m, preserve_topology=True)
        ring = _chaikin_ring(list(polygon.exterior.coords)[:-1], settings.chaikin_iterations)
        shape = Polygon(ring)
        if not shape.is_valid or shape.area < 0.5 * settings.min_area_m2:
            continue
        height = float(np.percentile(heights[component_slice][region & np.isfinite(heights[component_slice])], 90))
        entities.append({"type": "polyline", "points": [[float(x), float(y)] for x, y in ring], "closed": True, "holes": [], "_hedge": True, "_height_m": height})
        entities.append({"type": "text", "text": f"Gebüsch h={height:.1f} m", "point": list(shape.representative_point().coords[0]), "height": 0.4, "rotation": 0.0, "layer_role": "hedge_label"})
    return entities


def _chaikin_ring(points: list[tuple[float, float]], iterations: int) -> list[tuple[float, float]]:
    ring = [np.asarray(point, dtype=float) for point in points]
    for _ in range(max(0, iterations)):
        refined: list[np.ndarray] = []
        for index, first in enumerate(ring):
            second = ring[(index + 1) % len(ring)]
            refined.extend((first * 0.75 + second * 0.25, first * 0.25 + second * 0.75))
        ring = refined
    return [(float(point[0]), float(point[1])) for point in ring]


def max_turning_angle_deg(points: list[list[float]]) -> float:
    """Largest direction change (degrees) between consecutive edges of a closed ring."""
    array = np.asarray(points, dtype=float)
    edges = np.roll(array, -1, axis=0) - array
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    turn = np.abs((np.roll(angles, -1) - angles + math.pi) % (2.0 * math.pi) - math.pi)
    return float(np.degrees(turn.max()))
