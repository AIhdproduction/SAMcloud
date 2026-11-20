"""Upper kitchen cabinets (A-OBEN) from the 1.80-2.20 m band above a base-cabinet run.

The base run (A-EINB) stays a solid footprint.  Wall units hanging above it are
only seen as vertical surfaces in the upper z band, never connected to the
floor.  Where such surfaces stand off the wall above a run, a dashed outline of
the hanging unit (standard depth 0.35 m) is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .raster_grid import RasterGrid


@dataclass(frozen=True, slots=True)
class UpperCabinetSettings:
    """Metric thresholds (scaled by units per metre at use)."""

    band_min_m: float = 1.80
    band_max_m: float = 2.20
    depth_m: float = 0.35
    min_length_m: float = 0.40
    min_coverage: float = 0.50
    min_run_depth_m: float = 0.25  # a base run thinner than this is a sliver, not a counter: no unit hangs above it
    gap_fill_m: float = 0.15
    bins: int = 16


def upper_band_mask(bitmask: np.ndarray, storey_height: float, units_per_metre: float, settings: UpperCabinetSettings = UpperCabinetSettings()) -> np.ndarray:
    """Cells with any vertical-occupancy bin whose centre lies in the upper z band."""
    bits = np.asarray(bitmask, dtype=np.uint32)
    if storey_height <= 0.0:
        return np.zeros(bits.shape, dtype=bool)
    low = settings.band_min_m * units_per_metre / storey_height * settings.bins
    high = settings.band_max_m * units_per_metre / storey_height * settings.bins
    selected = 0
    for index in range(settings.bins):
        if low <= index + 0.5 <= high:
            selected |= 1 << index
    return (bits & np.uint32(selected)) != 0


def _polygon_raster(polygon: Polygon, grid: RasterGrid) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    points = np.asarray([[round(grid.local_to_pixel(x, y)[1]), round(grid.local_to_pixel(x, y)[0])] for x, y in polygon.exterior.coords], dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def detect_upper_cabinets(
    base_polygons: Iterable[Polygon],
    upper_mask: np.ndarray,
    floor_connected: np.ndarray,
    wall_evidence: np.ndarray,
    grid: RasterGrid,
    wall_lines: Iterable[LineString],
    units_per_metre: float,
    settings: UpperCabinetSettings = UpperCabinetSettings(),
) -> list[Polygon]:
    """Return hanging-cabinet footprints above each base run that shows upper-band evidence."""
    upm = max(float(units_per_metre), 1e-9)
    walls = list(wall_lines)
    # Upper fronts can project into a cell that also has lower cabinet evidence.
    # Reject only an entirely floor-connected candidate (a tall unit), rather
    # than masking such cells before their upper-band evidence is evaluated.
    candidate = np.asarray(upper_mask, dtype=bool) & ~np.asarray(wall_evidence, dtype=bool)
    grounded = np.asarray(floor_connected, dtype=bool)
    output: list[Polygon] = []
    for base in base_polygons:
        rectangle = base.minimum_rotated_rectangle
        if rectangle.is_empty or not walls:
            continue
        corners = np.asarray(rectangle.exterior.coords[:4], dtype=float)
        edges = [corners[(index + 1) % 4] - corners[index] for index in range(4)]
        long_index = int(np.argmax([float(np.hypot(*edges[0])), float(np.hypot(*edges[1]))]))
        run_length = float(np.hypot(*edges[long_index]))
        run_depth = float(np.hypot(*edges[1 - long_index]))
        if run_length < settings.min_length_m * upm or run_depth < settings.min_run_depth_m * upm:
            continue
        u = edges[long_index] / run_length
        v = np.asarray((-u[1], u[0]))
        long_edges = [(corners[long_index], corners[(long_index + 1) % 4]), (corners[(long_index + 2) % 4], corners[(long_index + 3) % 4])]
        midpoints = [(first + second) / 2.0 for first, second in long_edges]
        back = min(midpoints, key=lambda point: min(wall.distance(Point(float(point[0]), float(point[1]))) for wall in walls))
        centre = np.asarray(rectangle.centroid.coords[0], dtype=float)
        into = v if float(np.dot(centre - back, v)) > 0.0 else -v
        hits = candidate & _polygon_raster(rectangle.buffer(0.03 * upm), grid)
        rows, cols = np.nonzero(hits)
        if not len(rows):
            continue
        if float(grounded[rows, cols].mean()) >= 0.95:
            continue
        points = np.asarray([grid.pixel_to_local(float(r), float(c)) for r, c in zip(rows, cols, strict=True)])
        along = (points - back) @ u
        across = (points - back) @ into
        # Hanging units stand off the wall: ignore evidence hugging the wall plane.
        keep = across >= 0.05 * upm
        if not keep.any():
            continue
        along = along[keep]
        start = float(np.dot(corners[long_index] - back, u))
        end = float(np.dot(corners[(long_index + 1) % 4] - back, u))
        low, high = min(start, end), max(start, end)
        step = max(0.05 * upm, grid.pixel_size)
        count = max(1, int(math.ceil((high - low) / step)))
        covered = np.zeros(count, dtype=bool)
        covered[np.clip(((along - low) / step).astype(int), 0, count - 1)] = True
        if float(covered.mean()) < settings.min_coverage:
            continue
        gap = max(1, int(round(settings.gap_fill_m * upm / step)))
        filled = covered.copy()
        last = -10**9
        for index in range(count):
            if covered[index]:
                if 0 < index - last <= gap:
                    filled[last:index + 1] = True
                last = index
        depth = min(settings.depth_m * upm, run_depth)
        index = 0
        while index < count:
            if not filled[index]:
                index += 1
                continue
            stop = index
            while stop < count and filled[stop]:
                stop += 1
            if (stop - index) * step >= settings.min_length_m * upm:
                a0, a1 = low + index * step, min(high, low + stop * step)
                ring = [back + u * a0, back + u * a1, back + u * a1 + into * depth, back + u * a0 + into * depth]
                output.append(Polygon([tuple(float(c) for c in point) for point in ring]))
            index = stop
    return output
