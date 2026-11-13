"""North-up raster coordinates and lossless raster polygonisation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from shapely import make_valid
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import polygonize


@dataclass(frozen=True, slots=True)
class RasterGrid:
    """A north-up local-metre raster where row zero is located at ``max_y``."""

    min_x: float
    max_y: float
    pixel_size: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.pixel_size <= 0.0 or self.width <= 0 or self.height <= 0:
            raise ValueError("pixel_size, width and height must be positive")

    def pixel_to_local(self, row: float, col: float) -> tuple[float, float]:
        """Return the local coordinates of a pixel centre."""
        return (
            self.min_x + (float(col) + 0.5) * self.pixel_size,
            self.max_y - (float(row) + 0.5) * self.pixel_size,
        )

    def local_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Return fractional (row, col) coordinates measured at pixel centres."""
        return (
            (self.max_y - float(y)) / self.pixel_size - 0.5,
            (float(x) - self.min_x) / self.pixel_size - 0.5,
        )

    def transform(self) -> tuple[float, float, float, float, float, float]:
        """Return the GDAL-style affine tuple mapping (column, row) to metres."""
        return (self.pixel_size, 0.0, self.min_x, 0.0, -self.pixel_size, self.max_y)


def mask_to_polygons(mask: np.ndarray, grid: RasterGrid) -> list[Polygon]:
    """Convert a mask to valid, pixel-edge polygons in local metres.

    ``cv2.findContours`` with CCOMP supplies the component/hole hierarchy.  Its
    centreline vertices are converted into an exact grid-edge contour network:
    every foreground/background transition contributes one directed pixel edge,
    and polygonisation rebuilds the hierarchy with holes.  This rasterio-free
    approach preserves exact pixel area, including one-pixel components, and
    gives identically positioned borders when each class of a label raster is
    polygonised independently.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.shape != (grid.height, grid.width):
        raise ValueError("mask shape must match RasterGrid height and width")
    if not binary.any():
        return []
    contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    parents = hierarchy[0, :, 3]
    exterior_count = int(np.count_nonzero(parents == -1))
    hole_count = int(np.count_nonzero(parents != -1))
    if exterior_count == 0 or len(contours) != exterior_count + hole_count:
        return []
    return _pixel_edge_polygons(binary, grid)


def _pixel_edge_polygons(mask: np.ndarray, grid: RasterGrid) -> list[Polygon]:
    """Polygonise the exact directed pixel-edge network for a CCOMP mask."""
    padded = np.pad(mask, 1, constant_values=False)
    rows, cols = np.nonzero(mask)
    lines: list[LineString] = []
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        rr, cc = row + 1, col + 1
        if not padded[rr - 1, cc]:
            lines.append(LineString(((col, row), (col + 1, row))))
        if not padded[rr, cc + 1]:
            lines.append(LineString(((col + 1, row), (col + 1, row + 1))))
        if not padded[rr + 1, cc]:
            lines.append(LineString(((col + 1, row + 1), (col, row + 1))))
        if not padded[rr, cc - 1]:
            lines.append(LineString(((col, row + 1), (col, row))))
    pixel_polygons = polygonize(lines)
    result: list[Polygon] = []
    for pixel_polygon in pixel_polygons:
        point = pixel_polygon.representative_point()
        row, col = int(math.floor(point.y)), int(math.floor(point.x))
        if not (0 <= row < grid.height and 0 <= col < grid.width and mask[row, col]):
            continue
        exterior = [(grid.min_x + x * grid.pixel_size, grid.max_y - y * grid.pixel_size) for x, y in pixel_polygon.exterior.coords]
        holes = [[(grid.min_x + x * grid.pixel_size, grid.max_y - y * grid.pixel_size) for x, y in ring.coords] for ring in pixel_polygon.interiors]
        result.extend(_polygon_parts(make_valid(Polygon(exterior, holes))))
    return result


def _polygon_parts(geometry: object) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry] if not geometry.is_empty else []
    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms if not polygon.is_empty]
    if isinstance(geometry, GeometryCollection):
        result: list[Polygon] = []
        for member in geometry.geoms:
            result.extend(_polygon_parts(member))
        return result
    return []
