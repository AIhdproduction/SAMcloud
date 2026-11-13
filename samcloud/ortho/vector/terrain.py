"""Terrain sampling and 3D draping of 2D lines (breaklines) on a DTM raster."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Polygon

from .raster_grid import RasterGrid


class DtmSampler:
    """Bilinear DTM lookup in local coordinates; gaps are filled from the nearest valid cell."""

    def __init__(self, dtm: np.ndarray, grid: RasterGrid) -> None:
        array = np.asarray(dtm, dtype=np.float64)
        if array.shape != (grid.height, grid.width):
            raise ValueError("dtm shape must match RasterGrid")
        self.grid = grid
        self.valid_fraction = float(np.isfinite(array).mean()) if array.size else 0.0
        if np.isfinite(array).any() and not np.isfinite(array).all():
            nearest = ndimage.distance_transform_edt(~np.isfinite(array), return_distances=False, return_indices=True)
            array = array[tuple(nearest)]
        self._array = array
        self._measured = np.isfinite(np.asarray(dtm, dtype=np.float64))

    def sample(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """Local ground height at (x, y); NaN when the DTM has no data at all."""
        xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if not np.isfinite(self._array).any():
            return np.full(xs.shape, np.nan)
        rows = (self.grid.max_y - ys) / self.grid.pixel_size - 0.5
        cols = (xs - self.grid.min_x) / self.grid.pixel_size - 0.5
        return ndimage.map_coordinates(self._array, [rows, cols], order=1, mode="nearest")

    def is_measured(self, x: float, y: float) -> bool:
        """True where the DTM cell at (x, y) holds a real ground measurement (not a filled gap)."""
        row = int(round((self.grid.max_y - y) / self.grid.pixel_size - 0.5))
        col = int(round((x - self.grid.min_x) / self.grid.pixel_size - 0.5))
        return bool(0 <= row < self.grid.height and 0 <= col < self.grid.width and self._measured[row, col])


def drape_coordinates(coords: list[tuple[float, float]] | np.ndarray, sampler: DtmSampler, *, max_segment: float = 2.0) -> list[list[float]]:
    """3D vertices of a 2D line: segments longer than ``max_segment`` are subdivided, z comes from the DTM."""
    line = LineString(coords)
    if max_segment > 0.0 and line.length > max_segment:
        line = line.segmentize(max_segment)
    points = np.asarray(line.coords, dtype=np.float64)
    heights = sampler.sample(points[:, 0], points[:, 1])
    return [[float(x), float(y), float(z)] for (x, y), z in zip(points[:, :2], heights, strict=True)]


def drape_outline_heights(entity: dict[str, object], sampler: DtmSampler) -> None:
    """Attach DTM heights to a closed 2D outline entity: ``z`` for the exterior ring, ``hole_z`` for holes.

    The 2D ``points`` / ``holes`` are left untouched; the DXF writer turns an entity with ``z`` into one
    continuous 3D polyline per ring.
    """
    points = np.asarray(entity["points"], dtype=np.float64)
    heights = sampler.sample(points[:, 0], points[:, 1])
    if not np.isfinite(heights).all():
        return  # no terrain data: never invent heights, the outline stays 2D
    entity["z"] = [float(value) for value in heights]
    hole_heights = []
    for hole in entity.get("holes", []) or []:  # type: ignore[union-attr]
        ring = np.asarray(hole, dtype=np.float64)
        hole_heights.append([float(value) for value in sampler.sample(ring[:, 0], ring[:, 1])])
    entity["hole_z"] = hole_heights
    entity["_breakline"] = True


def polygon_from_points(points: list[list[float]]) -> Polygon:
    return Polygon(points)
