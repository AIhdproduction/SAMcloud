"""Raster-to-vector conversion with valid polygon output."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.validation import make_valid


@dataclass(frozen=True, slots=True)
class Raster:
    mask: np.ndarray
    origin: tuple[float, float]
    cell_size: float


def rasterize_points(points: np.ndarray, cell_size: float, *, closing_kernel: int = 3) -> Raster:
    """Rasterize XY points into a compact binary mask in local metres."""
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return Raster(np.zeros((0, 0), dtype=np.uint8), (0.0, 0.0), cell_size)
    if values.ndim != 2 or values.shape[1] < 2 or cell_size <= 0:
        raise ValueError("Points must be an N by 2-or-more array and cell_size must be positive")
    minimum = values[:, :2].min(axis=0)
    indices = np.floor((values[:, :2] - minimum) / cell_size + 0.5).astype(np.int64)
    shape = tuple((indices.max(axis=0)[::-1] + 3).tolist())
    mask = np.zeros(shape, dtype=np.uint8)
    mask[indices[:, 1] + 1, indices[:, 0] + 1] = 255
    if closing_kernel > 1:
        kernel_size = max(closing_kernel, _sampling_kernel(values[:, :2], cell_size))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return Raster(mask, (float(minimum[0] - cell_size), float(minimum[1] - cell_size)), cell_size)


def polygons_from_points(points: np.ndarray, cell_size: float, *, closing_kernel: int = 3, simplify_tolerance: float = 0.0, min_area: float = 0.0) -> list[Polygon]:
    """Return cleaned polygons, retaining contour hierarchy holes."""
    raster = rasterize_points(points, cell_size, closing_kernel=closing_kernel)
    return polygons_from_raster(raster, simplify_tolerance=simplify_tolerance, min_area=min_area)


def polygons_from_raster(raster: Raster, *, simplify_tolerance: float = 0.0, min_area: float = 0.0) -> list[Polygon]:
    """Convert CCOMP contours to valid Shapely polygons in local metres."""
    if raster.mask.size == 0:
        return []
    contours, hierarchy = cv2.findContours(raster.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    result: list[Polygon] = []
    entries = hierarchy[0]
    for index, contour in enumerate(contours):
        if entries[index][3] != -1:
            continue
        exterior = _contour_coordinates(contour, raster)
        if len(exterior) < 3:
            continue
        holes = [_contour_coordinates(candidate, raster) for child, candidate in enumerate(contours) if entries[child][3] == index]
        holes = [hole for hole in holes if len(hole) >= 3]
        polygon = Polygon(exterior, holes)
        if simplify_tolerance:
            polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)
        cleaned = make_valid(polygon)
        candidates = [cleaned] if cleaned.geom_type == "Polygon" else list(getattr(cleaned, "geoms", ()))
        for candidate in candidates:
            if candidate.geom_type == "Polygon" and candidate.area >= min_area:
                result.append(candidate)
    return result


def _contour_coordinates(contour: np.ndarray, raster: Raster) -> list[tuple[float, float]]:
    values = contour.reshape(-1, 2).astype(np.float64)
    return [(raster.origin[0] + x * raster.cell_size, raster.origin[1] + y * raster.cell_size) for x, y in values]


def _sampling_kernel(points: np.ndarray, cell_size: float) -> int:
    """Estimate a conservative odd kernel from point spacing for sparse scans."""
    if len(points) < 3:
        return 3
    sample = points if len(points) <= 20_000 else points[np.linspace(0, len(points) - 1, 20_000, dtype=int)]
    distances, _ = cKDTree(sample).query(sample, k=2)
    spacing = float(np.median(distances[:, 1]))
    size = min(101, max(3, int(np.ceil(spacing / cell_size)) * 2 + 1))
    return size if size % 2 else size + 1
