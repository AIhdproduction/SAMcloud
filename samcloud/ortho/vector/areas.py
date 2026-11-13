"""Raster area cleaning and coverage-preserving planar partition construction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from shapely import coverage_is_valid, coverage_simplify, coverage_union_all
from shapely.geometry import Polygon

from .raster_grid import RasterGrid, mask_to_polygons


@dataclass(frozen=True, slots=True)
class AreaParams:
    """Metric morphology and minimum-area settings for one area class."""

    open_radius: float = 0.0
    close_radius: float = 0.0
    min_area: float = 0.0
    courtyard_min_area: float = 0.0


def clean_label_raster(labels: np.ndarray, *, class_params: dict[int, AreaParams], grid: RasterGrid) -> np.ndarray:
    """Clean classes independently then assign removed covered pixels to nearest labels.

    Small enclosed holes are filled only where the input is unlabelled.  A small
    neighbouring class inside another class is therefore preserved regardless of
    numeric class-code order; if that neighbour is subsequently removed as a
    blob, normal nearest-label reassignment handles its pixels.
    """
    source = np.asarray(labels, dtype=np.uint8)
    if source.ndim != 2 or source.shape != (grid.height, grid.width):
        raise ValueError("labels shape must match RasterGrid")
    covered = source != 0
    result = np.zeros_like(source)
    for class_code, params in sorted(class_params.items()):
        if not 0 < int(class_code) < 256:
            raise ValueError("class codes must be uint8 values excluding zero")
        mask = source == class_code
        cleaned = _morphology(mask, params, grid.pixel_size)
        cleaned = _remove_small_components(cleaned, params.min_area, grid.pixel_size)
        cleaned = _fill_small_holes(cleaned, params.courtyard_min_area, grid.pixel_size, source == 0)
        assignable = cleaned & (result == 0)
        result[assignable] = np.uint8(class_code)
    missing = covered & (result == 0)
    if missing.any() and (result != 0).any():
        _distances, indices = ndimage.distance_transform_edt(result == 0, return_indices=True)
        result[missing] = result[tuple(indices[:, missing])]
    elif missing.any():
        result[missing] = source[missing]
    return result


def build_partition(
    labels: np.ndarray,
    grid: RasterGrid,
    *,
    class_to_layer: dict[int, str],
    tolerances: dict[str, float],
    partition_tolerance: float | None = None,
) -> dict[str, list[Polygon]]:
    """Polygonise a shared label raster and simplify its coverage once.

    Shared borders are simplified by one configurable ``partition_tolerance``
    with ``simplify_boundary=False``.  Layer tolerances are intentionally limited
    to free outer boundaries: changing one side of an inter-layer border with a
    per-layer tolerance creates a gap or overlap.  The current polygon ownership
    representation cannot replace only a subset of a polygon ring safely, so
    free-boundary simplification is applied only to isolated layer coverages;
    mixed coverages retain the valid shared partition tolerance.
    """
    source = np.asarray(labels, dtype=np.uint8)
    if source.shape != (grid.height, grid.width):
        raise ValueError("labels shape must match RasterGrid")
    geometries: list[Polygon] = []
    owners: list[str] = []
    for class_code, layer in sorted(class_to_layer.items()):
        for polygon in mask_to_polygons(source == class_code, grid):
            geometries.append(polygon)
            owners.append(layer)
    if not geometries:
        return {layer: [] for layer in dict.fromkeys(class_to_layer.values())}
    positive = [float(tolerances.get(layer, 0.0)) for layer in owners if float(tolerances.get(layer, 0.0)) > 0.0]
    tolerance = float(partition_tolerance) if partition_tolerance is not None else (min(positive) if positive else 0.0)
    if tolerance < 0.0:
        raise ValueError("partition_tolerance must be non-negative")
    simplified = coverage_simplify(geometries, tolerance, simplify_boundary=False)
    polygons = [polygon for polygon in np.asarray(simplified, dtype=object).tolist()]
    if not coverage_is_valid(polygons):
        raise ValueError("coverage_simplify returned an invalid coverage")
    partition: dict[str, list[Polygon]] = {layer: [] for layer in dict.fromkeys(class_to_layer.values())}
    for layer, polygon in zip(owners, polygons, strict=True):
        partition[layer].append(polygon)
    return _simplify_isolated_layer_boundaries(partition, tolerances)


def partition_entities(partition: dict[str, list[Polygon]]) -> dict[str, list[dict[str, object]]]:
    """Convert partition polygons to closed neutral polyline entities with holes."""
    output: dict[str, list[dict[str, object]]] = {}
    for layer, polygons in partition.items():
        entities: list[dict[str, object]] = []
        for polygon in polygons:
            entities.append(
                {
                    "type": "polyline",
                    "points": _ring_points(polygon.exterior.coords),
                    "closed": True,
                    "holes": [_ring_points(ring.coords) for ring in polygon.interiors],
                }
            )
        output[layer] = entities
    return output


def _morphology(mask: np.ndarray, params: AreaParams, pixel_size: float) -> np.ndarray:
    result = mask.astype(np.uint8)
    for radius, operation in ((params.open_radius, cv2.MORPH_OPEN), (params.close_radius, cv2.MORPH_CLOSE)):
        if radius > 0.0:
            pixels = max(1, int(round(radius / pixel_size)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
            result = cv2.morphologyEx(result, operation, kernel)
    return result.astype(bool)


def _remove_small_components(mask: np.ndarray, minimum_area: float, pixel_size: float) -> np.ndarray:
    if minimum_area <= 0.0:
        return mask
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    result = mask.copy()
    small = stats[:, cv2.CC_STAT_AREA] * pixel_size**2 < minimum_area
    small[0] = False
    result[small[labels]] = False
    return result


def _fill_small_holes(mask: np.ndarray, courtyard_min_area: float, pixel_size: float, fillable: np.ndarray) -> np.ndarray:
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return mask
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    result = mask.copy()
    fill_lut = stats[:, cv2.CC_STAT_AREA] * pixel_size**2 < courtyard_min_area
    fill_lut[0] = False
    result[fill_lut[labels] & fillable] = True
    return result


def _ring_points(coordinates: object) -> list[list[float]]:
    points = list(coordinates)[:-1]
    return [[float(point[0]), float(point[1])] for point in points]


def _simplify_isolated_layer_boundaries(
    partition: dict[str, list[Polygon]], tolerances: dict[str, float]
) -> dict[str, list[Polygon]]:
    """Apply layer tolerances where a layer has no shared coverage boundary."""
    layer_unions = {
        layer: coverage_union_all(polygons)
        for layer, polygons in partition.items()
        if polygons
    }
    for layer, polygons in partition.items():
        tolerance = float(tolerances.get(layer, 0.0))
        if tolerance <= 0.0 or not polygons:
            continue
        layer_union = layer_unions[layer]
        others = [other_union for other_layer, other_union in layer_unions.items() if other_layer != layer]
        if others and layer_union.boundary.intersection(coverage_union_all(others).boundary).length > 1e-9:
            continue
        candidate = coverage_simplify(polygons, tolerance, simplify_boundary=True)
        candidate_polygons = [polygon for polygon in np.asarray(candidate, dtype=object).tolist()]
        replacement = [polygon for other_layer, values in partition.items() for polygon in (candidate_polygons if other_layer == layer else values)]
        if coverage_is_valid(replacement):
            partition[layer] = candidate_polygons
    return partition
