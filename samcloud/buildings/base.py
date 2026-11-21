"""Building base: wall bottoms that follow the terrain."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

DEFAULT_BASE = {"mode": "terrain", "embed_m": 0.02, "max_edge_m": 2.0, "follow_tol_m": 0.03}
_INTERIOR_GRID_M = 1.0


def base_settings(settings: dict | None) -> dict:
    """The ``buildings.base`` block with defaults filled in."""
    block = dict((settings or {}).get("base", {}) or {})
    return {**DEFAULT_BASE, **block}


def _sample(sampler, x: float, y: float) -> float:
    try:
        return float(sampler(x, y))
    except TypeError:
        return float(sampler((x, y)))


def densify_ring(ring: np.ndarray, max_edge_m: float | None) -> np.ndarray:
    """Insert equally spaced vertices so that no edge of the closed ring exceeds ``max_edge_m``.

    ``ring`` is the open ring (first vertex not repeated); the result is open as well.
    """
    ring = np.asarray(ring, dtype=float)
    if max_edge_m is None or max_edge_m <= 0.0 or len(ring) < 2:
        return ring
    result: list[np.ndarray] = []
    for index in range(len(ring)):
        start, end = ring[index], ring[(index + 1) % len(ring)]
        pieces = max(1, int(math.ceil(float(np.linalg.norm(end - start)) / max_edge_m - 1e-9)))
        for step in range(pieces):
            result.append(start + (end - start) * (step / pieces))
    return np.asarray(result, dtype=float)


def _adaptive_nodes(start: np.ndarray, end: np.ndarray, ground, max_edge_m: float, tol_m: float, min_segment_m: float) -> list[float]:
    """Parameters t in (0, 1) at which the straight wall bottom start->end must get a vertex.

    A segment is split at its middle while the terrain deviates more than ``tol_m`` from the chord
    (checked at samples at most ``max_edge_m / 4`` apart), so flat ground adds no vertices.
    """
    length = float(np.linalg.norm(end - start))
    if length <= min_segment_m:
        return []
    count = max(3, int(math.ceil(length / (max_edge_m / 4.0))))
    ts = (np.arange(count) + 1.0) / (count + 1.0)
    z0, z1 = _sample(ground, float(start[0]), float(start[1])), _sample(ground, float(end[0]), float(end[1]))
    deviation = 0.0
    for t in ts:
        point = start + (end - start) * t
        deviation = max(deviation, abs(_sample(ground, float(point[0]), float(point[1])) - (z0 + (z1 - z0) * t)))
    if deviation <= tol_m:
        return []
    middle = (start + end) / 2.0
    left = [t / 2.0 for t in _adaptive_nodes(start, middle, ground, max_edge_m, tol_m, min_segment_m)]
    right = [0.5 + t / 2.0 for t in _adaptive_nodes(middle, end, ground, max_edge_m, tol_m, min_segment_m)]
    return left + [0.5] + right


def densify_footprint(footprint: Polygon, max_edge_m: float | None, ground=None, tol_m: float = 0.03, min_segment_m: float = 0.25) -> Polygon:
    """Footprint with extra collinear exterior vertices where the terrain bends away from the chord.

    Without a ``ground`` callable every edge is simply subdivided to at most ``max_edge_m``.
    """
    if max_edge_m is None or max_edge_m <= 0.0:
        return footprint
    ring = np.asarray(footprint.exterior.coords, dtype=float)[:-1]
    if ground is None:
        dense = densify_ring(ring, max_edge_m)
    else:
        points: list[np.ndarray] = []
        for index in range(len(ring)):
            start, end = ring[index], ring[(index + 1) % len(ring)]
            points.append(start)
            points.extend(start + (end - start) * t for t in _adaptive_nodes(start, end, ground, max_edge_m, tol_m, min_segment_m))
        dense = np.asarray(points, dtype=float)
    if len(dense) == len(ring):
        return footprint
    return Polygon(dense)


def terrain_under(sampler, footprint: Polygon, *, spacing: float = _INTERIOR_GRID_M, max_edge_m: float = 0.5) -> np.ndarray:
    """Terrain heights along the footprint outline and on a coarse interior grid."""
    ring = densify_ring(np.asarray(footprint.exterior.coords, dtype=float)[:-1], max_edge_m)
    points = [tuple(p) for p in ring]
    min_x, min_y, max_x, max_y = footprint.bounds
    grid_x, grid_y = np.meshgrid(np.arange(min_x + spacing / 2.0, max_x, spacing), np.arange(min_y + spacing / 2.0, max_y, spacing))
    inside = contains_xy(footprint, grid_x.ravel(), grid_y.ravel())
    points.extend(zip(grid_x.ravel()[inside].tolist(), grid_y.ravel()[inside].tolist(), strict=True))
    values = np.array([_sample(sampler, x, y) for x, y in points], dtype=float)
    return values[np.isfinite(values)]


def make_ground_function(sampler, footprint: Polygon, settings: dict | None = None) -> Callable[[float, float], float]:
    """Return ``ground(x, y)``: the z of the wall bottom / floor of a building.

    ``terrain``: terrain height at the point minus ``embed_m``.
    ``flat_min``: the lowest terrain height under the footprint minus ``embed_m`` everywhere.
    """
    cfg = base_settings(settings)
    embed = float(cfg["embed_m"])
    if sampler is None:
        return lambda _x, _y: -embed
    if str(cfg["mode"]) == "flat_min":
        values = terrain_under(sampler, footprint)
        level = (float(values.min()) if len(values) else 0.0) - embed
        return lambda _x, _y: level

    def ground(x: float, y: float) -> float:
        value = _sample(sampler, x, y)
        return (value if math.isfinite(value) else 0.0) - embed

    ground.follow_tol_m = float(cfg["follow_tol_m"])  # type: ignore[attr-defined]
    return ground


def make_tin_sampler(tin, fallback=None) -> Callable | None:
    """Terrain height sampler from a labelled TIN (linear interpolation); ``fallback`` fills gaps.

    Returns ``None`` when the TIN is unusable.  The sampler accepts ``(x, y)`` or an ``(N, 2)`` array.
    """
    try:
        from scipy.interpolate import LinearNDInterpolator

        vertices = np.asarray(tin.vertices, dtype=float)
        triangles = np.asarray(tin.triangles, dtype=int)
        if len(vertices) < 3 or len(triangles) == 0:
            return None
        used = np.unique(triangles.ravel())
        interpolator = LinearNDInterpolator(vertices[used, :2], vertices[used, 2])
    except Exception as exc:  # noqa: BLE001 - the DTM is a complete fallback
        logger.warning("terrain TIN sampler unavailable: %s", exc)
        return None

    def sample(xy, y=None):
        scalar = y is not None and np.asarray(xy).ndim == 0 and np.asarray(y).ndim == 0
        values = np.asarray((xy, y) if y is not None else xy, dtype=float).reshape(-1, 2)
        z = np.asarray(interpolator(values), dtype=float).reshape(-1)
        missing = ~np.isfinite(z)
        if missing.any() and fallback is not None:
            for index in np.flatnonzero(missing):
                z[index] = _sample(fallback, float(values[index, 0]), float(values[index, 1]))
        return float(z[0]) if scalar else z

    return sample
