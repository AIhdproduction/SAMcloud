"""DGM spot elevations: crosses with absolute-height text on S-ELEV."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class SpotSettings:
    """Spot-elevation selection (metres)."""

    grid_spacing_m: float = 17.5
    min_spacing_m: float = 4.5
    curb_spacing_m: float = 15.0
    cross_size_m: float = 0.6
    text_height_m: float = 0.7
    max_spots: int = 4000


@dataclass(frozen=True, slots=True)
class Spot:
    """One selected spot: ``kind`` is manhole, building_corner, curb or grid."""

    kind: str
    x: float
    y: float


def ground_grid_points(bounds: tuple[float, float, float, float], spacing: float, is_ground: Callable[[float, float], bool]) -> list[tuple[float, float]]:
    """Regular lattice (``spacing`` apart, centred in the bounds) restricted to ground positions."""
    min_x, min_y, max_x, max_y = bounds
    if spacing <= 0.0 or max_x <= min_x or max_y <= min_y:
        return []
    columns = max(1, int(math.floor((max_x - min_x) / spacing)))
    rows = max(1, int(math.floor((max_y - min_y) / spacing)))
    x0 = min_x + ((max_x - min_x) - (columns - 1) * spacing) / 2.0
    y0 = min_y + ((max_y - min_y) - (rows - 1) * spacing) / 2.0
    return [(x0 + column * spacing, y0 + row * spacing) for row in range(rows) for column in range(columns) if is_ground(x0 + column * spacing, y0 + row * spacing)]


def points_along(coords: Iterable[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
    """Start point, end point and a point every ``spacing`` metres along a polyline."""
    array = np.asarray(list(coords), dtype=float)[:, :2]
    if len(array) < 2:
        return [] if not len(array) else [(float(array[0, 0]), float(array[0, 1]))]
    steps = np.hypot(*np.diff(array, axis=0).T)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    length = float(cumulative[-1])
    distances = list(np.arange(0.0, length, spacing)) + [length]
    output = []
    for distance in distances:
        index = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(array) - 2)
        fraction = (distance - cumulative[index]) / max(float(steps[index]), 1e-12)
        point = array[index] * (1.0 - fraction) + array[index + 1] * fraction
        output.append((float(point[0]), float(point[1])))
    return output


def select_spots(groups: list[tuple[str, list[tuple[float, float]]]], settings: SpotSettings = SpotSettings()) -> list[Spot]:
    """Greedy selection in group priority order; no two spots closer than ``min_spacing_m``."""
    cell = max(settings.min_spacing_m, 1e-6)
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    selected: list[Spot] = []
    for kind, candidates in groups:
        for x, y in candidates:
            if len(selected) >= settings.max_spots:
                return selected
            key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
            clash = any(math.hypot(x - ox, y - oy) < settings.min_spacing_m for dx in (-1, 0, 1) for dy in (-1, 0, 1) for ox, oy in buckets.get((key[0] + dx, key[1] + dy), ()))
            if clash:
                continue
            buckets.setdefault(key, []).append((x, y))
            selected.append(Spot(kind, float(x), float(y)))
    return selected


def format_spot_height(height: float) -> str:
    """Absolute height with sign and two decimals, e.g. ``+452.35``."""
    return f"{height:+.2f}"


def spot_entities(spots: list[Spot], local_heights: list[float], absolute_heights: list[float], settings: SpotSettings = SpotSettings()) -> list[dict[str, object]]:
    """A cross (two 3D lines at the local ground height) and an absolute-height TEXT per spot."""
    half = settings.cross_size_m / 2.0
    entities: list[dict[str, object]] = []
    for spot, local_z, absolute_z in zip(spots, local_heights, absolute_heights, strict=True):
        if not (math.isfinite(local_z) and math.isfinite(absolute_z)):
            continue
        entities.append({"type": "line", "start": [spot.x - half, spot.y, local_z], "end": [spot.x + half, spot.y, local_z], "_spot": spot.kind})
        entities.append({"type": "line", "start": [spot.x, spot.y - half, local_z], "end": [spot.x, spot.y + half, local_z], "_spot": spot.kind})
        entities.append({"type": "text", "text": format_spot_height(absolute_z), "point": [spot.x + half + 0.2, spot.y, local_z], "height": settings.text_height_m, "rotation": 0.0, "align": "MIDDLE_LEFT", "layer_role": "spot_height", "_spot": spot.kind})
    return entities
