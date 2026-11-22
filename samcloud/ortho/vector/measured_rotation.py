"""When does an object keep the rotation it was measured with, and when does it snap to the wall axes?

A rotation is measured only with clear support: the object is elongated, its evidence fills the turned rectangle
(residual), the rectangle's sides are covered by the evidence, and the turned rectangle explains the evidence better
than the one along the walls.  Anything weaker snaps to the wall axes.  All lengths are metric.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon, box


@dataclass(frozen=True, slots=True)
class RotationSettings:
    min_deviation_deg: float = 5.0  # less is the noise of a traced outline
    min_elongation: float = 1.25  # a square has no direction
    min_fill: float = 0.88  # evidence area / rectangle area
    max_residual_m: float = 0.04  # p90 distance between evidence outline and rectangle
    min_gain: float = 0.05  # the wall-axis rectangle fills at least this much worse
    min_points: int = 40
    side_coverage: float = 0.80  # share of the rectangle's sides with evidence points beside them
    side_reach_m: float = 0.05


DEFAULT_ROTATION = RotationSettings()


def axis_deviation_deg(angle: float, theta: float) -> float:
    """Degrees between a direction and the nearest wall axis, 0..45."""
    return abs(((math.degrees(angle - theta) + 45.0) % 90.0) - 45.0)


def _outline_samples(polygon: Polygon, step: float) -> np.ndarray:
    ring = polygon.exterior
    return shapely.get_coordinates(shapely.line_interpolate_point(ring, np.arange(0.0, ring.length, step)))


def clear_rotation(evidence: Any, theta: float, units_per_metre: float, settings: RotationSettings = DEFAULT_ROTATION) -> Polygon | None:
    """The turned rectangle of a footprint whose rotation is clearly measured, else None."""
    upm = max(float(units_per_metre), 1e-9)
    if not isinstance(evidence, Polygon) or evidence.is_empty or not evidence.is_valid:
        return None
    rectangle = evidence.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon) or rectangle.area <= 0.0:
        return None
    corners = np.asarray(rectangle.exterior.coords[:4], dtype=float)
    sides = [corners[1] - corners[0], corners[2] - corners[1]]
    lengths = [float(np.hypot(*side)) for side in sides]
    edge = sides[int(np.argmax(lengths))]
    if max(lengths) < settings.min_elongation * min(lengths):
        return None
    if axis_deviation_deg(math.atan2(edge[1], edge[0]), theta) < settings.min_deviation_deg:
        return None
    fill = evidence.area / rectangle.area
    if fill < settings.min_fill:
        return None
    straight = box(*affinity.rotate(evidence, -theta, origin=(0.0, 0.0), use_radians=True).bounds)
    if fill - evidence.area / straight.area < settings.min_gain:
        return None
    step = 0.02 * upm
    for source, target in ((evidence, rectangle), (rectangle, evidence)):
        samples = _outline_samples(source, step)
        distances = shapely.distance(target.exterior, shapely.points(samples))
        if float(np.percentile(distances, 90)) > settings.max_residual_m * upm:
            return None
    return rectangle


def clear_point_rotation(points: np.ndarray, theta: float, units_per_metre: float, settings: RotationSettings = DEFAULT_ROTATION) -> Polygon | None:
    """``clear_rotation`` for label points (an outline or a filled patch): their sides must be covered by points."""
    upm = max(float(units_per_metre), 1e-9)
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(values) < settings.min_points:
        return None
    hull = MultiPoint(values).convex_hull
    rectangle = clear_rotation(hull, theta, upm, settings)
    if rectangle is None:
        return None
    samples = _outline_samples(rectangle, 0.02 * upm)
    nearest = np.min(np.hypot(samples[:, None, 0] - values[None, :, 0], samples[:, None, 1] - values[None, :, 1]), axis=1)
    return rectangle if float(np.mean(nearest <= settings.side_reach_m * upm)) >= settings.side_coverage else None
