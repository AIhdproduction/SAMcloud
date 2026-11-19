"""Small 2D vector helpers (float64, pure Python, deterministic)."""

from __future__ import annotations

import math
from collections.abc import Sequence

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

PARALLEL_EPS = 1e-9
"""Sine of the angle below which two directions count as parallel."""


def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Vec2, b: Vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def add_scaled(a: Vec2, d: Vec2, s: float) -> Vec2:
    """Return ``a + s * d``."""
    return (a[0] + s * d[0], a[1] + s * d[1])


def left_normal(d: Vec2) -> Vec2:
    """Rotate a direction by +90 degrees."""
    return (-d[1], d[0])


def intersect_lines(p1: Vec2, d1: Vec2, p2: Vec2, d2: Vec2) -> Vec2 | None:
    """Intersect the lines ``p1 + a d1`` and ``p2 + b d2``; None when (nearly) parallel.

    The directions must be unit vectors so that the parallel test is a sine.
    """
    denominator = cross(d1, d2)
    if abs(denominator) < PARALLEL_EPS:
        return None
    a = cross(sub(p2, p1), d2) / denominator
    return add_scaled(p1, d1, a)


def signed_area(points: Sequence[Vec2]) -> float:
    """Shoelace area, positive for counter-clockwise rings."""
    terms = []
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        terms.append(x1 * y2 - x2 * y1)
    return math.fsum(terms) / 2.0


def ring_length(points: Sequence[Vec2]) -> float:
    """Perimeter of a closed ring given without repeating the first point."""
    count = len(points)
    return math.fsum(
        math.hypot(
            points[(i + 1) % count][0] - points[i][0], points[(i + 1) % count][1] - points[i][1]
        )
        for i in range(count)
    )


def oriented(points: Sequence[Vec2], counter_clockwise: bool) -> tuple[Vec2, ...]:
    """Return the ring in the requested orientation (input order kept when it already matches)."""
    ring = tuple(points)
    if (signed_area(ring) > 0.0) == counter_clockwise:
        return ring
    return tuple(reversed(ring))


def shifted(points: Sequence[Vec2], origin: Vec2) -> list[Vec2]:
    """Subtract a local origin from all points."""
    return [(x - origin[0], y - origin[1]) for x, y in points]


def unshifted(points: Sequence[Vec2], origin: Vec2) -> tuple[Vec2, ...]:
    """Add the local origin back."""
    return tuple((x + origin[0], y + origin[1]) for x, y in points)
