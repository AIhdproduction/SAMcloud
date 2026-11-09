"""Metric scale state and explicit scale measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class ScaleState:
    """The explicit metric scale applied to local coordinates."""

    factor: float = 1.0
    source: Literal["none", "numeric", "two_point"] = "none"
    status_text: str = "No scale adjustment applied."

    def __post_init__(self) -> None:
        if not np.isfinite(self.factor) or self.factor <= 0:
            raise ValueError("Scale factor must be a finite positive number")


def numeric_scale(factor: float) -> ScaleState:
    """Create a scale state supplied directly by the user."""
    return ScaleState(float(factor), "numeric", f"Applied numeric scale factor {factor:.8g}.")


def two_point_scale(point_a: np.ndarray, point_b: np.ndarray, real_distance: float) -> ScaleState:
    """Compute scale from two picked points and their known real-world distance."""
    first = np.asarray(point_a, dtype=np.float64)
    second = np.asarray(point_b, dtype=np.float64)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError("Scale measurement points must each have exactly three coordinates")
    measured_distance = float(np.linalg.norm(second - first))
    if not np.isfinite(real_distance) or real_distance <= 0:
        raise ValueError("Real distance must be a finite positive number")
    if measured_distance <= np.finfo(float).eps:
        raise ValueError("Picked scale points must not coincide")
    factor = float(real_distance) / measured_distance
    return ScaleState(factor, "two_point", f"Measured {measured_distance:.4f} units; applied scale factor {factor:.8g}.")
