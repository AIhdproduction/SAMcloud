"""Fit quality of a furniture shape: distance of the instance points to its surface, coverage, volume."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from samcloud.bim.furniture_recon.geometry import Mesh, Solid, sample_shape, shape_volume


@dataclass(frozen=True, slots=True)
class FitMetrics:
    points: int
    mean_m: float
    p90_m: float
    coverage: float  # share of points within the coverage distance of the surface
    volume_m3: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "points": self.points, "mean_m": round(self.mean_m, 4), "p90_m": round(self.p90_m, 4),
            "coverage": round(self.coverage, 4), "volume_m3": round(self.volume_m3, 4),
        }


def evaluate(points: np.ndarray, solids: list[Solid], mesh: Mesh | None, coverage_distance: float = 0.03) -> FitMetrics | None:
    """Point-to-surface statistics of ``points`` (item frame) against the shape (surface sampled at ~1 cm)."""
    if not len(points):
        return None
    surface = sample_shape(solids, mesh)
    if not len(surface):
        return None
    distance, _ = cKDTree(surface).query(points, k=1)
    return FitMetrics(
        points=int(len(points)), mean_m=float(distance.mean()), p90_m=float(np.percentile(distance, 90)),
        coverage=float((distance <= coverage_distance).mean()), volume_m3=shape_volume(solids, mesh),
    )
