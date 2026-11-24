"""Fit quality of a furniture shape: distance of the instance points to its surface, coverage, volume."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from samcloud.bim.furniture_recon.geometry import Mesh, Solid, sample_shape, shape_volume

logger = logging.getLogger(__name__)


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
    try:
        surface = sample_shape(solids, mesh)
        if not len(surface):
            return None
        distance, _ = cKDTree(surface).query(points, k=1)
        try:
            volume = shape_volume(solids, mesh)
        except Exception as exc:  # noqa: BLE001 - a volume failure must not drop the fit
            logger.warning("furniture fit: volume failed (%s); using the sum of the part volumes", exc)
            volume = float(sum(s.volume() for s in solids)) + (abs(mesh.volume()) if mesh is not None else 0.0)
        return FitMetrics(
            points=int(len(points)), mean_m=float(distance.mean()), p90_m=float(np.percentile(distance, 90)),
            coverage=float((distance <= coverage_distance).mean()), volume_m3=volume,
        )
    except Exception as exc:  # noqa: BLE001 - metrics are advisory: None means "no metrics", callers handle it
        logger.warning("furniture fit: metrics failed (%s); item kept without metrics", exc)
        return None
