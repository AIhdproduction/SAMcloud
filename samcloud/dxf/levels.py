"""Indoor floor and ceiling peak detection with deterministic level naming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

from samcloud.dxf.cancel import CancelToken, ProgressCallback, check_cancel, report_progress
from samcloud.dxf.config import DxfExportConfig, LevelsConfig, load_dxf_config


@dataclass(frozen=True, slots=True)
class Level:
    """Detected indoor storey represented by its floor and optional ceiling elevation."""

    id: str
    name: str
    floor_z: float
    ceiling_z: float | None
    point_count: int
    confidence: float
    low_confidence: bool
    reason: str | None = None


def detect_levels(
    xyz: np.ndarray,
    classification: np.ndarray,
    *,
    config: DxfExportConfig | LevelsConfig | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> list[Level]:
    """Detect floors and ceilings; levels are named "1", "2", "3"... upward from the lowest floor.

    Users can rename levels in the GUI.
    """
    points, classes = _validate_cloud(xyz, classification)
    settings = _levels_settings(config)
    floor_z = points[classes == 102, 2]
    ceiling_z = points[classes == 101, 2]
    fallback = len(floor_z) == 0
    check_cancel(cancel)
    report_progress(progress, 0.05, "Detecting floor and ceiling surfaces")
    if fallback:
        floor_peaks = _surface_peaks(points[:, 2], settings, progress=progress, cancel=cancel)
        if len(ceiling_z):
            ceiling_peaks = _surface_peaks(ceiling_z, settings, progress=progress, cancel=cancel)
            floor_peaks = [peak for peak in floor_peaks if all(abs(peak[0] - ceiling[0]) > settings.bin_size for ceiling in ceiling_peaks)]
        else:
            ceiling_peaks = []
        reason = "No floor-class points were available; levels were inferred from all horizontal surface peaks."
    else:
        floor_peaks = _surface_peaks(floor_z, settings, progress=progress, cancel=cancel)
        ceiling_peaks = _surface_peaks(ceiling_z, settings, progress=progress, cancel=cancel)
        reason = None
    report_progress(progress, 0.65, "Matching floors and ceilings")
    levels: list[Level] = []
    for index, (floor, _peak_count, prominence) in enumerate(sorted(floor_peaks)):
        check_cancel(cancel)
        matching = [item for item in ceiling_peaks if settings.min_floor_to_ceiling <= item[0] - floor <= settings.max_floor_to_ceiling]
        ceiling = min(matching, key=lambda item: item[0])[0] if matching else None
        confidence = min(1.0, prominence / max(settings.min_prominence * 4.0, 1.0))
        if ceiling is not None:
            confidence = min(1.0, confidence + 0.2)
        point_count = int(np.count_nonzero(np.abs(floor_z - floor) <= settings.floor_point_tolerance))
        if fallback:
            point_count = int(np.count_nonzero(np.abs(points[:, 2] - floor) <= settings.floor_point_tolerance))
        levels.append(Level(f"level_{index + 1}", "", floor, ceiling, point_count, confidence, fallback, reason))
    result = _name_levels(levels)
    report_progress(progress, 1.0, "Level detection complete")
    return result


def assign_stairs(
    levels: Iterable[Level],
    stair_points: np.ndarray,
    *,
    config: DxfExportConfig | LevelsConfig | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> dict[int, str]:
    """Assign each DBSCAN stair cluster to the level at or below its lowest point."""
    selected = sorted(levels, key=lambda level: level.floor_z)
    points = np.asarray(stair_points, dtype=np.float64)
    check_cancel(cancel)
    if points.size == 0:
        return {}
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Stair points must have shape (N, 3)")
    if not selected:
        return {}
    settings = _levels_settings(config)
    report_progress(progress, 0.1, "Clustering stairs")
    labels = DBSCAN(eps=settings.stair_cluster_eps, min_samples=settings.stair_cluster_min_samples).fit_predict(points)
    check_cancel(cancel)
    result: dict[int, str] = {}
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        check_cancel(cancel)
        lowest = float(points[labels == label, 2].min())
        below = [level for level in selected if level.floor_z <= lowest + settings.bin_size]
        target = below[-1] if below else selected[0]
        result[label] = target.id
    report_progress(progress, 1.0, "Stair assignment complete")
    return result


def _surface_peaks(
    values: np.ndarray,
    settings: LevelsConfig,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> list[tuple[float, int, float]]:
    """Find horizontal-surface peaks while observing cancellation requests."""
    check_cancel(cancel)
    if len(values) == 0:
        return []
    lower = float(np.min(values)) - settings.bin_size / 2.0
    upper = float(np.max(values)) + settings.bin_size / 2.0
    edges = np.arange(lower, upper + settings.bin_size, settings.bin_size)
    if len(edges) < 3:
        edges = np.array([lower, lower + settings.bin_size, lower + 2 * settings.bin_size])
    counts, edges = np.histogram(values, bins=edges)
    check_cancel(cancel)
    report_progress(progress, 0.3, "Smoothing surface-height histogram")
    smooth = gaussian_filter1d(counts.astype(np.float64), settings.gaussian_sigma)
    check_cancel(cancel)
    padded = np.pad(smooth, 1, mode="constant")
    padded_indices, details = find_peaks(padded, height=settings.min_point_count, prominence=settings.min_prominence)
    valid = (padded_indices > 0) & (padded_indices <= len(smooth))
    indices = padded_indices[valid] - 1
    centres = (edges[:-1] + edges[1:]) / 2.0
    prominences = np.asarray(details["prominences"])[valid]
    check_cancel(cancel)
    return [(float(centres[index]), int(counts[index]), float(prominences[position])) for position, index in enumerate(indices)]


def _name_levels(levels: list[Level]) -> list[Level]:
    if not levels:
        return []
    ordered = sorted(levels, key=lambda level: level.floor_z)
    named: list[Level] = []
    for index, level in enumerate(ordered):
        name = str(index + 1)
        named.append(Level(level.id, name, level.floor_z, level.ceiling_z, level.point_count, level.confidence, level.low_confidence, level.reason))
    return named


def _validate_cloud(xyz: np.ndarray, classification: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=np.float64)
    classes = np.asarray(classification, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(classes):
        raise ValueError("XYZ points must have shape (N, 3) and match classification length")
    return points, classes


def _levels_settings(config: DxfExportConfig | LevelsConfig | None) -> LevelsConfig:
    if config is None:
        return load_dxf_config().levels
    return config.levels if isinstance(config, DxfExportConfig) else config
