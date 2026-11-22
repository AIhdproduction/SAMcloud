"""Shared pieces of the structure analysis: settings, the metric point set of one level, geometry helpers.

Every detector of this package works in *metres* in the plan frame: ``x, y`` = local plan coordinates divided by
``units_per_metre`` and ``z`` = height above the level floor in metres.  Results are converted back to local plan units
only when they are written into ``plan_model.json`` (see :mod:`.analysis`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

# Cyclone codes (config/cyclone_codes.yaml) used by the structure analysis.
WALL = 100
CEILING = 101
FLOOR = 102
DOOR = 103
WINDOW = 104
STAIRS = 105
COLUMN = 108
BEAM = 109
STAIRS_WALL = 126
# Classes that are never structure: furniture, sanitary / kitchen, clutter, construction material.
FOREIGN_CODES = frozenset({1, 117, 118, 119, 120, 121, 122, 123, 124, 125, 180})


@dataclass(frozen=True, slots=True)
class StructureSettings:
    """Thresholds of the structure analysis (metres / degrees / shares).  Physically motivated defaults.

    Read from the optional ``structure:`` section of ``config/ortho_plan.yaml``; unknown keys are ignored.
    """

    enabled: bool = True
    max_points: int = 8_000_000  # point budget of the level (stride decimation above it)
    # --- wall heights -------------------------------------------------------------------------------------------
    wall_bin_m: float = 0.20  # station length along a wall
    wall_min_points_per_bin: int = 8
    wall_min_bin_share: float = 0.30  # share of stations that must carry wall points
    wall_top_percentile: float = 98.0  # top of one station (robust against single high points)
    wall_top_quantile: float = 75.0  # wall top across stations (stations hidden by upper cabinets are ignored)
    wall_ceiling_tolerance_m: float = 0.10  # a wall within this of the ceiling reaches the ceiling
    wall_free_above_ratio: float = 0.15  # density above / below a low wall top that proves free space
    wall_min_height_m: float = 0.30
    # --- per-room floor / ceiling ---------------------------------------------------------------------------------
    room_erode_m: float = 0.15
    room_cell_m: float = 0.25
    room_min_points: int = 30
    room_min_ceiling_coverage: float = 0.15
    ceiling_min_clear_m: float = 1.50
    ceiling_slope_min_deg: float = 3.0
    ceiling_plane_min_inliers: float = 0.60
    ceiling_plane_min_rise_m: float = 0.10  # a sloped ceiling must rise at least this much across the room
    # two planes (gable / valley / slope + flat) only when clearly better than one surface
    ceiling_two_plane_gain: float = 0.15  # explained share must grow by at least this
    ceiling_two_plane_min_share: float = 0.75
    ceiling_two_plane_min_part: float = 0.15  # each plane owns at least this share of the points and of the room
    ceiling_two_plane_min_angle_deg: float = 5.0
    # --- columns ---------------------------------------------------------------------------------------------------
    column_cell_m: float = 0.04
    column_max_section_m: float = 0.80
    column_min_section_m: float = 0.08
    column_max_aspect: float = 2.5
    column_min_height_share: float = 0.70
    column_top_gap_m: float = 0.15
    column_wall_clearance_m: float = 0.10
    column_opening_clearance_m: float = 0.15
    column_min_points: int = 60
    column_max_foreign_share: float = 0.30
    column_max_furniture_overlap: float = 0.30
    column_max_instance_overlap: float = 0.60  # share of the column inside a non-structural scan instance (door, bed ...)
    column_attached_density: float = 0.20  # strip-to-wall point density / own face density that marks a wall-attached blob
    column_max_residual_m: float = 0.025  # median point distance to the fitted outline ...
    column_max_relative_residual: float = 0.10  # ... and at most this share of the smaller section side
    column_slab_extent_ratio: float = 1.40
    # --- beams -----------------------------------------------------------------------------------------------------
    beam_cell_m: float = 0.10
    beam_min_drop_m: float = 0.08
    beam_max_drop_m: float = 0.80
    beam_min_width_m: float = 0.08
    beam_max_width_m: float = 0.80
    beam_min_length_m: float = 1.00
    beam_min_elongation: float = 4.0
    beam_min_fill: float = 0.55
    beam_support_reach_m: float = 0.35
    beam_wall_adjacent_m: float = 0.15
    beam_side_clear_share: float = 0.60
    # --- stairs ----------------------------------------------------------------------------------------------------
    stair_min_points: int = 80
    stair_cluster_m: float = 0.25
    stair_riser_m: tuple[float, float] = (0.12, 0.23)
    stair_tread_m: tuple[float, float] = (0.18, 0.40)
    stair_comfort_m: tuple[float, float] = (0.50, 0.80)
    stair_min_width_m: float = 0.60
    stair_max_width_m: float = 3.00
    stair_min_risers: int = 3
    stair_min_r2: float = 0.70
    stair_min_level_share: float = 0.60  # share of tread levels that must carry points


def load_structure_settings(path: Path | str | None = None) -> StructureSettings:
    """``structure:`` of ``config/ortho_plan.yaml`` over the defaults."""
    import yaml

    if path is None:
        from samcloud.core.paths import CONFIG_DIRECTORY

        path = Path(CONFIG_DIRECTORY) / "ortho_plan.yaml"
    section: dict[str, Any] = {}
    if Path(path).is_file():
        section = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("structure") or {}
    names = {item.name for item in fields(StructureSettings)}
    values: dict[str, Any] = {}
    for key, value in section.items():
        if key not in names:
            continue
        values[key] = tuple(value) if isinstance(value, list) else value
    return StructureSettings(**values)


@dataclass(slots=True)
class LevelPoints:
    """Classified points of one level in metres (plan x / y, z above the level floor) with their Cyclone code."""

    xyz: np.ndarray
    codes: np.ndarray
    storey_height_m: float

    def of(self, codes: Iterable[int]) -> np.ndarray:
        return self.xyz[np.isin(self.codes, np.asarray(list(codes), dtype=self.codes.dtype))]

    def mask(self, codes: Iterable[int]) -> np.ndarray:
        return np.isin(self.codes, np.asarray(list(codes), dtype=self.codes.dtype))

    def spacing_m(self, codes: Iterable[int] = (WALL,), sample: int = 20_000) -> float:
        """Median nearest-neighbour distance of a deterministic sample (point spacing of the cloud)."""
        from scipy.spatial import cKDTree

        points = self.of(codes)
        if len(points) < 10:
            return 0.02
        step = max(1, len(points) // sample)
        subset = points[::step]
        distances, _ = cKDTree(subset).query(subset, k=2)
        # A stride sample of a surface thins it by ``step``: the spacing of the full cloud is sqrt(step) smaller.
        return float(np.clip(np.median(distances[:, 1]) / np.sqrt(step), 0.003, 0.10))


def load_level_points(
    source: Path | str | None, frame: Any, floor_z: float, units_per_metre: float, storey_height_m: float,
    settings: StructureSettings = StructureSettings(), cancel: Any = None,
) -> LevelPoints | None:
    """Stream the classified source (PLY or LAS) once and keep the points of the level's height range (metres)."""
    if source is None or not Path(source).is_file():
        return None
    from samcloud.dxf.las_source import iter_source_chunks

    upm = max(float(units_per_metre), 1e-9)
    low, high = -1.2 * storey_height_m, 1.6 * storey_height_m
    pieces: list[np.ndarray] = []
    code_pieces: list[np.ndarray] = []
    total = 0
    for chunk in iter_source_chunks(Path(source), 500_000, cancel=cancel, stage="Streaming points for structure analysis"):
        local = np.asarray(frame.to_local(np.column_stack((chunk.x, chunk.y, chunk.z))), dtype=np.float64)
        z = (local[:, 2] - float(floor_z)) / upm
        keep = (z >= low) & (z <= high)
        if not keep.any():
            continue
        metric = np.column_stack((local[keep, 0] / upm, local[keep, 1] / upm, z[keep]))
        pieces.append(metric)
        code_pieces.append(np.asarray(chunk.classification, dtype=np.uint16)[keep])
        total += len(metric)
    if not pieces:
        return None
    xyz = np.vstack(pieces)
    codes = np.concatenate(code_pieces)
    if total > settings.max_points:
        step = int(np.ceil(total / settings.max_points))
        xyz, codes = xyz[::step], codes[::step]
    return LevelPoints(xyz=xyz, codes=codes, storey_height_m=float(storey_height_m))


def to_metric(geometry: BaseGeometry, units_per_metre: float) -> BaseGeometry:
    factor = 1.0 / max(float(units_per_metre), 1e-9)
    return affinity.scale(geometry, factor, factor, origin=(0.0, 0.0))


def to_local(geometry: BaseGeometry, units_per_metre: float) -> BaseGeometry:
    factor = float(units_per_metre)
    return affinity.scale(geometry, factor, factor, origin=(0.0, 0.0))


def ring_local(polygon: Polygon, units_per_metre: float) -> list[list[float]]:
    """Exterior ring (first point not repeated) of a metric polygon in local plan units."""
    return [[float(x) * units_per_metre, float(y) * units_per_metre] for x, y in list(polygon.exterior.coords)[:-1]]


def contains(polygon: BaseGeometry, xy: np.ndarray) -> np.ndarray:
    import shapely

    if len(xy) == 0 or polygon.is_empty:
        return np.zeros(len(xy), dtype=bool)
    return shapely.contains_xy(polygon, xy[:, 0], xy[:, 1])


def histogram_mode(values: np.ndarray, bin_m: float = 0.01, smooth: int = 2) -> float:
    """Peak of a smoothed histogram, refined as the median of the values within two bins of it."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < bin_m:
        return float(np.median(values))
    bins = int(np.ceil((hi - lo) / bin_m)) + 1
    counts, edges = np.histogram(values, bins=bins, range=(lo, lo + bins * bin_m))
    if smooth > 0:
        kernel = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        counts = np.convolve(counts, kernel, mode="same")
    peak = int(np.argmax(counts))
    centre = 0.5 * (edges[peak] + edges[peak + 1])
    near = values[np.abs(values - centre) <= 2.0 * bin_m]
    return float(np.median(near)) if len(near) else float(centre)


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))
