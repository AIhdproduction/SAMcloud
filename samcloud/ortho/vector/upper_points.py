"""Upper kitchen cabinets from kitchen-class points (cloud + COLMAP depth), with or without a base run below.

The run-based detector (:mod:`.upper_cabinets`) needs a base-cabinet run and upper-band raster evidence.  A hanging unit
without a base run below it (a unit above a tall appliance or a bare wall) is only visible as furniture-class points in the
upper z band.  This module finds those from the points themselves and measures the unit's height from the robust (p5..p95)
band of its points, never from a loose instance bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon


@dataclass(frozen=True, slots=True)
class UpperPointSettings:
    """Point-evidence detection of hanging units (metres, scaled by units per metre at use)."""

    enabled: bool = True
    # core z band of a hanging unit, and the band below it that must stay free of furniture-class points
    band_min_m: float = 1.40
    band_max_m: float = 2.25
    gap_low_m: float = 1.00
    gap_high_m: float = 1.25
    cell_m: float = 0.05
    min_points_per_cell: int = 4
    max_wall_distance_m: float = 0.70
    min_length_m: float = 0.40
    max_length_m: float = 4.00
    min_depth_m: float = 0.20
    max_depth_m: float = 0.60
    # a unit whose measured band is thinner than this is no cabinet (a shelf edge, a lamp)
    min_z_span_m: float = 0.40
    # a hanging unit is mounted on the wall: it reaches back to the wall face, and evidence starting farther than this from the
    # wall plane is no wall unit (a shelf on a stand, a lamp)
    max_wall_gap_m: float = 0.25
    # existing (run-based) detections are kept when a point cluster overlaps them by this share
    duplicate_overlap: float = 0.50
    min_band_points: int = 40
    # a hanging unit leaves the gap band free: at most this share (of its upper-band points) may lie in it (a tall unit fills it)
    gap_max_ratio: float = 0.15


def load_class_points(source: "Path | str | None", frame: Any, floor_z: float, classes: Iterable[int] = (17, 18, 19), min_z: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(local xyz with z above the floor, scalar class, instance id) of the given classes of a classified PLY.

    Empty arrays when the source is not a PLY with ``scalar_class`` (the raster evidence then stays the only source).
    """
    empty = (np.empty((0, 3), np.float64), np.empty(0, np.uint8), np.empty(0, np.uint32))
    if source is None or str(source).lower().endswith((".las", ".laz")) or not Path(source).is_file():
        return empty
    try:
        from samcloud.engine.ply_io import read_vertex_array

        vertices = read_vertex_array(source)
        names = vertices.dtype.names or ()
        if "scalar_class" not in names:
            return empty
        keep = np.isin(vertices["scalar_class"], np.asarray(list(classes), dtype=np.uint8))
        if not keep.any():
            return empty
        xyz = np.column_stack((vertices["x"][keep], vertices["y"][keep], vertices["z"][keep])).astype(np.float64)
        classes_out = np.asarray(vertices["scalar_class"][keep], dtype=np.uint8)
        instance = np.asarray(vertices["instance_id"][keep], dtype=np.uint32) if "instance_id" in names else np.zeros(int(keep.sum()), np.uint32)
    except Exception:  # noqa: BLE001 - optional evidence
        return empty
    local = np.asarray(frame.to_local(xyz), dtype=np.float64)
    local[:, 2] -= float(floor_z)
    above = local[:, 2] >= float(min_z)
    return local[above], classes_out[above], instance[above]


def measured_z_band(points: np.ndarray, polygon: Polygon, units_per_metre: float, low_m: float = 1.20, minimum: int = 20, percentiles: tuple[float, float] = (5.0, 95.0)) -> tuple[float, float] | None:
    """Robust (p5, p95) height in metres of the points above ``low_m`` inside ``polygon`` (grown 3 cm); None with too few points."""
    pts = np.asarray(points, dtype=float)
    if not len(pts):
        return None
    upm = max(float(units_per_metre), 1e-9)
    grown = polygon.buffer(0.03 * upm)
    minx, miny, maxx, maxy = grown.bounds
    near = (pts[:, 0] >= minx) & (pts[:, 0] <= maxx) & (pts[:, 1] >= miny) & (pts[:, 1] <= maxy) & (pts[:, 2] / upm >= low_m)
    candidate = pts[near]
    if not len(candidate):
        return None
    inside = shapely.contains_xy(grown, candidate[:, 0], candidate[:, 1])
    if int(inside.sum()) < minimum:
        return None
    low, high = np.percentile(candidate[inside, 2] / upm, percentiles)
    return float(low), float(high)


def detect_upper_cabinets_from_points(
    points: np.ndarray,
    wall_lines: Iterable[LineString],
    zone: Any,
    existing: Iterable[Polygon],
    units_per_metre: float,
    settings: UpperPointSettings = UpperPointSettings(),
) -> list[tuple[Polygon, tuple[float, float]]]:
    """Hanging units from furniture-class points alone: (footprint, (z_bottom_m, z_top_m)) pairs.

    A hanging unit is evidence in the upper band (``band_min_m``..``band_max_m``) that leaves ``gap_low_m``..``gap_high_m`` free
    (a tall unit or a wall does not), close to a wall face and, when ``zone`` is given, inside the kitchen zone.  No base run is
    needed.  The footprint always reaches back to the wall (evidence starting more than ``max_wall_gap_m`` from it is no wall unit), its depth is
    measured (clamped to ``min_depth_m``..``max_depth_m``), its height is the robust measured band.  Clusters overlapping an
    ``existing`` detection are skipped.
    """
    from scipy import ndimage

    upm = max(float(units_per_metre), 1e-9)
    walls = [part for line in wall_lines if line is not None and not line.is_empty for part in getattr(line, "geoms", [line])]
    pts = np.asarray(points, dtype=float)
    if not settings.enabled or not len(pts) or not walls:
        return []
    zm = pts[:, 2] / upm
    upper = (zm >= settings.band_min_m) & (zm <= settings.band_max_m)
    gap = (zm >= settings.gap_low_m) & (zm <= settings.gap_high_m)
    if int(upper.sum()) < settings.min_band_points:
        return []
    step = settings.cell_m * upm
    focus = pts[upper | gap]
    x0, y0 = float(focus[:, 0].min()), float(focus[:, 1].min())
    nx = int((float(focus[:, 0].max()) - x0) / step) + 3
    ny = int((float(focus[:, 1].max()) - y0) / step) + 3
    if nx * ny > 4_000_000:
        return []
    count_upper = np.zeros((ny, nx), dtype=np.int32)
    count_gap = np.zeros((ny, nx), dtype=np.int32)
    for mask, target in ((upper, count_upper), (gap, count_gap)):
        if mask.any():
            cols = ((pts[mask, 0] - x0) / step).astype(np.int64) + 1
            rows = ((pts[mask, 1] - y0) / step).astype(np.int64) + 1
            np.add.at(target, (rows, cols), 1)
    cell = (count_upper >= settings.min_points_per_cell) & (count_gap <= np.maximum(1, settings.gap_max_ratio * count_upper))
    if not cell.any():
        return []
    rows_idx, cols_idx = np.nonzero(cell)
    centres = np.column_stack((x0 + (cols_idx - 1 + 0.5) * step, y0 + (rows_idx - 1 + 0.5) * step))
    wall_union = shapely.union_all(walls)
    keep = shapely.distance(wall_union, shapely.points(centres[:, 0], centres[:, 1])) <= settings.max_wall_distance_m * upm
    if zone is not None:
        keep &= shapely.contains_xy(zone, centres[:, 0], centres[:, 1])
    cell[rows_idx[~keep], cols_idx[~keep]] = False
    taken = [polygon for polygon in existing if polygon is not None and not polygon.is_empty]
    if taken:
        # Evidence already explained by a run-based detection is removed first, so a cluster that runs on beyond it (the next unit
        # along the wall) stays a unit of its own instead of being swallowed as a duplicate.
        covered = shapely.union_all([polygon.buffer(0.10 * upm) for polygon in taken])
        rows_all, cols_all = np.nonzero(cell)
        cell_centres = np.column_stack((x0 + (cols_all - 1 + 0.5) * step, y0 + (rows_all - 1 + 0.5) * step))
        inside = shapely.contains_xy(covered, cell_centres[:, 0], cell_centres[:, 1])
        cell[rows_all[inside], cols_all[inside]] = False
    cell = ndimage.binary_closing(cell, structure=np.ones((3, 3), bool), iterations=1) & (count_upper > 0)
    labels, count = ndimage.label(cell, structure=np.ones((3, 3), bool))
    point_rows = ((pts[:, 1] - y0) / step).astype(np.int64) + 1
    point_cols = ((pts[:, 0] - x0) / step).astype(np.int64) + 1
    valid = (point_rows >= 0) & (point_rows < ny) & (point_cols >= 0) & (point_cols < nx)
    point_label = np.zeros(len(pts), dtype=np.int32)
    point_label[valid] = labels[point_rows[valid], point_cols[valid]]
    output: list[tuple[Polygon, tuple[float, float]]] = []
    for label in range(1, count + 1):
        member = labels == label
        if int(member.sum()) * step * step < settings.min_length_m * settings.min_depth_m * upm * upm:
            continue
        # the whole cluster must leave the gap band free (random sparse cells of a tall unit pass the per-cell test)
        if float(count_gap[member].sum()) > settings.gap_max_ratio * max(float(count_upper[member].sum()), 1.0):
            continue
        selected = pts[(point_label == label) & (zm >= settings.gap_high_m)]
        if len(selected) < settings.min_band_points:
            continue
        centroid = Point(float(selected[:, 0].mean()), float(selected[:, 1].mean()))
        wall = min(walls, key=lambda line: line.distance(centroid))
        coordinates = np.asarray(wall.coords, dtype=float)
        direction = coordinates[-1] - coordinates[0]
        length = float(np.hypot(*direction))
        if length < 1e-9:
            continue
        u = direction / length
        v = np.asarray((-u[1], u[0]))
        relative = selected[:, :2] - coordinates[0]
        along, across = relative @ u, relative @ v
        if float(np.median(across)) < 0.0:
            v, across = -v, -across
        a_low, a_high = (float(value) for value in np.percentile(along, (2.0, 98.0)))
        b_low, b_high = (float(value) for value in np.percentile(across, (2.0, 98.0)))
        if not settings.min_length_m * upm <= (a_high - a_low) <= settings.max_length_m * upm:
            continue
        if b_low > settings.max_wall_gap_m * upm:
            continue
        back = 0.0
        front = b_high
        if front - back < settings.min_depth_m * upm:
            front = back + 0.35 * upm
        front = min(front, back + settings.max_depth_m * upm)
        origin = coordinates[0]
        ring = [origin + u * a_low + v * back, origin + u * a_high + v * back, origin + u * a_high + v * front, origin + u * a_low + v * front]
        polygon = Polygon([tuple(float(c) for c in point) for point in ring])
        if polygon.is_empty or polygon.area <= 0.0:
            continue
        if any(polygon.intersection(other).area >= settings.duplicate_overlap * min(polygon.area, other.area) for other in taken):
            continue
        z_low, z_high = (float(value) for value in np.percentile(selected[:, 2] / upm, (5.0, 95.0)))
        if z_high - z_low < settings.min_z_span_m:
            continue
        output.append((polygon, (z_low, z_high)))
        taken.append(polygon)
    return output
