"""Tight furniture footprints from an instance's own points (cloud + COLMAP depth voxels).

An instance's ``instances.json`` bounding box is a loose world AABB: rotated into the plan it is a diamond that
covers neighbouring floor.  The footprint measured from the instance's points is an oriented rectangle aligned
to the nearest wall (cabinets and appliances stand square to the wall), found with robust cell statistics, and
completed towards the wall behind it when only the front was seen (a thin slab of a tall appliance).

All coordinates are plan-local units; ``units_per_metre`` converts the metric thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon


@dataclass(frozen=True, slots=True)
class TightFootprintSettings:
    """Metric thresholds of the tight footprint measurement."""

    enabled: bool = True
    cell_m: float = 0.05
    # a cell needs this many points (cloud + depth voxels) to count; single stray points never make a footprint
    min_cell_points: int = 3
    min_points: int = 80
    # a footprint thinner than this (metres) is a slab: only the front of the object was seen
    thin_depth_m: float = 0.25
    # a slab is completed towards a wall closer than this behind it (the body reaches back to the wall)
    back_reach_m: float = 0.80
    # a measured or completed depth is limited to this (a cabinet or appliance is not deeper)
    max_depth_m: float = 0.80
    # a tall kitchen unit is at least this high and at most this wide on its narrow side
    tall_min_top_m: float = 1.80
    tall_max_width_m: float = 1.00


def nearest_wall_angle(centre: np.ndarray, wall_lines: list[LineString]) -> float | None:
    """Direction angle (radians, 0..pi) of the wall segment nearest to ``centre``; None without walls."""
    best: tuple[float, float] | None = None
    point = Point(float(centre[0]), float(centre[1]))
    for line in wall_lines:
        for part in getattr(line, "geoms", [line]):
            coordinates = np.asarray(part.coords, dtype=float)
            for first, second in zip(coordinates[:-1], coordinates[1:]):
                segment = LineString([tuple(first), tuple(second)])
                distance = segment.distance(point)
                if best is None or distance < best[0]:
                    delta = second - first
                    if float(np.hypot(*delta)) < 1e-9:
                        continue
                    best = (distance, math.atan2(delta[1], delta[0]) % math.pi)
    return None if best is None else best[1]


def tight_rectangle(points_xy: np.ndarray, angle: float, units_per_metre: float, settings: TightFootprintSettings = TightFootprintSettings()) -> Polygon | None:
    """Oriented rectangle (axes at ``angle``) around the main cluster of ``points_xy``; None without enough evidence.

    Cells (``cell_m``) with at least ``min_cell_points`` points are kept, the largest connected cluster wins, and its
    extent is the rectangle.  Sparse stray points (a lamp cable, a reflection) cannot stretch it.
    """
    upm = max(float(units_per_metre), 1e-9)
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < settings.min_points:
        return None
    cos, sin = math.cos(angle), math.sin(angle)
    a = pts[:, 0] * cos + pts[:, 1] * sin
    b = -pts[:, 0] * sin + pts[:, 1] * cos
    step = settings.cell_m * upm
    a0, b0 = float(a.min()), float(b.min())
    ia = ((a - a0) / step).astype(np.int64)
    ib = ((b - b0) / step).astype(np.int64)
    grid = np.zeros((int(ib.max()) + 1, int(ia.max()) + 1), dtype=np.int32)
    np.add.at(grid, (ib, ia), 1)
    keep = grid >= settings.min_cell_points
    if not keep.any():
        return None
    labels, count = ndimage.label(keep, structure=np.ones((3, 3), bool))
    if count > 1:
        weight = ndimage.sum(grid, labels, index=np.arange(1, count + 1))
        keep = labels == (int(np.argmax(weight)) + 1)
    rows, cols = np.nonzero(keep)
    a_low, a_high = a0 + cols.min() * step, a0 + (cols.max() + 1) * step
    b_low, b_high = b0 + rows.min() * step, b0 + (rows.max() + 1) * step
    corners = [(a_low, b_low), (a_high, b_low), (a_high, b_high), (a_low, b_high)]
    ring = [(ca * cos - cb * sin, ca * sin + cb * cos) for ca, cb in corners]
    polygon = Polygon(ring)
    return polygon if polygon.is_valid and not polygon.is_empty else None


def complete_towards_wall(rectangle: Polygon, wall_union: Any, units_per_metre: float, settings: TightFootprintSettings = TightFootprintSettings()) -> Polygon:
    """Extend the thin side of a slab-like rectangle back to the wall behind it (within ``back_reach_m``).

    Only a rectangle thinner than ``thin_depth_m`` on its short axis is touched; the side facing the nearer wall moves to
    that wall, and never so far that the depth exceeds ``max_depth_m``.
    """
    upm = max(float(units_per_metre), 1e-9)
    if wall_union is None or wall_union.is_empty:
        return rectangle
    corners = np.asarray(rectangle.exterior.coords[:4], dtype=float)
    edges = [corners[(index + 1) % 4] - corners[index] for index in range(4)]
    lengths = [float(np.hypot(*edge)) for edge in edges]
    short_index = 0 if lengths[0] <= lengths[1] else 1
    thickness = lengths[short_index]
    if thickness >= settings.thin_depth_m * upm or thickness < 1e-9:
        return rectangle
    along = edges[short_index] / thickness  # unit vector across the thin dimension
    centre = corners.mean(axis=0)
    best: tuple[float, int] | None = None
    for sign in (-1, 1):
        probe_start = centre + sign * along * thickness / 2.0
        probe = LineString([tuple(probe_start), tuple(probe_start + sign * along * settings.back_reach_m * upm)])
        hit = probe.intersection(wall_union)
        if hit.is_empty:
            continue
        distance = float(Point(probe_start).distance(hit))
        if best is None or distance < best[0]:
            best = (distance, sign)
    if best is None:
        return rectangle
    distance, sign = best
    new_thickness = min(thickness + distance, settings.max_depth_m * upm)
    grow = new_thickness - thickness
    if grow <= 0.0:
        return rectangle
    long_index = 1 - short_index
    other = edges[long_index] / max(lengths[long_index], 1e-9)
    half_long = lengths[long_index] / 2.0
    near = centre + sign * along * thickness / 2.0
    far_front = centre - sign * along * thickness / 2.0
    new_near = near + sign * along * grow
    ring = [far_front - other * half_long, far_front + other * half_long, new_near + other * half_long, new_near - other * half_long]
    grown = Polygon([tuple(float(c) for c in point) for point in ring])
    return grown if grown.is_valid and not grown.is_empty else rectangle


def tight_footprint(
    points_xy: np.ndarray,
    wall_lines: list[LineString],
    wall_union: Any,
    units_per_metre: float,
    settings: TightFootprintSettings = TightFootprintSettings(),
) -> Polygon | None:
    """Wall-aligned tight rectangle of ``points_xy``, completed towards the wall when it is a thin slab."""
    pts = np.asarray(points_xy, dtype=float)
    if not settings.enabled or len(pts) < settings.min_points:
        return None
    centre = np.median(pts, axis=0)
    angle = nearest_wall_angle(centre, wall_lines)
    if angle is None:
        return None
    rectangle = tight_rectangle(pts, angle, units_per_metre, settings)
    if rectangle is None:
        return None
    return complete_towards_wall(rectangle, wall_union, units_per_metre, settings)


def narrow_side_m(polygon: Polygon, units_per_metre: float) -> float:
    """Short side of the minimum rotated rectangle of ``polygon`` in metres."""
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        return 0.0
    corners = list(rectangle.exterior.coords)
    return min(math.dist(corners[0], corners[1]), math.dist(corners[1], corners[2])) / max(float(units_per_metre), 1e-9)


class InstanceFootprints:
    """Tight footprints of instances, measured once per instance from their points (cloud + depth) and cached."""

    def __init__(self, points_by_instance: dict[int, np.ndarray], wall_lines: list[LineString], wall_union: Any, units_per_metre: float,
                 settings: TightFootprintSettings = TightFootprintSettings()) -> None:
        self.points = points_by_instance
        self.wall_lines = wall_lines
        self.wall_union = wall_union
        self.upm = float(units_per_metre)
        self.settings = settings
        self._cache: dict[int, Polygon | None] = {}

    def has(self, instance_id: int | None) -> bool:
        return instance_id is not None and int(instance_id) in self.points

    def footprint(self, instance_id: int | None) -> Polygon | None:
        if not self.settings.enabled or instance_id is None or int(instance_id) not in self.points:
            return None
        key = int(instance_id)
        if key not in self._cache:
            self._cache[key] = tight_footprint(self.points[key][:, :2], self.wall_lines, self.wall_union, self.upm, self.settings)
        return self._cache[key]

    def top_m(self, instance_id: int | None, percentile: float = 95.0) -> float | None:
        if instance_id is None or int(instance_id) not in self.points:
            return None
        return float(np.percentile(self.points[int(instance_id)][:, 2], percentile) / max(self.upm, 1e-9))


def instance_point_sets(
    points: np.ndarray, instances: np.ndarray, depth_points: np.ndarray | None, units_per_metre: float,
    min_points: int = 80, radius_m: float = 0.08,
) -> dict[int, np.ndarray]:
    """Group cloud points by instance id and add the depth voxels (COLMAP furniture-class gap fill) lying within ``radius_m`` of them."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=float)
    ids = np.asarray(instances)
    result: dict[int, np.ndarray] = {}
    if not len(pts):
        return result
    upm = max(float(units_per_metre), 1e-9)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1, len(sorted_ids)]
    extra = None if depth_points is None or not len(depth_points) else np.asarray(depth_points, dtype=float)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        identifier = int(sorted_ids[start])
        if identifier <= 0 or stop - start < min_points:
            continue
        member = pts[order[start:stop]]
        if extra is not None:
            margin = (radius_m + 0.02) * upm
            low, high = member.min(axis=0) - margin, member.max(axis=0) + margin
            box = np.all((extra >= low) & (extra <= high), axis=1)
            if box.any():
                candidates = extra[box]
                distance, _ = cKDTree(member).query(candidates, distance_upper_bound=radius_m * upm)
                near = candidates[np.isfinite(distance)]
                if len(near):
                    member = np.vstack((member, near))
        result[identifier] = member
    return result
