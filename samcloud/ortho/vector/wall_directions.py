"""Indoor wall faces off the dominant orthogonal axis pair.

``extract_axis_wall_faces`` recovers the faces of the two dominant orthogonal
directions.  Slanted walls, trapezoid rooms or a second building wing at
another angle leave wall evidence those faces do not explain.  This module
finds further dominant directions in that residual evidence and extracts
their faces with the same projection criteria (occupancy, fill, vertical
span, empty side), generalised to an arbitrary direction.

Without significant residual evidence the axis faces are returned unchanged
(same objects, same coordinates).

Extra acceptance rules for the new faces (physical, not project specific):

* a wall reaches the ceiling: the face support must reach within
  ``top_reach_m`` of the top of the wall-evidence band and cover at least
  ``vertical_fill`` of its 10 cm height bins (rejects open door leaves,
  furniture sides and curtains, which end lower or hang near the ceiling);
* a face of a new direction is a wall when it is at least ``min_length_m``
  long, or at least ``min_joined_length_m`` long with both ends joined to
  other faces (chamfered corners, short slanted pieces between two walls).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import cv2
import numpy as np

from .raster_grid import RasterGrid
from .wall_lines import WallFace, _close_corners, _merge_axis_faces, projection_faces


@dataclass(frozen=True, slots=True)
class DirectionSettings:
    """Configuration of the multi-direction wall-face pass (``walls.extra_directions``)."""

    enabled: bool = True
    min_axis_offset_deg: float = 5.0
    explained_m: float = 0.05
    min_length_m: float = 1.0
    min_joined_length_m: float = 0.40
    join_reach_m: float = 0.30
    top_reach_m: float = 0.15
    vertical_fill: float = 0.50
    max_directions: int = 4
    corner_min_angle_deg: float = 15.0
    near_parallel_angle_deg: float = 20.0
    near_parallel_reach_m: float = 0.25
    axis_thin_wall_faces: bool = False
    min_slant_m: float = 0.40

    @classmethod
    def from_mapping(cls, data: Any) -> "DirectionSettings":
        if isinstance(data, bool):
            return cls(enabled=data)
        if not isinstance(data, dict):
            return cls()
        defaults = cls()
        values = {}
        for name in cls.__dataclass_fields__:
            raw = data.get(name, getattr(defaults, name))
            values[name] = type(getattr(defaults, name))(raw)
        return cls(**values)


def add_direction_faces(
    faces: list[WallFace],
    evidence: np.ndarray,
    theta: float,
    grid: RasterGrid,
    units_per_metre: float,
    settings: DirectionSettings,
    *,
    passable: np.ndarray | None = None,
) -> list[WallFace]:
    """Return ``faces`` plus measured faces of further dominant wall directions.

    1. Candidate directions: Hough segments of the vertically persistent evidence
       at least ``min_axis_offset_deg`` off the axes, TLS-refined and clustered.
    2. Faces of each direction with the axis criteria (side-aware suppression so
       both faces of a thin slanted wall survive), reaching the ceiling.
    3. Conflicts: binning a slanted wall on the axis grid (or an axis wall on a
       slanted grid) yields short stair-step faces whose support lies on the
       other family's faces.  A face whose support is mostly (>= 60 %) within
       the measured scatter of the other family is dropped.
    4. New faces next to a remaining face at a small angle are door leaves,
       curtains or wall irregularities; short new faces need joined ends.

    When no new face survives, the original ``faces`` list is returned untouched.
    """
    data = np.asarray(evidence, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3 or len(data) < 50:
        return faces
    points, heights = data[:, :2], data[:, 2]
    rms_values = [face.rms for face in faces if face.rms > 0.0]
    median_rms = float(np.median(rms_values)) if rms_values else 0.0
    # Points within the measured scatter of a face are explained by it.
    claim_radius = max(settings.explained_m * units_per_metre, 4.0 * median_rms)
    conflict_radius = max(0.6 * settings.explained_m * units_per_metre, 3.0 * median_rms)
    directions = candidate_directions(points, heights, theta, units_per_metre, settings)
    if not directions:
        return faces
    top, bottom = (float(value) for value in np.percentile(heights, (99.0, 1.0)))
    new_faces: list[WallFace] = []
    family: dict[int, int] = {}
    available = np.ones(len(points), dtype=bool)
    for angle in directions[: max(0, settings.max_directions)]:
        if int(available.sum()) < 30:
            break
        subset_points, subset_heights = points[available], heights[available]
        rotation = np.asarray(((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))))
        found = projection_faces(subset_points, subset_heights, subset_points @ rotation.T, rotation, 0, units_per_metre, axis_snapped=False, side_aware_nms=True)
        index = _PointIndex(subset_points, 0.50 * units_per_metre)
        # A slant must be visible over the face: its lateral deviation from the axis direction (length x sin(offset))
        # must exceed the surface spread of a smeared scan (unfused clouds: 85-115 mm), else it is a diagonal through a thick wall band.
        found = [face for face in found if _slant(face, theta) * face.length >= settings.min_slant_m * units_per_metre]
        found = [face for face in found if _reaches_ceiling(face, subset_points, subset_heights, top, bottom, units_per_metre, settings, index)]
        # Axis faces that are stair steps of these (much longer) faces do not count as a conflict.
        stair_steps = {id(face) for face in faces if _explained_share(face, [other for other in found if other.length > 1.5 * face.length], conflict_radius) >= 0.60}
        reference = [face for face in faces if id(face) not in stair_steps]
        accepted = [face for face in found if _explained_share(face, reference, conflict_radius) < 0.60 and not _inside_wall_body(face, reference, units_per_metre)]
        if not accepted:
            continue
        new_faces.extend(accepted)
        family.update({id(face): int(round(math.degrees(angle) * 10)) for face in accepted})
        available &= ~_near_faces(points, accepted, claim_radius)
    if not new_faces:
        return faces
    _merge_axis_faces(new_faces, units_per_metre, grid=grid, passable=passable)
    new_faces = [face for face in new_faces if face.line is not None]
    # Stair-step axis faces on slanted evidence give way to the (longer) slanted faces.
    surviving = [face for face in faces if _explained_share(face, [other for other in new_faces if other.length > 1.5 * face.length], conflict_radius) < 0.60]
    # Decide before any corner is closed: the axis faces stay untouched unless a new wall is kept.
    kept_new = [face for face in new_faces if not _shadows_face(face, surviving, units_per_metre, settings) and _is_wall(face, [*surviving, *new_faces], units_per_metre, settings)]
    for _attempt in range(3):
        # A direction is a wall direction only with at least one wall's length of kept faces.
        totals: dict[int, float] = {}
        for face in kept_new:
            totals[family.get(id(face), 0)] = totals.get(family.get(id(face), 0), 0.0) + face.length
        kept_new = [face for face in kept_new if totals[family.get(id(face), 0)] >= settings.min_length_m * units_per_metre]
        if not kept_new:
            return faces
        # Only faces explained by a kept slanted face are removed; the rest are copied so the caller's faces stay untouched.
        base = [replace(face) for face in faces if _explained_share(face, [other for other in kept_new if other.length > 1.5 * face.length], conflict_radius) < 0.60]
        additions = [replace(face) for face in kept_new]
        combined = [*base, *additions]
        _close_corners(combined, settings.join_reach_m * units_per_metre, minimum_angle_deg=settings.corner_min_angle_deg, involving={id(face) for face in additions})
        too_short = [original for original, closed in zip(kept_new, additions, strict=True) if closed.line is None or closed.length < settings.min_joined_length_m * units_per_metre]
        if not too_short:
            return [face for face in combined if face.line is not None and face.length > 0.0]
        short_ids = {id(face) for face in too_short}
        kept_new = [face for face in kept_new if id(face) not in short_ids]
    return faces


def _slant(face: WallFace, theta: float) -> float:
    """Sine of the angle between the face and the nearest axis direction."""
    start, end = np.asarray(face.line.coords[0], dtype=float), np.asarray(face.line.coords[-1], dtype=float)
    relative = (math.atan2(float(end[1] - start[1]), float(end[0] - start[0])) - theta) % (math.pi / 2.0)
    return math.sin(min(relative, math.pi / 2.0 - relative))


def _inside_wall_body(face: WallFace, references: list[WallFace], units_per_metre: float) -> bool:
    """The face lies between two parallel faces of one wall (a door reveal / jamb diagonal, not a wall)."""
    middle = np.asarray(face.line.interpolate(0.5, normalized=True).coords[0], dtype=float)
    lines = [reference for reference in references if reference.line is not None and reference.line.length > 1e-9]
    low, high = 0.05 * units_per_metre, 0.50 * units_per_metre
    for index, first in enumerate(lines):
        a0, a1 = np.asarray(first.line.coords[0], dtype=float), np.asarray(first.line.coords[-1], dtype=float)
        axis = (a1 - a0) / float(np.linalg.norm(a1 - a0))
        normal = np.asarray((-axis[1], axis[0]))
        t_first = float(np.dot(middle - a0, axis))
        if not -0.05 * units_per_metre <= t_first <= float(np.linalg.norm(a1 - a0)) + 0.05 * units_per_metre:
            continue
        offset_first = float(np.dot(middle - a0, normal))
        for second in lines[index + 1:]:
            b0, b1 = np.asarray(second.line.coords[0], dtype=float), np.asarray(second.line.coords[-1], dtype=float)
            other = (b1 - b0) / float(np.linalg.norm(b1 - b0))
            if abs(float(axis[0] * other[1] - axis[1] * other[0])) > math.sin(math.radians(2.0)):
                continue
            separation = float(np.dot(b0 - a0, normal))
            if not low <= abs(separation) <= high:
                continue
            t_second = float(np.dot(middle - b0, other))
            if not -0.05 * units_per_metre <= t_second <= float(np.linalg.norm(b1 - b0)) + 0.05 * units_per_metre:
                continue
            if 0.0 < offset_first / separation < 1.0:
                return True
    return False


def _explained_share(face: WallFace, others: list[WallFace], radius: float) -> float:
    """Share of the face support within ``radius`` of ``others``."""
    support = np.asarray(face.points, dtype=float).reshape((-1, 2)) if face.points is not None else np.empty((0, 2))
    candidates = [other for other in others if other is not face and other.line is not None]
    if not len(support) or not candidates:
        return 0.0
    return float(_near_faces(support, candidates, radius).mean())


def candidate_directions(points: np.ndarray, heights: np.ndarray, theta: float, units_per_metre: float, settings: DirectionSettings) -> list[float]:
    """Dominant line directions (radians, mod pi) of vertically persistent wall evidence.

    Directions within ``min_axis_offset_deg`` of the axis pair are skipped.
    Each probabilistic-Hough segment is refined by a total-least-squares fit of
    its points; segment angles are clustered (1.5 degrees) and weighted by the
    number of supporting points.
    """
    cell = 0.02 * units_per_metre
    if len(points) < 30:
        return []
    origin = points.min(axis=0)
    ij = np.floor((points - origin) / cell).astype(np.int64)
    width, height = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    if width * height > 40_000_000:
        return []
    key = ij[:, 1] * width + ij[:, 0]
    counts = np.bincount(key, minlength=width * height)
    z_index = np.clip(np.floor(heights / (0.10 * units_per_metre)).astype(np.int64), 0, 40)
    zmask = np.zeros(width * height, dtype=np.int64)
    np.bitwise_or.at(zmask, key, np.int64(1) << z_index)
    population = np.zeros(width * height, dtype=np.int16)
    remaining = zmask.copy()
    while remaining.any():
        population += (remaining & 1).astype(np.int16)
        remaining >>= 1
    image = ((population >= 3) & (counts >= 3)).reshape(height, width).astype(np.uint8) * 255
    if not image.any():
        return []
    minimum_length = max(3, int(round(settings.min_joined_length_m * units_per_metre / cell)))
    segments = cv2.HoughLinesP(image, 1, math.radians(0.5), max(8, int(round(0.30 * units_per_metre / cell))), minLineLength=minimum_length, maxLineGap=max(1, int(round(0.25 * units_per_metre / cell))))
    if segments is None:
        return []
    offset_limit = math.radians(settings.min_axis_offset_deg)
    estimates: list[tuple[float, float, float]] = []
    for x0, y0, x1, y1 in segments[:, 0]:
        start = origin + (np.asarray((x0, y0), dtype=float) + 0.5) * cell
        end = origin + (np.asarray((x1, y1), dtype=float) + 0.5) * cell
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        direction = vector / length
        raw_relative = (math.atan2(float(direction[1]), float(direction[0])) - theta) % (math.pi / 2.0)
        if min(raw_relative, math.pi / 2.0 - raw_relative) < offset_limit - math.radians(2.0):
            continue  # clearly an axis segment (Hough angle resolution 0.5 deg, cell quantisation)
        normal = np.asarray((-direction[1], direction[0]))
        margin = 0.03 * units_per_metre
        low, high = np.minimum(start, end) - margin, np.maximum(start, end) + margin
        box = np.flatnonzero((points[:, 0] >= low[0]) & (points[:, 0] <= high[0]) & (points[:, 1] >= low[1]) & (points[:, 1] <= high[1]))
        relative = points[box] - start
        along, across = relative @ direction, relative @ normal
        near = (np.abs(across) <= margin) & (along >= 0.0) & (along <= length)
        if int(near.sum()) < 10:
            continue
        support = points[box[near]]
        centre = support.mean(axis=0)
        _values, vectors = np.linalg.eigh(np.cov((support - centre).T))
        angle = math.atan2(float(vectors[1, -1]), float(vectors[0, -1])) % math.pi
        relative_angle = (angle - theta) % (math.pi / 2.0)
        if min(relative_angle, math.pi / 2.0 - relative_angle) < offset_limit:
            continue
        estimates.append((angle, float(near.sum()), length))
    clusters: list[list[tuple[float, float]]] = []
    lengths: list[float] = []
    for angle, weight, length in sorted(estimates, key=lambda item: -item[1]):
        for index, cluster in enumerate(clusters):
            reference = _weighted_angle(cluster)
            if _angle_difference(angle, reference) <= math.radians(1.5):
                cluster.append((angle, weight))
                lengths[index] += length
                break
        else:
            clusters.append([(angle, weight)])
            lengths.append(length)
    # A wall direction needs at least one wall's worth of line evidence (overlapping Hough segments count repeatedly,
    # the face criteria decide afterwards).
    ranked = sorted(((sum(weight for _angle, weight in cluster), cluster) for cluster, length in zip(clusters, lengths, strict=True) if length >= settings.min_length_m * units_per_metre), key=lambda item: -item[0])
    return [_weighted_angle(cluster) for _weight, cluster in ranked]


def _weighted_angle(values: list[tuple[float, float]]) -> float:
    """Weighted mean of axial angles (mod pi)."""
    vector = sum((weight * np.asarray((math.cos(2.0 * angle), math.sin(2.0 * angle))) for angle, weight in values), np.zeros(2))
    return (math.atan2(float(vector[1]), float(vector[0])) / 2.0) % math.pi


def _angle_difference(first: float, second: float) -> float:
    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def _near_faces(points: np.ndarray, faces: list[WallFace], radius: float) -> np.ndarray:
    """Mask of the points within ``radius`` of any face (raster lookup)."""
    lines = [face.geometry() for face in faces if face.line is not None or face.kind == "arc"]
    lines = [line for line in lines if not line.is_empty]
    if not lines or not len(points):
        return np.zeros(len(points), dtype=bool)
    cell = max(radius / 2.0, 1e-9)
    origin = points.min(axis=0) - 2.0 * radius
    extent = points.max(axis=0) + 2.0 * radius
    width, height = int(math.ceil((extent[0] - origin[0]) / cell)) + 1, int(math.ceil((extent[1] - origin[1]) / cell)) + 1
    if width * height > 60_000_000:
        cell = math.sqrt((extent[0] - origin[0]) * (extent[1] - origin[1]) / 60_000_000)
        width, height = int(math.ceil((extent[0] - origin[0]) / cell)) + 1, int(math.ceil((extent[1] - origin[1]) / cell)) + 1
    image = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, int(round(2.0 * radius / cell)))
    for line in lines:
        coords = np.round((np.asarray(line.coords) - origin) / cell * 16.0).astype(np.int32)
        cv2.polylines(image, [coords.reshape((-1, 1, 2))], False, 1, thickness=thickness, lineType=cv2.LINE_8, shift=4)
    ij = np.floor((points - origin) / cell).astype(np.int64)
    ij[:, 0] = np.clip(ij[:, 0], 0, width - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, height - 1)
    return image[ij[:, 1], ij[:, 0]].astype(bool)


def _reaches_ceiling(face: WallFace, points: np.ndarray, heights: np.ndarray, top: float, bottom: float, units_per_metre: float, settings: DirectionSettings, index: "_PointIndex | None" = None) -> bool:
    """The face support spans the wall band up to its top (a wall, not a door leaf or furniture side)."""
    if face.line is None or not len(face.points):
        return False
    support = _support_heights(face, points, heights, units_per_metre, index)
    if len(support) < 10:
        return False
    if float(np.percentile(support, 95.0)) < top - settings.top_reach_m * units_per_metre:
        return False
    step = 0.10 * units_per_metre
    bins = max(1, int(math.ceil((top - bottom) / step)))
    inside = support[(support >= bottom) & (support <= top)]
    occupied = np.unique(np.clip(np.floor((inside - bottom) / step).astype(np.int64), 0, bins - 1))
    return len(occupied) / bins >= settings.vertical_fill


class _PointIndex:
    """Coarse spatial bins of the evidence for fast neighbourhood queries."""

    def __init__(self, points: np.ndarray, cell: float) -> None:
        self.points, self.cell = points, cell
        self.origin = points.min(axis=0) if len(points) else np.zeros(2)
        cells = np.floor((points - self.origin) / cell).astype(np.int64)
        self.width = int(cells[:, 0].max()) + 1 if len(points) else 1
        keys = cells[:, 1] * self.width + cells[:, 0]
        self.order = np.argsort(keys, kind="stable")
        self.keys = keys[self.order]

    def within_box(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        if not len(self.points):
            return np.empty(0, dtype=np.int64)
        c0 = np.floor((low - self.origin) / self.cell).astype(np.int64)
        c1 = np.floor((high - self.origin) / self.cell).astype(np.int64)
        c0[0], c1[0] = max(0, c0[0]), min(self.width - 1, c1[0])
        parts = []
        for row in range(max(0, int(c0[1])), int(c1[1]) + 1):
            first, last = row * self.width + int(c0[0]), row * self.width + int(c1[0])
            if last < first:
                continue
            a, b = np.searchsorted(self.keys, first, side="left"), np.searchsorted(self.keys, last, side="right")
            if b > a:
                parts.append(self.order[a:b])
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _support_heights(face: WallFace, points: np.ndarray, heights: np.ndarray, units_per_metre: float, index: "_PointIndex | None" = None) -> np.ndarray:
    line = face.line
    start, end = np.asarray(line.coords[0], dtype=float), np.asarray(line.coords[-1], dtype=float)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return np.empty(0)
    direction = vector / length
    margin = 0.03 * units_per_metre
    candidates = index.within_box(np.minimum(start, end) - margin, np.maximum(start, end) + margin) if index is not None else np.arange(len(points))
    relative = points[candidates] - start
    along, across = relative @ direction, relative @ np.asarray((-direction[1], direction[0]))
    near = (np.abs(across) <= margin) & (along >= 0.0) & (along <= length)
    return heights[candidates[near]]


def _shadows_face(face: WallFace, references: list[WallFace], units_per_metre: float, settings: DirectionSettings) -> bool:
    """A face at a small angle next to an existing face is a door leaf, curtain or wall irregularity.

    Two walls cannot run within one wall thickness of each other at a small
    angle; the existing (axis) face is the measured wall.  The reference
    extent is widened by a door width, since a leaf stands in the door gap.
    """
    start, end = np.asarray(face.line.coords[0], dtype=float), np.asarray(face.line.coords[-1], dtype=float)
    direction = (end - start) / max(float(np.linalg.norm(end - start)), 1e-12)
    reach = settings.near_parallel_reach_m * units_per_metre
    widen = 1.0 * units_per_metre
    for reference in references:
        if reference.line is None:
            continue
        r0, r1 = np.asarray(reference.line.coords[0], dtype=float), np.asarray(reference.line.coords[-1], dtype=float)
        length = float(np.linalg.norm(r1 - r0))
        if length <= 1e-9:
            continue
        axis = (r1 - r0) / length
        if math.degrees(math.asin(min(1.0, abs(float(axis[0] * direction[1] - axis[1] * direction[0]))))) >= settings.near_parallel_angle_deg:
            continue
        normal = np.asarray((-axis[1], axis[0]))
        if max(abs(float(np.dot(start - r0, normal))), abs(float(np.dot(end - r0, normal)))) > reach:
            continue
        values = (float(np.dot(start - r0, axis)), float(np.dot(end - r0, axis)))
        if min(values) <= length + widen and max(values) >= -widen:
            return True
    return False


def _is_wall(face: WallFace, faces: list[WallFace], units_per_metre: float, settings: DirectionSettings) -> bool:
    if face.length >= settings.min_length_m * units_per_metre:
        return True
    if face.length < settings.min_joined_length_m * units_per_metre:
        return False
    from shapely.geometry import Point

    reach = settings.join_reach_m * units_per_metre
    others = [other.line for other in faces if other is not face and other.line is not None]
    ends = [Point(face.line.coords[0]), Point(face.line.coords[-1])]
    return all(any(other.distance(end) <= reach for other in others) for end in ends)


def trim_rooms_to_faces(rooms: list[Any], faces: list[WallFace], units_per_metre: float, *, max_tooth_m: float = 0.80, max_share: float = 0.35) -> list[Any]:
    """Cut the stair-step teeth of grid-aligned room polygons beyond measured slanted faces.

    Ceiling rooms are traced on the axis grid, so a slanted wall leaves a staircase
    of small pieces beyond its face.  Each room is split by every slanted face
    (extended by ``max_tooth_m``); a split-off part smaller than ``max_share`` of the
    room that lies within ``max_tooth_m`` of the face is removed.  Rooms without a
    slanted face nearby are returned unchanged (same objects).
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import split, unary_union

    slanted = [face for face in faces if not face.axis_snapped and face.kind == "line" and face.line is not None and face.line.length > 1e-9]
    if not slanted:
        return rooms
    reach = max_tooth_m * units_per_metre
    output: list[Any] = []
    for room in rooms:
        if not isinstance(room, Polygon) or room.is_empty:
            output.append(room)
            continue
        current = room
        for face in slanted:
            if face.line.distance(current) > reach:
                continue
            start, end = np.asarray(face.line.coords[0], dtype=float), np.asarray(face.line.coords[-1], dtype=float)
            direction = (end - start) / float(np.linalg.norm(end - start))
            cutter = LineString((tuple(start - direction * reach), tuple(end + direction * reach)))
            try:
                parts = list(split(current, cutter).geoms)
            except (ValueError, TypeError):
                continue
            if len(parts) < 2:
                continue
            area = current.area
            teeth = [part for part in parts if part.area < max_share * area and part.hausdorff_distance(face.line) <= reach + 0.5 * face.line.length and _max_distance_to_line(part, start, direction) <= reach]
            if not teeth:
                continue
            kept = [part for part in parts if not any(part is tooth for tooth in teeth)]
            merged = unary_union(kept)
            if isinstance(merged, Polygon) and merged.is_valid and not merged.is_empty:
                current = merged
        output.append(current)
    return output


def _max_distance_to_line(polygon: Any, origin: np.ndarray, direction: np.ndarray) -> float:
    coords = np.asarray(polygon.exterior.coords, dtype=float)
    normal = np.asarray((-direction[1], direction[0]))
    return float(np.max(np.abs((coords - origin) @ normal)))


@dataclass(frozen=True, slots=True)
class UnobservedEdgeSettings:
    """Ceiling wall pieces on the edge of unscanned space (``indoor_unobserved_edges``)."""

    enabled: bool = True
    min_vertical_support: float = 0.05
    max_observed_share: float = 0.30
    min_height_bins: int = 2
    skip_low_bins: int = 2

    @classmethod
    def from_mapping(cls, data: Any) -> "UnobservedEdgeSettings":
        if isinstance(data, bool):
            return cls(enabled=data)
        if not isinstance(data, dict):
            return cls()
        defaults = cls()
        return cls(**{name: type(getattr(defaults, name))(data.get(name, getattr(defaults, name))) for name in cls.__dataclass_fields__})


def drop_unobserved_edge_pieces(pieces: list[WallFace], bitmask: np.ndarray | None, observed: np.ndarray, grid: RasterGrid, units_per_metre: float, settings: UnobservedEdgeSettings = UnobservedEdgeSettings()) -> tuple[list[WallFace], int]:
    """Drop ceiling-completed wall pieces that only mark where the scan ends.

    A ceiling room traced up to the edge of an unscanned area makes that edge look like
    a wall.  A real wall has vertical point evidence on its line (any class above the
    floor: wall, window frame, sill, lintel, curtain), even where windows or furniture
    hide most of it.  A piece is dropped when fewer than ``min_vertical_support`` of
    its samples have points in at least ``min_height_bins`` height bins (ignoring the
    ``skip_low_bins`` floor bins) within 6 cm AND the space beyond it is unobserved
    (observed share below ``max_observed_share``).
    """
    if not settings.enabled or not pieces or bitmask is None:
        return pieces, 0
    bits = np.asarray(bitmask).astype(np.int64) >> max(0, settings.skip_low_bins)
    count = np.zeros(bits.shape, dtype=np.int16)
    while bits.any():
        count += (bits & 1).astype(np.int16)
        bits >>= 1
    vertical = count >= settings.min_height_bins
    observed_mask = np.asarray(observed, dtype=bool)

    def lookup(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
        cols = np.floor((points[:, 0] - grid.min_x) / grid.pixel_size).astype(np.int64)
        rows = np.floor((grid.max_y - points[:, 1]) / grid.pixel_size).astype(np.int64)
        inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        values = np.zeros(len(points), dtype=bool)
        values[inside] = mask[rows[inside], cols[inside]]
        return values

    kept: list[WallFace] = []
    dropped = 0
    for piece in pieces:
        if piece.line is None or piece.line.length <= 1e-9:
            kept.append(piece)
            continue
        start, end = np.asarray(piece.line.coords[0], dtype=float), np.asarray(piece.line.coords[-1], dtype=float)
        direction = (end - start) / float(np.linalg.norm(end - start))
        normal = np.asarray((-direction[1], direction[0]))
        base = start[None, :] + np.linspace(0.05, 0.95, 19)[:, None] * (end - start)[None, :]
        support = np.zeros(len(base), dtype=bool)
        for offset in np.linspace(-0.06, 0.06, 7) * units_per_metre:
            support |= lookup(vertical, base + offset * normal)
        observed_sides = [float(np.mean(lookup(observed_mask, np.vstack([base + sign * offset * units_per_metre * normal for offset in (0.08, 0.15, 0.25)])))) for sign in (1.0, -1.0)]
        if float(support.mean()) < settings.min_vertical_support and min(observed_sides) < settings.max_observed_share:
            dropped += 1
            continue
        kept.append(piece)
    return kept, dropped


def close_corners_through_unobserved(faces: list[WallFace], observed: np.ndarray, grid: RasterGrid, units_per_metre: float, *, max_extension_m: float = 2.0, min_angle_deg: float = 30.0, free_end_m: float = 0.30, max_observed_share: float = 0.15) -> tuple[list[WallFace], int]:
    """Extend two free wall ends to their corner when the gap lies entirely in unscanned space.

    An unscanned corner leaves both walls ending short of it; the rooms then cut the
    corner off with a chord.  Two straight faces whose free ends point at the
    intersection of their lines (each extension at most ``max_extension_m``, corner
    angle at least ``min_angle_deg``) are extended to it when the extensions and their
    surroundings (+-0.1 / 0.2 m) are unobserved: nothing measured contradicts a wall
    there, and the walls themselves are measured on both sides of the gap.
    Faces elsewhere (observed space) are never touched.
    """
    lines = [face for face in faces if face.kind == "line" and face.line is not None and face.line.length > 1e-9]
    if len(lines) < 2:
        return faces, 0
    observed_mask = np.asarray(observed, dtype=bool)

    def observed_share(points: np.ndarray) -> float:
        cols = np.floor((points[:, 0] - grid.min_x) / grid.pixel_size).astype(np.int64)
        rows = np.floor((grid.max_y - points[:, 1]) / grid.pixel_size).astype(np.int64)
        inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        values = np.zeros(len(points), dtype=bool)
        values[inside] = observed_mask[rows[inside], cols[inside]]
        return float(values.mean()) if len(values) else 1.0

    from shapely.geometry import LineString, Point

    def free_end(face: WallFace, point: np.ndarray) -> bool:
        return not any(other is not face and other.line is not None and other.line.distance(Point(point)) <= free_end_m * units_per_metre for other in faces)

    minimum_sine = math.sin(math.radians(min_angle_deg))
    reach = free_end_m * units_per_metre
    limit = max_extension_m * units_per_metre
    extended = 0
    updates: dict[int, list[np.ndarray]] = {}
    for index, first in enumerate(lines):
        for second in lines[index + 1:]:
            a0, a1 = (np.asarray(c, dtype=float) for c in (first.line.coords[0], first.line.coords[-1]))
            b0, b1 = (np.asarray(c, dtype=float) for c in (second.line.coords[0], second.line.coords[-1]))
            da, db = a1 - a0, b1 - b0
            cross = float(da[0] * db[1] - da[1] * db[0])
            if abs(cross) / (float(np.linalg.norm(da)) * float(np.linalg.norm(db))) < minimum_sine:
                continue
            delta = b0 - a0
            t = float(delta[0] * db[1] - delta[1] * db[0]) / cross
            s = float(delta[0] * da[1] - delta[1] * da[0]) / cross
            corner = a0 + da * t
            plan = []
            for face, start, end, value, vector in ((first, a0, a1, t, da), (second, b0, b1, s, db)):
                length = float(np.linalg.norm(vector))
                beyond = -value * length if value < 0.0 else (value - 1.0) * length
                if beyond <= reach or beyond > limit:
                    break  # the ordinary corner closing handles short gaps; the corner must lie beyond this end
                endpoint = start if value < 0.0 else end
                if not free_end(face, endpoint):
                    break
                plan.append((face, endpoint, value < 0.0))
            if len(plan) != 2:
                continue
            samples = []
            for face, endpoint, _ in plan:
                direction = (corner - endpoint) / max(float(np.linalg.norm(corner - endpoint)), 1e-12)
                normal = np.asarray((-direction[1], direction[0]))
                along = endpoint[None, :] + np.linspace(0.15, 1.0, 12)[:, None] * (corner - endpoint)[None, :]
                for offset in (-0.20, -0.10, 0.0, 0.10, 0.20):
                    samples.append(along + offset * units_per_metre * normal)
            if observed_share(np.vstack(samples)) >= max_observed_share:
                continue
            for face, endpoint, at_start in plan:
                coords = updates.setdefault(id(face), [np.asarray(face.line.coords[0], dtype=float), np.asarray(face.line.coords[-1], dtype=float)])
                coords[0 if at_start else 1] = corner
            extended += 1
    if not extended:
        return faces, 0
    output = []
    for face in faces:
        if id(face) in updates:
            copy = replace(face)
            copy.line = LineString([tuple(point) for point in updates[id(face)]])
            output.append(copy)
        else:
            output.append(face)
    return output, extended


def extend_rooms_into_unobserved(rooms: list[Any], faces: list[WallFace], observed: np.ndarray, grid: RasterGrid, units_per_metre: float, *, max_observed_share: float = 0.15, min_area_m2: float = 0.10, max_share: float = 0.35) -> tuple[list[Any], int]:
    """Grow rooms up to their measured walls across unscanned pockets.

    A ceiling room ends where the scan ends.  Where the measured walls (after closing
    corners through unscanned space) enclose a pocket next to the room that holds no
    data at all (observed share below ``max_observed_share``), the pocket belongs to
    that room.  Pockets smaller than ``min_area_m2`` or larger than ``max_share`` of the
    room are left alone; rooms without such a pocket are returned unchanged.
    """
    from shapely.geometry import Polygon
    from shapely.ops import polygonize, unary_union

    lines = [face.line for face in faces if face.kind == "line" and face.line is not None and face.line.length > 1e-9]
    if not lines or not rooms:
        return rooms, 0
    extended_lines = []
    for line in lines:
        start, end = np.asarray(line.coords[0], dtype=float), np.asarray(line.coords[-1], dtype=float)
        direction = (end - start) / float(np.linalg.norm(end - start))
        extended_lines.append(type(line)([tuple(start - direction * 0.02 * units_per_metre), tuple(end + direction * 0.02 * units_per_metre)]))
    cells = [cell for cell in polygonize(unary_union(extended_lines)) if cell.area > 0.0]
    if not cells:
        return rooms, 0
    observed_mask = np.asarray(observed, dtype=bool)

    def observed_share(polygon: Any) -> float:
        x0, y0, x1, y1 = polygon.bounds
        c0, c1 = max(0, int((x0 - grid.min_x) / grid.pixel_size)), min(grid.width, int((x1 - grid.min_x) / grid.pixel_size) + 1)
        r0, r1 = max(0, int((grid.max_y - y1) / grid.pixel_size)), min(grid.height, int((grid.max_y - y0) / grid.pixel_size) + 1)
        if c1 <= c0 or r1 <= r0:
            return 1.0
        import shapely

        rr, cc = np.mgrid[r0:r1, c0:c1]
        xs, ys = grid.min_x + (cc + 0.5) * grid.pixel_size, grid.max_y - (rr + 0.5) * grid.pixel_size
        inside = shapely.contains_xy(polygon, xs, ys)
        return float(observed_mask[r0:r1, c0:c1][inside].mean()) if inside.any() else 1.0

    output: list[Any] = []
    grown = 0
    for room in rooms:
        if not isinstance(room, Polygon) or room.is_empty:
            output.append(room)
            continue
        current = room
        for cell in cells:
            if cell.intersection(room).area < 0.5 * room.area:
                continue
            # Thin slivers between the room outline and the wall faces are no pockets (morphological opening).
            remainder = cell.difference(room).buffer(-0.05 * units_per_metre, join_style=2).buffer(0.05 * units_per_metre, join_style=2).intersection(cell)
            for pocket in getattr(remainder, "geoms", [remainder]):
                if not isinstance(pocket, Polygon) or pocket.area < min_area_m2 * units_per_metre**2 or pocket.area > max_share * room.area:
                    continue
                if observed_share(pocket) < max_observed_share:
                    merged = unary_union([current, pocket])
                    if isinstance(merged, Polygon) and merged.is_valid:
                        current = merged
                        grown += 1
        output.append(current)
    return output, grown
