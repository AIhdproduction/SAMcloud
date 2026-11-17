"""Pure-geometry tidy-up of the indoor A-WALL polylines.

The measured wall faces of an indoor plan contain collinear duplicates (two
detectors fitting the same face), tiny fragments inside longer faces and open
joints a few centimetres wide where two walls should meet.  This module merges
the duplicates, re-applies the door/window cuts, closes short perpendicular
joints and drops covered fragments.  Every distance is a multiple of
``units_per_metre``; there are no project constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point

_MERGE_ANGLE_DEG = 1.0
_MERGE_OFFSET_M = 0.02
_MERGE_GAP_M = 0.01
_OPENING_TOLERANCE_M = 0.15
_OPENING_AXIS_DEG = 10.0
_MIN_PIECE_M = 0.02
_FREE_DISTANCE_M = 0.01
_JOINT_REACH_M = 0.30
_JOINT_ANGLE_DEG = 2.0
_OPENING_MARGIN_M = 0.05
_FRAGMENT_M = 0.05
_FRAGMENT_ANGLE_DEG = 2.0
_FRAGMENT_OFFSET_M = 0.03
_FRAGMENT_TOLERANCE_M = 0.01


@dataclass(slots=True)
class _Segment:
    start: np.ndarray
    end: np.ndarray
    entity: dict[str, Any]
    order: int
    modified: bool = False
    members: list[int] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        vector = self.end - self.start
        return vector / max(float(np.linalg.norm(vector)), 1e-12)


def tidy_wall_entities(entities: list[dict], openings: list[object], units_per_metre: float) -> tuple[list[dict], dict[str, int | float]]:
    """Merge duplicate wall faces, keep openings open, close joints, drop fragments."""
    upm = float(units_per_metre)
    stats: dict[str, int | float] = {"walls_merged": 0, "walls_joints_closed": 0, "walls_extension_m": 0.0, "walls_fragments_dropped": 0}
    passthrough: list[tuple[int, dict]] = []
    segments: list[_Segment] = []
    for index, entity in enumerate(entities):
        segment = _as_segment(entity, index)
        if segment is None:
            passthrough.append((index, entity))
        else:
            segments.append(segment)
    if not segments:
        return list(entities), stats

    segments, merged = _merge_collinear(segments, upm)
    stats["walls_merged"] = merged
    segments = _recut_openings(segments, openings, upm)
    others = [geometry for _, entity in passthrough if (geometry := _entity_geometry(entity)) is not None]
    closed, extension = _close_joints(segments, others, openings, upm)
    stats["walls_joints_closed"] = closed
    stats["walls_extension_m"] = extension / max(upm, 1e-12)
    segments, dropped = _drop_covered_fragments(segments, upm)
    stats["walls_fragments_dropped"] = dropped

    emitted: list[tuple[int, int, dict]] = [(order, 0, entity) for order, entity in passthrough]
    for sequence, segment in enumerate(segments):
        emitted.append((segment.order, 1 + sequence, _emit(segment)))
    emitted.sort(key=lambda item: (item[0], item[1]))
    return [entity for _, _, entity in emitted], stats


def _as_segment(entity: dict, order: int) -> _Segment | None:
    if entity.get("type") != "polyline" or entity.get("closed"):
        return None
    points = entity.get("points", [])
    if len(points) != 2:
        return None
    start = np.asarray(points[0][:2], dtype=float)
    end = np.asarray(points[1][:2], dtype=float)
    if float(np.linalg.norm(end - start)) <= 1e-9:
        return None
    return _Segment(start, end, entity, order)


def _emit(segment: _Segment) -> dict:
    if not segment.modified:
        return segment.entity
    output = dict(segment.entity)
    output["points"] = [[float(segment.start[0]), float(segment.start[1])], [float(segment.end[0]), float(segment.end[1])]]
    return output


def _entity_geometry(entity: dict) -> LineString | None:
    try:
        if entity.get("type") == "polyline":
            points = [(float(p[0]), float(p[1])) for p in entity.get("points", [])]
            if entity.get("closed") and len(points) >= 3:
                points.append(points[0])
            return LineString(points) if len(points) >= 2 else None
        if entity.get("type") == "arc":
            centre = entity["center"]
            radius = float(entity["radius"])
            first, last = math.radians(float(entity["start_angle"])), math.radians(float(entity["end_angle"]))
            if last < first:
                last += 2.0 * math.pi
            angles = np.linspace(first, last, 16)
            return LineString([(float(centre[0]) + radius * math.cos(a), float(centre[1]) + radius * math.sin(a)) for a in angles])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return None


def _parallel(first: np.ndarray, second: np.ndarray, tolerance_deg: float) -> bool:
    cross = abs(float(first[0] * second[1] - first[1] * second[0]))
    return math.degrees(math.asin(min(1.0, cross))) <= tolerance_deg


def _opening_axis(opening: object) -> np.ndarray | None:
    angle = getattr(opening, "axis_angle", None)
    if angle is None:
        return None
    return np.asarray((math.cos(float(angle)), math.sin(float(angle))))


# --------------------------------------------------------------------------
# 1. collinear duplicates
# --------------------------------------------------------------------------

def _mergeable(first: _Segment, second: _Segment, upm: float) -> bool:
    longer, shorter = (first, second) if first.length >= second.length else (second, first)
    direction = longer.direction
    if not _parallel(direction, shorter.direction, _MERGE_ANGLE_DEG):
        return False
    normal = np.asarray((-direction[1], direction[0]))
    origin = longer.start
    t_short = [float(np.dot(point - origin, direction)) for point in (shorter.start, shorter.end)]
    o_short = [float(np.dot(point - origin, normal)) for point in (shorter.start, shorter.end)]
    low_s, high_s = min(t_short), max(t_short)
    gap = max(low_s - longer.length, 0.0 - high_s)
    if gap > _MERGE_GAP_M * upm:
        return False
    low, high = max(0.0, low_s), min(longer.length, high_s)
    middle = (low + high) / 2.0 if high >= low else (high + low) / 2.0
    span = t_short[1] - t_short[0]
    if abs(span) < 1e-12:
        offset = o_short[0]
    else:
        offset = o_short[0] + (o_short[1] - o_short[0]) * (middle - t_short[0]) / span
    return abs(offset) <= _MERGE_OFFSET_M * upm


def _merge_collinear(segments: list[_Segment], upm: float) -> tuple[list[_Segment], int]:
    count = len(segments)
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for i in range(count):
        for j in range(i + 1, count):
            if find(i) != find(j) and _mergeable(segments[i], segments[j], upm):
                parent[find(j)] = find(i)
    clusters: dict[int, list[int]] = {}
    for index in range(count):
        clusters.setdefault(find(index), []).append(index)
    output: list[_Segment] = []
    merged = 0
    for indices in sorted(clusters.values(), key=min):
        members = [segments[index] for index in indices]
        if len(members) == 1:
            output.append(members[0])
            continue
        merged += len(members) - 1
        longest = max(members, key=lambda item: item.length)
        reference = longest.direction
        weighted = np.zeros(2)
        for member in members:
            direction = member.direction
            if float(np.dot(direction, reference)) < 0.0:
                direction = -direction
            weighted += member.length * direction
        direction = weighted / max(float(np.linalg.norm(weighted)), 1e-12)
        normal = np.asarray((-direction[1], direction[0]))
        origin = (longest.start + longest.end) / 2.0
        total = sum(member.length for member in members)
        offset = sum(member.length * float(np.dot((member.start + member.end) / 2.0 - origin, normal)) for member in members) / max(total, 1e-12)
        positions = [float(np.dot(point - origin, direction)) for member in members for point in (member.start, member.end)]
        first_point = origin + direction * min(positions) + normal * offset
        last_point = origin + direction * max(positions) + normal * offset
        if float(np.dot(reference, last_point - first_point)) < 0.0:
            first_point, last_point = last_point, first_point
        output.append(_Segment(first_point, last_point, longest.entity, min(member.order for member in members), modified=True))
    return output, merged


# --------------------------------------------------------------------------
# 2. opening cuts
# --------------------------------------------------------------------------

def _opening_interval(start: np.ndarray, end: np.ndarray, opening: object, upm: float) -> tuple[float, float] | None:
    """Return the opening span in arc length along start->end, or None if it does not apply."""
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        return None
    direction = vector / length
    axis = _opening_axis(opening)
    if axis is not None and not _parallel(direction, axis, _OPENING_AXIS_DEG):
        return None
    normal = np.asarray((-direction[1], direction[0]))
    offset = np.asarray(getattr(opening, "center"), dtype=float)[:2] - start
    if abs(float(np.dot(offset, normal))) > _OPENING_TOLERANCE_M * upm:
        return None
    centre = float(np.dot(offset, direction))
    half = float(getattr(opening, "width")) / 2.0
    return centre - half, centre + half


def _recut_openings(segments: list[_Segment], openings: list[object], upm: float) -> list[_Segment]:
    output: list[_Segment] = []
    for segment in segments:
        length = segment.length
        cuts: list[tuple[float, float]] = []
        for opening in openings:
            interval = _opening_interval(segment.start, segment.end, opening, upm)
            if interval is None:
                continue
            low, high = max(0.0, interval[0]), min(length, interval[1])
            if high - low > 1e-6 * max(upm, 1e-12):
                cuts.append((low, high))
        if not cuts:
            output.append(segment)
            continue
        direction = segment.direction
        cursor = 0.0
        pieces: list[tuple[float, float]] = []
        for low, high in sorted(cuts):
            if low - cursor > 1e-9:
                pieces.append((cursor, low))
            cursor = max(cursor, high)
        if length - cursor > 1e-9:
            pieces.append((cursor, length))
        for low, high in pieces:
            if high - low < _MIN_PIECE_M * upm:
                continue
            output.append(_Segment(segment.start + direction * low, segment.start + direction * high, segment.entity, segment.order, modified=True))
    return output


# --------------------------------------------------------------------------
# 3. joints
# --------------------------------------------------------------------------

def _blocked_by_opening(first: np.ndarray, second: np.ndarray, openings: list[object], upm: float) -> bool:
    """True when the extension first->second would enter any opening span."""
    extension = LineString([tuple(first), tuple(second)])
    length = float(np.linalg.norm(second - first))
    for opening in openings:
        centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        half = float(getattr(opening, "width")) / 2.0 + _OPENING_MARGIN_M * upm
        axis = _opening_axis(opening)
        if axis is None:
            if Point(float(centre[0]), float(centre[1])).distance(extension) <= half:
                return True
        elif extension.intersects(LineString([tuple(centre - axis * half), tuple(centre + axis * half)])):
            return True
        interval = _opening_interval(first, second, opening, upm)
        if interval is not None:
            margin = _OPENING_MARGIN_M * upm
            if interval[0] - margin <= length and interval[1] + margin >= 0.0:
                return True
    return False


def _is_free(point: np.ndarray, owner: int, segments: list[_Segment], others: list[LineString], upm: float) -> bool:
    limit = _FREE_DISTANCE_M * upm
    probe = Point(float(point[0]), float(point[1]))
    for index, other in enumerate(segments):
        if index != owner and probe.distance(LineString([tuple(other.start), tuple(other.end)])) <= limit:
            return False
    return all(probe.distance(geometry) > limit for geometry in others)


def _close_joints(segments: list[_Segment], others: list[LineString], openings: list[object], upm: float) -> tuple[int, float]:
    reach = _JOINT_REACH_M * upm
    perpendicular = math.sin(math.radians(90.0 - _JOINT_ANGLE_DEG))
    closed = 0
    extension_total = 0.0
    for i, segment in enumerate(segments):
        for end_index in (0, 1):
            point = segment.end if end_index else segment.start
            tail = segment.start if end_index else segment.end
            if not _is_free(point, i, segments, others, upm):
                continue
            vector = point - tail
            u = vector / max(float(np.linalg.norm(vector)), 1e-12)
            best: tuple[float, int, np.ndarray, int | None] | None = None
            for j, other in enumerate(segments):
                if j == i:
                    continue
                d = other.direction
                cross = float(u[0] * d[1] - u[1] * d[0])
                if abs(cross) < perpendicular:
                    continue
                s = ((other.start[0] - point[0]) * d[1] - (other.start[1] - point[1]) * d[0]) / cross
                if s < -1e-9 or s > reach:
                    continue
                hit = point + u * max(s, 0.0)
                t = float(np.dot(hit - other.start, d))
                other_end: int | None = None
                if t < 0.0:
                    if -t > reach:
                        continue
                    other_end = 0
                elif t > other.length:
                    if t - other.length > reach:
                        continue
                    other_end = 1
                if other_end is not None:
                    far = other.start if other_end == 0 else other.end
                    if not _is_free(far, j, segments, others, upm):
                        continue
                if _blocked_by_opening(point, hit, openings, upm):
                    continue
                if other_end is not None:
                    far = other.start if other_end == 0 else other.end
                    if _blocked_by_opening(far, hit, openings, upm):
                        continue
                if best is None or s < best[0]:
                    best = (s, j, hit, other_end)
            if best is None:
                continue
            s, j, hit, other_end = best
            if end_index:
                segment.end = hit
            else:
                segment.start = hit
            segment.modified = True
            extension_total += s
            if other_end is not None:
                other = segments[j]
                far = other.start if other_end == 0 else other.end
                extension_total += float(np.linalg.norm(hit - far))
                if other_end == 0:
                    other.start = hit
                else:
                    other.end = hit
                other.modified = True
            closed += 1
    return closed, extension_total


# --------------------------------------------------------------------------
# 4. covered fragments
# --------------------------------------------------------------------------

def _drop_covered_fragments(segments: list[_Segment], upm: float) -> tuple[list[_Segment], int]:
    limit = _FRAGMENT_M * upm
    tolerance = _FRAGMENT_TOLERANCE_M * upm
    kept: list[_Segment] = []
    dropped = 0
    removed: set[int] = set()
    for index, segment in enumerate(segments):
        covered = False
        if segment.length < limit:
            for other_index, other in enumerate(segments):
                if other_index == index or other_index in removed or other.length < segment.length:
                    continue
                direction = other.direction
                if not _parallel(direction, segment.direction, _FRAGMENT_ANGLE_DEG):
                    continue
                normal = np.asarray((-direction[1], direction[0]))
                offsets = [abs(float(np.dot(point - other.start, normal))) for point in (segment.start, segment.end)]
                if max(offsets) > _FRAGMENT_OFFSET_M * upm:
                    continue
                positions = [float(np.dot(point - other.start, direction)) for point in (segment.start, segment.end)]
                if min(positions) >= -tolerance and max(positions) <= other.length + tolerance:
                    covered = True
                    break
        if covered:
            removed.add(index)
            dropped += 1
        else:
            kept.append(segment)
    return kept, dropped
