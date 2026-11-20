"""Sanitary blocks (WC, washbasin) always stand with their back on a wall and face the room.

The catalogue blocks are drawn with their mounting edge at the origin and +Y pointing into free space.  The
first placement (`snap_insert_to_wall`) works on measured evidence and may fall back to a block that is only
centred on the measured footprint, whose facing is arbitrary, or it snaps to a wall line that is not the final
wall.  This pass runs on the FINAL walls: the side of the block footprint that is nearest to a wall becomes its
back, the front looks away from that wall, and the back is snapped flush onto the wall face.  The result is the
single source of truth for DXF and IFC (``_facing_deg`` of the insert -> ``PlanFurniture.facing_deg``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

SANITARY_FACING_KINDS = frozenset({"WC_WALL", "WASHBASIN"})


@dataclass(frozen=True, slots=True)
class SanitaryFacingSettings:
    """Thresholds of the sanitary facing rule (metres / degrees)."""

    enabled: bool = True
    max_gap_m: float = 0.35  # back side farther than this from every wall: free standing, left alone
    tie_m: float = 0.03  # sides whose wall distance differs by less than this count as equally close
    snap_angle_deg: float = 12.0  # a block whose back edge is within this angle of the wall face is rotated onto it
    segment_tie_m: float = 0.02  # wall segments this close to the nearest one compete (corner): the most parallel one wins


def _segments(geometry: BaseGeometry | None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Straight boundary segments of a wall polygon / line collection."""
    if geometry is None or geometry.is_empty:
        return []
    kind = geometry.geom_type
    if kind in ("MultiPolygon", "GeometryCollection", "MultiLineString"):
        return [segment for part in geometry.geoms for segment in _segments(part)]
    if kind == "Polygon":
        rings = [geometry.exterior, *geometry.interiors]
    elif kind in ("LineString", "LinearRing"):
        rings = [geometry]
    else:
        return []
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for ring in rings:
        coords = np.asarray(ring.coords, dtype=float)
        for first, second in zip(coords[:-1], coords[1:], strict=True):
            if float(np.hypot(*(second - first))) > 1e-9:
                output.append((first, second))
    return output


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.hypot(*vector)), 1e-12)


def _side_distance(walls: BaseGeometry, segments: list[tuple[np.ndarray, np.ndarray]], midpoint: np.ndarray, axis: np.ndarray, cos_limit: float = 0.866) -> float:
    """Distance of an edge midpoint to the wall: 0 inside a wall body, else to the nearest wall segment parallel (30 degrees) to the edge."""
    point = Point(float(midpoint[0]), float(midpoint[1]))
    if walls.area > 0.0 and walls.contains(point):
        return 0.0
    best = math.inf
    for first, second in segments:
        if abs(float(np.dot(_unit(second - first), axis))) >= cos_limit:
            best = min(best, float(LineString([tuple(first), tuple(second)]).distance(point)))
    return best


def _wall_behind(walls: BaseGeometry, segment: tuple[np.ndarray, np.ndarray], facing: np.ndarray, probe: float) -> bool:
    """True when the wall body lies on the far side of ``segment`` seen from the direction the block faces."""
    middle = (segment[0] + segment[1]) / 2.0
    behind = middle - facing * probe
    ahead = middle + facing * probe
    return bool(walls.contains(Point(float(behind[0]), float(behind[1]))) and not walls.contains(Point(float(ahead[0]), float(ahead[1]))))


def _nearest_face(segments: list[tuple[np.ndarray, np.ndarray]], point: np.ndarray, back_axis: np.ndarray, tie: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Wall segment behind ``point``: the nearest one; at a corner the one most parallel to the back edge, then the longest."""
    scored: list[tuple[float, float, tuple[np.ndarray, np.ndarray]]] = []
    for first, second in segments:
        line = LineString([tuple(first), tuple(second)])
        scored.append((float(line.distance(Point(float(point[0]), float(point[1])))), float(line.length), (first, second)))
    if not scored:
        return None
    nearest = min(item[0] for item in scored)
    close = [item for item in scored if item[0] <= nearest + tie]
    # Round the alignment: 1 mm of angular noise must not beat a longer, equally aligned face.
    best = max(close, key=lambda item: (round(abs(float(np.dot(_unit(item[2][1] - item[2][0]), back_axis))), 3), item[1]))
    return best[2]


def orient_sanitary_inserts(layers: dict[str, list[dict[str, Any]]], walls: BaseGeometry | None, units_per_metre: float, settings: SanitaryFacingSettings) -> dict[str, Any]:
    """Turn / move every WC and washbasin insert so its back is on the nearest wall face; returns a report.

    ``walls`` are the final wall bodies (or wall face lines) in plan units.  An insert is changed in place:
    ``point`` / ``rotation`` and ``_facing_deg`` (direction the front looks, degrees counter-clockwise from +X).
    """
    report: dict[str, Any] = {"sanitary_facing_checked": 0, "sanitary_facing_flipped": 0, "sanitary_facing_rotated": 0, "sanitary_facing_free": []}
    if not settings.enabled or walls is None or walls.is_empty:
        return report
    upm = max(float(units_per_metre), 1e-9)
    segments = _segments(walls)
    if not segments:
        return report
    gap = settings.max_gap_m * upm
    tie = settings.tie_m * upm
    for entity in layers.get("P-SANI", []):
        catalogue = entity.get("_catalogue") or {}
        if entity.get("type") != "insert" or catalogue.get("kind") not in SANITARY_FACING_KINDS:
            continue
        report["sanitary_facing_checked"] += 1
        scale = float(entity.get("xscale", upm))
        width, depth = float(catalogue["width_m"]) * scale, float(catalogue["depth_m"]) * scale
        origin = np.asarray(entity["point"], dtype=float)
        rotation = math.radians(float(entity.get("rotation", 0.0)))
        x_axis = np.asarray((math.cos(rotation), math.sin(rotation)))
        y_axis = np.asarray((-x_axis[1], x_axis[0]))
        centre = origin + y_axis * depth / 2.0
        # Candidate backs in preference order: the current back, the opposite width edge, then the two long sides.
        candidates = [
            (centre - y_axis * depth / 2.0, y_axis, x_axis),
            (centre + y_axis * depth / 2.0, -y_axis, x_axis),
            (centre - x_axis * width / 2.0, x_axis, y_axis),
            (centre + x_axis * width / 2.0, -x_axis, y_axis),
        ]
        distances = [_side_distance(walls, segments, middle, axis) for middle, _facing, axis in candidates]
        # The block keeps its size and shape: a width edge (current back or its opposite) is the back whenever one is within reach of a
        # wall; a long side becomes the back (block turned 90 degrees) only when no width edge is.
        usable = [0, 1] if min(distances[:2]) <= gap else [2, 3]
        nearest = min(distances[i] for i in usable)
        if nearest > gap:
            report["sanitary_facing_free"].append({"name": str(entity.get("name")), "point": [round(float(origin[0]), 3), round(float(origin[1]), 3)]})
            continue
        index = next(i for i in usable if distances[i] <= nearest + tie)
        middle, facing, back_axis = candidates[index]
        # Only a face with the wall body BEHIND it (opposite to the facing) can carry the back: the far face of a wall the block
        # overlaps, or the face of a wall on the room side, is never the one.
        behind = [segment for segment in segments if _wall_behind(walls, segment, facing, 0.02 * upm)] if walls.area > 0.0 else segments
        face = _nearest_face(behind or segments, middle, back_axis, settings.segment_tie_m * upm)
        if face is None:
            continue
        tangent = _unit(face[1] - face[0])
        normal = np.asarray((-tangent[1], tangent[0]))
        if float(np.dot(normal, facing)) < 0.0:
            normal = -normal
        angle = math.degrees(math.acos(float(np.clip(abs(float(np.dot(tangent, back_axis))), 0.0, 1.0))))
        if angle <= settings.snap_angle_deg:
            facing = normal  # the face is the reference: block rotated square onto it
        # Flush: back edge midpoint onto the wall face line (no gap, no overlap into the wall).
        offset = float(np.dot(middle - face[0], normal))
        back_point = middle - normal * offset if angle <= settings.snap_angle_deg and abs(offset) <= gap else middle
        new_x = np.asarray((facing[1], -facing[0]))
        new_origin = back_point
        changed_back = index != 0
        if changed_back:
            report["sanitary_facing_flipped" if index == 1 else "sanitary_facing_rotated"] += 1
        entity["point"] = [float(new_origin[0]), float(new_origin[1])]
        entity["rotation"] = math.degrees(math.atan2(float(new_x[1]), float(new_x[0]))) % 360.0
        entity["_facing_deg"] = math.degrees(math.atan2(float(facing[1]), float(facing[0]))) % 360.0
        catalogue["wall_snapped"] = True
        catalogue["back_to_wall"] = True
    return report
