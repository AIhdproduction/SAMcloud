"""Tall kitchen rows (fridge, oven tower, tall cabinets) from measured furniture-class points.

A tall row is seen from the room: its front face (and sometimes the top) is scanned, the rest of the body - back to the wall - is
hidden.  The detector works on the cloud points of the furniture classes (+ COLMAP depth points) around the kitchen:

1. a plan cell (``cell_m``) is *tall* when it holds points above ``tall_min_m`` AND points in the mid band (``mid_low_m`` ..
   ``mid_high_m``): a hanging upper cabinet leaves that band free, a tall unit does not;
2. connected tall cells near the kitchen run are decomposed into straight front-face segments along the wall axes (an L-shaped
   cluster = two rows), longest first, an overlapping duplicate (the end face of a row) is dropped;
3. each segment is completed from its front face to the wall behind it (within ``wall_reach_m``, never deeper than ``max_depth_m``)
   and along the row to the next wall / kitchen leg (within ``along_reach_m``); without a scanned wall behind it (a partition
   hidden between two rows of cabinets) the row is as deep as its own scanned end face, and without that it is no row; a face
   stands at the median of its points, and a scanned end face is where the row ends;
4. the row top is the robust measured top of its points.

``split_row`` cuts a row into units around fixed items (a fridge) that already own part of it.  All coordinates are plan-local units;
``units_per_metre`` converts the metric thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union


@dataclass(frozen=True, slots=True)
class KitchenRowSettings:
    """Metric thresholds of the tall-row detection (config ``kitchen_rows`` in bim_export.yaml)."""

    enabled: bool = True
    cell_m: float = 0.05
    # a tall cell holds points higher than tall_min_m and points in the mid band (a hanging unit leaves the mid band free)
    tall_min_m: float = 1.80
    tall_max_m: float = 2.70
    mid_low_m: float = 1.00
    mid_high_m: float = 1.30
    min_cell_points: int = 3
    # only clusters this close to a kitchen run are kitchen rows
    kitchen_reach_m: float = 1.20
    # a front face is at least this long; runs split by up to gap_cells empty cells stay one run
    min_run_m: float = 0.30
    gap_cells: int = 2
    # the wall behind the front face (the row fills front -> wall) within this distance; never deeper than max_depth_m
    wall_reach_m: float = 0.90
    max_depth_m: float = 0.80
    # a row end snaps to a wall / kitchen leg within this distance along the row
    along_reach_m: float = 0.40
    min_length_m: float = 0.30
    min_area_m2: float = 0.15
    # a segment overlapping an accepted one by this share of its own area is a face of the same body (end face)
    duplicate_share: float = 0.30
    # a row piece shorter than this next to a fixed item (fridge) is absorbed by it
    min_unit_m: float = 0.30
    min_top_points: int = 8
    # a row of fewer cells than this fraction touching the mid band is dropped
    ceiling_margin_m: float = 0.02
    # No wall behind the front (a thin partition hidden between two rows of cabinets is never scanned): the row is as deep as its own
    # scanned end face.  A cabinet body is 0.30 m (shelf unit) to 0.80 m (max_depth_m) deep; the end face starts at the end of the
    # front and at the front line within end_face_join_m (two cells of the 0.05 m grid plus the 3 cm scan noise of a face).
    end_face: bool = True
    end_face_min_m: float = 0.30
    end_face_join_m: float = 0.15
    # A face lies at the median of its points above worktop level (mid_low_m and higher: no clutter of the floor zone), not on the
    # outer side of its cells: a face seen 2-3 cells thick (handles, an oven door, 3 cm noise) put the front 3-13 cm into the room
    # (against a robust line fit of the cloud: v12 3.0 and 4.5 cm, v13 10.0 and 12.6 cm).
    face_fit: bool = True


@dataclass(slots=True)
class TallRow:
    """One detected row: plan polygon, measured top (metres above the floor), along axis angle and front side."""

    polygon: Polygon
    top_m: float
    angle: float  # direction of the row (radians) in plan
    cells: int = 0
    front: str = ""


def dominant_angle(wall_lines: list[LineString], centre: tuple[float, float]) -> float:
    """Wall direction (radians, folded into 0..pi/2) of the nearest wall segment; 0 without walls."""
    from samcloud.bim.furniture_footprint import nearest_wall_angle

    angle = nearest_wall_angle(np.asarray(centre, dtype=float), wall_lines)
    return 0.0 if angle is None else float(angle % (math.pi / 2.0))


def _runs(mask: np.ndarray, along_axis: int, min_cells: int, gap: int) -> list[tuple[int, int, int]]:
    """(line index, first, last) of maximal runs of True along ``along_axis`` (empty gaps up to ``gap`` cells are bridged)."""
    runs: list[tuple[int, int, int]] = []
    lines = mask.shape[1 - along_axis]
    for line in range(lines):
        values = mask[:, line] if along_axis == 0 else mask[line, :]
        index = np.flatnonzero(values)
        if not len(index):
            continue
        splits = np.flatnonzero(np.diff(index) > gap + 1)
        for chunk in np.split(index, splits + 1):
            if int(chunk[-1] - chunk[0] + 1) >= min_cells:
                runs.append((line, int(chunk[0]), int(chunk[-1])))
    return runs


def _group_runs(runs: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    """Runs on neighbouring lines (<= 2 apart) that overlap along the line belong to one face (a face is seen 1-3 cells thick)."""
    parent = list(range(len(runs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(len(runs)):
        for second in range(first + 1, len(runs)):
            a, b = runs[first], runs[second]
            if abs(a[0] - b[0]) <= 2 and min(a[2], b[2]) - max(a[1], b[1]) + 1 >= 0.5 * min(a[2] - a[1] + 1, b[2] - b[1] + 1):
                parent[find(first)] = find(second)
    groups: dict[int, list[tuple[int, int, int]]] = {}
    for index, run in enumerate(runs):
        groups.setdefault(find(index), []).append(run)
    return list(groups.values())


def _ray(origin: np.ndarray, direction: np.ndarray, length: float, obstacle: Any) -> float | None:
    """Distance from ``origin`` along ``direction`` to the first contact with ``obstacle`` within ``length``."""
    if obstacle is None or obstacle.is_empty:
        return None
    line = LineString([tuple(origin), tuple(origin + direction * length)])
    hit = line.intersection(obstacle)
    if hit.is_empty:
        return None
    if Point(*origin).intersects(obstacle):
        return 0.0
    return float(Point(*origin).distance(hit))


def detect_tall_rows(
    points: np.ndarray,
    wall_union: Any,
    kitchen_area: Polygon,
    units_per_metre: float,
    angle: float,
    settings: KitchenRowSettings = KitchenRowSettings(),
    obstacles: Any = None,
) -> list[TallRow]:
    """Tall rows near ``kitchen_area`` from ``points`` (x, y, z in plan units, z above the floor).

    ``wall_union`` = wall bodies, ``obstacles`` = extra stops along a row (kitchen legs).  ``angle`` = wall axis (radians).
    """
    upm = max(float(units_per_metre), 1e-9)
    pts = np.asarray(points, dtype=float)
    if not settings.enabled or kitchen_area is None or kitchen_area.is_empty or len(pts) < settings.min_cell_points * 4:
        return []
    reach = settings.kitchen_reach_m * upm
    minx, miny, maxx, maxy = kitchen_area.buffer(reach + 1.0 * upm).bounds
    near = (pts[:, 0] >= minx) & (pts[:, 0] <= maxx) & (pts[:, 1] >= miny) & (pts[:, 1] <= maxy)
    pts = pts[near]
    if len(pts) < settings.min_cell_points * 4:
        return []
    cos, sin = math.cos(angle), math.sin(angle)
    u = pts[:, 0] * cos + pts[:, 1] * sin
    v = -pts[:, 0] * sin + pts[:, 1] * cos
    z = pts[:, 2] / upm
    step = settings.cell_m * upm
    u0, v0 = float(u.min()) - 2 * step, float(v.min()) - 2 * step
    nu, nv = int((u.max() - u0) / step) + 3, int((v.max() - v0) / step) + 3
    iu, iv = ((u - u0) / step).astype(np.int64), ((v - v0) / step).astype(np.int64)
    tall = np.zeros((nv, nu), dtype=np.int32)
    mid = np.zeros((nv, nu), dtype=np.int32)
    is_tall = (z >= settings.tall_min_m) & (z <= settings.tall_max_m)
    is_mid = (z >= settings.mid_low_m) & (z <= settings.mid_high_m)
    np.add.at(tall, (iv[is_tall], iu[is_tall]), 1)
    np.add.at(mid, (iv[is_mid], iu[is_mid]), 1)
    mask = (tall >= settings.min_cell_points) & (mid >= 2)
    if not mask.any():
        return []
    # the mid band of a front face is seen sparsely: close one-cell holes before looking for faces
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool)) | mask
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    wall_stop = unary_union([wall_union]) if wall_union is not None and not wall_union.is_empty else None
    stops = unary_union([item for item in (wall_stop, obstacles) if item is not None and not item.is_empty]) if (wall_stop is not None or (obstacles is not None and not obstacles.is_empty)) else None
    rows: list[TallRow] = []
    accepted: list[Polygon] = []
    min_cells = max(2, int(round(settings.min_run_m * upm / step)))

    def to_plan(a: float, b: float) -> tuple[float, float]:
        return (a * cos - b * sin, a * sin + b * cos)

    segments: list[dict[str, Any]] = []
    for number in range(1, count + 1):
        member = labels == number
        cells = int(member.sum())
        if cells < 4:
            continue
        rr, cc = np.nonzero(member)
        cluster = box(u0 + cc.min() * step, v0 + rr.min() * step, u0 + (cc.max() + 1) * step, v0 + (rr.max() + 1) * step)
        cluster_plan = affinity.rotate(cluster, angle, origin=(0.0, 0.0), use_radians=True)
        if cluster_plan.distance(kitchen_area) > reach:
            continue
        for along_axis, name in ((1, "u"), (0, "v")):
            for group in _group_runs(_runs(member, along_axis, min_cells, settings.gap_cells)):
                lines = [run[0] for run in group]
                first, last = min(run[1] for run in group), max(run[2] for run in group)
                segments.append({
                    "axis": name, "lines": (min(lines), max(lines) + 1), "along": (first, last + 1), "cells": sum(run[2] - run[1] + 1 for run in group),
                    "length": (last + 1 - first) * step, "member": number,
                })
    segments.sort(key=lambda item: (-item["length"], item["axis"], item["lines"], item["along"]))

    def spans(segment: dict[str, Any]) -> tuple[float, float, float, float]:
        """(a0, a1, c_low, c_high): along range and the two sides of the face cells, wall-frame units."""
        along0, across0 = (u0, v0) if segment["axis"] == "u" else (v0, u0)
        return (along0 + segment["along"][0] * step, along0 + segment["along"][1] * step,
                across0 + segment["lines"][0] * step, across0 + segment["lines"][1] * step)

    above = z >= settings.mid_low_m

    def cell_points(segment: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """(along, across) of the points above worktop level in the cells of a face."""
        a0, a1, c_low, c_high = spans(segment)
        along, across = (u, v) if segment["axis"] == "u" else (v, u)
        inside = above & (along >= a0) & (along < a1) & (across >= c_low) & (across < c_high)
        return along[inside], across[inside]

    def plane(segment: dict[str, Any]) -> float | None:
        """Where the scanned face stands across its cells (None with too few points to say)."""
        _, across = cell_points(segment)
        return float(np.median(across)) if settings.face_fit and len(across) >= settings.min_top_points else None

    # which side holds the wall behind each face: sample the rays at 20 / 50 / 80 % of the face
    for segment in segments:
        axis = segment["axis"]
        a0, a1, c_low, c_high = spans(segment)
        best: tuple[float, int] | None = None
        for sign in (1, -1):
            edge = c_low if sign == 1 else c_high
            hits: list[float] = []
            for share in (0.2, 0.5, 0.8):
                position = a0 + share * (a1 - a0)
                origin = np.asarray(to_plan(position, edge) if axis == "u" else to_plan(edge, position), dtype=float)
                direction = np.asarray(to_plan(0.0, sign) if axis == "u" else to_plan(sign, 0.0), dtype=float)
                hit = _ray(origin, direction, settings.wall_reach_m * upm, wall_union)
                if hit is not None:
                    hits.append(hit)
            if len(hits) >= 2:
                distance = float(np.median(hits))
                if best is None or distance < best[0]:
                    best = (distance, sign)
        segment["wall"] = best
    if settings.end_face:
        # on which side of each face its own body lies, as far as the face says so itself (its wall, else its own end faces)
        own = [None if segment["wall"] is not None else _end_face_depth(segment, segments, step, settings, upm) for segment in segments]
        for segment, found in zip(segments, own, strict=True):
            segment["side"] = segment["wall"][1] if segment["wall"] is not None else (None if found is None else found[1])
    for segment in segments:
        axis = segment["axis"]
        a0, a1, c_low, c_high = spans(segment)
        best = segment["wall"]
        end_faces: list[dict[str, Any]] = []
        walled = best is not None
        if not walled and settings.end_face:
            best = _end_face_depth(segment, segments, step, settings, upm, end_faces)
        elif walled and settings.end_face and settings.face_fit:
            # a row against a wall: its scanned end face still says where the row ends
            _end_face_depth(segment, segments, step, settings, upm, end_faces, side=best[1], limit=best[0])
        if best is None:
            continue
        distance, sign = best
        depth = min(distance, settings.max_depth_m * upm)
        front = c_low if sign == 1 else c_high
        far = front + sign * depth
        ends = [a0, a1]
        measured_ends: set[int] = set()
        if settings.face_fit:
            # the body ends where its end face ends (without a wall behind it), and at the plane of that end face
            reaches = []
            for other in end_faces:
                along_other, across_other = cell_points(other)
                if len(along_other) < settings.min_top_points:
                    continue
                # the end face reaches somewhere into its last cell: half a cell less than the cell side, on average
                low_other, high_other, _, _ = spans(other)
                reaches.append(high_other - 0.5 * step if sign == 1 else low_other + 0.5 * step)
                side = float(np.median(across_other))
                nearest = 0 if abs(side - a0) <= abs(side - a1) else 1
                ends[nearest] = side
                measured_ends.add(nearest)
            if reaches and not walled:
                far = min(reaches) if sign == 1 else max(reaches)
            measured = plane(segment)
            if measured is not None and sign * (far - measured) > 0.0:
                front = measured
            if abs(far - front) > settings.max_depth_m * upm:
                far = front + sign * settings.max_depth_m * upm
        c0, c1 = (min(front, far), max(front, far))
        # along the row: snap each end to a wall / kitchen leg within along_reach_m (an end with its scanned end face is measured)
        centre_across = (c0 + c1) / 2.0
        for index, direction_sign in ((0, -1), (1, 1)):
            # (a measured end still abuts a wall / leg closer than one cell: the cells cannot tell a smaller gap from none)
            reach = step if index in measured_ends else settings.along_reach_m * upm
            origin = np.asarray(to_plan(ends[index], centre_across) if axis == "u" else to_plan(centre_across, ends[index]), dtype=float)
            direction = np.asarray(to_plan(direction_sign, 0.0) if axis == "u" else to_plan(0.0, direction_sign), dtype=float)
            hit = _ray(origin, direction, reach, stops)
            if hit is not None:
                ends[index] = ends[index] + direction_sign * hit
        a0, a1 = ends
        if (a1 - a0) < settings.min_length_m * upm:
            continue
        rect = box(a0, c0, a1, c1) if axis == "u" else box(c0, a0, c1, a1)
        polygon = affinity.rotate(rect, angle, origin=(0.0, 0.0), use_radians=True)
        if stops is not None:
            polygon = polygon.difference(stops)
            pieces = [piece for piece in getattr(polygon, "geoms", [polygon]) if isinstance(piece, Polygon)]
            if not pieces:
                continue
            polygon = max(pieces, key=lambda piece: piece.area)
            # slivers left by a stop that overlaps the row obliquely are opened away; the rest stays a rectangle along the walls
            opened = polygon.buffer(-0.07 * upm, join_style=2).buffer(0.07 * upm, join_style=2)
            pieces = [piece for piece in getattr(opened, "geoms", [opened]) if isinstance(piece, Polygon) and not piece.is_empty]
            if not pieces:
                continue
            polygon = max(pieces, key=lambda piece: piece.area)
        if polygon.area < settings.min_area_m2 * upm * upm:
            continue
        if any(polygon.intersection(other).area > settings.duplicate_share * polygon.area for other in accepted):
            continue
        accepted.append(polygon)
        top = _row_top(pts, polygon, upm, settings)
        if top is None:
            continue
        rows.append(TallRow(polygon, top, angle + (0.0 if axis == "u" else math.pi / 2.0), segment["cells"], f"{axis}{'+' if sign == 1 else '-'}"))
    return rows


def _end_face_depth(
    segment: dict[str, Any], segments: list[dict[str, Any]], step: float, settings: KitchenRowSettings, upm: float,
    used: list[dict[str, Any]] | None = None, side: int | None = None, limit: float | None = None,
) -> tuple[float, int] | None:
    """(depth, side) of a front face from the scanned end face of its own body, for a row without a wall behind it.

    ``used`` collects the end faces the depth was read from.  With a known ``side`` (the wall behind the row) and wall distance
    ``limit`` only the end faces on that side, no longer than the way to the wall, are looked for.

    ``segments`` are the straight faces of the tall cells (cell indices).  An end face belongs to the same cluster, runs square to
    the front, starts at one of its ends and at the front line, and leaves it to one side only.  It is as long as the body is deep.
    Two end faces must agree on the side; the shallower one is the depth (a neighbour standing beside the row can prolong one of them).
    Nothing is assumed without an end face: a front alone has no measured depth.
    """
    join = max(1, int(round(settings.end_face_join_m * upm / step)))
    shortest, longest = settings.end_face_min_m * upm, settings.max_depth_m * upm
    if limit is not None:
        longest = min(longest, limit)
    line0, line1 = segment["lines"]
    along0, along1 = segment["along"]
    found: list[tuple[float, int]] = []
    faces: list[dict[str, Any]] = []
    for other in segments:
        if other is segment or other["member"] != segment["member"] or other["axis"] == segment["axis"] or other.get("wall") is not None:
            continue  # (a face with a wall behind it is the front of the neighbouring row, not the side of this one)
        # the end face stands at an end of the front ...
        position0, position1 = other["lines"]
        at_end = min(abs(position0 - along0), abs(position1 - along0), abs(position0 - along1), abs(position1 - along1)) <= join
        if not at_end or position0 < along0 - join or position1 > along1 + join:
            continue
        # ... and this front stands on its body side: a face whose own body lies on the other side is the front of the
        # neighbouring row in an inner corner (two rows meeting at a right angle), this front runs out into the room before it
        facing = other.get("side")
        if facing is not None and facing != (1 if along0 + along1 > position0 + position1 else -1):
            continue
        # ... and runs from the front line to one side of it
        start, stop = other["along"]
        behind, before = stop - line1, line0 - start
        for reach, sign, opposite in ((behind, 1, before), (before, -1, behind)):
            if opposite > join or reach <= 0 or (side is not None and sign != side):
                continue
            depth = (reach + (line1 - line0)) * step  # measured from the outer side of the front face cells
            if shortest <= depth <= longest + join * step:
                found.append((min(depth, longest), sign))
                faces.append(other)
    if not found or len({sign for _, sign in found}) != 1:
        return None
    if used is not None:
        used.extend(faces)
    return min(depth for depth, _ in found), found[0][1]


def _row_top(pts: np.ndarray, polygon: Polygon, upm: float, settings: KitchenRowSettings) -> float | None:
    grown = polygon.buffer(0.05 * upm)
    minx, miny, maxx, maxy = grown.bounds
    near = (pts[:, 0] >= minx) & (pts[:, 0] <= maxx) & (pts[:, 1] >= miny) & (pts[:, 1] <= maxy)
    candidate = pts[near]
    if not len(candidate):
        return None
    from shapely import contains_xy

    inside = candidate[contains_xy(grown, candidate[:, 0], candidate[:, 1])]
    z = inside[:, 2] / upm
    z = z[(z >= settings.tall_min_m) & (z <= settings.tall_max_m)]
    if len(z) < settings.min_top_points:
        return None
    return float(np.percentile(z, 90))


def split_row(row: Polygon, fixed: list[Polygon], units_per_metre: float, min_unit_m: float = 0.30) -> list[tuple[Polygon, int | None]]:
    """Units of a straight rectangular row around fixed items: [(polygon, index of the fixed item or None)].

    A fixed item (a fridge) owns the along-range of its footprint (clipped to the row) over the full depth of the row; the
    remaining along-ranges become free units; a piece shorter than ``min_unit_m`` is absorbed by the neighbouring fixed item.
    """
    upm = max(float(units_per_metre), 1e-9)
    rectangle = row.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        return [(row, None)]
    corners = list(rectangle.exterior.coords)[:4]
    edge_a = np.asarray(corners[1]) - np.asarray(corners[0])
    edge_b = np.asarray(corners[2]) - np.asarray(corners[1])
    long_edge = edge_a if np.hypot(*edge_a) >= np.hypot(*edge_b) else edge_b
    angle = math.atan2(long_edge[1], long_edge[0])
    local = affinity.rotate(row, -angle, origin=(0.0, 0.0), use_radians=True)
    x0, y0, x1, y1 = local.bounds
    ranges: list[tuple[float, float, int]] = []
    for index, item in enumerate(fixed):
        clipped = affinity.rotate(item, -angle, origin=(0.0, 0.0), use_radians=True).intersection(box(x0, y0, x1, y1))
        if clipped.is_empty:
            continue
        low, high = clipped.bounds[0], clipped.bounds[2]
        if high - low > 0.15 * upm:
            ranges.append((max(low, x0), min(high, x1), index))
    ranges.sort()
    # overlapping fixed ranges: the earlier one keeps its range
    cleaned: list[tuple[float, float, int]] = []
    for low, high, index in ranges:
        if cleaned and low < cleaned[-1][1]:
            low = cleaned[-1][1]
        if high - low > 0.15 * upm:
            cleaned.append((low, high, index))
    if not cleaned:
        return [(row, None)]
    # absorb short free pieces into the neighbouring fixed item
    spans: list[list[float | int | None]] = []
    cursor = x0
    for low, high, index in cleaned:
        if low - cursor > 1e-9:
            spans.append([cursor, low, None])
        spans.append([low, high, index])
        cursor = high
    if x1 - cursor > 1e-9:
        spans.append([cursor, x1, None])
    for position, span in enumerate(spans):
        if span[2] is None and float(span[1]) - float(span[0]) < min_unit_m * upm:
            for neighbour in (position - 1, position + 1):
                if 0 <= neighbour < len(spans) and spans[neighbour][2] is not None:
                    spans[neighbour][0] = min(float(spans[neighbour][0]), float(span[0]))
                    spans[neighbour][1] = max(float(spans[neighbour][1]), float(span[1]))
                    span[2] = -1
                    break
    output: list[tuple[Polygon, int | None]] = []
    for low, high, index in spans:
        if index == -1:
            continue
        piece = affinity.rotate(box(float(low), y0, float(high), y1), angle, origin=(0.0, 0.0), use_radians=True)
        output.append((piece, None if index is None else int(index)))
    return output
