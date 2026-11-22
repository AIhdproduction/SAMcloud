"""Beams: elongated soffit drops that span between supports.

The overhead surface (ceiling- and beam-class points) is rasterised (median height per cell) and compared with the room's
measured ceiling (flat or plane).  Cells lower by ``min_drop`` .. ``max_drop`` form drop regions.  A region is a beam only when

* it is a narrow band (``min_width <= width <= max_width``, length >= ``min_length``, length / width >= ``min_elongation``,
  filling its rectangle),
* it does not run along a wall face (a bulkhead / duct box beside a wall is no beam),
* the ceiling is back at the room height on both sides of the band,
* both ends reach a support (wall body or column) within ``support_reach``.

A wide lowered ceiling area therefore never becomes a beam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .common import BEAM, CEILING, LevelPoints, StructureSettings, contains, histogram_mode
from .levels import RoomLevel


@dataclass(slots=True)
class Beam:
    axis: tuple[tuple[float, float], tuple[float, float]]  # metric plan end points (on the supports)
    width_m: float
    depth_m: float
    z_bottom_m: float
    z_top_m: float
    footprint: Polygon
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


def _rectangle(points: np.ndarray) -> tuple[Polygon, np.ndarray, float, float, np.ndarray]:
    rectangle = MultiPoint([tuple(p) for p in points]).convex_hull.minimum_rotated_rectangle
    corners = np.asarray(rectangle.exterior.coords)[:4]
    first, second = corners[1] - corners[0], corners[2] - corners[1]
    if np.hypot(*first) >= np.hypot(*second):
        long_edge, short_edge = first, second
    else:
        long_edge, short_edge = second, first
    length, width = float(np.hypot(*long_edge)), float(np.hypot(*short_edge))
    centre = corners.mean(axis=0)
    return rectangle, centre, length, width, long_edge / max(length, 1e-12)


def detect_beams(
    points: LevelPoints, rooms: list[Polygon], levels: list[RoomLevel], walls: BaseGeometry | None, columns: list[Polygon],
    level_ceiling_m: float, settings: StructureSettings = StructureSettings(),
) -> tuple[list[Beam], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    measured = [(room, level) for room, level in zip(rooms, levels, strict=True) if level.ceiling_source != "level"]
    if not measured:
        return [], rejected
    storey = float(level_ceiling_m)
    overhead = points.xyz[points.mask((CEILING, BEAM)) & (points.xyz[:, 2] >= 0.6 * storey) & (points.xyz[:, 2] <= 1.3 * storey)]
    if len(overhead) < 20:
        return [], rejected
    cell = settings.beam_cell_m
    union = unary_union([room for room, _ in measured])
    x0, y0, x1, y1 = union.bounds
    nx, ny = int(math.ceil((x1 - x0) / cell)) + 1, int(math.ceil((y1 - y0) / cell)) + 1
    cols = np.floor((overhead[:, 0] - x0) / cell).astype(np.int64)
    rows = np.floor((overhead[:, 1] - y0) / cell).astype(np.int64)
    inside = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
    flat = rows[inside] * nx + cols[inside]
    z = overhead[inside, 2]
    surface = np.full(nx * ny, np.nan)
    counts = np.bincount(flat, minlength=nx * ny)
    labels = np.unique(flat)
    surface[labels] = ndimage.median(z, labels=flat, index=labels)
    surface[counts < 3] = np.nan
    surface = surface.reshape(ny, nx)
    gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * cell, y0 + (np.arange(ny) + 0.5) * cell)
    centres = np.column_stack((gx.ravel(), gy.ravel()))
    reference = np.full(nx * ny, np.nan)
    threshold = np.full(nx * ny, np.nan)
    for room, level in measured:
        member = contains(room, centres)
        reference[member] = level.ceiling_at(centres[member], storey)
        threshold[member] = max(settings.beam_min_drop_m, 4.0 * float(level.noise_m or 0.0))
    reference = reference.reshape(ny, nx)
    threshold = threshold.reshape(ny, nx)
    drop = reference - surface
    valid = np.isfinite(drop)
    mask = valid & (drop >= np.nan_to_num(threshold, nan=np.inf)) & (drop <= settings.beam_max_drop_m)
    clear = valid & (drop < np.nan_to_num(threshold, nan=-np.inf))
    regions, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    supports = [geometry for geometry in ([walls] if walls is not None and not walls.is_empty else []) + list(columns)]
    support_union = unary_union(supports) if supports else None
    beams: list[Beam] = []

    def lookup(xy: np.ndarray, grid: np.ndarray) -> np.ndarray:
        c = np.floor((xy[:, 0] - x0) / cell).astype(int)
        r = np.floor((xy[:, 1] - y0) / cell).astype(int)
        ok = (c >= 0) & (c < nx) & (r >= 0) & (r < ny)
        out = np.zeros(len(xy), dtype=bool)
        out[ok] = grid[r[ok], c[ok]]
        return out

    for index in range(1, count + 1):
        cells = np.argwhere(regions == index)
        area = len(cells) * cell * cell
        if area < settings.beam_min_length_m * settings.beam_min_width_m:
            continue
        corners = np.concatenate([
            np.column_stack((x0 + (cells[:, 1] + dx) * cell, y0 + (cells[:, 0] + dy) * cell)) for dx in (0, 1) for dy in (0, 1)
        ])
        rectangle, centre, length, width, direction = _rectangle(corners)
        info: dict[str, Any] = {"center": [round(float(centre[0]), 3), round(float(centre[1]), 3)], "length_m": round(length, 3), "width_m": round(width, 3)}
        fill = area / max(rectangle.area, 1e-9)
        info["fill"] = round(fill, 3)
        if not (settings.beam_min_width_m <= width <= settings.beam_max_width_m) or length < settings.beam_min_length_m \
                or length / max(width, 1e-9) < settings.beam_min_elongation or fill < settings.beam_min_fill:
            rejected.append({**info, "reason": "not_a_narrow_band"})
            continue
        normal = np.asarray((-direction[1], direction[0]))
        stations = centre + np.outer(np.linspace(-0.4, 0.4, 9) * length, direction)
        if walls is not None and not walls.is_empty:
            near = walls.buffer(settings.beam_wall_adjacent_m)
            along_wall = max(float(contains(near, stations + sign * normal * width / 2.0).mean()) for sign in (-1.0, 1.0))
            info["along_wall_share"] = round(along_wall, 3)
            if along_wall >= 0.5:
                rejected.append({**info, "reason": "along_wall"})
                continue
        side_ok = True
        for sign in (-1.0, 1.0):
            samples = stations + sign * normal * (width / 2.0 + 0.25)
            known = lookup(samples, valid)
            if known.sum() < 3 or lookup(samples, clear)[known].mean() < settings.beam_side_clear_share:
                side_ok = False
        if not side_ok:
            rejected.append({**info, "reason": "no_ceiling_beside"})
            continue
        ends = [centre - direction * length / 2.0, centre + direction * length / 2.0]
        reach = max(settings.beam_support_reach_m, 2.0 * width)
        distances = [float(support_union.distance(Point(*end))) if support_union is not None else float("inf") for end in ends]
        info["support_distance_m"] = [round(value, 3) if math.isfinite(value) else None for value in distances]
        if any(value > reach for value in distances):
            rejected.append({**info, "reason": "no_support"})
            continue
        # extend each end onto its support
        ends = [ends[0] - direction * distances[0], ends[1] + direction * distances[1]]
        component = regions == index
        depth = float(np.nanmedian(drop[component]))
        bottom = float(np.nanmedian(surface[component]))
        top = float(np.nanmedian(reference[component]))
        # Refine bottom, width and axis from the soffit points themselves (the raster widens a rotated band by up to one cell).
        near = overhead[contains(rectangle.buffer(cell), overhead[:, :2])]
        soffit = near[near[:, 2] <= top - float(np.nanmedian(threshold[component]))] if len(near) else near
        if len(soffit) >= 20:
            bottom = histogram_mode(soffit[:, 2])
            underside = soffit[np.abs(soffit[:, 2] - bottom) <= 0.03]
            if len(underside) >= 20:
                across = (underside[:, :2] - centre) @ normal
                lo, hi = (float(v) for v in np.percentile(across, [2, 98]))
                if settings.beam_min_width_m <= hi - lo <= width + 1e-9:
                    width = hi - lo
                    shift = normal * 0.5 * (lo + hi)
                    ends = [ends[0] + shift, ends[1] + shift]
            depth = top - bottom
        footprint = LineString([tuple(ends[0]), tuple(ends[1])]).buffer(width / 2.0, cap_style=2)
        confidence = float(np.clip(fill, 0.0, 1.0) * min(1.0, length / (settings.beam_min_elongation * width)))
        beams.append(Beam(
            axis=((float(ends[0][0]), float(ends[0][1])), (float(ends[1][0]), float(ends[1][1]))), width_m=width, depth_m=depth,
            z_bottom_m=bottom, z_top_m=top, footprint=footprint, confidence=round(confidence, 3), evidence={**info, "depth_m": round(depth, 3)},
        ))
    return beams, rejected


def merge_collinear_beams(beams: list[Beam], max_gap_m: float = 0.80, max_angle_deg: float = 3.0) -> list[Beam]:
    """Join beam pieces of one beam split by a column (or an unseen stretch): parallel, collinear, close end to end, same depth."""
    items = list(beams)
    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                da = np.subtract(a.axis[1], a.axis[0])
                db = np.subtract(b.axis[1], b.axis[0])
                la, lb = float(np.hypot(*da)), float(np.hypot(*db))
                if la < 1e-9 or lb < 1e-9:
                    continue
                ua, ub = da / la, db / lb
                if abs(float(ua @ ub)) < math.cos(math.radians(max_angle_deg)):
                    continue
                normal = np.asarray((-ua[1], ua[0]))
                offsets = [abs(float((np.asarray(p) - np.asarray(a.axis[0])) @ normal)) for p in b.axis]
                if max(offsets) > max(a.width_m, b.width_m) / 2.0 or abs(a.depth_m - b.depth_m) > 0.05:
                    continue
                along_b = sorted(float((np.asarray(p) - np.asarray(a.axis[0])) @ ua) for p in b.axis)
                gap = max(along_b[0] - la, -along_b[1], 0.0)
                if gap > max_gap_m:
                    continue
                lo, hi = min(0.0, along_b[0]), max(la, along_b[1])
                start = np.asarray(a.axis[0]) + ua * lo
                end = np.asarray(a.axis[0]) + ua * hi
                width = (a.width_m * la + b.width_m * lb) / (la + lb)
                footprint = LineString([tuple(start), tuple(end)]).buffer(width / 2.0, cap_style=2)
                items[i] = Beam(
                    axis=((float(start[0]), float(start[1])), (float(end[0]), float(end[1]))), width_m=width,
                    depth_m=(a.depth_m * la + b.depth_m * lb) / (la + lb), z_bottom_m=min(a.z_bottom_m, b.z_bottom_m), z_top_m=max(a.z_top_m, b.z_top_m),
                    footprint=footprint, confidence=max(a.confidence, b.confidence),
                    evidence={**a.evidence, "merged_pieces": int(a.evidence.get("merged_pieces", 1)) + int(b.evidence.get("merged_pieces", 1)), "merge_gap_m": round(gap, 3)},
                )
                del items[j]
                merged = True
                break
            if merged:
                break
    return items
