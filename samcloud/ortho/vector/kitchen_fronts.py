"""Measured fronts of a kitchen run from the scan points: diagonal corner units, recessed bases, upper-cabinet ends.

The raster run (``kitchen_run``) knows only rectangles along the walls.  This module reads the fronts where a scan
really shows them, as vertical faces between plinth and worktop, seen from the room:

* ``measure_corner_fronts``: a straight front across an inner corner at ANY angle (robust line fit).  It must run
  without gaps over plinth..worktop height, be straight, join both straight fronts, and the square corner behind it
  must be hidden.  Without that evidence the corner stays square.  Next to a measured corner front the two straight
  fronts take their measured offset as well, so the three fronts meet where the scan shows them.
* ``measure_recesses``: a stretch of a leg whose worktop is measured while the body below stands back (knee space,
  open end of a peninsula): the base steps back to the measured face, the worktop stays.
* ``seen_through_ends``: an end of an upper cabinet where the scan sees the wall behind it and no front is no cabinet.
* ``split_run``: the legs and corner triangles of a run as solids that abut and never overlap.

Coordinates are wall-frame plan units (u, v = local rotated by -theta), heights are metres above the floor, all
thresholds are metric and scaled by ``units_per_metre``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


@dataclass(frozen=True, slots=True)
class KitchenFrontSettings:
    """Physical limits and evidence thresholds (config ``indoor_kitchen_fronts:``)."""

    enabled: bool = True
    band_min_m: float = 0.12  # cabinet fronts: above the plinth ...
    band_max_m: float = 0.84  # ... and below the worktop
    band_bins: int = 6
    station_m: float = 0.04
    station_min_bins: int = 4  # a station shows a front when this many height bins hold points
    line_tolerance_m: float = 0.035  # scan noise plus handles and door gaps of one front
    max_shift_m: float = 0.08  # a straight front moves at most this far from the fitted run
    corner_min_cut_m: float = 0.15
    corner_max_cut_m: float = 1.10
    corner_min_angle_deg: float = 15.0  # flatter is the straight front itself
    corner_iterations: int = 600
    corner_min_coverage: float = 0.75
    corner_min_top_coverage: float = 0.60  # up to the worktop (a stool or a bin ends below it)
    corner_max_residual_m: float = 0.02
    corner_min_inlier_share: float = 0.70
    corner_max_bow_m: float = 0.025
    corner_max_hidden_share: float = 0.35  # share of the square corner behind the diagonal that may still show a front
    corner_min_join_share: float = 0.50  # the straight fronts are seen right beside the diagonal
    front_observed_share: float = 0.50
    recess: bool = True
    recess_min_depth_m: float = 0.12
    recess_min_length_m: float = 0.40
    recess_max_spread_m: float = 0.05
    recess_min_top_share: float = 0.50
    recess_anchor_m: float = 0.30
    upper_trim: bool = True
    upper_trim_min_m: float = 0.15
    upper_min_length_m: float = 0.40


@dataclass(slots=True)
class ScanPoints:
    """Scan points around the kitchen: wall-frame (u, v) in plan units, height above the floor in metres."""

    uv: np.ndarray
    z: np.ndarray

    @classmethod
    def of(cls, local_xyz: np.ndarray, theta: float, units_per_metre: float) -> "ScanPoints":
        """From local plan points whose z is the height above the floor in plan units."""
        points = np.asarray(local_xyz, dtype=float).reshape(-1, 3)
        cos, sin = math.cos(theta), math.sin(theta)
        uv = np.column_stack((points[:, 0] * cos + points[:, 1] * sin, -points[:, 0] * sin + points[:, 1] * cos))
        return cls(uv, points[:, 2] / max(float(units_per_metre), 1e-9))

    def band(self, low: float, high: float) -> "ScanPoints":
        keep = (self.z >= low) & (self.z < high)
        return ScanPoints(self.uv[keep], self.z[keep])


@dataclass(slots=True)
class CornerFront:
    """A diagonal front between ``first`` (on the horizontal leg's front) and ``second`` (on the vertical one's)."""

    corner: np.ndarray
    first: np.ndarray
    second: np.ndarray
    leg_h: int
    leg_v: int
    angle_deg: float  # against the horizontal leg, 0..90
    coverage: float
    residual_m: float

    @property
    def triangle(self) -> Polygon:
        return Polygon([tuple(self.corner), tuple(self.first), tuple(self.second)])

    @property
    def tangent(self) -> np.ndarray:
        vector = self.second - self.first
        return vector / float(np.hypot(*vector))

    @property
    def normal(self) -> np.ndarray:
        """Unit normal of the front, pointing into the room."""
        tangent = self.tangent
        normal = np.asarray((-tangent[1], tangent[0]))
        return normal if float((self.first - self.corner) @ normal) > 0.0 else -normal


@dataclass(slots=True)
class Recess:
    leg: int
    area: Polygon  # wall frame: what the worktop covers and the base does not
    step: list[tuple[float, float]]  # the stepped base front
    depth: float
    length: float


def load_kitchen_points(source: "Path | str | None", frame: Any, floor_z: float, bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Local points (z above the floor) of all classes inside ``bounds`` of a classified PLY; empty without one.

    All classes: what stands in a knee space is clutter and the wall seen behind an open shelf is wall.
    """
    empty = np.empty((0, 3), dtype=np.float64)
    if source is None or str(source).lower().endswith((".las", ".laz")) or not Path(source).is_file():
        return empty
    try:
        from samcloud.engine.ply_io import read_vertex_array

        vertices = read_vertex_array(source)
        if "scalar_class" not in (vertices.dtype.names or ()):
            return empty
        x0, y0, x1, y1 = bounds
        kept = [empty]
        for start in range(0, len(vertices), 2_000_000):
            part = vertices[start:start + 2_000_000]
            local = np.asarray(frame.to_local(np.column_stack((part["x"], part["y"], part["z"])).astype(np.float64)), dtype=np.float64)
            inside = (local[:, 0] >= x0) & (local[:, 0] <= x1) & (local[:, 1] >= y0) & (local[:, 1] <= y1)
            kept.append(local[inside])
    except Exception:  # noqa: BLE001 - optional evidence, the raster run stays
        return empty
    points = np.vstack(kept)
    points[:, 2] -= float(floor_z)
    return points


# --------------------------------------------------------------------------------------------- helpers

def _bins(z: np.ndarray, low: float, high: float, count: int) -> np.ndarray:
    return np.clip(((z - low) / (high - low) * count).astype(int), 0, count - 1)


def _cover(along: np.ndarray, bins: np.ndarray, start: float, stop: float, step: float, count: int) -> np.ndarray:
    """(stations, height bins) occupancy of the stretch start..stop."""
    stations = max(1, int(round((stop - start) / step)))
    occupied = np.zeros((stations, count), dtype=bool)
    inside = (along >= start) & (along < stop)
    index = np.minimum(((along[inside] - start) / (stop - start) * stations).astype(int), stations - 1)
    occupied[index, bins[inside]] = True
    return occupied


def _share(along: np.ndarray, bins: np.ndarray, start: float, stop: float, step: float, settings: KitchenFrontSettings) -> float:
    if stop - start <= 1e-9:
        return 0.0
    occupied = _cover(along, bins, start, stop, step, settings.band_bins)
    return float(np.mean(occupied.sum(axis=1) >= settings.station_min_bins))


def _cell_medians(along: np.ndarray, across: np.ndarray, bins: np.ndarray, step: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """One sample per (station, height bin), so dense patches do not pull a fit."""
    if not len(along):
        return along, across
    keys = np.floor(along / step).astype(np.int64) * count + bins
    order = np.argsort(keys, kind="stable")
    parts = np.split(order, np.flatnonzero(np.diff(keys[order])) + 1)
    return np.asarray([np.median(along[part]) for part in parts]), np.asarray([np.median(across[part]) for part in parts])


def _front_offset(along: np.ndarray, across: np.ndarray, bins: np.ndarray, start: float, stop: float, settings: KitchenFrontSettings, upm: float) -> tuple[float, float]:
    """(offset of a straight front from the fitted edge, share of the stretch that shows it)."""
    step = settings.station_m * upm
    shift, tolerance = settings.max_shift_m * upm, settings.line_tolerance_m * upm
    if stop - start < step:
        return 0.0, 0.0
    near = (along >= start) & (along < stop) & (np.abs(across) <= shift + tolerance)
    if near.sum() < 3:
        return 0.0, 0.0
    _, samples = _cell_medians(along[near], across[near], bins[near], step, settings.band_bins)
    # densest face first (an object standing in front is the minority), then the median around it
    support = [int(np.count_nonzero(np.abs(samples - value) <= tolerance)) for value in samples]
    best = np.flatnonzero(np.asarray(support) == max(support))
    offset = float(samples[best[np.argmin(np.abs(samples[best]))]])
    for width in (tolerance, tolerance, 0.7 * tolerance):
        close = np.abs(samples - offset) <= width
        if not close.any():
            break
        offset = float(np.median(samples[close]))
    offset = float(np.clip(offset, -shift, shift))
    face = np.abs(across - offset) <= tolerance
    return offset, _share(along[face], bins[face], start, stop, step, settings)


# --------------------------------------------------------------------------------------------- corners

@dataclass(slots=True)
class _Corner:
    h: int
    v: int
    cu: float
    cv: float
    su: float
    sv: float
    run_a: float  # free length of the horizontal front
    run_b: float


def _inner_corners(legs: Sequence[Any], obstacles: Any, room: Any, settings: KitchenFrontSettings, upm: float) -> list[_Corner]:
    touch, shortest = 0.05 * upm, settings.corner_min_cut_m * upm
    boxes = [box(leg.u0, leg.v0, leg.u1, leg.v1) for leg in legs]
    found: list[_Corner] = []
    for h, level in enumerate(legs):
        for v, upright in enumerate(legs):
            if not level.horizontal or upright.horizontal or boxes[h].distance(boxes[v]) > touch:
                continue
            for su, cu, run_a in ((1.0, upright.u1, level.u1 - upright.u1), (-1.0, upright.u0, upright.u0 - level.u0)):
                for sv, cv, run_b in ((1.0, level.v1, upright.v1 - level.v1), (-1.0, level.v0, level.v0 - upright.v0)):
                    if run_a < shortest or run_b < shortest:
                        continue
                    if level.back == ("v1" if sv > 0 else "v0") or upright.back == ("u1" if su > 0 else "u0"):
                        continue  # that side is the wall
                    probe = Point(cu + su * touch, cv + sv * touch)
                    if any(shape.contains(probe) for shape in boxes):
                        continue
                    if obstacles is not None and obstacles.contains(probe):
                        continue
                    if room is not None and not room.contains(probe):
                        continue
                    found.append(_Corner(h, v, cu, cv, su, sv, run_a, run_b))
    return found


def _fit_diagonal(a: np.ndarray, b: np.ndarray, bins: np.ndarray, la: float, lb: float, settings: KitchenFrontSettings, upm: float) -> tuple[np.ndarray, float] | None:
    """(unit normal, distance) of the best supported straight face across the corner (a, b >= 0 is the room)."""
    step, tolerance = settings.station_m * upm, settings.line_tolerance_m * upm
    inner = (a > tolerance) & (b > tolerance) & (a <= la) & (b <= lb)
    a, b, bins = a[inner], b[inner], bins[inner]
    if len(a) < 12:
        return None
    # one sample per 2 cm cell and height bin
    cells = np.column_stack((np.floor(a / (0.02 * upm)), np.floor(b / (0.02 * upm)), bins)).astype(np.int64)
    first = np.sort(np.unique(cells, axis=0, return_index=True)[1])
    a, b, bins = a[first], b[first], bins[first]
    xy = np.column_stack((a, b))
    steep, flat = math.sin(math.radians(settings.corner_min_angle_deg)), math.cos(math.radians(settings.corner_min_angle_deg))
    low, high = settings.corner_min_cut_m * upm, settings.corner_max_cut_m * upm
    rng = np.random.default_rng(0)
    best: tuple[int, np.ndarray, float] | None = None
    for i, j in rng.integers(0, len(xy), size=(settings.corner_iterations, 2)):
        vector = xy[j] - xy[i]
        length = math.hypot(vector[0], vector[1])
        if length < 2.0 * step:
            continue
        normal = np.asarray((-vector[1], vector[0])) / length
        if normal[0] < 0.0:
            normal = -normal
        if not (steep <= normal[0] <= flat and steep <= normal[1] <= flat):
            continue
        distance = float(xy[i] @ normal)
        cut_a, cut_b = distance / normal[0], distance / normal[1]
        if not (low <= cut_a <= min(high, la) and low <= cut_b <= min(high, lb)):
            continue
        along = xy @ np.asarray((-normal[1], normal[0]))
        face = np.abs(xy @ normal - distance) <= 0.7 * tolerance
        start, stop = -cut_a * normal[1], cut_b * normal[0]
        occupied = _cover(along[face], bins[face], start, stop, step, settings.band_bins)
        score = 2 * int(occupied.sum()) - occupied.size  # supported minus unsupported face
        if best is None or score > best[0]:
            best = (score, normal, distance)
    if best is None or best[0] <= 0:
        return None
    _, normal, distance = best
    for width in (1.5 * tolerance, tolerance, tolerance, 0.8 * tolerance):
        tangent = np.asarray((-normal[1], normal[0]))
        start, stop = -distance / normal[0] * normal[1], distance / normal[1] * normal[0]
        trim = min(0.04 * upm, 0.15 * (stop - start))  # the rounded ends belong to both fronts
        across, along = xy @ normal - distance, xy @ tangent
        use = (np.abs(across) <= width) & (along >= start + trim) & (along <= stop - trim)
        if use.sum() < 8:
            return None
        s, r = _cell_medians(along[use], across[use], bins[use], step, settings.band_bins)
        samples = s[:, None] * tangent + (r + distance)[:, None] * normal
        mean = samples.mean(axis=0)
        normal = np.linalg.svd(samples - mean, full_matrices=False)[2][1]
        if normal[0] < 0.0:
            normal = -normal
        if not (steep <= normal[0] <= flat and steep <= normal[1] <= flat):
            return None
        distance = float(mean @ normal)
    return normal, distance


def _density_ridge(a: np.ndarray, b: np.ndarray, normal: np.ndarray, distance: float, settings: KitchenFrontSettings, upm: float) -> tuple[np.ndarray, float] | None:
    """The fitted face pulled onto the densest line of ALL its points.

    The search keeps one sample per cell, so in a dense, slightly noisy scan the stray points weigh as much as the face
    and the line slides off it.  Here every point counts across the face (biweight) and every station counts equally
    along it.
    """
    step, reach = settings.station_m * upm, 2.0 * settings.line_tolerance_m * upm
    steep, flat = math.sin(math.radians(settings.corner_min_angle_deg)), math.cos(math.radians(settings.corner_min_angle_deg))
    xy = np.column_stack((a, b))
    for shrink in (1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.75, 0.75):
        tangent = np.asarray((-normal[1], normal[0]))
        start, stop = -distance / normal[0] * normal[1], distance / normal[1] * normal[0]
        trim = min(0.04 * upm, 0.15 * (stop - start))
        across, along = xy @ normal - distance, xy @ tangent
        use = (np.abs(across) < shrink * reach) & (along >= start + trim) & (along <= stop - trim)
        if use.sum() < 8:
            return None
        weight = (1.0 - (across[use] / (shrink * reach)) ** 2) ** 2
        _, station = np.unique(np.floor(along[use] / step).astype(np.int64), return_inverse=True)
        weight /= np.maximum(np.bincount(station, weights=weight)[station], 1e-12)
        near = xy[use]
        mean = (near * weight[:, None]).sum(axis=0) / weight.sum()
        centred = near - mean
        normal = np.linalg.eigh((centred * weight[:, None]).T @ centred)[1][:, 0]
        if normal[0] < 0.0:
            normal = -normal
        if not (steep <= normal[0] <= flat and steep <= normal[1] <= flat):
            return None
        distance = float(mean @ normal)
    return normal, distance


@dataclass(slots=True)
class _Measured:
    corner: _Corner
    normal: np.ndarray
    distance: float
    shift_a: float  # vertical front
    shift_b: float  # horizontal front
    share_a: float
    share_b: float
    coverage: float
    residual: float


def _measure_corner(corner: _Corner, band: ScanPoints, low: float, high: float, settings: KitchenFrontSettings, upm: float) -> tuple[_Measured | None, bool]:
    """(the measured diagonal or None, whether both straight fronts of this corner are observed at all).

    A rejected fit is tried once more on the density ridge of its points, under the same tests.
    """
    measured, observed = _judge_corner(corner, band, low, high, settings, upm, ridge=False)
    if measured is None:
        measured, _ = _judge_corner(corner, band, low, high, settings, upm, ridge=True)
    return measured, observed


def _judge_corner(
    corner: _Corner, band: ScanPoints, low: float, high: float, settings: KitchenFrontSettings, upm: float, *, ridge: bool,
) -> tuple[_Measured | None, bool]:
    step, tolerance, margin = settings.station_m * upm, settings.line_tolerance_m * upm, 0.15 * upm
    a, b = (band.uv[:, 0] - corner.cu) * corner.su, (band.uv[:, 1] - corner.cv) * corner.sv
    near = (a >= -margin) & (b >= -margin) & (a <= corner.run_a + margin) & (b <= corner.run_b + margin)
    a, b, bins = a[near], b[near], _bins(band.z[near], low, high, settings.band_bins)
    rough_b, seen_h = _front_offset(a, b, bins, 0.0, corner.run_a, settings, upm)
    rough_a, seen_v = _front_offset(b, a, bins, 0.0, corner.run_b, settings, upm)
    observed = min(seen_h, seen_v) >= settings.front_observed_share
    # origin on the two observed straight fronts (the fitted edges where there is none)
    oa = rough_a if seen_v >= 0.3 else 0.0
    ob = rough_b if seen_h >= 0.3 else 0.0
    reach = settings.corner_max_cut_m * upm
    la, lb = min(corner.run_a - oa, reach) + tolerance, min(corner.run_b - ob, reach) + tolerance
    fitted = _fit_diagonal(a - oa, b - ob, bins, la, lb, settings, upm)
    if fitted is not None and ridge:
        fitted = _density_ridge(a - oa, b - ob, *fitted, settings, upm)
    if fitted is None:
        return None, observed
    normal, distance = fitted
    cut_a, cut_b = oa + distance / normal[0], ob + distance / normal[1]
    shift_b, share_b = _front_offset(a, b, bins, cut_a, corner.run_a, settings, upm)
    shift_a, share_a = _front_offset(b, a, bins, cut_b, corner.run_b, settings, upm)
    joins = []
    for along, across, cut, run, shift in ((a, b, cut_a, corner.run_a, shift_b), (b, a, cut_b, corner.run_b, shift_a)):
        if run - cut < 0.10 * upm:
            continue  # the diagonal reaches the end of the leg
        face = np.abs(across - shift) <= tolerance
        joins.append(_share(along[face], bins[face], cut, min(cut + 0.25 * upm, run), step, settings))
    # without an observed straight front on either side a slanted face is just some object
    if not joins or min(joins) < settings.corner_min_join_share:
        return None, observed
    oa = shift_a if share_a >= settings.front_observed_share else oa
    ob = shift_b if share_b >= settings.front_observed_share else ob
    a, b = a - oa, b - ob
    distance += float(normal @ ((rough_a if seen_v >= 0.3 else 0.0) - oa, (rough_b if seen_h >= 0.3 else 0.0) - ob))
    cut_a, cut_b = distance / normal[0], distance / normal[1]
    if min(cut_a, cut_b) < settings.corner_min_cut_m * upm or max(cut_a, cut_b) > reach + tolerance or cut_a > corner.run_a - oa + tolerance or cut_b > corner.run_b - ob + tolerance:
        return None, observed
    along, across = -a * normal[1] + b * normal[0], a * normal[0] + b * normal[1] - distance
    start, stop = -cut_a * normal[1], cut_b * normal[0]
    trim = min(0.04 * upm, 0.15 * (stop - start))
    on = (along >= start + trim) & (along <= stop - trim)
    face, wide = on & (np.abs(across) <= tolerance), on & (np.abs(across) <= 0.10 * upm)
    if face.sum() < 8:
        return None, observed
    coverage = _share(along[face], bins[face], start + trim, stop - trim, step, settings)
    occupied = _cover(along[face], bins[face], start + trim, stop - trim, step, settings.band_bins)
    top_coverage = float(np.mean(occupied[:, -2:].any(axis=1)))  # the two highest bins, just below the worktop
    residual = float(np.median(np.abs(across[face])))
    if coverage < settings.corner_min_coverage or top_coverage < settings.corner_min_top_coverage:
        return None, observed
    if residual > settings.corner_max_residual_m * upm or face.sum() < settings.corner_min_inlier_share * wide.sum():
        return None, observed
    # straight, not a round object standing in the corner
    s, r = _cell_medians(along[wide], across[wide], bins[wide], step, settings.band_bins)
    if len(s) >= 6:
        half = (stop - start) / 2.0
        curve = np.polyfit((s - (start + stop) / 2.0) / half, r, 2)
        if abs(float(curve[0])) > settings.corner_max_bow_m * upm:
            return None, observed
    # the diagonal unit hides the square corner behind it
    for along_leg, across_leg, cut in ((a, b, cut_a), (b, a, cut_b)):
        stop_hidden = cut - max(0.08 * upm, 2.0 * tolerance)
        if stop_hidden > step:
            front = np.abs(across_leg) <= 0.8 * tolerance
            if _share(along_leg[front], bins[front], 0.0, stop_hidden, step, settings) > settings.corner_max_hidden_share:
                return None, observed
    # back to the coordinates of the fitted corner
    distance += float(normal @ (oa, ob))
    return _Measured(corner, normal, distance, oa, ob, share_a, share_b, coverage, residual / upm), observed


def _move_edge(legs: Sequence[Any], index: int, name: str, value: float, upm: float) -> None:
    """Move a front edge; the ends of the perpendicular legs that stop at that edge follow it."""
    leg = legs[index]
    old = getattr(leg, name)
    for other in legs:
        if other is leg or other.horizontal == leg.horizontal:
            continue
        for end in ("v0", "v1") if leg.horizontal else ("u0", "u1"):
            if abs(getattr(other, end) - old) <= 0.01 * upm:
                setattr(other, end, value)
    setattr(leg, name, value)


def measure_corner_fronts(
    legs: Sequence[Any], points: ScanPoints, obstacles: Any, room: Any, settings: KitchenFrontSettings, units_per_metre: float,
    counter_top_m: float | None = None,
) -> tuple[list[CornerFront], list[tuple[float, float]]]:
    """Diagonal fronts in the inner corners of ``legs`` and the corners that are observed to be square.

    ``legs`` are ``_LegBox`` in the wall frame; the fronts beside a measured diagonal are moved onto their measured
    face.  The second value lists the corner points whose straight fronts are observed without a diagonal between.
    """
    upm = max(float(units_per_metre), 1e-9)
    if not settings.enabled or not len(points.uv):
        return [], []
    low = settings.band_min_m
    high = settings.band_max_m if counter_top_m is None else min(settings.band_max_m, counter_top_m - 0.04)
    if high - low < 0.30:
        return [], []
    band = points.band(low, high)
    measured: list[_Measured] = []
    square: list[tuple[float, float]] = []
    for corner in _inner_corners(legs, obstacles, room, settings, upm):
        result, observed = _measure_corner(corner, band, low, high, settings, upm)
        if result is not None:
            measured.append(result)
        elif observed:
            square.append((corner.cu, corner.cv))
    moves: dict[tuple[int, str], tuple[float, float]] = {}  # (leg, edge) -> (support, target)
    for item in measured:
        c = item.corner
        for key, share, target in (
            ((c.h, "v1" if c.sv > 0 else "v0"), item.share_b * c.run_a, c.cv + c.sv * item.shift_b),
            ((c.v, "u1" if c.su > 0 else "u0"), item.share_a * c.run_b, c.cu + c.su * item.shift_a),
        ):
            enough = share >= settings.front_observed_share * (c.run_a if key[0] == c.h else c.run_b)
            if enough and (key not in moves or share > moves[key][0]):
                moves[key] = (share, target)
    for (index, name), (_, target) in moves.items():
        _move_edge(legs, index, name, target, upm)
    fronts: list[CornerFront] = []
    for item in measured:
        c = item.corner
        normal = item.normal * (c.su, c.sv)  # wall frame
        distance = item.distance + float(normal @ (c.cu, c.cv))
        cu = legs[c.v].u1 if c.su > 0 else legs[c.v].u0
        cv = legs[c.h].v1 if c.sv > 0 else legs[c.h].v0
        first = np.asarray(((distance - normal[1] * cv) / normal[0], cv))
        second = np.asarray((cu, (distance - normal[0] * cu) / normal[1]))
        if (first[0] - cu) * c.su <= 0.0 or (second[1] - cv) * c.sv <= 0.0:
            continue
        angle = math.degrees(math.atan2(abs(second[1] - first[1]), abs(second[0] - first[0])))
        fronts.append(CornerFront(np.asarray((cu, cv)), first, second, c.h, c.v, angle, item.coverage, item.residual))
    return fronts, square


# --------------------------------------------------------------------------------------------- recessed bases

def _runs(flags: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Index intervals [start, stop) of True, gaps up to ``gap`` bridged."""
    index = np.flatnonzero(flags)
    if not len(index):
        return []
    breaks = np.flatnonzero(np.diff(index) > gap + 1)
    return list(zip(np.r_[index[0], index[breaks + 1]].tolist(), (np.r_[index[breaks], index[-1]] + 1).tolist(), strict=True))


def measure_recesses(
    legs: Sequence[Any], corners: Sequence[CornerFront], points: ScanPoints, settings: KitchenFrontSettings, units_per_metre: float,
    counter_top_m: float = 0.90,
) -> list[Recess]:
    """Stretches of a leg front where the worktop is measured and the body below visibly stands back."""
    upm = max(float(units_per_metre), 1e-9)
    if not settings.enabled or not settings.recess or not len(points.uv):
        return []
    step, tolerance, count = settings.station_m * upm, settings.line_tolerance_m * upm, settings.band_bins
    low, high = settings.band_min_m, min(0.75, counter_top_m - 0.10)  # under the apron of the worktop
    if high - low < 0.30:
        return []
    below, top = points.band(low, high), points.band(counter_top_m - 0.12, counter_top_m + 0.15)
    found: list[Recess] = []
    for index, leg in enumerate(legs):
        axis = 0 if leg.horizontal else 1
        sides = ("v0", "v1") if leg.horizontal else ("u0", "u1")
        along0, along1 = (leg.u0, leg.u1) if leg.horizontal else (leg.v0, leg.v1)
        for side in sides:
            if leg.back == side:
                continue
            edge, inward = getattr(leg, side), (1.0 if side.endswith("0") else -1.0)
            # the part of this side that is a free front (perpendicular legs and corner units cover the rest)
            start, stop = along0, along1
            spans: list[tuple[float, float]] = []
            for other in legs:
                if other is leg or other.horizontal == leg.horizontal:
                    continue
                across = (other.v0, other.v1) if leg.horizontal else (other.u0, other.u1)
                if across[0] - 0.01 * upm <= edge <= across[1] + 0.01 * upm:
                    spans.append((other.u0, other.u1) if leg.horizontal else (other.v0, other.v1))
            for corner in corners:
                ends = [corner.corner] + ([corner.first] if corner.leg_h == index else [corner.second] if corner.leg_v == index else [])
                if len(ends) == 2 and abs(float(corner.corner[1 - axis]) - edge) <= 0.02 * upm:
                    spans.append((min(float(end[axis]) for end in ends), max(float(end[axis]) for end in ends)))
            for span in sorted(spans):
                if span[0] <= start + 0.05 * upm:
                    start = max(start, span[1])
            for span in sorted(spans, key=lambda value: -value[1]):
                if span[1] >= stop - 0.05 * upm:
                    stop = min(stop, span[0])
            if stop - start < settings.recess_anchor_m * upm + settings.recess_min_length_m * upm:
                continue
            stations = max(1, int(round((stop - start) / step)))

            def occupancy(data: ScanPoints, near: float, far: float, z_low: float, z_high: float) -> np.ndarray:
                inner = (data.uv[:, 1 - axis] - edge) * inward
                use = (inner >= near) & (inner <= far)
                return _cover(data.uv[use, axis], _bins(data.z[use], z_low, z_high, count), start, stop, (stop - start) / stations, count)

            front = occupancy(below, -settings.max_shift_m * upm, 0.06 * upm, low, high).sum(axis=1) >= settings.station_min_bins
            if front.sum() * step < settings.recess_anchor_m * upm:
                continue  # no observed front on this side: nothing to step back from
            outside = occupancy(below, -0.35 * upm, -settings.max_shift_m * upm, low, high).sum(axis=1) >= count // 2
            topped = occupancy(top, 0.0, settings.recess_min_depth_m * upm + tolerance, counter_top_m - 0.12, counter_top_m + 0.15).any(axis=1)
            inner = (below.uv[:, 1 - axis] - edge) * inward
            at = np.floor((below.uv[:, axis] - start) / (stop - start) * stations).astype(int)
            z_bins = _bins(below.z, low, high, count)
            possible = (inner >= 0.08 * upm) & (inner <= leg.depth - 0.05 * upm) & (at >= 0) & (at < stations)
            face = np.full(stations, np.nan)
            for station in np.flatnonzero(~front & ~outside):
                use = possible & (at == station)
                if use.sum() < 3:
                    continue
                depths, levels = inner[use], z_bins[use]
                for depth in np.arange(0.08 * upm, float(depths.max()) + 1e-9, 0.01 * upm):  # nearest face seen from the room
                    close = np.abs(depths - depth - tolerance) <= tolerance
                    if len(np.unique(levels[close])) >= max(2, count // 2):
                        face[station] = float(np.median(depths[close]))
                        break
            candidates = np.nan_to_num(face, nan=-1.0) >= settings.recess_min_depth_m * upm
            for first, last in _runs(candidates, 2):
                while first > 0 and not front[first - 1] and first - 1 <= 2:
                    first -= 1  # up to two unseen stations at the end of the leg
                while last < stations and not front[last] and stations - last <= 2:
                    last += 1
                if (last - first) * step < settings.recess_min_length_m * upm or front[first:last].any():
                    continue
                anchored = (first >= 1 and front[max(0, first - 3):first].any()) or (last < stations and front[last:last + 3].any())
                if not anchored:
                    continue
                seen = face[first:last][candidates[first:last]]
                depth = float(np.median(seen))
                if float(np.median(np.abs(seen - depth))) > settings.recess_max_spread_m * upm:
                    continue
                if float(np.mean(topped[first:last])) < settings.recess_min_top_share:
                    continue
                size = (stop - start) / stations
                a0, a1 = start + first * size, start + last * size
                if along1 - a1 <= step:
                    a1 = along1
                if a0 - along0 <= step:
                    a0 = along0
                back = edge + inward * depth
                low_side, high_side = min(edge - inward * 0.01 * upm, back), max(edge - inward * 0.01 * upm, back)
                area = box(a0, low_side, a1, high_side) if leg.horizontal else box(low_side, a0, high_side, a1)
                path = [(a0, back), (a1, back)]
                if a0 > along0:
                    path.insert(0, (a0, edge))
                if a1 < along1:
                    path.append((a1, edge))
                if not leg.horizontal:
                    path = [(v, u) for u, v in path]
                found.append(Recess(index, area, path, depth, a1 - a0))
    return found


# --------------------------------------------------------------------------------------------- upper cabinets

def seen_through_ends(unit: Any, band_m: tuple[float, float], points: ScanPoints, settings: KitchenFrontSettings, units_per_metre: float) -> tuple[float, float] | None:
    """New (along0, along1) of an upper cabinet whose end is seen through to the wall; None = keep it.

    Judged in the upper half of its height band: what stands on an open shelf is low, a cabinet front is not.
    Without points (an unseen glossy front) nothing is trimmed, only where the scan sees the wall behind.
    """
    upm = max(float(units_per_metre), 1e-9)
    if not settings.enabled or not settings.upper_trim or unit.back is None or not len(points.uv):
        return None
    low, high = (band_m[0] + band_m[1]) / 2.0, band_m[1] - 0.03
    if high - low < 0.15:
        return None
    half = points.band(low, high)
    axis = 0 if unit.horizontal else 1
    along0, along1 = (unit.u0, unit.u1) if unit.horizontal else (unit.v0, unit.v1)
    inward = 1.0 if unit.back.endswith("0") else -1.0
    out = (half.uv[:, 1 - axis] - getattr(unit, unit.back)) * inward
    z_bins = _bins(half.z, low, high, settings.band_bins)
    stations = max(1, int(round((along1 - along0) / (0.025 * upm))))
    size = (along1 - along0) / stations

    def share(near: float, far: float) -> np.ndarray:
        use = (out >= near) & (out <= far)
        return _cover(half.uv[use, axis], z_bins[use], along0, along1, size, settings.band_bins).mean(axis=1)

    front = share(0.6 * unit.depth, unit.depth + 0.10 * upm)
    behind = share(-0.10 * upm, 0.35 * unit.depth)
    cabinet = front >= 0.5
    free = (front <= 0.2) & (behind >= 0.4)
    # a cabinet starts with two stations of front (a single one is a bottle or a hook)
    solid = np.flatnonzero(cabinet & np.r_[cabinet[1:], False])
    if not len(solid):
        return None
    first, last = int(solid[0]), int(solid[-1]) + 2
    start, stop = along0, along1

    if first * size >= settings.upper_trim_min_m * upm and free[:first].mean() >= 0.6:
        start = along0 + first * size
    if (stations - last) * size >= settings.upper_trim_min_m * upm and free[last:].mean() >= 0.6:
        stop = along0 + last * size
    # what is left must still be most of the detected unit: otherwise the detection, not its end, is in question
    if (start, stop) == (along0, along1) or stop - start < max(settings.upper_min_length_m * upm, 0.5 * (along1 - along0)):
        return None
    return start, stop


# --------------------------------------------------------------------------------------------- solids

def _polygons(geometry: Any, minimum: float) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    return [part for part in getattr(geometry, "geoms", [geometry]) if isinstance(part, Polygon) and part.area > minimum]


def split_run(
    legs: Sequence[Any], triangles: Sequence[Polygon], recesses: Sequence[Recess], units_per_metre: float,
) -> list[tuple[list[Polygon], list[Polygon]]]:
    """(worktop pieces, base pieces) per leg in the wall frame: they abut and never overlap.

    The wall-backed, longer leg keeps a shared corner square together with the triangle of its corner unit.
    """
    upm = max(float(units_per_metre), 1e-9)
    boxes = [box(leg.u0, leg.v0, leg.u1, leg.v1) for leg in legs]
    order = sorted(range(len(legs)), key=lambda i: (legs[i].back is None, -max(legs[i].u1 - legs[i].u0, legs[i].v1 - legs[i].v0), i))
    own: list[list[Polygon]] = [[] for _ in legs]
    for triangle in triangles:
        touching = [i for i in order if boxes[i].buffer(0.005 * upm, join_style=2).intersection(triangle).area > 1e-6 * upm * upm]
        if touching:
            own[touching[0]].append(triangle)
    sliver, minimum = 0.001 * upm, 0.01 * upm * upm
    result: list[tuple[list[Polygon], list[Polygon]]] = [([], []) for _ in legs]
    taken: Any = None
    for index in order:
        if not own[index] and (taken is None or boxes[index].intersection(taken).area <= 1e-9 * upm * upm):
            shape = boxes[index]  # nothing to share: exactly the fitted rectangle
        else:
            shape = unary_union([boxes[index], *own[index]]).buffer(sliver, join_style=2).buffer(-sliver, join_style=2)
            if taken is not None:
                # opened by a millimetre: no hairline spikes along the shared edge
                shape = shape.difference(taken).buffer(-sliver, join_style=2).buffer(sliver, join_style=2).difference(taken)
        tops = _polygons(shape, minimum)
        cut = [item.area for item in recesses if item.leg == index]
        bases = _polygons(unary_union(tops).difference(unary_union(cut)), minimum) if cut and tops else tops
        result[index] = (tops, bases)
        if tops:
            taken = unary_union(tops) if taken is None else unary_union([taken, *tops])
    return result


__all__ = [
    "CornerFront", "KitchenFrontSettings", "Recess", "ScanPoints", "load_kitchen_points", "measure_corner_fronts",
    "measure_recesses", "seen_through_ends", "split_run",
]
