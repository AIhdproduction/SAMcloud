"""Kitchen run: a clean A-EINB countertop polygon along the walls, with appliance inserts.

Kitchen fine-label evidence (kitchen counter / cabinet, stove, cooktop, oven,
sink) is turned into a straight, L or U run of constant depth (0.60 m) whose back
edge lies flush on the wall.  Cooktop and sink blocks are placed on the run only
where an appliance label was observed; nothing is ever invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from .indoor import _uniform_insert
from .raster_grid import RasterGrid
from .symbols import CATALOGUE


@dataclass(frozen=True, slots=True)
class KitchenRunSettings:
    """Metric thresholds (scaled by units per metre at use)."""

    depth_m: float = 0.60
    wall_reach_m: float = 0.30
    strip_reach_m: float = 0.90
    min_leg_length_m: float = 0.60
    corner_reach_m: float = 0.50
    min_depth_m: float = 0.35  # thinnest bar that still counts as a counter leg
    max_depth_m: float = 1.30  # thickest bar (peninsula); anything deeper is not a counter leg
    depth_snap_tol_m: float = 0.07  # measured depth snaps to a common size within this tolerance
    depth_std_a_m: float = 0.60
    depth_std_b_m: float = 0.65
    depth_std_c_m: float = 0.90
    depth_std_d_m: float = 1.20
    counter_min_m: float = 0.80  # DSM counter-height band above the floor
    counter_max_m: float = 1.00
    raised_min_m: float = 0.55  # evidence below this height above the floor is not a counter
    close_m: float = 0.12  # closing radius: objects on the counter and small gaps belong to the footprint
    bridge_m: float = 0.25  # no-data gaps up to this width inside the counter are bridged before the legs are measured
    open_m: float = 0.12  # opening radius: protrusions (objects beside the counter) thinner than twice this are cut off
    long_leg_m: float = 1.00  # a column run this long marks a leg running along the columns (others are found as rows)
    run_gap_m: float = 0.40  # no-data gaps up to this length inside a leg do not interrupt it
    chamfer_min_m: float = 0.30  # smallest inner-corner chamfer that is still drawn
    chamfer_max_m: float = 0.60
    band_reach_m: float = 0.20  # counter-height surfaces join only this close to the labelled counter
    min_label_overlap: float = 0.40  # a height-only component needs this share of kitchen-label pixels


@dataclass(slots=True)
class KitchenLeg:
    """One straight leg: back edge on the wall line, depth into the room."""

    origin: np.ndarray  # a point on the wall line (along coordinate 0)
    tangent: np.ndarray
    normal: np.ndarray  # pointing into the room
    start: float
    end: float
    depth: float

    def polygon(self) -> Polygon:
        a, b = self.origin + self.tangent * self.start, self.origin + self.tangent * self.end
        return Polygon([tuple(a), tuple(b), tuple(b + self.normal * self.depth), tuple(a + self.normal * self.depth)])


@dataclass(slots=True)
class KitchenRun:
    """The clean run: union polygon(s), its legs and a shape name (straight / L / U / multi)."""

    pieces: list[Polygon]
    legs: list[KitchenLeg]
    shape: str


@dataclass(slots=True)
class _Bar:
    """Axis-aligned rectangle in the rotated (u, v) room frame."""

    u0: float
    u1: float
    v0: float
    v1: float
    back: str | None = None  # thickness edge flush on a wall: "u0" / "u1" / "v0" / "v1"
    fixed: set[str] = field(default_factory=set)  # every edge that was snapped onto a wall

    @property
    def horizontal(self) -> bool:
        return (self.u1 - self.u0) >= (self.v1 - self.v0)

    def polygon(self) -> Polygon:
        return box(self.u0, self.v0, self.u1, self.v1)


def build_kitchen_run(
    mask: np.ndarray,
    grid: RasterGrid,
    wall_lines: Iterable[LineString],
    units_per_metre: float,
    settings: KitchenRunSettings = KitchenRunSettings(),
    height_map: np.ndarray | None = None,
) -> KitchenRun | None:
    """Build the counter run that follows the measured counter edges; None when it is not wall based.

    The footprint (kitchen labels confirmed by the DSM, merged with connected counter-height surfaces, objects on
    the counter absorbed) is decomposed into rectangular legs along the dominant wall axes.  Legs are wall-attached
    or peninsulas attached to a wall-attached leg; depth and ends are measured (depth snaps to common sizes within a
    tolerance, ends and depth edges snap flush to walls).  A single chamfer at an inner corner is added only when
    the evidence fills it.
    """
    upm = max(float(units_per_metre), 1e-9)
    evidence = _counter_evidence(mask, height_map, upm, settings, grid.pixel_size)
    rows, cols = np.nonzero(evidence)
    if len(rows) < 8:
        return None
    walls = list(wall_lines)
    angle = _dominant_axis(walls)
    cos, sin = math.cos(angle), math.sin(angle)
    xs = grid.min_x + (cols + 0.5) * grid.pixel_size
    ys = grid.max_y - (rows + 0.5) * grid.pixel_size
    us, vs = xs * cos + ys * sin, -xs * sin + ys * cos
    res = max(grid.pixel_size, 0.02 * upm)
    radius = max(1, int(round(settings.close_m * upm / res)))
    margin = radius + 3
    u_min, v_min = float(us.min()) - (margin + 0.5) * res, float(vs.min()) - (margin + 0.5) * res
    width = int((us.max() - u_min) / res) + margin + 2
    height = int((vs.max() - v_min) / res) + margin + 2
    raster = np.zeros((height, width), dtype=np.uint8)
    raster[((vs - v_min) / res).astype(int), ((us - u_min) / res).astype(int)] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(raster, contours, -1, 1, thickness=cv2.FILLED)  # objects on the counter leave no holes
    fine_raster = raster.astype(bool)  # evidence for the chamfer test (a wide closing would fake corner fillets)
    bridge = max(radius, int(round(settings.bridge_m * upm / res)))
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge + 1, 2 * bridge + 1))
    raster = cv2.morphologyEx(np.pad(raster, bridge + 1), cv2.MORPH_CLOSE, bridge_kernel)[bridge + 1:-(bridge + 1), bridge + 1:-(bridge + 1)]
    opening = max(1, int(round(settings.open_m * upm / res)))
    raster = cv2.morphologyEx(np.pad(raster, opening + 1), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * opening + 1, 2 * opening + 1)))[opening + 1:-(opening + 1), opening + 1:-(opening + 1)]
    dmin, dmax = max(2, int(math.ceil(settings.min_depth_m * upm / res))), max(3, int(settings.max_depth_m * upm / res))
    lmin = max(dmin, int(math.ceil(settings.min_leg_length_m * upm / res)))
    boxes = _extract_bars(raster, dmin, dmax, lmin, max(lmin, int(round(settings.long_leg_m * upm / res))), max(1, int(round(settings.run_gap_m * upm / res))))
    bars = [_Bar(u_min + c0 * res, u_min + c1 * res, v_min + r0 * res, v_min + r1 * res) for r0, r1, c0, c1 in boxes]
    bars = _drop_redundant(bars)
    segments = _wall_segments(walls, cos, sin)
    for bar in bars:
        _snap_to_walls(bar, segments, settings, upm)
    attached = _attached_bars(bars, settings.run_gap_m * upm)
    if not attached:
        # Ragged evidence (diagonal corner fill, merged perpendicular legs) defeats the raster-shape extraction:
        # measure the legs from the walls instead.
        attached = _wall_anchored_bars(raster, u_min, v_min, res, segments, (dmin, dmax, lmin, opening), settings, upm)
    _bridge_gaps(attached, settings.run_gap_m * upm)
    if not attached:
        return None
    for bar in attached:
        _snap_depth(bar, settings, upm)
    _bridge_gaps(attached, settings.run_gap_m * upm)  # depths changed: re-close the joints
    _align_edges(attached, settings.depth_snap_tol_m * upm)
    union = unary_union([bar.polygon() for bar in attached]).buffer(0)
    if isinstance(union, Polygon):
        union = _chamfer_corners(union, fine_raster, u_min, v_min, res, settings, upm)
    parts = [union] if isinstance(union, Polygon) else [part for part in getattr(union, "geoms", ()) if isinstance(part, Polygon)]
    parts = [_to_local(part.simplify(1e-6 * upm, preserve_topology=True), cos, sin) for part in parts if not part.is_empty]
    if not parts:
        return None
    centre_u = (min(bar.u0 for bar in attached) + max(bar.u1 for bar in attached)) / 2.0
    centre_v = (min(bar.v0 for bar in attached) + max(bar.v1 for bar in attached)) / 2.0
    legs = [_leg_of(bar, centre_u, centre_v, cos, sin) for bar in sorted(attached, key=lambda item: max(item.u1 - item.u0, item.v1 - item.v0), reverse=True)]
    shape_name = "multi" if len(parts) > 1 else {1: "straight", 2: "L", 3: "U"}.get(len(legs), "U+")
    return KitchenRun(parts, legs, shape_name)


def _counter_evidence(mask: np.ndarray, height_map: np.ndarray | None, upm: float, settings: KitchenRunSettings, pixel_size: float) -> np.ndarray:
    """Counter footprint evidence: kitchen-label pixels confirmed by height, plus connected counter-height surfaces.

    Without usable heights (no DSM, or none of the labelled pixels stands counter-high) the labels are used as they are.
    Objects standing on the counter are absorbed by the closing; unlabelled raised pixels only join when they touch
    the labelled counter, so a freestanding table next to the run is not merged into it.
    """
    labelled = np.asarray(mask, dtype=bool)
    if height_map is None:
        return labelled
    height = np.nan_to_num(np.asarray(height_map, dtype=np.float32), nan=-1e9, posinf=-1e9, neginf=-1e9)
    raised = height >= settings.raised_min_m * upm
    if float((labelled & raised).sum()) < 0.30 * float(labelled.sum()):
        return labelled  # the DSM does not confirm a counter height here
    core = labelled & raised
    band = (height >= settings.counter_min_m * upm) & (height <= settings.counter_max_m * upm)
    radius = max(1, int(round(settings.close_m * upm / pixel_size)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    reach = max(1, int(round(settings.band_reach_m * upm / pixel_size)))
    reach_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    # Counter-height surfaces join only right next to the labelled counter: a freestanding table or a stool
    # standing apart from it never becomes part of the run, and objects beside it cannot widen its edges.
    near_core = cv2.dilate(core.astype(np.uint8), reach_kernel).astype(bool)
    seed = cv2.morphologyEx((core | (band & near_core)).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    count, labels = cv2.connectedComponents(seed, connectivity=8)
    keep = np.zeros_like(core)
    for index in range(1, count):
        component = labels == index
        if float((component & core).sum()) >= settings.min_label_overlap * float(component.sum()):
            keep |= component
    return keep


def _dominant_axis(wall_lines: Iterable[LineString], fallback: float = 0.0) -> float:
    """Length-weighted dominant wall direction (radians, modulo 90 degrees)."""
    total = 0.0 + 0.0j
    for line in wall_lines:
        coords = np.asarray(line.coords, dtype=float)
        for first, second in zip(coords[:-1], coords[1:], strict=True):
            vector = second - first
            length = float(np.hypot(*vector))
            if length > 1e-9:
                total += length * np.exp(4j * math.atan2(vector[1], vector[0]))
    return float(np.angle(total) / 4.0) if abs(total) > 1e-12 else fallback


def _longest_runs(raster: np.ndarray, gap: int) -> np.ndarray:
    """Per column the longest run along the rows as (length, start, stop); gaps up to ``gap`` pixels are bridged."""
    output = np.zeros((raster.shape[1], 3), dtype=np.int32)
    for column in range(raster.shape[1]):
        indices = np.flatnonzero(raster[:, column])
        if len(indices) == 0:
            continue
        pieces = np.split(indices, np.flatnonzero(np.diff(indices) > gap + 1) + 1)
        piece = max(pieces, key=lambda item: item[-1] - item[0])
        output[column] = (piece[-1] - piece[0] + 1, piece[0], piece[-1] + 1)
    return output


def _bar_groups(raster: np.ndarray, flag_length: int, gap: int, dmin: int, dmax: int) -> list[tuple[int, int, int, int]]:
    """Bars running along the rows axis of ``raster`` as (row0, row1, col0, col1) boxes.

    A bar is a group of neighbouring columns whose longest run is at least ``flag_length``; its depth is the group
    width and its ends are the median run ends, so ragged edges and objects beside the counter do not matter.
    """
    runs = _longest_runs(raster, gap)
    flagged = runs[:, 0] >= flag_length
    boxes: list[tuple[int, int, int, int]] = []
    column = 0
    while column < len(flagged):
        if not flagged[column]:
            column += 1
            continue
        stop = column
        while stop + 1 < len(flagged) and (flagged[stop + 1] or (stop + 2 < len(flagged) and flagged[stop + 2])):
            stop += 1
        columns = np.arange(column, stop + 1)
        lengths = runs[columns, 0]
        trusted = columns[lengths >= 0.8 * float(np.median(lengths[lengths > 0]))]
        column = stop + 1
        if len(trusted) == 0:
            continue
        c0, c1 = int(trusted.min()), int(trusted.max()) + 1
        width = c1 - c0
        median_length = float(np.median(runs[trusted, 0]))
        if width < dmin or width > dmax or median_length < 1.25 * width:
            continue
        boxes.append((int(np.median(runs[trusted, 1])), int(np.median(runs[trusted, 2])), c0, c1))
    return boxes


def _extract_bars(raster: np.ndarray, dmin: int, dmax: int, lmin: int, flag_length: int, gap: int) -> list[tuple[int, int, int, int]]:
    """Legs of the footprint as (row0, row1, col0, col1) boxes: long vertical legs first, then horizontal ones in the rest."""
    raster = np.asarray(raster, dtype=bool)
    vertical = _bar_groups(raster, flag_length, gap, dmin, dmax)
    rest = raster.copy()
    for r0, r1, c0, c1 in vertical:
        rest[r0:r1, c0:c1] = False
    # Horizontal legs: transpose so that the same routine measures rows (thickness = rows, length = columns).
    horizontal = [(c0, c1, r0, r1) for r0, r1, c0, c1 in _bar_groups(rest.T, lmin, gap, dmin, dmax)]
    return vertical + horizontal


def _perpendicular_depths(plane: np.ndarray, boundary: float, forward: bool, reach: int, bridge: int) -> np.ndarray:
    """Evidence depth (pixels) from a wall line into the room for every column of ``plane`` (rows = across the wall).

    ``boundary`` is the wall line as a fractional row edge.  The first evidence pixel must lie within ``reach``
    pixels of the wall; gaps up to ``bridge`` pixels inside the evidence are bridged.  0 where there is no leg.
    """
    edge = int(round(boundary))
    part = plane[edge:] if forward else plane[:edge][::-1]
    depths = np.zeros(plane.shape[1], dtype=np.int32)
    if part.shape[0] == 0:
        return depths
    for column in range(plane.shape[1]):
        indices = np.flatnonzero(part[:, column])
        if len(indices) == 0 or indices[0] > reach:
            continue
        breaks = np.flatnonzero(np.diff(indices) > bridge + 1)
        stop = indices[breaks[0]] if len(breaks) else indices[-1]
        depths[column] = int(stop) + 1
    return depths


def _depth_intervals(depths: np.ndarray, min_depth: int, gap: int) -> list[tuple[int, int]]:
    """Index intervals [start, stop) of columns that are at least ``min_depth`` deep; gaps up to ``gap`` are bridged."""
    good = np.flatnonzero(depths >= min_depth)
    if len(good) == 0:
        return []
    breaks = np.flatnonzero(np.diff(good) > gap + 1)
    starts = np.concatenate(([good[0]], good[breaks + 1]))
    stops = np.concatenate((good[breaks], [good[-1]])) + 1
    return [(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]


def _wall_anchored_bars(
    raster: np.ndarray,
    u_min: float,
    v_min: float,
    res: float,
    segments: list[tuple[str, float, float, float]],
    pixels: tuple[int, int, int, int],
    settings: KitchenRunSettings,
    upm: float,
) -> list[_Bar]:
    """Wall-attached legs measured from the walls, plus peninsulas attached to them (fallback for ragged evidence).

    Along every axis-parallel wall (extended by the corner reach) the evidence depth into the room is measured
    perpendicular to the wall.  Stretches that are deep enough and long enough become legs whose back edge is the wall
    line and whose depth is the median measured depth, so a perpendicular leg or a peninsula crossing the profile and
    a diagonal corner fill do not inflate it.  What remains of the evidence after the legs are removed is searched
    for peninsulas with the regular extraction; only those touching a wall leg are kept.
    """
    dmin, dmax, lmin, opening = pixels
    plane = np.asarray(raster, dtype=bool)
    bridge = max(1, int(round(settings.bridge_m * upm / res)))
    reach = max(1, int(round(settings.wall_reach_m * upm / res))) + bridge  # the no-data gap before the counter is bridged too
    gap = max(1, int(round(settings.run_gap_m * upm / res)))
    extension = settings.corner_reach_m * upm
    pad = reach + 2  # a wall may lie outside the evidence raster (counter set back from the wall)
    plane = np.pad(plane, pad)
    u_min, v_min = u_min - pad * res, v_min - pad * res
    legs: list[_Bar] = []
    for kind, coordinate, lo, hi in segments:
        view, along_min, across_min = (plane, u_min, v_min) if kind == "h" else (plane.T, v_min, u_min)
        first = max(0, int(math.floor((lo - extension - along_min) / res)))
        last = min(view.shape[1], int(math.ceil((hi + extension - along_min) / res)))
        boundary = (coordinate - across_min) / res
        if last - first < lmin or not 0.0 <= boundary <= view.shape[0]:
            continue
        for forward in (True, False):
            depths = np.zeros(view.shape[1], dtype=np.int32)
            depths[first:last] = _perpendicular_depths(view[:, first:last], boundary, forward, reach, bridge)
            for start, stop in _depth_intervals(depths, dmin, gap):
                length = stop - start
                if length < lmin:
                    continue
                measured = depths[start:stop]
                plausible = measured[(measured >= dmin) & (measured <= dmax)]  # deeper columns are a crossing leg, not this one
                depth = float(np.clip(np.median(plausible if len(plausible) else measured[measured >= dmin]), dmin, dmax)) * res
                if length * res < 1.25 * depth:
                    continue
                a0, a1 = along_min + start * res, along_min + stop * res
                side = depth if forward else -depth
                if kind == "h":
                    bar = _Bar(a0, a1, coordinate + min(0.0, side), coordinate + max(0.0, side), "v0" if forward else "v1")
                else:
                    bar = _Bar(coordinate + min(0.0, side), coordinate + max(0.0, side), a0, a1, "u0" if forward else "u1")
                bar.fixed.add(bar.back)  # type: ignore[arg-type]
                legs.append(bar)
    legs = _drop_redundant(legs)
    if not legs:
        return []
    # Peninsulas: evidence left over after the wall legs are cut out (a little wider than the legs themselves).
    rest = plane.copy()
    margin = max(1, int(round(settings.close_m * upm / res)))
    for bar in legs:
        c0, c1 = int((bar.u0 - u_min) / res) - margin, int(math.ceil((bar.u1 - u_min) / res)) + margin
        r0, r1 = int((bar.v0 - v_min) / res) - margin, int(math.ceil((bar.v1 - v_min) / res)) + margin
        rest[max(0, r0):max(0, r1), max(0, c0):max(0, c1)] = False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * opening + 1, 2 * opening + 1))
    rest = cv2.morphologyEx(np.pad(rest.astype(np.uint8), opening + 1), cv2.MORPH_OPEN, kernel)[opening + 1:-(opening + 1), opening + 1:-(opening + 1)].astype(bool)
    boxes = _extract_bars(rest, dmin, dmax, lmin, max(lmin, int(round(settings.long_leg_m * upm / res))), gap)
    extra = [_Bar(u_min + c0 * res, u_min + c1 * res, v_min + r0 * res, v_min + r1 * res) for r0, r1, c0, c1 in boxes]
    for bar in extra:
        _grow_to_evidence(bar, rest, u_min, v_min, res, dmax)
        _snap_to_walls(bar, segments, settings, upm)
    return _attached_bars(legs + extra, settings.run_gap_m * upm)


def _grow_to_evidence(bar: _Bar, plane: np.ndarray, u_min: float, v_min: float, res: float, max_depth: int) -> None:
    """Move both thickness edges of a peninsula outwards while at least half of the bar's length is still evidence.

    The extraction trusts only rows that are as long as the median row, so rounded or ragged long edges make the
    bar thinner than the counter; this puts the edge where the evidence thins out to half.
    """
    horizontal = bar.horizontal
    c0, c1 = int(round((bar.u0 - u_min) / res)), int(round((bar.u1 - u_min) / res))
    r0, r1 = int(round((bar.v0 - v_min) / res)), int(round((bar.v1 - v_min) / res))
    view = plane[:, c0:c1] if horizontal else plane[r0:r1, :].T  # lines across the thickness axis
    low, high = (r0, r1) if horizontal else (c0, c1)
    if view.shape[1] == 0:
        return
    while high - low < max_depth and low > 0 and view[low - 1].mean() >= 0.5:
        low -= 1
    while high - low < max_depth and high < view.shape[0] and view[high].mean() >= 0.5:
        high += 1
    if horizontal:
        bar.v0, bar.v1 = v_min + low * res, v_min + high * res
    else:
        bar.u0, bar.u1 = u_min + low * res, u_min + high * res


def _drop_redundant(bars: list[_Bar]) -> list[_Bar]:
    """Drop bars that lie (almost) completely inside the other bars."""
    kept = list(bars)
    for bar in sorted(bars, key=lambda item: item.polygon().area):
        others = [other for other in kept if other is not bar]
        if others and bar.polygon().intersection(unary_union([other.polygon() for other in others])).area >= 0.85 * bar.polygon().area:
            kept.remove(bar)
    return kept


def _wall_segments(walls: list[LineString], cos: float, sin: float) -> list[tuple[str, float, float, float]]:
    """Wall segments parallel to the room axes as (kind, coordinate, lo, hi); kind "h" runs along u (v constant)."""
    output: list[tuple[str, float, float, float]] = []
    for line in walls:
        coords = np.asarray(line.coords, dtype=float)
        for first, second in zip(coords[:-1], coords[1:], strict=True):
            a = (first[0] * cos + first[1] * sin, -first[0] * sin + first[1] * cos)
            b = (second[0] * cos + second[1] * sin, -second[0] * sin + second[1] * cos)
            du, dv = abs(b[0] - a[0]), abs(b[1] - a[1])
            if du >= 10.0 * dv and du > 1e-9:
                output.append(("h", (a[1] + b[1]) / 2.0, min(a[0], b[0]), max(a[0], b[0])))
            elif dv >= 10.0 * du and dv > 1e-9:
                output.append(("v", (a[0] + b[0]) / 2.0, min(a[1], b[1]), max(a[1], b[1])))
    return output


def _snap_to_walls(bar: _Bar, segments: list[tuple[str, float, float, float]], settings: KitchenRunSettings, upm: float) -> None:
    """Snap the back edge (and the ends) of a bar flush onto walls within reach; records the wall-snapped edges."""
    horizontal = bar.horizontal
    thick_kind, long_kind = ("h", "v") if horizontal else ("v", "h")
    lo_attr, hi_attr = ("v0", "v1") if horizontal else ("u0", "u1")
    end_lo, end_hi = ("u0", "u1") if horizontal else ("v0", "v1")
    along = (bar.u0, bar.u1) if horizontal else (bar.v0, bar.v1)
    across = (bar.v0, bar.v1) if horizontal else (bar.u0, bar.u1)
    best: tuple[float, str, float] | None = None
    for kind, coordinate, lo, hi in segments:
        if kind != thick_kind or min(hi, along[1]) - max(lo, along[0]) < 0.30 * upm:
            continue
        for attr, value in ((lo_attr, across[0]), (hi_attr, across[1])):
            gap = abs(value - coordinate)
            if gap <= settings.wall_reach_m * upm and (best is None or gap < best[0]):
                best = (gap, attr, coordinate)
    if best is not None:
        _, attr, coordinate = best
        trial = (coordinate, across[1]) if attr == lo_attr else (across[0], coordinate)
        if trial[1] - trial[0] >= settings.min_depth_m * upm * 0.9:
            setattr(bar, attr, coordinate)
            bar.back = attr
            bar.fixed.add(attr)
    across = (bar.v0, bar.v1) if horizontal else (bar.u0, bar.u1)
    for kind, coordinate, lo, hi in segments:
        if kind != long_kind or min(hi, across[1]) - max(lo, across[0]) < 0.20 * upm:
            continue
        for attr, value in ((end_lo, along[0]), (end_hi, along[1])):
            if abs(value - coordinate) <= settings.corner_reach_m * upm:
                setattr(bar, attr, coordinate)
                bar.fixed.add(attr)


def _attached_bars(bars: list[_Bar], reach: float) -> list[_Bar]:
    """Wall-attached bars plus peninsulas that touch (transitively) a wall-attached bar."""
    attached = [bar for bar in bars if bar.back is not None]
    if not attached:
        return []
    rest = [bar for bar in bars if bar.back is None]
    changed = True
    while changed:
        changed = False
        for bar in list(rest):
            if any(bar.polygon().distance(other.polygon()) <= reach for other in attached):
                attached.append(bar)
                rest.remove(bar)
                changed = True
    return attached


def _bridge_gaps(bars: list[_Bar], gap: float) -> None:
    """Close the no-data gap between the end of a leg and the side of a perpendicular leg it faces."""
    for horizontal in (bar for bar in bars if bar.horizontal):
        for vertical in (bar for bar in bars if not bar.horizontal):
            overlap = min(horizontal.v1, vertical.v1) - max(horizontal.v0, vertical.v0)
            if overlap < 0.5 * (horizontal.v1 - horizontal.v0):
                continue
            if 0.0 < horizontal.u0 - vertical.u1 <= gap:
                horizontal.u0 = vertical.u1
            if 0.0 < vertical.u0 - horizontal.u1 <= gap:
                horizontal.u1 = vertical.u0
    for vertical in (bar for bar in bars if not bar.horizontal):
        for horizontal in (bar for bar in bars if bar.horizontal):
            overlap = min(horizontal.u1, vertical.u1) - max(horizontal.u0, vertical.u0)
            if overlap < 0.5 * (vertical.u1 - vertical.u0):
                continue
            if 0.0 < horizontal.v0 - vertical.v1 <= gap:
                vertical.v1 = horizontal.v0
            if 0.0 < vertical.v0 - horizontal.v1 <= gap:
                vertical.v0 = horizontal.v1


def _snap_depth(bar: _Bar, settings: KitchenRunSettings, upm: float) -> None:
    """Snap the measured depth to a common size (0.60 preferred) within the tolerance, else keep the measurement."""
    standards = [value * upm for value in (settings.depth_std_a_m, settings.depth_std_b_m, settings.depth_std_c_m, settings.depth_std_d_m)]
    horizontal = bar.horizontal
    lo_attr, hi_attr = ("v0", "v1") if horizontal else ("u0", "u1")
    low, high = getattr(bar, lo_attr), getattr(bar, hi_attr)
    thickness = high - low
    within = [value for value in standards if abs(thickness - value) <= settings.depth_snap_tol_m * upm + 1e-9]
    if not within:
        return
    target = standards[0] if standards[0] in within else min(within, key=lambda value: abs(value - thickness))
    if bar.back == lo_attr or lo_attr in bar.fixed and hi_attr not in bar.fixed:
        setattr(bar, hi_attr, low + target)
    elif bar.back == hi_attr or hi_attr in bar.fixed and lo_attr not in bar.fixed:
        setattr(bar, lo_attr, high - target)
    else:
        centre = (low + high) / 2.0
        setattr(bar, lo_attr, centre - target / 2.0)
        setattr(bar, hi_attr, centre + target / 2.0)


def _align_edges(bars: list[_Bar], tolerance: float) -> None:
    """Merge parallel edges of different bars that lie within ``tolerance`` (wall-snapped edges win)."""
    for names in (("u0", "u1"), ("v0", "v1")):
        items = sorted(((getattr(bar, name), bar, name) for bar in bars for name in names), key=lambda item: item[0])
        group: list[tuple[float, _Bar, str]] = []
        for item in items + [None]:  # type: ignore[list-item]
            if item is not None and (not group or item[0] - group[-1][0] <= tolerance):
                group.append(item)
                continue
            if len(group) > 1 and len({id(member[1]) for member in group}) == len(group):
                fixed = [member[0] for member in group if member[2] in member[1].fixed]
                value = fixed[0] if fixed else sum(member[0] for member in group) / len(group)
                for _, bar, name in group:
                    setattr(bar, name, value)
            group = [item] if item is not None else []


def _chamfer_corners(polygon: Polygon, raster: np.ndarray, u_min: float, v_min: float, res: float, settings: KitchenRunSettings, upm: float) -> Polygon:
    """Add at most one diagonal at an inner corner, and only where the evidence fills that corner triangle."""
    ring = list(polygon.exterior.coords)[:-1]
    sign = 1.0 if polygon.exterior.is_ccw else -1.0
    best: tuple[float, Polygon] | None = None
    for index, vertex in enumerate(ring):
        previous, following = np.asarray(ring[index - 1]), np.asarray(ring[(index + 1) % len(ring)])
        vertex = np.asarray(vertex)
        e1, e2 = previous - vertex, following - vertex
        l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
        if l1 < 1e-9 or l2 < 1e-9 or sign * float(e1[0] * (-e2[1]) - e1[1] * (-e2[0])) >= 0.0:
            continue  # convex corner (cross of incoming and outgoing edge has the polygon's turning sign)
        e1, e2 = e1 / l1, e2 / l2
        c = min(settings.chamfer_max_m * upm, 0.8 * min(l1, l2))
        while c >= settings.chamfer_min_m * upm:
            steps = np.arange(res / 2.0, c, res)
            a, b = np.meshgrid(steps, steps)
            points = vertex[None, None, :] + a[..., None] * e1 + b[..., None] * e2
            iu = ((points[..., 0] - u_min) / res).astype(int)
            iv = ((points[..., 1] - v_min) / res).astype(int)
            inside = (iu >= 0) & (iu < raster.shape[1]) & (iv >= 0) & (iv < raster.shape[0])
            filled = np.zeros(a.shape, dtype=bool)
            filled[inside] = raster[iv[inside], iu[inside]]
            core = (a >= 0.25 * c) & (b >= 0.25 * c)  # ignore the slivers along the two counter edges
            triangle = (a + b) <= c
            if not (core & triangle).any() or not (core & ~triangle).any():
                break
            near, far = float(filled[core & triangle].mean()), float(filled[core & ~triangle].mean())
            if near >= 0.60 and far <= 0.35:
                score = c + 0.01 * (near - far)  # the largest evidenced chamfer wins
                if best is None or score > best[0]:
                    best = (score, Polygon([tuple(vertex), tuple(vertex + c * e1), tuple(vertex + c * e2)]))
                break
            c -= 0.05 * upm
    if best is None:
        return polygon
    merged = unary_union([polygon, best[1]]).buffer(0)
    return merged if isinstance(merged, Polygon) else polygon


def _to_local(polygon: Polygon, cos: float, sin: float) -> Polygon:
    def convert(coords: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(u * cos - v * sin, u * sin + v * cos) for u, v in coords]

    return Polygon(convert(polygon.exterior.coords), [convert(ring.coords) for ring in polygon.interiors])


def _leg_of(bar: _Bar, centre_u: float, centre_v: float, cos: float, sin: float) -> KitchenLeg:
    """Leg with its back edge on the wall (peninsulas: on the side away from the run centre) and normal into the room."""
    horizontal = bar.horizontal
    if horizontal:
        back = bar.back or ("v0" if (bar.v0 + bar.v1) / 2.0 < centre_v else "v1")
        sign = 1.0 if back == "v0" else -1.0
        origin_uv, tangent_uv, normal_uv = (bar.u0, bar.v0 if back == "v0" else bar.v1), (1.0, 0.0), (0.0, sign)
        length, depth = bar.u1 - bar.u0, bar.v1 - bar.v0
    else:
        back = bar.back or ("u0" if (bar.u0 + bar.u1) / 2.0 < centre_u else "u1")
        sign = 1.0 if back == "u0" else -1.0
        origin_uv, tangent_uv, normal_uv = (bar.u0 if back == "u0" else bar.u1, bar.v0), (0.0, 1.0), (sign, 0.0)
        length, depth = bar.v1 - bar.v0, bar.u1 - bar.u0

    def rotate(vector: tuple[float, float]) -> np.ndarray:
        return np.asarray((vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos))

    return KitchenLeg(rotate(origin_uv), rotate(tangent_uv), rotate(normal_uv), 0.0, float(length), float(depth))


def appliance_insert(
    run: KitchenRun,
    kinds: tuple[str, ...] | str,
    appliance_points: np.ndarray,
    units_per_metre: float,
    label: str,
) -> dict[str, object] | None:
    """Catalogue insert for a labelled appliance, mounted on the run's wall edge (+Y into the room).

    ``appliance_points`` are the local coordinates of the appliance's label pixels.  The catalogue
    width nearest to the observed extent along the run is used; the block is centred on the
    observed centre, clamped to stay on the run.
    """
    upm = max(float(units_per_metre), 1e-9)
    names = (kinds,) if isinstance(kinds, str) else tuple(kinds)
    points = np.asarray(appliance_points, dtype=float)
    # Mount on the leg that holds most of the appliance's label pixels (a label may straddle a corner).
    shapes = [item.polygon().buffer(0.02 * upm) for item in run.legs]
    counts = [sum(shape.covers(Point(float(x), float(y))) for x, y in points[:: max(1, len(points) // 300)]) for shape in shapes]
    if max(counts) > 0:
        leg = run.legs[int(np.argmax(counts))]
        points = np.asarray([point for point in points if leg.polygon().buffer(0.02 * upm).covers(Point(float(point[0]), float(point[1])))])
    else:
        leg = min(run.legs, key=lambda item: item.polygon().distance(Point(*map(float, points.mean(axis=0)))))
    if len(points) == 0:
        points = np.asarray(appliance_points, dtype=float)
    centre = points.mean(axis=0)
    along = (points - leg.origin) @ leg.tangent
    extent_m = float(along.max() - along.min()) / upm
    entry = min((entry for name in names for entry in CATALOGUE[name]), key=lambda item: abs(item.width_m - extent_m) / item.width_m)
    half = entry.width_m * upm / 2.0
    if leg.end - leg.start < 2.0 * half:
        return None
    position = float(np.clip(float((centre - leg.origin) @ leg.tangent), leg.start + half, leg.end - half))
    x_axis = leg.tangent if float(np.dot(np.asarray((-leg.tangent[1], leg.tangent[0])), leg.normal)) > 0.0 else -leg.tangent
    insert = _uniform_insert(entry, leg.origin + leg.tangent * position, x_axis, upm, {"attributes": {"LABEL": label}}, 0.0, True)
    insert["_catalogue"]["on_kitchen_run"] = True  # type: ignore[index]
    return insert
