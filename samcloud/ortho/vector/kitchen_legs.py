"""Kitchen legs that the label raster lost, established from the scanned cabinet front itself.

The raster run needs the labelled counter to be ``min_depth_m`` deep.  Where objects on the worktop or an unlabelled
back strip make it thinner, a whole leg is missing although its front is scanned perfectly.  Starting at an end of a
known leg, a perpendicular front becomes a leg only when ALL of this is measured:

* a straight vertical face from plinth to worktop, without long gaps, at least ``min_length_m`` long;
* the worktop of the same run behind it: points at the same level;
* kitchen labels along it (a table or a sideboard along the second wall has none);
* it is no tall row (no face above the worktop);
* it is joined to the known leg: the two fronts meet, or a diagonal between them with line support of its own;
* a wall at cabinet depth behind it, or, for a free-standing leg (peninsula), its own scanned back face.

Nothing is completed by symmetry or by a default.  Coordinates are wall-frame plan units, heights are metres.
"""

from __future__ import annotations

from collections.abc import MutableSequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import box

from .kitchen_fronts import KitchenFrontSettings, ScanPoints, _bins, _cover, _front_offset, _inner_corners, _measure_corner, _runs


@dataclass(frozen=True, slots=True)
class FrontLegSettings:
    """Evidence thresholds of a leg from its front (config ``indoor_kitchen_front_legs:``)."""

    enabled: bool = True
    min_length_m: float = 0.60  # straight front
    max_length_m: float = 4.00  # search range along the new leg
    end_before_m: float = 0.90  # the new front lies this far before / beyond the end of the known leg (fitted ends are rough)
    end_beyond_m: float = 1.10
    gap_m: float = 0.20  # doors gaps, handles and small occlusions inside one front
    min_coverage: float = 0.75  # stations of the front seen over the front height
    min_top_coverage: float = 0.60  # the face reaches the worktop
    worktop_tolerance_m: float = 0.07  # same level as the known run
    min_worktop_share: float = 0.60
    label_reach_m: float = 0.45  # kitchen labels this far behind the front
    min_label_share: float = 0.60
    tall_low_m: float = 1.15
    tall_high_m: float = 1.60
    max_tall_share: float = 0.40
    join_m: float = 0.15  # fronts that meet
    min_depth_m: float = 0.45  # wall behind the front
    max_depth_m: float = 0.80
    min_wall_share: float = 0.80
    free_min_depth_m: float = 0.35  # peninsula: front to its scanned back face
    free_max_depth_m: float = 0.90
    min_back_face_share: float = 0.60
    max_overlap: float = 0.20  # a place that a known leg already occupies
    # A free-standing leg of the label raster takes its measured body (``refit_free_legs``).  The labels of a peninsula often cover a
    # raised bar and what stands beside it, so the raster leg is only roughly placed; the measured body is the same leg when the two
    # share at least this much of the smaller one (half: a body one cabinet depth beside it shares nothing).
    refit_free: bool = True
    refit_min_overlap: float = 0.50


@dataclass(slots=True)
class _View:
    """Coordinates at one end of a known leg: ``a`` outwards past the end, ``b`` from its front into the room."""

    axis: int  # along axis of the known leg
    end: float
    front: float
    out: float
    into: float

    def ab(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (uv[:, self.axis] - self.end) * self.out, (uv[:, 1 - self.axis] - self.front) * self.into

    def bounds(self, a0: float, a1: float, b0: float, b1: float) -> tuple[float, float, float, float]:
        along = sorted((self.end + self.out * a0, self.end + self.out * a1))
        across = sorted((self.front + self.into * b0, self.front + self.into * b1))
        (u0, u1), (v0, v1) = (along, across) if self.axis == 0 else (across, along)
        return u0, v0, u1, v1


@dataclass(slots=True)
class _Face:
    offset: float  # a
    start: float  # b
    stop: float


def recover_front_legs(
    legs: MutableSequence[Any],
    points: ScanPoints,
    labels: np.ndarray,
    obstacles: Any,
    room: Any,
    front_settings: KitchenFrontSettings,
    units_per_metre: float,
    counter_top_m: float = 0.90,
    settings: FrontLegSettings = FrontLegSettings(),
    trace: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Append the legs measured from their fronts to ``legs`` (``_LegBox``); returns what supported each of them.

    ``labels`` are wall-frame points of the kitchen labels.  A known leg that meets a recovered one reaches into the
    shared corner.  A recovered leg can itself carry the next one (the U).  ``trace`` collects every rejected front
    with its reason.
    """
    upm = max(float(units_per_metre), 1e-9)
    labels = np.asarray(labels, dtype=float).reshape(-1, 2)
    if not settings.enabled or not front_settings.enabled or not len(points.uv) or not len(labels):
        return []
    low, high = front_settings.band_min_m, min(front_settings.band_max_m, counter_top_m - 0.04)
    if high - low < 0.30:
        return []
    band = points.band(low, high)
    worktop = points.band(counter_top_m - settings.worktop_tolerance_m, counter_top_m + settings.worktop_tolerance_m)
    tall = points.band(settings.tall_low_m, settings.tall_high_m)
    anchors = [index for index, leg in enumerate(legs) if leg.back is not None]  # grows with every wall-backed find
    found: list[dict[str, Any]] = []
    for index in anchors:
        for out in (-1.0, 1.0):
            leg = legs[index]
            axis = 0 if leg.horizontal else 1
            ends = ("u0", "u1") if leg.horizontal else ("v0", "v1")
            opposite = {"u0": "u1", "u1": "u0", "v0": "v1", "v1": "v0"}[leg.back]
            view = _View(axis, getattr(leg, ends[1] if out > 0 else ends[0]), getattr(leg, opposite), out, 1.0 if opposite.endswith("1") else -1.0)
            data = _Evidence(view, band, worktop, tall, labels, low, high, front_settings, settings, upm)
            for face in data.faces():
                made, reason = _leg_from_face(legs, index, data, face, obstacles, room)
                if made is None:
                    if trace is not None:
                        trace.append({"anchor": index, "out": out, "offset_m": round(face.offset / upm, 3), "start_m": round(face.start / upm, 3),
                                      "length_m": round((face.stop - face.start) / upm, 3), "reason": reason})
                    continue
                new, end, support = made
                setattr(leg, ends[1] if out > 0 else ends[0], end)
                legs.append(new)
                found.append(support)
                if new.back is not None:
                    anchors.append(len(legs) - 1)
                break
    return found


def refit_free_legs(
    legs: MutableSequence[Any],
    points: ScanPoints,
    labels: np.ndarray,
    obstacles: Any,
    room: Any,
    front_settings: KitchenFrontSettings,
    units_per_metre: float,
    counter_top_m: float = 0.90,
    settings: FrontLegSettings = FrontLegSettings(),
    trace: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Replace a free-standing raster leg (peninsula) by its body measured in the scan; returns what supported each of them.

    The same evidence as for a lost leg is asked for, seen from the wall-backed leg the peninsula is joined to: its scanned front
    from plinth to worktop, the worktop at the run's level, kitchen labels, no tall row, fronts that meet, and its own scanned back
    face.  The measured body must be the raster leg itself (``refit_min_overlap``).  Without all of it the raster leg stays as it is.
    Front, back and the end at the wall-backed leg are measured; the free end is never shorter than the raster leg's.

    The wall-backed leg then ends at the back face of the measured peninsula: it reaches into the shared corner, and a stretch
    beyond it that shows no cabinet front (objects labelled as kitchen beside the bar) is cut off.  Only the rough end of a fitted leg
    (``end_before_m``) is given up that way, and never where a front is scanned.
    """
    upm = max(float(units_per_metre), 1e-9)
    labels = np.asarray(labels, dtype=float).reshape(-1, 2)
    if not settings.enabled or not settings.refit_free or not front_settings.enabled or not len(points.uv) or not len(labels):
        return []
    low, high = front_settings.band_min_m, min(front_settings.band_max_m, counter_top_m - 0.04)
    if high - low < 0.30:
        return []
    band = points.band(low, high)
    worktop = points.band(counter_top_m - settings.worktop_tolerance_m, counter_top_m + settings.worktop_tolerance_m)
    tall = points.band(settings.tall_low_m, settings.tall_high_m)
    found: list[dict[str, Any]] = []
    for free in [leg for leg in legs if leg.back is None]:
        raster = box(free.u0, free.v0, free.u1, free.v1)
        for anchor in [leg for leg in legs if leg.back is not None and leg.horizontal != free.horizontal]:
            if raster.distance(box(anchor.u0, anchor.v0, anchor.u1, anchor.v1)) > settings.join_m * upm:
                continue
            axis = 0 if anchor.horizontal else 1
            ends = ("u0", "u1") if anchor.horizontal else ("v0", "v1")
            opposite = {"u0": "u1", "u1": "u0", "v0": "v1", "v1": "v0"}[anchor.back]
            centre = 0.5 * (getattr(free, ends[0]) + getattr(free, ends[1]))
            out = 1.0 if centre >= 0.5 * (getattr(anchor, ends[0]) + getattr(anchor, ends[1])) else -1.0
            # the view of a lost leg, set on the middle of the raster leg: both of its long faces lie in the search range
            view = _View(axis, centre, getattr(anchor, opposite), out, 1.0 if opposite.endswith("1") else -1.0)
            data = _Evidence(view, band, worktop, tall, labels, low, high, front_settings, settings, upm)
            others = [leg for leg in legs if leg is not free]
            index = next(number for number, leg in enumerate(others) if leg is anchor)
            made = None
            for face in data.faces():
                trial, reason = _leg_from_face(others, index, data, face, obstacles, room)
                if trial is not None:
                    body = box(trial[0].u0, trial[0].v0, trial[0].u1, trial[0].v1)
                    if body.intersection(raster).area >= settings.refit_min_overlap * min(body.area, raster.area):
                        made = trial
                        break
                    reason = "another place than the raster leg"
                if trace is not None:
                    trace.append({"refit": True, "offset_m": round(face.offset / upm, 3), "start_m": round(face.start / upm, 3),
                                  "length_m": round((face.stop - face.start) / upm, 3), "reason": reason})
            if made is None:
                continue
            new, _, support = made
            # The measured stretch ends where worktop and labels are no longer both seen (things standing on the end of a worktop hide
            # it); the raster end is where the labelled counter surface ends.  The free end keeps the farther of the two.
            far_name = ("v1" if view.into > 0 else "v0") if anchor.horizontal else ("u1" if view.into > 0 else "u0")
            far_end = max(getattr(new, far_name), getattr(free, far_name)) if view.into > 0 else min(getattr(new, far_name), getattr(free, far_name))
            free.u0, free.u1, free.v0, free.v1 = new.u0, new.u1, new.v0, new.v1
            setattr(free, far_name, far_end)
            support = {**support, "length_m": round(max(free.u1 - free.u0, free.v1 - free.v0) / upm, 3)}
            # the wall-backed leg ends at the back face of the peninsula
            end_name = ends[1] if out > 0 else ends[0]
            back_face = getattr(free, end_name)
            beyond = (getattr(anchor, end_name) - back_face) * out
            if beyond < 0.0:
                setattr(anchor, end_name, back_face)
            elif data.step <= beyond <= settings.end_before_m * upm:
                a, b = view.ab(band.uv)
                near = np.abs(b) <= front_settings.max_shift_m * upm + data.tolerance
                start = (back_face - centre) * out
                _, seen = _front_offset(a[near], b[near], _bins(band.z[near], low, high, front_settings.band_bins), start, start + beyond, front_settings, upm)
                if seen < settings.min_coverage:
                    setattr(anchor, end_name, back_face)
                    support = {**support, "anchor_cut_m": round(beyond / upm, 3), "anchor_front_share": round(float(seen), 3)}
            found.append(support)
            break
    return found


class _Evidence:
    """The scan seen from one end of a known leg, with the tests of a front found there."""

    def __init__(
        self, view: _View, band: ScanPoints, worktop: ScanPoints, tall: ScanPoints, labels: np.ndarray, low: float, high: float,
        front_settings: KitchenFrontSettings, settings: FrontLegSettings, upm: float,
    ) -> None:
        self.view, self.settings, self.fronts, self.upm, self.low, self.high = view, settings, front_settings, upm, low, high
        self.band = band
        self.step, self.tolerance = front_settings.station_m * upm, front_settings.line_tolerance_m * upm
        a0 = -(settings.end_before_m + front_settings.max_shift_m) * upm
        a1 = (settings.end_beyond_m + settings.free_max_depth_m + front_settings.max_shift_m) * upm
        b0, b1 = -0.30 * upm, (settings.max_length_m + 0.30) * upm

        def cut(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            a, b = view.ab(uv)
            near = (a >= a0) & (a <= a1) & (b >= b0) & (b <= b1)
            return a[near], b[near], near

        self.a, self.b, near = cut(band.uv)
        self.bins = _bins(band.z[near], low, high, front_settings.band_bins)
        self.top_a, self.top_b, _ = cut(worktop.uv)
        self.tall_a, self.tall_b, near = cut(tall.uv)
        self.tall_bins = _bins(tall.z[near], settings.tall_low_m, settings.tall_high_m, 4)
        self.label_a, self.label_b, _ = cut(labels)

    def faces(self) -> list[_Face]:
        """Straight vertical faces perpendicular to the known leg, longest first, one per position."""
        settings, upm, step, count = self.settings, self.upm, self.step, self.fronts.band_bins
        low, high, cell = -settings.end_before_m * upm, settings.end_beyond_m * upm, 0.01 * upm
        width, stations = int(round((high - low) / cell)), int(round(settings.max_length_m * upm / step))
        inside = (self.a >= low) & (self.a < high) & (self.b >= 0.0) & (self.b < stations * step)
        if inside.sum() < 12:
            return []
        seen = np.zeros((width, stations, count), dtype=bool)
        seen[np.minimum(((self.a[inside] - low) / cell).astype(int), width - 1), (self.b[inside] / step).astype(int), self.bins[inside]] = True
        seen = ndimage.maximum_filter1d(seen.astype(np.uint8), 2 * int(round(self.tolerance / cell)) + 1, axis=0, mode="constant")
        good = seen.sum(axis=2) >= self.fronts.station_min_bins
        gap = max(1, int(round(settings.gap_m * upm / step)))
        shortest = settings.min_length_m * upm
        ranked: list[tuple[int, int, int, int]] = []
        for position in np.flatnonzero(good.sum(axis=1) * step >= 0.5 * shortest):
            for first, last in _runs(good[position], gap):
                if (last - first) * step >= shortest:
                    ranked.append((int(good[position, first:last].sum()), -int(position), first, last))
        faces: list[_Face] = []
        def known(offset: float, start: float, stop: float) -> bool:
            return any(abs(offset - face.offset) <= 2.0 * self.tolerance and min(stop, face.stop) > max(start, face.start) for face in faces)

        for _, position, first, last in sorted(ranked, reverse=True):
            rough, start, stop = low + (0.5 - position) * cell, first * step, last * step
            if known(rough, start, stop):
                continue
            shift, _ = _front_offset(self.b, self.a - rough, self.bins, start, stop, self.fronts, upm)
            # the ends are read again on the refined line: the search line smears over what stands beside the front
            on = np.abs(self.a - rough - shift) <= self.tolerance
            here = _cover(self.b[on], self.bins[on], 0.0, stations * step, step, count).sum(axis=1) >= self.fronts.station_min_bins
            spans = [span for span in _runs(here, gap) if min(span[1], last) > max(span[0], first) and (span[1] - span[0]) * step >= shortest]
            if not spans:
                continue
            first, last = max(spans, key=lambda span: min(span[1], last) - max(span[0], first))
            if not known(rough + shift, first * step, last * step):
                faces.append(_Face(rough + shift, first * step, last * step))
        return faces

    def front(self, face: _Face) -> tuple[float, float]:
        """(coverage over the front height, coverage just below the worktop)."""
        on = np.abs(self.a - face.offset) <= self.tolerance
        seen = _cover(self.b[on], self.bins[on], face.start, face.stop, self.step, self.fronts.band_bins)
        return float(np.mean(seen.sum(axis=1) >= self.fronts.station_min_bins)), float(np.mean(seen[:, -2:].any(axis=1)))

    def below_tall(self, face: _Face) -> tuple[_Face | None, float]:
        """The stretch of a front below the worktop only (a flush oven tower or fridge ends it), and its share of tall stations."""
        on = np.abs(self.tall_a - face.offset) <= self.tolerance
        high = _cover(self.tall_b[on], self.tall_bins[on], face.start, face.stop, self.step, 4).sum(axis=1) >= 2
        runs = _runs(~high, 2)  # a bottle on the edge of the worktop is no tower
        if not runs:
            return None, 1.0
        first, last = max(runs, key=lambda run: run[1] - run[0])
        size = (face.stop - face.start) / len(high)
        if (last - first) * size < self.settings.min_length_m * self.upm - 0.5 * self.step:
            return None, float(high.mean())
        return _Face(face.offset, face.start + first * size, face.start + last * size), float(high[first:last].mean())

    def stations_behind(self, a: np.ndarray, b: np.ndarray, face: _Face, start: float, stop: float, near: float, far: float, step: float) -> np.ndarray:
        on = (a - face.offset >= near * self.upm) & (a - face.offset <= far * self.upm)
        return _cover(b[on], np.zeros(int(on.sum()), dtype=int), start, stop, step, 1)[:, 0]

    def labelled(self, face: _Face, start: float, stop: float, step: float) -> np.ndarray:
        return self.stations_behind(self.label_a, self.label_b, face, start, stop, -0.05, self.settings.label_reach_m, step)

    def topped(self, face: _Face, start: float, stop: float, step: float) -> np.ndarray:
        return self.stations_behind(self.top_a, self.top_b, face, start, stop, 0.02, 0.30, step)

    def back_face(self, face: _Face) -> tuple[float, float] | None:
        """(depth, coverage) of the scanned back of a free-standing leg."""
        low, high = self.settings.free_min_depth_m * self.upm, self.settings.free_max_depth_m * self.upm
        inner = self.a - face.offset
        best: tuple[float, float] | None = None
        for depth in np.arange(low, high + 1e-9, 0.01 * self.upm):
            on = np.abs(inner - depth) <= 1.5 * self.tolerance  # a back panel seen from the other room side is rougher
            share = float(np.mean(_cover(self.b[on], self.bins[on], face.start, face.stop, self.step, self.fronts.band_bins).sum(axis=1) >= max(2, self.fronts.band_bins // 2)))
            if best is None or share > best[1]:
                best = (float(depth), share)
        if best is None or best[1] < self.settings.min_back_face_share:
            return None
        on = (np.abs(inner - best[0]) <= 1.5 * self.tolerance) & (self.b >= face.start) & (self.b < face.stop)
        return float(np.clip(np.median(inner[on]), low, high)), best[1]

    def reach(self, face: _Face) -> float:
        """The end of a free-standing leg: its worktop goes on over a knee space while top and labels are measured."""
        limit = self.settings.max_length_m * self.upm
        if limit - face.stop < self.step:
            return face.stop
        both = self.topped(face, face.stop, limit, self.step) & self.labelled(face, face.stop, limit, self.step)
        runs = _runs(both, max(1, int(round(self.settings.gap_m * self.upm / self.step))))
        if not runs or runs[0][0] * self.step > self.settings.gap_m * self.upm:
            return face.stop
        return face.stop + runs[0][1] * (limit - face.stop) / len(both)


def _leg_from_face(
    legs: MutableSequence[Any], index: int, data: _Evidence, face: _Face, obstacles: Any, room: Any,
) -> tuple[tuple[Any, float, dict[str, Any]] | None, str]:
    """((new leg, new end of the known leg, support), "") or (None, the test that failed)."""
    settings, fronts, upm, view = data.settings, data.fronts, data.upm, data.view
    anchor = legs[index]
    coverage, top_coverage = data.front(face)
    if coverage < settings.min_coverage or top_coverage < settings.min_top_coverage:
        return None, "front"
    low_part, tall_share = data.below_tall(face)
    if low_part is None or tall_share >= settings.max_tall_share:
        return None, "tall"
    if low_part != face:
        face = low_part
        coverage, top_coverage = data.front(face)
        if coverage < settings.min_coverage or top_coverage < settings.min_top_coverage:
            return None, "front"
    wide = 0.05 * upm  # stations of the weaker evidence (sparse labels, objects on the worktop)
    label_share = float(data.labelled(face, face.start, face.stop, wide).mean())
    if label_share < settings.min_label_share:
        return None, "labels"
    top_share = float(data.topped(face, face.start, face.stop, wide).mean())
    if top_share < settings.min_worktop_share:
        return None, "worktop"

    # what is behind: a wall at cabinet depth, else the scanned back of a peninsula
    margin = 0.10 * (face.stop - face.start)
    b0, b1 = face.start + margin, face.stop - margin
    depth, back_share, backed = 0.0, 0.0, False
    if obstacles is not None and not obstacles.is_empty:
        strip = box(*view.bounds(face.offset + 0.05 * upm, face.offset + settings.max_depth_m * upm, b0, b1))
        hit = obstacles.intersection(strip)
        if not hit.is_empty and hit.area > 1e-9 * upm * upm:
            x0, y0, x1, y1 = hit.bounds
            a, b = view.ab(np.asarray(((x0, y0), (x1, y1))))
            back_share = float(np.ptp(b)) / (b1 - b0)
            depth = float(a.min()) - face.offset
            backed = back_share >= settings.min_wall_share
            if backed and depth < settings.min_depth_m * upm:
                return None, "wall too close"
            if not backed:
                return None, "broken wall behind"
    stop = face.stop
    if not backed:
        back = data.back_face(face)
        if back is None:
            return None, "nothing behind"
        depth, back_share = back
        stop = data.reach(face)

    u0, v0, u1, v1 = view.bounds(face.offset, face.offset + depth, -anchor.depth, stop)
    far_side = face.offset + depth
    if view.axis == 0:
        name = "u1" if view.out > 0 else "u0"
    else:
        name = "v1" if view.out > 0 else "v0"
    new = replace(anchor, u0=u0, u1=u1, v0=v0, v1=v1, back=name if backed else None)
    if new.horizontal == anchor.horizontal:
        return None, "shorter than deep"
    body = box(u0, v0, u1, v1)
    for number, other in enumerate(legs):
        if number == index or other.horizontal != new.horizontal:
            continue
        known = box(other.u0, other.v0, other.u1, other.v1)
        if body.intersection(known).area > settings.max_overlap * min(body.area, known.area):
            return None, "leg known"
    straight = box(*view.bounds(face.offset, far_side, max(face.start, 0.0), stop))
    if obstacles is not None and straight.intersection(obstacles).area > 0.10 * straight.area:
        return None, "blocked"
    if room is not None and straight.intersection(room).area < 0.80 * straight.area:
        return None, "outside the room"

    # the known leg reaches into the shared corner, unless something stands there
    end_a = max(0.0, far_side)
    if end_a > 0.0:
        added = box(*view.bounds(0.0, end_a, -anchor.depth, 0.0))
        if obstacles is not None and added.intersection(obstacles).area > 0.20 * added.area:
            return None, "corner blocked"
    end = view.end + view.out * end_a

    # joined to the known leg: the fronts meet, or a diagonal with line support of its own
    joined = ""
    if face.start <= settings.join_m * upm:
        back_along = -(data.a - face.offset)  # along the known front, away from the new leg
        near = np.abs(data.b) <= fronts.max_shift_m * upm + data.tolerance
        _, seen = _front_offset(back_along[near], data.b[near], data.bins[near], 0.0, 0.30 * upm, fronts, upm)
        if seen >= fronts.corner_min_join_share:
            joined = "fronts meet"
    if not joined:
        trial = list(legs)
        trial[index] = replace(anchor, **{name: end})
        trial.append(new)
        fresh = len(trial) - 1
        for corner in _inner_corners(trial, obstacles, room, fronts, upm):
            if {corner.h, corner.v} != {index, fresh}:
                continue
            measured, _ = _measure_corner(corner, data.band, data.low, data.high, fronts, upm)
            if measured is None:
                continue
            cut = measured.distance / float(measured.normal[0] if corner.h == fresh else measured.normal[1])
            # it reaches this front (the hidden-corner test already rules out a front that goes on behind it)
            if face.start - settings.gap_m * upm <= cut <= face.stop - 0.5 * settings.min_length_m * upm:
                joined = "diagonal"
                break
    if not joined:
        return None, "not joined"
    return (new, end, {
        "length_m": round(max(u1 - u0, v1 - v0) / upm, 3), "straight_front_m": round((face.stop - face.start) / upm, 3),
        "depth_m": round(depth / upm, 3), "behind": "wall" if backed else "back face", "behind_share": round(back_share, 3), "joined": joined,
        "front_share": round(coverage, 3), "top_share": round(top_share, 3), "label_share": round(label_share, 3),
    }), ""


__all__ = ["FrontLegSettings", "recover_front_legs", "refit_free_legs"]
