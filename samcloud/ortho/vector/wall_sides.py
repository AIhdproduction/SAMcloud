"""Which parallel measured faces are the two sides of one wall.

A wall body is the band between two faces that look away from each other: the room ceiling and the floor end at both
faces, and both continue on the outer sides.  A surface standing inside a room (cabinet front, curtain, shower screen,
tiles or a pre-wall, a mirror's double surface, a second registration of the same face) fails at least one part of
this test: the ceiling runs on behind it, no room follows on one outer side, or another face stands between.

:func:`two_faced_walls` finds these walls on the faces as extracted.  The later Flucht snap joins pieces up to 0.25 m
apart without floor between them, which both faces of a wall are, so the wall stage alone sees one face and picks a
side and a thickness for it.  :func:`apply_measured_bands` puts the runs built for such a wall onto its two faces: the
measurement wins over that side choice and over a default, learned or averaged thickness.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .opening_bands import ASSUMED_THICKNESS, MEASURED_FACES, _move_run, host_run, run_band
from .raster_grid import RasterGrid

_PARALLEL = 0.9986  # cos 3 degrees
_NO_WALL = frozenset({"niche", "closure"})  # never the wall itself; a closure may continue its band


@dataclass(frozen=True, slots=True)
class WallSideSettings:
    """Metres and shares of the 5 cm samples along the overlap of two faces."""

    min_thickness_m: float = 0.06
    max_thickness_m: float = 0.30
    min_overlap_m: float = 0.40
    sample_m: float = 0.05
    outer_gap_m: float = 0.025  # past the noise of the face itself
    outer_reach_m: float = 0.11
    plane_band_m: float = 0.03
    room_share: float = 0.40  # floor or ceiling right in front of each face
    body_ceiling_share: float = 0.30  # the ceiling does not run through a wall
    body_floor_share: float = 0.60  # more floor in the band only where a doorway fills the overlap (ceiling rule below)
    solid_floor_share: float = 0.25  # a face hidden at the top still needs a band without floor
    top_height_share: float = 0.80  # of the storey: a door leaf, mirror, screen or pre-wall ends lower
    top_share: float = 0.40
    refine_m: float = 0.04  # a face is measured again on the overlap: a long fitted face may average two layers
    refine_bin_m: float = 0.005
    # A face without an extracted partner (the extraction keeps one of two planes closer than its suppression distance, or
    # sees no empty side): the second plane is searched where both cover most of the face and are hollow between.
    partner_min_length_m: float = 0.60
    partner_min_m: float = 0.10
    partner_share: float = 0.60
    partner_rise_m: float = 0.30
    partner_valley_share: float = 0.30


@dataclass(frozen=True, slots=True)
class WallSideEvidence:
    """Rasters of the storey on the plan grid (``high``: wall-class points near the ceiling)."""

    grid: RasterGrid
    units_per_metre: float
    ceiling: np.ndarray
    floor: np.ndarray
    high: np.ndarray
    settings: WallSideSettings = WallSideSettings()


@dataclass(frozen=True, slots=True)
class MeasuredWall:
    """Two measured faces: ``start``-``end`` is their overlap on the first, the other lies ``thickness`` along ``towards``."""

    first: int
    second: int
    start: tuple[float, float]
    end: tuple[float, float]
    towards: tuple[float, float]  # unit vector from the first face into the body
    thickness: float  # plan units


def wall_side_evidence(vertical: Any, grid: RasterGrid, units_per_metre: float, bins: int, settings: WallSideSettings = WallSideSettings()) -> WallSideEvidence | None:
    """Evidence from the streamed storey rasters; None when the cloud gave no ceiling, floor or wall heights."""
    ceiling, floor, wall_bits = (getattr(vertical, name, None) for name in ("ceiling_count", "floor_points", "wall_bits"))
    shape = (grid.height, grid.width)
    if any(item is None or np.shape(item) != shape for item in (ceiling, floor, wall_bits)):
        return None
    first_high_bin = min(int(settings.top_height_share * bins), bins - 1)
    high = (np.asarray(wall_bits, dtype=np.uint32) >> np.uint32(first_high_bin)) != 0
    return WallSideEvidence(grid, max(float(units_per_metre), 1e-9), np.asarray(ceiling) > 0, np.asarray(floor, dtype=bool), high, settings)


def two_faced_walls(lines: Sequence[Any], evidence: WallSideEvidence | None, points: np.ndarray | None = None) -> list[MeasuredWall]:
    """Pairs of ``lines`` (two end points each, plan units) proven to look away from each other.

    With ``points`` (structural points x, y, height) each face is placed on the strongest plane of the overlap, and a
    face without a partner looks for the second plane of its wall (``second`` is then -1).
    """
    if evidence is None:
        return []
    settings, upm = evidence.settings, evidence.units_per_metre
    segments = [np.asarray(line, dtype=float)[[0, -1], :2] for line in lines]
    lengths = [float(np.linalg.norm(segment[1] - segment[0])) for segment in segments]
    found: list[MeasuredWall] = []
    for first, segment in enumerate(segments):
        if lengths[first] <= 1e-9:
            continue
        origin = segment[0]
        axis = (segment[1] - origin) / lengths[first]
        normal = np.asarray((-axis[1], axis[0]))
        placed: list[tuple[int, float, float, float]] = []
        for other, line in enumerate(segments):
            if other == first or lengths[other] <= 1e-9 or abs(float((line[1] - line[0]) @ axis)) / lengths[other] < _PARALLEL:
                continue
            along = np.sort((line - origin) @ axis)
            placed.append((other, float(np.mean((line - origin) @ normal)), max(float(along[0]), 0.0), min(float(along[1]), lengths[first])))
        for second, offset, start, end in placed:
            thickness = abs(offset)
            if second < first or end - start < settings.min_overlap_m * upm:
                continue
            if not settings.min_thickness_m * upm <= thickness <= settings.max_thickness_m * upm:
                continue
            margin = min(settings.plane_band_m * upm, thickness / 3.0)
            between = sum(
                max(0.0, min(high, end) - max(low, start))
                for other, value, low, high in placed
                if other != second and value * offset > 0.0 and margin < abs(value) < thickness - margin
            )
            if between >= 0.5 * (end - start):
                continue
            towards = normal if offset > 0.0 else -normal
            if _looks_apart(evidence, origin, axis, towards, start, end, thickness):
                near, far = 0.0, thickness
                if points is not None and len(points):
                    near, far = (_strongest_plane(np.asarray(points), origin, axis, towards, (start, end), value, evidence) for value in (near, far))
                    if not settings.min_thickness_m * upm <= far - near <= settings.max_thickness_m * upm:
                        near, far = 0.0, thickness
                ends = [tuple(float(value) for value in origin + axis * along + towards * near) for along in (start, end)]
                thickness = far - near
                found.append(MeasuredWall(first, second, ends[0], ends[1], (float(towards[0]), float(towards[1])), thickness))
    if points is not None and len(points) and np.shape(points)[1] >= 3:
        paired = {index for wall in found for index in (wall.first, wall.second)}
        for first, length in enumerate(lengths):
            if first in paired or length < settings.partner_min_length_m * upm:
                continue
            partner = _partner_wall(first, segments, np.asarray(points, dtype=float), evidence)
            if partner is not None:
                found.append(partner)
    return found


def _partner_wall(first: int, segments: list[np.ndarray], points: np.ndarray, evidence: WallSideEvidence) -> MeasuredWall | None:
    """The wall of a face whose second plane was not extracted.

    It passes the test of two extracted faces and more: no doorway excuses floor or ceiling between the planes, free
    space of a room (between two drift layers of one surface) never becomes a wall body.
    """
    settings, upm = evidence.settings, evidence.units_per_metre
    origin = segments[first][0]
    length = float(np.linalg.norm(segments[first][1] - origin))
    axis = (segments[first][1] - origin) / length
    normal = np.asarray((-axis[1], axis[0]))
    step, band = 0.01 * upm, settings.plane_band_m * upm
    reach = settings.max_thickness_m * upm + band
    relative = points[:, :2] - origin
    along, offset = relative @ axis, relative @ normal
    near = (along >= 0.0) & (along <= length) & (np.abs(offset) <= reach)
    if not near.any():
        return None
    along, offset, height = along[near], offset[near], points[near, 2]
    cells = max(int(round(length / (settings.sample_m * upm))), 1)

    def is_plane(value: float) -> bool:
        on_plane = np.abs(offset - value) <= band
        if not on_plane.any() or float(np.subtract(*np.percentile(height[on_plane], (90.0, 10.0)))) < settings.partner_rise_m * upm:
            return False
        covered = np.unique(np.clip((along[on_plane] / length * cells).astype(np.int64), 0, cells - 1))
        return len(covered) >= settings.partner_share * cells

    if not is_plane(0.0):
        return None
    count, edges = np.histogram(offset, bins=np.arange(-reach, reach + step, step))
    count = np.convolve(count, np.ones(3), mode="same")
    middles = (edges[:-1] + edges[1:]) / 2.0
    best: tuple[float, np.ndarray, float] | None = None
    for sign in (1.0, -1.0):
        depth = middles * sign
        zone = np.flatnonzero((depth >= settings.partner_min_m * upm) & (depth <= settings.max_thickness_m * upm))
        if not len(zone):
            continue
        peak = int(zone[int(np.argmax(count[zone]))])
        thickness = float(depth[peak])
        if count[peak] <= 0 or (best is not None and count[peak] <= best[0]) or not is_plane(float(middles[peak])):
            continue
        hollow = (depth > band) & (depth < thickness - band)
        if not hollow.any() or float(count[hollow].min()) > settings.partner_valley_share * float(count[peak]):
            continue
        towards = sign * normal
        between = 0.0
        for index, line in enumerate(segments):
            span = float(np.linalg.norm(line[1] - line[0]))
            if index == first or span <= 1e-9 or abs(float((line[1] - line[0]) @ axis)) / span < _PARALLEL:
                continue
            if band < float(np.mean((line - origin) @ towards)) < thickness - band:
                low, high = np.sort((line - origin) @ axis)
                between += max(0.0, min(float(high), length) - max(float(low), 0.0))
        if between < 0.5 * length and _looks_apart(evidence, origin, axis, towards, 0.0, length, thickness):
            best = (float(count[peak]), towards, thickness)
    if best is None:
        return None
    _, towards, thickness = best
    near_face, far_face = (_strongest_plane(points, origin, axis, towards, (0.0, length), value, evidence) for value in (0.0, thickness))
    if not settings.min_thickness_m * upm <= far_face - near_face <= settings.max_thickness_m * upm:
        return None
    body = (near_face + 0.3 * (far_face - near_face), near_face + 0.7 * (far_face - near_face))
    room = evidence.ceiling | evidence.floor
    if _share_along(room, evidence.grid, origin, axis, towards, (0.0, length), body, settings.sample_m * upm) > settings.solid_floor_share:
        return None
    ends = [tuple(float(value) for value in origin + axis * value_along + towards * near_face) for value_along in (0.0, length)]
    return MeasuredWall(first, -1, ends[0], ends[1], (float(towards[0]), float(towards[1])), far_face - near_face)


def _looks_apart(evidence: WallSideEvidence, origin: np.ndarray, axis: np.ndarray, towards: np.ndarray, start: float, end: float, thickness: float) -> bool:
    settings, upm = evidence.settings, evidence.units_per_metre

    def share(mask: np.ndarray, low: float, high: float) -> float:
        return _share_along(mask, evidence.grid, origin, axis, towards, (start, end), (low, high), settings.sample_m * upm)

    body = (0.3 * thickness, 0.7 * thickness)
    if share(evidence.ceiling, *body) > settings.body_ceiling_share:
        return False
    body_floor = share(evidence.floor, *body)
    gap, reach = settings.outer_gap_m * upm, settings.outer_reach_m * upm
    room = evidence.ceiling | evidence.floor
    outer = ((-reach, -gap), (thickness + gap, thickness + reach))
    if any(share(room, *strip) < settings.room_share for strip in outer):
        return False
    band = settings.plane_band_m * upm
    if all(share(evidence.high, value - band, value + band) >= settings.top_share for value in (0.0, thickness)):
        # Where a doorway fills the overlap its floor runs through; then the ceiling itself must end at both faces
        # (an open leaf or a screen ends below it, behind a curtain it runs on).
        return body_floor <= settings.body_floor_share or all(share(evidence.ceiling, *strip) >= settings.room_share for strip in outer)
    # A face whose top nobody saw: the floor itself must end at both faces.
    return body_floor <= settings.solid_floor_share and all(share(evidence.floor, *strip) >= settings.room_share for strip in outer)


def _strongest_plane(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, across: np.ndarray, span: tuple[float, float], guess: float, evidence: WallSideEvidence) -> float:
    """Offset of the plane with most points near ``guess`` that covers the overlap; ``guess`` without one."""
    settings, upm = evidence.settings, evidence.units_per_metre
    relative = points[:, :2] - origin
    along, offset = relative @ axis, relative @ across
    step, reach = settings.refine_bin_m * upm, settings.refine_m * upm
    near = (along >= span[0]) & (along <= span[1]) & (np.abs(offset - guess) <= reach)
    if not near.any():
        return guess
    along, offset = along[near], offset[near]
    count, edges = np.histogram(offset, bins=np.arange(guess - reach, guess + reach + step, step))
    peak = int(np.argmax(np.convolve(count, np.ones(3), mode="same")))
    on_plane = np.abs(offset - (edges[peak] + edges[peak + 1]) / 2.0) <= 3.0 * step
    bins = max(int(round((span[1] - span[0]) / (settings.sample_m * upm))), 1)
    covered = np.unique(np.clip(((along[on_plane] - span[0]) / (span[1] - span[0]) * bins).astype(np.int64), 0, bins - 1))
    if len(covered) < settings.top_share * bins:
        return guess
    return float(np.median(offset[on_plane]))


def _share_along(mask: np.ndarray, grid: RasterGrid, origin: np.ndarray, axis: np.ndarray, across: np.ndarray, span: tuple[float, float], strip: tuple[float, float], bin_size: float) -> float:
    """Share of bins along ``span`` with at least two hit samples in ``strip``."""
    bins = max(int(round((span[1] - span[0]) / bin_size)), 1)
    per_bin = max(2, int(np.ceil(bin_size / grid.pixel_size)))
    along = span[0] + (np.arange(bins * per_bin) + 0.5) * (span[1] - span[0]) / (bins * per_bin)
    offsets = np.linspace(strip[0], strip[1], max(2, int(np.ceil((strip[1] - strip[0]) / grid.pixel_size)) + 1))
    points = origin + along[:, None, None] * axis + offsets[None, :, None] * across
    rows = np.floor((grid.max_y - points[..., 1]) / grid.pixel_size).astype(np.int64)
    columns = np.floor((points[..., 0] - grid.min_x) / grid.pixel_size).astype(np.int64)
    inside = (rows >= 0) & (rows < grid.height) & (columns >= 0) & (columns < grid.width)
    hits = np.zeros(rows.shape, dtype=bool)
    hits[inside] = mask[rows[inside], columns[inside]]
    return float(np.mean(hits.reshape(bins, -1).sum(axis=1) >= 2))


def doorway_walls(runs: list[Any], faces: list[Any], openings: Sequence[Any], points: np.ndarray | None, storey_height: float, evidence: WallSideEvidence | None, upm: float, settings: Any) -> list[tuple[int, MeasuredWall]]:
    """(host run, wall) of the doors whose wall shows both faces above the head only.

    A doorway leaves no face to extract and its floor runs through the band.  Above the head the wall reaches the
    ceiling: two vertical planes over the span with the valley of a body between them, and the ceiling of a room ends
    at each of them.  Unlike the lintel band the planes may lie anywhere within a wall thickness of the detected line.
    """
    if evidence is None or points is None or not len(points) or storey_height <= 0.0:
        return []
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        return []
    face_of = {face.index: face for face in faces}
    limit, tolerance = settings.max_thickness_m * upm, settings.lintel_band_tolerance_m * upm
    rules = evidence.settings
    found: list[tuple[int, MeasuredWall]] = []
    for opening in openings:
        head = getattr(opening, "head_m", None)
        if str(getattr(opening, "kind", "")).upper() != "DOOR" or head is None:
            continue
        bottom = (float(head) + settings.lintel_band_clearance_m) * upm
        if storey_height - bottom < settings.lintel_band_min_height_m * upm:
            continue
        host = host_run(opening, runs, face_of, upm)
        if host is None or host.evidence not in ASSUMED_THICKNESS:
            continue
        face = face_of[host.face]
        width = float(opening.width)
        origin = np.asarray(opening.center, dtype=float)[:2] - face.direction * width / 2.0
        relative = points[:, :2] - origin
        along, across = relative @ face.direction, relative @ face.solid_normal
        above = (points[:, 2] >= bottom) & (along >= 0.0) & (along <= width) & (np.abs(across) <= limit)
        planes = _lintel_planes(along[above], across[above], points[above, 2], width, upm, settings)
        if planes is None:
            continue
        low, high = planes
        strips = (
            (low - rules.outer_reach_m * upm, low - rules.outer_gap_m * upm),
            (low + 0.3 * (high - low), low + 0.7 * (high - low)),
            (high + rules.outer_gap_m * upm, high + rules.outer_reach_m * upm),
        )
        first, body, second = (
            _share_along(evidence.ceiling, evidence.grid, origin, face.direction, face.solid_normal, (0.0, width), strip, rules.sample_m * upm)
            for strip in strips
        )
        if body > rules.body_ceiling_share or min(first, second) < rules.room_share:
            continue
        centre = float((face.point(0.0, host.thickness / 2.0) - origin) @ face.solid_normal)
        if abs(centre - (low + high) / 2.0) <= tolerance and abs(host.thickness - (high - low)) <= tolerance:
            continue  # nothing to correct beyond what the lintel band refines
        ends = [tuple(float(value) for value in origin + face.direction * value_along + face.solid_normal * low) for value_along in (0.0, width)]
        index = next(number for number, run in enumerate(runs) if run is host)
        found.append((index, MeasuredWall(-1, -1, ends[0], ends[1], tuple(float(value) for value in face.solid_normal), high - low)))
    return found


def _lintel_planes(along: np.ndarray, across: np.ndarray, height: np.ndarray, width: float, upm: float, settings: Any) -> tuple[float, float] | None:
    """Offsets of the strongest plane and of the strongest one a wall thickness beside it, hollow between them."""
    step, cell = 0.01 * upm, 0.05 * upm
    cells = max(1, int(np.ceil(width / cell)))
    if len(across) < 2 * cells:
        return None
    limit, tolerance = settings.max_thickness_m * upm, settings.lintel_band_tolerance_m * upm
    count, edges = np.histogram(across, bins=np.arange(-limit, limit + step, step))
    count = np.convolve(count, np.ones(3), mode="same")
    middles = (edges[:-1] + edges[1:]) / 2.0

    def is_plane(index: int) -> bool:
        on_plane = np.abs(across - middles[index]) <= tolerance
        if not on_plane.any() or float(np.subtract(*np.percentile(height[on_plane], (90.0, 10.0)))) < settings.lintel_band_min_plane_m * upm:
            return False
        return len(np.unique(np.floor(along[on_plane] / cell))) >= settings.lintel_band_min_share * cells

    first = int(np.argmax(count))
    if not is_plane(first):
        return None
    distance = np.abs(middles - middles[first])
    beside = np.flatnonzero((distance >= settings.lintel_band_min_m * upm) & (distance <= WallSideSettings().max_thickness_m * upm))
    if not len(beside):
        return None
    second = int(beside[int(np.argmax(count[beside]))])
    if count[second] <= 0 or not is_plane(second):
        return None
    low, high = sorted((middles[first], middles[second]))
    between = (middles > low + tolerance) & (middles < high - tolerance)
    if not between.any() or float(count[between].min()) > settings.lintel_band_valley_share * float(count[second]):
        return None
    sides = [float(np.median(across[np.abs(across - value) <= 1.5 * step])) for value in (low, high)]
    return sides[0], sides[1]


def apply_measured_bands(
    runs: list[Any], faces: list[Any], walls: Sequence[MeasuredWall], upm: float, settings: Any, hosts: Sequence[tuple[int, MeasuredWall]] = (),
    openings: Sequence[Any] = (),
) -> dict[str, int]:
    """The wall runs built for a two-faced wall take both measured faces; crossing runs keep ending on it.

    A run was built for the wall when it runs along the overlap and either shares half of the body or stands with one
    face on a measured face (then possibly on the wrong side of it).  Pieces with the very same band continue that wall.
    ``hosts`` name further (run index, wall) claims.
    """
    stats = {"walls": len(walls) + len(hosts), "runs_changed": 0}
    if not walls and not hosts:
        return stats
    face_of = {face.index: face for face in faces}
    target: dict[int, MeasuredWall] = {}
    claimed: dict[int, float] = {}
    tolerance = settings.opening_flush_m * upm
    for wall in walls:
        origin, normal = np.asarray(wall.start), np.asarray(wall.towards)
        span = np.asarray(wall.end) - origin
        length = float(np.linalg.norm(span))
        if length <= 1e-9:
            continue
        axis = span / length
        for index, run in enumerate(runs):
            face = face_of[run.face]
            if run.thickness <= 0.0 or run.evidence in _NO_WALL or abs(float(face.direction @ axis)) < _PARALLEL:
                continue
            low, high = sorted(float((face.point(value) - origin) @ axis) for value in (run.start, run.end))
            overlap = min(high, length) - max(low, 0.0)
            if overlap < 0.10 * upm or overlap <= claimed.get(index, 0.0):
                continue
            middle = origin + axis * (max(low, 0.0) + min(high, length)) / 2.0
            near, far = run_band(run, face, middle, normal)
            shared = min(far, wall.thickness) - max(near, 0.0)
            on_face = min(abs(one - two) for one in (near, far) for two in (0.0, wall.thickness)) <= settings.measured_band_face_m * upm
            if not on_face and shared < 0.5 * min(run.thickness, wall.thickness):
                continue
            needed = 0.5 * (high - low)
            if run.evidence not in MEASURED_FACES:
                # A piece beside a doorway carries the wall that is seen from both sides above the door: same body, same face.
                continues = on_face and shared > 0.0 and run.evidence in ASSUMED_THICKNESS
                needed = 0.0 if continues else min(needed, settings.measured_band_overlap_m * upm)
            if overlap < needed:
                continue
            claimed[index], target[index] = overlap, wall
    for index, wall in hosts:  # doors of walls seen above the head only (doorway_walls)
        if index not in target:
            claimed[index], target[index] = 0.0, wall
    queue = sorted(target, key=lambda index: -claimed[index])
    for index in queue:
        run = runs[index]
        face = face_of[run.face]
        for number, other in enumerate(runs):
            if number in target or other.thickness <= 0.0 or other.evidence == "niche" or other.evidence in MEASURED_FACES:
                continue
            other_face = face_of[other.face]
            if abs(float(other_face.direction @ face.direction)) < _PARALLEL:
                continue
            low, high = sorted(float((other_face.point(value) - face.start) @ face.direction) for value in (other.start, other.end))
            if max(low - run.end, run.start - high, 0.0) > settings.flush_chain_gap_m * upm:
                continue
            middle = other_face.point((other.start + other.end) / 2.0)
            own, theirs = run_band(run, face, middle, face.solid_normal), run_band(other, other_face, middle, face.solid_normal)
            if max(abs(own[0] - theirs[0]), abs(own[1] - theirs[1])) <= tolerance:
                target[number] = target[index]
                queue.append(number)

    target = _snap_to_jambs(target, runs, face_of, openings, upm, settings)
    moves: list[tuple[int, float, float]] = []
    for index, wall in target.items():
        run, face = runs[index], face_of[runs[index].face]
        towards = np.asarray(wall.towards)
        middle = face.point((run.start + run.end) / 2.0)
        first = float((np.asarray(wall.start) - middle) @ towards)
        sign = 1.0 if float(face.solid_normal @ towards) > 0.0 else -1.0
        near = min(sign * first, sign * (first + wall.thickness))
        far = near + wall.thickness
        # a face that stands on its measurement stays (a millimetre decides joins and shifts of openings downstream)
        near = 0.0 if abs(near) <= tolerance else near
        far = run.thickness if abs(far - run.thickness) <= tolerance else far
        if near != 0.0 or far != run.thickness:
            moves.append((index, near, far - near))
    joints = _crossing_ends(runs, face_of, moves, openings, upm, settings)
    for index, near, thickness in moves:
        _move_run(runs[index], faces, face_of, near, thickness)
        if runs[index].evidence != "door-host":  # a built piece stays "door-host": no wall line for the 3D refinement
            runs[index].evidence = "face"
        stats["runs_changed"] += 1
    for index, at_start, shift in joints:
        run = runs[index]
        if run.detected is None:
            run.detected = (face_of[run.face], run.thickness)
        if at_start:
            run.start -= shift
        else:
            run.end += shift
    return stats


def _snap_to_jambs(target: dict[int, MeasuredWall], runs: list[Any], face_of: dict[int, Any], openings: Sequence[Any], upm: float, settings: Any) -> dict[int, MeasuredWall]:
    """The walls with a face on the jamb of a crossing opening where both lie within ``lintel_corner_snap_m``.

    That is one corner measured twice.  Where the wall was seen mostly above its own doorway the jamb wins: the opening
    keeps its clear width and no step is left at it.  Faces seen along the wall itself always win over a jamb.
    """
    if not target or not openings:
        return target
    number_of = {id(run): number for number, run in enumerate(runs)}
    limit, corner = settings.max_thickness_m * upm, settings.lintel_corner_snap_m * upm
    planes: dict[int, list[float]] = {}
    for opening in openings:
        host = host_run(opening, runs, face_of, upm)
        if host is None or number_of[id(host)] not in target:
            continue
        wall = target[number_of[id(host)]]
        origin, towards = np.asarray(wall.start), np.asarray(wall.towards)
        centre, half = np.asarray(opening.center, dtype=float)[:2], float(opening.width) / 2.0
        axis = np.asarray((towards[1], -towards[0]))
        low, high = sorted(float((np.asarray(point) - centre) @ axis) for point in (wall.start, wall.end))
        if min(high, half) - max(low, -half) < 0.5 * (high - low):
            continue
        for other in openings:
            direction = np.asarray((np.cos(float(other.axis_angle)), np.sin(float(other.axis_angle))))
            if other is opening or abs(float(direction @ towards)) < _PARALLEL:
                continue
            for sign in (-1.0, 1.0):
                jamb = np.asarray(other.center, dtype=float)[:2] + sign * direction * float(other.width) / 2.0
                if abs(abs(float((jamb - centre) @ axis)) - half) <= limit:
                    planes.setdefault(id(wall), []).append(float((jamb - origin) @ towards))
    snapped: dict[int, MeasuredWall] = {}
    for wall in {id(item): item for item in target.values()}.values():
        faces = []
        for value in (0.0, wall.thickness):
            near = [plane for plane in planes.get(id(wall), ()) if abs(plane - value) <= corner]
            faces.append(min(near, key=lambda plane: abs(plane - value)) if near else value)
        shift = np.asarray(wall.towards) * faces[0]
        start, end = (tuple(float(value) for value in np.asarray(point) + shift) for point in (wall.start, wall.end))
        snapped[id(wall)] = replace(wall, start=start, end=end, thickness=faces[1] - faces[0])
    return {index: snapped[id(wall)] for index, wall in target.items()}


def _crossing_ends(runs: list[Any], face_of: dict[int, Any], moves: list[tuple[int, float, float]], openings: Sequence[Any], upm: float, settings: Any) -> list[tuple[int, bool, float]]:
    """How far each end that stands in a band follows when the band moves (positive: longer).

    The band reaches across the openings it hosts (their jamb is such an end); a stub that the wall bodies trim at the
    band (``min_stub_m``) stands in it as well.  An end at the far face ends on the measured one: one corner, no step.
    """
    reach = settings.joint_tolerance_m * upm
    stub, corner = max(settings.min_stub_m * upm, reach), settings.lintel_corner_snap_m * upm
    spans: dict[int, list[float]] = {}  # along each moved run: the openings it hosts
    moved = {id(runs[index]): index for index, _, _ in moves}
    for opening in openings if moves else ():
        host = host_run(opening, runs, face_of, upm)
        if host is not None and id(host) in moved:
            face, half = face_of[host.face], float(opening.width) / 2.0
            middle = float((np.asarray(opening.center, dtype=float)[:2] - face.start) @ face.direction)
            spans.setdefault(moved[id(host)], []).extend((middle - half, middle + half))
    inset = 0.01 * upm
    changes: dict[tuple[int, bool], float] = {}
    for index, near, thickness in moves:
        run, face = runs[index], face_of[runs[index].face]
        band_start = min([run.start - run.extend_start, *spans.get(index, ())])
        band_end = max([run.end + run.extend_end, *spans.get(index, ())])
        for number, other in enumerate(runs):
            other_face = face_of[other.face]
            across = float(other_face.direction @ face.solid_normal)
            if other.thickness <= 0.0 or abs(across) < _PARALLEL:
                continue
            for at_start in (True, False):
                position = other.start - other.extend_start if at_start else other.end + other.extend_end
                end = other_face.point(position, other.thickness / 2.0)
                depth, along = float((end - face.start) @ face.solid_normal), float((end - face.start) @ face.direction)
                outward = -across if at_start else across
                sign = 1.0 if outward > 0.0 else -1.0
                entering = depth if sign > 0.0 else run.thickness - depth  # 0 on the face it comes from
                if not -reach <= entering <= run.thickness + stub:
                    continue
                if not band_start - other.thickness <= along <= band_end + other.thickness:
                    continue
                low, high = near, near + thickness
                entered, passed = (low + inset, high) if sign > 0.0 else (high - inset, low)
                if (entered - depth) * sign > 0.0:  # the body went away: follow it
                    shift = (entered - depth) / outward
                elif (depth - passed) * sign > 0.0:  # the body came towards the end: nothing sticks out of it
                    shift = max((passed - depth) / outward, inset - (other.end - other.start))
                elif 0.0 < (passed - depth) * sign <= corner:  # two measurements of one outer corner
                    shift = (passed - depth) / outward
                else:
                    continue
                if abs(shift) > abs(changes.get((number, at_start), 0.0)):
                    changes[number, at_start] = shift
    return [(number, at_start, shift) for (number, at_start), shift in changes.items()]
