"""The wall band an opening interrupts: one pair of faces for lintel, jambs, frame and cut.

Openings are detected ON a measured wall face and carry their own thickness estimate.  Neither is geometry: an opening
always takes both faces of its host wall.  This module finds that host and keeps its band right:

* :func:`host_index` is the host rule of the plan model and of the wall stage (one rule, one band);
* :func:`apply_lintel_bands`: above a door head nothing occludes the wall, so the scan sees both of its faces there.  Where
  the host only has an assumed thickness (exterior default, learned, filled gap) the measured band wins for the wall and
  its lintel.  A host built between the jambs only (no wall run there) is moved onto its two measured faces;
* :func:`flush_interrupted_runs`: the pieces on both sides of an opening share the band of the better measured one; a step
  stays only when both pieces carry measured faces and differ clearly.

``runs`` / ``faces`` are the ``WallRun`` / ``_Face`` objects of :mod:`.double_walls`, ``settings`` its settings; lengths are
plan units (``upm`` per metre).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon

ASSUMED_THICKNESS = frozenset({"exterior", "exterior-reveal", "learned", "door-fill", "door-host", "room-edge"})
MEASURED_FACES = frozenset({"face", "depth"})
OTHER_ELEMENTS = frozenset({"niche", "closure", "partition"})  # pieces that are no continuation of a wall: their own band stays
_STRENGTH = {"face": 5, "depth": 4, "lintel": 4, "face-collinear": 3, "face-beside": 3, "room": 2}
_PARALLEL_HOST = 0.9  # |cos| between opening axis and host direction


def _axis(opening: Any) -> np.ndarray:
    angle = float(getattr(opening, "axis_angle", 0.0))
    return np.asarray((math.cos(angle), math.sin(angle)))


def _centre(opening: Any) -> np.ndarray:
    return np.asarray(getattr(opening, "center"), dtype=float)[:2]


def _kind(opening: Any) -> str:
    return str(getattr(opening, "kind", "")).upper()


def host_index(centre: np.ndarray, axis: np.ndarray, width: float, thickness: float, lines: Sequence[Any], thicknesses: Sequence[float], upm: float) -> int | None:
    """Nearest centreline along the opening axis whose body holds the opening centre; None without one.

    The centre may lie anywhere in the body (openings are detected on a face); along the wall the piece may end at the jamb.
    """
    best: tuple[float, int] | None = None
    margin = 0.05 * upm
    for index, (points, own) in enumerate(zip(lines, thicknesses, strict=True)):
        a, b = np.asarray(points[0], dtype=float), np.asarray(points[1], dtype=float)
        length = float(np.linalg.norm(b - a))
        if length <= 1e-9:
            continue
        direction = (b - a) / length
        if abs(float(np.dot(direction, axis))) < _PARALLEL_HOST:
            continue
        half = max(float(own), thickness) / 2.0
        if abs(float(np.dot(centre - a, np.asarray((-direction[1], direction[0]))))) > half + margin:
            continue
        distance = float(LineString(points).distance(Point(float(centre[0]), float(centre[1]))))
        if distance <= width / 2.0 + half + margin and (best is None or distance < best[0]):
            best = (distance, index)
    return None if best is None else best[1]


def host_run(opening: Any, runs: list[Any], face_of: dict[int, Any], upm: float) -> Any | None:
    """The wall run that hosts ``opening`` (the piece the plan model names as host)."""
    solid = [run for run in runs if run.thickness > 0.0]
    lines, widths = [], []
    for run in solid:
        # the band as detected: pieces often tie (both end at a jamb), a measured band must not flip the host
        face, thickness = run.detected if run.detected is not None else (face_of[run.face], run.thickness)
        lines.append([tuple(face.point(value, thickness / 2.0)) for value in (run.start - run.extend_start, run.end + run.extend_end)])
        widths.append(thickness)
    index = host_index(
        _centre(opening), _axis(opening), float(getattr(opening, "width")), float(getattr(opening, "wall_thickness", 0.0)),
        lines, widths, upm,
    )
    return None if index is None else solid[index]


def run_band(run: Any, face: Any, origin: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Offsets along ``normal`` (from ``origin``) of both faces of a run, beside ``origin``."""
    along = float(np.dot(origin - face.start, face.direction))
    first = float(np.dot(face.point(along) - origin, normal))
    second = float(np.dot(face.point(along, run.thickness) - origin, normal))
    return min(first, second), max(first, second)


def _move_run(run: Any, faces: list[Any], face_of: dict[int, Any], shift: float, thickness: float) -> None:
    """Give one run a new band (near face moved by ``shift`` along its solid normal) without moving other runs of its face."""
    if run.detected is None:
        run.detected = (face_of[run.face], run.thickness)
    if abs(shift) > 1e-9:
        face = face_of[run.face]
        offset = face.solid_normal * shift
        moved = replace(face, index=len(faces), start=face.start + offset, end=face.end + offset)
        faces.append(moved)
        face_of[moved.index] = moved
        run.face = moved.index
    run.thickness = float(thickness)


def passage_reveals(openings: Sequence[Any], slabs: Sequence[tuple[float, float]], hosted: Sequence[bool]) -> list[Polygon]:
    """The floor of every hosted passage between both faces of its wall (``slabs``: the faces across each opening).

    It always holds the line the passage was detected on, which lies beside a wall that moved onto its measured faces.
    """
    reveals = []
    for opening, (low, high), has_host in zip(openings, slabs, hosted, strict=True):
        if not has_host or _kind(opening) != "PASSAGE" or high <= low:
            continue
        axis, centre = _axis(opening), _centre(opening)
        normal, half = np.asarray((-axis[1], axis[0])), float(getattr(opening, "width")) / 2.0
        low, high = min(low, 0.0), max(high, 0.0)
        reveals.append(Polygon([tuple(centre + axis * a + normal * b) for a, b in ((-half, low), (half, low), (half, high), (-half, high))]))
    return reveals


def measure_two_faces(across: np.ndarray, along: np.ndarray, height: np.ndarray, length: float, upm: float, settings: Any) -> tuple[float, float] | None:
    """(near, far) offsets of the two faces of a wall seen in one stretch, None unless both are clear.

    ``across`` counts from a known face (positive into the wall).  The near face lies within the tolerance of it, the far face
    is the strongest plane behind it.  Both cover ``lintel_band_min_share`` of the stretch, both rise vertically over
    ``lintel_band_min_plane_m`` (the underside of a lintel and the top of a casing lie flat) and the wall between them is
    hollow (drift layers of one surface have no valley between them).
    """
    tolerance, low, high = settings.lintel_band_tolerance_m * upm, settings.lintel_band_min_m * upm, settings.max_thickness_m * upm
    step, cell = 0.01 * upm, 0.05 * upm
    cells = max(1, int(math.ceil(length / cell)))

    def covered(position: float) -> bool:
        on_plane = np.abs(across - position) <= tolerance
        here = along[on_plane]
        if not len(here):
            return False
        if float(np.subtract(*np.percentile(height[on_plane], (90.0, 10.0)))) < settings.lintel_band_min_plane_m * upm:
            return False
        index = np.clip(np.floor((here - along_start) / cell).astype(np.int64), 0, cells - 1)
        return len(np.unique(index)) >= settings.lintel_band_min_share * cells

    if len(across) < 2 * cells:
        return None
    along_start = -length / 2.0
    if not covered(0.0):
        return None
    edges = np.arange(-tolerance, high + step, step)
    counts = np.convolve(np.histogram(across, bins=edges)[0], np.ones(3), mode="same")
    middles = (edges[:-1] + edges[1:]) / 2.0
    candidates = np.flatnonzero(middles >= low)
    if not len(candidates):
        return None
    peak = int(candidates[int(np.argmax(counts[candidates]))])
    position = float(middles[peak])
    if counts[peak] <= 0 or not covered(position):
        return None
    between = (middles > tolerance) & (middles < position - tolerance)
    if not between.any() or float(counts[between].min()) > settings.lintel_band_valley_share * float(counts[peak]):
        return None
    near = across[np.abs(across) <= tolerance]
    far = across[np.abs(across - position) <= 1.5 * step]
    if not len(far):
        return None
    return float(np.median(near)), float(np.median(far))


def _corner_planes(opening: Any, openings: list[Any], face: Any, upm: float, settings: Any) -> list[float]:
    """Depths in the host band of jambs of crossing openings at this opening's ends (the end of the crossing wall)."""
    centre, axis = _centre(opening), _axis(opening)
    half, limit = float(getattr(opening, "width")) / 2.0, settings.max_thickness_m * upm
    planes = []
    for other in openings:
        other_axis = _axis(other)
        if other is opening or abs(float(other_axis @ axis)) > math.sin(math.radians(10.0)):
            continue
        for sign in (-1.0, 1.0):
            jamb = _centre(other) + sign * other_axis * float(getattr(other, "width")) / 2.0
            depth = float(np.dot(jamb - face.start, face.solid_normal))
            beside = abs(abs(float(np.dot(jamb - centre, axis))) - half)
            if 0.0 < depth <= limit + settings.lintel_corner_snap_m * upm and beside <= limit:
                planes.append(depth)
    return planes


def _same_wall(run: Any, runs: list[Any], face_of: dict[int, Any], upm: float, settings: Any) -> list[Any]:
    """``run`` and the assumed-thickness pieces chained to it with the very same band."""
    chain = [run]
    tolerance, reach = settings.opening_flush_m * upm, settings.flush_chain_gap_m * upm
    for member in chain:
        face = face_of[member.face]
        for other in runs:
            if any(other is item for item in chain) or other.evidence not in ASSUMED_THICKNESS or abs(other.thickness - member.thickness) > tolerance:
                continue
            other_face = face_of[other.face]
            if float(np.dot(other_face.solid_normal, face.solid_normal)) < math.cos(math.radians(settings.pair_angle_deg)):
                continue
            first, last = other_face.point(other.start), other_face.point(other.end)
            if max(abs(float(np.dot(point - face.start, face.solid_normal))) for point in (first, last)) > tolerance:
                continue
            low, high = sorted(float(np.dot(point - face.start, face.direction)) for point in (first, last))
            if max(low - member.end, member.start - high, 0.0) <= reach:
                chain.append(other)
    return chain


def apply_lintel_bands(runs: list[Any], faces: list[Any], openings: list[Any], points: np.ndarray | None, storey_height: float, upm: float, settings: Any) -> dict[str, int]:
    """Measured band above each door head -> thickness of its host wall (and of the pieces chained to it).

    Only doors with a measured head and hosts without a measured thickness take part.  The measured face of the host stays
    where the whole run fitted it; only the assumed face moves.  Within ``lintel_corner_snap_m`` the measured face ends on
    the jamb of the crossing wall (two measurements of one corner; the jamb keeps the opening widths).  When such a jamb
    already lies on the present face and the lintel says otherwise, the evidence conflicts and the wall is kept.
    """
    stats = {"measured": 0, "runs_changed": 0, "corner_conflicts": 0}
    if points is None or not len(points) or storey_height <= 0.0:
        return stats
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        return stats
    face_of = {face.index: face for face in faces}
    tolerance, limit = settings.lintel_band_tolerance_m * upm, settings.max_thickness_m * upm
    for opening in openings:
        head = getattr(opening, "head_m", None)
        if _kind(opening) != "DOOR" or head is None:
            continue
        bottom = (float(head) + settings.lintel_band_clearance_m) * upm
        if storey_height - bottom < settings.lintel_band_min_height_m * upm:
            continue
        host = host_run(opening, runs, face_of, upm)
        if host is None or host.evidence not in ASSUMED_THICKNESS:
            continue
        centre, face = _centre(opening), face_of[host.face]
        half = float(getattr(opening, "width")) / 2.0
        radius = half + 2.0 * limit
        near = points[(points[:, 2] >= bottom) & (np.abs(points[:, 0] - centre[0]) <= radius) & (np.abs(points[:, 1] - centre[1]) <= radius)]
        along = (near[:, :2] - centre) @ face.direction
        near, along = near[np.abs(along) <= half], along[np.abs(along) <= half]
        if not len(near):
            continue
        built = host.evidence == "door-host"
        if built:
            # The piece between the jambs is centred on the opening line, which is a face: the wall is on either side of it.
            offset = float(np.dot(centre - face.start, face.solid_normal))
            options = [(offset, 1.0), (offset, -1.0)]
        else:
            options = [(0.0, 1.0)]
        found: list[tuple[int, float, float]] = []  # (supporting points, near, far) from the host face along its solid normal
        for anchor, sign in options:
            across = ((near[:, :2] - face.start) @ face.solid_normal - anchor) * sign
            keep = (across >= -tolerance) & (across <= limit)
            across, stretch = across[keep], along[keep]
            measured = measure_two_faces(across, stretch, near[keep, 2], 2.0 * half, upm, settings)
            if measured is not None:
                support = int(np.count_nonzero(np.abs(across - measured[1]) <= tolerance))
                found.append((support, anchor + sign * measured[0], anchor + sign * measured[1]))
        if not found:
            continue
        stats["measured"] += 1
        _, first, second = max(found)
        low, high = (min(first, second), max(first, second)) if built else (0.0, second)
        snap = settings.lintel_corner_snap_m * upm
        planes = _corner_planes(opening, openings, face, upm, settings)
        joined = [plane for plane in planes if abs(plane - high) <= snap]
        if joined:
            high = min(joined, key=lambda plane: abs(plane - high))
        elif not built and any(abs(plane - host.thickness) <= settings.opening_flush_m * upm for plane in planes):
            stats["corner_conflicts"] += 1
            continue
        if not settings.min_thickness_m * upm <= high - low <= limit:
            continue
        if max(abs(low), abs(high - host.thickness)) <= settings.opening_flush_m * upm:
            continue
        for run in [host] if built else _same_wall(host, runs, face_of, upm, settings):
            _move_run(run, faces, face_of, low, high - low)
            if not built:
                run.evidence = "lintel"  # a built piece stays "door-host": it is no wall line for the 3D refinement
            stats["runs_changed"] += 1
    return stats


def _keep_joints(piece: Any, size: float, runs: list[Any], face_of: dict[int, Any], upm: float, settings: Any) -> None:
    """Crossing pieces that end just short of ``piece`` are extended to it before it moves by ``size``: the joint stays closed."""
    face = face_of[piece.face]
    first, last = piece.start - piece.extend_start, piece.end + piece.extend_end
    body = Polygon([tuple(face.point(first)), tuple(face.point(last)), tuple(face.point(last, piece.thickness)), tuple(face.point(first, piece.thickness))])
    for other in runs:
        other_face = face_of[other.face]
        if other is piece or other.thickness <= 0.0 or abs(float(other_face.direction @ face.direction)) > 0.5:
            continue
        for at_start in (True, False):
            position = other.start if at_start else other.end
            outward = -other_face.direction if at_start else other_face.direction
            gaps = []
            for fraction in (0.05, 0.5, 0.95):  # as the free-end closing of the wall stage
                end = other_face.point(position, other.thickness * fraction)
                hit = LineString([tuple(end), tuple(end + outward * (size + settings.joint_tolerance_m * upm))]).intersection(body)
                if not hit.is_empty:
                    gaps.append(float(Point(tuple(end)).distance(hit)))
            if not gaps or min(gaps) <= 0.01 * upm:
                continue
            gap = min(gaps)
            if at_start:
                other.extend_start = max(other.extend_start, gap + 0.01 * upm)
            else:
                other.extend_end = max(other.extend_end, gap + 0.01 * upm)


def flush_interrupted_runs(runs: list[Any], faces: list[Any], openings: list[Any], upm: float, settings: Any) -> int:
    """Pieces that continue the host beside a jamb share its band; returns how many were aligned.

    The better measured band wins (the host on a tie).  Steps above ``flush_line_offset_m`` are other walls; a step of two
    measured pieces of at least ``step_proof_offset_m`` is a real reveal.  Unlike the flushing by wall line this also joins
    pieces that were recorded from opposite faces.
    """
    face_of = {face.index: face for face in faces}
    tolerance, reach = settings.opening_flush_m * upm, settings.jamb_reach_m * upm
    parallel = math.cos(math.radians(settings.pair_angle_deg))
    changed = 0
    for opening in openings:
        host = host_run(opening, runs, face_of, upm)
        if host is None or host.evidence in OTHER_ELEMENTS:
            continue
        centre = _centre(opening)
        half = float(getattr(opening, "width")) / 2.0
        for other in runs:
            if other is host or other.thickness <= 0.0 or other.evidence in OTHER_ELEMENTS:
                continue
            face, other_face = face_of[host.face], face_of[other.face]
            if abs(float(face.direction @ other_face.direction)) < parallel:
                continue
            low, high = sorted(float(np.dot(other_face.point(value) - centre, face.direction)) for value in (other.start - other.extend_start, other.end + other.extend_end))
            if max(low - half, -half - high, 0.0) > reach:
                continue
            normal = face.solid_normal
            own, theirs = run_band(host, face, centre, normal), run_band(other, other_face, centre, normal)
            size = max(abs(own[0] - theirs[0]), abs(own[1] - theirs[1]))
            shared = min(own[1], theirs[1]) - max(own[0], theirs[0])
            if size <= tolerance or size > settings.flush_line_offset_m * upm or shared < 0.5 * min(host.thickness, other.thickness):
                continue
            if {host.evidence, other.evidence} <= MEASURED_FACES and size >= settings.step_proof_offset_m * upm:
                continue
            winner, loser = (other, host) if _STRENGTH.get(other.evidence, 1) > _STRENGTH.get(host.evidence, 1) else (host, other)
            target = theirs if winner is other else own
            loser_face = face_of[loser.face]
            same = float(loser_face.solid_normal @ normal) > 0.0
            current = run_band(loser, loser_face, centre, normal)
            shift = target[0] - current[0] if same else current[1] - target[1]
            _keep_joints(loser, size, runs, face_of, upm, settings)
            _move_run(loser, faces, face_of, shift, target[1] - target[0])
            changed += 1
    return changed
