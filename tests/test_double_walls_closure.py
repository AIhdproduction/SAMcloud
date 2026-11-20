"""Wall stage rules: an open door leaf is never a wall, steps are straightened and free ends across a step are joined."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from samcloud.ortho.vector.double_walls import (
    DoubleWallSettings,
    WallRun,
    _close_free_ends,
    _drop_leaf_runs,
    _free_ends,
    _merge_jogs,
    _run_centre,
    _Face,
)
from samcloud.ortho.vector.indoor import Opening

SETTINGS = DoubleWallSettings(double_line=True)
UNITS = pytest.mark.parametrize("upm", [1.0, 0.9891])


def _wall(faces, runs, p0, p1, upm, thickness=0.2, evidence="face", noise=0.0):
    """Wall along p0 -> p1 (metres); its face is offset by half a thickness so the centreline passes through p0 -> p1 (solid normal = left)."""
    start, end = np.asarray(p0, dtype=float) * upm, np.asarray(p1, dtype=float) * upm
    direction = (end - start) / np.linalg.norm(end - start)
    normal = np.asarray((-direction[1], direction[0]))
    shift = normal * thickness * upm / 2.0
    jitter = np.asarray((0.0, noise * upm))
    face = _Face(len(faces), start - shift + jitter, end - shift + jitter, {"type": "polyline", "points": [], "closed": False}, side=1)
    faces.append(face)
    runs.append(WallRun(face.index, 0.0, face.length, thickness * upm, "interior", evidence))
    return runs[-1]


def _leaf_door(upm, hinge, free):
    return Opening("DOOR", (2.0 * upm, 0.0), 0.85 * upm, 0.0, 0.2 * upm, leaf_detected=True, hinge_side="left", swing_side=1, leaf_angle_deg=80.0, leaf_hinge=(hinge[0] * upm, hinge[1] * upm), leaf_free=(free[0] * upm, free[1] * upm))


@UNITS
def test_wall_piece_lying_along_the_open_leaf_is_dropped(upm):
    faces, runs = [], []
    _wall(faces, runs, (0.0, 0.0), (4.0, 0.0), upm)  # the real wall the door sits in
    leaf_wall = _wall(faces, runs, (0.03, 0.9), (0.04, -0.15), upm, evidence="room", noise=0.01)  # thin piece standing where the leaf stands
    other = _wall(faces, runs, (6.0, 0.0), (6.0, 1.0), upm)  # unrelated wall of similar length, far away
    door = _leaf_door(upm, hinge=(0.0, 0.9), free=(0.1, -0.2))
    dropped = _drop_leaf_runs(runs, faces, [door], upm, SETTINGS)
    assert dropped == 1 and leaf_wall not in runs and other in runs and len(runs) == 2


@UNITS
def test_wall_longer_than_the_leaf_or_off_the_leaf_is_kept(upm):
    faces, runs = [], []
    long_wall = _wall(faces, runs, (0.0, 0.0), (0.0, 3.0), upm)  # three times the leaf length, along it
    offset = _wall(faces, runs, (0.8, 0.0), (0.8, 1.2), upm)  # as long as the leaf but 0.8 m away
    door = _leaf_door(upm, hinge=(0.0, 0.0), free=(0.0, 1.2))
    assert _drop_leaf_runs(runs, faces, [door], upm, SETTINGS) == 0
    assert long_wall in runs and offset in runs


@UNITS
def test_no_leaf_no_change(upm):
    faces, runs = [], []
    _wall(faces, runs, (0.0, 0.0), (0.0, 1.2), upm)
    plain = Opening("DOOR", (0.0, 0.6 * upm), 0.85 * upm, 0.0, 0.2 * upm)
    assert _drop_leaf_runs(runs, faces, [plain], upm, SETTINGS) == 0


@UNITS
def test_small_room_edge_step_is_straightened(upm):
    faces, runs = [], []
    long = _wall(faces, runs, (0.0, 14.65), (3.0, 14.65), upm, evidence="room-edge")
    stepped = _wall(faces, runs, (2.8, 14.45), (3.5, 14.45), upm, evidence="room-edge", noise=0.003)  # 0.2 m step, overlaps the long wall
    assert _merge_jogs(runs, faces, upm, SETTINGS) == 1
    centre = _run_centre(stepped, faces[stepped.face], stepped.start)
    assert centre[1] / upm == pytest.approx(14.65, abs=0.01)
    assert _run_centre(long, faces[long.face], long.start)[1] / upm == pytest.approx(14.65, abs=0.01)


@UNITS
def test_measured_walls_and_big_offsets_are_not_moved(upm):
    faces, runs = [], []
    _wall(faces, runs, (0.0, 14.65), (3.0, 14.65), upm, evidence="room-edge")
    measured = _wall(faces, runs, (2.8, 14.45), (3.5, 14.45), upm, evidence="face")  # measured: never moved
    far = _wall(faces, runs, (2.8, 13.9), (3.5, 13.9), upm, evidence="room-edge")  # 0.75 m step: a real offset
    assert _merge_jogs(runs, faces, upm, SETTINGS) == 0
    assert _run_centre(measured, faces[measured.face], measured.start)[1] / upm == pytest.approx(14.45, abs=0.01)
    assert _run_centre(far, faces[far.face], far.start)[1] / upm == pytest.approx(13.9, abs=0.01)


@UNITS
def test_free_ends_across_a_step_are_joined_by_a_wall_piece(upm):
    faces, runs = [], []
    a = _wall(faces, runs, (-3.0, 14.65), (-2.55, 14.65), upm)
    b = _wall(faces, runs, (-2.53, 14.04), (-1.9, 14.04), upm)
    before = len(_free_ends(runs, faces, [], upm, SETTINGS))
    assert before == 4  # all four ends are free
    assert _close_free_ends(runs, faces, [], upm, SETTINGS) == 1
    connector = runs[-1]
    assert connector.evidence == "closure"
    face = faces[connector.face]
    line = LineString([tuple(_run_centre(connector, face, connector.start)), tuple(_run_centre(connector, face, connector.end))])
    # the piece runs from the end of wall A to the end of wall B and overlaps both by half a thickness
    assert line.distance(Point(-2.55 * upm, 14.65 * upm)) < 0.02 * upm and line.distance(Point(-2.53 * upm, 14.04 * upm)) < 0.02 * upm
    after = _free_ends(runs, faces, [], upm, SETTINGS)
    assert len(after) == 2  # the two outer ends stay free
    assert a in runs and b in runs


@UNITS
def test_ends_at_an_opening_or_far_apart_are_not_joined(upm):
    faces, runs = [], []
    _wall(faces, runs, (0.0, 0.0), (2.0, 0.0), upm)
    _wall(faces, runs, (2.0, 2.5), (4.0, 2.5), upm)  # 2.5 m step: not a closing piece
    assert _close_free_ends(runs, faces, [], upm, SETTINGS) == 0
    faces, runs = [], []
    _wall(faces, runs, (0.0, 0.0), (2.0, 0.0), upm)
    _wall(faces, runs, (2.9, 0.0), (5.0, 0.0), upm)  # collinear wall pieces with a gap: that is a door gap, never closed here
    door = Opening("DOOR", (2.45 * upm, 0.0), 0.9 * upm, 0.0, 0.2 * upm)
    assert _close_free_ends(runs, faces, [door], upm, SETTINGS) == 0
    assert len(_free_ends(runs, faces, [door], upm, SETTINGS)) == 2  # only the outer ends; the jambs touch the opening


@UNITS
def test_a_t_junction_is_closed_not_free(upm):
    faces, runs = [], []
    _wall(faces, runs, (0.0, 0.0), (4.0, 0.0), upm)
    _wall(faces, runs, (2.0, 0.0), (2.0, 3.0), upm)
    free = _free_ends(runs, faces, [], upm, SETTINGS)
    ends = sorted((round(float(p[0]) / upm, 2), round(float(p[1]) / upm, 2)) for _r, _e, p, _d in free)
    assert (2.0, 0.0) not in ends and (2.0, 3.0) in ends


# --- a door is never a wall: enforced after the runs are built ------------------------------------------------------------------------------

from samcloud.ortho.vector.double_walls import _enforce_door_spans  # noqa: E402


def _door(upm, centre, width=0.85, axis=0.0, thickness=0.26):
    return Opening("DOOR", (centre[0] * upm, centre[1] * upm), width * upm, axis, thickness * upm)


def _covered(runs, faces, point):
    return [run for run in runs if LineString([tuple(_run_centre(run, faces[run.face], run.start)), tuple(_run_centre(run, faces[run.face], run.end))]).distance(Point(*point)) <= run.thickness / 2.0 + 1e-6]


@UNITS
def test_parallel_duplicate_run_beside_the_host_is_removed(upm):
    faces, runs = [], []
    host = _wall(faces, runs, (7.99, 2.17), (8.95, 2.17), upm, thickness=0.26, evidence="room")
    duplicate = _wall(faces, runs, (7.86, 2.26), (8.95, 2.26), upm, thickness=0.26, evidence="learned")
    counts = _enforce_door_spans(runs, faces, [_door(upm, (8.38, 2.17))], upm, SETTINGS)
    assert host in runs and duplicate not in runs and counts["runs_removed"] == 1


@UNITS
def test_parallel_run_is_cut_back_to_the_jambs_keeping_long_remnants(upm):
    faces, runs = [], []
    host = _wall(faces, runs, (0.0, 0.0), (6.0, 0.0), upm)
    other = _wall(faces, runs, (0.0, 0.1), (6.0, 0.1), upm)
    _enforce_door_spans(runs, faces, [_door(upm, (3.0, 0.0), 0.9)], upm, SETTINGS)
    assert host in runs and other not in runs
    pieces = [run for run in runs if run is not host]
    spans = sorted((faces[run.face].point(run.start)[0] / upm, faces[run.face].point(run.end)[0] / upm) for run in pieces)
    assert len(spans) == 2 and spans[0][1] == pytest.approx(2.55, abs=0.01) and spans[1][0] == pytest.approx(3.45, abs=0.01)


@UNITS
def test_collinear_pieces_of_the_host_wall_are_kept(upm):
    faces, runs = [], []
    first = _wall(faces, runs, (0.0, 0.0), (3.2, 0.0), upm)
    second = _wall(faces, runs, (2.8, 0.0), (6.0, 0.0), upm)
    _enforce_door_spans(runs, faces, [_door(upm, (3.0, 0.0), 0.9)], upm, SETTINGS)
    assert first in runs and second in runs


@UNITS
def test_open_leaf_wall_perpendicular_in_the_span_is_cut_back(upm):
    faces, runs = [], []
    host = _wall(faces, runs, (0.0, 0.0), (6.0, 0.0), upm)
    leaf = _wall(faces, runs, (3.0, -0.1), (3.0, 0.9), upm, thickness=0.1)  # thin piece that hangs from the span into the room (begins inside the span)
    _enforce_door_spans(runs, faces, [_door(upm, (3.0, 0.0), 0.9)], upm, SETTINGS)
    assert host in runs and leaf not in runs
    for run in runs:
        if run is host:
            continue
        for end in (run.start, run.end):
            point = _run_centre(run, faces[run.face], end) / upm
            assert not (2.55 < point[0] < 3.45 and abs(point[1]) < 0.12)


@UNITS
def test_t_wall_ending_at_the_jamb_is_untouched(upm):
    faces, runs = [], []
    _wall(faces, runs, (0.0, 0.0), (6.0, 0.0), upm)
    stub = _wall(faces, runs, (2.55, 0.0), (2.55, 3.0), upm)
    _enforce_door_spans(runs, faces, [_door(upm, (3.0, 0.0), 0.9)], upm, SETTINGS)
    assert stub in runs and (stub.start, stub.end) == (0.0, stub.end)


@UNITS
def test_door_without_any_wall_gets_a_door_fill_host(upm):
    faces, runs = [], []
    near = _wall(faces, runs, (6.66, 6.0), (6.66, 11.1), upm, thickness=0.3)  # wall ends 0.4 m before the door, 0.3 m beside its axis
    door = _door(upm, (6.36, 11.58), 0.85, axis=np.pi / 2, thickness=0.2)
    counts = _enforce_door_spans(runs, faces, [door], upm, SETTINGS)
    assert counts["hosts_built"] == 1
    filler = runs[-1]
    assert filler.evidence == "door-host" and filler.thickness == pytest.approx(0.3 * upm)  # thickness of the nearest wall
    point = _run_centre(filler, faces[filler.face], 0.5 * (filler.start + filler.end)) / upm
    assert point[0] == pytest.approx(6.36, abs=0.01) and point[1] == pytest.approx(11.58, abs=0.02)
    assert near in runs and _enforce_door_spans(runs, faces, [door], upm, SETTINGS)["hosts_built"] == 0  # idempotent


@UNITS
def test_door_in_a_gap_between_two_wall_pieces_has_its_host_and_builds_nothing(upm):
    faces, runs = [], []
    _wall(faces, runs, (0.0, 0.0), (2.55, 0.0), upm)
    _wall(faces, runs, (3.45, 0.0), (6.0, 0.0), upm)
    before = len(runs)
    counts = _enforce_door_spans(runs, faces, [_door(upm, (3.0, 0.0), 0.9)], upm, SETTINGS)
    assert counts == {"hosts_built": 0, "runs_cut": 0, "runs_removed": 0} and len(runs) == before


@UNITS
def test_wall_passing_the_end_of_the_span_at_a_junction_is_kept(upm):
    faces, runs = [], []
    _wall(faces, runs, (0.0, 0.0), (6.0, 0.0), upm)  # horizontal host; the door below is vertical, its span reaches the host wall
    crossing = _wall(faces, runs, (2.5, 0.0), (5.0, 0.0), upm, thickness=0.24)
    door = _door(upm, (3.0, 0.45), 0.95, axis=np.pi / 2, thickness=0.3)  # span y -0.03 .. 0.93: the crossing wall runs through its lower end
    _enforce_door_spans(runs, faces, [door], upm, SETTINGS)
    assert crossing in runs
