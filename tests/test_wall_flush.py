"""Measured wall-line flushing keeps door-adjacent bodies straight."""

from __future__ import annotations

import numpy as np
import pytest

from samcloud.ortho.vector.double_walls import DoubleWallSettings, WallRun, _Face, _flush_collinear_runs, _run_centre


def _run(faces: list[_Face], runs: list[WallRun], start: tuple[float, float], end: tuple[float, float], *, thickness: float, evidence: str) -> WallRun:
    """Build a horizontal face whose wall centre is the supplied line."""
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    direction /= np.linalg.norm(direction)
    normal = np.asarray((-direction[1], direction[0]))
    face = _Face(
        len(faces),
        np.asarray(start, dtype=float) - normal * thickness / 2.0,
        np.asarray(end, dtype=float) - normal * thickness / 2.0,
        {"type": "polyline", "points": [], "closed": False},
        side=1,
    )
    faces.append(face)
    run = WallRun(face.index, 0.0, face.length, thickness, "interior", evidence)
    runs.append(run)
    return run


@pytest.mark.parametrize("upm", [1.0, 0.9891])
def test_chained_door_fragments_take_the_measured_wall_line_and_thickness(upm: float) -> None:
    faces: list[_Face] = []
    runs: list[WallRun] = []
    measured = _run(faces, runs, (0.0, 12.30 * upm), (2.0 * upm, 12.30 * upm), thickness=0.20 * upm, evidence="face")
    inferred = _run(faces, runs, (2.85 * upm, 12.55 * upm), (6.0 * upm, 12.55 * upm), thickness=0.29 * upm, evidence="learned")
    settings = DoubleWallSettings(flush_line_offset_m=0.30, flush_chain_gap_m=1.50)

    assert _flush_collinear_runs(runs, faces, upm, settings) >= 2
    for run in (measured, inferred):
        point = _run_centre(run, faces[run.face], (run.start + run.end) / 2.0)
        assert point[1] == pytest.approx(12.30 * upm, abs=0.001 * upm)
        assert run.thickness == pytest.approx(0.20 * upm, abs=1e-9)


def test_two_long_independently_measured_faces_keep_a_proven_step() -> None:
    faces: list[_Face] = []
    runs: list[WallRun] = []
    first = _run(faces, runs, (0.0, 0.0), (2.0, 0.0), thickness=0.20, evidence="face")
    second = _run(faces, runs, (0.2, 0.12), (2.2, 0.12), thickness=0.24, evidence="face")

    assert _flush_collinear_runs(runs, faces, 1.0, DoubleWallSettings()) == 0
    assert _run_centre(first, faces[first.face], 1.0)[1] == pytest.approx(0.0)
    assert _run_centre(second, faces[second.face], 1.0)[1] == pytest.approx(0.12)


def test_offset_beyond_the_flush_limit_is_kept_as_a_step() -> None:
    # A 0.23 m offset between chained fragments is larger than the production limit (0.20 m): both lines stay where they were measured.
    faces: list[_Face] = []
    runs: list[WallRun] = []
    north = _run(faces, runs, (0.0, 1.00), (1.5, 1.00), thickness=0.23, evidence="exterior")
    south = _run(faces, runs, (1.9, 0.77), (4.0, 0.77), thickness=0.23, evidence="face-collinear")

    assert DoubleWallSettings().flush_line_offset_m == pytest.approx(0.20)
    assert _flush_collinear_runs(runs, faces, 1.0, DoubleWallSettings()) == 0
    assert _run_centre(north, faces[north.face], 0.75)[1] == pytest.approx(1.00)
    assert _run_centre(south, faces[south.face], 1.0)[1] == pytest.approx(0.77)
