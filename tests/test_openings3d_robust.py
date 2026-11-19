"""Opening detection must not depend on where a small change of the wall set puts the wall-line ends or the line split."""

from __future__ import annotations

import numpy as np
import pytest

from openings3d_synth import Scene, build_evidence, make_gap_plan
from samcloud.bim.openings3d.config import load_openings_config
from samcloud.bim.openings3d.detect import detect_openings
from samcloud.bim.openings3d.walls import wall_lines

WINDOW = (4.50, 6.30, 1.10, 2.20)  # world x range of a 1.8 m window that straddles the end of the plan wall on its left
MULLION = 5.40


def _scene() -> Scene:
    return Scene(holes=[WINDOW], windows=[WINDOW], mullions=[MULLION])


def _detect(plan, config=None):
    """Detect on the wall line that holds the window and return (window openings, wall line)."""
    config = config or load_openings_config()
    lines = wall_lines(plan, config)
    index = next(i for i, line in enumerate(lines) if line.a[0] <= 5.4 <= line.a[0] + line.length)
    evidence, _, config = build_evidence(_scene(), config, plan=plan, line_index=index)
    detection = detect_openings({evidence.line.id: evidence}, [], plan, config)
    return [o for o in detection.openings if o.kind == "window"], lines[index]


def test_gap_longer_than_bridge_limit_splits_the_line_and_pads_both_facing_ends() -> None:
    config = load_openings_config()
    gap = config.walls.max_gap_m + 0.25
    lines = wall_lines(make_gap_plan([(0.0, 2.0), (2.0 + gap, 9.0)]), config)
    assert len(lines) == 2
    left, right = sorted(lines, key=lambda line: line.a[0])
    pad = min(gap, config.walls.end_pad_m)
    assert left.a[0] == pytest.approx(0.0)  # a free end (no neighbour) is not padded
    assert left.length == pytest.approx(2.0 + pad)  # but the end towards the gap continues into it
    assert right.a[0] == pytest.approx(2.0 + gap - pad)
    assert right.a[0] + right.length == pytest.approx(9.0)


def test_pad_never_exceeds_the_gap() -> None:
    config = load_openings_config()
    gap = config.walls.max_gap_m + 0.25
    config = load_openings_config(overrides={"walls": {"end_pad_m": 10.0}})
    left, right = sorted(wall_lines(make_gap_plan([(0.0, 2.0), (2.0 + gap, 9.0)]), config), key=lambda line: line.a[0])
    assert left.a[0] + left.length == pytest.approx(2.0 + gap)  # reaches the neighbour, not past it
    assert right.a[0] == pytest.approx(2.0)


def test_gap_within_bridge_limit_is_one_unpadded_line() -> None:
    config = load_openings_config()
    lines = wall_lines(make_gap_plan([(0.0, 2.0), (2.0 + config.walls.max_gap_m - 0.2, 9.0)]), config)
    assert len(lines) == 1
    assert lines[0].a[0] == pytest.approx(0.0) and lines[0].length == pytest.approx(9.0)


def test_window_straddling_the_plan_end_is_not_cut_at_the_wall_end() -> None:
    config = load_openings_config()
    plan = make_gap_plan([(0.0, 2.0), (5.25, 9.0)])  # gap of 3.25 m: the line splits, the window starts 0.75 m before the right line
    windows, line = _detect(plan, config)
    assert len(windows) == 1, windows
    window = windows[0]
    assert abs(window.width_m - 1.8) <= 0.1
    assert abs(window.center[0] - 5.4) <= 0.1
    assert window.mullion_count == 1
    assert abs(window.sill_m - 1.10) < 0.05 and abs(window.head_m - 2.20) < 0.05
    # Without the padding the same evidence loses the part of the window left of the plan wall end (documents what the fix prevents).
    unpadded = load_openings_config(overrides={"walls": {"end_pad_m": 0.0}})
    cut, _ = _detect(plan, unpadded)
    assert not cut or cut[0].width_m < 1.4


@pytest.mark.parametrize("gap", [2.85, 3.05, 3.25, 3.45, 3.65])
def test_window_is_stable_when_the_plan_gap_changes_by_centimetres(gap: float) -> None:
    # Ends of the wall pieces moving by +-10 cm (or more) switch between one bridged line and two padded lines.
    plan = make_gap_plan([(0.0, 5.25 - gap), (5.25, 9.0)])
    windows, _ = _detect(plan)
    assert len(windows) == 1, (gap, windows)
    assert abs(windows[0].width_m - 1.8) <= 0.1
    assert abs(windows[0].center[0] - 5.4) <= 0.1
    assert windows[0].mullion_count == 1


@pytest.mark.parametrize("shift", [-0.10, -0.05, 0.05, 0.10])
def test_window_is_stable_when_the_right_piece_starts_earlier_or_later(shift: float) -> None:
    plan = make_gap_plan([(0.0, 2.0), (5.25 + shift, 9.0 + shift)])
    windows, _ = _detect(plan)
    assert len(windows) == 1, (shift, windows)
    assert abs(windows[0].width_m - 1.8) <= 0.1
    assert abs(windows[0].center[0] - 5.4) <= 0.1
    assert windows[0].mullion_count == 1


def test_padded_line_ids_and_pieces_are_unchanged() -> None:
    config = load_openings_config()
    lines = wall_lines(make_gap_plan([(0.0, 2.0), (5.5, 9.0)]), config)
    assert sorted(line.pieces for line in lines) == [("W001",), ("W002",)]
    assert all(np.allclose(line.d, [1.0, 0.0]) for line in lines)


def test_rectangle_cut_by_a_padded_line_end_is_dropped_on_that_line() -> None:
    # On the left line the window is only visible up to the artificial end of the padding: the whole window is on the right line.
    config = load_openings_config()
    plan = make_gap_plan([(0.0, 2.0), (5.25, 9.0)])
    lines = wall_lines(plan, config)
    index = next(i for i, line in enumerate(lines) if line.a[0] == 0.0)
    assert lines[index].pad_end_m > 0 and lines[index].pad_start_m == 0
    evidence, _, config = build_evidence(_scene(), config, plan=plan, line_index=index)
    detection = detect_openings({evidence.line.id: evidence}, [], plan, config)
    assert [o for o in detection.openings if o.kind == "window"] == []
    assert any("padded line end" in str(item.get("reason")) for item in detection.rejected)


def test_window_edges_follow_the_wall_points_not_a_photograph_that_is_a_few_cm_off() -> None:
    hole = (1.00, 2.40, 0.90, 2.10)  # where the wall points are missing: 1.40 m
    photo = (0.94, 2.47, 0.90, 2.10)  # the window class votes extend 6-7 cm further on both sides (1.53 m)
    evidence, plan, config = build_evidence(Scene(holes=[hole], windows=[photo]))
    detection = detect_openings({evidence.line.id: evidence}, [], plan, config)
    window = next(o for o in detection.openings if o.kind == "window")
    assert abs(window.width_m - 1.40) < abs(1.53 - 1.40) / 2  # clearly closer to the points than to the photograph
    assert abs(window.center[0] - 1.70) < 0.05


def test_window_edges_stay_with_the_photograph_when_points_disagree() -> None:
    # The wall points resume 40 cm away from the photographed edge: no agreement, the photograph decides.
    evidence, plan, config = build_evidence(Scene(holes=[(0.6, 2.4, 0.9, 2.1)], windows=[(1.0, 2.0, 0.9, 2.1)]))
    detection = detect_openings({evidence.line.id: evidence}, [], plan, config)
    window = next(o for o in detection.openings if o.kind == "window")
    assert abs(window.width_m - 1.0) <= 0.06


def test_empty_padding_is_not_an_opening_in_points_only_mode() -> None:
    # The padding continues the grid into a plan gap that has no wall points; that empty zone must not become a window or door.
    from samcloud.bim.openings3d.pointsonly import detect_point_openings

    config = load_openings_config()
    plan = make_gap_plan([(0.0, 2.0), (5.25, 9.0)])
    lines = wall_lines(plan, config)
    index = next(i for i, line in enumerate(lines) if line.a[0] == 0.0)
    scene = Scene(holes=[(2.0, 5.0, 0.0, 2.6)])  # the padded zone behind the end of the left piece: no points, nothing observed
    evidence, _, _ = build_evidence(scene, config, plan=plan, line_index=index)
    assert detect_point_openings(evidence, config) == []
