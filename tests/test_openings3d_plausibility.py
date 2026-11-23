"""Plausibility rules of the 3D opening refinement (synthetic walls, noisy evidence, no exact ties).

1. a plan door whose span runs on beyond the end of its wall line continues the line, so the door is not clipped to a sliver
2. plan windows beyond a short unmeasured pier continue an exterior wall line (a glazed wall the 2D plan did not draw as wall)
3. no door / passage narrower than any doorway is kept
4. a photographed floor-reaching rectangle narrower than a glazed door without wall points around it is no window
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from openings3d_synth import Scene, build_evidence
from samcloud.bim.openings3d.classify import classify_openings
from samcloud.bim.openings3d.config import load_openings_config
from samcloud.bim.openings3d.detect import detect_openings, narrow_floor_window
from samcloud.bim.openings3d.model import RefinedOpening
from samcloud.bim.openings3d.walls import wall_lines
from samcloud.bim.plan_model import BuildingPlan, PlanOpening, PlanWall

UNITS = pytest.mark.parametrize("upm", [1.0, 0.9891])
HEIGHT = 2.6


def _plan(walls, openings, upm):
    frame = {"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}
    return BuildingPlan("synthetic", "level_1", "Level 1", upm, frame, 0.0, HEIGHT * upm, walls=walls, openings=openings)


def _wall(identifier, p0, p1, upm, thickness=0.2, kind="interior"):
    return PlanWall(id=identifier, centreline=[[p0[0] * upm, p0[1] * upm], [p1[0] * upm, p1[1] * upm]], thickness_m=thickness, kind=kind)


def _opening(identifier, kind, center, width, axis, upm, host=None):
    return PlanOpening(id=identifier, kind=kind, center=[center[0] * upm, center[1] * upm], width_m=width, axis_angle=axis, wall_thickness_m=0.2, host_wall_id=host)


def _span(line, upm):
    """(low, high) of the line's extent along its main axis, in metres."""
    ends = [line.a, line.a + line.d * line.length]
    axis = 0 if abs(line.d[0]) > abs(line.d[1]) else 1
    values = sorted(float(point[axis]) / upm for point in ends)
    return values[0], values[1]


@UNITS
def test_door_seed_running_on_beyond_the_line_end_continues_the_line(upm):
    """The vertical wall piece starts at y = 3.03; the plan door (centre on the piece) reaches down to y = 2.63, over the line end."""
    config = load_openings_config()
    walls = [_wall("W001", (4.61, 3.03), (4.60, 6.02), upm), _wall("W002", (0.0, 2.21), (9.0, 2.23), upm)]
    seed = _opening("O001", "door", (4.61, 3.03), 0.81, math.pi / 2, upm)
    plain = next(line for line in wall_lines(_plan(walls, [], upm), config) if abs(line.d[1]) > 0.9)
    extended = next(line for line in wall_lines(_plan(walls, [seed], upm), config) if abs(line.d[1]) > 0.9)
    assert _span(plain, upm)[0] == pytest.approx(3.03, abs=0.01)
    assert _span(extended, upm)[0] <= 3.03 - 0.81 / 2 + 0.02  # the grid reaches the door's far edge
    assert extended.pad_start_m > 0.3 or extended.pad_end_m > 0.3  # the continuation is an artificial end
    # a door well inside the piece leaves the line as it is
    inside = _opening("O002", "door", (4.61, 4.52), 0.81, math.pi / 2, upm)
    same = next(line for line in wall_lines(_plan(walls, [inside], upm), config) if abs(line.d[1]) > 0.9)
    assert same.length_m == pytest.approx(plain.length_m, abs=1e-6)


@UNITS
def test_plan_windows_beyond_a_short_pier_continue_an_exterior_line(upm):
    """Exterior wall piece 5.03..9.02; two plan windows without host west of it (pier 0.7 m, then 0.24 m), crossing wall at x = 1.61."""
    config = load_openings_config()
    walls = [
        _wall("W001", (5.03, 0.01), (9.02, 0.03), upm, kind="exterior"),
        _wall("W002", (1.61, 0.02), (1.60, 6.03), upm, kind="exterior"),
    ]
    windows = [_opening("O001", "window", (3.98, 0.02), 0.64, 0.0, upm), _opening("O002", "window", (2.76, 0.01), 1.32, 0.0, upm)]
    line = next(line for line in wall_lines(_plan(walls, windows, upm), config) if abs(line.d[0]) > 0.9)
    low, high = _span(line, upm)
    assert high == pytest.approx(9.02, abs=0.01)
    # both windows lie on the line with room to spare: a window rectangle ending at a plan edge is not cut by the artificial end
    assert low <= 2.76 - 1.32 / 2 - config.windows.pad_edge_margin_m
    assert line.pad_start_m > 3.0
    # a plan window 1.5 m beyond the line end (more than a pier) does not continue the line
    far = [_opening("O003", "window", (3.03, 0.02), 1.0, 0.0, upm)]
    alone = next(line for line in wall_lines(_plan(walls, far, upm), config) if abs(line.d[0]) > 0.9)
    assert _span(alone, upm)[0] == pytest.approx(5.03, abs=0.01)


def test_door_clipped_by_the_line_end_is_measured_over_its_whole_gap():
    """Wall piece 0..3.02 m; the plan door (centre 3.07, 0.8 m) runs 0.45 m beyond it; the measured hole is 2.66..3.41 m."""
    walls = [PlanWall(id="W001", centreline=[[0.0, 0.0], [3.02, 0.0]], thickness_m=0.25, kind="interior")]
    seed = PlanOpening("O001", "door", [3.07, 0.0], 0.8, 0.0, 0.25, "W001")
    frame = {"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}
    plan = BuildingPlan("synthetic", "level_1", "Level 1", 1.0, frame, 0.0, HEIGHT, walls=walls, openings=[seed])
    hole = (2.66, 3.41, 0.0, 2.03)
    evidence, plan, config = build_evidence(Scene(length_m=3.6, kind="interior", holes=[hole], doors=[hole]), plan=plan)
    detection = detect_openings({evidence.line.id: evidence}, [seed], plan, config)
    doors = [o for o in detection.openings if o.kind in ("door", "passage")]
    assert len(doors) == 1
    assert doors[0].width_m == pytest.approx(0.75, abs=0.1)  # clipped at the old line end it was 0.36 m


def test_door_narrower_than_any_doorway_is_dropped():
    config = load_openings_config()
    frame = {"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}
    plan = BuildingPlan("synthetic", "level_1", "Level 1", 1.0, frame, 0.0, HEIGHT, walls=[], openings=[])

    def door(width, centre):
        return RefinedOpening(
            id="", kind="door", center=[centre, 0.0], width_m=width, axis_angle=0.0, wall_thickness_m=0.2, host_wall_id="W001", sill_m=0.0, head_m=2.0,
            height_source="points", confidence=0.55, line="W001", u0=centre - width / 2, u1=centre + width / 2, plan_seed="O001",
        )

    points = SimpleNamespace(grids={}, door_points=None)
    kept, dropped, _ = classify_openings([door(0.31, 1.03), door(0.83, 3.02)], [], points, plan, config)
    assert [round(o.width_m, 2) for o in kept] == [0.83]
    assert len(dropped) == 1 and "narrower than any doorway" in dropped[0]["reason"]


def test_narrow_floor_reaching_photo_window_is_no_window():
    # photographs vote "window" from the floor to 2.3 m over 0.55 m, the wall points run on behind it: a reflection / shadow, no window
    rect = (1.03, 1.58, 0.0, 2.31)
    evidence, plan, config = build_evidence(Scene(length_m=4.0, windows=[rect]))
    detection = detect_openings({evidence.line.id: evidence}, [], plan, config)
    assert [o for o in detection.openings if o.kind == "window"] == []
    assert any("narrower than a glazed door" in str(item.get("reason")) for item in detection.rejected)
    # the same rectangle framed by wall points (a real narrow glazed element) stays
    evidence, plan, config = build_evidence(Scene(length_m=4.0, windows=[rect], holes=[rect]))
    detection = detect_openings({evidence.line.id: evidence}, [], plan, config)
    assert len([o for o in detection.openings if o.kind == "window"]) == 1
    # wide floor-to-ceiling glazing seen on photographs only stays a window
    assert not narrow_floor_window({"v0": 0.0, "u0": 0.42, "u1": 2.47, "src": "photo"}, config)


def test_french_window_photographed_from_low_height_gets_sill_zero():
    """Glazing x 1.03..3.08 down to the floor; the photo votes start only at 0.27 m (threshold / frame rows blurred), no wall points below,
    window-class points of the bottom frame in the floor rows: the window reaches the floor (sill 0), as a points-confirmed measurement."""
    from openings3d_synth import _cells

    rect = (1.03, 3.08, 0.0, 2.37)
    evidence, plan, config = build_evidence(Scene(length_m=4.2, holes=[rect], windows=[(1.03, 3.08, 0.27, 2.37)]))
    rng = np.random.default_rng(5)
    rows, cols = _cells((1.03, 3.08, 0.0, 0.1))
    evidence.grids["win"][rows, cols] = rng.poisson(6.0, evidence.grids["win"][rows, cols].shape)
    detection = detect_openings({evidence.line.id: evidence}, [], plan, config)
    windows = [o for o in detection.openings if o.kind == "window"]
    assert len(windows) == 1 and windows[0].sill_m == 0.0 and windows[0].height_source == "photo+points"
    # without the floor-row frame points (glazing that ends above a hidden parapet) the photographed sill stays
    evidence, plan, config = build_evidence(Scene(length_m=4.2, holes=[rect], windows=[(1.03, 3.08, 0.27, 2.37)]))
    windows = [o for o in detect_openings({evidence.line.id: evidence}, [], plan, config).openings if o.kind == "window"]
    assert len(windows) == 1 and windows[0].sill_m >= 0.2
    # a parapet row of wall points under the rectangle keeps the sill as well
    evidence, plan, config = build_evidence(Scene(length_m=4.2, holes=[(1.03, 3.08, 0.2, 2.37)], windows=[(1.03, 3.08, 0.27, 2.37)]))
    evidence.grids["win"][rows, cols] = rng.poisson(6.0, evidence.grids["win"][rows, cols].shape)
    windows = [o for o in detect_openings({evidence.line.id: evidence}, [], plan, config).openings if o.kind == "window"]
    assert len(windows) == 1 and windows[0].sill_m >= 0.2


def _window_head(evidence, plan, config) -> float:
    windows = [o for o in detect_openings({evidence.line.id: evidence}, [], plan, config).openings if o.kind == "window"]
    assert len(windows) == 1
    return float(windows[0].head_m)


def test_window_head_is_where_the_wall_points_resume_not_where_the_photograph_rectangle_ends():
    """Window 1.03..3.08 m, sill 1.02 m; the wall points resume at 2.08 m, the photograph votes run on to 2.38 m (blind box,
    curtain rail): the head is at the wall (2.1 m), 0.3 m below the rectangle."""
    evidence, plan, config = build_evidence(Scene(length_m=4.2, holes=[(1.03, 3.08, 1.02, 2.08)], windows=[(1.03, 3.08, 1.02, 2.38)]))
    assert _window_head(evidence, plan, config) == pytest.approx(2.10, abs=0.05)
    # the two agree within the blending distance: blended as before, in between
    evidence, plan, config = build_evidence(Scene(length_m=4.2, holes=[(1.03, 3.08, 1.02, 2.27)], windows=[(1.03, 3.08, 1.02, 2.38)]))
    assert 2.27 <= _window_head(evidence, plan, config) <= 2.37


def test_window_head_is_where_the_glazing_ends_when_the_wall_above_is_only_partly_scanned():
    """No row of wall points closes the window below the photograph head (2.38 m), but the glazing is measured: window points
    fill the width up to 2.12 m, none stand above, and wall points stand there in patches: the head is where the glass ends."""
    from openings3d_synth import _cells

    rect = (1.03, 3.08, 1.02, 2.38)

    def scene(wall_above: bool):
        evidence, plan, config = build_evidence(Scene(length_m=4.2, holes=[rect], windows=[rect]))
        rng = np.random.default_rng(7)
        rows, cols = _cells((1.03, 3.08, 1.02, 2.12))
        evidence.grids["win"][rows, cols] = rng.poisson(6.0, evidence.grids["win"][rows, cols].shape)
        if wall_above:
            rows, cols = _cells((1.03, 3.08, 2.15, 2.38))
            evidence.solid[rows, cols] = rng.random(evidence.solid[rows, cols].shape) < 0.45
        return evidence, plan, config

    assert _window_head(*scene(wall_above=True)) == pytest.approx(2.12, abs=0.05)
    # nothing measured above the glass (the rows up to the photograph head are not observed by the points): the photograph stays
    assert _window_head(*scene(wall_above=False)) >= 2.3


def test_mirror_on_a_wall_stays_no_window_when_the_wall_points_resume_above_it():
    """A 0.55 m wide rectangle from the floor to 2.31 m on the photographs, no wall points behind it up to 2.02 m (a mirror),
    wall points above: the measured head moves down, but the rectangle is still seen on photographs only - no window."""
    evidence, plan, config = build_evidence(Scene(length_m=4.0, windows=[(1.03, 1.58, 0.0, 2.31)], holes=[(1.03, 1.58, 0.0, 2.02)]))
    detection = detect_openings({evidence.line.id: evidence}, [], plan, config)
    assert [o for o in detection.openings if o.kind == "window"] == []
    assert any("narrower than a glazed door" in str(item.get("reason")) for item in detection.rejected)
