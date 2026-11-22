"""Opening rules of the 3D refinement that the synthetic bench exposed (2026-11-21).

1. a lintel is recognised from the head's share of the storey height too (scale-independent), with 15 cm of headroom
2. a passage needs the gap to run (nearly) up to the ceiling; 20 cm of wall above a frameless gap is a lintel -> door
3. a lintel door that only the wall-plane measurement found needs its gap measured free (furniture / unscanned stretch)
4. points only: a floor-reaching hole narrower than a door is no doorway (wall projection, niche, furniture gap)
5. points only: a hole dominated by window-class points is a window spanning the glazing (furniture below the sill and a
   shelf beside the window do not make it a door or widen it)
6. points only: a measured hole that IS the plan's door keeps the plan seed
7. one opening seen on two collinear wall lines is kept once
8. a wall line continues over a plan window in the last stretch of the wall
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from samcloud.bim.openings3d.classify import classify_features
from samcloud.bim.openings3d.config import load_openings_config
from samcloud.bim.openings3d.detect import _deduplicate
from samcloud.bim.openings3d.evidence import build_line_evidence
from samcloud.bim.openings3d.frames import FrameFeatures
from samcloud.bim.openings3d.model import RefinedOpening
from samcloud.bim.openings3d.pointsonly import detect_point_openings, points_only_detection
from samcloud.bim.openings3d.walls import WallLine, wall_lines
from samcloud.bim.plan_model import BuildingPlan, PlanOpening, PlanWall

UNITS = pytest.mark.parametrize("upm", [1.0, 0.9891])
CELL = 0.05


def _features(ceiling: float, head: float | None, *, free: float = 1.0, side: float = 1.0) -> FrameFeatures:
    return FrameFeatures(
        ceiling_m=ceiling, head_m=head, headroom_m=None if head is None else max(0.0, ceiling - head), head_used_m=head or 2.0,
        jamb_cov={"left": 0.0, "right": 0.0}, side_wall_cov={"left": side, "right": side}, free_fraction=free, floor_fraction=1.0,
    )


def test_lintel_from_the_share_of_the_storey_height_is_scale_independent():
    config = load_openings_config()
    # a 2.1 m door head under a 2.83 m ceiling measured with a 13 % too large storey scale: 1.86 / 2.50 m
    kind, reason = classify_features(_features(2.50, 1.80), config, seeded=False)
    assert kind == "door" and reason.startswith("lintel")
    # the same opening measured at metric scale
    assert classify_features(_features(2.83, 2.04), config, seeded=False)[0] == "door"


def test_twenty_centimetres_of_wall_above_a_frameless_gap_is_a_lintel_not_a_passage():
    config = load_openings_config()
    kind, reason = classify_features(_features(2.40, 2.20), config, seeded=False)
    assert kind == "door" and reason.startswith("lintel")
    # a gap up to the ceiling (a few cm of noise) without frame is a passage
    kind, reason = classify_features(_features(2.40, 2.32), config, seeded=False)
    assert kind == "passage"
    assert classify_features(_features(2.40, None), config, seeded=False)[0] == "passage"


def test_unseeded_lintel_needs_a_free_gap():
    config = load_openings_config()
    # furniture / unscanned stretch in front of the wall: only half of the gap is measured free
    assert classify_features(_features(2.40, 1.85, free=0.55), config, seeded=False)[0] == "none"
    # the plan saw a door there: the lenient rule keeps it
    assert classify_features(_features(2.40, 1.85, free=0.55), config, seeded=True)[0] == "door"
    assert classify_features(_features(2.40, 1.85, free=0.95), config, seeded=False)[0] == "door"


# ----------------------------------------------------------------------------------------------- points only

def _line(length_m: float, upm: float = 1.0, thickness_m: float = 0.2, identifier: str = "L1", y: float = 0.0) -> WallLine:
    return WallLine(identifier, (identifier,), np.array([0.0, y * upm]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), length_m * upm, length_m, thickness_m * upm, "exterior")


def _grids(length_m: float, height_m: float = 2.5) -> dict[str, np.ndarray]:
    nu, nv = int(round(length_m / CELL)), int(round(height_m / CELL))
    rng = np.random.default_rng(7)
    wall = rng.integers(3, 6, size=(nv, nu)).astype(np.int32)  # dense wall points everywhere
    grids = {name: np.zeros((nv, nu), np.int32) for name in ("win", "door", "other")}
    grids["wall"] = wall
    return grids


def _cells(a_m: float, b_m: float) -> slice:
    return slice(int(round(a_m / CELL)), int(round(b_m / CELL)))


def _finish(grids: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    grids["any"] = grids["wall"] + grids["win"] + grids["door"] + grids["other"]
    return grids


def test_floor_hole_narrower_than_a_door_is_no_passage():
    config = load_openings_config()
    grids = _grids(6.0)
    grids["wall"][_cells(0.0, 2.45), _cells(2.62, 2.98)] = 0  # 0.36 m projection: no wall points in the band, floor to near ceiling
    grids["wall"][_cells(0.0, 2.0), _cells(4.0, 4.85)] = 0  # 0.85 m doorway with a lintel
    items = detect_point_openings(build_line_evidence(_line(6.0), _finish(grids), None, config), config)
    floor = [item for item in items if item["kind"] in ("door", "passage")]
    assert len(floor) == 1 and 3.95 <= floor[0]["u0"] <= 4.05 and 4.8 <= floor[0]["u1"] <= 4.9


def test_window_points_above_furniture_make_a_window_not_a_doorway():
    config = load_openings_config()
    grids = _grids(6.0)
    sofa, glazing, shelf = _cells(1.6, 3.4), _cells(1.83, 3.12), _cells(3.12, 3.62)
    grids["wall"][_cells(0.0, 0.92), sofa] = 0  # wall below the sill hidden by a sofa and its scan shadow
    grids["other"][_cells(0.5, 0.92), sofa] = 3
    grids["wall"][_cells(0.95, 2.15), glazing] = 0
    grids["win"][_cells(0.95, 2.15), glazing] = 4  # frame + glass returns
    grids["wall"][_cells(0.0, 1.85), shelf] = 0  # a tall shelf beside the window hides the wall
    grids["other"][_cells(0.0, 1.85), shelf] = 3
    items = detect_point_openings(build_line_evidence(_line(6.0), _finish(grids), None, config), config)
    windows = [item for item in items if item["kind"] == "window"]
    assert [item["kind"] for item in items].count("door") == 0 and [item["kind"] for item in items].count("passage") == 0
    assert len(windows) == 1
    window = windows[0]
    assert abs(window["u0"] - 1.83) <= 0.06 and abs(window["u1"] - 3.12) <= 0.06  # the glazing, not the shelf
    assert abs(window["v0"] - 0.95) <= 0.06 and abs(window["v1"] - 2.15) <= 0.06


def _plan(walls, openings, upm, height=2.5):
    frame = {"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}
    return BuildingPlan("synthetic", "level_1", "Level 1", upm, frame, 0.0, height * upm, walls=walls, openings=openings)


@UNITS
def test_measured_hole_keeps_the_plan_door_seed(upm):
    config = load_openings_config()
    grids = _grids(6.0)
    grids["wall"][_cells(0.0, 2.05), _cells(2.0, 2.85)] = 0  # doorway with a lintel, leaf open into the room (no door points in the plane)
    line = _line(6.0, upm)
    seed = PlanOpening(id="O007", kind="door", center=[2.43 * upm, 0.0], width_m=0.8, axis_angle=0.0, wall_thickness_m=0.2)
    plan = _plan([PlanWall(id="L1", centreline=[[0.0, 0.0], [6.0 * upm, 0.0]], thickness_m=0.2, kind="exterior")], [seed], upm)
    detection = points_only_detection({"L1": build_line_evidence(line, _finish(grids), None, config)}, [seed], plan, config)
    doors = [record for record in detection.openings if record.kind in ("door", "passage")]
    assert len(doors) == 1 and doors[0].plan_seed == "O007"
    assert doors[0].height_source == "points"  # the measured hole, not a copy of the plan gap


@UNITS
def test_one_window_on_two_collinear_wall_lines_is_kept_once(upm):
    config = load_openings_config()
    plan = _plan([], [], upm)

    def record(line, centre_x, width, confidence):
        return RefinedOpening(
            id="", kind="window", center=[centre_x * upm, 5.4 * upm], width_m=width, axis_angle=0.0, wall_thickness_m=0.3, host_wall_id=line,
            sill_m=0.2, head_m=2.3, height_source="points", confidence=confidence, line=line, u0=centre_x - width / 2, u1=centre_x + width / 2,
        )

    rejected: list = []
    kept = _deduplicate([record("W011", 3.26, 1.0, 0.45), record("W012", 3.92, 2.3, 0.5), record("W012", 1.6, 1.1, 0.5)], plan, config, rejected)
    assert sorted(round(item.width_m, 2) for item in kept) == [1.1, 2.3]
    # a window on a parallel wall 2 m away is a different opening
    kept = _deduplicate([record("W011", 3.26, 1.0, 0.45), RefinedOpening(
        id="", kind="window", center=[3.3 * upm, 3.4 * upm], width_m=1.0, axis_angle=0.0, wall_thickness_m=0.3, host_wall_id="W020",
        sill_m=0.9, head_m=2.1, height_source="points", confidence=0.4, line="W020", u0=0.0, u1=1.0)], plan, config, [])
    assert len(kept) == 2


@UNITS
def test_wall_line_continues_over_a_window_in_the_last_stretch_of_a_wall(upm):
    config = load_openings_config()
    walls = [
        PlanWall(id="W005", centreline=[[0.0, 4.45 * upm], [3.49 * upm, 4.45 * upm]], thickness_m=0.3, kind="exterior"),
        PlanWall(id="W001", centreline=[[5.63 * upm, 4.45 * upm], [5.63 * upm, 0.0]], thickness_m=0.3, kind="exterior"),
    ]
    window = PlanOpening(id="O002", kind="window", center=[4.47 * upm, 4.40 * upm], width_m=1.97, axis_angle=0.0, wall_thickness_m=0.3)
    plain = next(line for line in wall_lines(_plan(walls, [], upm), config) if abs(line.d[0]) > 0.9)
    extended = next(line for line in wall_lines(_plan(walls, [window], upm), config) if abs(line.d[0]) > 0.9)
    assert plain.length_m == pytest.approx(3.49, abs=0.01)
    assert extended.length_m >= 4.47 + 1.97 / 2 - 0.02 and extended.pad_end_m > 0.9
    far = PlanOpening(id="O003", kind="window", center=[8.0 * upm, 4.40 * upm], width_m=1.0, axis_angle=0.0, wall_thickness_m=0.3)
    assert next(line for line in wall_lines(_plan(walls, [far], upm), config) if abs(line.d[0]) > 0.9).length_m == pytest.approx(3.49, abs=0.01)


def test_refined_openings_not_applied_is_reported(tmp_path):
    import json

    from samcloud.bim.openings3d.pipeline import level_cache_dir, refined_openings_applied

    directory = level_cache_dir(tmp_path, "level_1")
    directory.mkdir(parents=True)
    assert refined_openings_applied(tmp_path, "level_1") == (True, "no refined openings")
    (directory / "openings_3d.json").write_text("{}", encoding="utf-8")
    (directory / "plan_model.json").write_text(json.dumps({"source": {"openings_3d": {"sha256": "", "applied": False, "reason": "no current pass-1 snapshot yet"}}}), encoding="utf-8")
    assert refined_openings_applied(tmp_path, "level_1") == (False, "no current pass-1 snapshot yet")
    (directory / "plan_model.json").write_text(json.dumps({"source": {"openings_3d": {"sha256": "abc", "applied": True}}}), encoding="utf-8")
    assert refined_openings_applied(tmp_path, "level_1")[0] is True


def test_scan_shadow_above_a_furniture_top_is_no_window():
    config = load_openings_config()
    grids = _grids(6.0)
    grids["wall"][_cells(1.10, 1.70), _cells(1.0, 1.5)] = 0  # wall hidden above a 1.1 m sideboard top, wall again above
    grids["other"][_cells(0.80, 1.10), _cells(0.9, 1.6)] = 3  # sideboard top points in the wall band
    grids["wall"][_cells(0.80, 1.10), _cells(0.9, 1.6)] = 0
    grids["wall"][_cells(0.90, 2.00), _cells(3.5, 4.3)] = 0  # a window hole without glass returns, wall below it
    items = detect_point_openings(build_line_evidence(_line(6.0), _finish(grids), None, config), config)
    windows = [item for item in items if item["kind"] == "window"]
    assert len(windows) == 1 and 3.45 <= windows[0]["u0"] <= 3.55


def test_two_windows_whose_holes_merged_stay_two_windows():
    config = load_openings_config()
    grids = _grids(6.0)
    grids["wall"][:, _cells(1.0, 4.6)] = 0  # wall band unmeasured from the floor up between the windows (scan shadow)
    grids["wall"][_cells(2.45, 2.5), _cells(1.0, 4.6)] = 5
    first, second = _cells(1.0, 2.3), _cells(2.95, 4.6)
    grids["win"][_cells(0.85, 2.20), first] = 4
    grids["win"][_cells(0.05, 2.30), second] = 4  # french window from the floor
    items = sorted((item for item in detect_point_openings(build_line_evidence(_line(6.0), _finish(grids), None, config), config) if item["kind"] == "window"), key=lambda item: item["u0"])
    assert len(items) == 2
    assert abs(items[0]["u0"] - 1.0) <= 0.06 and abs(items[0]["u1"] - 2.3) <= 0.06 and abs(items[0]["v0"] - 0.85) <= 0.06
    assert abs(items[1]["u0"] - 2.95) <= 0.06 and abs(items[1]["v0"] - 0.05) <= 0.06


@UNITS
def test_wall_line_continues_over_windows_side_by_side_without_a_host(upm):
    """The plan lost the wall piece of a glazed facade: only a stub is left, the windows beside each other continue it in turn."""
    config = load_openings_config()
    walls = [PlanWall(id="W003", centreline=[[11.47 * upm, 0.0], [11.47 * upm, 0.84 * upm]], thickness_m=0.3, kind="exterior")]
    windows = [
        PlanOpening(id="O002", kind="window", center=[11.32 * upm, 2.48 * upm], width_m=0.77, axis_angle=math.pi / 2, wall_thickness_m=0.3),
        PlanOpening(id="O005", kind="window", center=[11.32 * upm, 3.79 * upm], width_m=1.85, axis_angle=math.pi / 2, wall_thickness_m=0.3),
        PlanOpening(id="O006", kind="window", center=[11.32 * upm, 1.53 * upm], width_m=1.39, axis_angle=math.pi / 2, wall_thickness_m=0.3, host_wall_id="W003"),
    ]
    line = next(iter(wall_lines(_plan(walls, windows, upm), config)))
    top = max(line.a[1], line.a[1] + line.d[1] * line.length) / upm
    assert top >= 3.79 + 1.85 / 2 - 0.02


@UNITS
def test_door_found_on_both_faces_of_a_thick_wall_is_kept_once(upm):
    """A door in a 0.36 m wall is detected once per wall face: two parallel candidates 0.36 m apart across the wall."""
    from samcloud.ortho.plan import _deduplicate_plan_openings
    from samcloud.ortho.vector.indoor import Opening

    vertical = math.pi / 2
    inner = Opening("DOOR", (3.18 * upm, 2.51 * upm), 0.85 * upm, vertical, 0.03 * upm, class_evidence=0.4)
    outer = Opening("DOOR", (2.82 * upm, 2.49 * upm), 0.85 * upm, vertical, 0.03 * upm, class_evidence=0.2)
    kept = _deduplicate_plan_openings([outer, inner], 0.15 * upm, across_limit=0.45 * upm)
    assert len(kept) == 1 and kept[0].class_evidence == 0.4
    # without the across-wall rule both stay (the old behaviour)
    assert len(_deduplicate_plan_openings([outer, inner], 0.15 * upm)) == 2
    # two doors side by side along the same wall, and a door in a parallel wall 1 m away, are different openings
    beside = Opening("DOOR", (3.18 * upm, 3.60 * upm), 0.85 * upm, vertical, 0.03 * upm)
    parallel = Opening("DOOR", (4.20 * upm, 2.51 * upm), 0.85 * upm, vertical, 0.03 * upm)
    assert len(_deduplicate_plan_openings([inner, beside, parallel], 0.15 * upm, across_limit=0.45 * upm)) == 3
    # windows the same way
    win_a = Opening("WINDOW", (0.0, 5.0 * upm), 1.2 * upm, 0.0, 0.1 * upm)
    win_b = Opening("WINDOW", (0.05 * upm, 5.30 * upm), 1.1 * upm, 0.0, 0.1 * upm)
    assert len(_deduplicate_plan_openings([win_a, win_b], 0.15 * upm, across_limit=0.45 * upm)) == 1
