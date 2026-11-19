"""Detection rules of the 3D opening refinement on synthetic noisy wall-plane evidence."""

from __future__ import annotations

import numpy as np
import pytest

from openings3d_synth import Scene, build_evidence, make_plan, seed_door
from samcloud.bim.openings3d.detect import detect_openings
from samcloud.bim.openings3d.pointsonly import detect_point_openings, points_only_detection
from samcloud.bim.openings3d.walls import walls_fingerprint, walls_match, wall_lines
from samcloud.bim.openings3d.config import load_openings_config, config_hash


def _run(scene: Scene, seeds=()):
    evidence, plan, config = build_evidence(scene)
    detection = detect_openings({evidence.line.id: evidence}, list(seeds), plan, config)
    return detection, evidence


def test_two_pane_window_with_mullion_is_one_window() -> None:
    scene = Scene(
        holes=[(1.0, 2.4, 0.9, 2.1)], windows=[(1.0, 2.4, 0.9, 2.1)], mullions=[1.70],
    )
    detection, _ = _run(scene)
    windows = [o for o in detection.openings if o.kind == "window"]
    assert len(windows) == 1, [o.to_dict() for o in detection.openings]
    window = windows[0]
    assert window.mullion_count == 1
    assert abs(window.mullion_u[0] - 0.70) < 0.08  # from the left edge at u = 1.0
    assert abs(window.width_m - 1.40) < 0.06
    assert abs(window.sill_m - 0.90) < 0.03
    assert abs(window.head_m - 2.10) < 0.03
    assert window.height_source == "photo+points"
    assert not window.glazed
    assert 0.0 < window.confidence <= 0.99


def test_window_without_mullion_has_none() -> None:
    detection, _ = _run(Scene(holes=[(1.0, 2.0, 0.9, 2.1)], windows=[(1.0, 2.0, 0.9, 2.1)]))
    windows = [o for o in detection.openings if o.kind == "window"]
    assert len(windows) == 1
    assert windows[0].mullion_count == 0 and windows[0].mullion_u == []


def test_unobserved_hole_is_unknown_not_an_opening() -> None:
    # A gap in the wall points that no photograph saw: neither a window nor a door may appear.
    scene = Scene(unobserved=[(2.8, 3.6, 0.0, 2.1)], holes=[(0.5, 1.4, 0.9, 2.1)], windows=[(0.5, 1.4, 0.9, 2.1)])
    detection, _ = _run(scene)
    assert [o.kind for o in detection.openings] == ["window"]
    assert detection.openings[0].center[0] < 1.5  # only the observed window
    reasons = " ".join(str(item.get("reason")) for item in detection.rejected)
    assert "weakly observed" in reasons


def test_unobserved_window_region_without_votes_is_not_emitted() -> None:
    scene = Scene(unobserved=[(1.0, 2.0, 0.9, 2.1)])
    detection, _ = _run(scene)
    assert detection.openings == []


def test_door_gap_snapping_uses_the_measured_gap_width() -> None:
    gap = (0.80, 1.75, 0.0, 2.12)  # real doorway, 0.95 m wide, lintel at 2.12 m
    scene = Scene(holes=[gap], doors=[gap])
    detection, evidence = _run(scene, seeds=[seed_door("O001", 1.05, 0.70)])
    doors = [o for o in detection.openings if o.kind == "door"]
    assert len(doors) == 1, [o.to_dict() for o in detection.openings]
    door = doors[0]
    assert abs(door.width_m - 0.95) < 0.06  # snapped from the plan's 0.70 m
    assert abs(door.center[0] - 1.275) < 0.06
    assert door.sill_m == 0.0
    assert abs(door.head_m - 2.12) < 0.06
    assert door.height_source in {"points", "photo+points"}
    assert door.plan_seed == "O001"
    assert "width snapped" in door.evidence
    assert not door.glazed


def test_door_without_lintel_evidence_uses_default_head() -> None:
    gap = (0.80, 1.75, 0.0, 2.6)  # points never resume above the gap
    detection, _ = _run(Scene(holes=[gap]), seeds=[seed_door("O002", 1.275, 0.9)])
    door = next(o for o in detection.openings if o.kind in {"door", "passage"})
    assert door.height_source in {"default", "photo"}
    assert door.head_m <= 2.35


def test_glazed_door_is_flagged() -> None:
    gap = (0.80, 1.75, 0.0, 2.10)
    scene = Scene(holes=[gap], doors=[gap], windows=[gap])
    detection, _ = _run(scene, seeds=[seed_door("O003", 1.275, 0.95)])
    doors = [o for o in detection.openings if o.kind == "door"]
    assert len(doors) == 1
    assert doors[0].glazed is True
    assert doors[0].sill_m == 0.0 and abs(doors[0].width_m - 0.95) < 0.06


def test_curtain_region_is_flagged_and_bridges_two_windows() -> None:
    left, right = (0.2, 0.95, 0.4, 2.2), (1.65, 2.4, 0.4, 2.2)
    curtain = (0.95, 1.65, 0.1, 2.4)
    scene = Scene(holes=[left, right], windows=[left, right], furniture=[curtain])
    detection, _ = _run(scene)
    assert len(detection.curtain_regions) == 1
    region = detection.curtain_regions[0]
    assert region["flag"] == "curtain_covered"
    assert 0.8 < region["u_range"][0] < 1.1 and 1.5 < region["u_range"][1] < 1.8
    extended = [o for o in detection.openings if o.evidence == "curtain-extended"]
    assert len(extended) == 1 and extended[0].kind == "window"
    assert extended[0].confidence < 0.5
    assert len([o for o in detection.openings if o.kind == "window"]) == 3


def test_window_on_interior_wall_with_low_sill_is_rejected() -> None:
    rect = (1.0, 2.0, 0.1, 2.1)
    scene = Scene(kind="interior", holes=[rect], windows=[rect])
    detection, _ = _run(scene)
    assert all(o.kind != "window" for o in detection.openings)
    assert any("interior wall" in str(item.get("reason")) for item in detection.rejected)


def test_floor_to_ceiling_glazing_is_a_window_with_zero_sill() -> None:
    rect = (0.5, 2.5, 0.0, 2.3)
    detection, _ = _run(Scene(holes=[rect], windows=[rect]))
    glazing = [o for o in detection.openings if o.glazed]
    assert len(glazing) == 1 and glazing[0].kind == "window" and glazing[0].sill_m == 0.0


def test_see_through_gap_becomes_a_passage_without_seed() -> None:
    gap = (1.0, 1.9, 0.0, 2.1)
    detection, _ = _run(Scene(holes=[gap], see_through=[gap]))
    openings = [o for o in detection.openings if o.kind in {"door", "passage"}]
    assert len(openings) == 1
    assert abs(openings[0].width_m - 0.9) < 0.1


def test_detection_is_deterministic() -> None:
    scene = Scene(holes=[(1.0, 2.4, 0.9, 2.1)], windows=[(1.0, 2.4, 0.9, 2.1)], mullions=[1.70])
    first, _ = _run(scene)
    second, _ = _run(scene)
    assert [o.to_dict() for o in first.openings] == [o.to_dict() for o in second.openings]


# ---------------------------------------------------------------- points only


def test_points_only_window_hole_bounded_by_wall_points() -> None:
    scene = Scene(holes=[(1.0, 2.4, 0.9, 2.1)], windows=[(1.0, 2.4, 0.9, 2.1)])
    evidence, plan, config = build_evidence(scene)
    from samcloud.bim.openings3d.evidence import build_line_evidence

    points_only = build_line_evidence(evidence.line, evidence.grids, None, config)
    assert not points_only.obs.any()
    detection = points_only_detection({points_only.line.id: points_only}, [], plan, config)
    windows = [o for o in detection.openings if o.kind == "window"]
    assert len(windows) == 1
    assert abs(windows[0].sill_m - 0.90) < 0.08 and abs(windows[0].head_m - 2.10) < 0.08
    assert windows[0].height_source == "points"


def test_points_only_hole_touching_the_wall_end_is_unknown() -> None:
    # unobserved gap at the end of the wall: no bounding wall points on the far side
    scene = Scene(unobserved=[(3.2, 4.0, 0.0, 2.1)])
    evidence, plan, config = build_evidence(scene)
    from samcloud.bim.openings3d.evidence import build_line_evidence

    points_only = build_line_evidence(evidence.line, evidence.grids, None, config)
    assert detect_point_openings(points_only, config) == []


def test_points_only_door_seed_without_gap_stays_a_plan_gap() -> None:
    evidence, plan, config = build_evidence(Scene())
    from samcloud.bim.openings3d.evidence import build_line_evidence

    points_only = build_line_evidence(evidence.line, evidence.grids, None, config)
    detection = points_only_detection({points_only.line.id: points_only}, [seed_door("O009", 2.0, 0.9)], plan, config)
    assert [o.kind for o in detection.openings] == ["door"]
    assert detection.openings[0].height_source == "default" and detection.openings[0].confidence < 0.5


# ---------------------------------------------------------------- fingerprints


def test_wall_fingerprint_tolerates_small_shifts_but_not_new_walls() -> None:
    config = load_openings_config()
    lines = wall_lines(make_plan(4.0), config)
    base = walls_fingerprint(lines)
    shifted = walls_fingerprint(wall_lines(make_plan(4.08), config))
    longer = walls_fingerprint(wall_lines(make_plan(4.5), config))
    assert walls_match(base, shifted, 0.15)
    assert not walls_match(base, longer, 0.15)
    assert not walls_match(base, base + base, 0.15)


def test_config_hash_changes_with_a_threshold() -> None:
    assert config_hash(load_openings_config()) != config_hash(load_openings_config(overrides={"windows": {"threshold": 0.5}}))
    assert config_hash(load_openings_config()) == config_hash(load_openings_config())


def test_unknown_setting_raises() -> None:
    with pytest.raises(AttributeError):
        _ = load_openings_config().windows.no_such_threshold
