"""The plan opening carries the measured leaf segment (never a wall) and the evidence stamp the invariants read."""

from __future__ import annotations

import pytest

from samcloud.bim.invariants import door_backing
from samcloud.bim.openings3d.apply import to_plan_opening
from samcloud.bim.openings3d.model import RefinedOpening


def _refined(**changes):
    base = dict(
        id="C001", kind="door", center=[2.0, 0.0], width_m=0.85, axis_angle=0.0, wall_thickness_m=0.2, host_wall_id="W001", sill_m=0.0, head_m=2.0, height_source="points",
        evidence="seed O003; width snapped; " + "x" * 220 + "; door: frame (left jamb + head strip)", confidence=0.9, frame_jambs=["left"], head_frame=True,
        leaf_detected=True, hinge_side="left", swing_side=1, leaf_angle_deg=80.0, leaf={"hinge": [1.575, 0.1], "free": [1.9, 1.3]},
    )
    base.update(changes)
    return RefinedOpening(**base)


@pytest.mark.parametrize("upm", [1.0, 0.9891])
def test_leaf_segment_is_scaled_to_plan_units_and_backing_survives_truncation(upm):
    opening = to_plan_opening(_refined(), upm, (2.0 * upm, 0.0), 0.0)
    assert opening.leaf_hinge == pytest.approx((1.575 * upm, 0.1 * upm)) and opening.leaf_free == pytest.approx((1.9 * upm, 1.3 * upm))
    assert len(opening.evidence) <= 200 and door_backing(opening) == "frame"  # the long measurement text is cut, the backing stamp is not


def test_door_without_leaf_has_no_leaf_segment_and_unbacked_door_is_stamped_none():
    opening = to_plan_opening(_refined(leaf_detected=False, leaf=None, frame_jambs=[], head_frame=False, evidence="seed O001; door frac 0.8"), 1.0, (2.0, 0.0), 0.0)
    assert opening.leaf_hinge is None and opening.leaf_free is None
    assert door_backing(opening) is None
