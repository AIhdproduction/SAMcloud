"""Manhole diameters snap to the Swiss standard sizes (nearest size, tolerance and second-nearest rule)."""
from __future__ import annotations

import numpy as np
import pytest

from samcloud.bim.core.model.enums import ParamOrigin
from samcloud.drone import contracts as c
from samcloud.outdoor_model import civil

CFG = c.load_drone_config()


@pytest.mark.parametrize("measured, expected, snapped", [
    (0.80, 0.785, True),    # nearest 0.785, second 0.745
    (0.83, 0.85, True),     # nearest 0.85 (2 cm) is closer than half the distance to 0.785 (4.5 cm)
    (0.70, 0.70, False),    # between 0.62 and 0.745: 4.5 cm to the nearest is not below half of 8 cm to the second
    (0.29, 0.27, True),     # small size: 2 cm within the 4 cm tolerance
    (0.34, 0.34, False),    # small size: 7 cm from 0.27 is outside 4 cm, 8 cm from 0.42
    (1.30, 1.30, False),    # 25 cm beyond the largest size: far outside the tolerance
])
def test_snap_rules(measured, expected, snapped):
    value, label, did = civil.snap_manhole_diameter(measured, CFG)
    assert did is snapped and value == pytest.approx(expected)
    assert bool(label) is snapped


def _cover(rng, diameter, n=900, z=1.0):
    r = diameter / 2 * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([5 + r * np.cos(a), 5 + r * np.sin(a), z + rng.normal(0, 0.003, n)])


def _detect(diameter, seed=1):
    xyz = _cover(np.random.default_rng(seed), diameter)
    ps = c.PointSet(xyz, np.full(len(xyz), c.MANHOLE, dtype=np.uint16))
    return civil.manhole_detections(ps, config=CFG)


def test_detected_covers_get_the_standard_diameter_and_keep_the_measurement():
    (det,) = _detect(0.80)
    assert det.status == "ok" and det.snapped and det.diameter == pytest.approx(0.785) and abs(det.diameter_measured - 0.80) < 0.02
    (element,) = civil.build_civil_elements([], manholes=[det], config=CFG)[0]
    assert element.params.diameter == pytest.approx(0.785) and element.param_origin["diameter"] == ParamOrigin.SNAPPED
    assert element.evidence["snapped"] is True and element.evidence["standard"] == "Rahmen LW600 (Typ Viatop)"
    assert abs(element.evidence["diameter_measured_m"] - 0.80) < 0.02
    insert = civil.to_dxf_entities([element], c.LocalFrame(origin_lv95=(2_600_000.0, 1_200_000.0, 430.0)), config=CFG)["S-MANH"][0]
    assert insert["xscale"] == pytest.approx(0.785 / 0.6)  # the symbol is drawn at the snapped diameter


def test_small_valve_caps_are_accepted_and_far_sizes_stay_measured():
    (small,) = _detect(0.29, seed=2)
    assert small.status == "ok" and small.snapped and small.diameter == pytest.approx(0.27)
    (odd,) = _detect(0.70, seed=3)
    assert odd.status == "ok" and not odd.snapped and odd.diameter == pytest.approx(odd.diameter_measured)
    (element,) = civil.build_civil_elements([], manholes=[odd], config=CFG)[0]
    assert element.param_origin["diameter"] == ParamOrigin.FITTED and element.evidence["snapped"] is False
    (huge,) = _detect(1.30, seed=4)
    assert huge.status == "rejected"  # outside the 0.25-1.2 m range
