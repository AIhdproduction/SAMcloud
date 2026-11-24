"""Furniture reconstruction must survive touching / collinear part outlines and failing metrics."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import box as sbox

from samcloud.bim.furniture_recon import metrics
from samcloud.bim.furniture_recon.geometry import box, robust_union, shape_volume

Y = 0.894527


def _touching_solids():
    # two parts whose outlines share a collinear, overlapping edge along y=Y (x ranges overlap)
    a = box("a", -1.06356, -0.656536, Y - 0.4, Y, 0.0, 0.5)
    b = box("b", -0.771346, -0.2, Y, Y + 0.3, 0.0, 0.5)
    return a, b


def test_shape_volume_touching_collinear_outlines():
    a, b = _touching_solids()
    volume = shape_volume([a, b], None)
    expected = a.volume() + b.volume()  # they only touch: union volume == sum
    assert math.isfinite(volume) and volume > 0
    assert abs(volume - expected) <= 0.1 * expected


def test_robust_union_collinear_overlapping_edges():
    left = sbox(-1.06356, Y - 0.4, -0.656536, Y)
    right = sbox(-0.771346, Y, -0.2, Y + 0.3)
    merged = robust_union([left, right])
    assert not merged.is_empty
    assert abs(merged.area - (left.area + right.area)) < 1e-6
    assert robust_union([]).is_empty


def test_metric_failure_keeps_item(monkeypatch):
    a, b = _touching_solids()
    points = np.array([[-0.9, Y - 0.2, 0.25], [-0.4, Y + 0.1, 0.25]])

    def boom(*_args, **_kwargs):
        raise RuntimeError("volume failed")

    monkeypatch.setattr(metrics, "shape_volume", boom)
    result = metrics.evaluate(points, [a, b], None)
    assert result is not None and result.volume_m3 > 0

    monkeypatch.setattr(metrics, "sample_shape", boom)
    assert metrics.evaluate(points, [a, b], None) is None
