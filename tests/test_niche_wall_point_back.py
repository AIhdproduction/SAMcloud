"""Regression: _wall_point_back must survive points that fill only 1-2 bins."""

from types import SimpleNamespace

import numpy as np

from samcloud.ortho.vector.niche import NicheSettings, _wall_point_back

SETTINGS = NicheSettings()
UPM = 1.0
SPAN = (0.0, 3.0)


def _front():
    return SimpleNamespace(
        unit=np.array([1.0, 0.0]), normal=np.array([0.0, 1.0]), offset=0.0,
    )


def _points(depths, count=60):
    rng = np.random.default_rng(0)
    along = np.linspace(0.2, 2.8, count)
    across = rng.choice(np.asarray(depths, dtype=float), size=count)
    return np.column_stack([along, across, np.zeros(count)])


def _call(points):
    return _wall_point_back(points, _front(), 1.0, SPAN, 0.1, 1.0, UPM, SETTINGS)


def test_single_bin_does_not_crash():
    result = _call(_points([0.30]))
    assert result is None or isinstance(result[0], float)


def test_two_bins_do_not_crash():
    result = _call(_points([0.30, 0.34]))
    assert result is None or isinstance(result[0], float)


def test_normal_multi_bin_wall_line():
    rng = np.random.default_rng(1)
    along = np.linspace(0.2, 2.8, 120)
    across = 0.5 + rng.uniform(-0.01, 0.01, size=120)
    result = _call(np.column_stack([along, across, np.zeros(120)]))
    assert result is not None
    assert abs(result[0] - 0.5) < 0.02
    assert result[1] > 100
