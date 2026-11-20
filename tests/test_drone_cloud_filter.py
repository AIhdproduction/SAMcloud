"""Tests for samcloud.drone.cloud_filter (synthetic, noisy, seeded clouds)."""
from __future__ import annotations

import numpy as np

from samcloud.drone.cloud_filter import (
    filter_drone_cloud,
    project_planar,
    remove_outliers,
    smooth_ground,
)
from samcloud.drone.contracts import (
    BUILDING,
    CURB,
    LOW_VEGETATION,
    PointSet,
    ROAD,
    ROOF_INCLINED,
    SIDEWALK,
    TREE,
)

CFG: dict = {}  # defaults; avoids reading the repo config


def _ps(xyz: np.ndarray, classes: np.ndarray) -> PointSet:
    return PointSet(xyz=np.asarray(xyz, dtype=np.float64), classes=np.asarray(classes, dtype=np.uint16))


def _roof(rng, n=14000, noise=0.01, ridge=False):
    x = rng.uniform(0.013, 4.017, n)
    y = rng.uniform(0.007, 4.003, n)
    slope = np.tan(np.radians(35))
    z = slope * (2.0 - np.abs(x - 2.0)) if ridge else 0.3 * x + 0.2 * y
    z = z + rng.normal(0, noise, n)
    return np.column_stack([x, y, z])


def _road_curb(rng, noise=0.003):
    n = 6000
    road = np.column_stack([rng.uniform(-3, -0.003, n), rng.uniform(0, 3, n), rng.normal(0, noise, n)])
    side = np.column_stack([rng.uniform(0.003, 3, n), rng.uniform(0, 3, n), 0.12 + rng.normal(0, noise, n)])
    m = 600
    curb = np.column_stack([rng.normal(0, 0.003, m), rng.uniform(0, 3, m), rng.uniform(0, 0.12, m)])
    xyz = np.vstack([road, side, curb])
    cls = np.concatenate([np.full(n, ROAD), np.full(n, SIDEWALK), np.full(m, CURB)])
    return _ps(xyz, cls)


def test_outliers_far_from_plane_removed_inliers_kept():
    rng = np.random.default_rng(1)
    plane = _roof(rng, n=4000, noise=0.005)
    out = np.column_stack([rng.uniform(0.3, 3.7, 30), rng.uniform(0.3, 3.7, 30), rng.uniform(2.0, 5.0, 30)])
    ps = _ps(np.vstack([plane, out]), np.full(4030, BUILDING))
    res, rep = remove_outliers(ps)
    assert len(res) == 4000
    assert res.xyz[:, 2].max() < 3.0
    assert rep[BUILDING]["removed"] == 30
    assert rep[BUILDING]["removed_by_reason"]["statistical"] == 30


def test_noisy_roof_moves_bounded_and_ridge_untouched():
    rng = np.random.default_rng(2)
    flat = _ps(_roof(rng), np.full(14000, ROOF_INCLINED))
    res, rep = project_planar(flat, [ROOF_INCLINED])
    shift = np.linalg.norm(res.xyz - flat.xyz, axis=1)
    moved = res.adjusted
    assert moved.sum() > 0.3 * len(flat)
    assert np.all(shift[~moved] == 0.0)
    assert shift[moved].max() <= 0.02  # <= sigma_factor * rms (rms ~ 0.01)
    assert rep[ROOF_INCLINED]["moved"] == int(moved.sum())
    assert rep[ROOF_INCLINED]["max_shift_m"] <= 0.02

    ridge = _ps(_roof(rng, ridge=True, noise=0.005), np.full(14000, ROOF_INCLINED))
    res, _ = project_planar(ridge, [ROOF_INCLINED])
    at_edge = np.abs(ridge.xyz[:, 0] - 2.0) < 0.03
    assert at_edge.sum() > 100
    near_edge = np.abs(ridge.xyz[:, 0] - 2.0) < 0.06
    assert res.adjusted[near_edge].mean() < 0.1  # almost untouched out to ~half the fit radius
    assert not res.adjusted[at_edge].any()
    assert np.all(res.xyz[at_edge] == ridge.xyz[at_edge])
    assert res.adjusted.sum() > 0  # flat parts away from the ridge still get cleaned


def test_curb_step_preserved():
    rng = np.random.default_rng(3)
    ps = _road_curb(rng)
    res, _ = smooth_ground(ps, [ROAD, SIDEWALK, CURB])
    x = ps.xyz[:, 0]
    near = (np.abs(x) < 0.12) & (ps.classes != CURB)
    assert near.sum() > 100
    assert not res.adjusted[near].any()
    assert not res.adjusted[ps.classes == CURB].any()
    assert res.adjusted.sum() > 0.3 * len(ps)  # away from the curb it does smooth

    def band_z(lo, hi, cls):
        m = (res.xyz[:, 0] > lo) & (res.xyz[:, 0] < hi) & (res.classes == cls)
        return res.xyz[m, 2].mean()

    step = band_z(0.15, 0.4, SIDEWALK) - band_z(-0.4, -0.15, ROAD)
    assert abs(step - 0.12) < 0.01


def test_vegetation_never_moved():
    rng = np.random.default_rng(4)
    n = 5000
    veg = np.column_stack([rng.uniform(0.01, 4, n), rng.uniform(0.01, 4, n), rng.normal(0, 0.01, n)])
    ps = _ps(veg, np.where(rng.random(n) < 0.5, LOW_VEGETATION, TREE))
    for fn in (
        lambda p: project_planar(p, [LOW_VEGETATION, TREE]),
        lambda p: smooth_ground(p, [LOW_VEGETATION, TREE]),
        lambda p: filter_drone_cloud(p, CFG),
    ):
        res, rep = fn(ps)
        kept_xyz = res.xyz
        # survivors are exactly the original points (only outlier removal may drop some)
        assert not res.adjusted.any()
        assert np.isin(kept_xyz[:, 0], ps.xyz[:, 0]).all()
        assert all(r["moved"] == 0 for r in rep.values())


def test_outliers_only_mode_moves_nothing():
    rng = np.random.default_rng(5)
    roof = _roof(rng, n=6000)
    ps = _ps(roof, np.full(6000, BUILDING))
    res, rep = filter_drone_cloud(ps, CFG, mode="outliers_only")
    assert not res.adjusted.any()
    assert np.all(rep[BUILDING]["moved"] == 0)
    assert np.isin(res.xyz[:, 0], ps.xyz[:, 0]).all()
    full, rep_full = filter_drone_cloud(ps, CFG, mode="full")
    assert rep_full[BUILDING]["moved"] > 0


def test_report_consistent_input_equals_output_plus_removed():
    rng = np.random.default_rng(6)
    road = _road_curb(rng)
    roof = _ps(_roof(rng, n=5000), np.full(5000, BUILDING))
    out = _ps(np.column_stack([rng.uniform(0, 3, 20), rng.uniform(0, 3, 20), rng.uniform(4, 8, 20)]),
              np.full(20, BUILDING))
    ps = _ps(np.vstack([road.xyz, roof.xyz, out.xyz]),
             np.concatenate([road.classes, roof.classes, out.classes]))
    res, rep = filter_drone_cloud(ps, CFG)
    assert sum(r["input"] for r in rep.values()) == len(ps)
    for code, r in rep.items():
        left = int((res.classes == code).sum())
        assert r["input"] == left + r["removed"], code
        assert r["moved"] <= left
        assert r["median_shift_m"] <= r["max_shift_m"]
    assert rep[BUILDING]["removed"] >= 20
    assert int(res.adjusted.sum()) == sum(r["moved"] for r in rep.values())


def test_tiling_matches_untiled():
    rng = np.random.default_rng(7)
    road = _road_curb(rng)
    roof = _roof(rng, n=6000)
    roof[:, 0] += 5.0
    out = np.column_stack([rng.uniform(-3, 8, 25), rng.uniform(0, 4, 25), rng.uniform(3, 6, 25)])
    xyz = np.vstack([road.xyz, roof, out])
    cls = np.concatenate([road.classes, np.full(6000, ROOF_INCLINED), np.full(25, BUILDING)])
    ps = _ps(xyz, cls)
    cfg_small = {"cloud_filter": {"tile_size_m": 2.0, "halo_m": 1.0}}
    cfg_big = {"cloud_filter": {"tile_size_m": 1000.0, "halo_m": 1.0}}
    a, rep_a = filter_drone_cloud(ps, cfg_small, workers=1)
    b, rep_b = filter_drone_cloud(ps, cfg_big, workers=1)
    assert len(a) == len(b)
    assert np.abs(a.xyz - b.xyz).max() < 1e-3
    assert np.array_equal(a.adjusted, b.adjusted)
    assert rep_a == rep_b
    assert a.adjusted.sum() > 0
