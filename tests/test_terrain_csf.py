"""Tests for the cloth simulation filter."""
from __future__ import annotations

import time

import numpy as np

from samcloud.terrain.csf import cloth_simulation_filter


def _flat_scene(rng: np.random.Generator):
    """Flat ground 60x60 m, a 12x10x7 m box building and tree blobs."""
    g = rng.uniform(0, 60, (60000, 2))
    g_z = rng.normal(0, 0.01, len(g))
    ground = np.column_stack([g, g_z])
    # building footprint x 20..32, y 25..35: no ground points under it (aerial survey)
    under = (g[:, 0] > 19.9) & (g[:, 0] < 32.1) & (g[:, 1] > 24.9) & (g[:, 1] < 35.1)
    ground = ground[~under]
    roof_xy = np.column_stack([rng.uniform(20.013, 31.987, 6000), rng.uniform(25.013, 34.987, 6000)])
    roof = np.column_stack([roof_xy, 7.0 + rng.normal(0, 0.01, 6000)])
    wall_t = rng.uniform(0.3, 6.9, 1500)
    wall_x = np.column_stack([np.full(1500, 20.0), rng.uniform(25, 35, 1500), wall_t])
    building = np.vstack([roof, wall_x])
    trees = []
    for cx, cy in [(10.0, 10.0), (45.0, 15.0), (48.0, 48.0), (8.0, 45.0)]:
        r = rng.uniform(0, 1.8, 900)
        a = rng.uniform(0, 2 * np.pi, 900)
        trees.append(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a), rng.uniform(3.5, 8.0, 900)]))
        trunk_z = rng.uniform(0.4, 3.5, 60)
        trees.append(np.column_stack([cx + rng.normal(0, 0.05, 60), cy + rng.normal(0, 0.05, 60), trunk_z]))
    trees = np.vstack(trees)
    return ground, building, trees


def test_building_and_trees_not_ground_flat_recall():
    rng = np.random.default_rng(1)
    ground, building, trees = _flat_scene(rng)
    xyz = np.vstack([ground, building, trees])
    mask = cloth_simulation_filter(xyz)
    n_g, n_b = len(ground), len(building)
    assert mask[:n_g].mean() > 0.95
    assert (~mask[n_g:n_g + n_b]).mean() > 0.95
    assert (~mask[n_g + n_b:]).mean() > 0.95


def test_embankment_30_degrees_preserved():
    rng = np.random.default_rng(2)
    n = 120000
    x = rng.uniform(0, 40, n)
    y = rng.uniform(0, 30, n)
    t = np.tan(np.deg2rad(30))
    z = np.where(x < 10, 0.0, np.where(x < 22, (x - 10) * t, 12 * t))
    xyz = np.column_stack([x, y, z + rng.normal(0, 0.01, n)])
    slope = (x > 10.5) & (x < 21.5)
    # a building on the plateau so the cloth has something to bridge
    bx = rng.uniform(28, 36, 3000)
    by = rng.uniform(10, 20, 3000)
    bld = np.column_stack([bx, by, 12 * t + 6 + rng.normal(0, 0.01, 3000)])
    mask = cloth_simulation_filter(np.vstack([xyz, bld]))
    assert mask[:n][slope].mean() > 0.90
    assert mask[:n].mean() > 0.95
    assert (~mask[n:]).mean() > 0.95


def test_curb_step_12cm_both_sides_ground():
    rng = np.random.default_rng(3)
    n = 100000
    x = rng.uniform(0, 30, n)
    y = rng.uniform(0, 30, n)
    z = np.where(x < 15.013, 0.0, 0.12) + rng.normal(0, 0.008, n)
    xyz = np.column_stack([x, y, z])
    mask = cloth_simulation_filter(xyz)
    road = x < 14.7
    side = x > 15.3
    assert mask[road].mean() > 0.97
    assert mask[side].mean() > 0.97


def test_two_million_points_under_30_seconds():
    rng = np.random.default_rng(4)
    n = 2_000_000
    x = rng.uniform(0, 200, n)
    y = rng.uniform(0, 200, n)
    z = 0.02 * x + rng.normal(0, 0.02, n)
    lift = (x > 80.01) & (x < 120.01) & (y > 80.01) & (y < 100.01)
    z[lift] += 6.0
    xyz = np.column_stack([x, y, z])
    start = time.perf_counter()
    mask = cloth_simulation_filter(xyz)
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0
    assert mask[~lift].mean() > 0.95
    assert (~mask[lift]).mean() > 0.9
