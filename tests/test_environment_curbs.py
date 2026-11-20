"""Tests for samcloud.environment.curbs."""
import numpy as np

from samcloud.drone import contracts as c
from samcloud.environment import curbs as cb

STEP_H = 0.12


def _curb_points(rng, path_fn, length, n=6000, sigma_xy=0.01, sigma_z=0.006):
    """Curb points along a path: half on the vertical face, half on the 12 cm wide top."""
    t = rng.uniform(0, length, n)
    base = path_fn(t)
    nrm = np.gradient(base, axis=0) if False else None
    eps = 1e-3
    tang = (path_fn(t + eps) - path_fn(t - eps)) / (2 * eps)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    top = rng.random(n) < 0.5
    offset = np.where(top, rng.uniform(0, STEP_H, n), 0.0)
    xy = base + normal * offset[:, None] + rng.normal(0, sigma_xy, (n, 2))
    z = np.where(top, STEP_H, rng.uniform(0, STEP_H, n)) + rng.normal(0, sigma_z, n)
    return np.column_stack([xy, z]), base, normal


def _ps(xyz):
    return c.PointSet(xyz=xyz, classes=np.full(len(xyz), c.CURB, dtype=np.uint16))


def _line(t):
    return np.column_stack([t, 0.3 * t * 0 + 2.0])


def test_straight_curb_one_axis_top_edge_z():
    rng = np.random.default_rng(21)
    xyz, _, _ = _curb_points(rng, _line, 20.0, n=8000)
    axes = cb.curb_axes(_ps(xyz))
    assert len(axes) == 1
    pts = axes[0].points
    assert len(pts) <= 3
    assert abs(pts[:, 2] - STEP_H).max() < 0.01
    assert abs(pts[:, 0].min() - 0.0) < 0.3 and abs(pts[:, 0].max() - 20.0) < 0.3


R = 8.0


def _arc(t):
    ang = t / R
    return np.column_stack([R * np.sin(ang), R - R * np.cos(ang)])


def test_curved_curb_follows_arc():
    rng = np.random.default_rng(22)
    xyz, _, _ = _curb_points(rng, _arc, R * np.pi / 2, n=8000)
    # shift so that the radial offset of the top (inside the curve) is symmetric noise only
    axes = cb.curb_axes(_ps(xyz))
    assert len(axes) == 1
    pts = axes[0].points
    assert len(pts) >= 5
    # distance of the polyline (densified) to the circle through the axis with centre (0, R)
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        for u in np.linspace(0, 1, 20):
            dense.append(a[:2] * (1 - u) + b[:2] * u)
    dense = np.asarray(dense)
    radial = np.hypot(dense[:, 0], dense[:, 1] - R)
    # the top is offset towards +normal (centre side): mean radius = R - 0.5 * 0.12 * share
    ref = np.median(radial)
    assert abs(ref - R) < 0.08
    assert np.abs(radial - ref).max() < 0.02


def test_split_straight_and_arcs_line_arc_line():
    r = 8.0
    line1 = np.column_stack([np.arange(0, 10, 0.25), np.zeros(40)])
    ang = np.arange(0.25, r * np.pi / 2, 0.25) / r
    arc = np.column_stack([10 + r * np.sin(ang), r - r * np.cos(ang)])
    x_end, y_end = arc[-1]
    line2 = np.column_stack([np.full(40, x_end), y_end + np.arange(0.25, 10.25, 0.25)])
    pts = np.vstack([line1, arc, line2])
    rng = np.random.default_rng(23)
    pts = pts + rng.normal(0, 0.002, pts.shape)
    parts = cb.split_straight_and_arcs(pts)
    kinds = [k for k, _ in parts]
    assert kinds == ["straight", "arc", "straight"]
    arc_slice = parts[1][1]
    assert abs(arc_slice.start - 40) < 4 and abs(arc_slice.stop - (40 + len(arc))) < 6
    assert parts[0][1].stop - 1 == parts[1][1].start


def test_split_pure_straight_and_large_radius():
    x = np.arange(0, 20, 0.25)
    assert cb.split_straight_and_arcs(np.column_stack([x, 0 * x])) == [("straight", slice(0, len(x)))]
    ang = np.arange(0, 10, 0.25) / 200.0  # R = 200 m > arc_max_radius
    big = np.column_stack([200 * np.sin(ang), 200 - 200 * np.cos(ang)])
    assert [k for k, _ in cb.split_straight_and_arcs(big)] == ["straight"]


def test_two_separate_curbs_two_axes():
    rng = np.random.default_rng(24)
    a, _, _ = _curb_points(rng, _line, 15.0, n=5000)
    b, _, _ = _curb_points(rng, lambda t: np.column_stack([t, 0 * t + 12.0]), 15.0, n=5000)
    axes = cb.curb_axes(_ps(np.vstack([a, b])))
    assert len(axes) == 2
    assert sorted(round(float(ax.points[:, 1].mean())) for ax in axes) == [2, 12]


def test_t_junction_splits_and_small_cluster_dropped():
    rng = np.random.default_rng(25)
    main, _, _ = _curb_points(rng, _line, 20.0, n=8000)
    branch, _, _ = _curb_points(rng, lambda t: np.column_stack([0 * t + 10.0, t + 2.5]), 8.0, n=3500)
    stray = np.column_stack([rng.normal(50, 0.02, 10), rng.normal(50, 0.02, 10), np.zeros(10)])
    axes = cb.curb_axes(_ps(np.vstack([main, branch, stray])))
    assert len(axes) == 2


def test_douglas_peucker_3d():
    t = np.linspace(0, 10, 100)
    pts = np.column_stack([t, np.zeros(100), 0.5 * t])
    assert len(cb.douglas_peucker_3d(pts, 0.01)) == 2
    bent = np.vstack([pts, np.column_stack([10 + t, np.zeros(100), 5 + 0 * t])])
    assert len(cb.douglas_peucker_3d(bent, 0.01)) == 3
