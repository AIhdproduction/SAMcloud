"""Synthetic coverage for the shared COLMAP depth-edge prefilter."""

from __future__ import annotations

import numpy as np

from samcloud.colmap.depth_edges import geometric_valid_mask
from samcloud.colmap.plane_fill import View


def test_depth_edge_filter_drops_flying_pixels() -> None:
    view = View(0, "edge.png", np.eye(3), np.zeros(3), 80.0, 80.0, 50.0, 50.0, 100, 100)
    depth = np.full((100, 100), 3.0, np.float32)
    depth[:, 50:] = 5.0
    depth[20, 20] = 4.0
    before = depth.copy()

    keep = geometric_valid_mask(depth, view, {"depth_edge_filter": {"enabled": True, "max_incidence_deg": 89,
                                 "edge_px": 2, "edge_jump_rel": 0.05, "edge_jump_m": 0.03}})

    assert not keep[:, 48:52].any()
    assert not keep[20, 20]
    assert keep[10:18, 10:18].mean() > 0.995
    assert np.array_equal(depth, before)


def _plane_depth(view: View, normal: np.ndarray, size: int = 100) -> np.ndarray:
    """Depth of the plane n.X = 3 * n_z seen through ``view`` (dense)."""
    cols, rows = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    rays = np.stack(((cols - view.cx) / view.fx, (rows - view.cy) / view.fy, np.ones_like(cols)), -1)
    return (3.0 * normal[2] / (rays @ normal)).astype(np.float32)


def test_incidence_ignores_hole_neighbours() -> None:
    view = View(0, "flat.png", np.eye(3), np.zeros(3), 80.0, 80.0, 50.0, 50.0, 100, 100)
    depth = np.full((100, 100), 3.0, np.float32)
    rng = np.random.default_rng(0)
    depth[rng.random(depth.shape) < 0.85] = 0.0
    before = depth.copy()
    valid = depth > 0
    settings = {"depth_edge_filter": {"enabled": True, "max_incidence_deg": 80}}

    keep = geometric_valid_mask(depth, view, settings)

    assert 1.0 - keep[valid].mean() < 0.01
    assert np.array_equal(depth, before)


def test_grazing_plane_dropped() -> None:
    view = View(0, "graze.png", np.eye(3), np.zeros(3), 2000.0, 2000.0, 50.0, 50.0, 100, 100)
    angle = np.radians(85.0)
    depth = _plane_depth(view, np.array([0.0, np.sin(angle), np.cos(angle)]))
    assert (depth > 0).all()
    settings = {"depth_edge_filter": {"enabled": True, "max_incidence_deg": 80, "edge_jump_rel": 10.0,
                                      "edge_jump_m": 1e9}}

    keep = geometric_valid_mask(depth, view, settings)

    assert 1.0 - keep.mean() >= 0.9
