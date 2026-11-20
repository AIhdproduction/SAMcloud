"""Tests for samcloud.environment.vegetation on synthetic trees and shrubs."""
from __future__ import annotations

import numpy as np

from samcloud.drone.contracts import MEDIUM_VEGETATION, SHRUB, TREE, TREE_TRUNK, PointSet
from samcloud.environment.vegetation import (
    canopy_height_model,
    crown_detail_mesh,
    detect_shrubs,
    detect_trees,
    vegetation_qa,
)


def dtm(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return 0.02 * xy[:, 0] + 0.01 * xy[:, 1]


def make_tree(rng, cx, cy, *, crown_r=2.0, top=7.5, crown_c=2.5, n_crown=2500, n_trunk=1500,
              trunk_r=0.15, trunk_top=2.8):
    """Return (xyz, classes) of a tree: trunk cylinder + ellipsoid crown surface, 2 cm noise."""
    ground = float(dtm(np.array([[cx, cy]]))[0])
    chunks, classes = [], []
    if n_trunk > 0:
        ang = rng.uniform(0, 2 * np.pi, n_trunk)
        z = rng.uniform(0.0, trunk_top, n_trunk)
        pts = np.column_stack([cx + trunk_r * np.cos(ang), cy + trunk_r * np.sin(ang), ground + z])
        chunks.append(pts + rng.normal(0, 0.02, pts.shape))
        classes.append(np.full(n_trunk, TREE_TRUNK))
    v = rng.normal(size=(n_crown, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    centre_z = ground + top - crown_c
    pts = np.column_stack([cx + crown_r * v[:, 0], cy + crown_r * v[:, 1], centre_z + crown_c * v[:, 2]])
    chunks.append(pts + rng.normal(0, 0.02, pts.shape))
    classes.append(np.full(n_crown, TREE))
    return np.vstack(chunks), np.concatenate(classes)


def build(parts) -> PointSet:
    xyz = np.vstack([p[0] for p in parts])
    cls = np.concatenate([p[1] for p in parts]).astype(np.uint16)
    return PointSet(xyz=xyz, classes=cls)


def test_two_separated_trees_measured():
    rng = np.random.default_rng(1)
    ps = build([make_tree(rng, 0.0, 0.0), make_tree(rng, 12.0, 3.0)])
    trees = detect_trees(ps, dtm)
    assert len(trees) == 2
    assert all(t.status == "ok" for t in trees)
    for t in trees:
        assert abs(t.height_m - 7.5) < 0.2
        assert abs(t.crown_diameter_m - 4.0) < 0.6
        assert t.trunk_diameter_m is not None
        assert abs(t.trunk_diameter_m - 0.30) < 0.03
        assert t.evidence.notes["trunk_xy"] == "trunk_points"
        assert t.evidence.points > 200


def test_touching_crowns_split():
    rng = np.random.default_rng(2)
    ps = build([make_tree(rng, 0.0, 0.0, top=7.5), make_tree(rng, 3.6, 0.4, top=6.8)])
    trees = detect_trees(ps, dtm)
    assert len(trees) == 2
    heights = sorted(t.height_m for t in trees)
    assert abs(heights[0] - 6.8) < 0.3 and abs(heights[1] - 7.5) < 0.3
    owned = [set(t.point_indices.tolist()) for t in trees]
    assert not owned[0] & owned[1]


def test_tree_without_trunk_points():
    rng = np.random.default_rng(3)
    ps = build([make_tree(rng, 0.0, 0.0, n_trunk=0)])
    trees = detect_trees(ps, dtm)
    assert len(trees) == 1
    t = trees[0]
    assert t.trunk_diameter_m is None
    assert t.evidence.notes["trunk_xy"] == "crown_peak"
    assert np.hypot(t.trunk_xy[0], t.trunk_xy[1]) < 1.0


def test_small_bush_classified_as_tree_is_rejected():
    rng = np.random.default_rng(4)
    bush = make_tree(rng, 0.0, 0.0, crown_r=0.9, top=1.5, crown_c=0.75, n_crown=800, n_trunk=0)
    trees = detect_trees(build([bush]), dtm)
    assert len(trees) == 1
    assert trees[0].status == "rejected"
    assert "height" in trees[0].reason
    qa = vegetation_qa(trees, [])
    assert len(qa) == 1 and qa[0].category == "tree"


def test_shrub_cluster_outline_and_height():
    rng = np.random.default_rng(5)
    n = 500
    r = 0.8 * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    x, y = 5.0 + r * np.cos(a), 2.0 + r * np.sin(a)
    h = rng.uniform(0.2, 1.0, n)
    xyz = np.column_stack([x, y, dtm(np.column_stack([x, y])) + h + rng.normal(0, 0.02, n)])
    ps = PointSet(xyz=xyz, classes=np.full(n, SHRUB, dtype=np.uint16))
    shrubs = detect_shrubs(ps, dtm)
    assert len(shrubs) == 1
    s = shrubs[0]
    assert s.status == "ok"
    true_area = np.pi * 0.8 ** 2
    assert abs(s.outline.area - true_area) / true_area < 0.20
    assert abs(s.height_m - np.percentile(h, 95)) < 0.15


def test_medium_vegetation_under_tree_not_a_shrub():
    rng = np.random.default_rng(6)
    xyz_t, cls_t = make_tree(rng, 0.0, 0.0)
    n = 300
    x, y = rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n)
    mv = np.column_stack([x, y, dtm(np.column_stack([x, y])) + rng.uniform(0.5, 2.0, n)])
    ps = PointSet(xyz=np.vstack([xyz_t, mv]),
                  classes=np.concatenate([cls_t, np.full(n, MEDIUM_VEGETATION)]).astype(np.uint16))
    assert detect_shrubs(ps, dtm) == []


def test_crown_detail_mesh_uses_input_points():
    rng = np.random.default_rng(7)
    xyz, cls = make_tree(rng, 0.0, 0.0, n_trunk=0)
    pts = xyz[cls == TREE]
    verts, faces = crown_detail_mesh(pts)
    assert len(faces) > 0
    assert faces.max() < len(verts)
    pts_set = {tuple(p) for p in pts.tolist()}
    assert all(tuple(v) in pts_set for v in verts.tolist())


def test_nothing_invented_without_support():
    rng = np.random.default_rng(8)
    # 10 trunk points (< 20) and a tree without any: no trunk diameter, no trunk position from trunk points
    parts = [make_tree(rng, 0.0, 0.0, n_trunk=10), make_tree(rng, 12.0, 0.0, n_trunk=0)]
    trees = detect_trees(build(parts), dtm)
    assert len(trees) == 2
    for t in trees:
        assert t.trunk_diameter_m is None
        assert t.evidence.notes["trunk_xy"] == "crown_peak"
        assert t.evidence.rmse is None


def test_canopy_height_model_values():
    rng = np.random.default_rng(9)
    ps = build([make_tree(rng, 0.0, 0.0)])
    chm, grid = canopy_height_model(ps, dtm)
    assert chm.shape == (grid.height, grid.width)
    assert abs(np.nanmax(chm) - 7.5) < 0.3
    empty, g = canopy_height_model(PointSet(xyz=np.zeros((0, 3)), classes=np.zeros(0, np.uint16)), dtm)
    assert g is None
