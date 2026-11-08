"""Depth-aware class votes: a SAM mask only labels points at the depth of its own surface."""

from __future__ import annotations

import numpy as np

from samcloud.engine.classify import depth_consistent_probs

WALL, SOFA = 1, 16  # 0-based class columns (wall = id 2, furniture = id 17)


def _scene():
    h, w = 120, 160
    depth = np.full((h, w), 3.0, np.float32)
    depth[:, :70] = 2.0 + np.linspace(0, 0.1, 70)[None, :]  # sofa in front, left part of the image
    labels = np.full((h, w), WALL + 1, np.uint8)
    labels[:, :76] = SOFA + 1  # the sofa mask spills 6 px over the wall behind
    return depth, labels


def _probs(n, winner, other=WALL):
    probs = np.zeros((n, 19), np.float32)
    probs[:, winner] = 0.8
    probs[:, other] = np.maximum(probs[:, other], 0.5)
    return probs


def test_spilled_mask_does_not_label_the_wall_behind() -> None:
    depth, labels = _scene()
    rng = np.random.default_rng(1)
    vi = rng.integers(10, 110, 300)
    ui = rng.integers(71, 76, 300)  # wall pixels under the spilled sofa mask
    z = np.full(300, 3.0) + rng.normal(0, 0.005, 300)
    probs = _probs(300, SOFA)
    stats = depth_consistent_probs(probs, ui, vi, z, labels, depth, 0.4)
    assert stats["class_removed"] >= 290
    assert (probs.argmax(axis=1) == WALL).mean() > 0.95  # the wall wins again


def test_real_sofa_and_wall_points_keep_their_class() -> None:
    depth, labels = _scene()
    rng = np.random.default_rng(2)
    ui = np.concatenate((rng.integers(5, 60, 200), rng.integers(90, 155, 200)))
    vi = rng.integers(5, 115, 400)
    z = depth[vi, ui].astype(float) + rng.normal(0, 0.004, 400)
    probs = np.vstack((_probs(200, SOFA), _probs(200, WALL, other=SOFA)))
    before = probs.copy()
    stats = depth_consistent_probs(probs, ui, vi, z, labels, depth, 0.4)
    assert stats["class_removed"] <= 2
    assert (probs.argmax(axis=1) == before.argmax(axis=1)).mean() > 0.99


def test_hidden_points_get_no_vote_and_unmeasured_pixels_are_not_judged() -> None:
    depth, labels = _scene()
    ui = np.array([30, 120])
    vi = np.array([60, 60])
    z = np.array([3.0, 3.0])  # first: behind the sofa (hidden in this view)
    probs = _probs(2, WALL)
    stats = depth_consistent_probs(probs, ui, vi, z, labels, depth, 0.4)
    assert stats["occluded"] == 1 and probs[0].max() == 0 and probs[1].argmax() == WALL
    empty = np.zeros_like(depth)  # textureless: no depth at all
    probs = _probs(2, SOFA)
    stats = depth_consistent_probs(probs, np.array([73, 73]), np.array([50, 51]), np.array([3.0, 3.0]), labels, empty, 0.4)
    assert stats["checked"] == 0 and (probs.argmax(axis=1) == SOFA).all()


def test_label_raster_edges_move_onto_the_depth_edge() -> None:
    from samcloud.engine.classify import refine_labels_with_depth

    depth, labels = _scene()
    refined, changed = refine_labels_with_depth(labels, depth, num_classes=20, device="cpu")
    assert (refined[:, 70:76] == WALL + 1).mean() > 0.95  # spilled sofa pixels on the wall: wall again
    assert (refined[:, 5:66] == SOFA + 1).all() and (refined[:, 80:] == WALL + 1).all()
    assert changed >= 0.9 * 6 * depth.shape[0]
    empty = np.zeros_like(depth)
    same, moved = refine_labels_with_depth(labels, empty, num_classes=20, device="cpu")
    assert moved == 0 and (same == labels).all()
