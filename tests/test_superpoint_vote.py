"""Regression tests for the final indoor superpoint vote and suppression marker."""

import json
from pathlib import Path

import numpy as np

from samcloud.colmap.superpoint_vote import (
    DEFAULTS,
    resolve_relabel_class_ids,
    vote_classes,
    vote_classified_cloud,
)
from samcloud.classification.custom import build_custom_classes
from samcloud.engine.classify import map_suppressed_winners
from samcloud.engine.ply_io import read_classified_ply, write_classified_ply


def _noisy_planes() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two noisy walls with a coplanar coloured door, fine blob and label noise."""
    rng = np.random.default_rng(20261110)
    values = np.linspace(0.0, 2.0, 43)
    grid_x, grid_z = np.meshgrid(values, values, indexing="ij")
    door_region = (grid_x >= 0.75) & (grid_x <= 1.25) & (grid_z >= 0.10) & (grid_z <= 1.65)
    wall = np.column_stack((grid_x[~door_region], np.zeros(np.count_nonzero(~door_region)), grid_z[~door_region]))
    door = np.column_stack((grid_x[door_region], np.zeros(np.count_nonzero(door_region)), grid_z[door_region]))
    grid_y, grid_z2 = np.meshgrid(values, values, indexing="ij")
    other_wall = np.column_stack((np.zeros(grid_y.size), grid_y.ravel(), grid_z2.ravel()))
    xyz = np.vstack((wall, door, other_wall)).astype(np.float32)
    xyz += rng.normal(0.0, 0.0015, xyz.shape).astype(np.float32)
    rgb = np.empty((len(xyz), 3), dtype=np.uint8)
    rgb[:len(wall)] = (185, 185, 180)
    rgb[len(wall):len(wall) + len(door)] = (125, 80, 45)
    rgb[len(wall) + len(door):] = (180, 185, 185)
    expected = np.full(len(xyz), 2, dtype=np.uint8)
    expected[len(wall):len(wall) + len(door)] = 3
    noisy = expected.copy()
    wall_candidates = np.arange(len(wall))
    fine_blob = rng.choice(wall_candidates, size=50, replace=False)
    remaining = np.setdiff1d(wall_candidates, fine_blob)
    noise = rng.choice(remaining, size=45, replace=False)
    noisy[fine_blob] = 5
    noisy[noise] = 4
    return xyz, rgb, noisy, expected, noise


def test_vote_fixes_noise_but_keeps_coloured_door_fine_blob_and_scores(tmp_path: Path) -> None:
    xyz, rgb, noisy, expected, noise = _noisy_planes()
    ply = tmp_path / "classified.ply"
    scores = tmp_path / "classified_scores.npz"
    report_dir = tmp_path / "reports"
    write_classified_ply(ply, xyz, rgb, noisy)
    np.savez(
        scores, top1_class=noisy, top1_conf=np.ones(len(noisy), np.float16), top2_class=np.zeros(len(noisy), np.uint8),
        top2_conf=np.zeros(len(noisy), np.float16), view_count=np.ones(len(noisy), np.uint16),
        suppressed=np.zeros(len(noisy), dtype=bool),
    )
    names = ["construction material", "wall", "door", "furniture", "electrical"]
    report = vote_classified_cloud(ply, DEFAULTS, names, scores_path=scores, report_dir=report_dir)
    _xyz, _rgb, after = read_classified_ply(ply)

    assert np.count_nonzero(after[noise] == 2) >= 35
    assert np.all(after[expected == 3] == 3)
    assert np.all(after[noisy == 5] == 5)
    assert report["changed_points"]["count"] >= 35
    assert (report_dir / "superpoint_stats.json").is_file()
    with np.load(scores) as archive:
        np.testing.assert_array_equal(archive["top1_class"], after)


def test_suppressed_points_dominate_and_appended_own_class_is_untouched() -> None:
    classes = np.array([1] * 15 + [0] * 35 + [2] * 5, dtype=np.uint8)
    suppressed = np.array([False] * 15 + [True] * 35 + [False] * 5)
    segments = np.array([0] * 50 + [1] * 5, dtype=np.int32)
    relabel = resolve_relabel_class_ids(["wall", "own class"], ["unknown", "wall"])

    after = vote_classes(classes, segments, np.array([50, 5]), 30, 0.6, relabel, 200, suppressed)

    assert np.all(after[:50] == 0)
    assert np.all(after[50:] == 2)
    assert relabel == {0, 1}


def test_clothes_protected_from_superpoint_vote() -> None:
    """Clothing keeps its Loose clutter class (LAS 180) inside furniture (LAS 117)."""
    clutter_id, furniture_id = 1, 17
    classes = np.array([clutter_id] * 30 + [furniture_id] * 70, dtype=np.uint8)
    protected_clutter = np.zeros(100, dtype=bool)
    protected_clutter[:30] = True
    segments = np.zeros(100, dtype=np.int32)

    after = vote_classes(
        classes, segments, np.array([100]), 30, 0.6, {furniture_id}, 200,
        protected_clutter=protected_clutter,
    )

    assert np.all(after[:30] == clutter_id)
    assert np.all(after[30:] == furniture_id)
    assert after[30] == furniture_id  # A cushion query is an ordinary furniture winner.
    assert [180, 117] == [
        json.loads(Path("config/classes.json").read_text(encoding="utf-8"))["las_codes"]["indoor"][clutter_id],
        json.loads(Path("config/classes.json").read_text(encoding="utf-8"))["las_codes"]["indoor"][furniture_id],
    ]


def test_suppress_entries_are_internal_and_score_marker_survives_filter(tmp_path: Path, monkeypatch) -> None:
    config = json.loads(Path("config/classes.json").read_text(encoding="utf-8"))
    visible = config["indoor"] + config["outdoor"] + config["labels_de"]["indoor"] + config["labels_de"]["outdoor"]
    assert all("curtain" not in str(entry).lower() for entry in visible)
    assert len(config["las_codes"]["indoor"]) == len(config["indoor"]) + 1
    assert len(config["colors"]["indoor"]) == len(config["indoor"]) + 1
    assert build_custom_classes("indoor", {"added": [{"name": "own class"}]})["suppress"] == config["suppress"]

    mapped, suppressed = map_suppressed_winners(np.array([1, 3, 2], dtype=np.uint8), {3})
    assert mapped.tolist() == [1, 0, 2]
    assert suppressed.tolist() == [False, True, False]

    from samcloud.colmap import point_cleaning

    dense = tmp_path / "dense"
    (dense / "stereo" / "depth_maps").mkdir(parents=True)
    ply, scores = tmp_path / "classified.ply", tmp_path / "scores.npz"
    write_classified_ply(ply, np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float32), None, np.array([1, 0], np.uint8))
    np.savez(scores, top1_class=np.array([1, 0], np.uint8), suppressed=np.array([False, True]))
    monkeypatch.setattr(point_cleaning, "free_space_counts", lambda *_args, **_kwargs: (np.array([1, 0]), np.zeros(2)))

    report = point_cleaning.clean_classified_cloud(dense, ply, {"min_free_views": 1}, scores_path=scores, views=[])

    assert report["scores_filtered"] is True
    with np.load(scores) as archive:
        assert archive["suppressed"].tolist() == [True]
