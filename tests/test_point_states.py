"""Point-state LAS export behaviour."""

from pathlib import Path

import json
import laspy
import numpy as np

from samcloud.engine.export_las import PLY_DTYPE, export_las
from samcloud.engine.classify import resolve_weighted_votes


def _ply(path: Path, classes: list[int]) -> None:
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex " + str(len(classes)) + "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar scalar_class\nend_header\n").encode()
    data = np.zeros(len(classes), dtype=PLY_DTYPE)
    data["scalar_class"] = classes
    path.write_bytes(header + data.tobytes())


def test_export_distinguishes_unseen_and_unclassified_and_writes_provenance(tmp_path: Path) -> None:
    source, output = tmp_path / "classified.ply", tmp_path / "classified.las"
    _ply(source, [255, 0, 1])
    export_las(source, output, Path("config/classes.json"), 10, "LOCAL", "LOCAL", "outdoor")
    assert laspy.read(output).classification.tolist() == [0, 1, 2]
    assert output.with_suffix(".las.profile.json").is_file()


def test_old_ply_unknown_remains_unclassified(tmp_path: Path) -> None:
    source, output = tmp_path / "old.ply", tmp_path / "old.las"
    _ply(source, [0])
    export_las(source, output, Path("config/classes.json"), 10, "LOCAL", "LOCAL", "outdoor")
    assert int(laspy.read(output).classification[0]) == 1


def test_seen_counts_keep_vote_count_semantics_and_encode_only_unseen() -> None:
    scores = np.array([[0.0, 0.0], [0.4, 0.1], [0.9, 0.1]], dtype=np.float32)
    votes = np.array([0, 1, 1], dtype=np.uint16)
    seen = np.array([0, 3, 3], dtype=np.uint16)
    assert resolve_weighted_votes(scores, votes, 2, 0, seen).tolist() == [255, 0, 0]


def test_flurseg_profile_exports_model_ids_without_sam_aliases(tmp_path: Path) -> None:
    source, output = tmp_path / "flurseg.ply", tmp_path / "flurseg.las"
    _ply(source, [1, 3, 10])
    export_las(source, output, Path("config/classes.json"), 10, "LOCAL", "LOCAL", "outdoor", engine="flurseg")
    assert laspy.read(output).classification.tolist() == [64, 71, 82]


def test_one_confident_view_of_three_stays_unknown_with_two_minimum_votes() -> None:
    scores = np.array([[0.0, 0.0], [0.8, 0.1]], dtype=np.float32)
    confident_votes = np.array([1, 1], dtype=np.uint16)
    seen = np.array([3, 3], dtype=np.uint16)

    resolved = resolve_weighted_votes(scores, confident_votes, 2, 0, seen)

    assert resolved.tolist() == [0, 0]


def test_votes_resolve_and_export_all_point_states(tmp_path: Path) -> None:
    scores = np.array([[0.0, 0.0], [0.8, 0.1], [1.2, 0.1]], dtype=np.float32)
    confident_votes = np.array([0, 1, 3], dtype=np.uint16)
    seen = np.array([0, 3, 3], dtype=np.uint16)
    classes = resolve_weighted_votes(scores, confident_votes, 2, 0, seen)
    source, output = tmp_path / "votes.ply", tmp_path / "votes.las"

    _ply(source, classes.tolist())
    export_las(source, output, Path("config/classes.json"), 10, "LOCAL", "LOCAL", "outdoor")

    assert classes.tolist() == [255, 0, 1]
    assert laspy.read(output).classification.tolist() == [0, 1, 2]


def test_class_coherence_preserves_never_classified_points_for_las_export(tmp_path: Path) -> None:
    from samcloud.colmap.surface_fusion import class_coherence

    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    labels = np.array([255, 1, 2], dtype=np.uint8)
    settings = {
        "class_coherence_classes": ["unknown", "ground", "tree"], "class_cell_m": 1.0,
        "class_window_cells": 1, "class_min_share": 0.9, "class_majority_share": 1.0,
        "class_min_points": 0,
    }

    coherent, keep, _stats = class_coherence(points, labels, {"ground": 1, "tree": 2}, settings)
    source, output = tmp_path / "coherent.ply", tmp_path / "coherent.las"
    _ply(source, coherent.tolist())
    export_las(source, output, Path("config/classes.json"), 10, "LOCAL", "LOCAL", "outdoor")

    assert keep.all()
    assert coherent[0] == 255
    assert int(laspy.read(output).classification[0]) == 0


def test_point_cleaning_and_depth_fill_keep_existing_never_classified_rows(tmp_path: Path, monkeypatch) -> None:
    from samcloud.colmap.depth_fill import fill_classified_from_prior
    from samcloud.colmap.point_cleaning import clean_classified_cloud
    from samcloud.engine.ply_io import read_classified_ply, write_classified_ply
    from test_plane_fill import WALL_Y, _cleaning_workspace

    dense = tmp_path / "dense"
    views = _cleaning_workspace(dense)
    classified = tmp_path / "classified.ply"
    points = np.array([[0.0, WALL_Y, 1.2], [0.0, 2.0, 1.2]], dtype=np.float32)
    write_classified_ply(classified, points, None, np.array([255, 1], dtype=np.uint8))

    report = clean_classified_cloud(dense, classified, {"min_free_views": 3}, views=views)
    _xyz, _rgb, cleaned = read_classified_ply(classified)

    assert report["removed_points"] == 1
    assert cleaned.tolist() == [255]

    monkeypatch.setattr(
        "samcloud.colmap.depth_fill.fill_from_depth_prior",
        lambda *args, **kwargs: (
            np.array([[2.0, 2.0, 2.0]]), np.array([[10, 20, 30]], dtype=np.uint8),
            np.array([1], dtype=np.uint8), np.array([2], dtype=np.int32), {"added_points": 1},
        ),
    )
    fill_classified_from_prior(dense, classified, views, lambda _name: None)
    _xyz, _rgb, filled = read_classified_ply(classified)

    assert filled.tolist() == [255, 1]
