"""Synthetic regression coverage for SAM mirror filtering and cached reclassification."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from samcloud.colmap.mirror_filter import Mirror, MirrorCandidate, detect_mirrors, mirror_candidate, mirror_mask_reader, mirror_report, phantom_mask
from samcloud.colmap.plane_fill import View, write_colmap_array
from samcloud.engine import classify, pipeline


def _view(index: int = 0, centre: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> View:
    center = np.asarray(centre, dtype=float)
    return View(index, f"view_{index}.png", np.eye(3), -center, 50.0, 50.0, 50.0, 50.0, 100, 100)


def test_mirror_plane_from_frame_ring() -> None:
    rng = np.random.default_rng(20261111)
    view = _view()
    rows, _cols = np.mgrid[:100, :100]
    depth = np.zeros((100, 100), dtype=np.float32)
    valid = rows > 50
    depth[valid] = 50.0 / (rows[valid] + 0.5 - 50.0)
    depth += rng.normal(0.0, 0.002, depth.shape).astype(np.float32)
    mask = np.zeros((100, 100), dtype=bool)
    mask[65:85, 35:65] = True
    depth[mask] = 5.5

    candidate = mirror_candidate(view, depth, mask, {})

    assert candidate is not None
    assert abs(float(candidate.normal @ np.array([0.0, 1.0, 0.0]))) >= np.cos(np.deg2rad(2.0))
    assert abs(float(candidate.normal @ np.array([0.0, 1.0, 0.0]) + candidate.offset)) <= 0.01


def _write_support_depths(stereo_dir: Path, view: View, points: np.ndarray) -> None:
    depth = np.zeros((100, 100), dtype=np.float32)
    local = points @ view.rotation.T + view.translation
    u = ((view.fx * local[:, 0] / local[:, 2] + view.cx) * depth.shape[1] / view.width).astype(int)
    v = ((view.fy * local[:, 1] / local[:, 2] + view.cy) * depth.shape[0] / view.height).astype(int)
    for x, y, z in zip(u, v, local[:, 2]):
        if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
            depth[y, x] = z
    write_colmap_array(stereo_dir / "depth_maps" / f"{view.name}.geometric.bin", depth)


def _write_prior_depths(stereo_dir: Path, view: View, points: np.ndarray, depths: np.ndarray | None = None) -> None:
    """Write smooth aligned-prior samples at the projected point pixels."""
    depth = np.zeros((100, 100), dtype=np.float32)
    local = points @ view.rotation.T + view.translation
    u = ((view.fx * local[:, 0] / local[:, 2] + view.cx) * depth.shape[1] / view.width).astype(int)
    v = ((view.fy * local[:, 1] / local[:, 2] + view.cy) * depth.shape[0] / view.height).astype(int)
    values = local[:, 2] if depths is None else np.asarray(depths, dtype=np.float32)
    for x, y, z in zip(u, v, values):
        if 2 <= x < depth.shape[1] - 2 and 2 <= y < depth.shape[0] - 2:
            depth[y - 2:y + 3, x - 2:x + 3] = z
    prior_dir = stereo_dir.parent / "depth_prior"
    prior_dir.mkdir(parents=True, exist_ok=True)
    np.save(prior_dir / f"{view.name}.npy", depth)


def _write_mirror_masks(root: Path, views: list[View], mask: np.ndarray | None = None) -> Path:
    masks = root / "sam_mirror"
    masks.mkdir(parents=True, exist_ok=True)
    image = np.zeros((100, 100), np.uint8) if mask is None else mask.astype(np.uint8) * 255
    if mask is None:
        image[30:70, 30:70] = 255
    for view in views:
        assert cv2.imwrite(str(masks / f"{view.name}.png"), image)
    return masks


def test_mirror_phantom_points_removed(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    views = [_view(index, (float(index - 1), 0.0, 0.0)) for index in range(3)]
    phantom = np.array([0.0, 0.0, 5.0])
    real_front = np.array([0.0, 0.0, 2.0])
    direct_through_door = np.array([4.0, 0.0, 5.0])
    points = np.vstack((phantom, real_front, direct_through_door))
    for view in views:
        _write_support_depths(stereo, view, points)
    _write_mirror_masks(tmp_path, views)
    mirror = Mirror(
        normal=np.array([0.0, 0.0, -1.0]), offset=3.0, center=np.array([0.0, 0.0, 3.0]),
        polygons=[np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])],
        views=views[:2],
    )

    removed = phantom_mask(points, [mirror], views, stereo, {"through_share": 0.8})

    assert removed.tolist() == [True, False, False]


def test_phantom_mask_removes_unsupported_points(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    views = [_view(index, (float(index - 1) * 0.5, 0.0, 0.0)) for index in range(3)]
    for view in views:
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", np.zeros((100, 100), np.float32))
        _write_prior_depths(stereo, view, np.array([[0.0, 0.0, 5.0]]))
    _write_mirror_masks(tmp_path, views)
    mirror = Mirror(np.array([0.0, 0.0, -1.0]), 3.0, np.array([0.0, 0.0, 3.0]),
                    [np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])], views)

    removed = phantom_mask(np.array([[0.0, 0.0, 5.0], [3.0, 0.0, 5.0]]), [mirror], views, stereo, {})

    assert removed.tolist() == [True, False]


def test_mirror_union_area() -> None:
    mirror = Mirror(np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(3), [
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]),
        np.array([[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0]]),
        np.array([[0.5, 1.0], [2.5, 1.0], [2.5, 3.0], [0.5, 3.0]]),
    ], [])

    assert mirror.area == pytest.approx(8.0, rel=0.05)


def _bow_tie_mirror() -> Mirror:
    return Mirror(np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(3), [
        np.array([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
    ], [])


def test_mirror_area_bow_tie_polygon_is_finite() -> None:
    area = _bow_tie_mirror().area

    assert np.isfinite(area) and area > 0.0


def test_mirror_area_degenerate_polygons_do_not_raise() -> None:
    mirror = Mirror(np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(3), [
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [np.nan, 1.0], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]),
    ], [])

    assert mirror.area == pytest.approx(4.0)
    assert Mirror(np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(3), [np.zeros((2, 2))], []).area == 0.0


def test_mirror_report_with_self_intersecting_mirror() -> None:
    from samcloud.colmap.mirror_filter import mirror_report

    report = mirror_report([_bow_tie_mirror()], np.zeros(5, dtype=bool))

    assert np.isfinite(report["mirrors"][0]["area"]) and report["mirrors"][0]["area"] > 0.0
    assert report["mirror_points_removed"] == 0


def test_guarded_mirror_report_survives_failure(capsys: pytest.CaptureFixture[str]) -> None:
    from samcloud.colmap.surface_fusion import _guarded_mirror_report

    def broken(mirrors, removed):
        raise RuntimeError("TopologyException: side location conflict")

    report = _guarded_mirror_report(broken, [_bow_tie_mirror()], np.array([True, True, False]))

    assert report["mirror_points_removed"] == 2
    assert "side location conflict" in report["mirror_report_error"]
    assert "WARNING: Mirror report failed" in capsys.readouterr().out


def test_adjacent_frames_do_not_confirm_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stereo = tmp_path / "stereo" / "depth_maps"
    masks = tmp_path / "masks"
    stereo.mkdir(parents=True); masks.mkdir()

    def run(baseline: float) -> list[Mirror]:
        views = [_view(0), _view(1, (baseline, 0.0, 0.0))]
        x, z = np.meshgrid(np.linspace(-1.0, 1.0, 8), np.linspace(0.1, 1.5, 8))
        points = np.column_stack((x.ravel(), np.ones(x.size), z.ravel()))
        for view in views:
            write_colmap_array(stereo / f"{view.name}.geometric.bin", np.ones((10, 10), np.float32))
        monkeypatch.setattr("samcloud.colmap.mirror_filter.read_mirror_mask", lambda *_args: np.ones((10, 10), bool))
        monkeypatch.setattr("samcloud.colmap.mirror_filter.mirror_candidate", lambda view, *_args: MirrorCandidate(
            view, np.array([0.0, 1.0, 0.0]), -1.0, np.array([0.0, 1.0, 0.8]), np.ones((10, 10), bool),
            np.array([[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.6], [-1.0, 1.0, 1.6]]), points, np.ones(len(points))))
        return detect_mirrors(views, tmp_path / "stereo", masks, {"min_baseline_m": 0.3})

    assert not run(0.05)
    assert len(run(0.5)) == 1


def test_unsupported_point_seen_directly_is_kept(tmp_path: Path) -> None:
    scale = 0.8157
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    views = [_view(index, (x * scale, y * scale, 0.0)) for index, (x, y) in enumerate(((-0.31, 0.12), (0.17, -0.18), (0.42, 0.08), (-0.14, -0.33)))]
    phantom = np.array([0.0, 0.0, 5.0]) * scale
    direct_room = np.array([2.1, 0.0, 5.0]) * scale
    for view in views:
        _write_support_depths(stereo, view, np.asarray([direct_room]))
        _write_prior_depths(stereo, view, np.asarray([phantom]))
    _write_mirror_masks(tmp_path, views)
    mirror = Mirror(np.array([0.0, 0.0, -1.0]), 3.0 * scale, np.array([0.0, 0.0, 3.0]) * scale,
                    [np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * scale], views)

    removed = phantom_mask(np.vstack((phantom, direct_room)), [mirror], views, stereo,
                           {"units_per_metre": scale, "through_share": 0.8})

    assert removed.tolist() == [True, False]
    assert mirror.filter_stats["direct_kept"] == 0


def test_back_side_views_do_not_count(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    views = [_view(index, (0.2 * index, 0.0, 6.0)) for index in range(3)]
    for view in views:
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", np.ones((100, 100), np.float32) * 3.0)
    _write_mirror_masks(tmp_path, views, np.ones((100, 100), bool))
    mirror = Mirror(np.array([0.0, 0.0, -1.0]), 3.0, np.array([0.0, 0.0, 3.0]),
                    [np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])], views)

    removed = phantom_mask(np.array([[0.0, 0.0, 8.0]]), [mirror], views, stereo, {})

    assert not removed.any()
    assert mirror.filter_stats["through_points"] == 0


def test_phantom_mask_handles_rotated_scene(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    angle = np.deg2rad(37.0)
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
    base_centres = [np.array([x, 0.0, 0.0]) for x in (-0.7, -0.2, 0.35)]
    views = [View(index, f"view_{index}.png", rotation.T, -centre, 50.0, 50.0, 50.0, 50.0, 100, 100)
             for index, centre in enumerate(base_centres)]
    phantom = np.array([[0.0, 0.0, 5.0]]) @ rotation.T
    for view in views:
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", np.zeros((100, 100), np.float32))
        _write_prior_depths(stereo, view, phantom)
    _write_mirror_masks(tmp_path, views)
    normal = rotation @ np.array([0.0, 0.0, -1.0])
    mirror = Mirror(normal, 3.0, np.array([0.0, 0.0, 3.0]) @ rotation.T,
                    [np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])], views)

    removed = phantom_mask(phantom, [mirror], views, stereo, {})

    assert removed.tolist() == [True]


def test_phantom_mask_reads_each_depth_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    views = [_view(index, (float(index - 1) * 0.2, 0.0, 0.0)) for index in range(3)]
    for view in views:
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", np.ones((100, 100), np.float32) * 3.0)
        _write_prior_depths(stereo, view, np.array([[0.0, 0.0, 5.0]]))
    _write_mirror_masks(tmp_path, views, np.ones((100, 100), bool))
    mirrors = [Mirror(np.array([0.0, 0.0, -1.0]), 3.0, np.array([0.0, 0.0, 3.0]),
                      [np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])], views)
               for _ in range(3)]
    reader = __import__("samcloud.colmap.mirror_filter", fromlist=["read_colmap_array"]).read_colmap_array
    calls: list[Path] = []

    def counting_reader(path: Path) -> np.ndarray:
        calls.append(Path(path))
        return reader(path)

    monkeypatch.setattr("samcloud.colmap.mirror_filter.read_colmap_array", counting_reader)
    original_load = np.load
    prior_calls: list[Path] = []

    def counting_load(path: Path, *args, **kwargs):
        prior_calls.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr("samcloud.colmap.mirror_filter.np.load", counting_load)
    phantom_mask(np.array([[0.0, 0.0, 5.0]]), mirrors, views, stereo, {})

    assert len(calls) == len(views)
    assert len(prior_calls) == len(views)


def test_real_point_behind_mirror_wall_kept_without_direct_depth(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    views = [_view(index, (float(index - 1) * 0.3, 0.0, 0.0)) for index in range(3)]
    real_ceiling = np.array([[0.0, 0.0, 5.0]])
    for view in views:
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", np.zeros((100, 100), np.float32))
        _write_prior_depths(stereo, view, real_ceiling, np.array([4.0]))
    mirror = Mirror(np.array([0.0, 0.0, -1.0]), 3.0, np.array([0.0, 0.0, 3.0]),
                    [np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])], views)

    removed = phantom_mask(real_ceiling, [mirror], views, stereo, {})

    assert removed.tolist() == [False]
    assert mirror.filter_stats["through_points"] == 0


def test_da_only_phantom_matching_prior_removed(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    views = [_view(index, (float(index - 1) * 0.3, 0.0, 0.0)) for index in range(3)]
    phantom = np.array([[0.0, 0.0, 5.0]])
    for view in views:
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", np.zeros((100, 100), np.float32))
        _write_prior_depths(stereo, view, phantom)
    mirror = Mirror(np.array([0.0, 0.0, -1.0]), 3.0, np.array([0.0, 0.0, 3.0]),
                    [np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])], views)

    removed = phantom_mask(phantom, [mirror], views, stereo, {})

    assert removed.tolist() == [True]
    assert mirror.filter_stats["prior_through"] == 1


def test_adjacent_coplanar_mirrors_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stereo = tmp_path / "stereo" / "depth_maps"
    masks = tmp_path / "masks"
    stereo.mkdir(parents=True); masks.mkdir()
    views = [_view(index, (float(index), 0.0, 0.0)) for index in range(4)]
    for view in views:
        write_colmap_array(stereo / f"{view.name}.geometric.bin", np.ones((10, 10), np.float32))
    x, z = np.meshgrid(np.linspace(3.7, 4.4, 8), np.linspace(0.1, 1.5, 8))
    ring = np.column_stack((x.ravel(), np.ones(x.size), z.ravel()))

    def run(centres: list[float]) -> list[Mirror]:
        def candidate(view: View, *_args) -> MirrorCandidate:
            centre = np.array([centres[view.index], 1.0, 0.8])
            angle = np.deg2rad(2.0 if view.index % 2 else 0.0)
            normal = np.array([np.sin(angle), np.cos(angle), 0.0])
            outline = np.array([[centre[0] - 0.30, 1.0, 0.0], [centre[0] + 0.30, 1.0, 0.0],
                                [centre[0] + 0.30, 1.0, 1.6], [centre[0] - 0.30, 1.0, 1.6]])
            local_ring = ring + np.array([centre[0] - 4.0, 0.0, 0.0])
            return MirrorCandidate(view, normal, -float(normal @ centre), centre, np.ones((10, 10), bool),
                                   outline, local_ring, np.ones(len(local_ring)))
        monkeypatch.setattr("samcloud.colmap.mirror_filter.read_mirror_mask", lambda *_args: np.ones((10, 10), bool))
        monkeypatch.setattr("samcloud.colmap.mirror_filter.mirror_candidate", candidate)
        return detect_mirrors(views, tmp_path / "stereo", masks, {"min_baseline_m": 0.3})

    assert len(run([4.00, 4.05, 4.00, 4.05])) == 1
    assert len(run([4.00, 4.05, 5.50, 5.55])) == 2


def test_ring_plane_ignores_floor_pixels() -> None:
    view = _view()
    rows, cols = np.mgrid[:100, :100]
    depth = np.zeros((100, 100), np.float32)
    valid = rows > 50
    depth[valid] = 50.0 / (rows[valid] + 0.5 - 50.0)
    mask = np.zeros((100, 100), bool); mask[65:85, 35:65] = True
    depth[mask] = 5.5
    labels = np.full((100, 100), 2, np.uint8)
    labels[(cols < 50) & ~mask] = 4
    depth[(cols < 50) & ~mask] = 1.2

    candidate = mirror_candidate(view, depth, mask, {"ring_label_ids": [2, 5, 17]}, labels)

    assert candidate is not None
    assert abs(float(candidate.normal @ np.array([0.0, 1.0, 0.0]))) >= np.cos(np.deg2rad(2.0))


def test_mirror_detection_runs_before_depth_fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.colmap.depth_prior import availability, compute_depth_priors, prior_reader
    from samcloud.colmap.depth_fill import fill_classified_from_prior
    from samcloud.colmap.plane_fill import load_views
    from samcloud.colmap.surface_fusion import consolidate_classified_cloud

    calls: list[str] = []
    dense = tmp_path / "dense"
    (dense / "stereo" / "depth_maps").mkdir(parents=True)
    (dense / "sparse").mkdir()
    view = _view()
    args = SimpleNamespace(classes_json=str(Path(__file__).parents[1] / "config" / "classes.json"), class_set="indoor")
    summary = SimpleNamespace(settings={}, record_stage=lambda *_args: None)
    monkeypatch.setattr(pipeline, "snapshot_fine_terms", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_depth_prior_settings", lambda _args: {"model": "test"})
    monkeypatch.setattr(pipeline, "_depth_fill_settings", lambda _args: {"enabled": True})
    monkeypatch.setattr(pipeline, "_surface_fusion_settings", lambda _args: {"enabled": True, "mirror_filter": {"enabled": True}})
    monkeypatch.setattr(pipeline, "_cleaning_settings", lambda _args: None)
    monkeypatch.setattr(pipeline, "_plane_fill_settings", lambda _args: None)
    monkeypatch.setattr("samcloud.colmap.plane_fill.load_views", lambda _path: [view])
    monkeypatch.setattr("samcloud.colmap.depth_prior.availability", lambda _model: None)
    monkeypatch.setattr("samcloud.colmap.depth_prior.compute_depth_priors", lambda *_args, **_kwargs: {"model": "test", "aligned_views": 0, "global_fit_views": 0})
    monkeypatch.setattr("samcloud.colmap.depth_prior.prior_reader", lambda _path: lambda _name: np.ones((8, 8), np.float32))
    monkeypatch.setattr("samcloud.colmap.mirror_filter.detect_mirrors", lambda *_args, **_kwargs: calls.append("detect") or [])
    monkeypatch.setattr("samcloud.colmap.depth_fill.fill_classified_from_prior", lambda *_args, **_kwargs: calls.append("fill") or {"added_points": 0})
    monkeypatch.setattr("samcloud.colmap.surface_fusion.consolidate_classified_cloud", lambda *_args, **_kwargs: calls.append("consolidate") or {"snapped": 0, "final_points": 0})

    pipeline._refine_after_classification(args, dense, tmp_path / "classified.ply", tmp_path / "scores.npz", summary)

    assert calls == ["detect", "fill", "consolidate"]


def test_picture_without_depth_jump_not_mirror(tmp_path: Path) -> None:
    view = _view()
    depth = np.full((100, 100), 3.0, dtype=np.float32)
    mask = np.zeros((100, 100), dtype=bool)
    mask[35:65, 35:65] = True

    assert mirror_candidate(view, depth, mask, {}) is None
    assert not phantom_mask(np.array([[0.0, 0.0, 5.0]]), [], [], tmp_path, {}).any()


def test_mirror_mask_not_written_into_classes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLabeler:
        @staticmethod
        def load_image(_path):
            return np.zeros((10, 10, 3), dtype=np.uint8), 10, 10

        @staticmethod
        def score_points(_path, _u, _v, **_kwargs):
            scores = np.array([[0.1, 0.8, 0.95]], dtype=np.float32)
            labels = np.full((10, 10), 2, dtype=np.uint8)
            raw = labels.copy()
            raw[3:7, 3:7] = 3
            return scores, labels, raw

    written: list[np.ndarray] = []
    monkeypatch.setattr(classify, "make_point_labeler", lambda *_args, **_kwargs: FakeLabeler())
    monkeypatch.setattr(classify, "project_points", lambda *_args, **_kwargs: (
        np.array([0]), np.array([5]), np.array([5]), np.array([3.0]),
    ))
    monkeypatch.setattr(classify, "write_classified_ply", lambda _path, _xyz, _rgb, classes: written.append(classes.copy()))
    args = SimpleNamespace(
        background_threshold=0.4, threshold_sweep="", dense_dir=str(tmp_path), labels_dir=str(tmp_path / "sam_labels"),
        images_dir=str(tmp_path), depth_check=0, out=str(tmp_path / "classified.ply"), scores_out=None, min_votes=1,
        summary_file=None,
    )
    image = SimpleNamespace(name="view.png", camera_id=1)

    classify.run_segearth_ov3(
        args, np.array([[0.0, 0.0, 3.0]]), np.array([[1, 2, 3]], dtype=np.uint8), {1: object()}, {1: image},
        [(1, [0])], ["clutter", "wall"], 0, None, mirror_names=["mirror"], mirror_prompts=[["mirror"]],
    )

    labels = cv2.imread(str(tmp_path / "sam_labels" / "view.png.png"), cv2.IMREAD_UNCHANGED)
    mirror = cv2.imread(str(tmp_path / "sam_mirror" / "view.png.png"), cv2.IMREAD_GRAYSCALE)
    assert labels is not None and not np.any(labels == 3)
    assert mirror is not None and np.all(mirror[3:7, 3:7] == 255)
    assert written and written[0].tolist() == [2]


def test_reclassify_uses_cache_fused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outdir = tmp_path / "project" / "products"
    cache_fused = outdir.parent / "processing_cache" / "dense" / "fused.ply"
    cache_fused.parent.mkdir(parents=True)
    cache_fused.write_bytes(b"cache fused cloud")
    before = (cache_fused.stat().st_size, cache_fused.stat().st_mtime_ns)
    staged: list[Path] = []
    original_stage = pipeline._stage_reclassify_cache

    def stage(cache_dense: Path) -> Path:
        staged_path = original_stage(cache_dense)
        staged.append(staged_path)
        return staged_path

    monkeypatch.setattr(pipeline, "_stage_reclassify_cache", stage)
    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace())
    args = SimpleNamespace(images=None, panorama_workdir=None)
    with pytest.raises(ValueError, match="fused.ply.vis"):
        pipeline._reclassify_project(args, outdir, tmp_path / "reports", SimpleNamespace(settings={}), None, tmp_path / "classified.ply")

    assert staged and (staged[0] / "fused.ply").is_file()
    assert (cache_fused.stat().st_size, cache_fused.stat().st_mtime_ns) == before


def _wall_mirror(views: list[View], scale: float = 1.0) -> Mirror:
    return Mirror(np.array([0.0, 0.0, -1.0]), 3.0 * scale, np.array([0.0, 0.0, 3.0]) * scale,
                  [np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]) * scale], views)


def test_textureless_ceiling_protected_by_prior(tmp_path: Path) -> None:
    scale = 0.8157
    rng = np.random.default_rng(7)
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    front = [_view(index, (x * scale, y * scale, 0.0)) for index, (x, y) in enumerate(((-0.31, 0.12), (0.17, -0.18), (0.42, 0.08)))]
    room = [_view(10 + index, (x * scale, y * scale, z * scale))
            for index, (x, y, z) in enumerate(((-0.27, 0.21, 3.7), (0.33, -0.12, 3.9), (0.08, 0.31, 3.6), (-0.19, -0.26, 4.1)))]
    ceiling = np.array([[0.11, 0.07, 6.0], [-0.9, 0.5, 6.4]]) * scale + rng.normal(0.0, 0.002, (2, 3))
    for view in front + room:
        # No COLMAP depth anywhere: the ceiling is textureless.
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", np.zeros((100, 100), np.float32))
    for view in front:
        # The noisy prior happens to match the virtual depth in the mirror views: removal pressure.
        local = ceiling @ view.rotation.T + view.translation
        _write_prior_depths(stereo, view, ceiling, local[:, 2] * (1.0 + rng.normal(0.0, 0.004, len(ceiling))))
    for view in room:
        local = ceiling @ view.rotation.T + view.translation
        _write_prior_depths(stereo, view, ceiling, local[:, 2] * (1.0 + rng.normal(0.0, 0.004, len(ceiling))))
    _write_mirror_masks(tmp_path, front + room, np.zeros((100, 100), bool))
    mirror = _wall_mirror(front, scale)

    removed = phantom_mask(ceiling, [mirror], front + room, stereo, {"units_per_metre": scale})

    assert mirror.filter_stats["through_points"] == 2
    assert removed.tolist() == [False, False]
    assert mirror.filter_stats["removed"] == 0
    # Without the room views the same prior agreement is all the evidence: the points are reflections.
    only_front = _wall_mirror(front, scale)
    assert phantom_mask(ceiling, [only_front], front, stereo, {"units_per_metre": scale}).tolist() == [True, True]


def test_phantom_occluded_in_all_direct_views_removed(tmp_path: Path) -> None:
    scale = 0.8157
    rng = np.random.default_rng(11)
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    front = [_view(index, (x * scale, y * scale, 0.0)) for index, (x, y) in enumerate(((-0.31, 0.12), (0.17, -0.18), (0.42, 0.08)))]
    room = [_view(10 + index, (x * scale, y * scale, z * scale))
            for index, (x, y, z) in enumerate(((-0.27, 0.21, 3.7), (0.33, -0.12, 3.9), (0.08, 0.31, 3.6), (-0.19, -0.26, 4.1)))]
    phantom = np.array([[0.11, 0.07, 5.0], [-0.9, 0.5, 5.4]]) * scale + rng.normal(0.0, 0.002, (2, 3))
    for view in front:
        # Seen through the mirror polygon, but neither COLMAP nor the prior agrees with the virtual depth.
        _write_support_depths(stereo, view, np.asarray([[0.0, 0.0, 3.0 * scale]]))
        _write_prior_depths(stereo, view, phantom, np.full(len(phantom), 3.0 * scale))
    for view in room:
        centre = view.center
        blocker = centre + (phantom - centre) * 0.45
        _write_support_depths(stereo, view, blocker)
        local = blocker @ view.rotation.T + view.translation
        _write_prior_depths(stereo, view, phantom, local[:, 2] * (1.0 + rng.normal(0.0, 0.004, len(phantom))))
    _write_mirror_masks(tmp_path, front + room, np.zeros((100, 100), bool))
    mirror = _wall_mirror(front, scale)

    removed = phantom_mask(phantom, [mirror], front + room, stereo, {"units_per_metre": scale})

    assert removed.tolist() == [True, True]
    assert mirror.filter_stats["direct_kept"] == 0


def _write_room_depths(stereo: Path, room: list[View], scale: float, real: np.ndarray | None = None) -> None:
    """Real back wall at z = 8 * scale in every room view; ``real`` points are measured at their own pixels."""
    for view in room:
        depth = np.full((100, 100), 8.0 * scale - float(view.center[2]), dtype=np.float32)
        if real is not None:
            local = real @ view.rotation.T + view.translation
            for point in local:
                x = int((view.fx * point[0] / point[2] + view.cx) * 100 / view.width)
                y = int((view.fy * point[1] / point[2] + view.cy) * 100 / view.height)
                # A small block so the depth-edge filter keeps the object pixel.
                depth[y - 2:y + 3, x - 2:x + 3] = point[2]
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", depth)


def _free_space_scene(tmp_path: Path, scale: float, front_sees_points: bool):
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    front = [_view(index, (x * scale, y * scale, 0.0)) for index, (x, y) in enumerate(((-0.31, 0.12), (0.17, -0.18), (0.42, 0.08)))]
    room = [_view(10 + index, (x * scale, y * scale, z * scale))
            for index, (x, y, z) in enumerate(((-0.27, 0.21, 3.7), (0.33, -0.12, 3.9), (0.08, 0.31, 3.6), (-0.19, -0.26, 4.1)))]
    reflection = np.array([[0.11, 0.07, 5.0]]) * scale
    real = np.array([[-0.9, 0.5, 6.4]]) * scale
    for view in front:
        if front_sees_points:
            _write_support_depths(stereo, view, np.vstack((reflection, real)))
        else:
            write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", np.zeros((100, 100), np.float32))
    _write_room_depths(stereo, room, scale, real)
    _write_mirror_masks(tmp_path, front + room, np.zeros((100, 100), bool))
    return stereo, front, room, reflection, real


def test_reflection_inside_scanned_room_behind_mirror_removed(tmp_path: Path) -> None:
    scale = 0.978
    stereo, front, room, reflection, real = _free_space_scene(tmp_path, scale, front_sees_points=True)
    points = np.vstack((reflection, real))
    mirror = _wall_mirror(front, scale)

    removed = phantom_mask(points, [mirror], front + room, stereo, {"units_per_metre": scale})

    assert removed.tolist() == [True, False]
    assert mirror.filter_stats["free_space_removed"] == 1
    # The old rules count the plain views that look straight through the reflection as direct support.
    old = _wall_mirror(front, scale)
    assert phantom_mask(points, [old], front + room, stereo, {"units_per_metre": scale, "free_space_views": 0}).tolist() == [False, False]
    assert old.filter_stats["free_space_removed"] == 0


def test_free_space_rule_needs_mirror_evidence(tmp_path: Path) -> None:
    scale = 0.978
    stereo, front, room, reflection, _real = _free_space_scene(tmp_path, scale, front_sees_points=False)
    mirror = _wall_mirror(front, scale)

    removed = phantom_mask(reflection, [mirror], front + room, stereo, {"units_per_metre": scale})

    # Looked through by the room views but never seen in the mirror: a matter for the visibility check.
    assert removed.tolist() == [False]
    assert mirror.filter_stats["free_space_removed"] == 0


def test_mirror_mask_reader_dilates_and_skips_empty(tmp_path: Path) -> None:
    masks = tmp_path / "sam_mirror"
    masks.mkdir()
    block = np.zeros((100, 100), np.uint8)
    block[40:44, 40:44] = 255
    assert cv2.imwrite(str(masks / f"{_view(0).name}.png"), block)
    assert cv2.imwrite(str(masks / f"{_view(1).name}.png"), np.zeros((100, 100), np.uint8))
    reader = mirror_mask_reader(masks, 2)

    full = reader(_view(0), (100, 100))
    assert full is not None and full.dtype == bool and full.shape == (100, 100)
    assert full.sum() == 8 * 8 and full[38:46, 38:46].all()
    small = reader(_view(0), (50, 50))
    assert small is not None and small.dtype == bool and small.shape == (50, 50)
    assert small.sum() == 6 * 6 and small[18:24, 18:24].all()
    assert reader(_view(1), (50, 50)) is None  # empty mask
    assert reader(_view(2), (50, 50)) is None  # no mask file
    assert mirror_mask_reader(None, 2)(_view(0), (50, 50)) is None

    report = mirror_report([_wall_mirror([_view(0)]), _wall_mirror([_view(1)])], np.zeros(3, bool))
    assert [item["free_space_removed"] for item in report["mirrors"]] == [0, 0]


def test_bow_tie_union_raises_in_shapely_but_not_in_mirror_area() -> None:
    """Regression for the v10 'Surface fusion was skipped: TopologyException' (unguarded unary_union)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    bow = [(0, 0), (1, 1), (1, 0), (0, 1)]
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    with pytest.raises(Exception, match="side location conflict"):
        unary_union([Polygon(bow), Polygon(square)])
    mirror = Mirror(np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(3),
                    [np.array(bow, dtype=float), np.array(square, dtype=float)], [])
    assert mirror.area == pytest.approx(1.0)
