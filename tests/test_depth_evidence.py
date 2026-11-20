"""Compact depth evidence: streaming voxel aggregation, fusion rules, free space, gap fill."""

from __future__ import annotations

import json

import cv2
import numpy as np

import samcloud.bim.depth_evidence.core as core
from samcloud.bim.depth_evidence import DepthEvidence, build_depth_product
from samcloud.colmap.plane_fill import View, write_colmap_array
from samcloud.engine.ply_io import write_classified_ply

SIZE = 64
FOCAL = 64.0  # 53 degree field of view, plane at 2 m -> ~3 cm pixels
NO_EDGE = {"depth_edge_rel": 100.0}


def _view(index: int, name: str, shift: float = 0.0) -> View:
    # camera centre = -R.T @ t = (-shift, 0, 0) for R = I
    return View(index, name, np.eye(3), np.array([shift, 0.0, 0.0]), FOCAL, FOCAL, SIZE / 2, SIZE / 2, SIZE, SIZE)


def _write_view(dense, view: View, geometric, prior, labels) -> None:
    face, image = view.name.split("/")
    for sub in (f"stereo/depth_maps/{face}", f"depth_prior/{face}", f"sam_labels/{face}"):
        (dense / sub).mkdir(parents=True, exist_ok=True)
    if geometric is not None:
        write_colmap_array(dense / "stereo" / "depth_maps" / face / f"{image}.geometric.bin", geometric.astype(np.float32))
    if prior is not None:
        np.save(dense / "depth_prior" / face / f"{image}.npy", prior.astype(np.float32))
    assert cv2.imwrite(str(dense / "sam_labels" / face / f"{image}.png"), labels.astype(np.uint8))


def _project(tmp_path, views, geometric_fn, prior_fn=None, label_fn=None, cloud_xyz=None):
    """Plane z = 2 seen by every view; ``*_fn(view, rng)`` returns the per-view arrays."""
    dense = tmp_path / "processing_cache" / "dense"
    (dense / "sparse").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for view in views:
        geometric = geometric_fn(view, rng)
        prior = None if prior_fn is None else prior_fn(view, rng)
        labels = np.full((SIZE, SIZE), 17, np.uint8) if label_fn is None else label_fn(view, rng)
        _write_view(dense, view, geometric, prior, labels)
    (tmp_path / "products" / "classified").mkdir(parents=True, exist_ok=True)
    cloud = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.0]], np.float32) if cloud_xyz is None else cloud_xyz
    write_classified_ply(tmp_path / "products" / "classified" / "classified.ply", cloud, None,
                         np.full(len(cloud), 17, np.uint8))
    return dense


def _plane(depth: float = 2.0):
    def make(_view_, rng):
        return depth + rng.normal(0.0, 0.002, (SIZE, SIZE))
    return make


def _patch(monkeypatch, views):
    monkeypatch.setattr(core, "load_views", lambda _path: views)


# --- voxel aggregation / streaming ----------------------------------------------------------

def test_voxel_key_round_trip_with_negative_coordinates() -> None:
    rng = np.random.default_rng(1)
    points = rng.uniform(-40.0, 40.0, (500, 3))
    cells = core._decode_keys(core._voxel_keys(points, 0.015))
    assert np.array_equal(cells, np.floor(points / 0.015).astype(np.int32))
    keys = core._voxel_keys(points, 0.015)
    assert len(np.unique(keys)) == len(np.unique(cells, axis=0))


def test_aggregate_batch_mean_vote_views_and_source() -> None:
    rng = np.random.default_rng(2)
    voxel = 0.015
    centre = np.array([0.4125, 1.2075, -0.3075])  # centre of a voxel
    xyz = np.vstack([centre + rng.uniform(-0.005, 0.005, (30, 3)),
                     centre + np.array([0.105, 0, 0]) + rng.uniform(-0.005, 0.005, (10, 3))])
    labels = np.r_[np.full(20, 17), np.full(10, 2), np.full(10, 18)].astype(np.uint8)
    conf = np.full(40, 0.9, np.float32)
    sources = np.r_[np.full(15, 1), np.full(15, 2), np.full(10, 2)].astype(np.uint8)
    views = np.r_[np.arange(30) % 3, np.arange(10) % 2].astype(np.int64)
    state = core._aggregate_batch(xyz, labels, conf, sources, views, voxel)
    assert len(state["keys"]) == 2
    big = int(np.argmax(state["samples"]))
    assert state["samples"][big] == 30 and state["views"][big] == 3 and state["source"][big] == 3
    mean = state["sum"][big] / 30
    assert np.linalg.norm(mean - centre) < 0.004
    final_xyz, final_labels, confidence, n_views, _ = core._finalise_state(state)
    assert final_labels[big] == 17  # 20 of 30 votes
    assert np.isclose(confidence[big], 0.9 * 20 / 30, atol=1e-6)
    assert n_views[big] == 3


def test_streamed_merge_equals_single_pass() -> None:
    rng = np.random.default_rng(3)
    xyz = rng.uniform(0.0, 0.3, (6000, 3))
    labels = rng.choice([2, 17, 18], 6000).astype(np.uint8)
    conf = rng.uniform(0.4, 0.9, 6000).astype(np.float32)
    sources = rng.choice([1, 2], 6000).astype(np.uint8)
    views = rng.integers(0, 6, 6000).astype(np.int64)
    whole = core._aggregate_batch(xyz, labels, conf, sources, views, 0.015)
    state = core._empty_state()
    for first, last in ((0, 2500), (2500, 4100), (4100, 6000)):
        # distinct view ids per batch, as in the real stream
        state = core._merge_states(state, core._aggregate_batch(
            xyz[first:last], labels[first:last], conf[first:last], sources[first:last], views[first:last] + 100 * first, 0.015))
    assert np.array_equal(state["keys"], whole["keys"])
    assert np.allclose(state["sum"], whole["sum"]) and np.array_equal(state["samples"], whole["samples"])
    assert np.array_equal(state["votes"], whole["votes"])
    assert np.array_equal(state["source"], whole["source"])
    assert np.all(state["views"] >= whole["views"])  # a view split over batches is counted once per batch


def test_confidence_reflects_label_agreement_and_source() -> None:
    xyz = np.tile(np.array([[0.1, 0.1, 0.1]]), (10, 1)) + np.random.default_rng(4).uniform(-0.003, 0.003, (10, 3))
    views = np.arange(10).astype(np.int64)
    unanimous = core._aggregate_batch(xyz, np.full(10, 17, np.uint8), np.full(10, 0.9, np.float32),
                                      np.ones(10, np.uint8), views, 0.015)
    split = core._aggregate_batch(xyz, np.r_[np.full(6, 17), np.full(4, 2)].astype(np.uint8),
                                  np.full(10, 0.9, np.float32), np.ones(10, np.uint8), views, 0.015)
    prior = core._aggregate_batch(xyz, np.full(10, 17, np.uint8), np.full(10, 0.45, np.float32),
                                  np.full(10, 2, np.uint8), views, 0.015)
    conf = [core._finalise_state(state)[2][0] for state in (unanimous, split, prior)]
    assert conf[0] > conf[1] > 0.5 and conf[0] > conf[2] and np.isclose(conf[0], 0.9)


# --- fusion, filters ------------------------------------------------------------------------

def test_build_prefers_colmap_and_compact_product_opens(monkeypatch, tmp_path) -> None:
    views = [_view(0, "front/a.jpg")]
    dense = _project(tmp_path, views, lambda v, r: np.where(
        (np.add.outer(np.arange(SIZE), np.arange(SIZE)) >= 0) & (np.arange(SIZE)[:, None] > 15) & (np.arange(SIZE)[:, None] < 48)
        & (np.arange(SIZE)[None, :] > 15) & (np.arange(SIZE)[None, :] < 48), 2.0 + r.normal(0, 0.002, (SIZE, SIZE)), 0.0),
        prior_fn=lambda v, r: np.full((SIZE, SIZE), 3.0))
    _patch(monkeypatch, views)
    report = build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 1, "voxel_m": 0.02, "min_prior_views": 1, **NO_EDGE})
    assert report["written"] and report["counts_per_source"]["colmap"] > 500
    evidence = DepthEvidence.open(tmp_path)
    assert evidence is not None
    ring = [(-9, -9), (9, -9), (9, 9), (-9, 9)]
    near, labels, _conf, nviews = evidence.points(ring, (1.9, 2.1))
    assert len(near) > 500 and np.all(labels == 17) and np.all(nviews == 1)
    far, *_ = evidence.points(ring, (2.9, 3.1))
    assert len(far) > 100  # Depth-Anything only fills outside the COLMAP patch
    with np.load(tmp_path / "products" / "depth" / "depth_points.npz") as product:
        z = product["xyz"][:, 2]
        assert np.all(product["source"][np.abs(z - 2.0) < 0.05] == 1)
        assert np.all(product["source"][np.abs(z - 3.0) < 0.05] == 2)


def test_prior_only_voxels_need_several_views(monkeypatch, tmp_path) -> None:
    views = [_view(0, "front/a.jpg"), _view(1, "front/b.jpg", shift=-0.01)]
    dense = _project(tmp_path, views, lambda v, r: np.zeros((SIZE, SIZE)),
                     prior_fn=lambda v, r: 3.0 + r.normal(0, 0.002, (SIZE, SIZE)))
    _patch(monkeypatch, views)
    two = build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 1, "voxel_m": 0.05, "min_prior_views": 2, **NO_EDGE})
    assert two["points"] > 0
    evidence = DepthEvidence.open(tmp_path)
    assert evidence is not None and int(evidence.n_views.min()) >= 2
    assert np.all(evidence.source == 2)
    # a single view can never satisfy the two-view rule
    other = tmp_path / "single"
    dense_single = _project(other, views[:1], lambda v, r: np.zeros((SIZE, SIZE)),
                            prior_fn=lambda v, r: 3.0 + r.normal(0, 0.002, (SIZE, SIZE)))
    _patch(monkeypatch, views[:1])
    one = build_depth_product(other, dense_dir=dense_single, settings={"stride": 1, "voxel_m": 0.05, "min_prior_views": 2, **NO_EDGE})
    assert one["written"] is False  # nothing survives: every prior-only voxel has a single view


def test_depth_edges_and_unlabelled_pixels_are_dropped(monkeypatch, tmp_path) -> None:
    views = [_view(0, "front/a.jpg")]

    def stepped(_view_, rng):
        depth = 2.0 + rng.normal(0, 0.002, (SIZE, SIZE))
        depth[:, SIZE // 2:] = 3.0 + rng.normal(0, 0.002, (SIZE, SIZE // 2))  # 50 % depth jump
        return depth

    def labels(_view_, _rng):
        out = np.full((SIZE, SIZE), 17, np.uint8)
        out[:8] = 0  # unlabelled band
        return out

    dense = _project(tmp_path, views, stepped, label_fn=labels)
    _patch(monkeypatch, views)
    build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 1, "voxel_m": 0.01})
    evidence = DepthEvidence.open(tmp_path)
    ring = [(-9, -9), (9, -9), (9, 9), (-9, 9)]
    points, *_ = evidence.points(ring, (0.0, 9.0))
    pixel = np.arange(SIZE)
    # nothing within +-2 px of the depth jump is kept: the world x of columns 30..33 is absent
    cols_x = (pixel[30:34] + 0.5 - SIZE / 2) * 2.0 / FOCAL
    assert not np.any((points[:, 2] < 2.5) & (points[:, 0] > cols_x[0] - 0.01) & (points[:, 0] < cols_x[-1] + 0.01))
    # unlabelled rows 0..7 (world y < (8 - 32) * 2 / 64 = -0.75 at z = 2) never reach the product
    near = points[points[:, 2] < 2.5]
    assert len(near) > 100 and near[:, 1].min() > -0.75


def test_mirror_pixels_are_excluded(monkeypatch, tmp_path) -> None:
    views = [_view(0, "front/a.jpg")]
    dense = _project(tmp_path, views, _plane())
    mirror = np.zeros((SIZE, SIZE), np.uint8)
    mirror[:, :32] = 255
    (dense / "sam_mirror" / "front").mkdir(parents=True)
    assert cv2.imwrite(str(dense / "sam_mirror" / "front" / "a.jpg.png"), mirror)
    _patch(monkeypatch, views)
    build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 1, "voxel_m": 0.02, **NO_EDGE})
    evidence = DepthEvidence.open(tmp_path)
    assert evidence.xyz[:, 0].min() > -0.1  # left half (x < 0) is the mirror


# --- stride / voxel / all views -------------------------------------------------------------

def test_stride_voxel_and_all_views_settings(monkeypatch, tmp_path) -> None:
    views = [_view(index, f"front/v{index}.jpg", shift=-0.05 * index) for index in range(4)]
    dense = _project(tmp_path, views, _plane())
    _patch(monkeypatch, views)
    counts = {}
    for stride in (1, 2, 4):
        counts[stride] = build_depth_product(tmp_path, dense_dir=dense, settings={"stride": stride, "voxel_m": 0.02, **NO_EDGE})["samples"]
    assert counts[1] == 4 * counts[2] and counts[2] == 4 * counts[4]
    fine = build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 1, "voxel_m": 0.01, **NO_EDGE})["points"]
    coarse = build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 1, "voxel_m": 0.06, **NO_EDGE})["points"]
    assert fine > 2 * coarse
    all_views = build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 2, "max_views": 0, **NO_EDGE})
    capped = build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 2, "max_views": 2, **NO_EDGE})
    assert all_views["views_used"] == 4 and capped["views_used"] == 2


def test_batches_do_not_change_the_product(monkeypatch, tmp_path) -> None:
    views = [_view(index, f"front/v{index}.jpg", shift=-0.04 * index) for index in range(5)]
    dense = _project(tmp_path, views, _plane())
    _patch(monkeypatch, views)
    results = []
    for batch in (1, 2, 64):
        build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 2, "voxel_m": 0.02, "batch_views": batch, **NO_EDGE})
        with np.load(tmp_path / "products" / "depth" / "depth_points.npz") as product:
            order = np.lexsort(product["xyz"].T)
            results.append((product["xyz"][order], product["n_views"][order]))
    for xyz, n_views in results[1:]:
        assert xyz.shape == results[0][0].shape
        assert np.allclose(xyz, results[0][0], atol=1e-5) and np.array_equal(n_views, results[0][1])
    assert results[0][1].max() >= 2  # voxels seen by several views are merged across batches


# --- gap fill ---------------------------------------------------------------------------------

def test_gap_fill_mask_uses_radius() -> None:
    cloud = (np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.array([17, 17], np.uint8))
    probe = np.array([[0.02, 0.0, 0.0], [0.045, 0.0, 0.0], [1.0, 0.021, 0.0], [0.5, 0.5, 0.5]])
    assert list(core._gap_fill_mask(probe, cloud, 0.03)) == [False, True, False, True]
    assert core._gap_fill_mask(probe, None, 0.03).all()


def test_report_counts_gap_fill_per_class(monkeypatch, tmp_path) -> None:
    views = [_view(0, "front/a.jpg")]
    # the classified cloud only covers x > 0 of the plane z = 2
    xs, ys = np.meshgrid(np.arange(0.0, 1.0, 0.01), np.arange(-1.0, 1.0, 0.01))
    cloud = np.column_stack((xs.ravel(), ys.ravel(), np.full(xs.size, 2.0))).astype(np.float32)
    dense = _project(tmp_path, views, _plane(), cloud_xyz=cloud)
    _patch(monkeypatch, views)
    report = build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 1, "voxel_m": 0.02, **NO_EDGE})
    gap = report["gap_fill"]
    assert gap["radius_m"] == 0.03 and 0 < gap["points"] < report["points"]
    assert set(gap["per_class"]) == {"17"} and gap["per_class"]["17"] == gap["points"]
    evidence = DepthEvidence.open(tmp_path)
    ring = [(-9, -9), (9, -9), (9, 9), (-9, 9)]
    added, *_ = evidence.points(ring, (0.0, 9.0), gap_only=True)
    assert len(added) == gap["points"] and added[:, 0].max() < 0.05
    assert len(evidence.points(ring, (0.0, 9.0), min_views=2)[0]) == 0  # single view


def test_alignment_reports_offset_and_distance_per_class() -> None:
    rng = np.random.default_rng(5)
    plane = np.column_stack((rng.uniform(0, 3, 4000), rng.uniform(0, 3, 4000), np.full(4000, 1.0)))
    cloud = (plane, np.full(4000, 4, np.uint8))
    product = plane[:1500] + np.array([0.0, 0.0, 0.012]) + rng.normal(0, 0.002, (1500, 3)) * [0.2, 0.2, 1.0]
    aligned, residual, detail = core._alignment(product, np.full(1500, 4, np.uint8), cloud, 0.08, sources=np.ones(1500, np.uint8))
    assert 0.008 < residual < 0.016
    assert abs(np.median(aligned[:, 2]) - 1.0) < 0.003
    assert "4" in detail["median_nn_distance_per_class_m"] and "1" in detail["median_nn_distance_per_source_m"]


# --- free space ----------------------------------------------------------------------------------

def test_free_space_ray_marching_matches_brute_force() -> None:
    voxel = 0.05
    camera = np.array([0.013, 0.021, 0.007])
    rng = np.random.default_rng(6)
    points = np.column_stack((rng.uniform(-0.6, 0.6, 40), rng.uniform(-0.6, 0.6, 40), np.full(40, 2.0)))
    keys, through, end_keys, ends = core._free_space_records(camera, points, voxel)
    got = dict(zip(keys.tolist(), through.tolist()))
    expected: dict[int, int] = {}
    for point in points:
        delta = point - camera
        distance = np.linalg.norm(delta)
        for step in range(1, int(max(np.floor((distance - 1.5 * voxel) / voxel), 0)) + 1):
            key = int(core._voxel_keys((camera + delta / distance * step * voxel)[None], voxel)[0])
            expected[key] = expected.get(key, 0) + 1
    assert got == expected
    assert ends.sum() == 40
    # the surface voxel of every ray is never a pass-through voxel of the same ray's last step
    surface = set(end_keys.tolist())
    assert not (surface & set(keys.tolist())) or len(surface & set(keys.tolist())) < len(surface)


def test_free_space_between_camera_and_plane(monkeypatch, tmp_path) -> None:
    views = [_view(0, "front/a.jpg"), _view(1, "front/b.jpg", shift=-0.2)]
    dense = _project(tmp_path, views, _plane())
    _patch(monkeypatch, views)
    report = build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 4, "free_ray_stride": 2, **NO_EDGE})
    assert report["free_space_voxels"] > 500
    evidence = DepthEvidence.open(tmp_path)
    ring = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    free, through, ending = evidence.free_space(ring, (0.1, 1.8), min_through=2)
    assert len(free) > 200 and np.all(through > ending)
    assert free[:, 2].max() < 1.9  # nothing is "empty" at or behind the observed plane z = 2
    behind, *_ = evidence.free_space(ring, (2.2, 5.0), min_through=1)
    assert len(behind) == 0


# --- manifest, cache key, absence ----------------------------------------------------------------

def test_missing_project_is_graceful_and_manifest_is_valid_json(monkeypatch, tmp_path) -> None:
    views = [_view(0, "front/a.jpg")]
    dense = _project(tmp_path, views, _plane())
    _patch(monkeypatch, views)
    build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 2, **NO_EDGE})
    manifest = json.loads((tmp_path / "products" / "depth" / "depth_evidence.json").read_text(encoding="utf-8"))
    assert manifest["written"] and manifest["runtime_s"] >= 0 and manifest["size_bytes"] > 0
    assert manifest["depth_anything_vs_colmap_abs_depth_diff_m"] is None  # no prior in this project
    assert manifest["gap_fill"]["points"] >= 0 and manifest["views_total"] == 1
    evidence = DepthEvidence.open(tmp_path)
    key = evidence.cache_key([(0, 0), (1, 0), (1, 1)], (0, 1), {"stride": 2})
    assert key == evidence.cache_key([(0, 0), (1, 0), (1, 1)], (0, 1), {"stride": 2})
    assert key != evidence.cache_key([(0, 0), (1, 0), (1, 1)], (0, 1), {"stride": 3})
    assert DepthEvidence.open(tmp_path / "no_depth") is None
    assert build_depth_product(tmp_path / "no_depth")["reason"] == "depth maps missing"


def test_prior_vs_colmap_agreement_is_reported(monkeypatch, tmp_path) -> None:
    views = [_view(0, "front/a.jpg")]
    dense = _project(tmp_path, views, _plane(2.0), prior_fn=lambda v, r: 2.03 + r.normal(0, 0.004, (SIZE, SIZE)))
    _patch(monkeypatch, views)
    report = build_depth_product(tmp_path, dense_dir=dense, settings={"stride": 2, **NO_EDGE})
    diff = report["depth_anything_vs_colmap_abs_depth_diff_m"]
    assert 0.02 < diff["median"] < 0.04 and diff["p90"] >= diff["median"]


def test_world_points_match_matrix_reference_for_rotated_view() -> None:
    angle = np.deg2rad(37.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    view = View(0, "front/a.jpg", rotation, np.array([0.4, -1.1, 2.3]), FOCAL, FOCAL, SIZE / 2, SIZE / 2, SIZE, SIZE)
    rng = np.random.default_rng(8)
    depth = 1.0 + rng.uniform(0.0, 4.0, (SIZE, SIZE))
    rows, cols = np.mgrid[0:SIZE:3, 0:SIZE:3]
    rows, cols = rows.ravel(), cols.ravel()
    got = core._world_points(view, rows, cols, depth.astype(np.float32))
    z = depth.astype(np.float32)[rows, cols].astype(np.float64)
    camera = np.column_stack(((cols + 0.5 - SIZE / 2) * z / FOCAL, (rows + 0.5 - SIZE / 2) * z / FOCAL, z))
    assert np.allclose(got, (camera - view.translation) @ rotation, atol=1e-9)
    # projecting back with the pinhole model returns the pixel
    camera_back = got @ rotation.T + view.translation
    assert np.allclose(camera_back[:, 0] / camera_back[:, 2] * FOCAL + SIZE / 2 - 0.5, cols, atol=1e-6)


def test_pipeline_hook_runs_before_cache_cleanup_and_is_guarded() -> None:
    from pathlib import Path

    source = (Path(core.__file__).resolve().parents[2] / "engine" / "pipeline.py").read_text(encoding="utf-8")
    hook = source.index("build_depth_product(dense_dir.parent.parent")
    cleanup = source.index("remove_generated(generated, dense_dir.parent)", hook)
    assert hook < cleanup and source.index("_release_fusion_cache(dense_dir, summary)", hook) > hook
    guarded = source[source.rindex("try:", 0, hook):hook]
    after = source[hook:cleanup]
    assert "try:" in guarded and "except Exception" in after  # a failure only logs a warning
