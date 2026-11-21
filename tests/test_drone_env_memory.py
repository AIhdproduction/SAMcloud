"""Memory-bounded working-cloud and semantic regression tests."""
from __future__ import annotations

import pickle

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

from samcloud.drone.contracts import LocalFrame, PointSet
from samcloud.drone.run_environment import (
    _step_load,
    _thin_pointset_voxels,
    _transfer_classes_nearest,
    _working_cloud,
)
from samcloud.drone.semantics import _local_plane_metrics, _near_building_mask, run_semantics


def _pointset(xyz: np.ndarray) -> PointSet:
    count = len(xyz)
    values = np.arange(count)
    return PointSet(
        xyz=np.asarray(xyz, dtype=np.float64),
        classes=(values + 10).astype(np.uint16),
        confidence=(values + .1).astype(np.float32),
        margin=(values + .2).astype(np.float32),
        view_count=(values + 20).astype(np.uint16),
        normals=np.column_stack((values, values + 1, values + 2)).astype(float),
        rgb=np.column_stack((values, values + 1, values + 2)).astype(np.uint8),
        adjusted=values.astype(bool),
    )


def test_voxel_thinning_keeps_pointset_attributes_aligned() -> None:
    rng = np.random.default_rng(12)
    ps = _pointset(rng.uniform(0.001, 2.999, (400, 3)))
    thinned = _thin_pointset_voxels(ps, .3)
    source_index = thinned.classes.astype(int) - 10
    assert 0 < len(thinned) < len(ps)
    assert np.allclose(thinned.xyz, ps.xyz[source_index])
    assert np.allclose(thinned.confidence, ps.confidence[source_index])
    assert np.allclose(thinned.margin, ps.margin[source_index])
    assert np.array_equal(thinned.view_count, ps.view_count[source_index])
    assert np.allclose(thinned.normals, ps.normals[source_index])
    assert np.array_equal(thinned.rgb, ps.rgb[source_index])
    assert np.array_equal(thinned.adjusted, ps.adjusted[source_index])


def test_auto_working_voxel_respects_limit_and_skips_small_cloud() -> None:
    rng = np.random.default_rng(25)
    ps = _pointset(rng.uniform(0.0001, 25.9999, (300_000, 3)))
    working, voxel = _working_cloud(ps, {"working_voxel_m": "auto", "max_working_points": 50_000, "min_working_voxel_m": .03})
    assert len(working) <= 50_000
    assert voxel is not None and voxel >= .03
    unchanged, no_voxel = _working_cloud(ps.subset(np.arange(2_000)), {"working_voxel_m": "auto", "max_working_points": 50_000, "min_working_voxel_m": .03})
    assert len(unchanged) == 2_000 and no_voxel is None


def test_local_plane_metrics_recovers_noisy_plane() -> None:
    rng = np.random.default_rng(31)
    xy = rng.uniform(0.001, 1.999, (10_000, 2))
    xyz = np.column_stack((xy, rng.normal(0, .003, len(xy))))
    rms, normals, sizes = _local_plane_metrics(xyz, cKDTree(xyz), radius=.2, k=24)
    usable = sizes >= 6
    assert abs(np.median(rms[usable]) - .003) < .0015
    assert np.median(np.abs(normals[usable, 2])) > .99


def test_local_plane_metrics_chunked_matches_unchunked() -> None:
    rng = np.random.default_rng(43)
    xyz = rng.normal(size=(5_000, 3))
    tree = cKDTree(xyz)
    expected = _local_plane_metrics(xyz, tree, radius=.6, k=24, chunk_size=10_000)
    actual = _local_plane_metrics(xyz, tree, radius=.6, k=24, chunk_size=1_000)
    for left, right in zip(expected, actual, strict=True):
        assert np.allclose(left, right)


def test_vectorised_near_building_matches_covers_away_from_boundaries() -> None:
    rng = np.random.default_rng(55)
    outlines = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]).buffer(.25),
        Polygon([(4, 1), (7, 1), (7, 4), (4, 4)]).buffer(.25),
    ]
    xy = rng.uniform([-1.9, -1.8], [8.9, 5.8], (2_000, 2)) + rng.normal(0, 1e-5, (2_000, 2))
    old = np.array([any(outline.covers(Point(*point)) for outline in outlines) for point in xy])
    assert np.array_equal(_near_building_mask(xy, outlines), old)


def test_run_semantics_uses_supplied_cloud_without_reload(tmp_path, monkeypatch) -> None:
    import samcloud.drone.io as drone_io

    monkeypatch.setattr(drone_io, "load_classified_pointset_lv95", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("reload")))
    rng = np.random.default_rng(67)
    ground = np.column_stack((rng.uniform(0.01, .99, (100, 2)), rng.normal(0, .001, 100)))
    ps = PointSet(np.vstack((ground, [[.5, .5, 3.0]])), np.full(101, 2, dtype=np.uint16))
    frame = LocalFrame((2_600_000.0, 1_200_000.0, 400.0))
    result = run_semantics(tmp_path / "products", tmp_path / "out", pointset=ps, frame=frame, dtm_sampler=lambda xy: np.zeros(len(np.asarray(xy).reshape(-1, 2))))
    assert result["frame"] is frame and len(result["pointset"]) == len(ps)


def test_step_load_writes_working_and_full_cloud(tmp_path, monkeypatch) -> None:
    import samcloud.drone.io as drone_io
    import samcloud.drone.perimeter as perimeter

    rng = np.random.default_rng(79)
    ps = _pointset(rng.uniform(0.001, 9.999, (5_000, 3)))
    frame = LocalFrame((2_600_000.0, 1_200_000.0, 400.0))
    monkeypatch.setattr(drone_io, "load_classified_pointset_lv95", lambda *_args, **_kwargs: (ps, frame))
    monkeypatch.setattr(perimeter, "processing_boundary", lambda *_args, **_kwargs: (None, {"mode": "none"}))
    cache = tmp_path / "cache"
    info = _step_load(tmp_path / "project", tmp_path / "out", {"environment": {"working_voxel_m": "auto", "max_working_points": 100, "min_working_voxel_m": .03}}, cache)
    with (cache / "load.pkl").open("rb") as stream:
        working, _working_frame = pickle.load(stream)
    with (cache / "load_full.pkl").open("rb") as stream:
        full, _full_frame = pickle.load(stream)
    assert len(working) <= 100 and len(full) == len(ps)
    assert info["working_voxel_m"] is not None
    assert info["points_full"] == len(ps) and info["points_working"] == len(working)


def test_export_class_transfer_uses_nearest_working_point() -> None:
    working = PointSet(np.array([[.01, .02, .03], [10.02, .01, -.02]]), np.array([6, 145], dtype=np.uint16))
    full = PointSet(np.array([[.03, .01, .02], [-.02, .02, -.01], [9.98, .03, .01], [10.04, -.01, -.02]]), np.zeros(4, dtype=np.uint16))
    transferred = _transfer_classes_nearest(full, working, chunk_size=2)
    expected = working.classes[cKDTree(working.xyz).query(full.xyz)[1]]
    assert np.array_equal(transferred.classes, expected)
