"""Tests for 3D instance association, spatial splitting, snapshotting, and attachment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from samcloud.engine.instances import (
    JSON_NAME,
    SOURCE_NAME,
    associate_instances,
    attach_instances,
    snapshot_instances,
    write_instance_product,
)
from samcloud.engine.ply_io import read_vertex_array, write_classified_ply


def _generate_noisy_cluster(center: tuple[float, float, float], count: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Generate points densely inside a bounding sphere so cKDTree(0.05) connects them."""
    cx, cy, cz = center
    steps = int(np.ceil(count ** (1 / 3)))
    coords = np.linspace(-radius * 0.7, radius * 0.7, steps)
    grid = np.array([[x, y, z] for x in coords for y in coords for z in coords], dtype=np.float64)
    if len(grid) < count:
        grid = np.vstack([grid, grid[:count - len(grid)]])
    chosen = grid[:count]
    jitter = rng.uniform(-0.003, 0.003, chosen.shape)
    points = chosen + jitter + np.array([cx, cy, cz], dtype=np.float64)
    return points


def test_associate_four_chairs_multiview() -> None:
    """4 chairs separated ~0.6 m apart, seen in 3 views with partial overlap:

    - exactly 4 consecutive IDs (1..4)
    - specific label "chair" preferred over generic "furniture"
    - label_index 1..4 ordered by centroid coordinates
    """
    rng = np.random.default_rng(42)
    n_chairs = 4
    pts_per_chair = 300
    n_points = n_chairs * pts_per_chair

    centers = [(i * 0.6 + 0.01 * rng.uniform(-1, 1), 0.0, 0.0) for i in range(n_chairs)]
    chair_clouds = [_generate_noisy_cluster(center, pts_per_chair, radius=0.04, rng=rng) for center in centers]
    xyz = np.vstack(chair_clouds)

    point_rows = []
    vinst_rows = []
    vinst_classes = []
    vinst_confs = []
    vinst_prompts = []
    vinst_views = []

    # Each chair seen in 3 views with partial overlap:
    # View 0: points 0..249 of chair
    # View 1: points 50..299 of chair
    # View 2: points 0..100 and 200..299 of chair (151 points)
    # Every point has at least 2 views; VI 0 and VI 1 share 200 pts; VI 1 and VI 2 share 151 pts.
    vinst_counter = 0
    for c in range(n_chairs):
        base_pt = c * pts_per_chair

        # VI 0: view 0, generic prompt "furniture" (idx 0)
        vi0_pts = base_pt + np.arange(0, 250, dtype=np.int32)
        point_rows.append(vi0_pts)
        vinst_rows.append(np.full(len(vi0_pts), vinst_counter, dtype=np.int32))
        vinst_classes.append(17)
        vinst_confs.append(0.82)
        vinst_prompts.append(0)  # "furniture" (generic)
        vinst_views.append(0)
        vinst_counter += 1

        # VI 1: view 1, specific prompt "chair" (idx 1)
        vi1_pts = base_pt + np.arange(50, 300, dtype=np.int32)
        point_rows.append(vi1_pts)
        vinst_rows.append(np.full(len(vi1_pts), vinst_counter, dtype=np.int32))
        vinst_classes.append(17)
        vinst_confs.append(0.88)
        vinst_prompts.append(1)  # "chair" (specific)
        vinst_views.append(1)
        vinst_counter += 1

        # VI 2: view 2, specific prompt "chair" (idx 1)
        vi2_pts = base_pt + np.concatenate([np.arange(0, 101), np.arange(200, 300)]).astype(np.int32)
        point_rows.append(vi2_pts)
        vinst_rows.append(np.full(len(vi2_pts), vinst_counter, dtype=np.int32))
        vinst_classes.append(17)
        vinst_confs.append(0.85)
        vinst_prompts.append(1)  # "chair" (specific)
        vinst_views.append(2)
        vinst_counter += 1

    point_idx = np.concatenate(point_rows)
    vinst_idx = np.concatenate(vinst_rows)
    resolved_class = np.full(n_points, 17, dtype=np.uint8)

    params = {
        "merge_overlap": 0.5,
        "min_shared_points": 10,
        "min_views": 2,
        "min_points": 100,
        "split_radius_m": 0.05,
        "class_names": ["wall", "ceiling", "floor", "door", "window", "stairs", "railing",
                        "column", "beam", "steel structure", "pipe", "cable", "electrical",
                        "hvac", "ceiling equipment", "construction material", "furniture"],
    }
    prompt_texts = ["furniture", "chair"]
    generic_prompt_ids = {0}

    instance_id, metadata = associate_instances(
        point_idx=point_idx,
        vinst_idx=vinst_idx,
        vinst_class=np.asarray(vinst_classes, dtype=np.int32),
        vinst_conf=np.asarray(vinst_confs, dtype=np.float32),
        vinst_prompt=np.asarray(vinst_prompts, dtype=np.int32),
        n_points=n_points,
        resolved_class=resolved_class,
        xyz=xyz,
        params=params,
        prompt_texts=prompt_texts,
        generic_prompt_ids=generic_prompt_ids,
        vinst_view=np.asarray(vinst_views, dtype=np.int32),
    )

    unique_ids = np.unique(instance_id[instance_id > 0]).tolist()
    assert unique_ids == [1, 2, 3, 4]
    assert len(metadata) == 4

    for c in range(n_chairs):
        chair_slice = instance_id[c * pts_per_chair:(c + 1) * pts_per_chair]
        assert np.all(chair_slice == c + 1)

    labels = [m["label"] for m in metadata]
    assert labels == ["chair", "chair", "chair", "chair"]

    label_indices = [m["label_index"] for m in metadata]
    assert label_indices == [1, 2, 3, 4]

    for m in metadata:
        assert m["class_name"] == "furniture"
        assert m["class_id"] == 17
        assert m["point_count"] == pts_per_chair
        assert m["view_count"] == 3


def test_two_touching_same_class_objects_not_merged() -> None:
    """Distinct view-instances of two adjacent same-class objects that overlap marginally must not merge."""
    rng = np.random.default_rng(101)
    # Object A around x=0.0, Object B around x=0.3
    obj_a = _generate_noisy_cluster((0.0, 0.0, 0.0), 200, radius=0.03, rng=rng)
    obj_b = _generate_noisy_cluster((0.3, 0.0, 0.0), 200, radius=0.03, rng=rng)
    xyz = np.vstack([obj_a, obj_b])
    n_points = 400

    # Object A is seen in views 0 and 1 (sharing 180 points -> merged)
    # Object B is seen in views 2 and 3 (sharing 180 points -> merged)
    # View 1 marginally observes 5 points of Object B (shared = 5 < min_shared_points=10)
    point_idx = np.concatenate([
        np.arange(0, 190, dtype=np.int32),  # VI 0: view 0 on Obj A
        np.arange(10, 205, dtype=np.int32),  # VI 1: view 1 on Obj A + 5 pts of Obj B
        np.arange(200, 390, dtype=np.int32),  # VI 2: view 2 on Obj B
        np.arange(210, 400, dtype=np.int32),  # VI 3: view 3 on Obj B
    ])
    vinst_idx = np.concatenate([
        np.full(190, 0, dtype=np.int32),
        np.full(195, 1, dtype=np.int32),
        np.full(190, 2, dtype=np.int32),
        np.full(190, 3, dtype=np.int32),
    ])
    vinst_class = np.array([17, 17, 17, 17], dtype=np.int32)
    vinst_conf = np.array([0.85, 0.85, 0.85, 0.85], dtype=np.float32)
    vinst_prompt = np.array([1, 1, 1, 1], dtype=np.int32)
    vinst_view = np.array([0, 1, 2, 3], dtype=np.int32)

    params = {
        "merge_overlap": 0.5,
        "min_shared_points": 10,
        "min_views": 2,
        "min_points": 100,
        "split_radius_m": 0.05,
        "class_names": ["furniture"],
    }

    instance_id, metadata = associate_instances(
        point_idx=point_idx,
        vinst_idx=vinst_idx,
        vinst_class=vinst_class,
        vinst_conf=vinst_conf,
        vinst_prompt=vinst_prompt,
        n_points=n_points,
        resolved_class=np.full(n_points, 17, dtype=np.uint8),
        xyz=xyz,
        params=params,
        prompt_texts=["furniture", "chair"],
        generic_prompt_ids={0},
        vinst_view=vinst_view,
    )

    unique_ids = np.unique(instance_id[instance_id > 0]).tolist()
    assert unique_ids == [1, 2]
    assert len(metadata) == 2


def test_single_view_instance_dropped_when_min_views_is_two() -> None:
    """An instance seen in only one view must be dropped (all IDs 0) when min_views=2."""
    rng = np.random.default_rng(202)
    xyz = _generate_noisy_cluster((0.0, 0.0, 0.0), 200, radius=0.03, rng=rng)
    n_points = 200

    # Only a single view (view 0) observes the object
    point_idx = np.arange(200, dtype=np.int32)
    vinst_idx = np.zeros(200, dtype=np.int32)
    vinst_class = np.array([17], dtype=np.int32)
    vinst_conf = np.array([0.90], dtype=np.float32)
    vinst_prompt = np.array([1], dtype=np.int32)
    vinst_view = np.array([0], dtype=np.int32)

    params = {
        "merge_overlap": 0.5,
        "min_shared_points": 10,
        "min_views": 2,
        "min_points": 100,
        "split_radius_m": 0.05,
    }

    instance_id, metadata = associate_instances(
        point_idx=point_idx,
        vinst_idx=vinst_idx,
        vinst_class=vinst_class,
        vinst_conf=vinst_conf,
        vinst_prompt=vinst_prompt,
        n_points=n_points,
        resolved_class=np.full(n_points, 17, dtype=np.uint8),
        xyz=xyz,
        params=params,
        prompt_texts=["furniture", "chair"],
        generic_prompt_ids={0},
        vinst_view=vinst_view,
    )

    assert np.all(instance_id == 0)
    assert len(metadata) == 0


def test_point_resolved_class_mismatch_gets_zero() -> None:
    """A point whose resolved semantic class contradicts the instance majority class gets ID 0."""
    rng = np.random.default_rng(303)
    xyz = _generate_noisy_cluster((0.0, 0.0, 0.0), 200, radius=0.03, rng=rng)
    n_points = 200

    # Seen in views 0 and 1 with full overlap -> merges
    point_idx = np.concatenate([np.arange(200, dtype=np.int32), np.arange(200, dtype=np.int32)])
    vinst_idx = np.concatenate([np.zeros(200, dtype=np.int32), np.ones(200, dtype=np.int32)])
    vinst_class = np.array([17, 17], dtype=np.int32)  # furniture
    vinst_conf = np.array([0.85, 0.85], dtype=np.float32)
    vinst_prompt = np.array([1, 1], dtype=np.int32)
    vinst_view = np.array([0, 1], dtype=np.int32)

    # First 150 points resolved as furniture (17); last 50 points resolved as wall (2)
    resolved_class = np.full(n_points, 17, dtype=np.uint8)
    resolved_class[150:] = 2

    params = {
        "merge_overlap": 0.5,
        "min_shared_points": 10,
        "min_views": 2,
        "min_points": 100,
        "split_radius_m": 0.05,
    }

    instance_id, metadata = associate_instances(
        point_idx=point_idx,
        vinst_idx=vinst_idx,
        vinst_class=vinst_class,
        vinst_conf=vinst_conf,
        vinst_prompt=vinst_prompt,
        n_points=n_points,
        resolved_class=resolved_class,
        xyz=xyz,
        params=params,
        prompt_texts=["furniture", "chair"],
        generic_prompt_ids={0},
        vinst_view=vinst_view,
    )

    assert np.all(instance_id[:150] == 1)
    assert np.all(instance_id[150:] == 0)
    assert len(metadata) == 1
    assert metadata[0]["point_count"] == 150


def test_spatially_disjoint_instance_is_split() -> None:
    """A single view-instance cluster consisting of two spatially disconnected blobs 1 m apart is split."""
    rng = np.random.default_rng(404)
    blob1 = _generate_noisy_cluster((0.0, 0.0, 0.0), 150, radius=0.03, rng=rng)
    blob2 = _generate_noisy_cluster((1.0, 0.0, 0.0), 150, radius=0.03, rng=rng)
    xyz = np.vstack([blob1, blob2])
    n_points = 300

    # Both blobs are seen together in views 0 and 1
    point_idx = np.concatenate([np.arange(300, dtype=np.int32), np.arange(300, dtype=np.int32)])
    vinst_idx = np.concatenate([np.zeros(300, dtype=np.int32), np.ones(300, dtype=np.int32)])
    vinst_class = np.array([17, 17], dtype=np.int32)
    vinst_conf = np.array([0.90, 0.90], dtype=np.float32)
    vinst_prompt = np.array([1, 1], dtype=np.int32)
    vinst_view = np.array([0, 1], dtype=np.int32)

    params = {
        "merge_overlap": 0.5,
        "min_shared_points": 10,
        "min_views": 2,
        "min_points": 100,
        "split_radius_m": 0.05,
    }

    instance_id, metadata = associate_instances(
        point_idx=point_idx,
        vinst_idx=vinst_idx,
        vinst_class=vinst_class,
        vinst_conf=vinst_conf,
        vinst_prompt=vinst_prompt,
        n_points=n_points,
        resolved_class=np.full(n_points, 17, dtype=np.uint8),
        xyz=xyz,
        params=params,
        prompt_texts=["furniture", "chair"],
        generic_prompt_ids={0},
        vinst_view=vinst_view,
    )

    unique_ids = np.unique(instance_id[instance_id > 0]).tolist()
    assert unique_ids == [1, 2]
    assert len(metadata) == 2
    assert np.all(instance_id[:150] == 1)
    assert np.all(instance_id[150:] == 2)
    assert metadata[0]["point_count"] == 150
    assert metadata[1]["point_count"] == 150


def test_snapshot_and_attach_instances(tmp_path: Path) -> None:
    """Snapshot transfers IDs by nearest neighbour within 5 cm; drops outside 5 cm or on class mismatch; handles missing snapshot."""
    rng = np.random.default_rng(505)
    source_xyz = _generate_noisy_cluster((0.0, 0.0, 0.0), 120, radius=0.02, rng=rng)
    source_ply = tmp_path / "classified_source.ply"
    source_npz = tmp_path / SOURCE_NAME
    source_rgb = np.full((120, 3), 100, dtype=np.uint8)
    source_class = np.full(120, 17, dtype=np.uint8)

    # Write source classified PLY
    write_classified_ply(source_ply, source_xyz, source_rgb, source_class)
    source_instance_id = np.ones(120, dtype=np.uint32)
    source_meta = [{
        "id": 1,
        "class_id": 17,
        "class_name": "furniture",
        "label": "chair",
        "label_index": 1,
        "point_count": 120,
        "centroid": source_xyz.mean(axis=0).tolist(),
        "bbox_min": source_xyz.min(axis=0).tolist(),
        "bbox_max": source_xyz.max(axis=0).tolist(),
        "view_count": 3,
        "confidence": 0.88,
    }]
    write_instance_product(source_ply, source_instance_id, source_meta)

    # Snapshot source instances
    snapshot_ok = snapshot_instances(source_ply, source_npz)
    assert snapshot_ok is True
    assert source_npz.is_file()

    # Create target point cloud (step 6 output):
    # - points 0..119: near source points (offset 0.015 m < 0.05 m), same class 17 -> transfers ID 1
    # - point 120: near source point (offset 0.015 m < 0.05 m), but class 2 (wall) -> class guard sets 0
    # - point 121: far from source points (offset 0.25 m > 0.05 m), class 17 -> gets 0
    target_xyz = np.vstack([
        source_xyz + rng.uniform(-0.015, 0.015, source_xyz.shape),
        source_xyz[0] + [0.01, 0.01, 0.01],
        source_xyz[0] + [0.25, 0.25, 0.25],
    ])
    target_class = np.concatenate([
        np.full(120, 17, dtype=np.uint8),
        np.array([2], dtype=np.uint8),
        np.array([17], dtype=np.uint8),
    ])
    target_ply = tmp_path / "classified_final.ply"
    write_classified_ply(target_ply, target_xyz, np.full((122, 3), 100, dtype=np.uint8), target_class)

    attach_result = attach_instances(source_npz, target_ply, params={"attach_radius_m": 0.05, "min_points": 50})
    assert attach_result["enabled"] is True
    assert attach_result["count"] == 1
    assert attach_result["points"] == 120

    vertex = read_vertex_array(target_ply)
    assert "instance_id" in vertex.dtype.names
    attached_ids = np.asarray(vertex["instance_id"])
    assert np.all(attached_ids[:120] == 1)
    assert attached_ids[120] == 0  # class guard
    assert attached_ids[121] == 0  # > 5 cm

    # Check updated instances.json next to final PLY
    json_path = target_ply.with_name(JSON_NAME)
    assert json_path.is_file()
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_data["schema_version"] == 1
    assert len(json_data["instances"]) == 1
    record = json_data["instances"][0]
    assert record["id"] == 1
    assert record["point_count"] == 120
    assert record["label"] == "chair"

    # Missing snapshot file must be a no-op without crash and without writing instance property
    clean_target = tmp_path / "clean_classified.ply"
    write_classified_ply(clean_target, target_xyz[:10], np.full((10, 3), 100, dtype=np.uint8), target_class[:10])
    missing_result = attach_instances(tmp_path / "non_existent.npz", clean_target)
    assert missing_result["enabled"] is False
    assert "reason" in missing_result
    clean_vertex = read_vertex_array(clean_target)
    assert "instance_id" not in clean_vertex.dtype.names


def test_associate_instances_large_input_is_fast() -> None:
    """300k points / ~3M memberships (30 objects x 10 views) must stay vectorised."""
    import time

    objects, side, views_per_object = 30, 100, 10
    per_object = side * side
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2) * 0.008
    xyz = np.concatenate([
        np.column_stack((grid[:, 0] + 3.0 * index, grid[:, 1], np.zeros(per_object))) for index in range(objects)
    ])
    n_points = len(xyz)
    rng = np.random.default_rng(7)
    point_rows, vinst_rows = [], []
    for index in range(objects):
        for view in range(views_per_object):
            seen = np.flatnonzero(rng.random(per_object) < 0.92) + index * per_object
            point_rows.append(seen)
            vinst_rows.append(np.full(len(seen), index * views_per_object + view))
    point_idx = np.concatenate(point_rows)
    vinst_idx = np.concatenate(vinst_rows)
    assert len(point_idx) > 2_500_000
    count = objects * views_per_object
    started = time.perf_counter()
    instance_id, metadata = associate_instances(
        point_idx=point_idx, vinst_idx=vinst_idx,
        vinst_class=np.full(count, 17, dtype=np.int32),
        vinst_conf=np.full(count, 0.8, dtype=np.float32),
        vinst_prompt=np.zeros(count, dtype=np.int32),
        n_points=n_points, resolved_class=np.full(n_points, 17, dtype=np.uint8), xyz=xyz,
        params={"merge_overlap": 0.5, "min_shared_points": 10, "min_views": 2, "min_points": 100,
                "split_radius_m": 0.05, "class_names": ["x"] * 17},
        prompt_texts=["furniture"], vinst_view=np.arange(count, dtype=np.int32),
    )
    elapsed = time.perf_counter() - started
    print(f"associate_instances perf: {elapsed:.2f}s for {len(point_idx)} pairs")
    assert elapsed < 20.0
    assert len(metadata) == objects
    assert np.count_nonzero(instance_id) > 0.99 * n_points


def test_pipeline_discards_stale_snapshot_and_guards_attach(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace
    from samcloud.engine import pipeline

    raw = tmp_path / "classified_raw.ply"
    stale = tmp_path / SOURCE_NAME
    stale.write_bytes(b"stale")
    pipeline._discard_stale_instances(raw)
    assert not stale.exists()
    pipeline._discard_stale_instances(raw)  # idempotent when nothing is left

    disabled = SimpleNamespace(instances_enabled="0", no_instances=False)
    stale.write_bytes(b"stale")
    output = tmp_path / "classified.ply"
    output.write_bytes(b"ply")
    summary = SimpleNamespace(settings={})
    pipeline._attach_instances(disabled, tmp_path / "classified_scores.npz", output, summary)
    assert summary.settings == {}  # never attempted: stale IDs cannot leak into a disabled run

    enabled = SimpleNamespace(instances_enabled="1", no_instances=False, instance_attach_radius_m=0.05,
                              instance_min_points=100, instance_merge_overlap=0.5,
                              instance_min_shared_points=10, instance_min_views=2,
                              instance_split_radius_m=0.05)
    pipeline._attach_instances(enabled, tmp_path / "classified_scores.npz", output, summary)
    assert "instances" in summary.settings  # enabled: attempted (stale junk reported, not raised)


def test_snapshot_failure_is_logged_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from types import SimpleNamespace
    from samcloud.engine import pipeline

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "snapshot_instances", boom)
    args = SimpleNamespace(instances_enabled="1", no_instances=False)
    pipeline._snapshot_instances_safely(args, tmp_path / "classified_raw.ply")
    assert "WARNING: Instances could not be set aside" in capsys.readouterr().out


def test_instance_product_writes_leave_no_temporary_files(tmp_path: Path) -> None:
    ply = tmp_path / "classified.ply"
    xyz = np.random.default_rng(1).random((5, 3)).astype(np.float32)
    write_classified_ply(ply, xyz, np.zeros((5, 3), np.uint8), np.ones(5, np.uint8))
    write_instance_product(ply, np.array([0, 1, 1, 0, 1], dtype=np.uint32), [{"id": 1}])
    assert "instance_id" in read_vertex_array(ply).dtype.names
    assert not list(tmp_path.glob("*.tmp"))
