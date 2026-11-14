"""Tests for the instance refinement stage: aliases, soft furnishings, merging, region growing."""

from __future__ import annotations

import time

import numpy as np

from samcloud.engine.instances import associate_instances, refine_instance_ids

FURNITURE = 17
WINDOW = 5
CLASS_NAMES = ["wall", "ceiling", "floor", "door", "window", "stairs", "railing", "column", "beam",
               "steel structure", "pipe", "cable", "electrical", "hvac", "ceiling equipment",
               "construction material", "furniture"]


def _box(rng, origin, size, spacing=0.012, jitter=0.004):
    """Noisy grid-like point set filling an axis aligned box ."""
    axes = [np.arange(0.0, extent, spacing) for extent in size]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid + np.asarray(origin) + rng.uniform(-jitter, jitter, grid.shape)


def _cloud(parts):
    """parts: list of (xyz, class_id, instance_id) -> concatenated arrays."""
    xyz = np.vstack([part[0] for part in parts])
    cls = np.concatenate([np.full(len(part[0]), part[1]) for part in parts])
    ids = np.concatenate([np.full(len(part[0]), part[2]) for part in parts]).astype(np.uint32)
    return xyz, cls, ids


def _meta(identifier, label, class_id=FURNITURE):
    return {"id": identifier, "class_id": class_id, "class_name": "furniture", "label": label,
            "view_count": 2, "confidence": 0.8}


def test_alias_maps_prompt_variant_to_canonical_label():
    rng = np.random.default_rng(1)
    xyz = _box(rng, (0.31, 0.12, 0.05), (0.3, 0.1, 0.3), spacing=0.02)
    n = len(xyz)
    assert n >= 300
    point_idx = np.concatenate([np.arange(n), np.arange(n)]).astype(np.int32)
    vinst_idx = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int32)
    ids, metadata = associate_instances(
        point_idx, vinst_idx, np.array([WINDOW, WINDOW]), np.array([0.9, 0.8]), np.array([0, 0]), n,
        np.full(n, WINDOW), xyz, {"min_points": 100, "class_names": CLASS_NAMES},
        ["casement window"], vinst_view=np.array([0, 1]))
    assert len(metadata) == 1
    assert metadata[0]["label"] == "window"
    assert metadata[0]["label_index"] == 1
    assert np.count_nonzero(ids) == n


def test_pillow_touching_bed_is_absorbed_far_pillow_is_not():
    rng = np.random.default_rng(2)
    bed = _box(rng, (0.0, 0.0, 0.0), (1.0, 0.7, 0.2))
    pillow = _box(rng, (0.1, 0.1, 0.213), (0.3, 0.3, 0.08))        # resting on the bed (1.3 cm gap)
    far = _box(rng, (3.07, 0.03, 0.0), (0.3, 0.3, 0.08))
    xyz, cls, ids = _cloud([(bed, FURNITURE, 1), (pillow, FURNITURE, 2), (far, FURNITURE, 3)])
    meta = [_meta(1, "bed"), _meta(2, "bed pillow"), _meta(3, "pillow")]
    out, records = refine_instance_ids(xyz, cls, ids, meta, {"split_radius_m": 0.05})
    n_bed, n_pillow = len(bed), len(pillow)
    assert len(np.unique(out[:n_bed + n_pillow])) == 1
    assert len(records) == 2
    labels = sorted(record["label"] for record in records)
    assert labels == ["bed", "pillow"]
    assert out[-1] != out[0]
    merged = next(record for record in records if record["label"] == "bed")
    assert merged["point_count"] == n_bed + n_pillow


def test_sofa_fragments_merge_but_chairs_touching_at_small_patch_do_not():
    rng = np.random.default_rng(3)
    left = _box(rng, (0.0, 0.0, 0.0), (0.4, 0.5, 0.4))
    right = _box(rng, (0.413, 0.0, 0.0), (0.4, 0.5, 0.4))             # broad face contact
    chair_a = _box(rng, (5.0, 0.0, 0.0), (0.3, 0.3, 0.3))
    chair_b = _box(rng, (5.313, 0.27, 0.27), (0.3, 0.3, 0.3))         # touches only at a ~3 cm corner
    xyz, cls, ids = _cloud([(left, FURNITURE, 1), (right, FURNITURE, 2),
                            (chair_a, FURNITURE, 3), (chair_b, FURNITURE, 4)])
    meta = [_meta(1, "sofa"), _meta(2, "sofa"), _meta(3, "chair"), _meta(4, "chair")]
    out, records = refine_instance_ids(xyz, cls, ids, meta, {"split_radius_m": 0.05})
    n_sofa = len(left) + len(right)
    assert len(np.unique(out[:n_sofa])) == 1
    chair_ids = {out[n_sofa], out[-1]}
    assert len(chair_ids) == 2 and 0 not in chair_ids
    assert sorted(record["label"] for record in records) == ["chair", "chair", "sofa"]
    assert [record["id"] for record in records] == [1, 2, 3]


def test_fill_grows_over_connected_same_class_points_only():
    rng = np.random.default_rng(4)
    blob = _box(rng, (0.0, 0.0, 0.0), (0.3, 0.3, 0.25))
    seeded = np.where(blob[:, 0] < 0.3 * 0.2 + 0.0, 1, 0)             # ~20 % of the blob is seeded
    other_blob = _box(rng, (1.3, 0.0, 0.0), (0.3, 0.3, 0.25))         # 1 m away, same class, no id
    foreign = _box(rng, (0.0, 0.31, 0.0), (0.3, 0.2, 0.25))           # touching, different class
    xyz = np.vstack([blob, other_blob, foreign])
    cls = np.concatenate([np.full(len(blob), FURNITURE), np.full(len(other_blob), FURNITURE),
                          np.full(len(foreign), 3)])
    ids = np.concatenate([seeded, np.zeros(len(other_blob) + len(foreign))]).astype(np.uint32)
    share = np.count_nonzero(ids) / len(blob)
    assert 0.1 < share < 0.35
    out, records = refine_instance_ids(xyz, cls, ids, [_meta(1, "sofa")], {"split_radius_m": 0.05})
    assert np.count_nonzero(out[:len(blob)]) >= 0.95 * len(blob)
    assert not out[len(blob):].any()
    assert records[0]["point_count"] == np.count_nonzero(out)
    assert records[0]["point_count"] >= 0.95 * len(blob)


def test_growth_stops_at_instance_grow_max():
    rng = np.random.default_rng(5)
    bar = _box(rng, (0.0, 0.0, 0.0), (2.0, 0.1, 0.1))
    ids = (bar[:, 0] < 0.1).astype(np.uint32)
    cls = np.full(len(bar), FURNITURE)
    out, _ = refine_instance_ids(bar, cls, ids, [_meta(1, "sofa")],
                                 {"split_radius_m": 0.05, "instance_grow_max_m": 0.3})
    assert np.all(out[bar[:, 0] < 0.25] == 1)
    assert not out[bar[:, 0] > 0.1 + 0.3 + 0.15].any()
    assert out[bar[:, 0] > 0.1 + 0.3 + 0.15].size > 0
    unlimited, _ = refine_instance_ids(bar, cls, ids, [_meta(1, "sofa")], {"split_radius_m": 0.05})
    assert np.count_nonzero(unlimited) < len(bar)            # default 0.6 m cap still holds
    assert np.all(unlimited[bar[:, 0] < 0.6] == 1)


def test_refine_performance_one_million_points():
    rng = np.random.default_rng(6)
    side = 27
    local = np.stack(np.meshgrid(*[np.arange(side) * 0.012] * 3, indexing="ij"), -1).reshape(-1, 3)
    blocks, ids, classes = [], [], []
    meta = []
    for index in range(50):
        origin = np.array([(index % 10) * 1.0, (index // 10) * 1.0, 0.0]) + rng.uniform(0.0, 0.05, 3)
        blocks.append(local + origin + rng.uniform(-0.003, 0.003, local.shape))
        ids.append(np.where(local[:, 0] < 0.3 * side * 0.012, index + 1, 0))
        classes.append(np.full(len(local), FURNITURE))
        meta.append(_meta(index + 1, "chair" if index % 2 else "casement window"))
    xyz = np.vstack(blocks)
    ids = np.concatenate(ids).astype(np.uint32)
    cls = np.concatenate(classes)
    assert len(xyz) > 980_000
    start = time.perf_counter()
    out, records = refine_instance_ids(xyz, cls, ids, meta, {"split_radius_m": 0.05})
    elapsed = time.perf_counter() - start
    print(f"refine 1M points / 50 instances: {elapsed:.2f} s")
    assert elapsed < 30.0
    assert len(records) == 50
    assert np.count_nonzero(out) > 0.95 * len(xyz)
