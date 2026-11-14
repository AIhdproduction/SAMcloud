"""AprilTag squares never form or extend object instances (Paket D)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from samcloud.dxf.geometry.features import furniture_inserts
from samcloud.engine.instances import (
    SOURCE_NAME,
    attach_instances,
    refine_instance_ids,
    snapshot_instances,
    tag_point_mask,
    write_instance_product,
)
from samcloud.engine.ply_io import read_vertex_array, write_classified_ply

WALL = 2
FURNITURE = 17
STRAY_ID = 9
CHAIR_ID = 3
# Tag on the wall plane x = 0, centre (0, 1.0, 1.5), 16 cm square.
TAG_CORNERS = [[0.0, 0.92, 1.42], [0.0, 1.08, 1.42], [0.0, 1.08, 1.58], [0.0, 0.92, 1.58]]
TAG_CENTRE = np.array([0.0, 1.0, 1.5])


def _report(path: Path, with_tag: bool = True) -> Path:
    tags = [{"id": 2, "status": "insufficient", "images": 1}]
    if with_tag:
        tags.append({"id": 1, "status": "ok", "center_xyz": TAG_CENTRE.tolist(), "corners_xyz": TAG_CORNERS,
                     "size_mm_after": 160.0, "images": 5})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "scale", "tags": tags, "warnings": []}), encoding="utf-8")
    return path


def _scene():
    """Noisy wall + chair + tag points carrying stray ids (9 only on tag points, 3 on five of them)."""
    rng = np.random.default_rng(77)
    axes = [np.arange(0.0, 3.0, 0.02), np.arange(0.0, 2.5, 0.02)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    wall = np.column_stack((rng.uniform(-0.004, 0.004, len(grid)), grid[:, 0] + rng.uniform(-0.004, 0.004, len(grid)),
                            grid[:, 1] + rng.uniform(-0.004, 0.004, len(grid))))
    cell = np.arange(0.0, 0.5, 0.03)
    chair = np.stack(np.meshgrid(cell, cell, np.arange(0.0, 0.8, 0.03), indexing="ij"), axis=-1).reshape(-1, 3)
    chair = chair + np.array([2.0, 1.0, 0.0]) + rng.uniform(-0.005, 0.005, chair.shape)
    xyz = np.vstack((wall, chair))
    point_class = np.concatenate((np.full(len(wall), WALL), np.full(len(chair), FURNITURE)))
    ids = np.concatenate((np.zeros(len(wall)), np.full(len(chair), CHAIR_ID))).astype(np.uint32)
    on_tag = (np.abs(wall[:, 1] - 1.0) <= 0.08) & (np.abs(wall[:, 2] - 1.5) <= 0.08)
    tag_index = np.flatnonzero(on_tag)
    assert len(tag_index) > 30
    ids[tag_index] = STRAY_ID
    ids[tag_index[:5]] = CHAIR_ID
    return xyz, point_class, ids, tag_index, len(wall)


def test_tag_points_get_zero_and_chair_keeps_its_id(tmp_path: Path) -> None:
    xyz, point_class, ids, tag_index, wall_count = _scene()
    mask = tag_point_mask(xyz, _report(tmp_path / "reports" / "apriltag_report.json"))
    assert mask is not None and mask[tag_index].all()
    assert not mask[wall_count:].any()  # chair untouched
    # Wall points well outside the square (beyond the 2 cm margin) stay out of the mask.
    far = np.flatnonzero((np.abs(xyz[:wall_count, 1] - 1.0) > 0.12) | (np.abs(xyz[:wall_count, 2] - 1.5) > 0.12))
    assert not mask[far].any()

    baseline_ids, baseline_meta = refine_instance_ids(xyz, point_class, ids, [], {})
    assert len(baseline_meta) == 2  # without the mask the stray tag-only instance exists

    refined, metadata = refine_instance_ids(xyz, point_class, ids, [], {}, mask)
    assert np.all(refined[tag_index] == 0)
    chair_ids = np.unique(refined[wall_count:])
    assert chair_ids.tolist() == [1]
    assert len(metadata) == 1
    assert metadata[0]["class_id"] == FURNITURE
    assert metadata[0]["point_count"] == len(xyz) - wall_count
    assert int(np.count_nonzero(refined)) == len(xyz) - wall_count  # growth did not reclaim tag points


def test_without_report_output_is_identical(tmp_path: Path) -> None:
    xyz, point_class, ids, _tag_index, _wall_count = _scene()
    assert tag_point_mask(xyz, tmp_path / "missing" / "apriltag_report.json") is None
    assert tag_point_mask(xyz, None) is None
    # A report whose tags are all insufficient is equivalent to no report.
    assert tag_point_mask(xyz, _report(tmp_path / "r" / "apriltag_report.json", with_tag=False)) is None
    reference_ids, reference_meta = refine_instance_ids(xyz, point_class, ids, [], {})
    again_ids, again_meta = refine_instance_ids(xyz, point_class, ids, [], {}, tag_point_mask(xyz, tmp_path / "missing.json"))
    assert np.array_equal(reference_ids, again_ids)
    assert reference_meta == again_meta


def test_dxf_furniture_inserts_have_no_insert_at_tag(tmp_path: Path) -> None:
    xyz, point_class, ids, tag_index, wall_count = _scene()
    mask = tag_point_mask(xyz, _report(tmp_path / "reports" / "apriltag_report.json"))

    def inserts_for(final_ids):
        selected = (point_class == FURNITURE) | (final_ids > 0)  # inputs of the labelled furniture layer
        return furniture_inserts(xyz[selected], point_class[selected].astype(np.uint8),
                                 instance_ids=final_ids[selected], instance_labels={1: "chair", 2: "chair", STRAY_ID: "tag"})

    def near_tag(inserts):
        return [item for item in inserts if np.hypot(item["point"][0] - TAG_CENTRE[0], item["point"][1] - TAG_CENTRE[1]) < 0.4]

    before_ids, _ = refine_instance_ids(xyz, point_class, ids, [], {})
    assert near_tag(inserts_for(before_ids))  # the stray id would have produced an insert on the wall
    after_ids, _ = refine_instance_ids(xyz, point_class, ids, [], {}, mask)
    inserts = inserts_for(after_ids)
    assert near_tag(inserts) == []
    assert len(inserts) == 1 and inserts[0]["attributes"]["instance"] == 1


def test_attach_instances_reads_report_and_writes_clean_json(tmp_path: Path) -> None:
    xyz, point_class, ids, tag_index, wall_count = _scene()
    project = tmp_path / "project"
    classified = project / "products" / "classified"
    classified.mkdir(parents=True)
    rgb = np.full((len(xyz), 3), 120, dtype=np.uint8)
    source_ply = classified / "classified.ply"
    write_classified_ply(source_ply, xyz.astype(np.float32), rgb, point_class.astype(np.uint8))
    write_instance_product(source_ply, ids, [])
    assert snapshot_instances(source_ply)
    snapshot = classified / SOURCE_NAME
    assert snapshot.is_file()

    without = attach_instances(snapshot, source_ply, {"min_points": 10})
    assert without["enabled"] and without["count"] == 2

    _report(project / "reports" / "apriltag_report.json")
    report = attach_instances(snapshot, source_ply, {"min_points": 10})
    assert report["count"] == 1
    result_ids = np.asarray(read_vertex_array(source_ply)["instance_id"])
    assert np.all(result_ids[tag_index] == 0)
    assert np.all(result_ids[wall_count:] == 1)
    stored = json.loads((classified / "instances.json").read_text(encoding="utf-8"))["instances"]
    assert [item["id"] for item in stored] == [1]
