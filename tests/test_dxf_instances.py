"""Behavioural coverage for optional point-cloud instance identifiers."""

from __future__ import annotations

from pathlib import Path

import ezdxf
import laspy
import numpy as np

from samcloud.dxf.geometry.features import furniture_inserts
from samcloud.dxf.las_source import Frame, PlyPointChunk, iter_las_chunks
from samcloud.dxf.pipeline import _PointAccumulator
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import build_dxf_document


def _touching_chairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(44)
    first = np.column_stack((rng.uniform(0.02, 0.24, 20), rng.uniform(0.02, 0.24, 20), rng.uniform(0.45, 0.50, 20)))
    second = np.column_stack((rng.uniform(0.23, 0.45, 20), rng.uniform(0.02, 0.24, 20), rng.uniform(0.45, 0.50, 20)))
    points = np.vstack((first, second))
    return points, np.full(len(points), 119, dtype=np.uint8), np.repeat(np.asarray((1, 2), dtype=np.uint32), 20)


def test_touching_chairs_split_by_instance_id() -> None:
    points, codes, ids = _touching_chairs()
    inserts = furniture_inserts(points, codes, instance_ids=ids, instance_labels={1: "chair", 2: "chair"})
    assert len(inserts) == 2
    assert {int(item["attributes"]["instance"]) for item in inserts} == {1, 2}
    assert {item["attributes"]["label"] for item in inserts} == {"chair"}


def test_touching_chairs_without_instance_ids_use_existing_cluster_path() -> None:
    points, codes, _ids = _touching_chairs()
    legacy = furniture_inserts(points, codes)
    explicit_zero = furniture_inserts(points, codes, instance_ids=np.zeros(len(points), dtype=np.uint32))
    assert len(legacy) == 1
    assert explicit_zero == legacy


def test_ply_chunk_and_las_extra_dimension_expose_instance_ids(tmp_path: Path) -> None:
    xyz = np.asarray(((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)))
    ply = PlyPointChunk(xyz, np.asarray((119, 119), dtype=np.uint8), np.full((2, 3), 80, dtype=np.uint8), np.asarray((7, 8), dtype=np.uint32))
    missing_ply = PlyPointChunk(xyz, np.asarray((119, 119), dtype=np.uint8), np.full((2, 3), 80, dtype=np.uint8))
    assert np.array_equal(ply.instance_id, np.asarray((7, 8), dtype=np.uint32))
    assert np.array_equal(missing_ply.instance_id, np.zeros(2, dtype=np.uint32))

    path = tmp_path / "instances.las"
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.uint32))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cloud.classification = np.asarray((119, 119), dtype=np.uint8)
    cloud.instance_id = np.asarray((7, 8), dtype=np.uint32)
    cloud.write(path)
    loaded = next(iter_las_chunks(path, 100))
    assert np.array_equal(np.asarray(loaded.instance_id, dtype=np.uint32), np.asarray((7, 8), dtype=np.uint32))


def test_furniture_block_writes_instance_and_label_attributes(tmp_path: Path) -> None:
    result = DxfPlanResult(
        "test", "site", "indoor", None, 1.0, Frame(np.zeros(3)), "test",
        {"A-FURN": [{"type": "insert", "name": "FURNITURE", "point": [0.0, 0.0], "attributes": {"class": "119", "height": "0.50", "instance": 3, "label": "chair"}}]},
        {}, {"A-FURN": 7}, BackgroundImage(None, None, (0.0, 0.0, 1.0, 1.0)),
    )
    document = build_dxf_document(result)
    path = tmp_path / "instance-attributes.dxf"
    document.saveas(path)
    insert = next(entity for entity in ezdxf.readfile(path).modelspace() if entity.dxftype() == "INSERT")
    assert {attribute.dxf.tag: attribute.dxf.text for attribute in insert.attribs}["instance"] == "3"
    assert {attribute.dxf.tag: attribute.dxf.text for attribute in insert.attribs}["label"] == "chair"


def test_voxel_thinning_keeps_modal_instance_id() -> None:
    accumulator = _PointAccumulator(0.5)
    points = np.asarray(((0.11, 0.12, 0.13), (0.14, 0.15, 0.16), (0.17, 0.18, 0.19)))
    accumulator.add(points, np.full(3, 119, dtype=np.uint8), np.asarray((4, 4, 9), dtype=np.uint32))
    kept, codes, ids = accumulator.points_for((119,), with_instance_ids=True)
    assert kept.shape == (1, 3)
    assert np.array_equal(codes, np.asarray((119,), dtype=np.uint8))
    assert np.array_equal(ids, np.asarray((4,), dtype=np.uint32))
