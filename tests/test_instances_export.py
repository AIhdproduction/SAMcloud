"""Tests for PLY instance_id property writing, LAS export extra dimension, and instances.json schema."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from samcloud.engine import classify
from samcloud.engine.export_las import export_las
from samcloud.engine.instances import (
    JSON_NAME,
    write_instance_product,
    write_instances_json,
)
from samcloud.engine.ply_io import read_vertex_array, write_classified_ply

CLASSES_JSON = Path(__file__).resolve().parents[1] / "config" / "classes.json"


def test_classified_ply_has_uint32_instance_id_and_retains_properties(tmp_path: Path) -> None:
    """write_instance_product adds uint32 instance_id to PLY and keeps x, y, z, rgb, scalar_class and fine_term."""
    n_points = 50
    rng = np.random.default_rng(606)

    xyz = rng.uniform(-1.0, 1.0, (n_points, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, (n_points, 3), dtype=np.uint8)
    scalar_class = np.full(n_points, 17, dtype=np.uint8)
    fine_term = rng.integers(0, 5, n_points, dtype=np.uint8)

    # 1. Standard classified PLY with x, y, z, red, green, blue, scalar_class
    ply_standard = tmp_path / "classified_standard.ply"
    write_classified_ply(ply_standard, xyz, rgb, scalar_class)

    instance_id = np.arange(1, n_points + 1, dtype=np.uint32)
    metadata = [{"id": 1, "class_id": 17, "class_name": "furniture", "label": "chair", "label_index": 1,
                 "point_count": n_points, "centroid": xyz.mean(axis=0).tolist(),
                 "bbox_min": xyz.min(axis=0).tolist(), "bbox_max": xyz.max(axis=0).tolist(),
                 "view_count": 2, "confidence": 0.85}]

    write_instance_product(ply_standard, instance_id, metadata)

    vertex_std = read_vertex_array(ply_standard)
    assert "instance_id" in vertex_std.dtype.names
    assert vertex_std["instance_id"].dtype == np.uint32
    assert np.array_equal(vertex_std["instance_id"], instance_id)
    assert np.allclose(vertex_std["x"], xyz[:, 0], atol=1e-6)
    assert np.allclose(vertex_std["y"], xyz[:, 1], atol=1e-6)
    assert np.allclose(vertex_std["z"], xyz[:, 2], atol=1e-6)
    assert np.array_equal(vertex_std["red"], rgb[:, 0])
    assert np.array_equal(vertex_std["green"], rgb[:, 1])
    assert np.array_equal(vertex_std["blue"], rgb[:, 2])
    assert np.array_equal(vertex_std["scalar_class"], scalar_class)

    # 2. Classified PLY that also includes fine_term property
    ply_fine = tmp_path / "classified_fine.ply"
    dtype_with_fine = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("scalar_class", "u1"), ("fine_term", "u1"),
    ]
    data_fine = np.empty(n_points, dtype=dtype_with_fine)
    data_fine["x"], data_fine["y"], data_fine["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data_fine["red"], data_fine["green"], data_fine["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    data_fine["scalar_class"] = scalar_class
    data_fine["fine_term"] = fine_term
    PlyData([PlyElement.describe(data_fine, "vertex")], text=False).write(str(ply_fine))

    write_instance_product(ply_fine, instance_id, metadata)

    vertex_fine = read_vertex_array(ply_fine)
    assert "instance_id" in vertex_fine.dtype.names
    assert "fine_term" in vertex_fine.dtype.names
    assert vertex_fine["instance_id"].dtype == np.uint32
    assert np.array_equal(vertex_fine["instance_id"], instance_id)
    assert np.array_equal(vertex_fine["fine_term"], fine_term)
    assert np.array_equal(vertex_fine["scalar_class"], scalar_class)


def test_las_export_contains_instance_id_extra_dim(tmp_path: Path) -> None:
    """export_las writes extra dim instance_id (uint32) if and only if the PLY has the property."""
    n_points = 30
    rng = np.random.default_rng(707)

    xyz = rng.uniform(-1.0, 1.0, (n_points, 3)).astype(np.float32)
    rgb = np.full((n_points, 3), 120, dtype=np.uint8)
    scalar_class = np.full(n_points, 17, dtype=np.uint8)  # class 17 is furniture in indoor
    instance_id = np.array([1] * 15 + [2] * 15, dtype=np.uint32)

    # 1. PLY with instance_id
    ply_with_id = tmp_path / "with_instances.ply"
    las_with_id = tmp_path / "with_instances.las"
    write_classified_ply(ply_with_id, xyz, rgb, scalar_class)
    write_instance_product(ply_with_id, instance_id, [])

    count = export_las(ply_with_id, las_with_id, CLASSES_JSON, 10, "LOCAL", "LOCAL", "indoor")
    assert count == n_points

    las = laspy.read(las_with_id)
    assert "instance_id" in las.point_format.extra_dimension_names
    assert np.array_equal(np.asarray(las.instance_id), instance_id)
    assert np.asarray(las.instance_id).dtype == np.uint32

    # 2. PLY without instance_id
    ply_without_id = tmp_path / "without_instances.ply"
    las_without_id = tmp_path / "without_instances.las"
    write_classified_ply(ply_without_id, xyz, rgb, scalar_class)

    export_las(ply_without_id, las_without_id, CLASSES_JSON, 10, "LOCAL", "LOCAL", "indoor")

    las_no = laspy.read(las_without_id)
    assert "instance_id" not in las_no.point_format.extra_dimension_names


def test_instances_json_schema(tmp_path: Path) -> None:
    """instances.json matches schema_version 1 and contains all mandatory record fields."""
    metadata = [
        {
            "id": 1,
            "class_id": 17,
            "class_name": "furniture",
            "label": "chair",
            "label_index": 1,
            "point_count": 312,
            "centroid": [0.12, -0.45, 0.78],
            "bbox_min": [-0.10, -0.60, 0.50],
            "bbox_max": [0.35, -0.30, 1.05],
            "view_count": 4,
            "confidence": 0.89,
        },
        {
            "id": 2,
            "class_id": 17,
            "class_name": "furniture",
            "label": "chair",
            "label_index": 2,
            "point_count": 280,
            "centroid": [1.12, -0.45, 0.78],
            "bbox_min": [0.90, -0.60, 0.50],
            "bbox_max": [1.35, -0.30, 1.05],
            "view_count": 3,
            "confidence": 0.83,
        },
    ]

    json_path = tmp_path / JSON_NAME
    write_instances_json(json_path, metadata)

    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert "instances" in payload
    instances = payload["instances"]
    assert len(instances) == 2

    expected_keys = {
        "id", "class_id", "class_name", "label", "label_index",
        "point_count", "centroid", "bbox_min", "bbox_max", "view_count", "confidence",
    }

    for item in instances:
        assert set(item.keys()) == expected_keys
        assert isinstance(item["id"], int) and item["id"] >= 1
        assert isinstance(item["class_id"], int)
        assert isinstance(item["class_name"], str) and len(item["class_name"]) > 0
        assert isinstance(item["label"], str) and len(item["label"]) > 0
        assert isinstance(item["label_index"], int) and item["label_index"] >= 1
        assert isinstance(item["point_count"], int) and item["point_count"] > 0
        assert isinstance(item["centroid"], list) and len(item["centroid"]) == 3
        assert all(isinstance(v, (float, int)) for v in item["centroid"])
        assert isinstance(item["bbox_min"], list) and len(item["bbox_min"]) == 3
        assert all(isinstance(v, (float, int)) for v in item["bbox_min"])
        assert isinstance(item["bbox_max"], list) and len(item["bbox_max"]) == 3
        assert all(isinstance(v, (float, int)) for v in item["bbox_max"])
        assert isinstance(item["view_count"], int) and item["view_count"] >= 1
        assert isinstance(item["confidence"], (float, int)) and 0.0 <= item["confidence"] <= 1.0


class _MockClassifyLabeler:
    """Minimal labeler stand-in for classify.run_segearth_ov3."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names

    @staticmethod
    def load_image(_path):
        return np.zeros((16, 16, 3), dtype=np.uint8), 16, 16

    def score_points(self, _path, u, _v, instances_out=None, **_kwargs):
        probs = np.zeros((len(u), len(self.class_names)), dtype=np.float32)
        probs[:, 0] = 0.95  # first class
        if instances_out is not None:
            instances_out["point_local_instance"] = np.zeros(len(u), dtype=np.int32)
            instances_out["instances"] = [{
                "class_id": 1,
                "prompt_idx": 0,
                "conf": 0.90,
                "pixel_count": 25,
            }]
        return probs


def _run_mock_segearth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
                       instance_config: dict | None, no_instances_flag: bool = False) -> Path:
    class_names = ["furniture", "wall"]
    monkeypatch.setattr(classify, "make_point_labeler", lambda *_a, **_k: _MockClassifyLabeler(class_names))
    monkeypatch.setattr(classify, "project_points", lambda *_a, **_k: (
        np.arange(6), np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64), np.full(6, 2.0),
    ))

    out_ply = tmp_path / f"{name}.ply"
    args = SimpleNamespace(
        background_threshold=0.4,
        threshold_sweep="",
        dense_dir=str(tmp_path),
        labels_dir=None,
        images_dir=str(tmp_path),
        depth_check=0,
        out=str(out_ply),
        scores_out=None,
        min_votes=1,
        summary_file=None,
        class_set="indoor",
        engine="sam3",
        mode="segearth_ov3",
        query_threshold=0.4,
        no_instances=no_instances_flag,
    )

    xyz = np.column_stack([np.linspace(-0.5, 0.5, 6), np.zeros(6), np.zeros(6)]).astype(np.float64)
    rgb = np.full((6, 3), 128, dtype=np.uint8)
    items = [(view, [0, 1, 2, 3, 4, 5]) for view in (1, 2, 3)]
    images = {view: SimpleNamespace(name=f"view{view}.png", camera_id=1) for view in (1, 2, 3)}
    fine_config = {
        "instance_classes": {
            "indoor": ["furniture"],
        },
    }

    classify.run_segearth_ov3(
        args=args,
        xyz=xyz,
        rgb=rgb,
        cameras={1: object()},
        images=images,
        image_items=items,
        class_names=class_names,
        unknown_id=0,
        progress_path=None,
        prompts=[["furniture", "chair"], ["wall"]],
        fine_config=fine_config,
        instance_config=instance_config,
    )
    return out_ply


def test_instances_disabled_produces_no_instance_property_and_no_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With instances disabled (enabled false / no_instances), no instance_id property and no instances.json are produced."""
    # Case 1: instance_config with enabled: False
    ply_disabled = _run_mock_segearth(tmp_path, monkeypatch, "disabled",
                                      instance_config={"enabled": False, "class_names": ["furniture"]})
    assert ply_disabled.is_file()
    vertex_disabled = read_vertex_array(ply_disabled)
    assert "instance_id" not in vertex_disabled.dtype.names
    assert not ply_disabled.with_name(JSON_NAME).exists()

    # Case 2: args.no_instances = True, even though instance_config has enabled: True
    ply_no_inst = _run_mock_segearth(tmp_path, monkeypatch, "no_instances",
                                     instance_config={"enabled": True, "class_names": ["furniture"], "min_views": 1, "min_points": 1},
                                     no_instances_flag=True)
    assert ply_no_inst.is_file()
    vertex_no_inst = read_vertex_array(ply_no_inst)
    assert "instance_id" not in vertex_no_inst.dtype.names
    assert not ply_no_inst.with_name(JSON_NAME).exists()

    # Case 3: instance_config is None
    ply_none = _run_mock_segearth(tmp_path, monkeypatch, "none_config", instance_config=None)
    assert ply_none.is_file()
    vertex_none = read_vertex_array(ply_none)
    assert "instance_id" not in vertex_none.dtype.names
    assert not ply_none.with_name(JSON_NAME).exists()

    # Case 4: Control case - instances enabled produces both property and instances.json
    ply_enabled = _run_mock_segearth(tmp_path, monkeypatch, "enabled",
                                     instance_config={"enabled": True, "class_names": ["furniture"], "min_views": 1, "min_points": 1},
                                     no_instances_flag=False)
    assert ply_enabled.is_file()
    vertex_enabled = read_vertex_array(ply_enabled)
    assert "instance_id" in vertex_enabled.dtype.names
    assert ply_enabled.with_name(JSON_NAME).exists()
