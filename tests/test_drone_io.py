"""Tests for samcloud.drone.io (synthetic PLY/npz/classes.json)."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from samcloud.drone import io as dio

LAS = {"outdoor": [1, 50, 7, 33, 12]}  # internal 0..4, deliberately not monotonic
PROMPTS = {"outdoor": [["alpha", "a1"], ["beta"], ["gamma"], ["delta"]]}


def _classes_json(path):
    path.write_text(json.dumps({"las_codes": LAS, "prompts": PROMPTS}), encoding="utf-8")
    return path


def _write_ply(path, xyz, cls, normals=False):
    props = "property float x\nproperty float y\nproperty float z\n"
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals:
        props += "property float nx\nproperty float ny\nproperty float nz\n"
        dtype += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    props += "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar scalar_class\n"
    dtype += [("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n{props}end_header\n"
    data = np.zeros(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz.T
    if normals:
        data["nx"], data["ny"], data["nz"] = 0.0, 0.0, 1.0
    data["r"], data["g"], data["b"], data["c"] = 10, 20, 30, cls
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii") + data.tobytes())


def _project(tmp_path, n=100, scores_n=None, normals=False):
    rng = np.random.default_rng(0)
    products = tmp_path / "proj" / "products"
    (products / "classified").mkdir(parents=True)
    xyz = rng.uniform(0, 5, (n, 3))
    cls = (np.arange(n) % 5).astype(np.uint8)
    _write_ply(products / "classified" / "classified.ply", xyz, cls, normals)
    if scores_n is not None:
        top1 = rng.uniform(0.5, 1.0, scores_n).astype(np.float16)
        np.savez(products / "classified" / "classified_scores.npz", top1_conf=top1,
                 top2_conf=(top1 * 0.5).astype(np.float16),
                 view_count=np.arange(scores_n, dtype=np.uint16))
    return products, cls, _classes_json(tmp_path / "classes.json")


def test_mapping_uses_las_codes_in_json_order(tmp_path):
    products, cls, cj = _project(tmp_path)
    ps = dio.load_classified_pointset(products, classes_json=cj)
    assert ps.classes.dtype == np.uint16
    assert np.array_equal(ps.classes, np.asarray(LAS["outdoor"], dtype=np.uint16)[cls])
    assert ps.rgb is not None and tuple(ps.rgb[0]) == (10, 20, 30)
    assert ps.confidence is None and ps.normals is None


def test_scores_merged_and_normals(tmp_path):
    products, _, cj = _project(tmp_path, scores_n=100, normals=True)
    ps = dio.load_classified_pointset(products, classes_json=cj)
    assert ps.confidence is not None and ps.confidence.dtype == np.float32
    assert np.allclose(ps.margin, ps.confidence - ps.confidence * 0.5, atol=2e-3)
    assert np.array_equal(ps.view_count, np.arange(100, dtype=np.uint16))
    assert ps.normals.shape == (100, 3) and np.allclose(ps.normals[:, 2], 1.0)


def test_score_size_mismatch_gives_none_and_warning(tmp_path, caplog):
    products, _, cj = _project(tmp_path, scores_n=77)
    with caplog.at_level(logging.WARNING, logger=dio.LOGGER.name):
        ps = dio.load_classified_pointset(products, classes_json=cj)
    assert ps.confidence is None and ps.margin is None and ps.view_count is None
    assert any("does not match" in r.message for r in caplog.records)


def test_stride_is_deterministic_and_bounded(tmp_path):
    products, cls, cj = _project(tmp_path, n=100, scores_n=100)
    a = dio.load_classified_pointset(products, classes_json=cj, max_points=30)
    b = dio.load_classified_pointset(products, classes_json=cj, max_points=30)
    assert len(a) <= 30 and len(a) == len(b)
    assert np.array_equal(a.xyz, b.xyz)
    assert len(a.confidence) == len(a) == len(a.view_count) == len(a.classes)
    assert np.array_equal(a.view_count, np.arange(0, 100, 4, dtype=np.uint16))


def test_class_code_and_internal_to_las(tmp_path):
    cj = _classes_json(tmp_path / "c.json")
    assert dio.class_code("Gamma", classes_json=cj) == 33
    assert dio.class_code("a1", classes_json=cj) == 50
    with pytest.raises(KeyError):
        dio.class_code("nope", classes_json=cj)
    out = dio.internal_to_las(np.array([0, 4, 9, 255]), classes_json=cj)
    assert out.tolist() == [1, 12, 1, 0]


def test_global_config_building_is_las_6():
    assert dio.class_code("building") == 6
    assert dio.class_code("road") == 11
    assert dio.class_code("ground") == 2


def test_default_frame_is_enu_unchanged(tmp_path):
    products, _, cj = _project(tmp_path)
    a = dio.load_classified_pointset(products, classes_json=cj)
    b = dio.load_classified_pointset(products, classes_json=cj, frame="enu")
    assert np.array_equal(a.xyz, b.xyz) and isinstance(a, dio.PointSet)
    with pytest.raises(ValueError):
        dio.load_classified_pointset(products, classes_json=cj, frame="nope")
