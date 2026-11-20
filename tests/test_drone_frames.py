"""Tests for samcloud.drone.frames and the LV95-aligned loader."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pyproj import Transformer

from samcloud.drone import frames
from samcloud.drone import io as dio
from samcloud.drone.contracts import LocalFrame

LAS = {"outdoor": [1, 50, 7, 33, 12]}
PROMPTS = {"outdoor": [["alpha"], ["beta"], ["gamma"], ["delta"]]}


def _classes_json(path):
    path.write_text(json.dumps({"las_codes": LAS, "prompts": PROMPTS}), encoding="utf-8")
    return path


def _write_ply(path, xyz, cls, normals=False):
    names = ["x", "y", "z"] + (["nx", "ny", "nz"] if normals else [])
    dtype = [(n, "<f4") for n in names] + [("c", "u1")]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}"]
    header += [f"property float {n}" for n in names] + ["property uchar scalar_class", "end_header"]
    data = np.zeros(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz.T
    if normals:
        data["nx"], data["ny"], data["nz"] = 0.0, 0.0, 1.0
    data["c"] = cls
    with open(path, "wb") as fh:
        fh.write((chr(10).join(header) + chr(10)).encode("ascii") + data.tobytes())


LAT, LON, H0 = 47.134, 8.755, 450.0
_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
_ECEF_TO_LV95 = Transformer.from_crs("EPSG:4978", "EPSG:2056+5728", always_xy=True)


def _georef():
    lat, lon = np.radians(LAT), np.radians(LON)
    east = [-np.sin(lon), np.cos(lon), 0.0]
    north = [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    up = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    rotation = np.array([east, north, up]).T
    origin = np.array(_TO_ECEF.transform(LON, LAT, H0))
    return {"scale": 1.0, "rotation": rotation.tolist(), "translation": origin.tolist(),
            "pairs": 5, "residual_m": {"mean": 0.0, "p95": 0.0, "max": 0.0}}


def _reference(xyz_enu):
    g = _georef()
    ecef = (np.asarray(xyz_enu) @ np.asarray(g["rotation"]).T) * g["scale"] + np.asarray(g["translation"])
    return np.column_stack(_ECEF_TO_LV95.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2]))


def _project(tmp_path, n=200, normals=False):
    rng = np.random.default_rng(3)
    project = tmp_path / "proj"
    products = project / "products"
    (products / "classified").mkdir(parents=True)
    (products / "georeference.json").write_text(json.dumps(_georef()), encoding="utf-8")
    xyz = rng.uniform(-150, 150, (n, 3)) * [1, 1, 0.1]
    _write_ply(products / "classified" / "classified.ply", xyz.astype(np.float32),
               (np.arange(n) % 5).astype(np.uint8), normals)
    return products, _classes_json(tmp_path / "classes.json")


def test_enu_east_is_rotated_in_lv95_and_loader_matches_pyproj(tmp_path):
    g = _georef()
    pts = frames.enu_to_lv95(np.array([[0, 0, 0], [100.0, 0, 0]]), g)
    vec = pts[1] - pts[0]
    bearing = np.degrees(np.arctan2(vec[0], vec[1]))
    assert 0.7 < abs(bearing - 90.0) < 1.2
    assert abs(np.linalg.norm(vec[:2]) - 100.0) > 1e-4  # scale differs too
    print(f"bearing diff {bearing - 90.0:.4f} deg, scale {np.linalg.norm(vec[:2]) / 100.0:.7f}")

    products, cj = _project(tmp_path)
    ps, frame = dio.load_classified_pointset_lv95(products, classes_json=cj)
    enu = dio.load_classified_pointset(products, classes_json=cj).xyz
    assert np.abs(frame.to_lv95(ps.xyz) - _reference(enu)).max() < 1e-3
    assert abs(np.abs(ps.xyz[:, :2]).max()) < 500 and frame.origin_lv95[0] % 100 == 0
    single = dio.load_classified_pointset(products, classes_json=cj, frame="lv95_local")
    assert np.allclose(single.xyz, ps.xyz)


def test_local_frame_round_trip_exact():
    frame = frames.make_local_frame(np.array([[2_700_123.4, 1_200_987.6, 451.7], [2_700_200.0, 1_201_000.0, 452.0]]))
    assert frame.origin_lv95 == (2_700_100.0, 1_200_900.0, 451.0)
    xyz = np.random.default_rng(1).uniform(-300, 300, (50, 3))
    assert np.array_equal(frames.to_local(frames.to_lv95(xyz, frame), frame), xyz) or \
        np.abs(frames.to_local(frames.to_lv95(xyz, frame), frame) - xyz).max() < 1e-9
    assert isinstance(frame, LocalFrame) and frame.epsg == 2056


def test_normals_rotated_consistently(tmp_path):
    products, cj = _project(tmp_path, normals=True)
    # facade normal pointing ENU-east: replace stored normals by (1, 0, 0)
    import open3d as o3d
    ply = products / "classified" / "classified.ply"
    cloud = o3d.t.io.read_point_cloud(str(ply))
    cloud.point.normals = o3d.core.Tensor(np.tile([1.0, 0.0, 0.0], (len(cloud.point.positions), 1)).astype(np.float32))
    o3d.t.io.write_point_cloud(str(ply), cloud)
    ps, _ = dio.load_classified_pointset_lv95(products, classes_json=cj)
    n = ps.normals[0]
    assert np.isclose(np.linalg.norm(n), 1.0)
    pts = frames.enu_to_lv95(np.array([[0, 0, 0], [100.0, 0, 0]]), _georef())
    expected = np.degrees(np.arctan2(*(pts[1] - pts[0])[:2]))
    assert abs(np.degrees(np.arctan2(n[0], n[1])) - expected) < 0.01
    assert abs(expected - 90.0) > 0.5 and abs(n[2]) < 1e-3


def test_old_loader_unchanged(tmp_path):
    products, cj = _project(tmp_path, normals=True)
    ps = dio.load_classified_pointset(products, classes_json=cj)
    assert isinstance(ps.xyz, np.ndarray) and np.abs(ps.xyz).max() < 151
    assert np.allclose(ps.normals, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        dio.load_classified_pointset(products, classes_json=cj, frame="bogus")


def test_chunking_identical():
    xyz = np.random.default_rng(5).uniform(-100, 100, (1003, 3))
    g = _georef()
    assert np.array_equal(frames.enu_to_lv95(xyz, g, chunk=100), frames.enu_to_lv95(xyz, g, chunk=10**7))


def test_frame_report_values():
    xyz = np.random.default_rng(7).uniform(-200, 200, (100, 3))
    rep = frames.frame_report(xyz, _georef())
    assert 0.7 < abs(rep["bearing_diff_deg"]) < 1.2
    assert abs(rep["bearing_diff_deg"] - rep["diagonal_bearing_diff_deg"]) < 0.01
    assert rep["max_offset_m"] > 1.0 and rep["max_nonlinearity_m"] < 0.05
    print(rep)
