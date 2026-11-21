from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from samcloud.drone import run_environment as re_mod


def _json_chunk(data: bytes) -> dict:
    length = struct.unpack("<I", data[12:16])[0]
    text = data[20:20 + length].decode("utf-8")

    def _raise(token):
        raise AssertionError(f"non-finite constant in GLB JSON: {token}")

    return json.loads(text, parse_constant=_raise)


def _frame():
    return SimpleNamespace(origin_lv95=(2600000.0, 1200000.0, 400.0), epsg=2056, vertical_datum="LN02")


def test_dsm_with_nan_cells_writes_finite_glb(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    dsm = np.arange(36, dtype="float32").reshape(6, 6) + 400
    dsm[2:4, 2:4] = -9999
    path = tmp_path / "dsm.tif"
    with rasterio.open(path, "w", driver="GTiff", height=6, width=6, count=1, dtype="float32",
                       transform=from_origin(2600000, 1200010, 1, 1), nodata=-9999) as dst:
        dst.write(dsm, 1)
    frame = SimpleNamespace(from_lv95=lambda world: np.asarray(world, float) - np.array([2600000.0, 1200000.0, 400.0]),
                            **vars(_frame()))
    mesh = re_mod._dsm_mesh(path, frame)
    assert mesh is not None
    assert np.isfinite(mesh[0]).all()
    assert mesh[1].max() < len(mesh[0])

    glb = tmp_path / "environment.glb"
    re_mod._write_environment_glb({"dsm-surface": (*mesh, "dsm")}, glb, frame)
    document = _json_chunk(glb.read_bytes())
    assert document["meshes"]
    import trimesh
    loaded = trimesh.load(glb, file_type="glb", force="mesh")
    assert np.isfinite(loaded.vertices).all()


def test_writer_cleans_nan_vertices_and_skips_all_nan_nodes(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan, 1, 1], [5, 5, 5]], float)
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    bad = (np.full((3, 3), np.nan), np.array([[0, 1, 2]]))
    glb = tmp_path / "environment.glb"
    re_mod._write_environment_glb({"mixed": (vertices, faces, "x"), "allnan": (*bad, "y")}, glb, _frame())
    document = _json_chunk(glb.read_bytes())
    names = [node.get("name") for node in document["nodes"]]
    assert "mixed" in names and "allnan" not in names
    import trimesh
    loaded = trimesh.load(glb, file_type="glb", force="mesh")
    assert len(loaded.faces) == 1 and np.isfinite(loaded.vertices).all()
