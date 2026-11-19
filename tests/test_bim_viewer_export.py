"""Viewer export: parse the written GLB back and compare it with the source IFC."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import ifcopenshell
import numpy as np
import pytest

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene
from samcloud.bim.viewer_export import export_viewer_assets

COMPONENT_DTYPES = {5126: np.float32, 5125: np.uint32}


def _read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(data)
    assert length % 4 == 0
    offset = 12
    chunks: list[tuple[int, bytes]] = []
    while offset < length:
        size, kind = struct.unpack("<II", data[offset : offset + 8])
        assert size % 4 == 0
        chunks.append((kind, data[offset + 8 : offset + 8 + size]))
        offset += 8 + size
    assert offset == length
    assert [kind for kind, _ in chunks] == [0x4E4F534A, 0x004E4942]
    return json.loads(chunks[0][1].decode("ascii")), chunks[1][1]


def _accessor(document: dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    width = {"VEC3": 3, "SCALAR": 1}[accessor["type"]]
    dtype = COMPONENT_DTYPES[accessor["componentType"]]
    start = view["byteOffset"] + accessor.get("byteOffset", 0)
    values = np.frombuffer(binary, dtype=dtype, count=accessor["count"] * width, offset=start)
    return values.reshape(-1, width) if width > 1 else values


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("viewer_export")
    ifc = root / "room.ifc"
    write_ifc(build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE)), ifc, run_id="viewer-test")
    assets = export_viewer_assets(ifc, root / "viewer")
    return ifcopenshell.open(str(ifc)), assets


def test_glb_structure_matches_ifc(exported) -> None:
    model, assets = exported
    document, binary = _read_glb(assets.glb_path)
    props = json.loads(assets.props_path.read_text(encoding="utf-8"))
    assert document["asset"]["version"] == "2.0"
    assert len(document["buffers"]) == 1 and document["buffers"][0]["byteLength"] == len(binary)
    names = [node["name"] for node in document["nodes"]]
    assert len(names) == len(set(names)) > 0
    for node in document["nodes"]:
        element = model.by_guid(node["name"])
        assert node["extras"]["ifc_class"] == element.is_a()
        assert node["extras"]["name"] == (element.Name or "")
        assert node["name"] in props["elements"]
        assert len(document["meshes"][node["mesh"]]["primitives"]) >= 1
    expected = {e.GlobalId for e in model.by_type("IfcProduct") if e.Representation is not None}
    assert set(names) == expected
    assert set(props["elements"]) == expected
    for ifc_class in ("IfcWall", "IfcDoor", "IfcWindow", "IfcSpace", "IfcSlab", "IfcOpeningElement"):
        assert props["classes"][ifc_class] == len(model.by_type(ifc_class)) == assets.counts[ifc_class]


def test_accessors_min_max_and_yup_conversion(exported) -> None:
    model, assets = exported
    document, binary = _read_glb(assets.glb_path)
    all_positions = []
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            positions = _accessor(document, binary, primitive["attributes"]["POSITION"])
            normals = _accessor(document, binary, primitive["attributes"]["NORMAL"])
            indices = _accessor(document, binary, primitive["indices"])
            assert len(positions) > 0 and len(positions) == len(normals)
            assert indices.max() < len(positions) and len(indices) % 3 == 0
            accessor = document["accessors"][primitive["attributes"]["POSITION"]]
            np.testing.assert_array_equal(np.array(accessor["min"], dtype=np.float32), positions.min(axis=0))
            np.testing.assert_array_equal(np.array(accessor["max"], dtype=np.float32), positions.max(axis=0))
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
            all_positions.append(positions)
    stacked = np.concatenate(all_positions)

    import ifcopenshell.geom

    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    products = [e for e in model.by_type("IfcProduct") if e.Representation is not None]
    iterator = ifcopenshell.geom.iterator(settings, model, 1, include=products)
    verts = []
    assert iterator.initialize()
    while True:
        verts.append(np.asarray(iterator.get().geometry.verts).reshape(-1, 3))
        if not iterator.next():
            break
    ifc_points = np.concatenate(verts)
    assert stacked[:, 1].max() == pytest.approx(ifc_points[:, 2].max(), abs=1e-6)
    assert stacked[:, 1].min() == pytest.approx(ifc_points[:, 2].min(), abs=1e-6)
    assert stacked[:, 0].max() == pytest.approx(ifc_points[:, 0].max(), abs=1e-6)
    assert stacked[:, 2].min() == pytest.approx(-ifc_points[:, 1].max(), abs=1e-6)


def test_materials_blend_for_glass_and_space(exported) -> None:
    _model, assets = exported
    document, _binary = _read_glb(assets.glb_path)
    by_class: dict[str, set[str | None]] = {}
    for node in document["nodes"]:
        for primitive in document["meshes"][node["mesh"]]["primitives"]:
            material = document["materials"][primitive["material"]]
            by_class.setdefault(node["extras"]["ifc_class"], set()).add(material.get("alphaMode"))
    assert by_class["IfcWindow"] == {None, "BLEND"}  # opaque frame/mullions + translucent glass panes
    assert by_class["IfcSpace"] == {"BLEND"}
    assert by_class["IfcWall"] == {None}


def test_props_contents(exported) -> None:
    model, assets = exported
    props = json.loads(assets.props_path.read_text(encoding="utf-8"))
    assert props["schema_version"] == 1 and props["ifc_file"] == "room.ifc"
    wall = model.by_type("IfcWall")[0]
    entry = props["elements"][wall.GlobalId]
    assert entry["ifc_class"] == "IfcWall" and entry["name"] == wall.Name
    assert entry["storey"] == "Ground floor"
    assert "Qto_WallBaseQuantities" in entry["psets"]
    import ifcopenshell.util.element as util

    expected = util.get_psets(wall)["Qto_WallBaseQuantities"]
    assert entry["psets"]["Qto_WallBaseQuantities"]["Length"] == pytest.approx(expected["Length"])
    assert entry["psets"]["Qto_WallBaseQuantities"]["Height"] == pytest.approx(2.8)
    assert entry["psets"]["Pset_WallCommon"]["IsExternal"] is True


def test_pygltflib_loads_when_installed(exported) -> None:
    pygltflib = pytest.importorskip("pygltflib")
    _model, assets = exported
    gltf = pygltflib.GLTF2().load(str(assets.glb_path))
    assert len(gltf.nodes) == sum(assets.counts.values())
