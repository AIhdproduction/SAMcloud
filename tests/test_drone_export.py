"""Tests for samcloud.drone.export and the extended samcloud.engine.export_las."""
from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import pytest

from samcloud.drone.contracts import LocalFrame, PointSet
from samcloud.drone.export import (
    export_mesh,
    export_point_cloud,
    export_scene_glb,
    lv95_sample_check,
    validate_exports,
)
from samcloud.engine.export_las import export_las
from samcloud.engine.ply_io import write_classified_ply

CLASSES_JSON = Path(__file__).resolve().parents[1] / "config" / "classes.json"
FRAME = LocalFrame(origin_lv95=(2683000.123, 1248000.456, 435.789))


def _pointset(n: int = 200, seed: int = 1) -> PointSet:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-20.0, 20.0, (n, 3)) + rng.normal(0, 1e-4, (n, 3))
    return PointSet(
        xyz=xyz,
        classes=rng.choice([2, 5, 6], n).astype(np.uint16),
        confidence=rng.uniform(0.3, 0.99, n).astype(np.float32),
        margin=rng.uniform(0.0, 0.5, n).astype(np.float32),
        view_count=rng.integers(1, 30, n).astype(np.uint16),
        rgb=rng.integers(0, 256, (n, 3), dtype=np.uint8),
        adjusted=rng.random(n) > 0.8,
    )


def _extras(n: int) -> dict:
    rng = np.random.default_rng(2)
    return {"hag": rng.uniform(0, 12, n).astype(np.float32),
            "building_id": rng.integers(-1, 4, n),
            "tree_id": rng.integers(-1, 6, n)}


def _tetra() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[0.1, 0.2, 0.3], [4.2, 0.1, 0.4], [0.3, 3.9, 0.2], [1.1, 1.3, 3.7]])
    f = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    return v, f


def test_laz_roundtrip_with_extra_dims_and_dtypes(tmp_path: Path) -> None:
    ps = _pointset()
    path = export_point_cloud(ps, tmp_path / "cloud.laz", FRAME, extra=_extras(len(ps)))
    las = laspy.read(str(path))
    names = set(las.point_format.extra_dimension_names)
    assert {"confidence", "margin", "view_count", "adjusted", "hag", "building_id", "tree_id"} <= names
    assert len(las.points) == len(ps)
    assert np.asarray(las.confidence).dtype == np.float32
    assert np.asarray(las.view_count).dtype == np.uint16
    assert np.asarray(las.adjusted).dtype == np.uint8
    assert np.asarray(las.building_id).dtype == np.int32
    assert np.allclose(np.asarray(las.confidence), ps.confidence)
    assert np.array_equal(np.asarray(las.view_count), ps.view_count)
    assert np.array_equal(np.asarray(las.adjusted), ps.adjusted.astype(np.uint8))
    assert np.array_equal(np.asarray(las.classification), ps.classes.astype(np.uint8))
    assert np.allclose(las.header.scales, 0.001)
    assert np.allclose(las.header.offsets, np.floor(las.header.offsets))
    crs = las.header.parse_crs()
    assert crs is not None and "2056" in crs.to_wkt()


def test_export_las_laz_and_extra_dims_keep_plain_las_unchanged(tmp_path: Path) -> None:
    n = 30
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    rgb = np.full((n, 3), 100, dtype=np.uint8)
    cls = np.full(n, 17, dtype=np.uint8)
    ply = tmp_path / "in.ply"
    write_classified_ply(ply, xyz, rgb, cls)

    plain = tmp_path / "plain.las"
    assert export_las(ply, plain, CLASSES_JSON, 7, "LOCAL", "LOCAL", "indoor") == n
    las = laspy.read(str(plain))
    assert las.header.point_format.id == 7
    assert list(las.point_format.extra_dimension_names) == []

    extra = {"conf": rng.uniform(0, 1, n), "views": rng.integers(0, 9, n).astype(np.uint16),
             "flag": rng.random(n) > 0.5, "ident": rng.integers(0, 99, n).astype(np.uint32)}
    laz = tmp_path / "out.laz"
    export_las(ply, laz, CLASSES_JSON, 7, "LOCAL", "LOCAL", "indoor", extra_dims=extra)
    assert laz.read_bytes()[:4] == b"LASF"
    las = laspy.read(str(laz))
    assert np.asarray(las.conf).dtype == np.float32
    assert np.asarray(las.views).dtype == np.uint16
    assert np.asarray(las.flag).dtype == np.uint8
    assert np.asarray(las.ident).dtype == np.uint32
    assert np.array_equal(np.asarray(las.views), extra["views"])  # chunked (7) writing keeps the order
    assert np.array_equal(np.asarray(las.ident), extra["ident"])
    assert np.array_equal(np.asarray(las.flag), extra["flag"].astype(np.uint8))
    with pytest.raises(ValueError):
        export_las(ply, tmp_path / "bad.las", CLASSES_JSON, 7, "LOCAL", "LOCAL", "indoor",
                   extra_dims={"x": np.zeros(n + 1)})


def test_lv95_coordinates_match_frame_within_1mm(tmp_path: Path) -> None:
    ps = _pointset()
    path = export_point_cloud(ps, tmp_path / "c.laz", FRAME)
    las = laspy.read(str(path))
    stored = np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)))
    assert np.abs(stored - FRAME.to_lv95(ps.xyz)).max() < 1e-3


def test_obj_has_lv95_and_glb_has_local_with_origin(tmp_path: Path) -> None:
    import trimesh

    v, f = _tetra()
    obj = export_mesh(v, f, tmp_path / "b.obj", FRAME, fmt="obj", node_name="house")
    text = obj.read_text(encoding="utf-8")
    assert text.startswith("# CRS: EPSG:2056")
    rows = [list(map(float, ln.split()[1:])) for ln in text.splitlines() if ln.startswith("v ")]
    assert np.allclose(np.array(rows), FRAME.to_lv95(v), atol=6e-4)
    assert all(len(ln.split()[1].split(".")[1]) == 3 for ln in text.splitlines() if ln.startswith("v "))

    glb = export_mesh(v, f, tmp_path / "b.glb", FRAME, fmt="glb", node_name="house")
    scene = trimesh.load(str(glb), file_type="glb", force="scene", process=False)
    verts = np.vstack([g.vertices for g in scene.geometry.values()])
    assert verts.max() < 100  # local coordinates, no large floats
    assert np.allclose(np.sort(verts, axis=0), np.sort(v, axis=0), atol=1e-5)
    assert np.allclose(scene.metadata["origin_lv95"], FRAME.origin_lv95)
    assert "house" in scene.graph.nodes_geometry


def test_scene_glb_has_one_named_node_per_entry(tmp_path: Path) -> None:
    import trimesh

    v, f = _tetra()
    nodes = {"building_1": (v, f), "building_2": (v + 10, f), "tree_7": (v * 0.5, f)}
    path = export_scene_glb(nodes, tmp_path / "scene.glb", FRAME)
    scene = trimesh.load(str(path), file_type="glb", force="scene", process=False)
    assert set(scene.graph.nodes_geometry) == set(nodes)
    assert len(scene.geometry) == 3
    assert np.allclose(scene.metadata["origin_lv95"], FRAME.origin_lv95)


def _make_tif(path: Path, epsg: int = 2056, value: float = 3.0) -> None:
    import rasterio
    from rasterio.transform import from_origin

    data = (np.arange(16, dtype=np.float32).reshape(4, 4) * 0.37 + value)
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
                       crs=f"EPSG:{epsg}", transform=from_origin(2683000, 1248010, 1, 1), nodata=-9999) as dst:
        dst.write(data, 1)


def _make_ifc(path: Path, with_map: bool = True) -> None:
    import ifcopenshell
    import ifcopenshell.api

    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name="p")
    ifcopenshell.api.run("unit.assign_unit", model)
    ctx = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    if with_map:
        crs = model.create_entity("IfcProjectedCRS", Name="EPSG:2056")
        model.create_entity("IfcMapConversion", SourceCRS=ctx, TargetCRS=crs,
                            Eastings=2683000.0, Northings=1248000.0, OrthogonalHeight=435.0)
    model.write(str(path))


def _make_dxf(path: Path) -> None:
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.modelspace().add_line((0, 0), (5, 5))
    doc.saveas(str(path))


def test_validate_exports_passes_good_and_flags_broken(tmp_path: Path) -> None:
    ps = _pointset()
    v, f = _tetra()
    good = {
        "laz": export_point_cloud(ps, tmp_path / "g.laz", FRAME, extra=_extras(len(ps))),
        "las": export_point_cloud(ps, tmp_path / "g.las", FRAME),
        "obj": export_mesh(v, f, tmp_path / "g.obj", FRAME, fmt="obj"),
        "glb": export_mesh(v, f, tmp_path / "g.glb", FRAME, fmt="glb"),
    }
    _make_tif(tmp_path / "g.tif")
    good["tif"] = tmp_path / "g.tif"
    _make_ifc(tmp_path / "g.ifc")
    good["ifc"] = tmp_path / "g.ifc"
    _make_dxf(tmp_path / "g.dxf")
    good["dxf"] = tmp_path / "g.dxf"
    report = validate_exports(good)
    for key, entry in report.items():
        assert entry["ok"], (key, entry)
    assert report["laz"]["points"] == len(ps)
    assert "hag" in report["laz"]["extra_dims"]
    assert report["obj"]["watertight"] is True
    assert report["dxf"]["entities"] == 1

    # broken files
    (tmp_path / "empty.laz").write_bytes(b"")
    (tmp_path / "junk.glb").write_bytes(b"not a glb at all")
    (tmp_path / "empty.obj").write_text("# nothing\n", encoding="utf-8")
    _make_tif(tmp_path / "wrong_crs.tif", epsg=4326)
    _make_ifc(tmp_path / "nomap.ifc", with_map=False)
    local_cloud = PointSet(xyz=ps.xyz, classes=ps.classes)
    export_point_cloud(local_cloud, tmp_path / "far.las", LocalFrame(origin_lv95=(100.0, 100.0, 0.0)))
    bad = validate_exports({
        "empty": tmp_path / "empty.laz", "junk": tmp_path / "junk.glb", "emptyobj": tmp_path / "empty.obj",
        "crs": tmp_path / "wrong_crs.tif", "nomap": tmp_path / "nomap.ifc", "far": tmp_path / "far.las",
        "missing": tmp_path / "nope.dxf",
    })
    for key, entry in bad.items():
        assert not entry["ok"], (key, entry)
        assert entry["reason"]


def test_lv95_sample_check_passes_and_fails_when_shifted(tmp_path: Path) -> None:
    ps = _pointset()
    good = export_point_cloud(ps, tmp_path / "ok.laz", FRAME)
    result = lv95_sample_check(ps, good, FRAME, n=20)
    assert result["ok"] and result["n"] == 20 and result["max_err_m"] < 1e-3

    shifted_frame = LocalFrame(origin_lv95=(FRAME.origin_lv95[0] + 1.0, FRAME.origin_lv95[1],
                                            FRAME.origin_lv95[2]))
    shifted = export_point_cloud(ps, tmp_path / "shift.laz", shifted_frame)
    result = lv95_sample_check(ps, shifted, FRAME, n=20)
    assert not result["ok"]
    assert result["max_err_m"] > 1e-3
