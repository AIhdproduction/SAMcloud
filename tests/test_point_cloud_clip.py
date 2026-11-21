"""Point cloud products of outdoor/drone projects are clipped to the processing boundary (raw outputs stay untouched)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np
import pytest
import yaml
from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform as shapely_transform

from samcloud.drone import perimeter
from samcloud.drone.contracts import LocalFrame, PointSet
from samcloud.drone.export import CLIPPED_PLY_NAME, ensure_clipped_classified_ply
from samcloud.engine.export_las import export_las
from samcloud.engine.instances import write_instance_product
from samcloud.engine.ply_io import read_vertex_array, write_classified_ply

CLASSES_JSON = Path(__file__).resolve().parents[1] / "config" / "classes.json"
LAT, LON, H0 = 47.134, 8.755, 450.0
_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
_ECEF_TO_LV95 = Transformer.from_crs("EPSG:4978", "EPSG:2056+5728", always_xy=True)
HALF = 40.0  # boundary half edge in metres (LV95 axis-aligned box)


def _georef() -> dict:
    lat, lon = np.radians(LAT), np.radians(LON)
    east = [-np.sin(lon), np.cos(lon), 0.0]
    north = [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    up = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    rotation = np.array([east, north, up]).T
    origin = np.array(_TO_ECEF.transform(LON, LAT, H0))
    return {"scale": 1.0, "rotation": rotation.tolist(), "translation": origin.tolist(),
            "pairs": 5, "residual_m": {"mean": 0.0, "p95": 0.0, "max": 0.0}}


def _reference_lv95(xyz_enu: np.ndarray) -> np.ndarray:
    g = _georef()
    ecef = (np.asarray(xyz_enu) @ np.asarray(g["rotation"]).T) * g["scale"] + np.asarray(g["translation"])
    return np.column_stack(_ECEF_TO_LV95.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2]))


def _enu_cloud(n: int = 400, seed: int = 5) -> np.ndarray:
    """ENU points around the boundary edges (rotation of ~1 degree decides several of them)."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(-HALF - 6.0, HALF + 6.0, (n, 3)) * [1.0, 1.0, 0.05]).astype(np.float32)


def _boundary_lv95() -> Polygon:
    centre = _reference_lv95(np.zeros((1, 3)))[0]
    return box(centre[0] - HALF, centre[1] - HALF, centre[0] + HALF, centre[1] + HALF)


def _sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _write_geojson(path: Path, polygon) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(polygon)}]}
    path.write_text(json.dumps(doc), encoding="utf-8")


def _project(tmp_path: Path, *, class_set: str = "outdoor", boundary: bool = True, n: int = 400) -> tuple[Path, np.ndarray]:
    project = tmp_path / "proj"
    products = project / "products"
    (products / "classified").mkdir(parents=True)
    (project / "project.samcloud.yml").write_text(
        yaml.safe_dump({"acquisition": {"type": "drone_gps"}, "settings": {"classification": {"class_set": class_set}}}),
        encoding="utf-8")
    (products / "georeference.json").write_text(json.dumps(_georef()), encoding="utf-8")
    xyz = _enu_cloud(n)
    rgb = (np.arange(n * 3) % 251).astype(np.uint8).reshape(n, 3)
    classes = (1 + np.arange(n) % 5).astype(np.uint8)
    ply = products / "classified" / "classified.ply"
    write_classified_ply(ply, xyz, rgb, classes)
    write_instance_product(ply, np.arange(1, n + 1, dtype=np.uint32), [])
    if boundary:
        _write_geojson(project / perimeter.BOUNDARY_GEOJSON, _boundary_lv95())
    return project, xyz


def _inside_reference(xyz_enu: np.ndarray) -> np.ndarray:
    lv95 = _reference_lv95(xyz_enu.astype(np.float64))
    poly = _boundary_lv95()
    return np.array([poly.contains(__import__("shapely").geometry.Point(x, y)) for x, y, _ in lv95])


# ---------------------------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------------------------

def test_enu_cloud_is_clipped_with_the_georeference_not_a_translation() -> None:
    xyz = _enu_cloud()
    truth = _inside_reference(xyz)
    mask = perimeter.inside_mask_enu(xyz, _georef(), _boundary_lv95())
    assert np.array_equal(mask, truth)
    # a translation-only frame (ENU origin -> LV95 origin) decides some points differently: the rotation matters
    origin = _reference_lv95(np.zeros((1, 3)))[0]
    naive = perimeter.inside_mask_lv95(xyz[:, :2].astype(np.float64) + origin[:2], _boundary_lv95())
    assert int(np.sum(naive != truth)) > 0
    assert 0 < int(truth.sum()) < len(xyz)


# ---------------------------------------------------------------------------------------------
# LAS export of the main pipeline / app
# ---------------------------------------------------------------------------------------------

def test_export_las_keeps_only_inside_points_with_all_attributes(tmp_path: Path) -> None:
    project, xyz = _project(tmp_path)
    ply = project / "products" / "classified" / "classified.ply"
    before = _sha(ply)
    out = project / "products" / "classified" / "classified.las"
    truth = _inside_reference(xyz)
    written = export_las(ply, out, CLASSES_JSON, 70, "LOCAL", "EPSG:2056", "outdoor",
                         project / "products" / "georeference.json", clip_polygon_lv95=_boundary_lv95())
    las = laspy.read(out)
    assert written == int(truth.sum()) == len(las.points)
    source = read_vertex_array(ply)
    assert np.array_equal(np.asarray(las.instance_id), source["instance_id"][truth])
    assert np.array_equal(np.asarray(las.red), source["red"][truth].astype(np.uint16) * 257)
    assert len(set(np.asarray(las.classification).tolist())) > 1
    lv95 = _reference_lv95(xyz.astype(np.float64))[truth]
    assert np.column_stack((las.x, las.y)) == pytest.approx(lv95[:, :2], abs=0.002)
    assert _sha(ply) == before  # raw source untouched


def test_export_las_extra_dims_follow_the_clip(tmp_path: Path) -> None:
    project, xyz = _project(tmp_path)
    ply = project / "products" / "classified" / "classified.ply"
    truth = _inside_reference(xyz)
    marker = np.arange(len(xyz), dtype=np.float32)
    out = tmp_path / "out.laz"
    export_las(ply, out, CLASSES_JSON, 33, "LOCAL", "EPSG:2056", "outdoor", project / "products" / "georeference.json",
               extra_dims={"marker": marker}, clip_polygon_lv95=_boundary_lv95())
    las = laspy.read(out)
    assert np.array_equal(np.asarray(las.marker), marker[truth])


def test_export_las_without_clip_polygon_is_unchanged(tmp_path: Path) -> None:
    project, xyz = _project(tmp_path)
    ply = project / "products" / "classified" / "classified.ply"
    out = tmp_path / "all.las"
    written = export_las(ply, out, CLASSES_JSON, 100, "LOCAL", "EPSG:2056", "outdoor",
                         project / "products" / "georeference.json")
    assert written == len(xyz) == len(laspy.read(out).points)


def test_export_las_local_target_without_lv95_position_stays_unclipped(tmp_path: Path) -> None:
    project, xyz = _project(tmp_path)
    ply = project / "products" / "classified" / "classified.ply"
    out = tmp_path / "local.las"
    written = export_las(ply, out, CLASSES_JSON, 100, "LOCAL", "LOCAL", "outdoor", clip_polygon_lv95=_boundary_lv95())
    assert written == len(xyz)


def test_processing_export_command_gets_clip_only_for_outdoor_projects(tmp_path: Path) -> None:
    outdoor, _ = _project(tmp_path)
    assert perimeter.clip_polygon_lv95(outdoor, {"environment": {"clip_point_cloud": True}}) is not None
    assert perimeter.clip_polygon_lv95(outdoor, {"environment": {"clip_point_cloud": False}}) is None
    indoor_root = tmp_path / "indoor_case"
    indoor_root.mkdir()
    indoor, _ = _project(indoor_root, class_set="indoor")
    assert perimeter.clip_polygon_lv95(indoor) is None
    nobound_root = tmp_path / "nobound_case"
    nobound_root.mkdir()
    nobound, _ = _project(nobound_root, boundary=False)
    assert perimeter.clip_polygon_lv95(nobound) is None


# ---------------------------------------------------------------------------------------------
# viewer copy
# ---------------------------------------------------------------------------------------------

def test_clipped_classified_ply_for_viewer(tmp_path: Path) -> None:
    project, xyz = _project(tmp_path)
    raw = project / "products" / "classified" / "classified.ply"
    before = _sha(raw)
    truth = _inside_reference(xyz)
    clipped = ensure_clipped_classified_ply(project)
    assert clipped == raw.with_name(CLIPPED_PLY_NAME)
    source, result = read_vertex_array(raw), read_vertex_array(clipped)
    assert result.dtype == source.dtype  # all properties kept (instance_id, colours, classes)
    assert np.array_equal(result, source[truth])
    assert _sha(raw) == before
    stamp = clipped.stat().st_mtime_ns
    assert ensure_clipped_classified_ply(project) == clipped and clipped.stat().st_mtime_ns == stamp  # cached
    from samcloud.viewer.preview import _source_path

    assert _source_path(project, "dense") == clipped


def test_viewer_keeps_raw_cloud_for_indoor_and_without_boundary(tmp_path: Path) -> None:
    from samcloud.viewer.preview import _source_path

    indoor_root = tmp_path / "a"
    indoor_root.mkdir()
    indoor, _ = _project(indoor_root, class_set="indoor")
    assert ensure_clipped_classified_ply(indoor) is None
    assert _source_path(indoor, "dense") == indoor / "products" / "classified" / "classified.ply"
    nobound_root = tmp_path / "b"
    nobound_root.mkdir()
    nobound, _ = _project(nobound_root, boundary=False)
    assert ensure_clipped_classified_ply(nobound) is None
    assert _source_path(nobound, "dense") == nobound / "products" / "classified" / "classified.ply"
    assert not (nobound / "products" / "classified" / CLIPPED_PLY_NAME).exists()


# ---------------------------------------------------------------------------------------------
# environment LAZ (_step_export)
# ---------------------------------------------------------------------------------------------

def _run_step_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg: dict) -> tuple[dict, np.ndarray, Path]:
    import pickle

    from samcloud.drone import run_environment as env

    frame = LocalFrame(origin_lv95=(2_683_000.0, 1_222_000.0, 400.0))
    rng = np.random.default_rng(11)
    full_xyz = rng.uniform(-30.0, 30.0, (600, 3)) * [1.0, 1.0, 0.1]
    working = PointSet(xyz=full_xyz[::2].copy(), classes=np.full(300, 2, np.uint16),
                       confidence=rng.uniform(0.5, 1.0, 300).astype(np.float32),
                       view_count=np.full(300, 4, np.uint16), rgb=np.full((300, 3), 90, np.uint8))
    full = PointSet(xyz=full_xyz, classes=np.full(600, 2, np.uint16),
                    confidence=rng.uniform(0.5, 1.0, 600).astype(np.float32),
                    view_count=np.full(600, 4, np.uint16), rgb=np.full((600, 3), 90, np.uint8))
    boundary = box(-12.3, -9.7, 14.1, 11.9)  # local frame, off-grid edges
    cache, out = tmp_path / "cache", tmp_path / "out"
    cache.mkdir(), out.mkdir()
    for name, value in (("load.pkl", (working, frame)), ("load_full.pkl", (full, frame)), ("boundary.pkl", boundary),
                        ("scene.pkl", {"scene": SimpleNamespace(elements={}), "dxf": {}})):
        with (cache / name).open("wb") as handle:
            pickle.dump(value, handle)
    monkeypatch.setattr("samcloud.bim.core.ifc.writer.write_ifc", lambda scene, path: Path(path).write_bytes(b"ifc"))
    monkeypatch.setattr("samcloud.dxf.writer.build_dxf_document",
                        lambda *a, **k: SimpleNamespace(saveas=lambda path: Path(path).write_bytes(b"dxf")))
    monkeypatch.setattr("samcloud.dxf.writer.ensure_outdoor_layers", lambda *a, **k: None)
    monkeypatch.setattr(env, "_dsm_mesh", lambda *a, **k: None)
    monkeypatch.setattr("samcloud.drone.publish.publish_environment_exports", lambda *a, **k: {})
    result = env._step_export(tmp_path, out, cfg, cache)
    return result, full_xyz, out / "classified.laz"


def test_environment_laz_contains_only_points_inside_the_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, full_xyz, laz = _run_step_export(tmp_path, monkeypatch, {})
    inside = (full_xyz[:, 0] > -12.3) & (full_xyz[:, 0] < 14.1) & (full_xyz[:, 1] > -9.7) & (full_xyz[:, 1] < 11.9)
    las = laspy.read(laz)
    assert len(las.points) == int(inside.sum()) < len(full_xyz)
    assert result["point_source"] == "full_transferred"
    assert result["point_cloud_clip"] == {"points_before": 600, "points_after": int(inside.sum())}
    names = set(las.point_format.extra_dimension_names)
    assert {"confidence", "view_count", "hag", "building_id", "tree_id"} <= names
    assert set(np.asarray(las.classification).tolist()) == {2}
    local = np.column_stack((las.x, las.y)) - np.array([2_683_000.0, 1_222_000.0])
    assert local[:, 0].min() > -12.31 and local[:, 0].max() < 14.11
    assert local[:, 1].min() > -9.71 and local[:, 1].max() < 11.91


def test_environment_laz_switch_off_keeps_all_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, full_xyz, laz = _run_step_export(tmp_path, monkeypatch, {"environment": {"clip_point_cloud": False}})
    assert len(laspy.read(laz).points) == len(full_xyz)
    assert "point_cloud_clip" not in result


def test_share_prepare_points_follows_the_boundary(tmp_path: Path) -> None:
    from samcloud.share import export as share_export

    project, xyz = _project(tmp_path)
    truth = _inside_reference(xyz)
    (project / "products" / "classes.json").write_text(CLASSES_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    points, original = share_export.prepare_points(project, 10_000, lambda *a: None, lambda: None)
    assert original == int(truth.sum()) < len(xyz)
    other_root = tmp_path / "indoor_share"
    other_root.mkdir()
    indoor, _ = _project(other_root, class_set="indoor")
    (indoor / "products" / "classes.json").write_text(CLASSES_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    _points, indoor_original = share_export.prepare_points(indoor, 10_000, lambda *a: None, lambda: None)
    assert indoor_original == len(xyz)
