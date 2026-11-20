"""Perimeter (KML area of interest) loading, projection, clipping and pipeline hook."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon

from samcloud.drone.contracts import LocalFrame, PointSet, load_drone_config
from samcloud.drone.perimeter import (camera_hull_boundary, camera_positions_lv95, clip_pointset, find_project_perimeter,
                                      load_perimeter_kml, perimeter_to_local, processing_boundary)
from test_run_environment import _synthetic_project

NS = 'xmlns="http://www.opengis.net/kml/2.2"'
REAL_KML = Path(r"D:\SAMcloud\projects\drohnenrekonstruktion_3d_test\inputs\survey\Perimeter.kml")


def _kml(body: str) -> str:
    return f'<kml {NS}><Document><Placemark>{body}</Placemark></Document></kml>'


def _poly_xml(outer: str, inner: str | None = None) -> str:
    s = f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{outer}</coordinates></LinearRing></outerBoundaryIs>"
    if inner:
        s += f"<innerBoundaryIs><LinearRing><coordinates>{inner}</coordinates></LinearRing></innerBoundaryIs>"
    return s + "</Polygon>"


def test_kml_parse_polygon_with_hole_and_multigeometry(tmp_path: Path) -> None:
    outer = "8.0,47.0 8.01,47.0 8.01,47.01 8.0,47.01 8.0,47.0"
    inner = "8.004,47.004 8.006,47.004 8.006,47.006 8.004,47.006 8.004,47.004"
    f1 = tmp_path / "a.kml"; f1.write_text(_kml(_poly_xml(outer, inner)), encoding="utf-8")
    poly = load_perimeter_kml(f1)
    assert isinstance(poly, Polygon) and len(poly.interiors) == 1
    assert abs(poly.area - (0.01 * 0.01 - 0.002 * 0.002)) < 1e-12
    f2 = tmp_path / "b.kml"
    other = "8.5,47.0 8.51,47.0 8.51,47.01 8.5,47.01 8.5,47.0"
    f2.write_text(_kml("<MultiGeometry>" + _poly_xml(outer) + _poly_xml(other) + "</MultiGeometry>"), encoding="utf-8")
    multi = load_perimeter_kml(f2)
    assert multi.geom_type == "MultiPolygon" and len(multi.geoms) == 2


def test_wgs84_to_local_round_trip_matches_pyproj() -> None:
    frame = LocalFrame((2_680_000.0, 1_215_000.0, 450.0))
    lon, lat = 8.755, 47.134
    local = perimeter_to_local(Polygon([(lon, lat), (lon + .001, lat), (lon, lat + .001)]), frame)
    e, n = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform(lon, lat)
    first = np.asarray(local.exterior.coords)[0]
    assert abs(first[0] - (e - 2_680_000.0)) < 1e-3 and abs(first[1] - (n - 1_215_000.0)) < 1e-3
    back = frame.to_lv95(np.array([[first[0], first[1], 0.0]]))[0]
    assert abs(back[0] - e) < 1e-3 and abs(back[1] - n) < 1e-3


def test_clip_pointset_with_buffer() -> None:
    rng = np.random.default_rng(3)
    xy = rng.uniform(-10, 20, (4000, 2)) + 0.0137
    ps = PointSet(np.c_[xy, rng.normal(size=4000)], np.full(4000, 2, np.uint16))
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    inside, mask = clip_pointset(ps, square, 0.0)
    assert len(inside) == mask.sum() and (inside.xyz[:, :2] >= 0).all() and (inside.xyz[:, :2] <= 10).all()
    buffered, mask2 = clip_pointset(ps, square, 2.0)
    assert len(buffered) > len(inside) and mask2[mask].all()
    assert (buffered.xyz[:, :2] >= -2).all() and (buffered.xyz[:, :2] <= 12).all()
    assert not mask2[(xy[:, 0] > 12.01) | (xy[:, 0] < -2.01)].any()


def test_find_project_perimeter(tmp_path: Path) -> None:
    assert find_project_perimeter(tmp_path) is None
    (tmp_path / "inputs").mkdir()
    top = tmp_path / "inputs" / "top.kml"; top.write_text("x")
    assert find_project_perimeter(tmp_path) == top
    (tmp_path / "inputs" / "survey").mkdir()
    sv = tmp_path / "inputs" / "survey" / "Perimeter.kml"; sv.write_text("x")
    assert find_project_perimeter(tmp_path) == sv
    assert find_project_perimeter(tmp_path, "inputs/top.kml") == top


def _write_gps_ref(project: Path, local_xy: np.ndarray, frame: LocalFrame) -> None:
    lon, lat = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True).transform(
        local_xy[:, 0] + frame.origin_lv95[0], local_xy[:, 1] + frame.origin_lv95[1])
    (project / "products").mkdir(parents=True, exist_ok=True)
    (project / "products" / "gps_ref.txt").write_text(
        "".join(f"dir/img_{i:04d}.JPG {la:.9f} {lo:.9f} 973.3\n" for i, (la, lo) in enumerate(zip(lat, lon))), encoding="utf-8")


def test_builtin_default_boundary() -> None:
    assert load_drone_config()["boundary"] == {"mode": "camera_hull", "buffer_m": 10.0, "perimeter_path": None, "perimeter_buffer_m": 2.0}


def test_camera_hull_boundary_is_hull_plus_buffer(tmp_path: Path) -> None:
    from shapely.geometry import MultiPoint
    frame = LocalFrame((2_699_880.0, 1_221_200.0, 450.0))
    rng = np.random.default_rng(5)
    pts = rng.uniform(-30, 30, (60, 2)) + 0.0123
    _write_gps_ref(tmp_path, pts, frame)
    assert camera_positions_lv95(tmp_path).shape == (60, 2)
    poly = camera_hull_boundary(tmp_path, frame, 10.0)
    expected = MultiPoint(pts).convex_hull.buffer(10.0)
    # lon/lat text rounding (1e-9 deg ~ 0.1 mm) keeps the difference far below 1 cm
    assert expected.symmetric_difference(poly).area / poly.length < 0.01
    assert abs(poly.area - expected.area) < 0.5
    assert poly.hausdorff_distance(expected) < 0.01


def test_processing_boundary_modes(tmp_path: Path) -> None:
    frame = LocalFrame((2_699_880.0, 1_221_200.0, 450.0))
    _write_gps_ref(tmp_path, np.array([[-20, -20], [20, -20], [20, 20], [-20, 20.0]]) + 0.011, frame)
    corners = np.array([[0, -50], [50, -50], [50, 50], [0, 50], [0, -50]], float)
    lon, lat = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True).transform(
        corners[:, 0] + frame.origin_lv95[0], corners[:, 1] + frame.origin_lv95[1])
    (tmp_path / "inputs" / "survey").mkdir(parents=True)
    (tmp_path / "inputs" / "survey" / "P.kml").write_text(_kml(_poly_xml(" ".join(f"{a},{b}" for a, b in zip(lon, lat)))), encoding="utf-8")
    cfg = load_drone_config()
    hull, info = processing_boundary(tmp_path, frame, cfg)
    assert info["mode"] == "camera_hull" and info["cameras"] == 4 and abs(hull.area - (1600 + 1600 + np.pi * 100)) < 1.0
    cfg["boundary"]["mode"] = "intersection"
    inter, info = processing_boundary(tmp_path, frame, cfg)
    from shapely.geometry import box
    assert abs(inter.area - hull.intersection(box(0, -50, 50, 50)).area) < 0.5 and inter.bounds[0] > -1e-3
    cfg["boundary"].update(mode="perimeter", perimeter_buffer_m=0.0)
    kml, _ = processing_boundary(tmp_path, frame, cfg)
    assert abs(kml.area - 5000) < 1.0
    cfg["boundary"]["mode"] = "none"
    assert processing_boundary(tmp_path, frame, cfg)[0] is None


def test_run_environment_default_clips_to_camera_hull_and_masks_rasters(tmp_path: Path) -> None:
    import rasterio
    from samcloud.drone.io import load_classified_pointset_lv95
    from samcloud.drone.run_environment import HANDLERS, _apply_boundary, _load
    project = _synthetic_project(tmp_path, points=100_000)
    cfg = load_drone_config()
    cfg["boundary"]["buffer_m"] = 2.0
    cfg["terrain"].update({"ground_filter": "pmf", "dtm_resolution_m": .25, "dsm_resolution_m": .25})
    ps0, frame = load_classified_pointset_lv95(project / "products", project_dir=project)
    n0 = len(ps0)
    out = project / "products" / "environment"; cache = out / "cache"; cache.mkdir(parents=True)
    first = HANDLERS["load"](project, out, cfg, cache); assert first["points"] == n0 and "points_after" not in first.get("boundary", {})  # no gps_ref -> unchanged
    center = ps0.xyz[:, :2].mean(0)
    _write_gps_ref(project, np.array([[-10, -8], [10, -8], [10, 8], [-10, 8.0]]) + 0.0137 + center, frame)
    info = HANDLERS["load"](project, out, cfg, cache)
    b = info["boundary"]
    assert b["mode"] == "camera_hull" and b["points_before"] == n0 and b["points_after"] == info["points"] < n0 and info["points"] > 1000
    assert abs(b["area_m2"] - (20 * 16 + 2 * 2 * (20 + 16) + np.pi * 4)) < 0.5
    ps, _ = _load(cache / "load.pkl")
    rel = ps.xyz[:, :2] - center
    assert (np.abs(rel[:, 0]) <= 12.02).all() and (np.abs(rel[:, 1]) <= 10.02).all()
    gj = json.loads((out / "boundary.geojson").read_text(encoding="utf-8"))
    assert gj["features"][0]["geometry"]["type"] == "Polygon"
    HANDLERS["terrain"](project, out, cfg, cache)
    _apply_boundary("terrain", out, cache)
    with rasterio.open(out / "dtm.tif") as src:
        data = src.read(1); transform = src.transform; nodata = src.nodata
    rows, cols = np.indices(data.shape)
    east, north = transform * (cols + .5, rows + .5)
    local_x = east - frame.origin_lv95[0] - center[0]; local_y = north - frame.origin_lv95[1] - center[1]
    from shapely import contains_xy
    from shapely.geometry import box
    far = ~contains_xy(box(-10, -8, 10, 8).buffer(2.05), local_x, local_y)
    assert far.any() and (data[far] == nodata).all()
    inner = (np.abs(local_x) < 8) & (np.abs(local_y) < 6)
    assert (data[inner] != nodata).all()


def test_apply_boundary_filters_tin_and_pavement(tmp_path: Path) -> None:
    import pickle
    from shapely.geometry import box
    from samcloud.drone.contracts import LabeledTin, PavementPolygon
    from samcloud.drone.run_environment import _apply_boundary
    cache = tmp_path / "cache"; cache.mkdir()
    pickle.dump(box(0, 0, 10, 10), (cache / "boundary.pkl").open("wb"))
    verts = np.array([[1, 1, 0], [4, 1, 0], [1, 4, 0], [20, 20, 0], [24, 20, 0], [20, 24, 0.0]])
    pickle.dump(LabeledTin(verts, np.array([[0, 1, 2], [3, 4, 5]]), np.array([1, 1], np.uint8)), (cache / "tin.pkl").open("wb"))
    _apply_boundary("tin", tmp_path, cache)
    assert len(pickle.load((cache / "tin.pkl").open("rb")).triangles) == 1
    pave = {"polygons": [PavementPolygon(1, "road", box(5, 5, 15, 15)), PavementPolygon(2, "path", box(30, 30, 31, 31))],
            "lawns": [box(8, 8, 12, 12), box(40, 40, 41, 41)], "curbs": []}
    pickle.dump(pave, (cache / "pavement.pkl").open("wb"))
    _apply_boundary("pavement", tmp_path, cache)
    got = pickle.load((cache / "pavement.pkl").open("rb"))
    assert len(got["polygons"]) == 1 and abs(got["polygons"][0].polygon.area - 25) < 1e-9
    assert len(got["lawns"]) == 1 and abs(got["lawns"][0].area - 4) < 1e-9


def test_real_perimeter_kml_area() -> None:
    if not REAL_KML.is_file():
        return
    poly = load_perimeter_kml(REAL_KML)
    local = perimeter_to_local(poly, LocalFrame((0.0, 0.0, 0.0)))
    assert 1000 < local.area < 10000


def test_boundary_dxf_layer_with_one_closed_polyline(tmp_path: Path) -> None:
    import ezdxf
    from shapely.geometry import box
    from samcloud.drone.run_environment import _boundary_dxf_entities
    from samcloud.dxf.writer import build_dxf_document, ensure_outdoor_layers
    from samcloud.outdoor_model.scene import assemble_dxf
    frame = LocalFrame((2_699_800.0, 1_221_100.0, 450.0))
    entities = {"boundary": _boundary_dxf_entities(box(0, 0, 40, 30).buffer(2), frame)}
    result = assemble_dxf(entities, frame, load_drone_config())
    doc = build_dxf_document(result, coordinates="local", local_origin=False, origin_offset=frame.origin_lv95)
    ensure_outdoor_layers(doc)
    path = tmp_path / "b.dxf"; doc.saveas(path)
    back = ezdxf.readfile(path)
    polys = [e for e in back.modelspace() if e.dxf.layer == "S-ANNO-GRENZE"]
    assert len(polys) == 1 and polys[0].dxftype() == "LWPOLYLINE" and polys[0].closed
    assert back.layers.get("S-ANNO-GRENZE").dxf.color == 1
    assert min(p[0] for p in polys[0].get_points("xy")) > 2_699_000
    assert _boundary_dxf_entities(None, frame) == {}
