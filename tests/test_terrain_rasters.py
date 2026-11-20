"""Tests for terrain rasters, GeoTIFF export, filter comparison and the orchestrator."""
from __future__ import annotations

import json

import numpy as np
import pytest

from samcloud.drone.contracts import BUILDING, GROUND, ROAD, LocalFrame, PointSet
from samcloud.terrain.compare import compare_ground_filters, pick_best
from samcloud.terrain.rasters import (
    build_dsm_raster, build_dtm_raster, choose_resolution, export_terrain, ground_mask,
    write_geotiff,
)

BOUNDS = (0.0, 0.0, 40.0, 40.0)


def _scene(seed: int = 5) -> PointSet:
    """Flat road/ground 40x40 m with a 10x10 m, 6 m high building."""
    rng = np.random.default_rng(seed)
    n = 60000
    xy = rng.uniform(0.01, 39.99, (n, 2))
    under = (xy[:, 0] > 14.9) & (xy[:, 0] < 25.1) & (xy[:, 1] > 14.9) & (xy[:, 1] < 25.1)
    xy = xy[~under]
    ground = np.column_stack([xy, 100.0 + rng.normal(0, 0.01, len(xy))])
    g_cls = np.where(xy[:, 0] < 20, ROAD, GROUND).astype(np.uint16)
    m = 6000
    roof = np.column_stack([rng.uniform(15.013, 24.987, m), rng.uniform(15.013, 24.987, m),
                            106.0 + rng.normal(0, 0.01, m)])
    walls = np.column_stack([np.full(1200, 15.0), rng.uniform(15, 25, 1200), rng.uniform(100.3, 105.9, 1200)])
    bld = np.vstack([roof, walls])
    xyz = np.vstack([ground, bld])
    classes = np.concatenate([g_cls, np.full(len(bld), BUILDING, dtype=np.uint16)])
    return PointSet(xyz=xyz, classes=classes)


def _grid_points(res: float, keep) -> np.ndarray:
    """One slightly jittered point per cell centre of a 40x40 raster at ``res``."""
    rng = np.random.default_rng(11)
    n = int(40 / res)
    cx = (np.arange(n) + 0.5) * res
    gx, gy = np.meshgrid(cx, cx)
    pts = np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-0.2, 0.2, (n * n, 2)) * res
    pts = pts[keep(pts)]
    return np.column_stack([pts, 50.0 + 0.01 * pts[:, 0] + rng.normal(0, 0.001, len(pts))])


def test_dtm_validity_bands_hole_in_middle():
    res = 0.5
    pts = _grid_points(res, lambda p: ~((np.abs(p[:, 0] - 20) < 5) & (np.abs(p[:, 1] - 20) < 5)))
    dtm, validity, dist = build_dtm_raster(pts, resolution=res, bounds=BOUNDS, max_interp_distance=2.0)
    assert dtm.shape == (80, 80)
    centre = (40, 40)                                         # (20 m, 20 m), ~5 m from the hole border
    assert validity[centre] == 0 and np.isnan(dtm[centre])
    assert dist[centre] == pytest.approx(5.0, abs=0.6)
    near = (40, int(16.5 / res))                              # 1.5 m inside the hole edge at x = 15
    assert validity[near] == 2 and np.isfinite(dtm[near])
    assert dist[near] <= 2.0
    assert (validity == 1).sum() > 5000
    assert np.all(dist[validity == 1] == 0)
    measured = np.argwhere(validity == 1)
    for cell in [near, (35, 40), centre]:
        brute = np.sqrt(((measured - np.array(cell)) ** 2).sum(axis=1)).min() * res
        assert dist[cell] == pytest.approx(brute, abs=1e-4)
    x_near = (near[1] + 0.5) * res
    assert dtm[near] == pytest.approx(50.0 + 0.01 * x_near, abs=0.02)
    assert not np.isfinite(dtm[validity == 0]).any()
    assert (validity == 2).any() and (validity == 0).any()


def test_dsm_takes_roof_heights_and_fills_small_holes():
    ps = _scene()
    dsm, validity = build_dsm_raster(ps.xyz, resolution=1.0, bounds=BOUNDS)
    roof_cell = (20, 20)
    assert dsm[roof_cell] == pytest.approx(106.0, abs=0.05)
    assert dsm[(35, 5)] == pytest.approx(100.0, abs=0.05)
    assert validity[roof_cell] == 1
    pts = ps.xyz[ps.xyz[:, 2] < 101]

    def cell(x, y):
        return (np.floor(pts[:, 0]) == x) & (np.floor(pts[:, 1]) == y)

    drop = cell(3, 3) | cell(8, 8) | cell(9, 8) | cell(30, 30) | cell(31, 30) | cell(32, 30)
    d2, v2 = build_dsm_raster(pts[~drop], resolution=1.0, bounds=BOUNDS)
    assert v2[39 - 3, 3] == 2 and np.isfinite(d2[39 - 3, 3])
    assert v2[39 - 8, 8] == 2 and v2[39 - 8, 9] == 2
    assert v2[39 - 30, 30] == 0 and np.isnan(d2[39 - 30, 31])


def test_choose_resolution_never_finer_than_target_and_density_limited():
    rng = np.random.default_rng(8)
    dense = rng.uniform(0, 30, (400000, 2))
    dsm, dtm = choose_resolution(np.column_stack([dense, np.zeros(len(dense))]))
    assert dsm == pytest.approx(0.10) and dtm == pytest.approx(0.25)
    sparse = rng.uniform(0, 100, (30000, 2))                  # ~3 points / m2
    dsm, dtm = choose_resolution(np.column_stack([sparse, np.zeros(len(sparse))]))
    assert dsm >= 1.0 and dtm >= 1.0


def test_geotiff_round_trip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from pyproj import CRS

    frame = LocalFrame(origin_lv95=(2600000.0, 1200000.0, 430.0))
    rng = np.random.default_rng(3)
    arr = rng.normal(100, 1, (30, 50)).astype(np.float32)
    arr[3, 4] = np.nan
    bounds = (10.0, 5.0, 22.5, 12.5)                          # 50 cols x 30 rows at 0.25 m
    out = write_geotiff(tmp_path / "t.tif", arr, bounds_local=bounds, resolution=0.25, frame=frame)
    with rasterio.open(out) as src:
        assert src.crs is not None
        horizontal = src.crs.to_epsg() or CRS.from_wkt(src.crs.to_wkt()).sub_crs_list[0].to_epsg()
        assert horizontal == 2056
        assert src.transform.c == pytest.approx(2600000.0 + 10.0)
        assert src.transform.f == pytest.approx(1200000.0 + 12.5)
        assert src.transform.a == pytest.approx(0.25) and src.transform.e == pytest.approx(-0.25)
        assert src.tags()["VERTICAL_DATUM"] == "LN02"
        assert src.nodata == -9999
        back = src.read(1)
    assert back.shape == arr.shape and back.dtype == np.float32
    ok = np.isfinite(arr)
    assert np.allclose(back[ok], arr[ok] + 430.0, atol=1e-3)   # heights are LN02 = local + frame height
    assert back[3, 4] == -9999
    out2 = write_geotiff(tmp_path / "v.tif", np.ones((30, 50), dtype=np.uint8), bounds_local=bounds,
                         resolution=0.25, frame=frame)
    with rasterio.open(out2) as src:
        assert src.dtypes[0] == "uint8" and src.read(1).min() == 1


def test_compare_prefers_mask_excluding_building_points():
    ps = _scene()
    good = ps.classes != BUILDING
    bad = np.ones(len(ps), dtype=bool)
    metrics = compare_ground_filters(ps, {"good": good, "bad": bad})
    assert metrics["good"]["footprint_share"] < metrics["bad"]["footprint_share"]
    name, reason = pick_best(metrics)
    assert name == "good" and "penalty" in reason
    profile = np.array([[5.0, 20.0], [35.0, 20.0]])           # crosses the building
    with_profile = compare_ground_filters(ps, {"good": good, "bad": bad}, profiles=[profile])
    assert with_profile["good"]["profile_error_m"] < 0.1
    assert pick_best(with_profile)[0] == "good"


def test_ground_mask_never_marks_building_and_matches_filters():
    ps = _scene()
    for method in ("pmf", "csf"):
        mask = ground_mask(ps, method)
        assert not mask[ps.classes == BUILDING].any()
        assert mask[ps.classes != BUILDING].mean() > 0.9
    with pytest.raises(ValueError):
        ground_mask(ps, "nope")


def test_export_terrain_writes_all_files(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    ps = _scene()
    frame = LocalFrame(origin_lv95=(2683000.0, 1248000.0, 430.0))
    for chosen in ("csf", "auto"):
        cfg = {"terrain": {"dsm_resolution_m": 0.5, "dtm_resolution_m": 1.0, "ground_filter": chosen}}
        out_dir = tmp_path / chosen
        report = export_terrain(ps, out_dir, frame, config=cfg)
        for name in ("dtm.tif", "dtm_validity.tif", "dsm.tif", "dsm_validity.tif", "hillshade.png",
                     "terrain_report.json"):
            assert (out_dir / name).is_file(), name
        data = json.loads((out_dir / "terrain_report.json").read_text(encoding="utf-8"))
        assert data["filter"]["used"] in ("pmf", "csf")
        if chosen == "csf":
            assert data["filter"]["used"] == "csf"
        else:
            assert "reason" in data["filter"]
        assert data["dtm"]["resolution_m"] >= 1.0 and data["dsm"]["resolution_m"] >= 0.5
        assert 0 < data["dtm"]["validity"]["measured"] <= 1
        assert set(report["files"]) >= {"dtm", "dsm", "hillshade", "report"}
        with rasterio.open(out_dir / "dsm.tif") as src:
            assert src.read(1).max() == pytest.approx(536.0, abs=0.1)      # roof 106 + 430
            assert src.tags()["VERTICAL_DATUM"] == "LN02"
        with rasterio.open(out_dir / "dtm.tif") as src:
            vals = src.read(1)
            assert vals[vals != -9999].max() == pytest.approx(530.0, abs=0.1)   # no roof in the DTM
