"""Clean terrain model: surface polygons, DTM bumps, smooth pavements, deviation report, line/arc edges."""
from __future__ import annotations

import ezdxf
import numpy as np
import pytest
from ezdxf.math import bulge_to_arc
from shapely.geometry import LineString, Point, Polygon, box

from samcloud.drone import contracts as c
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import build_dxf_document
from samcloud.environment import topology
from samcloud.outdoor_model import pavement as pv
from samcloud.terrain.clean import clean_dtm, smooth_pavement_surfaces, tin_deviation
from samcloud.terrain.edges import arc_geometry, densify, fit_chain, fit_surfaces, tangent_gap_deg
from samcloud.terrain.tin import build_labeled_tin

CFG = c.load_drone_config()
E0, N0, H0 = 2_600_000.0, 1_200_000.0, 430.0


# ---------------------------------------------------------------------------------- 1. surface polygons


def _ragged(poly: Polygon, rng, sigma: float = 0.02) -> Polygon:
    """Densify the outline and jitter it (a rasterised outline is never straight)."""
    ring = np.asarray(poly.exterior.coords)
    pts = []
    for a, b in zip(ring[:-1], ring[1:]):
        n = max(int(np.hypot(*(b - a)) / 0.1), 1)
        pts.extend(a + (b - a) * k / n for k in range(n))
    pts = np.asarray(pts) + rng.normal(0, sigma, (len(pts), 2))
    return Polygon(pts)


def test_ragged_lawn_becomes_one_clean_polygon_next_to_the_sidewalk() -> None:
    rng = np.random.default_rng(11)
    main = _ragged(box(0.0, 3.0, 20.0, 12.0), rng)
    holes = [box(5.0, 6.0, 5.5, 6.5), box(12.0, 8.0, 12.4, 8.3)]            # < 2 m2: filled
    lawn = main
    for hole in holes:
        lawn = lawn.difference(hole)
    specks = [box(25.0, 5.0, 25.5, 5.5), box(27.0, 9.0, 27.6, 9.4), box(23.0, 2.0, 23.4, 2.5)]  # < 4 m2: dropped
    sliver = box(19.9, 6.0, 26.0, 6.3)                                       # 0.3 m wide: removed
    lawns = [lawn, *specks, sliver]
    sidewalk = c.PavementPolygon(0, "sidewalk", _ragged(box(0.0, 0.0, 20.0, 3.05), rng), evidence=c.Evidence(points=10))
    topo = pv.pavement_topology([sidewalk], lawns, [], config=CFG)
    assert len(topo.lawns) == 1, [p.area for p in topo.lawns]
    clean = topo.lawns[0]
    assert len(clean.interiors) == 0, "small holes must be filled"
    assert clean.area > 150.0
    assert clean.bounds[2] < 20.6, "the sliver must be gone"
    assert not clean.contains(Point(23.0, 6.15)) and not clean.intersects(box(24.5, 4.5, 28.0, 10.0))
    sidewalk_after = [p.polygon for p in topo.polygons if p.kind == "sidewalk"]
    assert len(sidewalk_after) == 1
    report = topology.gap_overlap_report(sidewalk_after, topo.lawns)
    assert report["overlap_cm2"] < 1.0 and report["gap_cm2"] < 1.0, report
    assert topo.worst_cm2() < CFG["pavement"]["max_gap_overlap_cm2"]
    assert all(p.is_valid for p in topo.lawns)


def test_narrow_sliver_is_merged_into_the_neighbour_with_the_longest_shared_edge() -> None:
    wide = box(0, 0, 10, 10)
    strip = box(10, 0, 10.4, 10)                      # 0.4 m wide next to the road, away from the lawn
    lawn = box(0, 10, 10, 20)
    groups, counts = topology.absorb_slivers([[wide], [], [], [strip, lawn]], 0.6, names=("sidewalk", "road", "path", "lawn"))
    assert counts.get("lawn", 0) >= 1
    assert groups[0][0].area == pytest.approx(100.0 + 4.0, rel=0.01)
    assert sum(p.area for p in groups[3]) == pytest.approx(100.0, rel=0.01)


# ---------------------------------------------------------------------------------- 2. DTM bumps


def _bump_scene(seed: int = 4):
    rng = np.random.default_rng(seed)
    res, n = 0.25, 240
    row, col = np.mgrid[0:n, 0:n]
    x, y = col * res, (n - 1 - row) * res
    truth = 0.05 * x + 0.02 * y + np.where(x > 35.0, 1.0, 0.0)        # slope + 1 m retaining wall step at x = 35
    z = truth + rng.normal(0.0, 0.01, truth.shape)
    centres = ((10.0, 10.0), (20.0, 45.0), (45.0, 30.0))
    for cx, cy in centres:                                            # 0.5 m high, 3 m wide
        z = z + 0.5 * np.clip(1.0 - np.hypot(x - cx, y - cy) / 1.5, 0.0, 1.0)
    return z, truth, x, y, centres, res


def test_mounds_are_removed_and_the_retaining_wall_stays() -> None:
    z, truth, x, y, centres, res = _bump_scene()
    out, report = clean_dtm(z, np.ones(z.shape, np.uint8), res, config=CFG)
    for cx, cy in centres:
        near = np.hypot(x - cx, y - cy) < 2.5
        assert float(np.abs(out - truth)[near].max()) < 0.03, (cx, cy)
    assert report["bumps_removed"] >= 3 and report["area_removed_m2"] > 10.0
    step = float(np.median(out[:, 150] - out[:, 130]))
    assert step == pytest.approx(float(np.median(truth[:, 150] - truth[:, 130])), abs=0.03)
    wall = (np.abs(x - 35.0) < 0.6) & (np.abs(y - 25.0) < 8.0)
    assert float(np.abs(out - truth)[wall].max()) < 0.05, "the step edge must not be rounded off"


def test_protected_cells_and_large_hills_are_left_alone() -> None:
    z, _truth, x, y, _centres, res = _bump_scene()
    protect = np.hypot(x - 10.0, y - 10.0) < 3.0
    out, _ = clean_dtm(z, np.ones(z.shape, np.uint8), res, protect=protect, config=CFG)
    assert np.allclose(out[protect], z[protect]), "pavement/curb cells are never modified"
    radius = np.hypot(x - 30.0, y - 15.0)
    hill = np.where(radius < 12.0, 0.5 * (1.0 + np.cos(np.pi * radius / 12.0)), 0.0)  # smooth, 24 m wide: real terrain
    out2, _ = clean_dtm(0.0 * z + hill, np.ones(z.shape, np.uint8), res, config=CFG)
    assert float(np.abs(out2 - hill).max()) < 0.2 and float(out2.max()) > 0.8


# ---------------------------------------------------------------------------------- 3. smooth road


def _noisy_road(seed: int = 8):
    rng = np.random.default_rng(seed)
    res = 0.25
    axis = np.arange(0.0, 20.0 + res / 2, res)
    gx, gy = np.meshgrid(axis, axis)
    plane = 0.02 * gx + 0.01 * gy
    return gx, gy, plane, plane + rng.normal(0.0, 0.03, plane.shape), res


def test_noisy_road_is_smoothed_and_needs_far_fewer_triangles() -> None:
    gx, gy, plane, noisy, res = _noisy_road()
    road = box(2.05, 2.05, 17.95, 17.95)
    bounds = (0.0, 0.0, 20.0, 20.0)
    raster = noisy[::-1]                                                  # north-up raster
    smooth_raster, report = smooth_pavement_surfaces(raster, bounds, res, [road], config=CFG)
    assert report["elements"] == 1
    smooth = smooth_raster[::-1]
    inside = np.array([road.contains(Point(a, b)) for a, b in zip(gx.ravel(), gy.ravel())]).reshape(gx.shape)
    rms = float(np.sqrt(np.mean((smooth - plane)[inside] ** 2)))
    assert rms < 0.01, rms

    def tin_for(heights, **kw):
        ground = np.column_stack((gx.ravel(), gy.ravel(), heights.ravel()))
        return build_labeled_tin(ground, regions={"road": [road]}, initial_spacing=2.0, max_vertices=20000, **kw)

    tin_smooth = tin_for(smooth, tolerance=0.04, tolerance_by_label={"road": 0.02, "terrain": 0.04})
    tin_noisy = tin_for(noisy, tolerance=0.03)
    road_code = c.TIN_LABELS["road"]
    smooth_road, noisy_road = int((tin_smooth.labels == road_code).sum()), int((tin_noisy.labels == road_code).sum())
    assert smooth_road * 4 < noisy_road, (smooth_road, noisy_road)
    truth_points = np.column_stack((gx.ravel() + res / 2, gy.ravel() + res / 2, (plane + 0.02 * res / 2 + 0.01 * res / 2).ravel()))
    deviation = tin_deviation(tin_smooth, truth_points)
    assert deviation["labels"]["road"]["rms_m"] < 0.01, deviation


# ---------------------------------------------------------------------------------- 4. deviation report


def test_deviation_is_reported_per_label() -> None:
    res = 0.5
    axis = np.arange(0.0, 20.0 + res / 2, res)
    gx, gy = np.meshgrid(axis, axis)
    ground = np.column_stack((gx.ravel(), gy.ravel(), 0.02 * gx.ravel()))
    tin = build_labeled_tin(ground, regions={"road": [box(2, 2, 10, 10)], "lawn": [box(12, 2, 18, 10)]})
    rng = np.random.default_rng(2)
    pts_xy = rng.uniform(1.3, 18.7, (4000, 2))
    z = 0.02 * pts_xy[:, 0]
    in_road = (pts_xy[:, 0] > 2.1) & (pts_xy[:, 0] < 9.9) & (pts_xy[:, 1] > 2.1) & (pts_xy[:, 1] < 9.9)
    z = z + np.where(in_road, 0.03, 0.0) + np.where(pts_xy[:, 0] > 12.1, 0.05, 0.0)
    deviation = tin_deviation(tin, np.column_stack((pts_xy, z)))
    labels = deviation["labels"]
    assert {"terrain", "road", "lawn"} <= set(labels)
    assert labels["road"]["rms_m"] == pytest.approx(0.03, abs=0.004)
    assert labels["lawn"]["rms_m"] == pytest.approx(0.05, abs=0.004)
    for row in labels.values():
        assert row["rms_m"] <= row["p95_m"] + 1e-9 <= row["max_m"] + 2e-9
        assert row["points"] > 0


# ---------------------------------------------------------------------------------- F5 edges


def _curve(rng, sigma: float = 0.0):
    """Line (0,0)-(10,0), tangent arc R=12 (centre (10,12)), line up to (22,22)."""
    ang = np.linspace(-np.pi / 2, 0.0, 200)
    arc = np.column_stack((10 + 12 * np.cos(ang), 12 + 12 * np.sin(ang)))
    l1 = np.column_stack((np.linspace(0, 10, 100, endpoint=False), np.zeros(100)))
    l2 = np.column_stack((np.full(100, 22.0), np.linspace(12, 22, 100)))
    xy = np.vstack((l1, arc, l2))
    noisy = xy + rng.normal(0.0, sigma, xy.shape)
    noisy[0], noisy[-1] = xy[0], xy[-1]
    return xy, noisy


def test_noisy_outline_is_fitted_as_two_lines_and_one_arc() -> None:
    truth, noisy = _curve(np.random.default_rng(5), 0.015)
    chain = fit_chain(noisy, config=CFG)
    assert (chain.lines, chain.arcs) == (2, 1), (chain.lines, chain.arcs, chain.report)
    dense, _seg = densify(chain, 0.01)
    deviation = float(max(np.max([Point(p).distance(LineString(dense)) for p in truth]),
                          np.max([Point(p).distance(LineString(truth)) for p in dense])))
    assert deviation < 0.05, deviation
    arc_index = int(np.flatnonzero(np.abs(chain.bulges) > 1e-9)[0])
    centre, radius, _a0, sweep = arc_geometry(chain.pts[arc_index], chain.pts[arc_index + 1], chain.bulges[arc_index])
    assert radius == pytest.approx(12.0, abs=0.4) and abs(np.degrees(sweep)) == pytest.approx(90.0, abs=3.0)
    assert tangent_gap_deg(chain) < 1.0, "line and arc must join tangentially"
    assert chain.report["vertices_after"] < chain.report["vertices_before"] / 20


def test_road_and_sidewalk_share_one_fitted_curb_without_gap_or_overlap() -> None:
    rng = np.random.default_rng(9)
    _truth, curve = _curve(rng, 0.012)
    road = Polygon(np.vstack((curve, [(30, 22), (30, -10), (0, -10)])))
    walk = Polygon(np.vstack((curve, [(0, 22)])))
    curb_xy = _curve(np.random.default_rng(10), 0.012)[1]
    curb = c.CurbAxis(3, np.column_stack((curb_xy, 0.1 + 0.01 * curb_xy[:, 0])))
    fit = fit_surfaces({"road": [road], "sidewalk": [walk]}, [curb], config=CFG)
    assert "fallback" not in fit.report, fit.report
    road_f, walk_f = fit.regions["road"][0], fit.regions["sidewalk"][0]
    assert len(road_f) == 1 and len(walk_f) == 1
    road_f, walk_f = road_f[0], walk_f[0]
    report = topology.gap_overlap_report([road_f], [walk_f])
    assert report["overlap_cm2"] < 1.0 and report["gap_cm2"] < 1.0, report
    assert road_f.union(walk_f).area == pytest.approx(road.union(walk).area, rel=0.01)
    assert len(road_f.union(walk_f).interiors) == 0
    assert fit.report["arcs"] >= 1
    (fitted_curb,) = fit.curbs
    assert fitted_curb.shared and fitted_curb.line.arcs == 1 and fitted_curb.line.lines == 2
    # both neighbours run exactly along the fitted curb (identical geometry, not two similar fits)
    for xy in fitted_curb.dense_xyz[:, :2]:
        assert road_f.boundary.distance(Point(xy)) < 1e-6 and walk_f.boundary.distance(Point(xy)) < 1e-6
    assert fitted_curb.dense_xyz[:, 2].min() >= 0.09


def test_lwpolyline_bulges_round_trip_through_dxf(tmp_path) -> None:
    centre, radius = np.array([10.0, 12.0]), 12.0
    start, end = centre + radius * np.array([0.0, -1.0]), centre + radius * np.array([1.0, 0.0])
    bulge = np.tan(np.pi / 2 / 4)                                           # counter-clockwise quarter circle
    pts = [(E0 + 0.0, N0 + 0.0, H0), (E0 + start[0], N0 + start[1], H0), (E0 + end[0], N0 + end[1], H0), (E0 + 22.0, N0 + 22.0, H0)]
    entity = {"type": "lwpolyline", "points": pts, "closed": False, "bulges": [0.0, bulge, 0.0], "elevation": H0}
    layers = {"S-PAVE": [entity]}
    result = DxfPlanResult("t", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "t", layers, {},
                           dict.fromkeys(layers, 7), BackgroundImage(None, None, (E0, N0, E0 + 10, N0 + 10)))
    path = tmp_path / "bulge.dxf"
    build_dxf_document(result).saveas(path)
    polyline = next(e for e in ezdxf.readfile(path).modelspace().query('LWPOLYLINE[layer=="S-PAVE"]'))
    points = polyline.get_points("xyb")
    assert len(points) == 4 and points[1][4 - 2] == pytest.approx(bulge)
    arc_centre, a0, a1, arc_radius = bulge_to_arc((points[1][0], points[1][1]), (points[2][0], points[2][1]), points[1][2])
    assert arc_radius == pytest.approx(radius, abs=1e-6)
    mid_angle = (a0 + (a1 - a0 + 2 * np.pi) % (2 * np.pi) / 2) if a1 < a0 else (a0 + a1) / 2
    mid = np.array(arc_centre) + radius * np.array([np.cos(mid_angle), np.sin(mid_angle)])
    expected = np.array([E0, N0]) + centre + radius * np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4)])
    assert np.hypot(*(mid - expected)) < 0.01
