"""Tests for samcloud.environment.pavement."""
import numpy as np

from samcloud.drone import contracts as c
from samcloud.environment import pavement as pv

DENSITY = 400.0  # points per m2


def _rect(rng, x0, y0, x1, y1, code, sigma=0.01):
    n = int((x1 - x0) * (y1 - y0) * DENSITY)
    xy = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)]) + rng.normal(0, sigma, (n, 2))
    xyz = np.column_stack([xy, rng.normal(0, 0.01, n)])
    return xyz, np.full(n, code, dtype=np.uint16)


def _ps(parts):
    return c.PointSet(xyz=np.vstack([p[0] for p in parts]), classes=np.concatenate([p[1] for p in parts]))


def test_road_and_sidewalk_areas():
    rng = np.random.default_rng(11)
    ps = _ps([_rect(rng, 0.03, 0.07, 30.03, 8.07, c.ROAD), _rect(rng, 0.03, 8.57, 30.03, 11.57, c.SIDEWALK)])
    polys = pv.pavement_polygons(ps)
    by_kind = {p.kind: p for p in polys}
    assert sorted(by_kind) == ["road", "sidewalk"] and len(polys) == 2
    assert abs(by_kind["road"].polygon.area - 240.0) / 240.0 < 0.03
    assert abs(by_kind["sidewalk"].polygon.area - 90.0) / 90.0 < 0.03
    assert all(p.polygon.is_valid and p.status == "ok" and p.evidence.points > 1000 for p in polys)


def test_small_blobs_dropped_and_reported():
    rng = np.random.default_rng(12)
    parts = [_rect(rng, 0, 0, 20, 6, c.ROAD), _rect(rng, 40, 0, 41.2, 1.2, c.SIDEWALK),
             _rect(rng, 50, 5, 50.9, 5.9, c.PATH)]
    stray = (np.column_stack([rng.uniform(60, 80, 15), rng.uniform(0, 20, 15), np.zeros(15)]),
             np.full(15, c.ROAD, dtype=np.uint16))
    dropped: list[dict] = []
    polys = pv.pavement_polygons(_ps(parts + [stray]), dropped=dropped)
    assert [p.kind for p in polys] == ["road"]
    kinds = sorted(d["kind"] for d in dropped)
    assert kinds == ["path", "sidewalk"]
    assert all(d["area"] < 2.0 for d in dropped)


def test_markings_and_manhole_do_not_create_holes():
    rng = np.random.default_rng(13)
    road_xyz, road_cls = _rect(rng, 0, 0, 20, 8, c.ROAD)
    mark = (road_xyz[:, 0] > 5) & (road_xyz[:, 0] < 8) & (road_xyz[:, 1] > 3) & (road_xyz[:, 1] < 3.4)
    man = np.hypot(road_xyz[:, 0] - 14, road_xyz[:, 1] - 4) < 0.35
    road_cls = road_cls.copy()
    road_cls[mark] = c.ROAD_MARKING
    road_cls[man] = c.MANHOLE
    assert mark.sum() > 100 and man.sum() > 100
    polys = pv.pavement_polygons(_ps([(road_xyz, road_cls)]))
    assert len(polys) == 1
    assert len(polys[0].polygon.interiors) == 0
    assert abs(polys[0].polygon.area - 160.0) / 160.0 < 0.03


def test_without_marking_class_hole_is_filled_when_small():
    rng = np.random.default_rng(14)
    xyz, cls = _rect(rng, 0, 0, 20, 8, c.ROAD)
    keep = np.hypot(xyz[:, 0] - 10, xyz[:, 1] - 4) > 0.6
    polys = pv.pavement_polygons(_ps([(xyz[keep], cls[keep])]))
    assert len(polys) == 1 and len(polys[0].polygon.interiors) == 0


def test_lawn_polygons_and_class_raster():
    rng = np.random.default_rng(15)
    ps = _ps([_rect(rng, 0, 0, 10, 10, c.LOW_VEGETATION), _rect(rng, 12, 0, 20, 6, c.ROAD)])
    lawns = pv.lawn_polygons(ps)
    assert len(lawns) == 1 and abs(lawns[0].area - 100.0) / 100.0 < 0.03
    mask, grid = pv.class_raster(ps, [c.ROAD], resolution=0.5)
    assert mask.shape == grid.shape and mask.any() and not mask[:, :20].any()
