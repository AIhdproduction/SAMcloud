"""Tests for samcloud.environment.topology."""
import numpy as np
from shapely.geometry import LineString, Polygon, box

from samcloud.drone import contracts as c
from samcloud.environment import topology as tp

CURB = LineString([(10.0, 0.0), (10.0, 20.0)])


def _road_sidewalk():
    # Road: 3 cm gap to the curb for y<10, 2 cm overlap past the curb for y>=10.
    road = Polygon([(0, 0), (9.97, 0), (9.97, 10), (10.02, 10), (10.02, 20), (0, 20)])
    sidewalk = box(10.0, 0.0, 14.0, 20.0)
    return road, sidewalk


def test_snap_removes_gap_and_overlap_and_shares_curb():
    road, sidewalk = _road_sidewalk()
    before = tp.gap_overlap_report([road], [sidewalk])
    assert before["gap_cm2"] > 100 and before["overlap_cm2"] > 100
    r = tp.snap_to_breakline([road], [CURB], tolerance=0.05)
    s = tp.snap_to_breakline([sidewalk], [CURB], tolerance=0.05)
    rep = tp.gap_overlap_report(r, s)
    assert rep["gap_cm2"] + rep["overlap_cm2"] < 1.0 and rep["ok"]
    for poly in (r[0], s[0]):
        along = poly.boundary.intersection(CURB.buffer(1e-6))
        assert along.length > 19.9


def test_all_outputs_valid_and_types_preserved():
    road, sidewalk = _road_sidewalk()
    pp = c.PavementPolygon(pavement_id=3, kind="road", polygon=road)
    out = tp.snap_to_breakline([pp], [CURB], tolerance=0.05)
    assert isinstance(out[0], c.PavementPolygon) and out[0].pavement_id == 3 and out[0].kind == "road"
    bow = Polygon([(9.9, 0), (10.1, 5), (9.9, 5), (10.1, 0)])  # self-intersecting
    res = tp.snap_to_breakline([road, sidewalk, bow], [CURB], tolerance=0.05)
    assert res and all(p.is_valid for p in res)
    groups = tp.resolve_overlaps([[sidewalk], [road]], gap_tolerance=0.05)
    assert all(p.is_valid for g in groups for p in g)


def test_report_known_overlap_and_gap():
    a = box(0, 0, 10, 10)
    b = box(9.95, 0, 20, 10)  # 0.05 x 10 = 0.5 m2 overlap
    rep = tp.gap_overlap_report([a], [b])
    assert abs(rep["overlap_m2"] - 0.5) < 1e-9 and rep["gap_m2"] < 1e-9 and not rep["ok"]
    assert rep["is_valid"]
    g = tp.gap_overlap_report([box(0, 0, 10, 10)], [box(10.04, 0, 20, 10)], tolerance=0.05)
    assert abs(g["gap_m2"] - 0.4) < 0.01 and g["overlap_m2"] == 0.0
    clipped = tp.gap_overlap_report([a], [b], region=box(0, 0, 10, 5))
    assert abs(clipped["overlap_m2"] - 0.25) < 1e-9


def test_resolve_overlaps_priority_and_gap_closing():
    sidewalk = box(10.0, 0, 14, 10)
    road = box(0, 0, 10.5, 10)        # overlaps sidewalk by 0.5 m
    lawn = box(14.03, 0, 20, 10)      # 3 cm gap to sidewalk
    out_sw, out_road, out_lawn = tp.resolve_overlaps([[sidewalk], [road], [lawn]], gap_tolerance=0.05)
    assert 40.0 - 1e-6 <= sum(p.area for p in out_sw) <= 40.31  # may receive the 0.3 m2 gap
    assert abs(sum(p.area for p in out_road) - 100.0) < 1e-6
    # gap goes to a neighbour with a long shared edge; no gap remains
    total = sum(p.area for g in (out_sw, out_road, out_lawn) for p in g)
    assert abs(total - 200.0) < 1e-6
    rep = tp.gap_overlap_report(out_sw, out_lawn)
    assert rep["gap_cm2"] + rep["overlap_cm2"] < 1.0


def test_pairwise_report_only_neighbours():
    named = {"road": [box(0, 0, 10, 10)], "sidewalk": [box(10, 0, 14, 10)], "far": [box(100, 100, 110, 110)]}
    rep = tp.pairwise_report(named)
    assert list(rep) == [("road", "sidewalk")]
    assert rep[("road", "sidewalk")]["ok"]
    assert np.isclose(rep[("road", "sidewalk")]["overlap_m2"], 0.0)
