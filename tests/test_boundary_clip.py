from __future__ import annotations

import pickle

import numpy as np
from shapely.geometry import box

from samcloud.bim.core.model import FenceElement, FenceParams, ManholeElement, ManholeParams, TreeElement, TreeParams
from samcloud.drone.contracts import CurbAxis
from samcloud.drone.run_environment import _apply_boundary
from samcloud.outdoor_model.vegetation import VegetationResult


def _dump(path, value):
    with path.open("wb") as fh:
        pickle.dump(value, fh)


def _load(path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _setup(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _dump(cache / "boundary.pkl", box(0, 0, 100, 100))
    return tmp_path, cache


def test_curb_crossing_boundary_twice_keeps_inside_parts_with_z(tmp_path):
    out, cache = _setup(tmp_path)
    # leaves at x=100 (z 10..), re-enters at x=100 again after a detour; tiny stub inside at start
    pts = np.array([[99.8, 50.0, 0.0], [100.5, 50.0, 7.0], [100.5, 60.0, 8.0], [90.0, 60.0, 9.0], [40.0, 60.0, 14.0],
                    [-10.0, 60.0, 19.0]])
    curb_path = np.array([[-5.0, 10.0, 0.0], [50.0, 10.0, 5.5], [105.0, 10.0, 11.0], [105.0, 20.0, 11.0],
                          [50.0, 20.0, 5.5]])
    _dump(cache / "pavement.pkl", {"polygons": [], "lawns": [],
                                   "curbs": [CurbAxis(curb_id=1, points=pts), CurbAxis(curb_id=2, points=curb_path),
                                             CurbAxis(curb_id=3, points=np.array([[200.0, 0.0, 0.0], [210.0, 0.0, 0.0]]))]})
    counts = _apply_boundary("pavement", out, cache)
    curbs = _load(cache / "pavement.pkl")["curbs"]
    assert counts["curbs_dropped"] == 1 and counts["curbs_clipped"] == 2
    # curb 1: first stub (0.2 m) dropped, only the part x in [0,100] on y=60 remains with interpolated z
    c1 = [c for c in curbs if c.curb_id == 1]
    assert len(c1) == 1
    xy = c1[0].points
    assert xy[:, 0].min() >= -1e-9 and xy[:, 0].max() <= 100 + 1e-9
    first = xy[np.argmin(xy[:, 0])]
    assert abs(first[0]) < 1e-6 and abs(first[2] - 18.0) < 1e-6  # z at x=0 on 90->-10 ... interpolated
    # curb 2 leaves and re-enters: two inside parts
    c2 = [c for c in curbs if c.curb_id != 1]
    assert len(c2) == 2
    assert len({c.curb_id for c in c2}) == 2
    for c in c2:
        assert np.all(c.points[:, 0] <= 100 + 1e-9)
    z_at_boundary = [c.points[np.isclose(c.points[:, 0], 100.0)][0, 2] for c in c2 if np.isclose(c.points[:, 0], 100.0).any()]
    assert z_at_boundary and abs(z_at_boundary[0] - (5.5 + (11.0 - 5.5) * 50 / 55)) < 1e-6
    assert all(len(c.points) >= 2 for c in curbs)


def test_tree_outside_dropped_inside_kept(tmp_path):
    out, cache = _setup(tmp_path)
    inside = TreeElement(id="t1", name="in", params=TreeParams(trunk_xy=(50, 50), base_z=0.0, height=5.0))
    outside = TreeElement(id="t2", name="out", params=TreeParams(trunk_xy=(150, 50), base_z=0.0, height=5.0))
    _dump(cache / "vegetation.pkl", VegetationResult([inside, outside], [], []))
    counts = _apply_boundary("vegetation", out, cache)
    res = _load(cache / "vegetation.pkl")
    assert [e.id for e in res.elements] == ["t1"]
    assert counts["trees_dropped"] == 1


def test_fence_partly_outside_clipped_manhole_outside_dropped(tmp_path):
    out, cache = _setup(tmp_path)
    fence = FenceElement(id="f1", name="Zaun", params=FenceParams(axis=[(50, 10, 1.0), (150, 10, 2.0)], height=1.5))
    mh_out = ManholeElement(id="m1", params=ManholeParams(center=(150, 50, 0.0), diameter=0.6))
    mh_in = ManholeElement(id="m2", params=ManholeParams(center=(50, 50, 0.0), diameter=0.6))
    _dump(cache / "site.pkl", {"elements": [fence, mh_out, mh_in], "qa": []})
    counts = _apply_boundary("site", out, cache)
    els = {e.id: e for e in _load(cache / "site.pkl")["elements"]}
    assert set(els) == {"f1", "m2"}
    axis = np.asarray(els["f1"].params.axis)
    assert abs(axis[-1][0] - 100.0) < 1e-6 and abs(axis[-1][2] - 1.5) < 1e-6
    assert counts["fences_clipped"] == 1 and counts["manholes_dropped"] == 1
