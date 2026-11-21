"""Contour height labels (S-TOPO-TEXT) and no spot heights (S-ELEV) in the terrain DXF."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point

from samcloud.drone import contracts as c
from samcloud.outdoor_model import terrain as tr

E0, N0, H0 = 2_600_000.0, 1_200_000.0, 430.0
FRAME = c.LocalFrame(origin_lv95=(E0, N0, H0))
RES = 0.25
BOUNDS = (0.0, 0.0, 60.0, 40.0)


def _dtm() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((160, 240))
    x, y = (cols + 0.5) * RES, 40.0 - (rows + 0.5) * RES
    z = 0.05 * x + 0.02 * y + 0.003 * np.sin(x * 1.3) + 1.0       # 5 % slope: major contours (1 m) every ~20 m
    return z, np.ones(z.shape, dtype=np.uint8)


def _entities(**dxf):
    dtm, validity = _dtm()
    cfg = {"dxf": {"contour_major_m": 1.0, "contour_minor_m": 0.5, "spline_contours": True, **dxf}}
    tin = c.LabeledTin(np.array([[0, 0, 1.0], [60, 0, 4.0], [60, 40, 4.8], [0, 40, 1.8]]),
                       np.array([[0, 1, 2], [0, 2, 3]]), np.zeros(2, np.uint8))
    entities, info = tr.terrain_dxf_entities([], FRAME, tin=tin, dtm=dtm, validity=validity, dtm_bounds=BOUNDS,
                                             resolution=RES, manholes=[(10.0, 10.0)], config=cfg)
    return entities, info


def test_no_spot_elevations_by_default() -> None:
    entities, _ = _entities()
    assert "S-ELEV" not in entities
    entities, _ = _entities(spot_heights=True, spot_spacing_m=10.0)
    assert entities.get("S-ELEV")


def test_every_long_major_contour_has_labels_about_ten_metres_apart() -> None:
    entities, info = _entities()
    labels = entities["S-TOPO-TEXT"]
    assert info["contour_labels"] == len(labels) > 0
    majors = entities["S-TOPO-MAJR"]
    for line in majors:
        shape = LineString(np.asarray(line["fit_points"])[:, :2])
        mine = [t for t in labels if abs(t["elevation"] - line["elevation"]) < 1e-9
                and shape.distance(Point(t["point"][:2])) < 0.3]
        if shape.length > 10.0:
            assert len(mine) >= 1, (shape.length, line["elevation"])
            assert len(mine) <= int(round(shape.length / 10.0)) + 1
            arcs = sorted(shape.project(Point(t["point"][:2])) for t in mine)
            if len(arcs) >= 2:
                assert 5.0 < float(np.median(np.diff(arcs))) < 15.0


def test_label_text_is_height_with_one_decimal_and_readable() -> None:
    entities, _ = _entities()
    for label in entities["S-TOPO-TEXT"]:
        assert label["type"] == "text"
        assert label["text"] == f"{label['elevation']:.1f}" and label["text"].endswith(".0")
        assert -90.0 < label["rotation"] <= 90.0           # never upside down
        assert abs(label["height"] - 0.25) < 1e-9


def test_contour_labels_follow_the_line_direction() -> None:
    diagonal = [{"layer_role": "major", "elevation": 432.0, "points": [[0.0, 0.0, 2.0], [30.0, 30.0, 2.0]]},
                {"layer_role": "major", "elevation": 433.0, "points": [[30.0, 0.0, 3.0], [0.0, 30.0, 3.0]]},
                {"layer_role": "major", "elevation": 434.0, "points": [[0.0, 0.0, 4.0], [4.0, 0.0, 4.0]]}]
    labels = tr.contour_labels(diagonal, spacing_m=10.0, text_height_m=0.25)
    by = {l["elevation"]: [x for x in labels if x["elevation"] == l["elevation"]] for l in labels}
    assert {round(x["rotation"], 3) for x in by[432.0]} == {45.0}
    assert {round(x["rotation"], 3) for x in by[433.0]} == {-45.0}     # 135 deg flipped to stay readable
    assert 3 <= len(by[432.0]) <= 5
    assert 434.0 not in by or len(by[434.0]) <= 1                      # a 4 m line: at most one label
    short = tr.contour_labels([{"layer_role": "major", "elevation": 5.0, "points": [[0, 0, 0], [8, 0, 0]]}])
    assert len(short) == 1 and abs(short[0]["point"][0] - 4.0) < 1e-9   # shorter than the spacing: one in the middle


def test_contours_and_labels_are_cut_at_building_footprints() -> None:
    from shapely.geometry import Point, box
    from shapely import affinity

    dtm, validity = _dtm()
    cfg = {"dxf": {"contour_major_m": 1.0, "contour_minor_m": 0.5, "spline_contours": True, "contour_building_gap_m": 0.10}}
    tin = c.LabeledTin(np.array([[0, 0, 1.0], [60, 0, 4.0], [60, 40, 4.8], [0, 40, 1.8]]), np.array([[0, 1, 2], [0, 2, 3]]), np.zeros(2, np.uint8))
    kw = dict(tin=tin, dtm=dtm, validity=validity, dtm_bounds=BOUNDS, resolution=RES, config=cfg)
    plain, info0 = tr.terrain_dxf_entities([], FRAME, **kw)
    house = box(20, 10, 40, 30)                                   # a 20 x 20 m house on the slope
    cut, info = tr.terrain_dxf_entities([], FRAME, footprints=[house], **kw)
    forbidden = affinity.translate(house.buffer(0.10), E0, N0)
    for layer in ("S-TOPO-MAJR", "S-TOPO-MINR"):
        assert len(cut[layer]) >= len(plain[layer])
        for entity in cut[layer]:
            xy = np.asarray(entity["fit_points"])[:, :2]
            assert not any(forbidden.buffer(-1e-6).contains(Point(p)) for p in xy)
            assert float(np.hypot(*np.diff(xy, axis=0).T).sum()) >= 0.5
    for label in cut["S-TOPO-TEXT"]:
        assert not forbidden.contains(Point(label["point"][:2]))
    assert info["contour_pieces"]["after"] is not None and info["contour_pieces"]["before"] == info0["contour_count"]
    assert len(cut["S-TOPO-TEXT"]) < len(plain["S-TOPO-TEXT"]) + 5
