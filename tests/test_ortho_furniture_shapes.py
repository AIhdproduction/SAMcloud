"""Furniture shape recognition (rectangle / L / U / round / rotated) and sofa detail from heights (synthetic)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from samcloud.ortho.vector.furniture_shapes import FurnitureShapeSettings, class_penalty, fit_furniture_shape, sofa_details
from samcloud.ortho.vector.plan_polish import regularize_furniture

UPMS = (1.0, 0.8124)


def _s(geometry, upm: float):
    return affinity.scale(geometry, upm, upm, origin=(0.0, 0.0))


def _ragged(polygon: Polygon, amplitude: float, seed: int, step: float) -> Polygon:
    """Traced outline: a point every ``step`` along the ring, each off the true edge by up to ``amplitude``."""
    rng = np.random.default_rng(seed)
    ring = polygon.exterior
    count = max(16, int(ring.length / step))
    points = []
    for k in range(count):
        point = ring.interpolate((k + 0.31) / count, normalized=True)
        points.append((point.x + rng.uniform(-amplitude, amplitude), point.y + rng.uniform(-amplitude, amplitude)))
    cleaned = Polygon(points).buffer(0)
    return max(getattr(cleaned, "geoms", [cleaned]), key=lambda part: part.area)


def _rectilinear(polygon: Polygon, theta: float = 0.0, tolerance: float = 0.5) -> bool:
    coords = list(polygon.exterior.coords)
    return all(abs(((math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) - theta + 45.0) % 90.0) - 45.0) <= tolerance for a, b in zip(coords[:-1], coords[1:]) if math.dist(a, b) > 1e-9)


# A corner sofa: 2.9 m long section along the west, a 0.8 m deep arm to the east at the north end.
CORNER_SOFA = Polygon([(0.0, 0.0), (1.2, 0.0), (1.2, 1.75), (2.0, 1.75), (2.0, 2.9), (0.0, 2.9)])
U_SOFA = Polygon([(0.0, 0.0), (0.9, 0.0), (0.9, 1.4), (2.1, 1.4), (2.1, 0.0), (3.0, 0.0), (3.0, 2.3), (0.0, 2.3)])


@pytest.mark.parametrize("upm", UPMS)
def test_ragged_rectangle_stays_a_rectangle(upm: float) -> None:
    true = _s(box(1.0, 1.0, 2.8, 1.9), upm)
    true = affinity.rotate(true, 0.2, origin=(0.0, 0.0))
    choice = fit_furniture_shape(_ragged(true, 0.03 * upm, 1, 0.04 * upm), math.radians(0.2), upm, "Sideboard")
    assert choice is not None and choice.kind == "rect"
    assert len(choice.polygon.exterior.coords) == 5 and _rectilinear(choice.polygon, 0.2, 0.1)
    assert choice.polygon.symmetric_difference(true).area <= 0.10 * true.area
    assert choice.iou >= 0.90 and choice.scores["rect"][0] == pytest.approx(choice.iou, abs=1e-3)


@pytest.mark.parametrize("upm", UPMS)
def test_corner_sofa_is_an_l_shape(upm: float) -> None:
    true = _s(affinity.translate(CORNER_SOFA, 3.0, 2.0), upm)
    choice = fit_furniture_shape(_ragged(true, 0.025 * upm, 2, 0.03 * upm), 0.0, upm, "Sofa")
    assert choice is not None and choice.kind == "L"
    assert len(choice.polygon.exterior.coords) - 1 == 6 and _rectilinear(choice.polygon)
    assert choice.polygon.symmetric_difference(true).area <= 0.08 * true.area
    # the rectangle was a real candidate and lost on the measured fit
    assert choice.scores["rect"][0] < choice.iou - 0.10
    assert choice.boundary_m <= 0.03


@pytest.mark.parametrize("upm", UPMS)
def test_u_shaped_sofa_is_a_u_shape(upm: float) -> None:
    true = _s(affinity.translate(U_SOFA, -1.0, 4.0), upm)
    choice = fit_furniture_shape(_ragged(true, 0.02 * upm, 3, 0.03 * upm), 0.0, upm, "Sofa")
    assert choice is not None and choice.kind == "U"
    assert len(choice.polygon.exterior.coords) - 1 == 8
    assert choice.polygon.symmetric_difference(true).area <= 0.08 * true.area


@pytest.mark.parametrize("upm", UPMS)
def test_class_prior_keeps_a_bed_rectangular_where_a_sofa_turns_l(upm: float) -> None:
    # 1.8 x 2.1 footprint with a 0.75 x 0.7 corner notch: a sofa with a chaise, but a bed stays a rectangle.
    outline = _s(Polygon([(0, 0), (1.05, 0), (1.05, 0.7), (1.8, 0.7), (1.8, 2.1), (0, 2.1)]), upm)
    evidence = _ragged(outline, 0.02 * upm, 4, 0.03 * upm)
    assert class_penalty("Bett") > class_penalty("Sofa")
    bed = fit_furniture_shape(evidence, 0.0, upm, "Bett")
    sofa = fit_furniture_shape(evidence, 0.0, upm, "Sofa")
    assert bed is not None and bed.kind == "rect"
    assert sofa is not None and sofa.kind == "L"


@pytest.mark.parametrize("upm", UPMS)
def test_trace_noise_notch_is_not_a_shape(upm: float) -> None:
    # a 0.12 m bite out of a corner (trace noise / an occluded corner) is below min_notch_m
    outline = _s(box(0.0, 0.0, 1.6, 0.9).difference(box(1.48, 0.78, 1.6, 0.9)), upm)
    choice = fit_furniture_shape(outline, 0.0, upm, "Sofa")
    assert choice is not None and choice.kind == "rect" and choice.iou >= 0.98


@pytest.mark.parametrize("upm", UPMS)
def test_round_table_is_a_circle_only_for_round_classes(upm: float) -> None:
    disc = _s(Point(5.0, 5.0).buffer(0.55, quad_segs=32), upm)
    evidence = _ragged(disc, 0.015 * upm, 5, 0.03 * upm)
    table = fit_furniture_shape(evidence, 0.0, upm, "Tisch")
    assert table is not None and table.kind == "circle"
    centre = table.polygon.centroid
    assert math.dist((centre.x, centre.y), (5.0 * upm, 5.0 * upm)) <= 0.03 * upm
    assert math.sqrt(table.polygon.area / math.pi) == pytest.approx(0.55 * upm, rel=0.05)
    shelf = fit_furniture_shape(evidence, 0.0, upm, "Regal")
    assert shelf is not None and shelf.kind == "rect"


@pytest.mark.parametrize("upm", UPMS)
def test_turned_rectangle_is_a_rotated_rectangle(upm: float) -> None:
    true = affinity.rotate(_s(box(2.0, 2.0, 3.4, 2.7), upm), 38.0, origin="centroid")
    choice = fit_furniture_shape(_ragged(true, 0.01 * upm, 6, 0.04 * upm), 0.0, upm, "Möbel")
    assert choice is not None and choice.kind == "rect_free"
    assert choice.polygon.symmetric_difference(true).area <= 0.12 * true.area


def _sofa_heights(outline: Polygon, backs: list[Polygon], arms: list[Polygon], upm: float):
    """Seat 0.42 m, backrests 0.80 m, armrests 0.62 m, floor 0 outside (all scaled)."""

    def height_of(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float).reshape(-1, 2)
        output = np.zeros(len(values))
        inside = shapely.contains_xy(outline, values[:, 0], values[:, 1])
        output[inside] = 0.42 * upm
        for shape, height in [(item, 0.62) for item in arms] + [(item, 0.80) for item in backs]:
            output[shapely.contains_xy(shape, values[:, 0], values[:, 1])] = height * upm
        return output

    return height_of


@pytest.mark.parametrize("upm", UPMS)
def test_corner_sofa_backrest_armrest_and_corner_module(upm: float) -> None:
    outline = _s(CORNER_SOFA, upm)
    backs = [_s(box(0.0, 0.0, 0.25, 2.9), upm), _s(box(0.0, 2.65, 2.0, 2.9), upm)]  # west + north backrest
    arms = [_s(box(0.25, 0.0, 1.2, 0.2), upm)]  # south armrest; the east end of the arm is open
    detail = sofa_details(outline, 0.0, upm, _sofa_heights(outline, backs, arms, upm))
    assert detail.raised_edges == 3
    lines = unary_union(detail.lines)
    # backrest fronts at x = 0.25 and y = 2.65, armrest at y = 0.20
    assert lines.intersection(_s(box(0.21, 0.4, 0.29, 2.5), upm)).length >= 1.8 * upm
    assert lines.intersection(_s(box(0.4, 2.61, 1.9, 2.69), upm)).length >= 1.2 * upm
    assert lines.intersection(_s(box(0.4, 0.16, 1.1, 0.24), upm)).length >= 0.6 * upm
    # corner module: the inner corner (1.2, 1.75) continued across the seat (to the backrests)
    assert lines.intersection(_s(box(0.3, 1.73, 1.15, 1.77), upm)).length >= 0.7 * upm
    assert lines.intersection(_s(box(1.18, 1.8, 1.22, 2.6), upm)).length >= 0.7 * upm
    # no line runs along the open east end or outside the outline
    assert lines.intersection(_s(box(1.95, 1.8, 2.05, 2.6), upm)).length <= 0.02 * upm
    assert outline.buffer(0.01 * upm).covers(lines)
    assert detail.seat is not None and detail.seat.within(outline) and detail.seat.area >= 0.8 * upm * upm


@pytest.mark.parametrize("upm", UPMS)
def test_sofa_without_raised_edges_gets_no_detail(upm: float) -> None:
    outline = _s(box(0.0, 0.0, 2.0, 0.9), upm)
    flat = _sofa_heights(outline, [], [], upm)
    assert sofa_details(outline, 0.0, upm, flat).lines == []


@pytest.mark.parametrize("upm", UPMS)
def test_regularize_furniture_draws_corner_sofa_with_backrests(upm: float) -> None:
    walls = _s(box(-1.0, -1.0, 6.0, 5.0).difference(box(-0.8, -0.8, 5.8, 4.8)), upm)
    room = _s(box(-0.8, -0.8, 5.8, 4.8), upm)
    true = _s(affinity.translate(CORNER_SOFA, 1.0, 0.5), upm)
    evidence = _ragged(true, 0.02 * upm, 7, 0.03 * upm).difference(_s(Point(2.0, 1.5).buffer(0.3), upm))  # a person on it
    traced = true.minimum_rotated_rectangle  # what the footprint fit made of it
    entity = {"type": "polyline", "points": [list(p) for p in list(traced.exterior.coords)[:-1]], "closed": True, "holes": [], "_label": "Sofa", "_semantic_label": "sofa", "_furn_id": 1, "_footprint_kind": "rect"}
    label = {"type": "text", "text": "Sofa h=0.80", "point": [2.0 * upm, 2.0 * upm], "height": 0.12 * upm, "rotation": 0.0, "_furn_id": 1}
    old_detail = {"type": "polyline", "points": [[1.3 * upm, 1.0 * upm], [1.9 * upm, 1.0 * upm]], "closed": False}
    layers = {"A-FURN": [entity], "A-ANNO": [label], "A-FURN-DETL": [old_detail]}
    backs = [_s(box(1.0, 0.5, 1.25, 3.4), upm), _s(box(1.0, 3.15, 3.0, 3.4), upm)]
    heights = _sofa_heights(true, backs, [], upm)
    stats = regularize_furniture(layers, [room], walls, 0.0, upm, evidence_of=lambda item: evidence, height_of=heights)
    outline = Polygon(layers["A-FURN"][0]["points"])
    assert layers["A-FURN"][0]["_footprint_kind"] == "L"
    assert len(outline.exterior.coords) - 1 == 6 and outline.symmetric_difference(true).area <= 0.08 * true.area
    assert stats["furniture_sofa_details"] == 1
    shape_lines = [item for item in layers["A-FURN-DETL"] if item.get("_shape_detail")]
    assert shape_lines and old_detail not in layers["A-FURN-DETL"]
    assert [row["kind"] for row in stats["furniture_shapes"]] == ["L"]
    from samcloud.ortho.vector.plan_polish import place_object_labels

    place_object_labels(layers, [room], upm, 0.0, lambda item: None)
    text = next(item for item in layers["A-ANNO"] if item.get("_furn_id") == 1)
    seat = Polygon(layers["A-FURN"][0]["_label_area"])
    assert seat.buffer(0.01 * upm).contains(Point(text["point"]))
    assert all(item.get("_shape_detail") for item in layers["A-FURN-DETL"])  # backrest lines survive the label


@pytest.mark.parametrize("upm", UPMS)
def test_sliver_glued_to_a_large_outline_is_dropped(upm: float) -> None:
    room = _s(box(0.0, 0.0, 6.0, 5.0), upm)
    big = _s(box(1.0, 1.0, 3.0, 2.0), upm)
    sliver = _s(box(1.2, 0.9, 2.0, 1.0), upm)  # 0.10 m deep strip under it
    furn = [
        {"type": "polyline", "points": [list(p) for p in list(shape.exterior.coords)[:-1]], "closed": True, "holes": [], "_label": name, "_furn_id": index, "_footprint_kind": "rect"}
        for index, (shape, name) in enumerate(((big, "Tisch"), (sliver, "Regal")), start=1)
    ]
    layers = {"A-FURN": furn, "A-ANNO": [], "A-FURN-DETL": []}
    regularize_furniture(layers, [room], None, 0.0, upm)
    assert [item["_furn_id"] for item in layers["A-FURN"]] == [1]
