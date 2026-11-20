"""CAD kitchen layout on the wall-fitted run: leg extension, raised bar, appliances on the counter, tall units,
upper-cabinet crosses and front/end counter lines (synthetic geometry and heights)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from samcloud.ortho.vector.kitchen_layout import KitchenLayoutSettings, _block_footprint, counter_lines, kitchen_cad_layout
from samcloud.ortho.vector.plan_polish import fit_kitchen_to_walls

UPMS = (1.0, 0.8124)


def _s(geometry, upm: float):
    return affinity.scale(geometry, upm, upm, origin=(0.0, 0.0))


def _pts(polygon: Polygon) -> list[list[float]]:
    return [list(point) for point in list(polygon.exterior.coords)[:-1]]


def _grid_points(polygon: Polygon, step: float) -> np.ndarray:
    x0, y0, x1, y1 = polygon.bounds
    gx, gy = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step))
    points = np.column_stack((gx.ravel(), gy.ravel()))
    return points[shapely.contains_xy(polygon, points[:, 0], points[:, 1])]


def _heights(parts: list[tuple[Polygon, float]], upm: float):
    def height_of(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float).reshape(-1, 2)
        output = np.zeros(len(values))
        for shape, height in parts:
            output[shapely.contains_xy(shape, values[:, 0], values[:, 1])] = height * upm
        return output

    return height_of


def _scene(upm: float, north_x1: float = 2.9):
    """6 x 5 m room (walls 0.25), wall stub x 3.6-3.85 below the north wall; U kitchen: west leg, north leg with a
    chamfered inner corner, peninsula from the west leg into the room."""
    walls = _s(unary_union([box(0.0, 0.0, 6.0, 5.0).difference(box(0.25, 0.25, 5.75, 4.75)), box(3.6, 4.0, 3.85, 4.75)]), upm)
    room = _s(box(0.25, 0.25, 5.75, 4.75), upm).difference(walls)
    north = _s(box(0.85, 4.15, north_x1, 4.75), upm)
    west = _s(box(0.25, 2.2, 0.85, 4.75), upm)
    peninsula = _s(box(0.85, 2.2, 3.0, 2.8), upm)
    chamfer = _s(Polygon([(0.85, 4.15), (0.85, 3.75), (1.25, 4.15)]), upm)
    run = unary_union([north, west, peninsula, chamfer, _s(box(0.85, 4.15, 0.95, 4.75), upm)])
    layers = {
        "A-EINB": [
            {"type": "polyline", "points": _pts(run), "closed": True, "holes": [], "_einb_kind": "fine_kitchen", "_run_legs": [_pts(leg) for leg in (north, west, peninsula)]},
            {"type": "text", "text": "Küche", "point": [0.5 * upm, 3.0 * upm], "height": 0.1 * upm, "rotation": 0.0},
        ],
        "A-OBEN": [{"type": "polyline", "points": _pts(_s(box(1.5, 4.4, 2.5, 4.75), upm)), "closed": True, "_upper_cabinet": True}],
        "A-FURN": [{"type": "polyline", "points": _pts(_s(box(1.5, 1.9, 2.2, 2.45), upm)), "closed": True, "holes": [], "_label": "Möbel", "_furn_id": 7}],
        "A-ANNO": [{"type": "text", "text": "Möbel h=1.21", "point": [1.85 * upm, 2.1 * upm], "height": 0.08 * upm, "rotation": 0.0, "_furn_id": 7}],
        "A-FURN-DETL": [],
    }
    return walls, room, layers


def _counter(layers) -> Polygon:
    return next(Polygon(entity["points"]) for entity in layers["A-EINB"] if entity.get("_run_legs") and entity.get("closed"))


@pytest.mark.parametrize("upm", UPMS)
@pytest.mark.parametrize("on_chamfer", [False, True])
def test_cad_kitchen_extends_legs_finds_bar_and_mounts_appliances(upm: float, on_chamfer: bool) -> None:
    walls, room, layers = _scene(upm)
    fit_kitchen_to_walls(layers, [room], walls, 0.0, upm)
    assert layers["A-EINB"][0]["_chamfers"], "the diagonal inner-corner unit survives the wall fit"
    heights = _heights([
        (_s(box(0.25, 2.2, 0.85, 4.75), upm), 0.90), (_s(box(0.85, 4.15, 3.6, 4.75), upm), 0.90),  # north counter runs on to the stub
        (_s(box(0.85, 2.2, 3.0, 2.8), upm), 0.90), (_s(box(0.85, 1.85, 3.0, 2.2), upm), 1.10),  # peninsula + raised bar
    ], upm)
    appliances = {
        "COOKTOP": _grid_points(_s(box(2.0, 4.25, 2.5, 4.6), upm), 0.01 * upm),
        "KITCHEN_SINK": _grid_points(_s(box(0.9, 3.85, 1.2, 4.1), upm), 0.01 * upm),
    }
    stats = kitchen_cad_layout(layers, [room], walls, 0.0, upm, heights, appliances, KitchenLayoutSettings(appliance_on_chamfer=on_chamfer))
    counter = _counter(layers)
    # north leg grown along the measured counter to the stub face (x 3.6), never into it
    assert stats["kitchen_legs_extended_m"] == pytest.approx(0.70, abs=0.06)
    assert counter.covers(_s(box(2.95, 4.2, 3.58, 4.7), upm)) and counter.intersection(walls).area <= 1e-6 * upm * upm
    # raised bar on the peninsula: part of the counter, the step drawn as an inner line
    assert stats["kitchen_bars"] >= 1 and stats["kitchen_bar_depth_m"] == pytest.approx(0.35, abs=0.05)
    assert counter.covers(_s(box(0.9, 1.9, 2.95, 2.18), upm))
    steps = [LineString(item["points"]) for item in layers["A-FURN-DETL"] if item.get("_kitchen_detail") == "bar_step"]
    assert steps and all(abs(line.coords[0][1] - 2.2 * upm) <= 0.01 * upm for line in steps)
    # the box standing on the bar belongs to the counter now
    assert not layers["A-FURN"] and not [item for item in layers["A-ANNO"] if item.get("_furn_id") == 7]
    inserts = {item["_catalogue"]["kind"]: item for item in layers["A-EINB"] if item.get("type") == "insert"}
    cooktop = _block_footprint(inserts["COOKTOP"], upm)
    assert counter.buffer(1e-4 * upm).covers(cooktop)
    assert cooktop.centroid.x == pytest.approx(2.25 * upm, abs=0.06 * upm)
    assert cooktop.centroid.y == pytest.approx(4.45 * upm, abs=0.02 * upm)  # centred in the 0.60 m counter depth
    sink = _block_footprint(inserts["KITCHEN_SINK"], upm)
    assert counter.buffer(1e-4 * upm).covers(sink)
    if on_chamfer:
        assert abs(((inserts["KITCHEN_SINK"]["rotation"] % 90.0) - 45.0)) <= 2.0  # opt-in: on the diagonal corner unit
    else:
        # default (owner feedback round 5): the sink is square to its run, lies inside a leg and never sticks out of the counter
        rotation = inserts["KITCHEN_SINK"]["rotation"] % 90.0
        assert min(rotation, 90.0 - rotation) <= 0.5
        assert counter.buffer(1e-4 * upm).covers(sink)
        assert sink.minimum_rotated_rectangle.area == pytest.approx(sink.area, rel=1e-6)
    assert stats["kitchen_appliances_mounted"] == ["KITCHEN_SINK", "COOKTOP"]
    assert stats["kitchen_upper_crosses"] == 1
    assert len([item for item in layers["A-OBEN"] if item.get("_upper_cross")]) == 2


@pytest.mark.parametrize("upm", UPMS)
def test_counter_drawn_by_front_and_end_lines(upm: float) -> None:
    walls, room, layers = _scene(upm)
    fit_kitchen_to_walls(layers, [room], walls, 0.0, upm)
    kitchen_cad_layout(layers, [room], walls, 0.0, upm, None, {})
    outline = _counter(layers)
    assert counter_lines(layers, walls, upm) == 1
    fronts = [item for item in layers["A-EINB"] if item.get("_counter_front")]
    assert fronts and all(not item["closed"] for item in fronts)
    drawn = unary_union([LineString(item["points"]) for item in fronts])
    assert drawn.intersection(walls.buffer(0.01 * upm)).length <= 0.03 * upm  # nothing doubles a wall face
    assert drawn.length == pytest.approx(outline.exterior.difference(walls.buffer(0.02 * upm)).length, rel=0.02)
    assert drawn.length < 0.8 * outline.exterior.length
    assert Polygon(fronts[0]["_counter_outline"]).equals_exact(outline, 1e-9)


@pytest.mark.parametrize("upm", UPMS)
def test_tall_unit_at_leg_end_is_full_depth_with_cross(upm: float) -> None:
    walls, room, layers = _scene(upm)
    fit_kitchen_to_walls(layers, [room], walls, 0.0, upm)
    heights = _heights([
        (_s(box(0.25, 2.2, 0.85, 4.75), upm), 0.90), (_s(box(0.85, 4.15, 2.9, 4.75), upm), 0.90),
        (_s(box(2.9, 4.15, 3.5, 4.75), upm), 2.10),  # fridge column after the north counter
        (_s(box(0.85, 2.2, 3.0, 2.8), upm), 0.90),
    ], upm)
    fridge = _grid_points(_s(box(2.95, 4.2, 3.45, 4.7), upm), 0.02 * upm)
    stats = kitchen_cad_layout(layers, [room], walls, 0.0, upm, heights, {"REFRIGERATOR": fridge})
    assert stats["kitchen_tall_units"] == 1 and stats["kitchen_legs_extended_m"] == pytest.approx(0.0, abs=1e-9)
    unit = next(Polygon(item["points"]) for item in layers["A-EINB"] if item.get("_einb_kind") == "tall_unit")
    assert unit.bounds == pytest.approx(tuple(v * upm for v in (2.9, 4.15, 3.5, 4.75)), abs=0.06 * upm)
    assert len([item for item in layers["A-EINB"] if item.get("_tall_unit_cross")]) == 2
    assert any(item.get("type") == "text" and item["text"] == "KS" for item in layers["A-EINB"])


@pytest.mark.parametrize("upm", UPMS)
def test_cooktop_hanging_off_a_moved_leg_is_remounted_inside(upm: float) -> None:
    walls, room, layers = _scene(upm)
    # the cooktop was mounted on the leg before the wall fit moved the leg back (half outside afterwards)
    layers["A-EINB"].append({
        "type": "insert", "name": "COOKTOP_580X510", "point": [2.0 * upm, 4.15 * upm], "rotation": 180.0, "xscale": upm, "yscale": upm,
        "attributes": {"LABEL": "Herd"}, "_catalogue": {"kind": "COOKTOP", "width_m": 0.58, "depth_m": 0.51, "layer": "A-EINB", "wall_snapped": True, "match_error": 0.0, "on_kitchen_run": True},
    })
    fit_kitchen_to_walls(layers, [room], walls, 0.0, upm)
    kitchen_cad_layout(layers, [room], walls, 0.0, upm, None, {})
    counter = _counter(layers)
    cooktop = next(item for item in layers["A-EINB"] if item.get("type") == "insert")
    footprint = _block_footprint(cooktop, upm)
    assert counter.buffer(1e-4 * upm).covers(footprint)
    assert footprint.centroid.x == pytest.approx(2.0 * upm, abs=0.05 * upm)
    assert math.isclose(cooktop["rotation"] % 360.0, 180.0, abs_tol=0.5)
