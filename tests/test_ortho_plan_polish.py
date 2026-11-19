"""Plan polish: rooms from wall cells, regular furniture, wall-hugging kitchen run and clean labels (synthetic)."""

from __future__ import annotations

import math
import os
from pathlib import Path
import uuid

import ezdxf
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
from samcloud.dxf.writer import export_plan
from samcloud.ortho.vector.labels import LabelSettings, resolve_label_overlaps
from samcloud.ortho.vector.plan_polish import (
    PlanPolishSettings,
    bridge_barriers,
    close_wall_cracks,
    collapse_short_edges,
    enclosed_cells,
    fit_kitchen_to_walls,
    place_object_labels,
    plausible_label,
    rectilinear_outline,
    regular_furniture_shape,
    regularize_furniture,
    rooms_from_wall_cells,
)
from samcloud.ortho.vector.raster_grid import RasterGrid
from samcloud.ortho.vector.room_names import RoomNameSettings, classify_room, name_halls, name_rooms

UPMS = (1.0, 0.8124)
CHAR = PlanPolishSettings().label_char_width


def _s(polygon: Polygon, upm: float) -> Polygon:
    return affinity.scale(polygon, upm, upm, origin=(0.0, 0.0))


def _ring_walls(outer: tuple[float, float, float, float], thickness: float) -> Polygon:
    x0, y0, x1, y1 = outer
    return box(x0, y0, x1, y1).difference(box(x0 + thickness, y0 + thickness, x1 - thickness, y1 - thickness))


def _rectilinear(polygon: Polygon, theta: float = 0.0, tolerance_deg: float = 0.5) -> bool:
    coords = list(polygon.exterior.coords)
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0)) - math.degrees(theta)
        if abs(((angle + 45.0) % 90.0) - 45.0) > tolerance_deg:
            return False
    return True


def _noisy_ring(polygon: Polygon, amplitude: float, seed: int, step: float) -> Polygon:
    """Ragged trace of an outline: points every ``step`` with a deterministic offset off the true edge."""
    rng = np.random.default_rng(seed)
    ring = polygon.exterior
    count = max(12, int(ring.length / step))
    points = []
    for k in range(count):
        point = ring.interpolate((k + 0.37) / count, normalized=True)
        points.append((point.x + rng.uniform(-amplitude, amplitude), point.y + rng.uniform(-amplitude, amplitude)))
    cleaned = Polygon(points).buffer(0)
    return max(getattr(cleaned, "geoms", [cleaned]), key=lambda part: part.area)


# ---------------------------------------------------------------- rooms

def _flat(upm: float) -> tuple[list[Polygon], list[Polygon]]:
    """A 10 x 6 m flat: open-plan living/kitchen (left), hall top right, closed room bottom right.

    Interior walls: a wall from x 6.0-6.2 for y 0.2-3.0 (closed room), a wall y 3.0-3.2 for x 6.0-9.8 with a
    door cut between x 7.0-7.9 (barrier), and a 0.6 m stub at x 4.0-4.2, y 4.4-5.8 below the north wall; the
    passage between the stub end (y 4.4) and the room corner (6.0, 3.2) separates the hall from the living area.
    """
    walls = [
        _ring_walls((0.0, 0.0, 10.0, 6.0), 0.2),
        box(6.0, 0.2, 6.2, 3.2),
        box(6.0, 3.0, 7.0, 3.2),
        box(7.9, 3.0, 9.8, 3.2),
        box(4.0, 4.4, 4.2, 5.8),
    ]
    barriers = [box(7.0, 3.0, 7.9, 3.2)]
    return [_s(item, upm) for item in walls], [_s(item, upm) for item in barriers]


@pytest.mark.parametrize("upm", UPMS)
def test_open_plan_with_stepped_ceiling_split_is_one_rectilinear_room(upm: float) -> None:
    walls, barriers = _flat(upm)
    # Kitchen and living ceiling rooms meet along a staircase through the open-plan interior.
    kitchen = Polygon([(0.2, 3.0), (2.0, 3.0), (2.0, 3.6), (3.2, 3.6), (3.2, 4.4), (4.0, 4.4), (4.0, 5.8), (0.2, 5.8)])
    living = Polygon([(0.2, 0.2), (6.0, 0.2), (6.0, 3.2), (4.0, 3.2), (4.0, 4.4), (3.2, 4.4), (3.2, 3.6), (2.0, 3.6), (2.0, 3.0), (0.2, 3.0)])
    hall = box(4.4, 3.4, 9.8, 5.8)
    closed = box(6.3, 0.3, 9.7, 2.9)
    rooms, stats = rooms_from_wall_cells([_s(item, upm) for item in (kitchen, living, hall, closed)], walls, barriers, None, None, upm, 0.0)
    assert len(rooms) == 3
    wall_union = unary_union(walls + barriers)
    open_plan = max(rooms, key=lambda room: room.area)
    # One room for kitchen + living: no boundary through the interior, it follows the inner wall faces.
    assert open_plan.contains(_s(Point(2.5, 3.5), upm)) and open_plan.contains(_s(Point(1.0, 1.0), upm)) and open_plan.contains(_s(Point(1.0, 5.0), upm))
    assert all(_rectilinear(room) for room in rooms)
    for room in rooms:
        assert room.intersection(wall_union).area <= 1e-6 * upm * upm
    hall_room = next(room for room in rooms if room.contains(_s(Point(8.0, 5.0), upm)))
    assert not hall_room.contains(_s(Point(2.0, 2.0), upm))
    # The cut runs through the passage between the stub end and the wall corner (<= 2.4 m), not further down.
    assert hall_room.contains(_s(Point(5.0, 4.0), upm)) and not open_plan.contains(_s(Point(5.0, 4.0), upm))
    assert stats["room_cells_split"] == 1
    total = sum(room.area for room in rooms)
    assert total == pytest.approx(_s(box(0.2, 0.2, 9.8, 5.8), upm).difference(wall_union).area, rel=1e-3)


@pytest.mark.parametrize("upm", UPMS)
def test_enclosed_cell_without_ceiling_room_is_a_room(upm: float) -> None:
    walls, barriers = _flat(upm)
    living = box(0.3, 0.3, 5.5, 5.5)
    rooms, stats = rooms_from_wall_cells([_s(living, upm)], walls, barriers, None, None, upm, 0.0)
    closed = [room for room in rooms if room.contains(_s(Point(8.0, 1.5), upm))]
    assert len(closed) == 1
    assert closed[0].area == pytest.approx(3.6 * 2.8 * upm * upm, rel=1e-3)


@pytest.mark.parametrize("upm", UPMS)
def test_cell_without_floor_is_not_a_room(upm: float) -> None:
    walls, barriers = _flat(upm)
    grid = RasterGrid(-1.0 * upm, 7.0 * upm, 0.05 * upm, int(12 / 0.05), int(8 / 0.05))
    floor = np.zeros((grid.height, grid.width), dtype=bool)
    rows = slice(int((7.0 - 5.8) / 0.05), int((7.0 - 0.2) / 0.05))
    floor[rows, int(1.2 / 0.05):int(6.8 / 0.05)] = True  # floor only left of x 5.8
    rooms, _ = rooms_from_wall_cells([], walls, barriers, floor, grid, upm, 0.0)
    assert not any(room.contains(_s(Point(8.0, 1.5), upm)) for room in rooms)
    assert any(room.contains(_s(Point(2.0, 2.0), upm)) for room in rooms)


@pytest.mark.parametrize("upm", UPMS)
def test_room_outside_wall_cells_is_simplified_rectilinear(upm: float) -> None:
    # Walls do not close (no cell); the ceiling room keeps its place but loses its stair steps.
    walls = [_s(box(0.0, 0.0, 6.0, 0.2), upm)]
    stepped = Polygon([(0.2, 0.2), (5.0, 0.2), (5.0, 2.0), (4.9, 2.0), (4.9, 2.1), (4.8, 2.1), (4.8, 3.0), (0.2, 3.0)])
    rooms, stats = rooms_from_wall_cells([_s(stepped, upm)], walls, [], None, None, upm, 0.0)
    assert stats["rooms_outside_cells"] == 1
    room = rooms[0]
    assert _rectilinear(room)
    coords = list(room.exterior.coords)
    shortest = min(math.dist(a, b) for a, b in zip(coords[:-1], coords[1:]))
    assert shortest >= 0.29 * upm
    assert room.symmetric_difference(_s(stepped, upm)).area <= 0.35 * upm * upm


# ---------------------------------------------------------------- room names

def test_room_names_open_plan_office_and_halls() -> None:
    assert classify_room({"kitchen", "living"}, 30.0) == "Wohnen / Küche"
    assert classify_room({"kitchen", "dining"}, 20.0) == "Küche / Essen"
    assert classify_room({"office"}, 12.0) == "Büro"
    rooms = [box(0, 0, 4, 3), box(5, 0, 9, 3)]
    tags = [((1.0, 1.0), "office"), ((2.0, 1.0), "office"), ((6.0, 1.0), "office")]
    names = name_rooms(rooms, tags, 1.0)
    assert names[0].name == "Büro"
    assert names[1].name is None  # one desk does not make an office


def test_small_corridor_and_wardrobe_hall_are_named_flur() -> None:
    corridor = box(0.0, 0.0, 1.3, 1.1)  # 1.4 m2 between three doors
    hall = box(2.0, 0.0, 6.0, 2.5)
    rooms = [corridor, hall]
    decisions = name_rooms(rooms, [((3.0, 1.0), "wardrobe")], 1.0)
    assert not decisions[0].keep and decisions[1].name == "Zimmer"
    doors = [((0.0, 0.5), math.pi / 2.0), ((1.3, 0.5), math.pi / 2.0), ((0.6, 1.1), 0.0), ((4.0, 2.5), 0.0), ((2.0, 1.0), math.pi / 2.0)]
    named = name_halls(rooms, decisions, doors, 1.0, RoomNameSettings())
    assert named[0].name == "Flur" and named[0].keep
    assert named[1].name == "Flur"  # entrance door (outside above) plus one more door


# ---------------------------------------------------------------- furniture shapes

@pytest.mark.parametrize("upm", UPMS)
def test_ragged_rectangle_becomes_wall_aligned_rectangle(upm: float) -> None:
    theta = math.radians(0.3)
    true = affinity.rotate(_s(box(1.0, 1.0, 2.6, 1.9), upm), 0.3, origin=(0, 0))
    ragged = _noisy_ring(true, 0.03 * upm, 1, 0.04 * upm)
    shape, kind = regular_furniture_shape(ragged, theta, upm)
    assert kind == "rect"
    assert len(shape.exterior.coords) == 5
    assert _rectilinear(shape, theta, 0.1)
    assert shape.symmetric_difference(true).area <= 0.12 * true.area


@pytest.mark.parametrize("upm", UPMS)
def test_ragged_l_shape_becomes_simple_rectilinear_outline(upm: float) -> None:
    true = _s(Polygon([(0, 0), (2.0, 0), (2.0, 0.6), (0.6, 0.6), (0.6, 1.8), (0, 1.8)]), upm)
    ragged = _noisy_ring(true, 0.025 * upm, 2, 0.03 * upm)
    shape, kind = regular_furniture_shape(ragged, 0.0, upm)
    assert kind == "rectilinear"
    assert len(shape.exterior.coords) - 1 <= PlanPolishSettings().furniture_max_vertices
    assert _rectilinear(shape)
    assert shape.symmetric_difference(true).area <= 0.20 * true.area


@pytest.mark.parametrize("upm", UPMS)
def test_truly_rotated_rectangle_stays_rotated(upm: float) -> None:
    true = affinity.rotate(_s(box(3.0, 3.0, 4.2, 3.6), upm), 40.0, origin="centroid")
    shape, kind = regular_furniture_shape(_noisy_ring(true, 0.01 * upm, 3, 0.04 * upm), 0.0, upm)
    assert kind == "rect_free"
    assert shape.symmetric_difference(true).area <= 0.15 * true.area


def test_collapse_short_edges_removes_steps_and_spikes() -> None:
    # A long edge with a 5 cm stair step and a 4 cm wide spike.
    polygon = Polygon([(0, 0), (2.0, 0), (2.0, 1.0), (1.2, 1.0), (1.2, 1.05), (0.62, 1.05), (0.62, 1.4), (0.58, 1.4), (0.58, 1.05), (0, 1.05)])
    result = collapse_short_edges(polygon, 0.15)
    coords = list(result.exterior.coords)
    assert min(math.dist(a, b) for a, b in zip(coords[:-1], coords[1:])) >= 0.15
    assert _rectilinear(result)
    assert result.symmetric_difference(polygon).area <= 0.08


@pytest.mark.parametrize("upm", UPMS)
def test_rectilinear_outline_keeps_position(upm: float) -> None:
    true = _s(box(3.0, 8.0, 4.0, 9.5), upm)
    shape = rectilinear_outline(true, math.radians(0.15), 0.15 * upm)
    assert shape.symmetric_difference(true).area <= 0.05 * true.area


def test_plausible_label_renames_oversized_soft_items() -> None:
    assert plausible_label("Kissen", 4.4, 1.49) == "Möbel"
    assert plausible_label("Kissen", 0.2, 0.5) == "Kissen"
    assert plausible_label("Bett", 0.8, 1.82) == "Möbel"
    assert plausible_label("Bett", 3.2, 0.6) == "Bett"
    assert plausible_label("Tisch", 4.0, 0.76) == "Tisch"


# ---------------------------------------------------------------- regularize_furniture

def _furn(points: Polygon, label: str, furn_id: int, **extra: object) -> dict:
    return {"type": "polyline", "points": [list(point) for point in list(points.exterior.coords)[:-1]], "closed": True, "holes": [], "_label": label, "_furn_id": furn_id, "_footprint_kind": "regularised", **extra}


def _label(text: str, polygon: Polygon, furn_id: int, upm: float) -> dict:
    point = polygon.representative_point()
    return {"type": "text", "text": text, "point": [point.x, point.y], "height": 0.08 * upm, "rotation": 0.0, "_furn_id": furn_id}


@pytest.mark.parametrize("upm", UPMS)
def test_regularize_furniture_clips_walls_drops_duplicates_and_clutter(upm: float) -> None:
    walls = unary_union([_s(_ring_walls((0.0, 0.0, 6.0, 4.0), 0.2), upm)])
    room = _s(box(0.2, 0.2, 5.8, 3.8), upm)
    sideboard = _noisy_ring(_s(box(4.0, 3.3, 5.6, 3.95), upm), 0.02 * upm, 4, 0.04 * upm)  # reaches into the north wall
    table = _s(box(1.0, 1.0, 2.6, 2.0), upm)
    inner = _noisy_ring(_s(box(1.3, 1.2, 2.2, 1.8), upm), 0.03 * upm, 5, 0.03 * upm)  # ragged trace inside the table
    crumb = _s(box(3.0, 1.0, 3.15, 1.2), upm)  # 0.03 m2 clutter
    cushion = _noisy_ring(_s(box(0.4, 2.6, 2.8, 3.6), upm), 0.02 * upm, 6, 0.04 * upm)  # 2.4 m2 "Kissen"
    fragment_a = _s(box(3.4, 0.4, 4.0, 1.0), upm)
    fragment_b = _s(box(4.05, 0.4, 4.6, 1.0), upm)
    detail_ragged = {"type": "polyline", "points": [[1.4 * upm, 1.3 * upm], [1.6 * upm, 1.5 * upm], [1.5 * upm, 1.7 * upm], [1.9 * upm, 1.6 * upm]], "closed": False}
    detail_straight = {"type": "polyline", "points": [[1.1 * upm, 1.5 * upm], [2.5 * upm, 1.5 * upm]], "closed": False}
    layers = {
        "A-FURN": [
            _furn(sideboard, "Sideboard", 1), {**_furn(table, "Tisch", 2), "_footprint_kind": "rect"}, _furn(inner, "Tisch", 3),
            _furn(crumb, "Objekt", 4), _furn(cushion, "Kissen", 5), _furn(fragment_a, "Regal", 6, _instance_id=9), _furn(fragment_b, "Regal", 7, _instance_id=9),
        ],
        "A-ANNO": [
            _label("Sideboard h=0.95", sideboard, 1, upm), _label("Tisch h=0.76", table, 2, upm), _label("Tisch h=0.80", inner, 3, upm),
            _label("Objekt h=0.30", crumb, 4, upm), _label("Kissen h=1.49", cushion, 5, upm), _label("Regal h=1.20", fragment_a, 6, upm), _label("Regal h=1.20", fragment_b, 7, upm),
        ],
        "A-FURN-DETL": [detail_ragged, detail_straight],
    }
    stats = regularize_furniture(layers, [room], walls, 0.0, upm)
    outlines = [Polygon(entity["points"]) for entity in layers["A-FURN"]]
    assert len(outlines) == 4  # sideboard, table, cushion, merged shelf
    for outline in outlines:
        assert outline.intersection(walls).area <= 1e-6 * upm * upm
        assert _rectilinear(outline)
        assert len(outline.exterior.coords) - 1 <= PlanPolishSettings().furniture_max_vertices
    texts = {entity["_furn_id"]: entity["text"] for entity in layers["A-ANNO"]}
    assert set(texts) == {1, 2, 5, 6}
    assert texts[5] == "Möbel h=1.49"
    assert stats["furniture_merged"] == 1 and stats["furniture_dropped_nested"] == 1 and stats["furniture_dropped_small"] >= 1
    shelf = next(Polygon(entity["points"]) for entity in layers["A-FURN"] if entity["_furn_id"] == 6)
    assert shelf.area >= 0.95 * (fragment_a.area + fragment_b.area)
    assert layers["A-FURN-DETL"] == [detail_straight]


@pytest.mark.parametrize("upm", UPMS)
def test_regularize_furniture_cuts_overlap_with_catalogue_block(upm: float) -> None:
    room = _s(box(0.2, 0.2, 5.8, 3.8), upm)
    bed = _s(box(1.0, 1.0, 2.6, 3.0), upm)
    shelf = _s(box(2.2, 2.6, 3.4, 3.2), upm)  # reaches 0.4 m into the bed block
    insert = {"type": "insert", "name": "BED_DOUBLE", "point": [0, 0], "_footprint": bed}
    layers = {"A-FURN": [insert, _furn(shelf, "Regal", 1)], "A-ANNO": [_label("Regal h=1.99", shelf, 1, upm)], "A-FURN-DETL": []}
    regularize_furniture(layers, [room], None, 0.0, upm, footprint_of=lambda entity: entity.get("_footprint"))
    polyline = next(entity for entity in layers["A-FURN"] if entity.get("type") == "polyline")
    assert Polygon(polyline["points"]).intersection(bed).area <= 1e-6 * upm * upm


# ---------------------------------------------------------------- kitchen

@pytest.mark.parametrize("upm", UPMS)
def test_kitchen_run_hugs_final_walls_with_one_depth(upm: float) -> None:
    walls = _s(_ring_walls((0.0, 0.0, 6.0, 5.0), 0.25), upm)
    room = _s(box(0.25, 0.25, 5.75, 4.75), upm)
    # L-run measured against a cabinet front: north leg 0.30 m in front of the wall and only 0.40 deep, west leg
    # 0.75 deep reaching 0.05 m into the west wall.
    north = _s(box(0.95, 4.05, 3.5, 4.45), upm)
    west = _s(box(0.20, 1.8, 0.95, 4.45), upm)
    run = unary_union([north, west])
    layers = {
        "A-EINB": [
            {"type": "polyline", "points": [list(p) for p in list(run.exterior.coords)[:-1]], "closed": True, "holes": [], "_einb_kind": "fine_kitchen", "_run_legs": [[list(p) for p in list(leg.exterior.coords)[:-1]] for leg in (north, west)]},
            {"type": "text", "text": "Küche", "point": [0.5 * upm, 3.0 * upm], "height": 0.1 * upm, "rotation": 0.0},
            {"type": "polyline", "points": [[1.0 * upm, 4.05 * upm], [3.4 * upm, 4.05 * upm]], "closed": False},  # front line retracing the run
            {"type": "polyline", "points": [[5.62 * upm, 1.0 * upm], [5.62 * upm, 2.0 * upm]], "closed": False},  # doubles the east wall face
            {"type": "polyline", "points": [[3.0 * upm, 2.0 * upm], [3.0 * upm, 2.8 * upm]], "closed": False},  # unrelated front
        ],
        "A-OBEN": [{"type": "polyline", "points": [[1.0 * upm, 4.1 * upm], [2.0 * upm, 4.1 * upm], [2.0 * upm, 4.45 * upm], [1.0 * upm, 4.45 * upm]], "closed": True, "_upper_cabinet": True}],
        "A-FURN": [_furn(_s(box(2.0, 3.9, 2.6, 4.6), upm), "Möbel", 1)],
        "A-ANNO": [_label("Möbel h=1.10", _s(box(2.0, 3.9, 2.6, 4.6), upm), 1, upm)],
    }
    stats = fit_kitchen_to_walls(layers, [room], walls, 0.0, upm)
    counter = Polygon(layers["A-EINB"][0]["points"])
    assert counter.intersection(walls).area <= 1e-6 * upm * upm
    assert stats["kitchen_depth_m"] == pytest.approx(0.60)
    # The north leg hugs the north wall face (y 4.75) and is 0.60 deep; the west leg hugs x 0.25.
    assert counter.covers(_s(box(1.0, 4.16, 3.4, 4.74), upm))
    assert not counter.intersects(_s(box(1.0, 3.9, 3.4, 4.12), upm))
    assert counter.covers(_s(box(0.26, 2.0, 0.84, 4.7), upm))
    assert not counter.intersects(_s(box(0.88, 2.0, 0.92, 3.9), upm))
    assert all(_rectilinear(Polygon(leg)) for leg in layers["A-EINB"][0]["_run_legs"])
    segments = [entity for entity in layers["A-EINB"] if entity["type"] == "polyline" and not entity.get("closed")]
    assert len(segments) == 1 and segments[0]["points"][0][0] == pytest.approx(3.0 * upm)
    upper = Polygon(layers["A-OBEN"][0]["points"])
    assert upper.bounds[3] == pytest.approx(4.75 * upm, abs=1e-6 * upm)
    assert upper.bounds[3] - upper.bounds[1] == pytest.approx(0.35 * upm, abs=1e-6 * upm)
    # Furniture standing mostly on the counter is absorbed with its label.
    assert not layers["A-FURN"] and not layers["A-ANNO"]


@pytest.mark.parametrize("upm", UPMS)
def test_kitchen_peninsula_keeps_position_and_joins_backed_leg(upm: float) -> None:
    walls = _s(_ring_walls((0.0, 0.0, 6.0, 5.0), 0.25), upm)
    room = _s(box(0.25, 0.25, 5.75, 4.75), upm)
    west = _s(box(0.25, 2.5, 0.95, 4.75), upm)
    peninsula = _s(box(0.95, 2.2, 2.4, 2.8), upm)
    run = unary_union([west, peninsula])
    layers = {"A-EINB": [{"type": "polyline", "points": [list(p) for p in list(run.exterior.coords)[:-1]], "closed": True, "holes": [], "_einb_kind": "fine_kitchen", "_run_legs": [[list(p) for p in list(leg.exterior.coords)[:-1]] for leg in (west, peninsula)]}]}
    fit_kitchen_to_walls(layers, [room], walls, 0.0, upm)
    counter = Polygon(layers["A-EINB"][0]["points"])
    assert counter.bounds[2] == pytest.approx(2.4 * upm, abs=0.01 * upm)  # the free end does not grow
    assert counter.covers(_s(box(0.3, 2.25, 2.35, 2.75), upm))  # the peninsula reaches the wall-backed leg
    assert counter.intersection(walls).area <= 1e-6 * upm * upm


# ---------------------------------------------------------------- labels

@pytest.mark.parametrize("upm", UPMS)
def test_object_labels_inside_object_or_room_never_under_walls(upm: float) -> None:
    room = _s(box(0.2, 0.2, 5.8, 3.8), upm)
    big = _s(box(0.2, 0.5, 0.8, 3.0), upm)  # tall narrow shelf on the west wall: label rotated inside
    small = _s(box(5.45, 1.0, 5.8, 1.3), upm)  # 0.1 m2 nightstand on the east wall: label dropped
    middle = _s(box(5.2, 2.6, 5.8, 3.2), upm)  # 0.36 m2: long label does not fit inside -> beside, inside the room
    layers = {
        "A-FURN": [_furn(big, "Regal", 1), _furn(small, "Nachttisch", 2), _furn(middle, "Sideboard", 3)],
        "A-ANNO": [
            {"type": "text", "text": "Regal h=1.87", "point": [0.25 * upm, 1.0 * upm], "height": 0.12 * upm, "rotation": 0.0, "_furn_id": 1},
            {"type": "text", "text": "Nachttisch h=0.52", "point": [5.7 * upm, 1.1 * upm], "height": 0.08 * upm, "rotation": 0.0, "_furn_id": 2},
            {"type": "text", "text": "Sideboard h=0.95", "point": [5.75 * upm, 2.9 * upm], "height": 0.08 * upm, "rotation": 0.0, "_furn_id": 3},
            {"type": "text", "text": "Raum  20.2 m²", "point": [3.0 * upm, 2.0 * upm], "height": 0.12 * upm, "rotation": 0.0, "_room_name": "Raum"},
        ],
        "A-FURN-DETL": [],
    }
    stats = place_object_labels(layers, [room], upm, 0.0, lambda entity: None)
    texts = {entity.get("_furn_id"): entity for entity in layers["A-ANNO"]}
    assert 2 not in texts and stats["labels_dropped_small"] == 1
    for key in (1, 3):
        entity = texts[key]
        width = len(entity["text"]) * CHAR * entity["height"]
        raw = box(entity["point"][0] - width / 2, entity["point"][1] - entity["height"] / 2, entity["point"][0] + width / 2, entity["point"][1] + entity["height"] / 2)
        label_box = affinity.rotate(raw, entity["rotation"], origin=tuple(entity["point"]))
        assert room.covers(label_box)
        assert entity["height"] >= 0.06 * upm - 1e-9
    shelf_label = texts[1]
    assert abs(shelf_label["rotation"] - 90.0) < 1e-6  # along the long side of the narrow shelf, inside it
    assert big.contains(Point(shelf_label["point"]))
    assert not middle.contains(Point(texts[3]["point"]))


@pytest.mark.parametrize("upm", UPMS)
def test_room_and_furniture_labels_do_not_cross_room_outline(upm: float) -> None:
    room = _s(box(0.0, 0.0, 4.0, 3.0), upm)
    layers = {
        "A-ANNO": [
            {"type": "text", "text": "Zimmer  12.0 m²", "point": [3.8 * upm, 1.5 * upm], "height": 0.12 * upm, "rotation": 0.0, "_room_name": "Zimmer"},
            {"type": "text", "text": "Sideboard h=0.95", "point": [0.1 * upm, 0.5 * upm], "height": 0.08 * upm, "rotation": 0.0},
        ],
    }
    resolve_label_overlaps(layers, [], [room], upm, LabelSettings())
    for entity in layers["A-ANNO"]:
        width = len(entity["text"]) * LabelSettings().char_width_factor * entity["height"]
        cx, cy = entity["point"]
        assert room.covers(box(cx - width / 2, cy - entity["height"] / 2, cx + width / 2, cy + entity["height"] / 2))


# ---------------------------------------------------------------- writer draw order

@pytest.fixture
def work_dir(request: pytest.FixtureRequest) -> Path:
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"plan_polish_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("plan_polish")


def test_writer_draws_fills_first_and_texts_last(work_dir: Path) -> None:
    layers = {
        "A-ANNO": [{"type": "text", "text": "Kratzbaum h=1.59", "point": [1.0, 1.0], "height": 0.08, "rotation": 0.0}],
        "A-WALL": [{"type": "polyline", "points": [[0, 0], [2, 0], [2, 0.2], [0, 0.2]], "closed": True, "holes": []}],
        "A-WALL-PATT": [{"type": "hatch", "points": [[0, 0], [2, 0], [2, 0.2], [0, 0.2]], "holes": []}],
    }
    frame = Frame(np.zeros(3, dtype=np.float64), 1.0, np.eye(3, dtype=np.float64))
    result = DxfPlanResult(
        project_name="P", level_name="EG", mode="indoor", profile_name="p", scale_factor=1.0, frame=frame, settings_fingerprint="f",
        layers=layers, statuses={name: LayerStatus(name, "ok", None, 1) for name in layers}, layer_colors={name: 7 for name in layers},
        background=BackgroundImage(None, None, (0.0, 0.0, 3.0, 3.0)), fingerprint_inputs={}, layer_linetypes={}, layer_hatches={},
        layer_styles={}, blocks=[], variant="ortho",
    )
    path = export_plan(result, work_dir, "local")
    order = [(entity.dxftype(), entity.dxf.layer) for entity in ezdxf.readfile(path).modelspace() if entity.dxf.layer != "X-NOTE"]
    assert order.index(("HATCH", "A-WALL-PATT")) < order.index(("LWPOLYLINE", "A-WALL")) < order.index(("TEXT", "A-ANNO"))


@pytest.mark.parametrize("upm", UPMS)
def test_object_label_keeps_clear_of_outline_and_detail_lines(upm: float) -> None:
    """A label inside an object keeps a gap to the outline and never touches the object's own detail lines
    (sofa seat / backrest lines, height steps)."""
    room = _s(box(0.0, 0.0, 5.0, 4.0), upm)
    sofa = _s(box(1.0, 1.0, 2.2, 2.0), upm)
    seat_line = [[1.0 * upm, 1.55 * upm], [2.2 * upm, 1.55 * upm]]  # a seat line across the middle
    module_line = [[1.6 * upm, 1.0 * upm], [1.6 * upm, 1.55 * upm]]
    layers = {
        "A-FURN": [{"type": "polyline", "points": [list(point) for point in sofa.exterior.coords[:-1]], "closed": True, "_furn_id": 1}],
        "A-FURN-DETL": [{"type": "polyline", "points": seat_line, "closed": False, "_shape_detail": True},
                        {"type": "polyline", "points": module_line, "closed": False, "_shape_detail": True}],
        "A-ANNO": [{"type": "text", "text": "Sofa h=0.80", "point": [1.6 * upm, 1.5 * upm], "height": 0.12 * upm, "rotation": 0.0, "_furn_id": 1}],
    }
    place_object_labels(layers, [room], upm, 0.0, lambda entity: None)
    label = layers["A-ANNO"][0]
    width = len(label["text"]) * CHAR * label["height"]
    raw = box(label["point"][0] - width / 2, label["point"][1] - label["height"] / 2, label["point"][0] + width / 2, label["point"][1] + label["height"] / 2)
    label_box = affinity.rotate(raw, label["rotation"], origin=tuple(label["point"]))
    margin = PlanPolishSettings().label_margin_m * upm
    assert sofa.buffer(-margin + 1e-6).covers(label_box)
    for line in (seat_line, module_line):
        assert LineString(line).distance(label_box) >= margin - 1e-6
    assert label["height"] >= 0.06 * upm - 1e-9 and len(layers["A-FURN-DETL"]) == 2


def test_room_name_sanitary_fixtures_do_not_name_a_large_open_plan_room() -> None:
    # A bath next to the living room was merged into it: 47 m2 holding a kitchen and sanitary fixtures is no "Bad".
    assert classify_room({"toilet", "washbasin", "shower", "kitchen", "living"}, 47.0) == "Wohnen / Küche"
    assert classify_room({"toilet", "kitchen"}, 30.0) == "Küche"
    assert classify_room({"toilet", "washbasin", "shower"}, 47.0) is None  # nothing but sanitary fixtures in a hall-sized room: no evidence
    assert classify_room({"toilet", "washbasin", "shower", "kitchen", "living"}, 5.0) == "Bad"  # a bath-sized room stays a bath
    assert classify_room({"toilet", "washbasin"}, 3.0) == "Gäste-WC"
    assert classify_room({"toilet", "washbasin"}, 14.9) == "Bad"
    assert classify_room({"toilet", "kitchen", "living"}, 20.0, RoomNameSettings(bath_max_area_m2=25.0)) == "Bad"
    rooms = [box(0, 0, 8, 6)]  # 48 m2 with a WC and a kitchen block inside
    names = name_rooms(rooms, [((1.0, 1.0), "toilet"), ((2.0, 1.0), "washbasin"), ((6.0, 4.0), "kitchen"), ((4.0, 3.0), "living")], 1.0)
    assert names[0].name == "Wohnen / Küche" and names[0].keep


def _bath_walls(wall: float, barriers_gap: tuple[float, float]) -> tuple[list[Polygon], list[Polygon]]:
    """Bath ring [10.0, 13.8] x [-3.4, -1.7] whose north wall is open between x 11.7 and 13.4, closed by two door-span barriers."""
    ring = box(10.0, -3.4, 13.8, -1.7).difference(box(10.0 + wall, -3.4 + wall, 13.8 - wall, -1.7 - wall))
    north_gap = box(11.7, -1.7 - wall, 13.4, -1.7)
    middle = barriers_gap
    barriers = [box(11.7, -1.7 - wall, middle[0], -1.7), box(middle[1], -1.7 - wall, 13.4, -1.7)]
    return [ring.difference(north_gap)], barriers


@pytest.mark.parametrize("upm", UPMS)
def test_hairline_crack_between_door_spans_does_not_open_the_room(upm: float) -> None:
    # Two adjacent door spans leave a 2 mm slit through the wall: the bath cell must stay closed (it merged with the hall).
    wall = 0.27
    walls, barriers = _bath_walls(wall, (12.5, 12.502))
    walls_s, barriers_s = [_s(item, upm) for item in walls], [_s(item, upm) for item in barriers]
    union = unary_union(walls_s + barriers_s)
    bath_point = _s(Point(11.5, -2.5), upm)
    assert not any(cell.contains(bath_point) for cell in enclosed_cells(union, upm))  # the crack really leaks without the closing
    closed = close_wall_cracks(union, 0.01 * upm, unary_union([_s(item, upm) for item in barriers]))
    cells = [cell for cell in enclosed_cells(closed, upm) if cell.contains(bath_point)]
    assert len(cells) == 1
    inner_area = (13.8 - wall - 10.0 - wall) * (-1.7 - wall + 3.4 - wall) * upm * upm
    assert cells[0].area == pytest.approx(inner_area, rel=0.02)
    bath_ceiling = box(10.5, -3.0, 13.2, -2.1)
    rooms, _ = rooms_from_wall_cells([_s(bath_ceiling, upm)], walls_s, barriers_s, None, None, upm, 0.0)
    bath = [room for room in rooms if room.contains(bath_point)]
    assert len(bath) == 1 and bath[0].area == pytest.approx(inner_area, rel=0.02)
    assert not bath[0].contains(_s(Point(12.0, -1.0), upm))
    # Rule off: the same input produces no bath room of the inner size.
    leaking, _ = rooms_from_wall_cells([_s(bath_ceiling, upm)], walls_s, barriers_s, None, None, upm, 0.0, PlanPolishSettings(wall_crack_m=0.0))
    assert not any(room.contains(bath_point) and abs(room.area - inner_area) < 0.02 * inner_area for room in leaking)


@pytest.mark.parametrize("upm", UPMS)
def test_crack_closing_keeps_real_openings_open(upm: float) -> None:
    # A 0.5 m (or wider) gap is a real passage, not a crack: the closing must leave it open, so no cell appears.
    wall = 0.27
    for gap in (0.5, 0.9):
        walls, barriers = _bath_walls(wall, (12.5, 12.5 + gap))
        union = unary_union([_s(item, upm) for item in walls + barriers])
        closed = close_wall_cracks(union, 0.01 * upm, unary_union([_s(item, upm) for item in barriers]))
        assert not any(cell.contains(_s(Point(11.5, -2.5), upm)) for cell in enclosed_cells(closed, upm)), gap
        assert closed.symmetric_difference(union).area <= 1e-6 * upm * upm * 100


def test_crack_closing_only_fills_seams_at_door_and_window_spans() -> None:
    left, right = box(0.0, 0.0, 2.0, 0.3), box(2.002, 0.0, 4.0, 0.3)  # 2 mm seam between two wall bodies
    door = box(6.0, 0.0, 7.0, 0.3)
    walls = unary_union([left, right, box(7.001, 0.0, 9.0, 0.3), door])  # 1 mm seam beside the door span
    closed = close_wall_cracks(walls, 0.01, door)
    seam_wall = box(2.0, 0.0, 2.002, 0.3)
    seam_door = box(7.0, 0.0, 7.001, 0.3)
    assert closed.intersection(seam_door).area == pytest.approx(seam_door.area, rel=1e-6)  # door seam filled
    assert closed.intersection(seam_wall).area == pytest.approx(0.0, abs=1e-9)  # wall-to-wall seam untouched
    assert close_wall_cracks(walls, 0.01).intersection(seam_wall).area == pytest.approx(seam_wall.area, rel=1e-6)  # no barriers: all closed
    assert close_wall_cracks(walls, 0.0, door) is walls


def test_cell_formed_only_by_crack_closing_must_be_small() -> None:
    # A 6 x 4 m space whose door spans leave a hairline crack stays with its measured ceiling room (a leak may be a real
    # passage); the same crack on a bath-sized cell is closed.
    wall = 0.2
    ring = box(0.0, 0.0, 6.0, 4.0).difference(box(wall, wall, 6.0 - wall, 4.0 - wall))
    gap = box(2.0, 4.0 - wall, 4.0, 4.0)
    barriers = [box(2.0, 4.0 - wall, 3.0, 4.0), box(3.002, 4.0 - wall, 4.0, 4.0)]
    ceiling = box(0.5, 0.5, 5.5, 3.5)
    rooms, stats = rooms_from_wall_cells([ceiling], [ring.difference(gap)], barriers, None, None, 1.0, 0.0)
    assert stats["room_cells"] == 0 and stats["rooms_outside_cells"] == 1
    assert len(rooms) == 1 and rooms[0].area == pytest.approx(ceiling.area, rel=0.02)  # the measured room (15 m2), not the 5.6 x 3.6 wall cell
    rooms_big, stats_big = rooms_from_wall_cells([ceiling], [ring.difference(gap)], barriers, None, None, 1.0, 0.0, PlanPolishSettings(wall_crack_cell_max_m2=30.0))
    assert stats_big["room_cells"] == 1 and rooms_big[0].area == pytest.approx(5.6 * 3.6, rel=0.01)


@pytest.mark.parametrize("upm", UPMS)
def test_short_wall_stretch_between_a_door_and_a_window_span_does_not_open_the_room(upm: float) -> None:
    # Door span 11.7-12.4 and window span 12.65-13.4 leave 0.25 m of north wall without any body (the measured faces
    # there were cut away): not a passage, the bath must stay closed.
    wall = 0.27
    walls, barriers = _bath_walls(wall, (12.4, 12.65))
    walls_s, barriers_s = [_s(item, upm) for item in walls], [_s(item, upm) for item in barriers]
    bath_point = _s(Point(11.5, -2.5), upm)
    assert not any(cell.contains(bath_point) for cell in enclosed_cells(unary_union(walls_s + barriers_s), upm))
    rooms, _ = rooms_from_wall_cells([_s(box(10.5, -3.0, 13.2, -2.1), upm)], walls_s, barriers_s, None, None, upm, 0.0)
    bath = [room for room in rooms if room.contains(bath_point)]
    inner_area = (13.8 - wall - 10.0 - wall) * (-1.7 - wall + 3.4 - wall) * upm * upm
    assert len(bath) == 1 and bath[0].area == pytest.approx(inner_area, rel=0.02)
    # a 0.5 m stretch (a real passage next to the door) is not closed
    walls_wide, barriers_wide = _bath_walls(wall, (12.4, 12.9))
    wide, _ = rooms_from_wall_cells([_s(box(10.5, -3.0, 13.2, -2.1), upm)], [_s(item, upm) for item in walls_wide], [_s(item, upm) for item in barriers_wide], None, None, upm, 0.0)
    assert not any(room.contains(bath_point) and abs(room.area - inner_area) < 0.02 * inner_area for room in wide)


@pytest.mark.parametrize("upm", UPMS)
def test_staggered_spans_with_a_wall_stub_gap_are_bridged_and_far_spans_are_not(upm: float) -> None:
    # Door span (y -1.97..-1.68) and window span (y -2.23..-1.94, offset wall) with 0.25 m of missing wall between them.
    door, window = box(11.7, -1.97, 12.4, -1.68), box(12.65, -2.23, 13.4, -1.94)
    bridges = bridge_barriers([_s(door, upm), _s(window, upm)], 0.30 * upm, 0.20 * upm * upm)
    assert bridges
    bridge = unary_union(bridges)
    assert bridge.covers(_s(Point(12.5, -1.95), upm)) and not bridge.intersects(_s(door, upm).buffer(-1e-6)) and not bridge.intersects(_s(window, upm).buffer(-1e-6))
    far = box(13.0, -2.23, 13.75, -1.94)  # 0.6 m from the door span: two separate openings
    assert bridge_barriers([_s(door, upm), _s(far, upm)], 0.30 * upm, 0.20 * upm * upm) == []
    assert bridge_barriers([_s(door, upm), _s(window, upm)], 0.20 * upm, 0.20 * upm * upm) == []  # reach below the 0.25 m gap
    assert bridge_barriers([_s(door, upm), _s(box(11.9, -1.7, 12.6, -1.4), upm)], 0.30 * upm, 0.20 * upm * upm) == []  # overlapping spans need no bridge


def test_a_single_kitchen_item_does_not_turn_a_bedroom_into_a_kitchen() -> None:
    """One upper cabinet at a door (inside the room outline) plus a bed: Schlafen.  Sink + hob: Küche."""
    room = box(0.0, 0.0, 4.0, 4.0)
    bed = ((1.0, 1.0), "bed")
    cabinet = ((3.9, 3.9), "kitchen")
    assert name_rooms([room], [bed, cabinet], 1.0)[0].name == "Schlafen"
    kitchen = name_rooms([room], [((1.0, 3.0), "kitchen"), ((2.0, 3.0), "kitchen")], 1.0)[0]
    assert kitchen.name == "Küche"
    assert name_rooms([room], [((1.0, 3.0), "kitchen")], 1.0)[0].name == "Küche"  # without a bed a single kitchen item still names the room
