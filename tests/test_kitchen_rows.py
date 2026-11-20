"""Round 5: tall kitchen rows from points, square kitchen inserts, hole-free furniture parts (synthetic noisy fixtures)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from samcloud.bim.furniture_parts import parts_hole_report, tile_parts_to_footprint
from samcloud.bim.kitchen_rows import KitchenRowSettings, detect_tall_rows, split_row
from samcloud.bim.plan_model import PlanFurniture, _apply_tall_rows, _parts_matching_footprint, _square_kitchen_inserts

UPMS = (1.0, 0.9891)


def _ring(polygon: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in list(polygon.exterior.coords)[:-1]]


def _noisy_ring(polygon: Polygon, rng: np.random.Generator, sigma: float) -> list[list[float]]:
    return [[float(x + rng.normal(0, sigma)), float(y + rng.normal(0, sigma))] for x, y in list(polygon.exterior.coords)[:-1]]


# --------------------------------------------------------------------------------------------- furniture parts

@pytest.mark.parametrize("upm", UPMS)
def test_sofa_parts_have_no_hole_where_something_lay_on_it(upm: float) -> None:
    """Measured seat has a hole (an item lay on it, unseen), a high blob (cushion pile) and the footprint is bigger than the measured parts."""
    rng = np.random.default_rng(11)
    footprint = affinity.scale(Polygon([(0, 0), (2.2, 0), (2.2, 0.95), (0, 0.95)]), upm, upm, origin=(0, 0))
    back = box(0.0, 0.70, 2.15, 0.95)
    seat_measured = box(0.03, 0.02, 2.12, 0.68).difference(box(0.9, 0.2, 1.4, 0.5))  # unseen hole in the middle of the seat
    blob = box(0.35, 0.25, 0.60, 0.45)  # cushions: higher island on the seat
    parts = [
        {"footprint": _noisy_ring(affinity.scale(seat_measured, upm, upm, origin=(0, 0)), rng, 0.004 * upm), "z_max": 0.45 * upm, "z_min": 0.0},
        {"footprint": _noisy_ring(affinity.scale(back, upm, upm, origin=(0, 0)), rng, 0.004 * upm), "z_max": 0.78 * upm, "z_min": 0.0},
        {"footprint": _noisy_ring(affinity.scale(blob, upm, upm, origin=(0, 0)), rng, 0.004 * upm), "z_max": 0.62 * upm, "z_min": 0.0},
    ]
    before = parts_hole_report(parts, footprint)
    assert before["missing_share"] > 0.05  # the fixture really has holes
    tiled = tile_parts_to_footprint(parts, footprint, upm)
    report = parts_hole_report(tiled, footprint)
    assert report["rings"] == 0 and report["missing_share"] < 0.02 and report["outside_share"] < 0.02 and report["overlap_share"] < 0.02
    levels = sorted({round(part["z_max"] / upm, 2) for part in tiled})
    assert levels == [0.45, 0.78], "the cushion island took the level around it; seat and back stay"
    seat = unary_union([Polygon(part["footprint"]) for part in tiled if abs(part["z_max"] / upm - 0.45) < 0.01])
    assert seat.buffer(0.01 * upm).covers(box(0.95 * upm, 0.25 * upm, 1.35 * upm, 0.45 * upm)), "the unseen hole is filled with seat height"


@pytest.mark.parametrize("upm", UPMS)
def test_parts_follow_a_regularised_footprint(upm: float) -> None:
    """The exported footprint is larger and L-shaped (regularised after the measurement): parts become exact tiles of it."""
    rng = np.random.default_rng(5)
    measured = [
        {"footprint": _noisy_ring(affinity.scale(box(0.2, 0.2, 1.2, 1.0), upm, upm, origin=(0, 0)), rng, 0.003 * upm), "z_max": 0.45 * upm},
        {"footprint": _noisy_ring(affinity.scale(box(0.2, 1.0, 1.2, 1.3), upm, upm, origin=(0, 0)), rng, 0.003 * upm), "z_max": 0.8 * upm},
    ]
    footprint = affinity.scale(Polygon([(0, 0), (1.4, 0), (1.4, 0.7), (1.0, 0.7), (1.0, 1.5), (0, 1.5)]), upm, upm, origin=(0, 0))
    tiled = _parts_matching_footprint(measured, footprint, minimum_iou=0.4, units_per_metre=upm)
    report = parts_hole_report(tiled, footprint)
    assert tiled and report["rings"] == 0 and report["missing_share"] < 0.02 and report["outside_share"] < 0.02
    assert {round(p["z_max"] / upm, 2) for p in tiled} == {0.45, 0.8}


def test_parts_matching_still_rejects_a_shifted_footprint() -> None:
    footprint = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    shifted = [{"footprint": [[1.6, 0], [3.6, 0], [3.6, 1], [1.6, 1]], "z_max": 0.4}]
    assert _parts_matching_footprint(shifted, footprint) == []


def test_edge_touching_high_part_is_kept_as_backrest() -> None:
    footprint = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    parts = [{"footprint": _ring(box(0, 0, 2, 0.6)), "z_max": 0.45}, {"footprint": _ring(box(0, 0.6, 2, 1.0)), "z_max": 0.8}]
    tiled = tile_parts_to_footprint(parts, footprint, 1.0)
    assert sorted(round(p["z_max"], 2) for p in tiled) == [0.45, 0.8]


# --------------------------------------------------------------------------------------------- tall rows

def _face(rng: np.random.Generator, axis: str, fixed: float, lo: float, hi: float, z0: float, z1: float, n: int, upm: float) -> np.ndarray:
    along = rng.uniform(lo, hi, n)
    z = rng.uniform(z0, z1, n)
    across = fixed + rng.normal(0.0, 0.008, n)
    xy = np.column_stack((across, along)) if axis == "v" else np.column_stack((along, across))
    return np.column_stack((xy * upm, z * upm))


def _scene(upm: float, rng: np.random.Generator):
    """North wall y 14.5..14.8, east wall x -2.0..-1.8 (from y 12.8 up), base run along the north wall, upper cabinets over it."""
    walls = unary_union([box(-6.4, 14.5, -1.8, 14.8), box(-2.0, 12.76, -1.8, 14.8)])
    walls = affinity.scale(walls, upm, upm, origin=(0, 0))
    leg_north = affinity.scale(box(-6.1, 13.9, -3.3, 14.5), upm, upm, origin=(0, 0))
    leg_west = affinity.scale(box(-6.1, 12.0, -5.5, 14.5), upm, upm, origin=(0, 0))
    points = [
        _face(rng, "v", -2.65, 12.35, 13.9, 0.1, 2.25, 2400, upm),  # east row front face (tall, continuous)
        _face(rng, "u", 13.95, -3.25, -2.7, 0.1, 2.25, 1000, upm),  # oven column front face
        _face(rng, "u", 14.2, -6.0, -3.4, 1.42, 2.18, 600, upm),  # upper cabinets over the base run (mid band free)
        _face(rng, "u", 13.92, -6.0, -3.4, 0.2, 0.9, 400, upm),  # base run front face
    ]
    return walls, unary_union([leg_north, leg_west]), np.vstack(points)


@pytest.mark.parametrize("upm", UPMS)
def test_tall_rows_fill_front_to_wall_and_split_l_clusters(upm: float) -> None:
    rng = np.random.default_rng(7)
    walls, legs, points = _scene(upm, rng)
    rows = detect_tall_rows(points, walls, legs, upm, 0.0, KitchenRowSettings(), legs)
    assert len(rows) == 2, [row.polygon.bounds for row in rows]
    by_x = sorted(rows, key=lambda row: row.polygon.bounds[0])
    oven, east = by_x
    ex0, ey0, ex1, ey1 = [value / upm for value in east.polygon.bounds]
    assert ex0 == pytest.approx(-2.66, abs=0.07) and ex1 == pytest.approx(-2.0, abs=0.07), "front face back to the east wall"
    assert ey0 == pytest.approx(12.35, abs=0.10) and ey1 == pytest.approx(13.9, abs=0.12)
    ox0, oy0, ox1, oy1 = [value / upm for value in oven.polygon.bounds]
    assert oy1 == pytest.approx(14.5, abs=0.07), "oven column reaches the north wall behind it"
    assert ox0 == pytest.approx(-3.3, abs=0.08), "oven column ends at the kitchen leg"
    assert east.top_m == pytest.approx(2.2, abs=0.1) and oven.top_m == pytest.approx(2.2, abs=0.1)
    for row in rows:
        assert row.polygon.minimum_rotated_rectangle.area == pytest.approx(row.polygon.area, rel=0.02), "rows are rectangles"
        assert row.polygon.intersection(walls).area < 1e-3 * upm * upm
    # the hanging upper cabinets over the base run (mid band free) are not tall rows
    assert all(row.polygon.bounds[3] / upm > 12.5 and row.polygon.bounds[0] / upm > -3.5 for row in rows)


def test_rows_need_a_kitchen_nearby() -> None:
    rng = np.random.default_rng(3)
    walls, legs, points = _scene(1.0, rng)
    far = box(10.0, 0.0, 12.0, 0.6)
    assert detect_tall_rows(points, walls, far, 1.0, 0.0, KitchenRowSettings(), far) == []


def test_split_row_around_a_fridge() -> None:
    row = box(-2.65, 12.35, -2.0, 13.9)
    fridge = box(-2.66, 12.29, -1.92, 13.04)
    units = split_row(row, [fridge], 1.0)
    assert [index for _, index in units] == [0, None]
    fridge_unit, rest = units[0][0], units[1][0]
    assert fridge_unit.bounds[3] == pytest.approx(13.04, abs=1e-6) and rest.bounds[1] == pytest.approx(13.04, abs=1e-6)
    assert unary_union([fridge_unit, rest]).area == pytest.approx(row.area, rel=1e-6)
    # a free piece shorter than min_unit_m is absorbed by the fridge
    tight = split_row(row, [box(-2.66, 12.3, -1.92, 13.75)], 1.0)
    assert [index for _, index in tight] == [0]


# --------------------------------------------------------------------------------------------- plan model

def _item(identifier: str, polygon: Polygon, **kwargs) -> PlanFurniture:
    return PlanFurniture(id=identifier, instance_id=kwargs.pop("instance_id", None), label=kwargs.pop("label", "x"), semantic_label=kwargs.pop("semantic_label", "x"),
                         catalogue_kind=kwargs.pop("catalogue_kind", None), layer=kwargs.pop("layer", "A-FURN"), footprint=_ring(polygon), **kwargs)


@pytest.mark.parametrize("upm", UPMS)
def test_diagonal_sink_is_squared_inside_its_leg(upm: float) -> None:
    leg_w = affinity.scale(box(-6.1, 12.0, -5.5, 14.5), upm, upm, origin=(0, 0))
    leg_n = affinity.scale(box(-6.1, 13.9, -3.3, 14.5), upm, upm, origin=(0, 0))
    diamond = affinity.rotate(affinity.scale(box(-0.49, -0.25, 0.49, 0.25), upm, upm, origin=(0, 0)), 45.0, origin=(0, 0))
    diamond = affinity.translate(diamond, -5.5 * upm, 13.85 * upm)
    sink = _item("F033", diamond, catalogue_kind="KITCHEN_SINK", layer="A-EINB")
    hob = _item("F034", affinity.scale(box(-4.2, 14.0, -3.6, 14.45), upm, upm, origin=(0, 0)), catalogue_kind="COOKTOP", layer="A-EINB")
    report: dict = {}
    _square_kitchen_inserts([sink, hob], [leg_w, leg_n], upm, report)
    polygon = Polygon(sink.footprint)
    assert unary_union([leg_w, leg_n]).buffer(1e-6).covers(polygon)
    assert polygon.minimum_rotated_rectangle.area == pytest.approx(polygon.area, rel=1e-6)
    ring = list(polygon.minimum_rotated_rectangle.exterior.coords)
    angle = math.degrees(math.atan2(ring[1][1] - ring[0][1], ring[1][0] - ring[0][0]))
    assert min(abs(angle) % 90.0, 90.0 - abs(angle) % 90.0) < 0.5
    assert [entry["id"] for entry in report["kitchen_inserts_squared"]] == ["F033"], "the straight hob is left alone"


@pytest.mark.parametrize("upm", UPMS)
def test_apply_tall_rows_replaces_owned_items_and_keeps_the_fridge(upm: float) -> None:
    rng = np.random.default_rng(9)
    walls, legs, points = _scene(upm, rng)
    leg_polygons = list(legs.geoms) if hasattr(legs, "geoms") else [legs]
    floor_z = -0.01 * upm
    hob = _item("F004", affinity.scale(box(-4.2, 14.0, -3.6, 14.45), upm, upm, origin=(0, 0)), catalogue_kind="COOKTOP", layer="A-EINB")
    fridge = _item("F006", affinity.scale(box(-2.66, 12.29, -1.92, 13.04), upm, upm, origin=(0, 0)), label="Kühlschrank", semantic_label="refrigerator", z_min=floor_z, z_max=floor_z + 2.29 * upm, z_source="instance_bbox")
    stray = _item("F005", affinity.scale(box(-1.9, 12.1, -1.42, 12.76), upm, upm, origin=(0, 0)), label="Schrank", semantic_label="cabinet", instance_id=38)

    class _Footprints:
        points = {38: np.column_stack((rng.uniform(-2.66, -2.6, 200), rng.uniform(12.4, 13.9, 200), rng.uniform(0.1, 0.9, 200))) * upm}

        def has(self, identifier):  # noqa: ANN001
            return identifier in self.points

    walls_union = walls
    items = [hob, fridge, stray]
    report: dict = {}
    result = _apply_tall_rows(items, leg_polygons, points, walls_union, [], _Footprints(), upm, floor_z, KitchenRowSettings(), report)
    ids = {item.id for item in result}
    assert "F005" not in ids, "the cabinet whose points stand in the tall row is replaced by the row"
    assert "F006" in ids
    labels = sorted(item.label for item in result if item.id not in {"F004", "F006"})
    assert labels == ["Backofen", "Hochschrank"], labels
    oven = next(item for item in result if item.label == "Backofen")
    assert oven.catalogue_kind == "oven_tower" and oven.z_source == "points" and (oven.z_max - floor_z) / upm == pytest.approx(2.2, abs=0.1)
    fridge_polygon = Polygon(fridge.footprint)
    cabinet = Polygon(next(item for item in result if item.label == "Hochschrank").footprint)
    row = unary_union([fridge_polygon, cabinet])
    assert row.bounds[0] / upm == pytest.approx(-2.65, abs=0.07) and row.bounds[2] / upm == pytest.approx(-2.0, abs=0.07)
    assert fridge_polygon.intersection(cabinet).area < 1e-6 * upm * upm
    assert report["kitchen_rows"] and len(report["kitchen_rows"]) == 2


@pytest.mark.parametrize("upm", UPMS)
def test_neighbouring_tall_unit_is_trimmed_clear_of_the_rows(upm: float) -> None:
    """A wall-stage tall unit that overlaps the measured oven row gives way and stays a rectangle."""
    rng = np.random.default_rng(13)
    walls, legs, points = _scene(upm, rng)
    leg_polygons = list(legs.geoms) if hasattr(legs, "geoms") else [legs]
    floor_z = 0.0
    corner = _item("F034", affinity.scale(box(-2.9, 13.8, -2.0, 14.5), upm, upm, origin=(0, 0)), label="tall_unit", semantic_label="kitchen cabinet", catalogue_kind="tall_unit", layer="A-EINB",
                   z_min=0.0, z_max=2.1 * upm, z_source="points")
    report: dict = {}
    result = _apply_tall_rows([corner], leg_polygons, points, walls, [], None, upm, floor_z, KitchenRowSettings(), report)
    oven_row = unary_union([Polygon(item.footprint) for item in result if item.id != "F034"])
    trimmed = Polygon(next(item for item in result if item.id == "F034").footprint)
    assert trimmed.intersection(oven_row).area < 0.02 * trimmed.area
    assert trimmed.minimum_rotated_rectangle.area == pytest.approx(trimmed.area, rel=0.02)
    assert report["kitchen_rows_trimmed_neighbours"] == ["F034"]
    # it overlaps BOTH rows (oven column and east row): what is left is still a sizeable rectangle clear of each
    east_row = Polygon(next(item for item in result if item.label == "Hochschrank").footprint)
    assert trimmed.intersection(east_row).area < 0.02 * trimmed.area and trimmed.area > 0.25 * upm * upm
