"""Furniture appears in the plan only where it was detected: catalogue blocks draw no stock chairs, pillows or blankets.

Synthetic scenes only (no real plan data).  Footprints carry a few millimetres of noise so no corner lies exactly on a
pixel border or a catalogue size.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from samcloud.dxf.result import expand_insert
from samcloud.ortho.plan import _furniture_catalogue_insert, _insert_footprint
from samcloud.ortho.vector import RasterGrid, centred_catalogue_insert, match_catalogue
from samcloud.ortho.vector.footprints import FootprintFit, object_instances, refine_dining_instances
from samcloud.ortho.vector.symbols import CATALOGUE, SYMBOLS, symbol_extents

UPM_V8 = 0.8157  # detector units per metre of an older project frame (same value as the catalogue tests)


def _noisy_rectangle(centre: tuple[float, float], length: float, depth: float, angle_deg: float, rng: np.random.Generator) -> Polygon:
    """A measured table top: rectangle with 3 mm corner noise, rotated off the axes."""
    corners = np.asarray([(-length / 2, -depth / 2), (length / 2, -depth / 2), (length / 2, depth / 2), (-length / 2, depth / 2)])
    corners = corners + rng.normal(0.0, 0.003, corners.shape)
    polygon = affinity.rotate(Polygon(corners), angle_deg, origin=(0.0, 0.0))
    return affinity.translate(polygon, *centre)


def _expanded_geometry(insert: dict[str, object]) -> list[object]:
    shapes: list[object] = []
    for entity in expand_insert(insert):
        if entity["type"] == "polyline":
            points = [tuple(point) for point in entity["points"]]
            shapes.append(Polygon(points) if entity.get("closed") and len(points) >= 3 else LineString(points))
        elif entity["type"] == "line":
            shapes.append(LineString([tuple(entity["start"]), tuple(entity["end"])]))
        elif entity["type"] == "circle":
            shapes.append(Point(entity["center"]).buffer(float(entity["radius"])))
    return shapes


def test_table_blocks_draw_the_top_only() -> None:
    for entry in CATALOGUE["TABLE_DINING"]:
        entities = SYMBOLS[entry.block].entities
        assert len(entities) == 1 and entities[0]["type"] == "polyline" and entities[0]["closed"]
        xmin, ymin, xmax, ymax = symbol_extents(entry.block)
        # nothing beyond the top: no chair outlines on the long sides
        assert (xmax - xmin, ymax - ymin) == pytest.approx((entry.width_m, entry.depth_m), abs=1e-9)
        assert (xmin, ymin) == pytest.approx((-entry.width_m / 2.0, -entry.depth_m / 2.0), abs=1e-9)


def test_bed_blocks_draw_no_pillows_or_blanket() -> None:
    for kind in ("BED_SINGLE", "BED_DOUBLE"):
        for entry in CATALOGUE[kind]:
            entities = SYMBOLS[entry.block].entities
            assert symbol_extents(entry.block) == pytest.approx((-entry.width_m / 2.0, 0.0, entry.width_m / 2.0, entry.depth_m), abs=1e-9)
            for entity in entities:
                # frame and head board span the full width; a pillow (narrower box) or a blanket fold line is gone
                assert entity["type"] == "polyline" and entity["closed"]
                xs = [point[0] for point in entity["points"]]
                ys = [point[1] for point in entity["points"]]
                assert (min(xs), max(xs)) == pytest.approx((-entry.width_m / 2.0, entry.width_m / 2.0), abs=1e-9)
                assert min(ys) == pytest.approx(0.0, abs=1e-9)
            assert len(entities) == 2


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_table_block_inserted_draws_nothing_around_its_top(upm: float) -> None:
    """The catalogue table block (still available) draws its top only: nothing outside the insert footprint."""
    rng = np.random.default_rng(20261123)
    top = affinity.scale(_noisy_rectangle((3.217, 2.143), 1.373, 0.787, 37.3, rng), upm, upm, origin=(0.0, 0.0))
    rectangle = top.minimum_rotated_rectangle
    corners = np.asarray(rectangle.exterior.coords[:4])
    edges = [corners[1] - corners[0], corners[2] - corners[1]]
    long_index = int(np.argmax([np.hypot(*edge) for edge in edges]))
    measured = {"type": "insert", "name": "FURNITURE", "point": [rectangle.centroid.x, rectangle.centroid.y],
                "rotation": math.degrees(math.atan2(edges[long_index][1], edges[long_index][0])) % 180.0,
                "xscale": float(np.hypot(*edges[long_index])), "yscale": float(np.hypot(*edges[1 - long_index])), "attributes": {}}
    insert = centred_catalogue_insert(measured, ("TABLE_DINING",), upm, tolerance=0.15, centred_origin=True)
    assert insert is not None and str(insert["name"]) == "TABLE_DINING_1400X800"
    footprint = _insert_footprint(insert)
    assert footprint is not None
    shapes = _expanded_geometry(insert)
    assert shapes
    for shape in shapes:
        assert footprint.buffer(1e-6 * upm).covers(shape)
    drawn = unary_union([shape for shape in shapes if isinstance(shape, Polygon)])
    assert drawn.area == pytest.approx(footprint.area, rel=1e-6)


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
@pytest.mark.parametrize("label", ["table", "dining table", "coffee table"])
def test_table_keeps_its_measured_outline_even_at_a_catalogue_size(upm: float, label: str) -> None:
    """A low table at a sofa that matches a dining-table size within the tolerance gets no catalogue block: its
    measured outline is drawn (the export then shows the reconstructed top), so no standard extent is invented."""
    rng = np.random.default_rng(7)
    top = affinity.scale(_noisy_rectangle((3.217, 2.143), 1.373, 0.787, 37.3, rng), upm, upm, origin=(0.0, 0.0))
    assert match_catalogue(("TABLE_DINING",), 1.373, 0.787, tolerance=0.15) is not None  # the size would match
    assert _furniture_catalogue_insert(label, FootprintFit("rect", top, 0.0, 1.0), [], upm) is None


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_bed_at_a_wall_still_gets_its_catalogue_block(upm: float) -> None:
    """Guard: the measured-outline rule is for tables only; a bed of standard size at a wall keeps its block."""
    rng = np.random.default_rng(3)
    bed = _noisy_rectangle((2.013, 1.517), 1.618, 1.973, 0.0, rng)
    wall = LineString([(0.4, 2.511), (3.6, 2.511)])  # head wall 7 mm behind the bed's head edge
    bed = affinity.scale(bed, upm, upm, origin=(0.0, 0.0))
    wall = affinity.scale(wall, upm, upm, origin=(0.0, 0.0))
    insert = _furniture_catalogue_insert("bed", FootprintFit("rect", bed, 0.0, 1.0), [wall], upm)
    assert insert is not None and str(insert["name"]).startswith("BED_DOUBLE")


def _grid() -> RasterGrid:
    return RasterGrid(-0.5, 3.5, 0.01, 500, 500)


def _rasterise(polygon: Polygon, grid: RasterGrid) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    pixels = np.asarray([[round((x - grid.min_x) / grid.pixel_size), round((grid.max_y - y) / grid.pixel_size)] for x, y in polygon.exterior.coords], dtype=np.int32)
    cv2.fillPoly(mask, [pixels], 1)
    return mask.astype(bool)


def _dining_scene(chair_boxes: list[Polygon], seed: int) -> tuple[list[object], list[Polygon]]:
    """Table top (0.77 m) with the given detected chairs (0.85 m), rotated 4 degrees, 3 mm height noise."""
    grid = _grid()
    origin = (1.403, 1.197)

    def rotated(polygon: Polygon) -> Polygon:
        return affinity.rotate(polygon, 4.0, origin=origin)

    table = _rasterise(rotated(box(0.803, 0.797, 2.004, 1.596)), grid)
    fine = np.zeros(table.shape, dtype=np.uint8)
    dsm = np.zeros(table.shape, dtype=np.float32)
    chairs = [rotated(chair) for chair in chair_boxes]
    for chair in chairs:
        mask = _rasterise(chair, grid)
        fine[mask] = 2
        dsm[mask] = 0.85
    fine[table] = 1
    dsm[table] = 0.77
    dsm += np.random.default_rng(seed).normal(0.0, 0.003, dsm.shape).astype(np.float32)
    occupied = fine > 0
    classes = np.where(occupied, 117, 0).astype(np.uint8)
    instances, _metrics, height_map = object_instances(dsm, 0.0, classes, fine, np.where(occupied, 0.9, 0.0).astype(np.float32), ["table", "chair"], grid, 1.0, 2.4)
    stats: dict[str, int] = {}
    refined = refine_dining_instances(instances, classes, grid, height_map, math.radians(4.0), 1.0, stats)
    return refined, chairs


def test_table_without_detected_chairs_keeps_no_chair() -> None:
    refined, _chairs = _dining_scene([], seed=7)
    assert [item.label for item in refined] == ["table"]


def test_table_with_detected_chairs_keeps_exactly_those_at_their_place() -> None:
    measured = [box(0.952, 1.553, 1.371, 2.012), box(1.452, 0.348, 1.869, 0.851)]  # one chair on each long side
    refined, chairs = _dining_scene(measured, seed=11)
    found = [item for item in refined if item.label == "chair"]
    assert len([item for item in refined if item.label == "table"]) == 1
    assert len(found) == len(chairs) == 2  # no chairs added on the free sides
    for chair in chairs:
        nearest = min(found, key=lambda item: item.polygon.centroid.distance(chair.centroid))
        # the seat box sits where the chair was detected (seat sizes are a class prior, the position is measured)
        assert nearest.polygon.centroid.distance(chair.centroid) <= 0.15
        assert nearest.polygon.intersects(chair)
