"""Built-in fronts: walls move behind tall built-in units, and only behind them.

The scenes are synthetic rooms (rasters plus wall lines) with noise, a few degrees
of rotation and a non-metric unit scale.  The room frame is in metres; every
coordinate is rotated, scaled by ``units_per_metre`` and shifted before use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

import ezdxf

from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
from samcloud.dxf.writer import export_plan
from samcloud.ortho.vector.builtin_fronts import resolve_builtin_fronts, tall_furniture_mask
from samcloud.ortho.vector.raster_grid import RasterGrid

FINE_NAMES = ["wardrobe", "shelf", "sideboard"]
ROOM_WIDTH, ROOM_DEPTH = 5.0, 3.5


@dataclass
class Scene:
    upm: float
    angle: float
    grid: RasterGrid
    classes: np.ndarray
    fine: np.ndarray
    height: np.ndarray
    free: np.ndarray
    layers: dict[str, list[dict[str, Any]]]
    room: Polygon
    transform: Any
    strip: Polygon | None = None
    back_line: LineString | None = None
    front_line: LineString | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _make_transform(upm: float, angle_deg: float):
    cosine, sine = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))

    def transform(x: float, y: float) -> tuple[float, float]:
        return (upm * (cosine * x - sine * y) + 1.0 * upm, upm * (sine * x + cosine * y) + 1.0 * upm)

    return transform


def _fill(mask: np.ndarray, grid: RasterGrid, polygon: list[tuple[float, float]], transform: Any, value: Any) -> np.ndarray:
    corners = [transform(x, y) for x, y in polygon]
    pixels = np.asarray([((x - grid.min_x) / grid.pixel_size - 0.5, (grid.max_y - y) / grid.pixel_size - 0.5) for x, y in corners])
    layer = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(layer, [np.rint(pixels).astype(np.int32)], 1)
    selected = layer.astype(bool)
    mask[selected] = value
    return selected


def _rect(x0: float, x1: float, y0: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _wall(transform: Any, rng: np.random.Generator, upm: float, start: tuple[float, float], end: tuple[float, float]) -> dict[str, Any]:
    jitter = rng.normal(0.0, 0.004 * upm, size=4)
    first, second = transform(*start), transform(*end)
    return {"type": "polyline", "points": [[first[0] + jitter[0], first[1] + jitter[1]], [second[0] + jitter[2], second[1] + jitter[3]]], "closed": False, "_source": "measured"}


def _scene(
    upm: float, angle: float, *, seed: int = 7, unit: tuple[float, float, float, float, str, float] | None = None,
    front_wall: bool = True, room_reaches_wall: bool = True, wall_front_extends: bool = False,
) -> Scene:
    """A 5 x 3.5 m room.  ``unit`` is (x0, x1, y0, y1, fine label, height in m) in room metres."""
    rng = np.random.default_rng(seed)
    transform = _make_transform(upm, angle)
    corners = [transform(x, y) for x in (-0.5, ROOM_WIDTH + 0.5) for y in (-0.5, ROOM_DEPTH + 0.5)]
    pixel = 0.02 * upm
    min_x, max_y = min(c[0] for c in corners) - 0.2 * upm, max(c[1] for c in corners) + 0.2 * upm
    width = int(math.ceil((max(c[0] for c in corners) + 0.2 * upm - min_x) / pixel))
    height = int(math.ceil((max_y - (min(c[1] for c in corners) - 0.2 * upm)) / pixel))
    grid = RasterGrid(min_x, max_y, pixel, width, height)
    classes = np.zeros((height, width), dtype=np.uint8)
    fine = np.zeros((height, width), dtype=np.uint8)
    heights = np.zeros((height, width), dtype=np.float32)
    _fill(classes, grid, _rect(0.0, ROOM_WIDTH, 0.0, ROOM_DEPTH), transform, 102)
    layers: dict[str, list[dict[str, Any]]] = {name: [] for name in ("A-WALL", "A-EINB", "A-FURN", "A-FURN-DETL", "A-ANNO")}
    strip = back_line = front_line = None
    walls = layers["A-WALL"]
    walls.append(_wall(transform, rng, upm, (0.0, 0.0), (ROOM_WIDTH, 0.0)))
    walls.append(_wall(transform, rng, upm, (0.0, 0.0), (0.0, ROOM_DEPTH)))
    walls.append(_wall(transform, rng, upm, (ROOM_WIDTH, 0.0), (ROOM_WIDTH, ROOM_DEPTH)))
    back_line = LineString([transform(0.0, ROOM_DEPTH), transform(ROOM_WIDTH, ROOM_DEPTH)])
    notch: list[tuple[float, float]] | None = None
    if unit is None:
        walls.append(_wall(transform, rng, upm, (0.0, ROOM_DEPTH), (ROOM_WIDTH, ROOM_DEPTH)))
    else:
        x0, x1, y0, y1, label, height_m = unit
        mask = _fill(classes, grid, _rect(x0, x1, y0, y1), transform, 117)
        label_mask = mask & (rng.random(mask.shape) > 0.12)
        fine[label_mask] = FINE_NAMES.index(label) + 1
        heights[mask] = (height_m + rng.normal(0.0, 0.04, size=int(mask.sum()))) * upm
        strip = Polygon([transform(x, y) for x, y in _rect(x0, x1, y0, y1)])
        builtin = abs(y1 - ROOM_DEPTH) < 1e-6
        if builtin:
            # The real wall is only visible beside the unit.
            walls.append(_wall(transform, rng, upm, (0.0, ROOM_DEPTH), (x0, ROOM_DEPTH)))
            walls.append(_wall(transform, rng, upm, (x1, ROOM_DEPTH), (ROOM_WIDTH, ROOM_DEPTH)))
            front_line = LineString([transform(x0, y0), transform(x1, y0)])
            if front_wall:
                start, stop = (0.0, ROOM_WIDTH) if wall_front_extends else (x0, x1)
                walls.append(_wall(transform, rng, upm, (start, y0), (stop, y0)))
            if not room_reaches_wall:
                # The ceiling is not seen above the unit: the room outline has a notch in front of it.
                notch = [(0.0, 0.0), (ROOM_WIDTH, 0.0), (ROOM_WIDTH, ROOM_DEPTH), (x1, ROOM_DEPTH), (x1, y0), (x0, y0), (x0, ROOM_DEPTH), (0.0, ROOM_DEPTH)]
        else:
            walls.append(_wall(transform, rng, upm, (0.0, ROOM_DEPTH), (ROOM_WIDTH, ROOM_DEPTH)))
        # Floor behind and beside furniture is not free.
        classes[mask] = 117
        outline = Polygon([transform(x, y) for x, y in _rect(x0 - 0.03, x1 + 0.02, y0 - 0.20, y1 - 0.05)])
        layers["A-FURN"].append({"type": "polyline", "points": [[float(px), float(py)] for px, py in outline.exterior.coords[:-1]], "closed": True, "holes": [], "_label": {"wardrobe": "Schrank", "shelf": "Regal", "sideboard": "Sideboard"}[label]})
        layers["A-ANNO"].append({"type": "text", "text": f"{ {'wardrobe': 'Schrank', 'shelf': 'Regal', 'sideboard': 'Sideboard'}[label] } h={height_m:.2f}", "point": list(Point(*transform((x0 + x1) / 2.0, (y0 + y1) / 2.0)).coords[0]), "height": 0.08 * upm, "rotation": 0.0})
    layers["A-ANNO"].append({"type": "text", "text": "Raum 1", "point": list(transform(2.5, 1.0)), "height": 0.12 * upm, "rotation": 0.0})
    room = Polygon([transform(x, y) for x, y in (notch or _rect(0.0, ROOM_WIDTH, 0.0, ROOM_DEPTH))])
    free = classes == 102
    return Scene(upm, angle, grid, classes, fine, heights, free, layers, room, transform, strip, back_line, front_line)


def _resolve(scene: Scene) -> dict[str, Any]:
    tall = tall_furniture_mask(scene.fine, FINE_NAMES, scene.classes, scene.height, (100,), scene.grid, scene.upm)
    return resolve_builtin_fronts(
        scene.layers, [scene.room], tall=tall, free_floor=scene.free, fine_classes=scene.fine, fine_names=FINE_NAMES,
        grid=scene.grid, units_per_metre=scene.upm,
    )


def _wall_lines(scene: Scene) -> list[LineString]:
    return [LineString(entity["points"]) for entity in scene.layers["A-WALL"] if entity.get("type") == "polyline" and len(entity["points"]) >= 2]


def _covered_length(lines: list[LineString], reference: LineString, tolerance: float) -> float:
    """Length of ``reference`` that lies within ``tolerance`` of one of ``lines`` (sampled)."""
    samples = [reference.interpolate(fraction, normalized=True) for fraction in np.linspace(0.0, 1.0, 400)]
    hits = [any(line.distance(point) <= tolerance for line in lines) for point in samples]
    return float(np.mean(hits)) * reference.length


CASES = [(1.0, 0.0), (0.8157, 3.0), (0.8157, -4.5)]


@pytest.mark.parametrize("room_reaches_wall", [True, False], ids=["ceiling_to_wall", "ceiling_stops_at_front"])
@pytest.mark.parametrize("upm,angle", CASES)
def test_wall_moves_behind_builtin_wardrobe(upm: float, angle: float, room_reaches_wall: bool) -> None:
    depth = 0.45
    x0, x1 = 0.6, 3.6
    scene = _scene(upm, angle, unit=(x0, x1, ROOM_DEPTH - depth, ROOM_DEPTH, "wardrobe", 1.95), front_wall=room_reaches_wall, room_reaches_wall=room_reaches_wall)
    stats = _resolve(scene)
    assert stats["builtin_fronts"] == 1
    strip = scene.strip
    assert strip is not None and scene.back_line is not None and scene.front_line is not None
    tolerance = 0.05 * upm
    walls = _wall_lines(scene)
    # The wall now covers the whole length of the unit at the back line.
    back_pieces = [line for line in walls if max(scene.back_line.distance(Point(point)) for point in line.coords) <= tolerance]
    unit_back = LineString([scene.back_line.interpolate(scene.back_line.project(Point(strip.exterior.coords[2]))), scene.back_line.interpolate(scene.back_line.project(Point(strip.exterior.coords[3])))])
    assert _covered_length(back_pieces, unit_back, 0.03 * upm) >= 0.97 * unit_back.length
    # Nothing is left at the front line or inside the strip.
    inner = strip.buffer(-0.02 * upm)
    assert not any(line.intersects(inner) for line in walls)
    assert not any(max(scene.front_line.distance(Point(point)) for point in line.coords) <= tolerance for line in walls)
    # One A-EINB rectangle labelled Schrank, equal to the strip.
    polygons = [entity for entity in scene.layers["A-EINB"] if entity.get("type") == "polyline" and entity.get("closed")]
    assert len(polygons) == 1
    rectangle = Polygon(polygons[0]["points"])
    assert len(rectangle.exterior.coords) - 1 == 4
    assert rectangle.symmetric_difference(strip).area <= 0.08 * strip.area
    texts = [entity for entity in scene.layers["A-EINB"] if entity.get("type") == "text"]
    assert [entity["text"] for entity in texts] == ["Schrank"]
    assert rectangle.contains(Point(texts[0]["point"]))
    # The furniture outline and its label are absorbed.
    assert not any(Polygon(entity["points"]).intersection(strip).area > 0.05 * strip.area for entity in scene.layers["A-FURN"] if entity.get("closed"))
    assert not any(str(entity.get("text", "")).startswith("Schrank") for entity in scene.layers["A-ANNO"])
    assert any(entity.get("text") == "Raum 1" for entity in scene.layers["A-ANNO"])


@pytest.mark.parametrize("upm,angle", CASES)
def test_freestanding_tall_furniture_keeps_wall(upm: float, angle: float) -> None:
    # A 1.8 m shelf stands 0.3 m in front of the wall; no wall line runs in front of it.
    scene = _scene(upm, angle, unit=(1.0, 3.0, ROOM_DEPTH - 0.3 - 0.35, ROOM_DEPTH - 0.3, "shelf", 1.8))
    before_walls = [list(map(list, entity["points"])) for entity in scene.layers["A-WALL"]]
    before_furniture = [dict(entity) for entity in scene.layers["A-FURN"]]
    stats = _resolve(scene)
    assert stats["builtin_fronts"] == 0
    assert [list(map(list, entity["points"])) for entity in scene.layers["A-WALL"]] == before_walls
    assert scene.layers["A-FURN"] == before_furniture
    assert not [entity for entity in scene.layers["A-EINB"] if entity.get("type") == "polyline"]
    assert any(entity.get("text", "").startswith("Regal") for entity in scene.layers["A-ANNO"])


@pytest.mark.parametrize("front_wall", [False, True], ids=["no_front_line", "spurious_front_line"])
@pytest.mark.parametrize("upm,angle", CASES)
def test_low_furniture_does_not_move_wall(upm: float, angle: float, front_wall: bool) -> None:
    # A 0.8 m sideboard against the wall; even a wall line measured at its front must stay.
    scene = _scene(upm, angle, unit=(0.6, 3.6, ROOM_DEPTH - 0.45, ROOM_DEPTH, "sideboard", 0.8), front_wall=front_wall)
    before_walls = [list(map(list, entity["points"])) for entity in scene.layers["A-WALL"]]
    stats = _resolve(scene)
    assert stats["builtin_fronts"] == 0
    assert [list(map(list, entity["points"])) for entity in scene.layers["A-WALL"]] == before_walls
    assert not [entity for entity in scene.layers["A-EINB"] if entity.get("type") == "polyline"]
    assert any(entity.get("_label") == "Sideboard" for entity in scene.layers["A-FURN"])


@pytest.mark.parametrize("upm,angle", [(1.0, 0.0), (0.8157, 3.0)])
def test_front_wall_continuing_beyond_unit_is_a_real_wall(upm: float, angle: float) -> None:
    # The unit sits in a recess behind a wall line that carries on past it: keep that wall.
    scene = _scene(upm, angle, unit=(1.5, 2.8, ROOM_DEPTH - 0.45, ROOM_DEPTH, "wardrobe", 1.95), wall_front_extends=True)
    before_walls = [list(map(list, entity["points"])) for entity in scene.layers["A-WALL"]]
    stats = _resolve(scene)
    assert stats["builtin_fronts"] == 0
    assert [list(map(list, entity["points"])) for entity in scene.layers["A-WALL"]] == before_walls


def _beyond_room(scene: Scene, x0: float, x1: float) -> Polygon:
    """A second ceiling room whose top edge is collinear with the unit's front line and carries on past the unit."""
    y_front = ROOM_DEPTH - 0.45
    return Polygon([scene.transform(x, y) for x, y in ((x0, y_front - 1.4), (x1, y_front - 1.4), (x1, y_front), (x0, y_front))])


@pytest.mark.parametrize("upm,angle", [(1.0, 0.0), (0.8157, 3.0)])
def test_collinear_ceiling_edge_beyond_unit_does_not_block_wardrobe(upm: float, angle: float, tmp_path: Any) -> None:
    # The bedroom outline stops at the unit; the next room's outline merely continues on the same line.
    # That is no measured wall, so the wardrobe still gets its back wall and its A-EINB rectangle.
    scene = _scene(upm, angle, unit=(0.6, 3.6, ROOM_DEPTH - 0.45, ROOM_DEPTH, "wardrobe", 1.95))
    neighbour = _beyond_room(scene, 3.5, 4.9)
    tall = tall_furniture_mask(scene.fine, FINE_NAMES, scene.classes, scene.height, (100,), scene.grid, scene.upm)
    stats = resolve_builtin_fronts(
        scene.layers, [scene.room, neighbour], tall=tall, free_floor=scene.free, fine_classes=scene.fine, fine_names=FINE_NAMES,
        grid=scene.grid, units_per_metre=scene.upm,
    )
    assert stats["builtin_fronts"] == 1
    polygons = [entity for entity in scene.layers["A-EINB"] if entity.get("type") == "polyline" and entity.get("closed")]
    assert len(polygons) == 1 and Polygon(polygons[0]["points"]).symmetric_difference(scene.strip).area <= 0.08 * scene.strip.area
    # Read the plan back from a DXF: the A-EINB rectangle is there and no wall runs in front of the unit.
    frame = Frame(np.zeros(3))
    layers = {name: scene.layers[name] for name in ("A-WALL", "A-EINB")}
    result = DxfPlanResult(
        "Bedroom", "level_1", "indoor", None, upm, frame, "fp", layers,
        {name: LayerStatus(name, "ok", None, len(items)) for name, items in layers.items()}, {"A-WALL": 1, "A-EINB": 30},
        BackgroundImage(None, None, (0.0, 0.0, 10.0, 10.0)),
    )
    document = ezdxf.readfile(export_plan(result, tmp_path / "dxf", coordinates="local"))
    einb = [entity for entity in document.modelspace() if entity.dxf.layer == "A-EINB" and entity.dxftype() == "LWPOLYLINE"]
    assert len(einb) == 1 and Polygon([(x, y) for x, y, *_ in einb[0].get_points()]).symmetric_difference(scene.strip).area <= 0.08 * scene.strip.area
    front = scene.front_line
    for entity in document.modelspace():
        if entity.dxf.layer == "A-WALL" and entity.dxftype() == "LWPOLYLINE":
            points = [(x, y) for x, y, *_ in entity.get_points()]
            assert not all(front.distance(Point(point)) <= 0.05 * upm for point in points)


@pytest.mark.parametrize("upm,angle", [(1.0, 0.0), (0.8157, 3.0)])
def test_measured_wall_beyond_unit_still_blocks_wardrobe(upm: float, angle: float) -> None:
    # Same neighbour room, but a measured wall line carries the front on past the unit: it is a real wall.
    scene = _scene(upm, angle, unit=(0.6, 3.6, ROOM_DEPTH - 0.45, ROOM_DEPTH, "wardrobe", 1.95))
    y_front = ROOM_DEPTH - 0.45
    scene.layers["A-WALL"].append({"type": "polyline", "points": [list(scene.transform(3.6, y_front)), list(scene.transform(4.9, y_front))], "closed": False, "_source": "measured"})
    neighbour = _beyond_room(scene, 3.5, 4.9)
    tall = tall_furniture_mask(scene.fine, FINE_NAMES, scene.classes, scene.height, (100,), scene.grid, scene.upm)
    stats = resolve_builtin_fronts(
        scene.layers, [scene.room, neighbour], tall=tall, free_floor=scene.free, fine_classes=scene.fine, fine_names=FINE_NAMES,
        grid=scene.grid, units_per_metre=scene.upm,
    )
    assert stats["builtin_fronts"] == 0


def _add_tall_block(scene: Scene, x0: float, x1: float, y0: float, y1: float) -> None:
    mask = _fill(scene.classes, scene.grid, _rect(x0, x1, y0, y1), scene.transform, 117)
    scene.fine[mask] = FINE_NAMES.index("shelf") + 1
    scene.height[mask] = 1.8 * scene.upm


def _resolve_with_neighbour(scene: Scene) -> dict[str, Any]:
    # The front line carries on past the unit as the outline of the next room, so the run is free to stretch.
    tall = tall_furniture_mask(scene.fine, FINE_NAMES, scene.classes, scene.height, (100,), scene.grid, scene.upm)
    return resolve_builtin_fronts(
        scene.layers, [scene.room, _beyond_room(scene, 3.5, 4.9)], tall=tall, free_floor=scene.free, fine_classes=scene.fine, fine_names=FINE_NAMES,
        grid=scene.grid, units_per_metre=scene.upm,
    )


def _strip_extent(scene: Scene) -> tuple[float, float]:
    """Start and end of the A-EINB rectangle along the wall, in room metres."""
    rectangle = next(Polygon(entity["points"]) for entity in scene.layers["A-EINB"] if entity.get("type") == "polyline" and entity.get("closed"))
    cosine, sine = math.cos(math.radians(scene.angle)), math.sin(math.radians(scene.angle))
    along = [((x / scene.upm - 1.0) * cosine + (y / scene.upm - 1.0) * sine) for x, y in rectangle.exterior.coords]
    return min(along), max(along)


@pytest.mark.parametrize("upm,angle", CASES)
def test_stray_tall_pixels_beside_unit_do_not_stretch_run(upm: float, angle: float) -> None:
    # A thin shelf edge (0.08 m deep) stands 0.5 m beyond the end of a 3 m wardrobe: the unit keeps its length.
    scene = _scene(upm, angle, unit=(0.6, 3.6, ROOM_DEPTH - 0.45, ROOM_DEPTH, "wardrobe", 1.95))
    _add_tall_block(scene, 3.7, 4.2, ROOM_DEPTH - 0.45, ROOM_DEPTH - 0.37)
    assert _resolve_with_neighbour(scene)["builtin_fronts"] == 1
    start, end = _strip_extent(scene)
    assert end <= 3.6 + 0.12 and start >= 0.6 - 0.12


@pytest.mark.parametrize("upm,angle", CASES)
def test_full_depth_neighbour_unit_stays_in_run(upm: float, angle: float) -> None:
    # A second tall unit of full depth right beside the wardrobe belongs to the same built-in run.
    scene = _scene(upm, angle, unit=(0.6, 3.6, ROOM_DEPTH - 0.45, ROOM_DEPTH, "wardrobe", 1.95))
    _add_tall_block(scene, 3.6, 4.2, ROOM_DEPTH - 0.45, ROOM_DEPTH)
    assert _resolve_with_neighbour(scene)["builtin_fronts"] == 1
    assert _strip_extent(scene)[1] >= 4.2 - 0.12


def test_tall_mask_rules() -> None:
    scene = _scene(1.0, 0.0, unit=(0.6, 3.6, ROOM_DEPTH - 0.45, ROOM_DEPTH, "sideboard", 0.8))
    low = tall_furniture_mask(scene.fine, FINE_NAMES, scene.classes, scene.height, (100,), scene.grid, 1.0)
    assert not low.any()
    scene = _scene(1.0, 0.0, unit=(0.6, 3.6, ROOM_DEPTH - 0.45, ROOM_DEPTH, "wardrobe", 1.95))
    tall = tall_furniture_mask(scene.fine, FINE_NAMES, scene.classes, scene.height, (100,), scene.grid, 1.0)
    assert tall.sum() > 0.8 * np.count_nonzero(scene.classes == 117)
    # Class 117 above 1.6 m counts without any fine label; below it does not.
    scene.fine[:] = 0
    by_class = tall_furniture_mask(scene.fine, FINE_NAMES, scene.classes, scene.height, (100,), scene.grid, 1.0)
    assert by_class.sum() > 0.8 * np.count_nonzero(scene.classes == 117)
    assert not tall_furniture_mask(scene.fine, FINE_NAMES, scene.classes, scene.height * 0.6, (100,), scene.grid, 1.0).any()
