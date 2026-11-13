"""Recessed built-in niche: the wall follows the niche, the back wall comes from measured evidence.

The scenes are real rasters, wall pieces, room outlines and wall points in a room frame in metres
(a 5 x 3.5 m room whose back wall has an opening to a recess), rotated by a few degrees and scaled by
``units_per_metre``.  Results are written with the DXF writer and read back with ezdxf.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import uuid

import cv2
import ezdxf
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
from samcloud.dxf.writer import export_plan
from samcloud.ortho.vector.niche import NicheSettings, content_mask, resolve_recessed_niches
from samcloud.ortho.vector.raster_grid import RasterGrid

ROOM_WIDTH, ROOM_DEPTH = 5.0, 3.5
OPENING = (1.6, 2.8)
NICHE_DEPTH = 0.55
CASES = [(1.0, 0.0), (0.8157, 3.0)]


@pytest.fixture
def work_dir() -> Path:
    root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    path = Path(root) / f"niche_{uuid.uuid4().hex[:8]}" if root else Path(os.environ.get("TEMP", ".")) / f"niche_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Scene:
    upm: float
    grid: RasterGrid
    content: np.ndarray
    free: np.ndarray
    layers: dict[str, list[dict[str, Any]]]
    room: Polygon
    wall_points: np.ndarray | None
    transform: Any
    inverse: Any
    extras: dict[str, Any] = field(default_factory=dict)


def _make_transform(upm: float, angle_deg: float):
    cosine, sine = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))

    def transform(x: float, y: float) -> tuple[float, float]:
        return (upm * (cosine * x - sine * y) + 1.0 * upm, upm * (sine * x + cosine * y) + 1.0 * upm)

    def inverse(x: float, y: float) -> tuple[float, float]:
        dx, dy = x / upm - 1.0, y / upm - 1.0
        return (cosine * dx + sine * dy, -sine * dx + cosine * dy)

    return transform, inverse


def _fill(mask: np.ndarray, grid: RasterGrid, polygon: list[tuple[float, float]], transform: Any, value: Any = True) -> np.ndarray:
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
    upm: float, angle: float, *, seed: int = 11, niche: bool = True, contents: bool = True, wall_points: bool = False, panels: bool = True,
    stray: bool = False, flat_wardrobe: bool = False, doorway: bool = False, jog: str | None = None,
) -> Scene:
    rng = np.random.default_rng(seed)
    transform, inverse = _make_transform(upm, angle)
    corners = [transform(x, y) for x in (-1.0, ROOM_WIDTH + 1.0) for y in (-1.0, ROOM_DEPTH + 3.0)]
    pixel = 0.02 * upm
    min_x, max_y = min(c[0] for c in corners), max(c[1] for c in corners)
    width = int(math.ceil((max(c[0] for c in corners) - min_x) / pixel))
    height = int(math.ceil((max_y - min(c[1] for c in corners)) / pixel))
    grid = RasterGrid(min_x, max_y, pixel, width, height)
    heights = np.zeros((height, width), dtype=np.float32)
    free = np.zeros((height, width), dtype=bool)
    _fill(free, grid, _rect(0.1, ROOM_WIDTH - 0.1, 0.1, ROOM_DEPTH - 0.05), transform)
    layers: dict[str, list[dict[str, Any]]] = {name: [] for name in ("A-WALL", "A-EINB", "A-FURN", "A-FURN-DETL", "A-ANNO")}
    walls = layers["A-WALL"]
    walls.append(_wall(transform, rng, upm, (0.0, 0.0), (ROOM_WIDTH, 0.0)))
    walls.append(_wall(transform, rng, upm, (0.0, 0.0), (0.0, ROOM_DEPTH)))
    walls.append(_wall(transform, rng, upm, (ROOM_WIDTH, 0.0), (ROOM_WIDTH, ROOM_DEPTH)))
    x0, x1 = OPENING
    back = ROOM_DEPTH + NICHE_DEPTH
    points: list[tuple[float, float, float]] = []
    extras: dict[str, Any] = {"back_y": back}
    if niche or doorway or jog:
        walls.append(_wall(transform, rng, upm, (0.0, ROOM_DEPTH), (x0, ROOM_DEPTH)))
        walls.append(_wall(transform, rng, upm, (x1, ROOM_DEPTH), (ROOM_WIDTH, ROOM_DEPTH)))
    else:
        walls.append(_wall(transform, rng, upm, (0.0, ROOM_DEPTH), (ROOM_WIDTH, ROOM_DEPTH)))
    if niche:
        if panels:
            # The side walls run past the niche into the neighbouring volume.
            for x in (x0, x1):
                walls.append(_wall(transform, rng, upm, (x, ROOM_DEPTH), (x, back + 0.6)))
        if contents:
            area = _fill(np.zeros(heights.shape, dtype=bool), grid, _rect(x0 + 0.04, x1 - 0.04, ROOM_DEPTH + 0.02, back - 0.03), transform)
            heights[area] = (1.9 + rng.normal(0.0, 0.05, size=int(area.sum()))) * upm
            heights[area & (rng.random(heights.shape) < 0.12)] = 0.3 * upm
        if wall_points:
            # Only the corners of the back wall are visible above / between the contents.
            for lo, hi in ((x0 + 0.05, x0 + 0.45), (x1 - 0.45, x1 - 0.05)):
                for x in np.linspace(lo, hi, 40):
                    wx, wy = transform(x + rng.normal(0.0, 0.004), back + rng.normal(0.0, 0.004))
                    points.append((wx, wy, float(rng.uniform(1.6, 2.2)) * upm))
        if stray:
            region = _fill(np.zeros(heights.shape, dtype=bool), grid, _rect(x0 + 0.02, x1 - 0.02, back + 0.10, back + 0.50), transform)
            noise = region & (rng.random(heights.shape) < 0.06)
            heights[noise] = 2.0 * upm
            blob = [transform(x, y) for x, y in _rect(1.9, 2.1, back + 0.25, back + 0.40)]
            layers["A-FURN"].append({"type": "polyline", "points": [[float(px), float(py)] for px, py in blob], "closed": True, "holes": [], "_label": "Objekt"})
            first, second = transform(x0 + 0.2, back + 0.2), transform(x0 + 0.5, back + 0.2)
            walls.append({"type": "polyline", "points": [[first[0], first[1]], [second[0], second[1]]], "closed": False, "_source": "measured"})
            layers["A-ANNO"].append({"type": "text", "text": "Objekt h=0.60", "point": list(transform(2.0, back + 0.3)), "height": 0.08 * upm, "rotation": 0.0})
    if flat_wardrobe:
        area = _fill(np.zeros(heights.shape, dtype=bool), grid, _rect(1.2, 2.8, ROOM_DEPTH - 0.6, ROOM_DEPTH - 0.02), transform)
        heights[area] = 2.0 * upm
        free[area] = False
    if doorway:
        # The doorway leads into a second room: floor continues, a floor lamp stands behind it.
        _fill(free, grid, _rect(x0 + 0.1, x1 - 0.1, ROOM_DEPTH - 0.05, ROOM_DEPTH + 2.5), transform)
        _fill(free, grid, _rect(0.1, ROOM_WIDTH - 0.1, ROOM_DEPTH + 0.2, ROOM_DEPTH + 2.5), transform)
        for x in (x0, x1):
            walls.append(_wall(transform, rng, upm, (x, ROOM_DEPTH), (x, ROOM_DEPTH + 0.2)))
        lamp = _fill(np.zeros(heights.shape, dtype=bool), grid, _rect(3.5, 3.75, ROOM_DEPTH + 1.0, ROOM_DEPTH + 1.25), transform)
        heights[lamp] = 1.8 * upm
        free[lamp] = False
    outline = _rect(0.0, ROOM_WIDTH, 0.0, ROOM_DEPTH)
    if jog:
        # A jog of the room outline (a pocket beside a corridor wall): measured back wall and a wall stub across it,
        # tall objects inside; the side edges come from the ceiling outline only ("outline") or are measured walls ("walls").
        if jog == "room_edges":
            # Ceiling-derived outline spurs run into the pocket: they are room edges, not measured walls.
            outline = [(0.0, 0.0), (ROOM_WIDTH, 0.0), (ROOM_WIDTH, ROOM_DEPTH), (x1 + 0.02, ROOM_DEPTH), (x1 + 0.02, back - 0.1), (x1, back - 0.1), (x1, ROOM_DEPTH), (x0 + 0.02, ROOM_DEPTH), (x0 + 0.02, back - 0.1), (x0, back - 0.1), (x0, ROOM_DEPTH), (0.0, ROOM_DEPTH)]
        walls.append(_wall(transform, rng, upm, (x0 - 0.1, back), (x1 + 0.1, back)))
        if jog != "room_edges":
            walls.append(_wall(transform, rng, upm, ((x0 + x1) / 2.0, ROOM_DEPTH + 0.02), ((x0 + x1) / 2.0, back)))
        if jog == "walls":
            for x in (x0, x1):
                walls.append(_wall(transform, rng, upm, (x, ROOM_DEPTH), (x, back + 0.3)))
        area = _fill(np.zeros(heights.shape, dtype=bool), grid, _rect(x0 + 0.04, x1 - 0.04, ROOM_DEPTH + 0.02, back - 0.05), transform)
        heights[area] = 2.0 * upm
    room = Polygon([transform(x, y) for x, y in outline])
    wall_array = np.asarray(points, dtype=float) if points else None
    content = content_mask(np.zeros(heights.shape, dtype=bool), heights, upm)
    return Scene(upm, grid, content, free, layers, room, wall_array, transform, inverse, extras)


def _resolve(scene: Scene) -> dict[str, Any]:
    return resolve_recessed_niches(
        scene.layers, [scene.room], content=scene.content, free_floor=scene.free, fine_classes=np.zeros(scene.content.shape, dtype=np.uint8), fine_names=[],
        grid=scene.grid, units_per_metre=scene.upm, wall_points=scene.wall_points,
    )


def _export(scene: Scene, work_dir: Path) -> ezdxf.document.Drawing:
    frame = Frame(np.zeros(3, dtype=np.float64), 1.0, np.eye(3, dtype=np.float64))
    layers = {name: items for name, items in scene.layers.items()}
    result = DxfPlanResult(
        project_name="Niche", level_name="EG", mode="indoor", profile_name="test", scale_factor=1.0, frame=frame, settings_fingerprint="x",
        layers=layers, statuses={name: LayerStatus(name, "ok", None, len(items)) for name, items in layers.items()}, layer_colors={name: 7 for name in layers},
        background=BackgroundImage(None, None, (0.0, 0.0, 20.0, 20.0)), layer_linetypes={}, layer_styles={}, blocks=[], variant="ortho",
    )
    return ezdxf.readfile(export_plan(result, work_dir, coordinates="local"))


def _room_frame_polylines(scene: Scene, document: ezdxf.document.Drawing, layer: str) -> list[tuple[list[tuple[float, float]], bool]]:
    result = []
    for entity in document.modelspace().query("LWPOLYLINE"):
        if entity.dxf.layer == layer:
            result.append(([scene.inverse(point[0], point[1]) for point in entity.get_points("xy")], bool(entity.closed)))
    return result


def _texts(document: ezdxf.document.Drawing, layer: str) -> list[str]:
    return [entity.dxf.text for entity in document.modelspace().query("TEXT") if entity.dxf.layer == layer]


def _covered(lines: list[LineString], reference: LineString, tolerance: float) -> float:
    samples = [reference.interpolate(fraction, normalized=True) for fraction in np.linspace(0.0, 1.0, 300)]
    return float(np.mean([any(line.distance(point) <= tolerance for line in lines) for point in samples]))


def _assert_closed_niche(scene: Scene, document: ezdxf.document.Drawing, depth_expected: float, tolerance: float) -> None:
    """Walls run opening edge -> back wall -> opening edge; A-EINB fills the recess."""
    x0, x1 = OPENING
    walls = [LineString(points) for points, _ in _room_frame_polylines(scene, document, "A-WALL") if len(points) >= 2]
    for reference in (
        LineString([(x0, ROOM_DEPTH), (x0, ROOM_DEPTH + depth_expected)]),
        LineString([(x1, ROOM_DEPTH), (x1, ROOM_DEPTH + depth_expected)]),
        LineString([(x0, ROOM_DEPTH + depth_expected), (x1, ROOM_DEPTH + depth_expected)]),
    ):
        assert _covered(walls, reference, tolerance) >= 0.97
    # Orthogonal to the main wall: every niche wall piece is axis aligned in the room frame.
    for line in walls:
        (xa, ya), (xb, yb) = line.coords[0], line.coords[-1]
        if min(ya, yb) >= ROOM_DEPTH - 1e-6 and line.length > 0.1 * scene.upm and x0 - 0.05 <= xa <= x1 + 0.05 and x0 - 0.05 <= xb <= x1 + 0.05:
            assert min(abs(xb - xa), abs(yb - ya)) <= 0.01 * scene.upm + 0.02 * line.length
    # No wall along the opening itself.
    opening = LineString([(x0 + 0.1, ROOM_DEPTH), (x1 - 0.1, ROOM_DEPTH)])
    assert _covered(walls, opening, 0.03 * scene.upm) <= 0.05
    einb = [Polygon(points) for points, closed in _room_frame_polylines(scene, document, "A-EINB") if closed and len(points) >= 3]
    niche = Polygon(_rect(x0, x1, ROOM_DEPTH, ROOM_DEPTH + depth_expected))
    assert len(einb) == 1
    assert einb[0].symmetric_difference(niche).area <= 0.12 * niche.area
    assert any(text in {"Garderobe", "Einbauschrank"} for text in _texts(document, "A-EINB"))


@pytest.mark.parametrize("upm,angle", CASES)
def test_niche_back_wall_from_visible_corners(upm: float, angle: float, work_dir: Path) -> None:
    scene = _scene(upm, angle, wall_points=True)
    stats = _resolve(scene)
    assert stats["niche_count"] == 1
    niche = stats["niches"][0]
    assert niche["evidence"] == "wall_points" and not niche["no_back_wall_seen"]
    assert abs(niche["depth_m"] - NICHE_DEPTH) <= 0.03
    assert abs(niche["opening_m"] - (OPENING[1] - OPENING[0])) <= 0.06
    document = _export(scene, work_dir)
    _assert_closed_niche(scene, document, NICHE_DEPTH, 0.03 * upm)
    # The back wall is at the true depth within 3 cm.
    walls = [LineString(points) for points, _ in _room_frame_polylines(scene, document, "A-WALL")]
    back_pieces = [line for line in walls if line.length > 0.5 * upm and abs(line.coords[0][1] - line.coords[-1][1]) < 0.02 * upm and line.coords[0][1] > ROOM_DEPTH + 0.2 * upm]
    assert back_pieces
    assert all(abs(line.coords[0][1] - (ROOM_DEPTH + NICHE_DEPTH)) <= 0.03 * upm for line in back_pieces)


@pytest.mark.parametrize("upm,angle", CASES)
def test_niche_from_contents_only_is_closed_and_flagged(upm: float, angle: float, work_dir: Path) -> None:
    scene = _scene(upm, angle)
    stats = _resolve(scene)
    assert stats["niche_count"] == 1
    niche = stats["niches"][0]
    assert niche["evidence"] == "contents" and niche["no_back_wall_seen"] is True
    # Closed at the far boundary of the contents (they stop 3 cm in front of the true back wall).
    expected = NICHE_DEPTH - 0.03
    assert abs(niche["depth_m"] - expected) <= 0.03
    _assert_closed_niche(scene, _export(scene, work_dir), expected, 0.03 * upm)


@pytest.mark.parametrize("wall_points", [False, True], ids=["contents_only", "with_wall_points"])
@pytest.mark.parametrize("upm,angle", CASES)
def test_stray_points_behind_the_back_wall_are_dropped(upm: float, angle: float, wall_points: bool, work_dir: Path) -> None:
    scene = _scene(upm, angle, wall_points=wall_points, stray=True)
    stats = _resolve(scene)
    assert stats["niche_count"] == 1
    niche = stats["niches"][0]
    assert abs(niche["depth_m"] - (NICHE_DEPTH if wall_points else NICHE_DEPTH - 0.03)) <= 0.03
    assert niche["removed_beyond"] >= 3
    document = _export(scene, work_dir)
    x0, x1 = OPENING
    limit = ROOM_DEPTH + NICHE_DEPTH + 0.06
    inside = Polygon(_rect(x0 + 0.06, x1 - 0.06, limit, limit + 1.0))
    for layer in ("A-FURN", "A-FURN-DETL", "A-WALL", "A-EINB"):
        for points, closed in _room_frame_polylines(scene, document, layer):
            geometry = Polygon(points) if closed and len(points) >= 3 else LineString(points)
            assert geometry.intersection(inside).is_empty or geometry.intersection(inside).length + geometry.intersection(inside).area < 1e-6, layer
    assert not any("Objekt" in text for text in _texts(document, "A-ANNO"))


@pytest.mark.parametrize("upm,angle", CASES)
def test_wardrobe_in_front_of_a_flat_wall_is_unchanged(upm: float, angle: float) -> None:
    scene = _scene(upm, angle, niche=False, flat_wardrobe=True)
    before = copy.deepcopy(scene.layers)
    stats = _resolve(scene)
    assert stats["niche_count"] == 0 and stats["strips"] == []
    assert scene.layers == before


@pytest.mark.parametrize("upm,angle", CASES)
def test_doorway_with_floor_beyond_is_not_a_niche(upm: float, angle: float) -> None:
    scene = _scene(upm, angle, niche=False, doorway=True)
    before = copy.deepcopy(scene.layers)
    stats = _resolve(scene)
    assert stats["niche_count"] == 0
    assert scene.layers == before


@pytest.mark.parametrize("jog", ["outline", "room_edges", "walls"])
@pytest.mark.parametrize("upm,angle", CASES)
def test_wall_jog_with_tall_objects_is_not_a_niche(upm: float, angle: float, jog: str) -> None:
    # Pocket whose side edges are only the room outline (or whose interior is cut by a measured wall stub).
    scene = _scene(upm, angle, niche=False, jog=jog)
    before = copy.deepcopy(scene.layers)
    stats = _resolve(scene)
    assert stats["niche_count"] == 0 and stats["strips"] == []
    assert scene.layers == before


def test_niche_needs_a_side_panel() -> None:
    # Contents behind an opening that no wall stub closes are left to the built-in front logic.
    scene = _scene(1.0, 0.0, panels=False)
    stats = _resolve(scene)
    assert stats["niche_count"] == 0
    settings = NicheSettings(require_panel=False)
    scene = _scene(1.0, 0.0, panels=False)
    stats = resolve_recessed_niches(
        scene.layers, [scene.room], content=scene.content, free_floor=scene.free, fine_classes=np.zeros(scene.content.shape, dtype=np.uint8), fine_names=[],
        grid=scene.grid, units_per_metre=1.0, settings=settings,
    )
    assert stats["niche_count"] == 1
