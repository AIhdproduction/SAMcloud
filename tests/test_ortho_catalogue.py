"""Indoor interior-architecture catalogue: symbols, wall snapping, islands, upper cabinets, room names, layers.

All tests build real geometry / real DXF documents and read the written DXF back with ezdxf.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import uuid

import ezdxf
from ezdxf import bbox as ezbbox
import numpy as np
import pytest
from shapely import contains_xy
from shapely.geometry import LineString, Polygon, box

from samcloud.dxf.las_source import Frame
from samcloud.dxf.layers import load_layer_config
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
from samcloud.dxf.writer import build_dxf_document, export_plan
from samcloud.ortho.plan import (
    _add_ortho_plan_layer_presentation,
    _add_room_labels,
    _catalogue_fixture,
    _collect_room_tags,
    _layer_presentation,
)
from samcloud.ortho.vector.footprints import IslandSettings, ObjectInstance, detect_kitchen_island
from samcloud.ortho.vector.indoor import centred_catalogue_insert, fit_block_insert, match_catalogue, snap_insert_to_wall
from samcloud.ortho.vector.raster_grid import RasterGrid
from samcloud.ortho.vector.room_names import RoomNameSettings, classify_room, name_rooms, room_label_text, tag_for_name
from samcloud.ortho.vector.symbols import CATALOGUE, SYMBOLS, add_symbol_blocks, catalogue_entry_for_block, symbol_extents
from samcloud.ortho.vector.upper_cabinets import detect_upper_cabinets, upper_band_mask

UPM_V8 = 0.8157


@pytest.fixture
def work_dir(request: pytest.FixtureRequest) -> Path:
    root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if root:
        path = Path(root) / f"catalogue_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("catalogue")


def _result(layers: dict[str, list[dict]], *, blocks: list[str] | None = None, styles: dict | None = None, colors: dict | None = None, linetypes: dict | None = None) -> DxfPlanResult:
    frame = Frame(np.zeros(3, dtype=np.float64), 1.0, np.eye(3, dtype=np.float64))
    return DxfPlanResult(
        project_name="Cat", level_name="EG", mode="indoor", profile_name="test", scale_factor=1.0, frame=frame,
        settings_fingerprint="x", layers=layers, statuses={name: LayerStatus(name, "ok", None, len(items)) for name, items in layers.items()},
        layer_colors=colors if colors is not None else {name: 7 for name in layers},
        background=BackgroundImage(None, None, (0.0, 0.0, 20.0, 20.0)),
        layer_linetypes=linetypes or {}, layer_styles=styles or {}, blocks=blocks if blocks is not None else sorted({str(e["name"]) for items in layers.values() for e in items if e.get("type") == "insert"}),
        variant="ortho",
    )


def _world(insert: dict, x: float, y: float) -> np.ndarray:
    angle = math.radians(float(insert["rotation"]))
    scale = float(insert["xscale"])
    px, py = insert["point"]
    return np.asarray((px + scale * (x * math.cos(angle) - y * math.sin(angle)), py + scale * (x * math.sin(angle) + y * math.cos(angle))))


def _block_box(document: ezdxf.document.Drawing, name: str) -> tuple[float, float, float, float]:
    entities = [entity for entity in document.blocks[name] if entity.dxftype() in {"LINE", "LWPOLYLINE", "CIRCLE", "ARC"}]
    extents = ezbbox.extents(entities)
    return extents.extmin.x, extents.extmin.y, extents.extmax.x, extents.extmax.y


# 1. standard catalogue dimensions -------------------------------------------------

EXPECTED_SIZES = {
    "WC_WALL": [(0.40, 0.56)],
    "WASHBASIN": [(0.60, 0.48), (0.80, 0.50)],
    "BATHTUB": [(1.70, 0.75), (1.80, 0.80)],
    "SHOWER": [(0.90, 0.90), (1.00, 1.00), (1.20, 0.90)],
    "COOKTOP": [(0.58, 0.51), (0.80, 0.52)],
    "KITCHEN_SINK": [(0.86, 0.50), (1.00, 0.50)],
    "BED_SINGLE": [(0.90, 2.00), (1.00, 2.00)],
    "BED_DOUBLE": [(1.60, 2.00), (1.80, 2.00)],
}


def test_catalogue_blocks_have_standard_metric_dimensions_and_wall_origin(work_dir: Path) -> None:
    document = ezdxf.new("R2018")
    add_symbol_blocks(document)
    path = work_dir / "catalogue.dxf"
    document.saveas(path)
    reopened = ezdxf.readfile(path)
    for kind, sizes in EXPECTED_SIZES.items():
        assert [(entry.width_m, entry.depth_m) for entry in CATALOGUE[kind]] == sizes
        for entry in CATALOGUE[kind]:
            xmin, ymin, xmax, ymax = _block_box(reopened, entry.block)
            # Dimensions in metres, origin centred on the wall-mounting edge, +Y into the room.
            assert xmax - xmin == pytest.approx(entry.width_m, abs=1e-6)
            assert ymax - ymin == pytest.approx(entry.depth_m, abs=1e-6)
            assert xmin == pytest.approx(-entry.width_m / 2.0, abs=1e-6)
            assert ymin == pytest.approx(0.0, abs=1e-6)
    # The bare type name is the default (first) size.
    assert _block_box(reopened, "BATHTUB") == pytest.approx(_block_box(reopened, "BATHTUB_1700X750"))
    # Kitchen island and furniture catalogue sizes.
    assert {(e.width_m, e.depth_m) for e in CATALOGUE["KITCHEN_ISLAND"]} == {(1.80, 0.90), (2.40, 1.20)}
    assert all(e.depth_m == 0.60 for e in CATALOGUE["WARDROBE"])
    table = _block_box(reopened, "TABLE_DINING_1600X900")
    assert table[2] - table[0] == pytest.approx(1.60) and table[3] - table[1] > 0.90  # table plus chair outlines
    # Sanitary blocks carry real detail: tub drain circle, shower diagonal cross, basin ellipse.
    assert any(entity.dxftype() == "CIRCLE" for entity in reopened.blocks["BATHTUB_1700X750"])
    assert sum(entity.dxftype() == "LINE" for entity in reopened.blocks["SHOWER_900X900"]) == 2
    assert sum(entity.dxftype() == "LWPOLYLINE" for entity in reopened.blocks["WASHBASIN_600X480"]) >= 2
    assert catalogue_entry_for_block("cooktop_800x520").kind == "COOKTOP"
    assert symbol_extents("WC_WALL")[3] == pytest.approx(0.56)


def test_match_catalogue_picks_nearest_standard_size() -> None:
    assert match_catalogue("BATHTUB", 1.78, 0.79).entry.width_m == 1.80
    assert match_catalogue("BATHTUB", 1.66, 0.74).entry.width_m == 1.70
    assert match_catalogue("COOKTOP", 1.9, 0.5, tolerance=0.3) is None  # a whole kitchen run is not a cooktop
    assert match_catalogue(("BED_SINGLE", "BED_DOUBLE"), 1.62, 1.97).entry.block == "BED_DOUBLE_1600X2000"


# 2. kitchen island ------------------------------------------------------------------


def _instance(grid: RasterGrid, polygon: Polygon, label: str | None, height_m: float, upm: float) -> ObjectInstance:
    cols, rows = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    xs = grid.min_x + (cols + 0.5) * grid.pixel_size
    ys = grid.max_y - (rows + 0.5) * grid.pixel_size
    mask = contains_xy(polygon, xs, ys)
    return ObjectInstance(mask, label, 117, label is not None, height_m * upm, 1.0)


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_freestanding_kitchen_island_without_backing_wall(upm: float) -> None:
    grid = RasterGrid(0.0, 10.0 * upm, 0.04 * upm, 250, 250)
    theta = math.radians(15.0)
    island_rect = box(-1.0 * upm, -0.46 * upm, 1.0 * upm, 0.46 * upm)
    from shapely import affinity
    island_rect = affinity.translate(affinity.rotate(island_rect, 15.0, origin=(0, 0)), 5.0 * upm, 5.0 * upm)
    walls = [LineString([(0.5 * upm, 0.5 * upm), (9.5 * upm, 0.5 * upm)])]  # the nearest wall is 4 m away
    zone = box(0.0, 0.0, 10.0 * upm, 10.0 * upm)
    island = detect_kitchen_island(_instance(grid, island_rect, None, 0.90, upm), grid, theta, upm, walls, zone)
    assert island is not None and island.reason == "kitchen_zone"
    assert island.depth_m == 0.90 and 1.9 <= island.length_m <= 2.1
    assert island.polygon.area / upm**2 == pytest.approx(island.length_m * island.depth_m, rel=1e-6)
    assert abs(((island.angle_rad - theta) + math.pi / 2) % math.pi - math.pi / 2) < math.radians(0.5)  # aligned to the room axis
    corners = list(island.polygon.exterior.coords)[:4]
    edge = np.asarray(corners[1]) - np.asarray(corners[0])
    assert abs(math.degrees(math.atan2(edge[1], edge[0])) - 15.0) % 90.0 < 0.5 or abs(math.degrees(math.atan2(edge[1], edge[0])) - 15.0) % 90.0 > 89.5
    # A kitchen fine label supports the island outside any kitchen zone as well.
    assert detect_kitchen_island(_instance(grid, island_rect, "kitchen counter", 0.90, upm), grid, theta, upm, walls, None) is not None
    # Without semantics (no label, no kitchen zone) it is not an island.
    assert detect_kitchen_island(_instance(grid, island_rect, None, 0.90, upm), grid, theta, upm, walls, None) is None


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_coffee_table_sofa_and_wall_counter_are_not_islands(upm: float) -> None:
    grid = RasterGrid(0.0, 10.0 * upm, 0.04 * upm, 250, 250)
    zone = box(0.0, 0.0, 10.0 * upm, 10.0 * upm)
    far_wall = [LineString([(0.5 * upm, 0.5 * upm), (9.5 * upm, 0.5 * upm)])]
    coffee = box(4.0 * upm, 4.0 * upm, 5.0 * upm, 4.5 * upm)
    assert detect_kitchen_island(_instance(grid, coffee, None, 0.45, upm), grid, 0.0, upm, far_wall, zone) is None  # low and small
    sofa = box(3.0 * upm, 4.0 * upm, 5.0 * upm, 4.9 * upm)
    assert detect_kitchen_island(_instance(grid, sofa, "sofa", 0.90, upm), grid, 0.0, upm, far_wall, zone) is None  # island size/height, wrong label
    unlabelled_low = box(3.0 * upm, 4.0 * upm, 5.0 * upm, 4.9 * upm)
    assert detect_kitchen_island(_instance(grid, unlabelled_low, None, 0.45, upm), grid, 0.0, upm, far_wall, zone) is None
    counter = box(3.0 * upm, 4.0 * upm, 5.0 * upm, 4.9 * upm)
    backed = [LineString([(2.0 * upm, 4.92 * upm), (6.0 * upm, 4.92 * upm)])]  # wall touches the long side
    assert detect_kitchen_island(_instance(grid, counter, None, 0.90, upm), grid, 0.0, upm, backed, zone) is None
    shallow = box(3.0 * upm, 4.0 * upm, 5.0 * upm, 4.6 * upm)  # 0.6 m deep wall counter depth
    assert detect_kitchen_island(_instance(grid, shallow, "kitchen counter", 0.90, upm), grid, 0.0, upm, far_wall, zone) is None
    assert IslandSettings().min_length_m == 1.20


# 3. wall snapping and orientation --------------------------------------------------

ROOM = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
WALLS = {
    "left": LineString([(0.0, 0.0), (0.0, 4.0)]),
    "right": LineString([(4.0, 0.0), (4.0, 4.0)]),
    "bottom": LineString([(0.0, 0.0), (4.0, 0.0)]),
    "top": LineString([(0.0, 4.0), (4.0, 4.0)]),
}
# wall name -> (inward normal, along-wall axis angle in degrees)
WALL_NORMALS = {"left": (1.0, 0.0), "right": (-1.0, 0.0), "bottom": (0.0, 1.0), "top": (0.0, -1.0)}


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
@pytest.mark.parametrize("wall_name", ["left", "right", "bottom", "top"])
@pytest.mark.parametrize("fixture, kinds, along_m, depth_m", [("WC", "WC_WALL", 0.40, 0.56), ("WASHBASIN", "WASHBASIN", 0.60, 0.48)])
def test_wall_mounted_fixture_snaps_flush_and_faces_into_room(wall_name: str, fixture: str, kinds: str, along_m: float, depth_m: float, upm: float) -> None:
    normal = np.asarray(WALL_NORMALS[wall_name], dtype=float) * 1.0
    wall = WALLS[wall_name]
    scaled_wall = LineString([(x * upm, y * upm) for x, y in wall.coords])
    wall_point = np.asarray(scaled_wall.interpolate(0.5, normalized=True).coords[0])
    along_axis = np.asarray((-normal[1], normal[0]))
    gap = 0.04 * upm  # measured rear edge floats 4 cm off the wall
    centre = wall_point + normal * (gap + depth_m * upm / 2.0) + along_axis * 0.3 * upm
    # Measured footprint, long axis perpendicular to the wall (like a real WC cluster): rotation of the long axis.
    long_axis = normal if depth_m >= along_m else along_axis
    measured = {"type": "insert", "name": "FURNITURE", "point": centre.tolist(), "rotation": math.degrees(math.atan2(long_axis[1], long_axis[0])) % 180.0,
                "xscale": max(along_m, depth_m) * upm * 1.03, "yscale": min(along_m, depth_m) * upm * 0.97, "attributes": {"LABEL": "x"}}
    placed = _catalogue_fixture(fixture, measured, [scaled_wall], upm)
    assert placed is not None and placed["name"].startswith(kinds)
    assert placed["xscale"] == placed["yscale"] == pytest.approx(upm)
    origin = _world(placed, 0.0, 0.0)
    # The mounting edge lies exactly on the wall line ...
    assert scaled_wall.distance(__import__("shapely").geometry.Point(*origin)) < 1e-9
    # ... along-wall axis (block +X) is parallel to the wall and +Y (the bowl) points into the room.
    block_x = _world(placed, 1.0, 0.0) - origin
    block_y = _world(placed, 0.0, 1.0) - origin
    assert abs(float(np.dot(block_x / upm, normal))) < 1e-9
    assert float(np.dot(block_y / upm, normal)) == pytest.approx(1.0, abs=1e-9)
    tip = _world(placed, 0.0, depth_m)
    assert scaled_wall.distance(__import__("shapely").geometry.Point(*tip)) == pytest.approx(depth_m * upm, abs=1e-9)
    assert Polygon([(x * upm, y * upm) for x, y in ROOM.exterior.coords]).contains(__import__("shapely").geometry.Point(*tip))
    # The block footprint corners all lie on the room side of the wall.
    for x, y in ((-along_m / 2, 0.0), (along_m / 2, 0.0), (-along_m / 2, depth_m), (along_m / 2, depth_m)):
        assert float(np.dot(_world(placed, x, y) - wall_point, normal)) >= -1e-9


def test_fixture_far_from_any_wall_is_not_wall_snapped() -> None:
    measured = {"type": "insert", "name": "FURNITURE", "point": [2.0, 2.0], "rotation": 0.0, "xscale": 1.7, "yscale": 0.75, "attributes": {}}
    assert snap_insert_to_wall(measured, "BATHTUB", [WALLS["left"]], 1.0) is None
    free = centred_catalogue_insert(measured, "BATHTUB", 1.0)
    assert free is not None and free["_catalogue"]["wall_snapped"] is False and free["xscale"] == free["yscale"] == 1.0
    # Footprint centre of the free-standing block equals the measured centre.
    centre = _world(free, 0.0, 0.75 / 2.0)
    assert np.allclose(centre, (2.0, 2.0), atol=1e-9)


# 4./6. uniform scale in the written DXF and catalogue size at upm 0.8157 ---------------


def _tub_mask(upm: float) -> tuple[np.ndarray, RasterGrid]:
    grid = RasterGrid(0.0, 6.0, 0.02, 300, 300)
    mask = np.zeros((300, 300), dtype=bool)
    # 1.70 x 0.75 m tub (in catalogue metres) scaled by the project units per metre.
    cols, rows = np.meshgrid(np.arange(300), np.arange(300))
    xs, ys = (cols + 0.5) * 0.02, 6.0 - (rows + 0.5) * 0.02
    mask = (xs >= 1.0) & (xs <= 1.0 + 1.70 * upm) & (ys >= 2.0) & (ys <= 2.0 + 0.75 * upm)
    return mask, grid


def test_inserts_are_not_distorted_and_use_catalogue_size_at_project_scale(work_dir: Path) -> None:
    mask, grid = _tub_mask(UPM_V8)
    insert = fit_block_insert(mask, grid, "BATHTUB", "Badewanne", units_per_metre=UPM_V8)
    assert insert["name"] == "BATHTUB_1700X750"
    assert insert["xscale"] == insert["yscale"] == pytest.approx(UPM_V8)  # no independent x/y stretching
    wc = centred_catalogue_insert({"point": [1.0, 1.0], "rotation": 30.0, "xscale": 0.56 * UPM_V8, "yscale": 0.40 * UPM_V8, "attributes": {}}, "WC_WALL", UPM_V8)
    sink = centred_catalogue_insert({"point": [3.0, 1.0], "rotation": 0.0, "xscale": 1.0 * UPM_V8, "yscale": 0.5 * UPM_V8, "attributes": {}}, "KITCHEN_SINK", UPM_V8)
    shower = centred_catalogue_insert({"point": [5.0, 1.0], "rotation": 90.0, "xscale": 0.9 * UPM_V8, "yscale": 0.9 * UPM_V8, "attributes": {}}, "SHOWER", UPM_V8)
    layers = {"P-SANI": [insert, wc, shower], "A-EINB": [sink]}
    path = export_plan(_result(layers), work_dir, coordinates="local")
    document = ezdxf.readfile(path)
    inserts = [entity for entity in document.modelspace() if entity.dxftype() == "INSERT"]
    assert len(inserts) == 4
    for reference in inserts:
        assert reference.dxf.xscale == pytest.approx(reference.dxf.yscale)  # uniform
        xmin, ymin, xmax, ymax = _block_box(document, reference.dxf.name)
        entry = catalogue_entry_for_block(reference.dxf.name)
        # Aspect ratio of the drawn block equals the catalogue ratio.
        assert (xmax - xmin) / (ymax - ymin) == pytest.approx(entry.width_m / entry.depth_m, rel=1e-6)
    tub = next(reference for reference in inserts if reference.dxf.name == "BATHTUB_1700X750")
    xmin, _ymin, xmax, _ymax = _block_box(document, "BATHTUB_1700X750")
    assert (xmax - xmin) * tub.dxf.xscale == pytest.approx(1.70 * UPM_V8, abs=1e-3)
    assert (_ymax - _ymin) * tub.dxf.yscale == pytest.approx(0.75 * UPM_V8, abs=1e-3)
    # Written to the layer that matches the symbol.
    assert {reference.dxf.layer for reference in inserts if reference.dxf.name.startswith("KITCHEN_SINK")} == {"A-EINB"}


def test_unit_blocks_without_catalogue_keep_legacy_measured_scales() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 30:70] = True
    insert = fit_block_insert(mask, RasterGrid(-1.0, 3.0, 0.05, 100, 100), "COLUMN", "Column", units_per_metre=UPM_V8)
    assert insert["name"] == "COLUMN" and insert["xscale"] != insert["yscale"]
    assert SYMBOLS["COLUMN"]


# 5. upper cabinets on A-OBEN -------------------------------------------------------------


def _kitchen_scene(upm: float, with_upper: bool, tall_unit: bool = False):
    grid = RasterGrid(0.0, 8.0 * upm, 0.025 * upm, 320, 320)
    wall = LineString([(1.0 * upm, 1.0 * upm), (7.0 * upm, 1.0 * upm)])
    base = box(1.5 * upm, 1.0 * upm, 5.5 * upm, 1.6 * upm)  # 4.0 x 0.6 m run against the wall (y = 1.0)
    height = 2.0305
    bits = np.zeros((320, 320), dtype=np.uint32)
    full = np.zeros((320, 320), dtype=bool)
    wall_evidence = np.zeros((320, 320), dtype=bool)
    row_wall = int((8.0 * upm - 1.0 * upm) / (0.025 * upm))
    wall_evidence[row_wall - 1:row_wall + 2, :] = True
    if with_upper:
        # Front face of the hanging units at 0.35 m depth: occupancy in the 1.80-2.20 m bins only, over x = 2.0 ... 5.0 m.
        row = int((8.0 * upm - 1.35 * upm) / (0.025 * upm))
        c0, c1 = int(2.0 * upm / (0.025 * upm)), int(5.0 * upm / (0.025 * upm))
        bin_index = int(2.0 * upm / height * 16)
        bits[row - 1:row + 2, c0:c1] = np.uint32(1) << np.uint32(bin_index)
        if tall_unit:
            full[row - 1:row + 2, c0:c1] = True
    return grid, wall, base, bits, full, wall_evidence, height


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_upper_cabinets_detected_only_with_upper_band_evidence(upm: float, work_dir: Path) -> None:
    results = {}
    for flag in (True, False):
        grid, wall, base, bits, full, wall_evidence, height = _kitchen_scene(upm, flag)
        mask = upper_band_mask(bits, height, upm)
        assert bool(mask.any()) == flag
        results[flag] = detect_upper_cabinets([base], mask, full, wall_evidence, grid, [wall], upm)
    assert len(results[False]) == 0
    assert len(results[True]) == 1
    cabinet = results[True][0]
    assert cabinet.bounds[2] - cabinet.bounds[0] == pytest.approx(3.0 * upm, abs=0.12 * upm)  # 3.0 m of hanging units
    assert cabinet.bounds[3] - cabinet.bounds[1] == pytest.approx(0.35 * upm, abs=1e-6)  # standard 0.35 m depth, from the wall
    assert cabinet.bounds[1] == pytest.approx(1.0 * upm, abs=1e-6) or cabinet.bounds[3] == pytest.approx(1.0 * upm, abs=1e-6)
    # Floor-connected (tall) units are not upper cabinets.
    grid, wall, base, bits, full, wall_evidence, height = _kitchen_scene(upm, True, tall_unit=True)
    assert detect_upper_cabinets([base], upper_band_mask(bits, height, upm), full, wall_evidence, grid, [wall], upm) == []
    # Written DXF: A-OBEN dashed outline only when upper cabinets exist; base run stays solid on A-EINB.
    presentation = _presentation()
    for flag in (True, False):
        layers = {"A-EINB": [{"type": "polyline", "points": [list(point) for point in list(base.exterior.coords)[:-1]], "closed": True, "holes": []}], "A-OBEN": [{"type": "polyline", "points": [list(point) for point in list(item.exterior.coords)[:-1]], "closed": True, "holes": []} for item in results[flag]]}
        path = export_plan(_result(layers, styles=presentation[0], colors=presentation[1], linetypes=presentation[2]), work_dir / f"oben_{flag}_{upm}", coordinates="local")
        document = ezdxf.readfile(path)
        oben = [entity for entity in document.modelspace() if entity.dxf.layer == "A-OBEN"]
        assert len(oben) == (1 if flag else 0)
        assert document.layers.get("A-OBEN").dxf.linetype == "DASHED"
        assert [entity for entity in document.modelspace() if entity.dxf.layer == "A-EINB" and entity.dxftype() == "LWPOLYLINE"]
        assert document.layers.get("A-EINB").dxf.linetype == "CONTINUOUS"


def test_upper_band_mask_uses_one_point_eight_to_two_point_two_metres() -> None:
    height, upm = 2.0305, UPM_V8
    def bit_for(z_m: float) -> np.ndarray:
        return np.asarray([[np.uint32(1) << np.uint32(int(z_m * upm / height * 16))]], dtype=np.uint32)
    assert upper_band_mask(bit_for(2.0), height, upm)[0, 0]
    assert upper_band_mask(bit_for(1.9), height, upm)[0, 0]
    assert not upper_band_mask(bit_for(1.2), height, upm)[0, 0]
    assert not upper_band_mask(bit_for(2.4), height, upm)[0, 0]


# 7. semantic room names ------------------------------------------------------------------


def test_semantic_room_names_and_text_format() -> None:
    assert classify_room({"toilet", "washbasin", "bathtub"}, 7.8) == "Bad"
    assert classify_room({"toilet", "washbasin"}, 1.8) == "Gäste-WC"
    assert classify_room({"toilet", "washbasin"}, 6.0) == "Bad"
    assert classify_room({"shower"}, 4.0) == "Bad"
    assert classify_room({"kitchen"}, 12.4) == "Küche"
    assert classify_room({"kitchen", "living"}, 30.0) == "Wohnen / Küche"
    assert classify_room({"bed", "wardrobe"}, 14.0) == "Schlafen"
    assert classify_room({"wardrobe"}, 9.0) == "Zimmer"
    assert classify_room({"living"}, 20.0) == "Wohnen"
    assert classify_room(set(), 20.0) is None
    assert room_label_text("Küche", 12.4) == "Küche  12.4 m²"
    assert room_label_text("Bad", 7.8) == "Bad  7.8 m²"
    assert tag_for_name("WC_WALL_400X560") == "toilet" and tag_for_name("COOKTOP_800X520") == "kitchen" and tag_for_name("Sofa h=0.83") == "living"
    assert tag_for_name("BED_DOUBLE_1800X2000") == "bed" and tag_for_name("Nachttisch") is None


def test_rooms_are_named_from_inserted_fixtures_and_tiny_rooms_dropped() -> None:
    upm = UPM_V8
    rooms = [box(0, 0, 4 * upm, 3.1 * upm), box(10 * upm, 0, 13 * upm, 2.6 * upm), box(20 * upm, 0, 22 * upm, 1.5 * upm), box(30 * upm, 0, 31 * upm, 1.2 * upm), box(40 * upm, 0, 45 * upm, 4 * upm)]
    layers = {
        "A-EINB": [
            {"type": "insert", "name": "COOKTOP_800X520", "point": [2.0 * upm, 0.2 * upm], "rotation": 0.0, "xscale": upm, "yscale": upm, "_catalogue": {"kind": "COOKTOP"}},
            {"type": "text", "text": "Küche", "point": [1.0 * upm, 1.0 * upm], "height": 0.1, "rotation": 0.0},
        ],
        "P-SANI": [
            {"type": "insert", "name": "BATHTUB_1700X750", "point": [11.5 * upm, 0.0], "rotation": 0.0, "xscale": upm, "yscale": upm, "_catalogue": {"kind": "BATHTUB"}},
            {"type": "insert", "name": "WC_WALL_400X560", "point": [21 * upm, 0.0], "rotation": 0.0, "xscale": upm, "yscale": upm, "_catalogue": {"kind": "WC_WALL"}},
        ],
        "A-FURN": [{"type": "insert", "name": "WARDROBE_1500X600", "point": [30.5 * upm, 0.0], "rotation": 0.0, "xscale": upm, "yscale": upm, "_catalogue": {"kind": "WARDROBE"}},
                   {"type": "insert", "name": "BED_DOUBLE_1800X2000", "point": [42.0 * upm, 0.5 * upm], "rotation": 0.0, "xscale": upm, "yscale": upm, "_catalogue": {"kind": "BED_DOUBLE"}}],
        "A-ANNO": [],
    }
    names = name_rooms(rooms, _collect_room_tags(layers, upm), upm, RoomNameSettings())
    assert [item.name for item in names] == ["Küche", "Bad", "Gäste-WC", "Zimmer", "Schlafen"]
    assert [item.keep for item in names] == [True, True, True, False, True]  # the 1.2 m2 wardrobe niche is no room
    assert room_label_text(names[0].name, 12.4) == "Küche  12.4 m²"


def test_room_label_text_in_dxf_uses_semantic_name_and_area() -> None:
    grid = RasterGrid(0.0, 10.0, 0.05, 200, 200)
    walls = np.zeros((200, 200), dtype=bool)
    # Interior exactly 80 x 62 pixels = 4.0 x 3.1 m = 12.4 m2, enclosed by a 2 px thick wall ring.
    walls[40:106, 40:124] = True
    walls[42:104, 42:122] = False
    target: list[dict] = []
    tags = [((3.0, 6.0), "kitchen")]
    count = _add_room_labels(target, walls, grid, 0.3, units_per_metre=1.0, tagged_points=tags)
    assert count == 1
    texts = [entity["text"] for entity in target if entity.get("type") == "text"]
    assert texts == ["Küche  12.4 m²"]
    # No tags -> generic label (existing behaviour).
    generic: list[dict] = []
    _add_room_labels(generic, walls, grid, 0.3)
    assert [entity["text"] for entity in generic] == ["Room 1 (12.4 m²)"]
    assert [entity["text"] for entity in generic if entity.get("type") == "text"]


# 8. layer lineweights and DASHED A-OBEN in the written DXF ----------------------------------


def _presentation() -> tuple[dict, dict, dict]:
    layers = load_layer_config().for_project_mode("indoor")
    styles, colors, linetypes, hatches = _layer_presentation(layers)
    _add_ortho_plan_layer_presentation(styles, colors, linetypes, hatches)
    return styles, colors, linetypes


def test_layer_lineweights_and_dashed_overhead_layer_in_written_dxf(work_dir: Path) -> None:
    styles, colors, linetypes = _presentation()
    names = ("A-WALL", "A-EINB", "P-SANI", "A-FURN", "A-FURN-DETL", "A-OBEN")
    layers = {name: [{"type": "line", "start": [0.0, float(index)], "end": [1.0, float(index)]}] for index, name in enumerate(names)}
    path = export_plan(_result(layers, styles=styles, colors=colors, linetypes=linetypes), work_dir, coordinates="local")
    document = ezdxf.readfile(path)
    expected = {"A-WALL": 50, "A-EINB": 35, "P-SANI": 25, "A-FURN": 18, "A-FURN-DETL": 13, "A-OBEN": 18}  # 0.50 / 0.35 / 0.25 / 0.18 / 0.13 / 0.18 mm
    assert {name: document.layers.get(name).dxf.lineweight for name in names} == expected
    assert document.layers.get("A-OBEN").dxf.linetype == "DASHED"
    assert "DASHED" in document.linetypes
    for name in names[:-1]:
        assert document.layers.get(name).dxf.linetype.upper() == "CONTINUOUS"


def test_synthetic_indoor_plan_inserts_uniform_catalogue_wc(work_dir: Path) -> None:
    from test_ortho_plan import _build_indoor_wall_face_scene
    from samcloud.ortho.plan import generate_clean_plan

    level, _room = _build_indoor_wall_face_scene(work_dir / "scene", units_per_metre=0.8)
    result = generate_clean_plan(work_dir / "scene", level=level, mode="indoor")
    wcs = [entity for entity in result.layers["P-SANI"] if entity.get("type") == "insert"]
    assert wcs and wcs[0]["name"] == "WC_WALL_400X560"
    assert wcs[0]["xscale"] == wcs[0]["yscale"] == pytest.approx(float(result.fingerprint_inputs["units_per_metre"]), abs=1e-9)
    assert "A-OBEN" in result.layers and result.layers["A-OBEN"] == []
    assert result.layer_linetypes["A-OBEN"] == "DASHED"
    assert [item["block"] for item in result.fingerprint_inputs["catalogue_inserts"]] == ["WC_WALL_400X560"]


# --- follow-up: kitchen run (L), washbasin snap within 0.30 m, bathtub orientation -------------------------------


def _mask_of(grid: RasterGrid, *polygons: Polygon) -> np.ndarray:
    cols, rows = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    xs = grid.min_x + (cols + 0.5) * grid.pixel_size
    ys = grid.max_y - (rows + 0.5) * grid.pixel_size
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for polygon in polygons:
        mask |= contains_xy(polygon, xs, ys)
    return mask


def _kitchen_l_scene(upm: float):
    grid = RasterGrid(-1.0 * upm, 6.0 * upm, 0.02 * upm, 350, 350)
    walls = [LineString([(0.0, 0.0), (5.0 * upm, 0.0)]), LineString([(0.0, 0.0), (0.0, 4.0 * upm)])]
    # Noisy counter evidence (deeper than a counter): the run must still become a clean 0.60 m L.
    bottom = box(0.0, 0.0, 4.0 * upm, 0.66 * upm)
    left = box(0.0, 0.0, 0.66 * upm, 3.0 * upm)
    cooktop = box(2.0 * upm, 0.02 * upm, 2.6 * upm, 0.55 * upm)  # 0.6 m wide hob region on the bottom leg
    sink = box(0.05 * upm, 1.5 * upm, 0.58 * upm, 2.5 * upm)  # 1.0 m sink region on the left leg
    return grid, walls, _mask_of(grid, bottom, left), _mask_of(grid, cooktop), _mask_of(grid, sink)


def _kitchen_instance(mask: np.ndarray, label: str, upm: float):
    from samcloud.ortho.vector.footprints import ObjectInstance

    return ObjectInstance(mask, label, 120, True, 0.9 * upm, 1.0)


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_kitchen_l_run_is_clean_and_carries_labelled_cooktop_and_sink(upm: float, work_dir: Path) -> None:
    from shapely.geometry import Point

    from samcloud.ortho.plan import _emit_kitchen_run
    from samcloud.ortho.vector.kitchen_run import KitchenRunSettings, build_kitchen_run

    grid, walls, counters, hob, sink = _kitchen_l_scene(upm)
    run = build_kitchen_run(counters | hob | sink, grid, walls, upm, KitchenRunSettings())
    assert run is not None and run.shape == "L" and len(run.pieces) == 1 and len(run.legs) == 2
    polygon = run.pieces[0]
    assert len(polygon.exterior.coords) - 1 == 6  # a clean L outline
    assert polygon.area / upm**2 == pytest.approx(0.60 * 4.0 + 0.60 * (3.0 - 0.60), rel=0.04)
    for leg in run.legs:
        rectangle = leg.polygon()
        assert rectangle.minimum_rotated_rectangle.area == pytest.approx(rectangle.area)
        assert leg.depth == pytest.approx(0.60 * upm)  # standard run depth, aligned to the wall
    assert all(wall.distance(polygon) < 1e-6 for wall in walls)  # back edges flush on the walls

    # With appliance labels: one cooktop and one sink insert on A-EINB, mounted on the wall edge, +Y into the room.
    layers: dict[str, list[dict]] = {"A-EINB": [], "A-FURN": [], "P-SANI": [], "A-ANNO": []}
    pending = [(_kitchen_instance(counters, "kitchen counter", upm), "kitchen counter", "Küche"), (_kitchen_instance(hob, "stove", upm), "stove", "Herd"), (_kitchen_instance(sink, "sink", upm), "sink", "Spüle")]
    metrics: dict = {}
    height_map = np.zeros(counters.shape, dtype=np.float32)
    emitted = _emit_kitchen_run(pending, layers, [], grid, height_map, 0.0, walls, upm, metrics, KitchenRunSettings())
    assert emitted == 1 and metrics["kitchen_run_shape"] == "L"
    placed = [entity for entity in layers["A-EINB"] if entity.get("type") == "insert"]
    assert sorted(entity["name"].rsplit("_", 1)[0] for entity in placed) == ["COOKTOP", "KITCHEN_SINK"]
    for entity in placed:
        assert entity["xscale"] == entity["yscale"] == pytest.approx(upm)
        wall = walls[0] if entity["name"].startswith("COOKTOP") else walls[1]
        assert wall.distance(Point(*_world(entity, 0.0, 0.0))) < 1e-6  # mounting edge on the wall
        entry = catalogue_entry_for_block(entity["name"])
        footprint = Polygon([_world(entity, x, y) for x, y in ((-entry.width_m / 2, 0.0), (entry.width_m / 2, 0.0), (entry.width_m / 2, entry.depth_m), (-entry.width_m / 2, entry.depth_m))])
        assert polygon.buffer(1e-6).contains(footprint)  # the block sits inside the run, front face into the room
        assert footprint.centroid.distance(wall) == pytest.approx(entry.depth_m * upm / 2.0, abs=1e-6)
    assert next(entity for entity in placed if entity["name"].startswith("COOKTOP"))["name"] == "COOKTOP_580X510"
    # Without appliance labels no cooktop / sink is invented.
    bare: dict[str, list[dict]] = {"A-EINB": [], "A-FURN": [], "P-SANI": [], "A-ANNO": []}
    _emit_kitchen_run([(_kitchen_instance(counters, "kitchen counter", upm), "kitchen counter", "Küche")], bare, [], grid, height_map, 0.0, walls, upm, {}, KitchenRunSettings())
    assert not [entity for entity in bare["A-EINB"] if entity.get("type") == "insert"]
    assert [entity for entity in bare["A-EINB"] if entity.get("type") == "polyline" and entity.get("_einb_kind") == "fine_kitchen"]
    # Written DXF: 6-vertex L on A-EINB plus COOKTOP / KITCHEN_SINK inserts.
    path = export_plan(_result(layers), work_dir / f"krun_{upm}", coordinates="local")
    document = ezdxf.readfile(path)
    polylines = [entity for entity in document.modelspace() if entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer == "A-EINB"]
    assert len(polylines) == 1 and len(list(polylines[0].get_points("xy"))) == 6
    assert {entity.dxf.name.rsplit("_", 1)[0] for entity in document.modelspace() if entity.dxftype() == "INSERT"} == {"COOKTOP", "KITCHEN_SINK"}


def test_kitchen_u_run_closes_three_legs() -> None:
    from samcloud.ortho.vector.kitchen_run import build_kitchen_run

    grid = RasterGrid(-1.0, 6.0, 0.02, 350, 350)
    walls = [LineString([(0.0, 0.0), (3.0, 0.0)]), LineString([(0.0, 0.0), (0.0, 3.0)]), LineString([(3.0, 0.0), (3.0, 3.0)])]
    mask = _mask_of(grid, box(0.0, 0.0, 3.0, 0.62), box(0.0, 0.0, 0.62, 3.0), box(2.38, 0.0, 3.0, 3.0))
    run = build_kitchen_run(mask, grid, walls, 1.0)
    assert run is not None and run.shape == "U" and len(run.pieces) == 1 and len(run.legs) == 3


def test_kitchen_evidence_without_a_wall_is_not_a_wall_run() -> None:
    from samcloud.ortho.vector.kitchen_run import build_kitchen_run

    grid = RasterGrid(-1.0, 6.0, 0.02, 350, 350)
    far = [LineString([(0.0, 0.0), (5.0, 0.0)])]
    assert build_kitchen_run(_mask_of(grid, box(1.0, 2.0, 3.0, 2.6)), grid, far, 1.0) is None


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
@pytest.mark.parametrize("gap_m, rotation_error, snapped", [(0.20, 0.0, True), (0.28, 0.0, True), (0.20, 25.0, True), (0.45, 0.0, False)])
def test_washbasin_within_30_cm_of_a_wall_snaps_flush(upm: float, gap_m: float, rotation_error: float, snapped: bool) -> None:
    wall = LineString([(0.0, 0.0), (0.0, 4.0 * upm)])
    centre = np.asarray((gap_m * upm + 0.24 * upm, 2.0 * upm))  # rear edge `gap_m` off the wall (basin depth 0.48 m)
    measured = {"type": "insert", "name": "FURNITURE", "point": centre.tolist(), "rotation": 90.0 + rotation_error, "xscale": 0.62 * upm, "yscale": 0.48 * upm, "attributes": {"LABEL": "Lavabo"}}
    placed = _catalogue_fixture("WASHBASIN", measured, [wall], upm)
    assert placed is not None and placed["name"].startswith("WASHBASIN")
    assert placed["_catalogue"]["wall_snapped"] is snapped
    origin = _world(placed, 0.0, 0.0)
    if snapped:
        assert abs(origin[0]) < 1e-9  # mounting edge exactly on the wall
        assert float(_world(placed, 0.0, 1.0)[0] - origin[0]) > 0.0  # +Y into the room
        assert abs(float(_world(placed, 1.0, 0.0)[0] - origin[0])) < 1e-9  # +X runs along the wall
    else:
        assert abs(origin[0]) > 0.05 * upm  # free-standing fallback: stays where it was measured


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_bathtub_long_side_along_wall_versus_short_end_against_wall(upm: float) -> None:
    from shapely.geometry import Point

    wall = LineString([(0.0, 0.0), (6.0 * upm, 0.0)])  # room above the wall (+y)
    corners = ((-0.85, 0.0), (0.85, 0.0), (0.85, 0.75), (-0.85, 0.75))
    # (a) long side along the wall (1.70 m along x, 0.75 m deep), 3 cm off the wall.
    along = {"type": "insert", "name": "FURNITURE", "point": [3.0 * upm, (0.03 + 0.375) * upm], "rotation": 0.0, "xscale": 1.70 * upm, "yscale": 0.75 * upm, "attributes": {}}
    mounted = _catalogue_fixture("BATHTUB", along, [wall], upm)
    assert mounted["name"] == "BATHTUB_1700X750" and "end_on" not in mounted["_catalogue"]
    assert wall.distance(Point(*_world(mounted, 0.0, 0.0))) < 1e-9  # mounting edge = the long side on the wall
    assert float(_world(mounted, 0.0, 1.0)[1] - _world(mounted, 0.0, 0.0)[1]) > 0.0
    foot_a = Polygon([_world(mounted, x, y) for x, y in corners])
    assert foot_a.bounds[2] - foot_a.bounds[0] == pytest.approx(1.70 * upm) and foot_a.bounds[3] - foot_a.bounds[1] == pytest.approx(0.75 * upm)
    # (b) short end against the wall: long axis perpendicular to the wall.
    end = {"type": "insert", "name": "FURNITURE", "point": [3.0 * upm, (0.03 + 0.85) * upm], "rotation": 90.0, "xscale": 1.70 * upm, "yscale": 0.75 * upm, "attributes": {}}
    placed = _catalogue_fixture("BATHTUB", end, [wall], upm)
    assert placed["name"] == "BATHTUB_1700X750" and placed["_catalogue"]["end_on"] is True
    foot_b = Polygon([_world(placed, x, y) for x, y in corners])
    # The long axis stays perpendicular to the wall (1.70 m across y, 0.75 m along x) ...
    assert foot_b.bounds[3] - foot_b.bounds[1] == pytest.approx(1.70 * upm, abs=1e-9)
    assert foot_b.bounds[2] - foot_b.bounds[0] == pytest.approx(0.75 * upm, abs=1e-9)
    # ... and only the short end touches the wall, flush.
    assert wall.distance(foot_b) == pytest.approx(0.0, abs=1e-9)
    assert foot_b.bounds[1] == pytest.approx(0.0, abs=1e-9)
    assert foot_b.centroid.x == pytest.approx(3.0 * upm, abs=1e-9)
    assert foot_a.bounds[3] - foot_a.bounds[1] < foot_b.bounds[3] - foot_b.bounds[1]  # not simply snapped long-side-first


# --------------------------------------------------------------------------- viewer / snapshot block expansion

_VIEWER_BLOCKS = ("WC_WALL", "WASHBASIN", "BATHTUB", "BED_DOUBLE", "COOKTOP")


def _viewer_inserts() -> list[dict]:
    inserts = []
    for index, name in enumerate(_VIEWER_BLOCKS):
        for turn, rotation in enumerate((0.0, 90.0, 180.0, 270.0)):
            inserts.append({"type": "insert", "name": name, "point": [4.0 + 5.0 * index, 4.0 + 5.0 * turn], "rotation": rotation, "xscale": UPM_V8, "yscale": UPM_V8, "attributes": {}})
    return inserts


def _footprint(insert: dict) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = symbol_extents(insert["name"])
    corners = np.asarray([_world(insert, x, y) for x in (x0, x1) for y in (y0, y1)])
    return float(corners[:, 0].min()), float(corners[:, 1].min()), float(corners[:, 0].max()), float(corners[:, 1].max())


def _world_inverse(insert: dict, x: float, y: float) -> tuple[float, float]:
    angle = math.radians(float(insert["rotation"]))
    dx, dy = x - insert["point"][0], y - insert["point"][1]
    scale = float(insert["xscale"])
    return ((dx * math.cos(angle) + dy * math.sin(angle)) / scale, (-dx * math.sin(angle) + dy * math.cos(angle)) / scale)


def test_vector_payload_expands_catalogue_inserts_into_symbol_primitives() -> None:
    from samcloud.dxf.result import vector_payload

    inserts = _viewer_inserts()
    payload = vector_payload(_result({"A-SYMB": inserts}), grid_size=16)
    layer = payload["layers"]["A-SYMB"]
    assert all(entity["type"] != "insert" for entity in layer["entities"])
    per_insert = [len(SYMBOLS[name].entities) for name in _VIEWER_BLOCKS for _ in range(4)]
    assert len(layer["entities"]) == sum(per_insert)
    cursor = 0
    for insert, count in zip(inserts, per_insert, strict=True):
        part = layer["entities"][cursor:cursor + count]
        cursor += count
        xs: list[float] = []
        ys: list[float] = []
        for entity in part:
            for point in entity.get("points", []):
                xs.append(point[0])
                ys.append(point[1])
            if entity["type"] == "circle":
                xs.extend((entity["center"][0] - entity["radius"], entity["center"][0] + entity["radius"]))
                ys.extend((entity["center"][1] - entity["radius"], entity["center"][1] + entity["radius"]))
        expected = _footprint(insert)
        assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx(expected, abs=1e-6)
        entry = CATALOGUE[insert["name"]][0]
        width, height = expected[2] - expected[0], expected[3] - expected[1]
        if int(insert["rotation"]) % 180:
            width, height = height, width
        assert (width, height) == pytest.approx((entry.width_m * UPM_V8, entry.depth_m * UPM_V8), abs=1e-6)
        # origin on the wall edge: the symbol lies entirely on the local +Y side of the insert point
        local_y = [_world_inverse(insert, px, py)[1] for px, py in zip(xs, ys, strict=True)]
        assert min(local_y) >= -1e-6
        assert max(local_y) == pytest.approx(entry.depth_m, abs=1e-6)
    bounds = layer["bounds"]
    assert bounds[0] <= min(_footprint(i)[0] for i in inserts) + 1e-9
    assert bounds[3] >= max(_footprint(i)[3] for i in inserts) - 1e-9
    assert layer["grid"] and max(index for cell in layer["grid"].values() for index in cell) < len(layer["entities"])


def test_vector_payload_mirrored_scale_arc_and_legacy_unit_block() -> None:
    from samcloud.dxf.result import vector_payload

    door = {"type": "insert", "name": "DOOR", "point": [1.0, 1.0], "rotation": 0.0, "xscale": -2.0, "yscale": 2.0, "attributes": {}}
    legacy = {"type": "insert", "name": "FURNITURE", "point": [10.0, 10.0], "rotation": 90.0, "xscale": 2.0, "yscale": 3.0, "attributes": {}}
    unknown = {"type": "insert", "name": "NOT_A_BLOCK", "point": [0.0, 20.0], "rotation": 0.0, "xscale": 1.0, "yscale": 1.0, "attributes": {}}
    entities = vector_payload(_result({"A-SYMB": [door, legacy, unknown]}), grid_size=8)["layers"]["A-SYMB"]["entities"]
    line, arc = entities[0], entities[1]
    assert line["type"] == "line" and line["start"] == pytest.approx([1.0, 1.0]) and line["end"] == pytest.approx([1.0, 3.0])
    assert arc["type"] == "polyline" and not arc["closed"]
    assert arc["points"][0] == pytest.approx([-1.0, 1.0]) and arc["points"][-1] == pytest.approx([1.0, 3.0])
    assert all(math.hypot(x - 1.0, y - 1.0) == pytest.approx(2.0, abs=1e-6) for x, y in arc["points"])
    rectangle = entities[2]
    assert rectangle["type"] == "polyline" and rectangle["closed"] and len(rectangle["points"]) == 4
    assert np.asarray(rectangle["points"]).ravel() == pytest.approx([10.0, 10.0, 10.0, 12.0, 7.0, 12.0, 7.0, 10.0])
    assert np.asarray(entities[3]["points"]).ravel() == pytest.approx([0.0, 20.0, 1.0, 20.0, 1.0, 21.0, 0.0, 21.0])


def test_overlay_raster_draws_symbol_pixels_inside_footprint() -> None:
    from samcloud.dxf.overlay import render_dxf_rgba

    insert = {"type": "insert", "name": "BATHTUB", "point": [2.0, 1.0], "rotation": 90.0, "xscale": UPM_V8, "yscale": UPM_V8, "attributes": {}}
    rendered = render_dxf_rgba(_result({"P-SANI": [insert]}), bounds=(0.0, 0.0, 4.0, 4.0), size=(400, 400))
    ink = np.argwhere(rendered[..., 3] > 0)
    assert len(ink) > 200
    x0, y0, x1, y1 = _footprint(insert)
    world_x, world_y = ink[:, 1] / 399 * 4.0, 4.0 - ink[:, 0] / 399 * 4.0
    assert world_x.min() >= x0 - 0.02 and world_x.max() <= x1 + 0.02
    assert world_y.min() >= y0 - 0.02 and world_y.max() <= y1 + 0.02
    # the inner basin outline (not only the outer rectangle) is drawn: ink exists well inside the footprint
    inner = (world_x > x0 + 0.04) & (world_x < x1 - 0.04) & (world_y > y0 + 0.04) & (world_y < y1 - 0.04)
    assert int(inner.sum()) > 20
