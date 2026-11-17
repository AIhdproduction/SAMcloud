"""Kitchen run that follows the measured counter edges (straight / L / U / U with peninsula).

Real geometry only: rasterised counter evidence + DSM heights go through ``build_kitchen_run`` /
``_emit_kitchen_run`` and the written DXF is read back with ezdxf.
"""

from __future__ import annotations

import os
from pathlib import Path
import uuid

import ezdxf
import numpy as np
import pytest
from shapely import contains_xy
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
from samcloud.dxf.writer import export_plan
from samcloud.ortho.plan import _emit_kitchen_run
from samcloud.ortho.vector.footprints import ObjectInstance
from samcloud.ortho.vector.kitchen_run import KitchenRunSettings, build_kitchen_run
from samcloud.ortho.vector.raster_grid import RasterGrid

UPM_V8 = 0.8157


@pytest.fixture
def work_dir(request: pytest.FixtureRequest) -> Path:
    root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if root:
        path = Path(root) / f"kitchen_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("kitchen")


def _grid(upm: float) -> RasterGrid:
    return RasterGrid(-1.0 * upm, 6.0 * upm, 0.02 * upm, 350, 350)


def _mask_of(grid: RasterGrid, *polygons: Polygon) -> np.ndarray:
    cols, rows = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    xs = grid.min_x + (cols + 0.5) * grid.pixel_size
    ys = grid.max_y - (rows + 0.5) * grid.pixel_size
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for polygon in polygons:
        mask |= contains_xy(polygon, xs, ys)
    return mask


def _scaled(upm: float, *coords: float) -> Polygon:
    return box(*(value * upm for value in coords))


def _result(layers: dict[str, list[dict]]) -> DxfPlanResult:
    frame = Frame(np.zeros(3, dtype=np.float64), 1.0, np.eye(3, dtype=np.float64))
    return DxfPlanResult(
        project_name="Kit", level_name="EG", mode="indoor", profile_name="test", scale_factor=1.0, frame=frame,
        settings_fingerprint="x", layers=layers, statuses={name: LayerStatus(name, "ok", None, len(items)) for name, items in layers.items()},
        layer_colors={name: 7 for name in layers}, background=BackgroundImage(None, None, (0.0, 0.0, 20.0, 20.0)),
        layer_linetypes={}, layer_styles={}, blocks=sorted({str(e["name"]) for items in layers.values() for e in items if e.get("type") == "insert"}),
        variant="ortho",
    )


def _instance(mask: np.ndarray, label: str, upm: float) -> ObjectInstance:
    return ObjectInstance(mask, label, 120, True, 0.9 * upm, 1.0)


def _u_peninsula_scene(upm: float):
    """Back wall y=0, right wall x=5: back leg 0.60 deep, right leg 0.60 deep (3.0 long), peninsula 0.90 deep (2.2 long) at the back leg's left end."""
    grid = _grid(upm)
    walls = [LineString([(0.0, 0.0), (5.0 * upm, 0.0)]), LineString([(5.0 * upm, 0.0), (5.0 * upm, 4.0 * upm)])]
    back, right, peninsula = _scaled(upm, 1.0, 0.0, 4.4, 0.6), _scaled(upm, 4.4, 0.0, 5.0, 3.0), _scaled(upm, 1.0, 0.0, 1.9, 2.2)
    return grid, walls, (back, right, peninsula)


def _heights(grid: RasterGrid, counters: np.ndarray, upm: float, *extra: tuple[np.ndarray, float]) -> np.ndarray:
    height = np.zeros(counters.shape, dtype=np.float32)
    height[counters] = 0.9 * upm
    for mask, metres in extra:
        height[mask] = metres * upm
    return height


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_u_run_with_peninsula_has_three_measured_legs_and_swallows_counter_objects(upm: float, work_dir: Path) -> None:
    grid, walls, (back, right, peninsula) = _u_peninsula_scene(upm)
    counters = _mask_of(grid, back, right, peninsula)
    objects = _scaled(upm, 1.2, 0.8, 1.6, 1.2)  # a kettle on the peninsula: not a kitchen label, raised above the counter
    labelled = counters & ~_mask_of(grid, objects)
    heights = _heights(grid, counters, upm, (_mask_of(grid, objects), 1.2))
    run = build_kitchen_run(labelled, grid, walls, upm, KitchenRunSettings(), heights)
    assert run is not None and run.shape == "U" and len(run.legs) == 3 and len(run.pieces) == 1
    polygon = run.pieces[0]
    assert not list(polygon.interiors)  # the object does not punch a hole
    assert len(polygon.exterior.coords) - 1 == 8  # a clean rectilinear U
    expected_area = (0.9 * 2.2 + 0.6 * 2.5 + 0.6 * 3.0) * upm**2
    assert polygon.area == pytest.approx(expected_area, rel=0.04)
    depths = sorted(round(leg.depth / upm, 2) for leg in run.legs)
    assert depths == pytest.approx([0.60, 0.60, 0.90], abs=0.02)
    lengths = {round(leg.depth / upm, 1): leg for leg in run.legs}
    peninsula_leg = next(leg for leg in run.legs if leg.depth > 0.8 * upm)
    assert (peninsula_leg.end - peninsula_leg.start) / upm == pytest.approx(2.2, abs=0.08)  # measured end, not a fixed length
    right_leg = next(leg for leg in run.legs if abs(leg.tangent[1]) > 0.9 and leg.depth < 0.8 * upm)
    assert (right_leg.end - right_leg.start) / upm == pytest.approx(3.0, abs=0.08)
    assert lengths
    assert all(wall.distance(polygon) < 1e-6 for wall in walls)  # back edges flush on the walls

    # Emission: the run replaces traced blobs standing on it (>= 50 % inside); a blob beside it stays; DXF round trip.
    layers: dict[str, list[dict]] = {"A-EINB": [], "A-FURN": [], "P-SANI": [], "A-ANNO": [], "A-FURN-DETL": []}
    on_counter = [[x * upm, y * upm] for x, y in ((1.2, 0.8), (1.6, 0.8), (1.6, 1.2), (1.2, 1.2))]
    beside = [[x * upm, y * upm] for x, y in ((2.5, 2.0), (3.2, 2.0), (3.2, 2.6), (2.5, 2.6))]
    layers["A-FURN"] = [
        {"type": "polyline", "points": on_counter, "closed": True, "holes": [], "_label": "Objekt"},
        {"type": "polyline", "points": beside, "closed": True, "holes": [], "_label": "Tisch"},
    ]
    layers["A-ANNO"] = [{"type": "text", "text": "Objekt h=1.20", "point": [1.4 * upm, 1.0 * upm], "height": 0.1, "rotation": 0.0}]
    pending = [(_instance(labelled, "kitchen counter", upm), "kitchen counter", "Küche")]
    metrics: dict = {}
    assert _emit_kitchen_run(pending, layers, [], grid, heights, 0.0, walls, upm, metrics, KitchenRunSettings()) == 1
    assert metrics["kitchen_run_shape"] == "U" and metrics["kitchen_run_blobs_absorbed"] == 1
    path = export_plan(_result(layers), work_dir / f"u_{upm}", coordinates="local")
    modelspace = ezdxf.readfile(path).modelspace()
    einb = [entity for entity in modelspace if entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer == "A-EINB"]
    assert len(einb) == 1 and len(list(einb[0].get_points("xy"))) == 8
    furniture = [entity for entity in modelspace if entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer == "A-FURN"]
    assert len(furniture) == 1  # only the table beside the counter
    assert not [entity for entity in modelspace if entity.dxftype() == "TEXT" and entity.dxf.layer == "A-ANNO" and "Objekt" in entity.dxf.text]


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_l_run_with_heights_stays_a_clean_060_l(upm: float) -> None:
    grid = _grid(upm)
    walls = [LineString([(0.0, 0.0), (5.0 * upm, 0.0)]), LineString([(0.0, 0.0), (0.0, 4.0 * upm)])]
    counters = _mask_of(grid, _scaled(upm, 0.0, 0.0, 4.0, 0.66), _scaled(upm, 0.0, 0.0, 0.66, 3.0))
    run = build_kitchen_run(counters, grid, walls, upm, KitchenRunSettings(), _heights(grid, counters, upm))
    assert run is not None and run.shape == "L" and len(run.legs) == 2
    assert len(run.pieces[0].exterior.coords) - 1 == 6
    assert run.pieces[0].area / upm**2 == pytest.approx(0.6 * 4.0 + 0.6 * 2.4, rel=0.04)
    assert all(leg.depth == pytest.approx(0.60 * upm) for leg in run.legs)


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_chamfer_only_where_the_evidence_fills_the_inner_corner(upm: float) -> None:
    grid = _grid(upm)
    walls = [LineString([(0.0, 0.0), (5.0 * upm, 0.0)]), LineString([(4.0 * upm, 0.0), (4.0 * upm, 4.0 * upm)])]
    legs = [_scaled(upm, 0.0, 0.0, 4.0, 0.6), _scaled(upm, 3.4, 0.0, 4.0, 3.0)]
    plain = _mask_of(grid, *legs)
    run = build_kitchen_run(plain, grid, walls, upm, KitchenRunSettings(), _heights(grid, plain, upm))
    assert run is not None and len(run.pieces[0].exterior.coords) - 1 == 6  # square inner corner without evidence

    corner = np.asarray([(3.4, 0.6), (2.8, 0.6), (3.4, 1.2)]) * upm  # the diagonal fill of Domenic's counter
    filled = plain | _mask_of(grid, Polygon(corner))
    run = build_kitchen_run(filled, grid, walls, upm, KitchenRunSettings(), _heights(grid, filled, upm))
    assert run is not None
    polygon = run.pieces[0]
    assert len(polygon.exterior.coords) - 1 == 7  # one diagonal edge
    assert polygon.area / upm**2 > 0.6 * 4.0 + 0.6 * 2.4 + 0.08
    diagonal = [(a, b) for a, b in zip(polygon.exterior.coords[:-1], polygon.exterior.coords[1:], strict=True) if abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6]
    assert len(diagonal) == 1


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_freestanding_table_near_the_kitchen_is_not_merged(upm: float) -> None:
    grid = _grid(upm)
    walls = [LineString([(0.0, 0.0), (5.0 * upm, 0.0)]), LineString([(0.0, 0.0), (0.0, 4.0 * upm)])]
    counters = _mask_of(grid, _scaled(upm, 0.0, 0.0, 4.0, 0.6), _scaled(upm, 0.0, 0.0, 0.6, 3.0))
    table = _scaled(upm, 1.2, 1.2, 2.2, 1.9)  # 0.60 m clear of the counter, table-top height
    height = _heights(grid, counters, upm, (_mask_of(grid, table), 0.75))
    with_table = build_kitchen_run(counters, grid, walls, upm, KitchenRunSettings(), height)
    without = build_kitchen_run(counters, grid, walls, upm, KitchenRunSettings(), _heights(grid, counters, upm))
    assert with_table is not None and without is not None
    assert with_table.pieces[0].intersection(table).area == 0.0
    assert with_table.pieces[0].area == pytest.approx(without.pieces[0].area, rel=1e-6)
    # Even when its (unlabelled) top is counter-high and it floats in the evidence region the table stays out.
    high_table = _heights(grid, counters, upm, (_mask_of(grid, table), 0.90))
    run = build_kitchen_run(counters, grid, walls, upm, KitchenRunSettings(), high_table)
    assert run is not None and run.pieces[0].intersection(table).area == 0.0


def _ragged_u_scene(upm: float):
    """Ragged real-world U: top leg on the top wall, left leg on the left wall, a 0.9 m diagonal fill in their inner
    corner and a bottom peninsula that only touches the left leg; every counter edge is noisy by 2-4 cm (seeded).

    Metric layout (before scaling): top wall y=4 (leg 0.6 deep, x 0..3.6), left wall x=0 (leg 0.6 deep, y 1..4),
    peninsula x 0.6..2.1, y 1.0..1.6.
    """
    rng = np.random.default_rng(20261115)
    grid = RasterGrid(-1.0 * upm, 6.0 * upm, 0.01 * upm, 700, 700)
    walls = [LineString([(0.0, 0.0), (0.0, 4.0 * upm)]), LineString([(0.0, 4.0 * upm), (5.0 * upm, 4.0 * upm)])]
    cols, rows = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    xs = (grid.min_x + (cols + 0.5) * grid.pixel_size) / upm
    ys = (grid.max_y - (rows + 0.5) * grid.pixel_size) / upm

    def wobble(size: int) -> np.ndarray:
        """Edge offset in metres: smooth +-3 cm waves plus +-1 cm jitter, correlated over about 10 cm."""
        noise = np.convolve(rng.uniform(-1.0, 1.0, size + 20), np.ones(10) / 10.0, mode="same")[10:-10]
        return 0.03 * noise / max(float(np.abs(noise).max()), 1e-9) + 0.01 * rng.uniform(-1.0, 1.0, size)

    along_x, along_y, along_x2, along_y2 = wobble(grid.width)[cols], wobble(grid.height)[rows], wobble(grid.width)[cols], wobble(grid.height)[rows]
    top = (ys <= 4.0) & (ys >= 3.4 + along_x) & (xs >= 0.0) & (xs <= 3.6 + along_y)
    left = (xs >= 0.0) & (xs <= 0.6 + along_y2) & (ys >= 1.0 + along_x2) & (ys <= 4.0)
    fill = (xs >= 0.6) & (ys <= 3.4) & (ys >= 2.4) & ((xs - 0.6) + (3.4 - ys) <= 0.9 + 0.5 * along_x)
    peninsula = (xs >= 0.6) & (xs <= 2.1 + along_y) & (ys >= 1.0 + along_x2) & (ys <= 1.6 + along_x)
    return grid, walls, top | left | fill | peninsula


def _assert_clean_ragged_u(run, counters: np.ndarray, grid: RasterGrid, walls: list[LineString], upm: float) -> None:
    assert run is not None and len(run.legs) >= 2 and run.shape in {"L", "U", "U+"}
    for piece in run.pieces:
        points = np.asarray(piece.exterior.coords)
        for first, second in zip(points[:-1], points[1:], strict=True):
            angle = float(np.degrees(np.arctan2(second[1] - first[1], second[0] - first[0])) % 90.0)
            assert min(angle, 90.0 - angle) < 1.0 or abs(angle - 45.0) < 1.0, (first, second)  # axis parallel or the 45 degree chamfer
    union = unary_union(run.pieces)
    left_wall, top_wall = walls
    # Back edges flush on the walls: the run lies along both wall lines over a long stretch and never crosses them.
    assert top_wall.intersection(union.buffer(0.03 * upm)).length >= 2.0 * upm
    assert left_wall.intersection(union.buffer(0.03 * upm)).length >= 2.0 * upm
    assert union.bounds[3] <= 4.0 * upm + 0.03 * upm and union.bounds[0] >= -0.03 * upm
    for leg in run.legs:
        assert 0.5 * upm <= leg.depth <= 0.8 * upm, leg.depth / upm
    cols, rows = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    inside = contains_xy(union, grid.min_x + (cols + 0.5) * grid.pixel_size, grid.max_y - (rows + 0.5) * grid.pixel_size)
    iou = float((inside & counters).sum()) / float((inside | counters).sum())
    assert iou >= 0.55, iou


@pytest.mark.parametrize("upm", [1.0, 0.8])
def test_ragged_u_with_diagonal_corner_fill_becomes_a_clean_wall_attached_run(upm: float) -> None:
    """Scale invariance: the same structure comes out at 1.0 and at 0.8 units per metre."""
    grid, walls, counters = _ragged_u_scene(upm)
    heights = _heights(grid, counters, upm)
    run = build_kitchen_run(counters, grid, walls, upm, KitchenRunSettings(), heights)
    _assert_clean_ragged_u(run, counters, grid, walls, upm)
    assert len(run.legs) >= 3  # top leg, left leg and the peninsula attached to the left leg


def test_counter_blob_with_the_nearest_wall_0_6_m_away_is_not_a_run() -> None:
    upm = 1.0
    grid = _grid(upm)
    walls = [LineString([(0.0, 0.0), (5.0, 0.0)]), LineString([(0.0, 0.0), (0.0, 4.0)])]
    counters = _mask_of(grid, _scaled(upm, 0.6, 1.4, 2.4, 2.0))  # 1.8 x 0.6 m, 0.6 m from the left wall, 1.4 m from the other
    assert build_kitchen_run(counters, grid, walls, upm, KitchenRunSettings(), _heights(grid, counters, upm)) is None


@pytest.mark.parametrize("upm", [1.0, UPM_V8])
def test_counter_set_back_from_the_walls_is_flushed_onto_them_by_the_wall_anchored_fallback(upm: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """The regular extraction only snaps bars within 0.30 m of a wall; the wall-anchored fallback bridges the 0.35 m gap."""
    import samcloud.ortho.vector.kitchen_run as module

    calls: list[int] = []
    original = module._wall_anchored_bars
    monkeypatch.setattr(module, "_wall_anchored_bars", lambda *args, **kwargs: (calls.append(1), original(*args, **kwargs))[1])
    grid = _grid(upm)
    walls = [LineString([(0.0, 0.0), (5.0 * upm, 0.0)]), LineString([(0.0, 0.0), (0.0, 4.0 * upm)])]
    counters = _mask_of(grid, _scaled(upm, 0.35, 0.35, 4.0, 0.85), _scaled(upm, 0.35, 0.35, 0.85, 3.0))
    run = build_kitchen_run(counters, grid, walls, upm, KitchenRunSettings(), _heights(grid, counters, upm))
    assert calls and run is not None and run.shape == "L" and len(run.legs) == 2 and len(run.pieces) == 1
    polygon = run.pieces[0]
    assert all(wall.distance(polygon) < 1e-6 for wall in walls)  # back edges flush on the walls
    assert [round(leg.depth / upm, 2) for leg in run.legs] == pytest.approx([0.90, 0.90], abs=0.03)  # measured from the wall, snapped to a standard depth
    lengths = sorted(round((leg.end - leg.start) / upm, 1) for leg in run.legs)
    assert lengths == pytest.approx([2.65, 3.65], abs=0.1)  # the measured ends of the counter, not the wall ends
