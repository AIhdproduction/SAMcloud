"""Measured stepped furniture and grounding rules."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from samcloud.bim.scene_builder import build_scene
from samcloud.ortho.vector.footprints import ObjectInstance, furniture_solid_parts
from samcloud.ortho.vector.raster_grid import RasterGrid


def test_noisy_measured_sofa_emits_clean_low_and_high_solids() -> None:
    grid = RasterGrid(0.0, 4.0, 0.1, 40, 40)
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:30] = True
    heights = np.zeros((40, 40), dtype=float)
    heights[mask] = 0.44
    heights[5:35, 5:10] = 0.80  # backrest
    heights[14:25, 20:30] = 0.76  # arm/chaise detail
    heights[8, 8] = 1.40  # isolated scan spike must not create a fifth solid
    instance = ObjectInstance(mask, "sofa", 17, True, 0.8, 1.0)
    outline = Polygon([(0.5, 0.5), (3.0, 0.5), (3.0, 3.5), (0.5, 3.5)])

    parts = furniture_solid_parts(instance, heights, grid, 1.0, outline)

    assert 2 <= len(parts) <= 4
    tops = sorted(part["z_max_m"] for part in parts)
    assert tops[0] == pytest.approx(0.44, abs=0.05)
    assert tops[-1] >= 0.75
    assert all(Polygon(part["footprint"]).area >= 0.08 for part in parts)


def test_floor_standing_bbox_is_grounded() -> None:
    from test_bim_scene_builder import make_plan

    scene = build_scene(make_plan())
    sofa = next(item for item in scene.elements.values() if item.name == "Sofa 01")
    assert sofa.params.model_extra["z_bottom"] == 0.5


def _noisy_grid(seed: int, shape: tuple[int, int] = (260, 260)) -> tuple[RasterGrid, np.random.Generator]:
    return RasterGrid(0.0, shape[0] * 0.01, 0.01, shape[1], shape[0]), np.random.default_rng(seed)


def test_sparse_realistic_sofa_gives_seat_and_back_arm_levels_in_few_parts() -> None:
    # 1 cm raster like the production DSM: 2.0 x 1.0 m sofa, seat 0.45, backrest + arms 0.78, holes and noise in the scan.
    grid, rng = _noisy_grid(7)
    mask = np.zeros((260, 260), dtype=bool)
    mask[60:160, 30:230] = True
    heights = np.full(mask.shape, np.nan)
    heights[mask] = 0.45 + rng.normal(0.0, 0.015, int(mask.sum()))
    heights[60:100, 30:230] = 0.78 + rng.normal(0.0, 0.015, (40, 200))  # backrest strip along the long side
    heights[100:160, 30:55] = 0.75 + rng.normal(0.0, 0.015, (60, 25))  # one arm
    heights[100:160, 205:230] = 0.74 + rng.normal(0.0, 0.015, (60, 25))  # other arm
    heights[(rng.random(mask.shape) < 0.35) & mask] = np.nan  # a sparse scan
    mask &= np.isfinite(heights)
    outline = Polygon([(0.30, 1.0), (2.30, 1.0), (2.30, 2.0), (0.30, 2.0)])
    parts = furniture_solid_parts(ObjectInstance(mask, "sofa", 17, True, 0.78, 1.0), heights, grid, 1.0, outline)

    assert 2 <= len(parts) <= 4
    tops = sorted(part["z_max_m"] for part in parts)
    assert 0.38 <= tops[0] <= 0.50 and tops[-1] >= 0.70
    covered = Polygon(parts[0]["footprint"]).buffer(0)
    for part in parts[1:]:
        covered = covered.union(Polygon(part["footprint"]).buffer(0))
    assert covered.symmetric_difference(outline).area < 0.02 * outline.area  # the parts together are the footprint


def test_floor_noise_level_is_not_a_part_and_flat_objects_stay_one_box() -> None:
    grid, rng = _noisy_grid(11)
    mask = np.zeros((260, 260), dtype=bool)
    mask[60:200, 40:200] = True
    flat = np.where(mask, 0.74 + rng.normal(0.0, 0.02, mask.shape), np.nan)
    outline = Polygon([(0.4, 0.6), (2.0, 0.6), (2.0, 2.0), (0.4, 2.0)])
    instance = ObjectInstance(mask, "table", 17, True, 0.74, 1.0)
    assert furniture_solid_parts(instance, flat, grid, 1.0, outline) == []
    # A desk whose scan only reaches down to 0.08 m near the floor: that level is noise, the top and a 0.45 shelf remain.
    stepped = flat.copy()
    stepped[60:200, 40:80] = 0.45 + rng.normal(0.0, 0.015, (140, 40))
    stepped[150:200, 120:200] = 0.08 + rng.normal(0.0, 0.01, (50, 80))
    parts = furniture_solid_parts(instance, stepped, grid, 1.0, outline)
    assert all(part["z_max_m"] > 0.30 for part in parts)


def test_many_instances_stay_fast() -> None:
    import time

    grid = RasterGrid(0.0, 12.0, 0.01, 1200, 1200)
    rng = np.random.default_rng(3)
    started = time.perf_counter()
    for index in range(40):
        mask = np.zeros((1200, 1200), dtype=bool)
        row, col = 100 + 20 * index % 900, 100 + 25 * index % 900
        mask[row:row + 90, col:col + 150] = True
        heights = np.where(mask, 0.45 + 0.35 * (np.arange(1200)[None, :] > col + 100) + rng.normal(0.0, 0.01, mask.shape), np.nan)
        x0, y1 = col * 0.01, 12.0 - row * 0.01
        outline = Polygon([(x0, y1 - 0.9), (x0 + 1.5, y1 - 0.9), (x0 + 1.5, y1), (x0, y1)])
        assert furniture_solid_parts(ObjectInstance(mask, "sofa", 17, True, 0.8, 1.0), heights, grid, 1.0, outline)
    assert time.perf_counter() - started < 20.0


def test_parts_that_do_not_match_the_exported_footprint_fall_back_to_one_box() -> None:
    from samcloud.bim.plan_model import _parts_matching_footprint

    footprint = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    matching = [{"footprint": [[0, 0], [2, 0], [2, 0.5], [0, 0.5]], "z_max": 0.4}, {"footprint": [[0, 0.5], [2, 0.5], [2, 1], [0, 1]], "z_max": 0.8}]
    shifted = [{"footprint": [[1.6, 0], [3.6, 0], [3.6, 1], [1.6, 1]], "z_max": 0.4}]
    assert len(_parts_matching_footprint(matching, footprint)) == 2
    assert _parts_matching_footprint(shifted, footprint) == []
    assert _parts_matching_footprint(None, footprint) == []


def test_upper_cabinet_is_not_placed_above_a_sliver_base_run() -> None:
    from samcloud.ortho.vector.upper_cabinets import detect_upper_cabinets
    from shapely.geometry import LineString

    grid = RasterGrid(0.0, 4.0, 0.02, 300, 200)
    upper = np.zeros((200, 300), dtype=bool)
    upper[130:140, 20:190] = True
    nothing = np.zeros_like(upper)
    wall = LineString([(0.0, 1.5), (6.0, 1.5)])
    thin = Polygon([(0.4, 1.0), (3.6, 1.0), (3.6, 1.02), (0.4, 1.02)])
    proper = Polygon([(0.4, 0.9), (3.6, 0.9), (3.6, 1.5), (0.4, 1.5)])
    assert detect_upper_cabinets([thin], upper, nothing, nothing, grid, [wall], 1.0) == []
    assert detect_upper_cabinets([proper], upper, nothing, nothing, grid, [wall], 1.0)


@pytest.mark.parametrize("upm", [1.0, 0.8124])
def test_upper_cabinet_next_to_a_wall_stub_never_becomes_a_sliver(upm: float) -> None:
    # A unit that overlaps a wall body is cut by it; what is left must still be a unit (>= half the detected area, >= 0.15 m deep).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_ortho_kitchen_layout import _pts, _s, _scene
    from samcloud.ortho.vector.plan_polish import fit_kitchen_to_walls
    from shapely.geometry import box

    for x0, x1 in ((3.55, 3.70), (3.45, 3.72), (3.30, 3.65), (3.78, 4.20)):
        walls, room, layers = _scene(upm)
        source = _s(box(x0, 4.4, x1, 4.75), upm)
        layers["A-OBEN"] = [{"type": "polyline", "points": _pts(source), "closed": True, "_upper_cabinet": True}]
        fit_kitchen_to_walls(layers, [room], walls, 0.0, upm)
        result = Polygon(layers["A-OBEN"][0]["points"])
        assert result.area >= 0.5 * source.area
        rectangle = result.minimum_rotated_rectangle
        corners = list(rectangle.exterior.coords)
        assert min(((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2) ** 0.5, ((corners[1][0] - corners[2][0]) ** 2 + (corners[1][1] - corners[2][1]) ** 2) ** 0.5) >= 0.04 * upm
