"""A peninsula with a raised bar: kept by the label raster, measured in the scan, and the leg it joins ends at its back.

Synthetic rasters and scan points only (seeded noise, edges off the pixel grid, production settings).
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from samcloud.ortho.vector import kitchen_run
from samcloud.ortho.vector.kitchen_layout import _tidy_uppers
from samcloud.ortho.vector.kitchen_legs import FrontLegSettings
from samcloud.ortho.vector.kitchen_run import KitchenRunSettings, _extract_bars, _is_cross_section, build_kitchen_run
from test_kitchen_fronts import NORTH, SCALES, _face, _plan, _ring, _top
from test_kitchen_legs import _bounds, _known, _labels, _layout
from test_ortho_kitchen import _grid, _mask_of, _scaled

# --------------------------------------------------------------------------------------------- the label raster keeps it

# west wall x = 0, north wall y = 4 (metres); the edges lie off the 2 cm pixel grid
WEST_LEG = (0.0, 0.993, 0.617, 4.0)
NORTH_LEG = (0.0, 3.387, 3.213, 4.0)
PENINSULA = (0.617, 0.993, 2.747, 1.853)  # counter and raised bar are both labelled: 0.86 m of evidence
STOOLS = (1.453, 0.787, 1.871, 0.993)  # labelled things beside the bar: 1.07 m of evidence over 0.42 m


def _ragged(mask: np.ndarray, rng: np.random.Generator, share: float = 0.25) -> np.ndarray:
    """Drop a random share of the border pixels: no edge of the evidence is a straight pixel row."""
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    border = mask & ~inner
    return mask & ~(border & (rng.random(mask.shape) < share))


def _peninsula_raster(upm: float, rng: np.random.Generator, *, stools: bool = True):
    grid = _grid(upm)
    walls = [LineString([(0.0, 0.0), (0.0, 4.0 * upm)]), LineString([(0.0, 4.0 * upm), (5.0 * upm, 4.0 * upm)])]
    shapes = [_scaled(upm, *WEST_LEG), _scaled(upm, *NORTH_LEG), _scaled(upm, *PENINSULA)] + ([_scaled(upm, *STOOLS)] if stools else [])
    labelled = _ragged(_mask_of(grid, *shapes), rng)
    heights = np.zeros(labelled.shape, dtype=np.float32)
    heights[_mask_of(grid, *shapes[:3])] = 0.91 * upm
    heights[_mask_of(grid, _scaled(upm, 0.617, 0.993, 2.747, 1.25), *shapes[3:])] = 1.13 * upm  # the bar and what stands at it
    return grid, walls, labelled, heights


@pytest.mark.parametrize("upm", [1.0, 0.8157, 0.9731])
@pytest.mark.parametrize("seed", [5, 23])
def test_peninsula_as_thick_as_a_long_leg_stays_a_leg(upm: float, seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
    grid, walls, labelled, heights = _peninsula_raster(upm, np.random.default_rng(seed))
    run = build_kitchen_run(labelled, grid, walls, upm, KitchenRunSettings(), heights)
    assert run is not None and len(run.legs) == 3 and run.shape == "U", None if run is None else (run.shape, len(run.legs))
    free = [leg for leg in run.legs if abs(leg.tangent[0]) > 0.9 and leg.origin[1] < 2.5 * upm]
    assert len(free) == 1
    x0, y0, x1, y1 = (value / upm for value in free[0].polygon().bounds)
    assert (x0, x1) == pytest.approx((0.617, 2.747), abs=0.06), "from the front of the west leg to its measured end"
    assert (y0, y1) == pytest.approx((0.993, 1.853), abs=0.08), "the raster keeps the labelled depth; the scan measures the body later"
    # what went wrong before: the stretch with the stools was cut out as a leg along the other axis, and nothing of the peninsula was left
    monkeypatch.setattr(kitchen_run, "_is_cross_section", lambda *args: False)
    lost = build_kitchen_run(labelled, grid, walls, upm, KitchenRunSettings(), heights)
    assert lost is not None and len(lost.legs) == 2


def _production_pixels() -> tuple[int, int, int, int, int]:
    settings, res = KitchenRunSettings(), 0.02
    dmin, dmax = int(np.ceil(settings.min_depth_m / res)), int(settings.max_depth_m / res)
    lmin = max(dmin, int(np.ceil(settings.min_leg_length_m / res)))
    return dmin, dmax, lmin, max(lmin, int(round(settings.long_leg_m / res))), max(1, int(round(settings.run_gap_m / res)))


def _raster(*boxes: tuple[float, float, float, float], seed: int = 3) -> np.ndarray:
    """Boolean raster of (x0, y0, x1, y1) metre boxes at 2 cm, rows = y, with ragged borders."""
    raster = np.zeros((260, 260), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        raster[int(round(y0 / 0.02)):int(round(y1 / 0.02)), int(round(x0 / 0.02)):int(round(x1 / 0.02))] = True
    return _ragged(raster, np.random.default_rng(seed), 0.2)


def _lengths(bars: list[tuple[int, int, int, int]]) -> list[tuple[float, float]]:
    return sorted((round((r1 - r0) * 0.02, 1), round((c1 - c0) * 0.02, 1)) for r0, r1, c0, c1 in bars)


def test_real_legs_along_the_columns_are_no_cross_sections() -> None:
    dmin, dmax, lmin, flag, gap = _production_pixels()
    # an L whose return leg is two units long: half of it lies inside the long leg
    corner = _raster((0.5, 0.5, 3.91, 1.11), (0.5, 0.5, 1.11, 1.71))
    assert _lengths(_extract_bars(corner, dmin, dmax, lmin, flag, gap)) == [(0.6, 2.8), (1.2, 0.6)]
    # a T: a leg crossing the middle of a longer one and going on for a metre
    tee = _raster((0.5, 0.5, 4.11, 1.11), (2.0, 0.5, 2.61, 2.11))
    assert (1.6, 0.6) in _lengths(_extract_bars(tee, dmin, dmax, lmin, flag, gap))
    # a deep peninsula (0.9 m) along the columns on a 0.6 m leg along the rows
    deep = _raster((0.5, 0.5, 3.91, 1.11), (0.5, 0.5, 1.41, 2.71))
    assert (2.2, 0.9) in _lengths(_extract_bars(deep, dmin, dmax, lmin, flag, gap))


def test_cross_section_is_judged_on_what_sticks_out() -> None:
    _, _, _, _, gap = _production_pixels()
    thick = _raster((0.5, 1.0, 3.11, 1.87), (1.6, 0.79, 2.03, 1.0))  # 0.87 m leg along the rows, 0.21 m of things beside it
    assert _is_cross_section(thick, (39, 94, 80, 101), gap), "0.21 m beside a 0.87 m leg is no leg"
    legged = _raster((0.5, 1.0, 3.11, 1.87), (1.6, 0.21, 2.21, 1.0))  # a 0.61 m wide leg going on for 0.79 m
    assert not _is_cross_section(legged, (10, 94, 80, 110), gap)


# --------------------------------------------------------------------------------------------- the scan measures it

# the real cabinets (metres): west leg front x = 0.587 down to the peninsula; peninsula front y = 1.617, back panel y = 0.963
BACK, FRONT, END = 0.963, 1.617, 1.613
WEST_KNOWN = box(0.0, 0.403, 0.587, 4.0)  # the raster leg overshoots: a chair at the end of the bar is labelled as kitchen
RASTER = box(0.571, 0.803, 1.713, 1.403)  # the raster peninsula: the run depth centred on counter + bar, 0.2 m off the body


def _scene(rng, *, back: bool = True, west_front_to: float = FRONT, chair: bool = True):
    """U with a diagonal corner; the peninsula carries a raised bar on its far side, things stand at the end of the bar."""
    points, labels = _known(rng)
    points += [
        _face(rng, (1.05, 3.413), (3.113, 3.413)), _face(rng, (0.587, west_front_to), (0.587, 2.95)), _face(rng, (0.587, 2.95), (1.05, 3.413)),
        _face(rng, (0.587, FRONT), (END, FRONT)),  # peninsula front
        _top(rng, box(0.25, 1.0, 0.57, 3.8)), _top(rng, box(0.6, 1.25, 1.60, 1.60)),
        _top(rng, box(0.03, 0.72, 1.60, 1.22), height=1.13),  # the raised bar
    ]
    if back:
        points.append(_face(rng, (0.60, BACK), (END, BACK), noise=0.012))  # its back panel, seen from the other side of the room
    if chair:
        points.append(np.column_stack((rng.uniform(0.08, 0.52, 700), rng.uniform(0.43, 0.70, 700), rng.uniform(0.35, 1.08, 700))))
    labels += [_labels(rng, box(0.35, FRONT, 0.587, 3.6)), _labels(rng, box(0.05, 0.72, 1.68, FRONT)), _labels(rng, box(0.05, 0.42, 0.55, 0.72))]
    return np.vstack(points), np.vstack(labels)


def _legs(run: dict, upm: float, theta: float):
    return [_bounds(ring, upm, theta) for ring in run["_run_legs"]]


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_raster_peninsula_takes_its_measured_body_and_the_leg_ends_at_its_back(upm: float, theta: float) -> None:
    points, labels = _scene(np.random.default_rng(71))
    _, stats, run = _layout(points, labels, upm, theta, known=(NORTH, WEST_KNOWN, RASTER))
    assert stats["kitchen_legs_from_fronts"] == [] and len(stats["kitchen_free_legs_measured"]) == 1
    support = stats["kitchen_free_legs_measured"][0]
    assert support["behind"] == "back face" and support["joined"] == "fronts meet"
    assert support["depth_m"] == pytest.approx(FRONT - BACK, abs=0.02)
    assert support["anchor_cut_m"] == pytest.approx(BACK - 0.403, abs=0.03) and support["anchor_front_share"] < 0.3
    _, west, peninsula = _legs(run, upm, theta)
    assert peninsula[1] == pytest.approx(BACK, abs=0.02) and peninsula[3] == pytest.approx(FRONT, abs=0.02), "back panel and front as scanned"
    assert peninsula[0] == pytest.approx(0.0, abs=1e-6), "into the corner on the wall"
    assert peninsula[2] == pytest.approx(1.713, abs=0.01), "the free end is never shorter than the raster leg"
    assert west[1] == pytest.approx(peninsula[1], abs=1e-6) and west[3] == pytest.approx(4.0, abs=1e-6), "the west leg ends at the back of the peninsula"
    assert run["_kitchen_run"] == "U"


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize(
    ("case", "why"),
    [
        ({"scene": {"back": False}}, "no scanned back face: the depth is not measured"),
        ({"raster": box(0.571, 0.303, 1.713, 0.903)}, "the measured body is another place than the raster leg"),
        ({"settings": FrontLegSettings(refit_free=False)}, "switched off"),
    ],
)
def test_raster_peninsula_stays_without_the_full_evidence(upm: float, theta: float, case: dict, why: str) -> None:
    points, labels = _scene(np.random.default_rng(73), **case.get("scene", {}))
    raster = case.get("raster", RASTER)
    _, stats, run = _layout(points, labels, upm, theta, known=(NORTH, WEST_KNOWN, raster), leg_settings=case.get("settings", FrontLegSettings()))
    assert stats["kitchen_free_legs_measured"] == [], why
    _, west, peninsula = _legs(run, upm, theta)
    assert peninsula == pytest.approx(raster.bounds, abs=1e-6), why
    assert west[1] == pytest.approx(0.403, abs=1e-6), "and the wall-backed leg keeps its end"


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_leg_going_on_behind_the_peninsula_is_not_cut(upm: float, theta: float) -> None:
    """A T: the cabinets of the west leg go on beyond the back of the peninsula, their front is scanned there."""
    points, labels = _scene(np.random.default_rng(79), chair=False)
    rng = np.random.default_rng(83)
    points = np.vstack((points, _face(rng, (0.587, 0.403), (0.587, BACK)), _top(rng, box(0.05, 0.42, 0.57, 0.70))))
    _, stats, run = _layout(points, labels, upm, theta, known=(NORTH, WEST_KNOWN, RASTER))
    assert len(stats["kitchen_free_legs_measured"]) == 1 and "anchor_cut_m" not in stats["kitchen_free_legs_measured"][0]
    assert _legs(run, upm, theta)[1][1] == pytest.approx(0.403, abs=1e-6)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_long_stretch_beyond_the_peninsula_is_not_given_up(upm: float, theta: float) -> None:
    """Only the rough end of a fitted leg (end_before_m) is cut without a front; a longer stretch may just be hidden."""
    points, labels = _scene(np.random.default_rng(89))
    known_west = box(0.0, 0.013, 0.587, 4.0)  # 0.95 m beyond the back of the peninsula
    _, stats, run = _layout(points, labels, upm, theta, known=(NORTH, known_west, RASTER))
    assert len(stats["kitchen_free_legs_measured"]) == 1 and "anchor_cut_m" not in stats["kitchen_free_legs_measured"][0]
    assert _legs(run, upm, theta)[1][1] == pytest.approx(0.013, abs=1e-6)


# --------------------------------------------------------------------------------------------- one outline per upper cabinet

def _upper(shape: Polygon, upm: float, theta: float, band: tuple[float, float]) -> dict:
    return {"type": "polyline", "points": _ring(_plan(shape, upm, theta)), "closed": True, "_upper_cabinet": True, "_z_range_m": list(band)}


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_uppers_of_one_unit_merge_and_the_top_of_a_tall_unit_is_no_upper(upm: float, theta: float) -> None:
    long_unit, again = box(1.243, 3.647, 2.703, 4.0), box(1.251, 3.652, 2.697, 4.0)  # the same unit from two detectors
    short_unit = box(1.121, 3.655, 1.641, 3.993)  # its trimmed end, found once more
    tower_top = box(2.803, 3.48, 3.47, 3.81)  # the upper band of an oven tower
    apart = box(3.693, 3.651, 4.413, 4.0)  # a separate unit beyond the tower, found 0.11 m short of it
    tower = {"type": "polyline", "points": _ring(_plan(box(2.79, 3.37, 3.58, 4.0), upm, theta)), "closed": True, "_einb_kind": "tall_unit"}
    layers = {
        "A-EINB": [tower],
        "A-OBEN": [_upper(short_unit, upm, theta, (1.45, 2.15)), _upper(long_unit, upm, theta, (1.455, 2.158)), _upper(again, upm, theta, (1.44, 2.16)),
                   _upper(tower_top, upm, theta, (1.32, 2.19)), _upper(apart, upm, theta, (1.41, 2.17))],
    }
    stats = _tidy_uppers(layers, theta, upm)
    assert stats == {"kitchen_uppers_in_tall_units": 1, "kitchen_uppers_merged": 2, "kitchen_uppers_to_tall_units": 2}
    kept = sorted((_bounds(entity["points"], upm, theta), tuple(entity["_z_range_m"])) for entity in layers["A-OBEN"])
    assert len(kept) == 2
    assert kept[0][0] == pytest.approx((1.121, 3.647, 2.79, 4.0), abs=1e-6) and kept[0][1] == (1.455, 2.158), "the band of the longest outline"
    assert kept[1][0] == pytest.approx((3.58, 3.651, 4.413, 4.0), abs=1e-6), "both reach the side of the tower between them"


@pytest.mark.parametrize(("upm", "theta"), SCALES)
@pytest.mark.parametrize("shift", [-0.004, 0.0, 0.003])
def test_two_outlines_of_one_row_that_overlap_are_one_run(upm: float, theta: float, shift: float) -> None:
    """A run found in two pieces that share 0.41 m (38 % of the shorter one, below the same-unit share): one outline over
    both, whatever a few millimetres do to either piece; a unit below the row (other height band) is left as it is."""
    long_piece, short_piece = box(2.938, 3.652, 4.514, 3.975), box(2.261 + shift, 3.654, 3.344 + shift, 3.938)
    lower = box(4.203, 3.66, 5.297, 3.93)
    layers = {"A-EINB": [], "A-OBEN": [_upper(short_piece, upm, theta, (1.34, 2.20)), _upper(long_piece, upm, theta, (1.36, 2.20)), _upper(lower, upm, theta, (0.95, 1.31))]}
    stats = _tidy_uppers(layers, theta, upm)
    assert stats["kitchen_uppers_merged"] == 1 and len(layers["A-OBEN"]) == 2 and "kitchen_uppers_abutted" not in stats
    run = next(entity for entity in layers["A-OBEN"] if entity["_z_range_m"][0] > 1.32)
    assert _bounds(run["points"], upm, theta) == pytest.approx((2.261 + shift, 3.652, 4.514, 3.975), abs=1e-6)
    assert tuple(run["_z_range_m"]) == (1.36, 2.20), "the band of the longer piece"
    below = next(entity for entity in layers["A-OBEN"] if entity["_z_range_m"][0] < 1.32)
    assert _bounds(below["points"], upm, theta) == pytest.approx((4.203, 3.66, 5.297, 3.93), abs=1e-6)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_units_that_touch_never_share_a_place(upm: float, theta: float) -> None:
    """Two units side by side whose outlines overlap by 3 cm (tracing noise): two outlines, the shorter one cut back to the
    end of the longer one."""
    first, second = box(1.203, 3.651, 2.431, 4.0), box(2.401, 3.655, 3.297, 3.996)
    layers = {"A-EINB": [], "A-OBEN": [_upper(second, upm, theta, (1.41, 2.19)), _upper(first, upm, theta, (1.40, 2.20))]}
    stats = _tidy_uppers(layers, theta, upm)
    assert stats["kitchen_uppers_merged"] == 0 and stats.get("kitchen_uppers_abutted") == 1
    boxes = sorted(_bounds(entity["points"], upm, theta) for entity in layers["A-OBEN"])
    assert boxes[0] == pytest.approx((1.203, 3.651, 2.431, 4.0), abs=1e-6) and boxes[1] == pytest.approx((2.431, 3.655, 3.297, 3.996), abs=1e-6)


def test_separate_uppers_are_left_alone() -> None:
    first, second = box(1.2, 3.65, 2.4, 4.0), box(2.41, 3.65, 3.3, 4.0)  # two units side by side
    layers = {"A-EINB": [], "A-OBEN": [_upper(first, 1.0, 0.0, (1.4, 2.2)), _upper(second, 1.0, 0.0, (1.4, 2.2))]}
    before = [list(entity["points"]) for entity in layers["A-OBEN"]]
    assert _tidy_uppers(layers, 0.0, 1.0) == {"kitchen_uppers_in_tall_units": 0, "kitchen_uppers_merged": 0, "kitchen_uppers_to_tall_units": 0}
    assert [entity["points"] for entity in layers["A-OBEN"]] == before
    # a tall unit farther than tall_snap_m from the end, or one that does not stand across the upper's depth, is not reached
    far = {"type": "polyline", "points": _ring(box(3.65, 3.37, 4.3, 4.0)), "closed": True, "_einb_kind": "tall_unit"}
    beside = {"type": "polyline", "points": _ring(box(3.4, 3.2, 4.0, 3.72)), "closed": True, "_einb_kind": "tall_unit"}
    layers["A-EINB"] = [far, beside]
    assert _tidy_uppers(layers, 0.0, 1.0)["kitchen_uppers_to_tall_units"] == 0
    assert [entity["points"] for entity in layers["A-OBEN"]] == before
