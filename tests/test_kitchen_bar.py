"""The raised bar top of a peninsula: measured in the scan, drawn at its depth, a carried solid in the model; nothing without evidence.

Synthetic scan points only (seeded noise, edges off the cells), production settings; the layout cases run at three scales / axes.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from samcloud.bim.furniture import load_furniture_mapping
from samcloud.bim.furniture_recon.families import family_of
from samcloud.bim.plan_model import PlanFurniture, _bar_top_item, _seat_bar_tops
from samcloud.ortho.vector.kitchen_bar import BarTopSettings, measure_bar_top
from samcloud.ortho.vector.kitchen_fronts import ScanPoints
from samcloud.ortho.vector.plan_polish import _LegBox
from test_kitchen_fronts import NORTH, SCALES, _face, _top
from test_kitchen_legs import _bounds, _known, _labels, _layout

# the peninsula (metres): body y 0.963..1.617, worktop at 0.905; the bar board y 0.737..1.183 at 1.113, 7 cm thick
BODY = box(0.0, 0.963, 2.213, 1.617)
WORKTOP, LEVEL, UNDER = 0.905, 1.113, 1.043
BAR = (0.737, 1.183)


def _objects(rng, area: Polygon, low: float, high: float, count: int) -> np.ndarray:
    """Things standing on a surface: points on their sides and tops between two heights."""
    x0, y0, x1, y1 = area.bounds
    return np.column_stack((rng.uniform(x0, x1, count), rng.uniform(y0, y1, count), rng.uniform(low, high, count)))


def _bar_scene(rng, *, bar: tuple[float, float] | None = BAR, hidden_from: float | None = None, chair: bool = True, level: float = LEVEL,
               edge_face: bool = True, worktop_from: float | None = None, length: tuple[float, float] = (0.0, 2.213)) -> np.ndarray:
    """Peninsula with worktop, things on it, and (optionally) a bar top with things on it and a chair standing at it."""
    x0, x1 = length
    seen_worktop = (bar[1] if bar is not None else 0.99) if worktop_from is None else worktop_from
    parts = [
        _top(rng, box(0.02, seen_worktop + 0.02, 2.19, 1.60), height=WORKTOP),
        _objects(rng, box(0.3, 1.30, 0.7, 1.55), WORKTOP, 1.33, 500),  # a coffee machine on the worktop
        _objects(rng, box(1.5, 1.25, 1.75, 1.45), WORKTOP, 1.21, 300),
        _face(rng, (0.0, 1.617), (2.213, 1.617)), _face(rng, (0.0, 0.963), (2.213, 0.963), noise=0.012),
    ]
    if bar is not None:
        top_to = bar[1] if hidden_from is None else hidden_from
        parts.append(_top(rng, box(x0 + 0.02, bar[0] + 0.012, x1 - 0.02, top_to - 0.012), height=level, density=900.0))
        if hidden_from is not None:  # the board near the worktop is hidden by what stands on it
            parts.append(_objects(rng, box(x0 + 0.05, hidden_from, x1 - 0.05, bar[1] - 0.02), level, level + 0.30, int(2500 * (x1 - x0))))
        parts.append(_objects(rng, box(x0 + 0.9, bar[0] + 0.1, x0 + 1.25, bar[1] - 0.1), level, level + 0.25, 600))  # a toaster on the bar
        if edge_face:
            parts.append(_face(rng, (x0, bar[0]), (x1, bar[0]), low=UNDER + (level - LEVEL), high=level, density=5000.0))
    if chair:  # the curved back of a chair standing at the bar: at bar level over 0.4 m only
        angle = rng.uniform(0.0, np.pi, 900)
        parts.append(np.column_stack((1.75 + 0.2 * np.cos(angle), 0.70 - 0.22 * np.sin(angle), rng.uniform(0.55, 1.16, 900))))
        parts.append(_top(rng, box(1.57, 0.50, 1.93, 0.86), height=0.67))  # its seat
    return np.vstack(parts)


def _scan(points: np.ndarray, upm: float = 1.0) -> ScanPoints:
    return ScanPoints(points[:, :2] * upm, points[:, 2].copy())


def _leg(upm: float = 1.0, back: str | None = None) -> _LegBox:
    x0, y0, x1, y1 = BODY.bounds
    return _LegBox(x0 * upm, x1 * upm, y0 * upm, y1 * upm, back)


# --------------------------------------------------------------------------------------------- measured

@pytest.mark.parametrize("upm", [1.0, 0.8124])
@pytest.mark.parametrize("seed", [3, 17, 41])
def test_bar_top_is_measured_with_level_edges_and_underside(upm: float, seed: int) -> None:
    bar = measure_bar_top(_leg(upm), _scan(_bar_scene(np.random.default_rng(seed)), upm), 0.93, upm)
    assert bar is not None
    assert bar.level_m == pytest.approx(LEVEL, abs=0.012) and bar.worktop_m == pytest.approx(WORKTOP, abs=0.012), "levels from the scan, not from the 0.93 handed in"
    assert bar.v0 / upm == pytest.approx(BAR[0], abs=0.025), "the free edge, not the chair standing 0.25 m further out"
    assert bar.v1 / upm == pytest.approx(BAR[1], abs=0.035)
    assert (bar.u0 / upm, bar.u1 / upm) == pytest.approx((0.0, 2.213), abs=0.045)
    assert bar.bottom_m == pytest.approx(UNDER, abs=0.025) and bar.evidence["edge_face"], "the underside read on the edge face"
    assert bar.support is not None and bar.support[1] / upm == pytest.approx(0.963, abs=1e-6) and bar.support[3] == pytest.approx(bar.v1), "it stands on the body"
    assert len(bar.steps) == 1 and bar.steps[0][0][1] == pytest.approx(bar.v1), "the step from worktop to bar lies inside the leg"
    assert bar.evidence["over_body_m"] == pytest.approx(BAR[1] - 0.963, abs=0.035)


@pytest.mark.parametrize("seed", [5, 23])
def test_board_hidden_by_what_stands_on_it_still_reaches_the_worktop(seed: int) -> None:
    """Only the outer 0.2 m of the board is seen; over the body it carries things. No worktop is seen there: the bar goes on."""
    bar = measure_bar_top(_leg(), _scan(_bar_scene(np.random.default_rng(seed), hidden_from=0.94, edge_face=False)), 0.93, 1.0)
    assert bar is not None
    assert (bar.v0, bar.v1) == pytest.approx(BAR, abs=0.04)
    assert bar.bottom_m == pytest.approx(bar.level_m - BarTopSettings().thickness_m, abs=1e-9) and not bar.evidence["edge_face"], "no edge face seen: a board"


@pytest.mark.parametrize(
    ("case", "why"),
    [
        ({"scene": {"bar": None}}, "a worktop with things on it and a chair beside it"),
        ({"scene": {"bar": None, "chair": False}}, "nothing raised at all"),
        ({"scene": {"bar": (0.30, 0.80), "worktop_from": 0.99}}, "a high table standing 0.16 m clear of the body is not carried by it"),
        ({"scene": {"bar": (0.85, 1.617), "worktop_from": 1.70}}, "the whole leg at that level is a high counter, not a bar on a counter"),
        ({"scene": {"bar": (0.90, 0.99)}}, "a 9 cm upstand is no top"),
        ({"scene": {"length": (0.0, 0.85)}}, "0.85 m of a 2.2 m leg: under half of it"),
        ({"scene": {"level": 0.96}}, "5 cm above the worktop is its upstand, not a bar"),
        ({"back": "v0"}, "a leg against a wall has no bar top of this kind"),
        ({"settings": BarTopSettings(enabled=False)}, "switched off"),
    ],
)
def test_no_bar_top_without_the_full_evidence(case: dict, why: str) -> None:
    points = _bar_scene(np.random.default_rng(29), **case.get("scene", {}))
    assert measure_bar_top(_leg(back=case.get("back")), _scan(points), 0.93, 1.0, case.get("settings", BarTopSettings())) is None, why
    assert measure_bar_top(_leg(), None, 0.93, 1.0) is None and measure_bar_top(_leg(), ScanPoints(np.zeros((0, 2)), np.zeros(0)), 0.93, 1.0) is None


# --------------------------------------------------------------------------------------------- in the layout (2D) and the model (3D)

WEST = box(0.0, 0.963, 0.587, 4.0)
PENINSULA = box(0.571, 0.963, 2.213, 1.617)


def _kitchen(rng) -> tuple[np.ndarray, np.ndarray]:
    """U kitchen of the leg tests with the bar scene on its peninsula (the bar runs from the wall to the free end)."""
    points, labels = _known(rng)
    points += [
        _face(rng, (1.05, 3.413), (3.113, 3.413)), _face(rng, (0.587, 1.617), (0.587, 2.95)), _face(rng, (0.587, 2.95), (1.05, 3.413)),
        _top(rng, box(0.25, 1.7, 0.57, 3.8), height=WORKTOP), _bar_scene(rng),
    ]
    labels += [_labels(rng, box(0.35, 1.617, 0.587, 3.6)), _labels(rng, box(0.05, 0.75, 2.2, 1.617))]
    return np.vstack(points), np.vstack(labels)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_layout_draws_the_bar_at_its_depth_and_hands_it_to_the_model(upm: float, theta: float) -> None:
    points, labels = _kitchen(np.random.default_rng(61))
    layers, stats, run = _layout(points, labels, upm, theta, known=(NORTH, WEST, PENINSULA))
    assert len(stats["kitchen_bar_tops"]) == 1 and stats["kitchen_bars"] == 1
    evidence = stats["kitchen_bar_tops"][0]
    assert evidence["depth_m"] == pytest.approx(BAR[1] - BAR[0], abs=0.05) and evidence["level_m"] == pytest.approx(LEVEL, abs=0.012)
    assert len(run["_run_bars"]) == 1
    bar = run["_run_bars"][0]
    x0, y0, x1, y1 = _bounds(bar["outline"], upm, theta)
    assert (y0, y1) == pytest.approx(BAR, abs=0.04) and x0 == pytest.approx(0.0, abs=0.05) and x1 == pytest.approx(2.213, abs=0.05)
    assert bar["bottom_m"] == pytest.approx(UNDER, abs=0.025) and bar["worktop_m"] == pytest.approx(WORKTOP, abs=0.012)
    assert _bounds(bar["support"], upm, theta)[1] == pytest.approx(0.963, abs=0.03), "the support is the part over the body"
    # 2D: the counter outline reaches the free edge of the board (not the chair), the step is drawn where the board ends on the worktop
    outline = _bounds(run["points"], upm, theta)
    assert outline[1] == pytest.approx(BAR[0], abs=0.04)
    steps = [_bounds(entity["points"], upm, theta) for entity in layers["A-FURN-DETL"] if entity.get("_kitchen_detail") == "bar_step"]
    assert len(steps) == 1 and steps[0][1] == pytest.approx(BAR[1], abs=0.04) and steps[0][3] == pytest.approx(BAR[1], abs=0.04)


@pytest.mark.parametrize(("upm", "theta"), SCALES)
def test_layout_without_a_scanned_bar_has_none(upm: float, theta: float) -> None:
    rng = np.random.default_rng(67)
    points, labels = _known(rng)
    points += [
        _face(rng, (1.05, 3.413), (3.113, 3.413)), _face(rng, (0.587, 1.617), (0.587, 2.95)), _face(rng, (0.587, 2.95), (1.05, 3.413)),
        _top(rng, box(0.25, 1.7, 0.57, 3.8), height=WORKTOP), _bar_scene(rng, bar=None),
    ]
    labels += [_labels(rng, box(0.35, 1.617, 0.587, 3.6)), _labels(rng, box(0.05, 0.99, 2.2, 1.617))]
    _, stats, run = _layout(np.vstack(points), np.vstack(labels), upm, theta, known=(NORTH, WEST, PENINSULA))
    assert stats["kitchen_bar_tops"] == [] and run["_run_bars"] == [] and stats["kitchen_bars"] == 0


def _ring(shape: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in list(shape.exterior.coords)[:-1]]


@pytest.mark.parametrize("upm", [1.0, 0.9891])
def test_bar_top_item_is_a_carried_solid_seated_on_the_final_worktop(upm: float) -> None:
    floor = -0.021 * upm
    board, over = box(0.0, 0.737 * upm, 2.213 * upm, 1.183 * upm), box(0.0, 0.963 * upm, 2.213 * upm, 1.183 * upm)
    bar = {"outline": _ring(board), "level_m": LEVEL, "bottom_m": UNDER, "worktop_m": 0.925, "support": _ring(over)}
    seen: set = set()
    item = _bar_top_item(bar, "F007", "A-EINB", floor, upm, seen)
    assert item is not None and _bar_top_item(bar, "F008", "A-EINB", floor, upm, seen) is None, "one item per board (a run in two pieces carries it twice)"
    assert (item.label, item.catalogue_kind, item.layer, item.mounting, item.shape_source, item.z_source) == ("Theke", "bar_top", "A-EINB", "resting", "parametric", "points")
    assert item.floor_contact is False and item.instance_id is None
    assert [solid["role"] for solid in item.solids] == ["top", "support"]
    assert (item.z_max - floor) / upm == pytest.approx(LEVEL) and (item.z_min - floor) / upm == pytest.approx(0.925)
    # the reconstruction then measures the worktop 2 cm lower: the support follows it, the board keeps its measured underside
    worktop = PlanFurniture(id="F006", instance_id=None, label="Arbeitsplatte", semantic_label="worktop", catalogue_kind="worktop", layer="A-EINB",
                            footprint=_ring(box(0.0, 0.963 * upm, 2.213 * upm, 1.617 * upm)), z_min=floor + 0.865 * upm, z_max=floor + 0.905 * upm, z_source="points")
    _seat_bar_tops([worktop, item], upm)
    top, support = item.solids
    assert (support["z_min"] - floor) / upm == pytest.approx(0.905) and support["z_max"] == pytest.approx(top["z_min"])
    assert ((top["z_min"] - floor) / upm, (top["z_max"] - floor) / upm) == pytest.approx((UNDER, LEVEL))
    assert (item.z_min - floor) / upm == pytest.approx(0.905)
    # a board lying on the worktop needs no support
    flat = _bar_top_item({**bar, "outline": _ring(over), "bottom_m": 0.925, "support": None}, "F009", "A-EINB", floor, upm, set())
    assert flat is not None and [solid["role"] for solid in flat.solids] == ["top"]
    assert _bar_top_item({**bar, "outline": []}, "F010", "A-EINB", floor, upm, set()) is None


def test_bar_top_settings_come_from_the_plan_config() -> None:
    from samcloud.ortho.plan import load_clean_plan_settings

    settings = load_clean_plan_settings().indoor_kitchen_bar_top
    assert settings == BarTopSettings(), "the shipped config states the defaults of the code"


def test_bar_top_has_its_own_family_and_ifc_type() -> None:
    assert family_of("Theke", "bar top", "bar_top") == "bar_top" and family_of("Theke", None, None) == "bar_top"
    klass, object_type = load_furniture_mapping().resolve(label="Theke", semantic_label="bar top", catalogue_kind="bar_top", layer="A-EINB")
    assert (klass.ifc_class, klass.predefined_type, object_type) == ("IfcFurniture", "USERDEFINED", "BarTop")


@pytest.mark.parametrize("upm", [1.0, 0.8124])
def test_measured_board_replaces_the_side_band_and_keeps_the_free_end_of_the_surface_model(upm: float) -> None:
    """Beyond the free end the surface model still says how far the top goes on (as without a measured board)."""
    from samcloud.ortho.vector.kitchen_layout import KitchenLayoutSettings, _Frame, _bars_of

    rng = np.random.default_rng(71)

    def height_of(points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0] / upm, points[:, 1] / upm
        heights = np.zeros(len(points))
        heights[(x >= 0.0) & (x <= 2.213) & (y >= 0.963) & (y <= 1.617)] = WORKTOP
        heights[(x >= 0.0) & (x <= 2.6) & (y >= 0.737) & (y <= 1.183)] = LEVEL  # the board, 0.39 m longer than the fitted leg
        heights[(x >= 2.213) & (x <= 2.6) & (y >= 1.183) & (y <= 1.617)] = WORKTOP  # and its worktop
        return (heights + rng.normal(0.0, 0.004, len(points)) * (heights > 0)) * upm

    leg = _leg(upm)
    frame, settings = _Frame(0.0, height_of), KitchenLayoutSettings()
    measured = measure_bar_top(leg, _scan(_bar_scene(np.random.default_rng(73)), upm), 0.93, upm)
    assert measured is not None
    plain = _bars_of(leg, [leg], None, None, frame, WORKTOP * upm, settings, upm)
    with_board = _bars_of(leg, [leg], None, None, frame, WORKTOP * upm, settings, upm, measured)
    assert len(plain) == 2 and len(with_board) == 1, "side band and end band without the board, the end band alone with it"
    end = with_board[0][0].bounds
    assert end[0] / upm == pytest.approx(2.213, abs=1e-6) and end[2] / upm == pytest.approx(2.6, abs=0.03)
    assert end[1] == pytest.approx(measured.v0) and end[3] / upm == pytest.approx(1.617, abs=1e-6), "across the leg and the measured board"
