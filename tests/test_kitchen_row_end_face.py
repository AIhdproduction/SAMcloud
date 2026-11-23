"""A tall kitchen row without a scanned wall behind it: as deep as its own scanned end face, and no row without one.

Synthetic noisy points only (seeded, faces with 8 mm noise, nothing on a cell border); production settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from samcloud.bim.kitchen_rows import KitchenRowSettings, detect_tall_rows
from test_kitchen_rows import UPMS, _face, _ring

# the row of the scene: front at x = -2.653, from y = 12.347 to the oven column at 13.9, body back to x = -2.003
FRONT, BACK, SOUTH, NORTH = -2.653, -2.003, 12.347, 13.9


def _scene(upm: float, rng: np.random.Generator, *, end_face: tuple[float, float] | None = (FRONT, BACK), front: bool = True):
    """North wall only: the partition behind the east row is hidden between two rows of cabinets and was never scanned."""
    walls = affinity.scale(box(-6.4, 14.5, -1.8, 14.8), upm, upm, origin=(0, 0))
    legs = affinity.scale(unary_union([box(-6.1, 13.9, -3.3, 14.5), box(-6.1, 12.0, -5.5, 14.5)]), upm, upm, origin=(0, 0))
    points = [
        _face(rng, "u", 13.95, -3.25, -2.7, 0.1, 2.25, 1000, upm),  # oven column front face (wall behind it)
        _face(rng, "u", 14.2, -6.0, -3.4, 1.42, 2.18, 600, upm),  # upper cabinets over the base run
        _face(rng, "u", 13.92, -6.0, -3.4, 0.2, 0.9, 400, upm),  # base run front face
    ]
    if front:
        points.append(_face(rng, "v", FRONT, SOUTH, NORTH, 0.1, 2.25, 2400, upm))
    if end_face is not None:
        points.append(_face(rng, "u", SOUTH, end_face[0], end_face[1], 0.1, 2.25, int(1500 * abs(end_face[1] - end_face[0])), upm))
    return walls, legs, np.vstack(points)


def _east(rows, upm: float):
    return [row for row in rows if row.polygon.bounds[1] / upm < 13.0]


@pytest.mark.parametrize("upm", UPMS)
@pytest.mark.parametrize("seed", (3, 7, 19))
def test_row_without_a_wall_is_as_deep_as_its_end_face(upm: float, seed: int) -> None:
    walls, legs, points = _scene(upm, np.random.default_rng(seed))
    rows = detect_tall_rows(points, walls, legs, upm, 0.0, KitchenRowSettings(), legs)
    east = _east(rows, upm)
    assert len(east) == 1, [row.polygon.bounds for row in rows]
    x0, y0, x1, y1 = [value / upm for value in east[0].polygon.bounds]
    # faces at the median of their points (8 mm noise), not on the 5 cm cells
    assert x0 == pytest.approx(FRONT, abs=0.015) and x1 == pytest.approx(BACK, abs=0.03), "front face to the far end of the end face"
    assert y0 == pytest.approx(SOUTH, abs=0.015) and y1 == pytest.approx(NORTH, abs=0.12), "the end face is the side of the row"
    assert east[0].top_m == pytest.approx(2.2, abs=0.1)
    assert east[0].polygon.minimum_rotated_rectangle.area == pytest.approx(east[0].polygon.area, rel=0.02)
    assert len(rows) == 2, "the end face is no second row, the oven column keeps its wall"


@pytest.mark.parametrize("upm", UPMS)
@pytest.mark.parametrize("seed", (13, 29))
def test_two_rows_in_an_inner_corner_without_any_wall(upm: float, seed: int) -> None:
    """No wall is scanned behind either row: each row takes its own end face, and the front of the neighbour is not one of them."""
    rng = np.random.default_rng(seed)
    legs = affinity.scale(unary_union([box(-6.1, 13.9, -3.3, 14.5), box(-6.1, 12.0, -5.5, 14.5)]), upm, upm, origin=(0, 0))
    points = np.vstack([
        _face(rng, "v", FRONT, SOUTH, NORTH, 0.1, 2.25, 2400, upm),  # east row front
        _face(rng, "u", SOUTH, FRONT, BACK, 0.1, 2.25, 1000, upm),  # its free end face
        _face(rng, "u", 13.953, -3.247, -2.7, 0.1, 2.25, 1000, upm),  # oven column front, running out into the room before the east row
        _face(rng, "v", -3.247, 13.953, 14.5, 1.0, 2.25, 700, upm),  # its side, seen above the worktop of the base run
        _face(rng, "u", 13.92, -6.0, -3.4, 0.2, 0.9, 400, upm),  # base run front face
    ])
    rows = detect_tall_rows(points, None, legs, upm, 0.0, KitchenRowSettings(), legs)
    east = _east(rows, upm)
    assert len(rows) == 2 and len(east) == 1, [row.polygon.bounds for row in rows]
    x0, y0, x1, y1 = [value / upm for value in east[0].polygon.bounds]
    assert (x0, y0) == pytest.approx((FRONT, SOUTH), abs=0.015) and x1 == pytest.approx(BACK, abs=0.03), "its body lies behind its front, not in the room"
    assert y1 == pytest.approx(NORTH, abs=0.12)
    oven = next(row for row in rows if row is not east[0])
    ox0, oy0, ox1, oy1 = [value / upm for value in oven.polygon.bounds]
    assert oy0 == pytest.approx(13.953, abs=0.015) and oy1 == pytest.approx(14.5, abs=0.03), "front to the far end of its own side face"
    assert ox0 == pytest.approx(-3.247, abs=0.03)


@pytest.mark.parametrize("upm", UPMS)
def test_wall_behind_gives_the_depth_and_the_end_face_the_end(upm: float) -> None:
    """With the wall scanned the row fills front -> wall as before; its scanned end face says where the row ends."""
    walls, legs, points = _scene(upm, np.random.default_rng(7))
    walls = unary_union([walls, affinity.scale(box(-2.0, 12.05, -1.8, 14.8), upm, upm, origin=(0, 0))])  # the wall goes on 0.3 m beyond the row
    with_face = _east(detect_tall_rows(points, walls, legs, upm, 0.0, KitchenRowSettings(), legs), upm)
    without = _east(detect_tall_rows(points, walls, legs, upm, 0.0, KitchenRowSettings(end_face=False), legs), upm)
    assert len(with_face) == len(without) == 1
    x0, y0, x1, y1 = [value / upm for value in with_face[0].polygon.bounds]
    assert x1 == pytest.approx(-2.0, abs=1e-6), "flush on the wall face"
    assert x0 == pytest.approx(FRONT, abs=0.015) and y0 == pytest.approx(SOUTH, abs=0.015), "front and end at their scanned planes"
    old = [value / upm for value in without[0].polygon.bounds]
    assert old[2] == pytest.approx(-2.0, abs=1e-6) and old[0] == pytest.approx(FRONT, abs=0.015)
    assert old[1] == pytest.approx(SOUTH, abs=0.051) and old[3] == pytest.approx(y1, abs=1e-9), "without the rule the end lies on its 5 cm cell"


@pytest.mark.parametrize("upm", UPMS)
@pytest.mark.parametrize(
    ("end_face", "why"),
    [
        (None, "a front alone has no measured depth"),
        ((FRONT, FRONT + 0.18), "a 0.18 m return is no cabinet side"),
        ((FRONT, FRONT + 1.25), "1.25 m is no cabinet depth: another row or a partition"),
        ((FRONT - 0.62, FRONT + 0.63), "a face passing the front line on both sides is no end face"),
    ],
)
def test_no_row_without_a_measured_depth(upm: float, end_face, why: str) -> None:
    walls, legs, points = _scene(upm, np.random.default_rng(11), end_face=end_face)
    rows = detect_tall_rows(points, walls, legs, upm, 0.0, KitchenRowSettings(), legs)
    assert _east(rows, upm) == [], why
    assert len(rows) == 1, "the oven column against the north wall stays"


def test_end_face_rule_can_be_switched_off() -> None:
    walls, legs, points = _scene(1.0, np.random.default_rng(7))
    assert _east(detect_tall_rows(points, walls, legs, 1.0, 0.0, KitchenRowSettings(end_face=False), legs), 1.0) == []


@pytest.mark.parametrize("upm", UPMS)
@pytest.mark.parametrize(("box_top", "expected"), [(2.29, 2.2), (1.62, 1.62)])
def test_fridge_of_a_row_never_stands_above_the_measured_row_top(upm: float, box_top: float, expected: float) -> None:
    """A loose instance box 9 cm above the row is cut to the row top; a fridge lower than its row keeps its height."""
    from samcloud.bim.plan_model import _apply_tall_rows
    from test_kitchen_rows import _item, _scene as _walled_scene

    walls, legs, points = _walled_scene(upm, np.random.default_rng(9))
    floor_z = -0.01 * upm
    fridge = _item("F006", affinity.scale(box(-2.66, 12.29, -1.92, 13.04), upm, upm, origin=(0, 0)), label="Kühlschrank", semantic_label="refrigerator",
                   z_min=floor_z, z_max=floor_z + box_top * upm, z_source="instance_bbox")
    report: dict = {}
    result = _apply_tall_rows([fridge], list(getattr(legs, "geoms", [legs])), points, walls, [], None, upm, floor_z, KitchenRowSettings(), report)
    assert fridge in result and (fridge.z_max - floor_z) / upm == pytest.approx(expected, abs=0.03 if box_top > 2.0 else 1e-9)
    assert all("footprint" in item for row in report["kitchen_rows"] for item in row["replaced"])
    kept = [item for row in report["kitchen_rows"] for item in row["kept"]]
    assert [item["id"] for item in kept] == ["F006"], "the fridge keeps its item"
    assert Polygon(kept[0]["footprint"]).bounds[0] / upm == pytest.approx(-2.66, abs=1e-6), "with the outline it had before"


# --------------------------------------------------------------------------------------------- the 2D plan shows the rows

@dataclass
class _Plan:
    layers: dict


def test_row_units_are_drawn_in_the_plan_once(tmp_path) -> None:
    from samcloud.bim.furniture_recon.dxf_outline import apply_reconstructed_outlines, row_units

    upm = 0.9891
    oven, tall, fridge = box(2.3, 6.0, 3.1, 6.6), box(3.0, 5.2, 3.62, 6.0), box(3.0, 4.5, 3.62, 5.2)
    wall_stage = box(2.31, 6.09, 3.07, 6.6)  # the oven column as the wall stage drew it, a few centimetres off the row unit
    cupboard = Polygon([(3.2, 4.2), (4.1, 4.2), (4.1, 4.6), (3.2, 4.6)])  # a traced blob whose points stand in the row
    furniture = [
        {"id": "F001", "catalogue_kind": "oven_tower", "label": "Backofen", "layer": "A-EINB", "footprint": _ring(oven)},
        {"id": "F002", "catalogue_kind": "tall_unit", "label": "Hochschrank", "layer": "A-EINB", "footprint": _ring(tall)},
        {"id": "F003", "catalogue_kind": None, "label": "Kühlschrank", "layer": "A-FURN", "footprint": _ring(fridge)},
        {"id": "F004", "catalogue_kind": "tall_unit", "label": "tall_unit", "layer": "A-EINB", "footprint": _ring(box(0.0, 0.0, 0.6, 0.6))},
    ]
    traced = box(3.0, 4.45, 3.7, 4.6)  # the fridge as the plan traced it: a sliver of its side face ...
    completed = box(3.0, 4.45, 3.7, 5.2)  # ... which was completed to a body before the row took it: only the instance id still matches
    rows = [
        {"units": ["F001"], "replaced": [{"id": "F028", "label": "tall_unit", "instance_id": None, "footprint": _ring(wall_stage)}], "kept": []},
        {"units": ["F003", "F002"], "replaced": [{"id": "F005", "label": "Schrank", "instance_id": 41, "footprint": _ring(cupboard)}],
         "kept": [{"id": "F003", "label": "Kühlschrank", "instance_id": 33, "footprint": _ring(completed)}]},
    ]
    model = {"units_per_metre": upm, "furniture": furniture, "source": {"depth_furniture": {"kitchen_rows": rows}}}
    (tmp_path / "plan_model.json").write_text(json.dumps(model), encoding="utf-8")
    units = row_units(tmp_path / "plan_model.json")
    assert [Polygon(outline).equals_exact(shape, 1e-9) for (outline, _, _), shape in zip(units, (oven, tall, fridge), strict=True)] == [True, True, True], (
        "the units of the rows, the fridge among them; a tall unit of no row is not listed")
    assert [label for _, label, _ in units] == ["HS", "HS", "KS"] and {scale for _, _, scale in units} == {upm}

    drawn = {"type": "polyline", "points": _ring(wall_stage), "closed": True, "holes": [], "_einb_kind": "tall_unit"}
    run = {"type": "polyline", "points": _ring(box(0.0, 5.9, 3.7, 6.6)), "closed": True, "holes": [], "_run_legs": [_ring(box(0.0, 5.9, 2.3, 6.6))]}
    front = {"type": "polyline", "points": [[3.006, 5.3], [3.004, 5.9]], "closed": False, "_einb_kind": "segment"}  # the row front, seen as a loose line
    elsewhere = {"type": "polyline", "points": [[0.5, 4.0], [1.1, 4.01]], "closed": False, "_einb_kind": "segment"}
    layers = {
        "A-EINB": [drawn, run, front, elsewhere],
        "A-FURN": [{"type": "polyline", "points": _ring(cupboard), "closed": True, "holes": []},
                   {"type": "polyline", "points": _ring(traced), "closed": True, "holes": [], "_instance_id": 33},
                   {"type": "polyline", "points": _ring(box(1.5, 3.0, 1.9, 3.7)), "closed": True, "holes": []}],
        "A-ANNO": [{"type": "text", "text": "Schrank h=2.2", "point": [3.7, 4.4]}, {"type": "text", "text": "Tisch", "point": [1.0, 2.0]}],
        "A-WALL": [{"type": "polyline", "points": _ring(cupboard), "closed": True}],
    }
    result = apply_reconstructed_outlines(_Plan(layers), tmp_path)
    einb = result.layers["A-EINB"]
    outlines = [entity for entity in einb if entity.get("_einb_kind") == "tall_unit"]
    assert len(outlines) == 3 and outlines[0] is drawn, "the wall-stage drawing of the oven column stays and is not doubled"
    assert Polygon(outlines[1]["points"]).symmetric_difference(tall).area < 1e-9 and outlines[1]["_kitchen_row"]
    assert Polygon(outlines[2]["points"]).symmetric_difference(fridge).area < 1e-9, "the fridge is drawn as the unit of the row it is"
    assert sum(1 for entity in einb if entity.get("_tall_unit_cross")) == 4 and [entity["text"] for entity in einb if entity["type"] == "text"] == ["HS", "KS"]
    assert run in einb and elsewhere in einb and front not in einb, "only the loose line that retraces the row front goes"
    assert [len(result.layers["A-FURN"]), [entity["text"] for entity in result.layers["A-ANNO"]]] == [1, ["Tisch"]], (
        "the replaced cupboard with its label and the traced sliver of the fridge are gone")
    assert result.layers["A-WALL"] == layers["A-WALL"], "other layers are never touched"
    assert row_units(tmp_path / "missing.json") == [] and apply_reconstructed_outlines(_Plan(layers), tmp_path / "missing").layers is layers
