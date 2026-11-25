"""Outdoor assistant edit operations on a synthetic environment-chain Scene: every op, its dependencies and validation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from shapely.geometry import LineString

from assistant_outdoor_synthetic import (
    BUILDING_RIDGE, FAKE_ORIGIN, FENCE_HEIGHT, MANHOLE, TREE_1, TREE_2, lv95, shared_project, terrain_z,
)
from samcloud.assistant.domains.outdoor import OutdoorDomain
from samcloud.assistant.domains.outdoor_edits import (
    BULK_OP, OPS, OutdoorProposal, building_heights, element_height, ops_for_tier, resolve_signature, signature_of,
)
from samcloud.assistant.edits import EditError, EditOp

TIN_TOLERANCE_M = 0.08  # the synthetic TIN approximates terrain_z within its simplification tolerance


@pytest.fixture(scope="module")
def domain() -> OutdoorDomain:
    return OutdoorDomain(shared_project(), "site", mode="outdoor")


@pytest.fixture(scope="module")
def base(domain: OutdoorDomain) -> Any:
    return domain.load_base()


@pytest.fixture
def proposal(domain: OutdoorDomain, base: Any) -> OutdoorProposal:
    return domain.new_proposal(base)


def _apply(proposal: OutdoorProposal, name: str, target: str | None = None, **params: Any) -> list[Any]:
    return proposal.apply(EditOp(name, target, params))


def _el(proposal: OutdoorProposal, element_id: str) -> Any:
    return proposal.plan.elements[element_id]


# ---------------------------------------------------------------------------------------------------------- trees

def test_set_tree_dimensions_and_validation(proposal: OutdoorProposal, base: Any) -> None:
    before = base.elements[TREE_1["id"]].params
    changes = _apply(proposal, "set_tree", TREE_1["id"], height=10.5, crown_diameter=6.0, trunk_diameter=0.4, crown_shape="cone")
    p = _el(proposal, TREE_1["id"]).params
    assert changes[0].change == "modified" and (p.height, p.crown_diameter, p.trunk_diameter, p.crown_shape) == (10.5, 6.0, 0.4, "cone")
    assert p.base_z == before.base_z and p.trunk_measured is True
    assert p.crown_base_z == pytest.approx(before.base_z + (before.crown_base_z - before.base_z) * 10.5 / before.height)
    assert _el(proposal, TREE_1["id"]).param_origin["height"].value == "manual"
    _apply(proposal, "set_tree", TREE_1["id"], height_delta=-0.5)
    assert _el(proposal, TREE_1["id"]).params.height == pytest.approx(10.0)
    for params, key in (({"height": 50.0}, "assistant.edit.invalid_value"), ({"crown_diameter": 0.1}, "assistant.edit.invalid_value"),
                        ({"trunk_diameter": 2.0, "crown_diameter": 1.0}, "assistant.edit.invalid_value"),
                        ({"crown_shape": "pyramid"}, "assistant.edit.bad_params"), ({}, "assistant.edit.bad_params"),
                        ({"height": 5.0, "height_delta": 1.0}, "assistant.edit.bad_params"), ({"colour": "red"}, "assistant.edit.bad_params")):
        with pytest.raises(EditError) as error:
            _apply(proposal, "set_tree", TREE_1["id"], **params)
        assert error.value.key == key
    assert base.elements[TREE_1["id"]].params.height == before.height  # the base is never mutated


def test_move_and_delete_tree_follow_the_terrain(proposal: OutdoorProposal, base: Any) -> None:
    height = base.elements[TREE_2["id"]].params.height
    _apply(proposal, "move_tree", TREE_2["id"], dx_m=1.5, dy_m=-1.0)
    p = _el(proposal, TREE_2["id"]).params
    x, y = TREE_2["xy"][0] + 1.5, TREE_2["xy"][1] - 1.0
    assert p.trunk_xy == pytest.approx((x, y)) and p.height == height
    assert p.base_z == pytest.approx(float(terrain_z(x, y)), abs=TIN_TOLERANCE_M)
    e, n = lv95(10.0, 10.0)
    _apply(proposal, "move_tree", TREE_2["id"], to={"e": e, "n": n})
    assert _el(proposal, TREE_2["id"]).params.trunk_xy == pytest.approx((10.0, 10.0))
    with pytest.raises(EditError):
        _apply(proposal, "move_tree", TREE_2["id"], dx_m=100.0)
    _apply(proposal, "delete_tree", TREE_2["id"])
    assert TREE_2["id"] not in proposal.plan.elements and proposal.needs_confirmation()
    assert [entry["change"] for entry in proposal.summary()] == ["removed"]
    with pytest.raises(EditError) as error:
        _apply(proposal, "set_tree", "no-such-tree", height=5.0)
    assert error.value.key == "assistant.edit.not_found"


# -------------------------------------------------------------------------------------------------------- manholes

def test_manhole_ops(proposal: OutdoorProposal, base: Any) -> None:
    cover = base.elements["manhole-0"].params.center[2]
    _apply(proposal, "set_manhole", "manhole-0", diameter=0.8, type="Einlaufschacht")
    p = _el(proposal, "manhole-0").params
    assert (p.diameter, p.type) == (0.8, "Einlaufschacht")
    _apply(proposal, "set_manhole", "manhole-0", cover_z=FAKE_ORIGIN[2] + cover + 0.05)
    assert _el(proposal, "manhole-0").params.center[2] == pytest.approx(cover + 0.05)
    for params in ({"diameter": 2.0}, {"diameter": 0.1}, {"cover_z": FAKE_ORIGIN[2] + cover + 5.0}):
        with pytest.raises(EditError):
            _apply(proposal, "set_manhole", "manhole-0", **params)
    _apply(proposal, "move_manhole", "manhole-0", dx_m=4.0)
    moved = _el(proposal, "manhole-0").params.center
    x, y = MANHOLE["xy"][0] + 4.0, MANHOLE["xy"][1]
    assert moved[:2] == pytest.approx((x, y))
    assert moved[2] - (cover + 0.05) == pytest.approx(float(terrain_z(x, y) - terrain_z(*MANHOLE["xy"])), abs=TIN_TOLERANCE_M)
    _apply(proposal, "delete_manhole", "manhole-0")
    assert "manhole-0" not in proposal.plan.elements


# ---------------------------------------------------------------------------------------------- fences and walls

def test_fence_ops_keep_the_bottom_on_the_terrain(proposal: OutdoorProposal, base: Any) -> None:
    _apply(proposal, "set_fence", "fence-0", height=1.6, thickness=0.06)
    p = _el(proposal, "fence-0").params
    axis = np.asarray(p.axis)
    assert (p.height, p.thickness) == (1.6, 0.06)
    assert axis[:, 2] - np.asarray(p.bottom_z) == pytest.approx(np.full(len(axis), 1.6))
    _apply(proposal, "offset_fence", "fence-0", distance_m=0.5)
    p = _el(proposal, "fence-0").params
    old = LineString([q[:2] for q in base.elements["fence-0"].params.axis])
    new = LineString([q[:2] for q in p.axis])
    assert new.hausdorff_distance(old) == pytest.approx(0.5, abs=0.02) and old.distance(new) == pytest.approx(0.5, abs=0.02)
    xy = np.asarray([q[:2] for q in p.axis])
    assert np.asarray(p.bottom_z) == pytest.approx(terrain_z(xy[:, 0], xy[:, 1]), abs=TIN_TOLERANCE_M)
    assert np.asarray(p.axis)[:, 2] - np.asarray(p.bottom_z) == pytest.approx(np.full(len(xy), 1.6))
    _apply(proposal, "move_fence_vertex", "fence-0", index=1, dx_m=-0.4)
    assert _el(proposal, "fence-0").params.axis[1][0] == pytest.approx(p.axis[1][0] - 0.4)
    for name, params in (("set_fence", {"height": 5.0}), ("offset_fence", {"distance_m": 9.0}), ("offset_fence", {"distance_m": 0.0}),
                         ("move_fence_vertex", {"index": 7, "dx_m": 0.2}), ("move_fence_vertex", {"index": 0})):
        with pytest.raises(EditError):
            _apply(proposal, name, "fence-0", **params)
    _apply(proposal, "delete_fence", "fence-0")
    assert "fence-0" not in proposal.plan.elements
    assert base.elements["fence-0"].params.height == FENCE_HEIGHT


def test_site_wall_ops(proposal: OutdoorProposal, base: Any) -> None:
    from shapely.geometry import Polygon

    _apply(proposal, "set_site_wall", "site-wall-0", height=1.4)
    p = _el(proposal, "site-wall-0").params
    assert np.asarray(p.top_z_axis) - np.asarray(p.base_z_axis) == pytest.approx(np.full(3, 1.4))
    assert element_height(_el(proposal, "site-wall-0")) == pytest.approx(p.top_z - p.base_z)
    area_before = Polygon(p.footprint).area
    _apply(proposal, "set_site_wall", "site-wall-0", thickness=0.45)
    p = _el(proposal, "site-wall-0").params
    assert p.thickness == 0.45 and Polygon(p.footprint).area == pytest.approx(area_before * 1.5, rel=0.05)
    _apply(proposal, "offset_site_wall", "site-wall-0", distance_m=-0.4)
    p = _el(proposal, "site-wall-0").params
    xy = np.asarray(p.axis)
    assert LineString(xy).distance(LineString(base.elements["site-wall-0"].params.axis)) == pytest.approx(0.4, abs=0.02)
    assert np.asarray(p.base_z_axis) == pytest.approx(terrain_z(xy[:, 0], xy[:, 1]), abs=TIN_TOLERANCE_M)
    assert np.asarray(p.top_z_axis) - np.asarray(p.base_z_axis) == pytest.approx(np.full(3, 1.4))
    assert Polygon(p.footprint).is_valid
    _apply(proposal, "move_site_wall_vertex", "site-wall-0", index=-1, dy_m=0.5)
    assert _el(proposal, "site-wall-0").params.axis[-1][1] == pytest.approx(p.axis[-1][1] + 0.5)
    for params in ({"height": 10.0}, {"thickness": 3.0}):
        with pytest.raises(EditError):
            _apply(proposal, "set_site_wall", "site-wall-0", **params)


# ------------------------------------------------------------------------------------------------------------ curbs

def test_curb_ops(proposal: OutdoorProposal, base: Any) -> None:
    _apply(proposal, "set_curb", "curb-0", height=0.16, width=0.15)
    p = _el(proposal, "curb-0").params
    assert np.asarray(p.axis)[:, 2] - np.asarray(p.foot_z) == pytest.approx(np.full(3, 0.16))
    assert p.width == 0.15 and element_height(_el(proposal, "curb-0")) == pytest.approx(0.16)
    _apply(proposal, "set_curb", "curb-0", gutter_width=0.3)
    p = _el(proposal, "curb-0").params
    assert p.gutter_width == 0.3 and _el(proposal, "curb-0").evidence["gutter"] == "manual"
    _apply(proposal, "set_curb", "curb-0", gutter_width=0.0)
    assert _el(proposal, "curb-0").params.gutter_width == 0.0 and _el(proposal, "curb-0").params.gutter_axis is None
    _apply(proposal, "offset_curb", "curb-0", distance_m=0.2)
    p = _el(proposal, "curb-0").params
    assert LineString([q[:2] for q in p.axis]).distance(LineString([q[:2] for q in base.elements["curb-0"].params.axis])) == pytest.approx(0.2, abs=0.02)
    assert all(math.isfinite(v) for q in p.axis for v in q)
    for params in ({"height": 0.5}, {"width": 0.01}, {"gutter_width": 2.0}):
        with pytest.raises(EditError):
            _apply(proposal, "set_curb", "curb-0", **params)
    _apply(proposal, "delete_curb", "curb-0")
    assert "curb-0" not in proposal.plan.elements


# ------------------------------------------------------------------------------------------------------ full tier

def test_full_tier_pavement_and_building_ops(proposal: OutdoorProposal, base: Any) -> None:
    pavement = next(e.id for e in base.elements.values() if e.kind.value == "pavement")
    _apply(proposal, "set_pavement_kind", pavement, kind="sidewalk")
    assert _el(proposal, pavement).params.kind == "sidewalk"
    with pytest.raises(EditError):
        _apply(proposal, "set_pavement_kind", pavement, kind="sidewalk")
    heights = building_heights(base.elements["building-1"])
    _apply(proposal, "set_building", "building-1", ridge_height=heights["ridge"] - heights["ground"] + 1.5)
    after = building_heights(_el(proposal, "building-1"))
    assert after["ridge"] == pytest.approx(heights["ridge"] + 1.5) and after["eave"] == pytest.approx(heights["eave"])
    assert after["ground"] == pytest.approx(heights["ground"])
    _apply(proposal, "set_building", "building-1", eave_height_delta=0.5)
    assert building_heights(_el(proposal, "building-1"))["eave"] == pytest.approx(heights["eave"] + 0.5)
    with pytest.raises(EditError):
        _apply(proposal, "set_building", "building-1", eave_height=BUILDING_RIDGE + 50.0, ridge_height=3.0)
    _apply(proposal, "delete_pavement", pavement)
    _apply(proposal, "delete_building", "building-1")
    assert pavement not in proposal.plan.elements and "building-1" not in proposal.plan.elements
    assert "building-1" not in proposal.plan.buildings and "building-1" in base.buildings
    assert not proposal.plan.check_references()


def test_full_tier_terrain_adjustment_moves_the_dependents(proposal: OutdoorProposal, base: Any) -> None:
    x, y = TREE_2["xy"]
    ring = [list(lv95(x - 3.0, y - 3.0)), list(lv95(x + 3.0, y - 3.0)), list(lv95(x + 3.0, y + 3.0)), list(lv95(x - 3.0, y + 3.0))]
    changes = _apply(proposal, "adjust_terrain", None, polygon=ring, dz_m=0.4, taper_m=0.5)
    assert {c.element_id for c in changes} >= {"terrain-0", TREE_2["id"]}
    before = np.asarray(base.elements["terrain-0"].params.vertices)
    after = np.asarray(_el(proposal, "terrain-0").params.vertices)
    inner = (np.abs(before[:, 0] - x) < 2.0) & (np.abs(before[:, 1] - y) < 2.0)
    outer = (np.abs(before[:, 0] - x) > 3.5) | (np.abs(before[:, 1] - y) > 3.5)
    assert inner.any() and after[inner, 2] - before[inner, 2] == pytest.approx(np.full(int(inner.sum()), 0.4))
    assert after[outer, 2] == pytest.approx(before[outer, 2])
    tree = _el(proposal, TREE_2["id"]).params
    assert tree.base_z - base.elements[TREE_2["id"]].params.base_z == pytest.approx(0.4, abs=0.02)
    assert tree.height == base.elements[TREE_2["id"]].params.height
    with pytest.raises(EditError):
        _apply(proposal, "adjust_terrain", None, polygon=ring, dz_m=9.0)
    with pytest.raises(EditError):
        _apply(proposal, "adjust_terrain", None, polygon=ring)
    far = [list(lv95(500.0, 500.0)), list(lv95(501.0, 500.0)), list(lv95(501.0, 501.0))]
    with pytest.raises(EditError) as error:
        _apply(proposal, "adjust_terrain", None, polygon=far, dz_m=0.2)
    assert error.value.key == "assistant.outdoor.no_match"


def test_bulk_operation_is_all_or_nothing(proposal: OutdoorProposal, base: Any) -> None:
    changes = _apply(proposal, BULK_OP, None, category="tree", op="set_tree", params={"height_delta": 0.5})
    assert {c.element_id for c in changes} == {TREE_1["id"], TREE_2["id"]}
    assert [op.op for op in proposal.ops] == ["set_tree", "set_tree"]  # stored per element (re-applied by signature)
    assert _el(proposal, TREE_1["id"]).params.height == pytest.approx(base.elements[TREE_1["id"]].params.height + 0.5)
    state = (proposal.plan, len(proposal.ops))
    with pytest.raises(EditError):
        _apply(proposal, BULK_OP, None, category="tree", op="set_tree", params={"height_delta": 35.0})
    assert (proposal.plan, len(proposal.ops)) == state
    with pytest.raises(EditError) as error:
        _apply(proposal, BULK_OP, None, category="tree", op="set_tree", filter={"min_height": 30.0}, params={"height": 5.0})
    assert error.value.key == "assistant.outdoor.no_match"
    with pytest.raises(EditError):
        _apply(proposal, BULK_OP, None, category="fence", op="set_tree", params={"height": 5.0})
    e0, n0 = lv95(TREE_1["xy"][0] - 1.0, TREE_1["xy"][1] - 1.0)
    e1, n1 = lv95(TREE_1["xy"][0] + 1.0, TREE_1["xy"][1] + 1.0)
    changes = _apply(proposal, BULK_OP, None, category="tree", op="delete_tree", filter={"box": {"e": [e0, e1], "n": [n0, n1]}})
    assert [c.element_id for c in changes] == [TREE_1["id"]]


def test_internal_params_and_unknown_ops(proposal: OutdoorProposal) -> None:
    with pytest.raises(EditError) as error:
        _apply(proposal, "paint_tree", TREE_1["id"])
    assert error.value.key == "assistant.edit.unknown_op"
    _apply(proposal, "set_tree", TREE_1["id"], height=9.0, _element={"id": "forged"})
    assert "_element" not in proposal.ops[-1].params


def test_tiers_expose_the_right_ops(domain: OutdoorDomain) -> None:
    elements = {spec.name for spec in ops_for_tier("elements")}
    full = {spec.name for spec in ops_for_tier("full")}
    assert elements < full and full - elements == {"delete_pavement", "set_pavement_kind", "delete_building", "set_building",
                                                    "adjust_terrain", BULK_OP}
    assert {name for name, spec in OPS.items() if spec.expensive} == {"adjust_terrain", BULK_OP}
    assert not ({tool.name for tool in domain.tools("elements")} & (full - elements))
    assert full <= {tool.name for tool in domain.tools("full")}
    for tool in domain.tools("full"):
        assert tool.parameters["type"] == "object" and tool.parameters.get("additionalProperties") is False


def test_signatures_find_renumbered_and_moved_elements(base: Any) -> None:
    sig = signature_of(base, "fence-0")
    renamed = base.model_copy(update={"elements": {("fence-9" if k == "fence-0" else k): (v.model_copy(update={"id": "fence-9"}) if k == "fence-0" else v)
                                                   for k, v in base.elements.items()}})
    assert resolve_signature(renamed, sig, "fence-0", strict=True) == "fence-9"
    tree_sig = signature_of(base, TREE_1["id"])
    assert resolve_signature(base, tree_sig, TREE_1["id"], strict=True) == TREE_1["id"]
    gone = base.model_copy(update={"elements": {k: v for k, v in base.elements.items() if k != TREE_1["id"]}})
    assert resolve_signature(gone, tree_sig, TREE_1["id"], strict=True) is None
    twin = base.elements[TREE_1["id"]].model_copy(update={"id": "twin"})
    doubled = base.model_copy(update={"elements": {**gone.elements, "twin": twin, "twin2": twin.model_copy(update={"id": "twin2"})}})
    with pytest.raises(EditError) as error:
        resolve_signature(doubled, tree_sig, TREE_1["id"], strict=True)
    assert error.value.key == "assistant.history.ambiguous"
