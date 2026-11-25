"""Full control (tool tier "full") of the indoor assistant: settings defaults, tool lists, the full-tier edit operations,
replay after a recompute and the DXF patch of the new operations.  Synthetic data only, no network."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest

from assistant_synthetic import LEVEL_ID, make_plan, make_project, make_result, renumbered_recompute
from samcloud.assistant import settings as ai_settings
from samcloud.assistant.domains.base import TIER_ELEMENTS, TIER_FULL
from samcloud.assistant.domains.indoor import IndoorDomain
from samcloud.assistant.domains.registry import resolve_tier
from samcloud.assistant.edits import OPS, EditError, EditOp, Proposal, diff_plans, op_schemas, op_spec
from samcloud.assistant.edits_full import FULL_OP_NAMES, FULL_OPS
from samcloud.assistant.history import History, ensure_applied
from samcloud.assistant.paths import plan_model_path
from samcloud.assistant.tools import tool_names, tool_specs
from samcloud.bim.plan_model import load_plan_model, save_plan_model
from test_assistant_agent import settings_for

PHASE1_EDIT_TOOLS = ["move_wall", "set_wall_thickness", "delete_wall", "set_opening", "delete_opening", "add_furniture",
                     "move_furniture", "rotate_furniture", "delete_furniture", "set_furniture_class", "rename_room"]
READ_TOOLS = ["list_elements", "element_details", "points_in_box", "section", "fit_surface"]


@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the real user's AI settings / credentials."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _apply(plan: Any, *ops: tuple[str, str | None, dict[str, Any]]) -> Proposal:
    proposal = Proposal(plan)
    for name, target, params in ops:
        proposal.apply(EditOp(name, target, params))
    return proposal


def _room(plan: Any, room_id: str) -> Any:
    return next(room for room in plan.rooms if room.id == room_id)


def _wall(plan: Any, wall_id: str) -> Any:
    return next(wall for wall in plan.walls if wall.id == wall_id)


def _with_ceiling_object(plan: Any) -> Any:
    """``plan`` with one ceiling lamp C001 (0.4 m square around (2, 2), 2.70-2.80 m absolute)."""
    x0, x1, y0, y1, z0, z1 = 1.8, 2.2, 1.8, 2.2, 2.70, 2.80
    vertices = [[x, y, z] for z in (z0, z1) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
             [3, 0, 4], [3, 4, 7]]
    plan.ceiling_objects = [{"id": "C001", "label": "Deckenleuchte", "semantic_label": "ceiling lamp",
                             "footprint": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "z_min": z0, "z_max": z1,
                             "z_source": "points", "mesh": {"vertices": vertices, "faces": faces}}]
    return plan


# ----------------------------------------------------------------------------------------------- settings / tiers

@pytest.mark.parametrize("provider,expected", [("ollama", False), ("anthropic", True), ("openai", True), ("openrouter", True)])
def test_default_per_provider_and_explicit_choice(provider: str, expected: bool) -> None:
    assert ai_settings.effective_full_control(ai_settings.AssistantSettings(provider=provider)) is expected
    assert ai_settings.effective_full_control(ai_settings.AssistantSettings(provider=provider, full_control=not expected)) is (not expected)


def test_explicit_choice_persists_across_provider_switch() -> None:
    settings = ai_settings.AssistantSettings(provider="openai", full_control=False)
    ai_settings.save_settings(settings)
    loaded = ai_settings.load_settings()
    loaded.provider = "anthropic"
    ai_settings.save_settings(loaded)
    again = ai_settings.load_settings()
    assert again.full_control is False and ai_settings.effective_full_control(again) is False
    again.provider, again.full_control = "ollama", True
    ai_settings.save_settings(again)
    final = ai_settings.load_settings()
    final.provider = "openrouter"
    ai_settings.save_settings(final)
    assert ai_settings.load_settings().full_control is True


def test_indoor_tiers_and_tool_lists(tmp_path: Path) -> None:
    domain = IndoorDomain(make_project(tmp_path, with_cloud=False), LEVEL_ID)
    assert domain.supported_tiers == (TIER_ELEMENTS, TIER_FULL)
    assert resolve_tier(domain, settings_for("openai")) == TIER_FULL
    assert resolve_tier(domain, settings_for("ollama")) == TIER_ELEMENTS
    assert resolve_tier(domain, settings_for("ollama", full_control=True)) == TIER_FULL
    assert resolve_tier(domain, settings_for("anthropic", full_control=False)) == TIER_ELEMENTS
    # elements tier: exactly the phase 1 tools; full tier: the same plus the full-control operations
    assert [s.name for s in tool_specs(domain, TIER_ELEMENTS)] == READ_TOOLS + PHASE1_EDIT_TOOLS
    assert tool_names() == READ_TOOLS + PHASE1_EDIT_TOOLS
    assert [s.name for s in tool_specs(domain, TIER_FULL)] == READ_TOOLS + PHASE1_EDIT_TOOLS + list(FULL_OP_NAMES)
    assert set(FULL_OP_NAMES) == {"add_wall", "set_room_polygon", "split_room", "merge_rooms", "move_opening_to_wall",
                                  "move_ceiling_object", "delete_ceiling_object", "set_ceiling_object_height", "apply_to_kind",
                                  "rename_rooms_bulk"}
    assert set(op_schemas()) == set(OPS) and not set(OPS) & set(FULL_OPS)  # OPS / op_schemas unchanged
    assert op_spec("add_wall") is FULL_OPS["add_wall"] and op_spec("move_wall") is OPS["move_wall"] and op_spec("x") is None


def test_full_tier_tool_rejected_on_the_elements_tier(tmp_path: Path) -> None:
    from samcloud.assistant.providers.base import ToolCall
    from samcloud.assistant.tools import dispatch

    domain = IndoorDomain(make_project(tmp_path, with_cloud=False), LEVEL_ID)
    ctx = domain.tool_context(domain.load_base(), tier=TIER_ELEMENTS)
    result = dispatch(ctx, ToolCall("1", "rename_rooms_bulk", {"names": [{"id": "R001", "name": "Küche"}]}))
    assert result.error and result.error_key == "assistant.agent.unknown_tool" and ctx.proposal.summary() == []
    full = domain.tool_context(domain.load_base(), tier=TIER_FULL)
    result = dispatch(full, ToolCall("2", "rename_rooms_bulk", {"names": [{"id": "R001", "name": "Küche"}]}))
    assert not result.error and [e["element_id"] for e in full.proposal.summary()] == ["R001"]


# ------------------------------------------------------------------------------------------------------- add_wall

def test_add_wall_joins_splits_room_and_updates_bodies() -> None:
    plan = make_plan()
    proposal = _apply(plan, ("add_wall", None, {"start": [4.05, 3.5], "end": [7.9, 3.5], "thickness_m": 0.10}))
    edited = proposal.plan
    wall = _wall(edited, "W006")
    assert wall.centreline == [[4.0, 3.5], [8.0, 3.5]]  # both ends joined onto the centrelines of W005 / W002
    assert wall.kind == "interior" and wall.thickness_m == pytest.approx(0.10) and wall.face_a and wall.face_b
    lower, upper = _room(edited, "R002"), _room(edited, "R003")
    assert lower.area_m2 == pytest.approx(3.75 * 3.30, abs=1e-6) and upper.area_m2 == pytest.approx(3.75 * 1.30, abs=1e-6)
    assert upper.name == "Schlafen 2" and lower.name == "Schlafen"
    assert {(e["element_id"], e["change"]) for e in proposal.summary()} == {("W006", "added"), ("R002", "modified"), ("R003", "added")}
    assert edited.wall_bodies and len(edited.wall_bodies) == len(plan.wall_bodies)
    assert make_plan().to_dict() == plan.to_dict()  # base untouched
    assert any("joined" in note for note in proposal.notes)


def test_add_wall_refuses_duplicates_blocked_openings_and_bad_values() -> None:
    plan = make_plan()
    cases = [
        ({"start": [4.0, 0.5], "end": [4.0, 4.5], "thickness_m": 0.2}, "assistant.edit.overlap"),
        ({"start": [4.05, 2.5], "end": [7.9, 2.5], "thickness_m": 0.1}, "assistant.edit.blocks_opening"),
        ({"start": [5.0, 1.0], "end": [5.0, 1.05], "thickness_m": 0.1}, "assistant.edit.invalid_value"),
        ({"start": [5.0, 1.0], "end": [5.0, 3.0], "thickness_m": 2.0}, "assistant.edit.invalid_value"),
        ({"start": [5.0, 1.0], "end": [5.0, 3.0], "thickness_m": 0.1, "wall_kind": "glass"}, "assistant.edit.bad_params"),
        ({"start": [5.0], "end": [5.0, 3.0], "thickness_m": 0.1}, "assistant.edit.bad_params"),
    ]
    for params, key in cases:
        proposal = Proposal(plan)
        with pytest.raises(EditError) as raised:
            proposal.apply(EditOp("add_wall", None, params))
        assert raised.value.key == key, params
        assert proposal.summary() == []


def test_add_wall_hosts_an_unhosted_opening() -> None:
    plan = make_plan()
    door = next(o for o in plan.openings if o.id == "O001")
    door.host_wall_id = None
    door.center = [6.0, 3.5]
    door.axis_angle = 0.0
    proposal = _apply(plan, ("add_wall", None, {"start": [4.05, 3.5], "end": [7.9, 3.5], "thickness_m": 0.12}))
    hosted = next(o for o in proposal.plan.openings if o.id == "O001")
    assert hosted.host_wall_id == "W006" and hosted.wall_thickness_m == pytest.approx(0.12)


# ----------------------------------------------------------------------------------------------------------- rooms

def test_set_room_polygon_split_and_overlap() -> None:
    plan = make_plan()
    proposal = _apply(plan, ("set_room_polygon", "R001", {"outline": [[0.15, 0.15], [3.0, 0.15], [3.0, 4.85], [0.15, 4.85]]}))
    assert _room(proposal.plan, "R001").area_m2 == pytest.approx(2.85 * 4.7, abs=1e-6)
    # drawn up to the wall centrelines: clipped to the wall faces
    clipped = _apply(plan, ("set_room_polygon", "R001", {"outline": [[0.0, 0.0], [4.0, 0.0], [4.0, 5.0], [0.0, 5.0]]}))
    assert _room(clipped.plan, "R001").area_m2 == pytest.approx(3.75 * 4.7, abs=1e-6)

    split = _apply(plan, ("split_room", "R002", {"line": [[4.1, 2.0], [7.85, 2.0]], "name": "Bad"}))
    upper, lower = _room(split.plan, "R002"), _room(split.plan, "R003")
    assert upper.area_m2 == pytest.approx(3.75 * 2.85, abs=1e-6) and lower.area_m2 == pytest.approx(3.75 * 1.85, abs=1e-6)
    assert lower.name == "Bad" and upper.name == "Schlafen"
    with pytest.raises(EditError) as raised:
        split.apply(EditOp("set_room_polygon", "R003", {"outline": [[4.1, 0.15], [7.85, 0.15], [7.85, 4.85], [4.1, 4.85]]}))
    assert raised.value.key == "assistant.edit.overlap"
    with pytest.raises(EditError) as raised:
        Proposal(plan).apply(EditOp("split_room", "R002", {"line": [[9.0, 1.0], [9.5, 1.0]]}))
    assert raised.value.key == "assistant.edit.split_failed"


def test_merge_rooms_needs_the_wall_removed() -> None:
    plan = make_plan()
    with pytest.raises(EditError) as raised:
        Proposal(plan).apply(EditOp("merge_rooms", "R001", {"rooms": ["R002"]}))
    assert raised.value.key == "assistant.edit.not_adjacent"
    proposal = _apply(plan, ("delete_wall", "W005", {}), ("merge_rooms", "R001", {"rooms": ["R002"], "name": "Wohnraum"}))
    merged = _room(proposal.plan, "R001")
    assert [room.id for room in proposal.plan.rooms] == ["R001"]
    assert merged.area_m2 == pytest.approx(7.7 * 4.7, abs=1e-3) and merged.name == "Wohnraum"
    changes = {(e["element_id"], e["change"]) for e in proposal.summary()}
    assert ("R002", "removed") in changes and ("R001", "modified") in changes


def test_rename_rooms_bulk() -> None:
    proposal = _apply(make_plan(), ("rename_rooms_bulk", None, {"names": [{"id": "R001", "name": "Küche"}, {"id": "R002", "name": "Bad"}]}))
    assert [(room.id, room.name) for room in proposal.plan.rooms] == [("R001", "Küche"), ("R002", "Bad")]
    with pytest.raises(EditError):
        Proposal(make_plan()).apply(EditOp("rename_rooms_bulk", None, {"names": [{"id": "R009", "name": "X"}]}))


# -------------------------------------------------------------------------------------------------------- openings

def test_move_opening_to_another_wall() -> None:
    plan = make_plan()
    proposal = _apply(plan, ("move_opening_to_wall", "O002", {"wall": "W003", "center": [6.0, 5.1]}))
    window = next(o for o in proposal.plan.openings if o.id == "O002")
    assert window.host_wall_id == "W003" and window.center == pytest.approx([6.0, 5.0])
    assert window.wall_thickness_m == pytest.approx(0.30) and math.cos(window.axis_angle) == pytest.approx(1.0)
    # it must fit and must not overlap another opening on the same wall
    with pytest.raises(EditError) as raised:
        Proposal(plan).apply(EditOp("move_opening_to_wall", "O002", {"wall": "W005", "center": [4.0, 2.6]}))
    assert raised.value.key == "assistant.edit.overlap"
    with pytest.raises(EditError):
        Proposal(plan).apply(EditOp("move_opening_to_wall", "O002", {"wall": "W009"}))


# ------------------------------------------------------------------------------------------------- ceiling objects

def test_ceiling_object_move_height_delete() -> None:
    plan = _with_ceiling_object(make_plan())
    moved = _apply(plan, ("move_ceiling_object", "C001", {"dx_m": 1.0, "dy_m": -0.5}))
    item = moved.plan.ceiling_objects[0]
    assert item["footprint"][0] == pytest.approx([2.8, 1.3]) and item["mesh"]["vertices"][0] == pytest.approx([2.8, 1.3, 2.7])
    lowered = _apply(plan, ("set_ceiling_object_height", "C001", {"top_m": 2.4}))
    item = lowered.plan.ceiling_objects[0]
    assert item["z_max"] == pytest.approx(0.20 + 2.4) and item["z_min"] == pytest.approx(0.20 + 2.3)
    assert max(v[2] for v in item["mesh"]["vertices"]) == pytest.approx(2.6)
    stretched = _apply(plan, ("set_ceiling_object_height", "C001", {"top_m": 2.6, "bottom_m": 2.2}))
    item = stretched.plan.ceiling_objects[0]
    assert (item["z_min"], item["z_max"]) == pytest.approx((2.4, 2.8))
    deleted = _apply(plan, ("delete_ceiling_object", "C001", {}))
    assert deleted.plan.ceiling_objects == [] and deleted.summary()[0]["change"] == "removed"
    with pytest.raises(EditError):
        Proposal(plan).apply(EditOp("set_ceiling_object_height", "C001", {}))


# ------------------------------------------------------------------------------------------------------------ bulk

def _two_interior_walls() -> Any:
    return _apply(make_plan(), ("add_wall", None, {"start": [4.05, 3.5], "end": [7.9, 3.5], "thickness_m": 0.10})).plan


def test_apply_to_kind_sets_every_interior_wall() -> None:
    plan = _two_interior_walls()
    proposal = _apply(plan, ("apply_to_kind", None, {"kind": "wall", "op": "set_wall_thickness", "params": {"thickness_m": 0.12},
                                                     "filter": {"wall_kind": "interior"}}))
    edited = proposal.plan
    assert [round(w.thickness_m, 3) for w in edited.walls if w.kind == "interior"] == [0.12, 0.12]
    assert [round(w.thickness_m, 3) for w in edited.walls if w.kind == "exterior"] == [0.3, 0.3, 0.3, 0.3]
    assert next(o for o in edited.openings if o.id == "O001").wall_thickness_m == pytest.approx(0.12)
    assert len(proposal.ops) == 1 and proposal.ops[0].op == "apply_to_kind"
    assert {"W005", "W006"} <= proposal.changed_ids()


def test_apply_to_kind_filters_and_errors() -> None:
    plan = make_plan()
    doors = _apply(plan, ("apply_to_kind", None, {"kind": "door", "op": "set_opening", "params": {"width_m": 1.0}}))
    assert [round(o.width_m, 2) for o in doors.plan.openings] == [1.0, 1.2]  # only the door
    by_ids = _apply(plan, ("apply_to_kind", None, {"kind": "opening", "op": "set_opening", "params": {"head_m": 2.2},
                                                   "filter": {"ids": ["O002"]}}))
    assert [round(o.head_m, 2) for o in by_ids.plan.openings] == [2.05, 2.2]
    for params, key in (
        ({"kind": "wall", "op": "set_wall_thickness", "params": {"thickness_m": 0.12}, "filter": {"wall_kind": "partition"}}, "assistant.edit.no_match"),
        ({"kind": "wall", "op": "rename_room", "params": {}}, "assistant.edit.bad_params"),
        ({"kind": "tree", "op": "move_wall", "params": {}}, "assistant.edit.bad_params"),
        ({"kind": "wall", "op": "set_wall_thickness", "params": {"thickness_m": 5.0}}, "assistant.edit.invalid_value"),
    ):
        proposal = Proposal(plan)
        with pytest.raises(EditError) as raised:
            proposal.apply(EditOp("apply_to_kind", None, params))
        assert raised.value.key == key, params
        assert proposal.summary() == [] and proposal.plan.to_dict() == plan.to_dict()


# ------------------------------------------------------------------------------------------- history and replay

def test_full_ops_replay_after_a_renumbering_recompute(tmp_path: Path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    path = plan_model_path(project, LEVEL_ID)
    history = History.load(project, LEVEL_ID)
    merge = _apply(load_plan_model(path), ("delete_wall", "W005", {}), ("merge_rooms", "R001", {"rooms": ["R002"], "name": "Wohnraum"}))
    history.accept(merge, prompt="ein Raum", provider="openai", model="fake")
    bulk = _apply(load_plan_model(path), ("apply_to_kind", None, {"kind": "wall", "op": "set_wall_thickness",
                                                                  "params": {"thickness_m": 0.28}, "filter": {"wall_kind": "exterior"}}))
    History.load(project, LEVEL_ID).accept(bulk, prompt="aussen 28", provider="openai", model="fake")
    # the pipeline recomputes the plan: new ids (rooms reversed: R002 <-> R001), a few millimetres off
    save_plan_model(renumbered_recompute(make_plan()), path.parent)
    report = ensure_applied(project, LEVEL_ID)
    assert report.changed and report.stale == [] and len(report.applied) == 2
    plan = load_plan_model(path)
    # one merged room; its edges followed the thinner exterior walls (each face 0.01 m outwards)
    assert len(plan.rooms) == 1 and plan.rooms[0].name == "Wohnraum" and plan.rooms[0].area_m2 == pytest.approx(7.72 * 4.72, abs=0.01)
    assert len(plan.walls) == 4 and all(w.thickness_m == pytest.approx(0.28) for w in plan.walls)
    assert not any(o.kind == "door" for o in plan.openings)


def test_move_opening_replays_with_its_wall_reference(tmp_path: Path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    path = plan_model_path(project, LEVEL_ID)
    proposal = _apply(load_plan_model(path), ("move_opening_to_wall", "O002", {"wall": "W004", "center": [0.0, 2.5]}))
    stored = proposal.ops[0].to_dict()
    assert "_refs" in stored["params"] and "W004" in stored["params"]["_refs"]
    History.load(project, LEVEL_ID).accept(proposal, prompt="Fenster", provider="openai", model="fake")
    save_plan_model(renumbered_recompute(make_plan()), path.parent)  # W004 -> W002, O002 -> O001
    report = ensure_applied(project, LEVEL_ID)
    assert report.stale == []
    plan = load_plan_model(path)
    window = next(o for o in plan.openings if o.kind == "window")
    west = min(plan.walls, key=lambda w: (w.centreline[0][0] + w.centreline[1][0]))
    assert window.host_wall_id == west.id and window.center[0] == pytest.approx(west.centreline[0][0])


# ----------------------------------------------------------------------------------------------------------- DXF

def test_dxf_patch_draws_new_rooms_and_moved_openings() -> None:
    from samcloud.assistant.dxf_patch import apply_dxf_patch

    plan = make_plan()
    result = make_result(plan)
    split = _apply(plan, ("add_wall", None, {"start": [4.05, 3.5], "end": [7.9, 3.5], "thickness_m": 0.10}))
    patched = apply_dxf_patch(result, plan, split.plan, diff_plans(plan, split.plan))
    names = [e.get("_room_name") for e in patched.layers["A-ANNO"] if e.get("_room_name")]
    assert sorted(names) == ["Schlafen", "Schlafen 2", "Wohnen"]
    assert len([e for e in patched.layers["A-AREA"] if e.get("type") == "polyline"]) == 3
    new_label = next(e for e in patched.layers["A-ANNO"] if e.get("_room_name") == "Schlafen 2")
    assert "4.88" in new_label["text"] or "4.9" in new_label["text"]
    from shapely.geometry import Point, Polygon
    mass = [Polygon(e["points"], e.get("holes") or []) for e in patched.layers["A-WALL"]]
    assert any(p.covers(Point(6.0, 3.5)) for p in mass)  # the new wall is in the wall mass

    moved = _apply(plan, ("move_opening_to_wall", "O002", {"wall": "W003", "center": [6.0, 5.0]}))
    patched = apply_dxf_patch(result, plan, moved.plan, diff_plans(plan, moved.plan))
    mass = [Polygon(e["points"], e.get("holes") or []) for e in patched.layers["A-WALL"]]
    assert any(p.covers(Point(2.0, 0.0)) for p in mass)  # the old void in the south wall is closed
    assert not any(p.covers(Point(6.0, 5.0)) for p in mass)  # the new void in the north wall is open
    untouched = apply_dxf_patch(result, plan, copy.deepcopy(plan), [])
    assert untouched is result
