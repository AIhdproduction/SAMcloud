"""AI assistant edit engine: every op changes the plan correctly and its dependencies follow (synthetic plan)."""

from __future__ import annotations

import copy
import json

import pytest
from shapely.geometry import Point, Polygon

from samcloud.assistant.edits import OPS, EditError, EditOp, Proposal, apply_ops, diff_plans, op_schemas, resolve_signature, signature_of
from assistant_synthetic import make_plan

AREA_ROOM = 3.75 * 4.7


def _room(plan, room_id):
    return next(r for r in plan.rooms if r.id == room_id)


def _wall(plan, wall_id):
    return next(w for w in plan.walls if w.id == wall_id)


def _opening(plan, opening_id):
    return next(o for o in plan.openings if o.id == opening_id)


def _furniture(plan, furniture_id):
    return next(f for f in plan.furniture if f.id == furniture_id)


def _flat(points):
    return [float(value) for point in points for value in point]


def _x_extent(room):
    xs = [p[0] for p in room.polygon["exterior"]]
    return min(xs), max(xs)


def test_move_wall_offset_moves_openings_rooms_and_faces() -> None:
    base = make_plan()
    before = copy.deepcopy(base.to_dict())
    proposal = Proposal(base)
    changes = proposal.apply(EditOp("move_wall", "W005", {"offset_m": 0.5}))
    plan = proposal.plan

    assert base.to_dict() == before  # the base plan is never mutated
    wall = _wall(plan, "W005")
    assert _flat(wall.centreline) == pytest.approx([4.5, 0.0, 4.5, 5.0])
    assert [p[0] for p in wall.face_a] == pytest.approx([4.6, 4.6])  # + = towards face_a (right of the direction, +x here)
    assert [p[0] for p in wall.face_b] == pytest.approx([4.4, 4.4])
    assert _opening(plan, "O001").center == pytest.approx([4.5, 2.5])
    r1, r2 = _room(plan, "R001"), _room(plan, "R002")
    assert _x_extent(r1) == pytest.approx((0.15, 4.4))
    assert _x_extent(r2) == pytest.approx((4.6, 7.85))
    assert r1.area_m2 == pytest.approx(4.25 * 4.7)
    assert r2.area_m2 == pytest.approx(3.25 * 4.7)
    for room in (r1, r2):
        assert Polygon(room.polygon["exterior"]).contains(Point(room.label_point))
    assert _wall(plan, "W005").detected_centreline is None  # as-detected fields untouched
    kinds = {(c.element_id, c.change) for c in changes}
    assert {("W005", "moved"), ("O001", "moved"), ("R001", "modified"), ("R002", "modified")} <= kinds
    assert next(c for c in changes if c.element_id == "O001").note == "follows host wall W005"
    # the other walls stay (the partition ends inside the exterior walls' band)
    assert _wall(plan, "W001").centreline == base.walls[0].centreline


def test_move_wall_dx_dy_and_wall_bodies() -> None:
    proposal = Proposal(make_plan())
    proposal.apply(EditOp("move_wall", "W005", {"dx_m": -0.3, "dy_m": 0.0}))
    plan = proposal.plan
    assert _flat(_wall(plan, "W005").centreline) == pytest.approx([3.7, 0.0, 3.7, 5.0])
    assert _room(plan, "R001").area_m2 == pytest.approx(3.45 * 4.7)
    body = Polygon(plan.wall_bodies[0]["exterior"], plan.wall_bodies[0]["holes"])
    assert body.contains(Point(3.7, 1.0)) and not body.contains(Point(4.0, 1.0))


def test_move_exterior_wall_extends_joining_walls() -> None:
    proposal = Proposal(make_plan())
    changes = proposal.apply(EditOp("move_wall", "W002", {"offset_m": -0.5}))  # face_a of W002 is outside (+x): inward
    plan = proposal.plan
    assert _flat(_wall(plan, "W002").centreline) == pytest.approx([7.5, 0.0, 7.5, 5.0])
    assert _wall(plan, "W001").centreline[1] == pytest.approx([7.5, 0.0])
    assert _wall(plan, "W003").centreline[0] == pytest.approx([7.5, 5.0])
    assert _room(plan, "R002").area_m2 == pytest.approx((7.35 - 4.1) * 4.7)
    assert {"W001", "W003"} <= {c.element_id for c in changes if c.note.startswith("end follows")}


def test_set_wall_thickness_centre_and_face_anchor() -> None:
    proposal = Proposal(make_plan())
    proposal.apply(EditOp("set_wall_thickness", "W005", {"thickness_m": 0.30}))
    plan = proposal.plan
    assert _wall(plan, "W005").thickness_m == pytest.approx(0.30)
    assert _flat(_wall(plan, "W005").centreline) == pytest.approx([4.0, 0.0, 4.0, 5.0])
    assert _opening(plan, "O001").wall_thickness_m == pytest.approx(0.30)
    assert _x_extent(_room(plan, "R001"))[1] == pytest.approx(3.85)
    assert _x_extent(_room(plan, "R002"))[0] == pytest.approx(4.15)

    proposal = Proposal(make_plan())
    proposal.apply(EditOp("set_wall_thickness", "W005", {"thickness_m": 0.40, "anchor": "face_a"}))
    wall = _wall(proposal.plan, "W005")
    assert [p[0] for p in wall.face_a] == pytest.approx([4.1, 4.1])  # face_a stays
    assert wall.centreline[0][0] == pytest.approx(3.9)
    assert _opening(proposal.plan, "O001").center[0] == pytest.approx(3.9)
    assert _x_extent(_room(proposal.plan, "R001"))[1] == pytest.approx(3.7)
    assert _x_extent(_room(proposal.plan, "R002"))[0] == pytest.approx(4.1)


def test_delete_wall_deletes_hosted_opening_with_note() -> None:
    proposal = Proposal(make_plan())
    changes = proposal.apply(EditOp("delete_wall", "W005", {}))
    assert "W005" not in {w.id for w in proposal.plan.walls}
    assert "O001" not in {o.id for o in proposal.plan.openings}
    removed = {c.element_id: c for c in changes}
    assert removed["W005"].change == "removed"
    assert removed["O001"].change == "removed" and "host wall W005" in removed["O001"].note
    assert proposal.needs_confirmation()


def test_set_opening_values_and_shift() -> None:
    proposal = Proposal(make_plan())
    proposal.apply(EditOp("set_opening", "O001", {"width_m": 1.1, "head_m": 2.2}))
    proposal.apply(EditOp("set_opening", "O001", {"shift_m": 0.5}))
    proposal.apply(EditOp("set_opening", "O002", {"sill_m": 0.8, "width_m": 1.6}))
    door, window = _opening(proposal.plan, "O001"), _opening(proposal.plan, "O002")
    assert door.width_m == pytest.approx(1.1) and door.head_m == pytest.approx(2.2)
    assert door.center == pytest.approx([4.0, 3.0])  # along the host wall (+y direction of W005)
    assert window.sill_m == pytest.approx(0.8) and window.width_m == pytest.approx(1.6)
    assert not proposal.needs_confirmation()


@pytest.mark.parametrize(
    ("target", "params", "key"),
    [
        ("O001", {"width_m": 0.2}, "assistant.edit.invalid_value"),
        ("O001", {"width_m": 7.0}, "assistant.edit.invalid_value"),
        ("O001", {"sill_m": 0.5}, "assistant.edit.invalid_value"),  # a door has no sill
        ("O002", {"head_m": 0.8}, "assistant.edit.invalid_value"),  # below the window sill
        ("O002", {"head_m": 3.0}, "assistant.edit.invalid_value"),  # above the storey height (2.6 m)
        ("O002", {"shift_m": 6.0}, "assistant.edit.opening_outside_wall"),
        ("O001", {"shift_m": 2.4}, "assistant.edit.opening_outside_wall"),
        ("O009", {"width_m": 1.0}, "assistant.edit.not_found"),
        ("O001", {}, "assistant.edit.bad_params"),
    ],
)
def test_set_opening_rejects_invalid_values(target: str, params: dict, key: str) -> None:
    proposal = Proposal(make_plan())
    with pytest.raises(EditError) as raised:
        proposal.apply(EditOp("set_opening", target, params))
    assert raised.value.key == key
    assert proposal.ops == [] and proposal.plan.to_dict() == make_plan().to_dict()


@pytest.mark.parametrize(
    ("op", "target", "params", "key"),
    [
        ("set_wall_thickness", "W005", {"thickness_m": 0.02}, "assistant.edit.invalid_value"),
        ("set_wall_thickness", "W005", {"thickness_m": 1.5}, "assistant.edit.invalid_value"),
        ("set_wall_thickness", "W005", {"thickness_m": 0.2, "anchor": "middle"}, "assistant.edit.bad_params"),
        ("move_wall", "W005", {}, "assistant.edit.bad_params"),
        ("move_wall", "W005", {"offset_m": 5.0}, "assistant.edit.invalid_value"),
        ("move_wall", "W005", {"offset_m": 0.2, "dx_m": 0.1}, "assistant.edit.bad_params"),
        ("move_wall", "W099", {"offset_m": 0.2}, "assistant.edit.not_found"),
        ("move_wall", "O001", {"offset_m": 0.2}, "assistant.edit.not_found"),
        ("move_wall", "W005", {"offset_m": "far"}, "assistant.edit.bad_params"),
        ("move_wall", "W005", {"offset_m": 0.2, "colour": "red"}, "assistant.edit.bad_params"),
        ("fly_away", "W005", {}, "assistant.edit.unknown_op"),
        ("rename_room", "R001", {"name": "   "}, "assistant.edit.invalid_value"),
        ("rotate_furniture", "F001", {"angle_deg": 0.0}, "assistant.edit.invalid_value"),
        ("set_furniture_class", "F001", {"kind": "spaceship"}, "assistant.edit.unknown_kind"),
        ("move_furniture", "F001", {"dx_m": 0.0, "dy_m": 0.0}, "assistant.edit.invalid_value"),
    ],
)
def test_invalid_edits_raise(op: str, target: str, params: dict, key: str) -> None:
    proposal = Proposal(make_plan())
    with pytest.raises(EditError) as raised:
        proposal.apply(EditOp(op, target, params))
    assert raised.value.key == key
    assert raised.value.to_dict()["message"]


def test_moving_a_wall_shorter_than_its_door_fails() -> None:
    plan = make_plan()
    proposal = Proposal(plan)
    # W001 (south, face_a outside) moved 2.8 m north: W005 then starts at y=2.8, its door (y=2.5) would hang outside
    with pytest.raises(EditError) as raised:
        proposal.apply(EditOp("move_wall", "W001", {"offset_m": -2.8}))
    assert raised.value.key == "assistant.edit.opening_outside_wall"
    assert proposal.plan.to_dict() == plan.to_dict()


def test_furniture_ops() -> None:
    proposal = Proposal(make_plan())
    proposal.apply(EditOp("move_furniture", "F001", {"dx_m": 0.5, "dy_m": 1.0}))
    table = _furniture(proposal.plan, "F001")
    assert Polygon(table.footprint).centroid.coords[0] == pytest.approx((2.1, 2.4))
    proposal.apply(EditOp("rotate_furniture", "F001", {"angle_deg": 90.0}))
    table = _furniture(proposal.plan, "F001")
    xs, ys = [p[0] for p in table.footprint], [p[1] for p in table.footprint]
    assert max(xs) - min(xs) == pytest.approx(0.8) and max(ys) - min(ys) == pytest.approx(1.2)
    assert Polygon(table.footprint).centroid.coords[0] == pytest.approx((2.1, 2.4))
    proposal.apply(EditOp("set_furniture_class", "F001", {"kind": "desk"}))
    table = _furniture(proposal.plan, "F001")
    assert (table.label, table.semantic_label, table.layer) == ("Schreibtisch", "desk", "A-FURN")
    proposal.apply(EditOp("set_furniture_class", "F001", {"kind": "WC"}))
    assert _furniture(proposal.plan, "F001").layer == "P-SANI"
    proposal.apply(EditOp("rename_room", "R002", {"name": "  Gast  zimmer "}))
    assert _room(proposal.plan, "R002").name == "Gast zimmer"
    proposal.apply(EditOp("delete_furniture", "F001", {}))
    assert proposal.plan.furniture == []
    summary = proposal.summary()
    assert [entry["element_id"] for entry in summary] == ["F001", "R002"]
    assert summary[0]["change"] == "removed" and summary[0]["before"]["label"] == "Tisch"


def test_delete_opening_and_diff() -> None:
    base = make_plan()
    proposal = apply_ops(base, [EditOp("delete_opening", "O002", {}), {"op": "rename_room", "target": "R001", "params": {"name": "Küche"}}])
    changes = {(c.element_id, c.change) for c in diff_plans(base, proposal.plan)}
    assert changes == {("O002", "removed"), ("R001", "modified")}
    assert proposal.needs_confirmation()  # a removal


def test_op_schemas_are_json_and_cover_every_op() -> None:
    schemas = op_schemas()
    assert set(schemas) == set(OPS) == {
        "move_wall", "set_wall_thickness", "delete_wall", "set_opening", "delete_opening", "add_furniture", "move_furniture",
        "rotate_furniture", "delete_furniture", "set_furniture_class", "rename_room",
    }
    json.dumps(schemas)
    op = EditOp.from_tool_call("move_wall", {"target": "W005", "offset_m": 0.1})
    assert (op.target, op.params) == ("W005", {"offset_m": 0.1})


def test_signatures_find_elements_again() -> None:
    plan = make_plan()
    from assistant_synthetic import renumbered_recompute

    recomputed = renumbered_recompute(plan)
    for element_id, expected in (("W005", "W001"), ("O001", "O002"), ("R002", "R001"), ("F001", "F011")):
        assert resolve_signature(recomputed, signature_of(plan, element_id), preferred_id=element_id) == expected
    moved = copy.deepcopy(recomputed)
    for wall in moved.walls:
        wall.centreline = [[x + 1.0, y] for x, y in wall.centreline]
    assert resolve_signature(moved, signature_of(plan, "W005")) is None  # 1 m across: not the same wall any more


def test_collinear_pieces_split_at_a_door_move_together() -> None:
    from assistant_synthetic import _wall as synthetic_wall

    plan = make_plan()
    plan.walls = [w for w in plan.walls if w.id != "W005"] + [
        synthetic_wall("W005", (4.0, 0.0), (4.0, 2.05), 0.20, "interior"),
        synthetic_wall("W006", (4.0, 2.95), (4.0, 5.0), 0.20, "interior"),
    ]
    proposal = Proposal(plan)
    changes = proposal.apply(EditOp("move_wall", "W006", {"offset_m": 0.3}))
    assert _flat(_wall(proposal.plan, "W005").centreline) == pytest.approx([4.3, 0.0, 4.3, 2.05])
    assert _flat(_wall(proposal.plan, "W006").centreline) == pytest.approx([4.3, 2.95, 4.3, 5.0])
    assert _opening(proposal.plan, "O001").center == pytest.approx([4.3, 2.5])
    assert next(c for c in changes if c.element_id == "W005").note == "collinear with W006"
    assert _room(proposal.plan, "R001").area_m2 == pytest.approx(4.05 * 4.7)
    single = Proposal(plan)
    single.apply(EditOp("move_wall", "W006", {"offset_m": 0.3, "chain": False}))
    assert _flat(_wall(single.plan, "W005").centreline) == pytest.approx([4.0, 0.0, 4.0, 2.05])


def test_long_collinear_chain_needs_confirmation() -> None:
    from assistant_synthetic import _wall as synthetic_wall

    plan = make_plan()
    plan.walls = [w for w in plan.walls if w.id != "W005"] + [
        synthetic_wall("W005", (4.0, 0.0), (4.0, 1.5), 0.20, "interior"),
        synthetic_wall("W006", (4.0, 1.52), (4.0, 2.05), 0.20, "interior"),
        synthetic_wall("W007", (4.0, 2.95), (4.0, 5.0), 0.20, "interior"),
    ]
    plan.openings[0].host_wall_id = "W006"  # the door bridges W006 and W007
    proposal = Proposal(plan)
    changes = proposal.apply(EditOp("move_wall", "W007", {"offset_m": 0.1}))
    assert {c.element_id for c in changes if c.kind == "wall"} >= {"W005", "W006", "W007"}
    assert "with 2 collinear pieces" in changes[0].note
    assert proposal.confirm_reasons and any("3 pieces" in note for note in proposal.notes)
    assert proposal.needs_confirmation()
    entry = next(e for e in proposal.summary() if e["element_id"] == "W007")
    assert any("collinear" in note for note in entry["notes"])
    short = Proposal(make_plan())
    short.apply(EditOp("move_wall", "W005", {"offset_m": 0.1}))
    assert not short.needs_confirmation()
