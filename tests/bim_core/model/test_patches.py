"""Patch engine: validation, layer sync, origins, undo/redo, groups and log replay."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from samcloud.bim.core.model import (
    DoorElement,
    OpeningElement,
    Scene,
    SpaceElement,
    WallElement,
)
from samcloud.bim.core.model.graph import recompute
from samcloud.bim.core.model.patches import (
    AuthorKind,
    PatchEngine,
    PatchError,
    PatchRecord,
    PatchRejected,
)
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene


class Clock:
    """Deterministic clock: one second per call."""

    def __init__(self) -> None:
        self.now = datetime(2026, 11, 7, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def make_scene() -> Scene:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    recompute(scene)
    return scene


def wall_id(scene: Scene, tag: str) -> str:
    matches = [e.id for e in scene.elements.values() if isinstance(e, WallElement) and e.tag == tag]
    assert len(matches) == 1
    return matches[0]


def one_of(scene: Scene, cls: type) -> str:
    return next(e.id for e in scene.elements.values() if isinstance(e, cls))


def thickness_op(element_id: str, value: float) -> dict[str, Any]:
    return {"op": "replace", "path": f"/elements/{element_id}/params/thickness", "value": value}


def dump(scene: Scene) -> str:
    return scene.model_dump_json()


@pytest.fixture
def scene() -> Scene:
    return make_scene()


@pytest.fixture
def engine(scene: Scene, tmp_path: Path) -> PatchEngine:
    return PatchEngine(scene, tmp_path / "scene" / "patches.jsonl", Clock())


def log_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_bytes().decode("utf-8").split("\n")[:-1]


# ------------------------------------------------------------- thickness change


def test_thickness_change_updates_everything_and_undo_is_exact(
    scene: Scene, engine: PatchEngine
) -> None:
    west = wall_id(scene, "exterior_west")
    opening = next(
        e.id
        for e in scene.elements.values()
        if isinstance(e, OpeningElement) and e.tag == "door_west"
    )
    door = next(
        e.id for e in scene.elements.values() if isinstance(e, DoorElement) and e.tag == "door_west"
    )
    space = one_of(scene, SpaceElement)
    original = dump(scene)
    original_derived = json.dumps(scene.derived)
    assert scene.derived["spaces"][space]["area"] == pytest.approx(17.39, abs=1e-9)

    record = engine.apply(
        [thickness_op(west, 0.5)], author="reviewer", author_kind="user", reason="measured"
    )

    wall = scene.elements[west]
    assert isinstance(wall, WallElement)
    assert wall.params.thickness == 0.5
    assert wall.param_origin["thickness"].value == "manual"
    assert wall.provenance.value == "manual"
    assert len(wall.params.layers) == 1
    assert wall.params.layers[0].thickness == 0.5  # layer sync
    assert not wall.locked  # auto lock is off by default
    assert scene.derived["spaces"][space]["area"] == pytest.approx(17.02, abs=1e-9)
    assert scene.derived["elements"][opening]["geometry"]["depth"] == pytest.approx(0.5)
    assert {west, opening, door, space} <= set(record.changed_derived)
    assert engine.last_recompute is record.recompute
    # the extra operations are part of the record, derived values are not
    paths = [op["path"] for op in record.forward]
    assert f"/elements/{west}/params/layers/0/thickness" in paths
    assert f"/elements/{west}/param_origin/thickness" in paths
    assert not any(p.startswith("/derived") for p in paths + [o["path"] for o in record.inverse])
    assert record.author == "reviewer"
    assert record.author_kind is AuthorKind.USER
    assert record.reason == "measured"
    assert record.timestamp == "2026-11-07T12:00:01.000Z"
    assert len(record.id) == 36

    after = dump(scene)
    undone = engine.undo()
    assert undone is not None
    assert undone.action == "undo"
    assert undone.ref == record.id
    assert dump(scene) == original
    assert json.dumps(scene.derived) == original_derived
    assert scene.derived["spaces"][space]["area"] == pytest.approx(17.39, abs=1e-9)
    assert space in undone.changed_derived

    redone = engine.redo()
    assert redone is not None
    assert redone.action == "redo"
    assert redone.ref == record.id
    assert dump(scene) == after
    assert engine.redo() is None
    assert engine.undo() is not None
    assert engine.undo() is None
    assert dump(scene) == original


def test_apply_after_undo_clears_redo(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    north = wall_id(scene, "exterior_north")
    engine.apply([thickness_op(west, 0.4)], "u", "user")
    engine.undo()
    assert engine.can_redo
    engine.apply([thickness_op(north, 0.35)], "u", "user")
    assert not engine.can_redo
    assert engine.redo() is None


def test_several_layers_need_layer_change(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    layers = [
        {"material": "Brick", "thickness": 0.2},
        {"material": "Plaster", "thickness": 0.1},
    ]
    engine.apply(
        [{"op": "replace", "path": f"/elements/{west}/params/layers", "value": layers}],
        "script",
        "script",
    )
    before = dump(scene)
    with pytest.raises(PatchRejected, match="layers"):
        engine.apply([thickness_op(west, 0.4)], "u", "user")
    assert dump(scene) == before
    # changing thickness and layers together is accepted
    new_layers = [
        {"material": "Brick", "thickness": 0.3},
        {"material": "Plaster", "thickness": 0.1},
    ]
    engine.apply(
        [
            thickness_op(west, 0.4),
            {"op": "replace", "path": f"/elements/{west}/params/layers", "value": new_layers},
        ],
        "u",
        "user",
    )
    wall = scene.elements[west]
    assert isinstance(wall, WallElement)
    assert wall.params.thickness == 0.4
    assert [layer.thickness for layer in wall.params.layers] == [0.3, 0.1]


# ------------------------------------------------------------------------ locks


def test_locked_element_is_rejected(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    engine.apply(
        [{"op": "replace", "path": f"/elements/{west}/locked", "value": True}], "s", "script"
    )
    before = dump(scene)
    log_before = log_lines(engine.log_path)  # type: ignore[arg-type]
    for kind in ("user", "ai", "script"):
        with pytest.raises(PatchRejected, match="locked"):
            engine.apply([thickness_op(west, 0.4)], "x", kind)
    with pytest.raises(PatchRejected, match="locked"):
        engine.apply([{"op": "remove", "path": f"/elements/{west}"}], "x", "user")
    assert dump(scene) == before
    assert log_lines(engine.log_path) == log_before  # type: ignore[arg-type]


def test_user_may_unlock_but_ai_may_not(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    unlock = [{"op": "replace", "path": f"/elements/{west}/locked", "value": False}]
    engine.apply(
        [{"op": "replace", "path": f"/elements/{west}/locked", "value": True}], "s", "script"
    )
    with pytest.raises(PatchRejected, match="locked"):
        engine.apply(unlock, "ai-agent", "ai")
    assert scene.elements[west].locked
    engine.apply(unlock, "reviewer", "user")
    assert not scene.elements[west].locked
    engine.apply([thickness_op(west, 0.4)], "u", "user")  # editable again


def test_unlock_with_other_changes_is_rejected(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    engine.apply(
        [{"op": "replace", "path": f"/elements/{west}/locked", "value": True}], "s", "script"
    )
    with pytest.raises(PatchRejected, match="locked"):
        engine.apply(
            [
                {"op": "replace", "path": f"/elements/{west}/locked", "value": False},
                thickness_op(west, 0.4),
            ],
            "u",
            "user",
        )
    assert scene.elements[west].locked


def test_auto_lock_after_manual_edit(scene: Scene, engine: PatchEngine) -> None:
    scene.settings.auto_lock_on_manual_edit = True
    west = wall_id(scene, "exterior_west")
    engine.apply([thickness_op(west, 0.4)], "u", "user")
    assert scene.elements[west].locked
    with pytest.raises(PatchRejected, match="locked"):
        engine.apply([thickness_op(west, 0.45)], "u", "user")
    engine.undo()
    assert not scene.elements[west].locked


def test_ai_patch_does_not_auto_lock(scene: Scene, engine: PatchEngine) -> None:
    scene.settings.auto_lock_on_manual_edit = True
    west = wall_id(scene, "exterior_west")
    engine.apply([thickness_op(west, 0.4)], "ai", "ai")
    assert not scene.elements[west].locked


# ------------------------------------------------------------ derived and schema


def test_derived_paths_are_rejected(scene: Scene, engine: PatchEngine) -> None:
    space = one_of(scene, SpaceElement)
    before = dump(scene)
    for op in (
        {"op": "replace", "path": f"/derived/spaces/{space}/area", "value": 1.0},
        {"op": "remove", "path": "/derived/spaces"},
        {"op": "add", "path": "/derived/extra", "value": 1},
        {"op": "move", "from": "/derived/spaces", "path": "/settings/crs"},
        {"op": "replace", "path": "", "value": {}},
    ):
        with pytest.raises(PatchRejected, match="derived"):
            engine.apply([op], "u", "user")
    assert dump(scene) == before
    assert log_lines(engine.log_path) == []  # type: ignore[arg-type]
    assert not engine.log_path.exists()  # type: ignore[union-attr]


def test_schema_breaking_patches_are_rejected(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    window = next(e.id for e in scene.elements.values() if e.kind.value == "window")
    opening = next(e.id for e in scene.elements.values() if isinstance(e, OpeningElement))
    engine.apply([thickness_op(west, 0.4)], "u", "user")
    before = dump(scene)
    log_before = log_lines(engine.log_path)  # type: ignore[arg-type]
    bad_patches = [
        [thickness_op(west, -0.1)],  # schema (negative thickness)
        [thickness_op(west, 5.0)],  # range (maximum 2 m)
        [
            {
                "op": "replace",
                "path": f"/elements/{opening}/params/host_wall_id",
                "value": "does-not-exist",
            }
        ],  # dangling reference
        [{"op": "replace", "path": f"/elements/{window}/params/opening_id", "value": west}],
        [{"op": "add", "path": f"/elements/{west}/bogus", "value": 1}],  # unknown field
        [{"op": "test", "path": f"/elements/{west}/name", "value": "nope"}],  # failing test
        [{"op": "remove", "path": f"/elements/{west}"}],  # joins/openings now dangle
    ]
    for ops in bad_patches:
        with pytest.raises(PatchRejected):
            engine.apply(ops, "u", "user")
        assert dump(scene) == before
        assert log_lines(engine.log_path) == log_before  # type: ignore[arg-type]
    assert len(engine.history) == 1


def test_rejection_reports_messages(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    with pytest.raises(PatchRejected) as caught:
        engine.apply([thickness_op(west, 5.0)], "u", "user")
    assert any("maximum" in message for message in caught.value.messages)


# ------------------------------------------------------------------ provenance


def test_ai_patch_sets_provenance_and_keeps_origin(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    wall = scene.elements[west]
    origin_before = wall.param_origin["thickness"]
    engine.apply([thickness_op(west, 0.4)], "ai-agent", "ai", reason="fitted")
    wall = scene.elements[west]
    assert wall.provenance.value == "ai"
    assert wall.param_origin["thickness"] == origin_before
    engine.apply(
        [
            thickness_op(west, 0.42),
            {
                "op": "replace",
                "path": f"/elements/{west}/param_origin/thickness",
                "value": "fitted",
            },
        ],
        "ai-agent",
        "ai",
    )
    wall = scene.elements[west]
    assert wall.param_origin["thickness"].value == "fitted"
    assert wall.provenance.value == "ai"


def test_script_patch_leaves_origin_and_provenance(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    wall = scene.elements[west]
    origin_before = wall.param_origin["thickness"]
    provenance_before = wall.provenance
    engine.apply([thickness_op(west, 0.4)], "pipeline", AuthorKind.SCRIPT)
    wall = scene.elements[west]
    assert wall.param_origin["thickness"] == origin_before
    assert wall.provenance == provenance_before


def test_user_patch_forces_manual_origin(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    engine.apply(
        [
            thickness_op(west, 0.4),
            {
                "op": "replace",
                "path": f"/elements/{west}/param_origin/thickness",
                "value": "fitted",
            },
        ],
        "u",
        "user",
    )
    assert scene.elements[west].param_origin["thickness"].value == "manual"


# ---------------------------------------------------------------------- groups


def test_undo_group(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    north = wall_id(scene, "exterior_north")
    south = wall_id(scene, "exterior_south")
    start = dump(scene)
    engine.apply([thickness_op(south, 0.31)], "u", "user")
    after_first = dump(scene)
    engine.apply([thickness_op(west, 0.4)], "pipeline", "script", group_id="rerun-1")
    engine.apply([thickness_op(north, 0.35)], "pipeline", "script", group_id="rerun-1")
    records = engine.undo_group("rerun-1")
    assert [r.action for r in records] == ["undo", "undo"]
    assert all(r.group_id == "rerun-1" for r in records)
    assert dump(scene) == after_first
    assert len(engine.applied) == 1
    with pytest.raises(PatchError):
        engine.undo_group("rerun-1")
    with pytest.raises(PatchError):
        engine.undo_group("unknown")
    engine.undo()
    assert dump(scene) == start


def test_undo_group_requires_group_on_top(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    north = wall_id(scene, "exterior_north")
    engine.apply([thickness_op(west, 0.4)], "p", "script", group_id="g")
    engine.apply([thickness_op(north, 0.35)], "u", "user")
    with pytest.raises(PatchError, match="top"):
        engine.undo_group("g")


# ------------------------------------------------------------------- log and replay


def test_log_format(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    north = wall_id(scene, "exterior_north")
    first = engine.apply([thickness_op(west, 0.4)], "u", "user", reason="ä€ reason")
    engine.apply([thickness_op(north, 0.35)], "u", "user")
    engine.undo()
    engine.redo()
    raw = engine.log_path.read_bytes()  # type: ignore[union-attr]
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    lines = raw.decode("utf-8").split("\n")[:-1]
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]
    assert [r["action"] for r in records] == ["apply", "apply", "undo", "redo"]
    assert records[0]["reason"] == "ä€ reason"
    assert "recompute" not in records[0]
    assert records[2]["ref"] == records[1]["id"]
    assert records[3]["ref"] == records[1]["id"]
    assert records[2]["forward"] == records[1]["inverse"]
    assert records[3]["forward"] == records[1]["forward"]
    assert records[0]["id"] == first.id
    assert PatchRecord.model_validate(records[0]).forward == first.forward
    assert [r.id for r in engine.history] == [r["id"] for r in records]


def test_log_replay_reproduces_scene(scene: Scene, tmp_path: Path) -> None:
    log = tmp_path / "patches.jsonl"
    base = make_scene()
    engine = PatchEngine(scene, log, Clock())
    west = wall_id(scene, "exterior_west")
    north = wall_id(scene, "exterior_north")
    south = wall_id(scene, "exterior_south")
    engine.apply([thickness_op(west, 0.5)], "u", "user")
    engine.apply([thickness_op(north, 0.4)], "ai", "ai")
    engine.apply([thickness_op(south, 0.35)], "u", "user")
    engine.undo()
    engine.redo()
    engine.undo()
    engine.apply(
        [{"op": "replace", "path": f"/elements/{south}/name", "value": "Südwand"}], "u", "user"
    )

    replayed = PatchEngine.load(base, log, Clock())
    assert dump(replayed.scene) == dump(scene)
    assert replayed.scene is base
    assert len(replayed.history) == len(engine.history)
    assert [r.id for r in replayed.applied] == [r.id for r in engine.applied]
    assert replayed.can_undo
    assert not replayed.can_redo
    # the replayed engine continues the same log and undoes exactly
    replayed.undo()
    engine.undo()
    assert dump(replayed.scene) == dump(scene)
    assert len(log_lines(log)) == len(engine.history) + 1  # +1: the undo of the replayed engine


def test_log_replay_with_open_redo(scene: Scene, tmp_path: Path) -> None:
    log = tmp_path / "patches.jsonl"
    engine = PatchEngine(scene, log, Clock())
    west = wall_id(scene, "exterior_west")
    engine.apply([thickness_op(west, 0.5)], "u", "user")
    engine.undo()
    replayed = PatchEngine.load(make_scene(), log, Clock())
    assert dump(replayed.scene) == dump(scene)
    assert replayed.can_redo
    replayed.redo()
    engine.redo()
    assert dump(replayed.scene) == dump(scene)


def test_load_missing_log_is_noop(scene: Scene, tmp_path: Path) -> None:
    before = dump(scene)
    engine = PatchEngine.load(scene, tmp_path / "none.jsonl")
    assert dump(scene) == before
    assert engine.history == []


def test_load_rejects_broken_log(tmp_path: Path) -> None:
    log = tmp_path / "patches.jsonl"
    log.write_text("not json\n", encoding="utf-8")
    with pytest.raises(PatchError, match=r"patches.jsonl:1"):
        PatchEngine.load(make_scene(), log)


def test_load_rejects_out_of_order_undo(scene: Scene, tmp_path: Path) -> None:
    log = tmp_path / "patches.jsonl"
    engine = PatchEngine(scene, log, Clock())
    west = wall_id(scene, "exterior_west")
    engine.apply([thickness_op(west, 0.5)], "u", "user")
    engine.undo()
    lines = log_lines(log)
    log.write_bytes((lines[1] + "\n").encode("utf-8"))  # an undo without its apply
    with pytest.raises(PatchError, match="out of order"):
        PatchEngine.load(make_scene(), log)


def test_engine_without_log_path(scene: Scene) -> None:
    engine = PatchEngine(scene)
    west = wall_id(scene, "exterior_west")
    start = dump(scene)
    record = engine.apply([thickness_op(west, 0.4)], "u", "user")
    assert record.timestamp.endswith("Z")
    engine.undo()
    assert dump(scene) == start


def test_structural_ops_add_and_remove_element(scene: Scene, engine: PatchEngine) -> None:
    west = wall_id(scene, "exterior_west")
    wall_data = scene.elements[west].model_dump(mode="json")
    new_id = "11111111-2222-4333-8444-555555555555"
    wall_data.update(id=new_id, tag="extra", locked=False)
    wall_data["params"].update(start=[0.0, 5.0], end=[5.0, 5.0])
    start = dump(scene)
    engine.apply([{"op": "add", "path": f"/elements/{new_id}", "value": wall_data}], "ai", "ai")
    assert new_id in scene.elements
    assert scene.elements[new_id].provenance.value == "ai"
    assert new_id in scene.derived["elements"]
    engine.undo()
    assert new_id not in scene.elements
    assert new_id not in scene.derived["elements"]
    assert dump(scene) == start
