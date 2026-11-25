"""AI assistant edit history: accept, undo, re-apply by signature after a recompute, stale edits, no-op without edits."""

from __future__ import annotations

import json

import pytest

from assistant_synthetic import CHAIR_BOX, LEVEL_ID, make_project, make_result, renumbered_recompute
from samcloud.assistant.edits import EditOp, Proposal
from samcloud.assistant.history import History, HistoryError, apply_to_result, edit_changes, ensure_applied
from samcloud.assistant.paths import assistant_dir, plan_model_path, result_cache_dir
from samcloud.assistant.pointcloud import ProjectCloud
from samcloud.bim.plan_model import load_plan_model, save_plan_model

CHAIR = {"x": [CHAIR_BOX[0] - 0.05, CHAIR_BOX[1] + 0.05], "y": [CHAIR_BOX[2] - 0.05, CHAIR_BOX[3] + 0.05]}


def _plan(project):
    return load_plan_model(plan_model_path(project, LEVEL_ID))


def _wall_x(plan, wall_id):
    wall = next(w for w in plan.walls if w.id == wall_id)
    return wall.centreline[0][0]


def _accept(project, *ops, cloud=None, prompt="test"):
    proposal = Proposal(_plan(project))
    for op in ops:
        proposal.apply(op, cloud=cloud)
    return History.load(project, LEVEL_ID).accept(proposal, prompt=prompt, provider="ollama", model="dummy-model")


def test_without_edits_nothing_is_touched(tmp_path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    plan_file = plan_model_path(project, LEVEL_ID)
    before = plan_file.read_bytes()
    report = ensure_applied(project, LEVEL_ID)
    assert (report.applied, report.stale, report.changed) == ([], [], False)
    assert not assistant_dir(project, LEVEL_ID).exists()
    result = make_result(_plan(project))
    assert apply_to_result(result, project, LEVEL_ID) is result
    assert edit_changes(project, LEVEL_ID) == (None, None, [])
    assert plan_file.read_bytes() == before
    assert not assistant_dir(project, LEVEL_ID).exists()


def test_accept_undo_and_files(tmp_path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    original = plan_model_path(project, LEVEL_ID).read_bytes()
    first = _accept(project, EditOp("move_wall", "W005", {"offset_m": 0.3}), prompt="Wand 30 cm nach rechts")
    second = _accept(project, EditOp("rename_room", "R002", {"name": "Gast"}))
    directory = assistant_dir(project, LEVEL_ID)
    assert (directory / "plan_model.base.json").read_bytes() == original
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    assert state["plan_model_sha256"]
    plan = _plan(project)
    assert _wall_x(plan, "W005") == pytest.approx(4.3)
    assert [r.name for r in plan.rooms] == ["Wohnen", "Gast"]
    data = json.loads((directory / "edits.json").read_text(encoding="utf-8"))
    assert data["version"] == 1 and [e["id"] for e in data["edits"]] == [first, second]
    record = data["edits"][0]
    assert set(record) >= {"id", "timestamp", "prompt", "provider", "model", "ops", "changes", "signatures", "status", "stale_reason"}
    assert record["prompt"] == "Wand 30 cm nach rechts" and record["status"] == "active"
    assert "W005" in record["signatures"]

    report = History.load(project, LEVEL_ID).undo(first)
    assert report.changed and report.applied == [second] and report.stale == []
    plan = _plan(project)
    assert _wall_x(plan, "W005") == pytest.approx(4.0)
    assert [r.name for r in plan.rooms] == ["Wohnen", "Gast"]
    history = History.load(project, LEVEL_ID)
    assert [e["status"] for e in history.edits()] == ["undone", "active"]
    History.load(project, LEVEL_ID).undo(second)
    # save_plan_model is canonical (compact json.dumps of to_dict(); floats round-trip exactly), and the synthetic
    # plan_model.json was written by save_plan_model too, so after undoing everything the file is byte-identical.
    assert plan_model_path(project, LEVEL_ID).read_bytes() == original
    with pytest.raises(HistoryError):
        History.load(project, LEVEL_ID).undo(second)


def test_no_credentials_in_edits_json(tmp_path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    proposal = Proposal(_plan(project))
    proposal.apply(EditOp("rename_room", "R001", {"name": "Bad"}))
    # leaks must never reach the file: key-like names are matched by substring, the prompt is redacted
    proposal.ops[0].params.update({"api_key": "sk-test-DUMMY-0000", "OpenAI_Key": "dummy-value-1111", "x_auth_token": "dummy-value-2222",
                                   "Authorization": "Bearer dummy-value-3333", "my_secret": "dummy-value-4444"})
    History.load(project, LEVEL_ID).accept(proposal, prompt="rename it, key sk-test-DUMMY-0000 and Bearer dummy-value-5555", provider="openai", model="m")
    text = (assistant_dir(project, LEVEL_ID) / "edits.json").read_text(encoding="utf-8")
    for leak in ("sk-test-DUMMY-0000", "dummy-value-1111", "dummy-value-2222", "dummy-value-3333", "dummy-value-4444", "dummy-value-5555",
                 "api_key", "OpenAI_Key", "x_auth_token", "Authorization", "my_secret"):
        assert leak not in text
    assert "rename it" in json.loads(text)["edits"][0]["prompt"]


def test_outdated_proposal_is_rebased_or_refused(tmp_path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    older = Proposal(_plan(project))
    older.apply(EditOp("rename_room", "R001", {"name": "Bad"}))
    conflicting = Proposal(_plan(project))
    conflicting.apply(EditOp("move_furniture", "F001", {"dx_m": 0.5, "dy_m": 0.0}))
    _accept(project, EditOp("rename_room", "R002", {"name": "Gast"}), EditOp("delete_furniture", "F001", {}))
    History.load(project, LEVEL_ID).accept(older, prompt="x", provider="p", model="m")  # still applies: re-applied by signature
    assert [r.name for r in _plan(project).rooms] == ["Bad", "Gast"]
    with pytest.raises(HistoryError) as raised:
        History.load(project, LEVEL_ID).accept(conflicting, prompt="x", provider="p", model="m")
    assert raised.value.key == "assistant.history.outdated"
    assert len(History.load(project, LEVEL_ID).edits()) == 2


def _recompute(project, shift=(0.012, -0.008)):
    base = load_plan_model(assistant_dir(project, LEVEL_ID) / "plan_model.base.json")
    recomputed = renumbered_recompute(base, shift)
    save_plan_model(recomputed, result_cache_dir(project, LEVEL_ID))
    return recomputed


def test_undo_after_recompute_keeps_the_new_automatic_plan(tmp_path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    first = _accept(project, EditOp("move_wall", "W005", {"offset_m": 0.3}))
    _accept(project, EditOp("rename_room", "R002", {"name": "Gast"}))
    recomputed = _recompute(project)  # the pipeline ran, nobody called ensure_applied
    report = History.load(project, LEVEL_ID).undo(first)
    assert report.stale == []
    plan = _plan(project)
    partition = next(w for w in plan.walls if w.id == "W001")  # renumbered by the recompute
    assert partition.centreline[0][0] == pytest.approx(4.012)  # the new automatic geometry, not the old base
    assert sorted(r.name for r in plan.rooms) == ["Gast", "Wohnen"]
    base = load_plan_model(assistant_dir(project, LEVEL_ID) / "plan_model.base.json")
    assert base.to_dict() == recomputed.to_dict()


def test_accept_after_recompute_rebases(tmp_path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    _accept(project, EditOp("move_wall", "W005", {"offset_m": 0.3}))
    proposal = Proposal(_plan(project))  # made before the recompute
    proposal.apply(EditOp("set_opening", "O002", {"width_m": 1.6}))
    _recompute(project)
    history = History.load(project, LEVEL_ID)
    report = history.accept_with_report(proposal, prompt="x", provider="p", model="m")
    plan = _plan(project)
    assert next(w for w in plan.walls if w.id == "W001").centreline[0][0] == pytest.approx(4.312)
    assert next(o for o in plan.openings if o.kind == "window").width_m == pytest.approx(1.6)
    assert report.rebased == {"O002": "O001"} == history.last_rebase  # the user must be told the window is O001 now
    assert any("O002 -> O001" in note for note in report.notes) and history.last_notes == report.notes
    record = History.load(project, LEVEL_ID).edits()[-1]
    assert record["ops"][0]["target"] == "O001"  # the window's id after the recompute
    assert record["rebased"] == {"O002": "O001"} and any("O002 -> O001" in note for note in record["notes"])
    plain = Proposal(_plan(project))
    plain.apply(EditOp("rename_room", "R001", {"name": "Bad"}))
    again = History.load(project, LEVEL_ID)
    assert again.accept(plain, prompt="x", provider="p", model="m") == History.load(project, LEVEL_ID).edits()[-1]["id"]
    assert again.last_rebase == {} and "rebased" not in History.load(project, LEVEL_ID).edits()[-1]


def test_ambiguous_target_is_never_guessed(tmp_path) -> None:
    import copy

    from samcloud.assistant.edits import EditError, resolve_signature, signature_of
    from samcloud.assistant.history import reapply_edit

    project = make_project(tmp_path, with_cloud=False)
    original = _plan(project)
    edit = _accept(project, EditOp("move_furniture", "F001", {"dx_m": 0.2, "dy_m": 0.0}))
    twins = renumbered_recompute(original, (0.0, 0.0))
    left, right = twins.furniture[0], copy.deepcopy(twins.furniture[0])
    left.footprint = [[x - 0.1, y] for x, y in left.footprint]  # two equal tables 0.1 m either side of the old one
    right.id, right.footprint = "F012", [[x + 0.1, y] for x, y in right.footprint]
    twins.furniture.append(right)
    signature = signature_of(original, "F001")
    assert resolve_signature(twins, signature) in ("F011", "F012")  # the lenient lookup still answers
    with pytest.raises(EditError) as raised:
        resolve_signature(twins, signature, preferred_id="F001", strict=True)
    assert raised.value.key == "assistant.history.ambiguous"
    with pytest.raises(HistoryError) as raised:
        reapply_edit(twins, History.load(project, LEVEL_ID).edit(edit))
    assert raised.value.key == "assistant.history.ambiguous"
    save_plan_model(twins, result_cache_dir(project, LEVEL_ID))
    report = ensure_applied(project, LEVEL_ID)
    assert [s["id"] for s in report.stale] == [edit] and report.stale[0]["key"] == "assistant.history.ambiguous"
    assert sorted(min(p[0] for p in f.footprint) for f in _plan(project).furniture) == pytest.approx([0.9, 1.1])  # nothing moved
    # one clear candidate (the twin 0.1 m away vs. one 0.25 m away) resolves again
    right.footprint = [[x + 0.15, y] for x, y in right.footprint]
    assert resolve_signature(twins, signature, preferred_id="F001", strict=True) == "F011"


def test_is_locked_is_non_blocking(tmp_path) -> None:
    import threading

    from samcloud.assistant.history import _locked, history_locked

    project = make_project(tmp_path, with_cloud=False)
    history = History.load(project, LEVEL_ID)
    assert not history.is_locked() and not history_locked(project, LEVEL_ID)
    assert not assistant_dir(project, LEVEL_ID).exists()  # asking never creates anything
    with _locked(assistant_dir(project, LEVEL_ID)):
        assert history.is_locked()  # same thread: the OS file lock is held on another handle
    assert not history.is_locked()
    holding, release = threading.Event(), threading.Event()

    def writer() -> None:
        with _locked(assistant_dir(project, LEVEL_ID)):
            holding.set()
            release.wait(10)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert holding.wait(10)
        assert history_locked(project, LEVEL_ID)
    finally:
        release.set()
        thread.join(10)
    assert not history_locked(project, LEVEL_ID)


def _fail(monkeypatch, name, *, after_calls=0):
    original = getattr(History, name)
    calls = {"n": 0}

    def failing(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > after_calls:
            raise OSError("simulated crash")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(History, name, failing)


@pytest.mark.parametrize(
    ("step", "after_calls", "applied"),
    [
        ("_write", 0, False),  # crash before edits.json: the edit is not stored, nothing applied
        ("_save_plan", 0, True),  # crash before plan_model.json: rebuilt from base + edits
        ("_write_state", 1, True),  # crash after plan_model.json, before the final state: pending hash resolves
    ],
)
def test_crash_windows_never_lose_or_double_an_edit(tmp_path, monkeypatch, step, after_calls, applied) -> None:
    project = make_project(tmp_path, with_cloud=False)
    _accept(project, EditOp("move_wall", "W005", {"offset_m": 0.3}))
    with monkeypatch.context() as patch:
        _fail(patch, step, after_calls=after_calls)
        with pytest.raises(OSError):
            _accept(project, EditOp("move_wall", "W005", {"offset_m": 0.2}))
    report = ensure_applied(project, LEVEL_ID)
    plan = _plan(project)
    assert _wall_x(plan, "W005") == pytest.approx(4.5 if applied else 4.3)
    assert len(History.load(project, LEVEL_ID).active_edits()) == (2 if applied else 1)
    again = ensure_applied(project, LEVEL_ID)
    assert not again.changed and _wall_x(_plan(project), "W005") == pytest.approx(4.5 if applied else 4.3)
    assert report.stale == []


@pytest.mark.parametrize("state_text", ["", "{not json", "{}", '{"plan_model_sha256": "0000"}'])
def test_missing_or_corrupt_state_does_not_double_apply(tmp_path, state_text) -> None:
    project = make_project(tmp_path, with_cloud=False)
    _accept(project, EditOp("move_wall", "W005", {"offset_m": 0.3}))
    state_file = assistant_dir(project, LEVEL_ID) / "state.json"
    state_file.write_text(state_text, encoding="utf-8")
    report = ensure_applied(project, LEVEL_ID)
    assert not report.changed and _wall_x(_plan(project), "W005") == pytest.approx(4.3)
    assert json.loads(state_file.read_text(encoding="utf-8"))["plan_model_sha256"]
    # an accept with a corrupt state checks too (and stacks the new edit exactly once)
    state_file.write_text("{not json", encoding="utf-8")
    _accept(project, EditOp("move_wall", "W005", {"offset_m": 0.2}))
    assert _wall_x(_plan(project), "W005") == pytest.approx(4.5)


def test_corrupt_state_with_a_recompute_rebases(tmp_path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    _accept(project, EditOp("move_wall", "W005", {"offset_m": 0.3}))
    _recompute(project)
    (assistant_dir(project, LEVEL_ID) / "state.json").write_text("{not json", encoding="utf-8")
    report = ensure_applied(project, LEVEL_ID)
    assert report.changed
    assert next(w for w in _plan(project).walls if w.id == "W001").centreline[0][0] == pytest.approx(4.312)


def test_stale_edits_come_back_when_their_element_returns(tmp_path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    original = _plan(project)
    edit = _accept(project, EditOp("move_furniture", "F001", {"dx_m": 0.2, "dy_m": 0.0}))
    without_table = renumbered_recompute(original)
    without_table.furniture = []
    save_plan_model(without_table, result_cache_dir(project, LEVEL_ID))
    assert [s["id"] for s in ensure_applied(project, LEVEL_ID).stale] == [edit]
    save_plan_model(renumbered_recompute(original, (0.004, 0.003)), result_cache_dir(project, LEVEL_ID))
    report = ensure_applied(project, LEVEL_ID)
    assert report.applied == [edit] and report.stale == []
    history = History.load(project, LEVEL_ID)
    assert history.edit(edit)["status"] == "active" and history.edit(edit)["stale_reason"] is None
    table = _plan(project).furniture[0]
    assert min(p[0] for p in table.footprint) == pytest.approx(1.0 + 0.004 + 0.2)


def test_recompute_reapplies_by_signature_and_reports_stale(tmp_path) -> None:
    project = make_project(tmp_path)
    cloud = ProjectCloud.load(project, LEVEL_ID)
    moved = _accept(project, EditOp("move_wall", "W005", {"offset_m": 0.3}), EditOp("set_opening", "O002", {"width_m": 1.6}))
    chair = _accept(project, EditOp("add_furniture", None, {"box": CHAIR, "kind": "chair"}), EditOp("rename_room", "R002", {"name": "Gast"}),
                    cloud=cloud)
    table = _accept(project, EditOp("move_furniture", "F001", {"dx_m": 0.2, "dy_m": 0.0}))
    added_chair = _plan(project).furniture[-1]

    # the pipeline recomputes: ids renumbered, geometry shifted a few millimetres, and the table is gone
    base = load_plan_model(assistant_dir(project, LEVEL_ID) / "plan_model.base.json")
    recomputed = renumbered_recompute(base)
    recomputed.furniture = []
    save_plan_model(recomputed, result_cache_dir(project, LEVEL_ID))

    report = ensure_applied(project, LEVEL_ID)
    assert report.changed and report.applied == [moved, chair] and [s["id"] for s in report.stale] == [table]
    assert report.stale[0]["key"] == "assistant.history.element_gone"
    plan = _plan(project)
    partition = next(w for w in plan.walls if abs(w.centreline[0][1] - w.centreline[1][1]) > 4.0 and 3.0 < w.centreline[0][0] < 5.0)
    assert partition.id == "W001" and partition.centreline[0][0] == pytest.approx(4.312)
    door = next(o for o in plan.openings if o.kind == "door")
    assert door.center[0] == pytest.approx(4.312) and door.host_wall_id == "W001"
    window = next(o for o in plan.openings if o.kind == "window")
    assert window.width_m == pytest.approx(1.6)
    assert sorted(r.name for r in plan.rooms) == ["Gast", "Wohnen"]
    assert next(r for r in plan.rooms if r.name == "Gast").label_point[0] > 4.3
    chair_item = plan.furniture[-1]
    assert chair_item.label == "Stuhl" and chair_item.footprint == added_chair.footprint and chair_item.fit["evidence"] == "measured"
    history = History.load(project, LEVEL_ID)
    assert [e["status"] for e in history.edits()] == ["active", "active", "stale"]
    assert history.stale_edits()[0]["stale_reason"]
    # nothing changed since: a second call is a no-op
    again = ensure_applied(project, LEVEL_ID)
    assert not again.changed and again.applied == [moved, chair]


def test_apply_to_result_patches_the_dxf(tmp_path) -> None:
    from samcloud.dxf.pipeline import load_cached_result

    project = make_project(tmp_path, with_cloud=False)
    _accept(project, EditOp("set_opening", "O002", {"width_m": 1.5}))
    result = load_cached_result(result_cache_dir(project, LEVEL_ID))
    patched = apply_to_result(result, project, LEVEL_ID, language="en")
    insert = next(e for e in patched.layers["A-WIND"] if e["type"] == "insert")
    assert insert["attributes"]["LABEL"] == "Window 1.50 m"
    assert all(a is b for a, b in zip(result.layers["A-DOOR"], patched.layers["A-DOOR"]))


def test_scrub_never_drops_an_engine_field() -> None:
    """No parameter, stored item field, snapshot or record key may look like a secret: _scrub would silently drop it."""
    from dataclasses import fields

    from samcloud.assistant.edits import OPS, Change, signature_of, snapshot
    from samcloud.assistant.history import _secret_key
    from samcloud.bim.plan_model import BuildingPlan, PlanFurniture, PlanOpening, PlanRoom, PlanWall
    from assistant_synthetic import make_plan

    names: set[str] = {"_item", "_created_ids", "op", "target", "params", "rebased", "stale_code", "stale_reason", "undone_at"}
    for spec in OPS.values():
        names |= set(spec.schema["properties"])
        for value in spec.schema["properties"].values():
            names |= set((value.get("properties") or {}))
    for cls in (PlanFurniture, PlanWall, PlanOpening, PlanRoom, BuildingPlan):
        names |= {f.name for f in fields(cls)}
    names |= set(Change("x", "wall", "added", None, None).to_dict())
    plan = make_plan()
    for kind, element_id in (("wall", "W001"), ("opening", "O001"), ("room", "R001"), ("furniture", "F001")):
        found = next(e for e in {"wall": plan.walls, "opening": plan.openings, "room": plan.rooms, "furniture": plan.furniture}[kind] if e.id == element_id)
        names |= set(snapshot(plan, kind, found)) | set(signature_of(plan, element_id))
    names |= {"id", "timestamp", "prompt", "provider", "model", "ops", "changes", "signatures", "notes", "status", "version", "edits"}
    assert sorted(name for name in names if _secret_key(name)) == []
    assert _secret_key("api_key") and _secret_key("X-Auth-Token") and _secret_key("Authorization") and _secret_key("client_secret")


def test_every_op_survives_accept_recompute_replay(tmp_path) -> None:
    project = make_project(tmp_path)
    cloud = ProjectCloud.load(project, LEVEL_ID)
    groups = [
        [EditOp("move_wall", "W005", {"offset_m": 0.25}), EditOp("set_wall_thickness", "W002", {"thickness_m": 0.35, "anchor": "face_a"})],
        [EditOp("set_opening", "O002", {"width_m": 1.4, "sill_m": 0.85, "shift_m": 0.2}), EditOp("rename_room", "R001", {"name": "Küche"})],
        [EditOp("add_furniture", None, {"box": CHAIR, "kind": "chair"}),
         EditOp("add_furniture", None, {"box": {"x": [6.6, 7.4], "y": [3.6, 4.4]}, "kind": "sofa", "manual": True, "size_m": [1.8, 0.8]})],
        [EditOp("move_furniture", "F001", {"dx_m": 0.3, "dy_m": 0.4}), EditOp("rotate_furniture", "F001", {"angle_deg": 15.0})],
        [EditOp("set_furniture_class", "F002", {"kind": "armchair"}), EditOp("delete_furniture", "F003", {})],
        [EditOp("delete_opening", "O002", {})],
        [EditOp("delete_wall", "W005", {})],
    ]
    accepted = []
    for ops in groups:
        edit_id = _accept(project, *ops, cloud=cloud)
        accepted.append(edit_id)
    before = _plan(project)
    stored = History.load(project, LEVEL_ID).edits()
    assert [e["id"] for e in stored] == accepted
    for record in stored:  # nothing an op needs was scrubbed away
        for op in record["ops"]:
            if op["op"] == "add_furniture":
                assert op["params"]["_item"]["fit"]["evidence"] in ("measured", "manual") and op["params"]["_created_ids"]
    _recompute(project, (0.006, -0.004))
    report = ensure_applied(project, LEVEL_ID)
    assert report.changed and report.applied == accepted and report.stale == []
    after = _plan(project)
    assert len(after.walls) == len(before.walls) and len(after.openings) == len(before.openings) == 0
    assert sorted((f.label, f.layer) for f in after.furniture) == sorted((f.label, f.layer) for f in before.furniture)
    assert sorted(r.name for r in after.rooms) == sorted(r.name for r in before.rooms)
    assert next(w for w in after.walls if abs(w.thickness_m - 0.35) < 1e-9)
