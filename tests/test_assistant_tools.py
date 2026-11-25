"""AI assistant tools: specs, read / measure results, enforced rules (manual consent, no internal parameters), errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant_synthetic import CHAIR_BOX, EMPTY_BOX, LEVEL_ID, make_plan, make_project
from samcloud.assistant.edits import OPS, Proposal
from samcloud.assistant.providers.base import ToolCall
from samcloud.assistant.tools import ToolContext, compact, dispatch, short_args, tool_specs

CHAIR = {"x": [CHAIR_BOX[0] - 0.05, CHAIR_BOX[1] + 0.05], "y": [CHAIR_BOX[2] - 0.05, CHAIR_BOX[3] + 0.05]}


@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return make_project(tmp_path)


def ctx_for(project: Path, **kwargs) -> ToolContext:
    from samcloud.bim.plan_model import load_plan_model
    from samcloud.assistant.paths import plan_model_path

    return ToolContext.for_project(project, LEVEL_ID, load_plan_model(plan_model_path(project, LEVEL_ID)), **kwargs)


def run(ctx: ToolContext, name: str, **arguments):
    return dispatch(ctx, ToolCall("id1", name, arguments))


def test_tool_specs_cover_read_tools_and_every_op() -> None:
    specs = {spec.name: spec for spec in tool_specs()}
    assert {"list_elements", "element_details", "points_in_box", "section", "fit_surface"} <= set(specs)
    assert set(OPS) <= set(specs)
    for spec in specs.values():
        assert spec.parameters["type"] == "object" and spec.description
    assert "face_a" in specs["move_wall"].description and "chain" in specs["move_wall"].description
    assert "manual=true" in specs["add_furniture"].description
    assert "image" not in specs["section"].description
    assert "image" in {s.name: s for s in tool_specs(images_allowed=True)}["section"].description
    json.dumps([spec.parameters for spec in specs.values()])  # serialisable


def test_read_tools_return_compact_json(project: Path) -> None:
    ctx = ctx_for(project)
    walls = json.loads(run(ctx, "list_elements", kind="wall").content)
    assert walls["total"] == 5 and {row["id"] for row in walls["elements"]} == {"W001", "W002", "W003", "W004", "W005"}
    in_room = json.loads(run(ctx, "list_elements", room="R001", kind="furniture").content)
    assert [row["id"] for row in in_room["elements"]] == ["F001"]
    details = json.loads(run(ctx, "element_details", id="O001").content)
    assert details["host_wall_id"] == "W005" and details["width_m"] == 0.9
    points = json.loads(run(ctx, "points_in_box", box=CHAIR).content)
    assert points["count"] > 100 and points["classes"][0]["name"] == "furniture"
    empty = json.loads(run(ctx, "points_in_box", box={**EMPTY_BOX, "z": [0.1, 2.0]}).content)
    assert empty["count"] == 0
    section = run(ctx, "section", z_m=1.0)
    assert not section.error and section.images == [] and json.loads(section.content)["points"] > 0
    table = json.loads(run(ctx, "fit_surface", box={"x": [0.9, 2.3], "y": [0.9, 1.9]}, kind="table").content)
    assert table["ok"] and table["height_m"] == pytest.approx(0.75, abs=0.03)


def test_section_image_only_when_allowed(project: Path) -> None:
    assert run(ctx_for(project), "section", z_m=1.0).images == []
    result = run(ctx_for(project, images_allowed=True), "section", z_m=1.0, box={"x": [3.0, 5.0], "y": [1.0, 4.0]})
    assert len(result.images) == 1 and result.images[0].startswith(b"\x89PNG")


def test_missing_cloud_is_a_tool_error(tmp_path: Path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    ctx = ctx_for(project)
    result = run(ctx, "points_in_box", box=CHAIR)
    assert result.error and result.error_key == "assistant.agent.cloud_unavailable"
    added = run(ctx, "add_furniture", box=CHAIR, kind="chair")
    assert added.error and added.error_key == "assistant.edit.cloud_required"
    assert ctx.proposal.summary() == [] and ctx.last_edit_key == "assistant.edit.cloud_required"
    assert not run(ctx, "list_elements").error  # plan tools still work


def test_edit_tools_change_only_the_proposal(project: Path) -> None:
    ctx = ctx_for(project)
    result = run(ctx, "set_wall_thickness", target="W005", thickness_m=0.25)
    payload = json.loads(result.content)
    assert not result.error and payload["ok"] and payload["op"] == "set_wall_thickness"
    assert any(change["element_id"] == "W005" for change in payload["changes"])
    assert next(w for w in ctx.proposal.plan.walls if w.id == "W005").thickness_m == pytest.approx(0.25)
    assert next(w for w in ctx.plan.walls if w.id == "W005").thickness_m == pytest.approx(0.20)
    # read tools see the proposal state
    assert json.loads(run(ctx, "element_details", id="W005").content)["thickness_m"] == 0.25
    assert ctx.last_edit_key is None


def test_manual_add_and_internal_parameters_are_enforced(project: Path) -> None:
    ctx = ctx_for(project)
    refused = run(ctx, "add_furniture", box=EMPTY_BOX, kind="chair", manual=True)
    assert refused.error and refused.error_key == "assistant.edit.manual_not_allowed"
    assert ctx.proposal.summary() == [] and ctx.last_edit_key == "assistant.edit.manual_not_allowed"
    smuggled = run(ctx, "add_furniture", box=EMPTY_BOX, kind="chair", _item={"label": "Stuhl", "footprint": [[0, 0], [1, 0], [1, 1]]})
    assert smuggled.error and smuggled.error_key == "assistant.edit.bad_params" and ctx.proposal.summary() == []
    no_evidence = run(ctx, "add_furniture", box=EMPTY_BOX, kind="chair")
    assert no_evidence.error and no_evidence.error_key == "assistant.edit.no_evidence"
    assert "ask whether to place it anyway" in json.loads(no_evidence.content)["hint"]

    allowed = ctx_for(project, allow_manual=True)
    result = run(allowed, "add_furniture", box=EMPTY_BOX, kind="chair", manual=True)
    assert not result.error and json.loads(result.content)["changes"][0]["after"]["evidence"] == "manual"
    measured = run(allowed, "add_furniture", box=CHAIR, kind="chair")
    assert json.loads(measured.content)["changes"][0]["after"]["evidence"] == "measured"


def test_dispatch_never_raises() -> None:
    plan = make_plan()
    ctx = ToolContext(plan=plan, proposal=Proposal(plan))
    cases = [
        ToolCall("a", "rm_rf", {}),
        ToolCall("b", "set_opening", {"_raw": "{not json"}),
        ToolCall("c", "set_opening", {"target": "O001"}),
        ToolCall("d", "move_wall", {"target": "W005", "offset_m": "far"}),
        ToolCall("e", "section", {}),
        ToolCall("f", "fit_surface", {"box": {"x": [0, 1], "y": [0, 1]}, "kind": "sofa"}),
        ToolCall("g", "element_details", {}),
        ToolCall("h", "rename_room", {"target": "R001", "name": "x" * 200}),
    ]
    for case in cases:
        result = dispatch(ctx, case)
        assert result.error, case.name
        assert json.loads(result.content)["key"].startswith("assistant.")
    assert ctx.proposal.summary() == []


def test_compact_helpers() -> None:
    value = compact({"a": 1.23456, "b": list(range(50)), "c": float("nan")})
    assert value["a"] == 1.235 and value["b"][-1] == "... 10 more" and value["c"] is None
    assert len(short_args({"text": "x" * 500})) <= 160
