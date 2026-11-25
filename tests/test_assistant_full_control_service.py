"""Full control in the assistant service: direct apply after a passed QA (history, exports, undo), QA failure and history
refusal apply nothing, proposal mode unchanged without full control, and the domain-neutral service with a fake
non-indoor domain (ask / accept / undo / preview without any plan_model).  Fake providers only, no network."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from assistant_synthetic import LEVEL_ID, make_plan, make_project
from samcloud.assistant import agent as agent_module
from samcloud.assistant import credentials as ai_creds
from samcloud.assistant import settings as ai_settings
from samcloud.assistant.domains import registry
from samcloud.assistant.domains.base import TIER_ELEMENTS, TIER_FULL, AssistantDomain, HistoryBackend, ToolDef
from samcloud.assistant.edits import Change, EditError, EditOp, Proposal
from samcloud.assistant.history import History, HistoryError
from samcloud.assistant.paths import assistant_dir, plan_model_path
from samcloud.assistant.qa import QaResult
from samcloud.assistant.service import AssistantService
from samcloud.assistant.tools import ToolResult
from samcloud.bim.plan_model import load_plan_model, save_plan_model
from test_assistant_agent import FakeProvider, call, say

DUMMY_KEY = "sk-test-DUMMY-0000"
SITE = "site"


@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the real user's AI settings / credentials."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _enable(provider: str = "openai", **overrides: Any) -> None:
    values: dict[str, Any] = {"enabled": True, "provider": provider, "model": "fake-model", "cost_limit_request_usd": 0.0,
                              "cost_limit_project_day_usd": 0.0, "privacy_ack_cloud": True, **overrides}
    ai_settings.save_settings(ai_settings.AssistantSettings(**values))
    ai_creds.save_ai_credentials({"openai_api_key": DUMMY_KEY, "ollama_url": "http://127.0.0.1:9"})


def _use(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> FakeProvider:
    monkeypatch.setattr(agent_module, "get_provider", lambda name, settings, creds: provider)
    return provider


def _project_with_two_interior_walls(tmp_path: Path) -> Path:
    project = make_project(tmp_path, with_cloud=False)
    proposal = Proposal(make_plan())
    proposal.apply(EditOp("add_wall", None, {"start": [4.05, 3.5], "end": [7.9, 3.5], "thickness_m": 0.10}))
    save_plan_model(proposal.plan, plan_model_path(project, LEVEL_ID).parent)
    return project


BULK = dict(kind="wall", op="set_wall_thickness", params={"thickness_m": 0.12}, filter={"wall_kind": "interior"})


# ------------------------------------------------------------------------------------------------- direct apply

def test_full_control_bulk_change_is_applied_after_qa_and_undone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable("openai")  # cloud provider: full control on by default
    project = _project_with_two_interior_walls(tmp_path)
    path = plan_model_path(project, LEVEL_ID)
    original = load_plan_model(path).to_dict()
    provider = _use(monkeypatch, FakeProvider([call("apply_to_kind", **BULK), say("Alle Innenwände werden 0.12 m dick.")]))
    service = AssistantService(project, LEVEL_ID)
    service.discard()
    assert service.tool_tier == TIER_FULL
    result = service.ask("alle Innenwände 12 cm", [])
    assert result.error is None and result.qa is not None and result.qa.ok  # the real QA chain
    assert result.applied and result.edit_id and result.tier == TIER_FULL and not result.needs_confirmation
    assert "apply_to_kind" in provider.tools_seen[0] and "add_wall" in provider.tools_seen[0]
    assert "Vollkontrolle aktiv" in provider.requests[0][0].content or "Full control active" in provider.requests[0][0].content
    assert service.pending() is None
    edited = load_plan_model(path)
    assert sorted(round(w.thickness_m, 3) for w in edited.walls if w.kind == "interior") == [0.12, 0.12]
    edits = service.edits()
    assert len(edits) == 1 and edits[0]["id"] == result.edit_id and edits[0]["status"] == "active"
    assert edits[0]["tools"] == ["apply_to_kind"] and {"W005", "W006"} <= {s["element_id"] for s in edits[0]["summary"]}
    assert service.export_actions("accept") == ["export", "export_ifc"]
    chat = service.chat()
    assert chat[-1]["meta"]["applied"] is True and chat[-1]["meta"]["edit_id"] == result.edit_id
    assert result.to_dict()["applied"] is True

    report = service.undo(result.edit_id)
    assert report["changed"] and [e["status"] for e in service.edits()] == ["undone"]
    assert load_plan_model(path).to_dict() == original


def test_qa_failure_applies_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable("anthropic")
    ai_creds.save_ai_credentials({"anthropic_api_key": DUMMY_KEY})
    project = make_project(tmp_path, with_cloud=False)
    path = plan_model_path(project, LEVEL_ID)
    before = path.read_bytes()
    finding = [{"check": "wall_overlap", "subject": "W005|W006", "detail": None}]
    monkeypatch.setattr(agent_module, "run_qa", lambda plan, *, baseline, work_dir: QaResult(False, finding, finding, None, 0, "x"))
    _use(monkeypatch, FakeProvider([call("rename_room", target="R001", name="Küche"), say("Wird angewendet."),
                                    say("Geht nicht.")]))
    service = AssistantService(project, LEVEL_ID)
    service.discard()
    result = service.ask("Raum umbenennen", [])
    assert not result.applied and result.edit_id is None and result.proposal is None
    assert path.read_bytes() == before and service.pending() is None
    assert not (assistant_dir(project, LEVEL_ID) / "edits.json").exists()


def test_history_refusal_keeps_the_proposal_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable("openai")
    project = make_project(tmp_path, with_cloud=False)
    path = plan_model_path(project, LEVEL_ID)
    before = path.read_bytes()

    def locked(self: History, proposal: Any, **_kwargs: Any) -> Any:
        raise HistoryError("Another assistant edit is being saved; try again.", key="assistant.history.locked")

    monkeypatch.setattr(History, "accept_with_report", locked)
    _use(monkeypatch, FakeProvider([call("rename_room", target="R001", name="Küche"), say("Wird angewendet.")]))
    service = AssistantService(project, LEVEL_ID)
    service.discard()
    result = service.ask("Raum umbenennen", [])
    assert not result.applied and result.proposal is not None and service.pending() is result
    assert any("Not applied" in note or "Nicht angewendet" in note for note in result.notes)
    assert path.read_bytes() == before
    service.discard()


def test_without_full_control_everything_stays_a_proposal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable("openai", full_control=False)
    project = make_project(tmp_path, with_cloud=False)
    path = plan_model_path(project, LEVEL_ID)
    before = path.read_bytes()
    provider = _use(monkeypatch, FakeProvider([call("rename_room", target="R001", name="Küche"), say("Vorschlag: Küche.")]))
    service = AssistantService(project, LEVEL_ID)
    service.discard()
    assert service.tool_tier == TIER_ELEMENTS
    result = service.ask("Raum umbenennen", [])
    assert not result.applied and result.tier == TIER_ELEMENTS and service.pending() is result
    assert "add_wall" not in provider.tools_seen[0]
    system = provider.requests[0][0].content
    assert "nur ein Vorschlag" in system or "only a proposal" in system
    assert path.read_bytes() == before
    assert "applied" not in service.chat()[-1]["meta"]
    edit_id = service.accept()
    assert load_plan_model(path).rooms[0].name == "Küche" and service.edits()[0]["id"] == edit_id


def test_ollama_default_is_proposal_mode(tmp_path: Path) -> None:
    _enable("ollama")
    service = AssistantService(make_project(tmp_path, with_cloud=False), LEVEL_ID)
    assert service.tool_tier == TIER_ELEMENTS
    _enable("ollama", full_control=True)
    assert service.tool_tier == TIER_FULL


# ------------------------------------------------------------------------------------- domain-neutral service

def _bytes(model: dict[str, Any]) -> bytes:
    return json.dumps(model, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _model_path(project: Path, level_id: str) -> Path:
    return Path(project) / "products" / "fake_site" / "model.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(model: dict[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(_bytes(model))


class _SiteProposal:
    """Proposal of the fake site model: ``set_height`` of a tree."""

    def __init__(self, base: dict[str, Any]) -> None:
        self.base = copy.deepcopy(base)
        self.plan = copy.deepcopy(base)
        self.ops: list[EditOp] = []
        self.changes: list[Change] = []
        self.notes: list[str] = []
        self.signatures: dict[str, Any] = {}

    def apply(self, op: Any, *, cloud: Any = None, replay: bool = False) -> list[Change]:
        op = EditOp.from_dict(op) if isinstance(op, dict) else EditOp(op.op, op.target, dict(op.params))
        if op.op != "set_height" or op.target not in self.plan["trees"]:
            raise EditError(f"No tree {op.target}.", key="assistant.edit.not_found", params={"id": op.target})
        before = dict(self.plan["trees"][op.target])
        self.plan["trees"][op.target]["height_m"] = float(op.params["height_m"])
        change = Change(op.target, "tree", "modified", before, dict(self.plan["trees"][op.target]))
        self.signatures.setdefault(op.target, {"kind": "tree", "xy": before["xy"]})
        self.ops.append(op)
        self.changes.append(change)
        return [change]

    def summary(self) -> list[dict[str, Any]]:
        return [{"element_id": c.element_id, "kind": c.kind, "change": c.change, "before": c.before, "after": c.after, "notes": []}
                for c in self.changes]

    def needs_confirmation(self) -> bool:
        return False


def _set_height_tool(ctx: Any, call_: Any, arguments: dict[str, Any]) -> ToolResult:
    target = arguments.pop("target", None)
    ctx.proposal.apply(EditOp("set_height", target, arguments))
    return ToolResult(json.dumps({"ok": True}))


class _SiteDomain(AssistantDomain):
    """A non-indoor domain with its own JSON model (stands in for the outdoor domain)."""

    kind = "fake_site"
    supported_tiers = (TIER_ELEMENTS, TIER_FULL)

    @property
    def path(self) -> Path:
        return _model_path(self.project, self.level_id)

    def availability(self) -> tuple[bool, str | None]:
        return (True, None) if self.path.is_file() else (False, "assistant.agent.no_plan")

    def load_base(self) -> Any:
        return _load(self.path)

    def new_proposal(self, base: Any) -> Any:
        return _SiteProposal(base)

    def view_frame(self) -> dict[str, Any]:
        return {"units_per_metre": 1.0, "floor_z": None, "origin_offset": [100.0, 200.0, 3.0], "crs": "LOCAL"}

    def element_list(self, filter: Any = None, model: Any = None) -> list[dict[str, Any]]:  # noqa: A002
        model = model if model is not None else self.load_base()
        return [{"id": key, "type": "tree", "label": key, **value} for key, value in model["trees"].items()]

    def element_details(self, element_id: str, model: Any = None) -> dict[str, Any]:
        model = model if model is not None else self.load_base()
        tree = model["trees"].get(element_id)
        return {"found": False, "id": element_id} if tree is None else {"found": True, "id": element_id, **tree}

    def hit_test(self, x: float, y: float, model: Any = None) -> dict[str, Any]:
        model = model if model is not None else self.load_base()
        for key, tree in model["trees"].items():
            if abs(tree["xy"][0] - x) < 1.0 and abs(tree["xy"][1] - y) < 1.0:
                return {"type": "element", "kind": "tree", "id": key, "label": f"tree {key}"}
        return {"type": "point", "x": x, "y": y}

    def selection_from_global_id(self, global_id: str, point: list[float] | None = None, model: Any = None) -> dict[str, Any]:
        return {"type": "unknown", "global_id": global_id}

    def chip_label(self, sel: dict[str, Any], lang: str | None = None) -> str:
        return str(sel.get("label") or sel.get("id") or "")

    def outline(self, sel: dict[str, Any], model: Any = None) -> list[list[list[float]]]:
        model = model if model is not None else self.load_base()
        tree = model["trees"].get(str(sel.get("id")))
        if tree is None:
            return []
        x, y = tree["xy"]
        return [[[x - 1, y - 1], [x + 1, y - 1], [x + 1, y + 1], [x - 1, y + 1]]]

    def z_range(self, sel: dict[str, Any], model: Any = None) -> tuple[float, float] | None:
        return None

    def global_ids_for(self, selections: list[dict[str, Any]], model: Any = None) -> list[str]:
        return []

    def read_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        return []

    def edit_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        schema = {"type": "object", "properties": {"target": {"type": "string"}, "height_m": {"type": "number"}},
                  "required": ["target", "height_m"]}
        return [ToolDef("set_height", "Set a tree height.", schema, _set_height_tool, "edit")]

    def run_qa(self, proposal: Any, baseline: Any, *, work_dir: Path) -> Any:
        return QaResult(True, [], [], None, 0, "fake site ok")

    def vocabulary(self, lang: str) -> dict[str, Any]:
        return {"project_type": "fake site", "model_phrase": "the site model", "element_types": {"tree": "tree"}, "rules": ["{tier}"]}

    def history_backend(self) -> HistoryBackend:
        return HistoryBackend(
            kind="fake_site", base_filename="model.base.json", model_path=_model_path, load=_load, save=_save,
            semantic_hash=lambda model: hashlib.sha256(_bytes(model)).hexdigest(),
            file_hash=lambda model: hashlib.sha256(_bytes(model)).hexdigest(),
            new_proposal=_SiteProposal, op_from_dict=EditOp.from_dict,
            resolve_signature=lambda model, sig, preferred_id=None, strict=False: preferred_id if preferred_id in model["trees"] else None,
            diff=lambda base, edited: [])

    def export_actions(self, after: str = "accept") -> list[str]:
        return ["export_site"]

    def preview(self, proposal: Any) -> dict[str, list[dict[str, Any]]]:
        return {"added": [], "removed": [], "moved": [{"id": e["element_id"], "kind": "tree", "change": e["change"]}
                                                       for e in proposal.summary()]}


@pytest.fixture
def site(tmp_path: Path) -> Any:
    registry._reset_for_tests()
    project = tmp_path / "site_project"
    _save({"trees": {"T001": {"xy": [10.0, 20.0], "height_m": 5.0}}}, _model_path(project, SITE))
    registry.register_domain("fake_site", lambda p, level, mode: _SiteDomain(p, level, mode=mode),
                             lambda p, level, mode: str(level) == SITE and _model_path(p, level).is_file(), priority=5)
    yield project
    registry._reset_for_tests()


def test_fake_domain_proposal_accept_undo_without_plan_model(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable("openai", full_control=False)
    path = _model_path(site, SITE)
    original = path.read_bytes()
    _use(monkeypatch, FakeProvider([call("set_height", target="T001", height_m=7.0), say("Proposal: T001 7 m.")]))
    service = AssistantService(site, SITE, mode="outdoor")
    service.discard()
    assert service.is_available() == (True, None) and service.domain_kind == "fake_site"
    assert service.model_path() == path and service.model()["trees"]["T001"]["height_m"] == 5.0
    frame = service.view_frame()
    assert frame["origin_offset"] == [100.0, 200.0, 3.0] and frame["crs"] == "LOCAL" and frame["units_per_metre"] == 1.0
    assert service.hit_test(10.2, 19.9)["id"] == "T001"
    assert service.outlines_for([{"type": "element", "kind": "tree", "id": "T001"}])[0]["id"] == "T001"
    result = service.ask("tree 2 m higher", [{"type": "element", "kind": "tree", "id": "T001", "label": "tree T001"}])
    assert result.error is None and not result.applied and service.pending() is result
    assert service.preview()["moved"] == [{"id": "T001", "kind": "tree", "change": "modified"}]
    assert path.read_bytes() == original
    edit_id = service.accept()
    assert _load(path)["trees"]["T001"]["height_m"] == 7.0 and service.edits()[0]["id"] == edit_id
    assert service.export_actions() == ["export_site"]
    service.undo(edit_id)
    assert _load(path)["trees"]["T001"]["height_m"] == 5.0
    assert not plan_model_path(site, SITE).exists()  # the indoor plan model was never touched


def test_fake_domain_full_control_applies_directly(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable("openai")
    path = _model_path(site, SITE)
    _use(monkeypatch, FakeProvider([call("set_height", target="T001", height_m=8.0), say("T001 becomes 8 m.")]))
    service = AssistantService(site, SITE, mode="outdoor")
    service.discard()
    assert service.tool_tier == TIER_FULL
    result = service.ask("tree 3 m higher", [])
    assert result.applied and result.edit_id and service.pending() is None
    assert _load(path)["trees"]["T001"]["height_m"] == 8.0
    service.undo(result.edit_id)
    assert _load(path)["trees"]["T001"]["height_m"] == 5.0
    assert not plan_model_path(site, SITE).exists()


def test_outdoor_without_domain_never_falls_back_to_indoor(tmp_path: Path) -> None:
    import yaml

    project = tmp_path / "ortho_outdoor"
    project.mkdir()
    document = {"project": {"name": "synthetic"}, "settings": {"classification": {"class_set": "outdoor"}}, "workflow": {}}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    service = AssistantService(project, SITE, mode="outdoor")
    assert service.unsupported_reason == "assistant.agent.needs_environment"
    assert service.export_actions() == [] and service.model_path() is None
    assert service.hit_test(1.0, 2.0) == {"type": "point", "x": 1.0, "y": 2.0}
    assert service.outlines_for([{"type": "element", "kind": "wall", "id": "W001"}]) == []
    assert service.view_frame() == {"units_per_metre": 1.0, "floor_z": None}
    assert service.edits() == [] and service.stale() == []
    # an indoor level without plan keeps the phase 1 indoor fallback
    indoor = AssistantService(tmp_path / "nothing", LEVEL_ID)
    assert indoor.export_actions() == ["export", "export_ifc"]


# ------------------------------------------------------------------------------------------- repeat and chain

def test_repeating_the_same_change_applies_nothing_the_second_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable("openai")
    project = make_project(tmp_path, with_cloud=False)
    path = plan_model_path(project, LEVEL_ID)
    op = dict(target="W005", thickness_m=0.12)
    _use(monkeypatch, FakeProvider([call("set_wall_thickness", **op), say("W005 wird 0.12 m dick."),
                                    call("set_wall_thickness", **op), say("W005 wird 0.12 m dick.")]))
    service = AssistantService(project, LEVEL_ID)
    service.discard()
    first = service.ask("Innenwand 12 cm", [])
    assert first.applied and first.edit_id
    after_first = path.read_bytes()
    second = service.ask("Innenwand 12 cm", [])
    assert not second.applied and second.edit_id is None and service.pending() is None
    assert path.read_bytes() == after_first
    assert [edit["id"] for edit in service.edits()] == [first.edit_id]


def test_edit_on_a_new_wall_replays_after_a_renumbering_recompute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from assistant_synthetic import renumbered_recompute
    from samcloud.assistant.history import ensure_applied

    _enable("openai")
    project = make_project(tmp_path, with_cloud=False)
    path = plan_model_path(project, LEVEL_ID)
    _use(monkeypatch, FakeProvider([
        call("add_wall", start=[4.05, 3.5], end=[7.9, 3.5], thickness_m=0.10), say("Neue Wand W006."),
        call("set_wall_thickness", target="W006", thickness_m=0.15), say("W006 wird 0.15 m dick."),
    ]))
    service = AssistantService(project, LEVEL_ID)
    service.discard()
    added = service.ask("neue Wand", [])
    thicker = service.ask("die neue Wand 15 cm", [])
    assert added.applied and thicker.applied
    # the pipeline recomputes the plan (without the assistant's wall): reversed order, new ids, millimetres off
    save_plan_model(renumbered_recompute(make_plan()), path.parent)
    report = ensure_applied(project, LEVEL_ID)
    plan = load_plan_model(path)
    new_walls = [w for w in plan.walls if w.evidence == "assistant"]
    assert len(new_walls) == 1
    wall = new_walls[0]
    assert wall.centreline[0][1] == pytest.approx(3.5) and wall.centreline[1][1] == pytest.approx(3.5)
    stale = {item["id"] for item in report.stale}
    if thicker.edit_id in stale:  # allowed: reported stale, never applied to another element
        assert wall.thickness_m == pytest.approx(0.10)
    else:
        assert report.stale == [] and set(report.applied) == {added.edit_id, thicker.edit_id}
        assert wall.thickness_m == pytest.approx(0.15)
    # no other wall picked up the second edit
    assert sorted(round(w.thickness_m, 3) for w in plan.walls if w is not wall) == [0.2, 0.3, 0.3, 0.3, 0.3]
    assert [w.id for w in plan.walls].count(wall.id) == 1
