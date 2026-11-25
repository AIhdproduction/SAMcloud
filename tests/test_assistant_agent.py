"""AI assistant agent loop with a scripted fake provider (no network): proposals, evidence rule, QA rounds, limits."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Callable

import pytest

from assistant_synthetic import EMPTY_BOX, LEVEL_ID, make_project
from samcloud.assistant import agent as agent_module
from samcloud.assistant.costs import CostLedger
from samcloud.assistant.paths import plan_model_path
from samcloud.assistant.providers.base import (
    AuthError,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    ProviderTimeout,
    RateLimited,
    ToolCall,
    Unreachable,
    Usage,
)
from samcloud.assistant.qa import QaResult
from samcloud.assistant.settings import AssistantSettings

DUMMY_KEY = "sk-test-DUMMY-0000"
DOOR = {"type": "element", "kind": "opening", "id": "O001", "label": "door O001", "subkind": "door", "width_m": 0.9}
_IDS = itertools.count(1)


# ------------------------------------------------------------------------------------------------------- fake provider

def call(tool: str, /, **arguments: Any) -> ChatMessage:
    """An assistant message with one tool call."""
    return ChatMessage("assistant", "", [ToolCall(f"call_{next(_IDS)}", tool, arguments)])


def calls(*items: tuple[str, dict[str, Any]]) -> ChatMessage:
    return ChatMessage("assistant", "", [ToolCall(f"call_{next(_IDS)}", name, args) for name, args in items])


def say(text: str) -> ChatMessage:
    return ChatMessage("assistant", text)


class FakeProvider:
    """Scripted provider implementing the Provider protocol; records every chat request."""

    name = "fake"

    def __init__(self, script: list[Any], *, tools: bool | None = True, images: bool | None = False,
                 price: tuple[float, float] | None = (0.0, 0.0), info_error: Exception | None = None,
                 usage: tuple[int, int] = (1000, 200)) -> None:
        self.script = list(script)
        self.requests: list[list[ChatMessage]] = []
        self.tools_seen: list[list[str]] = []
        self.supports_tools = tools
        self.supports_images = images
        self.price = price
        self.info_error = info_error
        self.usage = usage

    def list_models(self) -> list[ModelInfo]:
        return [self.model_info("fake-model")]

    def model_info(self, model: str) -> ModelInfo:
        if self.info_error is not None:
            raise self.info_error
        price = self.price or (None, None)
        return ModelInfo(model, self.supports_tools, self.supports_images, price[0], price[1])

    def chat(self, messages: list[ChatMessage], tools: list[Any], *, model: str, max_tokens: int, timeout: float) -> ChatResponse:
        self.requests.append(copy.deepcopy(messages))
        self.tools_seen.append([spec.name for spec in tools])
        if not self.script:
            step: Any = say("done")
        else:
            step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            step = step(messages)
        message = copy.deepcopy(step)
        return ChatResponse(message, Usage(*self.usage), "tool_use" if message.tool_calls else "end_turn", model)

    def test_connection(self, model: str | None) -> Any:  # pragma: no cover - protocol completeness
        raise NotImplementedError


def settings_for(provider: str = "ollama", **overrides: Any) -> AssistantSettings:
    settings = AssistantSettings(enabled=True, provider=provider, model="fake-model", cost_limit_request_usd=0.0,
                                 cost_limit_project_day_usd=0.0)
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read the real user's AI credentials (redaction loads them)."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return make_project(tmp_path)


def turn(project: Path, provider: FakeProvider, prompt: str = "test", selection: list[dict[str, Any]] | None = None,
         settings: AssistantSettings | None = None, **kwargs: Any):
    return agent_module.run_turn(project=project, level_id=LEVEL_ID, prompt=prompt, selection=selection or [],
                                 settings=settings or settings_for(), creds={}, provider=provider, **kwargs)


def tool_messages(request: list[ChatMessage]) -> list[ChatMessage]:
    return [m for m in request if m.role == "tool"]


# ------------------------------------------------------------------------------------------------------------- tests

def test_read_edit_answer_gives_checked_proposal_and_leaves_the_plan_file(project: Path) -> None:
    plan_file = plan_model_path(project, LEVEL_ID)
    before = plan_file.read_bytes()
    provider = FakeProvider([
        calls(("list_elements", {"kind": "wall"}), ("element_details", {"id": "W005"})),
        call("points_in_box", box={"x": [3.85, 4.15], "y": [1.0, 2.0]}),
        call("move_wall", target="W005", offset_m=0.25),
        say("Die Trennwand W005 liegt jetzt 25 cm weiter Richtung face_a."),
    ])
    result = turn(project, provider, "Verschiebe diese Wand um 25 cm",
                  [{"type": "element", "kind": "wall", "id": "W005", "label": "W005"}])
    assert result.error is None, result.error
    assert result.proposal is not None and result.qa is not None and result.qa.ok
    wall = next(entry for entry in result.summary if entry["element_id"] == "W005")
    assert wall["change"] == "moved"
    shift = abs(wall["after"]["centreline"][0][0] - wall["before"]["centreline"][0][0])
    assert shift == pytest.approx(0.25, abs=1e-6)
    assert "W005" in result.reply
    assert [entry["tool"] for entry in result.tool_log] == ["list_elements", "element_details", "points_in_box", "move_wall"]
    assert all(entry["ok"] for entry in result.tool_log)
    assert plan_file.read_bytes() == before
    # the system prompt holds the rules, the context the selection, every tool result is JSON
    first = provider.requests[0]
    assert first[0].role == "system" and "Evidenz" not in first[0].content  # English UI in tests
    assert "No evidence = nothing" in first[0].content
    assert '"W005"' in first[1].content and "Verschiebe" in first[-1].content
    assert len(tool_messages(provider.requests[1])) == 2
    assert {"move_wall", "add_furniture", "list_elements", "section"} <= set(provider.tools_seen[0])


def test_door_width_example_summary(project: Path) -> None:
    provider = FakeProvider([call("set_opening", target="O001", width_m=1.0), say("Die Tür ist jetzt 1.00 m breit.")])
    result = turn(project, provider, "mach diese Tür 1 m breit", [DOOR])
    assert result.error is None and result.proposal is not None
    assert result.summary == [entry for entry in result.summary if entry["element_id"] == "O001"]
    assert (result.summary[0]["before"]["width_m"], result.summary[0]["after"]["width_m"]) == (0.9, 1.0)
    assert result.needs_confirmation is False and result.asks_manual is False


def test_no_evidence_asks_and_manual_needs_consent(project: Path) -> None:
    plan_file = plan_model_path(project, LEVEL_ID)
    before = plan_file.read_bytes()
    point = {"type": "point", "x": 7.0, "y": 4.0}
    first = FakeProvider([
        call("points_in_box", box=EMPTY_BOX),
        call("add_furniture", box=EMPTY_BOX, kind="chair"),
        say("Dort ist in der Punktwolke kein Stuhl. Soll ich trotzdem einen setzen?"),
    ])
    result = turn(project, first, "da fehlt noch ein Stuhl", [point])
    assert result.error is None
    assert result.tool_log[-1]["error_key"] == "assistant.edit.no_evidence"
    assert result.asks_manual is True and result.proposal is None and result.summary == []
    assert '"key":"assistant.edit.no_evidence"' in tool_messages(first.requests[2])[-1].content

    # the user insists: the UI starts the next turn with allow_manual=True
    second = FakeProvider([call("add_furniture", box=EMPTY_BOX, kind="chair", manual=True), say("Stuhl manuell gesetzt.")])
    result = turn(project, second, "ja, trotzdem setzen", [point], allow_manual=True,
                  history_messages=[{"role": "user", "text": "da fehlt noch ein Stuhl"},
                                    {"role": "assistant", "text": "Soll ich trotzdem einen setzen?"}])
    assert result.error is None and result.proposal is not None and result.asks_manual is False
    added = [entry for entry in result.summary if entry["change"] == "added"]
    assert len(added) == 1 and added[0]["after"]["evidence"] == "manual"
    item = next(f for f in result.proposal.plan.furniture if f.id == added[0]["element_id"])
    assert item.layer == "A-FURN-MANUAL" and item.fit["evidence"] == "manual"
    assert any(m.role == "assistant" and "trotzdem" in m.content for m in second.requests[0])

    # without consent the code refuses a manual add, whatever the model wants
    third = FakeProvider([call("add_furniture", box=EMPTY_BOX, kind="chair", manual=True), say("Soll ich ihn manuell setzen?")])
    result = turn(project, third, "setz einen Stuhl", [point], allow_manual=False)
    assert result.proposal is None and result.asks_manual is True
    assert result.tool_log[0]["error_key"] == "assistant.edit.manual_not_allowed"
    assert plan_file.read_bytes() == before


def test_measured_add_where_the_cloud_shows_the_chair(project: Path) -> None:
    box = {"x": [5.25, 5.80], "y": [2.05, 2.65]}
    provider = FakeProvider([call("add_furniture", box=box, kind="chair"), say("Stuhl gemessen und gesetzt.")])
    result = turn(project, provider, "da fehlt ein Stuhl")
    assert result.error is None and result.proposal is not None
    added = [entry for entry in result.summary if entry["change"] == "added"]
    assert len(added) == 1 and added[0]["after"]["evidence"] == "measured"


def _qa(ok: bool) -> QaResult:
    findings = [] if ok else [{"check": "wall_overlap", "subject": "W001|W005", "detail": {"area_m2": 0.12}}]
    return QaResult(ok, findings, findings, None, 0, "synthetic")


def test_qa_failure_gets_one_correction_round(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = [_qa(False), _qa(True)]
    seen: list[Any] = []

    def fake_qa(plan: Any, *, baseline: Any, work_dir: Path) -> QaResult:
        seen.append(work_dir)
        return results.pop(0)

    monkeypatch.setattr(agent_module, "run_qa", fake_qa)
    provider = FakeProvider([
        call("move_wall", target="W005", offset_m=0.3),
        say("Wand verschoben."),
        call("move_wall", target="W005", offset_m=-0.1),
        say("Korrigiert: Wand nur 20 cm verschoben."),
    ])
    result = turn(project, provider, "Wand verschieben")
    assert result.error is None and result.proposal is not None and result.qa is not None and result.qa.ok
    assert len(seen) == 2 and seen[0].name == "qa_tmp"
    feedback = provider.requests[2][-1]
    assert feedback.role == "user" and "QA" in feedback.content and "wall_overlap" in feedback.content
    assert result.reply.startswith("Korrigiert")


def test_qa_failing_twice_discards_the_proposal(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "run_qa", lambda plan, *, baseline, work_dir: _qa(False))
    provider = FakeProvider([call("move_wall", target="W005", offset_m=0.3), say("Verschoben."), say("Ich sehe keinen Fehler.")])
    result = turn(project, provider, "Wand verschieben")
    assert result.proposal is None and result.summary == [] and result.needs_confirmation is False
    assert result.qa is not None and not result.qa.ok
    assert "wall_overlap" in result.reply and "W001|W005" in result.reply
    assert len(provider.requests) == 3


def test_tool_call_limit_stops_without_proposal(project: Path) -> None:
    provider = FakeProvider([call("move_wall", target="W005", offset_m=0.2)] + [call("list_elements")] * 10)
    result = turn(project, provider, "Wand verschieben", settings=settings_for(max_tool_calls=3))
    assert result.error_kind == "too_many_tools" and "3" in (result.error or "")
    assert result.proposal is None and result.summary == []
    assert len(result.tool_log) == 3 and len(provider.requests) == 4


def test_cost_limit_blocks_before_sending(project: Path) -> None:
    settings = settings_for("openai", cost_limit_request_usd=0.01, cost_limit_project_day_usd=5.0)
    expensive = FakeProvider([say("x")], price=(50.0, 200.0))
    result = turn(project, expensive, "Wand verschieben", settings=settings)
    assert expensive.requests == [] and result.error_kind == "cost_limit"
    assert "0.01" in (result.error or "") and result.proposal is None

    unknown = FakeProvider([say("x")], price=None)
    result = turn(project, unknown, "Wand verschieben", settings=settings)
    assert unknown.requests == [] and result.error_kind == "cost_limit" and "fake-model" in (result.error or "")

    # the daily limit counts what this project already spent today
    ledger = CostLedger(project)
    ledger.add(4.99)
    day = settings_for("openai", cost_limit_request_usd=1.0, cost_limit_project_day_usd=5.0)
    cheap = FakeProvider([say("x")], price=(1.0, 2.0))
    result = turn(project, cheap, "Wand verschieben", settings=day)
    assert cheap.requests == [] and result.error_kind == "cost_limit"

    # Ollama is never blocked by cost limits, even without a price
    local = FakeProvider([say("Nichts zu tun.")], price=None)
    result = turn(project, local, "hallo", settings=settings_for("ollama", cost_limit_request_usd=0.01, cost_limit_project_day_usd=0.01))
    assert result.error is None and len(local.requests) == 1


def test_real_cost_goes_to_the_ledger(project: Path) -> None:
    settings = settings_for("openai", cost_limit_request_usd=1.0, cost_limit_project_day_usd=5.0)
    provider = FakeProvider([call("list_elements"), say("ok")], price=(1.0, 2.0))
    result = turn(project, provider, "Liste", settings=settings)
    assert result.error is None
    expected = 2 * (1000 * 1.0 + 200 * 2.0) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (2000, 400)
    assert CostLedger(project).spent_today() == pytest.approx(expected)


def test_model_without_tools_and_provider_errors(project: Path) -> None:
    no_tools = FakeProvider([say("x")], tools=False)
    result = turn(project, no_tools, "Wand verschieben")
    assert no_tools.requests == [] and result.error_kind == "no_tools" and "fake-model" in (result.error or "")

    unreachable = FakeProvider([], info_error=Unreachable("Server http://127.0.0.1:9 is not reachable."))
    result = turn(project, unreachable, "x")
    assert unreachable.requests == [] and result.error_kind == "unreachable"

    for error, kind in ((AuthError(f"bad key {DUMMY_KEY}"), "auth"), (Unreachable("down"), "unreachable"),
                        (RateLimited("slow down", 3.0), "rate_limited"), (ProviderTimeout("no answer"), "timeout")):
        provider = FakeProvider([call("set_opening", target="O001", width_m=1.0), error])
        result = turn(project, provider, "Tür breiter", [DOOR])
        assert result.error_kind == kind and result.error and result.proposal is None
        assert DUMMY_KEY not in result.error and DUMMY_KEY not in result.reply

    broken = FakeProvider([RuntimeError("boom")])
    result = turn(project, broken, "x")
    assert result.error_kind == "error" and "RuntimeError" in (result.error or "")


def test_missing_model_or_plan(project: Path, tmp_path: Path) -> None:
    provider = FakeProvider([say("x")])
    result = turn(project, provider, "x", settings=settings_for(model=""))
    assert result.error_kind == "no_model" and provider.requests == []
    empty = tmp_path / "empty_project"
    empty.mkdir()
    result = turn(empty, provider, "x")
    assert result.error_kind == "no_plan" and provider.requests == []


def _images_sent(provider: FakeProvider) -> list[bytes]:
    return [image for request in provider.requests for message in request for image in message.images]


def test_images_only_when_allowed(project: Path) -> None:
    script: Callable[[], list[Any]] = lambda: [call("section", z_m=1.0, box={"x": [3.0, 5.0], "y": [1.0, 4.0]}), say("ok")]
    text_only = FakeProvider(script(), images=True)
    result = turn(project, text_only, "schau dir die Wand an", settings=settings_for(text_only=True))
    assert result.error is None and _images_sent(text_only) == []
    assert "image" not in tool_messages(text_only.requests[1])[0].content

    blind = FakeProvider(script(), images=False)
    turn(project, blind, "schau", settings=settings_for(text_only=False))
    assert _images_sent(blind) == []

    seeing = FakeProvider(script(), images=True)
    result = turn(project, seeing, "schau", settings=settings_for(text_only=False))
    images = _images_sent(seeing)
    assert result.error is None and len(images) == 1 and images[0].startswith(b"\x89PNG")
    carrier = next(m for m in seeing.requests[1] if m.images)
    assert carrier.role == "user"


def test_deletion_needs_confirmation(project: Path) -> None:
    provider = FakeProvider([call("delete_opening", target="O002"), say("Fenster O002 gelöscht.")])
    result = turn(project, provider, "dieses Fenster gibt es nicht")
    assert result.proposal is not None and result.needs_confirmation is True
    assert result.summary[0]["change"] == "removed"


def test_bad_tool_calls_become_error_results(project: Path) -> None:
    provider = FakeProvider([
        calls(("make_coffee", {}), ("set_opening", {"target": "O001", "width_m": 9.0}),
              ("add_furniture", {"box": EMPTY_BOX, "kind": "chair", "_item": {"label": "Stuhl"}}),
              ("points_in_box", {"box": {"x": [1.0]}}), ("element_details", {"id": "W999"})),
        say("Das ging nicht."),
    ])
    result = turn(project, provider, "x")
    assert result.error is None and result.proposal is None
    keys = [entry["error_key"] for entry in result.tool_log]
    assert keys == ["assistant.agent.unknown_tool", "assistant.edit.invalid_value", "assistant.edit.bad_params",
                    "assistant.agent.bad_arguments", "assistant.edit.not_found"]


def test_cost_limit_inside_the_loop_after_the_first_response(project: Path) -> None:
    ledger = CostLedger(project)
    ledger.add(4.98)
    settings = settings_for("openai", cost_limit_request_usd=1.0, cost_limit_project_day_usd=5.0)
    # the first request fits (about 0.015 USD estimated); its real cost (0.022 USD) crosses the daily limit
    provider = FakeProvider([call("set_opening", target="O001", width_m=1.0), say("ok")], price=(1.0, 2.0), usage=(20000, 1000))
    result = turn(project, provider, "Tür breiter", [DOOR], settings=settings)
    assert len(provider.requests) == 1
    assert result.error_kind == "cost_limit" and result.proposal is None and result.summary == []
    assert result.cost_usd == pytest.approx(0.022)
    assert CostLedger(project).spent_today() == pytest.approx(5.002)
    assert [entry["tool"] for entry in result.tool_log] == ["set_opening"]


def test_tool_call_cap_counts_the_qa_correction_round(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = [_qa(False), _qa(True)]
    monkeypatch.setattr(agent_module, "run_qa", lambda plan, *, baseline, work_dir: results.pop(0))
    provider = FakeProvider([
        call("move_wall", target="W005", offset_m=0.3),
        say("Verschoben."),
        calls(("move_wall", {"target": "W005", "offset_m": -0.1}), ("list_elements", {})),
        say("Korrigiert."),
    ])
    result = turn(project, provider, "Wand verschieben", settings=settings_for(max_tool_calls=2))
    assert result.error_kind == "too_many_tools" and result.proposal is None and result.summary == []
    assert [entry["tool"] for entry in result.tool_log] == ["move_wall", "move_wall"]
    assert len(provider.requests) == 3 and len(results) == 1  # the QA ran once, the correction round hit the cap


def test_credential_values_are_redacted(project: Path) -> None:
    secret = "custom-provider-secret-1234"
    provider = FakeProvider([call("list_elements", label=secret), say(f"Antwort mit {secret}")])
    result = agent_module.run_turn(project=project, level_id=LEVEL_ID, prompt=f"mein Key {secret}", selection=[],
                                   settings=settings_for(), creds={"openai_api_key": secret, "ollama_url": "http://localhost:11434"},
                                   provider=provider, history_messages=[{"role": "user", "text": f"vorher {secret}"}])
    assert secret not in result.reply and secret not in str(result.tool_log)
    assert all(secret not in m.content for m in provider.requests[0])
    failing = FakeProvider([RuntimeError("x")], info_error=AuthError(f"rejected {secret}"))
    result = agent_module.run_turn(project=project, level_id=LEVEL_ID, prompt="x", selection=[], settings=settings_for(),
                                   creds={"anthropic_api_key": secret}, provider=failing)
    assert result.error_kind == "auth" and secret not in (result.error or "")
