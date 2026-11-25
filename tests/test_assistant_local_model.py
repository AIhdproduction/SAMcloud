"""Local Ollama model over several questions: num_ctx options, context budget, fresh-context retry, model choice.

Everything runs against a fake Ollama HTTP server on 127.0.0.1 (no network, no real model).
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from assistant_synthetic import LEVEL_ID, make_plan, make_project
from samcloud.app.i18n import t
from samcloud.assistant import agent as agent_module
from samcloud.assistant.providers import ContextError, is_context_error
from samcloud.assistant.providers.base import (
    AuthError,
    BadResponse,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    Usage,
    estimate_tokens,
    get_provider,
    map_http_error,
)
from samcloud.assistant.providers.ollama import (
    OllamaProvider,
    context_window,
    model_warnings,
    rank_models,
    recommended_model,
    request_num_ctx,
)
from samcloud.assistant.settings import AssistantSettings, load_settings, normalize_num_ctx, save_settings
from samcloud.bim.plan_model import BuildingPlan, PlanFurniture
from test_assistant_providers import FakeServer

MODEL = "qwen-test:14b-instruct"
REPEAT_ERROR = (500, {"error": "prediction aborted, token repeat limit reached"})


# ------------------------------------------------------------------------------------------------------------ fixtures

@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read the real user's AI settings / credentials."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer()
    yield fake
    fake.close()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return make_project(tmp_path, with_cloud=False)


def big_plan(count: int = 200) -> BuildingPlan:
    """The synthetic plan plus ``count`` small furniture pieces (positions off any grid) -> a long element list."""
    plan = make_plan()
    for index in range(count):
        x0 = 0.37 + (index % 20) * 0.371 + 0.0013 * index
        y0 = 0.41 + (index // 20) * 0.433 + 0.0007 * index
        plan.furniture.append(PlanFurniture(f"F{index + 100:03d}", None, f"Box {index}", "box", None, "A-FURN",
                                            [[x0, y0], [x0 + 0.21, y0], [x0 + 0.21, y0 + 0.17], [x0, y0 + 0.17]]))
    return plan


def show_reply(context: int | None, *, tools: bool = True) -> dict[str, Any]:
    reply: dict[str, Any] = {"capabilities": ["completion", "tools"] if tools else ["completion"],
                             "details": {"family": "qwen2", "parameter_size": "14.8B"}}
    if context:
        reply["model_info"] = {"general.architecture": "qwen2", "qwen2.context_length": context}
    return reply


def chat_reply(content: str = "No change is needed.", *, calls: list[dict[str, Any]] | None = None,
               done_reason: str = "stop") -> tuple[int, dict[str, Any]]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return 200, {"model": MODEL, "message": message, "done_reason": done_reason, "prompt_eval_count": 50, "eval_count": 7}


def settings_for(**overrides: Any) -> AssistantSettings:
    settings = AssistantSettings(enabled=True, provider="ollama", model=MODEL, cost_limit_request_usd=0.0,
                                 cost_limit_project_day_usd=0.0)
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


def chat_requests(server: FakeServer) -> list[dict[str, Any]]:
    return [request["body"] for request in server.requests if request["path"] == "/api/chat"]


def request_tokens(body: dict[str, Any]) -> int:
    """The agent's own estimate of a request (content + tool calls + tool specs) for comparison with the budget."""
    chars = 0
    for message in body["messages"]:
        chars += len(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            chars += len(call["function"]["name"]) + len(json.dumps(call["function"]["arguments"], default=str))
    for tool in body.get("tools") or []:
        function = tool["function"]
        chars += len(function["name"]) + len(function["description"]) + len(json.dumps(function["parameters"]))
    return estimate_tokens(chars)


def turn(project: Path, server: FakeServer, *, prompt: str = "Is door O001 too wide?", plan: Any = None,
         settings: AssistantSettings | None = None, history: list[dict[str, Any]] | None = None,
         selection: list[dict[str, Any]] | None = None) -> agent_module.TurnResult:
    return agent_module.run_turn(project=project, level_id=LEVEL_ID, prompt=prompt, selection=selection or [],
                                 settings=settings or settings_for(), creds={"ollama_url": server.url},
                                 history_messages=history, plan=plan)


def long_history(pairs: int, chars: int = 2000) -> list[dict[str, Any]]:
    history = []
    for index in range(pairs):
        history.append({"role": "user", "text": f"Question {index}: " + "please check the wall lengths " * (chars // 30)})
        history.append({"role": "assistant", "text": f"Answer {index}: " + "the wall W001 measures 8.00 m " * (chars // 30)})
    return history


def context_elements(body: dict[str, Any]) -> list[Any]:
    context = body["messages"][1]["content"]
    return json.loads(context.split("\n", 1)[1])["Elements"]


# --------------------------------------------------------------------------------------------- num_ctx and options

def test_num_ctx_for_qwen_32k_and_a_16k_prompt() -> None:
    needed = 16384 + 4096 + 256
    assert context_window(32768) == 32768
    assert request_num_ctx(32768, "auto", None, needed) == 32768
    assert request_num_ctx(32768, "auto", 16384, needed) == needed  # a lower cap is raised so the prompt fits
    assert request_num_ctx(32768, "auto", None, 50000) == 32768  # never above the model maximum
    assert context_window(131072) == 32768  # auto stays at ollama_num_ctx_max for very large models
    assert context_window(131072, "auto", 65536) == 65536
    assert context_window(8192) == 8192
    assert context_window(32768, "16384") == 16384 and context_window(8192, "65536") == 8192
    assert context_window(None) == 32768


def test_chat_sends_num_ctx_num_predict_and_sampling(server: FakeServer) -> None:
    server.routes[("POST", "/api/show")] = (200, show_reply(32768))
    server.routes[("POST", "/api/chat")] = lambda r: chat_reply()
    provider = OllamaProvider(server.url)
    prompt = "x" * (16000 * 3)  # about 16k tokens
    provider.chat([ChatMessage("system", "rules"), ChatMessage("user", prompt)], [], model=MODEL, max_tokens=4096,
                  timeout=5)
    options = chat_requests(server)[-1]["options"]
    estimated = estimate_tokens(len(prompt) + len("rules"))
    assert estimated + 4096 <= options["num_ctx"] <= 32768
    assert options["num_ctx"] == 32768
    assert options["num_predict"] == 4096
    assert options["temperature"] == pytest.approx(0.2)
    assert options["repeat_penalty"] == pytest.approx(1.1)
    assert options["top_p"] == pytest.approx(0.9)
    provider.chat([ChatMessage("user", "short")], [], model=MODEL, max_tokens=256, timeout=5)
    assert chat_requests(server)[-1]["options"]["num_ctx"] == 32768  # stable: Ollama does not reload the model
    assert sum(1 for request in server.requests if request["path"] == "/api/show") == 1  # /api/show cached


def test_num_ctx_setting_and_model_maximum(server: FakeServer) -> None:
    server.routes[("POST", "/api/show")] = (200, show_reply(32768))
    server.routes[("POST", "/api/chat")] = lambda r: chat_reply()
    provider = get_provider("ollama", settings_for(ollama_num_ctx="16384"), {"ollama_url": server.url})
    assert isinstance(provider, OllamaProvider)
    provider.chat([ChatMessage("user", "short")], [], model=MODEL, max_tokens=1024, timeout=5)
    assert chat_requests(server)[-1]["options"]["num_ctx"] == 16384
    provider.chat([ChatMessage("user", "y" * 60000)], [], model=MODEL, max_tokens=1024, timeout=5)
    raised = chat_requests(server)[-1]["options"]["num_ctx"]
    assert estimate_tokens(60000) + 1024 <= raised <= 32768

    server.routes[("POST", "/api/show")] = (200, show_reply(8192))
    small = OllamaProvider(server.url, num_ctx="auto", num_ctx_max=32768)
    small.chat([ChatMessage("user", "short")], [], model="small:3b", max_tokens=1024, timeout=5)
    assert chat_requests(server)[-1]["options"]["num_ctx"] == 8192


@pytest.mark.parametrize("reply, expected", [
    ({"model_info": {"llama.context_length": 131072}}, 131072),
    ({"details": {"context_length": 40960}}, 40960),
    ({"parameters": "stop \"<|im_end|>\"\nnum_ctx 8192\ntemperature 0.7"}, 8192),
    ({"capabilities": ["completion", "tools"]}, None),
])
def test_context_length_from_api_show(server: FakeServer, reply: dict[str, Any], expected: int | None) -> None:
    server.routes[("POST", "/api/show")] = (200, reply)
    assert OllamaProvider(server.url).model_info("any").context_length == expected


# -------------------------------------------------------------------------------------------- context budget (agent)

def test_five_turns_stay_within_the_budget(project: Path, server: FakeServer) -> None:
    server.routes[("POST", "/api/show")] = (200, show_reply(8192))
    server.routes[("POST", "/api/chat")] = lambda r: chat_reply("Proposal: nothing to change.")
    plan = big_plan()
    history: list[dict[str, Any]] = []
    results = []
    for index in range(5):
        prompt = f"Question {index + 1}: is the table F001 placed correctly?"
        result = turn(project, server, prompt=prompt, plan=plan, history=history)
        assert result.ok, result.error
        assert not result.retried_fresh
        results.append(result)
        history += long_history(1)
        history[-2]["text"] = prompt + history[-2]["text"]
    budget = agent_module.context_budget(8192, 4096)
    requests = chat_requests(server)
    assert len(requests) == 5
    sizes = [request_tokens(body) for body in requests]
    for body, size, result in zip(requests, sizes, results):
        assert size <= budget.prompt_tokens
        assert body["options"]["num_ctx"] == 8192
        assert body["options"]["num_ctx"] >= result.prompt_tokens + body["options"]["num_predict"]
        assert body["options"]["num_predict"] == budget.max_output
        assert len(body["messages"]) - 3 <= 2 * agent_module.HISTORY_TURNS  # history messages
        assert result.context_window == 8192 and 0 < result.prompt_tokens <= budget.prompt_tokens
    assert sizes[4] <= sizes[2] * 1.05  # does not grow with the chat history
    assert len(context_elements(requests[0])) < 80  # the element list was shortened to fit
    assert results[0].to_dict()["context_window"] == 8192


def test_history_is_limited_to_the_configured_turns(project: Path, server: FakeServer) -> None:
    server.routes[("POST", "/api/show")] = (200, show_reply(32768))
    server.routes[("POST", "/api/chat")] = lambda r: chat_reply()
    result = turn(project, server, history=long_history(6, chars=300), settings=settings_for(context_history_turns=2))
    assert result.ok
    body = chat_requests(server)[-1]
    assert len(body["messages"]) == 3 + 4
    assert body["messages"][2]["content"].startswith("Question 4")  # the last two question/answer pairs


def test_selected_and_nearby_elements_come_first(project: Path, server: FakeServer) -> None:
    server.routes[("POST", "/api/show")] = (200, show_reply(32768))
    server.routes[("POST", "/api/chat")] = lambda r: chat_reply()
    selection = [{"type": "element", "kind": "furniture", "id": "F250", "label": "Box 150"}]
    assert turn(project, server, plan=big_plan(), selection=selection).ok
    elements = context_elements(chat_requests(server)[-1])
    assert elements[0]["id"] == "F250"
    first = elements[0]["position"]
    distances = [(row["position"][0] - first[0]) ** 2 + (row["position"][1] - first[1]) ** 2
                 for row in elements[1:6] if row.get("position")]
    assert distances == sorted(distances)


def test_long_tool_results_are_compressed_inside_one_turn(project: Path, server: FakeServer) -> None:
    server.routes[("POST", "/api/show")] = (200, show_reply(16384))
    replies = iter([chat_reply("", calls=[{"function": {"name": "list_elements", "arguments": {}}}])] * 4
                   + [chat_reply("Proposal: nothing to change.")])
    server.routes[("POST", "/api/chat")] = lambda r: next(replies)
    result = turn(project, server, plan=big_plan())
    assert result.ok, result.error
    budget = agent_module.context_budget(16384, 4096)
    requests = chat_requests(server)
    assert len(requests) == 5
    assert all(request_tokens(body) <= budget.prompt_tokens for body in requests)
    tool_contents = [m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_contents) == 4
    first_result = [m["content"] for m in requests[1]["messages"] if m["role"] == "tool"][0]
    assert len(first_result) > agent_module.TOOL_RESULT_CHARS[0]  # sent in full while it fit
    assert len(tool_contents[0]) <= agent_module.TOOL_RESULT_CHARS[0]  # older results compressed later
    assert json.loads(tool_contents[0])  # still valid JSON for the model


def test_compress_tool_result_keeps_json() -> None:
    payload = json.dumps({"total": 300, "elements": [{"id": f"F{i}", "position": [i * 1.23456, 2.34567]} for i in range(300)]})
    short = agent_module.compress_tool_result(payload, 1500)
    assert len(short) <= 1500
    data = json.loads(short)
    assert data["total"] == 300 and data["elements"][1]["position"] == [1.23, 2.35]
    assert agent_module.compress_tool_result("plain " * 100, 100).startswith('{"truncated": true')
    assert agent_module.compress_tool_result("small", 100) == "small"


# ------------------------------------------------------------------------------------------------ fresh-context retry

def test_repeat_limit_is_retried_once_in_a_fresh_context(project: Path, server: FakeServer) -> None:
    server.routes[("POST", "/api/show")] = (200, show_reply(32768))
    replies = iter([REPEAT_ERROR, chat_reply("Proposal: nothing to change.")])
    server.routes[("POST", "/api/chat")] = lambda r: next(replies)
    prompt = "Make door O001 narrower?"
    selection = [{"type": "element", "kind": "opening", "id": "O001", "label": "door O001", "subkind": "door"}]
    result = turn(project, server, prompt=prompt, plan=big_plan(), history=long_history(3, chars=300),
                  selection=selection)
    assert result.ok, result.error
    assert result.retried_fresh is True and result.to_dict()["retried_fresh"] is True
    assert result.notes[0] == t("assistant.agent.retried_fresh")
    assert result.reply == "Proposal: nothing to change."
    first, second = chat_requests(server)
    assert len(first["messages"]) == 3 + 6  # with the chat history
    assert len(second["messages"]) == 3  # system, context, question: no history, no tool history
    assert [m["role"] for m in second["messages"]] == ["system", "user", "user"]
    assert second["messages"][-1]["content"] == prompt
    assert second["messages"][0] == first["messages"][0]
    assert len(context_elements(second)) <= agent_module.FRESH_ELEMENT_CAP < len(context_elements(first))
    assert context_elements(second)[0]["id"] == "O001"  # same selection
    assert "O001" in second["messages"][1]["content"]


def test_second_failure_gives_a_clear_error_with_a_tip(project: Path, server: FakeServer) -> None:
    server.routes[("POST", "/api/show")] = (200, show_reply(32768))
    server.routes[("POST", "/api/chat")] = REPEAT_ERROR
    result = turn(project, server, history=long_history(2, chars=300))
    assert not result.ok
    assert result.error_kind == "context" and result.retried_fresh is True
    assert len(chat_requests(server)) == 2  # only one retry
    assert "larger model" in result.error and "new chat context" in result.error
    assert result.proposal is None


def test_other_server_errors_are_not_retried(project: Path, server: FakeServer) -> None:
    server.routes[("POST", "/api/show")] = (200, show_reply(32768))
    server.routes[("POST", "/api/chat")] = (500, {"error": "llama runner process has terminated"})
    result = turn(project, server)
    assert not result.ok and result.error_kind == "error" and not result.retried_fresh
    assert len(chat_requests(server)) == 1


def test_empty_or_truncated_answer_is_retried(project: Path, server: FakeServer) -> None:
    server.routes[("POST", "/api/show")] = (200, show_reply(32768))
    replies = iter([chat_reply("", done_reason="length"), chat_reply("Proposal: nothing to change.")])
    server.routes[("POST", "/api/chat")] = lambda r: next(replies)
    result = turn(project, server, history=long_history(1, chars=300))
    assert result.ok and result.retried_fresh
    assert len(chat_requests(server)) == 2


class ScriptedCloud:
    """Cloud-like provider (no context_window_for) that fails with a context-length error first."""

    name = "fake"

    def __init__(self, steps: list[Any]) -> None:
        self.steps = list(steps)
        self.requests: list[list[ChatMessage]] = []

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(model, True, False, 0.0, 0.0, context_length=200000)

    def chat(self, messages: list[ChatMessage], tools: list[Any], *, model: str, max_tokens: int, timeout: float) -> ChatResponse:
        self.requests.append(copy.deepcopy(messages))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return ChatResponse(ChatMessage("assistant", step), Usage(100, 10), "end_turn", model)


def test_cloud_context_error_is_retried_and_cost_limits_still_apply(project: Path) -> None:
    error = map_http_error(400, json.dumps({"error": {"type": "invalid_request_error",
                                                      "message": "prompt is too long: 210000 tokens > 200000 maximum"}}), {})
    assert isinstance(error, ContextError)
    provider = ScriptedCloud([error, "Proposal: nothing to change."])
    settings = settings_for(provider="anthropic", model="vendor-large")
    result = agent_module.run_turn(project=project, level_id=LEVEL_ID, prompt="check", selection=[], settings=settings,
                                   creds={}, provider=provider, history_messages=long_history(2, chars=200))
    assert result.ok and result.retried_fresh
    assert result.context_window == 200000
    assert len(provider.requests[1]) == 3

    # a cost limit stops the retry like any other request (unknown price while a limit is set)
    blocked = ScriptedCloud([error, "never sent"])
    blocked.model_info = lambda model: ModelInfo(model, True, False, None, None, context_length=200000)  # type: ignore[method-assign]
    limited = settings_for(provider="anthropic", model="vendor-unpriced", cost_limit_request_usd=0.5)
    result = agent_module.run_turn(project=project, level_id=LEVEL_ID, prompt="check", selection=[], settings=limited,
                                   creds={}, provider=blocked)
    assert result.error_kind == "cost_limit" and blocked.requests == []


@pytest.mark.parametrize("code, body, context", [
    (500, {"error": "prediction aborted, token repeat limit reached"}, True),
    (400, {"error": {"message": "This model's maximum context length is 128000 tokens.", "code": "context_length_exceeded"}}, True),
    (400, {"error": {"message": "prompt is too long: 210000 tokens > 200000 maximum"}}, True),
    (413, {"error": "input is too long for the context window"}, True),
    (500, {"error": "llama runner process has terminated"}, False),
    (401, {"error": {"message": "context length invalid key"}}, False),
])
def test_context_errors_are_classified(code: int, body: dict[str, Any], context: bool) -> None:
    error = map_http_error(code, json.dumps(body), {})
    assert is_context_error(error) is context
    assert (error.kind == "context") is context


def test_is_context_error_reads_other_error_classes() -> None:
    assert is_context_error(BadResponse("HTTP 500 (prediction aborted, token repeat limit reached)"))
    assert not is_context_error(AuthError("maximum context"))
    assert not is_context_error(ValueError("context length"))


# --------------------------------------------------------------------------------------------------------- model choice

def _models() -> list[ModelInfo]:
    return [
        ModelInfo("tiny-chat:1b", True, False, context_length=8192, size_bytes=800_000_000),
        ModelInfo("big-base:32b", True, False, context_length=32768, size_bytes=19_000_000_000),
        ModelInfo("mid:14b-instruct", True, False, context_length=32768, size_bytes=9_000_000_000),
        ModelInfo("small:7b-instruct", True, False, context_length=32768, size_bytes=4_700_000_000),
        ModelInfo("vision-only:12b", False, True, context_length=131072, size_bytes=8_000_000_000),
        ModelInfo("huge:480b-cloud", True, False, context_length=262144, size_bytes=380, remote=True),
        ModelInfo("unknown:3b", None, None),
    ]


def test_ranking_filters_tools_and_recommends_the_largest_instruct_model() -> None:
    ranked = [info.id for info in rank_models(_models())]
    assert "vision-only:12b" not in ranked
    assert ranked[:3] == ["mid:14b-instruct", "small:7b-instruct", "big-base:32b"]
    assert recommended_model(_models()) == "mid:14b-instruct"
    assert recommended_model([ModelInfo("tiny-chat:1b", True, context_length=8192)]) is None
    warnings = {info.id: model_warnings(info) for info in _models()}
    assert warnings["tiny-chat:1b"] == ["small_context"] and warnings["vision-only:12b"] == ["no_tools"]
    assert warnings["huge:480b-cloud"] == ["remote"] and warnings["unknown:3b"] == ["tools_unknown"]


def test_list_models_reads_tags_and_falls_back_to_show(server: FakeServer) -> None:
    server.routes[("GET", "/api/tags")] = (200, {"models": [
        {"name": "mid:14b-instruct", "size": 9_000_000_000, "capabilities": ["completion", "tools"],
         "details": {"context_length": 32768, "parameter_size": "14.8B"}},
        {"name": "huge:480b-cloud", "size": 380, "remote_host": "https://remote.invalid:443",
         "capabilities": ["completion", "tools"], "details": {"context_length": 262144}},
        {"name": "old:7b", "size": 4_000_000_000, "details": {"parameter_size": "7B"}},
    ]})
    server.routes[("POST", "/api/show")] = lambda r: (200, show_reply(4096, tools=False))
    models = {info.id: info for info in OllamaProvider(server.url).list_models()}
    assert models["mid:14b-instruct"].context_length == 32768 and models["mid:14b-instruct"].supports_tools is True
    assert models["mid:14b-instruct"].size_bytes == 9_000_000_000
    assert models["huge:480b-cloud"].remote is True
    assert models["old:7b"].supports_tools is False and models["old:7b"].context_length == 4096
    shows = [request["body"]["model"] for request in server.requests if request["path"] == "/api/show"]
    assert shows == ["old:7b"]  # only the model without capabilities in /api/tags


def test_settings_keep_ollama_num_ctx(tmp_path: Path) -> None:
    assert [normalize_num_ctx(v) for v in ("auto", 16384, "8192", "lots", 0, None)] == [
        "auto", "16384", "8192", "auto", "auto", "auto"]
    settings = load_settings()
    assert settings.ollama_num_ctx == "auto" and settings.ollama_num_ctx_max == 32768
    assert settings.context_history_turns == 3
    settings.ollama_num_ctx = "16384"
    path = save_settings(settings)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["ollama_num_ctx"] == "16384"
    assert load_settings().ollama_num_ctx == "16384"


# ------------------------------------------------------------------------------------------------------- settings card

class CardProvider:
    name = "fake"

    def __init__(self, models: list[ModelInfo]) -> None:
        self.models = models

    def list_models(self) -> list[ModelInfo]:
        return [copy.copy(info) for info in self.models]


async def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.05)


@pytest.mark.anyio
async def test_card_lists_tool_models_preselects_the_recommendation_and_warns(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app.components import assistant_settings as card
    from test_apriltag_gui import _element, _page_user, _set
    from test_assistant_settings_ui import _hidden, _settings_route

    fake = CardProvider(_models())
    monkeypatch.setattr(card, "get_provider", lambda name, settings, creds: fake)
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert _element(user, "assistant-ollama-num-ctx").value == "auto"
        user.find(marker="assistant-load-models").click()
        select = _element(user, "assistant-model-select")
        await _wait_for(lambda: len(select.options) == 6)
        assert "vision-only:12b" not in select.options
        assert list(select.options)[0] == "mid:14b-instruct"
        assert select.value == "mid:14b-instruct"  # nothing chosen yet: recommendation preselected
        assert "Recommended" in select.options["mid:14b-instruct"] and "32k context" in select.options["mid:14b-instruct"]
        status = _element(user, "assistant-models-status").text
        assert "6 models" in status and "1 without tool support hidden" in status and "mid:14b-instruct" in status
        assert _hidden(user, "assistant-model-warning")  # the recommendation has no warning
        _set(user, "assistant-model-select", "tiny-chat:1b")
        await asyncio.sleep(0.1)
        assert "smaller than 16k" in _element(user, "assistant-model-warning").text
        _set(user, "assistant-model-select", "huge:480b-cloud")
        await asyncio.sleep(0.1)
        assert "leaves this computer" in _element(user, "assistant-model-warning").text

        _set(user, "assistant-model-select", "mid:14b-instruct")
        _set(user, "assistant-ollama-num-ctx", "16384")
        user.find(marker="assistant-save").click()
        await asyncio.sleep(0.1)
    saved = load_settings()
    assert saved.ollama_num_ctx == "16384" and saved.model == "mid:14b-instruct"


@pytest.mark.anyio
async def test_card_keeps_a_chosen_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app.components import assistant_settings as card
    from test_apriltag_gui import _element, _page_user
    from test_assistant_settings_ui import _settings_route

    save_settings(settings_for(model="small:7b-instruct"))
    fake = CardProvider(_models())
    monkeypatch.setattr(card, "get_provider", lambda name, settings, creds: fake)
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        user.find(marker="assistant-load-models").click()
        select = _element(user, "assistant-model-select")
        await _wait_for(lambda: len(select.options) == 6)
        assert select.value == "small:7b-instruct"
