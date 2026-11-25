"""AI assistant provider adapters against a local fake HTTP server (no network)."""

from __future__ import annotations

import base64
import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from samcloud.assistant.providers import (
    AuthError,
    ChatMessage,
    ModelMissing,
    NoToolSupport,
    ProviderTimeout,
    RateLimited,
    ToolCall,
    ToolSpec,
    Unreachable,
    get_provider,
)
from samcloud.assistant.providers.anthropic import AnthropicProvider
from samcloud.assistant.providers.ollama import OllamaProvider
from samcloud.assistant.providers.openai_compat import OpenAICompatProvider
from samcloud.assistant.settings import AssistantSettings

KEY = "sk-test-DUMMY-0000"
PNG = b"\x89PNG\r\n\x1a\nfake"
TOOL = ToolSpec("measure", "Measure an element", {"type": "object", "properties": {"id": {"type": "string"}}})

Reply = tuple[int, Any] | tuple[int, Any, dict[str, str]]


class FakeServer:
    """Tiny JSON server: ``routes[(method, path)]`` is a reply tuple or a callable(request) -> reply."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Reply | Callable[[dict[str, Any]], Reply]] = {}
        self.requests: list[dict[str, Any]] = []
        self.delay = 0.0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep test output quiet
                pass

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                request = {"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                           "body": json.loads(raw) if raw else None}
                server.requests.append(request)
                if server.delay:
                    time.sleep(server.delay)
                route = server.routes.get((self.command, self.path))
                if route is None:
                    reply: Reply = (404, {"error": {"message": f"no route {self.path}"}})
                else:
                    reply = route(request) if callable(route) else route
                status, body = reply[0], reply[1]
                headers = reply[2] if len(reply) > 2 else {}
                payload = json.dumps(body).encode("utf-8") if not isinstance(body, bytes) else body
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:  # client gave up (timeout test)
                    pass

            do_GET = _handle
            do_POST = _handle

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeServer]:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))  # redact() reads the (empty) isolated store
    fake = FakeServer()
    yield fake
    fake.close()


def _closed_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _conversation() -> list[ChatMessage]:
    return [
        ChatMessage("system", "You edit plans."),
        ChatMessage("user", "How long is wall 3?", images=[PNG]),
        ChatMessage("assistant", "Let me measure.", tool_calls=[ToolCall("call_1", "measure", {"id": "3"})]),
        ChatMessage("tool", '{"length_m": 4.21}', tool_call_id="call_1"),
    ]


# --------------------------------------------------------------------------- Ollama


def test_ollama_models_capabilities_and_chat_round(server: FakeServer) -> None:
    server.routes[("GET", "/api/tags")] = (200, {"models": [{"name": "local-a:7b"}, {"model": "local-b"}]})
    server.routes[("POST", "/api/show")] = lambda r: (200, {"capabilities": ["completion", "tools", "vision"]}
                                                      if r["body"]["model"] == "local-a:7b" else {"details": {}})
    replies = iter([
        (200, {"model": "local-a:7b", "message": {"role": "assistant", "content": "",
                                                  "tool_calls": [{"function": {"name": "measure", "arguments": {"id": "3"}}}]},
               "done_reason": "stop", "prompt_eval_count": 120, "eval_count": 15}),
        (200, {"model": "local-a:7b", "message": {"role": "assistant", "content": "Wall 3 is 4.21 m."},
               "done_reason": "stop", "prompt_eval_count": 160, "eval_count": 9}),
    ])
    server.routes[("POST", "/api/chat")] = lambda r: next(replies)
    provider = OllamaProvider(server.url + "/")
    assert [m.id for m in provider.list_models()] == ["local-a:7b", "local-b"]
    info = provider.model_info("local-a:7b")
    assert info.supports_tools is True and info.supports_images is True
    unknown = provider.model_info("local-b")
    assert unknown.supports_tools is None and unknown.supports_images is None

    first = provider.chat([ChatMessage("user", "How long is wall 3?")], [TOOL], model="local-a:7b", max_tokens=256, timeout=5)
    assert first.stop_reason == "tool_use" and first.usage.input_tokens == 120 and first.usage.output_tokens == 15
    call = first.message.tool_calls[0]
    assert call.name == "measure" and call.arguments == {"id": "3"} and call.id  # id generated
    sent = server.requests[-1]["body"]
    assert sent["stream"] is False and sent["options"]["num_predict"] == 256
    assert sent["options"]["num_ctx"] == 32768  # context length unknown: the ollama_num_ctx_max default
    assert {"temperature", "repeat_penalty", "top_p"} <= set(sent["options"])
    assert sent["tools"][0] == {"type": "function", "function": {"name": "measure", "description": "Measure an element",
                                                                 "parameters": TOOL.parameters}}

    second = provider.chat(_conversation(), [TOOL], model="local-a:7b", max_tokens=256, timeout=5)
    assert second.message.content == "Wall 3 is 4.21 m." and second.stop_reason == "end_turn"
    messages = server.requests[-1]["body"]["messages"]
    assert messages[1]["images"] == [base64.b64encode(PNG).decode("ascii")]
    assert messages[2]["tool_calls"] == [{"function": {"name": "measure", "arguments": {"id": "3"}}}]
    assert messages[3] == {"role": "tool", "content": '{"length_m": 4.21}', "tool_name": "measure"}
    assert "authorization" not in server.requests[-1]["headers"]
    assert server.requests[-1]["headers"]["user-agent"] == "SAMcloud"


def test_ollama_connection_results(server: FakeServer) -> None:
    server.routes[("GET", "/api/tags")] = (200, {"models": [{"name": "local-a"}]})
    server.routes[("POST", "/api/show")] = lambda r: ((200, {"capabilities": ["completion"]}) if r["body"]["model"] == "plain"
                                                      else (404, {"error": f"model '{r['body']['model']}' not found"}))
    provider = OllamaProvider(server.url)
    result = provider.test_connection(None)
    assert result.ok and result.kind == "ok" and "1" in result.message
    assert provider.test_connection("missing").kind == "model_missing"
    no_tools = provider.test_connection("plain")
    assert no_tools.kind == "no_tools" and not no_tools.ok
    server.routes[("POST", "/api/chat")] = (400, {"error": "registry/plain does not support tools"})
    with pytest.raises(NoToolSupport):
        provider.chat([ChatMessage("user", "hi")], [TOOL], model="plain", max_tokens=10, timeout=5)
    refused = OllamaProvider(f"http://127.0.0.1:{_closed_port()}").test_connection("local-a")
    assert refused.kind == "unreachable" and not refused.ok


def test_ollama_timeout(server: FakeServer) -> None:
    server.routes[("POST", "/api/chat")] = (200, {"message": {"content": "late"}})
    server.delay = 1.5
    provider = OllamaProvider(server.url)
    with pytest.raises(ProviderTimeout):
        provider.chat([ChatMessage("user", "hi")], [], model="m", max_tokens=10, timeout=0.3)
    slow = OllamaProvider(server.url, timeout=0.3)
    assert slow.test_connection(None).kind == "timeout"


def test_ollama_missing_done_reason_is_normalised(server: FakeServer) -> None:
    replies = iter([
        (200, {"message": {"role": "assistant", "content": "plain answer"}}),
        (200, {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "measure", "arguments": {"id": "3"}}}]}}),
        (200, {"message": {"role": "assistant", "content": "cut"}, "done_reason": "length"}),
    ])
    server.routes[("POST", "/api/chat")] = lambda r: next(replies)
    provider = OllamaProvider(server.url)
    reasons = [provider.chat([ChatMessage("user", "hi")], [TOOL], model="m", max_tokens=8, timeout=5).stop_reason
               for _ in range(3)]
    assert reasons == ["end_turn", "tool_use", "max_tokens"]


# --------------------------------------------------------------------------- Anthropic


def test_anthropic_models_pagination_and_info(server: FakeServer) -> None:
    server.routes[("GET", "/v1/models?limit=100")] = (200, {"data": [{"id": "claude-a"}], "has_more": True, "last_id": "claude-a"})
    server.routes[("GET", "/v1/models?limit=100&after_id=claude-a")] = (200, {"data": [{"id": "claude-b"}], "has_more": False})
    server.routes[("GET", "/v1/models/claude-b")] = (200, {"id": "claude-b", "type": "model"})
    provider = AnthropicProvider(KEY, server.url + "/v1")
    models = provider.list_models()
    assert [m.id for m in models] == ["claude-a", "claude-b"] and all(m.supports_tools for m in models)
    assert provider.model_info("claude-b").id == "claude-b"
    headers = server.requests[-1]["headers"]
    assert headers["x-api-key"] == KEY and headers["anthropic-version"] == "2023-06-01"
    assert provider.test_connection("claude-b").kind == "ok"
    assert provider.test_connection("claude-x").kind == "model_missing"
    assert KEY not in repr(provider)


def test_anthropic_chat_round_with_tool_call_and_result(server: FakeServer) -> None:
    replies = iter([
        (200, {"model": "claude-a", "stop_reason": "tool_use", "usage": {"input_tokens": 300, "output_tokens": 40},
               "content": [{"type": "text", "text": "Measuring."},
                           {"type": "tool_use", "id": "toolu_1", "name": "measure", "input": {"id": "3"}}]}),
        (200, {"model": "claude-a", "stop_reason": "end_turn", "usage": {"input_tokens": 380, "output_tokens": 12},
               "content": [{"type": "text", "text": "4.21 m."}]}),
    ])
    server.routes[("POST", "/v1/messages")] = lambda r: next(replies)
    provider = AnthropicProvider(KEY, server.url)
    first = provider.chat(_conversation()[:2], [TOOL], model="claude-a", max_tokens=512, timeout=5)
    assert first.stop_reason == "tool_use" and first.message.content == "Measuring."
    assert first.message.tool_calls == [ToolCall("toolu_1", "measure", {"id": "3"})]
    assert (first.usage.input_tokens, first.usage.output_tokens) == (300, 40)
    body = server.requests[-1]["body"]
    assert body["system"] == "You edit plans." and body["max_tokens"] == 512
    assert body["tools"] == [{"name": "measure", "description": "Measure an element", "input_schema": TOOL.parameters}]
    user_blocks = body["messages"][0]["content"]
    assert user_blocks[0]["type"] == "image" and user_blocks[0]["source"]["data"] == base64.b64encode(PNG).decode("ascii")
    assert user_blocks[-1] == {"type": "text", "text": "How long is wall 3?"}

    conversation = _conversation() + [ChatMessage("user", "Thanks")]
    second = provider.chat(conversation, [TOOL], model="claude-a", max_tokens=512, timeout=5)
    assert second.message.content == "4.21 m." and second.stop_reason == "end_turn"
    wire = server.requests[-1]["body"]["messages"]
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert wire[1]["content"][1] == {"type": "tool_use", "id": "call_1", "name": "measure", "input": {"id": "3"}}
    assert wire[2]["content"][0] == {"type": "tool_result", "tool_use_id": "call_1", "content": '{"length_m": 4.21}'}
    assert wire[2]["content"][1] == {"type": "text", "text": "Thanks"}


def test_anthropic_error_mapping_never_leaks_key(server: FakeServer, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    provider = AnthropicProvider(KEY, server.url)
    server.routes[("POST", "/v1/messages")] = (401, {"error": {"message": f"invalid x-api-key {KEY}"}})
    with pytest.raises(AuthError) as auth:
        provider.chat([ChatMessage("user", "hi")], [], model="claude-a", max_tokens=8, timeout=5)
    assert KEY not in str(auth.value)
    server.routes[("POST", "/v1/messages")] = (429, {"error": {"message": "slow down"}}, {"retry-after": "7"})
    with pytest.raises(RateLimited) as limited:
        provider.chat([ChatMessage("user", "hi")], [], model="claude-a", max_tokens=8, timeout=5)
    assert limited.value.retry_after == 7 and "7" in str(limited.value)
    server.routes[("POST", "/v1/messages")] = (404, {"error": {"message": "model: claude-x"}})
    with pytest.raises(ModelMissing):
        provider.chat([ChatMessage("user", "hi")], [], model="claude-x", max_tokens=8, timeout=5)
    server.routes[("GET", "/v1/models?limit=100")] = (401, {"error": {"message": f"bad key {KEY}"}})
    result = provider.test_connection(None)
    assert result.kind == "auth" and KEY not in result.message
    assert AnthropicProvider(KEY, f"http://127.0.0.1:{_closed_port()}").test_connection(None).kind == "unreachable"
    assert KEY not in caplog.text


# --------------------------------------------------------------------------- OpenAI


def test_openai_chat_round_and_models(server: FakeServer) -> None:
    replies = iter([
        (200, {"model": "gpt-x", "usage": {"prompt_tokens": 50, "completion_tokens": 7},
               "choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
                   {"id": "call_9", "type": "function", "function": {"name": "measure", "arguments": "{\"id\": \"3\"}"}}]}}]}),
        (200, {"model": "gpt-x", "usage": {"prompt_tokens": 70, "completion_tokens": 5},
               "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "4.21 m"}}]}),
    ])
    server.routes[("POST", "/v1/chat/completions")] = lambda r: next(replies)
    server.routes[("GET", "/v1/models")] = (200, {"data": [{"id": "gpt-y"}, {"id": "gpt-x"}]})
    server.routes[("GET", "/v1/models/gpt-x")] = (200, {"id": "gpt-x"})
    provider = OpenAICompatProvider("openai", KEY, server.url + "/v1")
    first = provider.chat(_conversation()[:2], [TOOL], model="gpt-x", max_tokens=300, timeout=5)
    assert first.stop_reason == "tool_use" and first.message.tool_calls == [ToolCall("call_9", "measure", {"id": "3"})]
    assert (first.usage.input_tokens, first.usage.output_tokens) == (50, 7)
    request = server.requests[-1]
    assert request["headers"]["authorization"] == f"Bearer {KEY}" and "http-referer" not in request["headers"]
    body = request["body"]
    assert body["max_completion_tokens"] == 300 and "max_tokens" not in body
    assert body["tools"][0]["type"] == "function" and body["tools"][0]["function"]["name"] == "measure"
    parts = body["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "How long is wall 3?"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    second = provider.chat(_conversation(), [TOOL], model="gpt-x", max_tokens=300, timeout=5)
    assert second.message.content == "4.21 m" and second.stop_reason == "end_turn"
    wire = server.requests[-1]["body"]["messages"]
    assert wire[2]["tool_calls"][0] == {"id": "call_1", "type": "function",
                                        "function": {"name": "measure", "arguments": json.dumps({"id": "3"})}}
    assert wire[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"length_m": 4.21}'}
    assert [m.id for m in provider.list_models()] == ["gpt-x", "gpt-y"]
    assert provider.test_connection("gpt-x").kind == "ok"
    assert provider.test_connection("gpt-missing").kind == "model_missing"


def test_openai_tool_calls_without_id_get_unique_ids(server: FakeServer) -> None:
    server.routes[("POST", "/v1/chat/completions")] = (200, {"model": "gpt-x", "choices": [{
        "finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"type": "function", "function": {"name": "measure", "arguments": "{\"id\": \"3\"}"}},
            {"id": "", "type": "function", "function": {"name": "measure", "arguments": "{\"id\": \"4\"}"}}]}}]})
    provider = OpenAICompatProvider("openai", KEY, server.url + "/v1")
    response = provider.chat([ChatMessage("user", "measure walls 3 and 4")], [TOOL], model="gpt-x", max_tokens=64,
                             timeout=5)
    ids = [call.id for call in response.message.tool_calls]
    assert len(ids) == 2 and all(ids) and len(set(ids)) == 2
    assert response.stop_reason == "tool_use"
    # the generated ids round-trip as tool_call_id of the tool results
    follow_up = [ChatMessage("assistant", "", tool_calls=response.message.tool_calls),
                 *[ChatMessage("tool", "{}", tool_call_id=call_id) for call_id in ids]]
    provider.chat(follow_up, [TOOL], model="gpt-x", max_tokens=64, timeout=5)
    wire = server.requests[-1]["body"]["messages"]
    assert [m["id"] for m in wire[0]["tool_calls"]] == ids
    assert [m["tool_call_id"] for m in wire[1:]] == ids


def test_openai_error_kinds(server: FakeServer) -> None:
    provider = OpenAICompatProvider("openai", KEY, server.url)
    server.routes[("GET", "/models")] = (401, {"error": {"message": f"Incorrect API key provided: {KEY}"}})
    result = provider.test_connection(None)
    assert result.kind == "auth" and KEY not in result.message
    server.routes[("GET", "/models")] = (429, {"error": {"message": "quota"}})
    assert provider.test_connection(None).kind == "rate_limited"
    server.routes[("GET", "/models")] = (500, {"error": {"message": "boom"}})
    other = provider.test_connection(None)
    assert other.kind == "error" and "500" in other.message


# --------------------------------------------------------------------------- OpenRouter


OPENROUTER_MODELS = {"data": [
    {"id": "vendor/vision-tools", "pricing": {"prompt": "0.000003", "completion": "0.000015"},
     "supported_parameters": ["tools", "max_tokens"], "architecture": {"input_modalities": ["text", "image"]}},
    {"id": "vendor/text-only", "pricing": {"prompt": "0", "completion": "-1"},
     "supported_parameters": ["max_tokens"], "architecture": {"input_modalities": ["text"]}},
]}


def test_openrouter_models_prices_and_chat_flags(server: FakeServer) -> None:
    server.routes[("GET", "/api/v1/models")] = (200, OPENROUTER_MODELS)
    server.routes[("POST", "/api/v1/chat/completions")] = (200, {
        "model": "vendor/vision-tools", "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]})
    provider = OpenAICompatProvider("openrouter", KEY, server.url + "/api/v1/")
    models = {m.id: m for m in provider.list_models()}
    vision = models["vendor/vision-tools"]
    assert vision.input_usd_per_mtok == pytest.approx(3.0) and vision.output_usd_per_mtok == pytest.approx(15.0)
    assert vision.supports_tools is True and vision.supports_images is True
    text = models["vendor/text-only"]
    assert text.supports_tools is False and text.supports_images is False
    assert text.input_usd_per_mtok == 0.0 and text.output_usd_per_mtok is None
    response = provider.chat([ChatMessage("user", "hi")], [TOOL], model="vendor/vision-tools", max_tokens=64, timeout=5)
    assert response.message.content == "ok" and response.usage.input_tokens == 11
    request = server.requests[-1]
    assert request["body"]["max_tokens"] == 64 and "max_completion_tokens" not in request["body"]
    assert request["headers"]["http-referer"] == "SAMcloud" and request["headers"]["x-title"] == "SAMcloud"
    assert request["headers"]["user-agent"] == "SAMcloud"


def test_openrouter_connection_checks_key_model_and_tools(server: FakeServer) -> None:
    server.routes[("GET", "/api/v1/models")] = (200, OPENROUTER_MODELS)
    server.routes[("GET", "/api/v1/key")] = (200, {"data": {"label": "dummy"}})
    provider = OpenAICompatProvider("openrouter", KEY, server.url + "/api/v1")
    ok = provider.test_connection("vendor/vision-tools")
    assert ok.ok and ok.model_info is not None and ok.model_info.input_usd_per_mtok == pytest.approx(3.0)
    assert provider.test_connection("vendor/text-only").kind == "no_tools"
    assert provider.test_connection("vendor/unknown").kind == "model_missing"
    server.routes[("GET", "/api/v1/key")] = (401, {"error": {"message": f"No auth credentials found for {KEY}"}})
    bad = provider.test_connection("vendor/vision-tools")
    assert bad.kind == "auth" and KEY not in bad.message


# --------------------------------------------------------------------------- factory


def test_get_provider_builds_adapters_from_settings_and_credentials() -> None:
    settings = AssistantSettings(base_urls={"anthropic": "http://127.0.0.1:1", "openai": "http://127.0.0.1:2/v1",
                                            "openrouter": "http://127.0.0.1:3/api/v1"})
    creds = {"ollama_url": "", "anthropic_api_key": KEY, "openai_api_key": KEY, "openrouter_api_key": ""}
    ollama = get_provider("ollama", settings, creds)
    assert isinstance(ollama, OllamaProvider) and ollama.base_url == "http://localhost:11434"
    anthropic = get_provider("anthropic", settings, creds)
    assert isinstance(anthropic, AnthropicProvider) and anthropic.base_url == "http://127.0.0.1:1"
    openai = get_provider("openai", settings, creds)
    assert isinstance(openai, OpenAICompatProvider) and openai.base_url == "http://127.0.0.1:2/v1"
    with pytest.raises(AuthError) as missing:
        get_provider("openrouter", settings, creds)
    assert missing.value.kind == "auth"
    with pytest.raises(ValueError):
        get_provider("nonsense", settings, creds)
    with pytest.raises(Unreachable):
        OllamaProvider("not a url").list_models()
