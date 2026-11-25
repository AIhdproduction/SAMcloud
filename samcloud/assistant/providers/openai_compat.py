"""OpenAI and OpenRouter adapter (OpenAI-compatible Chat Completions API).

One class for both services, differing by base URL and small flags:

* OpenAI: ``max_completion_tokens``; ``GET {base}/models`` (ids only), ``GET {base}/models/{id}``.
* OpenRouter: ``max_tokens``; neutral ``HTTP-Referer``/``X-Title`` headers; ``GET {base}/models`` with live prices
  (USD per token, converted to USD per million tokens), tool and image support; ``GET {base}/key`` checks the key.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.parse
import uuid
from typing import Any

from samcloud.app.i18n import t
from samcloud.assistant.providers.base import (
    BadResponse,
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    ModelMissing,
    ToolCall,
    ToolSpec,
    Usage,
    http_json,
    parse_arguments,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_BASE_URLS = {"openai": "https://api.openai.com/v1", "openrouter": "https://openrouter.ai/api/v1"}
_STOP_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "function_call": "tool_use", "length": "max_tokens"}


def _per_mtok(value: Any) -> float | None:
    """USD per token (string or number) → USD per million tokens; negative/invalid → None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number * 1_000_000, 6) if number >= 0 else None


class OpenAICompatProvider(BaseProvider):
    """Chat Completions client for ``name`` ``"openai"`` or ``"openrouter"``."""

    def __init__(self, name: str, api_key: str, base_url: str | None = None, *, timeout: float = 30.0) -> None:
        if name not in DEFAULT_BASE_URLS:
            raise ValueError(f"unknown OpenAI-compatible provider: {name}")
        self.name = name
        self._api_key = str(api_key or "").strip()
        self.base_url = str(base_url or DEFAULT_BASE_URLS[name]).strip().rstrip("/")
        self.timeout = timeout
        self._models: dict[str, ModelInfo] | None = None

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatProvider(name={self.name!r}, base_url={self.base_url!r})"

    @property
    def is_openrouter(self) -> bool:
        return self.name == "openrouter"

    def _secrets(self) -> tuple[str, ...]:
        return (self._api_key,)

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if self.is_openrouter:
            headers.update({"HTTP-Referer": "SAMcloud", "X-Title": "SAMcloud"})
        return headers

    def _request(self, method: str, path: str, *, body: Any = None, timeout: float | None = None,
                 not_found_is_model: bool = True) -> Any:
        return http_json(method, f"{self.base_url}{path}", headers=self._headers(), body=body,
                         timeout=self.timeout if timeout is None else timeout, secrets=self._secrets(),
                         not_found_is_model=not_found_is_model)

    # -------------------------------------------------------------------------------- models

    def _parse_model(self, entry: dict[str, Any]) -> ModelInfo:
        info = ModelInfo(id=str(entry["id"]))
        if not self.is_openrouter:
            return info
        pricing = entry.get("pricing") if isinstance(entry.get("pricing"), dict) else {}
        info.input_usd_per_mtok = _per_mtok(pricing.get("prompt"))
        info.output_usd_per_mtok = _per_mtok(pricing.get("completion"))
        try:
            context = int(entry.get("context_length") or 0)
        except (TypeError, ValueError):
            context = 0
        info.context_length = context if context > 0 else None
        parameters = entry.get("supported_parameters")
        if isinstance(parameters, list):
            info.supports_tools = "tools" in parameters
        architecture = entry.get("architecture") if isinstance(entry.get("architecture"), dict) else {}
        modalities = architecture.get("input_modalities")
        if isinstance(modalities, list):
            info.supports_images = "image" in modalities
        return info

    def list_models(self) -> list[ModelInfo]:
        data = self._request("GET", "/models", not_found_is_model=False)
        entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            raise BadResponse(t("assistant.settings.error.bad_json"))
        models = [self._parse_model(entry) for entry in entries if isinstance(entry, dict) and entry.get("id")]
        models.sort(key=lambda info: info.id)
        self._models = {info.id: info for info in models}
        return models

    def model_info(self, model: str) -> ModelInfo:
        if self.is_openrouter:
            models = self._models if self._models is not None else {m.id: m for m in self.list_models()}
            info = models.get(model)
            if info is None:
                raise ModelMissing(t("assistant.settings.error.model_missing", detail=model))
            return info
        data = self._request("GET", "/models/" + urllib.parse.quote(model, safe=""))
        return ModelInfo(id=str(data.get("id") or model) if isinstance(data, dict) else model)

    def _check_auth(self) -> None:
        if self.is_openrouter:  # the OpenRouter model list is public; the key endpoint needs a valid key
            self._request("GET", "/key", not_found_is_model=False)

    # -------------------------------------------------------------------------------- chat

    @staticmethod
    def _wire_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                wire.append({"role": "tool", "tool_call_id": message.tool_call_id or "", "content": message.content or ""})
                continue
            if message.role == "assistant":
                item: dict[str, Any] = {"role": "assistant", "content": message.content or None}
                if message.tool_calls:
                    item["tool_calls"] = [{"id": call.id, "type": "function",
                                           "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                                          for call in message.tool_calls]
                wire.append(item)
                continue
            if message.images and message.role == "user":
                parts: list[dict[str, Any]] = []
                if message.content:
                    parts.append({"type": "text", "text": message.content})
                for image in message.images:
                    url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                wire.append({"role": "user", "content": parts})
                continue
            wire.append({"role": message.role, "content": message.content or ""})
        return wire

    def chat(self, messages: list[ChatMessage], tools: list[ToolSpec], *, model: str, max_tokens: int,
             timeout: float) -> ChatResponse:
        body: dict[str, Any] = {"model": model, "messages": self._wire_messages(messages)}
        body["max_tokens" if self.is_openrouter else "max_completion_tokens"] = int(max_tokens)
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": tool.name, "description": tool.description,
                                                               "parameters": tool.parameters}} for tool in tools]
        data = self._request("POST", "/chat/completions", body=body, timeout=timeout)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise BadResponse(t("assistant.settings.error.bad_json"))
        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        calls: list[ToolCall] = []
        for index, raw in enumerate(message.get("tool_calls") or []):
            function = raw.get("function") if isinstance(raw, dict) else None
            if not isinstance(function, dict) or not function.get("name"):
                continue
            call_id = str(raw.get("id") or "").strip() or f"call_{index}_{uuid.uuid4().hex[:8]}"
            calls.append(ToolCall(id=call_id, name=str(function["name"]),
                                  arguments=parse_arguments(function.get("arguments"))))
        finish = str(choice.get("finish_reason") or ("tool_calls" if calls else "stop"))
        reason = "tool_use" if calls else _STOP_REASONS.get(finish, finish)
        usage_data = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        usage = Usage(int(usage_data.get("prompt_tokens") or 0), int(usage_data.get("completion_tokens") or 0))
        return ChatResponse(ChatMessage("assistant", str(message.get("content") or ""), calls), usage, reason,
                            str(data.get("model") or model))
