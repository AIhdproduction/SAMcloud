"""Anthropic (Claude) adapter over the Messages API.

Endpoints: ``POST {base}/v1/messages``, ``GET {base}/v1/models`` (paginated), ``GET {base}/v1/models/{id}``.
"""

from __future__ import annotations

import base64
import logging
import urllib.parse
from typing import Any

from samcloud.app.i18n import t
from samcloud.assistant.providers.base import (
    BadResponse,
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    ToolCall,
    ToolSpec,
    Usage,
    http_json,
    parse_arguments,
)

LOGGER = logging.getLogger(__name__)
API_VERSION = "2023-06-01"
MAX_MODEL_PAGES = 5


class AnthropicProvider(BaseProvider):
    """Chat with Claude models; ``base_url`` without ``/v1`` (default ``https://api.anthropic.com``)."""

    name = "anthropic"

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com", *, timeout: float = 30.0) -> None:
        self._api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "https://api.anthropic.com").strip().rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.timeout = timeout

    def __repr__(self) -> str:  # never show the key
        return f"AnthropicProvider(base_url={self.base_url!r})"

    def _secrets(self) -> tuple[str, ...]:
        return (self._api_key,)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}

    def _request(self, method: str, path: str, *, body: Any = None, timeout: float | None = None,
                 not_found_is_model: bool = True) -> Any:
        return http_json(method, f"{self.base_url}{path}", headers=self._headers(), body=body,
                         timeout=self.timeout if timeout is None else timeout, secrets=self._secrets(),
                         not_found_is_model=not_found_is_model)

    # -------------------------------------------------------------------------------- models

    def list_models(self) -> list[ModelInfo]:
        result: list[ModelInfo] = []
        after: str | None = None
        for _page in range(MAX_MODEL_PAGES):
            query = {"limit": "100"}
            if after:
                query["after_id"] = after
            data = self._request("GET", "/v1/models?" + urllib.parse.urlencode(query), not_found_is_model=False)
            entries = data.get("data") if isinstance(data, dict) else None
            if not isinstance(entries, list):
                raise BadResponse(t("assistant.settings.error.bad_json"))
            for entry in entries:
                if isinstance(entry, dict) and entry.get("id"):
                    result.append(ModelInfo(id=str(entry["id"]), supports_tools=True))
            after = str(data.get("last_id") or (entries[-1].get("id") if entries and isinstance(entries[-1], dict) else ""))
            if not data.get("has_more") or not after:
                break
        return result

    def model_info(self, model: str) -> ModelInfo:
        data = self._request("GET", "/v1/models/" + urllib.parse.quote(model, safe=""))
        model_id = str(data.get("id") or model) if isinstance(data, dict) else model
        return ModelInfo(id=model_id, supports_tools=True)

    # -------------------------------------------------------------------------------- chat

    @staticmethod
    def _image_blocks(images: list[bytes]) -> list[dict[str, Any]]:
        return [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(image).decode("ascii")}} for image in images]

    def _wire(self, messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
        """``(system, messages)`` in Messages API form; consecutive same-role turns are merged."""
        system_parts: list[str] = []
        wire: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue
            blocks: list[dict[str, Any]] = []
            if message.role == "tool":
                role = "user"
                blocks.append({"type": "tool_result", "tool_use_id": message.tool_call_id or "",
                               "content": message.content or ""})
                blocks.extend(self._image_blocks(message.images))
            elif message.role == "assistant":
                role = "assistant"
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
            else:
                role = "user"
                blocks.extend(self._image_blocks(message.images))
                if message.content or not blocks:
                    blocks.append({"type": "text", "text": message.content or ""})
            if not blocks:
                continue
            if wire and wire[-1]["role"] == role:
                # tool_result blocks must come first in a user turn
                if role == "user" and any(block["type"] == "tool_result" for block in blocks):
                    wire[-1]["content"] = [*[b for b in wire[-1]["content"] if b["type"] == "tool_result"], *blocks,
                                           *[b for b in wire[-1]["content"] if b["type"] != "tool_result"]]
                else:
                    wire[-1]["content"].extend(blocks)
            else:
                wire.append({"role": role, "content": blocks})
        return "\n\n".join(system_parts), wire

    def chat(self, messages: list[ChatMessage], tools: list[ToolSpec], *, model: str, max_tokens: int,
             timeout: float) -> ChatResponse:
        system, wire = self._wire(messages)
        body: dict[str, Any] = {"model": model, "max_tokens": int(max_tokens), "messages": wire}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [{"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
                             for tool in tools]
        data = self._request("POST", "/v1/messages", body=body, timeout=timeout)
        content = data.get("content") if isinstance(data, dict) else None
        if not isinstance(content, list):
            raise BadResponse(t("assistant.settings.error.bad_json"))
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(id=str(block.get("id") or ""), name=str(block.get("name") or ""),
                                      arguments=parse_arguments(block.get("input"))))
        usage_data = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        usage = Usage(int(usage_data.get("input_tokens") or 0), int(usage_data.get("output_tokens") or 0))
        reason = str(data.get("stop_reason") or ("tool_use" if calls else "end_turn"))
        return ChatResponse(ChatMessage("assistant", "".join(texts), calls), usage, reason,
                            str(data.get("model") or model))
