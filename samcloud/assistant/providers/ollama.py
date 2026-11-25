"""Ollama adapter (local or LAN server, no API key, no costs).

Endpoints: ``GET /api/tags`` (models), ``POST /api/show`` (capabilities, context length), ``POST /api/chat``
(non-streaming chat with tools).

Ollama silently truncates a prompt longer than its ``num_ctx`` (default only 2048-4096 tokens) and the model then tends to
repeat itself until the server aborts ("token repeat limit reached").  Every chat request therefore sends ``options`` with an
explicit ``num_ctx`` (see :func:`request_num_ctx`), ``num_predict`` and conservative sampling values.  ``num_ctx`` stays the
same between requests of one model (the model's window, capped by the settings) so Ollama does not reload the model; it only
grows when a prompt would not fit.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from typing import Any, Iterable

from samcloud.app.i18n import t
from samcloud.assistant.providers.base import (
    BadResponse,
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ConnectionResult,
    ModelInfo,
    ToolCall,
    ToolSpec,
    Usage,
    estimate_tokens,
    http_json,
    parse_arguments,
    tool_names_by_id,
)

LOGGER = logging.getLogger(__name__)
_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}

DEFAULT_NUM_CTX_MAX = 32768  # settings ``ollama_num_ctx_max``: upper bound of the automatic context window
MIN_NUM_CTX = 2048
RECOMMENDED_MIN_CONTEXT = 16384  # models with a smaller window are not recommended (our prompts need ~8-17k tokens)
OUTPUT_MARGIN_TOKENS = 256  # template / special tokens on top of the estimated prompt and the answer
IMAGE_TOKENS = 1600  # an attached image counted as this many prompt tokens
MESSAGE_OVERHEAD_CHARS = 16  # role markers of the chat template per message
SAMPLING_OPTIONS: dict[str, float] = {"temperature": 0.2, "repeat_penalty": 1.1, "top_p": 0.9}
_INSTRUCT = re.compile(r"(instruct|chat|[-_:.]it(\b|[-_.]))", re.IGNORECASE)


def parse_num_ctx(value: Any) -> int | None:
    """The fixed context size of the ``ollama_num_ctx`` setting, or ``None`` for ``"auto"`` / invalid values."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = int(value)
    else:
        text = str(value).strip().lower().replace("_", "").replace(" ", "")
        if not text.isdigit():
            return None
        number = int(text)
    return number if number > 0 else None


def context_window(model_max: int | None, num_ctx: Any = "auto", num_ctx_max: int | None = None) -> int:
    """Effective context window (tokens) of a model: the model maximum capped by the settings.

    ``num_ctx`` is the ``ollama_num_ctx`` setting (``"auto"`` or a number), ``num_ctx_max`` the ``ollama_num_ctx_max``
    setting (default :data:`DEFAULT_NUM_CTX_MAX`).  The model maximum always wins over a larger setting.
    """
    limit = int(num_ctx_max or DEFAULT_NUM_CTX_MAX)
    cap = int(model_max) if model_max else limit
    manual = parse_num_ctx(num_ctx)
    base = manual if manual is not None else min(limit, cap)
    return int(min(cap, max(MIN_NUM_CTX, base)))


def request_num_ctx(model_max: int | None, num_ctx: Any, num_ctx_max: int | None, needed_tokens: int) -> int:
    """``options.num_ctx`` for one request: the :func:`context_window`, raised to ``needed_tokens`` but never above the
    model maximum (the agent keeps prompts inside the window, so the raise is only a safety net)."""
    window = context_window(model_max, num_ctx, num_ctx_max)
    cap = int(model_max) if model_max else max(window, int(num_ctx_max or DEFAULT_NUM_CTX_MAX))
    return int(min(max(window, int(needed_tokens)), cap))


def is_instruct(model_id: str) -> bool:
    """True for instruction / chat tuned model names (``...-instruct``, ``...-chat``, ``...-it``)."""
    return bool(_INSTRUCT.search(str(model_id or "")))


def is_suitable(info: ModelInfo) -> bool:
    """A local model with tool support and a context window of at least :data:`RECOMMENDED_MIN_CONTEXT` tokens."""
    return (info.supports_tools is True and not info.remote and info.context_length is not None
            and info.context_length >= RECOMMENDED_MIN_CONTEXT)


def model_warnings(info: ModelInfo | None) -> list[str]:
    """Warning codes for a model: ``no_tools``, ``tools_unknown``, ``small_context``, ``remote``."""
    if info is None:
        return []
    codes = []
    if info.supports_tools is False:
        codes.append("no_tools")
    elif info.supports_tools is None:
        codes.append("tools_unknown")
    if info.context_length is not None and info.context_length < RECOMMENDED_MIN_CONTEXT:
        codes.append("small_context")
    if info.remote:
        codes.append("remote")
    return codes


def rank_models(models: Iterable[ModelInfo]) -> list[ModelInfo]:
    """Models that can (or might) call tools, most suitable first: local models with tools and a large window, instruct
    models before others, larger before smaller.  Models known to lack tool support are left out."""
    candidates = [info for info in models if info.supports_tools is not False]

    def key(info: ModelInfo) -> tuple:
        return (0 if is_suitable(info) else 1, 1 if info.remote else 0, 0 if info.supports_tools else 1,
                0 if is_instruct(info.id) else 1, -(info.size_bytes or 0), -(info.context_length or 0), info.id)

    return sorted(candidates, key=key)


def recommended_model(models: Iterable[ModelInfo]) -> str | None:
    """Id of the recommended model (first suitable one of :func:`rank_models`), or ``None``."""
    for info in rank_models(models):
        if is_suitable(info):
            return info.id
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _capabilities(info: ModelInfo, capabilities: Any) -> None:
    if isinstance(capabilities, list):
        names = {str(item).lower() for item in capabilities}
        info.supports_tools = "tools" in names
        info.supports_images = "vision" in names


def _context_from_show(data: dict[str, Any]) -> int | None:
    """``model_info.<arch>.context_length``, else ``details.context_length``, else ``num_ctx`` of the parameters."""
    model_info = data.get("model_info")
    if isinstance(model_info, dict):
        for name, value in model_info.items():
            if str(name).endswith(".context_length") and _int_or_none(value):
                return _int_or_none(value)
    details = data.get("details")
    if isinstance(details, dict) and _int_or_none(details.get("context_length")):
        return _int_or_none(details.get("context_length"))
    parameters = data.get("parameters")
    if isinstance(parameters, str):
        match = re.search(r"^\s*num_ctx\s+(\d+)", parameters, re.MULTILINE)
        if match:
            return _int_or_none(match.group(1))
    return None


class OllamaProvider(BaseProvider):
    """Chat with models served by an Ollama server at ``base_url`` (e.g. ``http://localhost:11434``).

    ``num_ctx`` / ``num_ctx_max`` are the ``ollama_num_ctx`` / ``ollama_num_ctx_max`` settings (see :func:`context_window`).
    """

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", *, timeout: float = 30.0, num_ctx: Any = "auto",
                 num_ctx_max: int | None = None) -> None:
        self.base_url = str(base_url or "http://localhost:11434").strip().rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_ctx_max = _int_or_none(num_ctx_max) or DEFAULT_NUM_CTX_MAX
        self._show_cache: dict[str, ModelInfo] = {}  # /api/show answers per model (this adapter instance)
        self.last_options: dict[str, Any] = {}

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def context_window_for(self, info: ModelInfo | None) -> int:
        """Effective context window the chat requests use for ``info`` (see :func:`context_window`)."""
        return context_window(info.context_length if info is not None else None, self.num_ctx, self.num_ctx_max)

    # -------------------------------------------------------------------------------- models

    def list_models(self) -> list[ModelInfo]:
        """All models of the server with capabilities, context length and size.

        Newer servers report capabilities and context length in ``/api/tags``; for older ones ``/api/show`` is asked per
        model (failures leave the values unknown).
        """
        data = http_json("GET", self._url("/api/tags"), timeout=self.timeout, not_found_is_model=False)
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise BadResponse(t("assistant.settings.error.bad_json"))
        result = []
        for entry in models:
            if not isinstance(entry, dict) or not (entry.get("name") or entry.get("model")):
                continue
            info = ModelInfo(id=str(entry.get("name") or entry.get("model")), input_usd_per_mtok=0.0,
                             output_usd_per_mtok=0.0, size_bytes=_int_or_none(entry.get("size")),
                             remote=bool(entry.get("remote_host") or entry.get("remote_model")))
            details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
            info.context_length = _int_or_none(details.get("context_length"))
            info.parameter_size = str(details.get("parameter_size") or "") or None
            _capabilities(info, entry.get("capabilities"))
            if info.supports_tools is None or info.context_length is None:
                try:
                    shown: ModelInfo | None = self.model_info(info.id)
                except Exception as error:  # noqa: BLE001 - one broken model must not hide the others
                    LOGGER.info("Ollama /api/show for one model failed: %s", type(error).__name__)
                    shown = None
                if shown is not None:
                    if info.supports_tools is None:
                        info.supports_tools, info.supports_images = shown.supports_tools, shown.supports_images
                    info.context_length = info.context_length or shown.context_length
                    info.parameter_size = info.parameter_size or shown.parameter_size
            result.append(info)
        return result

    def model_info(self, model: str) -> ModelInfo:
        """Capabilities and context length of ``model`` from ``/api/show`` (cached per model on this adapter)."""
        cached = self._show_cache.get(model)
        if cached is not None:
            return ModelInfo(**vars(cached))
        data = http_json("POST", self._url("/api/show"), body={"model": model}, timeout=self.timeout)
        info = ModelInfo(id=model, input_usd_per_mtok=0.0, output_usd_per_mtok=0.0)
        if isinstance(data, dict):
            _capabilities(info, data.get("capabilities"))
            info.context_length = _context_from_show(data)
            details = data.get("details") if isinstance(data.get("details"), dict) else {}
            info.parameter_size = str(details.get("parameter_size") or "") or None
            info.remote = bool(data.get("remote_host") or data.get("remote_model"))
        self._show_cache[model] = info
        return ModelInfo(**vars(info))

    def _model_max(self, model: str) -> int | None:
        try:
            return self.model_info(model).context_length
        except Exception as error:  # noqa: BLE001 - the chat request itself reports server problems
            LOGGER.debug("Ollama context length of the model unknown (%s)", type(error).__name__)
            return None

    def options_for(self, model: str, wire_messages: list[dict[str, Any]], wire_tools: list[dict[str, Any]],
                    max_tokens: int) -> dict[str, Any]:
        """``options`` of one chat request (``num_ctx`` from the estimated prompt size, see :func:`request_num_ctx`)."""
        images = 0
        chars = 0
        for item in wire_messages:
            images += len(item.get("images") or [])
            chars += len(str(item.get("content") or "")) + MESSAGE_OVERHEAD_CHARS
            if item.get("tool_calls"):
                chars += len(json.dumps(item["tool_calls"], ensure_ascii=False, default=str))
        if wire_tools:
            chars += len(json.dumps(wire_tools, ensure_ascii=False))
        needed = estimate_tokens(chars) + images * IMAGE_TOKENS + int(max_tokens) + OUTPUT_MARGIN_TOKENS
        num_ctx = request_num_ctx(self._model_max(model), self.num_ctx, self.num_ctx_max, needed)
        return {"num_ctx": num_ctx, "num_predict": int(max_tokens), **SAMPLING_OPTIONS}

    # -------------------------------------------------------------------------------- chat

    @staticmethod
    def _wire_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        names = tool_names_by_id(messages)
        wire: list[dict[str, Any]] = []
        for message in messages:
            item: dict[str, Any] = {"role": message.role, "content": message.content or ""}
            if message.images:
                item["images"] = [base64.b64encode(image).decode("ascii") for image in message.images]
            if message.role == "assistant" and message.tool_calls:
                item["tool_calls"] = [{"function": {"name": call.name, "arguments": call.arguments}}
                                      for call in message.tool_calls]
            if message.role == "tool":
                item["tool_name"] = names.get(message.tool_call_id or "", "")
            wire.append(item)
        return wire

    def chat(self, messages: list[ChatMessage], tools: list[ToolSpec], *, model: str, max_tokens: int,
             timeout: float) -> ChatResponse:
        wire = self._wire_messages(messages)
        wire_tools = [{"type": "function", "function": {"name": tool.name, "description": tool.description,
                                                        "parameters": tool.parameters}} for tool in tools]
        options = self.options_for(model, wire, wire_tools, max_tokens)
        self.last_options = dict(options)
        body: dict[str, Any] = {"model": model, "messages": wire, "stream": False, "options": options}
        if wire_tools:
            body["tools"] = wire_tools
        data = http_json("POST", self._url("/api/chat"), body=body, timeout=timeout)
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise BadResponse(t("assistant.settings.error.bad_json"))
        calls = []
        for index, raw in enumerate(message.get("tool_calls") or []):
            function = raw.get("function") if isinstance(raw, dict) else None
            if not isinstance(function, dict) or not function.get("name"):
                continue
            call_id = str(raw.get("id") or f"call_{index}_{uuid.uuid4().hex[:8]}")
            calls.append(ToolCall(id=call_id, name=str(function["name"]),
                                  arguments=parse_arguments(function.get("arguments"))))
        done_reason = str(data.get("done_reason") or "").strip()
        if calls:
            reason = "tool_use"
        elif not done_reason:
            reason = "end_turn"
        else:
            reason = _STOP_REASONS.get(done_reason, done_reason)
        usage = Usage(int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0))
        return ChatResponse(ChatMessage("assistant", str(message.get("content") or ""), calls), usage, reason,
                            str(data.get("model") or model))

    # -------------------------------------------------------------------------------- test

    def _probe(self, model: str | None) -> ConnectionResult:
        # list the models first: a clear "unreachable" before any model question
        models = self.list_models()
        if not model:
            return ConnectionResult(True, "ok", t("assistant.settings.test.ok_models", count=len(models)))
        return super()._probe(model)


__all__ = ["DEFAULT_NUM_CTX_MAX", "OllamaProvider", "RECOMMENDED_MIN_CONTEXT", "context_window", "is_instruct",
           "is_suitable", "model_warnings", "parse_num_ctx", "rank_models", "recommended_model", "request_num_ctx"]
