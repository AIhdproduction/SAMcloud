"""Provider-neutral chat types, errors and the HTTP helper of the AI assistant adapters.

Every adapter speaks plain HTTP via :mod:`urllib.request` and converts between these neutral types and its wire format:

* :class:`ChatMessage` with ``role`` ``"system" | "user" | "assistant" | "tool"``; assistant messages may carry
  :class:`ToolCall` s, tool results reference them through ``tool_call_id``.
* :class:`ChatResponse.stop_reason` is normalised to ``"end_turn"``, ``"tool_use"`` or ``"max_tokens"`` (other values are
  passed through unchanged).
* Images (PNG bytes) are sent whenever a message carries them; the caller decides (``text_only`` / model capability).

All errors are :class:`ProviderError` subclasses with a translated, redacted message and a ``kind`` matching
:attr:`ConnectionResult.kind`. No exception message ever contains an API key.  Errors caused by a full context window or a
repeating model (Ollama "token repeat limit", "maximum context length", "prompt is too long", ...) are :class:`ContextError`
(``kind == "context"``); :func:`is_context_error` also recognises them in errors of other classes.
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Protocol

from samcloud.app.i18n import t
from samcloud.assistant.credentials import redact

if TYPE_CHECKING:
    from samcloud.assistant.settings import AssistantSettings

LOGGER = logging.getLogger(__name__)
USER_AGENT = "SAMcloud"
_DETAIL_LIMIT = 300


@dataclass
class ToolSpec:
    """A tool the model may call; ``parameters`` is a JSON schema of ``type: object``."""

    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatMessage:
    """One conversation message in neutral form."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    images: list[bytes] = field(default_factory=list)  # PNG bytes


@dataclass
class Usage:
    """Token usage of one request."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatResponse:
    """The model's answer to one :meth:`Provider.chat` call."""

    message: ChatMessage
    usage: Usage
    stop_reason: str
    model: str


@dataclass
class ModelInfo:
    """What is known about a model; ``None`` means unknown.

    ``context_length`` is the model's maximum context window in tokens, ``size_bytes`` the download size (Ollama),
    ``parameter_size`` a display text such as ``"14.8B"`` and ``remote`` marks a model that runs on another service
    although it is listed by the local server (Ollama cloud models).
    """

    id: str
    supports_tools: bool | None = None
    supports_images: bool | None = None
    input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None
    context_length: int | None = None
    size_bytes: int | None = None
    parameter_size: str | None = None
    remote: bool = False


@dataclass
class ConnectionResult:
    """Outcome of :meth:`Provider.test_connection`.

    ``kind`` is one of ``"ok", "auth", "unreachable", "model_missing", "no_tools", "rate_limited", "timeout", "error"``.
    """

    ok: bool
    kind: str
    message: str
    model_info: ModelInfo | None = None


class ProviderError(Exception):
    """Base error of the provider adapters; the message is translated and redacted."""

    kind = "error"

    def __init__(self, message: str) -> None:
        super().__init__(redact(message))

    @property
    def message(self) -> str:
        return str(self)


class AuthError(ProviderError):
    """Wrong, missing or insufficient API key (HTTP 401/403)."""

    kind = "auth"


class Unreachable(ProviderError):
    """The server could not be reached (connection refused, DNS, network)."""

    kind = "unreachable"


class ModelMissing(ProviderError):
    """The model does not exist on the server (HTTP 404)."""

    kind = "model_missing"


class NoToolSupport(ProviderError):
    """The model cannot call tools."""

    kind = "no_tools"


class RateLimited(ProviderError):
    """Too many requests (HTTP 429); ``retry_after`` in seconds when the server sent it."""

    kind = "rate_limited"

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderTimeout(ProviderError):
    """The server did not answer in time."""

    kind = "timeout"


class BadResponse(ProviderError):
    """Any other HTTP error or an unreadable answer."""

    kind = "error"


class ContextError(BadResponse):
    """The prompt did not fit the model's context window or the model got stuck repeating itself."""

    kind = "context"


class Provider(Protocol):
    """Interface of every adapter (used by the agent loop and the settings card)."""

    name: str

    def list_models(self) -> list[ModelInfo]: ...

    def model_info(self, model: str) -> ModelInfo: ...

    def chat(self, messages: list[ChatMessage], tools: list[ToolSpec], *, model: str, max_tokens: int,
             timeout: float) -> ChatResponse: ...

    def test_connection(self, model: str | None) -> ConnectionResult: ...


# ------------------------------------------------------------------------------------------------
# HTTP
# ------------------------------------------------------------------------------------------------

_NO_TOOLS_MARKERS = ("does not support tools", "support tool use", "tool use is not supported",
                     "tools is not supported", "does not support function")
# error texts of a full context window or a repeating model (Ollama, llama.cpp, OpenAI, Anthropic, OpenRouter)
CONTEXT_ERROR_MARKERS = ("token repeat limit", "repeat limit reached", "prediction aborted", "context length",
                         "context_length", "maximum context", "context window", "prompt too long", "prompt is too long",
                         "too many tokens", "exceeds the context", "exceed context", "input is too long",
                         "reduce the length of the messages", "num_ctx")
CHARS_PER_TOKEN = 3.0  # conservative: JSON with many digits needs more tokens per character than prose


def estimate_tokens(chars: int | float) -> int:
    """Upper estimate of the tokens of ``chars`` characters of prompt text."""
    return int(max(0.0, float(chars)) / CHARS_PER_TOKEN) + 1


def is_context_error(error: BaseException) -> bool:
    """True when ``error`` says the context window is full or the model aborted because it kept repeating."""
    if isinstance(error, ContextError):
        return True
    if not isinstance(error, ProviderError) or isinstance(error, (AuthError, Unreachable, ModelMissing, RateLimited,
                                                                  NoToolSupport, ProviderTimeout)):
        return False
    lowered = str(error).lower()
    return any(marker in lowered for marker in CONTEXT_ERROR_MARKERS)


def _retry_after(headers: Any) -> float | None:
    try:
        value = headers.get("retry-after") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _detail(body: str, secrets: Iterable[str]) -> str:
    """Short, redacted error text from a JSON or plain error body."""
    text = body.strip()
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            text = str(error.get("message") or error)
        elif error:
            text = str(error)
        elif data.get("message"):
            text = str(data["message"])
    return redact(text[:_DETAIL_LIMIT], secrets)


def _is_timeout(reason: Any) -> bool:
    return isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason).lower()


def map_http_error(code: int, body: str, headers: Any, *, secrets: Iterable[str] = (),
                   not_found_is_model: bool = True) -> ProviderError:
    """Translate an HTTP error status into the matching :class:`ProviderError`."""
    detail = _detail(body, secrets)
    lowered = detail.lower()
    if any(marker in lowered for marker in _NO_TOOLS_MARKERS):
        return NoToolSupport(t("assistant.settings.error.no_tools", detail=detail))
    if code in (401, 403):
        return AuthError(t("assistant.settings.error.auth", detail=detail))
    if code == 404 and not_found_is_model:
        return ModelMissing(t("assistant.settings.error.model_missing", detail=detail))
    if code in (400, 413, 422, 500, 503) and any(marker in lowered for marker in CONTEXT_ERROR_MARKERS):
        return ContextError(t("assistant.settings.error.context", code=code, detail=detail))
    if code == 429:
        retry = _retry_after(headers)
        if retry is not None:
            message = t("assistant.settings.error.rate_limited_retry", seconds=int(round(retry)), detail=detail)
        else:
            message = t("assistant.settings.error.rate_limited", detail=detail)
        return RateLimited(message, retry)
    return BadResponse(t("assistant.settings.error.http", code=code, detail=detail))


def http_json(method: str, url: str, *, headers: dict[str, str] | None = None, body: Any = None,
              timeout: float = 30.0, secrets: Iterable[str] = (), not_found_is_model: bool = True) -> Any:
    """Send one JSON request and return the decoded JSON answer; every failure becomes a :class:`ProviderError`."""
    secrets = tuple(item for item in secrets if item)
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    request_headers.update(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    try:
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-configured endpoints
            raw = response.read()
    except urllib.error.HTTPError as error:
        try:
            text = error.read().decode("utf-8", "replace")
        except Exception:  # body not readable
            text = ""
        mapped = map_http_error(error.code, text, error.headers, secrets=secrets, not_found_is_model=not_found_is_model)
        LOGGER.info("AI provider request %s %s failed: %s", method, redact(url, secrets), mapped)
        raise mapped from None
    except urllib.error.URLError as error:
        reason = redact(getattr(error, "reason", error), secrets)
        if _is_timeout(getattr(error, "reason", None)):
            raise ProviderTimeout(t("assistant.settings.error.timeout", seconds=int(round(timeout)))) from None
        raise Unreachable(t("assistant.settings.error.unreachable", url=_origin(url), detail=reason)) from None
    except (socket.timeout, TimeoutError):
        raise ProviderTimeout(t("assistant.settings.error.timeout", seconds=int(round(timeout)))) from None
    except (ConnectionError, OSError) as error:
        raise Unreachable(t("assistant.settings.error.unreachable", url=_origin(url),
                            detail=redact(error, secrets))) from None
    except http.client.HTTPException as error:
        raise Unreachable(t("assistant.settings.error.unreachable", url=_origin(url),
                            detail=redact(type(error).__name__, secrets))) from None
    except ValueError as error:  # malformed URL
        raise Unreachable(t("assistant.settings.error.unreachable", url=_origin(url),
                            detail=redact(error, secrets))) from None
    try:
        return json.loads(raw.decode("utf-8")) if raw else {}
    except ValueError:
        raise BadResponse(t("assistant.settings.error.bad_json")) from None


def _origin(url: str) -> str:
    """Scheme + host of ``url`` (no path or query) for messages."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
    except ValueError:
        return "?"
    return f"{parts.scheme}://{parts.netloc}" if parts.netloc else url[:80]


# ------------------------------------------------------------------------------------------------
# shared adapter behaviour
# ------------------------------------------------------------------------------------------------

def tool_names_by_id(messages: Iterable[ChatMessage]) -> dict[str, str]:
    """Map tool-call ids of the assistant messages to the tool names (some wire formats need the name)."""
    names: dict[str, str] = {}
    for message in messages:
        for call in message.tool_calls:
            names[call.id] = call.name
    return names


def parse_arguments(value: Any) -> dict:
    """Tool-call arguments as a dict (wire formats send an object or a JSON string)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {"_raw": value}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {}


class BaseProvider:
    """Shared ``test_connection`` logic; subclasses implement the wire format."""

    name = "base"

    def _secrets(self) -> tuple[str, ...]:
        return ()

    def list_models(self) -> list[ModelInfo]:  # pragma: no cover - overridden
        raise NotImplementedError

    def model_info(self, model: str) -> ModelInfo:  # pragma: no cover - overridden
        raise NotImplementedError

    def _check_auth(self) -> None:
        """Extra authentication check before the model test (default: none)."""

    def _probe(self, model: str | None) -> ConnectionResult:
        self._check_auth()
        if not model:
            models = self.list_models()
            return ConnectionResult(True, "ok", t("assistant.settings.test.ok_models", count=len(models)))
        info = self.model_info(model)
        if info.supports_tools is False:
            return ConnectionResult(False, "no_tools", t("assistant.settings.error.no_tools", detail=model), info)
        return ConnectionResult(True, "ok", t("assistant.settings.test.ok_model", model=model), info)

    def test_connection(self, model: str | None) -> ConnectionResult:
        """Tiny request that checks address, key, model and tool support; never raises."""
        try:
            return self._probe((model or "").strip() or None)
        except ProviderError as error:
            return ConnectionResult(False, error.kind, str(error))
        except Exception as error:  # never raise from a connection test
            LOGGER.info("AI provider connection test failed: %s", redact(type(error).__name__, self._secrets()))
            return ConnectionResult(False, "error", t("assistant.settings.error.unexpected",
                                                      detail=redact(error, self._secrets())[:_DETAIL_LIMIT]))


def get_provider(name: str, settings: "AssistantSettings", creds: dict[str, Any]) -> Provider:
    """Adapter for ``name`` built from the settings (base URLs) and credentials (keys, Ollama address).

    Raises :class:`AuthError` when a cloud provider has no key and :class:`ValueError` for an unknown name.
    """
    from samcloud.assistant.credentials import DEFAULT_OLLAMA_URL, KEY_FIELDS

    if name == "ollama":
        from samcloud.assistant.providers.ollama import OllamaProvider

        url = str(creds.get("ollama_url") or settings.ollama_url_default or DEFAULT_OLLAMA_URL)
        return OllamaProvider(url, num_ctx=getattr(settings, "ollama_num_ctx", "auto"),
                              num_ctx_max=getattr(settings, "ollama_num_ctx_max", None))
    if name not in KEY_FIELDS:
        raise ValueError(f"unknown AI provider: {name}")
    key = str(creds.get(KEY_FIELDS[name]) or "").strip()
    if not key:
        raise AuthError(t("assistant.settings.error.no_key"))
    base_url = str(settings.base_urls.get(name) or "")
    if name == "anthropic":
        from samcloud.assistant.providers.anthropic import AnthropicProvider

        return AnthropicProvider(key, base_url or "https://api.anthropic.com")
    from samcloud.assistant.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(name, key, base_url or None)
