"""Provider adapters of the AI assistant (Ollama, Anthropic, OpenAI, OpenRouter) over plain HTTP.

Use :func:`get_provider` to build the adapter for the configured provider; all adapters implement
:class:`~samcloud.assistant.providers.base.Provider`.
"""

from __future__ import annotations

from samcloud.assistant.providers.base import (
    AuthError,
    BadResponse,
    ChatMessage,
    ChatResponse,
    ConnectionResult,
    ContextError,
    ModelInfo,
    ModelMissing,
    NoToolSupport,
    Provider,
    ProviderError,
    ProviderTimeout,
    RateLimited,
    ToolCall,
    ToolSpec,
    Unreachable,
    Usage,
    get_provider,
    is_context_error,
)

__all__ = [
    "AuthError", "BadResponse", "ChatMessage", "ChatResponse", "ConnectionResult", "ContextError", "ModelInfo",
    "ModelMissing", "NoToolSupport", "Provider", "ProviderError", "ProviderTimeout", "RateLimited", "ToolCall", "ToolSpec",
    "Unreachable", "Usage", "get_provider", "is_context_error",
]
