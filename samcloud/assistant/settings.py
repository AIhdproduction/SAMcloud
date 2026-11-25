"""Settings of the optional AI assistant.

Defaults come from ``config/ai_assistant.yaml`` (committed, never written by the app). The user's choices are stored in
``<user config dir>/ai_settings.yaml`` (see :func:`samcloud.core.secretfile.config_dir`), outside the repository.
API keys are not part of the settings; they live in :mod:`samcloud.assistant.credentials`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from samcloud.core import secretfile
from samcloud.core.paths import CONFIG_DIRECTORY

LOGGER = logging.getLogger(__name__)

PROVIDERS: tuple[str, ...] = ("ollama", "anthropic", "openai", "openrouter")  # UI order; Ollama first and default
CLOUD_PROVIDERS: tuple[str, ...] = tuple(name for name in PROVIDERS if name != "ollama")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
DEFAULTS_PATH = CONFIG_DIRECTORY / "ai_assistant.yaml"
USER_FILE_NAME = "ai_settings.yaml"
# Fields the settings card changes; save_settings writes these (other keys already in the user file are kept).
USER_FIELDS: tuple[str, ...] = ("enabled", "provider", "model", "text_only", "cost_limit_request_usd",
                                "cost_limit_project_day_usd", "privacy_ack_cloud", "ollama_num_ctx", "full_control")
OLLAMA_NUM_CTX_AUTO = "auto"


@dataclass
class AssistantSettings:
    """Effective assistant settings (defaults merged with the user file)."""

    enabled: bool = False
    provider: str = "ollama"
    model: str = ""
    text_only: bool = True
    max_tool_calls: int = 20
    request_timeout_s: float = 120.0
    max_output_tokens: int = 4096
    cost_limit_request_usd: float = 0.50
    cost_limit_project_day_usd: float = 5.00
    ollama_url_default: str = DEFAULT_OLLAMA_URL
    base_urls: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BASE_URLS))
    prices_usd_per_mtok: dict[str, dict[str, float]] = field(default_factory=dict)
    privacy_ack_cloud: bool = False
    # context window: Ollama "auto" (model maximum capped by ollama_num_ctx_max) or a fixed number of tokens
    ollama_num_ctx: str = OLLAMA_NUM_CTX_AUTO
    ollama_num_ctx_max: int = 32768
    context_history_turns: int = 3  # previous question/answer pairs sent with a request (fewer when the window is small)
    # full control (tool tier "full"): None = default per provider (on for cloud providers, off for Ollama), else the
    # user's explicit choice, which a provider change never overwrites (see effective_full_control)
    full_control: bool | None = None

    @property
    def is_cloud(self) -> bool:
        """True for a provider that sends data to an external service."""
        return self.provider in CLOUD_PROVIDERS


def user_settings_path() -> Path:
    """Location of the user's settings file (not created)."""
    return secretfile.config_dir() / USER_FILE_NAME


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_number(value: Any, default: float, kind: type) -> Any:
    try:
        number = kind(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _prices(value: Any) -> dict[str, dict[str, float]]:
    """Valid entries of a ``prefix -> {input, output}`` table; broken entries are skipped."""
    table: dict[str, dict[str, float]] = {}
    if not isinstance(value, dict):
        return table
    for prefix, entry in value.items():
        if not isinstance(entry, dict):
            continue
        try:
            table[str(prefix)] = {"input": float(entry["input"]), "output": float(entry["output"])}
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("ignoring invalid price entry %s in the AI assistant settings", prefix)
    return table


def normalize_num_ctx(value: Any) -> str:
    """``"auto"`` or a positive whole number of tokens as text (anything else becomes ``"auto"``)."""
    if isinstance(value, bool) or value is None:
        return OLLAMA_NUM_CTX_AUTO
    text = str(int(value)) if isinstance(value, (int, float)) else str(value).strip().replace("_", "").replace(" ", "")
    if text.isdigit() and int(text) > 0:
        return str(int(text))
    return OLLAMA_NUM_CTX_AUTO


def normalize_full_control(value: Any) -> bool | None:
    """``True`` / ``False`` for an explicit choice; None (default per provider) for null, "default", "auto" or junk."""
    if value is None or isinstance(value, str) and value.strip().lower() in {"", "null", "none", "default", "auto"}:
        return None
    if isinstance(value, bool):
        return value
    parsed = _as_bool(value, None)  # type: ignore[arg-type]
    return parsed if isinstance(parsed, bool) else None


def effective_full_control(settings: Any) -> bool:
    """Whether full control is on: the user's explicit ``full_control`` choice, else on for cloud providers (Claude,
    OpenAI, OpenRouter) and off for Ollama."""
    explicit = normalize_full_control(getattr(settings, "full_control", None))
    if explicit is not None:
        return explicit
    return str(getattr(settings, "provider", "") or "") in CLOUD_PROVIDERS


def _apply(settings: AssistantSettings, data: dict[str, Any]) -> None:
    """Overlay the known keys of ``data`` onto ``settings`` with type coercion."""
    defaults = AssistantSettings()
    for item in fields(AssistantSettings):
        if item.name not in data:
            continue
        value = data[item.name]
        current = getattr(defaults, item.name)
        if item.name == "provider":
            text = str(value or "").strip().lower()
            if text in PROVIDERS:
                settings.provider = text
        elif item.name == "base_urls":
            if isinstance(value, dict):
                settings.base_urls = {**settings.base_urls,
                                      **{str(k): str(v).rstrip("/") for k, v in value.items() if v}}
        elif item.name == "prices_usd_per_mtok":
            settings.prices_usd_per_mtok = _prices(value)
        elif item.name == "ollama_num_ctx":
            settings.ollama_num_ctx = normalize_num_ctx(value)
        elif item.name == "full_control":
            settings.full_control = normalize_full_control(value)
        elif isinstance(current, bool):
            setattr(settings, item.name, _as_bool(value, getattr(settings, item.name)))
        elif isinstance(current, int):
            setattr(settings, item.name, int(_as_number(value, getattr(settings, item.name), float)))
        elif isinstance(current, float):
            setattr(settings, item.name, float(_as_number(value, getattr(settings, item.name), float)))
        elif isinstance(current, str):
            text = "" if value is None else str(value).strip()
            if item.name == "ollama_url_default":
                text = text.rstrip("/") or DEFAULT_OLLAMA_URL
            setattr(settings, item.name, text)


def load_settings() -> AssistantSettings:
    """Defaults from ``config/ai_assistant.yaml`` merged with the user file; never raises for broken files."""
    settings = AssistantSettings()
    defaults = secretfile.read_yaml(DEFAULTS_PATH)
    if defaults:
        _apply(settings, defaults)
    user = secretfile.read_yaml(user_settings_path())
    if user:
        _apply(settings, user)
    return settings


def save_settings(settings: AssistantSettings) -> Path:
    """Write the user-changeable fields to the user file (never a repository file); returns its path."""
    path = user_settings_path()
    data = dict(secretfile.read_yaml(path) or {})
    values = asdict(settings)
    for name in USER_FIELDS:
        data[name] = values[name]
    return secretfile.write_private_yaml(path, data)


def is_configured(settings: AssistantSettings, creds: dict[str, Any]) -> bool:
    """True when the selected provider can be reached: an API key, or for Ollama a server URL."""
    if settings.provider == "ollama":
        return bool(str(creds.get("ollama_url") or "").strip())
    if settings.provider in CLOUD_PROVIDERS:
        return bool(str(creds.get(f"{settings.provider}_api_key") or "").strip())
    return False


def is_active(settings: AssistantSettings, creds: dict[str, Any]) -> bool:
    """True when the assistant is enabled, its provider is configured and a model is chosen."""
    return bool(settings.enabled and is_configured(settings, creds) and settings.model.strip())
