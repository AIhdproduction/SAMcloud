"""AI assistant settings, credentials (keys never leaked), redaction and cost limits."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import pytest
import yaml

from samcloud.assistant import costs
from samcloud.assistant import credentials as ai_creds
from samcloud.assistant import settings as ai_settings
from samcloud.assistant.providers.base import ModelInfo, Usage
from samcloud.core import secretfile
from samcloud.core.paths import PROJECT_ROOT

ANTHROPIC_KEY = "sk-ant-test-DUMMY-0000"
OPENAI_KEY = "sk-test-DUMMY-0000"
OPENROUTER_KEY = "sk-or-test-DUMMY-0000"


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    return tmp_path / "appdata" / "SAMcloud"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- defaults


def test_provider_order_and_defaults(config_home: Path) -> None:
    assert ai_settings.PROVIDERS[0] == "ollama"
    assert ai_settings.PROVIDERS == ("ollama", "anthropic", "openai", "openrouter")
    settings = ai_settings.load_settings()
    assert settings.enabled is False and settings.provider == "ollama" and settings.model == ""
    assert settings.text_only is True and settings.max_tool_calls == 20
    assert settings.cost_limit_request_usd == pytest.approx(0.5)
    assert settings.cost_limit_project_day_usd == pytest.approx(5.0)
    assert settings.ollama_url_default == "http://localhost:11434"
    assert settings.base_urls["openrouter"] == "https://openrouter.ai/api/v1"
    assert settings.privacy_ack_cloud is False
    creds = ai_creds.load_ai_credentials()
    assert creds == {"ollama_url": "http://localhost:11434", "anthropic_api_key": "", "openai_api_key": "",
                     "openrouter_api_key": ""}
    assert not config_home.exists()  # loading never writes


def test_repo_defaults_file_matches_dataclass_defaults() -> None:
    data = yaml.safe_load(ai_settings.DEFAULTS_PATH.read_text(encoding="utf-8"))
    assert data["enabled"] is False and data["provider"] == "ollama" and data["text_only"] is True
    assert data["ollama_url_default"] == "http://localhost:11434"
    assert set(data["base_urls"]) == {"anthropic", "openai", "openrouter"}
    assert not data.get("prices_usd_per_mtok")  # no invented prices
    text = ai_settings.DEFAULTS_PATH.read_text(encoding="utf-8")
    assert "sk-" not in text and "api_key" not in text


# --------------------------------------------------------------------------- credentials


def test_keys_saved_owner_only_masked_and_never_logged(config_home: Path, monkeypatch: pytest.MonkeyPatch,
                                                       caplog: pytest.LogCaptureFixture) -> None:
    restricted: list[Path] = []
    monkeypatch.setattr(secretfile, "restrict", lambda path: restricted.append(path))
    caplog.set_level(logging.DEBUG)
    path = ai_creds.save_ai_credentials({"anthropic_api_key": f"  {ANTHROPIC_KEY} ", "openai_api_key": OPENAI_KEY})
    assert path == config_home / "ai_credentials.yaml" == ai_creds.credentials_path()
    assert restricted == [path]
    stored = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert stored["anthropic_api_key"] == ANTHROPIC_KEY and stored["openai_api_key"] == OPENAI_KEY
    masked = ai_creds.masked_ai_credentials()
    assert masked["anthropic_api_key"] == "********0000" and ANTHROPIC_KEY not in str(masked)
    assert masked["openrouter_api_key"] == "" and masked["ollama_url"] == "http://localhost:11434"
    assert ai_creds.key_for("anthropic") == ANTHROPIC_KEY and ai_creds.key_for("ollama") == ""
    for secret in (ANTHROPIC_KEY, OPENAI_KEY):
        assert secret not in caplog.text
    # the settings file never receives a key
    settings = ai_settings.load_settings()
    settings.provider = "anthropic"
    settings_path = ai_settings.save_settings(settings)
    text = settings_path.read_text(encoding="utf-8")
    assert ANTHROPIC_KEY not in text and OPENAI_KEY not in text and "api_key" not in text


def test_empty_key_keeps_stored_value_and_delete_removes_it(config_home: Path) -> None:
    ai_creds.save_ai_credentials({"openrouter_api_key": OPENROUTER_KEY})
    ai_creds.save_ai_credentials({"openrouter_api_key": "", "openai_api_key": None})
    assert ai_creds.key_for("openrouter") == OPENROUTER_KEY
    assert ai_creds.delete_ai_key("openrouter") is True
    assert ai_creds.key_for("openrouter") == ""
    assert ai_creds.delete_ai_key("openrouter") is False
    with pytest.raises(ValueError):
        ai_creds.delete_ai_key("unknown")


def test_ollama_url_changeable_and_persisted(config_home: Path) -> None:
    ai_creds.save_ai_credentials({"ollama_url": "http://192.168.1.20:11434/"})
    assert ai_creds.load_ai_credentials()["ollama_url"] == "http://192.168.1.20:11434"
    assert ai_creds.masked_ai_credentials()["ollama_url"] == "http://192.168.1.20:11434"
    assert ai_creds.delete_ai_key("ollama") is True
    assert ai_creds.load_ai_credentials()["ollama_url"] == "http://localhost:11434"
    ai_creds.save_ai_credentials({"ollama_url": "   "})
    assert ai_creds.load_ai_credentials()["ollama_url"] == "http://localhost:11434"


def test_redact_replaces_stored_and_token_like_secrets(config_home: Path) -> None:
    ai_creds.save_ai_credentials({"anthropic_api_key": "plainkey-without-prefix-1234"})
    text = ("key plainkey-without-prefix-1234, header Authorization: Bearer abc.def-123 and "
            f"{OPENAI_KEY} and x-api-key: zzzzzzzz9")
    redacted = ai_creds.redact(text, extra=("extra-secret-77",))
    assert "plainkey-without-prefix-1234" not in redacted
    assert "abc.def-123" not in redacted and OPENAI_KEY not in redacted and "zzzzzzzz9" not in redacted
    assert "Bearer ***" in redacted
    assert ai_creds.redact("extra-secret-77 here", extra=("extra-secret-77",)) == "*** here"
    assert ai_creds.redact("nothing secret") == "nothing secret"


# --------------------------------------------------------------------------- settings


def test_settings_saved_only_to_user_file_outside_repo(config_home: Path) -> None:
    before = _sha(ai_settings.DEFAULTS_PATH)
    settings = ai_settings.load_settings()
    settings.enabled = True
    settings.provider = "openrouter"
    settings.model = "vendor/model-a"
    settings.text_only = False
    settings.cost_limit_request_usd = 0.25
    settings.privacy_ack_cloud = True
    path = ai_settings.save_settings(settings)
    assert path == config_home / "ai_settings.yaml" == ai_settings.user_settings_path()
    assert PROJECT_ROOT.resolve() not in path.resolve().parents
    assert _sha(ai_settings.DEFAULTS_PATH) == before
    loaded = ai_settings.load_settings()
    assert (loaded.enabled, loaded.provider, loaded.model, loaded.text_only) == (True, "openrouter", "vendor/model-a", False)
    assert loaded.cost_limit_request_usd == pytest.approx(0.25) and loaded.privacy_ack_cloud is True
    assert loaded.base_urls["openai"] == "https://api.openai.com/v1"  # repo defaults still apply


def test_user_file_extra_keys_kept_and_broken_values_ignored(config_home: Path) -> None:
    config_home.mkdir(parents=True)
    path = config_home / "ai_settings.yaml"
    path.write_text("provider: nonsense\nmax_tool_calls: many\nrequest_timeout_s: 30\n"
                    "base_urls: {openai: 'http://127.0.0.1:9/v1/'}\n"
                    "prices_usd_per_mtok: {good: {input: 1, output: 2}, bad: {input: x}}\n", encoding="utf-8")
    loaded = ai_settings.load_settings()
    assert loaded.provider == "ollama" and loaded.max_tool_calls == 20 and loaded.request_timeout_s == 30
    assert loaded.base_urls["openai"] == "http://127.0.0.1:9/v1"
    assert loaded.prices_usd_per_mtok == {"good": {"input": 1.0, "output": 2.0}}
    ai_settings.save_settings(loaded)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["request_timeout_s"] == 30 and "base_urls" in saved and saved["provider"] == "ollama"


def test_broken_user_file_falls_back_to_defaults(config_home: Path) -> None:
    config_home.mkdir(parents=True)
    (config_home / "ai_settings.yaml").write_text("enabled: [\n", encoding="utf-8")
    assert ai_settings.load_settings().enabled is False


def test_is_active_requires_switch_configuration_and_model() -> None:
    creds = {"ollama_url": "http://localhost:11434", "anthropic_api_key": "", "openai_api_key": OPENAI_KEY,
             "openrouter_api_key": ""}
    settings = ai_settings.AssistantSettings(enabled=True, provider="ollama", model="local-model")
    assert ai_settings.is_active(settings, creds)
    assert not ai_settings.is_active(ai_settings.AssistantSettings(enabled=False, model="m"), creds)
    assert not ai_settings.is_active(ai_settings.AssistantSettings(enabled=True, model=" "), creds)
    assert not ai_settings.is_active(ai_settings.AssistantSettings(enabled=True, provider="anthropic", model="m"), creds)
    assert ai_settings.is_active(ai_settings.AssistantSettings(enabled=True, provider="openai", model="m"), creds)


# --------------------------------------------------------------------------- costs


def _cloud(**overrides) -> ai_settings.AssistantSettings:
    values = {"enabled": True, "provider": "openai", "model": "family-large-2", "cost_limit_request_usd": 0.5,
              "cost_limit_project_day_usd": 5.0,
              "prices_usd_per_mtok": {"family": {"input": 1.0, "output": 4.0},
                                      "family-large": {"input": 3.0, "output": 15.0}}}
    values.update(overrides)
    return ai_settings.AssistantSettings(**values)


def test_price_lookup_live_then_longest_prefix_then_none() -> None:
    settings = _cloud()
    live = ModelInfo("family-large-2", input_usd_per_mtok=0.7, output_usd_per_mtok=2.1)
    assert costs.price_for(live, settings) == (0.7, 2.1)
    assert costs.price_for(ModelInfo("family-large-2"), settings) == (3.0, 15.0)
    assert costs.price_for("family-small", settings) == (1.0, 4.0)
    assert costs.price_for("other-model", settings) is None
    assert costs.price_for("anything", _cloud(provider="ollama")) == (0.0, 0.0)


def test_estimates_and_actual_cost() -> None:
    estimate = costs.estimate_request_cost(4001, 1000, (3.0, 15.0))
    assert estimate == pytest.approx((1001 * 3.0 + 1000 * 15.0) / 1e6)
    assert costs.estimate_request_cost(100, 10, None) is None
    assert costs.cost_of(Usage(2000, 500), (3.0, 15.0)) == pytest.approx((2000 * 3 + 500 * 15) / 1e6)
    assert costs.cost_of(Usage(1, 1), None) is None


def test_cost_ledger_accumulates_per_day(tmp_path: Path) -> None:
    ledger = costs.CostLedger(tmp_path / "project")
    assert ledger.spent_today() == 0.0
    assert ledger.add(0.1234) == pytest.approx(0.1234)
    assert ledger.add(0.2) == pytest.approx(0.3234)
    ledger.add(-1.0)
    assert ledger.spent_today() == pytest.approx(0.3234)
    data = json.loads((tmp_path / "project" / "products" / "assistant" / "cost_ledger.json").read_text(encoding="utf-8"))
    assert data == {date.today().isoformat(): pytest.approx(0.3234)}


def test_check_limits_request_day_unknown_ollama_and_zero(tmp_path: Path) -> None:
    ledger = costs.CostLedger(tmp_path / "project")
    settings = _cloud()
    assert costs.check_limits(settings, 0.1, ledger) is None
    reason = costs.check_limits(settings, 0.73, ledger)
    assert reason is not None and "0.73" in reason and "0.50" in reason
    ledger.add(4.9)
    reason = costs.check_limits(settings, 0.2, ledger)
    assert reason is not None and "5.00" in reason and "4.90" in reason
    unknown = costs.check_limits(settings, None, ledger)
    assert unknown is not None and "config/ai_assistant.yaml" in unknown and "family-large-2" in unknown
    ollama = _cloud(provider="ollama", cost_limit_request_usd=0.01, cost_limit_project_day_usd=0.01)
    assert costs.check_limits(ollama, None, ledger) is None and costs.check_limits(ollama, 99.0, ledger) is None
    unlimited = _cloud(cost_limit_request_usd=0, cost_limit_project_day_usd=0)
    assert costs.check_limits(unlimited, 99.0, ledger) is None and costs.check_limits(unlimited, None, ledger) is None
    only_day = _cloud(cost_limit_request_usd=0)
    assert costs.check_limits(only_day, 0.05, ledger) is None and costs.check_limits(only_day, 0.2, ledger) is not None
