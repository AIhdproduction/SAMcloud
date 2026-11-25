"""AI assistant settings card on the settings page (NiceGUI user simulation, no network)."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from samcloud.app.components import assistant_settings as card
from samcloud.app.pages import settings as settings_page
from samcloud.assistant import credentials as ai_creds
from samcloud.assistant import settings as ai_settings
from samcloud.assistant.providers import ConnectionResult, ModelInfo
from test_apriltag_gui import _element, _page_user, _set

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

KEY = "sk-test-DUMMY-0000"


def _settings_route() -> None:
    settings_page.render()


_settings_route.__module__ = "tests.assistant_settings_route"


def _config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    return tmp_path / "appdata" / "SAMcloud"


def _shown(element: Any) -> bool:
    while element is not None:
        if not element.visible:
            return False
        slot = element.parent_slot
        element = slot.parent if slot is not None else None
    return True


def _hidden(user: Any, marker: str) -> bool:
    """True when no visible element carries ``marker`` (the simulated user only finds visible elements)."""
    return _markers_present(user, marker) == 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.05)


class FakeProvider:
    name = "fake"

    def __init__(self, models: list[ModelInfo], result: ConnectionResult | None = None) -> None:
        self.models = models
        self.result = result or ConnectionResult(True, "ok", "fine")

    def list_models(self) -> list[ModelInfo]:
        return self.models

    def test_connection(self, model: str | None) -> ConnectionResult:
        return self.result


@pytest.mark.anyio
async def test_card_defaults_to_ollama_and_opening_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _config_home(tmp_path, monkeypatch)
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert _element(user, "assistant-settings-card") is not None
        assert _element(user, "assistant-enabled").value is False
        provider = _element(user, "assistant-provider")
        assert provider.value == "ollama" and list(provider.options)[0] == "ollama"
        assert list(provider.options) == ["ollama", "anthropic", "openai", "openrouter"]
        assert _element(user, "assistant-ollama-url").value == "http://localhost:11434"
        assert _shown(_element(user, "assistant-ollama-url")) and _shown(_element(user, "assistant-ollama-hint"))
        assert _hidden(user, "assistant-cost-request") and _hidden(user, "assistant-cost-day")
        assert _hidden(user, "assistant-api-key")
        assert _element(user, "assistant-text-only").value is True
    assert not home.exists()


@pytest.mark.anyio
async def test_cloud_provider_shows_key_and_cost_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config_home(tmp_path, monkeypatch)
    ai_creds.save_ai_credentials({"anthropic_api_key": KEY})
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        _set(user, "assistant-provider", "anthropic")
        await asyncio.sleep(0.1)
        assert _shown(_element(user, "assistant-cost-request")) and _shown(_element(user, "assistant-cost-day"))
        assert _element(user, "assistant-cost-request").value == pytest.approx(0.5)
        key_field = _element(user, "assistant-api-key")
        assert _shown(key_field) and key_field.props.get("type") == "password" and not key_field.value
        status = _element(user, "assistant-key-status").text
        assert "********0000" in status and KEY not in status
        assert _hidden(user, "assistant-ollama-url")
        _set(user, "assistant-provider", "openai")
        await asyncio.sleep(0.1)
        assert "No key stored" in _element(user, "assistant-key-status").text
        assert not _element(user, "assistant-delete-key").enabled
        _set(user, "assistant-provider", "anthropic")
        await asyncio.sleep(0.1)
        assert _element(user, "assistant-delete-key").enabled
        user.find(marker="assistant-delete-key").click()
        await asyncio.sleep(0.1)
        assert "No key stored" in _element(user, "assistant-key-status").text
    assert ai_creds.key_for("anthropic") == ""


@pytest.mark.anyio
async def test_saving_ollama_writes_only_user_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _config_home(tmp_path, monkeypatch)
    defaults_before = _sha(ai_settings.DEFAULTS_PATH)
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        _set(user, "assistant-enabled", True)
        _set(user, "assistant-ollama-url", "http://192.168.1.20:11434/")
        _set(user, "assistant-model", "local-model:7b")
        user.find(marker="assistant-save").click()
        await asyncio.sleep(0.1)
        assert _markers_present(user, "assistant-privacy-dialog") == 0  # local provider: no privacy dialog
    assert sorted(p.name for p in home.iterdir()) == ["ai_credentials.yaml", "ai_settings.yaml"]
    saved = yaml.safe_load((home / "ai_settings.yaml").read_text(encoding="utf-8"))
    assert saved["enabled"] is True and saved["provider"] == "ollama" and saved["model"] == "local-model:7b"
    assert ai_creds.load_ai_credentials()["ollama_url"] == "http://192.168.1.20:11434"
    assert _sha(ai_settings.DEFAULTS_PATH) == defaults_before
    assert ai_settings.is_active(ai_settings.load_settings(), ai_creds.load_ai_credentials())


def _markers_present(user: Any, marker: str) -> int:
    try:
        return len(user.find(marker=marker).elements)
    except AssertionError:
        return 0


@pytest.mark.anyio
async def test_cloud_save_needs_privacy_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _config_home(tmp_path, monkeypatch)
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        _set(user, "assistant-enabled", True)
        _set(user, "assistant-provider", "openrouter")
        _set(user, "assistant-api-key", KEY)
        _set(user, "assistant-model", "vendor/model-a")
        user.find(marker="assistant-save").click()
        await asyncio.sleep(0.1)
        assert _markers_present(user, "assistant-privacy-dialog") == 1
        user.find(marker="assistant-privacy-cancel").click()
        await asyncio.sleep(0.1)
        assert not home.exists()  # cancelled: nothing saved
        assert "Not saved" in _element(user, "assistant-save-status").text

        user.find(marker="assistant-save").click()
        await asyncio.sleep(0.1)
        user.find(marker="assistant-privacy-confirm").click()
        await asyncio.sleep(0.1)
        status = _element(user, "assistant-key-status").text
        assert "********0000" in status and KEY not in status
        assert not _element(user, "assistant-api-key").value
    settings_text = (home / "ai_settings.yaml").read_text(encoding="utf-8")
    assert KEY not in settings_text
    saved = yaml.safe_load(settings_text)
    assert saved["privacy_ack_cloud"] is True and saved["provider"] == "openrouter" and saved["enabled"] is True
    assert ai_creds.key_for("openrouter") == KEY


@pytest.mark.anyio
async def test_load_models_fills_select_and_forces_text_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config_home(tmp_path, monkeypatch)
    fake = FakeProvider([ModelInfo("vision-model", supports_images=True), ModelInfo("text-model", supports_images=False)])
    monkeypatch.setattr(card, "get_provider", lambda name, settings, creds: fake)
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        user.find(marker="assistant-load-models").click()
        select = _element(user, "assistant-model-select")
        await _wait_for(lambda: len(select.options) == 2)
        assert "2 models" in _element(user, "assistant-models-status").text
        _set(user, "assistant-text-only", False)
        _set(user, "assistant-model-select", "text-model")
        await asyncio.sleep(0.1)
        text_only = _element(user, "assistant-text-only")
        assert text_only.value is True and not text_only.enabled
        assert _shown(_element(user, "assistant-text-only-hint"))
        _set(user, "assistant-model-select", "vision-model")
        await asyncio.sleep(0.1)
        assert _element(user, "assistant-text-only").enabled
        assert _hidden(user, "assistant-text-only-hint")


@pytest.mark.anyio
async def test_connection_test_shows_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config_home(tmp_path, monkeypatch)
    fake = FakeProvider([], ConnectionResult(False, "auth", f"rejected {KEY}"))
    monkeypatch.setattr(card, "get_provider", lambda name, settings, creds: fake)
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        user.find(marker="assistant-test").click()
        label = _element(user, "assistant-test-result")
        await _wait_for(lambda: label.text.startswith("Wrong key"))
        assert KEY not in label.text
        fake.result = ConnectionResult(False, "unreachable", "down")
        user.find(marker="assistant-test").click()
        await _wait_for(lambda: label.text.startswith("Server not reachable"))
