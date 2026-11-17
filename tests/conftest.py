"""Test workspace fixture compatible with the project's no-tmpdir test command."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Give each test a durable directory below SAMCLOUD_TEST_WORKDIR."""
    root = Path(os.environ["SAMCLOUD_TEST_WORKDIR"])
    root.mkdir(parents=True, exist_ok=True)
    root = Path(os.path.realpath(root))  # expand Windows 8.3 short names (HDPROD~1)
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)[:64] + "_"
    path = root / f"{prefix}{uuid.uuid4().hex}"
    path.mkdir()
    return path


def use_language(monkeypatch: pytest.MonkeyPatch, language: str) -> None:
    """Switch the UI language of the running test (e.g. ``"de"``) the same way ``_english_ui`` forces English."""
    from samcloud.app import i18n

    monkeypatch.setattr(i18n, "_read_config_language", lambda: language)
    i18n.reload_translations()


@pytest.fixture
def german_ui(_english_ui: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a test with the German UI (for tests that assert the German wording)."""
    use_language(monkeypatch, "de")


@pytest.fixture(autouse=True)
def _english_ui(monkeypatch: pytest.MonkeyPatch):
    """Run every test with the English UI and never touch the real user preferences file.

    The preferences path points to a per-test location that is only created when a test
    stores a language; config/defaults.yml is bypassed so the config level resolves to "en".
    """
    from samcloud.app import i18n

    base = Path(os.environ.get("SAMCLOUD_TEST_WORKDIR") or tempfile.gettempdir())
    preferences = base / f"i18n_{uuid.uuid4().hex}" / ".samcloud" / "user_preferences.json"
    monkeypatch.setattr(i18n, "PREFERENCES_PATH", preferences)
    monkeypatch.setattr(i18n, "_read_config_language", lambda: "en")
    i18n.reload_translations()
    yield
    i18n.reload_translations()
