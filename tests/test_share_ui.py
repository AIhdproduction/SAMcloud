"""Share UI: button availability, Cloudflare setup dialog, share dialog states (active / expired)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from samcloud.app.pages import dense_cloud, settings as settings_page
from samcloud.share import credentials as creds
from test_apriltag_gui import _element, _page_user
from test_share_export import make_project

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

CREDS = {"account_id": "acc123", "access_key_id": "AKIAEXAMPLEKEY1234", "secret_access_key": "s3cr3t",
         "bucket": "my-bucket", "public_base_url": "https://pub-abc.r2.dev"}


def _dense_route() -> None:
    dense_cloud.render()


def _settings_route() -> None:
    settings_page.render()


_dense_route.__module__ = "tests.share_ui_route"
_settings_route.__module__ = "tests.share_ui_route"


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, provider: str | None = None) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    if provider:
        monkeypatch.setenv("SAMCLOUD_SHARE_PROVIDER", provider)
    else:
        monkeypatch.delenv("SAMCLOUD_SHARE_PROVIDER", raising=False)


def _store_share(project: Path, *, expires_in_days: float) -> None:
    path = project / "project.samcloud.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    document["share"] = {"url": "https://pub-abc.r2.dev/share/AbCdEfGhIjKlMnOpQrStUv/index.html",
                         "share_id": "AbCdEfGhIjKlMnOpQrStUv", "created_at": now.isoformat(timespec="seconds"),
                         "provider": "local", "points": 1234, "size_mb": 5.5,
                         "expires_at": (now + timedelta(days=expires_in_days)).isoformat(timespec="seconds")}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")


def _markers(user: Any, marker: str) -> int:
    try:
        return len(user.find(marker=marker).elements)
    except AssertionError:  # NiceGUI's finder raises when nothing matches
        return 0


@pytest.mark.anyio
async def test_button_is_disabled_before_step_6_and_enabled_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    early = make_project(tmp_path, "early", refinement="running", with_cloud=False)
    async with _page_user("/dense", _dense_route, early, monkeypatch) as user:
        await asyncio.sleep(0.3)
        button = _element(user, "share-button")
        assert "disable" in button.props and button.props["disable"] is not False
    done = make_project(tmp_path, "done", refinement="completed", with_cloud=False)
    async with _page_user("/dense", _dense_route, done, monkeypatch) as user:
        await asyncio.sleep(0.3)
        button = _element(user, "share-button")
        assert not button.props.get("disable")
        assert button.text == "Share (QR)"


@pytest.mark.anyio
async def test_click_without_credentials_opens_the_setup_dialog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    project = make_project(tmp_path, "p", with_cloud=False)
    async with _page_user("/dense", _dense_route, project, monkeypatch) as user:
        await asyncio.sleep(0.3)
        user.find(marker="share-button").click()
        await user.should_see(marker="share-setup-dialog")
        for key in creds.FIELDS:
            assert _markers(user, f"share-field-{key}") == 1
        assert _markers(user, "share-test") == 1 and _markers(user, "share-save") == 1
        assert _markers(user, "share-dialog") == 0


@pytest.mark.anyio
async def test_setup_dialog_saves_credentials_to_appdata_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    project = make_project(tmp_path, "p", with_cloud=False)
    async with _page_user("/dense", _dense_route, project, monkeypatch) as user:
        await asyncio.sleep(0.3)
        user.find(marker="share-button").click()
        await user.should_see(marker="share-setup-dialog")
        user.find(marker="share-save").click()  # empty fields: refused
        assert creds.load_credentials() is None
        assert "alle Felder" in _element(user, "share-setup-result").text or "all fields" in _element(user, "share-setup-result").text
        for key, value in CREDS.items():
            with user:
                _element(user, f"share-field-{key}").value = value
        user.find(marker="share-save").click()
        await user.should_see(marker="share-dialog")
    stored = creds.load_credentials()
    assert stored is not None and stored["bucket"] == "my-bucket"
    assert creds.credentials_path() == tmp_path / "appdata" / "SAMcloud" / "share_credentials.yaml"


@pytest.mark.anyio
async def test_with_credentials_the_share_flow_starts_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    creds.save_credentials(CREDS)
    project = make_project(tmp_path, "p", with_cloud=False)
    async with _page_user("/dense", _dense_route, project, monkeypatch) as user:
        await asyncio.sleep(0.3)
        user.find(marker="share-button").click()
        await user.should_see(marker="share-dialog")
        assert _markers(user, "share-setup-dialog") == 0 and _markers(user, "share-start") == 1


@pytest.mark.anyio
async def test_active_share_shows_link_validity_and_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path, provider="local")
    project = make_project(tmp_path, "p", with_cloud=False)
    _store_share(project, expires_in_days=9)
    async with _page_user("/dense", _dense_route, project, monkeypatch) as user:
        await asyncio.sleep(0.3)
        user.find(marker="share-button").click()
        await user.should_see(marker="share-link")
        for marker in ("share-valid-until", "share-extend", "share-recreate", "share-delete", "share-copy", "share-save-png"):
            assert _markers(user, marker) == 1, marker
        assert "valid until" in _element(user, "share-valid-until").text
        assert _markers(user, "share-expired") == 0


@pytest.mark.anyio
async def test_expired_share_is_flagged_and_offers_only_recreate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path, provider="local")
    project = make_project(tmp_path, "p", with_cloud=False)
    _store_share(project, expires_in_days=-2)
    async with _page_user("/dense", _dense_route, project, monkeypatch) as user:
        await asyncio.sleep(0.3)
        user.find(marker="share-button").click()
        await user.should_see(marker="share-expired")
        assert "expired" in _element(user, "share-expired").text
        assert _markers(user, "share-recreate") == 1 and _markers(user, "share-extend") == 0
        assert _markers(user, "share-link") == 0


@pytest.mark.anyio
async def test_settings_page_has_cloudflare_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert _markers(user, "share-settings-card") == 1
        assert "No Cloudflare account" in _element(user, "share-settings-status").text
        assert not _element(user, "share-settings-delete").enabled
        creds.save_credentials(CREDS)
    async with _page_user("/settings", _settings_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert "my-bucket" in _element(user, "share-settings-status").text
        assert _element(user, "share-settings-delete").enabled
        user.find(marker="share-settings-delete").click()
        assert creds.load_credentials() is None


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows ACL test")
def test_delete_credentials_after_old_rw_acl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import subprocess

    _isolate(monkeypatch, tmp_path)
    path = creds.save_credentials(CREDS)
    user = os.environ["USERNAME"]
    subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(R,W)"], capture_output=True, check=False)
    assert creds.delete_credentials() is True
    assert not path.exists()
