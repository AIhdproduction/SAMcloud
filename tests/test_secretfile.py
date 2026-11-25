"""Shared private-file helpers (samcloud.core.secretfile) and the share credentials built on them."""

from __future__ import annotations

from pathlib import Path

import pytest

from samcloud.core import secretfile
from samcloud.share import credentials as share_creds


def test_config_dir_uses_appdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    assert secretfile.config_dir() == tmp_path / "appdata" / "SAMcloud"
    assert share_creds.credentials_path() == tmp_path / "appdata" / "SAMcloud" / "share_credentials.yaml"


def test_config_dir_falls_back_to_xdg_without_appdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(secretfile.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert secretfile.config_dir() == tmp_path / "xdg" / "samcloud"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert secretfile.config_dir() == Path.home() / ".config" / "samcloud"


def test_write_read_and_delete_private_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    restricted: list[Path] = []
    monkeypatch.setattr(secretfile, "restrict", lambda path: restricted.append(path))
    path = tmp_path / "nested" / "dir" / "file.yaml"
    assert secretfile.read_yaml(path) is None
    returned = secretfile.write_private_yaml(path, {"b": 2, "a": "x"})
    assert returned == path and restricted == [path]
    assert path.read_text(encoding="utf-8").splitlines() == ["b: 2", "a: x"]  # key order kept
    assert secretfile.read_yaml(path) == {"b": 2, "a": "x"}
    assert secretfile.delete_private(path) is True and not path.exists()
    assert secretfile.delete_private(path) is False


def test_read_yaml_rejects_broken_or_non_mapping_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("a: [1, 2\n", encoding="utf-8")
    listing = tmp_path / "list.yaml"
    listing.write_text("- 1\n- 2\n", encoding="utf-8")
    assert secretfile.read_yaml(broken) is None
    assert secretfile.read_yaml(listing) is None


def test_write_private_yaml_regrants_after_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original = Path.write_text

    def flaky_write(self: Path, *args, **kwargs):
        if not calls:
            calls.append("denied")
            raise PermissionError("denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky_write)
    path = tmp_path / "acl.yaml"
    secretfile.write_private_yaml(path, {"k": "v"}, restrict_fn=lambda p: calls.append("restrict"),
                                  regrant_fn=lambda p: calls.append("regrant"))
    assert calls == ["denied", "regrant", "restrict"]
    assert secretfile.read_yaml(path) == {"k": "v"}


def test_mask_behaviour() -> None:
    assert secretfile.mask("abcdefghijkl") == "********ijkl"
    assert secretfile.mask("ab") == "********"
    assert secretfile.mask("abcdefgh", 0) == "********"
    assert secretfile.mask(None) == "********"  # type: ignore[arg-type]
    assert share_creds.mask is secretfile.mask


def test_share_credentials_keep_patchable_aliases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    restricted: list[Path] = []
    monkeypatch.setattr(share_creds, "_restrict", lambda path: restricted.append(path))
    data = {"account_id": "acc", "access_key_id": "AKIA-DUMMY-0000", "secret_access_key": "dummy-secret-0000",
            "bucket": "bucket", "public_base_url": "https://example.invalid/"}
    path = share_creds.save_credentials(data)
    assert restricted == [path]
    assert share_creds.load_credentials()["public_base_url"] == "https://example.invalid"
    for name in ("FIELDS", "FILE_NAME", "credentials_path", "normalize", "is_complete", "load_credentials",
                 "save_credentials", "delete_credentials", "mask", "masked", "_restrict", "_regrant"):
        assert hasattr(share_creds, name), name
    assert share_creds.delete_credentials() is True and not path.exists()
