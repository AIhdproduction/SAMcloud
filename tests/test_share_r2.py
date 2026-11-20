"""Cloudflare R2 provider (no network: fake client and botocore Stubber) and the credentials file."""

from __future__ import annotations

import io
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pytest
from botocore.client import Config
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber

from samcloud.share import credentials as creds
from samcloud.share.providers import LocalProvider, ProviderError, get_provider
from samcloud.share.providers.cloudflare_r2 import CloudflareR2Provider, content_type

SHARE_ID = "AbCdEfGhIjKlMnOpQrStUv"
CREDS = {"account_id": "acc123", "access_key_id": "AKIAEXAMPLEKEY1234", "secret_access_key": "s3cr3t-value-xyz",
         "bucket": "my-bucket", "public_base_url": "https://pub-abc.r2.dev/"}


def _package(root: Path) -> Path:
    package = root / "package"
    (package / "pointcloud").mkdir(parents=True)
    (package / "lib").mkdir()
    (package / "index.html").write_text("<html></html>", encoding="utf-8")
    (package / "lib" / "a.js").write_text("x", encoding="utf-8")
    (package / "lib" / "d.wasm").write_bytes(b"\0asm")
    (package / "pointcloud" / "metadata.json").write_text("{}", encoding="utf-8")
    (package / "pointcloud" / "octree.bin").write_bytes(b"x" * 5000)
    return package


def _stub_client() -> object:
    return boto3.client("s3", endpoint_url="https://acc123.r2.cloudflarestorage.com", region_name="auto",
                        aws_access_key_id="x", aws_secret_access_key="y", config=Config(signature_version="s3v4"))


class _FakeClient:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, dict]] = []

    def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs: dict, Callback) -> None:
        size = Path(filename).stat().st_size
        Callback(size // 2)
        Callback(size - size // 2)
        self.uploads.append((bucket, key, ExtraArgs))


def test_upload_keys_content_types_progress_and_url(tmp_path: Path) -> None:
    client = _FakeClient()
    provider = CloudflareR2Provider(CREDS, client=client)
    calls: list[tuple[int, int]] = []
    url = provider.upload(_package(tmp_path), SHARE_ID, lambda done, total: calls.append((done, total)))
    assert url == f"https://pub-abc.r2.dev/share/{SHARE_ID}/index.html"
    prefix = f"share/{SHARE_ID}/"
    keys = {key: args["ContentType"] for bucket, key, args in client.uploads if bucket == "my-bucket"}
    assert keys == {prefix + "index.html": "text/html; charset=utf-8", prefix + "lib/a.js": "text/javascript; charset=utf-8",
                    prefix + "lib/d.wasm": "application/wasm", prefix + "pointcloud/metadata.json": "application/json",
                    prefix + "pointcloud/octree.bin": "application/octet-stream"}
    assert all(set(args) == {"ContentType"} for _b, _k, args in client.uploads), "no Content-Encoding: Range stays valid"
    assert keys[prefix + "pointcloud/octree.bin"] == "application/octet-stream"
    assert calls[0][0] == 0 and calls[-1][0] == calls[-1][1]
    assert [c[0] for c in calls] == sorted(c[0] for c in calls)
    assert content_type("x.svg") == "image/svg+xml" and content_type("x.png") == "image/png"


def test_delete_removes_the_whole_prefix() -> None:
    client = _stub_client()
    prefix = f"share/{SHARE_ID}/"
    with Stubber(client) as stub:
        stub.add_response("list_objects_v2", {"Contents": [{"Key": prefix + "index.html"}, {"Key": prefix + "a/b.bin"}],
                                              "IsTruncated": False}, {"Bucket": "my-bucket", "Prefix": prefix})
        stub.add_response("delete_objects", {}, {"Bucket": "my-bucket", "Delete": {
            "Objects": [{"Key": prefix + "index.html"}, {"Key": prefix + "a/b.bin"}], "Quiet": True}})
        CloudflareR2Provider(CREDS, client=client).delete(SHARE_ID)
        stub.assert_no_pending_responses()


def test_extend_copies_every_object_in_place_with_replace() -> None:
    client = _stub_client()
    provider = CloudflareR2Provider(CREDS, expiry_days=15, client=client)
    prefix = f"share/{SHARE_ID}/"
    keys = [prefix + "index.html", prefix + "pointcloud/octree.bin"]
    with Stubber(client) as stub:
        stub.add_response("list_objects_v2", {"Contents": [{"Key": k} for k in keys], "IsTruncated": False},
                          {"Bucket": "my-bucket", "Prefix": prefix})
        for key in keys:
            stub.add_response("copy_object", {}, {"Bucket": "my-bucket", "Key": key,
                                                  "CopySource": {"Bucket": "my-bucket", "Key": key},
                                                  "MetadataDirective": "REPLACE", "ContentType": content_type(key),
                                                  "Metadata": ANY})
        expiry = provider.extend(SHARE_ID)
        stub.assert_no_pending_responses()
    assert timedelta(days=14, hours=23) < expiry - datetime.now(timezone.utc) <= timedelta(days=15)


def test_extend_of_a_vanished_package_is_a_clear_error() -> None:
    client = _stub_client()
    with Stubber(client) as stub:
        stub.add_response("list_objects_v2", {"IsTruncated": False},
                          {"Bucket": "my-bucket", "Prefix": f"share/{SHARE_ID}/"})
        with pytest.raises(ProviderError, match="neu erzeugen"):
            CloudflareR2Provider(CREDS, client=client).extend(SHARE_ID)


def test_ensure_lifecycle_merges_with_existing_rules() -> None:
    client = _stub_client()
    keep = {"ID": "keep", "Status": "Enabled", "Filter": {"Prefix": "tmp/"}, "Expiration": {"Days": 3}}
    old = {"ID": "samcloud-share-expiry", "Status": "Enabled", "Filter": {"Prefix": "share/"}, "Expiration": {"Days": 99}}
    ours = {"ID": "samcloud-share-expiry", "Status": "Enabled", "Filter": {"Prefix": "share/"}, "Expiration": {"Days": 15}}
    with Stubber(client) as stub:
        stub.add_response("get_bucket_lifecycle_configuration", {"Rules": [keep, old]}, {"Bucket": "my-bucket"})
        stub.add_response("put_bucket_lifecycle_configuration", {},
                          {"Bucket": "my-bucket", "LifecycleConfiguration": {"Rules": [keep, ours]}})
        CloudflareR2Provider(CREDS, client=client).ensure_lifecycle(15)
        stub.assert_no_pending_responses()
    with Stubber(client) as stub:
        stub.add_client_error("get_bucket_lifecycle_configuration", "NoSuchLifecycleConfiguration")
        stub.add_response("put_bucket_lifecycle_configuration", {},
                          {"Bucket": "my-bucket", "LifecycleConfiguration": {"Rules": [ours]}})
        CloudflareR2Provider(CREDS, client=client).ensure_lifecycle()
        stub.assert_no_pending_responses()


def test_connection_test_reports_german_errors() -> None:
    client = _stub_client()
    provider = CloudflareR2Provider(CREDS, client=client)
    with Stubber(client) as stub:
        stub.add_client_error("put_object", "InvalidAccessKeyId")
        stub.add_response("delete_object", {})
        with pytest.raises(ProviderError, match="Access Key"):
            provider.test_connection()
    with Stubber(client) as stub:
        stub.add_client_error("put_object", "NoSuchBucket")
        stub.add_response("delete_object", {})
        with pytest.raises(ProviderError, match="nicht gefunden"):
            provider.test_connection()


def test_connection_test_public_access_failure_still_cleans_up() -> None:
    sent: dict[str, object] = {}

    class Echo:
        def put_object(self, **kw): sent["body"] = kw["Body"]
        def get_object(self, **kw): return {"Body": io.BytesIO(sent["body"])}
        def delete_object(self, **kw): sent["deleted"] = kw["Key"]

    provider = CloudflareR2Provider(CREDS, client=Echo())
    provider._fetch = lambda url: (_ for _ in ()).throw(urllib.error.HTTPError(url, 403, "Forbidden", {}, None))
    with pytest.raises(ProviderError, match="ffentlicher Zugriff"):
        provider.test_connection()
    assert "deleted" in sent


def test_connection_test_success_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, object] = {}

    class Echo:
        def put_object(self, **kw): sent["body"] = kw["Body"]
        def get_object(self, **kw): return {"Body": io.BytesIO(sent["body"])}
        def delete_object(self, **kw): sent["deleted"] = kw["Key"]

    provider = CloudflareR2Provider(CREDS, client=Echo())
    monkeypatch.setattr(provider, "_fetch", lambda url: sent["body"])
    assert "erfolgreich" in provider.test_connection()
    assert str(sent["deleted"]).startswith("share/_connection_test/")


def test_credentials_roundtrip_location_and_masking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    assert creds.load_credentials() is None and not creds.delete_credentials()
    path = creds.save_credentials(CREDS)
    assert path == tmp_path / "appdata" / "SAMcloud" / "share_credentials.yaml" and path.is_file()
    loaded = creds.load_credentials()
    assert loaded is not None and loaded["public_base_url"] == "https://pub-abc.r2.dev"
    assert loaded["secret_access_key"] == CREDS["secret_access_key"]
    shown = creds.masked(CREDS)
    assert CREDS["secret_access_key"] not in str(shown) and shown["access_key_id"].endswith("1234")
    assert creds.mask("abcdefghijkl") == "********ijkl" and creds.mask("ab") == "********"
    with pytest.raises(ValueError):
        creds.save_credentials({**CREDS, "bucket": ""})
    assert creds.delete_credentials() and creds.load_credentials() is None


def test_get_provider_selects_r2_with_credentials_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.delenv("SAMCLOUD_SHARE_PROVIDER", raising=False)
    with pytest.raises(ProviderError, match="Zugangsdaten"):
        get_provider({"provider": "cloudflare_r2"}, tmp_path)
    creds.save_credentials(CREDS)
    provider = get_provider({"provider": "cloudflare_r2", "expiry_days": 15}, tmp_path)
    assert isinstance(provider, CloudflareR2Provider) and provider.expiry_days == 15
    monkeypatch.setenv("SAMCLOUD_SHARE_PROVIDER", "local")
    assert isinstance(get_provider({"provider": "cloudflare_r2"}, tmp_path), LocalProvider)


def test_gitignore_protects_credentials_and_packages() -> None:
    lines = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in ("*credentials*.yaml", "*.local.yaml", "products/share/"):
        assert pattern in lines
