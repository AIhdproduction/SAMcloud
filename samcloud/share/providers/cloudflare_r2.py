"""Cloudflare R2 provider: every SAMcloud user publishes into their own R2 bucket (S3-compatible API)."""

from __future__ import annotations

import logging
import mimetypes
import secrets
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from samcloud.share.providers.base import (
    ProgressCallback,
    ProviderError,
    ShareProvider,
    directory_size,
    validate_share_id,
)

LOGGER = logging.getLogger(__name__)
PREFIX = "share/"
LIFECYCLE_RULE_ID = "samcloud-share-expiry"
_DENIED_CODES = {"AccessDenied", "Forbidden", "403", "Unauthorized", "NotImplemented"}
DEFAULT_EXPIRY_DAYS = 15
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".json": "application/json", ".bin": "application/octet-stream", ".wasm": "application/wasm",
    ".png": "image/png", ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8",
}


def content_type(name: str) -> str:
    """Content-Type of a package file (explicit table first, then ``mimetypes``)."""
    suffix = Path(name).suffix.lower()
    return _CONTENT_TYPES.get(suffix) or mimetypes.guess_type(name)[0] or "application/octet-stream"


def make_client(credentials: dict[str, str]) -> Any:
    """boto3 S3 client for ``https://<account_id>.r2.cloudflarestorage.com`` (region ``auto``)."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=f"https://{credentials['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id=credentials["access_key_id"],
        aws_secret_access_key=credentials["secret_access_key"],
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"},
                      connect_timeout=15, read_timeout=120),
    )


def _error_code(error: Exception) -> str:
    response = getattr(error, "response", None) or {}
    return str(response.get("Error", {}).get("Code", ""))


def german_error(error: Exception) -> str:
    """Clear German message for the typical R2 setup mistakes."""
    code = _error_code(error)
    if code in {"InvalidAccessKeyId", "SignatureDoesNotMatch", "InvalidAccessKey"}:
        return "Access Key ID oder Secret Access Key ist falsch."
    if code == "NoSuchBucket":
        return "Der Bucket wurde nicht gefunden. Prüfe den Bucket-Namen und die Account-ID."
    if code in {"AccessDenied", "Forbidden", "403"}:
        return "Zugriff verweigert: Der API-Token braucht Object Read & Write für diesen Bucket."
    name = error.__class__.__name__
    if name in {"EndpointConnectionError", "ConnectTimeoutError", "ConnectionClosedError"}:
        return "Keine Verbindung zu Cloudflare: Account-ID falsch oder keine Internetverbindung."
    return f"Cloudflare R2: {code or name}: {error}"


class CloudflareR2Provider(ShareProvider):
    """Publishes packages under ``share/<share_id>/`` of the user's bucket; packages expire after ``expiry_days``."""

    name = "cloudflare_r2"

    def __init__(self, credentials: dict[str, str], *, expiry_days: int = DEFAULT_EXPIRY_DAYS, client: Any | None = None) -> None:
        self.credentials = dict(credentials)
        self.bucket = self.credentials["bucket"]
        self.public_base_url = self.credentials["public_base_url"].rstrip("/")
        self.expiry_days = int(expiry_days)
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client(self.credentials)
        return self._client

    # urls / keys -------------------------------------------------------------------------------
    @staticmethod
    def prefix(share_id: str) -> str:
        return f"{PREFIX}{validate_share_id(share_id)}/"

    def public_url(self, share_id: str) -> str:
        return f"{self.public_base_url}/{self.prefix(share_id)}index.html"

    def _keys(self, share_id: str) -> list[str]:
        keys: list[str] = []
        token: str | None = None
        while True:
            params: dict[str, Any] = {"Bucket": self.bucket, "Prefix": self.prefix(share_id)}
            if token:
                params["ContinuationToken"] = token
            page = self.client.list_objects_v2(**params)
            keys.extend(item["Key"] for item in page.get("Contents", []))
            if not page.get("IsTruncated"):
                return keys
            token = page.get("NextContinuationToken")

    # interface ---------------------------------------------------------------------------------
    def upload(self, local_dir: Path, share_id: str, progress_cb: ProgressCallback | None = None) -> str:
        source = Path(local_dir)
        if not (source / "index.html").is_file():
            raise ProviderError("package has no index.html")
        prefix = self.prefix(share_id)
        files = sorted(p for p in source.rglob("*") if p.is_file())
        total = directory_size(source)
        done = 0
        if progress_cb:
            progress_cb(0, total)
        try:
            for path in files:
                key = prefix + path.relative_to(source).as_posix()
                base = done

                def callback(sent: int, base: int = base) -> None:
                    if progress_cb:
                        progress_cb(min(base + sent, total), total)

                self.client.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": content_type(path.name)},
                                        Callback=callback)
                done += path.stat().st_size
                if progress_cb:
                    progress_cb(done, total)
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(german_error(error)) from error
        return self.public_url(share_id)

    def delete(self, share_id: str) -> None:
        try:
            keys = self._keys(share_id)
            for start in range(0, len(keys), 1000):
                batch = keys[start:start + 1000]
                self.client.delete_objects(Bucket=self.bucket,
                                           Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True})
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(german_error(error)) from error

    def extend(self, share_id: str) -> datetime:
        """Restart the lifecycle age of every object by an in-place CopyObject (no re-upload).

        Bucket lifecycle rules count from an object's last-modified time, which CopyObject with
        ``MetadataDirective=REPLACE`` resets; this also works for whole-bucket rules created by hand. The expiry
        shown in the UI is therefore "last upload or extend + ``expiry_days``".
        """
        now = datetime.now(timezone.utc)
        try:
            keys = self._keys(share_id)
            if not keys:
                raise ProviderError("Das geteilte Paket existiert nicht mehr (abgelaufen?). Bitte neu erzeugen.")
            for key in keys:
                self.client.copy_object(
                    Bucket=self.bucket, Key=key, CopySource={"Bucket": self.bucket, "Key": key},
                    MetadataDirective="REPLACE", ContentType=content_type(key),
                    Metadata={"extended-at": now.strftime("%Y%m%dT%H%M%SZ")})
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(german_error(error)) from error
        return now + timedelta(days=self.expiry_days)

    @staticmethod
    def _rule_covers(rule: dict[str, Any], days: int) -> bool:
        """True for an enabled rule deleting objects after at most ``days`` that covers the prefix ``share/``."""
        if rule.get("Status") != "Enabled":
            return False
        expiration = rule.get("Expiration") or {}
        if not expiration.get("Days") or int(expiration["Days"]) > days:
            return False
        filt = rule.get("Filter") or {}
        if set(filt) - {"Prefix"}:  # tag / size filters do not cover everything
            return False
        prefix = str(filt.get("Prefix", rule.get("Prefix", "")) or "")
        return PREFIX.startswith(prefix)

    def ensure_lifecycle(self, days: int | None = None) -> str:
        """Best effort: find or create a rule deleting ``share/`` objects after ``days``.

        Returns ``"present"`` (a suitable rule exists), ``"set"`` (created, other rules kept), ``"unreadable"`` (the
        token may not even read lifecycle rules, typical for Object Read & Write: unknown whether the hand-made
        rule of guide step 5 exists) or ``"missing"`` (rules were readable, none matches, and none could be written).
        """
        days = int(days or self.expiry_days)
        try:
            try:
                rules = list(self.client.get_bucket_lifecycle_configuration(Bucket=self.bucket).get("Rules", []))
            except Exception as error:
                code = _error_code(error)
                if code in _DENIED_CODES:
                    return "unreadable"
                if code not in {"NoSuchLifecycleConfiguration", "NoSuchLifecycle"}:
                    raise
                rules = []
            if any(self._rule_covers(rule, days) for rule in rules):
                return "present"
            rules = [rule for rule in rules if rule.get("ID") != LIFECYCLE_RULE_ID]
            rules.append({"ID": LIFECYCLE_RULE_ID, "Status": "Enabled", "Filter": {"Prefix": PREFIX},
                          "Expiration": {"Days": days}})
            try:
                self.client.put_bucket_lifecycle_configuration(Bucket=self.bucket,
                                                               LifecycleConfiguration={"Rules": rules})
            except Exception as error:
                if _error_code(error) in _DENIED_CODES:
                    return "missing"
                raise
            return "set"
        except Exception as error:
            raise ProviderError(german_error(error)) from error

    def _fetch(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "SAMcloud"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read()

    def test_connection(self) -> str:
        """Upload, read through the public URL, and delete a tiny object; raises ``ProviderError`` in German."""
        token = secrets.token_hex(8)
        key = f"{PREFIX}_connection_test/{token}.txt"
        url = f"{self.public_base_url}/{key}"
        try:
            try:
                self.client.put_object(Bucket=self.bucket, Key=key, Body=token.encode(), ContentType="text/plain")
                body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
                if body != token.encode():
                    raise ProviderError("Der Bucket lieferte unerwartete Daten.")
            except ProviderError:
                raise
            except Exception as error:
                raise ProviderError(german_error(error)) from error
            try:
                public = self._fetch(url)
            except urllib.error.HTTPError as error:
                raise ProviderError(
                    f"Öffentlicher Zugriff nicht verfügbar (HTTP {error.code}): aktiviere r2.dev-Zugriff im Bucket "
                    "(Settings > Public access) und prüfe die öffentliche URL.") from error
            except (urllib.error.URLError, OSError) as error:
                raise ProviderError(f"Die öffentliche URL ist nicht erreichbar: {error}") from error
            if public != token.encode():
                raise ProviderError("Die öffentliche URL liefert nicht den Inhalt des Buckets: falsche URL?")
        finally:
            try:
                self.client.delete_object(Bucket=self.bucket, Key=key)
            except Exception as error:  # cleanup only
                LOGGER.info("connection test object not removed: %s", error)
        return "Verbindung erfolgreich."
