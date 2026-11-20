"""Upload providers for shared point clouds; the provider is selected in ``config/share.yaml``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from samcloud.share.providers.base import ProgressCallback, ProviderError, ShareProvider
from samcloud.share.providers.local import LocalProvider

__all__ = ["LocalProvider", "ProgressCallback", "ProviderError", "ShareProvider", "get_provider"]


def get_provider(config: dict[str, Any], project_dir: Path | str | None = None, name: str | None = None) -> ShareProvider:
    """Build the provider named ``name`` (default ``config["provider"]``) from its ``providers.<name>`` settings."""
    name = str(name or os.environ.get("SAMCLOUD_SHARE_PROVIDER") or config.get("provider") or "cloudflare_r2")
    settings = (config.get("providers") or {}).get(name) or {}
    if name == "local":
        directory = settings.get("directory")
        if not directory:
            if project_dir is None:
                raise ProviderError("local provider needs a directory or a project")
            directory = Path(project_dir) / "products" / "share" / "site"
        return LocalProvider(
            directory,
            serve=bool(settings.get("serve", False)),
            host=str(settings.get("host", "127.0.0.1")),
            port=int(settings.get("port", 0) or 0),
        )
    if name == "cloudflare_r2":
        from samcloud.share.credentials import load_credentials
        from samcloud.share.providers.cloudflare_r2 import CloudflareR2Provider

        credentials = load_credentials()
        if credentials is None:
            raise ProviderError("Cloudflare-Zugangsdaten fehlen. Bitte zuerst das Konto einrichten.")
        return CloudflareR2Provider(credentials, expiry_days=int(config.get("expiry_days") or 15))
    raise ProviderError(f"unknown share provider: {name}")
