"""Configuration of the 3D opening refinement (``config/bim_openings.yaml``).

Every threshold of the detection rules lives in the YAML file; the code only reads it through the
attribute view :class:`Settings`, so a typo in a key raises instead of silently using a default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import yaml

CONFIG_FILENAME = "bim_openings.yaml"


class Settings:
    """Read-only attribute access over a nested mapping (unknown keys raise ``AttributeError``)."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name not in data:
            raise AttributeError(f"bim_openings.yaml has no setting '{name}'")
        value = data[name]
        return Settings(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - guard
        raise AttributeError("Settings are read-only")

    def __iter__(self) -> Iterator[str]:
        return iter(object.__getattribute__(self, "_data"))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(object.__getattribute__(self, "_data")))


def default_config_path() -> Path:
    from samcloud.core.paths import CONFIG_DIRECTORY

    return Path(CONFIG_DIRECTORY) / CONFIG_FILENAME


def load_openings_config(path: Path | str | None = None, overrides: dict[str, dict[str, Any]] | None = None) -> Settings:
    """Load the YAML thresholds; ``overrides`` ({section: {key: value}}) patch single values (tests)."""
    source = Path(path) if path is not None else default_config_path()
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    for section, values in (overrides or {}).items():
        data.setdefault(section, {}).update(values)
    return Settings(data)


def config_hash(settings: Settings) -> str:
    """Stable hash of every threshold (part of the refinement fingerprint)."""
    payload = json.dumps(settings.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
