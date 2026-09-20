"""Read global YAML configuration files."""

from pathlib import Path
from typing import Any

import yaml

from samcloud.core.paths import CONFIG_DIRECTORY


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and return an empty mapping for an empty file."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return data


def load_global_config(name: str) -> dict[str, Any]:
    """Load a named configuration file from the central config directory."""
    return load_yaml(CONFIG_DIRECTORY / name)
