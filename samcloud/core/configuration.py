"""Read global YAML configuration files."""

from pathlib import Path
from typing import Any

import yaml
from pyproj import CRS
from pyproj.exceptions import CRSError

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


def load_dxf_export_config() -> dict[str, Any]:
    """Load DXF export defaults, including automatic project-mode detection."""
    return load_global_config("dxf_export.yaml")


def load_export_crs_presets() -> dict[str, dict[str, Any]]:
    """Load and validate ordered LAS export CRS presets."""
    configuration = load_global_config("export_crs.yml")
    presets = configuration.get("presets")
    if not isinstance(presets, dict):
        raise ValueError("Export CRS configuration must contain a 'presets' mapping")

    validated: dict[str, dict[str, Any]] = {}
    for key, preset in presets.items():
        entry = str(key)
        if not isinstance(preset, dict):
            raise ValueError(f"Invalid export CRS preset '{entry}': entry must be a mapping")
        label = preset.get("label")
        target = preset.get("target")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid export CRS preset '{entry}': label is required")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Invalid export CRS preset '{entry}': target is required")
        if target.strip().upper() != "LOCAL":
            try:
                CRS.from_user_input(target)
            except CRSError as error:
                raise ValueError(
                    f"Invalid export CRS preset '{entry}': target {target!r} is not a valid PROJ CRS"
                ) from error
        required_grids = preset.get("required_grids", [])
        if not isinstance(required_grids, list) or not all(
            isinstance(grid, str) and grid.strip() for grid in required_grids
        ):
            raise ValueError(
                f"Invalid export CRS preset '{entry}': required_grids must be a list of filenames"
            )
        validated[entry] = {
            **preset,
            "label": label,
            "target": target,
            "required_grids": required_grids,
        }
    return validated
