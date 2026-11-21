"""Outdoor category styles (layer name, colour) read from ``config/outdoor_style.yaml``.

One source of truth for the outdoor IFC (presentation layers and surface styles), the outdoor DXF
(layer colours) and the app's IFC viewer.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from samcloud.core.paths import CONFIG_DIRECTORY

STYLE_PATH = CONFIG_DIRECTORY / "outdoor_style.yaml"

# Categories that must exist in the style file (IFC and DXF).
REQUIRED_CATEGORIES: tuple[str, ...] = (
    "terrain", "road", "path", "sidewalk", "curb", "building_wall", "roof", "window", "door",
    "site_wall", "fence", "gutter", "tree", "shrub", "manhole", "marking",
    "contour_major", "contour_minor", "contour_label", "spot_elevation", "boundary",
)


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    styles: dict[str, dict[str, Any]] = {}
    for category, record in (raw.get("categories") or {}).items():
        rgb = tuple(int(value) for value in record["rgb"])
        if len(rgb) != 3 or any(not 0 <= value <= 255 for value in rgb):
            raise ValueError(f"outdoor style '{category}': rgb must be three values 0..255")
        styles[str(category)] = {
            "category": str(category),
            "label": str(record.get("label", category)),
            "layer": str(record["layer"]),
            "rgb": rgb,
            "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
            "aci": int(record.get("aci", 7)),
            "transparency": float(record.get("transparency", 0.0)),
            "ifc": bool(record.get("ifc", True)),
        }
    return styles


def load_outdoor_style(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Return ``{category: {label, layer, rgb, hex, aci, transparency, ifc}}`` (a fresh copy)."""
    return {key: dict(value) for key, value in _load(str(path or STYLE_PATH)).items()}


def category_style(category: str) -> dict[str, Any]:
    """Style of one category; raises ``KeyError`` for an unknown category."""
    return dict(_load(str(STYLE_PATH))[category])


def layer_styles() -> dict[str, dict[str, Any]]:
    """Styles keyed by layer name (the first category wins when a layer is shared)."""
    result: dict[str, dict[str, Any]] = {}
    for style in _load(str(STYLE_PATH)).values():
        result.setdefault(style["layer"], dict(style))
    return result
