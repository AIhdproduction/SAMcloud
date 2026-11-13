"""Layer definitions for deterministic DXF plan generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from samcloud.classification.profiles import load_cyclone_codes
from samcloud.core.paths import CONFIG_DIRECTORY

VALID_DXF_LINEWEIGHTS: tuple[int, ...] = (
    0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80, 90, 100, 106, 120, 140, 158, 200, 211
)

ProjectMode = Literal["indoor", "outdoor"]
LayerMode = Literal["cut_band", "footprint", "top_surface", "ground_surface", "ignore"]
GeometryType = Literal["wall", "opening", "polygon", "line", "furniture", "skeleton", "circle", "tree", "point", "ignore"]


@dataclass(frozen=True, slots=True)
class LayerDefinition:
    project_mode: ProjectMode
    mode: LayerMode
    codes: tuple[int, ...]
    name: str
    color: int
    linetype: str
    geometry: GeometryType
    z_band: tuple[float, float] | None
    min_area: float
    simplify_tolerance: float
    hatch: bool
    true_color: tuple[int, int, int] | None = None
    lineweight: int | None = None
    hatch_pattern: str | None = None
    hatch_scale: float | None = None


@dataclass(frozen=True, slots=True)
class LayerConfig:
    layers: tuple[LayerDefinition, ...]

    def for_project_mode(self, project_mode: ProjectMode, *, include_ignored: bool = False) -> tuple[LayerDefinition, ...]:
        return tuple(layer for layer in self.layers if layer.project_mode == project_mode and (include_ignored or layer.mode != "ignore"))

    def for_mode(self, project_mode: ProjectMode, *, include_ignored: bool = False) -> tuple[LayerDefinition, ...]:
        """Compatibility alias for selecting definitions by project mode."""
        return self.for_project_mode(project_mode, include_ignored=include_ignored)

    def by_name(self, name: str) -> LayerDefinition:
        return next(layer for layer in self.layers if layer.name == name)


def load_layer_config(path: Path | None = None) -> LayerConfig:
    """Load and validate class-to-layer definitions without performing export work."""
    source = CONFIG_DIRECTORY / "dxf_layers.yaml" if path is None else Path(path)
    if not source.is_file():
        raise ValueError(f"DXF layer configuration was not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    records = raw.get("layers") if isinstance(raw, dict) else None
    if not isinstance(records, list):
        raise ValueError("DXF layer configuration must contain a 'layers' list")
    registry = load_cyclone_codes()
    layers = tuple(_parse_layer(record, index, registry) for index, record in enumerate(records, start=1))
    names = [layer.name for layer in layers if layer.name != "IGNORE"]
    if len(names) != len(set(names)):
        raise ValueError("DXF layer names must be unique across modes")
    return LayerConfig(layers)


def _parse_layer(record: object, index: int, registry: dict[int, dict[str, object]]) -> LayerDefinition:
    if not isinstance(record, dict):
        raise ValueError(f"DXF layer entry {index} must be a mapping")
    required = {"mode", "codes", "name", "color", "linetype", "geometry", "min_area", "simplify_tolerance", "hatch"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"DXF layer entry {index} is missing: {', '.join(sorted(missing))}")
    project_mode = record.get("project_mode")
    mode = record["mode"]
    geometry = record["geometry"]
    if project_mode is None and mode in {"indoor", "outdoor"}:
        project_mode = mode
        mode = _legacy_layer_mode(record)
    if project_mode not in {"indoor", "outdoor"}:
        raise ValueError(f"DXF layer entry {index} has invalid project_mode '{project_mode}'")
    if mode not in {"cut_band", "footprint", "top_surface", "ground_surface", "ignore"}:
        raise ValueError(f"DXF layer entry {index} has invalid mode '{mode}'")
    valid_geometry = {"wall", "opening", "polygon", "line", "furniture", "skeleton", "circle", "tree", "point", "ignore"}
    if geometry not in valid_geometry:
        raise ValueError(f"DXF layer entry {index} has invalid geometry '{geometry}'")
    codes = tuple(int(code) for code in record["codes"])
    if not codes:
        raise ValueError(f"DXF layer entry {index} must define at least one code")
    unknown = sorted(set(codes) - set(registry))
    if unknown:
        raise ValueError(f"DXF layer entry {index} uses unknown Cyclone code(s): {unknown}")
    band = record.get("z_band")
    if band is not None:
        if not isinstance(band, list) or len(band) != 2 or float(band[0]) >= float(band[1]):
            raise ValueError(f"DXF layer entry {index} has an invalid z_band")
        z_band: tuple[float, float] | None = (float(band[0]), float(band[1]))
    else:
        z_band = None
    color = int(record["color"])
    if not 1 <= color <= 255:
        raise ValueError(f"DXF layer entry {index} has an invalid ACI color")
    min_area, tolerance = float(record["min_area"]), float(record["simplify_tolerance"])
    if min_area < 0 or tolerance < 0:
        raise ValueError(f"DXF layer entry {index} has a negative area or tolerance")
    if mode == "cut_band" and z_band is None:
        raise ValueError(f"DXF layer entry {index} mode 'cut_band' requires z_band")

    raw_true_color = record.get("true_color")
    if raw_true_color is not None:
        if not isinstance(raw_true_color, (list, tuple)) or len(raw_true_color) != 3:
            raise ValueError(f"DXF layer entry {index} has invalid true_color '{raw_true_color}'")
        try:
            tc_values = tuple(int(c) for c in raw_true_color)
            if not all(0 <= c <= 255 for c in tc_values):
                raise ValueError
            true_color: tuple[int, int, int] | None = (tc_values[0], tc_values[1], tc_values[2])
        except (ValueError, TypeError):
            raise ValueError(f"DXF layer entry {index} has invalid true_color components")
    else:
        true_color = None

    raw_lineweight = record.get("lineweight")
    if raw_lineweight is not None:
        try:
            lw = int(raw_lineweight)
        except (ValueError, TypeError):
            raise ValueError(f"DXF layer entry {index} has invalid lineweight '{raw_lineweight}'")
        if lw not in VALID_DXF_LINEWEIGHTS:
            raise ValueError(f"DXF layer entry {index} lineweight {lw} is not a valid DXF lineweight")
        lineweight: int | None = lw
    else:
        lineweight = None

    raw_hatch_pattern = record.get("hatch_pattern")
    hatch_pattern = str(raw_hatch_pattern) if raw_hatch_pattern is not None else None

    raw_hatch_scale = record.get("hatch_scale")
    if raw_hatch_scale is not None:
        try:
            hs = float(raw_hatch_scale)
            if hs <= 0:
                raise ValueError
            hatch_scale: float | None = hs
        except (ValueError, TypeError):
            raise ValueError(f"DXF layer entry {index} has invalid hatch_scale '{raw_hatch_scale}'")
    else:
        hatch_scale = None

    return LayerDefinition(
        project_mode,
        mode,
        codes,
        str(record["name"]),
        color,
        str(record["linetype"]),
        geometry,
        z_band,
        min_area,
        tolerance,
        bool(record["hatch"]),
        true_color,
        lineweight,
        hatch_pattern,
        hatch_scale,
    )


def _legacy_layer_mode(record: dict[object, object]) -> LayerMode:
    """Map pre-C2 configuration records while retaining a clear typed schema."""
    geometry = record.get("geometry")
    if geometry == "ignore":
        return "ignore"
    if record.get("z_band") is not None:
        return "cut_band"
    if record.get("name") in {"S-GRND", "S-PAVE", "S-WALK", "S-CURB"}:
        return "ground_surface"
    if record.get("name") == "S-BLDG":
        return "top_surface"
    return "footprint"
