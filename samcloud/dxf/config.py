"""Validated typed configuration for DXF export processing."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from samcloud.core.paths import CONFIG_DIRECTORY


@dataclass(frozen=True, slots=True)
class LoadingConfig:
    chunk_size: int = 1_000_000


@dataclass(frozen=True, slots=True)
class PreviewConfig:
    max_points: int = 5_000_000
    voxel_size: float | None = None


@dataclass(frozen=True, slots=True)
class PlausibilityConfig:
    indoor_min_extent: float = 2.0
    indoor_max_extent: float = 500.0
    outdoor_min_extent: float = 2.0
    outdoor_max_extent: float = 20_000.0


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    ransac_iterations: int = 600
    ransac_distance: float = 0.03
    min_floor_points: int = 100
    normal_sample_size: int = 20_000
    normal_neighbors: int = 16
    align_wall_yaw: bool = True


@dataclass(frozen=True, slots=True)
class LevelsConfig:
    bin_size: float = 0.05
    gaussian_sigma: float = 1.0
    min_point_count: int = 30
    min_prominence: float = 10.0
    floor_point_tolerance: float = 0.05
    min_floor_to_ceiling: float = 2.0
    max_floor_to_ceiling: float = 6.0
    stair_cluster_eps: float = 0.35
    stair_cluster_min_samples: int = 8


@dataclass(frozen=True, slots=True)
class ModeDetectionConfig:
    indoor_share: float = 0.15


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    indoor_cell_size: float = 0.02
    outdoor_cell_size: float = 0.05
    closing_kernel: int = 3
    simplify_tolerance: float = 0.03
    wall_direction_tolerance_degrees: float = 5.0
    wall_min_thickness: float = 0.08
    wall_merge_gap: float = 0.15
    wall_corner_distance: float = 0.2
    wall_max_thickness: float = 0.5
    wall_min_length: float = 0.5
    wall_opening_bridge: float = 3.0
    opening_max_distance: float = 0.5
    opening_cluster_eps: float = 0.35
    skeleton_min_length: float = 0.2
    indoor_default_height: float = 6.0
    vector_grid_size: int = 64
    fill_under_canopy: bool = False


@dataclass(frozen=True, slots=True)
class DxfExportConfig:
    loading: LoadingConfig = LoadingConfig()
    preview: PreviewConfig = PreviewConfig()
    plausibility: PlausibilityConfig = PlausibilityConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    levels: LevelsConfig = LevelsConfig()
    mode_detection: ModeDetectionConfig = ModeDetectionConfig()
    geometry: GeometryConfig = GeometryConfig()


def load_dxf_config(path: Path | None = None) -> DxfExportConfig:
    """Load DXF export configuration, using documented defaults for missing sections."""
    source = CONFIG_DIRECTORY / "dxf_export.yaml" if path is None else Path(path)
    if not source.is_file():
        if path is None:
            return DxfExportConfig()
        raise ValueError(f"DXF export configuration was not found: {source}")
    with source.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError("DXF export configuration must contain a mapping")
    return DxfExportConfig(
        loading=_section(LoadingConfig, raw, "loading"),
        preview=_section(PreviewConfig, raw, "preview"),
        plausibility=_section(PlausibilityConfig, raw, "plausibility"),
        alignment=_section(AlignmentConfig, raw, "alignment"),
        levels=_section(LevelsConfig, raw, "levels"),
        mode_detection=_section(ModeDetectionConfig, raw, "mode_detection"),
        geometry=_section(GeometryConfig, raw, "geometry"),
    )


def _section[T](section_type: type[T], raw: dict[str, Any], name: str) -> T:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"DXF export configuration section '{name}' must be a mapping")
    allowed = set(section_type.__dataclass_fields__)
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown DXF export configuration key(s) in '{name}': {', '.join(sorted(unknown))}")
    try:
        result = section_type(**value)
    except TypeError as error:
        raise ValueError(f"Invalid DXF export configuration section '{name}': {error}") from error
    _validate_section(name, result)
    return result


def _validate_section(name: str, value: object) -> None:
    if name == "loading" and value.chunk_size < 1:
        raise ValueError("loading.chunk_size must be at least 1")
    if name == "preview":
        if value.max_points < 1:
            raise ValueError("preview.max_points must be at least 1")
        if value.voxel_size is not None and value.voxel_size <= 0:
            raise ValueError("preview.voxel_size must be positive when set")
    if name == "plausibility":
        for field in fields(value):
            key, number = field.name, getattr(value, field.name)
            if number <= 0:
                raise ValueError(f"plausibility.{key} must be positive")
        if value.indoor_min_extent >= value.indoor_max_extent or value.outdoor_min_extent >= value.outdoor_max_extent:
            raise ValueError("plausibility minimum extent must be smaller than maximum extent")
    if name == "alignment":
        if value.ransac_iterations < 1 or value.min_floor_points < 3 or value.normal_sample_size < 3 or value.normal_neighbors < 3:
            raise ValueError("alignment counts must support plane estimation")
        if value.ransac_distance <= 0:
            raise ValueError("alignment.ransac_distance must be positive")
    if name == "levels":
        if value.bin_size <= 0 or value.gaussian_sigma < 0 or value.min_point_count < 1 or value.min_prominence < 0:
            raise ValueError("levels histogram settings are invalid")
        if value.min_floor_to_ceiling <= 0 or value.min_floor_to_ceiling >= value.max_floor_to_ceiling:
            raise ValueError("levels floor-to-ceiling bounds are invalid")
        if value.floor_point_tolerance <= 0:
            raise ValueError("levels.floor_point_tolerance must be positive")
        if value.stair_cluster_eps <= 0 or value.stair_cluster_min_samples < 1:
            raise ValueError("levels stair clustering settings are invalid")
    if name == "mode_detection" and not 0.0 <= value.indoor_share <= 1.0:
        raise ValueError("mode_detection.indoor_share must be between 0 and 1")
    if name == "geometry":
        if value.indoor_cell_size <= 0 or value.outdoor_cell_size <= 0:
            raise ValueError("geometry cell sizes must be positive")
        if value.closing_kernel < 1 or value.closing_kernel % 2 == 0:
            raise ValueError("geometry.closing_kernel must be a positive odd number")
        if value.simplify_tolerance < 0 or not 0 < value.wall_direction_tolerance_degrees <= 90:
            raise ValueError("geometry simplification or wall direction tolerance is invalid")
        if value.wall_min_thickness <= 0 or value.wall_merge_gap < 0 or value.wall_corner_distance < 0 or value.wall_max_thickness < value.wall_min_thickness or value.wall_min_length <= 0 or value.wall_opening_bridge < 0:
            raise ValueError("geometry wall settings are invalid")
        if value.opening_max_distance <= 0 or value.opening_cluster_eps <= 0 or value.skeleton_min_length < 0:
            raise ValueError("geometry opening or skeleton settings are invalid")
        if value.indoor_default_height <= 0 or value.vector_grid_size < 1:
            raise ValueError("geometry background or grid settings are invalid")
