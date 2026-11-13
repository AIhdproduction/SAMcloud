"""Validated configuration for the orthophoto compute pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from samcloud.core.configuration import load_yaml
from samcloud.core.paths import CONFIG_DIRECTORY


@dataclass(frozen=True, slots=True)
class IndoorConfig:
    cut_below_ceiling: float = 0.15
    cut_fallback_above_floor: float = 2.0
    floor_margin: float = 0.10
    ceiling_codes: tuple[int, ...] = (101, 116)


@dataclass(frozen=True, slots=True)
class DsmConfig:
    cell_outdoor: float = 0.06
    cell_indoor: float = 0.02
    hole_fill_max_area_m2: float | None = 1.0
    hole_fill_max_area_m2_indoor: float | None = 0.25
    hole_fill_max_cells: int = 25
    surface_fill_indoor: bool = True
    surface_fill_close_m: float = 0.40
    surface_fill_max_hole_m2: float = 6.0
    surface_fill_method: str = "laplace"
    ground_codes: tuple[int, ...] = (2, 11, 71, 73)


@dataclass(frozen=True, slots=True)
class ZBufferConfig:
    voxel_outdoor: float = 0.03
    voxel_indoor: float = 0.015
    max_long_side: int = 1600
    rel_tol: float = 0.004
    abs_tol_voxels: float = 2.0
    min_filter_px: int = 2
    discontinuity_voxels: float = 4.0
    footprint_tolerance_scale: float = 1.0
    cache_mb: int = 4096


@dataclass(frozen=True, slots=True)
class BlendConfig:
    top_k: int = 3
    candidates: int = 6
    consistency_threshold: float = 45.0
    max_incidence_deg: float = 60.0
    max_incidence_deg_indoor: float = 82.0
    feather_px: int = 24
    weight_angle: float = 1.0
    weight_distance: float = 0.5
    weight_center: float = 0.3
    score_power: float = 2.0
    score_gap_sigma: float = 0.15
    nadir_soft_limit_deg: float = 45.0
    nadir_penalty: float = 0.35
    ray_check_enabled: bool = False
    ray_check_samples: int = 4
    ray_check_distance: float = 15.0
    ray_check_clearance: float = 0.15


@dataclass(frozen=True, slots=True)
class OrthoConfig:
    pixel_size_outdoor: float = 0.02
    pixel_size_indoor: float = 0.01
    max_side: int = 16000
    indoor: IndoorConfig = IndoorConfig()
    dsm: DsmConfig = DsmConfig()
    zbuffer: ZBufferConfig = ZBufferConfig()
    blend: BlendConfig = BlendConfig()
    tile_size: int = 2048
    jpeg_long_sides: tuple[int, ...] = (8000, 4096)
    jpeg_quality: int = 92
    affine_max_residual: float = 0.002
    max_model_cloud_distance: float = 0.10


def load_ortho_config(path: Path | None = None) -> OrthoConfig:
    """Read and validate a YAML orthophoto configuration file."""
    data = load_yaml(path or CONFIG_DIRECTORY / "ortho.yaml")
    indoor = dict(data.get("indoor", {}))
    dsm = dict(data.get("dsm", {}))
    zbuffer = dict(data.get("zbuffer", {}))
    blend = dict(data.get("blend", {}))
    ground_codes = tuple(int(value) for value in dsm.get("ground_codes", _ground_codes()))
    has_area = "hole_fill_max_area_m2" in dsm or "hole_fill_max_area_m2_indoor" in dsm
    has_legacy_only = "hole_fill_max_cells" in dsm and not has_area
    surface_fill_indoor = dsm.get("surface_fill_indoor", True)
    area_outdoor = None if has_legacy_only else float(dsm.get("hole_fill_max_area_m2", 1.0))
    area_indoor = None if has_legacy_only else float(dsm.get("hole_fill_max_area_m2_indoor", 0.25))
    result = OrthoConfig(
        pixel_size_outdoor=float(data.get("pixel_size_outdoor", 0.02)),
        pixel_size_indoor=float(data.get("pixel_size_indoor", 0.01)),
        max_side=int(data.get("max_side", 16000)),
        indoor=IndoorConfig(
            cut_below_ceiling=float(indoor.get("cut_below_ceiling", 0.15)),
            cut_fallback_above_floor=float(indoor.get("cut_fallback_above_floor", 2.0)),
            floor_margin=float(indoor.get("floor_margin", 0.10)),
            ceiling_codes=tuple(int(value) for value in indoor.get("ceiling_codes", (101, 116))),
        ),
        dsm=DsmConfig(
            cell_outdoor=float(dsm.get("cell_outdoor", 0.06)),
            cell_indoor=float(dsm.get("cell_indoor", 0.02)),
            hole_fill_max_area_m2=area_outdoor,
            hole_fill_max_area_m2_indoor=area_indoor,
            hole_fill_max_cells=int(dsm.get("hole_fill_max_cells", 25)),
            surface_fill_indoor=surface_fill_indoor,
            surface_fill_close_m=float(dsm.get("surface_fill_close_m", 0.40)),
            surface_fill_max_hole_m2=float(dsm.get("surface_fill_max_hole_m2", 6.0)),
            surface_fill_method=str(dsm.get("surface_fill_method", "laplace")),
            ground_codes=ground_codes,
        ),
        zbuffer=ZBufferConfig(
            voxel_outdoor=float(zbuffer.get("voxel_outdoor", 0.03)),
            voxel_indoor=float(zbuffer.get("voxel_indoor", 0.015)),
            max_long_side=int(zbuffer.get("max_long_side", 1600)),
            rel_tol=float(zbuffer.get("rel_tol", 0.004)),
            abs_tol_voxels=float(zbuffer.get("abs_tol_voxels", 2.0)),
            min_filter_px=int(zbuffer.get("min_filter_px", 2)),
            discontinuity_voxels=float(zbuffer.get("discontinuity_voxels", 4.0)),
            footprint_tolerance_scale=float(zbuffer.get("footprint_tolerance_scale", 1.0)),
            cache_mb=int(zbuffer.get("cache_mb", 4096)),
        ),
        blend=BlendConfig(
            top_k=int(blend.get("top_k", 3)),
            candidates=int(blend.get("candidates", 6)),
            consistency_threshold=float(blend.get("consistency_threshold", 45.0)),
            max_incidence_deg=float(blend.get("max_incidence_deg", 60.0)),
            max_incidence_deg_indoor=float(blend.get("max_incidence_deg_indoor", 82.0)),
            feather_px=int(blend.get("feather_px", 24)),
            weight_angle=float(blend.get("weight_angle", 1.0)),
            weight_distance=float(blend.get("weight_distance", 0.5)),
            weight_center=float(blend.get("weight_center", 0.3)),
            score_power=float(blend.get("score_power", 2.0)),
            score_gap_sigma=float(blend.get("score_gap_sigma", 0.15)),
            nadir_soft_limit_deg=float(blend.get("nadir_soft_limit_deg", 45.0)),
            nadir_penalty=float(blend.get("nadir_penalty", 0.35)),
            ray_check_enabled=bool(blend.get("ray_check_enabled", False)),
            ray_check_samples=int(blend.get("ray_check_samples", 4)),
            ray_check_distance=float(blend.get("ray_check_distance", 15.0)),
            ray_check_clearance=float(blend.get("ray_check_clearance", 0.15)),
        ),
        tile_size=int(data.get("tile_size", 2048)),
        jpeg_long_sides=tuple(int(value) for value in data.get("jpeg_long_sides", (8000, 4096))),
        jpeg_quality=int(data.get("jpeg_quality", 92)),
        affine_max_residual=float(data.get("affine_max_residual", 0.002)),
        max_model_cloud_distance=float(data.get("max_model_cloud_distance", 0.10)),
    )
    if (
        min(result.pixel_size_outdoor, result.pixel_size_indoor) <= 0
        or result.max_side < 1
        or result.tile_size < 1
        or result.blend.top_k < 1
        or result.blend.candidates < 1
        or result.blend.consistency_threshold < 0
        or not 0.0 <= result.blend.max_incidence_deg <= 90.0
        or not 0.0 <= result.blend.max_incidence_deg_indoor <= 90.0
        or result.blend.score_gap_sigma <= 0
        or result.zbuffer.max_long_side < 1
        or result.zbuffer.cache_mb < 1
        or result.zbuffer.rel_tol < 0
        or result.zbuffer.min_filter_px < 0
        or result.zbuffer.discontinuity_voxels < 0
        or result.zbuffer.footprint_tolerance_scale < 0
        or not 0.0 <= result.blend.nadir_soft_limit_deg <= 90.0
        or result.blend.nadir_penalty < 0
        or result.blend.ray_check_samples < 1
        or result.blend.ray_check_distance <= 0
        or result.blend.ray_check_clearance < 0
        or not 1 <= result.jpeg_quality <= 100
        or min(result.dsm.cell_outdoor, result.dsm.cell_indoor) <= 0
        or (result.dsm.hole_fill_max_area_m2 is not None and result.dsm.hole_fill_max_area_m2 < 0)
        or (result.dsm.hole_fill_max_area_m2_indoor is not None and result.dsm.hole_fill_max_area_m2_indoor < 0)
        or result.dsm.surface_fill_close_m <= 0
        or result.dsm.surface_fill_max_hole_m2 < 0
        or not isinstance(result.dsm.surface_fill_indoor, bool)
        or result.dsm.surface_fill_method not in {"laplace", "floor_plane"}
    ):
        raise ValueError("Invalid orthophoto configuration values")
    return result


def _ground_codes() -> tuple[int, ...]:
    """Read outdoor ground-like classes from the existing DXF layer configuration."""
    try:
        layers: dict[str, Any] = load_yaml(CONFIG_DIRECTORY / "dxf_layers.yaml")
        codes: set[int] = set()
        for layer in layers.get("layers", []):
            if layer.get("mode") == "outdoor" and str(layer.get("name", "")) in {"S-GRND", "S-PAVE", "S-WALK", "S-VEG"}:
                codes.update(int(value) for value in layer.get("codes", []))
        return tuple(sorted(codes)) or (2, 11, 71, 73)
    except (OSError, ValueError, TypeError):
        return (2, 11, 71, 73)
