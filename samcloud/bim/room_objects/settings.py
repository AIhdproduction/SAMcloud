"""Settings of the room-object detection (``config/room_objects.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ExtentSettings:
    enabled: bool = True
    min_points: int = 80
    min_bottom_m: float = 0.25
    wall_gap_m: float = 0.15
    plan_face_error_m: float = 0.25
    wall_band_m: float = 0.06
    seen_band_m: float = 0.60
    seen_share: float = 0.50
    support_band_m: float = 0.15
    support_min_points: int = 8
    carrier_gap_m: float = 0.08
    min_protrusion_m: float = 0.07
    below_used_share: float = 0.30
    max_item_share: float = 3.0
    small_item_m2: float = 0.10
    discover_min_points: int = 150
    discover_min_bottom_m: float = 0.60


@dataclass(frozen=True, slots=True)
class ScreenSettings:
    enabled: bool = True
    min_points: int = 60
    link_m: float = 0.06
    plane_tolerance_m: float = 0.035
    curved_rms_m: float = 0.04
    merge_gap_m: float = 0.30
    max_tilt_deg: float = 30.0
    min_planar_share: float = 0.60
    min_term_share: float = 0.15
    width_m: tuple[float, float] = (0.22, 3.20)
    height_m: tuple[float, float] = (0.14, 1.90)
    aspect: tuple[float, float] = (0.5, 2.8)
    ultrawide_aspect: float = 3.8
    grow_m: float = 0.35
    complete_fill: float = 0.60
    grow_cell_m: float = 0.04
    thickness_m: float = 0.04
    wall_gap_m: float = 0.12
    stand_reach_m: float = 0.35
    body_row_share: float = 0.50
    small_screen_m: float = 0.85
    monitor_term_share: float = 0.05
    stand_gap_m: float = 0.25
    unlabelled_term_share: float = 0.08


@dataclass(frozen=True, slots=True)
class CeilingSettings:
    enabled: bool = True
    band_m: float = 0.80
    min_height_m: float = 1.50
    above_tolerance_m: float = 0.10
    attach_gap_m: float = 0.08
    pendant_gap_m: float = 0.50
    pendant_wall_m: float = 0.40
    pendant_min_points: int = 200
    neck_gap_m: float = 0.15
    link_m: float = 0.05
    min_points: int = 50
    min_area_m2: float = 0.003
    max_size_m: float = 3.0
    cell_m: float = 0.02
    density_share: float = 0.20
    flat_drop_m: float = 0.06
    flat_depth_m: float = 0.03
    vent_drop_m: float = 0.10
    luminous_level: float = 225.0
    glare_size_m: float = 0.35
    spot_core_share: float = 0.50
    spot_max_m: float = 0.22
    rail_piece_m: float = 0.12
    rail_aspect: float = 4.0
    rail_wall_m: float = 0.25
    curtain_window_m: float = 0.50
    curtain_reach_m: float = 1.50
    wall_touch_m: float = 0.06
    detector_size_m: float = 0.22


@dataclass(frozen=True, slots=True)
class ReflectionSettings:
    match_m: float = 0.30
    size_ratio: float = 0.50
    lateral_m: float = 1.80
    twin_point_share: float = 0.10
    twin_size_share: float = 0.20
    twin_wall_m: float = 1.00


@dataclass(frozen=True, slots=True)
class ScreenFrameSettings:
    enabled: bool = True
    min_conf: float = 0.5
    median_conf: float = 0.65
    quad_fill: float = 0.86
    min_edge_px: float = 10.0
    min_views: int = 3
    min_places: int = 3
    monitor_conf: float = 0.8
    monitor_share: float = 0.45
    place_m: float = 0.20
    min_parallax_deg: float = 8.0
    max_distance_m: float = 6.0
    ray_gap_m: float = 0.05
    corner_share: float = 0.035
    corner_px: float = 4.0
    hypotheses: int = 30
    rectangle_tolerance: float = 0.12
    level_share: float = 0.08
    max_tilt_deg: float = 20.0
    flat_m: float = 0.04
    width_m: tuple[float, float] = (0.22, 3.20)
    height_m: tuple[float, float] = (0.14, 1.90)
    aspect: tuple[float, float] = (0.5, 3.8)
    min_bottom_m: float = 0.25
    support_distance_m: float = 3.5
    support_cosine: float = 0.5
    support_share: float = 0.35
    same_size_share: float = 0.10


@dataclass(frozen=True, slots=True)
class RoomObjectSettings:
    enabled: bool = True
    depth_min_views: int = 2
    extent: ExtentSettings = field(default_factory=ExtentSettings)
    screens: ScreenSettings = field(default_factory=ScreenSettings)
    ceiling: CeilingSettings = field(default_factory=CeilingSettings)
    reflection: ReflectionSettings = field(default_factory=ReflectionSettings)
    screen_frames: ScreenFrameSettings = field(default_factory=ScreenFrameSettings)


def _section(cls: type, values: dict[str, Any] | None) -> Any:
    known = {item.name for item in fields(cls)}
    unknown = set(values or {}) - known
    if unknown:
        raise ValueError(f"room_objects.yaml: unknown keys {sorted(unknown)} for {cls.__name__}")
    return cls(**{key: tuple(value) if isinstance(value, list) else value for key, value in (values or {}).items()})


def load_room_object_settings(path: Path | str | None = None) -> RoomObjectSettings:
    """``config/room_objects.yaml``; built-in defaults fill whatever is missing."""
    if path is None:
        from samcloud.core.paths import CONFIG_DIRECTORY

        path = Path(CONFIG_DIRECTORY) / "room_objects.yaml"
    data = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}) if Path(path).is_file() else {}
    return RoomObjectSettings(
        enabled=bool(data.get("enabled", True)),
        depth_min_views=int(data.get("depth_min_views", 2)),
        extent=_section(ExtentSettings, data.get("extent")),
        screens=_section(ScreenSettings, data.get("screens")),
        ceiling=_section(CeilingSettings, data.get("ceiling")),
        reflection=_section(ReflectionSettings, data.get("reflection")),
        screen_frames=_section(ScreenFrameSettings, data.get("screen_frames")),
    )
