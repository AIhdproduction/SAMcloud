"""Settings of the furniture reconstruction (``config/furniture_recon.yaml``, metric, never project specific)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_WINDOWS: dict[str, tuple[float, float | None]] = {
    "table": (0.0, 1.30), "chair": (0.0, 1.40), "sofa": (0.0, 1.40), "bed": (0.0, 1.60), "kitchen": (0.0, 1.25),
    "box": (0.0, None), "mesh": (0.0, None), "toilet": (0.0, 1.30),
}


@dataclass(frozen=True, slots=True)
class ReconSettings:
    enabled: bool = True
    classes: tuple[int, ...] = (17, 18, 19)
    footprint_margin_m: float = 0.03
    voxel_m: float = 0.01
    max_points: int = 40000
    outlier_k: int = 10
    outlier_std_ratio: float = 2.5
    run_gap_m: float = 0.12
    tall_run_gap_m: float = 0.5
    min_points: int = 60
    min_xy_coverage: float = 0.15
    max_coverage_drop: float = 0.15
    max_mean_factor: float = 1.5
    coverage_distance_m: float = 0.03
    mesh_ring_vertices: int = 24
    mesh_slice_m: float = 0.05
    mesh_max_triangles: int = 3000
    worktop_thickness_m: float = 0.04
    table_top_max_thickness_m: float = 0.08
    leg_prior_size_m: float = 0.045
    leg_prior_inset_m: float = 0.05
    chair_seat_prior_m: float = 0.46
    grow_m: float = 0.40
    max_grow_ratio: float = 1.6
    grow_extent_margin_m: float = 0.10
    grow_max_volume_factor: float = 1.25
    grow_min_coverage_gain: float = 0.05
    bed_trim_gap_m: float = 0.12
    bed_surface_share: float = 0.6
    discover_instances: bool = True
    windows: dict[str, tuple[float, float | None]] = field(default_factory=lambda: dict(_DEFAULT_WINDOWS))

    def window(self, family: str, ceiling_m: float) -> tuple[float, float]:
        low, high = self.windows.get(family, self.windows.get("box", (0.0, None)))
        return float(low), float(ceiling_m if high is None else min(float(high), ceiling_m))


def load_recon_settings(path: Path | str | None = None) -> ReconSettings:
    """Settings from ``config/furniture_recon.yaml`` (missing keys keep the defaults)."""
    if path is None:
        from samcloud.core.paths import CONFIG_DIRECTORY

        path = Path(CONFIG_DIRECTORY) / "furniture_recon.yaml"
    raw: dict[str, Any] = {}
    if Path(path).is_file():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    names = {f.name for f in fields(ReconSettings)}
    values: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in names:
            continue
        if key == "classes":
            values[key] = tuple(int(item) for item in value)
        elif key == "windows":
            windows = dict(_DEFAULT_WINDOWS)
            for name, pair in (value or {}).items():
                windows[str(name)] = (float(pair[0]), None if pair[1] is None else float(pair[1]))
            values[key] = windows
        else:
            values[key] = value
    import os

    if os.environ.get("SAMCLOUD_FURNITURE_RECON", "").strip().lower() in ("0", "false", "off", "no"):
        values["enabled"] = False  # kill switch (A/B runs on the same tree)
    return ReconSettings(**values)
