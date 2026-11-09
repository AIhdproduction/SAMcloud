"""Page controller for DXF preview, inspection, configuration and export without GUI dependencies."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from samcloud.classification.profiles import ProfileInfo
from samcloud.core.configuration import load_yaml
from samcloud.dxf.config import load_dxf_config
from samcloud.dxf.job_runner import DxfJobRunner
from samcloud.dxf.jobs import export_manifest_path
from samcloud.dxf.las_source import Frame, read_las_header_origin, read_las_header_profile
from samcloud.dxf.layers import load_layer_config
from samcloud.dxf.levels import Level
from samcloud.dxf.overlay import composite, render_dxf_rgba
from samcloud.dxf.pipeline import is_stale as pipeline_is_stale, load_cached_result
from samcloud.dxf.result import DxfPlanResult, vector_payload
from samcloud.dxf.scale import numeric_scale, two_point_scale
from samcloud.dxf.state import DxfProjectState, load_state, plan_settings_for, save_state
from samcloud.dxf.writer import sanitize_dxf_name

MODEL_PROFILES: tuple[str, ...] = ("sam3_outdoor", "sam3_indoor", "flurseg_v1", "legacy_v1")
"""Profiles offered when a LAS file carries no provenance."""

LAYER_MODES: tuple[str, ...] = ("cut_band", "footprint", "top_surface", "ground_surface")
"""Layer extraction modes that can be chosen per layer in the review settings."""

REVIEW_BACKGROUND: tuple[int, int, int] = (30, 34, 38)
"""Neutral backdrop shared by the review canvas and saved snapshots (same compositing math)."""

DEFAULT_CUT_BAND: tuple[float, float] = (0.9, 1.3)
"""Height band assigned when a layer is switched to cut_band without a band of its own."""


def _level_from_dict(item: dict[str, Any]) -> Level:
    """Construct a Level dataclass instance from a dictionary stored in project state."""
    return Level(
        id=str(item["id"]),
        name=str(item["name"]),
        floor_z=float(item["floor_z"]),
        ceiling_z=float(item["ceiling_z"]) if item.get("ceiling_z") is not None else None,
        point_count=int(item.get("point_count", 0)),
        confidence=float(item.get("confidence", 1.0)),
        low_confidence=bool(item.get("low_confidence", False)),
    )


DXF_NO_PROJECT_HINT = "Open a project first"
"""Shown on disabled DXF entry points while no project is open."""

DXF_NO_LAS_HINT = "Export a classified LAS first"
"""Shown on disabled DXF entry points while the project has no usable LAS file."""


def classified_las_path(project_dir: Path | str) -> Path | None:
    """Return the project's classified LAS (products/classified/classified.las first) using existence checks only."""
    classified_dir = Path(project_dir) / "products" / "classified"
    classified_default = classified_dir / "classified.las"
    if classified_default.is_file():
        return classified_default
    if classified_dir.is_dir():
        candidates = sorted(candidate for candidate in classified_dir.glob("*.las") if candidate.is_file())
        if candidates:
            return candidates[0]
    return None


def chosen_las_path(project_dir: Path | str) -> Path | None:
    """Return the LAS file previously chosen on the DXF page (stored in the DXF state), if any."""
    stored = load_state(project_dir).view.get("las_path")
    return Path(str(stored)).resolve() if stored else None


def dxf_unavailable_reason(project_dir: Path | str | None) -> str | None:
    """Explain why DXF export cannot be used yet, or return None when a project and a LAS file are available.

    Cheap by design: only file existence and the small DXF state.json are checked, no LAS data is read.
    """
    if project_dir is None:
        return DXF_NO_PROJECT_HINT
    chosen = chosen_las_path(project_dir)
    if chosen is not None and chosen.is_file():
        return None
    if classified_las_path(project_dir) is not None:
        return None
    return DXF_NO_LAS_HINT


class DxfPageController:
    """Manages project state, caching, freshness inspection, and subprocess jobs for DXF export."""

    def __init__(
        self,
        project_dir: Path | str,
        runner: DxfJobRunner | None = None,
        las_path: Path | str | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.runner = runner if runner is not None else DxfJobRunner(self.project_dir)
        self._las_path = Path(las_path).resolve() if las_path is not None else None
        self._state: DxfProjectState | None = None
        self._header_cache: dict[tuple[str, int, int], tuple[ProfileInfo | None, np.ndarray]] = {}
        self._result_memo: dict[str, tuple[tuple[int, int], DxfPlanResult]] = {}
        self.refresh_state()
        if self._las_path is None and self.state.view.get("las_path"):
            self._las_path = Path(str(self.state.view["las_path"])).resolve()

    def _load_las_info_cache(self) -> dict[str, Any] | None:
        """Read cached products/dxf/las_info.json without computing anything.

        The cache is ignored when it was written for a different LAS file than the active one.
        """
        path = self.project_dir / "products" / "dxf" / "las_info.json"
        if not path.is_file():
            return None
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        recorded = info.get("las_path")
        current = self.las_path
        if recorded and (current is None or Path(str(recorded)).resolve() != current.resolve()):
            return None
        return info

    def _las_header(self) -> tuple[ProfileInfo | None, np.ndarray] | None:
        """Return (provenance profile, local origin) of the active LAS from its header, memoised by file stat."""
        current = self.las_path
        if current is None or not current.is_file():
            return None
        stat = current.stat()
        key = (str(current.resolve()), stat.st_mtime_ns, stat.st_size)
        if key not in self._header_cache:
            try:
                self._header_cache[key] = (read_las_header_profile(current), read_las_header_origin(current))
            except Exception:
                return None
        return self._header_cache[key]

    def _las_profile(self) -> ProfileInfo | None:
        """Provenance of the active LAS (sidecar or VLR) without reading its points."""
        header = self._las_header()
        return None if header is None else header[0]

    # -------------------------------------------------------------------------
    # Properties for header and system status
    # -------------------------------------------------------------------------

    @property
    def project_name(self) -> str:
        """Return the current project folder name."""
        return self.project_dir.name

    @property
    def state(self) -> DxfProjectState:
        """Return the current in-memory DXF project state."""
        if self._state is None:
            self._state = load_state(self.project_dir)
        return self._state

    def refresh_state(self) -> DxfProjectState:
        """Reload project state from products/dxf/state.json."""
        self._state = load_state(self.project_dir)
        return self._state

    @property
    def las_path(self) -> Path | None:
        """Return the active LAS file path, defaulting to standard project outputs."""
        if self._las_path is not None:
            return self._las_path
        classified = classified_las_path(self.project_dir)
        if classified is not None:
            return classified
        inputs_dir = self.project_dir / "inputs"
        if inputs_dir.is_dir():
            candidates = sorted(inputs_dir.glob("*.las"))
            if candidates:
                return candidates[0]
        return None

    @las_path.setter
    def las_path(self, path: Path | str | None) -> None:
        self._las_path = Path(path).resolve() if path is not None else None

    def set_las_path(self, path: Path | str | None) -> None:
        """Select an external or alternative LAS file (None = project default) and persist the choice."""
        if path is not None:
            candidate = Path(path)
            if candidate.suffix.lower() != ".las":
                raise ValueError("Only uncompressed .las files are supported (no LAZ).")
            if not candidate.is_file():
                raise ValueError(f"LAS file was not found: {candidate}")
        self.las_path = path
        if path is None:
            self.state.view.pop("las_path", None)
        else:
            self.state.view["las_path"] = str(Path(path).resolve())
        save_state(self.project_dir, self.state)

    @property
    def las_is_custom(self) -> bool:
        """True when an explicitly chosen LAS file replaces the project default."""
        return self._las_path is not None

    @property
    def profile(self) -> str | None:
        """Return the effective model profile name, or None if unspecified."""
        if self.state.profile_name:
            return self.state.profile_name
        cached_info = self._load_las_info_cache()
        if cached_info and cached_info.get("profile"):
            return str(cached_info["profile"])
        las_profile = self._las_profile()
        if las_profile is not None:
            return las_profile.profile_name
        project_file = self.project_dir / "project.samcloud.yml"
        if project_file.is_file():
            try:
                data = load_yaml(project_file)
                profile_name = data.get("settings", {}).get("classification", {}).get("model_profile")
                if profile_name:
                    return str(profile_name)
            except Exception:
                pass
        return None

    @property
    def profile_required(self) -> bool:
        """True if the LAS source lacks profile provenance and user selection is mandatory."""
        return self.profile is None

    @property
    def mode(self) -> str:
        """Return project mode: 'indoor' or 'outdoor'."""
        if self.state.mode_override:
            return self.state.mode_override
        current_profile = self.profile
        if current_profile == "sam3_indoor":
            return "indoor"
        if current_profile in ("sam3_outdoor", "flurseg_v1"):
            return "outdoor"
        cached_info = self._load_las_info_cache()
        if cached_info and cached_info.get("mode"):
            return str(cached_info["mode"])
        las_profile = self._las_profile()
        if las_profile is not None:
            return str(las_profile.mode)
        project_file = self.project_dir / "project.samcloud.yml"
        if project_file.is_file():
            try:
                data = load_yaml(project_file)
                class_set = data.get("settings", {}).get("classification", {}).get("class_set")
                if class_set in ("indoor", "outdoor"):
                    return str(class_set)
            except Exception:
                pass
        return "outdoor"

    @property
    def blocked_message(self) -> str | None:
        """Return an explanation if current configuration is blocked, otherwise None."""
        if self.profile == "flurseg_v1" and self.mode == "indoor":
            return "The FlurSeg v1 profile supports outdoor projects only. Please switch mode to outdoor or select another profile."
        return None

    @property
    def scale_status(self) -> str:
        """Human-readable description of current scale factor and source."""
        return str(self.state.scale.get("status_text", "No scale adjustment applied."))

    @property
    def scale_factor(self) -> float:
        """Numerical scale factor applied to local coordinates."""
        return float(self.state.scale.get("factor", 1.0))

    @property
    def scale_source(self) -> str:
        """Source of scale factor: 'none', 'numeric', or 'two_point'."""
        return str(self.state.scale.get("source", "none"))

    @property
    def alignment(self) -> dict[str, Any] | None:
        """Return stored gravity alignment result, or None if unaligned."""
        return self.state.alignment

    @property
    def plausibility_warnings(self) -> list[str]:
        """Plausibility warnings generated during voxel preview generation."""
        info = self._load_las_info_cache()
        if info and "warnings" in info:
            return list(info["warnings"])
        return []

    @property
    def has_preview(self) -> bool:
        """Whether products/dxf/preview.npz is available on disk."""
        return (self.project_dir / "products" / "dxf" / "preview.npz").is_file()

    # -------------------------------------------------------------------------
    # State mutators (all cheap, strictly persist to state.json, never compute)
    # -------------------------------------------------------------------------

    def set_profile(self, profile_name: str) -> None:
        """Set active model profile and persist to project state."""
        self.state.profile_name = str(profile_name)
        save_state(self.project_dir, self.state)
        project_file = self.project_dir / "project.samcloud.yml"
        if project_file.is_file():
            try:
                project_data = load_yaml(project_file)
                project_data.setdefault("settings", {}).setdefault("classification", {})["model_profile"] = str(profile_name)
                project_file.write_text(yaml.dump(project_data, sort_keys=False), encoding="utf-8")
            except Exception:
                pass

    def set_mode_override(self, mode: str | None) -> None:
        """Set project mode override ('indoor', 'outdoor', or None for auto)."""
        if mode is not None and mode not in ("indoor", "outdoor"):
            raise ValueError(f"Invalid mode override '{mode}'. Must be 'indoor', 'outdoor', or None.")
        self.state.mode_override = mode
        save_state(self.project_dir, self.state)

    def set_scale_factor(self, factor: float) -> None:
        """Apply a manual numeric scale factor."""
        state_obj = numeric_scale(factor)
        self._rescale_levels(state_obj.factor)
        self.state.scale = {
            "factor": state_obj.factor,
            "source": state_obj.source,
            "status_text": state_obj.status_text,
        }
        save_state(self.project_dir, self.state)

    def _rescale_levels(self, new_factor: float) -> None:
        """Keep stored level heights consistent: they live in the scaled local frame."""
        old_factor = self.scale_factor
        if old_factor <= 0 or np.isclose(old_factor, new_factor, rtol=0.0, atol=1e-12):
            return
        ratio = float(new_factor) / old_factor
        for lvl in self.state.levels:
            lvl["floor_z"] = float(lvl["floor_z"]) * ratio
            if lvl.get("ceiling_z") is not None:
                lvl["ceiling_z"] = float(lvl["ceiling_z"]) * ratio

    def measure_scale(self, p1: Any, p2: Any, real_distance: float, points_scale: float | None = None) -> float:
        """Derive metric scale factor from two picked coordinates and measured distance.

        Picked points live in a frame already scaled by ``points_scale`` (default: the currently
        applied factor, which the side view uses). They are converted back to unscaled source units
        first, so a second measurement does not compound the previous factor.
        """
        pt1 = np.asarray(p1, dtype=np.float64)
        pt2 = np.asarray(p2, dtype=np.float64)
        if pt1.shape == (2,):
            pt1 = np.array([pt1[0], pt1[1], 0.0], dtype=np.float64)
        if pt2.shape == (2,):
            pt2 = np.array([pt2[0], pt2[1], 0.0], dtype=np.float64)
        applied = float(points_scale) if points_scale is not None else self.scale_factor
        if not np.isfinite(applied) or applied <= 0:
            raise ValueError("The scale of the picked points must be a finite positive number")
        state_obj = two_point_scale(pt1 / applied, pt2 / applied, float(real_distance))
        self._rescale_levels(state_obj.factor)
        self.state.scale = {
            "factor": state_obj.factor,
            "source": state_obj.source,
            "status_text": state_obj.status_text,
        }
        save_state(self.project_dir, self.state)
        return state_obj.factor

    def toggle_level_checked(self, level_id: str, checked: bool | None = None) -> bool:
        """Toggle or set level export selection flag."""
        for lvl in self.state.levels:
            if lvl["id"] == level_id:
                new_val = not lvl.get("checked", True) if checked is None else bool(checked)
                lvl["checked"] = new_val
                save_state(self.project_dir, self.state)
                return new_val
        raise ValueError(f"Level '{level_id}' not found in state levels")

    def set_level_checked(self, level_id: str, checked: bool) -> None:
        """Explicitly set level export selection flag."""
        self.toggle_level_checked(level_id, checked=checked)

    def rename_level(self, level_id: str, new_name: str) -> None:
        """Rename an indoor level in state."""
        for lvl in self.state.levels:
            if lvl["id"] == level_id:
                lvl["name"] = str(new_name).strip()
                save_state(self.project_dir, self.state)
                return
        raise ValueError(f"Level '{level_id}' not found in state levels")

    def move_level(self, level_id: str, floor_z: float | None = None, ceiling_z: float | None = None) -> None:
        """Adjust floor and/or ceiling elevations for an indoor level."""
        for lvl in self.state.levels:
            if lvl["id"] == level_id:
                if floor_z is not None:
                    lvl["floor_z"] = float(floor_z)
                if ceiling_z is not None:
                    lvl["ceiling_z"] = float(ceiling_z)
                save_state(self.project_dir, self.state)
                return
        raise ValueError(f"Level '{level_id}' not found in state levels")

    def add_level(self, floor_z: float, ceiling_z: float | None = None, name: str | None = None) -> dict[str, Any]:
        """Add a new indoor level to state with unique ID."""
        existing_ids = {lvl["id"] for lvl in self.state.levels}
        idx = len(self.state.levels) + 1
        while f"level_{idx}" in existing_ids:
            idx += 1
        new_id = f"level_{idx}"
        level_name = str(name) if name is not None else str(idx)
        new_level = {
            "id": new_id,
            "name": level_name,
            "floor_z": float(floor_z),
            "ceiling_z": float(ceiling_z) if ceiling_z is not None else None,
            "checked": True,
        }
        self.state.levels.append(new_level)
        save_state(self.project_dir, self.state)
        return dict(new_level)

    def add_level_at(self, floor_z: float) -> dict[str, Any]:
        """Add a level at a clicked height; its ceiling defaults to the next floor above (or +2.6 m)."""
        floor = float(floor_z)
        above = sorted(float(lvl["floor_z"]) for lvl in self.state.levels if float(lvl["floor_z"]) > floor + 0.5)
        ceiling = above[0] if above and above[0] - floor <= 6.0 else floor + 2.6
        return self.add_level(floor, ceiling)

    @property
    def align_walls(self) -> bool:
        """Whether the stored alignment also rotates walls onto the X/Y axes (yaw)."""
        alignment = self.state.alignment
        return True if alignment is None else bool(alignment.get("align_walls", True))

    def set_align_walls(self, enabled: bool) -> None:
        """Apply or remove the wall yaw of the stored alignment without recomputing it."""
        alignment = self.state.alignment
        if alignment is None or "rotation" not in alignment:
            raise ValueError("Run 'Align to gravity' before choosing the wall alignment")
        full = np.asarray(alignment.get("full_rotation", alignment["rotation"]), dtype=np.float64)
        yaw = np.radians(float(alignment.get("yaw_degrees", 0.0)))
        undo_yaw = np.array(((np.cos(-yaw), -np.sin(-yaw), 0.0), (np.sin(-yaw), np.cos(-yaw), 0.0), (0.0, 0.0, 1.0)))
        rotation = full if enabled else undo_yaw @ full
        alignment["full_rotation"] = full.tolist()
        alignment["rotation"] = rotation.tolist()
        alignment["align_walls"] = bool(enabled)
        save_state(self.project_dir, self.state)

    def delete_level(self, level_id: str) -> None:
        """Remove a level from state."""
        orig_len = len(self.state.levels)
        self.state.levels = [lvl for lvl in self.state.levels if lvl["id"] != level_id]
        if len(self.state.levels) == orig_len:
            raise ValueError(f"Level '{level_id}' not found in state levels")
        save_state(self.project_dir, self.state)

    def set_view_option(self, key: str, value: Any) -> None:
        """Set a view configuration option in state."""
        self.state.view[key] = value
        save_state(self.project_dir, self.state)

    def set_cell_size(self, cell_size: float | None) -> None:
        """Override vector plan rasterization cell size."""
        self.state.overrides["cell_size"] = float(cell_size) if cell_size is not None else None
        save_state(self.project_dir, self.state)

    def set_layer_override(
        self,
        layer_name: str,
        z_band: Any = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Override layer z_band, mode, or other parameters."""
        layers = self.state.overrides.setdefault("layers", {})
        entry = layers.setdefault(layer_name, {})
        if z_band is not None:
            entry["z_band"] = list(z_band)
        if mode is not None:
            entry["mode"] = str(mode)
        for k, v in kwargs.items():
            entry[k] = v
        save_state(self.project_dir, self.state)

    # -------------------------------------------------------------------------
    # Cached results & freshness queries
    # -------------------------------------------------------------------------

    def available_levels(self) -> list[dict[str, Any]]:
        """Return copies of all levels stored in state."""
        return [dict(lvl) for lvl in self.state.levels]

    def checked_levels(self) -> list[dict[str, Any]]:
        """Return levels selected for export."""
        return [dict(lvl) for lvl in self.state.levels if lvl.get("checked", True)]

    def cache_target_id(self, level_id: str | None = None) -> str:
        """Return the cache folder name for a level (indoor) or the whole site (outdoor)."""
        if self.mode == "indoor":
            if level_id is not None:
                target_id = str(level_id)
            elif self.state.levels:
                checked = next((lvl for lvl in self.state.levels if lvl.get("checked", True)), self.state.levels[0])
                target_id = str(checked["id"])
            else:
                target_id = "site"
        else:
            target_id = "site" if level_id is None else str(level_id)
        if not target_id or any(not (char.isalnum() or char in "_-") for char in target_id):
            raise ValueError(f"Invalid level identifier '{target_id}'")
        return target_id

    def cache_dir(self, level_id: str | None = None) -> Path:
        """Return products/dxf/<level id or site>/ (not created)."""
        return self.project_dir / "products" / "dxf" / self.cache_target_id(level_id)

    def cache_version(self, level_id: str | None = None) -> str:
        """Short token that changes whenever the cached result on disk changes (for browser cache busting)."""
        result_file = self.cache_dir(level_id) / "result.json"
        if not result_file.is_file():
            return "none"
        stat = result_file.stat()
        return f"{stat.st_mtime_ns:x}-{stat.st_size:x}"

    def cached_result(self, level_id: str | None = None) -> DxfPlanResult | None:
        """Load cached vector plan from disk without recomputing (memoised by file stat)."""
        cache_dir = self.cache_dir(level_id)
        result_file = cache_dir / "result.json"
        if not result_file.is_file():
            return None
        stat = result_file.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        memo = self._result_memo.get(str(cache_dir))
        if memo is not None and memo[0] == key:
            return memo[1]
        try:
            result = load_cached_result(cache_dir)
        except Exception:
            return None
        self._result_memo[str(cache_dir)] = (key, result)
        return result

    def cached_payload(self, level_id: str | None = None, grid_size: int | None = None) -> dict[str, Any] | None:
        """Return browser vector payload for cached result, or None if cache missing.

        The culling grid defaults to geometry.vector_grid_size from config/dxf_export.yaml.
        """
        result = self.cached_result(level_id)
        if result is None:
            return None
        size = int(grid_size) if grid_size is not None else int(load_dxf_config().geometry.vector_grid_size)
        return vector_payload(result, grid_size=size)

    def is_stale(self, level_id: str | None = None) -> bool:
        """Check whether cached vector plan differs from current settings."""
        cached = self.cached_result(level_id)
        if cached is None:
            return False

        if self.mode != cached.mode:
            return True

        current_profile = self.profile
        if current_profile is not None and cached.profile_name is not None and current_profile != cached.profile_name:
            return True

        cached_las = cached.fingerprint_inputs.get("las") if cached.fingerprint_inputs else None
        current_las = self.las_path
        if cached_las and current_las is not None and Path(str(cached_las[0])).resolve() != current_las.resolve():
            return True

        factor = float(self.state.scale.get("factor", 1.0))
        if not np.isclose(factor, cached.scale_factor, atol=1e-6):
            return True

        level_obj: Level | None = None
        if self.mode == "indoor":
            if level_id is not None:
                matched = next((lvl for lvl in self.state.levels if lvl["id"] == level_id), None)
                if matched is not None:
                    level_obj = _level_from_dict(matched)
            elif self.state.levels:
                matched = next((lvl for lvl in self.state.levels if lvl.get("checked", True)), self.state.levels[0])
                level_obj = _level_from_dict(matched)

            if level_obj is not None and cached.level_name != level_obj.name:
                return True

        header = self._las_header()
        origin = header[1] if header is not None else np.asarray(cached.frame.origin, dtype=np.float64)
        # Generation uses the LAS frame origin; without it every georeferenced cache would look stale.
        settings = plan_settings_for(self.state, level_obj, base_frame=Frame(origin), project_directory=self.project_dir)
        # Generation records the LAS provenance profile, so the fingerprint is compared against that.
        return pipeline_is_stale(cached, settings, level=level_obj, profile_name=None)

    def export_enabled(self, level_id: str | None = None) -> bool:
        """True only if cached plan exists, is fresh, and configuration is unblocked."""
        if self.blocked_message is not None:
            return False
        if self.profile_required:
            return False
        cached = self.cached_result(level_id)
        if cached is None:
            return False
        if self.is_stale(level_id):
            return False
        return True

    # -------------------------------------------------------------------------
    # Subprocess jobs (heavy work only through DxfJobRunner.start)
    # -------------------------------------------------------------------------

    def start_job(self, action: str, level_id: str | None = None) -> Any:
        """Launch background subprocess job via the runner."""
        if self.blocked_message is not None:
            raise ValueError(self.blocked_message)
        if self.profile_required:
            raise ValueError("A model profile must be selected before starting a job")
        las = self.las_path
        if las is None or not Path(las).is_file():
            raise ValueError(f"No valid LAS file found to execute '{action}'")
        return self.runner.start(action, las=las, level=level_id, mode=self.mode)

    def cancel_job(self) -> None:
        """Signal background job to cancel and terminate process if needed."""
        self.runner.cancel()

    def job_status(self) -> dict[str, Any]:
        """Poll and return current subprocess job status."""
        return self.runner.status()

    def is_running(self) -> bool:
        """Check whether a DXF background job is currently running."""
        return self.runner.is_running()

    def log_tail(self, n: int = 50) -> list[str]:
        """Return the latest lines from products/dxf/job.log."""
        return self.runner.log_tail(n)

    # -------------------------------------------------------------------------
    # Review helpers (cheap reads of cached data and settings, no generation)
    # -------------------------------------------------------------------------

    def level_label(self, level_id: str | None) -> str:
        """Display name of a level id, or 'Site' for outdoor/None."""
        if level_id is None:
            return "Site"
        matched = next((lvl for lvl in self.state.levels if lvl["id"] == level_id), None)
        return str(matched["name"]) if matched is not None else str(level_id)

    def review_targets(self) -> list[str | None]:
        """Level ids offered in the DXF review: checked levels (indoor) or [None] for the outdoor site."""
        if self.mode == "indoor":
            return [str(lvl["id"]) for lvl in self.checked_levels()]
        return [None]

    def stale_levels(self) -> list[str]:
        """Display names of review targets whose cached DXF preview no longer matches the settings."""
        return [self.level_label(level_id) for level_id in self.review_targets() if self.is_stale(level_id)]

    def all_exports_enabled(self) -> bool:
        """True when every review target has a fresh cached preview, so a batch export can run."""
        targets = self.review_targets()
        return bool(targets) and all(self.export_enabled(level_id) for level_id in targets)

    def background_path(self, level_id: str | None, source: str) -> Path | None:
        """Return the cached orthographic background PNG for 'rgb' or 'classes', if present."""
        if source not in ("rgb", "classes"):
            raise ValueError("Background source must be 'rgb' or 'classes'")
        path = self.cache_dir(level_id) / ("background_rgb.png" if source == "rgb" else "background_classes.png")
        return path if path.is_file() else None

    def layer_settings(self) -> list[dict[str, Any]]:
        """Effective per-layer mode and z band for the current project mode (defaults plus project overrides)."""
        overrides = self.state.overrides.get("layers", {}) or {}
        rows: list[dict[str, Any]] = []
        for layer in load_layer_config().for_project_mode(self.mode):  # type: ignore[arg-type]
            entry = overrides.get(layer.name, {})
            z_band = entry.get("z_band", list(layer.z_band) if layer.z_band is not None else None)
            rows.append(
                {
                    "name": layer.name,
                    "mode": str(entry.get("mode", layer.mode)),
                    "z_band": None if z_band is None else [float(z_band[0]), float(z_band[1])],
                    "default_mode": layer.mode,
                    "default_z_band": None if layer.z_band is None else [float(layer.z_band[0]), float(layer.z_band[1])],
                    "overridden": bool(entry),
                }
            )
        return rows

    def set_layer_mode(self, layer_name: str, mode: str) -> None:
        """Override a layer's extraction mode; cut_band without a band gets DEFAULT_CUT_BAND."""
        if mode not in LAYER_MODES:
            raise ValueError(f"Unsupported layer mode '{mode}'")
        current = next((row for row in self.layer_settings() if row["name"] == layer_name), None)
        if current is None:
            raise ValueError(f"Unknown layer '{layer_name}'")
        z_band = DEFAULT_CUT_BAND if mode == "cut_band" and current["z_band"] is None else None
        self.set_layer_override(layer_name, z_band=z_band, mode=mode)

    def set_layer_z_band(self, layer_name: str, lower: float, upper: float) -> None:
        """Override a layer's height band in metres above the level floor."""
        low, high = float(lower), float(upper)
        if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
            raise ValueError("The z band needs a lower value below the upper value")
        self.set_layer_override(layer_name, z_band=(low, high))

    def reset_layer_override(self, layer_name: str) -> None:
        """Drop a project-level layer override so the global dxf_layers.yaml values apply again."""
        layers = self.state.overrides.setdefault("layers", {})
        if layers.pop(layer_name, None) is not None:
            save_state(self.project_dir, self.state)

    def last_export(self) -> dict[str, Any] | None:
        """Summary of the last successful export job (files and audit result), if any."""
        path = export_manifest_path(self.project_dir)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def snapshot_directory(self) -> Path:
        """Folder for review snapshots: products/dxf/snapshots/."""
        return self.project_dir / "products" / "dxf" / "snapshots"

    def save_snapshot(
        self,
        level_id: str | None,
        image_source: str = "rgb",
        image_opacity: float = 1.0,
        dxf_opacity: float = 1.0,
        hidden_layers: list[str] | tuple[str, ...] = (),
    ) -> Path:
        """Write a PNG of the cached review (background + DXF) with the same compositing as the canvas."""
        result = self.cached_result(level_id)
        if result is None:
            raise ValueError("No cached DXF preview exists for this level. Generate it first.")
        png = result.background.rgb_png if image_source == "rgb" else result.background.class_png
        png = png or result.background.rgb_png or result.background.class_png
        image: np.ndarray | None = None
        if png is not None:
            decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is not None:
                image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        if image is None:
            min_x, min_y, max_x, max_y = result.background.bounds
            width = 1600
            height = max(1, int(round(width * max(1e-3, max_y - min_y) / max(1e-3, max_x - min_x))))
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:] = REVIEW_BACKGROUND
            image_opacity = 0.0
        hidden = set(hidden_layers)
        visible = replace(result, layers={name: entities for name, entities in result.layers.items() if name not in hidden})
        height, width = image.shape[:2]
        overlay_rgba = render_dxf_rgba(visible, size=(width, height))
        merged = composite(image, overlay_rgba, image_opacity, dxf_opacity, background=REVIEW_BACKGROUND)
        directory = self.snapshot_directory()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target = directory / f"{sanitize_dxf_name(self.level_label(level_id))}_{stamp}.png"
        success, encoded = cv2.imencode(".png", cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("PNG encoding of the snapshot failed")
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(target)
        return target
