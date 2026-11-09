"""Persistent DXF project state and mapping to plan settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from samcloud.dxf.alignment import AlignmentResult
from samcloud.dxf.config import DxfExportConfig, load_dxf_config
from samcloud.dxf.las_source import Frame
from samcloud.dxf.layers import LayerConfig, LayerDefinition, load_layer_config
from samcloud.dxf.levels import Level
from samcloud.dxf.pipeline import PlanSettings
from samcloud.dxf.scale import ScaleState

DEFAULT_VIEW: dict[str, Any] = {
    "side_axis": "x",
    "slab_center": None,
    "slab_thickness": None,
    "image_source": "rgb",
    "image_opacity": 1.0,
    "dxf_opacity": 1.0,
    "coordinates": "local",
    "wall_centerlines": True,
    "door_swing_arc": False,
}

DEFAULT_SCALE: dict[str, Any] = {
    "factor": 1.0,
    "source": "none",
    "status_text": "No scale adjustment applied.",
}


@dataclass
class DxfProjectState:
    """Serializable DXF settings and analysis results persisted per project."""

    mode_override: str | None = None
    profile_name: str | None = None
    scale: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SCALE))
    alignment: dict[str, Any] | None = None
    levels: list[dict[str, Any]] = field(default_factory=list)
    view: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_VIEW))
    overrides: dict[str, Any] = field(default_factory=lambda: {"cell_size": None, "layers": {}})

    def to_dict(self) -> dict[str, Any]:
        """Return a clean JSON-serializable dictionary."""
        return {
            "mode_override": self.mode_override,
            "profile_name": self.profile_name,
            "scale": dict(self.scale),
            "alignment": None if self.alignment is None else dict(self.alignment),
            "levels": [dict(level) for level in self.levels],
            "view": dict(self.view),
            "overrides": dict(self.overrides),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DxfProjectState:
        """Construct state from dictionary with backward-compatible defaults."""
        view = {**DEFAULT_VIEW, **data.get("view", {})}
        scale = {**DEFAULT_SCALE, **data.get("scale", {})}
        overrides = {"cell_size": None, "layers": {}, **data.get("overrides", {})}
        return cls(
            mode_override=data.get("mode_override"),
            profile_name=data.get("profile_name"),
            scale=scale,
            alignment=data.get("alignment"),
            levels=list(data.get("levels", [])),
            view=view,
            overrides=overrides,
        )


def state_path(project_dir: Path | str) -> Path:
    """Return the state.json location under products/dxf/."""
    return Path(project_dir) / "products" / "dxf" / "state.json"


def load_state(project_dir: Path | str) -> DxfProjectState:
    """Load DXF project state from disk. Never computes anything."""
    path = state_path(project_dir)
    if not path.is_file():
        return DxfProjectState()
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        return DxfProjectState.from_dict(content)
    except Exception:
        return DxfProjectState()


def save_state(project_dir: Path | str, state: DxfProjectState) -> Path:
    """Atomically persist DXF state to state.json using a temporary file and replace."""
    target = state_path(project_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_dict(), indent=2).encode("utf-8")
    descriptor, temp_path = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=target.parent)
    try:
        with open(descriptor, "wb") as file:
            file.write(payload)
        for attempt in range(50):
            try:
                os.replace(temp_path, target)
                break
            except PermissionError:
                # A concurrent reader (GUI poll vs. job subprocess) briefly holds the file on Windows.
                if attempt == 49:
                    raise
                time.sleep(0.02)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return target


def plan_settings_for(
    state: DxfProjectState,
    level: Level | dict[str, Any] | None = None,
    *,
    base_frame: Frame | None = None,
    origin: np.ndarray | None = None,
    project_directory: Path | None = None,
    config: DxfExportConfig | None = None,
    layer_config: LayerConfig | None = None,
) -> PlanSettings:
    """Construct PlanSettings from state scale, alignment, view and layer overrides."""
    factor = float(state.scale.get("factor", 1.0))
    if state.alignment is not None and "rotation" in state.alignment:
        rotation = np.asarray(state.alignment["rotation"], dtype=np.float64)
    else:
        rotation = np.eye(3, dtype=np.float64)

    if base_frame is not None:
        frame_origin = np.asarray(base_frame.origin, dtype=np.float64)
    elif origin is not None:
        frame_origin = np.asarray(origin, dtype=np.float64)
    else:
        frame_origin = np.zeros(3, dtype=np.float64)

    frame = Frame(origin=frame_origin, scale_factor=factor, rotation=rotation)

    cell_size = state.overrides.get("cell_size")
    base_layers = layer_config or load_layer_config()
    layers_overrides = state.overrides.get("layers", {})
    if layers_overrides:
        from dataclasses import replace

        modified_layers: list[LayerDefinition] = []
        for layer in base_layers.layers:
            if layer.name in layers_overrides:
                override = layers_overrides[layer.name]
                z_band = override.get("z_band", layer.z_band)
                if isinstance(z_band, list):
                    z_band = tuple(z_band)
                mode = override.get("mode", layer.mode)
                modified_layers.append(replace(layer, z_band=z_band, mode=mode))
            else:
                modified_layers.append(layer)
        effective_layers = LayerConfig(tuple(modified_layers))
    else:
        effective_layers = base_layers

    wall_centerlines = bool(state.view.get("wall_centerlines", True))
    door_swing_arc = bool(state.view.get("door_swing_arc", False))

    return PlanSettings(
        frame=frame,
        config=config,
        layer_config=effective_layers,
        cell_size=cell_size,
        project_directory=Path(project_directory) if project_directory else None,
        wall_centerlines=wall_centerlines,
        door_swing_arc=door_swing_arc,
    )
