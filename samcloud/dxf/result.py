"""Serializable neutral DXF plan result and browser vector payload."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import base64
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ezdxf.colors import aci2rgb

from samcloud.dxf.las_source import Frame

Entity = dict[str, Any]
LayerState = Literal["ok", "empty", "failed"]


@dataclass(frozen=True, slots=True)
class LayerStatus:
    name: str
    state: LayerState
    reason: str | None
    entity_count: int


@dataclass(frozen=True, slots=True)
class BackgroundImage:
    rgb_png: bytes | None
    class_png: bytes | None
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class DxfPlanResult:
    project_name: str
    level_name: str
    mode: Literal["indoor", "outdoor"]
    profile_name: str | None
    scale_factor: float
    frame: Frame
    settings_fingerprint: str
    layers: dict[str, list[Entity]]
    statuses: dict[str, LayerStatus]
    layer_colors: dict[str, int]
    background: BackgroundImage
    fingerprint_inputs: dict[str, Any] = field(default_factory=dict)
    layer_linetypes: dict[str, str] = field(default_factory=dict)
    layer_hatches: dict[str, bool] = field(default_factory=dict)


def result_to_dict(result: DxfPlanResult, *, include_images: bool = True) -> dict[str, Any]:
    """Convert a result to a JSON-safe payload suitable for cache persistence."""
    background: dict[str, Any] = {"bounds": result.background.bounds}
    if include_images:
        background.update({"rgb_png": _encode_bytes(result.background.rgb_png), "class_png": _encode_bytes(result.background.class_png)})
    return {"project_name": result.project_name, "level_name": result.level_name, "mode": result.mode, "profile_name": result.profile_name, "scale_factor": result.scale_factor, "frame": {"origin": result.frame.origin.tolist(), "scale_factor": result.frame.scale_factor, "rotation": result.frame.rotation.tolist()}, "settings_fingerprint": result.settings_fingerprint, "fingerprint_inputs": result.fingerprint_inputs, "layers": result.layers, "statuses": {name: asdict(status) for name, status in result.statuses.items()}, "layer_colors": result.layer_colors, "layer_linetypes": result.layer_linetypes, "layer_hatches": result.layer_hatches, "background": background}


def result_from_dict(payload: dict[str, Any], *, rgb_png: bytes | None = None, class_png: bytes | None = None) -> DxfPlanResult:
    """Rebuild a result from its JSON-safe representation."""
    frame_data = payload["frame"]
    frame = Frame(np.asarray(frame_data["origin"], dtype=np.float64), float(frame_data["scale_factor"]), np.asarray(frame_data["rotation"], dtype=np.float64))
    image = payload["background"]
    background = BackgroundImage(rgb_png if rgb_png is not None else _decode_bytes(image.get("rgb_png")), class_png if class_png is not None else _decode_bytes(image.get("class_png")), tuple(float(value) for value in image["bounds"]))
    return DxfPlanResult(str(payload["project_name"]), str(payload["level_name"]), payload["mode"], payload.get("profile_name"), float(payload["scale_factor"]), frame, str(payload["settings_fingerprint"]), {str(name): list(entities) for name, entities in payload["layers"].items()}, {str(name): LayerStatus(**status) for name, status in payload["statuses"].items()}, {str(name): int(color) for name, color in payload["layer_colors"].items()}, background, dict(payload.get("fingerprint_inputs", {})), {str(name): str(linetype) for name, linetype in payload.get("layer_linetypes", {}).items()}, {str(name): bool(hatch) for name, hatch in payload.get("layer_hatches", {}).items()})


def vector_payload(result: DxfPlanResult, *, grid_size: int = 64) -> dict[str, Any]:
    """Build a JSON-serializable browser payload with RGB colors and bounds."""
    payload_layers: dict[str, dict[str, Any]] = {}
    for name, entities in result.layers.items():
        bounds = _entity_bounds(entities)
        payload_layers[name] = {"color": aci_to_rgb(result.layer_colors[name]), "entities": entities, "bounds": bounds, "grid": _grid_index(entities, bounds, grid_size), "cell_size": _grid_cell_size(bounds, grid_size)}
    return {"project_name": result.project_name, "level": result.level_name, "mode": result.mode, "layers": payload_layers, "background_bounds": result.background.bounds}


def visible_cells(layer_payload: dict[str, Any], viewport: tuple[float, float, float, float]) -> list[str]:
    """Return the grid keys of one payload layer whose cells intersect viewport (min_x, min_y, max_x, max_y).

    This is the same culling rule the browser review canvas applies before drawing a layer.
    """
    bounds = layer_payload.get("bounds")
    cell_size = layer_payload.get("cell_size")
    grid = layer_payload.get("grid") or {}
    if bounds is None or not cell_size or not grid:
        return []
    left, bottom = float(bounds[0]), float(bounds[1])
    size = float(cell_size)
    view_left, view_bottom, view_right, view_top = (float(value) for value in viewport)
    if view_right < view_left or view_top < view_bottom:
        return []
    first_x, last_x = int((view_left - left) // size), int((view_right - left) // size)
    first_y, last_y = int((view_bottom - bottom) // size), int((view_top - bottom) // size)
    keys: list[str] = []
    for key in grid:
        x_text, y_text = key.split(":")
        x, y = int(x_text), int(y_text)
        if first_x <= x <= last_x and first_y <= y <= last_y:
            keys.append(key)
    return sorted(keys, key=lambda item: tuple(int(part) for part in item.split(":")))


def visible_entity_indices(layer_payload: dict[str, Any], viewport: tuple[float, float, float, float]) -> list[int]:
    """Return the sorted, de-duplicated entity indices registered in cells intersecting the viewport."""
    grid = layer_payload.get("grid") or {}
    indices: set[int] = set()
    for key in visible_cells(layer_payload, viewport):
        indices.update(int(index) for index in grid[key])
    return sorted(indices)


def aci_to_rgb(color: int) -> tuple[int, int, int]:
    """Map common AutoCAD ACI values to stable browser RGB colors."""
    value = aci2rgb(max(1, min(255, int(color))))
    return int(value.r), int(value.g), int(value.b)


def _entity_bounds(entities: list[Entity]) -> tuple[float, float, float, float] | None:
    values: list[tuple[float, float]] = []
    for entity in entities:
        for key in ("point", "start", "end", "center"):
            candidate = entity.get(key)
            if isinstance(candidate, list) and len(candidate) >= 2:
                values.append((float(candidate[0]), float(candidate[1])))
        if entity.get("type") == "circle" and isinstance(entity.get("center"), list):
            x, y = entity["center"][:2]
            radius = float(entity["radius"])
            values.extend([(float(x) - radius, float(y) - radius), (float(x) + radius, float(y) + radius)])
        for candidate in entity.get("points", []):
            if len(candidate) >= 2:
                values.append((float(candidate[0]), float(candidate[1])))
    if not values:
        return None
    array = np.asarray(values)
    return (float(array[:, 0].min()), float(array[:, 1].min()), float(array[:, 0].max()), float(array[:, 1].max()))


def _grid_cell_size(bounds: tuple[float, float, float, float] | None, grid_size: int) -> float | None:
    if bounds is None:
        return None
    left, bottom, right, top = bounds
    return max((right - left), (top - bottom), 1.0) / max(grid_size, 1)


def _grid_index(entities: list[Entity], bounds: tuple[float, float, float, float] | None, grid_size: int) -> dict[str, list[int]]:
    size = _grid_cell_size(bounds, grid_size)
    if bounds is None or size is None:
        return {}
    left, bottom, _right, _top = bounds
    grid: dict[str, list[int]] = {}
    for index, entity in enumerate(entities):
        candidate = _entity_bounds([entity])
        if candidate is None:
            continue
        start_x, start_y = int((candidate[0] - left) // size), int((candidate[1] - bottom) // size)
        end_x, end_y = int((candidate[2] - left) // size), int((candidate[3] - bottom) // size)
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                grid.setdefault(f"{x}:{y}", []).append(index)
    return grid


def _encode_bytes(value: bytes | None) -> str | None:
    return None if value is None else base64.b64encode(value).decode("ascii")


def _decode_bytes(value: object) -> bytes | None:
    return None if value is None else base64.b64decode(str(value))
