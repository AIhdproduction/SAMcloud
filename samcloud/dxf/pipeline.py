"""Pure LAS-to-plan generation and explicit on-disk result cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import cv2
import numpy as np

from samcloud.dxf.cancel import CancelToken, ProgressCallback, check_cancel, report_progress
from samcloud.dxf.config import DxfExportConfig, load_dxf_config
from samcloud.dxf.geometry import bridge_collinear, circles_from_clusters, furniture_inserts, opening_entities, opening_specs, points_from_cluster, polygons_from_points, skeleton_lines, tree_entities, wall_lines, wall_polygons
from samcloud.dxf.las_source import Frame, iter_las_chunks, read_las_header_profile, read_las_info
from samcloud.dxf.layers import LayerConfig, LayerDefinition, load_layer_config
from samcloud.dxf.levels import Level
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus, result_from_dict, result_to_dict


CEILING_CODES = (101, 116)
"""Ceiling classes hidden from the indoor top-view background."""


@dataclass(frozen=True, slots=True)
class PlanSettings:
    frame: Frame | None = None
    config: DxfExportConfig | None = None
    layer_config: LayerConfig | None = None
    cell_size: float | None = None
    fill_under_canopy: bool | None = None
    project_directory: Path | None = None
    wall_centerlines: bool = True
    door_swing_arc: bool = False


def generate_plan(las_path: Path, mode: Literal["indoor", "outdoor"], level: Level | None, settings: PlanSettings | Mapping[str, Any] | None = None, progress: ProgressCallback | None = None, cancel: CancelToken | None = None) -> DxfPlanResult:
    """Generate a GUI-free vector plan. Per-layer failures are isolated."""
    source = Path(las_path)
    chosen = _settings(settings)
    config = chosen.config or load_dxf_config()
    layers = chosen.layer_config or load_layer_config()
    info = read_las_info(source, config=config, cancel=cancel)
    frame = chosen.frame or info.frame
    cell_size = chosen.cell_size or (config.geometry.indoor_cell_size if mode == "indoor" else config.geometry.outdoor_cell_size)
    accumulated, background = _stream_local_points(
        source,
        frame,
        info.bounds_min,
        info.bounds_max,
        config,
        mode,
        level,
        cell_size,
        cancel,
        progress,
    )
    level_name = "site" if level is None else level.name
    fingerprint_inputs = _fingerprint_inputs(source, mode, level, chosen, cell_size, layers, info.profile.profile_name if info.profile else None, config)
    fingerprint = _hash_fingerprint_inputs(fingerprint_inputs)
    output: dict[str, list[dict[str, Any]]] = {}
    statuses: dict[str, LayerStatus] = {}
    colors: dict[str, int] = {}
    linetypes: dict[str, str] = {}
    hatches: dict[str, bool] = {}
    active_layers = layers.for_project_mode(mode)
    for index, definition in enumerate(active_layers):
        check_cancel(cancel)
        colors[definition.name] = definition.color
        linetypes[definition.name] = definition.linetype
        hatches[definition.name] = definition.hatch
        points, class_values = accumulated.points_for(definition.codes)
        if len(points) == 0:
            statuses[definition.name] = LayerStatus(definition.name, "empty", _empty_reason(definition, info.profile, info.class_histogram), 0)
            output[definition.name] = []
            continue
        points, class_values = _mode_points(points, class_values, definition, mode, level, cell_size)
        if len(points) == 0:
            statuses[definition.name] = LayerStatus(definition.name, "empty", f"class {definition.codes[0]} has no points in the selected elevation band", 0)
            output[definition.name] = []
            continue
        try:
            entities = _layer_entities(definition, points, class_values, cell_size, config, chosen)
            output[definition.name] = entities
            statuses[definition.name] = LayerStatus(definition.name, "ok" if entities else "empty", None if entities else "no geometry survived filtering", dxf_entity_count(entities))
        except Exception as error:
            output[definition.name] = []
            statuses[definition.name] = LayerStatus(definition.name, "failed", str(error), 0)
        report_progress(progress, (index + 1) / max(len(active_layers), 1), f"Generating {definition.name}")
    _add_openings(output, statuses, active_layers, accumulated, mode, level, cell_size, config, chosen)
    if "A-WALL" in statuses:
        status = statuses["A-WALL"]
        statuses["A-WALL"] = LayerStatus("A-WALL", "ok" if output.get("A-WALL") else "empty", status.reason, dxf_entity_count(output.get("A-WALL", [])))
    if chosen.fill_under_canopy if chosen.fill_under_canopy is not None else config.geometry.fill_under_canopy:
        _fill_ground_holes(output, active_layers)
        for definition in active_layers:
            if definition.mode == "ground_surface" and definition.name in statuses:
                previous_status = statuses[definition.name]
                statuses[definition.name] = LayerStatus(previous_status.name, previous_status.state, previous_status.reason, dxf_entity_count(output.get(definition.name, [])))
    report_progress(progress, 1.0, "DXF plan generation complete")
    return DxfPlanResult(source.stem, level_name, mode, info.profile.profile_name if info.profile else None, frame.scale_factor, frame, fingerprint, output, statuses, colors, background, fingerprint_inputs, linetypes, hatches)


def settings_fingerprint(las_path: Path, mode: str, level: Level | None, settings: PlanSettings, cell_size: float, layers: LayerConfig) -> str:
    """Hash every input which changes generated plan geometry."""
    profile = read_las_info(las_path, config=settings.config or load_dxf_config()).profile
    config = settings.config or load_dxf_config()
    return _hash_fingerprint_inputs(_fingerprint_inputs(las_path, mode, level, settings, cell_size, layers, profile.profile_name if profile else None, config))


def _fingerprint_inputs(las_path: Path, mode: str, level: Level | None, settings: PlanSettings, cell_size: float, layers: LayerConfig, profile_name: str | None, config: DxfExportConfig) -> dict[str, Any]:
    source = Path(las_path)
    stat = source.stat()
    frame = settings.frame
    return {"las": [str(source.resolve()), stat.st_mtime_ns, stat.st_size], "profile": profile_name, "mode": mode, "level": None if level is None else [level.id, level.name, level.floor_z, level.ceiling_z], "frame": None if frame is None else [frame.origin.tolist(), frame.scale_factor, frame.rotation.tolist()], "cell_size": cell_size, "plan_settings": _settings_snapshot(settings), "geometry": asdict(config.geometry), "layers": [{"project_mode": item.project_mode, "mode": item.mode, "codes": item.codes, "name": item.name, "color": item.color, "linetype": item.linetype, "geometry": item.geometry, "band": item.z_band, "area": item.min_area, "tolerance": item.simplify_tolerance, "hatch": item.hatch} for item in layers.layers]}


def _hash_fingerprint_inputs(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def cache_directory(las_path: Path, level: Level | None, settings: PlanSettings | None = None) -> Path:
    """Return the explicit cache location without creating it."""
    project = settings.project_directory if settings and settings.project_directory else Path(las_path).parent
    return Path(project) / "products" / "dxf" / ("site" if level is None else level.id)


def save_cached_result(result: DxfPlanResult, directory: Path) -> Path:
    """Persist a generated result; loading this cache never triggers generation."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload = result_to_dict(result, include_images=False)
    _atomic_bytes(target / "result.json", json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    _atomic_bytes(target / "fingerprint.txt", result.settings_fingerprint.encode("utf-8"))
    if result.background.rgb_png is not None:
        _atomic_bytes(target / "background_rgb.png", result.background.rgb_png)
    if result.background.class_png is not None:
        _atomic_bytes(target / "background_classes.png", result.background.class_png)
    return target


def load_cached_result(directory: Path) -> DxfPlanResult:
    """Load only persisted data; this function deliberately never regenerates."""
    target = Path(directory)
    payload = json.loads((target / "result.json").read_text(encoding="utf-8"))
    rgb = (target / "background_rgb.png").read_bytes() if (target / "background_rgb.png").is_file() else None
    classes = (target / "background_classes.png").read_bytes() if (target / "background_classes.png").is_file() else None
    return result_from_dict(payload, rgb_png=rgb, class_png=classes)


def _atomic_bytes(destination: Path, content: bytes) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)


def is_stale(
    result: DxfPlanResult,
    current_settings: Mapping[str, Any] | PlanSettings | str,
    level: Level | None = None,
    profile_name: str | None = None,
) -> bool:
    """Compare a cached fingerprint to a current precomputed fingerprint or mapping."""
    if isinstance(current_settings, str):
        return result.settings_fingerprint != current_settings
    if isinstance(current_settings, Mapping) and "fingerprint" in current_settings:
        return result.settings_fingerprint != str(current_settings["fingerprint"])
    settings = _settings(current_settings)
    inputs = result.fingerprint_inputs
    if not inputs or "las" not in inputs:
        return True
    source = Path(inputs["las"][0])
    config = settings.config or load_dxf_config()
    layers = settings.layer_config or load_layer_config()
    cell_size = settings.cell_size or (config.geometry.indoor_cell_size if result.mode == "indoor" else config.geometry.outdoor_cell_size)
    current = json.loads(json.dumps(inputs))
    if source.is_file():
        stat = source.stat()
        current["las"] = [str(source.resolve()), stat.st_mtime_ns, stat.st_size]
        # Header-only provenance lookup: identical to read_las_info().profile without scanning all points.
        profile = read_las_header_profile(source)
        current_profile_name = profile.profile_name if profile else None
    else:
        current_profile_name = inputs.get("profile")
    if profile_name is not None:
        current["profile"] = profile_name
    else:
        current["profile"] = current_profile_name
    if level is not None:
        current["level"] = [level.id, level.name, level.floor_z, level.ceiling_z]
    if settings.frame is not None:
        current["frame"] = [settings.frame.origin.tolist(), settings.frame.scale_factor, settings.frame.rotation.tolist()]
    current["cell_size"] = cell_size
    current["plan_settings"] = _settings_snapshot(settings)
    current["geometry"] = asdict(config.geometry)
    current["layers"] = [{"project_mode": item.project_mode, "mode": item.mode, "codes": item.codes, "name": item.name, "color": item.color, "linetype": item.linetype, "geometry": item.geometry, "band": item.z_band, "area": item.min_area, "tolerance": item.simplify_tolerance, "hatch": item.hatch} for item in layers.layers]
    return result.settings_fingerprint != _hash_fingerprint_inputs(current)


def _settings(value: PlanSettings | Mapping[str, Any] | None) -> PlanSettings:
    if value is None or isinstance(value, PlanSettings):
        return value or PlanSettings()
    accepted = {field.name for field in PlanSettings.__dataclass_fields__.values()}
    return PlanSettings(**{key: value[key] for key in accepted & set(value)})


def _settings_snapshot(settings: PlanSettings) -> dict[str, Any]:
    return {"frame": None if settings.frame is None else [settings.frame.origin.tolist(), settings.frame.scale_factor, settings.frame.rotation.tolist()], "config": None if settings.config is None else asdict(settings.config), "layer_config": None if settings.layer_config is None else [{"name": layer.name, "color": layer.color, "mode": layer.mode} for layer in settings.layer_config.layers], "cell_size": settings.cell_size, "fill_under_canopy": settings.fill_under_canopy, "project_directory": None if settings.project_directory is None else str(settings.project_directory), "wall_centerlines": settings.wall_centerlines, "door_swing_arc": settings.door_swing_arc}


@dataclass(slots=True)
class _PointAccumulator:
    """Keep one point per (class, voxel) with memory bounded by occupied voxels.

    Chunks are de-duplicated per class with ``np.unique``; buffered parts are
    consolidated whenever they grow past the consolidated size, so no array
    ever holds the raw cloud.
    """

    cell_size: float
    _parts: dict[int, list[np.ndarray]] = field(default_factory=dict)
    _consolidated: dict[int, int] = field(default_factory=dict)
    _pending: dict[int, int] = field(default_factory=dict)

    def add(self, points: np.ndarray, codes: np.ndarray) -> None:
        if len(points) == 0:
            return
        for code in np.unique(codes):
            key = int(code)
            members = _first_per_voxel(points[codes == code], self.cell_size)
            self._parts.setdefault(key, []).append(members)
            self._pending[key] = self._pending.get(key, 0) + len(members)
            if self._pending[key] > max(self._consolidated.get(key, 0), 50_000):
                self._consolidate(key)

    def _consolidate(self, code: int) -> None:
        merged = np.concatenate(self._parts[code]) if len(self._parts[code]) > 1 else self._parts[code][0]
        merged = _first_per_voxel(merged, self.cell_size)
        self._parts[code] = [merged]
        self._consolidated[code] = len(merged)
        self._pending[code] = 0

    def points_for(self, codes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        points: list[np.ndarray] = []
        classes: list[np.ndarray] = []
        for code in codes:
            if code not in self._parts:
                continue
            if len(self._parts[code]) > 1:
                self._consolidate(code)
            values = self._parts[code][0]
            points.append(values)
            classes.append(np.full(len(values), code, dtype=np.uint8))
        if not points:
            return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.uint8)
        if len(points) == 1:
            return points[0], classes[0]
        return np.concatenate(points), np.concatenate(classes)


def _first_per_voxel(points: np.ndarray, cell_size: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    _, index = np.unique(np.floor(points / cell_size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(index)]


def _highest_per_cell(points: np.ndarray, cell_size: float) -> np.ndarray:
    """Indices of the highest point in every XY cell (vectorised)."""
    cells = np.floor(points[:, :2] / cell_size).astype(np.int64)
    order = np.lexsort((-points[:, 2], cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)
    return order[first]


BACKGROUND_MAX_PIXELS = 2048
"""Longest side of the cached top-view background image."""


@dataclass(slots=True)
class _BackgroundAccumulator:
    """Orthographic top view: per pixel the colour and class of the highest point.

    Dense arrays of at most BACKGROUND_MAX_PIXELS per side; updated per chunk
    with vectorised operations. The PNG is north-up (row 0 = maximum y).
    """

    low: np.ndarray
    high: np.ndarray
    cell_size: float
    width: int = field(init=False)
    height: int = field(init=False)
    pixel_size: float = field(init=False)
    top_z: np.ndarray = field(init=False)
    classes: np.ndarray = field(init=False)
    rgb: np.ndarray = field(init=False)
    has_rgb: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        extent = np.maximum(self.high[:2] - self.low[:2], self.cell_size)
        self.pixel_size = max(self.cell_size, float(extent.max()) / BACKGROUND_MAX_PIXELS)
        self.width = max(1, int(np.ceil(extent[0] / self.pixel_size)) + 1)
        self.height = max(1, int(np.ceil(extent[1] / self.pixel_size)) + 1)
        self.top_z = np.full((self.height, self.width), -np.inf, dtype=np.float32)
        self.classes = np.zeros((self.height, self.width), dtype=np.uint8)
        self.rgb = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        self.has_rgb = np.zeros((self.height, self.width), dtype=bool)

    def add(self, points: np.ndarray, codes: np.ndarray, rgb: np.ndarray | None) -> None:
        """Add one transformed and elevation-filtered LAS chunk."""
        if len(points) == 0:
            return
        columns = np.clip(np.floor((points[:, 0] - self.low[0]) / self.pixel_size).astype(np.int64), 0, self.width - 1)
        rows = np.clip(self.height - 1 - np.floor((points[:, 1] - self.low[1]) / self.pixel_size).astype(np.int64), 0, self.height - 1)
        flat = rows * self.width + columns
        order = np.lexsort((-points[:, 2], flat))
        first = np.ones(len(order), dtype=bool)
        first[1:] = flat[order][1:] != flat[order][:-1]
        top = order[first]
        cell = flat[top]
        z_values = points[top, 2].astype(np.float32)
        current = self.top_z.reshape(-1)
        higher = z_values > current[cell]
        top, cell = top[higher], cell[higher]
        current[cell] = z_values[higher]
        self.classes.reshape(-1)[cell] = codes[top]
        if rgb is not None:
            self.rgb.reshape(-1, 3)[cell] = rgb[top]
            self.has_rgb.reshape(-1)[cell] = True

    def build(self) -> BackgroundImage:
        """Render RGB and class-colour images (BGR order for OpenCV encoding)."""
        occupied = np.isfinite(self.top_z)
        palette = np.zeros((256, 3), dtype=np.uint8)
        codes = np.arange(256)
        palette[:, 0], palette[:, 1], palette[:, 2] = (codes * 47) % 255, (codes * 83) % 255, (codes * 131) % 255
        image_cls = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        image_cls[occupied] = palette[self.classes[occupied]]
        image_rgb = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        image_rgb[occupied] = image_cls[occupied]
        image_rgb[self.has_rgb] = self.rgb[self.has_rgb][:, ::-1]
        bounds = (float(self.low[0]), float(self.low[1]), float(self.low[0] + self.width * self.pixel_size), float(self.low[1] + self.height * self.pixel_size))
        return BackgroundImage(_png(image_rgb), _png(image_cls), bounds)


def _stream_local_points(
    path: Path,
    frame: Frame,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    config: DxfExportConfig,
    mode: Literal["indoor", "outdoor"],
    level: Level | None,
    cell_size: float,
    cancel: CancelToken | None,
    progress: ProgressCallback | None,
) -> tuple[_PointAccumulator, BackgroundImage]:
    """Stream transformed chunks into bounded feature voxels and background cells."""
    accumulator = _PointAccumulator(cell_size)
    low, high = _transformed_bounds(frame, bounds_min, bounds_max)
    background = _BackgroundAccumulator(low, high, cell_size)
    for chunk in iter_las_chunks(path, config.loading.chunk_size, progress=progress, cancel=cancel):
        check_cancel(cancel)
        local = frame.to_local(np.column_stack((chunk.x, chunk.y, chunk.z)))
        codes = np.asarray(chunk.classification, dtype=np.uint8)
        selected = np.ones(len(local), dtype=bool)
        if mode == "indoor" and level is not None:
            ceiling = level.ceiling_z if level.ceiling_z is not None else level.floor_z + config.geometry.indoor_default_height
            selected = (local[:, 2] >= level.floor_z - 0.1) & (local[:, 2] <= ceiling + 0.1)
        local, codes = local[selected], codes[selected]
        accumulator.add(local, codes)
        rgb = None
        if len(local) and all(name in chunk.point_format.dimension_names for name in ("red", "green", "blue")):
            rgb = (np.column_stack((chunk.red, chunk.green, chunk.blue)).astype(np.uint16) // 257)[selected].astype(np.uint8)
        visible = ~np.isin(codes, CEILING_CODES) if mode == "indoor" else np.ones(len(codes), dtype=bool)
        background.add(local[visible], codes[visible], None if rgb is None else rgb[visible])
    return accumulator, background.build()


def _transformed_bounds(frame: Frame, bounds_min: np.ndarray, bounds_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform the eight LAS-bound corners once before streaming point chunks."""
    corners = np.asarray(list(product(*zip(np.asarray(bounds_min, dtype=np.float64), np.asarray(bounds_max, dtype=np.float64), strict=True))), dtype=np.float64)
    transformed = frame.to_local(corners)
    return transformed.min(axis=0), transformed.max(axis=0)


def _mode_points(points: np.ndarray, codes: np.ndarray, layer: LayerDefinition, project_mode: str, level: Level | None, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply a layer's semantic selection mode before geometry rasterization."""
    if layer.mode == "ignore":
        return np.empty((0, 3)), np.empty(0, dtype=np.uint8)
    selected = np.ones(len(points), dtype=bool)
    if project_mode == "indoor" and level is not None:
        ceiling = level.ceiling_z if level.ceiling_z is not None else level.floor_z + 6.0
        selected &= (points[:, 2] >= level.floor_z) & (points[:, 2] <= ceiling)
        if layer.mode == "cut_band" and layer.z_band is not None:
            selected &= (points[:, 2] >= level.floor_z + layer.z_band[0]) & (points[:, 2] <= level.floor_z + layer.z_band[1])
    if layer.mode == "ground_surface":
        selected &= ~np.isin(codes, [3, 4, 5, 95])
    result_points, result_codes = points[selected], codes[selected]
    if layer.mode == "top_surface" and len(result_points):
        indices = _highest_per_cell(result_points, cell_size)
        result_points, result_codes = result_points[indices], result_codes[indices]
    return result_points, result_codes


def _layer_entities(layer: LayerDefinition, points: np.ndarray, codes: np.ndarray, cell_size: float, config: DxfExportConfig, settings: PlanSettings) -> list[dict[str, Any]]:
    if layer.geometry == "wall":
        lines = _wall_lines(points, cell_size, config)
        return [{"type": "line", "start": list(start), "end": list(end)} for start, end in lines] if settings.wall_centerlines else wall_polygons(lines, points, minimum_thickness=config.geometry.wall_min_thickness)
    if layer.geometry == "furniture":
        return furniture_inserts(points, codes)
    if layer.geometry == "circle":
        return circles_from_clusters(points)
    if layer.geometry == "tree":
        return tree_entities(points, codes)
    if layer.geometry == "point":
        return points_from_cluster(points, eps=0.5)
    if layer.geometry == "skeleton":
        return skeleton_lines(points, cell_size, simplify_tolerance=layer.simplify_tolerance, min_length=config.geometry.skeleton_min_length)
    if layer.geometry == "line":
        # Linear objects (curbs, railings, walls, wires) are traced like skeletons so
        # several separate instances never collapse into one zig-zag polyline.
        return skeleton_lines(points, cell_size, simplify_tolerance=layer.simplify_tolerance, min_length=config.geometry.skeleton_min_length)
    polygons = polygons_from_points(points, cell_size, closing_kernel=config.geometry.closing_kernel, simplify_tolerance=layer.simplify_tolerance, min_area=layer.min_area)
    return [{"type": "polyline", "points": [list(value) for value in polygon.exterior.coords], "holes": [[list(value) for value in ring.coords] for ring in polygon.interiors], "closed": True} for polygon in polygons]


def _add_openings(
    output: dict[str, list[dict[str, Any]]],
    statuses: dict[str, LayerStatus],
    layers: tuple[LayerDefinition, ...],
    accumulated: _PointAccumulator,
    project_mode: str,
    level: Level | None,
    cell_size: float,
    config: DxfExportConfig,
    settings: PlanSettings,
) -> None:
    """Project opening classes onto freshly extracted walls and cut every gap."""
    wall_layer = next((layer for layer in layers if layer.geometry == "wall"), None)
    wall_points = np.empty((0, 3), dtype=np.float64)
    walls: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if wall_layer is not None:
        wall_points, wall_codes = accumulated.points_for(wall_layer.codes)
        wall_points, _ = _mode_points(wall_points, wall_codes, wall_layer, project_mode, level, cell_size)
        walls = _wall_lines(wall_points, cell_size, config)
    bridged = bridge_collinear(walls, config.geometry.wall_opening_bridge)
    all_specs: list[dict[str, object]] = []
    for layer in layers:
        if layer.geometry != "opening":
            continue
        try:
            points, opening_codes = accumulated.points_for(layer.codes)
            points, _ = _mode_points(points, opening_codes, layer, project_mode, level, cell_size)
            specs, reason = opening_specs(points, bridged, kind="window" if layer.name == "A-WIND" else "door", cluster_eps=config.geometry.opening_cluster_eps, max_distance=config.geometry.opening_max_distance)
            all_specs.extend(specs)
            entities = opening_entities(specs, swing_arc=settings.door_swing_arc)
            output[layer.name] = entities
            statuses[layer.name] = LayerStatus(layer.name, "ok" if entities else "empty", reason, dxf_entity_count(entities))
        except Exception as error:
            output[layer.name] = []
            statuses[layer.name] = LayerStatus(layer.name, "failed", str(error), 0)
    if wall_layer is None or not walls:
        return
    cut_lines = [{"type": "line", "start": list(start), "end": list(end)} for start, end in walls]
    _cut_openings(cut_lines, all_specs)
    output[wall_layer.name] = cut_lines if settings.wall_centerlines else wall_polygons(_line_entity_pairs(cut_lines), wall_points, minimum_thickness=config.geometry.wall_min_thickness)
    previous = statuses.get(wall_layer.name)
    if previous is not None:
        statuses[wall_layer.name] = LayerStatus(wall_layer.name, "ok" if output[wall_layer.name] else "empty", previous.reason, dxf_entity_count(output[wall_layer.name]))


def _wall_lines(points: np.ndarray, cell_size: float, config: DxfExportConfig) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    geometry = config.geometry
    return wall_lines(
        points,
        cell_size,
        direction_tolerance_degrees=geometry.wall_direction_tolerance_degrees,
        merge_gap=geometry.wall_merge_gap,
        corner_distance=geometry.wall_corner_distance,
        max_thickness=geometry.wall_max_thickness,
        min_length=geometry.wall_min_length,
    )


def dxf_entity_count(entities: list[dict[str, Any]]) -> int:
    """Number of DXF primitives the writer emits (polygon holes are separate polylines; hatches excluded)."""
    return sum(1 + len(entity.get("holes", [])) if entity.get("type") == "polyline" else 1 for entity in entities)


def _line_entity_pairs(lines: list[dict[str, Any]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Convert neutral line entities to wall polygon source segments."""
    return [(tuple(entity["start"]), tuple(entity["end"])) for entity in lines if entity.get("type") == "line"]


def _empty_reason(layer: LayerDefinition, profile: object, histogram: Mapping[int, int]) -> str:
    if profile is not None:
        available = {int(code) for code in getattr(profile, "code_to_name", {})}
        if not available.intersection(layer.codes):
            return f"model profile {getattr(profile, 'profile_name')} does not separate this class"
    return f"class {layer.codes[0]} not present in LAS"


def _cut_openings(walls: list[dict[str, Any]], openings: list[dict[str, Any]]) -> None:
    """Split every wall once around all projected opening intervals."""
    replacement: list[dict[str, Any]] = []
    for wall in walls:
        if wall.get("type") != "line":
            replacement.append(wall)
            continue
        start = np.asarray(wall["start"], dtype=np.float64)
        end = np.asarray(wall["end"], dtype=np.float64)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            replacement.append(wall)
            continue
        unit = vector / length
        intervals: list[tuple[float, float]] = []
        for opening in openings:
            centre = np.asarray(opening["center"], dtype=np.float64)
            direction = np.asarray(opening["direction"], dtype=np.float64)
            offset = centre - start
            if abs(float(offset[0] * unit[1] - offset[1] * unit[0])) > 0.05 or abs(float(direction @ unit)) < 0.99:
                continue
            centre_distance = float(np.dot(offset, unit))
            half_width = float(opening["width"]) / 2.0
            lower, upper = max(0.0, centre_distance - half_width), min(length, centre_distance + half_width)
            if upper > lower:
                intervals.append((lower, upper))
        if not intervals:
            replacement.append(wall)
            continue
        merged: list[list[float]] = []
        for lower, upper in sorted(intervals):
            if merged and lower <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], upper)
            else:
                merged.append([lower, upper])
        cursor = 0.0
        for lower, upper in merged:
            if lower > cursor:
                replacement.append({"type": "line", "start": (start + unit * cursor).tolist(), "end": (start + unit * lower).tolist()})
            cursor = max(cursor, upper)
        if cursor < length:
            replacement.append({"type": "line", "start": (start + unit * cursor).tolist(), "end": end.tolist()})
    walls[:] = replacement


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    factor = np.clip(np.dot(point - start, vector) / max(float(np.dot(vector, vector)), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(point - (start + factor * vector)))


def _fill_ground_holes(output: dict[str, list[dict[str, Any]]], layers: tuple[LayerDefinition, ...]) -> None:
    """Fill ground-surface polygon holes when canopy filling was explicitly enabled."""
    for layer in layers:
        if layer.mode != "ground_surface":
            continue
        for entity in output.get(layer.name, []):
            if entity.get("type") == "polyline":
                entity["holes"] = []


def _png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Unable to encode plan background PNG")
    return bytes(encoded)
