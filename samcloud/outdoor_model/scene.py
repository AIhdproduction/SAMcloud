"""Assembly of outdoor BIM scenes and neutral DXF results."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import numpy as np
import yaml

from samcloud.bim.core.model import Building, BuildingElement, CrsInfo, DataSource, ProjectSettings, Scene, Site, Scope
from samcloud.drone.contracts import LocalFrame
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus


def assemble_scene(*element_lists: Iterable[object], frame: LocalFrame) -> Scene:
    """Create an outdoor-only LV95 scene with one site and one spatial building per envelope."""
    site = Site(id="site-0", name="Survey site")
    settings = ProjectSettings(data_source=DataSource.PHOTOGRAMMETRY, scope=Scope.OUTDOOR_ONLY,
        origin_offset=tuple(float(v) for v in frame.origin_lv95),
        crs=CrsInfo(epsg=frame.epsg, name=f"EPSG:{frame.epsg}", vertical_datum=frame.vertical_datum, source="input"))
    scene = Scene(settings=settings, sites={site.id: site})
    for group in element_lists:
        for element in group or ():
            if isinstance(element, BuildingElement):
                scene.buildings[element.id] = Building(id=element.id, name=element.name or element.id, site_id=site.id)
            if getattr(element, "site_id", None) is None:
                element.site_id = site.id
            if getattr(element, "building_id", None) is not None and element.building_id not in scene.buildings:
                # This can occur when a caller supplies facade records before the envelope.
                scene.buildings[element.building_id] = Building(id=element.building_id, name=element.building_id, site_id=site.id)
            scene.add(element)
    return scene


def _styles() -> tuple[dict[str, int], dict[str, str], dict[str, dict]]:
    config = Path("config/dxf_layers.yaml")
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) if config.exists() else {}
    records = list(raw.get("layers", [])) + list(raw.get("outdoor_extra_layers", []))
    colors: dict[str, int] = {}; types: dict[str, str] = {}; styles: dict[str, dict] = {}
    for record in records:
        name = str(record["name"])
        colors[name] = int(record.get("color", 7)); types[name] = str(record.get("linetype", "CONTINUOUS"))
        styles[name] = {key: record[key] for key in ("lineweight", "true_color") if key in record}
    return colors, types, styles


def assemble_dxf(entities_by_builder, frame: LocalFrame, config: dict | None = None) -> DxfPlanResult:
    """Merge already-LV95 builder entities into the writer's neutral outdoor DXF result."""
    layers: dict[str, list[dict]] = {}
    values = entities_by_builder.values() if isinstance(entities_by_builder, dict) else entities_by_builder
    for value in values:
        if not value: continue
        for layer, entities in value.items(): layers.setdefault(layer, []).extend(entities)
    colors, types, styles = _styles()
    for layer in layers: colors.setdefault(layer, 7); types.setdefault(layer, "CONTINUOUS"); styles.setdefault(layer, {})
    all_points: list[np.ndarray] = []
    for entities in layers.values():
        for entity in entities:
            for key in ("points", "fit_points"):
                if key in entity and entity[key]: all_points.append(np.asarray(entity[key], float).reshape(-1, len(entity[key][0])))
            for key in ("point", "start", "end", "center"):
                if key in entity: all_points.append(np.asarray(entity[key], float).reshape(1, -1))
    joined = np.vstack([x[:, :3] if x.shape[1] >= 3 else np.column_stack([x, np.zeros(len(x))]) for x in all_points]) if all_points else np.zeros((1, 3))
    bounds = (float(joined[:, 0].min()), float(joined[:, 1].min()), float(joined[:, 0].max()), float(joined[:, 1].max()))
    blocks = sorted({str(entity.get("name")) for values in layers.values() for entity in values if entity.get("type") == "insert"})
    return DxfPlanResult("SAMcloud", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "outdoor-scene",
        layers, {name: LayerStatus(name, "ok", None, len(items)) for name, items in layers.items()}, colors,
        BackgroundImage(None, None, bounds), {}, types, {name: False for name in layers}, styles, blocks, "environment")
