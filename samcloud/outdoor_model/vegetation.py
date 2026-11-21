"""Vegetation scene elements (trees, shrubs) and lawn areas for the outdoor DXF/IFC chain.

:func:`build_vegetation` returns lawn areas as :class:`LawnArea` records; the terrain builder labels
the TIN triangles inside ``LawnArea.polygon`` with ``"lawn"`` and the DXF gets closed boundary
polylines on ``S-VEG``. :func:`lawn_elements` creates one ``VegetationAreaElement`` per area (outline
only); the IFC export gives each its own surface from the lawn-labelled terrain triangles inside its
outline and removes those triangles from the terrain product.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon

from samcloud.bim.core.model import (ParamOrigin, ShrubElement, ShrubParams, Status, TreeElement, TreeParams,
                                     VegetationAreaElement, VegetationAreaParams)
from samcloud.drone.contracts import LocalFrame, QAEntry, ShrubInstance, TreeInstance, load_drone_config
from samcloud.dxf.writer import outdoor_symbol_insert

logger = logging.getLogger(__name__)

DtmSampler = Callable[[np.ndarray], np.ndarray]
Entity = dict

LAYER_TREE = "S-TREE"
LAYER_SHRUB = "S-SHRUB"
LAYER_VEG = "S-VEG"
LAWN_3D_RANGE_M = 0.3
_STATUS = {"ok": Status.OK, "warning": Status.WARNING}


@dataclass
class LawnArea:
    """One lawn / vegetation area: boundary polygon plus the terrain elevations along it."""

    polygon: Polygon
    elevation: float                      # median DTM height over the boundary vertices
    z_range: float                        # max - min DTM height over the boundary vertices
    ring_z: list[float] = field(default_factory=list)  # DTM height per exterior vertex (closing vertex dropped)


@dataclass
class VegetationResult:
    """Everything the vegetation builder produces."""

    elements: list[TreeElement | ShrubElement]
    qa: list[QAEntry]
    lawn_areas: list[LawnArea]


def _cfg(config: dict | None) -> dict:
    cfg = load_drone_config() if config is None else config
    return cfg.get("vegetation", cfg)


def _sample(dtm_sampler: DtmSampler | None, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if dtm_sampler is None:
        return np.zeros(len(xy))
    return np.asarray(dtm_sampler(xy), dtype=np.float64).reshape(-1)


def _ring(polygon: Polygon) -> list[tuple[float, float]]:
    coords = [(float(x), float(y)) for x, y in polygon.exterior.coords]
    return coords[:-1] if len(coords) > 1 and coords[0] == coords[-1] else coords


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _evidence(instance: TreeInstance | ShrubInstance, key: str, ident: int) -> dict:
    ev = instance.evidence
    out: dict = {key: ident, "views": ev.views, "points": ev.points}
    if ev.rmse is not None:
        out["rmse"] = ev.rmse
    return out


def _tree_reject_reason(tree: TreeInstance, cfg: dict) -> str | None:
    if tree.status in ("rejected", "failed"):
        return tree.reason or f"status {tree.status}"
    if not _finite(tree.height_m):
        return "height not measured"
    if not _finite(tree.base_z):
        return "base height unknown"
    min_height = float(cfg.get("tree_min_height_m", 2.0))
    if float(tree.height_m) < min_height:
        return f"height {float(tree.height_m):.2f} m below {min_height:.2f} m"
    points = max(len(tree.point_indices), int(tree.evidence.points))
    min_points = int(cfg.get("tree_min_points", 200))
    if points < min_points:
        return f"{points} points below {min_points}"
    return None


def _build_tree(tree: TreeInstance, cfg: dict) -> TreeElement:
    origin: dict[str, ParamOrigin] = {"trunk_xy": ParamOrigin.MEASURED, "base_z": ParamOrigin.MEASURED,
                                      "height": ParamOrigin.MEASURED}
    crown = float(tree.crown_diameter_m) if _finite(tree.crown_diameter_m) and tree.crown_diameter_m > 0 else None
    trunk = float(tree.trunk_diameter_m) if _finite(tree.trunk_diameter_m) and tree.trunk_diameter_m > 0 else None
    if crown is not None:
        origin["crown_diameter"] = ParamOrigin.MEASURED
    else:
        crown = max(1.5, float(tree.height_m) * 0.5)
        origin["crown_diameter"] = ParamOrigin.ASSUMED
    if trunk is not None:
        origin["trunk_diameter"] = ParamOrigin.MEASURED
    else:
        trunk = float(cfg.get("default_trunk_diameter_m", 0.30))
        origin["trunk_diameter"] = ParamOrigin.ASSUMED
    params = TreeParams(trunk_xy=(float(tree.trunk_xy[0]), float(tree.trunk_xy[1])), base_z=float(tree.base_z),
                        height=float(tree.height_m), crown_diameter=crown, trunk_diameter=trunk)
    assumed = [k for k, v in origin.items() if v is ParamOrigin.ASSUMED]
    status = _STATUS.get(tree.status, Status.UNSURE)
    if assumed and status is Status.OK:
        status = Status.WARNING
    return TreeElement(
        name=f"Baum {tree.tree_id}", params=params, param_origin=origin, status=status,
        source_instance_ids=[f"tree-{tree.tree_id}"], evidence=_evidence(tree, "tree_id", tree.tree_id),
        notes=[f"assumed: {', '.join(assumed)}"] if assumed else [],
    )


def _build_shrub(shrub: ShrubInstance, dtm_sampler: DtmSampler | None) -> ShrubElement | str:
    if shrub.status in ("rejected", "failed"):
        return shrub.reason or f"status {shrub.status}"
    if not _finite(shrub.height_m) or float(shrub.height_m) <= 0:
        return "height not measured"
    if shrub.outline.is_empty or shrub.outline.area <= 0:
        return "empty outline"
    ring = _ring(shrub.outline)
    if len(ring) < 3:
        return "degenerate outline"
    samples = _sample(dtm_sampler, np.asarray(ring + [(shrub.outline.centroid.x, shrub.outline.centroid.y)]))
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return "no terrain height"
    base_z = float(np.median(samples))
    origin = {"outline": ParamOrigin.MEASURED, "base_z": ParamOrigin.MEASURED, "height": ParamOrigin.MEASURED}
    return ShrubElement(
        name=f"Busch {shrub.shrub_id}", params=ShrubParams(outline=ring, base_z=base_z, height=float(shrub.height_m)),
        param_origin=origin, status=_STATUS.get(shrub.status, Status.UNSURE),
        source_instance_ids=[f"shrub-{shrub.shrub_id}"], evidence=_evidence(shrub, "shrub_id", shrub.shrub_id),
    )


def build_lawn_areas(
    lawn_polygons: Sequence[Polygon], dtm_sampler: DtmSampler | None, *, qa: list[QAEntry] | None = None,
) -> list[LawnArea]:
    """Wrap lawn polygons with the DTM elevation (median) and height range of their boundary."""
    areas: list[LawnArea] = []
    for polygon in lawn_polygons:
        if polygon is None or polygon.is_empty or polygon.geom_type != "Polygon" or polygon.area <= 0:
            continue
        ring = _ring(polygon)
        z = _sample(dtm_sampler, np.asarray(ring))
        finite = z[np.isfinite(z)]
        if len(finite) == 0:
            logger.warning("lawn polygon without DTM height skipped")
            continue
        z = np.where(np.isfinite(z), z, float(np.median(finite)))
        areas.append(LawnArea(polygon, float(np.median(z)), float(z.max() - z.min()), [float(v) for v in z]))
    kept: list[LawnArea] = []
    for index, area in enumerate(areas, start=1):
        # A closed DXF ring inside another lawn exterior is necessarily counted as that
        # exterior's hole by the CAD QA. It is duplicate lawn geometry, not a second area.
        nested = any(
            other is not area and Polygon(other.polygon.exterior).contains(area.polygon)
            and other.polygon.intersection(area.polygon).area < 1e-6  # an island inside a hole; overlapping lawns are not duplicates of a hole
            for other in areas
        )
        if nested:
            message = "dropped lawn nested in another lawn outline"
            logger.warning("%s: lawn-%d", message, index)
            if qa is not None:
                qa.append(QAEntry("lawn", f"lawn-{index}", message, {"area_m2": round(float(area.polygon.area), 4)}))
            continue
        kept.append(area)
    return kept


def lawn_elements(lawn_areas: Sequence[LawnArea]) -> list[VegetationAreaElement]:
    """Create BIM vegetation-area records that correspond one-to-one with DXF lawns."""
    result: list[VegetationAreaElement] = []
    for index, area in enumerate(lawn_areas, start=1):
        outline = _ring(area.polygon)
        if len(outline) < 3:
            continue
        result.append(VegetationAreaElement(
            id=f"lawn-{index}", tag="lawn", params=VegetationAreaParams(outline=outline, base_z=area.elevation),
            param_origin={"outline": ParamOrigin.MEASURED, "base_z": ParamOrigin.MEASURED},
            evidence={"area_m2": round(float(area.polygon.area), 4)},
        ))
    return result


def build_vegetation(
    trees: Sequence[TreeInstance], shrubs: Sequence[ShrubInstance], lawn_polygons: Sequence[Polygon] = (),
    dtm_sampler: DtmSampler | None = None, *, config: dict | None = None,
) -> VegetationResult:
    """Build tree and shrub elements, QA entries for rejected ones, and the lawn areas."""
    cfg = _cfg(config)
    elements: list[TreeElement | ShrubElement] = []
    qa: list[QAEntry] = []
    for tree in trees:
        reason = _tree_reject_reason(tree, cfg)
        if reason is not None:
            qa.append(QAEntry("tree", f"tree-{tree.tree_id}", reason,
                              {"height_m": tree.height_m, "trunk_xy": tuple(tree.trunk_xy), "status": tree.status}))
        else:
            elements.append(_build_tree(tree, cfg))
    for shrub in shrubs:
        built = _build_shrub(shrub, dtm_sampler)
        if isinstance(built, str):
            qa.append(QAEntry("shrub", f"shrub-{shrub.shrub_id}", built,
                              {"height_m": shrub.height_m, "status": shrub.status}))
        else:
            elements.append(built)
    return VegetationResult(elements, qa, build_lawn_areas(lawn_polygons, dtm_sampler))


def build_vegetation_elements(
    trees: Sequence[TreeInstance], shrubs: Sequence[ShrubInstance], lawn_polygons: Sequence[Polygon] = (),
    dtm_sampler: DtmSampler | None = None, *, config: dict | None = None,
) -> tuple[list[TreeElement | ShrubElement], list[QAEntry]]:
    """Common builder signature ``(elements, qa)``; use :func:`build_vegetation` to also get lawn areas."""
    result = build_vegetation(trees, shrubs, lawn_polygons, dtm_sampler, config=config)
    return result.elements, result.qa


def _lv95(frame: LocalFrame, x: float, y: float, z: float) -> list[float]:
    return [float(v) for v in frame.to_lv95(np.array([x, y, z]))]


def to_dxf_entities(
    elements: Sequence[object], frame: LocalFrame, dtm_sampler: DtmSampler | None = None,
    lawn_areas: Sequence[LawnArea] = (),
) -> dict[str, list[Entity]]:
    """DXF entities per layer (S-TREE, S-SHRUB (insert + closed outline), S-VEG) with every coordinate already in LV95."""
    out: dict[str, list[Entity]] = {LAYER_TREE: [], LAYER_SHRUB: [], LAYER_VEG: []}
    for element in elements:
        if isinstance(element, TreeElement):
            p = element.params
            origin = element.param_origin
            crown_ok = origin.get("crown_diameter") is not ParamOrigin.ASSUMED
            trunk_ok = origin.get("trunk_diameter") is not ParamOrigin.ASSUMED
            attrs = {"HEIGHT": p.height, "CROWN_DIA": p.crown_diameter if crown_ok else None,
                     "TRUNK_DIA": p.trunk_diameter if trunk_ok else None}
            out[LAYER_TREE].append(outdoor_symbol_insert(
                "TREE", _lv95(frame, p.trunk_xy[0], p.trunk_xy[1], p.base_z), attrs,
                crown_radius=p.crown_diameter / 2.0 if crown_ok else None))
        elif isinstance(element, ShrubElement):
            ring = element.params.outline
            polygon = Polygon(ring)
            centroid = polygon.centroid
            elevation = float(element.params.base_z)
            radius = math.sqrt(polygon.area / math.pi)
            out[LAYER_SHRUB].append(outdoor_symbol_insert(
                "SHRUB", _lv95(frame, centroid.x, centroid.y, elevation), {"HEIGHT": element.params.height},
                crown_radius=radius))
            points = [_lv95(frame, x, y, elevation) for x, y in ring]
            out[LAYER_SHRUB].append({"type": "lwpolyline", "closed": True, "elevation": points[0][2],
                                     "points": [pt[:2] for pt in points]})
    for area in lawn_areas:
        rings = [(_ring(area.polygon), area.ring_z)]
        for hole in area.polygon.interiors:
            coords = [(float(x), float(y)) for x, y in hole.coords][:-1]
            hz = _sample(dtm_sampler, np.asarray(coords))
            hz = np.where(np.isfinite(hz), hz, area.elevation)
            rings.append((coords, [float(v) for v in hz]))
        for coords, zs in rings:
            if area.z_range > LAWN_3D_RANGE_M and len(zs) == len(coords):
                pts = [_lv95(frame, x, y, z) for (x, y), z in zip(coords, zs, strict=True)]
                out[LAYER_VEG].append({"type": "polyline3d", "closed": True, "points": pts})
            else:
                pts = [_lv95(frame, x, y, area.elevation) for x, y in coords]
                out[LAYER_VEG].append({"type": "lwpolyline", "closed": True, "elevation": pts[0][2],
                                       "points": [pt[:2] for pt in pts]})
    return {layer: ents for layer, ents in out.items() if ents}


def category_counts(elements: Sequence[object], lawn_areas: Sequence[LawnArea] = ()) -> dict[str, int]:
    """Counts per scene category (keys for qa_cad.count_consistency)."""
    return {
        "tree": sum(isinstance(e, TreeElement) for e in elements),
        "shrub": sum(isinstance(e, ShrubElement) for e in elements),
        "vegetation_area": sum(isinstance(e, VegetationAreaElement) for e in elements) or len(lawn_areas),
    }
