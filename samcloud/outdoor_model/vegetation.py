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
    trunk_measured = trunk is not None
    if trunk_measured:
        origin["trunk_diameter"] = ParamOrigin.MEASURED
    else:
        trunk = float(cfg.get("default_trunk_diameter_m", 0.30))
        origin["trunk_diameter"] = ParamOrigin.ASSUMED
    shape = str(getattr(tree, "crown_shape", None) or "ellipsoid")
    shape_status = str(getattr(tree, "crown_shape_status", "assumed"))
    origin["crown_shape"] = ParamOrigin.MEASURED if shape_status == "measured" else ParamOrigin.ASSUMED
    crown_base = getattr(tree, "crown_base_z", None)
    if not _finite(crown_base):
        crown_base = None
    elif not float(tree.base_z) < float(crown_base) < float(tree.base_z) + float(tree.height_m):
        crown_base = None                      # inconsistent with trunk/top: fall back to the stylised rule
    trunk_xy = (float(tree.trunk_xy[0]), float(tree.trunk_xy[1]))
    offset = tuple(getattr(tree, "crown_offset_xy", (0.0, 0.0)) or (0.0, 0.0))
    notes: list[str] = []
    # The trunk must stand under the crown footprint: clamp a trunk outside it to the footprint edge minus its radius.
    reach = max(crown / 2.0 - trunk / 2.0, 0.0)
    off_len = math.hypot(offset[0], offset[1])               # trunk -> crown centre distance
    if off_len > reach and off_len > 1e-9:
        k = (off_len - reach) / off_len
        trunk_xy = (trunk_xy[0] + offset[0] * k, trunk_xy[1] + offset[1] * k)
        offset = (offset[0] * (1.0 - k), offset[1] * (1.0 - k))
        notes.append("trunk position clamped to the crown footprint")
    params = TreeParams(trunk_xy=trunk_xy, base_z=float(tree.base_z),
                        height=float(tree.height_m), crown_diameter=crown, trunk_diameter=trunk,
                        crown_shape=shape, crown_base_z=None if crown_base is None else float(crown_base),
                        crown_offset=(float(offset[0]), float(offset[1])), trunk_measured=trunk_measured)
    # The crown shape is a stylisation choice, it does not downgrade the element status.
    assumed = [k for k, v in origin.items() if v is ParamOrigin.ASSUMED and k != "crown_shape"]
    status = _STATUS.get(tree.status, Status.UNSURE)
    if assumed and status is Status.OK:
        status = Status.WARNING
    return TreeElement(
        name=f"Baum {tree.tree_id}", params=params, param_origin=origin, status=status,
        source_instance_ids=[f"tree-{tree.tree_id}"], evidence=_evidence(tree, "tree_id", tree.tree_id),
        notes=([f"assumed: {', '.join(assumed)}"] if assumed else []) + notes,
    )


def tin_height_sampler(tin: object) -> Callable[[np.ndarray], np.ndarray]:
    """Height sampler of a TIN (``vertices`` (V,3), ``triangles`` (T,3)): barycentric interpolation, NaN outside."""
    from scipy.spatial import cKDTree
    vertices = np.asarray(tin.vertices, dtype=np.float64)
    triangles = np.asarray(tin.triangles, dtype=np.int64)
    corners = vertices[triangles]                                   # (T,3,3)
    kd = cKDTree(corners[:, :, :2].mean(axis=1))

    def sample(xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = np.full(len(pts), np.nan)
        k = min(24, len(triangles))
        _, candidates = kd.query(pts, k=k)
        candidates = np.asarray(candidates).reshape(len(pts), -1)
        for i, (x, y) in enumerate(pts):
            for t in candidates[i]:
                (ax, ay, az), (bx, by, bz), (cx, cy, cz) = corners[t]
                det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                if abs(det) < 1e-12:
                    continue
                w1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det
                w2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det
                w3 = 1.0 - w1 - w2
                if min(w1, w2, w3) >= -1e-9:
                    out[i] = w1 * az + w2 * bz + w3 * cz
                    break
        return out

    return sample


def rebase_trees_on_tin(elements: Sequence[object], tin: object) -> list[object]:
    """Put every tree's trunk base on the final TIN surface (top of the tree and crown base stay where they are)."""
    sample = tin_height_sampler(tin)
    out: list[object] = []
    for element in elements:
        if isinstance(element, TreeElement):
            p = element.params
            z = float(sample(np.array([p.trunk_xy]))[0])
            top = float(p.base_z) + float(p.height)
            if math.isfinite(z) and top - z > 0.5:
                element = element.model_copy(update={"params": p.model_copy(update={"base_z": z, "height": top - z})})
        out.append(element)
    return out


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


def _text_height(config: dict | None) -> float | None:
    """``dxf.block_text_height_m`` from the given (full) config, else None (writer default / config file)."""
    dxf = (config or {}).get("dxf") if isinstance(config, dict) else None
    value = (dxf or {}).get("block_text_height_m")
    return float(value) if value else None


def to_dxf_entities(
    elements: Sequence[object], frame: LocalFrame, dtm_sampler: DtmSampler | None = None,
    lawn_areas: Sequence[LawnArea] = (), config: dict | None = None,
) -> dict[str, list[Entity]]:
    """DXF entities per layer (S-TREE, S-SHRUB (insert + closed outline), S-VEG) with every coordinate already in LV95.

    A TREE insert sits at the crown centre, carries the visible labels KRONE/STAMM/HOEHE and
    is scaled to the measured crown radius.  The trunk circle is a separate TREE_STEM insert (to scale)
    at the trunk position and only exists for a measured trunk.
    """
    out: dict[str, list[Entity]] = {LAYER_TREE: [], LAYER_SHRUB: [], LAYER_VEG: []}
    text_height = _text_height(config)
    for element in elements:
        if isinstance(element, TreeElement):
            p = element.params
            origin = element.param_origin
            crown_ok = origin.get("crown_diameter") is not ParamOrigin.ASSUMED
            trunk_ok = origin.get("trunk_diameter") is not ParamOrigin.ASSUMED
            if p.trunk_measured is not None:
                trunk_ok = bool(p.trunk_measured)
            attrs = {"HEIGHT": p.height, "CROWN_DIA": p.crown_diameter if crown_ok else None,
                     "TRUNK_DIA": p.trunk_diameter if trunk_ok else None}
            offset = p.crown_offset or (0.0, 0.0)
            out[LAYER_TREE].append(outdoor_symbol_insert(          # crown circle + labels at the crown centre
                "TREE", _lv95(frame, p.trunk_xy[0] + offset[0], p.trunk_xy[1] + offset[1], p.base_z), attrs,
                crown_radius=p.crown_diameter / 2.0 if crown_ok else None, text_height=text_height))
            if trunk_ok:
                out[LAYER_TREE].append(outdoor_symbol_insert(
                    "TREE_STEM", _lv95(frame, p.trunk_xy[0], p.trunk_xy[1], p.base_z), None,
                    crown_radius=p.trunk_diameter / 2.0))
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
