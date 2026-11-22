"""Reconstruct every plan furniture item from its scan points and write the result into the PlanFurniture fields.

Per item: points inside the footprint (furniture classes, ownership between overlapping items) -> cleaning (voxel,
statistical outliers, vertical runs inside the family's height window) -> class model (parametric parts or slice-hull
mesh) -> fit metrics against the previous footprint body -> accept or keep the footprint body.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from samcloud.bim.furniture_recon import parametric
from samcloud.bim.furniture_recon.cloud import (
    ItemDescriptor, SceneCloud, clean_item_points, grown_indices, load_scene_cloud, owned_indices, xy_coverage,
)
from samcloud.bim.furniture_recon.families import STANDING_FAMILIES, WINDOW_KEY, allows_tall_gaps, family_of, is_low_piece, may_hang
from samcloud.bim.furniture_recon.geometry import (
    ItemFrame, Mesh, Solid, extrusion, frame_for_footprint, mesh_to_plan, outline_to_plan, projected_outline, solid_to_plan,
)
from samcloud.bim.furniture_recon.mesh import fit_mesh
from samcloud.bim.furniture_recon.metrics import FitMetrics, evaluate
from samcloud.bim.furniture_recon.settings import ReconSettings, load_recon_settings

_LOGGER = logging.getLogger(__name__)

_ELEVATED_KINDS = frozenset({"worktop", "kitchen_sink", "cooktop"})
_MESH_STANDING = frozenset({"toilet"})


@dataclass(slots=True)
class _Work:
    item: Any
    polygon: Polygon  # local units
    family: str
    frame: ItemFrame
    hanging: bool
    key: tuple = ()  # footprint key at the start (kitchen base and worktop of one run share it)


def _polygon(points: list[list[float]]) -> Polygon | None:
    try:
        polygon = Polygon(points)
        polygon = polygon if polygon.is_valid else polygon.buffer(0)
    except (TypeError, ValueError):
        return None
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda item: item.area)
    return polygon if isinstance(polygon, Polygon) and not polygon.is_empty and polygon.area > 0 else None


def _footprint_key(item: Any) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(x), 3), round(float(y), 3)) for x, y in item.footprint)


def _previous_shape(work: _Work, upm: float, floor_z: float, fallback_height: float) -> list[Solid]:
    """The body the IFC export drew before: stepped parts or the footprint prism (scene_builder rules)."""
    item = work.item
    frame = work.frame
    elevated = str(item.catalogue_kind or "").lower() in _ELEVATED_KINDS
    if item.z_source in ("instance_bbox", "points") and item.z_min is not None and item.z_max is not None:
        z0, z1 = (float(item.z_min) - floor_z) / upm, (float(item.z_max) - floor_z) / upm
    elif item.default_z_m:
        z0, z1 = float(item.default_z_m[0]), float(item.default_z_m[1])
    else:
        z0, z1 = 0.0, float(item.default_height_m or fallback_height)
    if not work.hanging and not elevated:
        z0 = 0.0
    solids: list[Solid] = []
    for part in item.parts or []:
        polygon = _polygon(part.get("footprint") or [])
        if polygon is None:
            continue
        high = (float(part.get("z_max", item.z_max if item.z_max is not None else floor_z + z1 * upm)) - floor_z) / upm
        low = 0.0 if not work.hanging else (float(part.get("z_min", item.z_min if item.z_min is not None else floor_z)) - floor_z) / upm
        if high - low >= 0.02:
            solids.append(extrusion("part", frame.polygon_to_item(polygon), low, min(high, z1)))
    if solids:
        return solids
    return [extrusion("body", frame.polygon_to_item(work.polygon), z0, max(z1, z0 + 0.02))]


def _accept(new: FitMetrics | None, old: FitMetrics | None, settings: ReconSettings, model: str = "") -> bool:
    if new is None:
        return False
    if old is None:
        return True
    if model == "box":
        # a box replaces the footprint body only when it fits the points at least as well (it adds no detail)
        return new.mean_m <= old.mean_m + 0.005 and new.coverage >= old.coverage - 0.05
    if new.coverage < old.coverage - settings.max_coverage_drop:
        return False
    return new.mean_m <= max(old.mean_m * settings.max_mean_factor, settings.coverage_distance_m)


def _window(work: _Work, settings: ReconSettings, ceiling_m: float, upm: float, floor_z: float) -> tuple[float, float]:
    item = work.item
    if item.layer == "A-OBEN":
        if item.z_min is not None and item.z_max is not None:
            return (float(item.z_min) - floor_z) / upm - 0.10, (float(item.z_max) - floor_z) / upm + 0.10
        if item.default_z_m:
            return float(item.default_z_m[0]) - 0.10, float(item.default_z_m[1]) + 0.10
    return settings.window(WINDOW_KEY.get(work.family, "box"), ceiling_m)


def _low(work: _Work) -> bool:
    return is_low_piece(work.item.label, work.item.semantic_label, work.item.catalogue_kind)


def _fit_family(work: _Work, points: np.ndarray, footprint: Polygon, settings: ReconSettings, window: tuple[float, float]) -> tuple[str, list[Solid], Mesh | None]:
    family = work.family
    if family == "table":
        solids = parametric.fit_table(points, footprint, settings, window)
        if solids:
            return "table", solids, None
    elif family == "chair":
        solids = parametric.fit_chair(points, footprint, settings, window)
        if solids:
            return "chair", solids, None
    elif family == "sofa":
        solids = parametric.fit_sofa(points, footprint, settings, window)
        if solids:
            return "sofa", solids, None
    elif family == "bed":
        solids = parametric.fit_bed(points, footprint, settings, window)
        if solids:
            return "bed", solids, None
    elif family in ("mesh", "toilet"):
        z_low, z_high = (float(v) for v in np.percentile(points[:, 2], [1, 99.5]))
        # what cannot hang stands on the floor: the lowest slice is carried down to it
        standing = (family in _MESH_STANDING and z_low < 0.10) or not work.hanging
        z_bottom = 0.0 if standing else z_low
        mesh = fit_mesh(points, z_bottom=z_bottom, z_top=z_high, ring_vertices=settings.mesh_ring_vertices, slice_m=settings.mesh_slice_m, max_triangles=settings.mesh_max_triangles)
        if mesh is not None:
            return "mesh", [], mesh
    solids = parametric.fit_box(points, footprint, settings, hanging=work.hanging, low_piece=_low(work))
    if solids and family == "counter":
        top = parametric.worktop_level(points)
        if top is not None and top > solids[0].z0 + 0.3:
            solids[0].z1 = top  # the measured worktop surface, not appliances standing on it
    return ("box", solids, None) if solids else ("none", [], None)


def _kitchen_tops(works: list[_Work], cloud: SceneCloud, descriptors: list[ItemDescriptor], fine_family: np.ndarray, settings: ReconSettings, upm: float) -> dict[tuple, float]:
    """Measured worktop height (metres above floor) per kitchen-run footprint."""
    tops: dict[tuple, float] = {}
    counts: dict[tuple, int] = {}
    runs: list[_Work] = []
    for work in works:
        if work.family != "kitchen_base":
            continue
        key = work.key
        if key in tops:
            continue
        descriptor = next(d for d in descriptors if d.key == work.item.id)
        indices = owned_indices(cloud, descriptor, descriptors, settings.footprint_margin_m * upm, fine_family)
        pts = work.frame.to_item(cloud.xyz[indices])
        pts = clean_item_points(pts, settings.window("kitchen", 3.0), settings, standing=True, tall=False)
        top = parametric.worktop_level(pts) if len(pts) >= settings.min_points else None
        if top is not None:
            tops[key] = top
            counts[key] = len(pts)
            runs.append(work)
    _level_abutting_tops(runs, tops, counts, settings.worktop_thickness_m, upm)
    return tops


def _level_abutting_tops(runs: list[_Work], tops: dict[tuple, float], counts: dict[tuple, int], tolerance_m: float, upm: float) -> None:
    """Abutting pieces of one run carry one worktop: levels closer than its thickness become the level measured by most points."""
    measured = dict(tops)
    waiting = sorted(runs, key=lambda work: -counts[work.key])
    while waiting:
        group = [waiting.pop(0)]
        for member in group:  # grows while abutting pieces at the same level are found
            for other in list(waiting):
                same_level = abs(measured[other.key] - measured[group[0].key]) <= tolerance_m
                if same_level and other.polygon.distance(member.polygon) <= 0.01 * upm:
                    group.append(other)
                    waiting.remove(other)
        for member in group:
            tops[member.key] = measured[group[0].key]


def _run_top_for(work: _Work, works: list[_Work], tops: dict[tuple, float]) -> float | None:
    best: tuple[float, float] | None = None
    for other in works:
        if other.family != "kitchen_base":
            continue
        top = tops.get(other.key)
        if top is None:
            continue
        overlap = work.polygon.intersection(other.polygon).area / max(work.polygon.area, 1e-12)
        if overlap > 0.3 and (best is None or overlap > best[0]):
            best = (overlap, top)
    return None if best is None else best[1]


def _apply(work: _Work, model: str, solids: list[Solid], mesh: Mesh | None, fit: dict[str, Any]) -> None:
    item = work.item
    frame = work.frame
    outline = projected_outline(solids, mesh)
    item.solids = [solid_to_plan(solid, frame) for solid in solids]
    item.mesh = mesh_to_plan(mesh, frame) if mesh is not None else None
    item.shape_source = "mesh" if mesh is not None else "parametric"
    lows = [s.z0 for s in solids] + ([float(mesh.vertices[:, 2].min())] if mesh is not None else [])
    highs = [s.z1 for s in solids] + ([float(mesh.vertices[:, 2].max())] if mesh is not None else [])
    item.z_min, item.z_max = frame.z_to_local(min(lows)), frame.z_to_local(max(highs))
    item.z_source = "points"
    if outline is not None and outline.area > 1e-4:
        # the outline it started from: the 2D DXF redraws exactly that polyline (samcloud.bim.furniture_recon.dxf_outline)
        fit["plan_footprint"] = [[round(float(x), 4), round(float(y), 4)] for x, y in item.footprint]
        # a kitchen piece that the simplification would move keeps its exact outline: abutting pieces must not overlap
        if model != "kitchen" or outline.hausdorff_distance(frame.polygon_to_item(work.polygon)) <= 1e-3:
            item.footprint = outline_to_plan(outline, frame)
    item.fit = fit


def _fit_candidates(
    work: _Work, points: np.ndarray, footprint_fit: Polygon, footprint_item: Polygon, settings: ReconSettings, window: tuple[float, float],
    before: FitMetrics | None,
) -> tuple[str, list[Solid], Mesh | None, FitMetrics | None, bool]:
    """Class model, then mesh, then box: the first candidate accepted against the previous body (``before``)."""
    candidates = [_fit_family(work, points, footprint_fit, settings, window)]
    if candidates[0][0] == "box" and footprint_fit is not footprint_item and work.family in ("table", "chair", "sofa", "bed"):
        # the class model may fail on the grown outline (a top surface diluted by it): retry on the plan outline
        retry = _fit_family(work, points, footprint_item, settings, window)
        if retry[0] != "box":
            candidates = [retry]
    if candidates[0][0] == "mesh":
        # a mesh must explain the points at least as well as a plain box of them
        box_solids = parametric.fit_box(points, footprint_fit, settings, hanging=work.hanging, low_piece=_low(work)) or []
        mesh_metrics = evaluate(points, [], candidates[0][2], settings.coverage_distance_m)
        box_metrics = evaluate(points, box_solids, None, settings.coverage_distance_m) if box_solids else None
        if mesh_metrics is not None and box_metrics is not None and mesh_metrics.coverage < box_metrics.coverage - 0.10:
            candidates = [("box", box_solids, None)]
    elif work.family in ("table", "chair", "sofa", "bed"):
        # an unusual piece (lounge chair, bean bag) the class model does not describe: compact mesh, then box
        z_high = float(np.percentile(points[:, 2], 99.5))
        fallback = fit_mesh(points, z_bottom=0.0, z_top=z_high, ring_vertices=settings.mesh_ring_vertices, slice_m=settings.mesh_slice_m, max_triangles=settings.mesh_max_triangles)
        if fallback is not None:
            candidates.insert(0 if candidates[0][0] == "box" else 1, ("mesh", [], fallback))
    if all(candidate[0] != "box" for candidate in candidates):
        candidates.append(("box", parametric.fit_box(points, footprint_fit, settings, hanging=work.hanging, low_piece=_low(work)) or [], None))
    model, solids, mesh, after = "none", [], None, None
    for c_model, c_solids, c_mesh in candidates:
        if not c_solids and c_mesh is None:
            continue
        c_after = evaluate(points, c_solids, c_mesh, settings.coverage_distance_m)
        if model == "none":
            model, solids, mesh, after = c_model, c_solids, c_mesh, c_after
        if _accept(c_after, before, settings, c_model):
            return c_model, c_solids, c_mesh, c_after, True
    return model, solids, mesh, after, False


def _choose_growth(plain: tuple, grown: tuple, own_points: np.ndarray, settings: ReconSettings) -> str:
    """Use the grown fit only when it is accepted and explains the item's own points (core + grown) better, or as well
    without adding much volume: growth that adds volume without improving coverage is rejected."""
    _g_model, g_solids, g_mesh, _g_after, g_ok = grown
    if not g_ok:
        return "plain: grown fit not accepted"
    _p_model, p_solids, p_mesh, _p_after, p_ok = plain
    g_metrics = evaluate(own_points, g_solids, g_mesh, settings.coverage_distance_m)
    p_metrics = evaluate(own_points, p_solids, p_mesh, settings.coverage_distance_m) if (p_solids or p_mesh is not None) else None
    if g_metrics is None:
        return "plain: no grown metrics"
    if p_metrics is None or not p_ok:
        return "grown"
    gain = g_metrics.coverage - p_metrics.coverage
    volume = g_metrics.volume_m3 / max(p_metrics.volume_m3, 1e-9)
    if gain < -0.02:
        return "plain: grown explains fewer own points"
    if volume > settings.grow_max_volume_factor and gain < settings.grow_min_coverage_gain:
        return f"plain: volume x{volume:.2f} without coverage gain ({gain:+.2f})"
    return "grown"


def reconstruct_items(
    items: list[Any], cloud: SceneCloud | None, *, units_per_metre: float, floor_z: float, ceiling_z: float | None,
    settings: ReconSettings | None = None, fallback_height_m: float = 0.8,
) -> dict[str, Any]:
    """Reconstruct ``items`` (PlanFurniture, modified in place) from ``cloud``; return a summary report."""
    settings = settings or load_recon_settings()
    upm = max(float(units_per_metre), 1e-9)
    report: dict[str, Any] = {"items": 0, "parametric": 0, "mesh": 0, "footprint": 0, "heights_measured": 0, "by_family": {}}
    if not settings.enabled or cloud is None or not len(cloud.xyz):
        report["skipped"] = "disabled" if not settings.enabled else "no furniture points"
        return report
    ceiling_m = (float(ceiling_z) - float(floor_z)) / upm if ceiling_z is not None else 3.0
    works: list[_Work] = []
    for item in items:
        polygon = _polygon(item.footprint)
        if polygon is None:
            continue
        family = family_of(item.label, item.semantic_label, item.catalogue_kind)
        mounting = getattr(item, "mounting", None)
        if mounting:
            hanging = mounting != "standing"
        else:
            hanging = may_hang(item.label, item.semantic_label, item.catalogue_kind, item.layer) and item.floor_contact is not True
        works.append(_Work(item, polygon, family, frame_for_footprint(polygon, upm, float(floor_z)), hanging, _footprint_key(item)))
    descriptors = [ItemDescriptor(w.item.id, w.polygon, w.family, w.item.instance_id) for w in works]
    fine_family = cloud.fine_families()
    margin = settings.footprint_margin_m * upm
    tops = _kitchen_tops(works, cloud, descriptors, fine_family, settings, upm)
    for work, descriptor in zip(works, descriptors):
        item = work.item
        if getattr(item, "mounting", None):
            # measured by samcloud.bim.room_objects (wall shelf, screen): it only keeps its points from the other items
            report["items"] += 1
            family_report = report["by_family"].setdefault(work.family, {"items": 0, "parametric": 0, "mesh": 0, "footprint": 0})
            family_report["items"] += 1
            kind = item.shape_source if item.shape_source in ("parametric", "mesh") else "footprint"
            report[kind] += 1
            family_report[kind] += 1
            continue
        report["items"] += 1
        family_report = report["by_family"].setdefault(work.family, {"items": 0, "parametric": 0, "mesh": 0, "footprint": 0})
        family_report["items"] += 1
        footprint_item = work.frame.polygon_to_item(work.polygon)
        before_shape = _previous_shape(work, upm, float(floor_z), fallback_height_m)
        was_default = item.z_source == "default"
        indices = owned_indices(cloud, descriptor, descriptors, margin, fine_family)
        kitchen = work.family in ("kitchen_base", "worktop", "kitchen_insert")
        window = _window(work, settings, ceiling_m, upm, float(floor_z))
        standing = work.family in STANDING_FAMILIES or not work.hanging
        low_piece = _low(work) and not work.hanging
        tall = allows_tall_gaps(work.family, item.label, item.semantic_label, item.catalogue_kind) and not low_piece

        def cleaned(selection: np.ndarray) -> np.ndarray:
            return clean_item_points(work.frame.to_item(cloud.xyz[selection]), window, settings, standing=standing, tall=tall, anchor_floor=low_piece)

        points = cleaned(indices)
        footprint_fit = footprint_item
        growth: dict[str, Any] = {}
        grown_points = None
        if not kitchen and settings.grow_m > 0:
            own_terms = {str(text or "").strip().lower() for text in (item.semantic_label, item.label)} - {""}
            grown = grown_indices(
                cloud, descriptor, descriptors, indices, settings.grow_m * upm, 0.05 * upm, fine_family, own_terms,
                extent_margin=settings.grow_extent_margin_m * upm,
            )
            if len(grown) >= 20:
                # the plan outline cut the object short: the fit may use the outline grown to the connected own-label points
                from shapely.geometry import MultiPoint

                grown_xy = work.frame.to_item(cloud.xyz[grown])[:, :2]
                hull = MultiPoint([tuple(p) for p in np.unique(np.round(grown_xy, 2), axis=0)]).convex_hull
                gap = settings.footprint_margin_m + 0.02  # the grown points start beyond the selection margin
                merged = footprint_item.union(hull).buffer(gap).buffer(-gap) if isinstance(hull, Polygon) else footprint_item
                # growth is evidence that the outline was cut short, not licence to swallow a neighbour: at most max_grow_ratio x the outline
                if isinstance(merged, Polygon) and footprint_item.area * 1.05 < merged.area <= footprint_item.area * (1.0 + settings.max_grow_ratio):
                    footprint_fit = Polygon(merged.exterior).simplify(0.01)
                    if work.family not in ("sofa", "bed"):
                        from shapely.geometry import box as _bounds_box

                        footprint_fit = _bounds_box(*footprint_fit.bounds)  # boxes, tops and seats are fitted in the item frame
                    grown_points = cleaned(np.concatenate((indices, grown)))
                    growth = {"grown_points": int(len(grown)), "grown_area_m2": round(footprint_fit.area - footprint_item.area, 3)}
                else:
                    growth = {"growth_rejected": "area cap", "growth_points": int(len(grown))}
        coverage_xy = xy_coverage(points, footprint_item)
        fit: dict[str, Any] = {"family": work.family, "points": int(len(points)), "xy_coverage": round(coverage_xy, 3), **growth}
        before = evaluate(points, before_shape, None, settings.coverage_distance_m) if len(points) else None
        if before is not None:
            fit["before"] = before.as_dict()
        model, solids, mesh = "none", [], None
        after: FitMetrics | None = None
        accepted = False
        if kitchen:
            model, solids = _kitchen_solids(work, footprint_item, tops, works, settings)
            accepted = bool(solids)
            if work.family == "kitchen_base" and solids and len(points):
                # the run's points belong to base AND worktop: both are measured together on the base item
                top = solids[0].z1 + settings.worktop_thickness_m
                worktop = extrusion("worktop", footprint_item, solids[0].z1, top)
                old_top = before_shape[0].z1
                old_worktop = extrusion("worktop", footprint_item, old_top, old_top + 0.04)
                before = evaluate(points, [*before_shape, old_worktop], None, settings.coverage_distance_m)
                if before is not None:
                    fit["before"] = before.as_dict()
                after = evaluate(points, [*solids, worktop], None, settings.coverage_distance_m)
                fit["metrics_scope"] = "kitchen run (base + worktop)"
            elif solids:
                fit.pop("before", None)
                before = None
                fit["metrics_scope"] = "measured with the kitchen base item"
        elif len(points) >= settings.min_points and coverage_xy >= settings.min_xy_coverage:
            model, solids, mesh, after, accepted = _fit_candidates(work, points, footprint_item, footprint_item, settings, window, before)
            if grown_points is not None and len(grown_points) >= settings.min_points:
                g_before = evaluate(grown_points, before_shape, None, settings.coverage_distance_m)
                g = _fit_candidates(work, grown_points, footprint_fit, footprint_item, settings, window, g_before)
                choice = _choose_growth((model, solids, mesh, after, accepted), g, grown_points, settings)
                fit["growth_decision"] = choice
                if choice == "grown":
                    model, solids, mesh, after, accepted = g
                    points, before = grown_points, g_before
                    fit["points"] = int(len(points))
                    if before is not None:
                        fit["before"] = before.as_dict()
        else:
            fit["reason"] = "too few points" if len(points) < settings.min_points else "footprint poorly covered"
        if accepted:
            fit["model"] = model
            if after is not None:
                fit["after"] = after.as_dict()
                if before is not None and before.volume_m3 > 0:
                    fit["volume_ratio"] = round(after.volume_m3 / before.volume_m3, 3)
            fit["parts"] = {"measured": sum(s.source == "measured" for s in solids), "symmetry": sum(s.source == "symmetry" for s in solids), "prior": sum(s.source == "prior" for s in solids)}
            _apply(work, model, solids, mesh, fit)
            kind = item.shape_source
            report[kind] += 1
            family_report[kind] += 1
            if was_default:
                report["heights_measured"] += 1
            continue
        if solids or mesh is not None:
            fit["reason"] = "fit worse than footprint body"
            fit["rejected_model"] = model
            if after is not None:
                fit["rejected"] = after.as_dict()
        item.shape_source = "footprint"
        if was_default and len(points) >= 30 and work.family not in ("kitchen_base", "worktop", "kitchen_insert"):
            z_low, z_high = (float(v) for v in np.percentile(points[:, 2], [2, 98]))
            if z_high >= 0.10:
                item.z_min = work.frame.z_to_local(z_low if work.hanging and z_low > 0.15 else 0.0)
                item.z_max = work.frame.z_to_local(z_high)
                item.z_source = "points"
                fit["heights"] = "measured"
                report["heights_measured"] += 1
        item.fit = fit
        report["footprint"] += 1
        family_report["footprint"] += 1
    report["default_heights_left"] = sum(1 for w in works if w.item.z_source == "default")
    return report


def _kitchen_solids(work: _Work, footprint_item: Polygon, tops: dict[tuple, float], works: list[_Work], settings: ReconSettings) -> tuple[str, list[Solid]]:
    """Kitchen run parts at the measured worktop height (structure of the plan kept)."""
    if work.family in ("kitchen_base", "worktop"):
        top = tops.get(work.key)
        if top is None and work.family == "worktop":
            top = _run_top_for(work, works, tops)  # a base standing back under its worktop has its own outline
    else:
        top = _run_top_for(work, works, tops)
    if top is None:
        return "none", []
    thickness = settings.worktop_thickness_m
    kind = str(work.item.catalogue_kind or "").lower()
    if work.family == "kitchen_base":
        return "kitchen", [extrusion("base", footprint_item, 0.0, top - thickness)]
    if work.family == "worktop":
        return "kitchen", [extrusion("worktop", footprint_item, top - thickness, top, source="measured")]
    if kind == "kitchen_sink":
        depth = 0.10
        if work.item.default_z_m:
            depth = max(0.05, float(work.item.default_z_m[1]) - float(work.item.default_z_m[0]))
        return "kitchen", [extrusion("sink", footprint_item, top - depth, top, source="prior")]
    if kind == "cooktop":
        height = 0.03
        if work.item.default_z_m:
            height = max(0.01, float(work.item.default_z_m[1]) - float(work.item.default_z_m[0]))
        return "kitchen", [extrusion("hob", footprint_item, top, top + height, source="prior")]
    return "none", []


def reconstruct_from_project_files(
    items: list[Any], *, frame: Any, floor_z: float, ceiling_z: float | None, units_per_metre: float,
    instances_path: Path | str | None, settings: ReconSettings | None = None,
) -> dict[str, Any] | None:
    """Plan hook: load ``classified.ply`` next to ``instances.json`` and reconstruct the items; None without a cloud."""
    settings = settings or load_recon_settings()
    if not settings.enabled or instances_path is None:
        return None
    ply = Path(instances_path).with_name("classified.ply")
    if not ply.is_file():
        return None
    cloud = load_scene_cloud(ply, frame, settings.classes)
    if cloud is None:
        return None
    from samcloud.bim.furniture_recon.discover import canonical_semantic_labels, discover_missing_items, load_instance_labels, merge_split_appliances

    relabelled = canonical_semantic_labels(items)
    added: list[Any] = []
    merged: list[dict[str, Any]] = []
    if settings.discover_instances:
        merged = merge_split_appliances(items, cloud, units_per_metre=units_per_metre)
        from samcloud.bim.plan_model import PlanFurniture

        added = discover_missing_items(
            items, cloud, load_instance_labels(instances_path), units_per_metre=units_per_metre, floor_z=floor_z, item_factory=PlanFurniture,
        )
    report = reconstruct_items(items, cloud, units_per_metre=units_per_metre, floor_z=floor_z, ceiling_z=ceiling_z, settings=settings)
    for item in added:  # keep the discovery provenance next to the fit metrics
        item.fit = {**(item.fit or {}), "added": "instance"}
    report["added_from_instances"] = [{"id": item.id, "label": item.semantic_label, "instance_id": item.instance_id} for item in added]
    report["semantic_labels_canonical"] = relabelled
    report["appliances_merged"] = merged
    return {key: value for key, value in report.items()}
