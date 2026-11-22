"""Combined multi-storey IFC of a project: one IfcBuilding with one IfcBuildingStorey per level.

Only written when a project has more than one level with a ``plan_model.json`` (single-level projects keep exactly their
per-level IFC).  Every level is built with :func:`samcloud.bim.scene_builder.build_scene` (same elements as its own IFC) and
the scenes are merged under the site / building of the lowest level.  The floor slab of a level gets an opening where a stair
of the level below arrives (``slab_openings_at_stairs``).

The levels must share the plan frame (origin, rotation, scale) and the units per metre (within 1 %); otherwise their metric
coordinates do not line up and the combined file is skipped with a reason.

Output: ``products/bim/building/<project>.ifc`` (+ ``building.json`` with counts and the per-storey elevations).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from samcloud.bim.core.model import Scene, SlabElement, StairElement
from samcloud.bim.plan_model import BuildingPlan, load_plan_model

BUILDING_DIRECTORY = ("products", "bim", "building")
STAIR_ARRIVAL_TOLERANCE_M = 0.35


def building_output_dir(project_root: Path | str) -> Path:
    return Path(project_root).joinpath(*BUILDING_DIRECTORY)


def level_plan_models(project_root: Path | str) -> list[BuildingPlan]:
    """All readable level plan models of the project (``products/dxf/<level>/ortho/plan_model.json``), lowest level first."""
    plans: list[BuildingPlan] = []
    for path in sorted(Path(project_root).joinpath("products", "dxf").glob("*/ortho/plan_model.json")):
        if path.parent.parent.name == "site":
            continue
        try:
            plan = load_plan_model(path)
        except (OSError, ValueError, KeyError):
            continue
        if plan.level_id != path.parent.parent.name:
            continue
        plans.append(plan)
    return sorted(plans, key=lambda plan: float(plan.floor_z) / max(float(plan.units_per_metre), 1e-9))


def _compatible(plans: list[BuildingPlan]) -> str | None:
    """None when all plans share frame and units; else the reason why they cannot be combined."""
    first = plans[0]
    for plan in plans[1:]:
        if abs(plan.units_per_metre - first.units_per_metre) > 0.01 * first.units_per_metre:
            return f"level {plan.level_id} uses {plan.units_per_metre:.4f} units per metre, level {first.level_id} {first.units_per_metre:.4f}"
        for key in ("origin", "rotation"):
            if not np.allclose(np.asarray(plan.frame.get(key), dtype=float), np.asarray(first.frame.get(key), dtype=float), atol=1e-6):
                return f"level {plan.level_id} has a different plan frame ({key})"
        if not math.isclose(float(plan.frame.get("scale_factor", 1.0)), float(first.frame.get("scale_factor", 1.0)), rel_tol=1e-6):
            return f"level {plan.level_id} has a different frame scale"
    return None


def _stair_footprint(stair: StairElement) -> Polygon:
    params = stair.params
    angle = math.radians(params.direction)
    d = np.asarray((math.cos(angle), math.sin(angle)))
    n = np.asarray((-d[1], d[0]))
    run = sum(max(flight.riser_count - 1, 0) * params.tread_length + flight.landing_length for flight in params.flights)
    start = np.asarray(params.start, dtype=float)
    half = params.width / 2.0
    return Polygon([tuple(start + d * a + n * b) for a, b in ((0.0, -half), (run + params.tread_length, -half), (run + params.tread_length, half), (0.0, half))])


def slab_openings_at_stairs(scene: Scene, margin_m: float = 0.05) -> int:
    """Cut the floor slab of the storey a stair arrives at (stair top within tolerance of its elevation); returns the cut count."""
    storeys = sorted(scene.storeys.values(), key=lambda storey: storey.elevation)
    cuts = 0
    for element in list(scene.elements.values()):
        if not isinstance(element, StairElement):
            continue
        own = scene.storeys.get(element.storey_id or "")
        if own is None:
            continue
        top = own.elevation + element.params.base_offset + element.params.step_count * element.params.riser_height
        upper = [storey for storey in storeys if storey.elevation > own.elevation + 0.5 and abs(storey.elevation - top) <= STAIR_ARRIVAL_TOLERANCE_M]
        if not upper:
            continue
        hole = _stair_footprint(element).buffer(margin_m, join_style=2)
        for key, slab in list(scene.elements.items()):
            if not isinstance(slab, SlabElement) or slab.storey_id != upper[0].id:
                continue
            outline = Polygon(slab.params.outline, [list(ring) for ring in slab.params.openings])
            if not outline.is_valid or not outline.intersects(hole):
                continue
            cut = outline.difference(hole)
            pieces = [cut] if cut.geom_type == "Polygon" else sorted(getattr(cut, "geoms", []), key=lambda piece: -piece.area)
            if not pieces or pieces[0].is_empty:
                continue
            piece = pieces[0]
            exterior = [(float(x), float(y)) for x, y in list(piece.exterior.coords)[:-1]]
            holes = [[(float(x), float(y)) for x, y in list(ring.coords)[:-1]] for ring in piece.interiors]
            scene.elements[key] = slab.model_copy(update={"params": slab.params.model_copy(update={"outline": exterior, "openings": holes})})
            cuts += 1
    return cuts


def combine_level_scenes(scenes: list[Scene]) -> Scene:
    """Merge per-level scenes (lowest first) into one scene with the site / building of the first."""
    first = scenes[0]
    site = next(iter(first.sites.values()))
    building = next(iter(first.buildings.values()))
    combined = Scene(settings=first.settings, sites={site.id: site}, buildings={building.id: building})
    for scene in scenes:
        for storey in scene.storeys.values():
            combined.storeys[storey.id] = storey.model_copy(update={"building_id": building.id})
        for element in scene.elements.values():
            update: dict[str, Any] = {}
            if element.building_id is not None:
                update["building_id"] = building.id
            if element.site_id is not None:
                update["site_id"] = site.id
            combined.add(element.model_copy(update=update) if update else element)
        for key, value in scene.types.items():
            combined.types.setdefault(key, value)
    frame = dict(first.derived.get("samcloud_frame") or {})
    frame["LevelId"] = "building"
    frame["LevelName"] = ", ".join(str(storey.name) for storey in sorted(combined.storeys.values(), key=lambda item: item.elevation))
    combined.derived["samcloud_frame"] = frame
    return combined


def export_building_ifc(project_root: Path | str) -> dict[str, Any] | None:
    """Write the combined IFC when the project has more than one level; None for single-level projects."""
    from samcloud.bim.core.ifc import write_ifc
    from samcloud.bim.scene_builder import build_scene

    plans = level_plan_models(project_root)
    if len(plans) < 2:
        return None
    out_dir = building_output_dir(project_root)
    reason = _compatible(plans)
    if reason is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {"written": False, "reason": reason, "levels": [plan.level_id for plan in plans]}
        (out_dir / "building.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    scenes = [build_scene(plan) for plan in plans]
    combined = combine_level_scenes(scenes)
    cuts = slab_openings_at_stairs(combined)
    path = out_dir / f"{Path(project_root).name}.ifc"
    result = write_ifc(combined, path)
    summary = {
        "written": True, "ifc": path.name, "levels": [plan.level_id for plan in plans],
        "storeys": [{"name": storey.name, "elevation_m": round(storey.elevation, 4)} for storey in sorted(combined.storeys.values(), key=lambda item: item.elevation)],
        "slab_openings_at_stairs": cuts, "counts": dict(result.counts), "warnings": list(result.warnings)[:50], "incomplete": bool(result.incomplete),
    }
    (out_dir / "building.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

