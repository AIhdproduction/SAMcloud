"""Resumable end-to-end outdoor environment pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import time
import traceback
import threading
from typing import Any

import numpy as np
import psutil

from samcloud.drone import contracts as c
from samcloud.drone.contracts import PointSet, load_drone_config

LOGGER = logging.getLogger(__name__)

STEPS = ("load", "semantics", "terrain", "vegetation", "pavement", "tin", "buildings", "site", "scene", "export", "qa", "report")
DEPENDENCIES = {"semantics": ("load",), "terrain": ("load",), "vegetation": ("terrain",), "pavement": ("terrain",),
                "tin": ("terrain", "pavement"), "buildings": ("terrain", "semantics"), "site": ("terrain",),
                "scene": ("vegetation", "pavement", "tin", "buildings", "site"), "export": ("scene",), "qa": ("export",), "report": ("qa",)}


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh: pickle.dump(value, fh)


def _load(path: Path, default=None):
    try:
        with path.open("rb") as fh: return pickle.load(fh)
    except FileNotFoundError: return default


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


class _PeakRssMonitor:
    """Sample process RSS while a pipeline step runs."""

    def __init__(self, process: psutil.Process) -> None:
        self.process = process
        self.peak = process.memory_info().rss
        self.running = False
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while self.running:
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(.05)

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def stop(self) -> float:
        self.running = False
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        return self.peak / (1024 * 1024)


def _sampler(report: dict):
    dtm = np.asarray(report["dtm"], float); bounds = tuple(report["bounds"]); resolution = float(report["resolution"])
    if not np.isfinite(dtm).all():
        from scipy import ndimage
        valid = np.isfinite(dtm)
        if not valid.any():
            raise ValueError("DTM has no finite samples")
        nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        dtm = dtm[tuple(nearest)]
    def sample(xy, y=None):
        scalar = y is not None and np.asarray(xy).ndim == 0 and np.asarray(y).ndim == 0
        values = np.asarray((xy, y) if y is not None else xy, float).reshape(-1, 2)
        col = np.clip(np.rint((values[:, 0] - bounds[0]) / resolution).astype(int), 0, dtm.shape[1] - 1)
        row = np.clip(np.rint((bounds[3] - values[:, 1]) / resolution).astype(int), 0, dtm.shape[0] - 1)
        sampled = dtm[row, col]
        return float(sampled[0]) if scalar else sampled
    return sample


def _dtm_support(report: dict, support_spacing_m: float | None = None) -> np.ndarray:
    """Return every valid DTM cell centre as the terrain-TIN candidate surface."""
    dtm = np.asarray(report["dtm"], dtype=float)
    validity = np.asarray(report["validity"])
    xmin, _ymin, _xmax, ymax = map(float, report["bounds"])
    resolution = float(report["resolution"])
    rows, cols = np.nonzero((validity > 0) & np.isfinite(dtm))
    if len(rows) < 3:
        raise ValueError("DTM has fewer than three valid support cells")
    xy = np.column_stack((xmin + (cols + 0.5) * resolution, ymax - (rows + 0.5) * resolution))
    return np.column_stack((xy, dtm[rows, cols]))


def _thin_pointset_voxels(pointset: PointSet, voxel_m: float) -> PointSet:
    """Keep one deterministic representative per 3-D voxel."""
    if voxel_m <= 0:
        raise ValueError("working voxel size must be positive")
    xyz = np.asarray(pointset.xyz, dtype=np.float64)
    if len(xyz) == 0:
        return pointset.subset(np.empty(0, dtype=np.int64))
    lower = xyz.min(axis=0)
    cells = np.floor((xyz - lower) / float(voxel_m)).astype(np.int64)
    _unique, selected = np.unique(cells, axis=0, return_index=True)
    return pointset.subset(np.sort(selected))


def _working_cloud(pointset: PointSet, environment: dict) -> tuple[PointSet, float | None]:
    """Build the bounded working cloud from the boundary-clipped source cloud."""
    maximum = int(environment.get("max_working_points", 8_000_000))
    if maximum < 1:
        raise ValueError("environment.max_working_points must be at least one")
    setting = environment.get("working_voxel_m", "auto")
    if len(pointset) <= maximum and setting == "auto":
        return pointset, None
    if setting != "auto":
        voxel = float(setting)
        return _thin_pointset_voxels(pointset, voxel), voxel
    voxel = float(environment.get("min_working_voxel_m", 0.03))
    if voxel <= 0:
        raise ValueError("environment.min_working_voxel_m must be positive")
    while True:
        working = _thin_pointset_voxels(pointset, voxel)
        if len(working) <= maximum:
            return working, voxel
        voxel *= 1.25


def _working_pointset(cache: Path) -> PointSet:
    """Return semantic classes when available, otherwise the raw working cloud."""
    semantic = _load(cache / "semantics.pkl")
    if isinstance(semantic, dict) and isinstance(semantic.get("pointset"), PointSet):
        return semantic["pointset"]
    return _load(cache / "load.pkl")[0]


def _step_load(project: Path, out: Path, cfg: dict, cache: Path):
    from samcloud.drone.io import load_classified_pointset_lv95
    ps, frame = load_classified_pointset_lv95(project / "products", project_dir=project)
    info = {"points": len(ps)}
    from samcloud.drone import perimeter as per
    poly, binfo = per.processing_boundary(project, frame, cfg)
    (cache / "boundary.pkl").parent.mkdir(parents=True, exist_ok=True)
    _dump(cache / "boundary.pkl", poly)
    if poly is not None:
        before = len(ps)
        ps, _mask = per.clip_pointset(ps, poly, 0.0)
        per.write_perimeter_geojson(poly, frame, out / "boundary.geojson")
        info["boundary"] = {**binfo, "points_before": before, "points_after": len(ps), "geojson": str(out / "boundary.geojson")}
        info["points"] = len(ps)
    elif binfo.get("mode") != "none":
        info["boundary"] = binfo
    full = ps
    working, voxel = _working_cloud(full, cfg.get("environment", {}))
    _dump(cache / "load.pkl", (working, frame))
    if voxel is not None:
        _dump(cache / "load_full.pkl", (full, frame))
    info.update({"points": len(working), "points_full": len(full), "points_working": len(working),
                 "working_voxel_m": voxel})
    return info


def _step_semantics(project, out, cfg, cache):
    ps, frame = _load(cache / "load.pkl")
    try:
        from samcloud.drone.semantics import run_semantics
        result = run_semantics(project / "products", out / "semantics", config=cfg, pointset=ps, frame=frame)
    except ImportError:
        result = {"pointset": ps, "instances": []}
    _dump(cache / "semantics.pkl", result); return {"available": True}


def _step_terrain(project, out, cfg, cache):
    from samcloud.terrain.rasters import export_terrain
    from samcloud.terrain.clean import clean_dtm, protect_mask
    ps, frame = _working_pointset(cache), _load(cache / "load.pkl")[1]
    raw = export_terrain(ps, out, frame, cfg)
    # Retain the arrays needed by later steps even when the public report only lists files.
    terrain = dict(raw)
    ground = None
    if not isinstance(terrain.get("dtm"), np.ndarray):
        from samcloud.terrain.rasters import build_dtm_raster, ground_mask
        ground = ps.xyz[ground_mask(ps, method="pmf")]
        resolution = float(cfg.get("terrain", {}).get("dtm_resolution_m", .25))
        lo, hi = ps.xyz[:, :2].min(0), ps.xyz[:, :2].max(0)
        bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
        terrain["dtm"], terrain["validity"], _ = build_dtm_raster(ground, resolution=resolution, bounds=bounds)
        terrain.update(bounds=bounds, resolution=resolution)
    # residual non-ground bumps are re-interpolated; pavement/curb cells and long steps stay untouched
    clean_cfg = cfg.get("terrain", {}).get("clean", {})
    dtm_raw = np.asarray(terrain["dtm"], dtype=float)
    pave_codes = (c.ROAD, c.SIDEWALK, c.PATH, c.CURB, c.MANHOLE, c.SEALED_TERRAIN, c.ROAD_MARKING)
    protect = protect_mask(ps.xyz[np.isin(ps.classes, pave_codes)], terrain["bounds"], float(terrain["resolution"]),
                           dtm_raw.shape, float(clean_cfg.get("protect_dilate_m", .5)))
    dtm_clean, clean_report = clean_dtm(dtm_raw, terrain["validity"], float(terrain["resolution"]), protect=protect, config=cfg)
    terrain.update(dtm_raw=dtm_raw, dtm_clean=dtm_clean, dtm=dtm_clean.copy(), clean_report=clean_report)
    if ground is not None:
        step = int(np.ceil(len(ground) / int(cfg.get("terrain", {}).get("deviation", {}).get("max_points", 400_000))))
        terrain["ground_sample"] = np.asarray(ground[::max(step, 1)], dtype=np.float64)
    _dump(cache / "terrain.pkl", terrain); return {"files": terrain.get("files", {}), "clean": clean_report}


def _step_vegetation(project, out, cfg, cache):
    from samcloud.environment.vegetation import detect_shrubs, detect_trees
    from samcloud.outdoor_model.vegetation import build_vegetation
    ps, _frame = _working_pointset(cache), _load(cache / "load.pkl")[1]; terrain = _load(cache / "terrain.pkl"); sample = _sampler(terrain)
    trees, shrubs = detect_trees(ps, sample, config=cfg), detect_shrubs(ps, sample)
    result = build_vegetation(trees, shrubs, (), sample, config=cfg)
    _dump(cache / "vegetation.pkl", result); return {"trees": len(trees), "shrubs": len(shrubs)}


def _step_pavement(project, out, cfg, cache):
    from samcloud.environment.pavement import lawn_polygons, pavement_polygons
    from samcloud.environment.curbs import curb_axes
    ps, _frame = _working_pointset(cache), _load(cache / "load.pkl")[1]
    result = {"polygons": pavement_polygons(ps), "lawns": lawn_polygons(ps), "curbs": curb_axes(ps)}
    _dump(cache / "pavement.pkl", result); return {k: len(v) for k, v in result.items()}


def _apply_edge_fit(topo, pave: dict, cfg: dict):
    """Replace the cleaned polygons and curb axes by their line/arc fits (shared edges are fitted once)."""
    from dataclasses import replace
    from samcloud.environment import topology as env_topology
    from samcloud.terrain.edges import fit_surfaces
    kinds = {k: [p for p in topo.polygons if p.kind == k] for k in c.PAVEMENT_KINDS}
    regions = {**{k: [p.polygon for p in v] for k, v in kinds.items()}, "lawn": list(topo.lawns)}
    fit = fit_surfaces(regions, pave.get("curbs", ()), config=cfg)
    if "fallback" in fit.report:
        return fit
    polygons = []
    for kind, items in kinds.items():
        for item, faces in zip(items, fit.regions[kind]):
            for n, face in enumerate(faces, start=1):
                polygons.append(replace(item, polygon=face, pavement_id=item.pavement_id if len(faces) == 1 else f"{item.pavement_id}_{n}"))
    topo.polygons = polygons
    topo.lawns = [face for faces in fit.regions["lawn"] for face in faces]
    named = {k: [p.polygon for p in polygons if p.kind == k] for k in c.PAVEMENT_KINDS}
    named["lawn"] = topo.lawns
    topo.report = {f"{a}|{b}": r for (a, b), r in env_topology.pairwise_report({k: v for k, v in named.items() if v}, tolerance=0.05).items()}
    by_id = {fc.curb_id: fc for fc in fit.curbs}
    pave["curbs"] = [replace(cb, points=by_id[cb.curb_id].dense_xyz) if cb.curb_id in by_id else cb for cb in pave.get("curbs", ())]
    return fit


def _label_stats(tin) -> dict:
    labels = np.asarray(tin.labels); tris = np.asarray(tin.triangles)
    return {name: {"triangles": int((labels == code).sum()),
                   "vertices": int(len(np.unique(tris[labels == code]))) if (labels == code).any() else 0}
            for name, code in c.TIN_LABELS.items()}


def _step_tin(project, out, cfg, cache):
    from samcloud.outdoor_model.pavement import pavement_topology
    from samcloud.terrain.clean import smooth_pavement_surfaces, tin_deviation
    from samcloud.terrain.tin import build_labeled_tin
    _ps, _frame = _working_pointset(cache), _load(cache / "load.pkl")[1]; terrain = _load(cache / "terrain.pkl"); pave = _load(cache / "pavement.pkl")
    topo = pavement_topology(pave["polygons"], pave["lawns"], pave["curbs"], config=cfg)
    fit = None
    if bool(cfg.get("edges", {}).get("enabled", False)):
        fit = _apply_edge_fit(topo, pave, cfg)
    pave.update(polygons=topo.polygons, lawns=topo.lawns, topology_qa=topo.dropped, edge_fit=fit,
                edge_report=getattr(fit, "report", None))
    _dump(cache / "pavement.pkl", pave)
    # smooth the pavement surfaces on the bump-cleaned DTM (the boundary clip of the terrain step is kept)
    if isinstance(terrain.get("dtm_clean"), np.ndarray):
        base = np.where(np.isfinite(np.asarray(terrain["dtm"], float)), terrain["dtm_clean"], np.nan)
    else:
        base = np.asarray(terrain["dtm"], float)
    smooth_report = {"enabled": False}
    if cfg.get("terrain", {}).get("smooth", {}).get("enabled", True):
        base, smooth_report = smooth_pavement_surfaces(base, terrain["bounds"], float(terrain["resolution"]), topo.polygons, config=cfg)
    terrain["dtm"] = base; terrain["smooth_report"] = smooth_report
    _dump(cache / "terrain.pkl", terrain)
    sample = _sampler(terrain)
    regions = topo.regions()
    tin_cfg = cfg.get("terrain", {}).get("tin", {})
    ground = _dtm_support(terrain, tin_cfg.get("support_spacing_m"))
    tol_terrain = float(tin_cfg.get("tolerance_terrain_m", tin_cfg.get("tolerance_m", .04)))
    tol_pavement = float(tin_cfg.get("tolerance_pavement_m", tol_terrain))
    by_label = {"road": tol_pavement, "sidewalk": tol_pavement, "path": tol_pavement, "lawn": tol_terrain, "terrain": tol_terrain}
    started = time.perf_counter()
    tin = build_labeled_tin(ground, regions=regions, dtm_sampler=sample,
        buffer=float(tin_cfg.get("constraint_buffer_m", .05)), min_angle_deg=float(tin_cfg.get("min_angle_deg", 20)),
        max_steiner_per_edge=int(tin_cfg.get("max_steiner_per_edge", 50)),
        thin_spacing=tin_cfg.get("thin_spacing_m"), tolerance=tol_terrain, tolerance_by_label=by_label,
        initial_spacing=tin_cfg.get("initial_spacing_m"), max_vertices=tin_cfg.get("max_vertices"),
        max_refinement_rounds=int(tin_cfg.get("max_refinement_rounds", 2)))
    _dump(cache / "tin.pkl", tin)
    dev_cfg = cfg.get("terrain", {}).get("deviation", {})
    deviation = tin_deviation(tin, terrain["ground_sample"], max_points=int(dev_cfg.get("max_points", 400_000))) if isinstance(terrain.get("ground_sample"), np.ndarray) else {}
    for name, row in deviation.get("labels", {}).items():
        limit = float(dev_cfg.get("max_rms_pavement_m", .04)) if name in c.PAVEMENT_KINDS else float(dev_cfg.get("max_rms_terrain_m", .06))
        row["limit_rms_m"] = limit; row["ok"] = bool(row["rms_m"] <= limit)
    report = {"clean": terrain.get("clean_report"), "smooth": smooth_report, "deviation": deviation, "labels": _label_stats(tin),
              "edges": getattr(fit, "report", None), "topology_worst_cm2": topo.worst_cm2()}
    (out / "terrain_clean_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return {"triangles": len(tin.triangles), "vertices": len(tin.vertices),
            "simplify_report": getattr(tin, "simplify_report", None), "seconds": round(time.perf_counter() - started, 3),
            "support_cells": int(len(ground)), "constraint_report": getattr(tin, "constraint_report", None),
            "deviation": deviation, "labels": report["labels"], "edges": report["edges"], "smooth": smooth_report}


def _building_lod1_fallback(inst, points: PointSet, building_id: int, sampler, reason: str):
    """Create a recoverable LoD1 envelope when one building derivation fails."""
    from shapely.geometry import Polygon
    from samcloud.buildings.extrude import extrude_lod1
    from samcloud.drone.contracts import BuildingModel

    footprint = getattr(inst, "footprint", None)
    if not isinstance(footprint, Polygon) or footprint.is_empty or footprint.area <= 1e-8:
        return None
    ground = float(sampler(footprint.representative_point().x, footprint.representative_point().y))
    height = max(float(np.nanpercentile(np.asarray(points.xyz, dtype=float)[:, 2], 95) - ground), 1.0)
    vertices, faces, roles = extrude_lod1(footprint, height, ground)
    if len(vertices) < 4 or len(faces) == 0:
        return None
    return BuildingModel(building_id, "1.2", vertices, faces, roles, "lod1_fallback", reason,
                         heights={"ground": ground, "height": height})


def _step_buildings(project, out, cfg, cache):
    import rasterio
    from shapely.geometry import Point
    from samcloud.outdoor_model.buildings import build_building_elements
    from samcloud.terrain.tin import build_labeled_tin
    from samcloud.buildings.reconstruct import reconstruct_building
    from samcloud.buildings.facades import facade_surfaces
    from samcloud.buildings.openings import detect_building_openings
    from samcloud.drone import contracts as c
    ps, _frame = _working_pointset(cache), _load(cache / "load.pkl")[1]; terrain = _load(cache / "terrain.pkl"); sample = _sampler(terrain)
    semantic = _load(cache / "semantics.pkl", {})
    instances = semantic.get("instances", semantic.get("buildings", [])) if isinstance(semantic, dict) else []
    dsm_points = np.empty((0, 3), dtype=float)
    dsm_resolution = float(cfg.get("terrain", {}).get("dsm_resolution_m", 0.10))
    try:
        with rasterio.open(out / "dsm.tif") as source:
            dsm = source.read(1, masked=True).filled(np.nan).astype(float)
            transform = source.transform
            dsm_resolution = float(abs(transform.a))
        validity_path = out / "dsm_validity.tif"
        measured = np.isfinite(dsm)
        if validity_path.is_file():
            with rasterio.open(validity_path) as source:
                measured &= source.read(1) == 1
        rows, columns = np.nonzero(measured)
        east, north = transform * (columns + 0.5, rows + 0.5)
        dsm_points = _frame.from_lv95(np.column_stack((east, north, dsm[rows, columns])))
    except Exception as exc:
        LOGGER.warning("DSM roof sampling is unavailable, falling back to point-cloud roofs: %s", exc)
    boundary = _load(cache / "boundary.pkl", None)
    models=[]; opens={}; facades={}; qa=[]; facade_reports={}; failure_reasons=[]; roof_reports={}
    fallback_count = dropped_count = 0
    for inst in instances:
        bid = int(getattr(inst, "building_id", -1))
        building_points = ps.subset(inst.point_indices)
        source_footprint = getattr(inst, "footprint", None)
        if source_footprint is not None and not source_footprint.is_empty and len(dsm_points):
            min_x, min_y, max_x, max_y = source_footprint.bounds
            dsm_mask = (dsm_points[:, 0] >= min_x - 1.0) & (dsm_points[:, 0] <= max_x + 1.0) & (dsm_points[:, 1] >= min_y - 1.0) & (dsm_points[:, 1] <= max_y + 1.0)
            building_dsm = dsm_points[dsm_mask]
        else:
            building_dsm = np.empty((0, 3), dtype=float)
        boundary_contacts = 0
        if source_footprint is not None and boundary is not None:
            ring = list(source_footprint.exterior.coords)[:-1]
            boundary_contacts = sum(boundary.boundary.distance(Point(x, y)) <= dsm_resolution * 1.5 for x, y in ring)
            cut_by_boundary = bool(ring and boundary_contacts / len(ring) >= 0.45)
        else:
            cut_by_boundary = False
        try:
            model = reconstruct_building(
                building_points,
                bid,
                dtm_sampler=sample,
                config=cfg,
                dsm_roof_points=building_dsm,
                dsm_cell_area_m2=dsm_resolution * dsm_resolution,
                cut_by_boundary=cut_by_boundary,
            )
            if model.status == "failed":
                if model.reason.startswith("plausibility rejected:"):
                    plausibility = model.heights.get("plausibility", {})
                    qa.append(c.QAEntry("building", f"building-{bid}", model.reason, plausibility))
                    roof_reports[str(bid)] = {"status": "rejected", "reason": model.reason, "roof_points": len(building_dsm), "plausibility": plausibility}
                    dropped_count += 1
                    continue
                raise RuntimeError(model.reason or "reconstruction failed")
            facade_report: dict = {}
            surfaces = facade_surfaces(model, getattr(inst, "facades", []), report=facade_report)
            detected, opening_qa = detect_building_openings(building_points, model, getattr(inst, "facades", []), dtm_sampler=sample, config=cfg)
            models.append(model); facades[bid] = surfaces; opens[bid] = detected; qa.extend(opening_qa)
            facade_reports[bid] = facade_report
            plausibility = model.heights.get("plausibility", {})
            roof_reports[str(bid)] = {
                "status": model.status,
                "faces": int(len(model.faces)),
                "roof_rmse": model.rmse,
                "rmse_lod2": model.heights.get("rmse_lod2"),
                "rmse_all": model.heights.get("rmse_all"),
                "superstructures": model.heights.get("superstructures", {}),
                "roof_planes": model.heights.get("roof_planes", []),
                "partition_cells": model.heights.get("partition_cells", 0),
                "labels": model.heights.get("labels", []),
                "roof_points": len(building_dsm),
                "plausibility": plausibility,
                "regularization": model.heights.get("regularization", {}),
            }
            if plausibility.get("cut_by_boundary"):
                qa.append(c.QAEntry("building", f"building-{bid}", "cut_by_boundary", plausibility))
            if facade_report.get("degenerate_wall_triangles", 0):
                qa.append(c.QAEntry("facade", f"building-{bid}", "degenerate wall triangles skipped", facade_report))
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            failure_reasons.append(f"building {bid}: {message}")
            fallback = _building_lod1_fallback(inst, building_points, bid, sample, f"LoD1 fallback after: {message}")
            if fallback is None:
                qa.append(c.QAEntry("building", f"building-{bid}", message))
                dropped_count += 1
                continue
            models.append(fallback); facades[bid] = []; opens[bid] = []
            qa.append(c.QAEntry("building", f"building-{bid}", f"LoD1 fallback after: {message}"))
            fallback_count += 1
    elements = []
    for model in models:
        try:
            built, more = build_building_elements([model], opens, facades, config=cfg)
            elements.extend(built); qa.extend(more)
        except Exception as exc:
            qa.append(c.QAEntry("building", f"building-{model.building_id}", f"element build failed: {exc}"))
            dropped_count += 1
    _dump(cache / "buildings.pkl", {"elements": elements, "qa": qa})
    produced = sum(type(item).__name__ == "BuildingElement" for item in elements)
    if instances and not produced:
        detail = "; ".join(failure_reasons) or "no reconstruction result was usable"
        raise RuntimeError(f"no building could be produced: {detail}")
    return {"buildings": produced, "fallback": fallback_count, "dropped": dropped_count, "qa": len(qa),
            "facade_report": facade_reports, "roof_report": roof_reports}


def _step_site(project, out, cfg, cache):
    from samcloud.outdoor_model.site_walls import build_site_elements, fence_features, site_wall_features
    from samcloud.outdoor_model.civil import build_civil_elements, manhole_detections, marking_features
    ps, _frame = _working_pointset(cache), _load(cache / "load.pkl")[1]; sample = _sampler(_load(cache / "terrain.pkl")); pave = _load(cache / "pavement.pkl")
    site, qa = build_site_elements(site_wall_features(ps, sample), fence_features(ps, sample), config=cfg)
    civil, civil_qa = build_civil_elements(pave["curbs"], marking_features(ps), manhole_detections(ps), config=cfg)
    _dump(cache / "site.pkl", {"elements": site + civil, "qa": qa + civil_qa}); return {"elements": len(site) + len(civil)}


def _boundary_dxf_entities(boundary, frame) -> dict:
    """Closed LWPOLYLINE(s) of the processing boundary on layer S-ANNO-GRENZE (LV95 like all DXF entities)."""
    if boundary is None or boundary.geom_type not in ("Polygon", "MultiPolygon"): return {}
    rings = [boundary] if boundary.geom_type == "Polygon" else list(boundary.geoms)
    items = []
    for ring in rings:
        xy = np.asarray(ring.exterior.coords)[:-1]
        items.append({"type": "lwpolyline", "closed": True, "elevation": float(frame.origin_lv95[2]),
                      "points": frame.to_lv95(np.column_stack([xy, np.zeros(len(xy))])).tolist()})
    return {"S-ANNO-GRENZE": items}


def _step_scene(project, out, cfg, cache):
    from samcloud.outdoor_model.scene import assemble_dxf, assemble_scene
    from samcloud.outdoor_model import buildings, site_walls, vegetation, pavement, civil, terrain as terrain_builder
    ps, frame = _working_pointset(cache), _load(cache / "load.pkl")[1]; veg = _load(cache / "vegetation.pkl"); pave = _load(cache / "pavement.pkl"); tin = _load(cache / "tin.pkl"); bld = _load(cache / "buildings.pkl"); site = _load(cache / "site.pkl"); terrain_data = _load(cache / "terrain.pkl")
    pavement_elements, pavement_qa = pavement.build_pavement_elements(
        pave["polygons"], pave["lawns"], pave["curbs"], tin, config=cfg, topology_ready=True,
    )
    terrain_elements, terrain_qa = terrain_builder.build_terrain_elements(tin, dtm=terrain_data["dtm"], validity=terrain_data["validity"], resolution=terrain_data["resolution"], config=cfg)
    lawn_qa = []
    lawn_areas = vegetation.build_lawn_areas(pave["lawns"], _sampler(terrain_data), qa=lawn_qa)
    lawn_elements = vegetation.lawn_elements(lawn_areas)
    groups=[veg.elements, lawn_elements, pavement_elements, terrain_elements, bld["elements"], site["elements"]]
    scene=assemble_scene(*groups, frame=frame)
    manholes = [tuple(item.params.center[:2]) for item in site["elements"] if hasattr(item.params, "center")]
    terrain_dxf, _terrain_info = terrain_builder.terrain_dxf_entities(terrain_elements, frame,
        dtm=terrain_data["dtm"], validity=terrain_data["validity"], dtm_bounds=tuple(terrain_data["bounds"]),
        resolution=terrain_data["resolution"], curbs=pave["curbs"], manholes=manholes, config=cfg)
    entities={"vegetation": vegetation.to_dxf_entities(veg.elements, frame, lawn_areas=lawn_areas), "pavement": pavement.to_dxf_entities(pavement_elements, frame, config=cfg, edge_fit=pave.get("edge_fit")), "terrain": terrain_dxf, "buildings": buildings.to_dxf_entities(bld["elements"], frame), "site": site_walls.to_dxf_entities(site["elements"], frame), "civil": civil.to_dxf_entities(site["elements"], frame, config=cfg)}
    fitted_curbs = getattr(pave.get("edge_fit"), "curbs", None)  # curbs as lines/arcs (bulges), fitted with the outlines
    if fitted_curbs: entities["civil"] = {**entities["civil"], **pavement.curb_dxf_entities(fitted_curbs, frame, config=cfg)}
    boundary_dxf = _boundary_dxf_entities(_load(cache / "boundary.pkl"), frame)
    if boundary_dxf: entities["boundary"] = boundary_dxf
    dxf=assemble_dxf(entities, frame, cfg); _dump(cache / "scene.pkl", {"scene": scene, "dxf": dxf, "qa": pave.get("topology_qa", []) + pavement_qa + lawn_qa + terrain_qa + bld["qa"] + site["qa"]}); return {"elements": len(scene.elements)}


def _element_category(element) -> str:
    """Map model elements to the viewer's stable, explicit mesh categories."""
    name = type(element).__name__.lower()
    if "building" in name or name in {"windowelement", "doorelement"}:
        return "building"
    if "terrain" in name:
        return "terrain"
    if "pavement" in name:
        return "pavement"
    if "tree" in name or "shrub" in name or "vegetation" in name:
        return "vegetation"
    return "other"


def _dsm_mesh(path: Path, frame) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a decimated local DSM grid mesh, capped at 200k triangles."""
    import rasterio

    with rasterio.open(path) as src:
        dsm = src.read(1, masked=True).filled(np.nan).astype(float)
        transform = src.transform
    rows, cols = dsm.shape
    if rows < 2 or cols < 2:
        return None
    step = max(1, int(np.ceil(np.sqrt((2 * (rows - 1) * (cols - 1)) / 200_000))))
    rr = np.arange(0, rows, step)
    cc = np.arange(0, cols, step)
    if rr[-1] != rows - 1: rr = np.r_[rr, rows - 1]
    if cc[-1] != cols - 1: cc = np.r_[cc, cols - 1]
    grid = dsm[np.ix_(rr, cc)]
    valid = np.isfinite(grid)
    if valid.sum() < 4:
        return None
    cc_grid, rr_grid = np.meshgrid(cc, rr)
    east, north = transform @ (cc_grid + .5, rr_grid + .5)
    world = np.column_stack((east.ravel(), north.ravel(), grid.ravel()))
    vertices = frame.from_lv95(world)
    width = len(cc)
    faces = []
    for row in range(len(rr) - 1):
        for col in range(width - 1):
            quad = np.array([row * width + col, row * width + col + 1, (row + 1) * width + col, (row + 1) * width + col + 1])
            if np.isfinite(world[quad, 2]).all():
                faces.extend(((quad[0], quad[2], quad[1]), (quad[1], quad[2], quad[3])))
    return _finite_mesh(vertices, np.asarray(faces, dtype=np.int64)) if faces else None


def _finite_mesh(vertices, faces) -> tuple[np.ndarray, np.ndarray] | None:
    """Drop faces touching non-finite vertices, remove unreferenced vertices and reindex."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        return None
    ok = np.isfinite(vertices).all(axis=1)
    if ((faces < 0) | (faces >= len(vertices))).any():
        return None
    faces = faces[ok[faces].all(axis=1)]
    if len(faces) == 0:
        return None
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def _write_environment_glb(nodes, path: Path, frame) -> None:
    import trimesh

    scene = trimesh.Scene()
    nodes = dict(nodes)
    for name in list(nodes):
        vertices, faces, category = nodes[name]
        cleaned = _finite_mesh(vertices, faces)
        if cleaned is None:
            LOGGER.warning("environment GLB: skipping node %s (no finite geometry)", name)
            del nodes[name]
        else:
            nodes[name] = (*cleaned, category)
    for name, (vertices, faces, category) in nodes.items():
        scene.add_geometry(trimesh.Trimesh(vertices=vertices, faces=faces, process=False), node_name=name, geom_name=name)

    origin = [float(value) for value in frame.origin_lv95]
    def decorate(document):
        document.setdefault("asset", {}).setdefault("extras", {}).update({"origin_lv95": origin, "epsg": int(frame.epsg), "vertical_datum": frame.vertical_datum})
        for node in document.get("nodes", []):
            if "mesh" in node:
                node.setdefault("extras", {})["category"] = nodes[str(node.get("name"))][2]
    path.write_bytes(trimesh.exchange.gltf.export_glb(scene, tree_postprocessor=decorate))


def _transfer_classes_nearest(full: PointSet, working: PointSet, *, chunk_size: int = 2_000_000) -> PointSet:
    """Copy refined working-cloud classes to every full-resolution point."""
    from scipy.spatial import cKDTree

    result = full.subset(np.arange(len(full)))
    tree = cKDTree(np.asarray(working.xyz, dtype=np.float64))
    for start in range(0, len(result), chunk_size):
        stop = min(start + chunk_size, len(result))
        indices = tree.query(result.xyz[start:stop], workers=-1)[1]
        result.classes[start:stop] = working.classes[indices]
    return result


def _step_export(project, out, cfg, cache):
    from samcloud.bim.core.ifc.writer import write_ifc
    from samcloud.bim.core.model import BuildingElement
    from samcloud.drone.export import export_mesh, export_point_cloud
    from samcloud.dxf.writer import build_dxf_document, ensure_outdoor_layers
    ps, frame = _working_pointset(cache), _load(cache / "load.pkl")[1]; data = _load(cache / "scene.pkl")
    full = _load(cache / "load_full.pkl")
    source = "working"
    if full is not None:
        ps = _transfer_classes_nearest(full[0], ps)
        source = "full_transferred"
    clip_info = None
    boundary = _load(cache / "boundary.pkl", None)
    if boundary is not None and bool(cfg.get("environment", {}).get("clip_point_cloud", True)):
        from samcloud.drone import perimeter as per
        before = len(ps)
        ps, _mask = per.clip_pointset(ps, boundary, 0.0)  # local frame polygon, local frame cloud
        clip_info = {"points_before": before, "points_after": len(ps)}
        if len(ps) == 0:
            raise ValueError("processing boundary contains no points; cannot export classified.laz")
    ifc_path=out / "outdoor.ifc"; dxf_path=out / "outdoor.dxf"; laz_path=out / "classified.laz"; glb_path=out / "environment.glb"
    write_ifc(data["scene"], ifc_path)
    doc=build_dxf_document(data["dxf"], coordinates="local", local_origin=bool(cfg.get("dxf", {}).get("local_origin", False)), origin_offset=frame.origin_lv95); ensure_outdoor_layers(doc, tin_frozen=bool(cfg.get("dxf", {}).get("tin_layer_frozen", True))); doc.saveas(dxf_path)
    extras={name: getattr(ps, name) for name in ("confidence", "margin", "view_count", "adjusted") if getattr(ps, name) is not None}
    extras.update({"hag": np.zeros(len(ps), np.float32), "building_id": np.zeros(len(ps), np.int32), "tree_id": np.zeros(len(ps), np.int32)})
    export_point_cloud(ps, laz_path, frame, extra=extras)
    nodes = {}
    for element in data["scene"].elements.values():
        if getattr(element.params, "triangles", None) and getattr(element.params, "vertices", None):
            vertices=np.asarray(element.params.vertices, float); faces=np.asarray(element.params.triangles, int)
            nodes[element.id]=(vertices, faces, _element_category(element))
            if isinstance(element, BuildingElement): export_mesh(vertices, faces, out / f"{element.id}.obj", frame, node_name=element.id)
    dsm = _dsm_mesh(out / "dsm.tif", frame)
    if dsm is not None:
        nodes["dsm-surface"] = (*dsm, "dsm")
    if nodes: _write_environment_glb(nodes, glb_path, frame)
    else: glb_path.write_bytes(b"glTF")
    published = None
    publish_error = None
    try:
        from samcloud.drone.publish import publish_environment_exports
    except ImportError:
        LOGGER.info("environment export publishing is unavailable")
    else:
        try:
            published = publish_environment_exports(project, {"ifc": ifc_path, "dxf": dxf_path, "glb": glb_path, "laz": laz_path})
        except Exception as exc:
            publish_error = str(exc)
            LOGGER.warning("environment export publishing failed: %s", exc)
    result = {"ifc": str(ifc_path), "dxf": str(dxf_path), "laz": str(laz_path), "glb": str(glb_path),
              "nodes": len(nodes), "point_source": source, "published": published}
    if clip_info is not None:
        result["point_cloud_clip"] = clip_info
    if publish_error is not None:
        result["publish_error"] = publish_error
    return result


def _step_qa(project, out, cfg, cache):
    from samcloud.drone.export import validate_exports, lv95_sample_check
    from samcloud.drone.qa_cad import check_dxf, check_ifc, count_consistency, write_qa_report
    from samcloud.outdoor_model import buildings, civil, pavement, vegetation
    ps, frame = _working_pointset(cache), _load(cache / "load.pkl")[1]; data=_load(cache / "scene.pkl")
    elements = list(data["scene"].elements.values())
    counts = {}
    for getter in (buildings.category_counts, civil.category_counts, pavement.category_counts, vegetation.category_counts):
        counts.update(getter(elements))
    results=check_dxf(out / "outdoor.dxf", cfg)+check_ifc(out / "outdoor.ifc", cfg)+count_consistency(counts, out / "outdoor.dxf", out / "outdoor.ifc")
    validation = validate_exports({"ifc": out / "outdoor.ifc", "dxf": out / "outdoor.dxf", "laz": out / "classified.laz", "glb": out / "environment.glb"})
    results += [{"rule": "exports", "ok": all(item.get("ok", False) for item in validation.values()), "value": validation, "threshold": "all exports valid", "offenders": []}]
    write_qa_report(results, out); _dump(cache / "qa.pkl", results); return {"results": len(results)}


def _step_report(project, out, cfg, cache, state):
    data={"steps": state.get("steps", {}), "qa": _load(cache / "qa.pkl", []), "rejected": [q.__dict__ for q in _load(cache / "scene.pkl", {}).get("qa", [])] if (cache / "scene.pkl").exists() else []}
    _json(out / "environment_report.json", data)
    lines=["# Environment report", "", "## Steps", *[f"- {n}: {v.get('status')} ({v.get('seconds', 0):.2f}s)" for n,v in data["steps"].items()]]
    (out / "environment_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8"); return {"report": str(out / "environment_report.json")}


HANDLERS={"load":_step_load,"semantics":_step_semantics,"terrain":_step_terrain,"vegetation":_step_vegetation,"pavement":_step_pavement,"tin":_step_tin,"buildings":_step_buildings,"site":_step_site,"scene":_step_scene,"export":_step_export,"qa":_step_qa}


_MIN_CLIPPED_LINE_M = 0.5


def _clip_polyline(points, poly, min_len: float = _MIN_CLIPPED_LINE_M):
    """Inside parts of a 2D/3D polyline as arrays (extra dims interpolated along the line); parts < min_len dropped."""
    from shapely.geometry import LineString, Point
    pts = np.asarray(points, float)
    if len(pts) < 2: return []
    line = LineString(pts[:, :2])
    if line.length <= 0: return []
    inter = line.intersection(poly)
    if inter.is_empty: return []
    geoms = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1); cum = np.concatenate([[0.0], np.cumsum(seg)])
    parts = []
    for g in geoms:
        if g.geom_type != "LineString" or g.length < min_len: continue
        xy = np.asarray(g.coords, float)[:, :2]
        s = np.array([line.project(Point(x, y)) for x, y in xy])
        cols = [xy] + [np.interp(s, cum, pts[:, d])[:, None] for d in range(2, pts.shape[1])]
        parts.append(np.hstack(cols))
    return parts


def _is_unchanged(parts, pts) -> bool:
    return len(parts) == 1 and len(parts[0]) == len(pts) and np.allclose(parts[0], pts)


def _clip_elements(elements, poly, counts):
    """Clip/drop outdoor scene elements: trees, manholes (point rule), walls, fences, curbs, open markings
    (line clipping, split into parts), shrubs, vegetation areas, closed markings (outline clipping)."""
    from shapely import contains_xy
    from shapely.geometry import Polygon
    out = []
    for el in elements:
        p = el.params; kind = type(el).__name__.replace("Element", "").lower()
        def renamed(k, pts, attr, _el=el):
            new = _el.model_copy(update={"params": _el.params.model_copy(update={attr: [tuple(map(float, q)) for q in pts]})})
            return new.model_copy(update={"id": f"{_el.id}-p{k}"}) if k else new
        if getattr(p, "trunk_xy", None) is not None or getattr(p, "center", None) is not None:
            xy = p.trunk_xy if getattr(p, "trunk_xy", None) is not None else p.center
            if contains_xy(poly, float(xy[0]), float(xy[1])): out.append(el)
            else: counts[f"{kind}s_dropped"] += 1
            continue
        line_attr = "axis" if getattr(p, "axis", None) is not None else ("points" if getattr(p, "points", None) is not None and not getattr(p, "closed", False) else None)
        if line_attr:
            pts = np.asarray(getattr(p, line_attr), float); parts = _clip_polyline(pts, poly)
            if not parts: counts[f"{kind}s_dropped"] += 1
            elif _is_unchanged(parts, pts): out.append(el)
            else:
                counts[f"{kind}s_clipped"] += 1
                out.extend(renamed(k, part, line_attr) for k, part in enumerate(parts))
            continue
        attr = next((a for a in ("outline", "footprint", "points") if getattr(p, a, None)), None)
        ring = getattr(p, attr) if attr else None
        if not ring or len(ring) < 3: out.append(el); continue
        geom = Polygon([tuple(q[:2]) for q in ring])
        if not geom.is_valid: geom = geom.buffer(0)
        clipped = geom.intersection(poly)
        if clipped.geom_type == "MultiPolygon": clipped = max(clipped.geoms, key=lambda g: g.area)
        if clipped.is_empty or clipped.geom_type != "Polygon" or clipped.area < 1e-6:
            counts[f"{kind}s_dropped"] += 1; continue
        if abs(clipped.area - geom.area) > 1e-9:
            new_ring = [(float(x), float(y)) for x, y in list(clipped.exterior.coords)[:-1]]
            out.append(el.model_copy(update={"params": p.model_copy(update={attr: new_ring})})); counts[f"{kind}s_clipped"] += 1
        else: out.append(el)
    return out


def _apply_boundary(name: str, out: Path, cache: Path) -> dict | None:
    """Restrict terrain rasters, pavement/lawn polygons, curbs, vegetation, site elements and TIN triangles to the processing boundary.
    Buildings are intentionally not clipped. Returns clip counters for pavement/vegetation/site."""
    poly = _load(cache / "boundary.pkl")
    if poly is None: return None
    from collections import defaultdict
    from shapely import contains_xy
    counts: dict = defaultdict(int)
    if name == "terrain":
        from shapely.affinity import translate
        from samcloud.drone.perimeter import mask_geotiff_outside
        terrain = _load(cache / "terrain.pkl"); frame = _load(cache / "load.pkl")[1]
        if isinstance(terrain, dict) and isinstance(terrain.get("dtm"), np.ndarray) and "bounds" in terrain:
            dtm = np.array(terrain["dtm"], float); b = terrain["bounds"]; res = float(terrain["resolution"])
            gx, gy = np.meshgrid(b[0] + np.arange(dtm.shape[1]) * res, b[3] - np.arange(dtm.shape[0]) * res)
            inside = contains_xy(poly, gx, gy)
            if inside.any():
                dtm[~inside] = np.nan; terrain["dtm"] = dtm; _dump(cache / "terrain.pkl", terrain)
        lv95_poly = translate(poly, float(frame.origin_lv95[0]), float(frame.origin_lv95[1]))
        for tif in ("dtm.tif", "dsm.tif"):
            if (out / tif).is_file(): mask_geotiff_outside(out / tif, lv95_poly)
    elif name == "pavement":
        from dataclasses import replace
        pave = _load(cache / "pavement.pkl"); kept = []
        for item in pave["polygons"]:
            clipped = item.polygon.intersection(poly)
            if not clipped.is_empty and clipped.geom_type in ("Polygon", "MultiPolygon"):
                kept.append(replace(item, polygon=clipped))
        pave["polygons"] = kept
        pave["lawns"] = [g for g in (lawn.intersection(poly) for lawn in pave["lawns"]) if not g.is_empty]
        if pave.get("curbs"):
            next_id = max(int(c.curb_id) for c in pave["curbs"]) + 1
            curbs = []
            for c in pave["curbs"]:
                pts = np.asarray(c.points, float); parts = _clip_polyline(pts, poly)
                if not parts: counts["curbs_dropped"] += 1; continue
                if _is_unchanged(parts, pts): curbs.append(c); continue
                counts["curbs_clipped"] += 1
                for k, part in enumerate(parts):
                    cid = c.curb_id if k == 0 else next_id
                    if k: next_id += 1
                    curbs.append(replace(c, curb_id=cid, points=part))
            pave["curbs"] = curbs
        _dump(cache / "pavement.pkl", pave)
    elif name == "vegetation":
        veg = _load(cache / "vegetation.pkl")
        if veg is not None:
            veg.elements = _clip_elements(veg.elements, poly, counts)
            if getattr(veg, "lawn_areas", None):
                veg.lawn_areas = [a for a in veg.lawn_areas if not a.polygon.intersection(poly).is_empty]
            _dump(cache / "vegetation.pkl", veg)
    elif name == "site":
        site = _load(cache / "site.pkl")
        if site is not None:
            site["elements"] = _clip_elements(site["elements"], poly, counts)
            _dump(cache / "site.pkl", site)
    elif name == "tin":
        tin = _load(cache / "tin.pkl"); tris = np.asarray(tin.triangles)
        cen = np.asarray(tin.vertices)[tris][:, :, :2].mean(1)
        keep = contains_xy(poly, cen[:, 0], cen[:, 1])
        if keep.any() and not keep.all(): _dump(cache / "tin.pkl", tin._filter(keep))
    return dict(counts) if name in ("pavement", "vegetation", "site") else None


def run_environment(project_dir, *, out_dir=None, config=None, steps=None, resume=True) -> dict:
    """Run the resumable outdoor chain; failures are recorded and independent steps continue."""
    project=Path(project_dir); out=Path(out_dir) if out_dir else project / "products" / "environment"; out.mkdir(parents=True, exist_ok=True); cache=out / "cache"; cache.mkdir(exist_ok=True)
    cfg=load_drone_config() if config is None else config; state_path=out / "state.json"; state=json.loads(state_path.read_text(encoding="utf-8")) if resume and state_path.exists() else {"steps": {}}
    requested=set(STEPS if not steps else ([x.strip() for x in steps.split(",")] if isinstance(steps,str) else steps)); active=set()
    for wanted in requested:
        if wanted not in STEPS: raise ValueError(f"unknown step {wanted!r}")
        active.add(wanted)
    # A targeted run invalidates only the requested checkpoints and their true descendants,
    # never merely the steps that happen to occur later in the display order.
    changed = True
    while changed:
        changed = False
        for name, dependencies in DEPENDENCIES.items():
            if name not in active and any(item in active for item in dependencies):
                active.add(name); changed = True
    for name in STEPS:
        if name not in active: continue
        previous=state["steps"].get(name, {})
        if resume and previous.get("status") == "done" and not steps:
            state["steps"][name]={**previous,"status":"skipped","reason":"resume: completed checkpoint"}; _json(state_path,state); continue
        failed=[] if name == "report" else [d for d in DEPENDENCIES.get(name, ()) if state["steps"].get(d,{}).get("status") in ("failed","skipped") and not (cache / f"{d}.pkl").exists()]
        if failed:
            state["steps"][name]={"status":"skipped","reason":"dependency unavailable: "+", ".join(failed),"seconds":0}; _json(state_path,state); continue
        started=time.perf_counter()
        monitor = _PeakRssMonitor(psutil.Process())
        monitor.start()
        try:
            outputs=_step_report(project,out,cfg,cache,state) if name=="report" else HANDLERS[name](project,out,cfg,cache)
            if name in ("terrain", "pavement", "tin", "vegetation", "site"):
                clip = _apply_boundary(name, out, cache)
                if clip is not None and isinstance(outputs, dict): outputs["boundary_clip"] = clip
            peak_rss_mb = monitor.stop()
            state["steps"][name]={"status":"done","seconds":round(time.perf_counter()-started,3),"peak_rss_mb":round(peak_rss_mb,3),"outputs":outputs}
        except Exception as exc:
            peak_rss_mb = monitor.stop()
            state["steps"][name]={"status":"failed","seconds":round(time.perf_counter()-started,3),"peak_rss_mb":round(peak_rss_mb,3),"error":str(exc),"traceback":traceback.format_exc(limit=3)}
        _json(state_path,state)
    return {"out_dir":str(out), **state}


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--project",required=True); parser.add_argument("--steps"); parser.add_argument("--no-resume",action="store_true")
    args=parser.parse_args(argv); print(json.dumps(run_environment(args.project,steps=args.steps,resume=not args.no_resume),indent=2,default=str))

if __name__ == "__main__": main()
