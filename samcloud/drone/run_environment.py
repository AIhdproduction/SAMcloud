"""Resumable end-to-end outdoor environment pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time
import traceback
from typing import Any

import numpy as np

from samcloud.drone.contracts import PointSet, load_drone_config

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
    _dump(cache / "load.pkl", (ps, frame)); return info


def _step_semantics(project, out, cfg, cache):
    ps, frame = _load(cache / "load.pkl")
    try:
        from samcloud.drone.semantics import run_semantics
        result = run_semantics(project / "products", out / "semantics", config=cfg)
    except ImportError:
        result = {"pointset": ps, "instances": []}
    _dump(cache / "semantics.pkl", result); return {"available": True}


def _step_terrain(project, out, cfg, cache):
    from samcloud.terrain.rasters import export_terrain
    ps, frame = _load(cache / "load.pkl")
    raw = export_terrain(ps, out, frame, cfg)
    # Retain the arrays needed by later steps even when the public report only lists files.
    terrain = dict(raw)
    if not isinstance(terrain.get("dtm"), np.ndarray):
        from samcloud.terrain.rasters import build_dtm_raster, ground_mask
        ground = ps.xyz[ground_mask(ps, method="pmf")]
        resolution = float(cfg.get("terrain", {}).get("dtm_resolution_m", .25))
        lo, hi = ps.xyz[:, :2].min(0), ps.xyz[:, :2].max(0)
        bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
        terrain["dtm"], terrain["validity"], _ = build_dtm_raster(ground, resolution=resolution, bounds=bounds)
        terrain.update(bounds=bounds, resolution=resolution)
    _dump(cache / "terrain.pkl", terrain); return {"files": terrain.get("files", {})}


def _step_vegetation(project, out, cfg, cache):
    from samcloud.environment.vegetation import detect_shrubs, detect_trees
    from samcloud.outdoor_model.vegetation import build_vegetation
    ps, _frame = _load(cache / "load.pkl"); terrain = _load(cache / "terrain.pkl"); sample = _sampler(terrain)
    trees, shrubs = detect_trees(ps, sample, config=cfg), detect_shrubs(ps, sample)
    result = build_vegetation(trees, shrubs, (), sample, config=cfg)
    _dump(cache / "vegetation.pkl", result); return {"trees": len(trees), "shrubs": len(shrubs)}


def _step_pavement(project, out, cfg, cache):
    from samcloud.environment.pavement import lawn_polygons, pavement_polygons
    from samcloud.environment.curbs import curb_axes
    ps, _frame = _load(cache / "load.pkl")
    result = {"polygons": pavement_polygons(ps), "lawns": lawn_polygons(ps), "curbs": curb_axes(ps)}
    _dump(cache / "pavement.pkl", result); return {k: len(v) for k, v in result.items()}


def _step_tin(project, out, cfg, cache):
    from samcloud.terrain.tin import build_labeled_tin
    ps, _frame = _load(cache / "load.pkl"); terrain = _load(cache / "terrain.pkl"); pave = _load(cache / "pavement.pkl")
    sample = _sampler(terrain); regions = {"lawn": list(pave["lawns"])}
    for poly in pave["polygons"]: regions.setdefault(poly.kind, []).append(poly.polygon)
    ground = ps.xyz[np.isin(ps.classes, [2, 11, 71, 73, 83])]
    tin_cfg = cfg.get("terrain", {}).get("tin", {})
    tin = build_labeled_tin(ground, regions=regions, dtm_sampler=sample,
        buffer=float(tin_cfg.get("constraint_buffer_m", .05)), min_angle_deg=float(tin_cfg.get("min_angle_deg", 20)),
        max_steiner_per_edge=int(tin_cfg.get("max_steiner_per_edge", 50)),
        thin_spacing=tin_cfg.get("thin_spacing_m"))
    _dump(cache / "tin.pkl", tin); return {"triangles": len(tin.triangles)}


def _step_buildings(project, out, cfg, cache):
    from samcloud.outdoor_model.buildings import build_building_elements
    from samcloud.terrain.tin import build_labeled_tin
    from samcloud.buildings.reconstruct import reconstruct_building
    from samcloud.buildings.facades import facade_surfaces
    from samcloud.buildings.openings import detect_building_openings
    from samcloud.drone import contracts as c
    ps, _frame = _load(cache / "load.pkl"); terrain = _load(cache / "terrain.pkl"); sample = _sampler(terrain)
    semantic = _load(cache / "semantics.pkl", {})
    instances = semantic.get("instances", semantic.get("buildings", [])) if isinstance(semantic, dict) else []
    models=[]; opens={}; facades={}; qa=[]
    failures = []
    for inst in instances:
        try:
            bid=int(inst.building_id); model=reconstruct_building(ps.subset(inst.point_indices), bid, dtm_sampler=sample, config=cfg)
            models.append(model)
            if model.status != "failed":
                surfaces=facade_surfaces(model, getattr(inst, "facades", [])); facades[bid]=surfaces
                detected, opening_qa=detect_building_openings(ps.subset(inst.point_indices), model, getattr(inst, "facades", []), dtm_sampler=sample, config=cfg)
                opens[bid]=detected; qa.extend(opening_qa)
        except Exception as exc:
            message = str(exc)
            qa.append(c.QAEntry("building", f"building-{getattr(inst, 'building_id', '?')}", message))
            failures.append(message)
    elements, more = build_building_elements(models, opens, facades, config=cfg); qa.extend(more)
    _dump(cache / "buildings.pkl", {"elements": elements, "qa": qa})
    if failures:
        raise RuntimeError("building reconstruction failed: " + "; ".join(failures))
    return {"buildings": len(models), "qa": len(qa)}


def _step_site(project, out, cfg, cache):
    from samcloud.outdoor_model.site_walls import build_site_elements, fence_features, site_wall_features
    from samcloud.outdoor_model.civil import build_civil_elements, manhole_detections, marking_features
    ps, _frame = _load(cache / "load.pkl"); sample = _sampler(_load(cache / "terrain.pkl")); pave = _load(cache / "pavement.pkl")
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
    ps, frame = _load(cache / "load.pkl"); veg = _load(cache / "vegetation.pkl"); pave = _load(cache / "pavement.pkl"); tin = _load(cache / "tin.pkl"); bld = _load(cache / "buildings.pkl"); site = _load(cache / "site.pkl")
    pavement_elements, pavement_qa = pavement.build_pavement_elements(pave["polygons"], pave["lawns"], pave["curbs"], tin, config=cfg)
    terrain_elements, terrain_qa = terrain_builder.build_terrain_elements(tin, config=cfg)
    groups=[veg.elements, pavement_elements, terrain_elements, bld["elements"], site["elements"]]
    scene=assemble_scene(*groups, frame=frame)
    terrain_dxf, _terrain_info = terrain_builder.terrain_dxf_entities(terrain_elements, frame, tin=tin, config=cfg)
    entities={"vegetation": vegetation.to_dxf_entities(veg.elements, frame, lawn_areas=veg.lawn_areas), "pavement": pavement.to_dxf_entities(pavement_elements, frame, config=cfg), "terrain": terrain_dxf, "buildings": buildings.to_dxf_entities(bld["elements"], frame), "site": site_walls.to_dxf_entities(site["elements"], frame), "civil": civil.to_dxf_entities(site["elements"], frame, config=cfg)}
    boundary_dxf = _boundary_dxf_entities(_load(cache / "boundary.pkl"), frame)
    if boundary_dxf: entities["boundary"] = boundary_dxf
    dxf=assemble_dxf(entities, frame, cfg); _dump(cache / "scene.pkl", {"scene": scene, "dxf": dxf, "qa": pavement_qa + terrain_qa + bld["qa"] + site["qa"]}); return {"elements": len(scene.elements)}


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
    return (vertices, np.asarray(faces, dtype=np.int64)) if faces else None


def _write_environment_glb(nodes, path: Path, frame) -> None:
    import trimesh

    scene = trimesh.Scene()
    for name, (vertices, faces, category) in nodes.items():
        scene.add_geometry(trimesh.Trimesh(vertices=vertices, faces=faces, process=False), node_name=name, geom_name=name)

    origin = [float(value) for value in frame.origin_lv95]
    def decorate(document):
        document.setdefault("asset", {}).setdefault("extras", {}).update({"origin_lv95": origin, "epsg": int(frame.epsg), "vertical_datum": frame.vertical_datum})
        for node in document.get("nodes", []):
            if "mesh" in node:
                node.setdefault("extras", {})["category"] = nodes[str(node.get("name"))][2]
    path.write_bytes(trimesh.exchange.gltf.export_glb(scene, tree_postprocessor=decorate))


def _step_export(project, out, cfg, cache):
    from samcloud.bim.core.ifc.writer import write_ifc
    from samcloud.bim.core.model import BuildingElement
    from samcloud.drone.export import export_mesh, export_point_cloud
    from samcloud.dxf.writer import build_dxf_document, ensure_outdoor_layers
    ps, frame = _load(cache / "load.pkl"); data = _load(cache / "scene.pkl")
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
    return {"ifc": str(ifc_path), "dxf": str(dxf_path), "laz": str(laz_path), "glb": str(glb_path), "nodes": len(nodes)}


def _step_qa(project, out, cfg, cache):
    from samcloud.drone.export import validate_exports, lv95_sample_check
    from samcloud.drone.qa_cad import check_dxf, check_ifc, count_consistency, write_qa_report
    from samcloud.outdoor_model import buildings, civil, pavement, vegetation
    ps, frame = _load(cache / "load.pkl"); data=_load(cache / "scene.pkl")
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


def _apply_boundary(name: str, out: Path, cache: Path) -> None:
    """Restrict terrain rasters, pavement/lawn polygons and TIN triangles to the processing boundary."""
    poly = _load(cache / "boundary.pkl")
    if poly is None: return
    from shapely import contains_xy
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
        _dump(cache / "pavement.pkl", pave)
    elif name == "tin":
        tin = _load(cache / "tin.pkl"); tris = np.asarray(tin.triangles)
        cen = np.asarray(tin.vertices)[tris][:, :, :2].mean(1)
        keep = contains_xy(poly, cen[:, 0], cen[:, 1])
        if keep.any() and not keep.all(): _dump(cache / "tin.pkl", tin._filter(keep))


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
        try:
            outputs=_step_report(project,out,cfg,cache,state) if name=="report" else HANDLERS[name](project,out,cfg,cache)
            if name in ("terrain", "pavement", "tin"): _apply_boundary(name, out, cache)
            state["steps"][name]={"status":"done","seconds":round(time.perf_counter()-started,3),"outputs":outputs}
        except Exception as exc:
            state["steps"][name]={"status":"failed","seconds":round(time.perf_counter()-started,3),"error":str(exc),"traceback":traceback.format_exc(limit=3)}
        _json(state_path,state)
    return {"out_dir":str(out), **state}


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--project",required=True); parser.add_argument("--steps"); parser.add_argument("--no-resume",action="store_true")
    args=parser.parse_args(argv); print(json.dumps(run_environment(args.project,steps=args.steps,resume=not args.no_resume),indent=2,default=str))

if __name__ == "__main__": main()
