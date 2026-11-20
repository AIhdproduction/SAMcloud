"""Post-classification geometric semantics for outdoor drone point clouds.

Roof classes are exported as ``ROOF_INCLINED`` and ``ROOF_FLAT``.  Facades
deliberately remain ``BUILDING``: the LAS class scheme has no separate facade
code, so their geometric role is exposed through the ``is_facade`` mask in the
split report instead.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely import concave_hull
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.wkt import dumps as wkt_dumps, loads as wkt_loads

from samcloud.buildings.footprint import detect_facade_planes
from samcloud.drone.contracts import (
    BUILDING,
    GROUND,
    LOW_VEGETATION,
    MEDIUM_VEGETATION,
    ROAD,
    ROOF_FLAT,
    ROOF_INCLINED,
    TREE,
    BuildingInstance,
    PointSet,
    load_drone_config,
)

LOGGER = logging.getLogger(__name__)
_BUILDING_CODES = np.array((BUILDING, ROOF_INCLINED, ROOF_FLAT), dtype=np.uint16)
_VEGETATION_CODES = np.array((LOW_VEGETATION, MEDIUM_VEGETATION, TREE), dtype=np.uint16)


def _copy_pointset(ps: PointSet) -> PointSet:
    """Copy every populated PointSet array so semantic edits are explicit."""
    values = {}
    for name in ("xyz", "classes", "confidence", "margin", "view_count", "normals", "rgb", "adjusted"):
        value = getattr(ps, name)
        values[name] = None if value is None else np.asarray(value).copy()
    return PointSet(**values)


def _normals(xyz: np.ndarray, supplied: np.ndarray | None) -> np.ndarray:
    """Return upward-facing unit KNN-16 normals, estimating missing values with Open3D."""
    points = np.asarray(xyz, dtype=np.float64)
    result = np.zeros_like(points)
    valid = np.zeros(len(points), dtype=bool)
    if supplied is not None and np.asarray(supplied).shape == points.shape:
        result = np.asarray(supplied, dtype=np.float64).copy()
        lengths = np.linalg.norm(result, axis=1)
        valid = lengths > 1e-9
        result[valid] /= lengths[valid, None]
    missing = ~valid
    if missing.any() and len(points) >= 3:
        try:
            import open3d as o3d

            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=min(16, len(points))))
            estimated = np.asarray(cloud.normals)
        except Exception:  # pragma: no cover - exercised only without Open3D
            count = min(16, len(points))
            indices = cKDTree(points).query(points, k=count, workers=-1)[1]
            local = points[indices] - points[indices].mean(axis=1, keepdims=True)
            covariance = np.einsum("nki,nkj->nij", local, local) / max(count - 1, 1)
            _values, vectors = np.linalg.eigh(covariance)
            estimated = vectors[:, :, 0]
        result[missing] = estimated[missing]
    lengths = np.linalg.norm(result, axis=1)
    usable = lengths > 1e-9
    result[usable] /= lengths[usable, None]
    result[result[:, 2] < 0] *= -1.0
    return result


def _sample_dtm(dtm_sampler: Callable, xy: np.ndarray) -> np.ndarray:
    """Call the lightweight DTM sampler accepted by this module."""
    try:
        values = dtm_sampler(np.asarray(xy, dtype=np.float64))
    except TypeError:
        values = dtm_sampler(np.asarray(xy, dtype=np.float64)[:, 0], np.asarray(xy, dtype=np.float64)[:, 1])
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(len(xy), float(values))
    return values.reshape(-1)


def _hag(ps: PointSet, dtm_sampler: Callable) -> np.ndarray:
    return np.asarray(ps.xyz, dtype=np.float64)[:, 2] - _sample_dtm(dtm_sampler, ps.xyz[:, :2])


def _slope_degrees(normals: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0)))


def split_roof_facade(
    ps: PointSet,
    dtm_sampler: Callable,
    *,
    facade_max_nz: float = 0.3,
    roof_min_nz: float = 0.5,
    flat_max_slope_deg: float = 5,
) -> tuple[PointSet, dict]:
    """Geometrically refine building points into roof classes and facade roles."""
    result = _copy_pointset(ps)
    normals = _normals(result.xyz, result.normals)
    result.normals = normals
    building = result.classes == BUILDING
    vertical = building & (np.abs(normals[:, 2]) < facade_max_nz)
    roof = building & (normals[:, 2] > roof_min_nz)
    slopes = _slope_degrees(normals)
    flat = roof & (slopes < flat_max_slope_deg)
    inclined = roof & ~flat
    result.classes[flat] = ROOF_FLAT
    result.classes[inclined] = ROOF_INCLINED
    report = {
        "is_facade": vertical,
        "facade": int(vertical.sum()),
        "roof_flat": int(flat.sum()),
        "roof_inclined": int(inclined.sum()),
        "ambiguous_building": int((building & ~vertical & ~roof).sum()),
        "hag_valid": int(np.isfinite(_hag(result, dtm_sampler)).sum()),
    }
    return result, report


def _occupancy_components(xy: np.ndarray, z: np.ndarray, resolution: float, gap: float) -> list[np.ndarray]:
    """Component labels on a 2-D occupancy grid, retaining DSM discontinuities as gaps."""
    if len(xy) == 0:
        return []
    lower = xy.min(axis=0)
    cells = np.floor((xy - lower) / resolution).astype(np.int64)
    shape = tuple((cells.max(axis=0) + 1).tolist())
    flat = cells[:, 0] + shape[0] * cells[:, 1]
    unique, inverse = np.unique(flat, return_inverse=True)
    height = np.zeros(len(unique), dtype=float)
    variation = np.zeros(len(unique), dtype=float)
    for index in range(len(unique)):
        values = z[inverse == index]
        height[index] = np.median(values)
        variation[index] = np.std(values)
    occupied = np.zeros(shape, dtype=bool)
    ux, uy = unique % shape[0], unique // shape[0]
    occupied[ux, uy] = True
    # Break occupied-cell bridges at steep local DSM steps.  This deliberately
    # separates adjoining buildings that share a facade and have different roofs.
    # A cell that contains two roof levels is the sampled form of the DSM
    # valley at a shared facade.  It must not bridge the two occupancies.
    allowed = occupied.copy()
    allowed[ux[variation > 0.75], uy[variation > 0.75]] = False
    grid_height = np.full(shape, np.nan)
    grid_height[ux, uy] = height
    for axis in (0, 1):
        current = [slice(None), slice(None)]
        neighbour = [slice(None), slice(None)]
        current[axis] = slice(0, -1)
        neighbour[axis] = slice(1, None)
        a, b = tuple(current), tuple(neighbour)
        step = allowed[a] & allowed[b] & (np.abs(grid_height[a] - grid_height[b]) > 1.5)
        # Remove the lower side of a discontinuity.  This creates a one-cell
        # valley while retaining each roof component.
        lower_side = step & (grid_height[a] <= grid_height[b])
        higher_side = step & ~lower_side
        view_a, view_b = allowed[a], allowed[b]
        view_a[lower_side] = False
        view_b[higher_side] = False
    labels, number = ndimage.label(allowed, structure=np.ones((3, 3), dtype=np.uint8))
    labels_for_points = labels[cells[:, 0], cells[:, 1]]
    components = [np.flatnonzero(labels_for_points == label) for label in range(1, number + 1)]
    # Cells deliberately removed for a valley join their nearest surviving side.
    orphan = np.flatnonzero(labels_for_points == 0)
    if len(orphan) and number:
        surviving_cells = np.column_stack(np.nonzero(allowed))
        centres = surviving_cells * resolution + lower + resolution / 2.0
        tree = cKDTree(centres)
        near = tree.query(xy[orphan], workers=-1)[1]
        nearest_labels = labels[surviving_cells[near, 0], surviving_cells[near, 1]]
        for point, label in zip(orphan, nearest_labels, strict=False):
            components[int(label) - 1] = np.append(components[int(label) - 1], point)
    return components


def _footprint(xy: np.ndarray) -> Polygon:
    if len(xy) < 3:
        return Polygon()
    # A high ratio keeps regular roof grids complete while still allowing real
    # concavities to survive; footprint regularisation happens downstream.
    polygon = concave_hull(MultiPoint(xy), ratio=0.75, allow_holes=False)
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        polygon = MultiPoint(xy).convex_hull
    return polygon.buffer(0)


def segment_buildings(
    ps: PointSet,
    dtm_sampler: Callable,
    *,
    min_points: int = 500,
    min_height: float = 2.0,
    min_footprint_m2: float = 10,
    separation_gap_m: float = 0.5,
) -> list[BuildingInstance]:
    """Segment elevated building/roof supports into persistent-instance candidates.

    Rejected components are omitted.  Their count and reasons are exposed in
    ``segment_buildings.last_report`` because the public result remains a list.
    """
    hag = _hag(ps, dtm_sampler)
    selected = np.flatnonzero(np.isin(ps.classes, _BUILDING_CODES) & (hag > 1.0))
    report = {"candidates": 0, "accepted": 0, "dropped": [], "facade_planes": 0}
    if len(selected) == 0:
        segment_buildings.last_report = report
        return []
    xyz = np.asarray(ps.xyz)[selected]
    components = _occupancy_components(xyz[:, :2], hag[selected], 0.25, separation_gap_m)
    result: list[BuildingInstance] = []
    for local_indices in components:
        indices = selected[local_indices]
        component_xyz = ps.xyz[indices]
        footprint = _footprint(component_xyz[:, :2])
        height = float(np.nanmax(hag[indices]))
        report["candidates"] += 1
        reason = ""
        if len(indices) < min_points:
            reason = f"too few points ({len(indices)} < {min_points})"
        elif footprint.area < min_footprint_m2:
            reason = f"footprint too small ({footprint.area:.2f} m2)"
        elif height < min_height:
            reason = f"height too small ({height:.2f} m)"
        if reason:
            report["dropped"].append({"point_count": int(len(indices)), "reason": reason})
            continue
        local = ps.subset(indices)
        facades = detect_facade_planes(local, distance=0.08, min_points=max(12, min(80, len(local) // 10)), max_planes=12)
        report["facade_planes"] += len(facades)
        result.append(BuildingInstance(
            building_id=-1,
            point_indices=np.asarray(indices, dtype=np.int64),
            footprint=footprint,
            facades=facades,
            status="ok",
        ))
    result.sort(key=lambda item: (item.footprint.centroid.x if item.footprint else 0, item.footprint.centroid.y if item.footprint else 0))
    report["accepted"] = len(result)
    segment_buildings.last_report = report
    return result


segment_buildings.last_report = {"candidates": 0, "accepted": 0, "dropped": [], "facade_planes": 0}


def load_instances(path: str | Path) -> list[BuildingInstance]:
    """Load the compact persistent instance interchange JSON."""
    source = Path(path)
    if not source.is_file():
        return []
    raw = json.loads(source.read_text(encoding="utf-8"))
    records = raw.get("instances", raw) if isinstance(raw, dict) else raw
    loaded = []
    for record in records:
        outline = wkt_loads(record["footprint_wkt"]) if record.get("footprint_wkt") else Polygon()
        loaded.append(BuildingInstance(
            building_id=int(record["building_id"]),
            point_indices=np.empty(int(record.get("point_count", 0)), dtype=np.int64),
            footprint=outline if isinstance(outline, Polygon) else Polygon(),
            status=record.get("status", "ok"),
            reason=record.get("reason", ""),
        ))
    return loaded


def save_instances(instances: list[BuildingInstance], path: str | Path) -> Path:
    """Persist IDs and spatial matching support without serialising cloud indices."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for instance in instances:
        footprint = instance.footprint if instance.footprint is not None else Polygon()
        records.append({
            "building_id": int(instance.building_id),
            "footprint_wkt": wkt_dumps(footprint, rounding_precision=6),
            "point_count": int(len(instance.point_indices)),
            "bbox": list(map(float, footprint.bounds)) if not footprint.is_empty else None,
            "status": instance.status,
            "reason": instance.reason,
        })
    destination.write_text(json.dumps({"instances": records}, indent=2) + "\n", encoding="utf-8")
    return destination


def assign_persistent_ids(instances: list[BuildingInstance], previous_path: str | Path | None = None) -> list[BuildingInstance]:
    """Match instances to prior footprints by IoU and allocate monotonically new IDs."""
    previous = load_instances(previous_path or Path("buildings") / "instances.json")
    unused = set(range(len(previous)))
    for instance in instances:
        footprint = instance.footprint
        if footprint is None or footprint.is_empty:
            continue
        best_index, best_iou = None, 0.5
        for index in unused:
            old = previous[index].footprint
            if old is None or old.is_empty or not footprint.intersects(old):
                continue
            union = footprint.union(old).area
            iou = footprint.intersection(old).area / union if union else 0.0
            if iou > best_iou:
                best_index, best_iou = index, iou
        if best_index is not None:
            instance.building_id = previous[best_index].building_id
            unused.remove(best_index)
    maximum = max((item.building_id for item in previous), default=0)
    for instance in instances:
        if instance.building_id < 0:
            maximum += 1
            instance.building_id = maximum
    return instances


def _local_plane_metrics(xyz: np.ndarray, tree: cKDTree, radius: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local PCA RMS, normal and neighbourhood size for every point."""
    rms = np.full(len(xyz), np.inf)
    normals = np.zeros((len(xyz), 3), dtype=float)
    sizes = np.zeros(len(xyz), dtype=np.int32)
    for index, neighbours in enumerate(tree.query_ball_point(xyz, radius, workers=-1)):
        sizes[index] = len(neighbours)
        if len(neighbours) < 6:
            continue
        local = xyz[neighbours] - xyz[neighbours].mean(axis=0)
        values, vectors = np.linalg.eigh(local.T @ local / len(local))
        rms[index] = float(np.sqrt(max(values[0], 0.0)))
        normals[index] = vectors[:, 0]
    normals[normals[:, 2] < 0] *= -1.0
    return rms, normals, sizes


def plausibility_rules(ps: PointSet, dtm_sampler: Callable, *, building_instances: list[BuildingInstance]) -> tuple[PointSet, dict]:
    """Apply logged geometry-only corrections around existing building footprints."""
    result = _copy_pointset(ps)
    xyz = result.xyz
    hag = _hag(result, dtm_sampler)
    tree = cKDTree(xyz)
    rms, local_normals, sizes = _local_plane_metrics(xyz, tree)
    normals = _normals(xyz, result.normals)
    normals[np.linalg.norm(local_normals, axis=1) > 0] = local_normals[np.linalg.norm(local_normals, axis=1) > 0]
    slopes = _slope_degrees(normals)
    outlines = [item.footprint.buffer(0.25) for item in building_instances if item.footprint and not item.footprint.is_empty]
    near_building = np.zeros(len(result), dtype=bool)
    if outlines:
        for index, xy in enumerate(xyz[:, :2]):
            near_building[index] = any(outline.covers(Point(float(xy[0]), float(xy[1]))) for outline in outlines)
    high_confidence = np.zeros(len(result), dtype=bool)
    if result.confidence is not None and result.margin is not None:
        high_confidence = (result.confidence > 0.8) & (result.margin > 0.3)
    vegetation = np.isin(result.classes, _VEGETATION_CODES)
    planar_roof = vegetation & near_building & (hag > 2.0) & (rms < 0.03) & (sizes >= 6) & (normals[:, 2] > 0.5)
    unambiguous_roof = planar_roof & (rms < 0.01) & (sizes >= 12)
    planar_roof &= ~high_confidence | unambiguous_roof
    target_roof = np.where(slopes < 5.0, ROOF_FLAT, ROOF_INCLINED).astype(np.uint16)
    result.classes[planar_roof] = target_roof[planar_roof]
    surface = np.isin(result.classes, (GROUND, ROAD))
    facade = surface & near_building & (hag > 0.5) & (np.abs(normals[:, 2]) < 0.3)
    unambiguous_facade = facade & (np.abs(normals[:, 2]) < 0.15) & (rms < 0.02)
    facade &= ~high_confidence | unambiguous_facade
    result.classes[facade] = BUILDING
    report = {
        "vegetation_to_roof": int(planar_roof.sum()),
        "ground_road_to_facade": int(facade.sum()),
        "high_confidence_protected": int((high_confidence & ((vegetation & near_building) | (surface & near_building)) & ~(planar_roof | facade)).sum()),
    }
    LOGGER.info("plausibility changes: vegetation_to_roof=%d ground_road_to_facade=%d", report["vegetation_to_roof"], report["ground_road_to_facade"])
    return result, report


def uncertainty_arrays(ps: PointSet) -> dict[str, np.ndarray | None]:
    """Return LAZ-extra-dimension ready uncertainty arrays with stable dtypes."""
    return {
        "confidence": None if ps.confidence is None else np.asarray(ps.confidence, dtype=np.float32),
        "margin": None if ps.margin is None else np.asarray(ps.margin, dtype=np.float32),
        "view_count": None if ps.view_count is None else np.asarray(ps.view_count, dtype=np.uint16),
    }


def _dtm_sampler_from_pointset(ps: PointSet, config: dict) -> Callable:
    """Build a bilinear local DTM sampler using the terrain raster implementation."""
    from samcloud.terrain.rasters import build_dtm_raster

    ground = ps.xyz[np.isin(ps.classes, (GROUND, ROAD))]
    if len(ground) < 3:
        raise ValueError("cannot build a DTM sampler without at least three ground or road points")
    resolution = float(config["terrain"].get("dtm_resolution_m", 0.25))
    lower, upper = ps.xyz[:, :2].min(axis=0) - resolution, ps.xyz[:, :2].max(axis=0) + resolution
    bounds = (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]))
    dtm, _validity, _distance = build_dtm_raster(ground, resolution=resolution, bounds=bounds)
    known = np.isfinite(dtm)
    if not known.any():
        raise ValueError("DTM raster contains no sampled heights")
    rows, cols = np.nonzero(known)
    known_xy = np.column_stack((bounds[0] + (cols + 0.5) * resolution, bounds[3] - (rows + 0.5) * resolution))
    nearest = cKDTree(known_xy)

    def sample(xy: np.ndarray) -> np.ndarray:
        query = np.asarray(xy, dtype=float).reshape(-1, 2)
        _distance, index = nearest.query(query, workers=-1)
        return dtm[rows[index], cols[index]].astype(np.float64)

    return sample


def run_semantics(products_dir: str | Path, out_dir: str | Path, *, dtm_sampler: Callable | None = None, config: dict | None = None) -> dict:
    """Run semantic refinement and persist ``buildings/instances.json`` under ``out_dir``."""
    from samcloud.drone.io import load_classified_pointset_lv95

    cfg = config or load_drone_config()
    pointset, frame = load_classified_pointset_lv95(products_dir)
    sampler = dtm_sampler or _dtm_sampler_from_pointset(pointset, cfg)
    split, split_report = split_roof_facade(pointset, sampler)
    provisional = segment_buildings(split, sampler)
    corrected, plausibility_report = plausibility_rules(split, sampler, building_instances=provisional)
    instances = segment_buildings(corrected, sampler)
    output = Path(out_dir) / "buildings" / "instances.json"
    assign_persistent_ids(instances, output)
    save_instances(instances, output)
    return {
        "pointset": corrected,
        "instances": instances,
        "frame": frame,
        "report": {
            "split": split_report,
            "plausibility": plausibility_report,
            "segmentation": segment_buildings.last_report,
            "instances_path": str(output),
        },
    }
